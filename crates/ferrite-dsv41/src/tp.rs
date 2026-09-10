//! Tensor parallelism for DeepSeek-V4.1-Flash: one process, one rank per thread.
//!
//! The CUDA runtime's device selection is **per-thread**, so a thread that calls
//! `cudaSetDevice(r)` drives device `r` for all of its subsequent calls. That
//! makes the simplest correct TP bring-up: spawn `world` threads, bind thread `r`
//! to device `r`, give it its own weights slice (the loader already slices by
//! `(world, rank)`) and its own `DevChain`, and let a barrier keep the ranks in
//! lockstep.
//!
//! Communication is peer copies plus a local reduction — no NCCL. Every rank
//! owns a staging buffer of `world` slots; rank `r` writes its values into slot
//! `r` of *every* rank's staging, then each rank reduces (all-reduce) or reads
//! all slots (all-gather). Payloads here are small (a few KB per site; the
//! largest is the vocabulary gather at sampling), so the copy cost is far below
//! what the layer maths costs.
//!
//! The reduction sums the slots in **rank order**, identically on every rank, so
//! the result is bit-identical across ranks and reproducible run to run.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Spin barrier. `std::sync::Barrier` parks and unparks threads through the
/// futex, which costs microseconds and, worse, an unbounded wakeup latency: with
/// ~3 waits per all-reduce and ~90 all-reduces per step that alone can dominate
/// the step. The ranks are a handful of threads on the same machine, so a
/// generation-counting spin is both correct and far cheaper.
pub struct SpinBarrier {
    n: usize,
    count: AtomicUsize,
    gen: AtomicUsize,
}

impl SpinBarrier {
    pub fn new(n: usize) -> Self {
        Self {
            n,
            count: AtomicUsize::new(0),
            gen: AtomicUsize::new(0),
        }
    }

    pub fn wait(&self) {
        if self.n <= 1 {
            return;
        }
        let g = self.gen.load(Ordering::Acquire);
        if self.count.fetch_add(1, Ordering::AcqRel) + 1 == self.n {
            // last in: reset the count before releasing the others
            self.count.store(0, Ordering::Release);
            self.gen.fetch_add(1, Ordering::Release);
        } else {
            let mut spins = 0u32;
            while self.gen.load(Ordering::Acquire) == g {
                spins += 1;
                if spins < 4096 {
                    std::hint::spin_loop();
                } else {
                    std::thread::yield_now();
                }
            }
        }
    }
}

use ferrite_types::{FerriteError, Result};

use crate::device::{DevBuf, Device};

/// Collective with a fixed payload size.
pub struct Collective {
    pub world: usize,
    pub rank: usize,
    pub bytes: usize,
    staging: DevBuf,
    /// Base address of each rank's staging buffer, indexed by rank.
    peers: Vec<u64>,
    barrier: Arc<SpinBarrier>,
    dev: Arc<Device>,
}

impl Collective {
    /// `staging` is `world * bytes`. The peer addresses are not known until
    /// every rank has allocated theirs, so this starts empty and `set_peers`
    /// fills it after the handshake.
    pub fn new(
        dev: Arc<Device>,
        world: usize,
        rank: usize,
        bytes: usize,
        barrier: Arc<SpinBarrier>,
    ) -> Result<Self> {
        let depth = bytes * std::mem::size_of::<f32>() / 4;
        let staging = dev.alloc(world * bytes)?;
        let _ = depth;
        Ok(Collective {
            world,
            rank,
            bytes,
            staging,
            peers: vec![0; world],
            barrier,
            dev,
        })
    }

    /// Install the peers' staging base addresses (from the startup handshake).
    pub fn set_peers(&mut self, peers: Vec<u64>) -> Result<()> {
        if peers.len() != self.world {
            return Err(FerriteError::Config(format!(
                "collective expects {} peer addresses, got {}",
                self.world,
                peers.len()
            )));
        }
        if peers.iter().any(|&p| p == 0) {
            return Err(FerriteError::Config("a peer address is 0".into()));
        }
        self.peers = peers;
        Ok(())
    }

    pub fn staging_base(&self) -> u64 {
        self.staging.ptr as u64
    }

    /// Publish `len` bytes from `src` into slot `rank` of every rank. `len`
    /// must not exceed the slot size: a site with a shorter payload than the
    /// staging would otherwise publish unrelated memory.
    fn publish(&self, src: *const std::ffi::c_void, len: usize) -> Result<()> {
        assert!(len <= self.bytes, "collective payload {len} > slot {}", self.bytes);
        for p in 0..self.world {
            let dst = (self.peers[p] + (self.rank * self.bytes) as u64) as *mut std::ffi::c_void;
            if p == self.rank {
                self.dev.memcpy_d2d(dst, src as *const std::ffi::c_void, len)?;
            } else {
                self.dev
                    .memcpy_peer(p as i32, dst, src as *const std::ffi::c_void, len)?;
            }
        }
        // The peer copies above use the SYNCHRONOUS cuMemcpyPeer (see Device::
        // memcpy_peer), so they have already landed by the time this returns; the
        // old comment here claimed they were asynchronous and defended a
        // cudaDeviceSynchronize. That sync ran on EVERY collective call — ~90
        // per decode step — and each one drains the whole pipeline, which is the
        // dominant per-layer cost. Stream ordering plus the barrier below is
        // enough: this rank's copies are complete, and every rank's writes are
        // issued before anyone reduces.
        self.barrier.wait();
        Ok(())
    }

    /// sum over ranks, written back to `dst` (which may be the same address as
    /// `src`).
    pub fn all_reduce_inplace(&self, buf: *mut std::ffi::c_void, len: usize) -> Result<()> {
        self.publish(buf as *const std::ffi::c_void, len)?;
        let slot0 = (self.staging.ptr as *mut u8);
        let n = (len / 4) as i64;
        for i in 1..self.world {
            let src = slot0.wrapping_add(i * self.bytes);
            self.dev
                .add_inplace_raw(slot0 as *mut std::ffi::c_void, src as *const std::ffi::c_void, n)?;
        }
        self.dev
            .memcpy_d2d(buf, slot0 as *const std::ffi::c_void, len)?;
        self.barrier.wait();
        Ok(())
    }

    /// Each rank's `len` bytes end up available in slot order (rank r's data in
    /// slot r). No reduction: the caller reads whichever slots it needs.
    pub fn all_gather(&self, src: *const std::ffi::c_void, len: usize) -> Result<()> {
        self.publish(src, len)?;
        Ok(())
    }

    pub fn slot(&self, i: usize) -> *const std::ffi::c_void {
        (self.staging.ptr as *const u8).wrapping_add(i * self.bytes) as *const std::ffi::c_void
    }

    pub fn slot_mut(&self, i: usize) -> *mut std::ffi::c_void {
        (self.staging.ptr as *mut u8).wrapping_add(i * self.bytes) as *mut std::ffi::c_void
    }

    /// Release a round (pairs with `publish`, to keep the next round from
    /// overwriting slots another rank is still reading).
    pub fn end_round(&self) {
        self.barrier.wait();
    }
}

/// Shared state a rank thread needs from its siblings.
pub struct RankLinks {
    pub barrier: Arc<SpinBarrier>,
    /// Filled by each rank at startup: `peers[rank] = staging base address`.
    pub peers_small: Vec<u64>,
    pub peers_big: Vec<u64>,
}

impl RankLinks {
    pub fn new(world: usize) -> Self {
        RankLinks {
            barrier: Arc::new(SpinBarrier::new(world)),
            peers_small: vec![0; world],
            peers_big: vec![0; world],
        }
    }
}

/// Run `f` on `world` threads, one per rank, each bound to its own device.
/// Returns rank 0's result.
pub fn run_ranks<F, T>(world: usize, f: F) -> Result<T>
where
    F: Fn(usize) -> Result<T> + Send + Sync + 'static,
    T: Send + 'static,
{
    let f = Arc::new(f);
    let mut handles = Vec::with_capacity(world);
    for r in 0..world {
        let f = Arc::clone(&f);
        handles.push(
            std::thread::Builder::new()
                .name(format!("dsv41-rank{r}"))
                .spawn(move || -> Result<T> { f(r) })
                .map_err(|e| FerriteError::Config(format!("spawn rank {r}: {e}")))?,
        );
    }
    let mut out: Vec<Option<Result<T>>> = (0..world).map(|_| None).collect();
    for (r, h) in handles.into_iter().enumerate() {
        let v = h
            .join()
            .map_err(|_| FerriteError::Config(format!("rank {r} panicked")))?;
        out[r] = Some(v);
    }
    out.into_iter()
        .next()
        .flatten()
        .ok_or_else(|| FerriteError::Config("no rank result".into()))?
}

/// Exchange staging addresses so every rank can reach every peer's buffer.
///
/// `slots` is a shared `world`-element table the ranks fill in turn; the barrier
/// before and after makes the handshake deterministic.
pub fn exchange(
    world: usize,
    rank: usize,
    small: u64,
    big: u64,
    table_small: &std::sync::Mutex<Vec<u64>>,
    table_big: &std::sync::Mutex<Vec<u64>>,
    barrier: &Arc<SpinBarrier>,
) -> (Vec<u64>, Vec<u64>) {
    {
        let mut t = table_small.lock().unwrap();
        t[rank] = small;
        let mut b = table_big.lock().unwrap();
        b[rank] = big;
    }
    barrier.wait();
    let s = table_small.lock().unwrap().clone();
    let b = table_big.lock().unwrap().clone();
    barrier.wait();
    let _ = world;
    (s, b)
}
