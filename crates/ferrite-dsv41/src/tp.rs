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

use std::ffi::c_uint;
use std::sync::atomic::{AtomicU32, Ordering as AtOrd};

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
    /// Byte offset (from each rank's staging base) of its `stored` stamp array:
    /// `world` u32 slots, slot `w` = the round rank `w` has finished publishing.
    stamps_at: usize,
    /// Same, for the `reduced` stamps written after the reduce completes.
    reduced_at: usize,
    /// Byte offset of the two last-block counters (store, reduce) for this rank.
    ctr_at: usize,
    /// Device copy of the peers' STAMP bases, so the stamp kernel can write into
    /// every rank's array (including our own) without host involvement.
    peer_stamps: DevBuf,
    /// Device copy of the peers' STAGING bases, so one store kernel can publish
    /// into every rank's slot instead of `world` host-issued peer copies.
    peer_slots: DevBuf,
    /// Device copy of the peers' `reduced` stamp areas (written after their
    /// reduce completes) — the credit the store's device-side wait consumes.
    peer_reduced: DevBuf,
    /// Monotonic round counter: each all-reduce is one round. Atomic because
    /// the collective is shared behind an Arc and only `&self` is available.
    round: AtomicU32,
    barrier: Arc<SpinBarrier>,
    dev: Arc<Device>,
}

impl std::fmt::Debug for Collective {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Collective(world={}, rank={})", self.world, self.rank)
    }
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
        // Layout per rank: [parity 0: world*bytes][parity 1: world*bytes]
        //                 [stored: world u32][reduced: world u32]
        // Double buffering by round parity is what lets the host barrier go: a
        // writer targets the half last used by round-2, and the store kernel
        // waits on the peers' `reduced` stamps before touching it.
        let stamps_at = 2 * world * bytes;
        let reduced_at = stamps_at + world * 4;
        let ctr_at = reduced_at + world * 4;
        let staging = dev.alloc(ctr_at + 64)?;
        // dev.alloc is cudaMalloc, which does NOT zero. The stamp arrays, the
        // `reduced` marks and the two last-block counters must start at 0: the
        // store kernel decides "am I the last block" with atomicAdd(ctr), so a
        // garbage counter means the stamp is never published and the reduce
        // spins forever — exactly the dev-path hang the micro-benchmark found.
        dev.zero_at(staging.ptr, ctr_at + 64)?;
        let peer_stamps = dev.alloc(world * 8)?;
        let peer_slots = dev.alloc(world * 8)?;
        let peer_reduced = dev.alloc(world * 8)?;
        let _ = depth;
        Ok(Collective {
            world,
            rank,
            bytes,
            staging,
            peers: vec![0; world],
            stamps_at,
            reduced_at,
            ctr_at,
            peer_stamps,
            peer_slots,
            peer_reduced,
            round: AtomicU32::new(0),
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
        // the stamp kernel writes into every rank's stamp area directly, so keep
        // a device copy of those bases
        let bases: Vec<u64> = peers.iter().map(|p| p + self.stamps_at as u64).collect();
        let mut buf = vec![0u8; self.world * 8];
        for (i, b) in bases.iter().enumerate() {
            buf[i * 8..i * 8 + 8].copy_from_slice(&b.to_le_bytes());
        }
        self.dev.upload_bytes_at(&self.peer_stamps, &buf)?;
        let mut buf2 = vec![0u8; self.world * 8];
        for (i, b) in peers.iter().enumerate() {
            buf2[i * 8..i * 8 + 8].copy_from_slice(&b.to_le_bytes());
        }
        self.dev.upload_bytes_at(&self.peer_slots, &buf2)?;
        let mut buf3 = vec![0u8; self.world * 8];
        for (i, b) in peers.iter().enumerate() {
            let rb = b + self.reduced_at as u64;
            buf3[i * 8..i * 8 + 8].copy_from_slice(&rb.to_le_bytes());
        }
        self.dev.upload_bytes_at(&self.peer_reduced, &buf3)?;
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
        let round = self.round.fetch_add(1, AtOrd::AcqRel) + 1;
        let dev_side = std::env::var("DSV41_AR_DEV").map(|v| v != "0").unwrap_or(false);
        if dev_side {
            // ⛔ DO NOT ENABLE: this path HANGS. It was an attempt at a fully
            // device-side collective (parity staging + a credit wait on the
            // peers' `reduced` stamps + in-kernel stamping, no host barrier) to
            // make the layer graph-capturable. Two bugs were found and fixed
            // (the release needed the fence and the signal in the same threads;
            // the in-kernel stamping dereferenced a null counter on the default
            // path) but it still deadlocks, so the default remains the
            // host-barrier path and this one is left behind the switch for the
            // next session to finish. See STATUS.md.
            let parity_off = ((round as usize % 2) * self.world * self.bytes) as i64;
            let reduced_local =
                (self.staging.ptr as *const u8).wrapping_add(self.reduced_at) as *const c_uint;
            // BISECT: pass a null counter so the store kernel does NOT stamp; the
            // standalone ar_stamp below does it instead (the combination that is
            // independently verified by the micro-benchmark's default path).
            let ctr = std::ptr::null_mut::<c_uint>();
            self.dev.ar_store2(
                self.peer_slots.ptr as *const u64,
                self.world as i32,
                self.rank as i32,
                src as *const f32,
                (len / 4) as i64,
                (self.bytes / 4) as i64,
                parity_off / 4,
                reduced_local,
                round,
                self.peer_stamps.ptr as *const u64,
                ctr,
            )?;
        } else {
            self.dev.ar_store(
                self.peer_slots.ptr as *const u64,
                self.world as i32,
                self.rank as i32,
                src as *const f32,
                (len / 4) as i64,
                (self.bytes / 4) as i64,
            )?;
        }
        // The peer copies are asynchronous on both devices (Device::memcpy_peer
        // maps to the async peer copy), so a device sync is REQUIRED here to make
        // this rank's issue complete before the barrier lets anyone reduce.
        // Removing it produced wrong output (" toll id " instead of " Paris.") —
        // verified. The real speedup must come from replacing this drain with
        // stream-ordered completion + a device-side stamp, not from dropping it.
        // The stamp kernel runs on the same stream AFTER the peer copies, so it
        // cannot execute before they complete — its write is the guarantee that
        // this rank's data has landed. The old cudaDeviceSynchronize here blocked
        // the host (so nothing overlapped) ~90 times per decode step.
        let round = self.round.fetch_add(1, AtOrd::AcqRel) + 1;
        self.dev.ar_stamp(
            self.peer_stamps.ptr as *const u64,
            self.world as i32,
            self.rank as i32,
            round,
        )?;
        if !dev_side {
            self.barrier.wait();
        }
        Ok(())
    }

    /// sum over ranks, written back to `dst` (which may be the same address as
    /// `src`).
    pub fn all_reduce_inplace(&self, buf: *mut std::ffi::c_void, len: usize) -> Result<()> {
        self.publish(buf as *const std::ffi::c_void, len)?;
        let slot0 = (self.staging.ptr as *mut u8);
        let n = (len / 4) as i64;
        let slot_f = (self.bytes / 4) as i64;
        let round = self.round.load(AtOrd::Acquire);
        let dev_side = std::env::var("DSV41_AR_DEV").map(|v| v != "0").unwrap_or(false);
        let base = if dev_side {
            (self.staging.ptr as *const u8)
                .wrapping_add((round as usize % 2) * self.world * self.bytes)
        } else {
            self.staging.ptr as *const u8
        };
        let stamps = (self.staging.ptr as *const u8).wrapping_add(self.stamps_at) as *const c_uint;
        let ctr2 = (self.staging.ptr as *mut u8).wrapping_add(self.ctr_at + 4) as *mut c_uint;
        self.dev.ar_reduce2(
            base as *mut f32,
            base as *const f32,
            n,
            slot_f,
            self.world as i32,
            stamps,
            round,
            self.peer_reduced.ptr as *const u64,
            self.rank as i32,
            std::ptr::null_mut::<c_uint>(),
            0, // BISECT: the standalone ar_mark below does the marking
        )?;
        if dev_side {
            self.dev.ar_mark(
                self.peer_reduced.ptr as *const u64,
                self.world as i32,
                self.rank as i32,
                round,
            )?;
        }
        self.dev
            .memcpy_d2d(buf, base as *const std::ffi::c_void, len)?;
        if !dev_side {
            self.barrier.wait();
        }
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
