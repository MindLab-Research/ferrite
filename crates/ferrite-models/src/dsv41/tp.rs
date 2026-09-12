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

use std::sync::atomic::{AtomicI32, AtomicUsize, Ordering};
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
    /// The arm vote — an INDEPENDENT generation, deliberately kept out of
    /// `count`/`gen` above. `wait` counts ARRIVALS per generation, so folding a
    /// payload-carrying rendezvous into the same counter would let a vote change
    /// the epoch sequence the callers of `wait` see (exactly the coupling that
    /// makes the arms' rendezvous counts load-bearing in `chain_dev`). See
    /// [`RankVote`] for the protocol.
    vote: RankVote,
}

impl SpinBarrier {
    pub fn new(n: usize) -> Self {
        Self {
            n,
            count: AtomicUsize::new(0),
            gen: AtomicUsize::new(0),
            vote: RankVote::new(n),
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

    /// This rank's `value`, returned as the UNANIMOUS value iff every rank voted
    /// the same, `None` otherwise. ONE rendezvous, and it does NOT touch `wait`'s
    /// generation. See [`RankVote::unanimous_i32`].
    pub fn unanimous_i32(&self, rank: usize, value: i32) -> Option<i32> {
        self.vote.unanimous_i32(rank, value)
    }
}

/// A one-`i32`-per-rank **unanimity vote**: every rank publishes one code and
/// learns whether the ranks agreed, in a single rendezvous.
///
/// # Why a vote rather than a barrier
///
/// The verify-graph gate in `chain_dev` is per-RANK (a capture-failure latch, a
/// shape pool, a `compress_len` host mirror), while the arms it chooses between
/// take DIFFERENT numbers of `host_barrier` arrivals per block. `SpinBarrier`
/// counts arrivals, so two ranks in different arms misphase every later barrier
/// (`ar5-hang`). The fix is to make the ranks agree on the arm before any of
/// them walks it — and, when they cannot agree, to fall back to the one arm that
/// exists in every state (the direct launches). A plain barrier cannot carry
/// that decision, hence the payload.
///
/// # Protocol (and why it is lap-safe)
///
/// Round `n` uses the buffer selected by `n & 1`, so a rank that has already
/// started round `n + 1` writes into the OTHER half and cannot be observed by a
/// slow peer still reading round `n`. Only one round of lapping is possible
/// (round `n + 1` needs every rank to arrive, and the slow rank has not), which
/// is exactly what the parity covers. The arrival counter is reset by the last
/// arriver BEFORE it bumps the generation, so a lap cannot be miscounted.
///
/// The result is deliberately all-or-nothing: a single dissenting rank sends the
/// whole world to the fallback. That is the conservative direction — the
/// alternative (majority) would leave a minority walking an arm whose rendezvous
/// count the majority does not share.
pub struct RankVote {
    world: usize,
    /// Arrivals in the CURRENT round.
    arrived: AtomicUsize,
    /// Completed rounds; also the buffer parity for the current round.
    gen: AtomicUsize,
    /// `2 * world` slots: round `n` uses `votes[(n & 1) * world .. + world]`.
    votes: Vec<AtomicI32>,
}

impl RankVote {
    pub fn new(world: usize) -> Self {
        let world = world.max(1);
        Self {
            world,
            arrived: AtomicUsize::new(0),
            gen: AtomicUsize::new(0),
            votes: (0..2 * world).map(|_| AtomicI32::new(0)).collect(),
        }
    }

    /// Publish `value` for `rank` and return `Some(v)` iff every rank published
    /// `v` this round, `None` on any disagreement. A world of one is unanimity by
    /// construction and skips the rendezvous entirely.
    pub fn unanimous_i32(&self, rank: usize, value: i32) -> Option<i32> {
        if self.world <= 1 {
            return Some(value);
        }
        let g = self.gen.load(Ordering::Acquire);
        let base = (g & 1) * self.world;
        self.votes[base + rank].store(value, Ordering::Release);
        if self.arrived.fetch_add(1, Ordering::AcqRel) + 1 == self.world {
            // Last in: clear the counter for the next round FIRST (a rank that
            // laps cannot pass the generation bump below until this store is
            // visible), then release the others with the bump.
            self.arrived.store(0, Ordering::Release);
            self.gen.store(g + 1, Ordering::Release);
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
        let first = self.votes[base].load(Ordering::Acquire);
        let all = self.votes[base..base + self.world]
            .iter()
            .all(|v| v.load(Ordering::Acquire) == first);
        all.then_some(first)
    }
}

use std::ffi::{c_int, c_uint};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering as AtOrd};

use ferrite_types::{FerriteError, Result};

use crate::dsv41::device::{DevBuf, Device};

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
        // `ctr_at + 4` (the word right after the v5 epoch) is the A4 broadcast
        // flag: `DSV41_AR_SINGLE_POLL` makes the pubred kernels' block 0 wait on
        // the peers' stamps and then publish `e + 1` there, so the other blocks
        // wait on ONE local word instead of on 8 stamps (160 pollers -> 8). It is
        // written by nothing else in the tree — the v2 path binds it as
        // `_ctr2_unused` (see `all_reduce_inplace_inner`) and its reader no-ops
        // unless `do_mark` is set. Zeroing it here is what makes the monotonic
        // `(int)(flag - (e+1)) < 0` wait in the kernel start from a clean slate
        // (the epoch itself is never reset at runtime).
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

    /// Device copy of the peers' staging bases (one u64 per rank) — what a
    /// cross-rank publish kernel walks to land a payload in every rank's slot.
    pub fn peer_slots_dev(&self) -> *const std::ffi::c_void {
        self.peer_slots.ptr as *const std::ffi::c_void
    }

    /// This rank's own staging base, for a kernel that has to read the slots
    /// peers published into.
    pub fn staging_dev(&self) -> *const std::ffi::c_void {
        self.staging.ptr as *const std::ffi::c_void
    }

    /// Peers' staging bases as a u64 pointer array — the typed form the
    /// vocabulary-sliced argmax exchange walks (`staging_tbl`).
    pub fn peer_slots_u64(&self) -> *const *mut u64 {
        self.peer_slots.ptr as *const *mut u64
    }

    /// Peers' ready-row bases (staging base + stamps_at) as a u32 pointer
    /// array — the argmax exchange stamps `ready_tbl[r][my_rank]`.
    pub fn peer_stamps_u32(&self) -> *const *mut u32 {
        self.peer_stamps.ptr as *const *mut u32
    }

    /// This rank's device epoch (`staging + ctr_at`) — the v5 round counter the
    /// argmax exchange reads and advances.
    pub fn epoch_dev(&self) -> *mut std::ffi::c_uint {
        (self.staging.ptr as *mut u8).wrapping_add(self.ctr_at) as *mut std::ffi::c_uint
    }

    /// This rank's own ready row (`staging + stamps_at`).
    pub fn ready_local_dev(&self) -> *const std::ffi::c_uint {
        (self.staging.ptr as *const u8).wrapping_add(self.stamps_at)
            as *const std::ffi::c_uint
    }

    /// True iff the device-side v5 protocol is live (epoch/stamps meaningful).
    /// The sliced lm_head must not run its exchange outside that protocol.
    pub fn uses_v5(&self) -> bool {
        ar_v5()
    }

    /// Peers' staging bases as an f32 pointer table — the `staging_tbl` form the
    /// fused epilogue store walks (`staging_tbl[rr][...] = v`). Bit-identical to
    /// `peer_slots_u64` (both are 8-byte device addresses), typed for the kernel.
    pub fn peer_slots_f32(&self) -> *const *mut f32 {
        self.peer_slots.ptr as *const *mut f32
    }

    /// This rank's device epoch as a READ-ONLY pointer — what the fused store in
    /// the producer kernel's epilogue dereferences (`*epoch`); the write side
    /// (pubred advancing it) goes through `epoch_dev`.
    pub fn epoch_u32(&self) -> *const c_uint {
        self.epoch_dev() as *const c_uint
    }

    /// Per-slot stride in FLOAT ELEMENTS (`bytes/4`) — the unit every v5 kernel
    /// indexes with. Passing `bytes` instead walks 4x past the slot into the
    /// stamp row and silently corrupts it.
    pub fn slot_stride_elems(&self) -> i32 {
        (self.bytes / 4) as i32
    }

    /// AR v5 publish + reduce ONLY: the store half already ran inside the producer
    /// kernel's epilogue (see `Device::gemm_fp8_mx_ar`), so this skips
    /// `p2p_ar_store_v5_kernel` and goes straight to the pubred kernel. `len` must
    /// equal the payload the fused store wrote. The fused producer and this call
    /// MUST be adjacent on the stream (no other all-reduce in between): both read
    /// the same `*epoch`, and pubred advances it at the end.
    pub fn all_reduce_inplace_pubred_only(
        &self,
        buf: *mut std::ffi::c_void,
        len: usize,
    ) -> Result<()> {
        debug_assert!(
            ar_v5(),
            "fused-store AR requires AR v5 (the producer kernel read *epoch)"
        );
        let n = (len / 4) as c_int;
        let stride = self.slot_stride_elems();
        let base8 = self.staging.ptr as *const u8;
        let staging_local = self.staging.ptr as *const f32;
        let ready_local = base8.wrapping_add(self.stamps_at) as *const c_uint;
        let epoch = (self.staging.ptr as *mut u8).wrapping_add(self.ctr_at) as *mut c_uint;
        self.dev.p2p_ar_pubred_v5(
            self.peer_stamps.ptr as *const *mut u32,
            epoch,
            staging_local,
            ready_local,
            buf as *mut f32,
            n,
            self.world as c_int,
            self.rank as c_int,
            stride,
        )
    }

    /// AR v5 with the segment-C `hc_post_inplace` fused into the pubred
    /// epilogue (see `ferrite_p2p_ar_v5_hcpost`). Returns `Ok(true)` when the
    /// fused path ran — the caller MUST then skip the standalone
    /// `hc_post_inplace`; `Ok(false)` means the protocol or the shape declined
    /// and the caller must use the plain `all_reduce_inplace` +
    /// `hc_post_inplace` pair instead.
    ///
    /// The fused kernel reads and writes `res` in place (one thread owns each
    /// payload column and walks all `hc_n` rows itself), so `res` must be the
    /// residual stream (`[hc_n][hc_h]`, row stride `hc_h`) and `len` must be
    /// `hc_h` floats: the AR payload axis and the hc-post column axis are the
    /// same one.
    #[allow(clippy::too_many_arguments)]
    pub fn all_reduce_inplace_hcpost(
        &self,
        buf: *mut std::ffi::c_void,
        len: usize,
        res: *mut f32,
        post: *const f32,
        comb: *const f32,
        hc_n: i32,
        hc_h: i32,
    ) -> Result<bool> {
        // Same shape gate as `dsv41_hc_post_inplace`: h % 4 == 0 (float4 path)
        // and 1 <= n <= 8 (the register-staging bound).
        if !ar_v5()
            || hc_n <= 0
            || hc_n > 8
            || hc_h <= 0
            || (hc_h & 3) != 0
            || (len / 4) as i32 != hc_h
        {
            return Ok(false);
        }
        let n = (len / 4) as c_int;
        let stride = (self.bytes / 4) as c_int;
        let base8 = self.staging.ptr as *const u8;
        let staging_local = self.staging.ptr as *const f32;
        let ready_local = base8.wrapping_add(self.stamps_at) as *const c_uint;
        let epoch = (self.staging.ptr as *mut u8).wrapping_add(self.ctr_at) as *mut c_uint;
        if !self.dev.p2p_ar_v5_hcpost(
            buf as *const f32,
            self.peer_slots.ptr as *const *mut f32,
            self.peer_stamps.ptr as *const *mut u32,
            epoch,
            staging_local,
            ready_local,
            buf as *mut f32,
            n,
            self.world as c_int,
            self.rank as c_int,
            stride,
            res,
            post,
            comb,
            hc_n as c_int,
            hc_h as c_int,
        )? {
            return Ok(false);
        }
        Ok(true)
    }

    /// AR v5 with the elementwise residual `add_in` folded into the STORE
    /// epilogue (`ferrite_p2p_ar_v5_add`, chain_dev.rs `add_epi()`): the value
    /// published to every peer is `buf[i] + add_in[i]` instead of `buf[i]`, which
    /// is exactly what the standalone `add_inplace(buf, add_in)` immediately
    /// before this AR produced. The pubred/reduce is the unchanged v5 one, so the
    /// result is BIT-IDENTICAL (same operands, same ascending-rank order) and one
    /// launch shorter. `Ok(false)` when the .so lacks the symbol, so the caller
    /// runs the standalone add + `all_reduce_inplace` as before.
    ///
    /// `add_in` must be final on this stream before the call (the MoE dual-chain
    /// join already guarantees it at the only call site).
    pub fn all_reduce_inplace_add(
        &self,
        buf: *mut std::ffi::c_void,
        len: usize,
        add_in: *const f32,
    ) -> Result<bool> {
        if !ar_v5() {
            return Ok(false);
        }
        let n = (len / 4) as c_int;
        let stride = self.slot_stride_elems();
        let base8 = self.staging.ptr as *const u8;
        let staging_local = self.staging.ptr as *const f32;
        let ready_local = base8.wrapping_add(self.stamps_at) as *const c_uint;
        let epoch = (self.staging.ptr as *mut u8).wrapping_add(self.ctr_at) as *mut c_uint;
        self.dev.p2p_ar_v5_add(
            buf as *const f32,
            add_in,
            self.peer_slots.ptr as *const *mut f32,
            self.peer_stamps.ptr as *const *mut u32,
            epoch,
            staging_local,
            ready_local,
            buf as *mut f32,
            n,
            self.world as c_int,
            self.rank as c_int,
            stride,
        )
    }

    /// `all_reduce_inplace_hcpost` + the ADD_EPI residual in one launch (see
    /// [`Self::all_reduce_inplace_add`] and [`Self::all_reduce_inplace_hcpost`]).
    /// Same shape gate as the plain hcpost fold; `Ok(false)` when it declines or
    /// the .so is stale, so the caller falls back to add + `all_reduce_inplace` +
    /// `hc_post_inplace`.
    #[allow(clippy::too_many_arguments)]
    pub fn all_reduce_inplace_hcpost_add(
        &self,
        buf: *mut std::ffi::c_void,
        len: usize,
        add_in: *const f32,
        res: *mut f32,
        post: *const f32,
        comb: *const f32,
        hc_n: i32,
        hc_h: i32,
    ) -> Result<bool> {
        if !ar_v5()
            || hc_n <= 0
            || hc_n > 8
            || hc_h <= 0
            || (hc_h & 3) != 0
            || (len / 4) as i32 != hc_h
        {
            return Ok(false);
        }
        let n = (len / 4) as c_int;
        let stride = self.slot_stride_elems();
        let base8 = self.staging.ptr as *const u8;
        let staging_local = self.staging.ptr as *const f32;
        let ready_local = base8.wrapping_add(self.stamps_at) as *const c_uint;
        let epoch = (self.staging.ptr as *mut u8).wrapping_add(self.ctr_at) as *mut c_uint;
        if !self.dev.p2p_ar_v5_hcpost_add(
            buf as *const f32,
            add_in,
            self.peer_slots.ptr as *const *mut f32,
            self.peer_stamps.ptr as *const *mut u32,
            epoch,
            staging_local,
            ready_local,
            buf as *mut f32,
            n,
            self.world as c_int,
            self.rank as c_int,
            stride,
            res,
            post,
            comb,
            hc_n as c_int,
            hc_h as c_int,
        )? {
            return Ok(false);
        }
        Ok(true)
    }

    /// The MULTI-ROW form of [`Self::all_reduce_inplace_hcpost`]
    /// (`ferrite_p2p_ar_v5_hcpost_rows`): the verify forward's `hc_post` folded
    /// into the AR that produced its `x`. `buf` is the `rows * hc_h`-float AR
    /// payload (`wo_out_r` / `moe_out_r`, row stride `hc_h`, reduced in place —
    /// the result is exactly the `x` the standalone post consumes); `res` is the
    /// residual block `[rows, hc_n, hc_h]` and `post` / `comb` are its per-row
    /// coefficient slices `[rows, hc_n]` / `[rows, hc_n, hc_n]`, i.e. the layout
    /// `hc_post_inplace_rows` takes.
    ///
    /// Returns `Ok(true)` when the fused path ran — the caller MUST then skip the
    /// standalone `hc_post_inplace_rows`; `Ok(false)` means the protocol or the
    /// shape declined and the caller must use the
    /// `all_reduce_inplace` + `hc_post_inplace_rows` pair instead.
    #[allow(clippy::too_many_arguments)]
    pub fn all_reduce_inplace_hcpost_rows(
        &self,
        buf: *mut std::ffi::c_void,
        len: usize,
        res: *mut f32,
        post: *const f32,
        comb: *const f32,
        rows: i32,
        hc_n: i32,
        hc_h: i32,
    ) -> Result<bool> {
        // Same shape gate as `dsv41_hc_post_inplace_rows`: `h % 4 == 0` (the
        // float4 path, and what keeps a payload float4 inside one row) and
        // `1 <= n <= 8` (the register-staging bound); plus `n == rows * hc_h`
        // (the payload is exactly the block's rows).
        if !ar_v5()
            || rows <= 0
            || hc_n <= 0
            || hc_n > 8
            || hc_h <= 0
            || (hc_h & 3) != 0
            || (len / 4) as i32 != rows * hc_h
        {
            return Ok(false);
        }
        let n = (len / 4) as c_int;
        let stride = self.slot_stride_elems();
        let base8 = self.staging.ptr as *const u8;
        let staging_local = self.staging.ptr as *const f32;
        let ready_local = base8.wrapping_add(self.stamps_at) as *const c_uint;
        let epoch = (self.staging.ptr as *mut u8).wrapping_add(self.ctr_at) as *mut c_uint;
        if !self.dev.p2p_ar_v5_hcpost_rows(
            buf as *const f32,
            self.peer_slots.ptr as *const *mut f32,
            self.peer_stamps.ptr as *const *mut u32,
            epoch,
            staging_local,
            ready_local,
            buf as *mut f32,
            n,
            self.world as c_int,
            self.rank as c_int,
            stride,
            res,
            post,
            comb,
            hc_n as c_int,
            hc_h as c_int,
            rows as c_int,
        )? {
            return Ok(false);
        }
        Ok(true)
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
        // AR v5: fully device-side (store -> publish/reduce), NO host barrier,
        // NO copy-back, and the epoch lives in device memory so a captured graph
        // replays correctly. The protection is the publish chain, not a credit.
        // ONE protocol, ONE kernel set: this now calls the SHARED
        // ferrite_p2p_ar_v5 (ferrite_kernels.cu). DSV41's staging is already the
        // shared layout — parity halves [2][world][bytes] at offset 0, the
        // [world] ready row at `stamps_at`, the device epoch at `ctr_at` — and
        // its `peer_slots` / `peer_stamps` u64 tables are bit-compatible with the
        // shared float*/u32* pointer tables (both are 8-byte device addresses),
        // so no buffer change and no kernel parameterization were needed.
        if ar_v5() {
            let n = (len / 4) as c_int;
            let stride = (self.bytes / 4) as c_int;
            let base8 = self.staging.ptr as *const u8;
            let staging_local = self.staging.ptr as *const f32;
            let ready_local = base8.wrapping_add(self.stamps_at) as *const c_uint;
            let epoch = (self.staging.ptr as *mut u8).wrapping_add(self.ctr_at) as *mut c_uint;
            self.dev.p2p_ar_v5(
                buf as *const f32,
                self.peer_slots.ptr as *const *mut f32,
                self.peer_stamps.ptr as *const *mut u32,
                epoch,
                staging_local,
                ready_local,
                buf as *mut f32,
                n,
                self.world as c_int,
                self.rank as c_int,
                stride,
            )?;
            return Ok(());
        }
        let _t_ar = std::time::Instant::now();
        let r = self.all_reduce_inplace_inner(buf, len);
        AR_HOST_NS.fetch_add(_t_ar.elapsed().as_nanos() as u64, AtOrd::Relaxed);
        AR_CALLS.fetch_add(1, AtOrd::Relaxed);
        if AR_CALLS.load(AtOrd::Relaxed) % 512 == 0 {
            let n = AR_CALLS.load(AtOrd::Relaxed);
            let us = AR_HOST_NS.load(AtOrd::Relaxed) as f64 / 1000.0;
            eprintln!(
                "[ar] calls={n} host_total={:.1}ms avg={:.1}us | barriers={} bar_total={:.1}ms avg={:.1}us",
                us / 1000.0,
                us / n as f64,
                AR_BAR_CALLS.load(AtOrd::Relaxed),
                AR_BAR_NS.load(AtOrd::Relaxed) as f64 / 1e6,
                AR_BAR_NS.load(AtOrd::Relaxed) as f64 / 1000.0
                    / (AR_BAR_CALLS.load(AtOrd::Relaxed).max(1)) as f64
            );
        }
        r
    }

    fn all_reduce_inplace_inner(&self, buf: *mut std::ffi::c_void, len: usize) -> Result<()> {
        self.publish(buf as *const std::ffi::c_void, len)?;
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
        // ctr_at + 4: unused on this path (the reader below `do_mark == 0` does not
        // touch it) and now also the A4 broadcast flag word (`DSV41_AR_SINGLE_POLL`,
        // see `Collective::new`); v2 and the v5 pubred path never run the same round.
        let _ctr2_unused = (self.staging.ptr as *mut u8).wrapping_add(self.ctr_at + 4) as *mut c_uint;
        // In the device-side path the reduce writes STRAIGHT into the caller's
        // buffer: keeping dst inside the staging half meant the copy-back below
        // read the half the reduce had just been writing, and any ordering slip
        // there returns the rank's own input (the micro-benchmark saw exactly
        // that: 0.0 for rank 0). Reading from the staging parity half and
        // writing to the caller separates the two roles completely.
        let reduce_dst = if dev_side { buf as *mut f32 } else { base as *mut f32 };
        self.dev.ar_reduce2(
            reduce_dst,
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
        if !dev_side {
            // the non-dev path reduces in place in slot 0, so copy it out
            self.dev
                .memcpy_d2d(buf, base as *const std::ffi::c_void, len)?;
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

    /// Host-side rendezvous, used around a whole-step CUDA graph capture.
    ///
    /// A capture RECORDS instead of executing, so a rank that is recording is not
    /// publishing its all-reduce stamps. The device-side AR deliberately has no
    /// host rendezvous of its own (`end_round` returns early under `ar_v5`, because
    /// the publish chain covers the normal case), which leaves the ranks free to
    /// drift apart around a capture: a peer that is EXECUTING its AR polls for a
    /// stamp the recording rank is only writing down, gives up, and reads staging
    /// that was never published. Measured before this was added: with the
    /// whole-step graph on, a single request was bit-identical to the per-kernel
    /// path, but the fourth of six sequential requests failed with an illegal
    /// memory access on a rank that varied between runs - the signature of a race.
    pub fn host_barrier(&self) {
        self.barrier.wait();
    }

    /// Publish this rank's arm code and return the UNANIMOUS code, or `None` when
    /// the ranks disagree. The caller then falls back to the direct launches.
    ///
    /// This is the ONLY cross-rank decision primitive here that carries a
    /// payload; it rides the shared [`SpinBarrier`] (every rank already holds the
    /// same `Arc`, so no new plumbing) but on an INDEPENDENT generation, so a
    /// vote never perturbs the arrival epochs `host_barrier` hands out. See
    /// [`RankVote`] for why that separation is load-bearing.
    pub fn unanimous_i32(&self, value: i32) -> Option<i32> {
        self.barrier.unanimous_i32(self.rank, value)
    }

    /// Release a round (pairs with `publish`, to keep the next round from
    /// overwriting slots another rank is still reading).
    pub fn end_round(&self) {
        // v5 needs no host barrier: the publish chain already guarantees every
        // peer is past the previous round's reduce before the next store runs.
        if ar_v5() {
            return;
        }
        let _t = std::time::Instant::now();
        self.barrier.wait();
        AR_BAR_NS.fetch_add(_t.elapsed().as_nanos() as u64, AtOrd::Relaxed);
        AR_BAR_CALLS.fetch_add(1, AtOrd::Relaxed);
    }
}

/// Process-wide host-time accounting for the collectives. The ranks are threads
/// of one process, so plain atomics are enough; this exists because the segment
/// graph experiment proved that launch counts are NOT the host bottleneck, so the
/// remaining candidate has to be measured rather than assumed.
pub static AR_HOST_NS: AtomicU64 = AtomicU64::new(0);
pub static AR_CALLS: AtomicU64 = AtomicU64::new(0);
pub static AR_BAR_NS: AtomicU64 = AtomicU64::new(0);
pub static AR_BAR_CALLS: AtomicU64 = AtomicU64::new(0);

/// AR v5 (the graph-capturable all-reduce) switch, read ONCE: this is on the
/// per-call hot path, and a per-call getenv is a hot-path slip.
static AR_V5: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
pub(crate) fn ar_v5() -> bool {
    *AR_V5.get_or_init(|| {
        // The whole-step CUDA graph REQUIRES the device-side AR: a host barrier is
        // not a CUDA call, so it would not be recorded and the replayed graph
        // would silently lose the inter-rank synchronisation. The graph is
        // DSV41_GRAPH_STEP (default ON since 2026-09-11, see chain_dev.rs
        // step_impl), while the device-side
        // AR is DEFAULT ON in its own right: it is verified correct with the graph
        // both off and on (a full 32-step generation is bit-identical to the host
        // barrier path) and it removes ~21 of the ~29 us each all-reduce costs.
        // DSV41_AR_V5=0 restores the host barrier for A/B — but ONLY together with
        // DSV41_GRAPH_STEP=0: the graph leg above is an `||`, so with the (default
        // ON) whole-step graph this function is true no matter what DSV41_AR_V5
        // says. That short-circuit is deliberate (a captured graph cannot record a
        // host barrier), and it is also the reason a reader must not treat
        // `ar_v5()` as opt-in when auditing a capture gate (chain_dev.rs
        // verify_graph_gate).
        let graph = std::env::var("DSV41_GRAPH_STEP").map(|v| v != "0").unwrap_or(true);
        graph || std::env::var("DSV41_AR_V5").map(|v| v != "0").unwrap_or(true)
    })
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

/// The arm vote's protocol is concurrency code, so it is pinned here rather than
/// only through the engine: the ranks are threads of one process, which makes the
/// real thing reproducible as plain host threads.
#[cfg(test)]
mod vote_tests {
    use super::RankVote;
    use std::sync::Arc;

    /// `rounds` of `world` threads, each voting `f(rank, round)`; returns every
    /// rank's per-round answer.
    fn run(world: usize, rounds: usize, f: impl Fn(usize, usize) -> i32 + Send + Sync + 'static) -> Vec<Vec<Option<i32>>> {
        let vote = Arc::new(RankVote::new(world));
        let f = Arc::new(f);
        let hs: Vec<_> = (0..world)
            .map(|r| {
                let vote = vote.clone();
                let f = f.clone();
                std::thread::spawn(move || {
                    (0..rounds).map(|i| vote.unanimous_i32(r, f(r, i))).collect::<Vec<_>>()
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    }

    #[test]
    fn equal_votes_are_unanimous_on_every_rank() {
        let all = run(8, 512, |_, _| 3);
        for rank in &all {
            assert!(rank.iter().all(|v| *v == Some(3)));
        }
    }

    /// One dissenting rank must reach EVERY rank as a disagreement — never a
    /// mixed answer, which is what the fallback depends on.
    #[test]
    fn a_single_dissent_returns_none_everywhere() {
        let all = run(8, 512, |r, _| if r == 3 { 1 } else { 2 });
        for rank in &all {
            assert!(rank.iter().all(|v| v.is_none()));
        }
    }

    /// The votes are per-ROUND: a rank whose vote alternates must not bleed into
    /// a neighbour's round (the buffer parity is what guarantees this).
    #[test]
    fn rounds_do_not_bleed_into_each_other() {
        let all = run(4, 512, |r, i| if r == 0 && i % 2 == 1 { 1 } else { 0 });
        for (i, want) in (0..512).map(|i| if i % 2 == 0 { Some(0) } else { None }).enumerate() {
            for (r, rank) in all.iter().enumerate() {
                assert_eq!(rank[i], want, "round {i}, rank {r}");
            }
        }
    }

    #[test]
    fn a_one_rank_world_is_unanimous_without_a_rendezvous() {
        let vote = RankVote::new(1);
        for i in 0..8 {
            assert_eq!(vote.unanimous_i32(0, i), Some(i));
        }
    }
}
