//! NCCL bindings — dlopen-based FFI to libnccl.so.2 (no link-time NCCL
//! dependency; resolves at first use, same pattern as the CUDA driver API
//! in `cuda.rs`).
//!
//! Enabled with `--features cuda`. This is the TP data-plane for the B300
//! deployment: `TpCluster`'s CPU-simulated collectives (all_reduce_sum /
//! all_gather_rows) swap to these calls once shards each own a device.
//!
//! Usage model (single process, multiple GPUs — B300 single-node TP):
//!   1. `NcclApi::get()` resolves libnccl.
//!   2. `ncclCommInitAll(&mut comms, devices)` creates one comm per device.
//!   3. Enqueue collectives on each device's stream; the calls are
//!      asynchronous (NCCL stream semantics), sync via the stream.
//!
//! Multi-process (multi-node later): rank 0 generates a UniqueId, wire it
//! to peers (TCP/SSH/file), then `ncclCommInitRank` per rank.
//!
//! dtype/op enum values are the NCCL 2.x ABI (nccl.h):
//!   ncclFloat32 = 7, ncclBfloat16 = 9; ncclSum = 0.

#![cfg(feature = "cuda")]

use std::ffi::c_void;

use ferrite_types::{FerriteError, Result};

/// cudaStream_t (ABI-identical to CuStream in cuda.rs).
pub type CuStream = *mut c_void;
/// Opaque NCCL communicator handle.
pub type NcclComm = *mut c_void;
/// NCCL unique id (NCCL_UNIQUE_ID_BYTES = 128).
#[repr(C)]
pub struct NcclUniqueId {
    pub bytes: [u8; 128],
}

// ncclResult_t
pub const NCCL_SUCCESS: i32 = 0;
// ncclDataType_t
pub const NCCL_FLOAT32: u32 = 7;
pub const NCCL_BFLOAT16: u32 = 9;
pub const NCCL_FLOAT16: u32 = 6;
// ncclRedOp_t
pub const NCCL_SUM: u32 = 0;

type FnGetUniqueId = unsafe extern "C" fn(*mut NcclUniqueId) -> i32;
type FnCommInitRank = unsafe extern "C" fn(*mut NcclComm, i32, i32, NcclUniqueId) -> i32;
type FnCommInitAll = unsafe extern "C" fn(*mut NcclComm, i32, *const i32) -> i32;
type FnCommDestroy = unsafe extern "C" fn(NcclComm) -> i32;
type FnAllReduce = unsafe extern "C" fn(*const c_void, *mut c_void, usize, u32, u32, NcclComm, CuStream) -> i32;
type FnAllGather = unsafe extern "C" fn(*const c_void, *mut c_void, usize, u32, NcclComm, CuStream) -> i32;
type FnBroadcast = unsafe extern "C" fn(*mut c_void, *mut c_void, usize, u32, i32, NcclComm, CuStream) -> i32;
type FnReduceScatter = unsafe extern "C" fn(*const c_void, *mut c_void, usize, u32, u32, NcclComm, CuStream) -> i32;
type FnGetErrorString = unsafe extern "C" fn(i32) -> *const std::os::raw::c_char;

extern "C" {
    #[link_name = "dlopen"]
    fn libc_dlopen(filename: *const std::os::raw::c_char) -> *mut c_void;
    #[link_name = "dlsym"]
    fn libc_dlsym(handle: *mut c_void, symbol: *const std::os::raw::c_char) -> *mut c_void;
}

#[allow(non_snake_case)]
struct NcclApi {
    ncclGetUniqueId: FnGetUniqueId,
    ncclCommInitRank: FnCommInitRank,
    ncclCommInitAll: FnCommInitAll,
    ncclCommDestroy: FnCommDestroy,
    ncclAllReduce: FnAllReduce,
    ncclAllGather: FnAllGather,
    ncclBroadcast: FnBroadcast,
    ncclReduceScatter: FnReduceScatter,
    ncclGetErrorString: FnGetErrorString,
}

impl NcclApi {
    /// Resolve libnccl.so.2 (falling back to libnccl.so) once, dlsym the
    /// collective entry points. `None` when NCCL is absent (e.g. dev boxes
    /// without the toolkit — CPU-only builds never call this).
    pub fn get() -> Option<&'static NcclApi> {
        static API: std::sync::OnceLock<Option<NcclApi>> = std::sync::OnceLock::new();
        API.get_or_init(|| {
            let name = c"libnccl.so.2";
            let mut h = unsafe { libc_dlopen(name.as_ptr()) };
            if h.is_null() {
                let name2 = c"libnccl.so";
                h = unsafe { libc_dlopen(name2.as_ptr()) };
            }
            if h.is_null() {
                // Absolute-path fallbacks: the toolkit lib dirs are not on
                // the default loader path and serve's env may not have them
                // (observed: python ctypes loads fine but serve's dlopen
                // misses — the CUDA toolkit ships nccl in versioned dirs).
                for p in [
                    c"/usr/local/cuda-13.2/lib/libnccl.so.2",
                    c"/usr/local/cuda-12.9/lib/libnccl.so.2",
                    c"/usr/lib/x86_64-linux-gnu/libnccl.so.2",
                ] {
                    h = unsafe { libc_dlopen(p.as_ptr()) };
                    if !h.is_null() {
                        break;
                    }
                }
            }
            if h.is_null() {
                return None;
            }
            let sym = |c: &std::ffi::CStr| unsafe {
                let p = libc_dlsym(h, c.as_ptr());
                if p.is_null() { None } else { Some(p) }
            };
            let f = |p: *mut c_void, name: &str| -> String {
                let _ = name;
                format!("nccl symbol missing: {name}")
            };
            let _ = f;
            // NOTE: individual symbol misses fall through to None (a partial
            // NCCL is not usable); the closure above documents each name.
            let i_id: *mut c_void = sym(c"ncclGetUniqueId")?;
            let i_ir: *mut c_void = sym(c"ncclCommInitRank")?;
            let i_ia: *mut c_void = sym(c"ncclCommInitAll")?;
            let i_cd: *mut c_void = sym(c"ncclCommDestroy")?;
            let i_ar: *mut c_void = sym(c"ncclAllReduce")?;
            let i_ag: *mut c_void = sym(c"ncclAllGather")?;
            let i_bc: *mut c_void = sym(c"ncclBroadcast")?;
            let i_rs: *mut c_void = sym(c"ncclReduceScatter")?;
            let i_es: *mut c_void = sym(c"ncclGetErrorString")?;
            Some(NcclApi {
                ncclGetUniqueId: unsafe { std::mem::transmute::<*mut c_void, FnGetUniqueId>(i_id) },
                ncclCommInitRank: unsafe { std::mem::transmute::<*mut c_void, FnCommInitRank>(i_ir) },
                ncclCommInitAll: unsafe { std::mem::transmute::<*mut c_void, FnCommInitAll>(i_ia) },
                ncclCommDestroy: unsafe { std::mem::transmute::<*mut c_void, FnCommDestroy>(i_cd) },
                ncclAllReduce: unsafe { std::mem::transmute::<*mut c_void, FnAllReduce>(i_ar) },
                ncclAllGather: unsafe { std::mem::transmute::<*mut c_void, FnAllGather>(i_ag) },
                ncclBroadcast: unsafe { std::mem::transmute::<*mut c_void, FnBroadcast>(i_bc) },
                ncclReduceScatter: unsafe { std::mem::transmute::<*mut c_void, FnReduceScatter>(i_rs) },
                ncclGetErrorString: unsafe { std::mem::transmute::<*mut c_void, FnGetErrorString>(i_es) },
            })
        })
        .as_ref()
    }

    fn err_str(&self, r: i32) -> String {
        unsafe {
            let p = (self.ncclGetErrorString)(r);
            if p.is_null() {
                format!("nccl error {r}")
            } else {
                std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
            }
        }
    }

    fn ck(&self, r: i32, op: &str) -> Result<()> {
        if r == NCCL_SUCCESS {
            Ok(())
        } else {
            Err(FerriteError::InvalidArg(format!("{op} failed: {}", self.err_str(r))))
        }
    }
}

/// A resolved NCCL communicator bound to one (device, stream) pair.
/// Single-process TP on B300: create one per device via
/// [`NcclGroup::init_all`].
pub struct NcclChannel {
    api: &'static NcclApi,
    comm: NcclComm,
    pub stream: CuStream,
    pub rank: usize,
    pub world: usize,
}

/// Builder for a set of single-process communicators (one per GPU).
pub struct NcclGroup;

impl NcclGroup {
    /// Single-process multi-GPU: one communicator per listed CUDA device
    /// ordinal. Caller keeps every returned channel alive together.
    /// Uses ncclCommInitRank from per-rank threads (PyTorch-style) —
    /// ncclCommInitAll fails with "unhandled cuda error" when the calling
    /// thread already has CUDA contexts from CudaBackend creation.
    pub fn init_all(devices: &[i32], streams: &[CuStream]) -> Result<Vec<NcclChannel>> {
        let api = NcclApi::get()
            .ok_or_else(|| FerriteError::InvalidArg("libnccl not loadable (no NCCL installed?)".into()))?;
        if devices.len() != streams.len() {
            return Err(FerriteError::InvalidArg("devices/streams length mismatch".into()));
        }
        let world = devices.len();

        // Generate unique ID (rank 0 style — all ranks share it)
        let mut id = NcclUniqueId { bytes: [0u8; 128] };
        let r = unsafe { (api.ncclGetUniqueId)(&mut id) };
        api.ck(r, "ncclGetUniqueId")?;

        // Spawn a thread per rank, each calls ncclCommInitRank concurrently
        // (this is how PyTorch initializes NCCL — it works on machines where
        // ncclCommInitAll fails due to pre-existing CUDA contexts).
        let mut handles = vec![];
        for rank in 0..world {
            let id_bytes = id.bytes; // Copy ([u8;128] is Send)
            let world_i = world as i32;
            let rank_i = rank as i32;
            let device = devices[rank];

            handles.push(std::thread::spawn(move || {
                // Set CUDA device for this thread (fresh context)
                unsafe {
                    crate::cuda::cuda_set_device(device);
                }
                let api = match NcclApi::get() {
                    Some(a) => a,
                    None => return (rank, 0usize, -1i32), // api lost
                };
                let mut comm: NcclComm = std::ptr::null_mut();
                let local_id = NcclUniqueId { bytes: id_bytes };
                let r = unsafe {
                    (api.ncclCommInitRank)(&mut comm, rank_i, world_i, local_id)
                };
                (rank, comm as usize, r)
            }));
        }

        // Collect results (usize comm pointers are Send)
        let mut comms: Vec<usize> = vec![0; world];
        let mut first_err: Option<String> = None;
        for h in handles {
            let (rank, comm, r) = h
                .join()
                .map_err(|_| FerriteError::InvalidArg("nccl init thread panicked".into()))?;
            if r != 0 {
                let err_str = api.err_str(r);
                if first_err.is_none() {
                    first_err = Some(format!("ncclCommInitRank rank {rank}: {err_str}"));
                }
            } else {
                comms[rank] = comm;
            }
        }
        if let Some(e) = first_err {
            return Err(FerriteError::InvalidArg(e));
        }
        if comms.iter().any(|c| *c == 0) {
            return Err(FerriteError::InvalidArg("ncclCommInitRank returned null comm".into()));
        }

        Ok(comms
            .into_iter()
            .zip(streams.iter())
            .enumerate()
            .map(|(rank, (comm, &stream))| NcclChannel {
                api,
                comm: comm as NcclComm,
                stream,
                rank,
                world,
            })
            .collect())
    }

    /// Multi-process: generate a unique id on rank 0 (broadcast it out of
    /// band — SSH/file/TCP, NCCL never does transport for bootstrap), then
    /// every rank calls init_rank.
    pub fn make_unique_id() -> Result<NcclUniqueId> {
        let api = NcclApi::get()
            .ok_or_else(|| FerriteError::InvalidArg("libnccl not loadable".into()))?;
        let mut id = NcclUniqueId { bytes: [0u8; 128] };
        let r = unsafe { (api.ncclGetUniqueId)(&mut id) };
        api.ck(r, "ncclGetUniqueId")?;
        Ok(id)
    }

    pub fn init_rank(id: NcclUniqueId, rank: i32, world: i32, stream: CuStream) -> Result<NcclChannel> {
        let api = NcclApi::get()
            .ok_or_else(|| FerriteError::InvalidArg("libnccl not loadable".into()))?;
        let mut comm: NcclComm = std::ptr::null_mut();
        let r = unsafe { (api.ncclCommInitRank)(&mut comm, rank, world, id) };
        api.ck(r, "ncclCommInitRank")?;
        Ok(NcclChannel {
            api,
            comm,
            stream,
            rank: rank as usize,
            world: world as usize,
        })
    }
}

impl NcclChannel {
    /// All-reduce (sum) over f32 device buffers. Asynchronous — NCCL enqueues
    /// on the channel's stream; sync the stream when results are needed.
    /// `send` and `recv` are device pointers (e.g. DevBuf::as_f32()).
    pub fn all_reduce_f32(
        &self,
        send: *const f32,
        recv: *mut f32,
        count: usize,
    ) -> Result<()> {
        let r = unsafe {
            (self.api.ncclAllReduce)(
                send as *const c_void,
                recv as *mut c_void,
                count,
                NCCL_FLOAT32,
                NCCL_SUM,
                self.comm,
                self.stream,
            )
        };
        self.api.ck(r, "ncclAllReduce")
    }

    /// All-gather f32 device buffers: `recv` must hold world × count elements.
    pub fn all_gather_f32(
        &self,
        send: *const f32,
        recv: *mut f32,
        count: usize,
    ) -> Result<()> {
        let r = unsafe {
            (self.api.ncclAllGather)(
                send as *const c_void,
                recv as *mut c_void,
                count,
                NCCL_FLOAT32,
                self.comm,
                self.stream,
            )
        };
        self.api.ck(r, "ncclAllGather")
    }

    /// Broadcast f32 device buffer from `root` rank.
    pub fn broadcast_f32(
        &self,
        send: *mut f32, // in-place: also the recv buffer
        count: usize,
        root: i32,
    ) -> Result<()> {
        let r = unsafe {
            (self.api.ncclBroadcast)(
                send as *mut c_void,
                send as *mut c_void,
                count,
                NCCL_FLOAT32,
                root,
                self.comm,
                self.stream,
            )
        };
        self.api.ck(r, "ncclBroadcast")
    }

    /// Reduce-scatter (sum) f32: `send` holds world × count, `recv` gets
    /// this rank's count-sized slice of the sum. The RS-then-AG pair is the
    /// bandwidth-optimal TP residual update.
    pub fn reduce_scatter_f32(
        &self,
        send: *const f32,
        recv: *mut f32,
        count: usize,
    ) -> Result<()> {
        let r = unsafe {
            (self.api.ncclReduceScatter)(
                send as *const c_void,
                recv as *mut c_void,
                count,
                NCCL_FLOAT32,
                NCCL_SUM,
                self.comm,
                self.stream,
            )
        };
        self.api.ck(r, "ncclReduceScatter")
    }
}

impl Drop for NcclChannel {
    fn drop(&mut self) {
        if !self.comm.is_null() {
            unsafe { (self.api.ncclCommDestroy)(self.comm) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The FFI table must resolve (or cleanly report absence) — never
    /// transmute a null symbol. On boxes without libnccl this is a no-op
    /// pass; on B300 it exercises the full dlopen path.
    #[test]
    fn nccl_api_resolves_or_absent() {
        match NcclApi::get() {
            Some(_) => {} // B300 / NCCL present: table is complete
            None => {} // dev box: absence is fine, callers check Option
        }
    }
}

// Cross-thread use: NCCL communicators are thread-safe for collective calls
// (each channel is bound to one stream; the TP fan_out workers each drive
// their own rank's channel). The raw comm pointer is not inherently Send,
// but single-process init_all comms are valid process-wide.
unsafe impl Send for NcclChannel {}
unsafe impl Sync for NcclChannel {}
