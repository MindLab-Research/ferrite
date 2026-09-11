//! Runtime-dlopen CUDA device runtime, shared by every model crate.
//!
//! This is the single home for the generic device plumbing a model needs when
//! it drives kernels at the raw-pointer level: dlopen of libcudart/libcublas
//! and of the loaded kernel `.so`, a stream, byte-granular device allocation,
//! H2D/D2H/D2D copies, peer access, and the CUDA-graph capture primitives.
//!
//! It is deliberately independent of the `cuda` feature and of any link-time
//! CUDA dependency: every symbol resolves through dlopen/dlsym at first use,
//! exactly like the rest of ferrite. A crate that only links this module keeps
//! building and unit-testing on a box with no CUDA; on the B300 the symbols
//! come from the real driver. `ferrite-kernel::cuda` is the tensor-level
//! `KernelBackend` (GLM); this module is the pointer-level runtime underneath
//! both the GLM chain and the DeepSeek chain.

use std::ffi::{c_char, c_int, c_uint, c_void, CString};

use ferrite_types::{FerriteError, Result};

/// Opaque stream handle (cudaStream_t == void* at the ABI level).
pub type CuStream = *mut c_void;

pub const CUDA_R_32F: c_int = 0;
pub const CUDA_R_16BF: c_int = 14;
const CUBLAS_OP_N: c_int = 0;
const CUBLAS_OP_T: c_int = 1;

const CUDA_MEMCPY_H2D: c_int = 1;
const CUDA_MEMCPY_D2H: c_int = 2;
const CUDA_MEMCPY_D2D: c_int = 3;

// ---------------------------------------------------------------- fn tables

// The full cudart surface a device runtime needs. A couple of entries have no
// caller yet (`memcpy_2d_async` — the strided upload is synchronous by design;
// `memcpy_peer` — the peer copy path uses the async form): they stay so the
// table is the complete primitive set, not a snapshot of today's call sites.
#[allow(dead_code)]
struct Cudart {
    malloc: unsafe extern "C" fn(*mut *mut c_void, usize) -> c_int,
    free: unsafe extern "C" fn(*mut c_void) -> c_int,
    memcpy: unsafe extern "C" fn(*mut c_void, *const c_void, usize, c_int) -> c_int,
    memcpy_async: Option<unsafe extern "C" fn(*mut c_void, *const c_void, usize, c_int, CuStream) -> c_int>,
    memset: unsafe extern "C" fn(*mut c_void, c_int, usize) -> c_int,
    memset_async: Option<unsafe extern "C" fn(*mut c_void, c_int, usize, CuStream) -> c_int>,
    stream_create: unsafe extern "C" fn(*mut CuStream) -> c_int,
    /// `cudaStreamCreateWithPriority` — the hc tail split's side stream is created
    /// at the device's GREATEST priority, so the graph replay prefers its 1-block
    /// LATE half over a full projection wave (see [`DevRuntime::side_stream`]).
    /// Optional: absent → the plain create is used and the split keeps its old
    /// (default-priority) scheduling.
    stream_create_priority:
        Option<unsafe extern "C" fn(*mut CuStream, c_int, c_int) -> c_int>,
    /// `cudaDeviceGetStreamPriorityRange` — "greatest" is a device property
    /// (usually -5, but never hardcode it), queried once at open.
    stream_priority_range: Option<unsafe extern "C" fn(*mut c_int, *mut c_int) -> c_int>,
    stream_sync: unsafe extern "C" fn(CuStream) -> c_int,
    dev_sync: unsafe extern "C" fn() -> c_int,
    last_error: unsafe extern "C" fn() -> c_int,
    strerror: unsafe extern "C" fn(c_int) -> *const c_char,
    set_device: unsafe extern "C" fn(c_int) -> c_int,
    get_device: unsafe extern "C" fn(*mut c_int) -> c_int,
    device_count: unsafe extern "C" fn(*mut c_int) -> c_int,
    enable_peer: unsafe extern "C" fn(c_int, c_uint) -> c_int,
    memcpy_peer_async: unsafe extern "C" fn(*mut c_void, c_int, *const c_void, c_int, usize, CuStream) -> c_int,
    memcpy_2d_async: unsafe extern "C" fn(*mut c_void, usize, *const c_void, usize, usize, usize, c_int, CuStream) -> c_int,
    memcpy_2d: unsafe extern "C" fn(*mut c_void, usize, *const c_void, usize, usize, usize, c_int) -> c_int,
    mem_info: unsafe extern "C" fn(*mut usize, *mut usize) -> c_int,
    memcpy_peer: unsafe extern "C" fn(*mut c_void, c_int, *const c_void, c_int, usize) -> c_int,
    // ---- CUDA graph capture (segment graphs: the layers are
    // [hc+attn] -> AR -> [hc+MoE] -> AR and nothing between the two ARs needs the
    // host, so each segment can be captured while the ARs stay host-issued) ----
    stream_begin_capture: Option<unsafe extern "C" fn(CuStream, c_int) -> c_int>,
    stream_end_capture: Option<unsafe extern "C" fn(CuStream, *mut *mut c_void) -> c_int>,
    graph_instantiate: Option<unsafe extern "C" fn(*mut *mut c_void, *mut c_void, u64) -> c_int>,
    graph_launch: Option<unsafe extern "C" fn(*mut c_void, CuStream) -> c_int>,
    graph_exec_destroy: Option<unsafe extern "C" fn(*mut c_void) -> c_int>,
    graph_destroy: Option<unsafe extern "C" fn(*mut c_void) -> c_int>,
    // ---- side-stream fork/join (DSV41_HC_TAIL_SPLIT) ----
    // The kernel launcher records fork/join and the model waits the join; Rust
    // only needs the record/wait primitives and the disable-timing create (an
    // event used inside a capture MUST be created with cudaEventDisableTiming).
    event_create_flags: Option<unsafe extern "C" fn(*mut *mut c_void, c_uint) -> c_int>,
    event_record: Option<unsafe extern "C" fn(*mut c_void, CuStream) -> c_int>,
    event_destroy: Option<unsafe extern "C" fn(*mut c_void) -> c_int>,
    stream_wait_event: Option<unsafe extern "C" fn(CuStream, *mut c_void, c_uint) -> c_int>,
}

/// `cudaEventDisableTiming` — required for any event recorded inside a stream
/// capture (a timing event makes `cudaStreamEndCapture` fail).
pub const CUDA_EVENT_DISABLE_TIMING: c_uint = 0x02;

/// `cudaStreamNonBlocking`: the side stream must not implicitly synchronise with
/// the legacy default stream (it never did — both streams here are explicit — but
/// the flag also documents the intent).
const CUDA_STREAM_NON_BLOCKING: c_int = 0x01;

/// `cudaGraphInstantiateFlagUseNodePriority` (CUDA 12.0+, driver_types.h).
/// Without it a replay runs every node at the LAUNCH stream's priority, and the
/// per-node priority captured from the side stream is silently ignored — which
/// is precisely the hole the hc tail split fell into (round 41: −0.20ms realised
/// of −0.86ms). Verified against CUDA 13.2's header: the flag is 8, and
/// "priorities ... are copied from stream priority during stream capture".
const CUDA_GRAPH_INSTANTIATE_USE_NODE_PRIORITY: u64 = 8;

struct Cublas {
    create: unsafe extern "C" fn(*mut *mut c_void) -> c_int,
    set_stream: unsafe extern "C" fn(*mut c_void, CuStream) -> c_int,
    #[allow(clippy::type_complexity)]
    gemm_ex: unsafe extern "C" fn(
        *mut c_void,
        c_int,
        c_int,
        c_int,
        c_int,
        c_int,
        *const c_void,
        *const c_void,
        c_int,
        c_int,
        *const c_void,
        c_int,
        c_int,
        *const c_void,
        *mut c_void,
        c_int,
        c_int,
        c_int,
        c_int,
    ) -> c_int,
}

extern "C" {
    #[link_name = "dlopen"]
    fn libc_dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    #[link_name = "dlsym"]
    fn libc_dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    #[link_name = "dlerror"]
    fn libc_dlerror() -> *const c_char;
}

const RTLD_NOW: c_int = 2;
const RTLD_GLOBAL: c_int = 0x100;

/// Resolve `name` in a dlopen handle, with a clear error when it is absent.
pub fn sym(handle: *mut c_void, name: &str) -> Result<*mut c_void> {
    let c = CString::new(name).map_err(|_| FerriteError::Config("bad symbol name".into()))?;
    let p = unsafe { libc_dlsym(handle, c.as_ptr()) };
    if p.is_null() {
        return Err(FerriteError::Config(format!("dlsym({name}) not found")));
    }
    Ok(p)
}

macro_rules! f {
    ($h:expr, $name:expr) => {
        std::mem::transmute_copy(&sym($h, $name)?)
    };
}

fn dlopen_or_err(path: &str) -> Result<*mut c_void> {
    // RTLD_GLOBAL so the kernel .so's own dependencies resolve.
    let c = CString::new(path).map_err(|_| FerriteError::Config("bad library path".into()))?;
    let h = unsafe { libc_dlopen(c.as_ptr(), RTLD_NOW | RTLD_GLOBAL) };
    if h.is_null() {
        let e = unsafe { libc_dlerror() };
        let msg = if e.is_null() {
            "unknown".to_string()
        } else {
            unsafe { std::ffi::CStr::from_ptr(e) }.to_string_lossy().into_owned()
        };
        return Err(FerriteError::Config(format!("dlopen({path}) failed: {msg}")));
    }
    Ok(h)
}

// ------------------------------------------------------------------- DevBuf

/// VERSION GATE for the dlopen-only paths (DSV41 and any other `devrt`
/// consumer). `CudaBackend::verify_kernel_build` guards the GLM serve path;
/// this mirrors it for the paths that never construct a `CudaBackend`.
///
/// User rule (2026-09-11, after a session of invalid measurements): the .so and
/// the binary MUST come from the same build. `kernels/cuda/build.sh` stamps the
/// .so with `git HEAD + sha256(.cu)` through `ferrite_kernel_build_id()`, and
/// `build.rs` bakes the same string into this binary as `FERRITE_BUILD_ID`.
/// Any mismatch — or a .so predating the stamp — is a hard error: a mismatched
/// pair produces numbers that mean nothing.
///
/// Unlike the GLM path this cannot require a LINKED image: a `devrt` consumer
/// deliberately does not link `libferrite_kernels.so` (that is the point of the
/// dlopen design). So the linked-image belt-and-braces is skipped here and the
/// stamp + ABI + same-source comparisons do the work.
unsafe fn verify_kernel_build(handle: *mut c_void, so_path: &str) -> Result<()> {
    const EXPECTED_ABI: u32 = 1;
    let get_id = sym(handle, "ferrite_kernel_build_id").ok();
    let get_abi = sym(handle, "ferrite_kernel_abi_version").ok();
    let (get_id, get_abi) = match (get_id, get_abi) {
        (Some(a), Some(b)) => (a, b),
        _ => {
            return Err(FerriteError::Config(format!(
                "kernel version gate: {so_path} carries NO build stamp (it predates the gate). \
                 Rebuild both artifacts from one checkout: \
                 `cd kernels/cuda && bash build.sh 103a && cd ../.. && cargo build --release`."
            )))
        }
    };
    let id_fn: extern "C" fn() -> *const c_char = std::mem::transmute(get_id);
    let abi_fn: extern "C" fn() -> u32 = std::mem::transmute(get_abi);
    let so_abi = abi_fn();
    let so_id = std::ffi::CStr::from_ptr(id_fn()).to_string_lossy().to_string();
    let bin_id = env!("FERRITE_BUILD_ID");
    if so_abi != EXPECTED_ABI {
        return Err(FerriteError::Config(format!(
            "kernel ABI mismatch: {so_path} abi={so_abi}, this binary expects {EXPECTED_ABI}. \
             Rebuild both artifacts."
        )));
    }
    if so_id != bin_id {
        return Err(FerriteError::Config(format!(
            "kernel build-id mismatch — REFUSING TO START (so and binary must be the same build): \
             .so {so_path} build_id={so_id} vs binary build_id={bin_id}. \
             Rebuild both from the same checkout: \
             `cd kernels/cuda && bash build.sh 103a && cd ../.. && cargo build --release`."
        )));
    }
    // Belt and braces: more than one mapped copy of the kernel .so means symbol
    // resolution could bind to a different image than the one just checked.
    let maps = std::fs::read_to_string("/proc/self/maps").unwrap_or_default();
    let mut seen: Vec<&str> = Vec::new();
    for line in maps.lines() {
        if let Some(p) = line.split_whitespace().last() {
            if p.contains("libferrite_kernels.so") && !seen.contains(&p) {
                seen.push(p);
            }
        }
    }
    if seen.len() > 1 {
        return Err(FerriteError::Config(format!(
            "more than one libferrite_kernels.so is mapped into this process ({seen:?}) — \
             symbol resolution may bind to a different image than --lib/DSV41_KERNELS. \
             Use one tree only."
        )));
    }
    Ok(())
}

/// A plain device allocation. No pooling: the load-and-run path is
/// latency-tolerant, and skipping the pool keeps this layer free of any
/// interaction with the capture-sensitive activation allocator.
#[derive(Clone)]
pub struct DevBuf {
    pub ptr: *mut c_void,
    pub bytes: usize,
    owned: bool,
}

impl DevBuf {
    /// Borrow an existing pointer (e.g. a slice of a bigger allocation).
    pub fn view(ptr: *mut c_void, bytes: usize) -> DevBuf {
        DevBuf { ptr, bytes, owned: false }
    }
    #[inline]
    pub fn as_u8(&self) -> *const u8 {
        self.ptr as *const u8
    }
    #[inline]
    pub fn as_f32(&self) -> *const f32 {
        self.ptr as *const f32
    }
    #[inline]
    pub fn as_mut_f32(&self) -> *mut f32 {
        self.ptr as *mut f32
    }
    #[inline]
    pub fn as_i32(&self) -> *const i32 {
        self.ptr as *const i32
    }
    #[inline]
    pub fn as_mut_i32(&self) -> *mut i32 {
        self.ptr as *mut i32
    }
    /// Byte view at an offset, e.g. to alias a slice of a packed weight.
    #[inline]
    pub fn as_u8_at(&self, off: usize) -> *const u8 {
        (self.ptr as *const u8).wrapping_add(off)
    }
    #[inline]
    pub fn as_f32_at(&self, off: usize) -> *const f32 {
        (self.ptr as *const f32).wrapping_add(off / 4)
    }
    /// True when this buffer owns its allocation (`Device::free` frees it).
    #[inline]
    pub fn is_owned(&self) -> bool {
        self.owned
    }
}

impl Drop for DevBuf {
    fn drop(&mut self) {
        // The owning runtime frees; DevBuf itself holds no handle, so freeing
        // happens through `DevRuntime::free`.
    }
}

// -------------------------------------------------------------- DevRuntime

/// The pointer-level CUDA device runtime: dlopen'd cudart/cublas + the loaded
/// kernel `.so`, one stream, byte-granular allocation and the CUDA-graph
/// capture primitives. Shared by every model crate — model-specific kernel
/// *symbols* stay in the model, only the platform plumbing lives here.
pub struct DevRuntime {
    /// bytes this handle has allocated (for the OOM diagnostic)
    allocated: std::cell::Cell<usize>,
    cudart: Cudart,
    cublas: Cublas,
    stream: CuStream,
    /// Side stream for the hc tail split (DSV41_HC_TAIL_SPLIT). Created once, only
    /// ever fed the LATE tail half, and joined back onto `stream` before hc_post.
    /// Null when `cudaStreamCreate` is unavailable, in which case the model keeps
    /// the single-launch path (`side_stream()` returns null).
    side_stream: CuStream,
    /// Priority the side stream was created with (0 = device default). Kept so
    /// the graph instantiation can decide whether the node-priority flag is
    /// worth passing, and so a log can state what was actually in effect.
    side_prio: c_int,
    /// Flags for `cudaGraphInstantiate`. Non-zero only when the side stream is a
    /// priority stream AND `DSV41_GRAPH_NODE_PRIORITY` is not 0.
    graph_instantiate_flags: u64,
    /// Fork/join events for the tail split, created with `cudaEventDisableTiming`
    /// so they are legal inside a stream capture. Recorded by the kernel launcher,
    /// waited by the model. Reused for every tail call: each record/wait pair is
    /// fully ordered within the capture, so program order disambiguates them.
    fork_ev: *mut c_void,
    join_ev: *mut c_void,
    handle: *mut c_void,
    /// the kernel `.so` — model crates resolve their own symbols in it
    kernel_handle: *mut c_void,
    debug_sync: bool,
    _cudart_handle: *mut c_void,
    _libs: (*mut c_void, *mut c_void, *mut c_void),
}

impl DevRuntime {
    /// `kernel_so` is the path to `libferrite_kernels.so` (every model's
    /// kernels are linked into that one object).
    pub fn open(kernel_so: &str) -> Result<DevRuntime> {
        let h_cudart = dlopen_or_err("libcudart.so")?;
        let h_cublas = dlopen_or_err("libcublas.so")?;
        let h_k = dlopen_or_err(kernel_so)
            .map_err(|e| FerriteError::Config(format!("{e} — run kernels/cuda/build.sh first")))?;
        // VERSION GATE (user rule 2026-09-11: so and binary MUST be the same
        // build). The DSV41 path dlopens its kernels through THIS function, not
        // through CudaBackend, so without this check a stale .so silently runs
        // beside a fresh binary (or vice versa) and every measurement is
        // meaningless. build.sh stamps the .so with `git HEAD + .cu hash` and
        // build.rs embeds the same string in this binary.
        unsafe { verify_kernel_build(h_k, kernel_so)? };
        unsafe {
            let cudart = Cudart {
                malloc: f!(h_cudart, "cudaMalloc"),
                free: f!(h_cudart, "cudaFree"),
                memcpy: f!(h_cudart, "cudaMemcpy"),
                memcpy_async: sym(h_cudart, "cudaMemcpyAsync")
                    .ok()
                    .map(|p| std::mem::transmute_copy(&p)),
                memset: f!(h_cudart, "cudaMemset"),
                memset_async: sym(h_cudart, "cudaMemsetAsync")
                    .ok()
                    .map(|p| std::mem::transmute_copy(&p)),
                stream_create: f!(h_cudart, "cudaStreamCreate"),
                stream_create_priority: sym(h_cudart, "cudaStreamCreateWithPriority")
                    .ok()
                    .map(|p| std::mem::transmute_copy(&p)),
                stream_priority_range: sym(h_cudart, "cudaDeviceGetStreamPriorityRange")
                    .ok()
                    .map(|p| std::mem::transmute_copy(&p)),
                stream_sync: f!(h_cudart, "cudaStreamSynchronize"),
                dev_sync: f!(h_cudart, "cudaDeviceSynchronize"),
                last_error: f!(h_cudart, "cudaGetLastError"),
                strerror: f!(h_cudart, "cudaGetErrorString"),
                set_device: f!(h_cudart, "cudaSetDevice"),
                get_device: f!(h_cudart, "cudaGetDevice"),
                device_count: f!(h_cudart, "cudaGetDeviceCount"),
                enable_peer: f!(h_cudart, "cudaDeviceEnablePeerAccess"),
                memcpy_peer_async: f!(h_cudart, "cudaMemcpyPeerAsync"),
                memcpy_2d_async: f!(h_cudart, "cudaMemcpy2DAsync"),
                memcpy_2d: f!(h_cudart, "cudaMemcpy2D"),
                mem_info: f!(h_cudart, "cudaMemGetInfo"),
                memcpy_peer: f!(h_cudart, "cudaMemcpyPeer"),
                stream_begin_capture: sym(h_cudart, "cudaStreamBeginCapture")
                    .ok()
                    .map(|p| std::mem::transmute_copy(&p)),
                stream_end_capture: sym(h_cudart, "cudaStreamEndCapture")
                    .ok()
                    .map(|p| std::mem::transmute_copy(&p)),
                graph_instantiate: sym(h_cudart, "cudaGraphInstantiate")
                    .ok()
                    .map(|p| std::mem::transmute_copy(&p)),
                graph_launch: sym(h_cudart, "cudaGraphLaunch")
                    .ok()
                    .map(|p| std::mem::transmute_copy(&p)),
                graph_exec_destroy: sym(h_cudart, "cudaGraphExecDestroy")
                    .ok()
                    .map(|p| std::mem::transmute_copy(&p)),
                graph_destroy: sym(h_cudart, "cudaGraphDestroy")
                    .ok()
                    .map(|p| std::mem::transmute_copy(&p)),
                event_create_flags: sym(h_cudart, "cudaEventCreateWithFlags")
                    .ok()
                    .map(|p| std::mem::transmute_copy(&p)),
                event_record: sym(h_cudart, "cudaEventRecord")
                    .ok()
                    .map(|p| std::mem::transmute_copy(&p)),
                event_destroy: sym(h_cudart, "cudaEventDestroy")
                    .ok()
                    .map(|p| std::mem::transmute_copy(&p)),
                stream_wait_event: sym(h_cudart, "cudaStreamWaitEvent")
                    .ok()
                    .map(|p| std::mem::transmute_copy(&p)),
            };
            let cublas = Cublas {
                create: f!(h_cublas, "cublasCreate_v2"),
                set_stream: f!(h_cublas, "cublasSetStream_v2"),
                gemm_ex: f!(h_cublas, "cublasGemmEx"),
            };

            let mut stream: CuStream = std::ptr::null_mut();
            check_cudart((cudart.stream_create)(&mut stream), &cudart, "cudaStreamCreate")?;
            let mut handle: *mut c_void = std::ptr::null_mut();
            let st = (cublas.create)(&mut handle);
            if st != 0 {
                return Err(FerriteError::Config(format!("cublasCreate: {st}")));
            }
            let st = (cublas.set_stream)(handle, stream);
            if st != 0 {
                return Err(FerriteError::Config(format!("cublasSetStream: {st}")));
            }
            // Side stream + fork/join events for the hc tail split. BEST EFFORT:
            // any failure (or a cudart without the symbols) leaves them null and
            // the model keeps the single-launch tail, so the GLM path and every
            // other consumer are unaffected. The events MUST be disable-timing:
            // a timing event makes cudaStreamEndCapture fail once the fork/join
            // lands inside the whole-step graph.
            let mut side_stream: CuStream = std::ptr::null_mut();
            // Priority of the side stream (0 = default). Non-zero means the graph
            // capture stamped its kernel nodes with this priority, and the graph
            // MUST be instantiated with the node-priority flag for that to mean
            // anything at replay time.
            let mut side_prio: c_int = 0;
            // `DSV41_HC_TAIL_PRIO=0` pins the side stream back to the default
            // priority (A/B knob: the scheduling hint is the only thing it changes).
            let prio_wanted = std::env::var("DSV41_HC_TAIL_PRIO")
                .map(|v| v != "0")
                .unwrap_or(true);
            if prio_wanted {
                if let (Some(create_prio), Some(prio_range)) =
                    (cudart.stream_create_priority, cudart.stream_priority_range)
                {
                    // greatest < least: the API returns a NEGATIVE number for the
                    // greatest priority and, on a device without priority support,
                    // both are 0 - in which case there is nothing to gain.
                    let (mut least, mut greatest) = (0 as c_int, 0 as c_int);
                    if prio_range(&mut least, &mut greatest) == 0 && greatest < least {
                        if create_prio(&mut side_stream, CUDA_STREAM_NON_BLOCKING, greatest) != 0 {
                            let _ = (cudart.last_error)();
                            side_stream = std::ptr::null_mut();
                        } else {
                            side_prio = greatest;
                        }
                    }
                }
            }
            if side_stream.is_null() {
                // clear any sticky error from the priority attempt before retrying
                let _ = (cudart.last_error)();
                if (cudart.stream_create)(&mut side_stream) != 0 {
                    let _ = (cudart.last_error)();   // clear the sticky flag
                    side_stream = std::ptr::null_mut();
                }
            }
            if side_stream.is_null() {
                eprintln!("[hc_tail] side stream unavailable — tail split stays single-launch");
            } else if side_prio != 0 {
                // (this line matters: it is the only evidence in a log that the
                // priority actually reached the capture)
                eprintln!("[hc_tail] side stream priority = {side_prio} (greatest)");
            }
            // Node-priority instantiation is what makes the captured priority
            // effective at replay. `DSV41_GRAPH_NODE_PRIORITY=0` pins it off.
            let node_prio = side_prio != 0
                && std::env::var("DSV41_GRAPH_NODE_PRIORITY")
                    .map(|v| v != "0")
                    .unwrap_or(true);
            let graph_instantiate_flags: u64 = if node_prio {
                CUDA_GRAPH_INSTANTIATE_USE_NODE_PRIORITY
            } else {
                0
            };
            let mut fork_ev: *mut c_void = std::ptr::null_mut();
            let mut join_ev: *mut c_void = std::ptr::null_mut();
            if let Some(make_ev) = cudart.event_create_flags {
                if make_ev(&mut fork_ev, CUDA_EVENT_DISABLE_TIMING) != 0 {
                    let _ = (cudart.last_error)();
                    fork_ev = std::ptr::null_mut();
                }
                if make_ev(&mut join_ev, CUDA_EVENT_DISABLE_TIMING) != 0 {
                    let _ = (cudart.last_error)();
                    join_ev = std::ptr::null_mut();
                }
                if (fork_ev.is_null() || join_ev.is_null()) && !fork_ev.is_null() {
                    if let Some(d) = cudart.event_destroy {
                        let _ = d(fork_ev);
                        fork_ev = std::ptr::null_mut();
                    }
                }
            }
            Ok(DevRuntime {
                cudart,
                cublas,
                stream,
                side_stream,
                side_prio,
                graph_instantiate_flags,
                fork_ev,
                join_ev,
                handle,
                kernel_handle: h_k,
                debug_sync: std::env::var("FERRITE_DEBUG_SYNC")
                    .or_else(|_| std::env::var("DSV41_DEBUG_SYNC"))
                    .map(|v| v != "0")
                    .unwrap_or(false),
                allocated: std::cell::Cell::new(0),
                _cudart_handle: h_cudart,
                _libs: (h_cudart, h_cublas, h_k),
            })
        }
    }

    /// This runtime's stream (model kernel launches submit here).
    pub fn stream(&self) -> CuStream {
        self.stream
    }

    /// The side stream the hc tail split feeds its LATE half (null when the
    /// runtime could not create one). Callers must gate on `is_null`.
    pub fn side_stream(&self) -> CuStream {
        self.side_stream
    }

    /// The priority the side stream was created with (0 = device default).
    /// Informational: the model does not branch on it, but a run log / a probe
    /// needs it to tell "the hint was never installed" apart from "the driver
    /// ignored it".
    pub fn side_stream_priority(&self) -> c_int {
        self.side_prio
    }

    /// Fork event for the hc tail split: recorded on the main stream by the
    /// kernel launcher, waited on the side stream. Null when unavailable.
    pub fn fork_event(&self) -> *mut c_void {
        self.fork_ev
    }

    /// Join event for the hc tail split: recorded on the side stream by the
    /// kernel launcher, waited on the main stream before hc_post. Null when
    /// unavailable.
    pub fn join_event(&self) -> *mut c_void {
        self.join_ev
    }

    /// Make `stream` wait for `ev`. Legal inside a capture (becomes a graph
    /// dependency edge); this is the join half of the tail split.
    pub fn stream_wait_event(&self, stream: CuStream, ev: *mut c_void) -> Result<()> {
        let f = self
            .cudart
            .stream_wait_event
            .ok_or_else(|| FerriteError::Config("cudaStreamWaitEvent missing".into()))?;
        let rc = unsafe { f(stream, ev, 0) };
        self.kerr(rc, "cudaStreamWaitEvent")
    }

    /// Resolve a kernel symbol in the loaded `.so` (required).
    pub fn kernel_sym(&self, name: &str) -> Result<*mut c_void> {
        sym(self.kernel_handle, name)
    }

    /// Resolve an optional kernel symbol (staged bring-up: absent = None).
    pub fn kernel_sym_opt(&self, name: &str) -> Option<*mut c_void> {
        sym(self.kernel_handle, name).ok()
    }

    /// This context's device ordinal.
    pub fn device_id(&self) -> i32 {
        let mut d: c_int = -1;
        unsafe {
            (self.cudart.get_device)(&mut d);
        }
        d
    }

    pub fn device_count(&self) -> i32 {
        let mut n: c_int = 0;
        unsafe {
            (self.cudart.device_count)(&mut n);
        }
        n
    }

    /// Bind this thread to `gpu` and let it reach every other device. Device
    /// selection in the CUDA runtime is per-thread, which is what lets one
    /// process drive several devices — one rank per thread — without juggling
    /// contexts by hand.
    pub fn bind_to(gpu: i32) -> Result<()> {
        // ONLY bind the device. Enabling peer access here is actively harmful:
        // the peers' contexts are being created concurrently by the other rank
        // threads, so the calls cannot succeed and racing context creation can
        // poison this context — after which every cudaMalloc fails as a bogus
        // "out of memory". Peer access is enabled once, after every rank has a
        // context (enable_peer_access, called past the load barrier).
        let d = Self::open_dummy_cudart()?;
        let rc = unsafe { (d.set_device)(gpu) };
        if rc != 0 {
            return Err(FerriteError::Config(format!("cudaSetDevice({gpu}): {rc}")));
        }
        Ok(())
    }

    fn open_dummy_cudart() -> Result<Cudart> {
        let h = dlopen_or_err("libcudart.so")?;
        unsafe {
            Ok(Cudart {
                malloc: f!(h, "cudaMalloc"),
                free: f!(h, "cudaFree"),
                memcpy: f!(h, "cudaMemcpy"),
                memcpy_async: None,
                memset: f!(h, "cudaMemset"),
                memset_async: None,
                stream_create: f!(h, "cudaStreamCreate"),
                stream_create_priority: None,
                stream_priority_range: None,
                stream_sync: f!(h, "cudaStreamSynchronize"),
                dev_sync: f!(h, "cudaDeviceSynchronize"),
                last_error: f!(h, "cudaGetLastError"),
                strerror: f!(h, "cudaGetErrorString"),
                set_device: f!(h, "cudaSetDevice"),
                get_device: f!(h, "cudaGetDevice"),
                device_count: f!(h, "cudaGetDeviceCount"),
                enable_peer: f!(h, "cudaDeviceEnablePeerAccess"),
                memcpy_peer_async: f!(h, "cudaMemcpyPeerAsync"),
                memcpy_2d_async: f!(h, "cudaMemcpy2DAsync"),
                memcpy_2d: f!(h, "cudaMemcpy2D"),
                mem_info: f!(h, "cudaMemGetInfo"),
                memcpy_peer: f!(h, "cudaMemcpyPeer"),
                stream_begin_capture: None,
                stream_end_capture: None,
                graph_instantiate: None,
                graph_launch: None,
                graph_exec_destroy: None,
                graph_destroy: None,
                event_create_flags: None,
                event_record: None,
                event_destroy: None,
                stream_wait_event: None,
            })
        }
    }

    /// Enable peer access to every other device.
    ///
    /// This cannot happen at bind time: `cudaDeviceEnablePeerAccess` needs the
    /// *peer's* context to exist, and at bind time no rank has opened a device
    /// yet, so the call fails silently and the later peer copy faults. Call it
    /// once every rank has created its context.
    pub fn enable_peer_access(&self) -> Result<usize> {
        let n = self.device_count();
        let me = self.device_id();
        let mut enabled = 0usize;
        for p in 0..n {
            if p == me {
                continue;
            }
            let rc = unsafe { (self.cudart.enable_peer)(p, 0) };
            // 704 = already enabled; both are fine, clear the sticky error
            unsafe {
                (self.cudart.last_error)();
            }
            if rc == 0 || rc == 704 {
                enabled += 1;
            }
        }
        Ok(enabled)
    }

    /// Copy `bytes` from this device to `dst_dev` over the GPU interconnect
    /// (NVLink/PCIe peer DMA). The copy is issued on this rank's stream, so it
    /// never passes through host memory; the caller synchronises before reading
    /// the result.
    pub fn memcpy_peer(
        &self,
        dst_dev: i32,
        dst: *mut c_void,
        src: *const c_void,
        bytes: usize,
    ) -> Result<()> {
        let st = unsafe {
            (self.cudart.memcpy_peer_async)(dst, dst_dev, src, self.device_id(), bytes, self.stream)
        };
        check_cudart(st, &self.cudart, "cudaMemcpyPeerAsync")
    }

    pub fn sync(&self) -> Result<()> {
        check_cudart(unsafe { (self.cudart.stream_sync)(self.stream) }, &self.cudart, "sync")
    }

    /// Device-wide synchronisation (the all-reduce must know that its peer
    /// copies have landed before summing them).
    pub fn dev_sync(&self) -> Result<()> {
        check_cudart(unsafe { (self.cudart.dev_sync)() }, &self.cudart, "cudaDeviceSynchronize")
    }

    pub fn alloc(&self, bytes: usize) -> Result<DevBuf> {
        let mut p: *mut c_void = std::ptr::null_mut();
        let st = unsafe { (self.cudart.malloc)(&mut p, bytes.max(1)) };
        if st != 0 {
            // name the failure: how big was the request, what has this rank
            // already taken, and what does the device still have
            let free = self.mem_free();
            return Err(FerriteError::Config(format!(
                "cudaMalloc({:.1} MiB) failed on device {}, free={:.1} MiB \n  \
                 (this rank has already allocated {:.1} GiB)",
                bytes as f64 / (1u64 << 20) as f64,
                self.device_id(),
                free as f64 / (1u64 << 20) as f64,
                self.allocated.get() as f64 / (1u64 << 30) as f64
            )));
        }
        self.allocated.set(self.allocated.get() + bytes);
        Ok(DevBuf { ptr: p, bytes, owned: true })
    }

    /// Free device memory in MiB at this instant (`cudaMemGetInfo`).
    pub fn mem_free(&self) -> usize {
        let mut f: usize = 0;
        let mut t: usize = 0;
        unsafe {
            (self.cudart.mem_info)(&mut f, &mut t);
        }
        f
    }

    pub fn upload(&self, bytes: &[u8]) -> Result<DevBuf> {
        let b = self.alloc(bytes.len())?;
        if !bytes.is_empty() {
            let st = unsafe {
                (self.cudart.memcpy)(
                    b.ptr,
                    bytes.as_ptr() as *const c_void,
                    bytes.len(),
                    CUDA_MEMCPY_H2D,
                )
            };
            check_cudart(st, &self.cudart, "cudaMemcpy H2D")?;
        }
        Ok(b)
    }

    pub fn upload_f32(&self, v: &[f32]) -> Result<DevBuf> {
        let bytes =
            unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) };
        self.upload(bytes)
    }

    pub fn upload_f32_at(&self, dst: *mut c_void, off: usize, v: &[f32]) -> Result<()> {
        let bytes =
            unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) };
        let st = unsafe {
            (self.cudart.memcpy)(
                (dst as *mut u8).add(off) as *mut c_void,
                bytes.as_ptr() as *const c_void,
                bytes.len(),
                CUDA_MEMCPY_H2D,
            )
        };
        check_cudart(st, &self.cudart, "cudaMemcpy H2D (at)")
    }

    pub fn download_f32(&self, src: &DevBuf, out: &mut [f32]) -> Result<()> {
        let bytes = out.len() * 4;
        if bytes > src.bytes {
            return Err(FerriteError::Config(format!("download {bytes} > buffer {}", src.bytes)));
        }
        let st = unsafe {
            (self.cudart.memcpy)(out.as_mut_ptr() as *mut c_void, src.ptr, bytes, CUDA_MEMCPY_D2H)
        };
        check_cudart(st, &self.cudart, "cudaMemcpy D2H")
    }

    /// Raw H2D upload into an existing device buffer at offset 0.
    pub fn upload_bytes_at(&self, dst: &DevBuf, bytes: &[u8]) -> Result<()> {
        let n = bytes.len().min(dst.bytes);
        let st = unsafe {
            (self.cudart.memcpy)(dst.ptr, bytes.as_ptr() as *const c_void, n, CUDA_MEMCPY_H2D)
        };
        check_cudart(st, &self.cudart, "cudaMemcpy H2D (bytes)")
    }

    pub fn download_u8(&self, src: &DevBuf, out: &mut [u8]) -> Result<()> {
        if out.len() > src.bytes {
            return Err(FerriteError::Config(format!(
                "download {} > buffer {}",
                out.len(),
                src.bytes
            )));
        }
        let st = unsafe {
            (self.cudart.memcpy)(out.as_mut_ptr() as *mut c_void, src.ptr, out.len(), CUDA_MEMCPY_D2H)
        };
        check_cudart(st, &self.cudart, "cudaMemcpy D2H (bytes)")
    }

    /// DMA upload straight from mapped host memory (no intermediate buffer):
    /// the path the weights take, mirroring mmap + cudaMemcpy.
    pub fn upload_from(&self, dst: *mut c_void, src: *const c_void, bytes: usize) -> Result<()> {
        let st = unsafe { (self.cudart.memcpy)(dst, src, bytes, CUDA_MEMCPY_H2D) };
        check_cudart(st, &self.cudart, "cudaMemcpy H2D (mapped)")
    }

    /// Strided DMA upload: `height` rows of `width` bytes, each row `spitch`
    /// apart in the source and `dpitch` apart in the destination. This is how a
    /// column-sliced tensor (row-parallel weights) is transferred without
    /// staging it through the CPU.
    pub fn upload_from_2d(
        &self,
        dst: *mut c_void,
        dpitch: usize,
        src: *const c_void,
        spitch: usize,
        width: usize,
        height: usize,
    ) -> Result<()> {
        // SYNCHRONOUS: the async form returns before the copy runs, so a fault
        // inside it only surfaces at some later call — where it reappears as a
        // misleading "out of memory" on the next cudaMalloc (a sticky context
        // error). Synchronous reports the real failure, with the real sizes.
        let st = unsafe {
            (self.cudart.memcpy_2d)(dst, dpitch, src, spitch, width, height, CUDA_MEMCPY_H2D)
        };
        check_cudart(st, &self.cudart, "cudaMemcpy2D H2D")
    }

    /// Zero `bytes` at `ptr` (lay the padding down before the DMA overwrites
    /// the real part, so no host-side assembly is needed).
    pub fn zero_at(&self, ptr: *mut c_void, bytes: usize) -> Result<()> {
        // MUST be the async variant on our own stream: the synchronous cudaMemset
        // runs on the legacy stream (0), and CUDA forbids a capturing stream from
        // depending on it — "operation would make the legacy stream depend on a
        // capturing blocking stream" is exactly the error the MoE segment graph
        // hit. Stream-ordered is also the correct semantics here anyway.
        if let Some(f) = self.cudart.memset_async {
            let st = unsafe { f(ptr, 0, bytes, self.stream) };
            return check_cudart(st, &self.cudart, "cudaMemsetAsync");
        }
        let st = unsafe { (self.cudart.memset)(ptr, 0, bytes) };
        check_cudart(st, &self.cudart, "cudaMemset(pad)")
    }

    pub fn zero(&self, b: &DevBuf) -> Result<()> {
        self.zero_at(b.ptr, b.bytes)
    }

    /// Device-to-device copy (small row moves: the KV ring append).
    pub fn memcpy_d2d(&self, dst: *mut c_void, src: *const c_void, bytes: usize) -> Result<()> {
        // Async on OUR stream — a synchronous cudaMemcpy runs on the legacy stream
        // and is illegal while capturable work is being recorded.
        if let Some(f) = self.cudart.memcpy_async {
            let st = unsafe { f(dst, src, bytes, CUDA_MEMCPY_D2D, self.stream) };
            return check_cudart(st, &self.cudart, "cudaMemcpyAsync D2D");
        }
        let st = unsafe { (self.cudart.memcpy)(dst, src, bytes, CUDA_MEMCPY_D2D) };
        check_cudart(st, &self.cudart, "cudaMemcpy D2D")
    }

    /// The single 4-byte host read per decode step (EOS check + printing).
    pub fn download_u32(&self, ptr: *const c_void) -> Result<u32> {
        let mut b = [0u8; 4];
        let st =
            unsafe { (self.cudart.memcpy)(b.as_mut_ptr() as *mut c_void, ptr, 4, CUDA_MEMCPY_D2H) };
        check_cudart(st, &self.cudart, "cudaMemcpy D2H token")?;
        Ok(u32::from_le_bytes(b))
    }

    pub fn free(&self, b: &DevBuf) {
        if b.owned && !b.ptr.is_null() {
            let rc = unsafe { (self.cudart.free)(b.ptr) };
            if rc != 0 {
                // report the failure: an unchecked failed free leaves the context
                // poisoned and shows up later as a bogus "out of memory"
                let msg = unsafe {
                    let p = (self.cudart.strerror)(rc);
                    std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
                };
                eprintln!("[ferrite] cudaFree({:p}) failed: {msg}", b.ptr);
            }
        }
    }

    /// Launch-status check. `debug_sync` (FERRITE_DEBUG_SYNC=1, legacy
    /// DSV41_DEBUG_SYNC=1) synchronises after every launch so an ASYNC fault
    /// (illegal address inside a kernel) is attributed to the kernel that
    /// caused it instead of surfacing at the next sync.
    pub fn kerr(&self, rc: c_int, what: &str) -> Result<()> {
        if rc != 0 {
            return Err(FerriteError::Config(format!("{what}: cuda error {rc}")));
        }
        check_cudart(unsafe { (self.cudart.last_error)() }, &self.cudart, what)?;
        if self.debug_sync {
            let rc = unsafe { (self.cudart.stream_sync)(self.stream) };
            if rc != 0 {
                let msg = unsafe {
                    let p = (self.cudart.strerror)(rc);
                    if p.is_null() {
                        format!("cuda error {rc}")
                    } else {
                        std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
                    }
                };
                return Err(FerriteError::Config(format!("ASYNC FAULT in {what}: {msg}")));
            }
        }
        Ok(())
    }

    // -------------------------------------------------- cuBLAS dense GEMMs

    /// f32 x f32 -> f32 GEMM (the release runs some projections in fp32).
    pub fn gemm_f32(
        &self,
        x: *const c_void,
        w: *const c_void,
        out: *mut f32,
        rows: i32,
        n_out: i32,
        k: i32,
    ) -> Result<()> {
        self.gemm_ex(x, CUDA_R_32F, w, CUDA_R_32F, out, rows, n_out, k)
            .map_err(|e| FerriteError::Config(format!("cublasGemmEx(f32): {e}")))
    }

    /// bf16 x bf16 -> f32 GEMM for the weights that are natively bf16 (the
    /// head, the compressor and indexer projections, the vision tower). This
    /// is not a dequantisation: those tensors are bf16 in the checkpoint.
    ///
    /// Layout: `w` is `[n_out, k]`, `x` is `[rows, k]`, `out` is
    /// `[rows, n_out]` (row-major), computed with cuBLAS in the same
    /// transposed form GLM uses.
    pub fn gemm_bf16(
        &self,
        x: *const c_void,
        w: *const c_void,
        out: *mut f32,
        rows: i32,
        n_out: i32,
        k: i32,
    ) -> Result<()> {
        self.gemm_ex(x, CUDA_R_16BF, w, CUDA_R_16BF, out, rows, n_out, k)?;
        // cuBLAS has no cudaGetLastError-style check here, so a fault inside the
        // GEMM would otherwise surface at the NEXT kernel's sync and be
        // attributed to the wrong op (the routing kernel was blamed for an
        // earlier cuBLAS fault).
        if self.debug_sync {
            let rc = unsafe { (self.cudart.stream_sync)(self.stream) };
            if rc != 0 {
                let msg = unsafe {
                    let p = (self.cudart.strerror)(rc);
                    if p.is_null() {
                        format!("cuda error {rc}")
                    } else {
                        std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
                    }
                };
                return Err(FerriteError::Config(format!(
                    "ASYNC FAULT in gemm (cuBLAS): {msg}"
                )));
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn gemm_ex(
        &self,
        x: *const c_void,
        xt: c_int,
        w: *const c_void,
        wt: c_int,
        out: *mut f32,
        rows: i32,
        n_out: i32,
        k: i32,
    ) -> Result<()> {
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        let st = unsafe {
            (self.cublas.gemm_ex)(
                self.handle,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                n_out,
                rows,
                k,
                &alpha as *const f32 as *const c_void,
                w,
                wt,
                k,
                x,
                xt,
                k,
                &beta as *const f32 as *const c_void,
                out as *mut c_void,
                CUDA_R_32F,
                n_out,
                CUDA_R_32F,
                99,
            )
        };
        if st != 0 {
            return Err(FerriteError::Config(format!("cublasGemmEx: {st}")));
        }
        Ok(())
    }

    // -------------------------------------------- CUDA graph capture

    /// Begin capturing work queued on this device's stream.
    ///
    /// `mode` is the cudart capture mode. Default callers pass
    /// `cudaStreamCaptureModeRelaxed` (2), NOT Global: Global invalidates the
    /// capture on ANY CUDA activity in ANY thread of the context, and TP8 runs
    /// one rank thread per device with its own all-reduce work, so the other
    /// ranks' traffic invalidated every capture
    /// (cudaErrorStreamCaptureUnjoined, 901). Relaxed only rejects the
    /// capturing thread's own illegal calls.
    pub fn capture_begin(&self) -> Result<()> {
        let f = self
            .cudart
            .stream_begin_capture
            .ok_or_else(|| FerriteError::Config("cudaStreamBeginCapture missing".into()))?;
        let rc = unsafe { f(self.stream, 2 /* cudaStreamCaptureModeRelaxed */) };
        self.kerr(rc, "cudaStreamBeginCapture")
    }

    /// End the capture and return the graph handle.
    pub fn capture_end(&self) -> Result<*mut c_void> {
        let f = self
            .cudart
            .stream_end_capture
            .ok_or_else(|| FerriteError::Config("cudaStreamEndCapture missing".into()))?;
        let mut g: *mut c_void = std::ptr::null_mut();
        let rc = unsafe { f(self.stream, &mut g) };
        self.kerr(rc, "cudaStreamEndCapture")?;
        Ok(g)
    }

    /// Instantiate a captured graph into an executable.
    ///
    /// Passes [`CUDA_GRAPH_INSTANTIATE_USE_NODE_PRIORITY`] when the runtime owns a
    /// priority side stream: stream capture copies each stream's priority onto its
    /// kernel nodes, but a replay only honours those per-node priorities when the
    /// graph is instantiated with this flag — otherwise every node runs at the
    /// LAUNCH stream's priority and the hc tail split's scheduling hint is inert.
    /// Any rejection (an older cudart, a flag the driver does not know) falls back
    /// to the plain instantiation: the flag is a hint, never a correctness input.
    pub fn graph_instantiate(&self, g: *mut c_void) -> Result<*mut c_void> {
        let f = self
            .cudart
            .graph_instantiate
            .ok_or_else(|| FerriteError::Config("cudaGraphInstantiate missing".into()))?;
        let mut e: *mut c_void = std::ptr::null_mut();
        let mut rc = unsafe { f(&mut e, g, self.graph_instantiate_flags) };
        if rc != 0 && self.graph_instantiate_flags != 0 {
            let _ = unsafe { (self.cudart.last_error)() };   // clear, do not report
            e = std::ptr::null_mut();
            eprintln!(
                "[hc_tail] cudaGraphInstantiate rejected the node-priority flag \
                 (rc={rc}) — replaying at the launch stream's priority"
            );
            rc = unsafe { f(&mut e, g, 0) };
        }
        self.kerr(rc, "cudaGraphInstantiate")?;
        Ok(e)
    }

    pub fn graph_launch(&self, e: *mut c_void) -> Result<()> {
        let f = self
            .cudart
            .graph_launch
            .ok_or_else(|| FerriteError::Config("cudaGraphLaunch missing".into()))?;
        let rc = unsafe { f(e, self.stream) };
        self.kerr(rc, "cudaGraphLaunch")
    }

    pub fn graph_free(&self, g: *mut c_void, e: *mut c_void) -> Result<()> {
        if !e.is_null() {
            if let Some(f) = self.cudart.graph_exec_destroy {
                let rc = unsafe { f(e) };
                self.kerr(rc, "cudaGraphExecDestroy")?;
            }
        }
        if !g.is_null() {
            if let Some(f) = self.cudart.graph_destroy {
                let rc = unsafe { f(g) };
                self.kerr(rc, "cudaGraphDestroy")?;
            }
        }
        Ok(())
    }
}

fn check_cudart(rc: c_int, c: &Cudart, what: &str) -> Result<()> {
    if rc == 0 {
        return Ok(());
    }
    let msg = unsafe {
        let p = (c.strerror)(rc);
        if p.is_null() {
            format!("cuda error {rc}")
        } else {
            std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
        }
    };
    Err(FerriteError::Config(format!("{what}: {msg}")))
}
