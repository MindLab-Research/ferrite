//! CudaBackend — FFI bridge to libferrite_kernels.so (sm_100a).
//!
//! Enabled with `--features cuda`. The .so is produced by
//! `kernels/cuda/build.sh` (nvcc; compiling needs no GPU).
//!
//! v1 contract: host-tensor in → cudaMemcpy H2D → kernel → cudaMemcpy D2H
//! → host-tensor out. This is the *correctness* path for the B300 golden
//! harness (diff against CpuBackend); the performance path (device-
//! resident tensors + CUDA graph replay via `cuStreamBeginCapture`) keeps
//! the same extern contract — see the GraphCapable section.
//!
//! Numerical parity target: f32 tolerance 1e-5 vs the CPU backend.

#![cfg(feature = "cuda")]

use std::ffi::CString;
use std::sync::Arc;

use ferrite_types::{DType, FerriteError, Result, Shape, Tensor};

/// Opaque stream handle (cudaStream_t == void* at the ABI level).
pub type CuStream = *mut std::ffi::c_void;

extern "C" {
    // cudart (linked into libferrite_kernels.so's dependency closure)
    fn cudaSetDevice(dev: i32) -> i32;
    fn cudaMalloc(ptr: *mut *mut std::ffi::c_void, size: usize) -> i32;
    fn cudaFree(ptr: *mut std::ffi::c_void) -> i32;
    fn cudaMemcpy(dst: *mut std::ffi::c_void, src: *const std::ffi::c_void, count: usize, kind: i32) -> i32;
    fn cudaStreamCreate(stream: *mut CuStream) -> i32;
    fn cudaStreamSynchronize(stream: CuStream) -> i32;
    fn cudaGetErrorString(err: i32) -> *const std::os::raw::c_char;

    // ferrite kernels (ferrite_kernels.cu bridge)
    fn ferrite_matmul(x: *const f32, w: *const f32, bias: *const f32, out: *mut f32,
                      n: i32, in_f: i32, out_f: i32, s: CuStream) -> i32;
    fn ferrite_matmul_bf16(x: *const f32, w: *const std::ffi::c_void,
                            bias: *const f32, out: *mut f32,
                            n: i32, in_f: i32, out_f: i32, s: CuStream) -> i32;
    fn ferrite_gemv_bf16(x: *const f32, w: *const std::ffi::c_void,
                          bias: *const f32, out: *mut f32,
                          in_f: i32, out_f: i32, s: CuStream) -> i32;
    fn ferrite_gemv_bf16_v2(x: *const f32, w: *const std::ffi::c_void,
                             bias: *const f32, out: *mut f32,
                             in_f: i32, out_f: i32, s: CuStream) -> i32;
    fn ferrite_layernorm_affine(x: *const f32, w: *const f32, b: *const f32,
                                 out: *mut f32, n: i32, dim: i32, s: CuStream) -> i32;
    fn ferrite_dsa_cache_append(kvb: *const f32, ki: *const f32, gate: *const f32,
                                 k_nope: *mut f32, v: *mut f32, k_idx: *mut f32, k_gate: *mut f32,
                                 t0_ptr: *const i32, n: i32, h: i32, dk: i32, dv: i32, idm: i32,
                                 s: CuStream) -> i32;
    fn ferrite_kpool_compress(k_idx: *const f32, k_gate: *const f32, ape: *const f32,
                               pool_keys: *mut f32, total_ptr: *const i32, npools: i32, kpool: i32,
                               idm: i32, s: CuStream) -> i32;
    fn ferrite_pool_expand(idx_pools: *const f32, idx: *mut f32,
                            n: i32, select_k: i32, kpool: i32, max_npools: i32,
                            total_ptr: *const i32, n_fixed: i32,
                            s: CuStream) -> i32;
    fn ferrite_scale_inplace(x: *mut f32, s: f32, n: i32, st: CuStream) -> i32;
    fn ferrite_graph_begin(s: CuStream) -> i32;
    fn ferrite_event_create(ev: *mut *mut std::ffi::c_void) -> i32;
    fn ferrite_event_record(ev: *mut std::ffi::c_void, s: CuStream) -> i32;
    fn ferrite_event_elapsed(a: *mut std::ffi::c_void, b: *mut std::ffi::c_void, ms: *mut f32) -> i32;
    fn ferrite_event_destroy(ev: *mut std::ffi::c_void) -> i32;
    fn ferrite_graph_end(s: CuStream, g: *mut *mut std::ffi::c_void) -> i32;
    fn ferrite_graph_instantiate(e: *mut *mut std::ffi::c_void, g: *mut std::ffi::c_void) -> i32;
    fn ferrite_graph_launch(e: *mut std::ffi::c_void, s: CuStream) -> i32;
    fn ferrite_f32_to_bf16(in_: *const f32, out: *mut std::ffi::c_void,
                            n: i64, s: CuStream) -> i32;
    fn ferrite_rmsnorm(x: *const f32, w: *const f32, out: *mut f32,
                       n: i32, dim: i32, eps: f32, s: CuStream) -> i32;
    fn ferrite_hc_contract(x: *const f32, out: *mut f32,
                           s: i32, n: i32, h: i32, stream: CuStream) -> i32;
    fn ferrite_gemv5_bf16(x: *const f32, w1: *const std::ffi::c_void, w2: *const std::ffi::c_void,
                          w3: *const std::ffi::c_void, w4: *const std::ffi::c_void, w5: *const std::ffi::c_void,
                          o1: *mut f32, o2: *mut f32, o3: *mut f32, o4: *mut f32, o5: *mut f32,
                          in_f: i32, of1: i32, of2: i32, of3: i32, of4: i32, of5: i32,
                          stream: CuStream) -> i32;
    fn ferrite_gated_rmsnorm(x: *const f32, gate: *const f32, w: *const f32, out: *mut f32,
                             n: i32, dim: i32, eps: f32, s: CuStream) -> i32;
    fn ferrite_swiglu(gu: *const f32, out: *mut f32, n: i32, inter: i32,
                      limit: f32, s: CuStream) -> i32;
    fn ferrite_swiglu2(gate: *const f32, up: *const f32, out: *mut f32,
                       n: i32, inter: i32, limit: f32, s: CuStream) -> i32;
    fn ferrite_causal_conv1d(x: *const f32, w: *const f32, state_in: *const f32,
                             out: *mut f32, state_out: *mut f32,
                             n: i32, ch: i32, conv: i32, s: CuStream) -> i32;
    fn ferrite_gdn_chunk(q: *const f32, k: *const f32, v: *const f32,
                         beta: *const f32, gate: *const f32, a_log: *const f32,
                         state: *mut f32, out: *mut f32,
                         n: i32, h: i32, dk: i32, dv: i32, s: CuStream) -> i32;
    fn ferrite_gdn_chunk_v2(q: *const f32, k: *const f32, v: *const f32,
                            beta: *const f32, gate: *const f32, a_log: *const f32,
                            state: *mut f32, out: *mut f32,
                            n: i32, h: i32, dk: i32, dv: i32, s: CuStream) -> i32;
    fn ferrite_gdn_chunk_wyf(q: *const f32, k: *const f32, v: *const f32,
                             beta: *const f32, gate: *const f32, a_log: *const f32,
                             state_in: *mut f32, out: *mut f32, state_out: *mut f32,
                             n: i32, h: i32, dk: i32, dv: i32, s: CuStream) -> i32;
    fn ferrite_moe_route(logits: *const f32, bias: *const f32, probs: *mut f32, ids: *mut f32,
                         n: i32, e: i32, topk: i32,
                         scale: f32, s: CuStream) -> i32;
    fn ferrite_indexer_topk(qi: *const f32, ki: *const f32, w: *const f32, idx: *mut f32,
                            n: i32, h: i32, d: i32, topk: i32,
                            total_ptr: *const i32, kpool_val: i32, n_fixed: i32, s: CuStream) -> i32;
    fn ferrite_sparse_attn(q: *const f32, k: *const f32, v: *const f32, idx: *const f32,
                           out: *mut f32, n: i32, t_ptr: *const i32, h: i32, d: i32, dv: i32,
                           topk: i32, s: CuStream) -> i32;
    fn ferrite_sparse_attn_v2(q: *const f32, k: *const f32, v: *const f32, idx: *const f32,
                              out: *mut f32, n: i32, t_ptr: *const i32, h: i32, d: i32, dv: i32,
                              topk: i32, s: CuStream) -> i32;
    fn ferrite_argmax(logits: *const f32, out: *mut f32, n: i32, dim: i32, s: CuStream) -> i32;
    fn ferrite_softmax(logits: *const f32, out: *mut f32, n: i32, dim: i32, s: CuStream) -> i32;
    fn ferrite_hc_pre(res: *const f32, fw: *const f32, scale: *const f32, base: *const f32,
                      li: *mut f32, post: *mut f32, comb: *mut f32,
                      s: i32, n: i32, h: i32, mix: i32,
                      rms_eps: f32, hc_eps: f32, iters: i32, stream: CuStream) -> i32;
    fn ferrite_hc_pre_split(res: *const f32, fw: *const f32, scale: *const f32, base: *const f32,
                             li: *mut f32, post: *mut f32, comb: *mut f32, mx_scratch: *mut f32,
                             s: i32, n: i32, h: i32, mix: i32,
                             rms_eps: f32, hc_eps: f32, iters: i32, stream: CuStream) -> i32;
    fn ferrite_hc_post(x: *const f32, res: *const f32, post: *const f32, comb: *const f32,
                       out: *mut f32, s: i32, n: i32, h: i32, stream: CuStream) -> i32;
    fn ferrite_gdn_prep(conv_out: *const f32, b_raw: *const f32, fb: *const f32,
                        dt_bias: *const f32, a_log: *const f32,
                        q: *mut f32, k: *mut f32, v: *mut f32, beta: *mut f32, gate: *mut f32,
                        n: i32, h: i32, dk: i32, lb: f32, stream: CuStream) -> i32;
    fn cudaMemset(ptr: *mut std::ffi::c_void, val: i32, bytes: usize) -> i32;
}

const CUDA_MEMCPY_H2D: i32 = 1;
const CUDA_MEMCPY_D2H: i32 = 2;
const CUDA_MEMCPY_D2D: i32 = 3;
extern "C" {
    fn cudaMemcpyAsync(dst: *mut std::ffi::c_void, src: *const std::ffi::c_void,
                        count: usize, kind: i32, stream: CuStream) -> i32;
    fn cudaMemsetAsync(ptr: *mut std::ffi::c_void, val: i32, count: usize, stream: CuStream) -> i32;
}

fn ck(err: i32, what: &str) -> Result<()> {
    if err == 0 {
        Ok(())
    } else {
        let msg = unsafe {
            let p = cudaGetErrorString(err);
            if p.is_null() {
                "unknown".into()
            } else {
                std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
            }
        };
        Err(FerriteError::InvalidArg(format!("CUDA {what}: {msg} (err {err})")))
    }
}

// ============================================================
// Activation buffer pool — per-device, size-class bucketed.
// Per-op cudaMalloc/cudaFree are device-synchronising calls that dominate the
// op-latency budget (tens of thousands of ops per token across a TP cluster);
// pooled reuse removes them entirely after warmup.
//
// GLOBAL (Mutex<HashMap>) — NOT thread-local: fan_out spawns fresh threads
// per layer (90 spawns × 4 ranks per token); a thread-local pool was EMPTY
// in every worker, so every DevBuf::alloc paid cudaMalloc+cudaMallocHost
// (~70μs each, ~14k allocs/token ≈ 1s of pure allocation per token) and
// the buffers LEAKED when the thread exited (pool dropped, never freed).
// The global pool also makes CUDA graph capture possible (cudaMallocHost
// during capture is illegal — with a warm global pool, capture allocates
// nothing).
//
// CUDA-graph capture support: every DevBuf owns a PINNED host staging buffer
// (cudaMallocHost, allocated with the device buffer, pooled with it).
// upload/download go through it — cudaMemcpyAsync from pageable memory is
// ILLEGAL during stream capture (cudaErrorStreamCaptureUnsupported) and the
// tensor's Vec address changes every call, which would bake a stale pointer
// into the graph. The pinned stage is the fixed-address rendezvous: the CPU
// writes it (outside the graph), the recorded memcpy moves stage→device.
// ============================================================
static BUF_POOL: std::sync::OnceLock<
    std::sync::Mutex<
        std::collections::HashMap<(i32, u32), Vec<PoolPtrs>>,
    >,
> = std::sync::OnceLock::new();

/// Raw device/pinned-stage pointer pair — Send+Sync because the pool's
/// Mutex serialises all take/release, and CUDA device pointers are
/// process-global (not thread-bound).
#[derive(Clone, Copy)]
struct PoolPtrs(*mut std::ffi::c_void, *mut std::ffi::c_void);
unsafe impl Send for PoolPtrs {}
unsafe impl Sync for PoolPtrs {}

fn pool() -> &'static std::sync::Mutex<
    std::collections::HashMap<(i32, u32), Vec<PoolPtrs>>,
> {
    BUF_POOL.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

/// True while THIS thread is inside a stream capture. THREAD-LOCAL: the
/// per-layer graphs capture inside fan_out workers — 4 ranks capture
/// concurrently and each ends independently; a GLOBAL flag would let the
/// first finisher re-enable sync for the others mid-capture (segfault).
/// Download skips its synchronisation while capturing — the graph's
/// end/replay syncs once at the tail.
thread_local! {
    static CAPTURING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

pub(crate) fn set_capturing(v: bool) {
    CAPTURING.with(|c| c.set(v));
}

fn is_capturing() -> bool {
    CAPTURING.with(|c| c.get())
}

fn buf_pool_release(dev: i32, class: u32, ptr: *mut std::ffi::c_void, stage: *mut std::ffi::c_void) {
    pool().lock().unwrap().entry((dev, class)).or_default().push(PoolPtrs(ptr, stage));
}

fn buf_pool_take(dev: i32, class: u32) -> Option<(*mut std::ffi::c_void, *mut std::ffi::c_void)> {
    pool().lock().unwrap().get_mut(&(dev, class)).and_then(|v| v.pop()).map(|p| (p.0, p.1))
}

/// Drop all pooled activation buffers (weights are owned by the weight cache).
/// Called from CudaBackend::Drop; leaks of freed devices are reclaimed by CUDA
/// context teardown at exit.
pub fn clear_activation_pool() {
    let mut p = pool().lock().unwrap();
    for ((dev, _), ptrs) in p.drain() {
        unsafe { cudaSetDevice(dev) };
        for pp in ptrs {
            unsafe { cudaFree(pp.0) };
            if !pp.1.is_null() {
                unsafe { cudaFreeHost(pp.1) };
            }
        }
    }
}

extern "C" {
    fn cudaMallocHost(ptr: *mut *mut std::ffi::c_void, bytes: usize) -> i32;
    fn cudaFreeHost(ptr: *mut std::ffi::c_void) -> i32;
}

/// A device buffer (pooled) with its pinned host stage. `len` is the
/// requested length; `class` the size class (next power of two ≥ len).
///
/// PUBLIC: the device-resident op-chain phase (whole-layer DevBuf pipelines
/// feeding a single CUDA graph) composes ops at this level — matmul_dev and
/// friends take/return DevBuf so activations never cross the bus inside a
/// layer.
pub struct DevBuf {
    pub ptr: *mut std::ffi::c_void,
    pub len: usize,
    pub class: u32,
    pub dev: i32,
    pub stream: CuStream,
    /// Pinned host staging (cudaMallocHost) — the fixed-address rendezvous
    /// for graph-capturable H2D/D2H (see the module comment above).
    pub stage: *mut std::ffi::c_void,
}

impl DevBuf {
    /// Pooled alloc: reuse a released (device, stage) pair of the same size
    /// class when available, else cudaMalloc + cudaMallocHost. The caller
    /// must have `enter()`ed the backend's device.
    pub fn alloc(dev: i32, stream: CuStream, len: usize) -> Result<Self> {
        let class = (len.max(1) as u32).next_power_of_two();
        if let Some((ptr, stage)) = buf_pool_take(dev, class) {
            return Ok(DevBuf { ptr, len, class, dev, stream, stage });
        }
        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        ck(unsafe { cudaMalloc(&mut ptr, class as usize * std::mem::size_of::<f32>()) }, "pooled malloc")?;
        let mut stage: *mut std::ffi::c_void = std::ptr::null_mut();
        ck(unsafe { cudaMallocHost(&mut stage, class as usize * std::mem::size_of::<f32>()) }, "pinned stage malloc")?;
        Ok(DevBuf { ptr, len, class, dev, stream, stage })
    }
    /// H2D via the pinned stage — graph-capturable: the CPU copy into the
    /// stage happens outside any graph; the recorded memcpy moves
    /// stage→device at fixed addresses on both ends.
    pub fn upload(&self, host: &[f32]) -> Result<()> {
        assert!(host.len() <= self.len);
        unsafe {
            std::ptr::copy_nonoverlapping(host.as_ptr(), self.stage as *mut f32, host.len());
        }
        ck(unsafe {
            cudaMemcpyAsync(self.ptr, self.stage, host.len() * 4, CUDA_MEMCPY_H2D, self.stream)
        }, "memcpy H2D")
    }
    /// D2H via the pinned stage; synchronises the stream (the op tail) —
    /// EXCEPT during capture, when sync is illegal and the graph's
    /// end_verify does the single tail sync instead.
    pub fn download(&self, host: &mut [f32]) -> Result<()> {
        assert!(host.len() <= self.len);
        ck(unsafe {
            cudaMemcpyAsync(self.stage, self.ptr, host.len() * 4, CUDA_MEMCPY_D2H, self.stream)
        }, "memcpy D2H")?;
        if !is_capturing() {
            ck(unsafe { cudaStreamSynchronize(self.stream) }, "sync after D2H")?;
        }
        unsafe {
            std::ptr::copy_nonoverlapping(self.stage as *const f32, host.as_mut_ptr(), host.len());
        }
        Ok(())
    }
    pub fn as_f32(&self) -> *mut f32 {
        self.ptr as *mut f32
    }
    pub fn as_const_f32(&self) -> *const f32 {
        self.ptr as *const f32
    }
}

impl Drop for DevBuf {
    /// Return the (device, stage) pair to the pool instead of freeing.
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            buf_pool_release(self.dev, self.class, self.ptr, self.stage);
        }
    }
}

/// Cached device-resident weight: keyed by the host Arc buffer pointer
/// (stable because the tensor is immutable), the host Arc is kept alive in
/// the cache so the pointer can never dangle.
struct CachedBuf {
    keep: Arc<Vec<f32>>,
    dev: *mut std::ffi::c_void,
    len: usize,
}

unsafe impl Send for CachedBuf {}
unsafe impl Sync for CachedBuf {}

/// A borrowed device pointer (no ownership — the cache frees it).
#[derive(Clone, Copy)]
struct DevRef {
    ptr: *mut std::ffi::c_void,
    len: usize,
}

impl DevRef {
    pub fn as_const_f32(&self) -> *const f32 {
        self.ptr as *const f32
    }
}

/// CUDA backend (v1: host↔device per op for activations; *weights* are
/// device-resident via the pointer-keyed cache — repeated uploads of the
/// same Arc'd weight tensor hit the cache, which is also the precondition
/// for CUDA-graph capture (stable device pointers across replays)).
pub struct CudaBackend {
    stream: CuStream,
    /// Device index this backend is bound to. cudaSetDevice is THREAD-LOCAL:
    /// a TP cluster drives N backends from one thread, so every op must
    /// re-bind before allocating/launching (buffers must live on the same
    /// device as the stream).
    dev: i32,
    weights: std::sync::Mutex<std::collections::HashMap<(usize, usize), CachedBuf>>,
    /// CUDA graph capture state (driver-API handle for the instantiated
    /// graph exec; see the GraphCapable impl below).
    graph: std::sync::Mutex<GraphState>,
    /// Device-resident recurrent states, keyed (seq, layer) — GDN
    /// [h,dk,dk] state and conv tails. NOT pooled (must persist across
    /// tokens; pooled buffers would be reused by other ops).
    gdn_states: std::sync::Mutex<std::collections::HashMap<(u64, usize), DeviceState>>,
    conv_states: std::sync::Mutex<std::collections::HashMap<(u64, usize), DeviceState>>,
    /// DSA caches: device-resident k_nope/v/k_idx/k_gate per (seq, family),
    /// pre-allocated to max tokens. The CPU path grew host Vecs and cloned
    /// them per call (~MBs memcpy per DSA layer per token).
    dsa_caches: std::sync::Mutex<std::collections::HashMap<(u64, usize), DsaCacheState>>,
    /// MoE expert POINTER TABLES (fused GPU dispatch): per layer, three
    /// device buffers of e_local raw pointers (gate/up/down) into the
    /// dev_weight_bf16 cache — the fused kernels gather the selected
    /// experts' rows through them with zero host round-trips. Keyed by the
    /// first expert's gate tensor pointer (stable per layer).
    moe_ptrs: std::sync::Mutex<std::collections::HashMap<usize, MoePtrTable>>,
    /// Named CUDA graphs (per layer-segment): FERRITE_GRAPH_LAYER captures
    /// each segment's op sequence once and replays per token — the per-op
    /// launch gaps (~30μs × ~19 ops/layer) are the decode bottleneck after
    /// the device chains.
    graph_execs: std::sync::Mutex<std::collections::HashMap<String, usize>>,
    /// Fixed IO pointers of captured segment graphs (per name).
    graph_io: std::sync::Mutex<std::collections::HashMap<String, GraphIO>>,
}

// cudaStream_t is thread-safe (CUDA runtime serialises ops on a stream);
// the raw pointer is just an opaque handle.
unsafe impl Send for CudaBackend {}
unsafe impl Sync for CudaBackend {}

impl Drop for CudaBackend {
    fn drop(&mut self) {
        self.clear_weight_cache();
        for store in [self.gdn_states.get_mut().unwrap(), self.conv_states.get_mut().unwrap()] {
            for (_, st) in store.drain() {
                unsafe { cudaFree(st.ptr) };
            }
        }
    }
}

impl Default for CudaBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl CudaBackend {
    /// Loads libferrite_kernels.so's dependency closure (cudart) and
    /// creates a stream. Call after the .so is in the loader path.
    pub fn new() -> Self {
        // The extern symbols resolve from libcudart which is linked by
        // libferrite_kernels.so; loading that .so first is the caller's
        // job (see `with_library`).
        let mut stream: CuStream = std::ptr::null_mut();
        let e = unsafe { cudaStreamCreate(&mut stream) };
        if e != 0 {
            panic!("cudaStreamCreate failed: {e} (is libferrite_kernels.so loaded? see CudaBackend::with_library)");
        }
        CudaBackend {
            stream,
            dev: 0,
            weights: std::sync::Mutex::new(std::collections::HashMap::new()),
            graph: std::sync::Mutex::new(GraphState::default()),
            gdn_states: std::sync::Mutex::new(std::collections::HashMap::new()),
            conv_states: std::sync::Mutex::new(std::collections::HashMap::new()),
            dsa_caches: std::sync::Mutex::new(std::collections::HashMap::new()),
            moe_ptrs: std::sync::Mutex::new(std::collections::HashMap::new()),
            graph_execs: std::sync::Mutex::new(std::collections::HashMap::new()),
            graph_io: std::sync::Mutex::new(std::collections::HashMap::new()),
        }
    }

    /// Bind this backend's device as the calling thread's current device.
    /// cudaSetDevice is thread-local state; TP ranks all call ops from the
    /// main thread, so each op entry re-binds before cudaMalloc/launch.
    /// PUBLIC: device-chain call sites (attn_shard) allocate DevBufs
    /// BEFORE the first op — cudaMalloc binds to the CALLING thread's
    /// current device, which in a fan_out thread is another rank's.
    #[inline]
    pub fn enter(&self) {
        unsafe {
            cudaSetDevice(self.dev);
        }
    }

    /// Upload a weight tensor to the device ONCE — subsequent calls with
    /// the same host Arc buffer hit the cache (pointer + length keyed; the
    /// Arc is kept alive inside the cache so the key can never dangle).
    /// This kills the per-op weight H2D of the naive path and is the
    /// precondition for CUDA-graph capture (stable device pointers).
    fn dev_weight(&self, t: &Tensor) -> Result<DevRef> {
        let key = (t.as_slice().as_ptr() as usize, t.numel());
        let mut cache = self.weights.lock().unwrap();
        if let Some(cb) = cache.get(&key) {
            if cb.len == t.numel() {
                return Ok(DevRef { ptr: cb.dev, len: cb.len });
            }
        }
        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        ck(unsafe { cudaMalloc(&mut ptr, t.numel() * 4) }, "weight malloc")?;
        ck(
            unsafe { cudaMemcpy(ptr, t.as_slice().as_ptr() as *const _, t.numel() * 4, CUDA_MEMCPY_H2D) },
            "weight H2D",
        )?;
        cache.insert(key, CachedBuf { keep: t.data.clone(), dev: ptr, len: t.numel() });
        Ok(DevRef { ptr, len: t.numel() })
    }

    /// Upload a weight tensor to the device ONCE in **bf16** — the resident
    /// layout for large matmul weights. A 285GB/TP4-rank f32 shard does not
    /// fit a 275GB B300; bf16 halves it to 142GB (TileRT's resident-weights
    /// model). The kernel converts bf16→f32 in registers; x/out stay f32.
    ///
    /// Large weights (≥8M elements = 32MB f32) convert ON THE GPU: the f32
    /// source is streamed to a scratch buffer in chunks and a kernel packs
    /// bf16 in place — CPU-side packing of 292GB/rank was the warmup
    /// bottleneck (~150s/thread). Small weights pack on the CPU (the
    /// bit-shift loop is vector-friendly). Both paths use identical
    /// truncation semantics (f32 high bits), so parity holds.
    fn dev_weight_bf16(&self, t: &Tensor) -> Result<DevRef> {
        let key = (t.as_slice().as_ptr() as usize, t.numel() << 1 | 1);
        let mut cache = self.weights.lock().unwrap();
        if let Some(cb) = cache.get(&key) {
            if cb.len == t.numel() {
                return Ok(DevRef { ptr: cb.dev, len: cb.len });
            }
        }
        let n = t.numel();
        let src = t.as_slice();
        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        ck(unsafe { cudaMalloc(&mut ptr, n * 2) }, "weight bf16 malloc")?;
        if n >= 8 << 20 {
            // GPU-side conversion: stream f32 chunks to a scratch buffer and
            // pack bf16 in place into the resident allocation. **Each chunk's
            // H2D must wait for the previous kernel** — blocking cudaMemcpy
            // does NOT order against stream-queued kernels (the next chunk's
            // H2D overwrites the scratch while the previous kernel is still
            // reading it — corrupted weights → NaN downstream; this exact
            // race is why the equivalence tests (small weights, CPU pack
            // path) all passed while serve (25M-element qkv_proj, 4 chunks)
            // produced all-NaN attention outputs).
            const CHUNK: usize = 32 << 20; // 32M f32 = 128MB per step
            let mut scratch: *mut std::ffi::c_void = std::ptr::null_mut();
            ck(unsafe { cudaMalloc(&mut scratch, CHUNK * 4) }, "bf16 scratch malloc")?;
            let conv = (|| -> Result<()> {
                for (i, chunk) in src.chunks(CHUNK).enumerate() {
                    self.sync()?; // previous kernel finished reading scratch
                    ck(
                        unsafe { cudaMemcpy(scratch, chunk.as_ptr() as *const _, chunk.len() * 4, CUDA_MEMCPY_H2D) },
                        "bf16 chunk H2D",
                    )?;
                    ck(
                        unsafe {
                            ferrite_f32_to_bf16(
                                scratch as *const f32,
                                (ptr as *mut u8).add(i * CHUNK * 2) as *mut _,
                                chunk.len() as i64,
                                self.stream,
                            )
                        },
                        "bf16 GPU convert",
                    )?;
                }
                self.sync()
            })();
            unsafe { cudaFree(scratch) };
            conv?;
        } else {
            // pack f32 → bf16 on the CPU (truncate to high bits; PyTorch
            // bf16 semantics — matches ferrite_f32_to_bf16 exactly)
            let mut packed: Vec<u16> = vec![0u16; src.len()];
            for (dst, v) in packed.iter_mut().zip(src.iter()) {
                *dst = (v.to_bits() >> 16) as u16;
            }
            ck(
                unsafe { cudaMemcpy(ptr, packed.as_ptr() as *const _, packed.len() * 2, CUDA_MEMCPY_H2D) },
                "weight bf16 H2D",
            )?;
        }
        cache.insert(key, CachedBuf { keep: t.data.clone(), dev: ptr, len: n });
        Ok(DevRef { ptr, len: n })
    }

    /// Preload a weight into device-resident storage (bf16 for 2-D matmul
    /// weights — run_matmul reads bf16 exclusively; 1-D tensors stay f32
    /// for the elementwise kernels). Serve calls this over every shard
    /// weight at startup so inference never uploads weights again (the
    /// TileRT model: weights resident, only activations cross the bus).
    pub fn preload_weight(&self, t: &Tensor) -> Result<()> {
        if t.numel() == 0 {
            return Ok(()); // TP shard placeholder (empty expert slice)
        }
        // cudaSetDevice is THREAD-LOCAL: a TP cluster drives N backends from
        // one thread, so every entry point must re-bind before cudaMalloc —
        // without this, ALL ranks' weights malloc onto whatever device the
        // thread last touched (observed: 4 ranks' 142GB each piling onto
        // GPU 7 at 247GB).
        self.enter();
        if t.shape.0.len() >= 2 {
            self.dev_weight_bf16(t).map(|_| ())
        } else {
            self.dev_weight(t).map(|_| ())
        }
    }

    /// Number of cached (device-resident) weights.
    pub fn cached_weights(&self) -> usize {
        self.weights.lock().unwrap().len()
    }

    /// Free all cached device weights (explicit; the Drop impl does it too).
    pub fn clear_weight_cache(&self) {
        let mut cache = self.weights.lock().unwrap();
        for (_, cb) in cache.drain() {
            unsafe { cudaFree(cb.dev) };
        }
    }

    /// Load `libferrite_kernels.so` (and its cudart dependency) explicitly,
    /// binding the backend to CUDA device `device` (cudaSetDevice). Each rank
    /// of a TP deployment constructs one backend per GPU.
    pub fn with_device(so_path: &str, device: i32) -> Result<Self> {
        let c = CString::new(so_path).map_err(|_| FerriteError::InvalidArg("bad path".into()))?;
        let handle = unsafe { libc_dlopen(c.as_ptr(), 2) };
        if handle.is_null() {
            return Err(FerriteError::InvalidArg(format!(
                "dlopen({so_path}) failed — run kernels/cuda/build.sh first"
            )));
        }
        let err = unsafe { cudaSetDevice(device) };
        if err != 0 {
            return Err(FerriteError::InvalidArg(format!(
                "cudaSetDevice({device}) failed: {err}"
            )));
        }
        let mut b = Self::new();
        b.dev = device;
        Ok(b)
    }

    /// Load `libferrite_kernels.so` (and its cudart dependency) explicitly.
    pub fn with_library(so_path: &str) -> Result<Self> {
        let c = CString::new(so_path).map_err(|_| FerriteError::InvalidArg("bad path".into()))?;
        let handle = unsafe { libc_dlopen(c.as_ptr(), 2) };
        if handle.is_null() {
            return Err(FerriteError::InvalidArg(format!(
                "dlopen({so_path}) failed — run kernels/cuda/build.sh first"
            )));
        }
        Ok(Self::new())
    }

    /// Synchronise this backend's stream (public: tests and callers of the
    /// device-chain APIs need a barrier before wall-clock timings).
    pub fn sync(&self) -> Result<()> {
        ck(unsafe { cudaStreamSynchronize(self.stream) }, "sync")
    }

    /// Device ordinal this backend is bound to (for DevBuf::alloc at the
    /// device-chain call sites).
    pub fn dev(&self) -> i32 {
        self.dev
    }

    /// This backend's CUDA stream (for NCCL comm init / external enqueue).
    pub fn stream_handle(&self) -> CuStream {
        self.stream
    }

    /// This backend's stream (device-chain ops submit here; the graph
    /// capture/replay uses it too).
    pub fn stream(&self) -> CuStream {
        self.stream
    }

    /// Device-resident matmul: x already on device, w uploaded here (the
    /// BufferCache will dedupe repeated weight uploads), result stays on
    /// device. Building block for fused op chains (expert FFN).
    /// Weights are resident in bf16 (dev_weight_bf16).
    pub fn matmul_dev(&self, x_dev: &DevBuf, w: &Tensor, n: i32, in_f: i32, out_f: i32) -> Result<DevBuf> {
        let dw = self.dev_weight_bf16(w)?;
        let do_ = DevBuf::alloc(self.dev, self.stream, n as usize * out_f as usize)?;
        let dbias: *const f32 = std::ptr::null();
        if n == 1 {
            // Decode GEMV v2: uint4 vectorized + K-split WPR — 2.09x over v1
            // (bench gemv_v2_bench: 3.11→6.80TB/s lm_head, 2.20→3.91 o_proj,
            // all shapes 1.45-2.18x, maxd 3.8e-6). v1 kept for A/B.
            ck(unsafe {
                ferrite_gemv_bf16_v2(x_dev.as_const_f32(), dw.ptr as *const _,
                                      dbias, do_.as_f32(), in_f, out_f, self.stream)
            }, "gemv_dev")?;
        } else {
            ck(unsafe {
                ferrite_matmul_bf16(x_dev.as_const_f32(), dw.ptr as *const _,
                                     dbias, do_.as_f32(), n, in_f, out_f, self.stream)
            }, "matmul_dev")?;
        }
        Ok(do_)
    }

    /// GEMV v2 (vectorized uint4 + K-split): the decode weight-streaming
    /// upgrade of matmul_dev's n==1 path — uint4 (8 bf16) loads + WPR
    /// warps/row K-split to cover HBM latency on medium matrices.
    /// Benchmarked 2.2-3.1 TB/s (v1) → target 6+ TB/s. A/B via gemv_v2_bench.
    pub fn gemv_v2_dev(&self, x_dev: &DevBuf, w: &Tensor, n: i32, in_f: i32, out_f: i32) -> Result<DevBuf> {
        let dw = self.dev_weight_bf16(w)?;
        let do_ = DevBuf::alloc(self.dev, self.stream, n as usize * out_f as usize)?;
        ck(unsafe {
            ferrite_gemv_bf16_v2(x_dev.as_const_f32(), dw.ptr as *const _,
                                 std::ptr::null(), do_.as_f32(), in_f, out_f, self.stream)
        }, "gemv_v2_dev")?;
        Ok(do_)
    }

    /// sparse_attn v2 (256-thread block, float4 dots, smem idx/bitmap dedup):
    /// the dsa attention core — v1 ran block=32 (ONE warp) over topk≈8K
    /// slots with serial scalar dots + O(topk²) global idx rereads. Parity
    /// tested against the CPU reference (dedup first-wins, padding, softmax)
    /// in gpu_smoke::sparse_attn_v2_parity.
    pub fn sparse_mla_attn_v2(&self, q: &Tensor, k_nope: &Tensor, v: &Tensor, idx: &Tensor, out: &mut Tensor) -> Result<()> {
        self.enter();
        let n = q.shape.0[0] as i32;
        let t = k_nope.shape.0[0] as i32;
        let h = q.shape.0[1] as i32;
        let d = *q.shape.0.last().unwrap() as i32;
        let dv = *v.shape.0.last().unwrap() as i32;
        let topk = *idx.shape.0.last().unwrap() as i32;
        let dq = DevBuf::alloc(self.dev, self.stream, q.numel())?; dq.upload(q.as_slice())?;
        let dk = DevBuf::alloc(self.dev, self.stream, k_nope.numel())?; dk.upload(k_nope.as_slice())?;
        let dv_ = DevBuf::alloc(self.dev, self.stream, v.numel())?; dv_.upload(v.as_slice())?;
        let di = DevBuf::alloc(self.dev, self.stream, idx.numel())?; di.upload(idx.as_slice())?;
        let do_ = DevBuf::alloc(self.dev, self.stream, out.numel())?;
        let t_ptr = &t as *const i32;
        ck(unsafe { ferrite_sparse_attn_v2(dq.as_const_f32(), dk.as_const_f32(), dv_.as_const_f32(), di.as_const_f32(), do_.as_f32(), n, t_ptr, h, d, dv, topk, self.stream) }, "sparse_attn_v2")?;
        let ov = Arc::get_mut(&mut out.data).expect("unique out");
        do_.download(ov)?;
        Ok(())
    }

    /// gdn_step v2 parity hook (kernel-level): runs the v2 recurrent core
    /// (state staged in smem, padded stride) and returns out + new state.
    /// Golden is the sequential CPU recurrence in the gpu_smoke test.
    pub fn gdn_step_v2_dev(&self, q: &Tensor, k: &Tensor, v: &Tensor, beta: &Tensor,
                           gate: &Tensor, a_log: &Tensor, state_in: &Tensor, n: usize,
                           h: usize, dk: usize, dv: usize,
                           out: &mut Tensor, state_out: &mut Tensor) -> Result<()> {
        self.enter();
        let nn = n as i32;
        let dq = DevBuf::alloc(self.dev, self.stream, q.numel())?; dq.upload(q.as_slice())?;
        let dk_ = DevBuf::alloc(self.dev, self.stream, k.numel())?; dk_.upload(k.as_slice())?;
        let dv_ = DevBuf::alloc(self.dev, self.stream, v.numel())?; dv_.upload(v.as_slice())?;
        let db = DevBuf::alloc(self.dev, self.stream, beta.numel())?; db.upload(beta.as_slice())?;
        let dg = DevBuf::alloc(self.dev, self.stream, gate.numel())?; dg.upload(gate.as_slice())?;
        let dal = DevBuf::alloc(self.dev, self.stream, a_log.numel())?; dal.upload(a_log.as_slice())?;
        let dst = DevBuf::alloc(self.dev, self.stream, state_in.numel())?; dst.upload(state_in.as_slice())?;
        let do_ = DevBuf::alloc(self.dev, self.stream, out.numel())?;
        ck(unsafe { ferrite_gdn_chunk_v2(dq.as_const_f32(), dk_.as_const_f32(), dv_.as_const_f32(),
                                         db.as_const_f32(), dg.as_const_f32(), dal.as_const_f32(),
                                         dst.as_f32(), do_.as_f32(), nn, h as i32, dk as i32, dv as i32,
                                         self.stream) }, "gdn_chunk_v2")?;
        let ov = Arc::get_mut(&mut out.data).expect("unique out");
        do_.download(ov)?;
        let sv = Arc::get_mut(&mut state_out.data).expect("unique state_out");
        dst.download(sv)?;
        Ok(())
    }

    /// Fused SwiGLU on device: reads two independent matmul outputs.
    pub fn swiglu2_dev(&self, gate: &DevBuf, up: &DevBuf, n: i32, inter: i32, limit: f32) -> Result<DevBuf> {
        let out = DevBuf::alloc(self.dev, self.stream, n as usize * inter as usize)?;
        ck(unsafe {
            ferrite_swiglu2(gate.as_const_f32(), up.as_const_f32(), out.as_f32(), n, inter, limit, self.stream)
        }, "swiglu2")?;
        Ok(out)
    }

    fn run_matmul(&self, x: &Tensor, w: &Tensor, bias: Option<&Tensor>, out: &mut Tensor) -> Result<()> {
        let n = x.shape.0[0] as i32;
        let in_f = x.shape.0[1] as i32;
        let out_f = w.shape.0[0] as i32;
        let dx = DevBuf::alloc(self.dev, self.stream, x.numel())?; dx.upload(x.as_slice())?;
        // weights resident in bf16 (half the f32 footprint — the TP4 shard
        // does not fit a 275GB B300 in f32); kernel converts to f32 in registers
        let dw = self.dev_weight_bf16(w)?;
        let db = match bias {
            Some(b) => Some(self.dev_weight(b)?),
            None => None,
        };
        let do_ = DevBuf::alloc(self.dev, self.stream, out.numel())?;
        ck(unsafe {
            ferrite_matmul_bf16(dx.as_const_f32(), dw.ptr as *const _,
                                 db.as_ref().map_or(std::ptr::null(), |b| b.as_const_f32()),
                                 do_.as_f32(), n, in_f, out_f, self.stream)
        }, "matmul")?;
        let ov = Arc::get_mut(&mut out.data).expect("unique out");
        do_.download(ov)?;
        Ok(())
    }
}

extern "C" {
    #[link_name = "dlopen"]
    fn libc_dlopen(filename: *const std::os::raw::c_char, flags: i32) -> *mut std::ffi::c_void;
    #[link_name = "dlsym"]
    fn libc_dlsym(handle: *mut std::ffi::c_void, symbol: *const std::os::raw::c_char) -> *mut std::ffi::c_void;
}

// ============================================================
// CUDA graph capture — driver API via dlopen/dlsym (no link-time CUDA
// dependency; resolves libcuda.so.1 at first use).
// Contract mapping onto GraphCapable:
//   begin_capture → cuStreamBeginCapture(THREAD_LOCAL)
//   end_capture   → cuStreamEndCapture + cuGraphInstantiate (exec kept)
//   begin_verify  → cuGraphLaunch (replay into the SAME device buffers)
//   end_verify    → stream sync
// Precondition: replay is only correct when kernel argument pointers are
// stable — weights are (BufferCache device-resident); activations must be
// arena-allocated (engine-level, wired on the B300 validation harness).
// ============================================================
type FnStreamBeginCapture = unsafe extern "C" fn(*mut std::ffi::c_void, i32) -> i32;
type FnStreamEndCapture = unsafe extern "C" fn(*mut std::ffi::c_void, *mut *mut std::ffi::c_void) -> i32;
type FnGraphInstantiate = unsafe extern "C" fn(*mut *mut std::ffi::c_void, *mut std::ffi::c_void, u64) -> i32;
type FnGraphLaunch = unsafe extern "C" fn(*mut std::ffi::c_void, *mut std::ffi::c_void) -> i32;
type FnGraphDestroy = unsafe extern "C" fn(*mut std::ffi::c_void) -> i32;

#[allow(non_snake_case)]
struct DriverApi {
    cuStreamBeginCapture: FnStreamBeginCapture,
    cuStreamEndCapture: FnStreamEndCapture,
    cuGraphInstantiate: FnGraphInstantiate,
    cuGraphLaunch: FnGraphLaunch,
    cuGraphDestroy: FnGraphDestroy,
}

impl DriverApi {
    fn get() -> Option<&'static DriverApi> {
        static API: std::sync::OnceLock<Option<DriverApi>> = std::sync::OnceLock::new();
        API.get_or_init(|| {
            let name = c"libcuda.so.1";
            let h = unsafe { libc_dlopen(name.as_ptr(), 2) };
            if h.is_null() {
                // try without the .1
                let name2 = c"libcuda.so";
                let h2 = unsafe { libc_dlopen(name2.as_ptr(), 2) };
                if h2.is_null() {
                    return None;
                }
                return DriverApi::from_handle(h2);
            }
            DriverApi::from_handle(h)
        })
        .as_ref()
    }

    fn from_handle(h: *mut std::ffi::c_void) -> Option<DriverApi> {
        let sym = |name: &std::ffi::CStr| unsafe {
            let p = libc_dlsym(h, name.as_ptr());
            if p.is_null() { None } else { Some(p) }
        };
        let s_bc: &std::ffi::CStr = c"cuStreamBeginCapture";
        let s_ec: &std::ffi::CStr = c"cuStreamEndCapture";
        let s_gi: &std::ffi::CStr = c"cuGraphInstantiate";
        let s_gl: &std::ffi::CStr = c"cuGraphLaunch";
        let s_gd: &std::ffi::CStr = c"cuGraphDestroy";
        let bc = sym(s_bc)?;
        let ec = sym(s_ec)?;
        let gi = sym(s_gi)?;
        let gl = sym(s_gl)?;
        let gd = sym(s_gd)?;
        Some(DriverApi {
            cuStreamBeginCapture: unsafe { std::mem::transmute::<*mut std::ffi::c_void, FnStreamBeginCapture>(bc) },
            cuStreamEndCapture: unsafe { std::mem::transmute::<*mut std::ffi::c_void, FnStreamEndCapture>(ec) },
            cuGraphInstantiate: unsafe { std::mem::transmute::<*mut std::ffi::c_void, FnGraphInstantiate>(gi) },
            cuGraphLaunch: unsafe { std::mem::transmute::<*mut std::ffi::c_void, FnGraphLaunch>(gl) },
            cuGraphDestroy: unsafe { std::mem::transmute::<*mut std::ffi::c_void, FnGraphDestroy>(gd) },
        })
    }
}

/// Graph capture state for the CUDA backend.
#[derive(Default)]
struct GraphState {
    capturing: bool,
    graph_exec: Option<*mut std::ffi::c_void>,
}

impl CudaBackend {
    /// Named-graph replay: launch a previously captured graph on this
    /// backend's stream. Returns false if the name has no captured graph
    /// yet (caller should run+capture instead). One graph per (layer,
    /// segment, rank) — the per-op launch gaps (~30μs × ~19 ops/layer) are
    /// the decode bottleneck after the device chains.
    pub fn graph_replay(&self, name: &str) -> bool {
        let exec = self.graph_execs.lock().unwrap().get(name).copied();
        match exec {
            Some(exec) => {
                let r = unsafe { ferrite_graph_launch(exec as *mut std::ffi::c_void, self.stream) };
                r == 0
            }
            None => false,
        }
    }

    /// Begin stream capture (THREAD_LOCAL mode). Ops enqueued until
    /// graph_capture_end are RECORDED, not executed. The pool and weight
    /// caches must be warm (prefill does this) — cudaMalloc during capture
    /// is illegal.
    pub fn graph_capture_begin(&self) {
        // RUNTIME API wrappers (the driver-API dlopen path SIGSEGV'd inside
        // cuGraphInstantiate on worker-thread captures)
        let r = unsafe { ferrite_graph_begin(self.stream) };
        if r != 0 {
            panic!("cudaStreamBeginCapture failed: {r}");
        }
        set_capturing(true);
    }

    /// End capture, instantiate, store under `name`. The recorded ops did
    /// NOT execute — replay immediately if this pass's results are needed.
    pub fn graph_capture_end(&self, name: &str) {
        set_capturing(false);
        let mut graph: *mut std::ffi::c_void = std::ptr::null_mut();
        let r = unsafe { ferrite_graph_end(self.stream, &mut graph) };
        if r != 0 {
            panic!("cudaStreamEndCapture failed: {r}");
        }
        if graph.is_null() {
            panic!("cudaStreamEndCapture returned a NULL graph (capture invalidated?)");
        }
        let mut exec: *mut std::ffi::c_void = std::ptr::null_mut();
        let r = unsafe { ferrite_graph_instantiate(&mut exec, graph) };
        if r != 0 {
            panic!("cudaGraphInstantiate failed: {r}");
        }
        self.graph_execs.lock().unwrap().insert(name.to_string(), exec as usize);
    }

    // ---- event-in-graph timing (FERRITE_MEGA_EVTS) ----
    // Events recorded during capture become graph nodes; replay updates
    // them, and post-replay elapsed gives TRUE in-graph segment times
    // (the DRY sync-timing drains the queue → contaminated numbers).

    /// Create a timing-enabled event (opaque handle; destroy with
    /// event_destroy). Capture-safe: recording it inside graph_capture
    /// inserts an event-record node.
    pub fn event_create(&self) -> Result<*mut std::ffi::c_void> {
        let mut ev: *mut std::ffi::c_void = std::ptr::null_mut();
        ck(unsafe { ferrite_event_create(&mut ev) }, "event_create")?;
        Ok(ev)
    }
    pub fn event_destroy(&self, ev: *mut std::ffi::c_void) {
        let _ = unsafe { ferrite_event_destroy(ev) };
    }
    /// Record on this backend's stream. Inside a capture pass this
    /// becomes an event-record node in the graph.
    pub fn event_record(&self, ev: *mut std::ffi::c_void) {
        let r = unsafe { ferrite_event_record(ev, self.stream) };
        if r != 0 { panic!("cudaEventRecord failed: {r}"); }
    }
    /// elapsed ms from event a → b (both updated by the same replay).
    pub fn event_elapsed_ms(&self, a: *mut std::ffi::c_void, b: *mut std::ffi::c_void) -> f32 {
        let mut ms = 0f32;
        let r = unsafe { ferrite_event_elapsed(a, b, &mut ms) };
        if r != 0 { panic!("cudaEventElapsedTime failed: {r}"); }
        ms
    }
}

/// Mega-graph segment events (rank-0): created during the capture pass
/// (mega_chain_dev, FERRITE_MEGA_EVTS), read after each replay
/// (decode_step_mega) — [e_layer_start, e_attn_end, e_mid_end, e_ffn_end]
/// per layer × 45 + head pair. Handles stored as usize (raw pointers
/// are not Send; a static Mutex needs Send contents).
pub static MEGA_EVTS: std::sync::Mutex<Vec<usize>> = std::sync::Mutex::new(Vec::new());


impl crate::graph::GraphCapable for CudaBackend {
    fn begin_capture(&self) {
        let api = DriverApi::get().expect("libcuda not loadable (no GPU present?)");
        let r = unsafe { (api.cuStreamBeginCapture)(self.stream, 1) }; // 1 = THREAD_LOCAL
        if r != 0 {
            panic!("cuStreamBeginCapture failed: {r}");
        }
        set_capturing(true);
        let mut g = self.graph.lock().unwrap();
        g.capturing = true;
    }

    fn end_capture(&self) -> crate::graph::OpTrace {
        set_capturing(false);
        let api = DriverApi::get().expect("libcuda not loadable");
        let mut graph: *mut std::ffi::c_void = std::ptr::null_mut();
        let r = unsafe { (api.cuStreamEndCapture)(self.stream, &mut graph) };
        if r != 0 {
            panic!("cuStreamEndCapture failed: {r}");
        }
        let mut exec: *mut std::ffi::c_void = std::ptr::null_mut();
        let r = unsafe { (api.cuGraphInstantiate)(&mut exec, graph, 0) };
        if r != 0 {
            unsafe { (api.cuGraphDestroy)(graph) };
            panic!("cuGraphInstantiate failed: {r}");
        }
        unsafe { (api.cuGraphDestroy)(graph) }; // exec is independent
        let mut g = self.graph.lock().unwrap();
        g.capturing = false;
        g.graph_exec = Some(exec);
        // The CUDA graph handle IS the trace (hardware-recorded op sequence);
        // the CPU-side OpTrace recorder is the CPU backend's equivalent.
        crate::graph::OpTrace::default()
    }

    fn begin_verify(&self, _trace: &crate::graph::OpTrace) {
        let api = DriverApi::get().expect("libcuda not loadable");
        let exec = self.graph.lock().unwrap().graph_exec;
        if let Some(exec) = exec {
            let r = unsafe { (api.cuGraphLaunch)(exec, self.stream) };
            if r != 0 {
                panic!("cuGraphLaunch failed: {r}");
            }
        }
    }

    fn end_verify(&self) -> bool {
        self.sync().is_ok()
    }
}

impl crate::KernelBackend for CudaBackend {
    #[cfg(feature = "cuda")]
    fn as_cuda(&self) -> Option<&CudaBackend> {
        Some(self)
    }

    fn matmul(&self, x: &Tensor, w: &Tensor, bias: Option<&Tensor>, out: &mut Tensor) -> Result<()> {
        self.enter();
        self.run_matmul(x, w, bias, out)
    }

    fn rmsnorm(&self, x: &Tensor, w: &Tensor, eps: f32, out: &mut Tensor) -> Result<()> {
        self.enter();
        let n = (x.numel() / w.numel()) as i32;
        let dim = w.numel() as i32;
        let dx = DevBuf::alloc(self.dev, self.stream, x.numel())?; dx.upload(x.as_slice())?;
        let dw = self.dev_weight(w)?;
        let do_ = DevBuf::alloc(self.dev, self.stream, out.numel())?;
        ck(unsafe { ferrite_rmsnorm(dx.as_const_f32(), dw.as_const_f32(), do_.as_f32(), n, dim, eps, self.stream) }, "rmsnorm")?;
        let ov = Arc::get_mut(&mut out.data).expect("unique out");
        do_.download(ov)?;
        Ok(())
    }

    fn gated_rmsnorm(&self, x: &Tensor, gate: &Tensor, w: &Tensor, eps: f32, out: &mut Tensor) -> Result<()> {
        self.enter();
        let n = (x.numel() / w.numel()) as i32;
        let dim = w.numel() as i32;
        let dx = DevBuf::alloc(self.dev, self.stream, x.numel())?; dx.upload(x.as_slice())?;
        let dg = DevBuf::alloc(self.dev, self.stream, gate.numel())?; dg.upload(gate.as_slice())?;
        let dw = self.dev_weight(w)?;
        let do_ = DevBuf::alloc(self.dev, self.stream, out.numel())?;
        ck(unsafe { ferrite_gated_rmsnorm(dx.as_const_f32(), dg.as_const_f32(), dw.as_const_f32(), do_.as_f32(), n, dim, eps, self.stream) }, "gated_rmsnorm")?;
        let ov = Arc::get_mut(&mut out.data).expect("unique out");
        do_.download(ov)?;
        Ok(())
    }

    fn swiglu_limited(&self, gate_up: &Tensor, limit: f32, out: &mut Tensor) -> Result<()> {
        self.enter();
        let n = out.shape.0[0] as i32;
        let inter = out.shape.0[1] as i32;
        let dgu = DevBuf::alloc(self.dev, self.stream, gate_up.numel())?; dgu.upload(gate_up.as_slice())?;
        let do_ = DevBuf::alloc(self.dev, self.stream, out.numel())?;
        ck(unsafe { ferrite_swiglu(dgu.as_const_f32(), do_.as_f32(), n, inter, limit, self.stream) }, "swiglu")?;
        let ov = Arc::get_mut(&mut out.data).expect("unique out");
        do_.download(ov)?;
        Ok(())
    }

    fn causal_conv1d(&self, x: &Tensor, w: &Tensor, state_in: &Tensor, out: &mut Tensor, state_out: &mut Tensor) -> Result<()> {
        self.enter();
        let n = x.shape.0[0] as i32;
        let ch = x.shape.0[1] as i32;
        let conv = w.shape.0[1] as i32;
        let dx = DevBuf::alloc(self.dev, self.stream, x.numel())?; dx.upload(x.as_slice())?;
        let dw = self.dev_weight(w)?;
        let dsi = DevBuf::alloc(self.dev, self.stream, state_in.numel())?; dsi.upload(state_in.as_slice())?;
        let do_ = DevBuf::alloc(self.dev, self.stream, out.numel())?;
        let dso = DevBuf::alloc(self.dev, self.stream, state_out.numel())?;
        ck(unsafe { ferrite_causal_conv1d(dx.as_const_f32(), dw.as_const_f32(), dsi.as_const_f32(), do_.as_f32(), dso.as_f32(), n, ch, conv, self.stream) }, "conv1d")?;
        let ov = Arc::get_mut(&mut out.data).expect("unique out");
        do_.download(ov)?;
        let sv = Arc::get_mut(&mut state_out.data).expect("unique state");
        dso.download(sv)?;
        Ok(())
    }

    fn gated_deltanet_step(&self, q: &Tensor, k: &Tensor, v: &Tensor, beta: &Tensor, gate: &Tensor, a_log: &Tensor, state_in: &Tensor, out: &mut Tensor, state_out: &mut Tensor) -> Result<()> {
        self.enter();
        self.gated_deltanet_chunk(q, k, v, beta, gate, a_log, state_in, out, state_out)
    }

    fn gated_deltanet_chunk(&self, q: &Tensor, k: &Tensor, v: &Tensor, beta: &Tensor, gate: &Tensor, a_log: &Tensor, state_in: &Tensor, out: &mut Tensor, state_out: &mut Tensor) -> Result<()> {
        self.enter();
        let n = q.shape.0[0] as i32;
        let h = a_log.numel() as i32;
        let dk = *q.shape.0.last().unwrap() as i32;
        let dv = *v.shape.0.last().unwrap() as i32;
        let dq = DevBuf::alloc(self.dev, self.stream, q.numel())?; dq.upload(q.as_slice())?;
        let dk_ = DevBuf::alloc(self.dev, self.stream, k.numel())?; dk_.upload(k.as_slice())?;
        let dv_ = DevBuf::alloc(self.dev, self.stream, v.numel())?; dv_.upload(v.as_slice())?;
        let db = DevBuf::alloc(self.dev, self.stream, beta.numel())?; db.upload(beta.as_slice())?;
        let dg = DevBuf::alloc(self.dev, self.stream, gate.numel())?; dg.upload(gate.as_slice())?;
        let dal = self.dev_weight(a_log)?;
        // WYF chunkwise: state ping-pong buffers (chunk chain), tail chunk
        // falls back to the exact per-token kernel inside the launcher.
        let dst_a = DevBuf::alloc(self.dev, self.stream, state_in.numel())?; dst_a.upload(state_in.as_slice())?;
        let do_ = DevBuf::alloc(self.dev, self.stream, out.numel())?;
        ck(unsafe { ferrite_gdn_chunk_v2(dq.as_const_f32(), dk_.as_const_f32(), dv_.as_const_f32(), db.as_const_f32(), dg.as_const_f32(), dal.as_const_f32(), dst_a.as_f32(), do_.as_f32(), n, h, dk, dv, self.stream) }, "gdn_chunk_v2")?;
        let ov = Arc::get_mut(&mut out.data).expect("unique out");
        do_.download(ov)?;
        let sv = Arc::get_mut(&mut state_out.data).expect("unique state");
        dst_a.download(sv)?;
        Ok(())
    }

    fn indexer_topk(
        &self,
        q_idx: &Tensor,
        k_idx: &Tensor,
        w: &Tensor,
        topk: usize,
        ctx0: usize,
        idx: &mut Tensor,
    ) -> Result<()> {
        self.enter();
        let n = q_idx.shape.0[0] as i32;
        let hd = q_idx.shape.0[1] as i32;
        let d = k_idx.shape.0[1] as i32;
        let t = k_idx.shape.0[0] as i32;
        let h = w.shape.0[1] as i32;
        if hd != h * d {
            return Err(FerriteError::InvalidArg(
                "indexer_topk: q_idx [n,H*D] vs w [n,H] head mismatch".into(),
            ));
        }
        let dq = DevBuf::alloc(self.dev, self.stream, q_idx.numel())?; dq.upload(q_idx.as_slice())?;
        let dk = DevBuf::alloc(self.dev, self.stream, k_idx.numel())?; dk.upload(k_idx.as_slice())?;
        let dw = DevBuf::alloc(self.dev, self.stream, w.numel())?; dw.upload(w.as_slice())?;
        let di = DevBuf::alloc(self.dev, self.stream, idx.numel())?;
        let total_i32 = (t * 4) as i32; // total = npools * kpool (approximate: use t*4 as total for the pinned path)
        let total_ptr = &total_i32 as *const i32;
        let kpool_const = 4i32;
        ck(unsafe { ferrite_indexer_topk(dq.as_const_f32(), dk.as_const_f32(), dw.as_const_f32(), di.as_f32(), n, h, d, topk as i32, total_ptr, kpool_const, n, self.stream) }, "indexer_topk")?;
        let ov = Arc::get_mut(&mut idx.data).expect("unique idx");
        di.download(ov)?;
        Ok(())
    }

    fn sparse_mla_attn(&self, q: &Tensor, k_nope: &Tensor, v: &Tensor, idx: &Tensor, out: &mut Tensor) -> Result<()> {
        self.enter();
        let n = q.shape.0[0] as i32;
        let t = k_nope.shape.0[0] as i32;
        let h = q.shape.0[1] as i32;
        let d = *q.shape.0.last().unwrap() as i32;
        let dv = *v.shape.0.last().unwrap() as i32;
        let topk = *idx.shape.0.last().unwrap() as i32;
        let dq = DevBuf::alloc(self.dev, self.stream, q.numel())?; dq.upload(q.as_slice())?;
        let dk = DevBuf::alloc(self.dev, self.stream, k_nope.numel())?; dk.upload(k_nope.as_slice())?;
        let dv_ = DevBuf::alloc(self.dev, self.stream, v.numel())?; dv_.upload(v.as_slice())?;
        let di = DevBuf::alloc(self.dev, self.stream, idx.numel())?; di.upload(idx.as_slice())?;
        let do_ = DevBuf::alloc(self.dev, self.stream, out.numel())?;
        let t_ptr = &t as *const i32;
        ck(unsafe { ferrite_sparse_attn(dq.as_const_f32(), dk.as_const_f32(), dv_.as_const_f32(), di.as_const_f32(), do_.as_f32(), n, t_ptr, h, d, dv, topk, self.stream) }, "sparse_attn")?;
        let ov = Arc::get_mut(&mut out.data).expect("unique out");
        do_.download(ov)?;
        Ok(())
    }

    fn moe_route(&self, logits: &Tensor, bias: &Tensor, topk: usize, routed_scaling: f32, probs: &mut Tensor, ids: &mut Tensor) -> Result<()> {
        self.enter();
        let n = logits.shape.0[0] as i32;
        let e = logits.shape.0[1] as i32;
        let dl = DevBuf::alloc(self.dev, self.stream, logits.numel())?; dl.upload(logits.as_slice())?;
        let db = self.dev_weight(bias)?;
        let dp = DevBuf::alloc(self.dev, self.stream, probs.numel())?;
        // ids on the CPU backend are f32-valued; the kernel writes i32.
        let di = DevBuf::alloc(self.dev, self.stream, n as usize * topk)?;
        ck(unsafe { ferrite_moe_route(dl.as_const_f32(), db.as_const_f32(), dp.as_f32(), di.as_f32(), n, e, topk as i32, routed_scaling, self.stream) }, "moe_route")?;
        let pv = Arc::get_mut(&mut probs.data).expect("unique probs");
        dp.download(pv)?;
        let iv = Arc::get_mut(&mut ids.data).expect("unique ids");
        di.download(iv)?;
        Ok(())
    }

    fn expert_ffn(&self, x: &Tensor, gate_w: &Tensor, up_w: &Tensor, down_w: &Tensor, swiglu_limit: f32, out: &mut Tensor) -> Result<()> {
        self.enter();
        // Fused device-resident chain: upload x once, two matmuls + swiglu2
        // + down matmul all on device, single D2H at the end. (The old path
        // did two host round-trips plus a host-side gate/up gather.)
        let n = x.shape.0[0] as i32;
        let in_f = x.shape.0[1] as i32;
        let inter = gate_w.shape.0[0] as i32;
        let dx = DevBuf::alloc(self.dev, self.stream, x.numel())?;
        dx.upload(x.as_slice())?;
        let gate = self.matmul_dev(&dx, gate_w, n, in_f, inter)?;
        let up = self.matmul_dev(&dx, up_w, n, in_f, inter)?;
        let act = self.swiglu2_dev(&gate, &up, n, inter, swiglu_limit)?;
        let dout = self.matmul_dev(&act, down_w, n, inter, in_f)?;
        let ov = Arc::get_mut(&mut out.data).expect("unique out");
        dout.download(ov)?;
        Ok(())
    }

    fn argmax_lastdim(&self, logits: &Tensor, out: &mut Tensor) -> Result<()> {
        self.enter();
        let dim = *logits.shape.0.last().unwrap() as i32;
        let n = (logits.numel() / dim as usize) as i32;
        let dl = DevBuf::alloc(self.dev, self.stream, logits.numel())?; dl.upload(logits.as_slice())?;
        let do_ = DevBuf::alloc(self.dev, self.stream, out.numel())?;
        ck(unsafe { ferrite_argmax(dl.as_const_f32(), do_.as_f32(), n, dim, self.stream) }, "argmax")?;
        let ov = Arc::get_mut(&mut out.data).expect("unique out");
        do_.download(ov)?;
        Ok(())
    }

    fn softmax_lastdim(&self, logits: &Tensor, out: &mut Tensor) -> Result<()> {
        self.enter();
        let dim = *logits.shape.0.last().unwrap() as i32;
        let n = (logits.numel() / dim as usize) as i32;
        let dl = DevBuf::alloc(self.dev, self.stream, logits.numel())?; dl.upload(logits.as_slice())?;
        let do_ = DevBuf::alloc(self.dev, self.stream, out.numel())?;
        ck(unsafe { ferrite_softmax(dl.as_const_f32(), do_.as_f32(), n, dim, self.stream) }, "softmax")?;
        let ov = Arc::get_mut(&mut out.data).expect("unique out");
        do_.download(ov)?;
        Ok(())
    }

    // MHC hyper-connections on the GPU — replaces the per-token host loops
    // (24×16384 mixes dot + sinkhorn + weighted combine) that dominated the
    // layer boundary between the fan_out attention/FFN segments.
    fn hc_pre(
        &self,
        residual_flat: &Tensor,
        fn_w: &Tensor,
        scale: &Tensor,
        base: &Tensor,
        rms_eps: f32,
        hc_eps: f32,
        sinkhorn_iters: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        self.enter();
        let s = residual_flat.shape.0[0] as i32;
        let nh = residual_flat.shape.0[1] as i32;
        let mix = fn_w.shape.0[0] as i32;
        let n = ((-2.0 + (4.0 + 4.0 * mix as f64).sqrt()) / 2.0) as i32;
        let h = nh / n;
        let dr = DevBuf::alloc(self.dev, self.stream, residual_flat.numel())?;
        dr.upload(residual_flat.as_slice())?;
        let dfw = self.dev_weight(fn_w)?;
        let dsc = self.dev_weight(scale)?;
        let dba = self.dev_weight(base)?;
        let dli = DevBuf::alloc(self.dev, self.stream, (s * h) as usize)?;
        let dpost = DevBuf::alloc(self.dev, self.stream, (s * n) as usize)?;
        let dcomb = DevBuf::alloc(self.dev, self.stream, (s * n * n) as usize)?;
        ck(
            unsafe {
                ferrite_hc_pre(
                    dr.as_const_f32(), dfw.as_const_f32(), dsc.as_const_f32(), dba.as_const_f32(),
                    dli.as_f32(), dpost.as_f32(), dcomb.as_f32(),
                    s, n, h, mix, rms_eps, hc_eps, sinkhorn_iters as i32, self.stream,
                )
            },
            "hc_pre",
        )?;
        let mut li = Tensor::zeros(Shape::new([s as usize, h as usize]), DType::F32);
        {
            let v = Arc::get_mut(&mut li.data).expect("unique");
            dli.download(v)?;
        }
        let mut post = Tensor::zeros(Shape::new([s as usize, n as usize]), DType::F32);
        {
            let v = Arc::get_mut(&mut post.data).expect("unique");
            dpost.download(v)?;
        }
        let mut comb = Tensor::zeros(Shape::new([s as usize, n as usize, n as usize]), DType::F32);
        {
            let v = Arc::get_mut(&mut comb.data).expect("unique");
            dcomb.download(v)?;
        }
        Ok((li, post, comb))
    }

    fn hc_post(&self, x: &Tensor, residual: &Tensor, post: &Tensor, comb: &Tensor) -> Result<Tensor> {
        self.enter();
        let s = x.shape.0[0] as i32;
        let h = x.shape.0[1] as i32;
        let n = residual.shape.0[1] as i32;
        let dx = DevBuf::alloc(self.dev, self.stream, x.numel())?;
        dx.upload(x.as_slice())?;
        let drs = DevBuf::alloc(self.dev, self.stream, residual.numel())?;
        drs.upload(residual.as_slice())?;
        let dp = DevBuf::alloc(self.dev, self.stream, post.numel())?;
        dp.upload(post.as_slice())?;
        let dc = DevBuf::alloc(self.dev, self.stream, comb.numel())?;
        dc.upload(comb.as_slice())?;
        let do_ = DevBuf::alloc(self.dev, self.stream, (s * n * h) as usize)?;
        ck(
            unsafe { ferrite_hc_post(dx.as_const_f32(), drs.as_const_f32(), dp.as_const_f32(), dc.as_const_f32(), do_.as_f32(), s, n, h, self.stream) },
            "hc_post",
        )?;
        let mut out = Tensor::zeros(Shape::new([s as usize, n as usize, h as usize]), DType::F32);
        {
            let v = Arc::get_mut(&mut out.data).expect("unique");
            do_.download(v)?;
        }
        Ok(out)
    }
}

// ============================================================
// GDN layer device chain — the whole linear-attention forward as one
// DevBuf pipeline (zero host round-trips inside the layer): six
// projections → causal conv (resident state) → fused prep (silu+split+
// l2norm+beta+gate, one kernel) → gated-deltanet core (resident state)
// → gated rmsnorm → o_proj. The caller (TpCluster's device path / the
// future single CUDA graph) feeds [n, hidden] and gets the TP partial
// [n, hidden] back, both as DevBuf.
// ============================================================

/// Device-resident recurrent state (GDN [h,dk,dk] / conv tails) — NOT
/// pooled: it must persist across tokens, and pooled buffers get reused
/// by other ops between tokens.
pub struct DeviceState {
    pub ptr: *mut std::ffi::c_void,
    pub len: usize, // floats
}
unsafe impl Send for DeviceState {}
unsafe impl Sync for DeviceState {}

/// Per-layer MoE expert pointer table (device buffers of e_local raw
/// pointers into the bf16 weight cache) — the fused kernels' indirect
/// addressing for GPU-side expert dispatch.
pub struct MoePtrTable {
    pub gate_dev: *mut std::ffi::c_void,
    pub up_dev: *mut std::ffi::c_void,
    pub down_dev: *mut std::ffi::c_void,
    pub e_local: usize,
}
unsafe impl Send for MoePtrTable {}
unsafe impl Sync for MoePtrTable {}

/// Weight set for one GDN layer's device chain (borrowed from the shard's
/// Engine weights — all hit the dev_weight caches after warmup preload).
pub struct GdnLayerWeights<'a> {
    pub qkv_proj: &'a Tensor,
    pub b_proj: &'a Tensor,
    pub f_a: &'a Tensor,
    pub f_b: &'a Tensor,
    pub g_a: &'a Tensor,
    pub g_b: &'a Tensor,
    pub conv_w: &'a Tensor,
    pub dt_bias: &'a Tensor,
    pub a_log: &'a Tensor,
    pub o_norm: &'a Tensor,
    pub o_proj: &'a Tensor,
}

/// Device-resident DSA cache: k_nope [max_t, h, dk], v [max_t, h, dv],
/// k_idx/k_gate [max_t, idm] — pre-allocated, appended in place by
/// ferrite_dsa_cache_append. The CPU path grew host Vecs and cloned them
/// per layer per token (MBs of memcpy per call).
pub struct DsaCacheState {
    pub k_nope: *mut std::ffi::c_void,
    pub v: *mut std::ffi::c_void,
    pub k_idx: *mut std::ffi::c_void,
    pub k_gate: *mut std::ffi::c_void,
    pub max_tokens: usize,
    /// tokens appended so far (device-side counter; the CPU Vecs are gone)
    pub t_count: usize,
    /// PINNED t0/total (graph-safe): the CPU writes these before each
    /// graph replay; kernels read them zero-copy from host memory.
    /// [t0, total] — 2 ints, cudaMallocHost'd.
    pub pinned_t0: *mut i32,
    pub pinned_total: *mut i32,
}
unsafe impl Send for DsaCacheState {}
unsafe impl Sync for DsaCacheState {}

impl DsaCacheState {
    fn clone_raw(&self) -> (*mut std::ffi::c_void, *mut std::ffi::c_void, *mut std::ffi::c_void, *mut std::ffi::c_void, usize) {
        (self.k_nope, self.v, self.k_idx, self.k_gate, self.max_tokens)
    }
}

/// Weight set for one DSA layer's device chain (borrowed from the shard
/// Engine's weights — all hit the dev_weight caches after preload).
pub struct DsaLayerWeights<'a> {
    pub q_a: &'a Tensor,
    pub q_a_ln: &'a Tensor,
    pub q_b: &'a Tensor,
    pub kv_a: &'a Tensor,
    pub kv_a_ln: &'a Tensor,
    pub kv_b: &'a Tensor,
    pub wq_b: &'a Tensor,
    pub wk: &'a Tensor,
    pub k_norm_w: &'a Tensor,
    pub k_norm_b: &'a Tensor,
    pub weights_proj: &'a Tensor,
    pub gate: &'a Tensor,
    pub ape: &'a Tensor,
    pub o_proj: &'a Tensor,
    // dims
    pub h: usize,
    pub dk: usize,
    pub dv: usize,
    pub ih: usize,
    pub idm: usize,
    pub kpool: usize,
    pub topk: usize,
    pub rms_eps: f32,
}

impl CudaBackend {
    fn dev_state(
        &self,
        store: &std::sync::Mutex<std::collections::HashMap<(u64, usize), DeviceState>>,
        key: (u64, usize),
        len: usize,
    ) -> Result<*mut f32> {
        let mut m = store.lock().unwrap();
        if let Some(st) = m.get(&key) {
            return Ok(st.ptr as *mut f32);
        }
        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        ck(unsafe { cudaMalloc(&mut ptr, len * 4) }, "state malloc")?;
        ck(unsafe { cudaMemset(ptr, 0, len * 4) }, "state zero")?;
        m.insert(key, DeviceState { ptr, len });
        Ok(ptr as *mut f32)
    }

    /// Whole GDN (linear-attention) layer on device. `x` is the layer's
    /// normed input [n, hidden]; returns the o_proj partial [n, hidden]
    /// (TP all-reduce happens at the caller).
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_layer_dev(
        &self,
        x: &DevBuf,
        w: &GdnLayerWeights,
        seq: u64,
        layer: usize,
        n: usize,
        hidden: usize,
        h: usize,
        dk: usize,
        lb: f32,
        rms_eps: f32,
        conv_size: usize,
    ) -> Result<DevBuf> {
        self.enter();
        let proj = h * dk;
        let ni = n as i32;
        // 1. six projections (bf16-resident weights). NOTE: the gemv5 fused
        // same-input kernel measured SLOWER than separate tiled GEMVs
        // (48.7us vs 40.5us — block-per-row co-op dot loses to the tiled
        // smem kernel's throughput; graph-replay has no launch tail to
        // save) — kept separate. gemv5_dev stays for a future tiled fused
        // version (weight concat at load time).
        let qkv = self.matmul_dev(x, w.qkv_proj, ni, hidden as i32, (3 * proj) as i32)?;
        // PROBE: dump x (input) + qkv (first matmul output) — pinpoints
        // divergence to upload (x wrong) vs matmul/weights (qkv wrong)
        if std::env::var_os("FERRITE_GDN_PROBE").is_some() && layer == 0 && n > 1 {
            let dir = std::env::var("FERRITE_PROBE_DIR").unwrap_or_else(|_| "/tmp/orion".into());
            let d = |name: &str, v: &[f32]| {
                let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
                std::fs::write(format!("{dir}/gdn_dev_{name}_r{}.f32", crate::shard_idx()), b).ok();
            };
            let mut xh = vec![0f32; x.len];
            let _ = x.download(&mut xh);
            d("x", &xh);
            let mut qh = vec![0f32; n * 3 * proj];
            let _ = qkv.download(&mut qh);
            d("qkv", &qh);
            eprintln!("[gdn_probe] dev L0 x/qkv dumped: x {} qkv {} (n={} proj={})", xh.len(), qh.len(), n, proj);
        }
        let b_raw = self.matmul_dev(x, w.b_proj, ni, hidden as i32, h as i32)?;
        let fa = self.matmul_dev(x, w.f_a, ni, hidden as i32, dk as i32)?;
        let fb = self.matmul_dev(&fa, w.f_b, ni, dk as i32, proj as i32)?;
        let ga = self.matmul_dev(x, w.g_a, ni, hidden as i32, dk as i32)?;
        let gb = self.matmul_dev(&ga, w.g_b, ni, dk as i32, proj as i32)?;
        // 2. causal conv — resident tail state (RMW in place: the kernel
        // reads state_in into smem at block start, writes state_out at end,
        // one block per channel, so in==out is safe)
        let ch = 3 * proj;
        let hist = conv_size.saturating_sub(1).max(1);
        let dw_conv = self.dev_weight(w.conv_w)?;
        let conv_state = self.dev_state(&self.conv_states, (seq, layer), ch * hist)?;
        let conv_out = DevBuf::alloc(self.dev, self.stream, n * ch)?;
        ck(
            unsafe {
                ferrite_causal_conv1d(
                    qkv.as_const_f32(), dw_conv.as_const_f32(), conv_state,
                    conv_out.as_f32(), conv_state, ni, ch as i32, conv_size as i32, self.stream,
                )
            },
            "conv1d_dev",
        )?;
        // 3. GPU pre-processing (ferrite_gdn_prep): silu + split + per-head L2
        // norm + KDA q-scale (rsqrt(dk), k NOT scaled — its absence here was the
        // root cause of the garbage-output bug: dev_q norms 1.0 vs CPU 0.0884
        // = 1/sqrt(128)) + beta + gate — ONE kernel, zero host round-trips.
        // (The old hybrid path downloaded conv/b_raw/fb, computed silu/split/
        // L2/beta/gate on CPU, re-uploaded q/k/v/beta/gate: 9 host crossings.)
        let q = DevBuf::alloc(self.dev, self.stream, n * proj)?;
        let k = DevBuf::alloc(self.dev, self.stream, n * proj)?;
        let v = DevBuf::alloc(self.dev, self.stream, n * proj)?;
        let beta = DevBuf::alloc(self.dev, self.stream, n * h)?;
        let gate = DevBuf::alloc(self.dev, self.stream, n * proj)?;
        let dw_dt = self.dev_weight(w.dt_bias)?;
        let dw_al = self.dev_weight(w.a_log)?;
        ck(
            unsafe {
                ferrite_gdn_prep(
                    conv_out.as_const_f32(), b_raw.as_const_f32(), fb.as_const_f32(),
                    dw_dt.as_const_f32(), dw_al.as_const_f32(),
                    q.as_f32(), k.as_f32(), v.as_f32(), beta.as_f32(), gate.as_f32(),
                    ni, h as i32, dk as i32, lb, self.stream,
                )
            },
            "gdn_prep",
        )?;
        // PROBE (rank-isolated, prefill-only): download intermediates for
        // CPU-vs-dev divergence diff; normal path stays zero-crossing.
        if std::env::var_os("FERRITE_GDN_PROBE").is_some() && layer == 0 && n > 1 {
            let dir = std::env::var("FERRITE_PROBE_DIR").unwrap_or_else(|_| "/tmp/orion".into());
            let d = |name: &str, v: &[f32]| {
                let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
                std::fs::write(format!("{dir}/gdn_dev_{name}_r{}.f32", crate::shard_idx()), b).ok();
            };
            let mut ch = vec![0f32; n * ch];
            let _ = conv_out.download(&mut ch);
            d("conv", &ch);
            let mut bh0 = vec![0f32; n * h];
            let _ = b_raw.download(&mut bh0);
            d("braw", &bh0);
            let mut fh = vec![0f32; n * proj];
            let _ = fb.download(&mut fh);
            d("fb", &fh);
            let mut qh = vec![0f32; n * proj];
            let _ = q.download(&mut qh);
            d("q", &qh);
            let mut kh = vec![0f32; n * proj];
            let _ = k.download(&mut kh);
            d("k", &kh);
            let mut bth = vec![0f32; n * h];
            let _ = beta.download(&mut bth);
            d("beta", &bth);
            let mut gth = vec![0f32; n * proj];
            let _ = gate.download(&mut gth);
            d("gate", &gth);
            eprintln!(
                "[gdn_probe] dev L0 gpu-prep dumped r{} (conv {} q {} — ferrite_gdn_prep path)",
                crate::shard_idx(), ch.len(), qh.len()
            );
        }
        // 4. gated-deltanet core — resident [h, dk, dk] state (per-head
        // blocks read-modify-write their own slice; single buffer is safe).
        // v2: state staged in smem (padded stride dk+1) — HBM 7 passes → 2.
        let gdn_state = self.dev_state(&self.gdn_states, (seq, layer), h * dk * dk)?;
        let core = DevBuf::alloc(self.dev, self.stream, n * proj)?;
        ck(
            unsafe {
                ferrite_gdn_chunk_v2(
                    q.as_const_f32(), k.as_const_f32(), v.as_const_f32(),
                    beta.as_const_f32(), gate.as_const_f32(), self.dev_weight(w.a_log)?.as_const_f32(),
                    gdn_state, core.as_f32(), ni, h as i32, dk as i32, dk as i32, self.stream,
                )
            },
            "gdn_chunk_dev",
        )?;
        // probe: core output NaN check
        if std::env::var_os("FERRITE_GDN_PROBE").is_some() && layer == 0 {
            let mut cb = vec![0f32; n * proj];
            core.download(&mut cb)?;
            let nan_c = cb.iter().filter(|x| x.is_nan()).count();
            let cmax = cb.iter().fold(0f32, |a, v| if v.is_finite() { a.max(v.abs()) } else { a });
            // also check gdn state
            let mut sb = vec![0f32; h * dk * dk];
            unsafe {
                ck(cudaMemcpy(sb.as_mut_ptr() as *mut std::ffi::c_void, gdn_state as *const std::ffi::c_void, h * dk * dk * 4, CUDA_MEMCPY_D2H), "state probe")?;
            }
            let nan_s = sb.iter().filter(|x| x.is_nan()).count();
            let smax = sb.iter().fold(0f32, |a, v| if v.is_finite() { a.max(v.abs()) } else { a });
            eprintln!(
                "[gdn_dev] L{layer} core NaN {nan_c}/{} core_max {cmax:.3e} state NaN {nan_s}/{} state_max {smax:.3e}",
                n * proj, h * dk * dk
            );
        }
        // 5. gated rmsnorm (core [n,h,dk] flat = [n*h, dk]; gb the gate)
        let o_norm_w = self.dev_weight(w.o_norm)?;
        let normed = DevBuf::alloc(self.dev, self.stream, n * proj)?;
        ck(
            unsafe {
                ferrite_gated_rmsnorm(
                    core.as_const_f32(), gb.as_const_f32(), o_norm_w.as_const_f32(),
                    normed.as_f32(), (n * h) as i32, dk as i32, rms_eps, self.stream,
                )
            },
            "gdn_norm_dev",
        )?;
        // 6. o_proj — TP partial out (probe: dump core + partial rank-isolated)
        let partial = self.matmul_dev(&normed, w.o_proj, ni, proj as i32, hidden as i32)?;
        if std::env::var_os("FERRITE_GDN_PROBE").is_some() && layer == 0 && n > 1 {
            let dir = std::env::var("FERRITE_PROBE_DIR").unwrap_or_else(|_| "/tmp/orion".into());
            let d = |name: &str, v: &[f32]| {
                let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
                std::fs::write(format!("{dir}/gdn_dev_{name}_r{}.f32", crate::shard_idx()), b).ok();
            };
            let mut ch = vec![0f32; n * proj];
            let _ = core.download(&mut ch);
            d("core", &ch);
            let mut ph = vec![0f32; n * hidden];
            let _ = partial.download(&mut ph);
            d("partial", &ph);
            eprintln!("[gdn_probe] dev L0 core/partial dumped r{} (n={} proj={} hidden={})", crate::shard_idx(), n, proj, hidden);
        }
        Ok(partial)
    }

    /// Whole DSA (sparse attention) layer on device, zero host round-trips:
    /// gemv projections → layernorm → cache append (device-resident KV +
    /// index caches) → kpool compress → indexer topk → pool expand →
    /// sparse attention → o_proj. The CPU path did 10 Tensor-level ops
    /// (each a sync) + host-side cache clones per layer per token
    /// (2.8ms/layer measured).
    #[allow(clippy::too_many_arguments)]
    pub fn dsa_layer_dev(
        &self,
        x: &DevBuf,
        w: &DsaLayerWeights,
        seq: u64,
        family: usize,
        n: usize,
        hidden: usize,
    ) -> Result<DevBuf> {
        self.enter();
        let ni = n as i32;
        let (h, dk, dv, ih, idm, kpool) = (w.h, w.dk, w.dv, w.ih, w.idm, w.kpool);

        // 1. query path: qa → rmsnorm → qb [n, h*dk]. (gemv5 fused same-input
        // GEMV measured SLOWER than separate tiled — see gdn note.)
        let qa = self.matmul_dev(x, w.q_a, ni, hidden as i32, (w.q_a.shape.0[0]) as i32)?;
        let qa_ln = self.rmsnorm_dev(&qa, w.q_a_ln, w.rms_eps, n, w.q_a.shape.0[0])?;
        let qb = self.matmul_dev(&qa_ln, w.q_b, ni, w.q_a.shape.0[0] as i32, (h * dk) as i32)?;

        // 2. kv path: latent → rmsnorm → kvb [n, h*(dk+dv)]
        let latent = self.matmul_dev(x, w.kv_a, ni, hidden as i32, (w.kv_a.shape.0[0]) as i32)?;
        let kv_ln = self.rmsnorm_dev(&latent, w.kv_a_ln, w.rms_eps, n, w.kv_a.shape.0[0])?;
        let kvb = self.matmul_dev(&kv_ln, w.kv_b, ni, w.kv_a.shape.0[0] as i32, (h * (dk + dv)) as i32)?;

        // 3. indexer queries: qi = qa @ wq_b [n, ih*idm]
        let qi = self.matmul_dev(&qa_ln, w.wq_b, ni, w.q_a.shape.0[0] as i32, (ih * idm) as i32)?;

        // 4. index keys: ki = LayerNorm(x @ wk)(k_norm affine) [n, idm]
        let ki_raw = self.matmul_dev(x, w.wk, ni, hidden as i32, idm as i32)?;
        let ki = DevBuf::alloc(self.dev, self.stream, n * idm)?;
        let knw = self.dev_weight(w.k_norm_w)?;
        let knb = self.dev_weight(w.k_norm_b)?;
        ck(
            unsafe { ferrite_layernorm_affine(ki_raw.as_const_f32(), knw.as_const_f32(), knb.as_const_f32(), ki.as_f32(), ni, idm as i32, self.stream) },
            "dsa_layernorm",
        )?;

        // 5. per-head score weights: w_idx = (x @ weights_proj) × ih^-0.5 [n, ih]
        let w_idx = self.matmul_dev(x, w.weights_proj, ni, hidden as i32, ih as i32)?;
        ck(
            unsafe { ferrite_scale_inplace(w_idx.as_f32(), (ih as f32).sqrt().recip(), (n * ih) as i32, self.stream) },
            "dsa_widx_scale",
        )?;

        // 6. kpool gate scores: gate = x @ compress_gate [n, idm]
        let gate = self.matmul_dev(x, w.gate, ni, hidden as i32, idm as i32)?;

        // 7. cache append (device-resident, in place at slot t0)
        let (k_nope_dev, v_dev, k_idx_dev, k_gate_dev, t0) = {
            let mut m = self.dsa_caches.lock().unwrap();
            let (t0, ptrs) = match m.get(&(seq, family)) {
                Some(c) => (c.t_count, (c.k_nope, c.v, c.k_idx, c.k_gate)),
                None => (0usize, (std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut())),
            };
            if ptrs.0.is_null() {
                let max_tokens = 8192usize;
                let kn = self.dsa_alloc(max_tokens * h * dk)?;
                let vv = self.dsa_alloc(max_tokens * h * dv)?;
                let ki_ = self.dsa_alloc(max_tokens * idm)?;
                let kg = self.dsa_alloc(max_tokens * idm)?;
                // pinned t0/total (graph-safe zero-copy)
                let mut pt0: *mut i32 = std::ptr::null_mut();
                let mut ptot: *mut i32 = std::ptr::null_mut();
                ck(unsafe { cudaMallocHost(&mut pt0 as *mut *mut i32 as *mut *mut std::ffi::c_void, 4) }, "pinned t0")?;
                ck(unsafe { cudaMallocHost(&mut ptot as *mut *mut i32 as *mut *mut std::ffi::c_void, 4) }, "pinned total")?;
                unsafe { *pt0 = 0; *ptot = 0; }
                m.insert(
                    (seq, family),
                    DsaCacheState { k_nope: kn, v: vv, k_idx: ki_, k_gate: kg, max_tokens, t_count: n, pinned_t0: pt0, pinned_total: ptot },
                );
                (kn, vv, ki_, kg, 0)
            } else {
                m.get_mut(&(seq, family)).unwrap().t_count += n;
                (ptrs.0, ptrs.1, ptrs.2, ptrs.3, t0)
            }
        };
        // Write t0/total to pinned memory (graph-safe: kernels read zero-copy,
        // CPU writes before each replay)
        let (pinned_t0, pinned_total) = {
            let m = self.dsa_caches.lock().unwrap();
            let c = m.get(&(seq, family)).unwrap();
            unsafe {
                *c.pinned_t0 = t0 as i32;
                *c.pinned_total = (t0 + n) as i32;
            }
            (c.pinned_t0 as *const i32, c.pinned_total as *const i32)
        };
        ck(
            unsafe {
                ferrite_dsa_cache_append(
                    kvb.as_const_f32(), ki.as_const_f32(), gate.as_const_f32(),
                    k_nope_dev as *mut f32, v_dev as *mut f32, k_idx_dev as *mut f32, k_gate_dev as *mut f32,
                    pinned_t0, ni, h as i32, dk as i32, dv as i32, idm as i32, self.stream,
                )
            },
            "dsa_cache_append",
        )?;
        let total = t0 + n;

        // 8. kpool compression: pool_keys [npools, idm]
        // max_npools for graph safety: the grid is sized for the MAX
        // possible pools; the kernel derives the ACTUAL npools from the
        // pinned total (a frozen grid with actual npools would miss pools
        // as the context grows).
        let max_npools = (8192 + kpool - 1) / kpool; // max_tokens / kpool
        let npools = (total + kpool - 1) / kpool;
        let pool_keys = DevBuf::alloc(self.dev, self.stream, max_npools * idm)?;
        let dape = self.dev_weight(w.ape)?;
        ck(
            unsafe {
                ferrite_kpool_compress(
                    k_idx_dev as *const f32, k_gate_dev as *const f32, dape.as_const_f32(), pool_keys.as_f32(),
                    pinned_total, max_npools as i32, kpool as i32, idm as i32, self.stream,
                )
            },
            "dsa_kpool",
        )?;

        // 9. indexer topk over pools
        let select_k = (w.topk / kpool).min(npools);
        let idx_pools = DevBuf::alloc(self.dev, self.stream, n * select_k)?;
        let ctx0 = total - n;
        // graph-safe: pass pinned total_ptr instead of frozen npools/ctx0 values
        ck(
            unsafe {
                ferrite_indexer_topk(
                    qi.as_const_f32(), pool_keys.as_const_f32(), w_idx.as_const_f32(),
                    idx_pools.as_f32(), ni, ih as i32, idm as i32,
                    select_k as i32, pinned_total, kpool as i32, ni, self.stream,
                )
            },
            "dsa_topk",
        )?;

        // 10. expand pools to token indices [n, out_width]
        let out_width = select_k * kpool + (kpool - 1);
        let idx = DevBuf::alloc(self.dev, self.stream, n * out_width)?;
        ck(
            unsafe {
                ferrite_pool_expand(
                    idx_pools.as_const_f32(), idx.as_f32(),
                    ni, select_k as i32, kpool as i32, max_npools as i32, pinned_total,
                    ni, self.stream,
                )
            },
            "dsa_pool_expand",
        )?;

        // 11. sparse attention: q [n,h,dk] × k [T,h,dk] × v [T,h,dv] → out [n, h*dv]
        // v2: 256-thread block (v1 was 32 — one warp over topk≈8K slots with
        // serial scalar dots + O(topk²) global idx rereads for dedup).
        let attn_out = DevBuf::alloc(self.dev, self.stream, n * h * dv)?;
        ck(
            unsafe {
                ferrite_sparse_attn_v2(
                    qb.as_const_f32(), k_nope_dev as *const f32, v_dev as *const f32, idx.as_const_f32(),
                    attn_out.as_f32(), ni, pinned_total, h as i32, dk as i32, dv as i32,
                    out_width as i32, self.stream,
                )
            },
            "dsa_sparse_attn",
        )?;

        // 12. o_proj — TP partial [n, hidden]
        let partial = self.matmul_dev(&attn_out, w.o_proj, ni, (h * dv) as i32, hidden as i32)?;
        Ok(partial)
    }

    /// Mega-graph host-side DSA advance: write the pinned t0/total that the
    /// captured graph's kernels read zero-copy, and advance t_count — the
    /// same bookkeeping dsa_layer_dev's host logic does per call, minus the
    /// kernels (the graph executes those at replay). Call BEFORE every
    /// graph replay.
    pub fn dsa_host_advance(&self, seq: u64, family: usize, n: usize) {
        let mut m = self.dsa_caches.lock().unwrap();
        if let Some(c) = m.get_mut(&(seq, family)) {
            let t0 = c.t_count;
            unsafe {
                *c.pinned_t0 = t0 as i32;
                *c.pinned_total = (t0 + n) as i32;
            }
            c.t_count += n;
        }
    }

    /// Mega-graph capture rollback: the capture pass ran dsa_layer_dev's
    /// host bookkeeping (t_count += n, pinned write) but its kernels were
    /// only RECORDED, not executed — undo the virtual advance so t_count
    /// equals the tokens actually in the cache, then advance before every
    /// replay keeps the invariant.
    pub fn dsa_host_rollback(&self, seq: u64, family: usize, n: usize) {
        let mut m = self.dsa_caches.lock().unwrap();
        if let Some(c) = m.get_mut(&(seq, family)) {
            c.t_count -= n;
        }
    }

    fn dsa_alloc(&self, floats: usize) -> Result<*mut std::ffi::c_void> {
        let mut p: *mut std::ffi::c_void = std::ptr::null_mut();
        ck(unsafe { cudaMalloc(&mut p, floats * 4) }, "dsa cache malloc")?;
        ck(unsafe { cudaMemset(p, 0, floats * 4) }, "dsa cache zero")?;
        Ok(p)
    }
}

// ============================================================
// TP all-reduce on device: sum N partial outputs in-place (graph-
// capturable, no H2D/D2H). For the decode-step device op chain —
// the fan-out produces world partial [n, hidden] DevBufs; this sums
// them into the first partial's buffer.
// ============================================================
extern "C" {
    fn ferrite_tp_all_reduce(partials: *const f32, out: *mut f32,
                               total: i32, world: i32, s: CuStream) -> i32;
    fn ferrite_moe_weighted_sum(probs: *const f32, eouts: *const f32,
                                  out: *mut f32, n: i32, topk: i32, hidden: i32,
                                  s: CuStream) -> i32;
    fn ferrite_moe_fused_act(x: *const f32, ids_f: *const f32,
                              gate_ptrs: *const *const std::ffi::c_void,
                              up_ptrs: *const *const std::ffi::c_void,
                              shared_gate: *const std::ffi::c_void,
                              shared_up: *const std::ffi::c_void,
                              act: *mut f32, expert_start: i32, e_local: i32,
                              hidden: i32, inter: i32, inter_shared: i32,
                              topk: i32, n: i32, limit: f32,
                              s: CuStream) -> i32;
    fn ferrite_moe_fused_down_sum(ids_f: *const f32, probs: *const f32,
                                   down_ptrs: *const *const std::ffi::c_void,
                                   shared_down: *const std::ffi::c_void,
                                   act: *const f32, out: *mut f32,
                                   expert_start: i32, e_local: i32,
                                   hidden: i32, inter: i32, inter_shared: i32,
                                   topk: i32, n: i32,
                                   s: CuStream) -> i32;
}

impl CudaBackend {
    /// Sum `world` partial [total] buffers (contiguous) into `out` — the
    /// GPU all-reduce for the TP decode-step device op chain. Graph-capturable.
    pub fn tp_all_reduce_dev(&self, partials: &DevBuf, out: &mut DevBuf, total: usize, world: usize) -> Result<()> {
        self.enter();
        ck(unsafe {
            ferrite_tp_all_reduce(partials.as_const_f32(), out.as_f32(),
                                   total as i32, world as i32, self.stream)
        }, "tp_all_reduce")
    }

    /// MoE weighted sum: out[t, h] = Σ_j probs[t, j] * eouts[t, j, h].
    /// Graph-capturable (replaces the CPU expert accumulation loop).
    pub fn moe_weighted_sum_dev(&self, probs: &DevBuf, eouts: &DevBuf, out: &mut DevBuf, n: usize, topk: usize, hidden: usize) -> Result<()> {
        self.enter();
        ck(unsafe {
            ferrite_moe_weighted_sum(probs.as_const_f32(), eouts.as_const_f32(),
                                      out.as_f32(), n as i32, topk as i32, hidden as i32, self.stream)
        }, "moe_weighted_sum")
    }
}

// ============================================================
// MoE layer device chain: routing + expert FFNs + weighted sum,
// all on device (zero H2D/D2H inside the layer). The caller
// (CUDA graph capture) feeds [n, hidden] DevBuf and gets the
// TP partial [n, hidden] DevBuf back.
// ============================================================

/// Weight set for one expert's device chain.
pub struct ExpertWeights<'a> {
    pub gate: &'a Tensor,
    pub up: &'a Tensor,
    pub down: &'a Tensor,
}

impl CudaBackend {
    /// Full MoE layer on device: x_dev [n, hidden] → partial [n, hidden].
    /// routing (moe_route) → top-k expert FFNs (matmul_dev + swiglu2_dev)
    /// → weighted sum → + shared expert. All DevBuf, graph-capturable.
    /// Lazily build (and cache) this layer's expert POINTER TABLE: three
    /// device buffers of e_local raw pointers into the dev_weight_bf16
    /// cache. The fused MoE kernels gather the selected experts' rows
    /// through these with GPU-side dispatch — zero host round-trips, zero
    /// duplicated weight memory. Keyed by the first expert's gate tensor
    /// pointer (stable per layer).
    pub fn moe_expert_ptrs(
        &self,
        experts: &[ExpertWeights],
    ) -> Result<(usize, *mut std::ffi::c_void, *mut std::ffi::c_void, *mut std::ffi::c_void)> {
        self.enter();
        let key = experts.first().map(|e| e.gate.as_slice().as_ptr() as usize).unwrap_or(0);
        let e_local = experts.len();
        {
            let m = self.moe_ptrs.lock().unwrap();
            if let Some(t) = m.get(&key) {
                return Ok((t.e_local, t.gate_dev, t.up_dev, t.down_dev));
            }
        }
        let mut gates: Vec<*mut std::ffi::c_void> = Vec::with_capacity(e_local);
        let mut ups: Vec<*mut std::ffi::c_void> = Vec::with_capacity(e_local);
        let mut downs: Vec<*mut std::ffi::c_void> = Vec::with_capacity(e_local);
        for e in experts {
            gates.push(self.dev_weight_bf16(e.gate)?.ptr);
            ups.push(self.dev_weight_bf16(e.up)?.ptr);
            downs.push(self.dev_weight_bf16(e.down)?.ptr);
        }
        let mk = |v: &[*mut std::ffi::c_void]| -> Result<*mut std::ffi::c_void> {
            let mut p: *mut std::ffi::c_void = std::ptr::null_mut();
            ck(unsafe { cudaMalloc(&mut p, v.len() * std::mem::size_of::<*mut std::ffi::c_void>()) }, "moe ptr table malloc")?;
            ck(unsafe {
                cudaMemcpy(p, v.as_ptr() as *const _, v.len() * std::mem::size_of::<*mut std::ffi::c_void>(), CUDA_MEMCPY_H2D)
            }, "moe ptr table H2D")?;
            Ok(p)
        };
        let (g, u, d) = (mk(&gates)?, mk(&ups)?, mk(&downs)?);
        self.moe_ptrs.lock().unwrap().insert(key, MoePtrTable { gate_dev: g, up_dev: u, down_dev: d, e_local });
        Ok((e_local, g, u, d))
    }

    pub fn moe_layer_dev(
        &self,
        x_dev: &DevBuf,
        gate_w: &Tensor,           // router [e, hidden]
        bias_w: &Tensor,            // router bias [e] (f32)
        shared: &ExpertWeights,     // shared expert
        experts: &[ExpertWeights],  // routed experts (this rank's slice)
        expert_start: usize,        // first expert id on this rank
        probs_out: &mut DevBuf,     // [n, topk] routing probabilities
        n: usize,
        hidden: usize,
        topk: usize,
        e_total: usize,
        routed_scaling: f32,
        swiglu_limit: f32,
    ) -> Result<DevBuf> {
        self.enter();
        let ni = n as i32;
        let hi = hidden as i32;

        // 1. routing: x @ gate_w → logits [n, e_total]
        let logits = self.matmul_dev(x_dev, gate_w, ni, hi, e_total as i32)?;

        // 2. moe_route on device: logits → probs [n, topk], ids [n, topk]
        let dprobs = DevBuf::alloc(self.dev, self.stream, n * topk)?;
        let dids = DevBuf::alloc(self.dev, self.stream, n * topk)?;

        // ---- FUSED PATH (TileRT ExpertSelect idea): GPU-side expert dispatch
        // via the pointer table — ids/probs NEVER cross to the host; two
        // kernels (act + down_sum) replace the per-expert kernel chains, the
        // D2D gather and the probs_ext upload. Now batch-capable: grid carries
        // the token dim (n==1 decode, n>1 chunked prefill).
        {
            let dbias = self.dev_weight(bias_w)?;
            ck(unsafe {
                ferrite_moe_route(logits.as_const_f32(), dbias.as_const_f32(),
                                  dprobs.as_f32(), dids.as_f32(),
                                  ni, e_total as i32, topk as i32, routed_scaling, self.stream)
            }, "moe_route_fused")?;
            if probs_out.len >= n * topk {
                let (dst, src) = (probs_out.as_f32(), dprobs.as_const_f32());
                ck(unsafe {
                    cudaMemcpyAsync(dst as *mut _, src as *const _, n * topk * 4, CUDA_MEMCPY_D2D, self.stream)
                }, "probs D2D (fused)")?;
            }
            let (e_local, g_ptrs, u_ptrs, d_ptrs) = self.moe_expert_ptrs(experts)?;
            // Routed experts keep the FULL inter; the shared expert's inter is
            // TP-sharded (moe_intermediate_size / world). The act buffer is
            // [n, topk*inter + inter_shared] (see the kernels' slot layout).
            let inter = experts.first()
                .map(|e| e.gate.shape.0[0])
                .unwrap_or(shared.gate.shape.0[0]) as i32;
            let inter_shared = shared.gate.shape.0[0] as i32;
            let dsg = self.dev_weight_bf16(shared.gate)?;
            let dsu = self.dev_weight_bf16(shared.up)?;
            let dsd = self.dev_weight_bf16(shared.down)?;
            let act = DevBuf::alloc(self.dev, self.stream, n * (topk * inter as usize + inter_shared as usize))?;
            ck(unsafe {
                ferrite_moe_fused_act(
                    x_dev.as_const_f32(), dids.as_const_f32(),
                    g_ptrs as *const *const _, u_ptrs as *const *const _,
                    dsg.ptr, dsu.ptr, act.as_f32(),
                    expert_start as i32, e_local as i32, hi, inter, inter_shared,
                    topk as i32, ni, swiglu_limit, self.stream,
                )
            }, "moe_fused_act")?;
            let out = DevBuf::alloc(self.dev, self.stream, n * hidden)?;
            ck(unsafe {
                ferrite_moe_fused_down_sum(
                    dids.as_const_f32(), dprobs.as_const_f32(),
                    d_ptrs as *const *const _, dsd.ptr,
                    act.as_const_f32(), out.as_f32(),
                    expert_start as i32, e_local as i32, hi, inter, inter_shared,
                    topk as i32, ni, self.stream,
                )
            }, "moe_fused_down_sum")?;
            return Ok(out);
        }

        ck(unsafe {
            let dbias = self.dev_weight(bias_w)?;
            ferrite_moe_route(logits.as_const_f32(), dbias.as_const_f32(),
                              dprobs.as_f32(), dids.as_f32(),
                              ni, e_total as i32, topk as i32, routed_scaling, self.stream)
        }, "moe_route_dev")?;

        // 3. shared expert: x → gate/up/swiglu/down → shared_out [n, hidden]
        let shared_gate = self.matmul_dev(x_dev, shared.gate, ni, hi, shared.gate.shape.0[0] as i32)?;
        let shared_up = self.matmul_dev(x_dev, shared.up, ni, hi, shared.up.shape.0[0] as i32)?;
        let shared_inter = shared.gate.shape.0[0] as i32; // gate/up have same inter
        let shared_act = self.swiglu2_dev(&shared_gate, &shared_up, ni, shared_inter, swiglu_limit)?;
        let shared_out = self.matmul_dev(&shared_act, shared.down, ni, shared_inter, hi)?;

        // 4. routed experts — the ONLY CPU↔GPU boundary in the layer:
        //    download ids+probs (small: n×topk), CPU dispatches expert FFN
        //    chains (device-resident, no syncs inside), D2D-copies each
        //    output into the gather buffer. For CUDA graph capture this
        //    becomes a static all-experts run + GPU-side gather instead.
        let mut ids_host = vec![0f32; n * topk];
        dids.download(&mut ids_host)?;
        let mut probs_host = vec![0f32; n * topk];
        dprobs.download(&mut probs_host)?;
        if probs_out.len >= n * topk {
            let (dst, src) = (probs_out.as_f32(), dprobs.as_const_f32());
            ck(unsafe {
                cudaMemcpyAsync(dst as *mut _, src as *const _, n * topk * 4, CUDA_MEMCPY_D2D, self.stream)
            }, "probs D2D")?;
        }

        // 5. gather buffer [n, (topk+1) * hidden]: slots 0..topk-1 = routed
        //    expert outputs, slot topk = shared output. probs_ext = [probs, 1.0]
        //    so ONE weighted_sum call folds the shared expert in.
        //    Zero-fill upfront: experts NOT owned by this rank (TP shard) leave
        //    their slots zero — the all-reduce sums partials across ranks.
        let slots = topk + 1;
        let mut eouts = DevBuf::alloc(self.dev, self.stream, n * slots * hidden)?;
        ck(unsafe { cudaMemsetAsync(eouts.as_f32() as *mut _, 0, n * slots * hidden * 4, self.stream) }, "eouts zero")?;
        let mut probs_ext = vec![0f32; n * slots];
        probs_ext[..n * topk].copy_from_slice(&probs_host);
        for t in 0..n {
            probs_ext[t * slots + topk] = 1.0; // shared weight
        }
        let dprobs_ext = DevBuf::alloc(self.dev, self.stream, n * slots)?;
        dprobs_ext.upload(&probs_ext)?;

        // 6. per-token expert dispatch: for decode (n=1) exactly `topk`
        //    expert chains; each = 3 matmuls + swiglu2 (no H2D/D2H inside).
        //    Experts not owned by this rank (TP shard) leave their slots ZERO
        //    (upfront memset above) — the all-reduce sums partials across ranks.
        let e_count = experts.len();
        for t in 0..n {
            for j in 0..topk {
                let slot = t * slots + j;
                let eid = ids_host[t * topk + j] as usize;
                let local = eid.saturating_sub(expert_start);
                if local >= e_count {
                    continue; // another rank owns this expert → zero slot
                }
                let w = &experts[local];
                let inter = w.gate.shape.0[0] as i32;
                let g = self.matmul_dev(x_dev, w.gate, ni, hi, inter)?;
                let u = self.matmul_dev(x_dev, w.up, ni, hi, inter)?;
                let a = self.swiglu2_dev(&g, &u, ni, inter, swiglu_limit)?;
                let d = self.matmul_dev(&a, w.down, ni, inter, hi)?;
                // D2D gather into slot (graph-capturable, no host round-trip)
                ck(unsafe {
                    cudaMemcpyAsync(
                        (eouts.as_f32() as *mut std::ffi::c_void).add(slot * hidden * 4),
                        d.as_const_f32() as *const std::ffi::c_void,
                        hidden * 4, // one token's row (decode: n==1 contiguous)
                        CUDA_MEMCPY_D2D, self.stream,
                    )
                }, "expert out gather")?;
            }
        }
        // shared expert output → slot topk (same layout)
        for t in 0..n {
            let slot = t * slots + topk;
            ck(unsafe {
                cudaMemcpyAsync(
                    (eouts.as_f32() as *mut std::ffi::c_void).add(slot * hidden * 4),
                    (shared_out.as_const_f32() as *const std::ffi::c_void).add(t * hidden * 4),
                    hidden * 4, CUDA_MEMCPY_D2D, self.stream,
                )
            }, "shared out gather")?;
        }

        // 7. ONE weighted-sum kernel: out[t] = Σ_j probs_ext[t,j]·eouts[t,j] + shared
        let mut out = DevBuf::alloc(self.dev, self.stream, n * hidden)?;
        self.moe_weighted_sum_dev(&dprobs_ext, &eouts, &mut out, n, slots, hidden)?;
        let _ = dprobs;
        Ok(out)
    }
}

// ============================================================
// MHC hyper-connections + rmsnorm at the DevBuf level — the layer-chain
// components for the full decode-step device op chain (graph capture):
// hc_pre_dev → rmsnorm_dev → gdn_layer_dev / moe_layer_dev →
// tp_all_reduce_dev → hc_post_dev, ALL DevBuf (zero host round-trips).
// The kernels are the same ferrite_hc_pre/hc_post/rmsnorm the Tensor-level
// path uses — these wrappers just keep activations on device.
// ============================================================
impl CudaBackend {
    /// MHC pre-step on device: residual_flat [s, n*h] → (li [s,h],
    /// post [s,n], comb [s,n,n]). Weights (fn_w/scale/base) hit the
    /// dev_weight cache (f32, resident after preload). DevBuf in/out —
    /// the CPU sinkhorn/dot loops this replaces took ~0.9ms/layer.
    #[allow(clippy::too_many_arguments)]
    pub fn hc_pre_dev(
        &self,
        res: &DevBuf,
        fn_w: &Tensor,
        scale: &Tensor,
        base: &Tensor,
        s: usize,
        nh: usize,
        rms_eps: f32,
        hc_eps: f32,
        sinkhorn_iters: usize,
    ) -> Result<(DevBuf, DevBuf, DevBuf)> {
        self.enter();
        let mix = fn_w.shape.0[0];
        let n = ((-2.0 + (4.0 + 4.0 * mix as f64).sqrt()) / 2.0) as usize;
        let h = nh / n;
        let dfw = self.dev_weight(fn_w)?;
        let dsc = self.dev_weight(scale)?;
        let dba = self.dev_weight(base)?;
        let li = DevBuf::alloc(self.dev, self.stream, s * h)?;
        let post = DevBuf::alloc(self.dev, self.stream, s * n)?;
        let comb = DevBuf::alloc(self.dev, self.stream, s * n * n)?;
        let mx_scratch = DevBuf::alloc(self.dev, self.stream, s * mix)?;
        // SPLIT version (grid(s, mix) — one block per mix row): the
        // single-block kernel ran on ONE SM (~6GB/s of 8TB/s HBM); the mix
        // GEMV (16384×18432) was 57% of the decode step (A_hc+C_hc 24ms of
        // 42ms). The original err-900 capture concern (mx_scratch pool
        // class not warm in capture thread) is gone under the mega-graph:
        // the dry-run warms the pool on the SAME worker that captures.
        ck(
            unsafe {
                ferrite_hc_pre_split(
                    res.as_const_f32(), dfw.as_const_f32(), dsc.as_const_f32(), dba.as_const_f32(),
                    li.as_f32(), post.as_f32(), comb.as_f32(),
                    mx_scratch.as_f32(),
                    s as i32, n as i32, h as i32, mix as i32,
                    rms_eps, hc_eps, sinkhorn_iters as i32, self.stream,
                )
            },
            "hc_pre_dev",
        )?;
        Ok((li, post, comb))
    }

    /// MHC post-step on device: x [s, h_out] + residual [s, n, h] →
    /// out [s, n, h] (per DevBuf — the all-reduce partial feeds straight
    /// in, the next hc_pre's residual feeds straight out).
    pub fn hc_post_dev(
        &self,
        x: &DevBuf,
        res: &DevBuf,
        post: &DevBuf,
        comb: &DevBuf,
        s: usize,
        n: usize,
        h: usize,
    ) -> Result<DevBuf> {
        self.enter();
        let out = DevBuf::alloc(self.dev, self.stream, s * n * h)?;
        ck(
            unsafe {
                ferrite_hc_post(
                    x.as_const_f32(), res.as_const_f32(), post.as_const_f32(), comb.as_const_f32(),
                    out.as_f32(), s as i32, n as i32, h as i32, self.stream,
                )
            },
            "hc_post_dev",
        )?;
        Ok(out)
    }

    /// RMSNorm on device: x [n, dim] → out (weight resident f32).
    /// DevBuf in/out — the layer chain's input_layernorm/post_attention
    /// layernorm without a host round-trip.
    pub fn rmsnorm_dev(&self, x: &DevBuf, w: &Tensor, eps: f32, n: usize, dim: usize) -> Result<DevBuf> {
        self.enter();
        let dw = self.dev_weight(w)?;
        let out = DevBuf::alloc(self.dev, self.stream, x.len)?;
        ck(
            unsafe { ferrite_rmsnorm(x.as_const_f32(), dw.as_const_f32(), out.as_f32(), n as i32, dim as i32, eps, self.stream) },
            "rmsnorm_dev",
        )?;
        Ok(out)
    }

    /// hc_contract on device: x [s, n*h] → out [s, h] — mean over the n MHC
    /// flows (mirror of mhc::hc_expand). The mega-graph's head-chain bridge:
    /// last layer's residual → contract → rmsnorm → lm_head, zero host
    /// crossings.
    pub fn hc_contract_dev(&self, x: &DevBuf, s: usize, n: usize, h: usize) -> Result<DevBuf> {
        self.enter();
        let out = DevBuf::alloc(self.dev, self.stream, s * h)?;
        ck(
            unsafe { ferrite_hc_contract(x.as_const_f32(), out.as_f32(), s as i32, n as i32, h as i32, self.stream) },
            "hc_contract_dev",
        )?;
        Ok(out)
    }

    /// Fused multi-GEMV (same input x, up to 5 weight matrices) — ONE launch
    /// for decode chains that project x through several matrices (gdn: qkv/b/
    /// fa/ga share x; dsa: qa/latent/ki/w_idx/gate share x). Kills N-1 kernel
    /// launches + their tail latencies (~10-15us each on B300). w5=None →
    /// of5=0 (4-matrix case). Returns the output DevBufs in order.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv5_dev(
        &self,
        x: &DevBuf,
        w1: &Tensor, w2: &Tensor, w3: &Tensor, w4: &Tensor,
        w5: Option<&Tensor>,
        in_f: i32,
        of1: i32, of2: i32, of3: i32, of4: i32,
    ) -> Result<(DevBuf, DevBuf, DevBuf, DevBuf, Option<DevBuf>)> {
        self.enter();
        let d1 = self.dev_weight_bf16(w1)?;
        let d2 = self.dev_weight_bf16(w2)?;
        let d3 = self.dev_weight_bf16(w3)?;
        let d4 = self.dev_weight_bf16(w4)?;
        let o1 = DevBuf::alloc(self.dev, self.stream, of1 as usize)?;
        let o2 = DevBuf::alloc(self.dev, self.stream, of2 as usize)?;
        let o3 = DevBuf::alloc(self.dev, self.stream, of3 as usize)?;
        let o4 = DevBuf::alloc(self.dev, self.stream, of4 as usize)?;
        let of5 = w5.map(|t| t.shape.0[0] as i32).unwrap_or(0);
        let d5: *const std::ffi::c_void = match w5 {
            Some(w5t) => self.dev_weight_bf16(w5t)?.ptr,
            None => std::ptr::null(),
        };
        let o5 = match w5 {
            Some(w5t) => DevBuf::alloc(self.dev, self.stream, w5t.numel())?,
            None => DevBuf::alloc(self.dev, self.stream, 1)?, // unused 4B slot
        };
        ck(
            unsafe {
                ferrite_gemv5_bf16(
                    x.as_const_f32(),
                    d1.ptr, d2.ptr, d3.ptr, d4.ptr, d5,
                    o1.as_f32(), o2.as_f32(), o3.as_f32(), o4.as_f32(),
                    if w5.is_some() { o5.as_f32() } else { std::ptr::null_mut() },
                    in_f, of1, of2, of3, of4, of5,
                    self.stream,
                )
            },
            "gemv5_dev",
        )?;
        Ok((o1, o2, o3, o4, if w5.is_some() { Some(o5) } else { None }))
    }
}

// ============================================================
// Per-layer-segment CUDA graphs (FERRITE_GRAPH_LAYER): each segment's
// op sequence (upload memcpy + kernels) is captured ONCE and replayed
// per token. The pool is per-device (fan_out ranks don't share) and each
// rank's op sequence is deterministic → buffer addresses are stable
// across tokens. The segment's INPUT staging and OUTPUT device buffers
// are registered as GraphIO and LEAKED (never returned to the pool —
// replay writes them; pool reuse would corrupt).
// ============================================================
/// Fixed IO pointers of a captured segment graph: the CPU writes the
/// input into `x_stage` (pinned, the recorded memcpy's source), launches
/// the graph, then downloads `out_dev`.
pub struct GraphIO {
    pub x_stage: *mut std::ffi::c_void,
    pub x_len: usize,
    pub out_dev: *mut std::ffi::c_void,
    pub out_len: usize,
}
unsafe impl Send for GraphIO {}
unsafe impl Sync for GraphIO {}
impl Clone for GraphIO {
    fn clone(&self) -> Self {
        *self
    }
}
impl Copy for GraphIO {}

impl CudaBackend {
    pub fn graph_io_put(&self, name: &str, io: GraphIO) {
        self.graph_io.lock().unwrap().insert(name.to_string(), io);
    }
    pub fn graph_io_get(&self, name: &str) -> Option<GraphIO> {
        self.graph_io.lock().unwrap().get(name).cloned()
    }
    /// Replay a segment graph with fresh input: write `input` to the
    /// captured staging, launch, download the output. (capture never
    /// executes — this is the steady-state path)
    pub fn graph_run(&self, name: &str, input: &[f32], out: &mut [f32]) -> Result<bool> {
        let Some(io) = self.graph_io_get(name) else { return Ok(false); };
        if input.len() != io.x_len || out.len() != io.out_len {
            return Err(FerriteError::InvalidArg(format!(
                "graph_run {name}: input {} != stage {} or out {} != {}",
                input.len(), io.x_len, out.len(), io.out_len
            )));
        }
        unsafe {
            std::ptr::copy_nonoverlapping(input.as_ptr(), io.x_stage as *mut f32, io.x_len);
        }
        if !self.graph_replay(name) {
            return Ok(false);
        }
        self.enter();
        ck(
            unsafe {
                cudaMemcpyAsync(
                    out.as_mut_ptr() as *mut std::ffi::c_void,
                    io.out_dev,
                    io.out_len * 4,
                    CUDA_MEMCPY_D2H,
                    self.stream,
                )
            },
            "graph_run D2H",
        )?;
        self.sync()?;
        Ok(true)
    }
}

/// Blocking device→host copy (raw pointers — used by the graph capture
/// path where the DevBuf was forgotten but its address is registered).
pub fn memcpy_d2h_sync(src: *mut std::ffi::c_void, dst: *mut f32, floats: usize, s: CuStream) -> i32 {
    unsafe { cudaMemcpyAsync(dst as *mut _, src, floats * 4, CUDA_MEMCPY_D2H, s) };
    unsafe { cudaStreamSynchronize(s) }
}

/// Global serialization for graph capture (experiment): fan_out's 4 rank
/// workers capture CONCURRENTLY and cuGraphInstantiate crashed inside
/// libcuda (gdb: SIGSEGV in cuGraphInstantiate from worker #2+). Capture
/// is a one-time cost per segment — serializing the capture passes (not
/// the replays) costs nothing steady-state.
pub fn capture_lock() -> &'static std::sync::Mutex<()> {
    static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
    LOCK.get_or_init(|| std::sync::Mutex::new(()))
}

impl CudaBackend {
    /// Argmax over the last dim of a device buffer [n, dim] → out [n].
    /// Device-to-device (the Tensor-level path downloaded the full logits
    /// row — 620KB for GLM's 154880 vocab).
    pub fn argmax_dev(&self, logits: &DevBuf, out: &mut DevBuf, n: usize, dim: usize) -> Result<()> {
        self.enter();
        ck(
            unsafe { ferrite_argmax(logits.as_const_f32(), out.as_f32(), n as i32, dim as i32, self.stream) },
            "argmax_dev",
        )?;
        Ok(())
    }
}

// ============================================================
// P2P all-reduce via NVLink (B300 GPU4-7 = NV18): rank 0 collects
// the other ranks' partials with cudaMemcpyPeerAsync, then the
// existing tp_all_reduce kernel sums on-device. Replaces the host
// download→CPU-sum→re-upload round-trip per attention/ffn segment.
// ============================================================
extern "C" {
    fn ferrite_p2p_copy(dst: *mut f32, dst_dev: i32, src: *const f32, src_dev: i32,
                         count: usize, s: CuStream) -> i32;
    fn ferrite_p2p_enable(dev: i32, peer: i32) -> i32;
}

impl CudaBackend {
    /// Enable P2P access between this device and `peer` (NVLink).
    pub fn p2p_enable(&self, peer: i32) -> Result<()> {
        self.enter();
        ck(unsafe { ferrite_p2p_enable(self.dev, peer) }, "p2p_enable")?;
        Ok(())
    }

    /// P2P all-reduce: collect `partials` (device pointers from each rank)
    /// into a contiguous buffer on THIS device, then sum with the
    /// tp_all_reduce kernel. All pointers must be [n] floats.
    pub fn p2p_all_reduce(
        &self,
        partial_ptrs: &[usize],  // device pointers, index = rank
        n: usize,
    ) -> Result<DevBuf> {
        self.enter();
        let world = partial_ptrs.len();
        if world <= 1 {
            let out = DevBuf::alloc(self.dev, self.stream, n)?;
            ck(unsafe {
                cudaMemcpyAsync(out.as_f32() as *mut _, partial_ptrs[0] as *const _,
                                 n * 4, CUDA_MEMCPY_D2D, self.stream)
            }, "p2p single copy")?;
            return Ok(out);
        }
        // staging: [world, n] contiguous on this device — LEAKED (not returned
        // to the pool): the pool would reuse this memory for the NEXT op's
        // allocation while the GPU is still executing the tp_all_reduce
        // kernel that reads from it (async race). Per-token leak is
        // world * n * 4 bytes = 4 × 4096 × 4 = 64KB → acceptable.
        let mut staging = DevBuf::alloc(self.dev, self.stream, world * n)?;
        for (rank, &ptr) in partial_ptrs.iter().enumerate() {
            if rank as i32 == self.dev {
                // same device: plain D2D copy
                ck(unsafe {
                    cudaMemcpyAsync(
                        (staging.as_f32() as *mut std::ffi::c_void).add(rank * n * 4),
                        ptr as *const std::ffi::c_void,
                        n * 4, CUDA_MEMCPY_D2D, self.stream)
                }, "p2p local copy")?;
            } else {
                ck(unsafe {
                    ferrite_p2p_copy(
                        staging.as_f32().add(rank * n),
                        self.dev,
                        ptr as *const f32,
                        rank as i32,
                        n, self.stream)
                }, "p2p peer copy")?;
            }
        }
        // sum on this device
        let out = DevBuf::alloc(self.dev, self.stream, n)?;
        let mut out_mut = out;
        self.tp_all_reduce_dev(&staging, &mut out_mut, n, world)?;
        // CRITICAL: sync before returning — staging goes back to the pool on
        // drop, and the NEXT allocation would reuse its memory while the
        // GPU is still reading it (tp_all_reduce kernel is async).
        self.sync()?;
        Ok(out_mut)
    }
}

impl CudaBackend {
    /// Graph replay WITHOUT the D2H download: write the input to the
    /// captured staging, launch, return the output DEVICE pointer (for
    /// P2P all-reduce — the result stays on GPU).
    pub fn graph_run_dev(&self, name: &str, input: &[f32]) -> Result<Option<usize>> {
        let Some(io) = self.graph_io_get(name) else { return Ok(None); };
        if input.len() != io.x_len {
            return Err(FerriteError::InvalidArg(format!(
                "graph_run_dev {name}: input {} != stage {}", input.len(), io.x_len
            )));
        }
        unsafe {
            std::ptr::copy_nonoverlapping(input.as_ptr(), io.x_stage as *mut f32, io.x_len);
        }
        if !self.graph_replay(name) {
            return Ok(None);
        }
        Ok(Some(io.out_dev as usize))
    }
}

impl CudaBackend {
    /// Raw P2P copy (NVLink): copy `count` floats from `src` (on `src_dev`)
    /// to `dst` (on this device). Public for the hn broadcast in the P2P
    /// decode chain.
    pub fn p2p_copy_raw(&self, dst: *mut f32, src: *const f32, src_dev: i32, count: usize) -> Result<()> {
        self.enter();
        ck(unsafe { ferrite_p2p_copy(dst, self.dev, src, src_dev, count, self.stream) }, "p2p_copy_raw")?;
        Ok(())
    }
}

/// Set the current CUDA device (for NCCL init — needs device 0 context).
pub fn cuda_set_device(dev: i32) {
    unsafe { cudaSetDevice(dev) };
}
