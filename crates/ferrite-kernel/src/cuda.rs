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
    fn ferrite_f32_to_bf16(in_: *const f32, out: *mut std::ffi::c_void,
                            n: i64, s: CuStream) -> i32;
    fn ferrite_rmsnorm(x: *const f32, w: *const f32, out: *mut f32,
                       n: i32, dim: i32, eps: f32, s: CuStream) -> i32;
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
    fn ferrite_gdn_chunk_wyf(q: *const f32, k: *const f32, v: *const f32,
                             beta: *const f32, gate: *const f32, a_log: *const f32,
                             state_in: *mut f32, out: *mut f32, state_out: *mut f32,
                             n: i32, h: i32, dk: i32, dv: i32, s: CuStream) -> i32;
    fn ferrite_moe_route(logits: *const f32, bias: *const f32, probs: *mut f32, ids: *mut f32,
                         n: i32, e: i32, topk: i32,
                         scale: f32, s: CuStream) -> i32;
    fn ferrite_indexer_topk(qi: *const f32, ki: *const f32, w: *const f32, idx: *mut f32,
                            n: i32, t: i32, h: i32, d: i32, topk: i32, ctx0: i32, s: CuStream) -> i32;
    fn ferrite_sparse_attn(q: *const f32, k: *const f32, v: *const f32, idx: *const f32,
                           out: *mut f32, n: i32, t: i32, h: i32, d: i32, dv: i32,
                           topk: i32, s: CuStream) -> i32;
    fn ferrite_argmax(logits: *const f32, out: *mut f32, n: i32, dim: i32, s: CuStream) -> i32;
    fn ferrite_softmax(logits: *const f32, out: *mut f32, n: i32, dim: i32, s: CuStream) -> i32;
    fn ferrite_hc_pre(res: *const f32, fw: *const f32, scale: *const f32, base: *const f32,
                      li: *mut f32, post: *mut f32, comb: *mut f32,
                      s: i32, n: i32, h: i32, mix: i32,
                      rms_eps: f32, hc_eps: f32, iters: i32, stream: CuStream) -> i32;
    fn ferrite_hc_post(x: *const f32, res: *const f32, post: *const f32, comb: *const f32,
                       out: *mut f32, s: i32, n: i32, h: i32, stream: CuStream) -> i32;
}

const CUDA_MEMCPY_H2D: i32 = 1;
const CUDA_MEMCPY_D2H: i32 = 2;
extern "C" {
    fn cudaMemcpyAsync(dst: *mut std::ffi::c_void, src: *const std::ffi::c_void,
                        count: usize, kind: i32, stream: CuStream) -> i32;
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
// CUDA-graph capture support: every DevBuf owns a PINNED host staging buffer
// (cudaMallocHost, allocated with the device buffer, pooled with it).
// upload/download go through it — cudaMemcpyAsync from pageable memory is
// ILLEGAL during stream capture (cudaErrorStreamCaptureUnsupported) and the
// tensor's Vec address changes every call, which would bake a stale pointer
// into the graph. The pinned stage is the fixed-address rendezvous: the CPU
// writes it (outside the graph), the recorded memcpy moves stage→device.
// ============================================================
thread_local! {
    static BUF_POOL: std::cell::RefCell<
        std::collections::HashMap<(i32, u32), Vec<(*mut std::ffi::c_void, *mut std::ffi::c_void)>>,
    > = std::cell::RefCell::new(std::collections::HashMap::new());
}

/// True while the current thread is inside a stream capture (THREAD_LOCAL
/// mode). Download skips its synchronisation then — cudaStreamSynchronize
/// is illegal mid-capture; the graph's end_verify syncs once at the tail.
thread_local! {
    static CAPTURING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

pub(crate) fn set_capturing(v: bool) {
    CAPTURING.with(|c| c.set(v));
}

fn buf_pool_release(dev: i32, class: u32, ptr: *mut std::ffi::c_void, stage: *mut std::ffi::c_void) {
    BUF_POOL.with_borrow_mut(|p| p.entry((dev, class)).or_default().push((ptr, stage)));
}

fn buf_pool_take(dev: i32, class: u32) -> Option<(*mut std::ffi::c_void, *mut std::ffi::c_void)> {
    BUF_POOL.with_borrow_mut(|p| p.get_mut(&(dev, class)).and_then(|v| v.pop()))
}

/// Drop all pooled activation buffers (weights are owned by the weight cache).
/// Called from CudaBackend::Drop; leaks of freed devices are reclaimed by CUDA
/// context teardown at exit.
pub fn clear_activation_pool() {
    BUF_POOL.with_borrow_mut(|p| {
        for ((dev, _), ptrs) in p.drain() {
            unsafe { cudaSetDevice(dev) };
            for (ptr, stage) in ptrs {
                unsafe { cudaFree(ptr) };
                if !stage.is_null() {
                    unsafe { cudaFreeHost(stage) };
                }
            }
        }
    });
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
        if !CAPTURING.with(|c| c.get()) {
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
}

// cudaStream_t is thread-safe (CUDA runtime serialises ops on a stream);
// the raw pointer is just an opaque handle.
unsafe impl Send for CudaBackend {}
unsafe impl Sync for CudaBackend {}

impl Drop for CudaBackend {
    fn drop(&mut self) {
        self.clear_weight_cache();
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
        }
    }

    /// Bind this backend's device as the calling thread's current device.
    /// cudaSetDevice is thread-local state; TP ranks all call ops from the
    /// main thread, so each op entry re-binds before cudaMalloc/launch.
    #[inline]
    fn enter(&self) {
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
            // pack bf16 in place into the resident allocation. Blocking
            // memcpy between kernels serialises per chunk (fine — the H2D
            // dominates); a final stream sync covers the last kernel.
            const CHUNK: usize = 32 << 20; // 32M f32 = 128MB per step
            let mut scratch: *mut std::ffi::c_void = std::ptr::null_mut();
            ck(unsafe { cudaMalloc(&mut scratch, CHUNK * 4) }, "bf16 scratch malloc")?;
            let conv = (|| -> Result<()> {
                for (i, chunk) in src.chunks(CHUNK).enumerate() {
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

    fn sync(&self) -> Result<()> {
        ck(unsafe { cudaStreamSynchronize(self.stream) }, "sync")
    }

    /// Device-resident matmul: x already on device, w uploaded here (the
    /// BufferCache will dedupe repeated weight uploads), result stays on
    /// device. Building block for fused op chains (expert FFN).
    /// Weights are resident in bf16 (dev_weight_bf16).
    pub fn matmul_dev(&self, x_dev: &DevBuf, w: &Tensor, n: i32, in_f: i32, out_f: i32) -> Result<DevBuf> {
        let dw = self.dev_weight_bf16(w)?;
        let do_ = DevBuf::alloc(self.dev, self.stream, n as usize * out_f as usize)?;
        let dbias: *const f32 = std::ptr::null();
        ck(unsafe {
            ferrite_matmul_bf16(x_dev.as_const_f32(), dw.ptr as *const _,
                                 dbias, do_.as_f32(), n, in_f, out_f, self.stream)
        }, "matmul_dev")?;
        Ok(do_)
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
        ck(unsafe { ferrite_gdn_chunk(dq.as_const_f32(), dk_.as_const_f32(), dv_.as_const_f32(), db.as_const_f32(), dg.as_const_f32(), dal.as_const_f32(), dst_a.as_f32(), do_.as_f32(), n, h, dk, dv, self.stream) }, "gdn_chunk")?;
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
        ck(unsafe { ferrite_indexer_topk(dq.as_const_f32(), dk.as_const_f32(), dw.as_const_f32(), di.as_f32(), n, t, h, d, topk as i32, ctx0 as i32, self.stream) }, "indexer_topk")?;
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
        ck(unsafe { ferrite_sparse_attn(dq.as_const_f32(), dk.as_const_f32(), dv_.as_const_f32(), di.as_const_f32(), do_.as_f32(), n, t, h, d, dv, topk, self.stream) }, "sparse_attn")?;
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
