//! Device layer for DeepSeek-V4.1-Flash.
//!
//! Everything CUDA is resolved at **runtime** (dlopen/dlsym), exactly like
//! `ferrite-kernel` does: the crate keeps building and unit-testing on a box
//! with no CUDA, and on the B300 the symbols come from the real driver. No
//! link-time dependency is added to any GLM crate, and nothing in this crate
//! modifies GLM code.
//!
//! It also reuses a handful of GLM kernels **read-only** where the geometry is
//! identical: the hyper-connection chain (`ferrite_hc_pre` / `ferrite_hc_post`
//! — hc_mult 4 / sinkhorn 20 / eps 1e-6 is the same configuration), `rmsnorm`,
//! the embedding expander and the f32->bf16 cast. Reuse means calling the
//! existing symbol; no GLM source is touched.

use std::ffi::{c_char, c_int, c_long, c_uint, c_void, CString};

use ferrite_types::{FerriteError, Result};

type CuStream = *mut c_void;

const CUDA_R_32F: c_int = 0;
const CUDA_R_16BF: c_int = 14;
const CUBLAS_OP_N: c_int = 0;
const CUBLAS_OP_T: c_int = 1;

// ---------------------------------------------------------------- fn tables

struct Cudart {
    malloc: unsafe extern "C" fn(*mut *mut c_void, usize) -> c_int,
    free: unsafe extern "C" fn(*mut c_void) -> c_int,
    memcpy: unsafe extern "C" fn(*mut c_void, *const c_void, usize, c_int) -> c_int,
    memcpy_async: Option<unsafe extern "C" fn(*mut c_void, *const c_void, usize, c_int, CuStream) -> c_int>,
    memset: unsafe extern "C" fn(*mut c_void, c_int, usize) -> c_int,
    memset_async: Option<unsafe extern "C" fn(*mut c_void, c_int, usize, CuStream) -> c_int>,
    stream_create: unsafe extern "C" fn(*mut CuStream) -> c_int,
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
}

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

/// The DeepSeek kernels (this crate's own) plus the read-only GLM reuse set.
struct Kernels {
    // ---- dsv41 (this crate) ----
    gemm_fp8_mx: unsafe extern "C" fn(
        *const u8, *const f32, *const u8, *const u8, *const f32, *mut f32,
        c_int, c_int, c_int, CuStream,
    ) -> c_int,
    quant_fp8: unsafe extern "C" fn(
        *const f32, *mut u8, *mut f32, c_int, c_int, c_int, c_int, CuStream,
    ) -> c_int,
    quant_fp4: unsafe extern "C" fn(
        *const f32, *mut u8, *mut f32, c_int, c_int, c_int, c_int, CuStream,
    ) -> c_int,
    expert_gate_up_fp4: unsafe extern "C" fn(
        *const u8, *const f32, *const u8, *const u8, *const u8, *const u8, *mut f32,
        c_int, c_int, c_int, f32, CuStream,
    ) -> c_int,
    expert_down_fp4: unsafe extern "C" fn(
        *const f32, *const u8, *const u8, *const f32, *mut f32,
        c_int, c_int, c_int, CuStream,
    ) -> c_int,
    engram_hash: unsafe extern "C" fn(
        *const i32, *mut i64, *const i64, *const i64, *const i64, *const i32, *const u8,
        *mut i64, c_int, c_int, c_int, c_int, c_int, c_int, c_int, i64, CuStream,
    ) -> c_int,
    engram_gather: unsafe extern "C" fn(
        *const u8, *const u8, *const i64, *mut f32,
        c_int, c_int, c_int, i64, i64, CuStream,
    ) -> c_int,
    sparse_attn: unsafe extern "C" fn(
        *const f32, *const f32, *const f32, *const i32, *mut f32,
        c_int, c_int, c_int, c_int, *const c_int, c_int, c_int, f32, CuStream,
    ) -> c_int,
    indexer_topk: unsafe extern "C" fn(
        *const f32, *const f32, *const f32, *const u8, *const i32, *mut i32,
        c_int, c_int, c_int, c_int, c_int, c_int, c_int, f32, f32, c_int, CuStream,
    ) -> c_int,
    candidate_blocks: unsafe extern "C" fn(
        *const f32, *const i32, *mut u8, c_int, c_int, c_int, c_int, CuStream,
    ) -> c_int,
    compressor: unsafe extern "C" fn(
        *const f32, *const u8, *const u8, *const u8, *const u8, *const f32,
        *mut f32, *mut f32, *mut f32, *mut i32,
        c_int, c_int, c_int, c_int, c_int, c_int, *const c_int, f32, CuStream,
    ) -> c_int,
    rope_precompute: unsafe extern "C" fn(
        *mut f32, *mut f32, c_int, c_int, c_int, f32, f32, f32, f32, CuStream,
    ) -> c_int,
    apply_rope: unsafe extern "C" fn(
        *mut f32, *const f32, *const f32, c_int, c_int, c_int, c_int, *const c_int, c_int, c_int,
        c_int, c_int, CuStream,
    ) -> c_int,
    hc_mixes: unsafe extern "C" fn(
        *const f32, *const f32, *const f32, *const f32, *mut f32, *mut f32, *mut f32,
        c_int, c_int, c_int, c_int, f32, CuStream,
    ) -> c_int,
    moe_route: unsafe extern "C" fn(
        *const f32, *const u8, *const u8, *const f32, *mut f32, *mut i32, *mut i32,
        c_int, c_int, c_int, c_int, f32, c_int, f32, c_int, CuStream,
    ) -> c_int,
    add_inplace: Option<unsafe extern "C" fn(*const f32, *const f32, *mut f32, c_int, CuStream) -> c_int>,
    hc_collapse: Option<unsafe extern "C" fn(*const f32, *const f32, *mut f32, c_int, c_int, c_int, CuStream) -> c_int>,
    ar_stamp: Option<unsafe extern "C" fn(*const u64, c_int, c_int, c_uint, CuStream) -> c_int>,
    ar_store: Option<
        unsafe extern "C" fn(*const u64, c_int, c_int, *const f32, i64, i64, CuStream) -> c_int,
    >,
    ar_store2: Option<
        unsafe extern "C" fn(
            *const u64, c_int, c_int, *const f32, i64, i64, i64,
            *const c_uint, c_uint, *const u64, *mut c_uint, CuStream,
        ) -> c_int,
    >,
    ar_reduce2: Option<
        unsafe extern "C" fn(
            *mut f32, *const f32, i64, i64, c_int, *const c_uint, c_uint,
            *const u64, c_int, *mut c_uint, c_int, CuStream,
        ) -> c_int,
    >,
    ar_mark: Option<unsafe extern "C" fn(*const u64, c_int, c_int, c_uint, CuStream) -> c_int>,
    gemv_bf16: Option<unsafe extern "C" fn(*const c_void, *const f32, *mut f32, c_int, c_int, CuStream) -> c_int>,
    gemv_f32: Option<unsafe extern "C" fn(*const f32, *const f32, *mut f32, c_int, c_int, CuStream) -> c_int>,
    argmax: Option<unsafe extern "C" fn(*const f32, *mut c_int, c_int, *mut c_int, CuStream) -> c_int>,
    ar_v5_store: Option<
        unsafe extern "C" fn(*const u64, c_int, c_int, *const f32, i64, i64, *const c_uint, CuStream) -> c_int,
    >,
    ar_v5_publish: Option<
        unsafe extern "C" fn(*const u64, *const c_uint, c_int, c_int, *const c_uint, CuStream) -> c_int,
    >,
    ar_v5_reduce: Option<
        unsafe extern "C" fn(*mut f32, *const f32, i64, i64, c_int, *mut c_uint, *mut c_uint, CuStream) -> c_int,
    >,
    window_idxs: Option<unsafe extern "C" fn(*mut i32, *const c_int, c_int, CuStream) -> c_int>,
    comp_placeholder:
        Option<unsafe extern "C" fn(*mut i32, *const c_int, c_int, c_int, CuStream) -> c_int>,
    ring_append:
        Option<unsafe extern "C" fn(*mut f32, *const f32, *const c_int, c_int, c_int, CuStream) -> c_int>,
    compress_commit: Option<
        unsafe extern "C" fn(
            *const f32,
            *const f32,
            *const f32,
            *mut f32,
            *const c_int,
            *mut c_int,
            c_int,
            c_int,
            c_int,
            c_int,
            c_int,
            CuStream,
        ) -> c_int,
    >,
    engram_hash_step: Option<
        unsafe extern "C" fn(
            *const i64,
            *mut i64,
            *const i64,
            *const u64,
            *const u64,
            *mut i64,
            *const i32,
            *const i32,
            i64,
            c_int,
            c_int,
            c_int,
            i64,
            CuStream,
        ) -> c_int,
    >,
    expert_gate_up_fp4_indirect: Option<
        unsafe extern "C" fn(
            *const u8, *const f32, *mut f32, c_int, c_int, c_int, f32,
            *const u8, i64, *const u8, i64, *const u8, i64, *const u8, i64,
            *const c_int, c_int, CuStream,
        ) -> c_int,
    >,
    expert_down_fp4_indirect: Option<
        unsafe extern "C" fn(
            *const f32, *mut f32, c_int, c_int, c_int, *const f32,
            *const u8, i64, *const u8, i64, *const c_int, c_int, CuStream,
        ) -> c_int,
    >,
    ar_reduce: Option<
        unsafe extern "C" fn(*mut f32, *const f32, i64, i64, c_int, *const c_uint, c_uint, CuStream) -> c_int,
    >,
    compressor_pool: Option<
        unsafe extern "C" fn(*const f32, *const f32, *const f32, *mut f32, *mut f32, *mut f32, *mut c_int, c_int, c_int, c_int, c_int, c_int, *const c_int, f32, CuStream) -> c_int,
    >,
    route_topk: Option<
        unsafe extern "C" fn(*const f32, *const f32, *mut f32, *mut c_int, *mut c_int, c_int, c_int, c_int, c_int, f32, c_int, CuStream) -> c_int,
    >,
    engram_apply: Option<
        unsafe extern "C" fn(*mut f32, *const f32, *const f32, *const f32, *const u8, c_int, c_int, c_int, f32, CuStream) -> c_int,
    >,
    swiglu_limit: Option<unsafe extern "C" fn(*mut f32, c_int, c_int, f32, CuStream) -> c_int>,
    gather_rows: Option<unsafe extern "C" fn(*const f32, *const i32, *mut f32, c_int, c_int, CuStream) -> c_int>,
    scatter_add_rows: Option<
        unsafe extern "C" fn(*const f32, *const i32, *const f32, *mut f32, c_int, c_int, CuStream) -> c_int,
    >,
    window_append: Option<unsafe extern "C" fn(
        *const f32, *mut u8, *mut f32, c_int, c_int, c_int, c_int, CuStream,
    ) -> c_int>,
    // ---- GLM kernels reused read-only (identical geometry) ----
    rmsnorm: unsafe extern "C" fn(*const f32, *const f32, *mut f32, c_int, c_int, f32, CuStream) -> c_int,
    hc_pre: unsafe extern "C" fn(
        *const f32, *const f32, *const f32, *const f32, *mut f32, *mut f32, *mut f32,
        c_int, c_int, c_int, c_int, f32, f32, c_int, CuStream,
    ) -> c_int,
    hc_post: unsafe extern "C" fn(
        *const f32, *const f32, *const f32, *const f32, *mut f32,
        c_int, c_int, c_int, CuStream,
    ) -> c_int,
    embed_expand_dev: unsafe extern "C" fn(
        *const c_void, *const c_int, *mut f32, c_int, c_int, c_int, c_int, CuStream,
    ) -> c_int,
    f32_to_bf16: unsafe extern "C" fn(*const f32, *mut c_void, c_long, CuStream) -> c_int,
    bf16_to_f32: unsafe extern "C" fn(*const c_void, *mut c_void, c_long, CuStream) -> c_int,
}

fn sym(handle: *mut c_void, name: &str) -> Result<*mut c_void> {
    let c = CString::new(name).map_err(|_| FerriteError::Config("bad symbol name".into()))?;
    let p = unsafe { libc::dlsym(handle, c.as_ptr()) };
    if p.is_null() {
        return Err(FerriteError::Config(format!("dlsym({name}) not found")));
    }
    Ok(p)
}

macro_rules! f {
    ($h:expr, $name:expr) => {
        unsafe { std::mem::transmute_copy(&sym($h, $name)?) }
    };
}

// ------------------------------------------------------------------- Device

pub struct Device {
    /// bytes this handle has allocated (for the OOM diagnostic)
    allocated: std::cell::Cell<usize>,
    cudart: Cudart,
    cublas: Cublas,
    kernels: Kernels,
    stream: CuStream,
    handle: *mut c_void,
    debug_sync: bool,
    _cudart_handle: *mut c_void,
    _libs: (*mut c_void, *mut c_void, *mut c_void),
}

/// A plain device allocation. No pooling: the load-and-run path is
/// latency-tolerant, and skipping the pool keeps this crate free of any
/// interaction with GLM's capture-sensitive allocator.
#[derive(Clone)]
pub struct DevBuf {
    pub ptr: *mut c_void,
    pub bytes: usize,
    owned: bool,
}

impl DevBuf {
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
}

const CUDA_MEMCPY_H2D: c_int = 1;
const CUDA_MEMCPY_D2H: c_int = 2;
const CUDA_MEMCPY_D2D: c_int = 3;

impl Device {
    /// `kernel_so` is the path to `libferrite_kernels.so` (the dsv41 kernels
    /// are linked into the same object as GLM's).
    pub fn open(kernel_so: &str) -> Result<Device> {
        unsafe {
            // RTLD_GLOBAL so the kernel .so's own dependencies resolve
            let h_cudart = libc::dlopen(
                CString::new("libcudart.so").unwrap().as_ptr(),
                libc::RTLD_NOW | libc::RTLD_GLOBAL,
            );
            if h_cudart.is_null() {
                return Err(FerriteError::Config("dlopen(libcudart.so) failed".into()));
            }
            let h_cublas = libc::dlopen(
                CString::new("libcublas.so").unwrap().as_ptr(),
                libc::RTLD_NOW | libc::RTLD_GLOBAL,
            );
            if h_cublas.is_null() {
                return Err(FerriteError::Config("dlopen(libcublas.so) failed".into()));
            }
            let h_k = libc::dlopen(
                CString::new(kernel_so).unwrap().as_ptr(),
                libc::RTLD_NOW | libc::RTLD_GLOBAL,
            );
            if h_k.is_null() {
                let e = libc::dlerror();
                let msg = if e.is_null() {
                    "unknown".to_string()
                } else {
                    std::ffi::CStr::from_ptr(e).to_string_lossy().into_owned()
                };
                return Err(FerriteError::Config(format!(
                    "dlopen({kernel_so}) failed: {msg} — run kernels/cuda/build.sh first"
                )));
            }

            let cudart = Cudart {
                malloc: f!(h_cudart, "cudaMalloc"),
                free: f!(h_cudart, "cudaFree"),
                memcpy: f!(h_cudart, "cudaMemcpy"),
                memcpy_async: sym(h_cudart, "cudaMemcpyAsync")
                    .ok()
                    .map(|p| unsafe { std::mem::transmute_copy(&p) }),
                memset: f!(h_cudart, "cudaMemset"),
                memset_async: sym(h_cudart, "cudaMemsetAsync")
                    .ok()
                    .map(|p| unsafe { std::mem::transmute_copy(&p) }),
                stream_create: f!(h_cudart, "cudaStreamCreate"),
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
                    .map(|p| unsafe { std::mem::transmute_copy(&p) }),
                stream_end_capture: sym(h_cudart, "cudaStreamEndCapture")
                    .ok()
                    .map(|p| unsafe { std::mem::transmute_copy(&p) }),
                graph_instantiate: sym(h_cudart, "cudaGraphInstantiate")
                    .ok()
                    .map(|p| unsafe { std::mem::transmute_copy(&p) }),
                graph_launch: sym(h_cudart, "cudaGraphLaunch")
                    .ok()
                    .map(|p| unsafe { std::mem::transmute_copy(&p) }),
                graph_exec_destroy: sym(h_cudart, "cudaGraphExecDestroy")
                    .ok()
                    .map(|p| unsafe { std::mem::transmute_copy(&p) }),
                graph_destroy: sym(h_cudart, "cudaGraphDestroy")
                    .ok()
                    .map(|p| unsafe { std::mem::transmute_copy(&p) }),
            };
            let cublas = Cublas {
                create: f!(h_cublas, "cublasCreate_v2"),
                set_stream: f!(h_cublas, "cublasSetStream_v2"),
                gemm_ex: f!(h_cublas, "cublasGemmEx"),
            };
            let kernels = Kernels {
                gemm_fp8_mx: f!(h_k, "dsv41_gemm_fp8_mx"),
                quant_fp8: f!(h_k, "dsv41_quant_fp8"),
                quant_fp4: f!(h_k, "dsv41_quant_fp4"),
                expert_gate_up_fp4: f!(h_k, "dsv41_expert_gate_up_fp4"),
                expert_down_fp4: f!(h_k, "dsv41_expert_down_fp4"),
                engram_hash: f!(h_k, "dsv41_engram_hash"),
                engram_gather: f!(h_k, "dsv41_engram_gather"),
                sparse_attn: f!(h_k, "dsv41_sparse_attn"),
                indexer_topk: f!(h_k, "dsv41_indexer_topk"),
                candidate_blocks: f!(h_k, "dsv41_candidate_blocks"),
                compressor: f!(h_k, "dsv41_compressor"),
                rope_precompute: f!(h_k, "dsv41_rope_precompute"),
                apply_rope: f!(h_k, "dsv41_apply_rope"),
                hc_mixes: f!(h_k, "dsv41_hc_mixes"),
                moe_route: f!(h_k, "dsv41_moe_route"),
                add_inplace: sym(h_k, "ferrite_add").ok().map(|p| unsafe { std::mem::transmute_copy(&p) }),
                hc_collapse: sym(h_k, "dsv41_hc_collapse").ok().map(|p| unsafe { std::mem::transmute_copy(&p) }),
                ar_stamp: sym(h_k, "dsv41_ar_stamp").ok().map(|p| unsafe { std::mem::transmute_copy(&p) }),
                ar_store: sym(h_k, "dsv41_ar_store").ok().map(|p| unsafe { std::mem::transmute_copy(&p) }),
                ar_store2: sym(h_k, "dsv41_ar_store2").ok().map(|p| unsafe { std::mem::transmute_copy(&p) }),
                ar_reduce2: sym(h_k, "dsv41_ar_reduce2").ok().map(|p| unsafe { std::mem::transmute_copy(&p) }),
                ar_mark: sym(h_k, "dsv41_ar_mark").ok().map(|p| unsafe { std::mem::transmute_copy(&p) }),
                gemv_bf16: sym(h_k, "dsv41_gemv_bf16").ok().map(|p| unsafe { std::mem::transmute_copy(&p) }),
                gemv_f32: sym(h_k, "dsv41_gemv_f32").ok().map(|p| unsafe { std::mem::transmute_copy(&p) }),
                argmax: sym(h_k, "dsv41_argmax").ok().map(|p| unsafe { std::mem::transmute_copy(&p) }),
                ar_v5_store: sym(h_k, "dsv41_ar_v5_store").ok().map(|p| unsafe { std::mem::transmute_copy(&p) }),
                ar_v5_publish: sym(h_k, "dsv41_ar_v5_publish").ok().map(|p| unsafe { std::mem::transmute_copy(&p) }),
                ar_v5_reduce: sym(h_k, "dsv41_ar_v5_reduce").ok().map(|p| unsafe { std::mem::transmute_copy(&p) }),
                engram_hash_step: sym(h_k, "dsv41_engram_hash_step").ok().map(|p| unsafe { std::mem::transmute_copy(&p) }),
                window_idxs: sym(h_k, "dsv41_window_idxs").ok().map(|p| unsafe { std::mem::transmute_copy(&p) }),
                comp_placeholder: sym(h_k, "dsv41_comp_placeholder").ok().map(|p| unsafe { std::mem::transmute_copy(&p) }),
                compress_commit: sym(h_k, "dsv41_compress_commit").ok().map(|p| unsafe { std::mem::transmute_copy(&p) }),
                ring_append: sym(h_k, "dsv41_ring_append").ok().map(|p| unsafe { std::mem::transmute_copy(&p) }),
                expert_gate_up_fp4_indirect: sym(h_k, "dsv41_expert_gate_up_fp4_indirect")
                    .ok()
                    .map(|p| unsafe { std::mem::transmute_copy(&p) }),
                expert_down_fp4_indirect: sym(h_k, "dsv41_expert_down_fp4_indirect")
                    .ok()
                    .map(|p| unsafe { std::mem::transmute_copy(&p) }),
                ar_reduce: sym(h_k, "dsv41_ar_reduce").ok().map(|p| unsafe { std::mem::transmute_copy(&p) }),
                route_topk: sym(h_k, "dsv41_route_topk").ok().map(|p| unsafe { std::mem::transmute_copy(&p) }),
                compressor_pool: sym(h_k, "dsv41_compressor_pool").ok().map(|p| unsafe { std::mem::transmute_copy(&p) }),
                engram_apply: sym(h_k, "dsv41_engram_apply").ok().map(|p| unsafe { std::mem::transmute_copy(&p) }),
                swiglu_limit: sym(h_k, "dsv41_swiglu_limit").ok().map(|p| unsafe { std::mem::transmute_copy(&p) }),
                gather_rows: sym(h_k, "dsv41_gather_rows").ok().map(|p| unsafe { std::mem::transmute_copy(&p) }),
                scatter_add_rows: sym(h_k, "dsv41_scatter_add_rows").ok().map(|p| unsafe { std::mem::transmute_copy(&p) }),
                window_append: sym(h_k, "dsv41_window_append").ok().map(|p| unsafe { std::mem::transmute_copy(&p) }),
                rmsnorm: f!(h_k, "ferrite_rmsnorm"),
                hc_pre: f!(h_k, "ferrite_hc_pre"),
                hc_post: f!(h_k, "ferrite_hc_post"),
                embed_expand_dev: f!(h_k, "ferrite_embed_expand_dev"),
                f32_to_bf16: f!(h_k, "ferrite_f32_to_bf16"),
                bf16_to_f32: f!(h_k, "ferrite_bf16_to_f32"),
            };

            let mut stream: CuStream = std::ptr::null_mut();
            check_cudart(
                unsafe { (cudart.stream_create)(&mut stream) },
                &cudart,
                "cudaStreamCreate",
            )?;
            let mut handle: *mut c_void = std::ptr::null_mut();
            let st = (cublas.create)(&mut handle);
            if st != 0 {
                return Err(FerriteError::Config(format!("cublasCreate: {st}")));
            }
            let st = (cublas.set_stream)(handle, stream);
            if st != 0 {
                return Err(FerriteError::Config(format!("cublasSetStream: {st}")));
            }
            Ok(Device {
                cudart,
                cublas,
                kernels,
                stream,
                handle,
                debug_sync: std::env::var("DSV41_DEBUG_SYNC").map(|v| v != "0").unwrap_or(false),
                allocated: std::cell::Cell::new(0),
                _cudart_handle: h_cudart,
                _libs: (h_cudart, h_cublas, h_k),
            })
        }
    }

    pub fn stream(&self) -> CuStream {
        self.stream
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
        let d = Device::open_dummy_cudart()?;
        let rc = unsafe { (d.set_device)(gpu) };
        if rc != 0 {
            return Err(FerriteError::Config(format!("cudaSetDevice({gpu}): {rc}")));
        }
        Ok(())
    }

    fn open_dummy_cudart() -> Result<Cudart> {
        unsafe {
            let h = libc::dlopen(
                CString::new("libcudart.so").unwrap().as_ptr(),
                libc::RTLD_NOW | libc::RTLD_GLOBAL,
            );
            if h.is_null() {
                return Err(FerriteError::Config("dlopen(libcudart.so) failed".into()));
            }
            Ok(Cudart {
                malloc: f!(h, "cudaMalloc"),
                free: f!(h, "cudaFree"),
                memcpy: f!(h, "cudaMemcpy"),
                memcpy_async: None,
                memset: f!(h, "cudaMemset"),
                memset_async: None,
                stream_create: f!(h, "cudaStreamCreate"),
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
    /// (NVLink/PCIe peer DMA — the same mechanism GLM's collectives use). The
    /// copy is issued on this rank's stream, so it never passes through host
    /// memory; the caller synchronises before reading the result.
    pub fn memcpy_peer(
        &self,
        dst_dev: i32,
        dst: *mut c_void,
        src: *const c_void,
        bytes: usize,
    ) -> Result<()> {
        let st = unsafe {
            (self.cudart.memcpy_peer_async)(
                dst,
                dst_dev,
                src,
                self.device_id(),
                bytes,
                self.stream,
            )
        };
        check_cudart(st, &self.cudart, "cudaMemcpyPeerAsync")
    }

    pub fn sync(&self) -> Result<()> {
        check_cudart(
            unsafe { (self.cudart.stream_sync)(self.stream) },
            &self.cudart,
            "sync",
        )
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

    /// Borrow an existing pointer (e.g. a slice of a bigger allocation).
    pub fn view(ptr: *mut c_void, bytes: usize) -> DevBuf {
        DevBuf { ptr, bytes, owned: false }
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
        let bytes = unsafe {
            std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v))
        };
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
            return Err(FerriteError::Config(format!(
                "download {bytes} > buffer {}",
                src.bytes
            )));
        }
        let st = unsafe {
            (self.cudart.memcpy)(
                out.as_mut_ptr() as *mut c_void,
                src.ptr,
                bytes,
                CUDA_MEMCPY_D2H,
            )
        };
        check_cudart(st, &self.cudart, "cudaMemcpy D2H")
    }

    /// Raw byte download (used to prove the loader reproduces the file bytes).
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
    /// this is the path the weights take, mirroring GLM's mmap + cudaMemcpy.
    pub fn upload_from(&self, dst: *mut c_void, src: *const c_void, bytes: usize) -> Result<()> {
        let st = unsafe {
            (self.cudart.memcpy)(
                dst,
                src,
                bytes,
                CUDA_MEMCPY_H2D,
            )
        };
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

    /// Zero `bytes` at `ptr` (used to lay the padding down before the DMA
    /// overwrites the real part, so no host-side assembly is needed).
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

    /// `dst += src` elementwise over `n` f32 (GLM's `ferrite_add` with z == x:
    /// each thread reads its own operands before writing, so it is safe
    /// in place).
    pub fn add_inplace(&self, dst: &DevBuf, src: &DevBuf, n: i64) -> Result<()> {
        let f = self.need(self.kernels.add_inplace, "ferrite_add")?;
        let rc = unsafe {
            f(dst.ptr as *const f32, src.ptr as *const f32, dst.ptr as *mut f32, n as c_int, self.stream)
        };
        self.kerr(rc, "ferrite_add")
    }

    /// Device-wide synchronisation (used by the all-reduce, which must know
    /// that its peer copies have landed before summing them).
    pub fn dev_sync(&self) -> Result<()> {
        check_cudart(unsafe { (self.cudart.dev_sync)() }, &self.cudart, "cudaDeviceSynchronize")
    }

    /// `dst += src` elementwise over `n` f32 at raw addresses.
    pub fn add_inplace_raw(&self, dst: *mut c_void, src: *const c_void, n: i64) -> Result<()> {
        let f = self.need(self.kernels.add_inplace, "ferrite_add")?;
        let rc = unsafe {
            f(dst as *const f32, src as *const f32, dst as *mut f32, n as c_int, self.stream)
        };
        self.kerr(rc, "ferrite_add")
    }

    pub fn zero(&self, b: &DevBuf) -> Result<()> {
        self.zero_at(b.ptr, b.bytes)
    }

    // ------------------------------------------------------------ launches

    /// Fetch an optional kernel, failing with a clear message if the .so was
    /// built before that kernel existed (staged bring-up).
    fn need<T: Copy>(&self, f: Option<T>, name: &str) -> Result<T> {
        f.ok_or_else(|| {
            FerriteError::Config(format!(
                "kernel {name} is not in the loaded .so — rebuild kernels/cuda (bash build.sh 103a)"
            ))
        })
    }

    fn kerr(&self, rc: c_int, what: &str) -> Result<()> {
        if rc != 0 {
            return Err(FerriteError::Config(format!("{what}: cuda error {rc}")));
        }
        check_cudart(unsafe { (self.cudart.last_error)() }, &self.cudart, what)?;
        // DSV41_DEBUG_SYNC=1 synchronises after every launch so an ASYNC fault
        // (illegal address inside a kernel) is attributed to the kernel that
        // caused it instead of surfacing at the next sync.
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
                    "ASYNC FAULT in {what}: {msg}"
                )));
            }
        }
        Ok(())
    }

    /// fp8 e4m3 dense GEMM: `out[m,n] = a[m,k] @ w[n,k]^T`. `a_scale` is f32
    /// per (row, k/32); `w_scale` is a ue8m0 byte per (n/32, k/32).
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_fp8_mx(
        &self,
        a: *const u8,
        a_scale: *const f32,
        w: *const u8,
        w_scale: *const u8,
        bias: *const f32,
        out: *mut f32,
        m: i32,
        n: i32,
        k: i32,
    ) -> Result<()> {
        let rc = unsafe {
            (self.kernels.gemm_fp8_mx)(a, a_scale, w, w_scale, bias, out, m, n, k, self.stream)
        };
        self.kerr(rc, "dsv41_gemm_fp8_mx")
    }

    pub fn quant_fp8(
        &self,
        x: *const f32,
        y: *mut u8,
        scale: *mut f32,
        rows: i32,
        cols: i32,
        block: i32,
        round_scale: bool,
    ) -> Result<()> {
        let rc = unsafe {
            (self.kernels.quant_fp8)(
                x,
                y,
                scale,
                rows,
                cols,
                block,
                round_scale as i32,
                self.stream,
            )
        };
        self.kerr(rc, "dsv41_quant_fp8")
    }

    pub fn quant_fp4(
        &self,
        x: *const f32,
        y: *mut u8,
        scale: *mut f32,
        rows: i32,
        cols: i32,
        block: i32,
        round_scale: bool,
    ) -> Result<()> {
        let rc = unsafe {
            (self.kernels.quant_fp4)(
                x,
                y,
                scale,
                rows,
                cols,
                block,
                round_scale as i32,
                self.stream,
            )
        };
        self.kerr(rc, "dsv41_quant_fp4")
    }

    /// Native fp4 experts (tcgen05 MXFP4): gate and up in one pass.
    #[allow(clippy::too_many_arguments)]
    pub fn expert_gate_up_fp4(
        &self,
        a: *const u8,
        a_scale: *const f32,
        w1: *const u8,
        w1_scale: *const u8,
        w3: *const u8,
        w3_scale: *const u8,
        out: *mut f32,
        rows: i32,
        dim: i32,
        inter: i32,
        limit: f32,
    ) -> Result<()> {
        let rc = unsafe {
            (self.kernels.expert_gate_up_fp4)(
                a, a_scale, w1, w1_scale, w3, w3_scale, out, rows, dim, inter, limit, self.stream,
            )
        };
        self.kerr(rc, "dsv41_expert_gate_up_fp4")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn expert_down_fp4(
        &self,
        act: *const f32,
        w2: *const u8,
        w2_scale: *const u8,
        weight: *const f32,
        out: *mut f32,
        rows: i32,
        dim: i32,
        inter: i32,
    ) -> Result<()> {
        let rc = unsafe {
            (self.kernels.expert_down_fp4)(act, w2, w2_scale, weight, out, rows, dim, inter, self.stream)
        };
        self.kerr(rc, "dsv41_expert_down_fp4")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn engram_hash(
        &self,
        token_map: *const i32,
        cache: *mut i64,
        primes: *const i64,
        offsets: *const i64,
        multipliers: *const i64,
        input_ids: *const i32,
        mask: *const u8,
        out: *mut i64,
        batch_row: i32,
        seqlen: i32,
        max_seq: i32,
        start_pos: i32,
        n_layers: i32,
        max_ngram: i32,
        n_heads: i32,
        pad_id: i64,
    ) -> Result<()> {
        let rc = unsafe {
            (self.kernels.engram_hash)(
                token_map, cache, primes, offsets, multipliers, input_ids, mask, out, batch_row,
                seqlen, max_seq, start_pos, n_layers, max_ngram, n_heads, pad_id, self.stream,
            )
        };
        self.kerr(rc, "dsv41_engram_hash")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn engram_gather(
        &self,
        table: *const u8,
        table_scale: *const u8,
        hash_ids: *const i64,
        out: *mut f32,
        rows: i32,
        n_cols: i32,
        head_dim: i32,
        part_start: i64,
        part_rows: i64,
    ) -> Result<()> {
        let rc = unsafe {
            (self.kernels.engram_gather)(
                table, table_scale, hash_ids, out, rows, n_cols, head_dim, part_start, part_rows,
                self.stream,
            )
        };
        self.kerr(rc, "dsv41_engram_gather")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn sparse_attn(
        &self,
        q: *const f32,
        kv: *const f32,
        sink: *const f32,
        idxs: *const i32,
        out: *mut f32,
        b: i32,
        m: i32,
        h: i32,
        d: i32,
        clen: *const c_int,
        window: i32,
        index_topk: i32,
        scale: f32,
    ) -> Result<()> {
        let rc = unsafe {
            (self.kernels.sparse_attn)(
                q, kv, sink, idxs, out, b, m, h, d, clen, window, index_topk, scale, self.stream,
            )
        };
        self.kerr(rc, "dsv41_sparse_attn")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn indexer_topk(
        &self,
        q: *const f32,
        index_k: *const f32,
        weights: *const f32,
        candidates: *const u8,
        compress_lens: *const i32,
        out: *mut i32,
        b: i32,
        m: i32,
        nh: i32,
        hd: i32,
        n_pos: i32,
        topk: i32,
        offset: i32,
        softmax_scale: f32,
        head_scale: f32,
        uses_candidates: bool,
    ) -> Result<()> {
        let rc = unsafe {
            (self.kernels.indexer_topk)(
                q, index_k, weights, candidates, compress_lens, out, b, m, nh, hd, n_pos, topk,
                offset, softmax_scale, head_scale, uses_candidates as i32, self.stream,
            )
        };
        self.kerr(rc, "dsv41_indexer_topk")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn candidate_blocks(
        &self,
        logits: *const f32,
        compress_lens: *const i32,
        mask: *mut u8,
        rows: i32,
        n_pos: i32,
        topk_blocks: i32,
        block_size: i32,
    ) -> Result<()> {
        let rc = unsafe {
            (self.kernels.candidate_blocks)(
                logits, compress_lens, mask, rows, n_pos, topk_blocks, block_size, self.stream,
            )
        };
        self.kerr(rc, "dsv41_candidate_blocks")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn compressor(
        &self,
        x: *const f32,
        wkv: *const u8,
        wkv_scale: *const u8,
        wgate: *const u8,
        wgate_scale: *const u8,
        norm_w: *const f32,
        state_kv: *mut f32,
        state_score: *mut f32,
        latents: *mut f32,
        out_rows: *mut i32,
        b: i32,
        seqlen: i32,
        dim: i32,
        head_dim: i32,
        ratio: i32,
        start_pos: i32,
        pos_ctr: *const c_int,
        eps: f32,
    ) -> Result<()> {
        let rc = unsafe {
            (self.kernels.compressor)(
                x, wkv, wkv_scale, wgate, wgate_scale, norm_w, state_kv, state_score, latents,
                out_rows, b, seqlen, dim, head_dim, ratio, start_pos, pos_ctr, eps, self.stream,
            )
        };
        self.kerr(rc, "dsv41_compressor")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn rope_precompute(
        &self,
        cos: *mut f32,
        sin: *mut f32,
        dim: i32,
        seqlen: i32,
        original_seq_len: i32,
        base: f32,
        factor: f32,
        beta_fast: f32,
        beta_slow: f32,
    ) -> Result<()> {
        let rc = unsafe {
            (self.kernels.rope_precompute)(
                cos, sin, dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow,
                self.stream,
            )
        };
        self.kerr(rc, "dsv41_rope_precompute")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn apply_rope(
        &self,
        x: *mut f32,
        cos: *const f32,
        sin: *const f32,
        rows: i32,
        row_len: i32,
        dim: i32,
        half: i32,
        base: *const c_int,
        mul: i32,
        off: i32,
        step: i32,
        inverse: bool,
    ) -> Result<()> {
        let rc = unsafe {
            (self.kernels.apply_rope)(
                x, cos, sin, rows, row_len, dim, half, base, mul, off, step, inverse as i32,
                self.stream,
            )
        };
        self.kerr(rc, "dsv41_apply_rope")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn hc_mixes(
        &self,
        x: *const f32,
        hc_fn: *const f32,
        hc_scale: *const f32,
        hc_base: *const f32,
        pre: *mut f32,
        post: *mut f32,
        comb: *mut f32,
        rows: i32,
        hc_dim: i32,
        hc: i32,
        sinkhorn_iters: i32,
        eps: f32,
    ) -> Result<()> {
        let rc = unsafe {
            (self.kernels.hc_mixes)(
                x, hc_fn, hc_scale, hc_base, pre, post, comb, rows, hc_dim, hc, sinkhorn_iters,
                eps, self.stream,
            )
        };
        self.kerr(rc, "dsv41_hc_mixes")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn moe_route(
        &self,
        x: *const f32,
        gate_w: *const u8,
        gate_w_scale: *const u8,
        gate_bias: *const f32,
        weights: *mut f32,
        indices: *mut i32,
        hist: *mut i32,
        rows: i32,
        dim: i32,
        n_experts: i32,
        topk: i32,
        gate_temp: f32,
        norm_topk_prob: bool,
        route_scale: f32,
        score_func: i32,
    ) -> Result<()> {
        let rc = unsafe {
            (self.kernels.moe_route)(
                x, gate_w, gate_w_scale, gate_bias, weights, indices, hist, rows, dim, n_experts,
                topk, gate_temp, norm_topk_prob as i32, route_scale, score_func, self.stream,
            )
        };
        self.kerr(rc, "dsv41_moe_route")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn window_append(
        &self,
        kv: *const f32,
        cache: *mut u8,
        cache_scale: *mut f32,
        rows: i32,
        head_dim: i32,
        window: i32,
        start_pos: i32,
    ) -> Result<()> {
        let f = self.need(self.kernels.window_append, "dsv41_window_append")?;
        let rc = unsafe { f(kv, cache, cache_scale, rows, head_dim, window, start_pos, self.stream) };
        self.kerr(rc, "dsv41_window_append")
    }

    /// MoE routing from pre-computed (bf16-gate) scores.
    #[allow(clippy::too_many_arguments)]
    pub fn route_topk(
        &self,
        scores: *const f32,
        bias: *const f32,
        weights: *mut f32,
        indices: *mut i32,
        hist: *mut i32,
        rows: i32,
        n_experts: i32,
        topk: i32,
        norm_topk_prob: bool,
        route_scale: f32,
        score_func: i32,
    ) -> Result<()> {
        let f = self.need(self.kernels.route_topk, "dsv41_route_topk")?;
        let rc = unsafe {
            f(
                scores, bias, weights, indices, hist, rows, n_experts, topk,
                norm_topk_prob as i32, route_scale, score_func, self.stream,
            )
        };
        self.kerr(rc, "dsv41_route_topk")
    }

    /// Compressor pooling half (bf16 projections are done by the caller).
    #[allow(clippy::too_many_arguments)]
    pub fn compressor_pool(
        &self,
        kvp: *const f32,
        scp: *const f32,
        norm_w: *const f32,
        state_kv: *mut f32,
        state_score: *mut f32,
        latents: *mut f32,
        out_rows: *mut i32,
        b: i32,
        seqlen: i32,
        head_dim: i32,
        ratio: i32,
        start_pos: i32,
        pos_ctr: *const c_int,
        eps: f32,
    ) -> Result<()> {
        let f = self.need(self.kernels.compressor_pool, "dsv41_compressor_pool")?;
        let rc = unsafe {
            f(
                kvp, scp, norm_w, state_kv, state_score, latents, out_rows, b, seqlen, head_dim,
                ratio, start_pos, pos_ctr, eps, self.stream,
            )
        };
        self.kerr(rc, "dsv41_compressor_pool")
    }

    // --------------------------------------------------- glue op wrappers

    pub fn gather_rows(
        &self,
        src: *const f32,
        idx: *const i32,
        out: *mut f32,
        n: i32,
        dim: i32,
    ) -> Result<()> {
        let f = self.need(self.kernels.gather_rows, "dsv41_gather_rows")?;
        let rc = unsafe { f(src, idx, out, n, dim, self.stream) };
        self.kerr(rc, "dsv41_gather_rows")
    }

    pub fn scatter_add_rows(
        &self,
        src: *const f32,
        idx: *const i32,
        weight: *const f32,
        dst: *mut f32,
        n: i32,
        dim: i32,
    ) -> Result<()> {
        let f = self.need(self.kernels.scatter_add_rows, "dsv41_scatter_add_rows")?;
        let rc = unsafe { f(src, idx, weight, dst, n, dim, self.stream) };
        self.kerr(rc, "dsv41_scatter_add_rows")
    }

    pub fn swiglu_limit(&self, gate_up: *mut f32, rows: i32, inter: i32, limit: f32) -> Result<()> {
        let f = self.need(self.kernels.swiglu_limit, "dsv41_swiglu_limit")?;
        let rc = unsafe { f(gate_up, rows, inter, limit, self.stream) };
        self.kerr(rc, "dsv41_swiglu_limit")
    }

    pub fn hc_collapse(
        &self,
        x: *const f32,
        pre: *const f32,
        out: *mut f32,
        rows: i32,
        hc: i32,
        dim: i32,
    ) -> Result<()> {
        let f = self.need(self.kernels.hc_collapse, "dsv41_hc_collapse")?;
        let rc = unsafe { f(x, pre, out, rows, hc, dim, self.stream) };
        self.kerr(rc, "dsv41_hc_collapse")
    }

    /// Stamp `round` into every rank's stamp array (including our own) after the
    /// data peer copies have completed on this stream.
    pub fn ar_stamp(&self, peer_stamps: *const u64, world: i32, rank: i32, round: u32) -> Result<()> {
        let f = self.need(self.kernels.ar_stamp, "dsv41_ar_stamp")?;
        let rc = unsafe { f(peer_stamps, world, rank, round, self.stream) };
        self.kerr(rc, "dsv41_ar_stamp")
    }

    // ---- CUDA graph capture helpers (segment graphs) ----
    /// Begin capturing work queued on this device's stream.
    pub fn capture_begin(&self) -> Result<()> {
        let f = self.need(self.cudart.stream_begin_capture, "cudaStreamBeginCapture")?;
        // cudaStreamCaptureModeRelaxed (2), NOT Global. Global invalidates the
        // capture on ANY CUDA activity in ANY thread of the context — and TP8 runs
        // one rank thread per device with its own all-reduce work, so the other
        // ranks' traffic invalidated every capture (cudaErrorStreamCaptureUnjoined,
        // 901). Relaxed only rejects the capturing thread's own illegal calls.
        let rc = unsafe { f(self.stream, 2 /* cudaStreamCaptureModeRelaxed */) };
        self.kerr(rc, "cudaStreamBeginCapture")
    }

    /// End the capture and return the graph handle.
    pub fn capture_end(&self) -> Result<*mut c_void> {
        let f = self.need(self.cudart.stream_end_capture, "cudaStreamEndCapture")?;
        let mut g: *mut c_void = std::ptr::null_mut();
        let rc = unsafe { f(self.stream, &mut g) };
        self.kerr(rc, "cudaStreamEndCapture")?;
        Ok(g)
    }

    /// Instantiate a captured graph into an executable.
    pub fn graph_instantiate(&self, g: *mut c_void) -> Result<*mut c_void> {
        let f = self.need(self.cudart.graph_instantiate, "cudaGraphInstantiate")?;
        let mut e: *mut c_void = std::ptr::null_mut();
        let rc = unsafe { f(&mut e, g, 0) };
        self.kerr(rc, "cudaGraphInstantiate")?;
        Ok(e)
    }

    pub fn graph_launch(&self, e: *mut c_void) -> Result<()> {
        let f = self.need(self.cudart.graph_launch, "cudaGraphLaunch")?;
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

    /// Publish with a device-side credit wait (replaces the host barrier).
    #[allow(clippy::too_many_arguments)]
    pub fn ar_store2(
        &self,
        peer_slots: *const u64,
        world: i32,
        rank: i32,
        src: *const f32,
        n: i64,
        slot_f: i64,
        parity_off: i64,
        reduced: *const c_uint,
        round: u32,
        peer_stamps: *const u64,
        ctr: *mut c_uint,
    ) -> Result<()> {
        let f = self.need(self.kernels.ar_store2, "dsv41_ar_store2")?;
        let rc = unsafe {
            f(peer_slots, world, rank, src, n, slot_f, parity_off, reduced, round, peer_stamps, ctr,
              self.stream)
        };
        self.kerr(rc, "dsv41_ar_store2")
    }

    /// Reduce with the same-threads release: the kernel that reads the slots also
    /// announces `reduced` once every block is done.
    #[allow(clippy::too_many_arguments)]
    pub fn ar_reduce2(
        &self,
        dst: *mut f32,
        staging: *const f32,
        n: i64,
        slot_f: i64,
        world: i32,
        stamps: *const c_uint,
        round: u32,
        peer_reduced: *const u64,
        rank: i32,
        ctr2: *mut c_uint,
        do_mark: i32,
    ) -> Result<()> {
        let f = self.need(self.kernels.ar_reduce2, "dsv41_ar_reduce2")?;
        let rc = unsafe {
            f(dst, staging, n, slot_f, world, stamps, round, peer_reduced, rank, ctr2, do_mark,
              self.stream)
        };
        self.kerr(rc, "dsv41_ar_reduce2")
    }

    /// Announce that this rank has finished reducing `round`.
    pub fn ar_mark(&self, peer_reduced: *const u64, world: i32, rank: i32, round: u32) -> Result<()> {
        let f = self.need(self.kernels.ar_mark, "dsv41_ar_mark")?;
        let rc = unsafe { f(peer_reduced, world, rank, round, self.stream) };
        self.kerr(rc, "dsv41_ar_mark")
    }

    /// Lean M=1 GEMV over bf16 weights with an f32 activation (out[n] = W[n,k]·x).
    /// Replaces cuBLAS's gemv2T path, which ran at ~40 GFLOP/s for a single row.
    pub fn gemv_bf16(&self, w: *const c_void, x: *const f32, out: *mut f32, n: i32, k: i32) -> Result<()> {
        let f = self.need(self.kernels.gemv_bf16, "dsv41_gemv_bf16")?;
        let rc = unsafe { f(w, x, out, n, k, self.stream) };
        self.kerr(rc, "dsv41_gemv_bf16")
    }

    /// Same, f32 weights.
    /// The decode-step n-gram hash on the device (removes the last per-step H2D
    /// and makes the step graph-capturable). Single thread, bit-identical to the
    /// host reference.
    #[allow(clippy::too_many_arguments)]
    pub fn engram_hash_step(
        &self,
        map: *const i64,
        cache: *mut i64,
        mults: *const i64,
        lms: *const u64,
        offs: *const u64,
        eng_ids: *mut i64,
        token: *const i32,
        pos_ctr: *const i32,
        map_len: i64,
        n_layers: c_int,
        max_ngram: c_int,
        n_heads: c_int,
        pad_id: i64,
    ) -> Result<()> {
        let f = self.need(self.kernels.engram_hash_step, "dsv41_engram_hash_step")?;
        let rc = unsafe {
            f(map, cache, mults, lms, offs, eng_ids, token, pos_ctr, map_len, n_layers, max_ngram,
              n_heads, pad_id, self.stream)
        };
        self.kerr(rc, "dsv41_engram_hash_step")
    }

    /// AR v5 (graph-capturable): store(e) - write my buffer to every peer's
    /// staging half e&1. The epoch is read from DEVICE memory at runtime.
    pub fn ar_v5_store(
        &self,
        peer_slots: *const u64,
        world: c_int,
        rank: c_int,
        src: *const f32,
        n: i64,
        slot_f: i64,
        epoch: *const c_uint,
    ) -> Result<()> {
        let f = self.need(self.kernels.ar_v5_store, "dsv41_ar_v5_store")?;
        let rc = unsafe { f(peer_slots, world, rank, src, n, slot_f, epoch, self.stream) };
        self.kerr(rc, "dsv41_ar_v5_store")
    }

    /// AR v5: publish(e) - stamp e+1 to every peer (system scope), then poll my
    /// own stamps until every peer has published; replaces the host barrier.
    pub fn ar_v5_publish(
        &self,
        peer_stamps: *const u64,
        stamps: *const c_uint,
        world: c_int,
        rank: c_int,
        epoch: *const c_uint,
    ) -> Result<()> {
        let f = self.need(self.kernels.ar_v5_publish, "dsv41_ar_v5_publish")?;
        let rc = unsafe { f(peer_stamps, stamps, world, rank, epoch, self.stream) };
        self.kerr(rc, "dsv41_ar_v5_publish")
    }

    /// AR v5: reduce(e) - sum the peers' staging half e&1 straight into the
    /// caller's buffer; the last block advances the device epoch.
    pub fn ar_v5_reduce(
        &self,
        dst: *mut f32,
        staging: *const f32,
        n: i64,
        slot_f: i64,
        world: c_int,
        epoch: *mut c_uint,
        ctr: *mut c_uint,
    ) -> Result<()> {
        let f = self.need(self.kernels.ar_v5_reduce, "dsv41_ar_v5_reduce")?;
        let rc = unsafe { f(dst, staging, n, slot_f, world, epoch, ctr, self.stream) };
        self.kerr(rc, "dsv41_ar_v5_reduce")
    }

    /// Stable argmax (ties -> lowest index); writes the winning index as i32 and
    /// advances the device position counter (the argmax is the step's last
    /// kernel, so the counter is stable during the step).
    pub fn argmax(&self, v: *const f32, out: *mut c_int, n: i32, pos_ctr: *mut c_int) -> Result<()> {
        let f = self.need(self.kernels.argmax, "dsv41_argmax")?;
        let rc = unsafe { f(v, out, n, pos_ctr, self.stream) };
        self.kerr(rc, "dsv41_argmax")
    }

    /// The decode-step window indices (the trailing window in ring-slot order),
    /// read from the device position counter - replaces the host computation
    /// and its per-layer H2D upload.
    pub fn window_idxs(&self, idxs: *mut i32, pos_ctr: *const c_int, window: i32) -> Result<()> {
        let f = self.need(self.kernels.window_idxs, "dsv41_window_idxs")?;
        let rc = unsafe { f(idxs, pos_ctr, window, self.stream) };
        self.kerr(rc, "dsv41_window_idxs")
    }

    /// Append the KV row into the window ring at the DEVICE-derived slot. This
    /// replaces a cudaMemcpy whose destination address was host-computed and
    /// therefore frozen by a graph capture (every replay wrote the same slot).
    pub fn ring_append(
        &self,
        ring: *mut f32,
        kv: *const f32,
        pos_ctr: *const c_int,
        window: i32,
        hd: i32,
    ) -> Result<()> {
        let f = self.need(self.kernels.ring_append, "dsv41_ring_append")?;
        let rc = unsafe { f(ring, kv, pos_ctr, window, hd, self.stream) };
        self.kerr(rc, "dsv41_ring_append")
    }

    /// The fused compressor commit: reads `out_rows` on the device, ropes the
    /// latent at (*clen) * ratio, stores it into the ring at row window + *clen
    /// and advances the counter. Replaces the host's download + branch + rope +
    /// copy, which was a sync D2H per layer per step and uncapturable.
    #[allow(clippy::too_many_arguments)]
    pub fn compress_commit(
        &self,
        latent: *const f32,
        cos: *const f32,
        sin: *const f32,
        ring: *mut f32,
        out_rows: *const c_int,
        clen: *mut c_int,
        hd: i32,
        rope_dim: i32,
        half: i32,
        window: i32,
        ratio: i32,
    ) -> Result<()> {
        let f = self.need(self.kernels.compress_commit, "dsv41_compress_commit")?;
        let rc = unsafe {
            f(latent, cos, sin, ring, out_rows, clen, hd, rope_dim, half, window, ratio,
              self.stream)
        };
        self.kerr(rc, "dsv41_compress_commit")
    }

    /// The recency placeholder for the compressed rows (the no-indexer safety
    /// net): idxs[win + j] = win + clen - take + j - replaces an upload.
    pub fn comp_placeholder(
        &self,
        idxs: *mut i32,
        clen: *const c_int,
        window: i32,
        index_topk: i32,
    ) -> Result<()> {
        let f = self.need(self.kernels.comp_placeholder, "dsv41_comp_placeholder")?;
        let rc = unsafe { f(idxs, clen, window, index_topk, self.stream) };
        self.kerr(rc, "dsv41_comp_placeholder")
    }

    /// The single 4-byte host read per decode step (EOS check + printing). The
    /// token itself stays on the device; only this value crosses back.
    pub fn download_u32(&self, ptr: *const c_void) -> Result<u32> {
        let mut b = [0u8; 4];
        let st =
            unsafe { (self.cudart.memcpy)(b.as_mut_ptr() as *mut c_void, ptr, 4, CUDA_MEMCPY_D2H) };
        check_cudart(st, &self.cudart, "cudaMemcpy D2H token")?;
        Ok(u32::from_le_bytes(b))
    }

    pub fn gemv_f32(&self, w: *const f32, x: *const f32, out: *mut f32, n: i32, k: i32) -> Result<()> {
        let f = self.need(self.kernels.gemv_f32, "dsv41_gemv_f32")?;
        let rc = unsafe { f(w, x, out, n, k, self.stream) };
        self.kerr(rc, "dsv41_gemv_f32")
    }

    /// Indirect expert gate/up: the weights come from the per-layer pools plus the
    /// device-side expert id, so the launch arguments do not depend on the routing.
    #[allow(clippy::too_many_arguments)]
    pub fn expert_gate_up_fp4_indirect(
        &self,
        a: *const u8,
        a_scale: *const f32,
        out: *mut f32,
        rows: i32,
        dim: i32,
        inter: i32,
        limit: f32,
        w1_base: *const u8,
        w1_stride: i64,
        w1s_base: *const u8,
        w1s_stride: i64,
        w3_base: *const u8,
        w3_stride: i64,
        w3s_base: *const u8,
        w3s_stride: i64,
        ids: *const i32,
        slot: i32,
    ) -> Result<()> {
        let f = self.need(
            self.kernels.expert_gate_up_fp4_indirect,
            "dsv41_expert_gate_up_fp4_indirect",
        )?;
        let rc = unsafe {
            f(
                a, a_scale, out, rows, dim, inter, limit, w1_base, w1_stride, w1s_base, w1s_stride,
                w3_base, w3_stride, w3s_base, w3s_stride, ids, slot, self.stream,
            )
        };
        self.kerr(rc, "dsv41_expert_gate_up_fp4_indirect")
    }

    /// Indirect expert down; accumulates into `out`.
    #[allow(clippy::too_many_arguments)]
    pub fn expert_down_fp4_indirect(
        &self,
        act: *const f32,
        out: *mut f32,
        rows: i32,
        dim: i32,
        inter: i32,
        row_weight: *const f32,
        w2_base: *const u8,
        w2_stride: i64,
        w2s_base: *const u8,
        w2s_stride: i64,
        ids: *const i32,
        slot: i32,
    ) -> Result<()> {
        let f = self.need(
            self.kernels.expert_down_fp4_indirect,
            "dsv41_expert_down_fp4_indirect",
        )?;
        let rc = unsafe {
            f(
                act, out, rows, dim, inter, row_weight, w2_base, w2_stride, w2s_base, w2s_stride,
                ids, slot, self.stream,
            )
        };
        self.kerr(rc, "dsv41_expert_down_fp4_indirect")
    }

    /// Publish `src` into every rank's staging slot for this rank, from the device.
    pub fn ar_store(
        &self,
        peer_slots: *const u64,
        world: i32,
        rank: i32,
        src: *const f32,
        n: i64,
        slot_f: i64,
    ) -> Result<()> {
        let f = self.need(self.kernels.ar_store, "dsv41_ar_store")?;
        let rc = unsafe { f(peer_slots, world, rank, src, n, slot_f, self.stream) };
        self.kerr(rc, "dsv41_ar_store")
    }

    /// Sum the `world` staging slots into `dst`, spinning on the local stamps
    /// until every peer has published `round` — the host never waits.
    pub fn ar_reduce(
        &self,
        dst: *mut f32,
        staging: *const f32,
        n: i64,
        slot_f: i64,
        world: i32,
        stamps: *const c_uint,
        round: u32,
    ) -> Result<()> {
        let f = self.need(self.kernels.ar_reduce, "dsv41_ar_reduce")?;
        let rc = unsafe { f(dst, staging, n, slot_f, world, stamps, round, self.stream) };
        self.kerr(rc, "dsv41_ar_reduce")
    }

    pub fn engram_apply(
        &self,
        x: *mut f32,
        kv: *const f32,
        q_weight: *const f32,
        k_weight: *const f32,
        token_mask: *const u8,
        rows: i32,
        hc: i32,
        dim: i32,
        eps: f32,
    ) -> Result<()> {
        let f = self.need(self.kernels.engram_apply, "dsv41_engram_apply")?;
        let rc = unsafe { f(x, kv, q_weight, k_weight, token_mask, rows, hc, dim, eps, self.stream) };
        self.kerr(rc, "dsv41_engram_apply")
    }

    // ------------------------------------------- read-only GLM reuse set

    pub fn rmsnorm(
        &self,
        x: *const f32,
        w: *const f32,
        out: *mut f32,
        n: i32,
        dim: i32,
        eps: f32,
    ) -> Result<()> {
        let rc = unsafe { (self.kernels.rmsnorm)(x, w, out, n, dim, eps, self.stream) };
        self.kerr(rc, "ferrite_rmsnorm")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn hc_pre(
        &self,
        res: *const f32,
        fw: *const f32,
        scale: *const f32,
        base: *const f32,
        li: *mut f32,
        post: *mut f32,
        comb: *mut f32,
        s: i32,
        n: i32,
        h: i32,
        mix: i32,
        rms_eps: f32,
        hc_eps: f32,
        iters: i32,
    ) -> Result<()> {
        let rc = unsafe {
            (self.kernels.hc_pre)(
                res, fw, scale, base, li, post, comb, s, n, h, mix, rms_eps, hc_eps, iters,
                self.stream,
            )
        };
        self.kerr(rc, "ferrite_hc_pre")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn hc_post(
        &self,
        x: *const f32,
        res: *const f32,
        post: *const f32,
        comb: *const f32,
        out: *mut f32,
        s: i32,
        n: i32,
        h: i32,
    ) -> Result<()> {
        let rc = unsafe { (self.kernels.hc_post)(x, res, post, comb, out, s, n, h, self.stream) };
        self.kerr(rc, "ferrite_hc_post")
    }

    /// Embedding gather + hc expansion, straight onto the residual stream.
    pub fn embed_expand_dev(
        &self,
        table: *const c_void,
        ids_dev: *const c_int,
        out: *mut f32,
        n: i32,
        hidden: i32,
        mult: i32,
        vocab: i32,
    ) -> Result<()> {
        let rc = unsafe {
            (self.kernels.embed_expand_dev)(table, ids_dev, out, n, hidden, mult, vocab, self.stream)
        };
        self.kerr(rc, "ferrite_embed_expand_dev")
    }

    /// bf16 -> f32 on the device (GLM's kernel). Used to widen weights without
    /// staging them through host memory.
    pub fn bf16_to_f32(&self, src: *const c_void, dst: *mut c_void, n: i64) -> Result<()> {
        let rc = unsafe { (self.kernels.bf16_to_f32)(src, dst, n, self.stream) };
        self.kerr(rc, "ferrite_bf16_to_f32")?;
        // load-time op: synchronise so a fault inside the kernel is attributed
        // here rather than to whatever call happens to run next
        self.dev_sync().map_err(|e| {
            FerriteError::Config(format!("ferrite_bf16_to_f32 (n={n}): {e}"))
        })
    }

    pub fn f32_to_bf16(&self, src: *const f32, dst: *mut c_void, n: i64) -> Result<()> {
        let rc = unsafe { (self.kernels.f32_to_bf16)(src, dst, n, self.stream) };
        self.kerr(rc, "ferrite_f32_to_bf16")
    }

    /// f32 x f32 -> f32 GEMM. Used where the release runs a projection in fp32
    /// (the compressor's, for the pooling's accuracy): the checkpoint stores
    /// those as bf16, widening is exact, so this reproduces the promotion.
    pub fn gemm_f32(
        &self,
        x: *const c_void,
        w: *const c_void,
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
                CUDA_R_32F,
                k,
                x,
                CUDA_R_32F,
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
            return Err(FerriteError::Config(format!("cublasGemmEx(f32): {st}")));
        }
        Ok(())
    }

    /// bf16 x bf16 -> f32 GEMM for the weights that are natively bf16 (the
    /// head, the compressor and indexer projections, the vision tower).
    /// This is not a dequantisation: those tensors are bf16 in the checkpoint.
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
                CUDA_R_16BF,
                k,
                x,
                CUDA_R_16BF,
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
                    "ASYNC FAULT in gemm_bf16 (cuBLAS): {msg}"
                )));
            }
        }
        Ok(())
    }
}

impl Drop for DevBuf {
    fn drop(&mut self) {
        // The owning Device frees; DevBuf itself holds no handle, so freeing
        // happens through `Device::free`.
    }
}

impl Device {
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
                eprintln!("[dsv41] cudaFree({:p}) failed: {msg}", b.ptr);
            }
        }
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
