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

use std::ffi::{c_char, c_int, c_long, c_uint, c_void, CString};

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
        unsafe { std::mem::transmute_copy(&sym($h, $name)?) }
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
