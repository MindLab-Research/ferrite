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
            Ok(DevRuntime {
                cudart,
                cublas,
                stream,
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
    pub fn graph_instantiate(&self, g: *mut c_void) -> Result<*mut c_void> {
        let f = self
            .cudart
            .graph_instantiate
            .ok_or_else(|| FerriteError::Config("cudaGraphInstantiate missing".into()))?;
        let mut e: *mut c_void = std::ptr::null_mut();
        let rc = unsafe { f(&mut e, g, 0) };
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
