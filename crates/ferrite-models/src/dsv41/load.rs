//! Real-checkpoint loader: safetensors -> device buffers, in the checkpoint's
//! own formats.
//!
//! Config-driven: the tensor list comes from [`crate::dsv41::weights::tensor_specs`]
//! (itself derived from [`crate::dsv41::config::Dsv41Config`]), so a different
//! geometry/TP degree needs no code change here.
//!
//! **No dequantisation.** Whatever the file holds is what lands in device
//! memory: fp8 e4m3 weights next to their ue8m0 32x32 scale tensors, fp4 e2m1
//! weights (I8-packed, 2 values per byte) next to their per-(row, 32) e8m0
//! scales, and bf16/f32 tensors untouched. Slicing for tensor parallelism is a
//! pure byte-range operation.

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use ferrite_types::{FerriteError, Result};

use crate::dsv41::config::Dsv41Config;
use crate::dsv41::device::{DevBuf, Device};
use crate::dsv41::weights::{gateup_ilv, local_shape, tensor_specs, SafetensorsIndex, Shard, TensorSpec};

/// Tensors whose consumer is a bf16 tensor-core GEMM: they stay bf16 verbatim.
/// Everything else that arrives as bf16 is widened to f32 on the way in —
/// an exact conversion (bf16 -> f32 is lossless), NOT a dequantisation. It is
/// required because the consuming kernels (rmsnorm weights, the hyper-connection
/// parameters, the routing bias, the embedding table) take f32 pointers;
/// handing them bf16 bytes made each of them read twice its length, which
/// corrupted values and walked off the end of the allocation.
const KEEP_BF16: &[&str] = &[
    "ffn.gate.weight",
    "attn.indexer.wq_b.weight",
    "attn.indexer.wk.weight",
    "attn.indexer.weights_proj.weight",
];

/// Byte-level bf16 widening (the loader's other call paths use it).
#[allow(dead_code)]
fn bf16_to_f32_bytes(b: &[u8]) -> Vec<u8> {
    let n = b.len() / 2;
    let mut out = Vec::with_capacity(n * 4);
    for i in 0..n {
        let h = u16::from_le_bytes([b[i * 2], b[i * 2 + 1]]);
        let f = f32::from_bits((h as u32) << 16);
        out.extend_from_slice(&f.to_le_bytes());
    }
    out
}

fn io<T>(r: std::io::Result<T>, what: &str) -> Result<T> {
    r.map_err(|e| FerriteError::Config(format!("{what}: {e}")))
}

/// On-disk element sizes for the dtypes the release uses.
fn dtype_size(dt: &str) -> usize {
    match dt {
        "F32" => 4,
        "BF16" => 2,
        "F8_E4M3" | "F8_E5M2" | "F8_E8M0" | "I8" | "U8" => 1,
        "I32" | "U32" => 4,
        "I64" => 8,
        _ => 1,
    }
}


/// Everything needed to place one tensor's slice on the device WITHOUT
/// allocating: the geometry of the local slice plus where it lives in the
/// mmap'd shard. Extracted from load_tensor so the expert pool can plan
/// thousands of slices, allocate ONE buffer, then DMA them all in.
pub struct TensorPlan {
    pub local: Vec<usize>,
    pub bytes: usize,
    pub dtype: String,
    pub shard: String,
    pub begin: u64,
    pub global: Vec<usize>,
    pub shard_rule: Shard,
    pub widen: bool,
}

impl TensorPlan {
    pub fn esz(&self) -> usize {
        dtype_size(&self.dtype)
    }
}

/// A tensor living on the device in its checkpoint format.
#[derive(Clone)]
pub struct DevTensor {
    pub buf: DevBuf,
    pub shape: Vec<usize>,
    pub dtype: String,
}

impl DevTensor {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
    pub fn ptr(&self) -> *mut std::ffi::c_void {
        self.buf.ptr
    }
    pub fn as_u8(&self) -> *const u8 {
        self.buf.ptr as *const u8
    }
    pub fn as_f32(&self) -> *const f32 {
        self.buf.ptr as *const f32
    }
    pub fn as_mut_f32(&self) -> *mut f32 {
        self.buf.ptr as *mut f32
    }
    pub fn as_i32(&self) -> *const i32 {
        self.buf.ptr as *const i32
    }
}

/// One expert's on-device weights. The tensors are VIEWS into a single pooled
/// allocation per layer — one cudaMalloc per layer instead of six per expert
/// (40 layers x 384 experts x 6 = ~92k individual allocations was exhausting the
/// 4 GB host's driver bookkeeping, which is what made a 16 MiB cudaMalloc 'fail'
/// with 186 GB free on the device).
pub struct DevExpert {
    pub w1: DevTensor,
    pub w1_scale: DevTensor,
    pub w3: DevTensor,
    pub w3_scale: DevTensor,
    pub w2: DevTensor,
    pub w2_scale: DevTensor,
}

/// A big device allocation that experts are carved out of. Dropped when the
/// weights are dropped, which frees the whole pool at once.
pub struct ExpertPool(pub DevBuf);

/// Per-layer device weights. Every optional group mirrors a config test
/// (`is_kv_source`, `is_index_source`, engram layers, ...).
#[derive(Default)]
pub struct LayerDev {
    pub hc_attn_fn: Option<DevTensor>,
    pub hc_attn_base: Option<DevTensor>,
    pub hc_attn_scale: Option<DevTensor>,
    pub hc_ffn_fn: Option<DevTensor>,
    pub hc_ffn_base: Option<DevTensor>,
    pub hc_ffn_scale: Option<DevTensor>,
    pub attn_norm: Option<DevTensor>,
    pub ffn_norm: Option<DevTensor>,
    pub wq_a: Option<DevTensor>,
    pub wq_a_scale: Option<DevTensor>,
    pub q_norm: Option<DevTensor>,
    pub wq_b: Option<DevTensor>,
    pub wq_b_scale: Option<DevTensor>,
    pub wkv: Option<DevTensor>,
    pub wkv_scale: Option<DevTensor>,
    pub kv_norm: Option<DevTensor>,
    pub wo_a: Option<DevTensor>,
    pub wo_a_scale: Option<DevTensor>,
    pub wo_b: Option<DevTensor>,
    pub wo_b_scale: Option<DevTensor>,
    pub attn_sink: Option<DevTensor>,
    /// compressor (kv sources only): wkv fp8 + (ratio>1) wgate fp8 + norm
    pub comp_wkv: Option<DevTensor>,
    pub comp_wkv_scale: Option<DevTensor>,
    pub comp_wgate: Option<DevTensor>,
    pub comp_wgate_scale: Option<DevTensor>,
    pub comp_norm: Option<DevTensor>,
    /// indexer (index sources only)
    pub idx_wq_b: Option<DevTensor>,
    pub idx_wq_b_scale: Option<DevTensor>,
    pub idx_weights: Option<DevTensor>,
    pub idx_wk: Option<DevTensor>,
    pub idx_k_norm: Option<DevTensor>,
    pub gate_w: Option<DevTensor>,
    pub gate_w_scale: Option<DevTensor>,
    pub gate_bias: Option<DevTensor>,
    pub gate_bias_vl: Option<DevTensor>,
    pub experts: Vec<DevExpert>,
    /// owns the memory the experts' tensors view into
    pub expert_pool: Option<DevBuf>,
    /// The routed experts' w1/w3 are stored INTERLEAVED in one region
    /// (DSV41_EXPERT_ILV). The consumer MUST pass `ilv = 1` to the batched
    /// gate/up launcher; the sequential (unfused) fallback cannot read this
    /// layout and refuses to run on it.
    pub experts_ilv: bool,
    pub shared_w1: Option<DevTensor>,
    pub shared_w1_scale: Option<DevTensor>,
    pub shared_w3: Option<DevTensor>,
    pub shared_w3_scale: Option<DevTensor>,
    pub shared_w2: Option<DevTensor>,
    pub shared_w2_scale: Option<DevTensor>,
    /// engram (this layer carries a table)
    pub engram_embed: Option<DevTensor>,
    pub engram_embed_scale: Option<DevTensor>,
    pub engram_wkv: Option<DevTensor>,
    pub engram_wkv_scale: Option<DevTensor>,
    pub engram_q_weight: Option<DevTensor>,
    pub engram_k_weight: Option<DevTensor>,
}

/// All weights a rank needs.
#[derive(Default)]
pub struct Dsv41DevWeights {
    pub embed: Option<DevTensor>,
    pub norm: Option<DevTensor>,
    pub head: Option<DevTensor>,
    pub layers: Vec<LayerDev>,
    pub mtp: Vec<LayerDev>,
    /// `mtp.0.main_proj` / `main_norm`
    pub main_proj: Option<DevTensor>,
    pub main_proj_scale: Option<DevTensor>,
    pub main_norm: Option<DevTensor>,
    /// `mtp.last.norm`
    pub dspark_norm: Option<DevTensor>,
    pub markov_embed: Option<DevTensor>,
    pub markov_head: Option<DevTensor>,
    pub confidence_proj: Option<DevTensor>,
    /// vision (replicated)
    pub vision: Vec<(String, DevTensor)>,
}

/// Reads a checkpoint shard-by-shard on demand, uploading the rank's slice.
pub struct Loader<'a> {
    dir: PathBuf,
    /// tensor name -> shard file
    index: HashMap<String, String>,
    /// shard file -> header
    headers: HashMap<String, SafetensorsIndex>,
    /// shard file -> open handle, kept for the whole load. Re-opening per tensor
    /// cost ~2 syscalls x ~96k tensors x 8 ranks and dominated the load time
    /// (410 s for 38.6 GiB/rank); the handles and the data-section offsets are
    /// both cached now.
    files: HashMap<String, std::fs::File>,
    /// shard file -> byte offset of the data section
    data_base: HashMap<String, u64>,
    /// shard file -> read-only mmap, so tensors go to the GPU by DMA
    maps: HashMap<String, (*const u8, usize)>,
    /// (bytes, name) per tensor, for `DSV41_LOAD_TRACE`
    trace: Vec<(usize, String)>,
    dev: &'a Device,
    /// bytes uploaded so far (for the report)
    pub uploaded: u64,
    /// tensor-name prefixes to skip entirely (bring-up: the 189 GiB engram
    /// tables while the engram write-back is not yet wired)
    pub skip_prefixes: Vec<String>,
}

impl<'a> Loader<'a> {
    pub fn new(dir: &Path, dev: &'a Device) -> Result<Self> {
        let idx_path = dir.join("model.safetensors.index.json");
        let txt = std::fs::read_to_string(&idx_path).map_err(|e| {
            FerriteError::Config(format!("read {}: {e}", idx_path.display()))
        })?;
        let v: serde_json::Value = serde_json::from_str(&txt)
            .map_err(|e| FerriteError::Config(format!("index json: {e}")))?;
        let mut index = HashMap::new();
        for (k, shard) in v["weight_map"]
            .as_object()
            .ok_or_else(|| FerriteError::Config("no weight_map".into()))?
        {
            if let Some(s) = shard.as_str() {
                index.insert(k.clone(), s.to_string());
            }
        }
        Ok(Loader {
            dir: dir.to_path_buf(),
            index,
            headers: HashMap::new(),
            files: HashMap::new(),
            data_base: HashMap::new(),
            maps: HashMap::new(),
            trace: Vec::new(),
            dev,
            uploaded: 0,
            skip_prefixes: Vec::new(),
        })
    }

    fn header(&mut self, name: &str) -> Result<&SafetensorsIndex> {
        let shard = self
            .index
            .get(name)
            .ok_or_else(|| FerriteError::Config(format!("{name} not in the checkpoint")))?
            .clone();
        if !self.headers.contains_key(&shard) {
            let h = SafetensorsIndex::read_header(&self.dir.join(&shard))?;
            self.headers.insert(shard.clone(), h);
        }
        Ok(self.headers.get(&shard).unwrap())
    }

    /// Raw bytes of one contiguous run inside a tensor's data.
    fn read_at(&mut self, name: &str, off: u64, len: usize) -> Result<Vec<u8>> {
        let shard = self.index.get(name).unwrap().clone();
        let h = self.header(name)?.tensors.get(name).cloned().ok_or_else(|| {
            FerriteError::Config(format!("{name} missing in {shard}"))
        })?;
        if !self.files.contains_key(&shard) {
            let path = self.dir.join(&shard);
            let mut f = io(File::open(&path), "open shard")?;
            let mut b = [0u8; 8];
            io(f.read_exact(&mut b), "read header len")?;
            let hlen = u64::from_le_bytes(b);
            self.data_base.insert(shard.clone(), 8 + hlen);
            self.files.insert(shard.clone(), f);
        }
        let base = *self.data_base.get(&shard).unwrap() + h.begin;
        let f = self.files.get_mut(&shard).unwrap();
        io(f.seek(SeekFrom::Start(base + off)), "seek")?;
        let mut buf = vec![0u8; len];
        io(f.read_exact(&mut buf), "read tensor bytes")?;
        self.uploaded += len as u64;
        Ok(buf)
    }

    /// Upload `spec`'s slice for `rank` of `world`.
    /// mmap a shard once; every tensor read comes straight out of it, so the
    /// data reaches the GPU by DMA instead of a read() into a host buffer.
    fn map_shard(&mut self, shard: &str) -> Result<(*const u8, usize)> {
        if !self.maps.contains_key(shard) {
            let path = self.dir.join(shard);
            let f = io(File::open(&path), "open shard for mmap")?;
            let len = io(f.metadata(), "stat shard")?.len() as usize;
            let p = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    len,
                    libc::PROT_READ,
                    libc::MAP_SHARED,
                    std::os::unix::io::AsRawFd::as_raw_fd(&f),
                    0,
                )
            };
            if p == libc::MAP_FAILED {
                return Err(FerriteError::Config(format!("mmap {path:?} failed")));
            }
            // the header length gives the data-section base
            let hdr = unsafe { std::slice::from_raw_parts(p as *const u8, 8) };
            let hlen = u64::from_le_bytes(hdr.try_into().unwrap());
            self.data_base.insert(shard.to_string(), 8 + hlen);
            self.maps.insert(shard.to_string(), (p as *const u8, len));
        }
        Ok(*self.maps.get(shard).unwrap())
    }

    /// Load one tensor onto the device.
    ///
    /// GPU DMA throughout (the GLM approach): the zeroed device buffer receives
    /// the slice with `cudaMemcpy` (row-contiguous slices) or `cudaMemcpy2D`
    /// (column slices, whose rows are strided), and a bf16 tensor that must be
    /// widened is widened ON THE DEVICE. Nothing is staged through host memory.
    /// `DSV41_CPU_LOAD=1` restores the older read-into-a-Vec path as a fallback.
    fn load_tensor(&mut self, spec: &TensorSpec, world: usize, rank: usize) -> Result<DevTensor> {
        if self
            .skip_prefixes
            .iter()
            .any(|p| spec.name.starts_with(p.as_str()))
        {
            return Ok(DevTensor {
                buf: self.dev.alloc(4)?,
                shape: vec![0],
                dtype: "SKIPPED".into(),
            });
        }
        let cpu_path = std::env::var("DSV41_CPU_LOAD").map(|v| v != "0").unwrap_or(false);
        let h = self.header(&spec.name)?.tensors.get(&spec.name).cloned().ok_or_else(|| {
            FerriteError::Config(format!("{} absent", spec.name))
        })?;
        let esz = dtype_size(&h.dtype);
        // Take the GLOBAL shape from the checkpoint header, not from the caller's
        // spec: several shard rules read spec.shape, and a caller that passes an
        // empty shape (as the parity tests do) would then slice wrongly.
        let spec_full = TensorSpec {
            name: spec.name.clone(),
            shape: h.shape.clone(),
            shard: spec.shard.clone(),
        };
        let local = local_shape(&Dsv41Config::production(), &spec_full, world, rank);
        let n_local: usize = local.iter().product();
        let global = &h.shape;
        let inner: usize = global[1..].iter().product();
        let row_bytes = inner * esz;
        // The head is kept in bf16 by EXACT name, not the KEEP_BF16 suffix list:
        // "head.weight" is also the suffix of "markov_head.head.weight", whose
        // consumer expects the widened f32 layout. The head gemv widens each
        // weight losslessly in-kernel (bf16 is a truncated f32), so the logits
        // stay bit-identical while the step's largest single weight read - the
        // full-vocab head, replicated on every rank - halves.
        let widen = h.dtype == "BF16" && spec.name != "head.weight"
            && !KEEP_BF16.iter().any(|k| spec.name.ends_with(k));
        let out_bytes = n_local * if widen { 4 } else { esz };

        self.uploaded += out_bytes as u64;
        if std::env::var("DSV41_LOAD_TRACE").map(|v| v != "0").unwrap_or(false) {
            self.trace.push((out_bytes, spec.name.clone()));
        }
        // destination, pre-zeroed so any padding the slice needs stays zero
        let buf = self.dev.alloc(out_bytes)?;
        self.dev.zero_at(buf.ptr, out_bytes)?;
        // where the bytes land: the output, or a bf16 scratch when widening
        let scratch = if widen { Some(self.dev.alloc(n_local * 2)?) } else { None };
        let dst = scratch.as_ref().map(|b| b.ptr).unwrap_or(buf.ptr);

        // global slice coordinates (identical to the CPU path)
        let (row0, rows) = match spec.shard {
            Shard::Replicated => (0usize, global[0]),
            Shard::Rows => {
                if spec.name.contains("engram.embed") {
                    let per = global[0].div_ceil(world);
                    (rank * per, per.min(global[0].saturating_sub(rank * per)))
                } else {
                    let per = global[0] / world;
                    (rank * per, per)
                }
            }
            Shard::Heads | Shard::Groups | Shard::Experts | Shard::ExpertRows => {
                let per = global[0] / world;
                (rank * per, per)
            }
            Shard::Cols | Shard::ExpertCols => (0, global[0]),
        };

        if cpu_path {
            // ---- fallback: read into a host buffer, then upload ----
            let raw = if matches!(spec.shard, Shard::Cols | Shard::ExpertCols) {
                let per = inner / world;
                let mut out = Vec::with_capacity(global[0] * per * esz);
                for r in 0..global[0] {
                    let off = (r * inner + rank * per) as u64 * esz as u64;
                    out.extend_from_slice(&self.read_at(&spec.name, off, per * esz)?);
                }
                out
            } else {
                self.read_at(&spec.name, row0 as u64 * row_bytes as u64, rows * row_bytes)?
            };
            // the CPU path fills the SAME `dst` the DMA path would (the bf16
            // scratch when widening); the device-side conversion below then
            // produces the f32 in `buf`. Widening on the host here too would
            // convert twice.
            let dstb = Device::view(dst, raw.len().min(out_bytes));
            self.dev.upload_bytes_at(&dstb, &raw)?;
        } else {
            // ---- DMA path ----
            let shard = self.index.get(&spec.name).unwrap().clone();
            let (base, _len) = self.map_shard(&shard)?;
            // data_base = the shard's data-section start; h.begin = THIS tensor's
            // offset inside it. Omitting h.begin made every tensor read from the
            // start of the data section (garbage values, and reads past the end
            // of the mapping for the later tensors).
            let data = base
                .wrapping_add(*self.data_base.get(&shard).unwrap() as usize)
                .wrapping_add(h.begin as usize);
            match spec.shard {
                Shard::Cols | Shard::ExpertCols => {
                    let per = inner / world;
                    self.dev.upload_from_2d(
                        dst,
                        local[1] * esz,
                        data.wrapping_add(rank * per * esz) as *const std::ffi::c_void,
                        row_bytes,
                        per * esz,
                        global[0],
                    )?;
                }
                _ => {
                    self.dev.upload_from(
                        dst,
                        data.wrapping_add(row0 * row_bytes) as *const std::ffi::c_void,
                        rows * row_bytes,
                    )?;
                }
            }
        }

        if widen {
            let tmp = scratch.unwrap();
            self.dev.bf16_to_f32(tmp.ptr as *const std::ffi::c_void, buf.ptr, n_local as i64)?;
            self.dev.free(&tmp);
        }
        Ok(DevTensor {
            buf,
            shape: local,
            dtype: if widen { "F32".into() } else { h.dtype },
        })
    }


    /// Plan a tensor's placement: computes the local slice geometry exactly as
    /// load_tensor does, but allocates nothing. The expert pool plans all of a
    /// layer's experts first, then allocates one block for all of them.
    fn plan_tensor(&mut self, spec: &TensorSpec, world: usize) -> Result<TensorPlan> {
        let h = self
            .header(&spec.name)?
            .tensors
            .get(&spec.name)
            .cloned()
            .ok_or_else(|| FerriteError::Config(format!("{} absent", spec.name)))?;
        // local_shape reads the GLOBAL shape off the spec — the caller passes
        // the checkpoint's shape, not the empty placeholder
        let spec_full = TensorSpec {
            name: spec.name.clone(),
            shape: h.shape.clone(),
            shard: spec.shard.clone(),
        };
        let local = local_shape(&Dsv41Config::production(), &spec_full, world, 0);
        let widen = h.dtype == "BF16" && !KEEP_BF16.iter().any(|k| spec.name.ends_with(k));
        let bytes = local.iter().product::<usize>() * if widen { 4 } else { dtype_size(&h.dtype) };
        Ok(TensorPlan {
            local,
            bytes,
            dtype: h.dtype.clone(),
            shard: self.index.get(&spec.name).unwrap().clone(),
            begin: h.begin,
            global: h.shape,
            shard_rule: spec.shard.clone(),
            widen,
        })
    }

    /// DMA a planned tensor's slice into `dst` (which must be at least
    /// `plan.bytes` for the raw form, or `plan.bytes` f32 when widening).
    /// This is the transfer half of load_tensor; the expert pool calls it with
    /// offsets into its single allocation.
    fn dma_plan(&mut self, plan: &TensorPlan, dst: *mut std::ffi::c_void, world: usize, rank: usize) -> Result<()> {
        let esz = plan.esz();
        let inner: usize = plan.global[1..].iter().product();
        let row_bytes = inner * esz;
        // widen needs a bf16 staging area then an on-device conversion
        let (target, scratch) = if plan.widen {
            let sc = self.dev.alloc(plan.local.iter().product::<usize>() * 2)?;
            (sc.ptr, Some(sc))
        } else {
            (dst, None)
        };
        let (row0, rows) = match plan.shard_rule {
            Shard::Replicated => (0usize, plan.global[0]),
            Shard::Rows => {
                if plan.shard.contains("engram.embed") {
                    let per = plan.global[0].div_ceil(world);
                    (rank * per, per.min(plan.global[0].saturating_sub(rank * per)))
                } else {
                    let per = plan.global[0] / world;
                    (rank * per, per)
                }
            }
            Shard::Heads | Shard::Groups | Shard::Experts | Shard::ExpertRows => {
                let per = plan.global[0] / world;
                (rank * per, per)
            }
            Shard::Cols | Shard::ExpertCols => (0, plan.global[0]),
        };
        let (map_base, _) = self.map_shard(&plan.shard)?;
        let data = map_base
            .wrapping_add(*self.data_base.get(&plan.shard).unwrap_or(&0) as usize)
            .wrapping_add(plan.begin as usize);
        match plan.shard_rule {
            Shard::Cols | Shard::ExpertCols => {
                let per = inner / world;
                self.dev.upload_from_2d(
                    target,
                    plan.local[1] * esz,
                    data.wrapping_add(rank * per * esz) as *const std::ffi::c_void,
                    row_bytes,
                    per * esz,
                    plan.global[0],
                )?;
            }
            _ => {
                self.dev.upload_from(
                    target,
                    data.wrapping_add(row0 * row_bytes) as *const std::ffi::c_void,
                    rows * row_bytes,
                )?;
            }
        }
        if plan.widen {
            let sc = scratch.unwrap();
            let n: i64 = plan.local.iter().product::<usize>() as i64;
            self.dev.bf16_to_f32(sc.ptr as *const std::ffi::c_void, dst, n)?;
            self.dev.free(&sc);
        }
        Ok(())
    }


    /// All of one layer's experts in ONE device allocation.
    ///
    /// ~92k individual cudaMallocs (384 experts x 6 tensors x 40 layers x ...)
    /// exhaust a 4 GB host's driver bookkeeping — a 16 MiB cudaMalloc 'fails'
    /// with 186 GB free on the device. One pooled buffer per layer instead.
    /// Expert weights are fp4-packed and their scales e8m0, never bf16, so the
    /// widening pass never triggers here.
    ///
    /// `ilv` (DSV41_EXPERT_ILV, decided by [`Loader::ilv_ok`]) stores w1 (gate)
    /// and w3 (up) in ONE region with an 8-byte granule alternation
    /// (`gate[0..8] + up[0..8] + gate[8..16] + ...`) so the fused gate/up GEMV
    /// fetches both halves of a row chunk with ONE LDG.128 instead of two
    /// LDG.64s. The pool's TOTAL size and every other plane's bytes are
    /// unchanged — this is a permutation plus a per-expert reorder, and it is
    /// bit-identical by construction (the same bytes, decoded the same way).
    /// The scales stay in their own blocks: the kernel reads gate/up scales from
    /// separate pointers regardless of how the weights are stored.
    fn load_expert_pool(
        &mut self,
        prefix: &str,
        n_routed: usize,
        _layer: usize,
        world: usize,
        rank: usize,
        ilv: bool,
    ) -> Result<(Vec<DevExpert>, DevBuf, bool)> {
        const NAMES: [&str; 6] = [
            "w1.weight", "w1.scale", "w3.weight", "w3.scale", "w2.weight", "w2.scale",
        ];
        // pass 1: plan every slice (no allocation)
        let mut plans: Vec<TensorPlan> = Vec::with_capacity(n_routed * 6);
        for e in 0..n_routed {
            for n in NAMES {
                let name = format!("{prefix}.ffn.experts.{e}.{n}");
                let spec = TensorSpec {
                    name: name.clone(),
                    shape: vec![], // unused by plan_tensor (it reads the header)
                    shard: if n.starts_with("w2") { Shard::ExpertCols } else { Shard::ExpertRows },
                };
                plans.push(self.plan_tensor(&spec, world)?);
            }
        }
        let total: usize = plans.iter().map(|p| p.bytes).sum();
        // The interleave needs the gate and up planes to be byte-identical (they
        // are: both are ExpertRows [padded_inter, dim/2]) and an 8-byte granule.
        // Any surprise falls back to the plain layout instead of mis-indexing.
        let w1b = if plans.len() >= 3 { plans[0].bytes } else { 0 };
        let ilv = ilv
            && n_routed > 0
            && plans.len() == n_routed * 6
            && w1b > 0
            && w1b % 8 == 0
            && plans[2].bytes == w1b
            && plans[0].local.len() == 2
            && plans[0].local[1] * dtype_size(&plans[0].dtype) % 8 == 0
            && !plans[0].widen
            && !plans[2].widen;
        // Per-expert byte offsets of the six planes.
        //   plain        : [w1][w1.scale][w3][w3.scale][w2][w2.scale]
        //   interleaved  : [w1||w3 interleaved][w1.scale][w3.scale][w2][w2.scale]
        // (w3.weight shares offset 0 with w1.weight in the interleaved case: one
        // region holds both, and the kernel derives the up bytes from the gate
        // pointer. Rust's per-expert STRIDES are pointer differences between
        // experts, which stay uniform in both layouts.)
        let (block, poff) = if ilv {
            let mut o = [0usize; 6];
            o[1] = 2 * w1b;
            o[3] = o[1] + plans[1].bytes;
            o[4] = o[3] + plans[3].bytes;
            o[5] = o[4] + plans[4].bytes;
            (o[5] + plans[5].bytes, o)
        } else {
            let mut o = [0usize; 6];
            let mut acc = 0usize;
            for (k, slot) in o.iter_mut().enumerate() {
                *slot = acc;
                acc += plans[k].bytes;
            }
            (acc, o)
        };
        debug_assert!(block * n_routed <= total.max(1));
        // one allocation; the K padding is already zero
        let pool = self.dev.alloc(total.max(1))?;
        self.dev.zero_at(pool.ptr, total.max(1))?;
        let base = pool.ptr as *mut u8;
        let mut views: Vec<DevTensor> = Vec::with_capacity(plans.len());
        // Interleaved: the gate/up pair is staged in a scratch (the permute
        // kernel may not write into either source). Zeroed ONCE: `dma_plan`
        // writes only the real rows, so the K padding rows stay zero in the
        // scratch and every expert's doubled region is rebuilt from them.
        let tmp = if ilv { Some(self.dev.alloc(2 * w1b)?) } else { None };
        if let Some(t) = &tmp {
            self.dev.zero_at(t.ptr, 2 * w1b)?;
        }
        for e in 0..n_routed {
            let i = e * 6;
            let pb = base.wrapping_add(e * block);
            if let Some(t) = &tmp {
                let tpb = t.ptr as *mut u8;
                self.dma_plan(&plans[i], tpb as *mut std::ffi::c_void, world, rank)?;
                self.dma_plan(
                    &plans[i + 2],
                    tpb.wrapping_add(w1b) as *mut std::ffi::c_void,
                    world,
                    rank,
                )?;
                self.dev.interleave_gateup_fp4(
                    tpb as *const u8,
                    tpb.wrapping_add(w1b) as *const u8,
                    pb,
                    w1b as i64,
                )?;
                for k in [1usize, 3, 4, 5] {
                    self.dma_plan(
                        &plans[i + k],
                        pb.wrapping_add(poff[k]) as *mut std::ffi::c_void,
                        world,
                        rank,
                    )?;
                }
            } else {
                for k in 0..6 {
                    self.dma_plan(
                        &plans[i + k],
                        pb.wrapping_add(poff[k]) as *mut std::ffi::c_void,
                        world,
                        rank,
                    )?;
                }
            }
            // views. k == 0 (w1) and k == 2 (w3) share the doubled interleaved
            // region; the doubled shape documents its true extent.
            for k in 0..6 {
                let p = &plans[i + k];
                let (ptr, bytes, shape) = if ilv && (k == 0 || k == 2) {
                    (pb, 2 * w1b, vec![p.local[0], p.local[1] * 2])
                } else {
                    (pb.wrapping_add(poff[k]), p.bytes, p.local.clone())
                };
                views.push(DevTensor {
                    buf: Device::view(ptr as *mut std::ffi::c_void, bytes),
                    shape,
                    dtype: if p.widen { "F32".into() } else { p.dtype.clone() },
                });
            }
        }
        if let Some(t) = tmp {
            self.dev.free(&t);
        }
        let mut experts = Vec::with_capacity(n_routed);
        for e in 0..n_routed {
            let i = e * 6;
            experts.push(DevExpert {
                w1: views[i].clone(),
                w1_scale: views[i + 1].clone(),
                w3: views[i + 2].clone(),
                w3_scale: views[i + 3].clone(),
                w2: views[i + 4].clone(),
                w2_scale: views[i + 5].clone(),
            });
        }
        Ok((experts, pool, ilv))
    }

    /// Whether the routed experts' w1/w3 can be stored INTERLEAVED
    /// (DSV41_EXPERT_ILV). Every term is a LOAD-TIME guarantee that the only
    /// consumers able to read that layout — the FUSED batched gate/up GEMV — are
    /// the ones that will actually run. The sequential fallback and the unfused
    /// batched body walk the gate and up pools separately and would read the
    /// wrong bytes, so they must not be reachable:
    ///   * DSV41_MOE_BATCH not disabled and the batched symbol set present;
    ///   * DSV41_GATEUP_FUSE not disabled and the fused signature present;
    ///   * fp4 expert mode 2 (the shared-LUT body the fused loop mirrors);
    ///   * dim % 512 == 0 (the launcher's own fuse gate);
    ///   * DSV41_NO_GEMV_FP4 unset — that knob routes the rows==1 call through
    ///     the tcgen05 GEMM, which assumes the plain layout.
    /// All of these are cached once (`OnceLock`), so the runtime `batched`
    /// decision in chain_dev.rs cannot drift from the layout fixed here.
    fn ilv_ok(&self, cfg: &Dsv41Config) -> bool {
        use crate::dsv41::chain_dev as cd;
        gateup_ilv()
            && cd::moe_batch()
            && cd::gateup_fuse()
            && cd::expert_fp4_mode() == 2
            && !cd::no_gemv_fp4()
            && cfg.dim % 512 == 0
            && self.dev.supports_moe_batch()
            && self.dev.supports_gateup_fuse()
            && self.dev.supports_expert_ilv()
    }

    pub fn load_single(&mut self, spec: &TensorSpec, world: usize, rank: usize) -> Result<DevTensor> {
        self.load_tensor(spec, world, rank)
    }

    /// Load everything the rank needs. `specs` come from `tensor_specs`.
    pub fn load(&mut self, cfg: &Dsv41Config, world: usize, rank: usize) -> Result<Dsv41DevWeights> {
        let specs = tensor_specs(cfg, world);
        // expert ownership: the rank keeps n_routed/world consecutive experts
        let _expert_lo = |n: usize| (rank * (n / world), n / world);
        let mut w = Dsv41DevWeights {
            layers: (0..cfg.n_layers).map(|_| LayerDev::default()).collect(),
            mtp: (0..cfg.n_mtp_layers).map(|_| LayerDev::default()).collect(),
            ..Default::default()
        };
        let want = |n: &str| specs.iter().find(|s| s.name == n).cloned();
        // ONE layout decision for the whole model (routed experts' gate/up):
        // every layer and the MTP blocks share it, so a mixed layout can never
        // arise from a per-layer difference. See `ilv_ok`.
        let ilv_ok = self.ilv_ok(cfg);
        // A macro rather than a closure: a closure capturing `self` mutably
        // would conflict with the very next `self.load_tensor` call, and it has
        // to span both the backbone and the draft loops.
        macro_rules! take {
            ($pfx:expr, $ld:expr, $key:expr, $field:ident) => {
                if let Some(sp) = want(&format!("{}.{}", $pfx, $key)) {
                    $ld.$field = Some(self.load_tensor(&sp, world, rank)?);
                }
            };
        }

        // ---- globals ----
        w.embed = Some(self.load_tensor(&want("embed.weight").unwrap(), world, rank)?);
        w.norm = Some(self.load_tensor(&want("norm.weight").unwrap(), world, rank)?);
        w.head = Some(self.load_tensor(&want("head.weight").unwrap(), world, rank)?);

        for l in 0..cfg.n_layers {
            let p = format!("layers.{l}");
            let p = p.as_str();
            let mut ld = LayerDev::default();
            let take = |ld: &mut LayerDev, key: &str, dst: fn(&mut LayerDev) -> &mut Option<DevTensor>| -> Result<()> {
                if let Some(s) = want(&format!("{p}.{key}")) {
                    *dst(ld) = Some(self.load_tensor(&s, world, rank)?);
                }
                Ok(())
            };
            take!(p, ld, "attn_norm.weight", attn_norm);
            take!(p, ld, "ffn_norm.weight", ffn_norm);
            take!(p, ld, "hc_attn_fn", hc_attn_fn);
            take!(p, ld, "hc_attn_base", hc_attn_base);
            take!(p, ld, "hc_attn_scale", hc_attn_scale);
            take!(p, ld, "hc_ffn_fn", hc_ffn_fn);
            take!(p, ld, "hc_ffn_base", hc_ffn_base);
            take!(p, ld, "hc_ffn_scale", hc_ffn_scale);
            take!(p, ld, "attn.wq_a.weight", wq_a);
            take!(p, ld, "attn.wq_a.scale", wq_a_scale);
            take!(p, ld, "attn.q_norm.weight", q_norm);
            take!(p, ld, "attn.wq_b.weight", wq_b);
            take!(p, ld, "attn.wq_b.scale", wq_b_scale);
            take!(p, ld, "attn.wkv.weight", wkv);
            take!(p, ld, "attn.wkv.scale", wkv_scale);
            take!(p, ld, "attn.kv_norm.weight", kv_norm);
            take!(p, ld, "attn.wo_a.weight", wo_a);
            take!(p, ld, "attn.wo_a.scale", wo_a_scale);
            take!(p, ld, "attn.wo_b.weight", wo_b);
            take!(p, ld, "attn.wo_b.scale", wo_b_scale);
            take!(p, ld, "attn.attn_sink", attn_sink);
            take!(p, ld, "attn.compressor.wkv.weight", comp_wkv);
            take!(p, ld, "attn.compressor.wkv.scale", comp_wkv_scale);
            take!(p, ld, "attn.compressor.wgate.weight", comp_wgate);
            take!(p, ld, "attn.compressor.wgate.scale", comp_wgate_scale);
            take!(p, ld, "attn.compressor.norm.weight", comp_norm);
            take!(p, ld, "attn.indexer.wq_b.weight", idx_wq_b);
            take!(p, ld, "attn.indexer.wq_b.scale", idx_wq_b_scale);
            take!(p, ld, "attn.indexer.weights_proj.weight", idx_weights);
            take!(p, ld, "attn.indexer.wk.weight", idx_wk);
            take!(p, ld, "attn.indexer.k_norm.weight", idx_k_norm);
            take!(p, ld, "ffn.gate.weight", gate_w);
            take!(p, ld, "ffn.gate.bias", gate_bias);
            take!(p, ld, "ffn.gate.bias_vl", gate_bias_vl);
            take!(p, ld, "ffn.shared_experts.w1.weight", shared_w1);
            take!(p, ld, "ffn.shared_experts.w1.scale", shared_w1_scale);
            take!(p, ld, "ffn.shared_experts.w3.weight", shared_w3);
            take!(p, ld, "ffn.shared_experts.w3.scale", shared_w3_scale);
            take!(p, ld, "ffn.shared_experts.w2.weight", shared_w2);
            take!(p, ld, "ffn.shared_experts.w2.scale", shared_w2_scale);
            take!(p, ld, "engram.embed.weight", engram_embed);
            take!(p, ld, "engram.embed.scale", engram_embed_scale);
            take!(p, ld, "engram.wkv.weight", engram_wkv);
            take!(p, ld, "engram.wkv.scale", engram_wkv_scale);
            take!(p, ld, "engram.q_weight", engram_q_weight);
            take!(p, ld, "engram.k_weight", engram_k_weight);

            // EVERY expert, TP-split along `inter` (no expert parallelism):
            // each rank holds a slice of all of them, and the MoE's all-reduce
            // sums slices of the same experts. All 384 experts live in ONE
            // pooled allocation — 6 cudaMallocs per expert (92k over the model)
            // exhausted the 4 GB host's driver bookkeeping.
            let (n_routed, _) = cfg.moe_config(l);
            let (experts, pool, experts_ilv) =
                self.load_expert_pool(&p, n_routed, l, world, rank, ilv_ok)?;
            ld.experts = experts;
            ld.expert_pool = Some(pool);
            ld.experts_ilv = experts_ilv;
            w.layers[l] = ld;
        }

        // ---- DSpark draft layers ----
        for s in 0..cfg.n_mtp_layers {
            let p = format!("mtp.{s}");
            let p = p.as_str();
            let mut ld = LayerDev::default();
            take!(p, ld, "attn_norm.weight", attn_norm);
            take!(p, ld, "ffn_norm.weight", ffn_norm);
            take!(p, ld, "hc_attn_fn", hc_attn_fn);
            take!(p, ld, "hc_attn_base", hc_attn_base);
            take!(p, ld, "hc_attn_scale", hc_attn_scale);
            take!(p, ld, "hc_ffn_fn", hc_ffn_fn);
            take!(p, ld, "hc_ffn_base", hc_ffn_base);
            take!(p, ld, "hc_ffn_scale", hc_ffn_scale);
            take!(p, ld, "attn.wq_a.weight", wq_a);
            take!(p, ld, "attn.wq_a.scale", wq_a_scale);
            take!(p, ld, "attn.q_norm.weight", q_norm);
            take!(p, ld, "attn.wq_b.weight", wq_b);
            take!(p, ld, "attn.wq_b.scale", wq_b_scale);
            take!(p, ld, "attn.wkv.weight", wkv);
            take!(p, ld, "attn.wkv.scale", wkv_scale);
            take!(p, ld, "attn.kv_norm.weight", kv_norm);
            take!(p, ld, "attn.wo_a.weight", wo_a);
            take!(p, ld, "attn.wo_a.scale", wo_a_scale);
            take!(p, ld, "attn.wo_b.weight", wo_b);
            take!(p, ld, "attn.wo_b.scale", wo_b_scale);
            take!(p, ld, "attn.attn_sink", attn_sink);
            take!(p, ld, "ffn.gate.weight", gate_w);
            take!(p, ld, "ffn.gate.bias", gate_bias);
            take!(p, ld, "ffn.gate.bias_vl", gate_bias_vl);
            take!(p, ld, "ffn.shared_experts.w1.weight", shared_w1);
            take!(p, ld, "ffn.shared_experts.w1.scale", shared_w1_scale);
            take!(p, ld, "ffn.shared_experts.w3.weight", shared_w3);
            take!(p, ld, "ffn.shared_experts.w3.scale", shared_w3_scale);
            take!(p, ld, "ffn.shared_experts.w2.weight", shared_w2);
            take!(p, ld, "ffn.shared_experts.w2.scale", shared_w2_scale);
            let (n_routed, _) = cfg.moe_config(cfg.n_layers + s);
            let (experts, pool, experts_ilv) =
                self.load_expert_pool(&p, n_routed, cfg.n_layers + s, world, rank, ilv_ok)?;
            ld.experts = experts;
            ld.expert_pool = Some(pool);
            ld.experts_ilv = experts_ilv;
            if s == 0 {
                // THE ROOT CAUSE of the 100% q divergence (unit-diff verdict):
                // this block once "parked" main_proj's spec in the attn_norm
                // FIELD as a placeholder — a later take!-free load then left
                // the field pointing at main_proj's fp8 bytes, which rmsnorm
                // read as f32 weights → 1e27 explosions. main_proj has its own
                // dedicated fields (w.main_proj*) loaded below; the placeholder
                // overwrite of attn_norm is deleted.
                if let Some(sp) = want(&format!("{p}.main_proj.weight")) {
                    w.main_proj = Some(self.load_tensor(&sp, world, rank)?);
                }
                if let Some(sp) = want(&format!("{p}.main_proj.scale")) {
                    w.main_proj_scale = Some(self.load_tensor(&sp, world, rank)?);
                }
                if let Some(sp) = want(&format!("{p}.main_norm.weight")) {
                    w.main_norm = Some(self.load_tensor(&sp, world, rank)?);
                }
            }
            if s + 1 == cfg.n_mtp_layers {
                if let Some(sp) = want(&format!("{p}.norm.weight")) {
                    w.dspark_norm = Some(self.load_tensor(&sp, world, rank)?);
                }
                if let Some(sp) = want(&format!("{p}.markov_head.embed.weight")) {
                    w.markov_embed = Some(self.load_tensor(&sp, world, rank)?);
                }
                if let Some(sp) = want(&format!("{p}.markov_head.head.weight")) {
                    w.markov_head = Some(self.load_tensor(&sp, world, rank)?);
                }
                if let Some(sp) = want(&format!("{p}.confidence_head.proj.weight")) {
                    w.confidence_proj = Some(self.load_tensor(&sp, world, rank)?);
                }
            }
            w.mtp[s] = ld;
        }

        // ---- vision (replicated; only loaded when enabled) ----
        if cfg.vision_enabled() {
            for s in &specs {
                if s.name.starts_with("vision.") || s.name.starts_with("aligner.") || s.name.starts_with("image_") {
                    let t = self.load_tensor(s, world, rank)?;
                    w.vision.push((s.name.clone(), t));
                }
            }
        }
        if std::env::var("DSV41_LOAD_TRACE").map(|v| v != "0").unwrap_or(false) {
            let mut t = self.trace.clone();
            t.sort_by(|a, b| b.0.cmp(&a.0));
            for (b, n) in t.iter().take(12) {
                eprintln!("[load] {:10.2} MiB  {n}", *b as f64 / (1u64 << 20) as f64);
            }
            let total: usize = self.trace.iter().map(|(b, _)| *b).sum();
            eprintln!(
                "[load] TOTAL {:.2} GiB over {} tensors",
                total as f64 / (1u64 << 30) as f64,
                self.trace.len()
            );
        }
        Ok(w)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_sizes_match_the_release() {
        assert_eq!(dtype_size("F8_E4M3"), 1);
        assert_eq!(dtype_size("F8_E8M0"), 1);
        assert_eq!(dtype_size("I8"), 1);
        assert_eq!(dtype_size("BF16"), 2);
        assert_eq!(dtype_size("F32"), 4);
    }

    #[test]
    fn bf16_widening_is_exact() {
        // every bf16 pattern must round-trip exactly through the widening we do
        // on the way to the device
        let mut raw = Vec::new();
        for h in [0u16, 1, 0x3f80, 0xbf80, 0x7f7f, 0x0080, 0x8000, 0xc000] {
            raw.extend_from_slice(&h.to_le_bytes());
        }
        let w = bf16_to_f32_bytes(&raw);
        assert_eq!(w.len(), raw.len() * 2);
        for i in 0..(raw.len() / 2) {
            let h = u16::from_le_bytes([raw[i * 2], raw[i * 2 + 1]]);
            let got = f32::from_le_bytes([w[i * 4], w[i * 4 + 1], w[i * 4 + 2], w[i * 4 + 3]]);
            assert_eq!(got.to_bits(), (h as u32) << 16, "bf16 {h:#06x} widened wrong");
        }
    }

    #[test]
    fn expert_ownership_is_consecutive_and_complete() {
        // the loader keeps n_routed/world consecutive experts per rank; across
        // all ranks that must cover every expert exactly once
        let cfg = Dsv41Config::production();
        for world in [1usize, 2, 4, 8] {
            let n = cfg.n_routed_experts;
            let per = n / world;
            let mut seen = vec![0usize; n];
            for rank in 0..world {
                let lo = rank * per;
                for e in lo..(lo + per).min(n) {
                    seen[e] += 1;
                }
            }
            assert!(
                seen.iter().all(|&c| c == 1),
                "world {world}: expert ownership must partition (got {:?})",
                seen.iter().filter(|&&c| c != 1).count()
            );
        }
    }
}
