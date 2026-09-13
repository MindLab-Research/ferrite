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
use crate::dsv41::weights::{
    check_bulk_geometry, gateup_ilv, local_shape, tensor_specs, SafetensorsIndex, Shard, TensorSpec,
};
// the SF-pitch half of the pool layout: `sf_pitch_plane` / `sf_plane_pitch` /
// `sf_stride_pad` (the w2 10-byte-row fix, see `plan_pitch` below)
use crate::dsv41::weights;

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
    /// PHYSICAL row stride in bytes. Equal to `local[1..]*esz` for every plane
    /// except the expert e8m0 plane this fix re-pitches (`w2.scale`:
    /// `sf_plane_pitch(10) = 16`), where the rows are laid out wider than the
    /// kernel's logical `k/32` indexing width — see
    /// [`crate::dsv41::weights::sf_plane_pitch`] and
    /// docs/agent/tcgen05-rank7-verdict.md §10.
    pub pitch: usize,
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
    /// A 16-byte-aligned byte view at `off` (see [`DevBuf::u8_aligned_at`]).
    /// Used at every site whose pointer feeds a `cp.async.bulk*` /
    /// `LDG.128` source, so a 4-byte slip into a pooled weight plane trips a
    /// `debug_assert` in CI instead of silently misplacing bytes.
    #[inline]
    pub fn u8_aligned_at(&self, off: usize) -> *const u8 {
        self.buf.u8_aligned_at(off)
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

/// `DSV41_ALIGN_AUDIT=1`: the load-time 16B audit switch (default OFF). Read ONCE
/// (house rule for a non-hot-path gate). See
/// docs/agent/tcgen05-tma-bulk-align-design.md §6 V2.
pub fn align_audit() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("DSV41_ALIGN_AUDIT").map(|v| v == "1").unwrap_or(false))
}

/// The six expert planes in the loader's own order — the names a misalignment
/// report has to use to be actionable at the call site.
const ALIGN_PLANES: [&str; 6] = ["w1", "w1.scale", "w3", "w3.scale", "w2", "w2.scale"];

/// The PHYSICAL row stride of one planned plane, in bytes.
///
/// Identity for every tensor in the model except the ONE plane the SF-pitch fix
/// re-pitches: the routed experts' `w2.scale` (`Shard::ExpertCols` + `.scale`),
/// whose logical row is `padded_inter(inter/world)/32 = 10` bytes at the
/// production shape and whose physical row becomes
/// [`weights::sf_plane_pitch`]`(10) = 16`. The pad bytes are never read (the
/// kernels index the first `k/32` bytes of each row) and take no part in any
/// dot-product, so the numerics are untouched — only the addresses move.
///
/// The `.cu` mirrors this with one formula, `dsv41_sf_pitch(k) = align16(k >> 5)`
/// (`kernels/cuda/dsv41_experts_mxf4.cu`), driven by the same
/// `DSV41_SF_STRIDE_PAD` variable; `DSV41_ALIGN_AUDIT=1` prints the physical
/// pitch at load time so the two sides are comparable in one run.
fn plan_pitch(name: &str, shard: &Shard, logical_pitch: usize) -> usize {
    if weights::sf_pitch_plane(name, *shard) {
        weights::sf_plane_pitch(logical_pitch)
    } else {
        logical_pitch
    }
}

/// `DSV41_ALIGN_AUDIT=1` (`load_expert_pool`): print the 16B-grid position of
/// every operand a tcgen05 / `cp.async.bulk` arm will derive from this layer's
/// pool, and report every violation it finds.
///
/// One line per plane at expert 0 (the layout is per-expert uniform, so e0 IS the
/// layout), each carrying the four quantities that can move a pointer off the
/// grid:
///   * `base&15`     — `pool + poff[k]`
///   * `worst_base&15` — the WORST expert (`pool + e*block + poff[k]`); a `block`
///                     that is not a 16B multiple slides every later expert
///   * `stride%16`   — `block`
///   * `pitch%16`    — the plane's row pitch, plus `row1&15` = `base + pitch`,
///                     i.e. where row 1 of the plane actually starts. A pitch
///                     that is not a 16B multiple (w2's e8m0 plane is 10 B/row at
///                     the production shape) leaves row 0 aligned and EVERY
///                     following row misaligned — the one failure the base and
///                     stride checks alone cannot see.
///
/// This runs at LOAD time, so a violation costs a second instead of an
/// unexplained `err 716` 191 ms into the first prefill, detached at the next
/// sync. It is arm-agnostic: it does not depend on any single arm's launcher
/// gate, which is the whole point (§6 V2 is described as the highest-value
/// change of the design).
fn audit_expert_pool(
    layer: usize,
    rank: usize,
    n_routed: usize,
    block: usize,
    poff: &[usize; 6],
    plans: &[TensorPlan],
    base: *mut u8,
    ilv: bool,
) {
    if plans.len() < 6 {
        return; // nothing to audit; a malformed plan list fails elsewhere
    }
    let base_usize = base as usize;
    let mut bad: Vec<String> = Vec::new();
    for (k, name) in ALIGN_PLANES.iter().enumerate() {
        let p = &plans[k]; // expert 0's plan for this plane
        // PHYSICAL row stride (bytes) — what the pool actually laid out, which
        // for `w2.scale` is `sf_plane_pitch(10) = 16` after the fix (it used to
        // be the logical 10, and that WAS the reported violation). `p.pitch` is
        // the one number the `.cu`'s `dsv41_sf_pitch(k)` has to agree with.
        let pitch = p.pitch;
        let logical = p.local.get(1..).map(|d| d.iter().product::<usize>()).unwrap_or(1) * p.esz();
        // interleaved: w3 shares w1's doubled region, so the kernel reads it at
        // offset 0 — report what the kernel will use, not the plan's own slot
        let off = if ilv && k == 2 { 0 } else { poff[k] };
        let a = (base_usize + off) & 0xF;
        // scan every expert: a `block` that is not a 16B multiple slides all the
        // experts after e0, so e0 alone would miss it
        let worst =
            (0..n_routed).map(|e| (base_usize + e * block + off) & 0xF).max().unwrap_or(0);
        let pitch_rem = pitch % 16;
        let row1 = (base_usize + off + pitch) & 0xF;
        eprintln!(
            "[align] L{layer} r{rank} e0 {name:<9} base&15={a} worst_base&15={worst} \
             stride%16={} pitch={pitch} pitch%16={pitch_rem} row1&15={row1} logical_pitch={logical}",
            block % 16
        );
        if a != 0 || worst != 0 {
            bad.push(format!(
                "L{layer} {name}: base off the 16B grid (e0={a} B, worst={worst} B; \
                 pool base&15={}, poff[{k}]={off}, block%16={})",
                base_usize & 0xF,
                block % 16
            ));
        }
        if pitch_rem != 0 {
            bad.push(format!(
                "L{layer} {name}: row pitch {pitch} B is not a multiple of 16 (row 1 starts \
                 {row1} B off the grid) — every row past the first of this plane is \
                 unaddressable by the bulk/uint4 paths (DSV41_SF_STRIDE_PAD=0 restores this)"
            ));
        }
    }
    if bad.is_empty() {
        eprintln!("[align] L{layer} r{rank}: pool geometry OK (block={block}, 6 planes, 16B)");
    } else {
        eprintln!(
            "[align] L{layer} r{rank}: {} 16B violation(s) in the expert pool — the tcgen05/bulk \
             operands CANNOT be addressed: {}",
            bad.len(),
            bad.join(" | ")
        );
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
    /// `DSV41_MOE_BF16_DEQUANT=1` only: the **bf16 copy of gate‖up**
    /// (`[2*inter_local, dim]`, rows `[0, inter_local)` = w1/gate and
    /// `[inter_local, 2*inter_local)` = w3/up) that the TileLang grouped-GEMM arm
    /// streams (`dsv41_moe_tilelang_gate_up_bf16`). Views into one contiguous
    /// `[E, 2*inter_local, dim]` pool per layer, so the shim's `base + e*N*K` expert
    /// arithmetic holds — the arm VERIFIES that stride before it fires
    /// (`moe_tilelang_weights`). The fp4 planes above are untouched (the proven
    /// paths still read them), so this is purely additive memory.
    pub up_bf16: Option<DevTensor>,
    /// The bf16 copy of w2 (`[dim, inter_local]`), same pool, for
    /// `dsv41_moe_tilelang_down_bf16`.
    pub dn_bf16: Option<DevTensor>,
    /// `DSV41_MOE_TILELANG_BS=1` only: the **group-major packed** ue8m0 words of
    /// this expert's gate plane (`w1.scale`), `[sf_words * inter_local]` u32, the
    /// layout `T.tcgen05_gemm_blockscaled`'s SF operand wants. The fp4 `w1` plane
    /// above is untouched and is fed to the arm **as-is** (zero dequant, zero copy
    /// of the weight itself); only the scale bytes are permuted, once, at load.
    pub wsf1: Option<DevTensor>,
    /// Same for the up plane (`w3.scale`).
    pub wsf3: Option<DevTensor>,
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
    /// owns the memory the experts' `up_bf16`/`dn_bf16` views point into
    /// (`DSV41_MOE_BF16_DEQUANT` only; `None` otherwise — no allocation, no
    /// behaviour change).
    pub expert_bf16_pool: Option<DevBuf>,
    /// owns the memory the experts' `wsf1`/`wsf3` packed-SF views point into
    /// (`DSV41_MOE_TILELANG_BS` only; `None` otherwise — no allocation, no
    /// behaviour change). Same pattern as [`Self::expert_bf16_pool`].
    pub expert_bs_pool: Option<DevBuf>,
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
        let esz = if widen { 4 } else { dtype_size(&h.dtype) };
        // LOGICAL row width = the kernel's `k/32` indexing width (or `k/2` for a
        // packed fp4 plane), PHYSICAL row stride = the same unless this is the
        // one expert plane the SF-pitch fix re-pitches.
        let logical_pitch = local.get(1..).map(|d| d.iter().product::<usize>()).unwrap_or(1) * esz;
        let physical_pitch = plan_pitch(&spec.name, &spec.shard, logical_pitch);
        let rows = local.first().copied().unwrap_or(1);
        let bytes = rows * physical_pitch;
        Ok(TensorPlan {
            local,
            bytes,
            pitch: physical_pitch,
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
                // DESTINATION pitch is the PHYSICAL row stride (== local[1]*esz
                // for every plane but `w2.scale`, which is laid out 16 B/row —
                // see `plan_pitch`). The SOURCE pitch stays the checkpoint's
                // `per*esz`: the real bytes land at the start of each wide row
                // and the 6 B tail keeps the zero it was allocated with.
                self.dev.upload_from_2d(
                    target,
                    plan.pitch,
                    data.wrapping_add(rank * per * esz) as *const std::ffi::c_void,
                    row_bytes,
                    per * esz,
                    plan.global[0],
                )?;
            }
            _ => {
                debug_assert_eq!(
                    plan.pitch, row_bytes,
                    "{}: a row-sharded plane must keep its contiguous layout",
                    plan.shard
                );
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
        layer: usize,
        world: usize,
        rank: usize,
        ilv: bool,
    ) -> Result<(Vec<DevExpert>, DevBuf, bool, Option<DevBuf>, Option<DevBuf>)> {
        const NAMES: [&str; 6] = [
            "w1.weight", "w1.scale", "w3.weight", "w3.scale", "w2.weight", "w2.scale",
        ];
        // DSV41_MOE_BF16_DEQUANT (§task 4): build the bf16 copies the TileLang MoE
        // arm streams. Both halves must agree — the runtime gate AND the symbol —
        // so a stale `.so` leaves the copies absent and the arm declines loudly
        // rather than reading bytes that are not there.
        let want_bf16 = crate::dsv41::chain_dev::moe_bf16_dequant()
            && self.dev.supports_moe_bf16_dequant();
        // DSV41_MOE_TILELANG_BS: the block-scaled (native fp4) MoE arm needs no
        // bf16 copy at all — it streams the fp4 planes above AND their ue8m0 planes.
        // What it DOES need is the e8m0 words in `T.tcgen05_gemm_blockscaled`'s
        // group-major packing, which is a one-time byte permutation done below.
        let want_bs =
            crate::dsv41::weights::moe_tilelang_bs() && self.dev.supports_moe_tilelang_bs();
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
        // ---- 16B pool geometry: the bulk-copy contract --------------------
        // Every pointer an expert kernel derives is
        //     pool + e*block + poff[k] + row*pitch [+ j*16]
        // and `cp.async.bulk{.prefetch}` demands 16 B on BOTH usable sides
        // (src, and dst for the G2S form). cudaMalloc supplies `pool`; `block`,
        // `poff` and `pitch` are OURS, so they are padded here instead of
        // being "accidentally right" for the shipped shape (the current
        // 2,611,200 B block is a multiple of 128 only by arithmetic luck, and
        // nothing asserts it).
        // See docs/agent/tcgen05-tma-bulk-align-design.md §4.
        const ALIGN: usize = 128; // >= 16 (PTX); 128 also keeps a full L2 sector
        let up = |x: usize| (x + ALIGN - 1) & !(ALIGN - 1);
        let (block, poff) = if ilv {
            let mut o = [0usize; 6];
            // o[2] stays 0: w3 reuses w1's doubled region (0 is trivially aligned).
            o[1] = up(2 * w1b);
            o[3] = up(o[1] + plans[1].bytes);
            o[4] = up(o[3] + plans[3].bytes);
            o[5] = up(o[4] + plans[4].bytes);
            (up(o[5] + plans[5].bytes), o)
        } else {
            let mut o = [0usize; 6];
            let mut acc = 0usize;
            for (k, slot) in o.iter_mut().enumerate() {
                *slot = up(acc);
                acc = *slot + plans[k].bytes;
            }
            (up(acc), o)
        };
        debug_assert_eq!(block % ALIGN, 0);
        debug_assert!(poff.iter().all(|o| o % ALIGN == 0));
        debug_assert!(block * n_routed >= total);
        // The SF-pitch invariant (the 2026-09-13 fix): with DSV41_SF_STRIDE_PAD
        // ON every plane's PHYSICAL row stride is a 16 B multiple, which is what
        // lets `pool + e*block + poff[k] + row*pitch` stay on the grid for EVERY
        // row (`w2.scale` used to be 10 B). The `.cu` addresses the same stride
        // (`dsv41_sf_pitch(k) = align16(k >> 5)`), so a violation here would be a
        // silent wrong answer there — hence the assert, not a comment.
        for (k, name) in ALIGN_PLANES.iter().enumerate() {
            if weights::sf_stride_pad() {
                debug_assert_eq!(
                    plans[k].pitch % 16,
                    0,
                    "{name}: physical row pitch {} is off the 16B grid",
                    plans[k].pitch
                );
            }
        }
        // one allocation; BOTH the K padding and the alignment padding are zero
        let pool_bytes = (block * n_routed).max(1);
        let pool = self.dev.alloc(pool_bytes)?;
        self.dev.zero_at(pool.ptr, pool_bytes)?;
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
        // ---- DSV41_ALIGN_AUDIT=1: the load-time 16B audit --------------------
        // Every operand a tcgen05/bulk arm consumes is
        //     pool + e*block + poff[k] + row*pitch [+ j*16]
        // so the ONLY things that can move it off the 16B grid are the pool base,
        // `block`, `poff[k]` and `pitch`. Audit all four HERE, per layer, at load
        // time (seconds) — instead of 191 ms into the first prefill, where the
        // err 716 surfaces detached at the next sync and names no buffer.
        // Default OFF; this is the §6 V2 hook of
        // docs/agent/tcgen05-tma-bulk-align-design.md, and it is arm-agnostic
        // (it does not depend on any single arm's launcher gate).
        // (Ordered ABOVE the DSV41_MOE_BF16_DEQUANT block below: the dequant
        // consumes `plans[].pitch`, which is exactly what this audit validates.)
        if align_audit() {
            audit_expert_pool(layer, rank, n_routed, block, &poff, &plans, base, ilv);
        }
        // ---- DSV41_MOE_BF16_DEQUANT: the bf16 copies the TileLang arm streams ----
        // DEFAULT OFF (the memory decision is the owner's; PROVENANCE.md §8). When
        // armed, expand every expert's fp4 planes ONCE here (load time, seconds) into
        // one contiguous bf16 pool per layer:
        //   up : [E, 2*inter_local, dim]  rows [0, il) = w1/gate, [il, 2*il) = w3/up
        //   dn : [E, dim, inter_local]    w2
        // The TileLang shim derives each expert's base ARITHMETICALLY (`e*N*K`), so
        // the pool has to be exactly this regular (the arm re-checks the stride).
        // ⚠️ Requires the PLAIN (non-interleaved) gate/up planes: with DSV41_EXPERT_ILV
        // the w1 view aliases the doubled interleaved region and this dequant would
        // read interleaved bytes as if they were a contiguous gate plane — a silent
        // wrong answer. `ilv_ok` therefore refuses to interleave when this gate is on.
        let mut bf16_pool: Option<DevBuf> = None;
        let mut up_views: Vec<Option<DevTensor>> = vec![None; n_routed];
        let mut dn_views: Vec<Option<DevTensor>> = vec![None; n_routed];
        if want_bf16 && !ilv && !plans.is_empty() {
            let il = plans[0].local[0]; // inter_local (padded)
            let dim = plans[0].local[1] * 2; // fp4 packs 2 values/byte
            let up_bytes = 2 * il * dim * 2;
            let dn_bytes = dim * il * 2;
            debug_assert!(plans[2].local[0] == il && plans[4].local[0] == dim);
            let pool_bf = self.dev.alloc((up_bytes + dn_bytes) * n_routed)?;
            let base_bf = pool_bf.ptr as *mut u8;
            let mut ok = true;
            for e in 0..n_routed {
                let eb = base_bf.wrapping_add(e * (up_bytes + dn_bytes));
                let i = e * 6;
                // w1 -> up[0..il) ; w3 -> up[il..2*il) ; w2 -> dn.
                for (qk, sk, dst, n, k) in [
                    (i, i + 1, eb, il, dim),
                    (i + 2, i + 3, eb.wrapping_add(il * dim * 2), il, dim),
                    (i + 4, i + 5, eb.wrapping_add(up_bytes), dim, il),
                ] {
                    let q = views[qk].ptr() as *const std::ffi::c_void;
                    let s = views[sk].ptr() as *const std::ffi::c_void;
                    let o = dst as *mut std::ffi::c_void;
                    let ok_one = self.dev.moe_fp4_to_bf16(
                        q,
                        s,
                        o,
                        n as i32,
                        k as i32,
                        plans[qk].pitch as i32,
                        plans[sk].pitch as i32,
                    )?;
                    ok &= ok_one;
                }
                up_views[e] = Some(DevTensor {
                    buf: Device::view(eb as *mut std::ffi::c_void, up_bytes),
                    shape: vec![2 * il, dim],
                    dtype: "BF16".into(),
                });
                dn_views[e] = Some(DevTensor {
                    buf: Device::view(
                        eb.wrapping_add(up_bytes) as *mut std::ffi::c_void,
                        dn_bytes,
                    ),
                    shape: vec![dim, il],
                    dtype: "BF16".into(),
                });
            }
            if ok {
                bf16_pool = Some(pool_bf);
            } else {
                // The kernel declined (shape/alignment) or the symbol is missing:
                // drop the pool and leave the experts without bf16 copies — the
                // TileLang arm then declines loudly instead of reading garbage.
                self.dev.free(&pool_bf);
                up_views = vec![None; n_routed];
                dn_views = vec![None; n_routed];
                eprintln!(
                    "[load] DSV41_MOE_BF16_DEQUANT armed but `dsv41_moe_fp4_to_bf16` declined \
                     (missing symbol, or a shape/pitch outside its contract) -> the TileLang MoE \
                     arm will decline"
                );
            }
        } else if want_bf16 && ilv {
            eprintln!(
                "[load] DSV41_MOE_BF16_DEQUANT is armed but the gate/up planes are INTERLEAVED \
                 (DSV41_EXPERT_ILV) -> no bf16 copy; the TileLang MoE arm will decline"
            );
        }

        // ---- DSV41_MOE_TILELANG_BS: the LOAD-TIME group-major ue8m0 pack --------
        // The block-scaled arm feeds `T.tcgen05_gemm_blockscaled` the e8m0 bytes in
        // **group-major packed uint32** form (`word[g*rows + row]` = 4 consecutive
        // ue8m0 bytes covering 128 K), while the pool stores them row-major
        // (`[rows, k/32]` u8). This is the ONE place that repack happens, ONCE per
        // expert per layer: it is a pure byte permutation of already-e8m0 data, so
        // it is bit-exact and lossless (nothing is decoded or re-rounded), and it
        // takes the hot path to zero extra launches.
        //
        // Cost: 2 planes x sf_words(40) x inter_local(320) x 4 B = 51200 B/expert,
        // i.e. ~1.54 GiB/rank at the production shape (40 layers x 384 experts) —
        // 1.5% of what the bf16 arm's mirror costs (`DSV41_MOE_BF16_DEQUANT`,
        // +105 GiB/rank) and 4% of the fp4 pool it lives next to.
        // ⚠️ Requires the PLAIN gate/up planes (same reason as the bf16 copy above:
        // under DSV41_EXPERT_ILV the w1 view aliases the doubled region and the w1
        // plane is not a clean `[inter_local, k/2]` face).
        let mut bs_pool: Option<DevBuf> = None;
        let mut wsf1_views: Vec<Option<DevTensor>> = vec![None; n_routed];
        let mut wsf3_views: Vec<Option<DevTensor>> = vec![None; n_routed];
        if want_bs && !ilv && !plans.is_empty() {
            let il = plans[0].local[0]; // inter_local (padded)
            let k = plans[0].local[1] * 2; // fp4 packs 2 values/byte
            let words = crate::dsv41::weights::moe_bs_sf_words(k);
            if words == 0 {
                eprintln!(
                    "[load] DSV41_MOE_TILELANG_BS armed but k={k} is not a multiple of 128 (the \
                     packed e8m0 word) -> no SF pool; the block-scaled arm will decline"
                );
            } else {
                let plane = words * il * 4; // u32 words x rows
                // FIX(pool-geometry): allocate as TWO CONTIGUOUS segments — all wsf1
                // first, then all wsf3 — so that consecutive experts' wsf1 (and wsf3)
                // are exactly `plane` bytes apart. The previous interleaved layout
                // ([wsf1[e], wsf3[e], wsf1[e+1], ...]) put them 2*plane apart, which
                // fails moe_bs_weights()'s contiguous-pool validation and the shim's
                // `sfw1 + e*sf_words*NP` indexing contract.
                let pool_bs = self.dev.alloc(plane * 2 * n_routed)?;
                let base_bs = pool_bs.ptr as *mut u8;
                let wsf1_seg = base_bs;
                let wsf3_seg = base_bs.wrapping_add(n_routed * plane);
                let mut ok = true;
                for e in 0..n_routed {
                    let wsf1_dst = wsf1_seg.wrapping_add(e * plane);
                    let wsf3_dst = wsf3_seg.wrapping_add(e * plane);
                    let i = e * 6;
                    for (sk, dst, tag) in [
                        (i + 1, wsf1_dst, "w1.scale"),
                        (i + 3, wsf3_dst, "w3.scale"),
                    ] {
                        // The routed gate/up scale planes are `Shard::ExpertRows`, which
                        // the SF-pitch fix does NOT re-pitch (only `w2.scale` is), so the
                        // source row stride is the logical `k/32`. `moe_bs_sf_src_pitch`
                        // keeps this from drifting from `sf_pitch_plane`.
                        let pitch = crate::dsv41::weights::moe_bs_sf_src_pitch(
                            tag,
                            Shard::ExpertRows,
                            k,
                        );
                        debug_assert_eq!(pitch, k / 32);
                        let _ = pitch; // the kernel derives nsc from k; kept for the assert
                        let ok_one = self.dev.moe_bs_pack_wsf(
                            views[sk].ptr() as *const std::ffi::c_void,
                            dst as *mut std::ffi::c_void,
                            il as i32,
                            k as i32,
                        )?;
                        ok &= ok_one;
                    }
                    wsf1_views[e] = Some(DevTensor {
                        buf: Device::view(wsf1_dst as *mut std::ffi::c_void, plane),
                        shape: vec![words * il],
                        dtype: "U32".into(),
                    });
                    wsf3_views[e] = Some(DevTensor {
                        buf: Device::view(wsf3_dst as *mut std::ffi::c_void, plane),
                        shape: vec![words * il],
                        dtype: "U32".into(),
                    });
                }
                if ok {
                    bs_pool = Some(pool_bs);
                } else {
                    // Declined (missing symbol / shape outside the contract): drop the
                    // pool and leave the experts without packed SF — the arm then
                    // declines loudly instead of reading garbage.
                    self.dev.free(&pool_bs);
                    wsf1_views = vec![None; n_routed];
                    wsf3_views = vec![None; n_routed];
                    eprintln!(
                        "[load] DSV41_MOE_TILELANG_BS armed but `dsv41_moe_bs_pack_wsf` declined \
                         (missing symbol, or k % 128 != 0) -> the block-scaled MoE arm will decline"
                    );
                }
            }
        } else if want_bs && ilv {
            eprintln!(
                "[load] DSV41_MOE_TILELANG_BS is armed but the gate/up planes are INTERLEAVED \
                 (DSV41_EXPERT_ILV) -> no packed SF pool; the block-scaled MoE arm will decline"
            );
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
                up_bf16: up_views[e].clone(),
                dn_bf16: dn_views[e].clone(),
                wsf1: wsf1_views[e].clone(),
                wsf3: wsf3_views[e].clone(),
            });
        }
        Ok((experts, pool, ilv, bf16_pool, bs_pool))
    }

    /// Whether the routed experts' w1/w3 can be stored INTERLEAVED
    /// (DSV41_EXPERT_ILV). Every term is a LOAD-TIME guarantee that the only
    /// consumers able to read that layout — the batched gate/up PAIR body, in
    /// EITHER of its two epilogues (swiglu'd `[inter]` or the raw gate|up pair
    /// in `[2*inter]`; ILV selects the READ path, `fuse` the WRITE, and the two
    /// are decoupled as of 2026-09-12) — are the ones that will actually run.
    /// The sequential fallback — and the batched body's plain-layout split arm,
    /// which walks the gate and up pools separately — would read the wrong
    /// bytes, so they must not be reachable:
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
            // DSV41_MOE_BF16_DEQUANT (§task 4): the load-time bf16 copy reads each
            // expert's gate/up planes as CONTIGUOUS `[inter_local, dim]` blocks.
            // Under ILV the w1 view aliases the doubled interleaved region, so the
            // dequant would read interleaved bytes as a plain gate plane — a silent
            // wrong answer. The two layouts are therefore mutually exclusive, and
            // this is the one place the decision is made (`load_expert_pool` also
            // guards it, so a drift cannot slip through).
            && !cd::moe_bf16_dequant()
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
        // The bulk/16B geometry is a LOAD-TIME property of the config: fail here
        // (seconds) instead of at the first prefill (191 ms in, with a detached
        // err 716). See docs/agent/tcgen05-tma-bulk-align-design.md §4.1b.
        check_bulk_geometry(cfg, world)?;
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
            let (experts, pool, experts_ilv, bf16_pool, bs_pool) =
                self.load_expert_pool(&p, n_routed, l, world, rank, ilv_ok)?;
            ld.experts = experts;
            ld.expert_pool = Some(pool);
            ld.experts_ilv = experts_ilv;
            ld.expert_bf16_pool = bf16_pool;
            ld.expert_bs_pool = bs_pool;
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
            let (experts, pool, experts_ilv, bf16_pool, bs_pool) =
                self.load_expert_pool(&p, n_routed, cfg.n_layers + s, world, rank, ilv_ok)?;
            ld.experts = experts;
            ld.expert_pool = Some(pool);
            ld.experts_ilv = experts_ilv;
            ld.expert_bf16_pool = bf16_pool;
            ld.expert_bs_pool = bs_pool;
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
