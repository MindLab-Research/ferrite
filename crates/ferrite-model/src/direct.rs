//! Direct mmap loading — weights flow disk→GPU without CPU materialization.
//!
//! The legacy path (`checkpoint::load_hf_checkpoint`) reads each tensor
//! (open+seek+read), expands it to f32 on the CPU (660 GB peak on the
//! box, ~80s), converts fp8→bf16 on the CPU again at preload, and
//! uploads. This module replaces the *storage* leg of that pipeline:
//!
//! ```text
//! legacy:  read(2GB) → Vec<u8> → f32 Vec(4×) → f32→bf16 pack → H2D
//! direct:  mmap (page cache) ──────────────── H2D ──▶ GPU dequant/kernel
//! ```
//!
//! - **Mmap shards**: every `*.safetensors` is mapped `PROT_READ,
//!   MAP_PRIVATE` once; tensor bytes are borrowed slices (the page cache
//! is the only CPU-side copy, and it is reclaimable).
//! - **`WeightView`** is the preload instruction per weight: bf16
//!   segments memcpy H2D (no conversion — the checkpoint's bf16 IS the
//!   resident layout), fp8 blocks go H2D raw + the GPU dequant kernel
//!   (e4m3 × block-128 scales → bf16), the f32-consumers (embed table)
//!   expand bf16→f32 on the GPU. The CPU never materializes a weight.
//! - **Placeholders**: the runtime tensor table carries shape-real,
//!   data-empty `Tensor`s (the `fp8 bypass` pattern — device caches key
//!   on the placeholder's data pointer, so the engine's zero-change
//!   `matmul_dev` lookups hit the direct-uploaded buffers).
//! - Name mapping is shared with the legacy loader (`checkpoint_jobs`)
//! — one source of truth for fused qkv/conv, prefixes, skips.
//!
//! What still crosses the CPU (deliberately, by size):
//! - fused GDN `qkv_conv1d`/`qkv_proj` segment *bounds* are computed on
//!   the host, but the bytes still H2D from the mmap slices directly.
//! - 1-D bf16 norms/biases (KBs) convert on the host — the mmap slice
//!   to a small Vec, bounded by the actual tensor size (single-digit MB
//!   across the whole model).
//!
//! `DirectWeights` (the mmap holder) is `Arc`-shared with the serve
//! preload loop — it must outlive every `WeightView` slice.

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::Arc;

use ferrite_types::{DType, FerriteError, Result, Shape, Tensor};

use crate::checkpoint::{checkpoint_jobs, is_fp8_eligible, layer_idx_of};

/// One contiguous byte range inside a mapped shard (mmap coordinates —
/// resolved to `&[u8]` by `DirectWeights::slice`).
#[derive(Debug, Clone, Copy)]
pub struct Seg {
    pub shard: usize,
    /// Absolute byte offset in the shard (header-relative data offsets
    /// are pre-added; H2D can copy straight from the slice).
    pub off: usize,
    pub len: usize,
}

/// Raw dtype of a checkpoint entry (the mmap bytes' on-disk format).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RawDType2 {
    F32,
    F16,
    Bf16,
    Fp8E4m3,
}

impl RawDType2 {
    fn from_str(s: &str) -> Option<Self> {
        match s {
            "F32" => Some(RawDType2::F32),
            "F16" => Some(RawDType2::F16),
            "BF16" => Some(RawDType2::Bf16),
            "F8_E4M3" => Some(RawDType2::Fp8E4m3),
            _ => None,
        }
    }
}

/// A parsed checkpoint entry (header view — no data touched).
#[derive(Debug, Clone)]
pub struct DirEntry {
    pub dtype: RawDType2,
    pub shape: Vec<usize>,
    pub seg: Seg,
}

/// A single mapped `*.safetensors` shard (RAII).
struct Mmap {
    ptr: *mut u8,
    len: usize,
    // file kept open (mmap lifetime; pages are the page cache's business)
    _file: fs::File,
}

// SAFETY: the mapping is read-only and never mutated; `&[u8]` slices
// handed out borrow `self` (the struct is not moved between slice use
// and Drop — Rust borrowck enforces it).
unsafe impl Send for Mmap {}
unsafe impl Sync for Mmap {}

impl Mmap {
    fn new(path: &Path) -> Result<Self> {
        use std::os::unix::io::AsRawFd;
        let file = fs::File::open(path)
            .map_err(|e| FerriteError::Config(format!("mmap: open {}: {e}", path.display())))?;
        let len = file
            .metadata()
            .map_err(|e| FerriteError::Config(format!("mmap: stat {}: {e}", path.display())))?
            .len() as usize;
        if len == 0 {
            return Err(FerriteError::Config(format!("mmap: empty {}", path.display())));
        }
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_PRIVATE,
                file.as_raw_fd(),
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(FerriteError::Config(format!(
                "mmap: map {} ({} B) failed: {}",
                path.display(),
                len,
                std::io::Error::last_os_error()
            )));
        }
        Ok(Mmap { ptr: ptr as *mut u8, len, _file: file })
    }

    fn slice(&self, off: usize, len: usize) -> &[u8] {
        assert!(off + len <= self.len, "mmap slice out of range");
        unsafe { std::slice::from_raw_parts(self.ptr.add(off), len) }
    }

    fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl Drop for Mmap {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.ptr as *mut libc::c_void, self.len);
        }
    }
}

/// All shards mapped + the full header index (name → entry). The data
/// never leaves the page cache through this type — slices are views.
pub struct DirectWeights {
    maps: Vec<Mmap>,
    entries: HashMap<String, DirEntry>,
}

impl DirectWeights {
    /// Map every `*.safetensors` in `dir` and parse headers from the
    /// mapping itself (header read = one small page-in, no pre-read).
    pub fn open(dir: &Path) -> Result<Self> {
        let mut paths: Vec<std::path::PathBuf> = fs::read_dir(dir)
            .map_err(|e| FerriteError::Config(format!("direct: read dir {}: {e}", dir.display())))?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().map(|x| x == "safetensors").unwrap_or(false))
            .collect();
        paths.sort();
        if paths.is_empty() {
            return Err(FerriteError::Config(format!(
                "direct: no *.safetensors in {}",
                dir.display()
            )));
        }
        let mut maps = Vec::with_capacity(paths.len());
        let mut entries = HashMap::new();
        for (shard, path) in paths.iter().enumerate() {
            let m = Mmap::new(path)?;
            let bytes = m.as_slice();
            if bytes.len() < 8 {
                return Err(FerriteError::Config("direct: shard too short".into()));
            }
            let hlen = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
            if 8 + hlen > bytes.len() {
                return Err(FerriteError::Config("direct: header length out of range".into()));
            }
            let header: serde_json::Value = serde_json::from_slice(&bytes[8..8 + hlen])
                .map_err(|e| FerriteError::Config(format!("direct: header parse: {e}")))?;
            let obj = header
                .as_object()
                .ok_or_else(|| FerriteError::Config("direct: header not an object".into()))?;
            let data_base = 8 + hlen;
            for (name, v) in obj {
                if name == "__metadata__" {
                    continue;
                }
                let vo = match v.as_object() {
                    Some(o) => o,
                    None => continue,
                };
                let dtype = vo
                    .get("dtype")
                    .and_then(|d| d.as_str())
                    .and_then(RawDType2::from_str);
                let (Some(dtype), Some(shape), Some((start, end))) = (
                    dtype,
                    vo.get("shape").and_then(|s| s.as_array()).map(|a| {
                        a.iter().map(|x| x.as_u64().unwrap_or(0) as usize).collect::<Vec<_>>()
                    }),
                    vo.get("data_offsets").and_then(|o| o.as_array()).and_then(|a| {
                        if a.len() == 2 {
                            Some((a[0].as_u64()? as usize, a[1].as_u64()? as usize))
                        } else {
                            None
                        }
                    }),
                ) else {
                    continue; // visual tensors etc. — same skip set as the legacy loader
                };
                entries.insert(
                    name.clone(),
                    DirEntry {
                        dtype,
                        shape,
                        seg: Seg { shard, off: data_base + start, len: end - start },
                    },
                );
            }
            maps.push(m);
        }
        Ok(DirectWeights { maps, entries })
    }

    pub fn entry(&self, name: &str) -> Option<&DirEntry> {
        self.entries.get(name)
    }

    /// Borrow a segment (the H2D source — CPU cost is the page cache's).
    pub fn slice(&self, seg: &Seg) -> &[u8] {
        &self.maps[seg.shard].slice(seg.off, seg.len)
    }

    /// fp8 block-scale sibling (`{name}_scale_inv`) — the dequant kernel
    /// reads it on the device right after its own H2D.
    pub fn scale_entry(&self, name: &str) -> Option<&DirEntry> {
        self.entries.get(&format!("{name}_scale_inv"))
    }

    /// Decode an F32 scale segment (mmap bytes → Vec<f32>) — the scales are
    /// tiny ([rows/128, cols/128] f32, KBs per weight); this is the ONE
    /// small CPU conversion on the direct path (the weight bytes never
    /// materialize — only the scales, whose total over the whole model is
    /// a few MB).
    pub fn scale_f32(&self, e: &DirEntry) -> Vec<f32> {
        let bytes = self.slice(&e.seg);
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }
}

/// The preload instruction per ferrite weight — what the serve loop does
/// with the mmap bytes. Segment lists (fused qkv/conv) keep their order
/// (q, k, v — the legacy concat order).
#[derive(Debug, Clone)]
pub enum WeightView {
    /// bf16 raw bytes (one segment or fused-row segments) — H2D verbatim;
    /// the resident layout IS the checkpoint layout (zero conversion).
    Bf16Segs { segs: Vec<Seg>, shape: Vec<usize> },
    /// fp8 block-quantized bytes + its `_scale_inv` — H2D raw + the GPU
    /// dequant kernel (e4m3 × block-128 scale → bf16 resident).
    Fp8 { data: Seg, scale: Seg, shape: Vec<usize> },
    /// bf16 bytes the runtime consumes as f32 (embed tables, 1-D norms —
    /// the GPU bf16→f32 expand kernel widens on device; no CPU round-trip
    /// for 2.5GB embed or any norm).
    Bf16ToF32 { seg: Seg, shape: Vec<usize> },
    /// f32 bytes the runtime consumes as f32 — H2D verbatim (the checkpoint
    /// already stores f32: rare 1-D weights like scales/biases). NO bf16→f32
    /// conversion (the mmap bytes ARE the device layout).
    F32Seg { seg: Seg, shape: Vec<usize> },
}

/// The direct-loading result: the runtime placeholder table (shape-real,
/// data-empty `Tensor`s — the device caches key on the placeholder's
/// pointer, exactly the fp8-bypass pattern) + per-name preload views.
pub struct DirectView {
    /// ferrite name → placeholder (the `Weights` table the engine queries).
    pub placeholders: HashMap<String, Tensor>,
    /// ferrite name → preload instruction (mmap coordinates).
    pub views: HashMap<String, WeightView>,
    /// Arc to the mmap holder — the serve loop shares it (views borrow).
    pub direct: Arc<DirectWeights>,
    /// W8A8-eligible fp8 views (name → (data, scale) mmap segs): when
    /// FERRITE_W8A8 is on these register fp8-raw instead of dequanting.
    pub fp8_raw: HashMap<String, (Seg, Seg)>,
}

/// Placeholder tensor (data is a 4-zero shared vec — never read; the
/// device caches key on its pointer, and `Tensor::numel` carries shape).
fn placeholder(shape: Vec<usize>) -> Tensor {
    Tensor {
        shape: Shape::new(shape),
        dtype: DType::F32,
        data: Arc::new(vec![0f32; 4]),
    }
}

/// Build the direct view over a mapped checkpoint: name mapping shared
/// with the legacy loader (`checkpoint_jobs`), fp8 detection by the
/// `_scale_inv` sibling presence, fused qkv/conv as multi-segment bf16.
///
/// `w8a8` mirrors the legacy `is_fp8_eligible` gate (env-driven; off by
/// default — the fp8-raw path is the W8A8 opt-in, everything else
/// dequants to bf16 on the GPU).
pub fn load_direct(dir: &Path, cfg: &crate::config::Glm53FlashConfig) -> Result<DirectView> {
    let direct = Arc::new(DirectWeights::open(dir)?);
    let jobs = checkpoint_jobs(cfg);
    let lm = "model.language_model";
    let mut placeholders = HashMap::new();
    let mut views = HashMap::new();
    let mut fp8_raw = HashMap::new();
    for (name, src) in jobs {
        // fused qkv (GDN): q/k/v bf16 segments row-concatenated — the
        // device-side H2D writes each segment at its row offset; bytes
        // are bf16 in the checkpoint and resident (no conversion).
        if let Some(base) = src.strip_suffix("__FUSED_QKV__") {
            // base is already the full checkpoint prefix (e.g.
            // "model.language_model.layers.0.self_attn.") — do NOT strip
            // "model." or re-prepend {lm}. (the legacy loader at
            // checkpoint.rs:464 searches "{base}q_proj.weight" directly).
            let mut segs = Vec::with_capacity(3);
            let mut rows = 0usize;
            let mut cols = 0usize;
            for part in ["q_proj", "k_proj", "v_proj"] {
                let e = direct
                    .entry(&format!("{base}{part}.weight"))
                    .ok_or_else(|| {
                        FerriteError::Config(format!("direct: fused qkv missing {base}{part}.weight"))
                    })?;
                if e.dtype != RawDType2::Bf16 {
                    return Err(FerriteError::Config(format!(
                        "direct: fused qkv {base}{part} is {:?} (expected BF16 — fp8 qkv needs the GDN bf16 path)",
                        e.dtype
                    )));
                }
                segs.push(e.seg);
                rows += e.shape[0];
                cols = e.shape[1];
            }
            placeholders.insert(name.clone(), placeholder(vec![rows, cols]));
            views.insert(name, WeightView::Bf16Segs { segs, shape: vec![rows, cols] });
            continue;
        }
        // fused conv (GDN short conv): [c,1,k] squeezed — same byte
        // layout as [c,k]; q/k/v segments concatenated.
        if let Some(base) = src.strip_suffix("__FUSED_CONV__") {
            // base is already the full checkpoint prefix (e.g.
            // "model.language_model.layers.0.self_attn.") — same fix as
            // FUSED_QKV: no strip, no {lm}. prefix.
            let mut segs = Vec::with_capacity(3);
            let mut rows = 0usize;
            let mut cols = 0usize;
            for part in ["q", "k", "v"] {
                let e = direct
                    .entry(&format!("{base}{part}_conv1d.weight"))
                    .ok_or_else(|| {
                        FerriteError::Config(format!("direct: fused conv missing {base}{part}_conv1d.weight"))
                    })?;
                let (c, k) = (e.shape[0], *e.shape.get(2).unwrap_or(&1));
                segs.push(e.seg);
                rows += c;
                cols = k;
            }
            placeholders.insert(name.clone(), placeholder(vec![rows, cols]));
            views.insert(name, WeightView::Bf16Segs { segs, shape: vec![rows, cols] });
            continue;
        }
        let Some(e) = direct.entry(&src) else {
            // the legacy loader reports these as missing/skipped — same set
            continue;
        };
        let shape = e.shape.clone();
        match e.dtype {
            RawDType2::Fp8E4m3 => {
                let Some(sc) = direct.scale_entry(&src) else {
                    return Err(FerriteError::Config(format!(
                        "direct: fp8 tensor {src} without _scale_inv"
                    )));
                };
                if sc.dtype != RawDType2::F32 {
                    return Err(FerriteError::Config(format!(
                        "direct: {src}_scale_inv is {:?} (expected F32)",
                        sc.dtype
                    )));
                }
                let (data, scale) = (e.seg, sc.seg);
                // W8A8 opt-in: eligible weights register fp8-raw (the mmap
                // bytes ARE the resident layout); the default dequants on
                // the GPU (block-128 e4m3 × scale).
                if std::env::var_os("FERRITE_W8A8").is_some()
                    && is_fp8_eligible(&src, layer_idx_of(&src), cfg)
                {
                    fp8_raw.insert(name.clone(), (data, scale));
                }
                placeholders.insert(name.clone(), placeholder(shape.clone()));
                views.insert(
                    name,
                    WeightView::Fp8 {
                        data,
                        scale,
                        shape: shape.clone(),
                    },
                );
            }
            RawDType2::Bf16 => {
                // f32-consumers (embed table — 2.5GB, and the 1-D norms/biases)
                // expand on the GPU (bf16 H2D raw + widen kernel — the f32
                // resident cache); bf16-consumers (2-D GDN/DSA weights) copy
                // verbatim (the checkpoint's bf16 IS the resident layout).
                if name.contains("embed_tokens") || shape.len() < 2 {
                    placeholders.insert(name.clone(), placeholder(shape.clone()));
                    views.insert(
                        name,
                        WeightView::Bf16ToF32 { seg: e.seg, shape: shape.clone() },
                    );
                } else {
                    placeholders.insert(name.clone(), placeholder(shape.clone()));
                    views.insert(
                        name,
                        WeightView::Bf16Segs { segs: vec![e.seg], shape: shape.clone() },
                    );
                }
            }
            other => {
                // F32/F16 checkpoint entries: the checkpoint stores these
                // directly as f32 (rare 1-D weights like scales/biases/A_log).
                // F32 bytes are ALREADY the device layout — H2D verbatim,
                // NO bf16→f32 conversion (the old Bf16ToF32 classification
                // was the 2x-byte-mismatch bug: it expected numel*2 bf16
                // bytes but the mmap segment had numel*4 f32 bytes).
                if shape.len() >= 2 {
                    // 2-D f32 in the checkpoint (never hit on this release):
                    // treat as bf16-seg-pass — will byte-mismatch if it ever
                    // fires, loudly identifying the case.
                    placeholders.insert(name.clone(), placeholder(shape.clone()));
                    views.insert(
                        name,
                        WeightView::Bf16Segs { segs: vec![e.seg], shape: shape.clone() },
                    );
                } else {
                    // 1-D f32: direct H2D (the bytes are the device layout).
                    placeholders.insert(name.clone(), placeholder(shape.clone()));
                    views.insert(
                        name,
                        WeightView::F32Seg { seg: e.seg, shape: shape.clone() },
                    );
                }
                let _ = other;
            }
        }
    }
    Ok(DirectView { placeholders, views, direct, fp8_raw })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mmap_opens_and_parses_headers() {
        // the real model dir when present (b300); a no-op skip otherwise —
        // the header parse is exercised by the direct e2e on the deploy box.
        let dir = Path::new("/opt/dlami/nvme/models/GLM-5.3-Flash");
        if !dir.join("config.json").exists() {
            eprintln!("model not present, skipping");
            return;
        }
        let dw = DirectWeights::open(dir).unwrap();
        assert!(!dw.entries.is_empty());
        // one bf16 and one fp8 entry with sane shapes (the model is fp8
        // MoE + bf16 GDN — both families must be visible).
        assert!(dw.entries.values().any(|e| e.dtype == RawDType2::Bf16));
        assert!(dw.entries.values().any(|e| e.dtype == RawDType2::Fp8E4m3));
        // slices resolve inside their shards and lengths match the header
        for e in dw.entries.values() {
            let s = dw.slice(&e.seg);
            assert_eq!(s.len(), e.seg.len);
        }
    }
}
