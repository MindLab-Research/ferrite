//! HF checkpoint adapter for the real zai-org/GLM-5.3-Flash release.
//!
//! The published checkpoint is a multimodal (Glm5NextForConditionalGeneration)
//! FP8-quantised model whose tensor naming/shape differs from ferrite's
//! WeightLayout in several ways. This module converts it into the layout the
//! Engine consumes (f32, dequantised, fused where needed):
//!
//! 1. prefix: `model.language_model.*` → `model.*`; `model.visual.*` skipped.
//! 2. MTP layer (`layers.{num_hidden_layers}.*` with eh_proj/enorm/hnorm/
//!    shared_head) skipped — ferrite runs the 45 decoder layers only.
//! 3. linear-attn: `q_proj/k_proj/v_proj` (BF16, separate) are concatenated
//!    into ferrite's fused `qkv_proj`; `q/k/v_conv1d` [c,1,k] squeezed and
//!    concatenated into `qkv_conv1d` [3c, k] (order: q, k, v).
//! 4. DSA indexer keeps the real names (`indexer.wq_b`, `indexer.wk`,
//!    `indexer.k_norm.weight/bias`, `indexer.weights_proj`) — the Engine
//!    forward reads them directly.
//! 5. MoE: `shared_experts.*` → `shared_expert.*`; `gate.e_score_correction_bias`
//!    is kept (noaux-tc routing bias).
//! 6. FP8 (F8_E4M3) weights are dequantised with their `weight_scale_inv`
//!    ([rows/128, cols/128] block scales) into f32.
//! 7. `o_norm.weight` stays [head_dim] (per-head, Engine reshapes).
//! 8. Skipped as unused by ferrite v1: `dt_bias`, `indexer.index_kpool_*`
//!    (logged in the report; kpool compression & dt handling are TODO).

use std::collections::HashMap;
use std::fs;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use ferrite_types::{DType, FerriteError, Result, Shape, Tensor};
use rayon::prelude::*;

use crate::config::{Glm53FlashConfig, LayerType, MlpType};
use crate::weights::{weight_layout, Weights};

const FP8_BLOCK: usize = 128;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RawDType {
    F32,
    F16,
    Bf16,
    Fp8E4m3,
}

impl RawDType {
    fn from_str(s: &str) -> Option<Self> {
        match s {
            "F32" => Some(RawDType::F32),
            "F16" => Some(RawDType::F16),
            "BF16" => Some(RawDType::Bf16),
            "F8_E4M3" => Some(RawDType::Fp8E4m3),
            _ => None,
        }
    }
    fn elem(self) -> usize {
        match self {
            RawDType::F32 => 4,
            RawDType::F16 | RawDType::Bf16 => 2,
            RawDType::Fp8E4m3 => 1,
        }
    }
}

struct Entry {
    file: usize,
    dtype: RawDType,
    shape: Vec<usize>,
    /// absolute byte offset of this file's data section (8 + header_len);
    /// `start`/`end` are relative to it.
    data_base: u64,
    start: u64,
    end: u64,
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as i32;
    let frac = (bits & 0x3ff) as u32;
    if exp == 0 {
        if frac == 0 {
            return if sign == 1 { -0.0 } else { 0.0 };
        }
        let mut e = -1i32;
        let mut m = frac;
        while m & 0x400 == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x3ff;
        let adj = (127 - 15 + e) as u32;
        let f = f32::from_bits((sign << 31) | (adj << 23) | (m << 13));
        if sign == 1 { -f } else { f }
    } else if exp == 0x1f {
        let f = f32::from_bits((sign << 31) | (0xffu32 << 23) | (frac << 13));
        if sign == 1 { -f } else { f }
    } else {
        let adj = (exp - 15 + 127) as u32;
        let f = f32::from_bits((sign << 31) | (adj << 23) | (frac << 13));
        if sign == 1 { -f } else { f }
    }
}

fn e4m3_to_f32(b: u8) -> f32 {
    let sign = ((b >> 7) & 1) as u32;
    let exp = ((b >> 3) & 0xf) as i32;
    let frac = (b & 0x7) as u32;
    let v = if exp == 0 && frac == 0 {
        0.0f32
    } else if exp == 0 {
        (frac as f32) * (2.0f32).powi(-9)
    } else if exp == 0xf && frac == 0x7 {
        f32::NAN
    } else {
        (1.0 + (frac as f32) / 8.0) * (2.0f32).powi(exp - 7)
    };
    if sign == 1 { -v } else { v }
}

/// Summary of what the adapter did (for the load report).
#[derive(Debug, Default)]
pub struct CheckpointReport {
    pub tensors_loaded: usize,
    pub fp8_dequantized: usize,
    pub fp8_bypass: usize,
    pub fp8_placeholder: usize,
    pub fused_concat: usize,
    pub skipped_unsupported: Vec<String>,
    pub missing: Vec<String>,
}

/// Scan all *.safetensors headers in `dir` → (file list, name → entry).
fn scan_headers(dir: &Path) -> Result<(Vec<PathBuf>, HashMap<String, Entry>)> {
    let mut files: Vec<PathBuf> = fs::read_dir(dir)
        .map_err(|e| FerriteError::Config(format!("ckpt: read dir {}: {e}", dir.display())))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map(|x| x == "safetensors").unwrap_or(false))
        .collect();
    files.sort();
    if files.is_empty() {
        return Err(FerriteError::Config(format!(
            "ckpt: no *.safetensors in {}",
            dir.display()
        )));
    }
    let mut index = HashMap::new();
    for (fi, path) in files.iter().enumerate() {
        let mut f = File::open(path)
            .map_err(|e| FerriteError::Config(format!("ckpt: open {}: {e}", path.display())))?;
        let mut len_buf = [0u8; 8];
        f.read_exact(&mut len_buf)
            .map_err(|e| FerriteError::Config(format!("ckpt: read header len: {e}")))?;
        let hlen = u64::from_le_bytes(len_buf) as usize;
        let mut hdr = vec![0u8; hlen];
        f.read_exact(&mut hdr)
            .map_err(|e| FerriteError::Config(format!("ckpt: read header: {e}")))?;
        let v: serde_json::Value = serde_json::from_slice(&hdr)
            .map_err(|e| FerriteError::Config(format!("ckpt: parse header json: {e}")))?;
        let obj = v
            .as_object()
            .ok_or_else(|| FerriteError::Config("ckpt: header not an object".into()))?;
        for (name, e) in obj {
            if name == "__metadata__" {
                continue;
            }
            let vo = e.as_object().unwrap();
            let dtype = vo
                .get("dtype")
                .and_then(|d| d.as_str())
                .and_then(RawDType::from_str)
                .ok_or_else(|| FerriteError::Config(format!("ckpt: bad dtype for {name}")))?;
            let shape: Vec<usize> = vo
                .get("shape")
                .and_then(|s| s.as_array())
                .map(|a| a.iter().map(|x| x.as_u64().unwrap_or(0) as usize).collect())
                .unwrap_or_default();
            let offs = vo.get("data_offsets").and_then(|o| o.as_array()).unwrap();
            let (start, end) = (
                offs[0].as_u64().unwrap_or(0),
                offs[1].as_u64().unwrap_or(0),
            );
            index.insert(
                name.clone(),
                Entry { file: fi, dtype, shape, data_base: 8 + hlen as u64, start, end },
            );
        }
    }
    Ok((files, index))
}

fn read_entry(files: &[PathBuf], e: &Entry) -> Result<Vec<u8>> {
    let mut f = File::open(&files[e.file])
        .map_err(|er| FerriteError::Config(format!("ckpt: open shard: {er}")))?;
    // data_offsets are relative to the END of the header (8-byte len prefix +
    // hlen bytes of JSON), not to the file start.
    f.seek(SeekFrom::Start(e.data_base + e.start))
        .map_err(|er| FerriteError::Config(format!("ckpt: seek: {er}")))?;
    let n = (e.end - e.start) as usize;
    let mut buf = vec![0u8; n];
    f.read_exact(&mut buf)
        .map_err(|er| FerriteError::Config(format!("ckpt: read tensor: {er}")))?;
    Ok(buf)
}

fn to_f32(bytes: &[u8], dtype: RawDType) -> Vec<f32> {
    let n = bytes.len() / dtype.elem();
    match dtype {
        RawDType::F32 => (0..n)
            .map(|i| f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap()))
            .collect(),
        RawDType::Bf16 => (0..n)
            .map(|i| bf16_to_f32(u16::from_le_bytes(bytes[i * 2..i * 2 + 2].try_into().unwrap())))
            .collect(),
        RawDType::F16 => (0..n)
            .map(|i| f16_to_f32(u16::from_le_bytes(bytes[i * 2..i * 2 + 2].try_into().unwrap())))
            .collect(),
        RawDType::Fp8E4m3 => (0..n).map(|i| e4m3_to_f32(bytes[i])).collect(),
    }
}

/// Dequantise FP8 block-scaled weights: w[i,j] * s[i/B, j/B].
fn dequant_block(w: &[f32], s: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let srows = rows.div_ceil(FP8_BLOCK);
    let scols = cols.div_ceil(FP8_BLOCK);
    let mut out = vec![0.0f32; w.len()];
    for i in 0..rows {
        let sr = (i / FP8_BLOCK).min(srows - 1);
        for j in 0..cols {
            out[i * cols + j] = w[i * cols + j] * s[sr * scols + (j / FP8_BLOCK).min(scols - 1)];
        }
    }
    out
}

/// Load one named tensor from the checkpoint (with FP8 dequant if scaled).
/// `src` is the checkpoint-side name; scale is looked up as `{src}.weight_scale_inv`-style
/// (`{src}` ends with `.weight` for proj matrices; non-.weight params are
/// never FP8 in this checkpoint).
/// sglang `modules_to_not_convert`-aligned fp8 eligibility (GLM-5.3-Flash,
/// activation_scheme=dynamic fmt=e4m3): fp8 ONLY for the MoE/Dense expert
/// GEMMs + the MLA full-attention main projections. Everything else — GDN
/// linear attention (all submodules), MLA indexer components, the MoE router
/// (mlp.gate / e_score_correction_bias), the shared expert (checkpoint-native
/// bf16 — no `*_scale_inv` sibling), lm_head, embed and every norm — stays
/// bf16. This decides whether a `*_scale_inv` tensor's weight is served from
/// the fp8 bypass (eligible ⇒ placeholder + Fp8Weight) or bf16-recovered
/// (dequant_block → f32, which the preload bf16-encodes).
pub fn is_fp8_eligible(src: &str, layer_idx: Option<usize>, cfg: &Glm53FlashConfig) -> bool {
    // non-layer globals: never fp8
    if src == "lm_head.weight" || src == "model.embed_tokens.weight" { return false; }
    if src.ends_with("_layernorm.weight") || src.ends_with(".norm.weight")
        || src.ends_with(".enorm.weight") || src.ends_with(".hnorm.weight")
        || src.ends_with(".shared_head.norm.weight") { return false; }
    // MoE router: bf16 (+ fp32 activations per moe_router_dtype)
    if src.ends_with(".mlp.gate.weight") || src.ends_with(".mlp.e_score_correction_bias") { return false; }
    // MoE routed experts: fp8
    if src.contains(".experts.") { return true; }
    // shared expert: checkpoint-native bf16
    if src.contains(".shared_expert.") { return false; }
    let Some(li) = layer_idx else { return false; };
    // dense-MLP expert GEMMs (first_k_dense_replace)
    if matches!(cfg.mlp_types.get(li), Some(MlpType::Dense))
        && (src.ends_with(".gate_proj.weight") || src.ends_with(".up_proj.weight")
            || src.ends_with(".down_proj.weight"))
    {
        return true;
    }
    // MLA (deepseek_sparse_attention, or the MTP/nextn layer 45 — treated as
    // DSA): main projections fp8; indexer components bf16; GDN bf16.
    let is_dsa = matches!(cfg.layer_types.get(li), Some(LayerType::DeepseekSparseAttention))
        || li >= cfg.layer_types.len();
    if is_dsa {
        if src.contains("self_attn.indexer.") { return false; }
        // e2e 2026-09-06: MLA main projections measured NEGATIVE (dsa at=6.0ms
        // /layer vs ~4 bf16 — 44 small-matrix W8A8 gemvs at 0.6x; accept
        // 2.38->2.17 from x-e4m3 argmax flips). lm_head (154880 rows) and MoE
        // experts stay fp8 (HBM-bound wins). Re-enable with the 6-matrix mega
        // gemv (shared xq quant, N=8 real columns) that fixes the 0.6x.
        return false;
    }
    false
}

fn layer_idx_of(src: &str) -> Option<usize> {
    let m = src.find("layers.")?;
    let after = &src[m + 7..];
    let end = after.find('.')?;
    after[..end].parse().ok()
}

fn load_named(
    files: &[PathBuf],
    index: &HashMap<String, Entry>,
    src: &str,
    rep: &mut CheckpointReport,
    cfg: &Glm53FlashConfig,
) -> Result<(Tensor, Option<crate::weights::Fp8Weight>)> {
    let e = index
        .get(src)
        .ok_or_else(|| FerriteError::Config(format!("ckpt: missing tensor {src}")))?;
    let shape: Vec<usize> = e.shape.clone();
    if let Some(sc) = index.get(&format!("{src}_scale_inv")) {
        // FP8 block dequant (weight → {src}.weight_scale_inv; src ends with ".weight")
        let raw = read_entry(files, e)?;
        let sraw = read_entry(files, sc)?;
        if sc.dtype != RawDType::F32 {
            return Err(FerriteError::Config(format!("ckpt: {src}_scale_inv not F32")));
        }
        let s = to_f32(&sraw, RawDType::F32);
        let rows = shape[0];
        let cols = if shape.len() > 1 { shape[1] } else { 1 };
        let expect = rows.div_ceil(FP8_BLOCK) * cols.div_ceil(FP8_BLOCK);
        if s.len() != expect {
            return Err(FerriteError::Config(format!(
                "ckpt: {src}_scale_inv len {} != {} (block {FP8_BLOCK})",
                s.len(),
                expect
            )));
        }
        // Single-store: fp8-served weights never materialize the dequantized
        // f32 (the checkpoint's native precision IS the fp8+scales — the bf16
        // path re-quantized it anyway). The placeholder keeps a unique heap
        // pointer (fp8_map keys on (ptr, numel)) and the real shape (dims are
        // read by shard/device-chain code); its data is NEVER touched on the
        // cuda path (fp8 hit ⇒ bf16 upload skipped, matmul serves fp8).
        // keep_f32 = consumers that read the f32 directly (fused-MoE bf16
        // pointer tables, hc dev_weight f32, embed lookup, fused-qkv concat).
        let fp8 = crate::weights::Fp8Weight {
            rows,
            cols,
            data: raw, // the F8 bytes (pre-f32-conversion)
            scale: s,
        };
        if is_fp8_eligible(src, layer_idx_of(src), cfg) {
            // sglang-aligned fp8 single-store: eligible weights (MoE/Dense
            // experts, MLA main projections) never materialize the dequantized
            // f32 — the checkpoint-native fp8+scales IS the precision (the
            // bf16 path re-quantized it anyway). The placeholder keeps a unique
            // heap ptr (fp8_map keys on (ptr, numel)) + the real shape (dims
            // are read by shard/device-chain code); data is untouched on the
            // cuda path (fp8 hit ⇒ bf16 upload skipped, matmul serves fp8).
            rep.fp8_placeholder += 1;
            let placeholder = Tensor {
                shape: Shape::new(shape),
                dtype: DType::F32,
                data: std::sync::Arc::new(vec![0f32; 4]),
            };
            Ok((placeholder, Some(fp8)))
        } else {
            // ineligible-but-`_scale_inv` present (MoE router, shared expert if
            // any): bf16-recover — dequant_block → f32 (preload bf16-encodes).
            rep.fp8_dequantized += 1;
            let w = to_f32(&fp8.data, RawDType::Fp8E4m3);
            let sraw = read_entry(files, sc)?;
            let s = to_f32(&sraw, RawDType::F32);
            let data = dequant_block(&w, &s, rows, cols);
            Ok((Tensor::new(Shape::new(shape), DType::F32, data), Some(fp8)))
        }
    } else {
        let raw = read_entry(files, e)?;
        let data = to_f32(&raw, e.dtype);
        Ok((Tensor::new(Shape::new(shape), DType::F32, data), None))
    }
}

/// Concatenate two row-major [rows, cols] tensors along dim 0.
fn concat_rows(a: &[f32], ra: usize, b: &[f32], rb: usize, cols: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(a.len() + b.len());
    out.extend_from_slice(a);
    out.extend_from_slice(b);
    debug_assert_eq!(out.len(), (ra + rb) * cols);
    out
}

/// Load the real HF GLM-5.3-Flash checkpoint into ferrite's WeightLayout.
///
/// `dir` is the model directory (config.json + model-*.safetensors).
/// Returns the f32 weight set (all names/quantities exactly matching
/// `weight_layout(cfg)`) plus a report.
pub fn load_hf_checkpoint(
    dir: &Path,
    cfg: &Glm53FlashConfig,
) -> Result<(Weights, crate::weights::Weights8, CheckpointReport)> {
    let (files, index) = scan_headers(dir)?;
    let mut rep = CheckpointReport::default();
    let lm = "model.language_model";

    // ---- build the job list (pure name mapping, no I/O) ----
    let mut jobs: Vec<(String, String)> = Vec::new(); // (ferrite name, checkpoint src or "")
    for spec in weight_layout(cfg).specs {
        let name = spec.name;
        let src = if let Some(pfx) = name.strip_suffix("qkv_proj.weight") {
            let base = pfx.strip_prefix("model.").unwrap_or(pfx).to_string();
            let src = format!("{lm}.{base}__FUSED_QKV__");
            jobs.push((name, src));
            continue;
        } else if let Some(pfx) = name.strip_suffix("qkv_conv1d.weight") {
            let base = pfx.strip_prefix("model.").unwrap_or(pfx).to_string();
            let src = format!("{lm}.{base}__FUSED_CONV__");
            jobs.push((name, src));
            continue;
        } else if name == "model.embed_tokens.weight" {
            format!("{lm}.embed_tokens.weight")
        } else if name == "model.norm.weight" {
            format!("{lm}.norm.weight")
        } else if name == "lm_head.weight" {
            "lm_head.weight".to_string()
        } else if let Some(rest) = name.strip_prefix("model.layers.") {
            let (l, r) = match rest.split_once('.') {
                Some(x) => x,
                None => {
                    rep.skipped_unsupported.push(name.clone());
                    continue;
                }
            };
            let lidx: usize = l.parse().unwrap_or(usize::MAX);
            if lidx > cfg.num_hidden_layers {
                rep.skipped_unsupported.push(name.clone());
                continue;
            }
            // lidx == num_hidden_layers: the MTP (nextn) layer — eh_proj /
            // enorm / hnorm / DSA attn / MoE mlp / shared_head.norm flow
            // through the same name mapping as decoder layers.
            // shared_expert (ferrite) → shared_experts (checkpoint, plural)
            let r = r.replacen("shared_expert.", "shared_experts.", 1);
            format!("{lm}.layers.{l}.{r}")
        } else {
            rep.skipped_unsupported.push(name.clone());
            continue;
        };
        jobs.push((name, src));
    }

    // ---- run jobs in parallel (rayon): each job reads + converts its own
    // tensors; FP8 dequant happens per-tensor inside load_named ----
    let results: Vec<Result<(String, Tensor, Option<crate::weights::Fp8Weight>)>> = jobs
        .into_par_iter()
        .map(|(name, src)| {
            if let Some(base) = src.strip_suffix("__FUSED_QKV__") {
                let mut r = CheckpointReport::default();
                let (q, _) = load_named(&files, &index, &format!("{base}q_proj.weight"), &mut r, cfg)?;
                let (k, _) = load_named(&files, &index, &format!("{base}k_proj.weight"), &mut r, cfg)?;
                let (v, _) = load_named(&files, &index, &format!("{base}v_proj.weight"), &mut r, cfg)?;
                let rows = q.shape.0[0] + k.shape.0[0] + v.shape.0[0];
                let c = q.shape.0[1];
                let data = concat_rows(
                    &concat_rows(q.as_slice(), q.shape.0[0], k.as_slice(), k.shape.0[0], c),
                    rows,
                    v.as_slice(),
                    v.shape.0[0],
                    c,
                );
                // fp8 bypass for fused-qkv is NOT concatenable per-block (the
                // per-part 128-block grid breaks at row seams) → bf16 path.
                return Ok((name, Tensor::new(Shape::new(vec![rows, c]), DType::F32, data), None));
            }
            if let Some(base) = src.strip_suffix("__FUSED_CONV__") {
                let mut r = CheckpointReport::default();
                let mut parts: Vec<Tensor> = Vec::new();
                for b in ["q", "k", "v"] {
                    let (t, _) = load_named(&files, &index, &format!("{base}{b}_conv1d.weight"), &mut r, cfg)?;
                    let (c1, c2, c3) = (
                        t.shape.0[0],
                        *t.shape.0.get(1).unwrap_or(&1),
                        *t.shape.0.get(2).unwrap_or(&1),
                    );
                    if c2 != 1 {
                        return Err(FerriteError::Config(format!(
                            "ckpt: {base}{b}_conv1d shape {c1}x{c2}x{c3} unexpected"
                        )));
                    }
                    parts.push(Tensor::new(Shape::new(vec![c1, c3]), DType::F32, t.as_slice().to_vec()));
                }
                let rows: usize = parts.iter().map(|t| t.shape.0[0]).sum();
                let c = parts[0].shape.0[1];
                let mut data = Vec::new();
                for p in &parts {
                    data.extend_from_slice(p.as_slice());
                }
                return Ok((name, Tensor::new(Shape::new(vec![rows, c]), DType::F32, data), None));
            }
            let mut r = CheckpointReport::default();
            // fp8 eligibility is decided inside load_named via is_fp8_eligible
            // (sglang modules_to_not_convert aligned): MoE/Dense experts + MLA
            // main projections → fp8 single-store (placeholder); everything
            // else (GDN, MLA indexer, MoE router, lm_head, embed, norms, hc)
            // → bf16. Pass cfg so the layer-type classification is available.
            let (t, fp8) = load_named(&files, &index, &src, &mut r, cfg)?;
            Ok((name, t, fp8))
        })
        .collect();

    // ---- collect + validate shapes ----
    let layout: std::collections::HashMap<String, Vec<usize>> = weight_layout(cfg)
        .specs
        .into_iter()
        .map(|s| (s.name.clone(), s.shape.0.clone()))
        .collect();
    let mut w: Weights = HashMap::new();
    let mut w8: crate::weights::Weights8 = HashMap::new();
    for res in results {
        let (name, t, fp8) = res?;
        if let Some(expect) = layout.get(&name) {
            if t.shape.0 != *expect {
                return Err(FerriteError::Config(format!(
                    "ckpt: {name}: shape {:?} != expected {:?}",
                    t.shape.0, expect
                )));
            }
        }
        if name.ends_with("qkv_proj.weight") || name.ends_with("qkv_conv1d.weight") {
            rep.fused_concat += 1;
        } else {
            rep.tensors_loaded += 1;
        }
        if let Some(f8) = fp8 {
            rep.fp8_bypass += 1;
            w8.insert(name.clone(), f8);
        }
        w.insert(name, t);
    }
    Ok((w, w8, rep))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_roundtrip() {
        assert_eq!(bf16_to_f32(0x3F80), 1.0);
        assert_eq!(bf16_to_f32(0xC000), -2.0);
        assert_eq!(f16_to_f32(0x3C00), 1.0);
        assert_eq!(f16_to_f32(0x4000), 2.0);
        assert_eq!(e4m3_to_f32(0x38), 1.0);
        assert_eq!(e4m3_to_f32(0x00), 0.0);
    }

    #[test]
    fn dequant_block_simple() {
        // 2x4 "matrix", block 128 → single scale covering everything
        let w = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let s = vec![0.5f32];
        let out = dequant_block(&w, &s, 2, 4);
        assert_eq!(out, vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]);
    }

    /// Probe a real checkpoint tensor end-to-end through load_named (needs the
    /// GLM-5.3-Flash download present; skipped otherwise).
    #[test]
    fn real_ckpt_probe() {
        let dir = Path::new("/opt/dlami/nvme/models/GLM-5.3-Flash");
        if !dir.join("config.json").exists() {
            eprintln!("model not present, skipping");
            return;
        }
        let (files, index) = scan_headers(dir).unwrap();
        let mut rep = CheckpointReport::default();
        // down_proj is fp8-ELIGIBLE → load_named serves the placeholder (no
        // dequantized f32). Directly dequant here to validate the block-scale
        // math against the reference f32 values.
        let de = &index["model.language_model.layers.5.mlp.experts.51.down_proj.weight"];
        let raw = read_entry(&files, de).unwrap();
        let w = to_f32(&raw, RawDType::Fp8E4m3);
        let se = &index["model.language_model.layers.5.mlp.experts.51.down_proj.weight_scale_inv"];
        let sraw = read_entry(&files, se).unwrap();
        let s = to_f32(&sraw, RawDType::F32);
        let rows = de.shape[0];
        let cols = if de.shape.len() > 1 { de.shape[1] } else { 1 };
        let data = dequant_block(&w, &s, rows, cols);
        let t = Tensor::new(Shape::new(vec![rows, cols]), DType::F32, data);
        let sl = t.as_slice();
        let nbad = sl.iter().filter(|v| !v.is_finite()).count();
        let first: Vec<usize> = sl
            .iter()
            .enumerate()
            .filter(|(_, v)| !v.is_finite())
            .map(|(i, _)| i)
            .take(4)
            .collect();
        println!(
            "down_proj: dtype={:?} shape={:?} n_bad={} first_bad={:?} w[0..4]={:?}",
            t.dtype,
            t.shape.0,
            nbad,
            first,
            &sl[..4]
        );
        let s = s.clone();
        // dump the raw entry the Rust scanner resolved, to compare with python
        let probe_e = index
            .get("model.language_model.layers.5.mlp.experts.51.down_proj.weight_scale_inv")
            .unwrap();
        let raw_s = read_entry(&files, probe_e).unwrap();
        let head_f32: Vec<f32> = raw_s[..16]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        println!(
            "scale entry: file={} start={} end={} dtype={:?} shape={:?}",
            probe_e.file, probe_e.start, probe_e.end, probe_e.dtype, probe_e.shape
        );
        println!("scale raw first-16-bytes as f32: {head_f32:?}");
        let (mn, mx) = s
            .as_slice()
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), v| (a.min(*v), b.max(*v)));
        println!(
            "scale: dtype=F32 len={} min={mn} max={mx} n_bad={}",
            s.len(),
            s.iter().filter(|v| !v.is_finite()).count()
        );
        assert!(nbad == 0, "expert weight has non-finite values after dequant");
    }
}
#[cfg(test)]
mod fp8_eligibility_tests {
    use super::*;
    use crate::config::{Glm53FlashConfig, LayerType, MlpType};

    #[test]
    fn fp8_eligibility_sglang_aligned() {
        let cfg = Glm53FlashConfig::test_config();
        let n = cfg.num_hidden_layers;
        // globals -> never fp8
        assert!(!is_fp8_eligible("lm_head.weight", None, &cfg));
        assert!(!is_fp8_eligible("model.embed_tokens.weight", None, &cfg));
        assert!(!is_fp8_eligible("model.norm.weight", None, &cfg));
        assert!(!is_fp8_eligible("model.language_model.layers.5.mlp.gate.weight", None, &cfg));
        // GDN linear-attention layer (idx 0 = LinearAttention): all bf16
        let gdn = cfg.layer_types.iter().position(|t| *t == LayerType::LinearAttention).unwrap();
        assert!(!is_fp8_eligible(&format!("model.layers.{gdn}.self_attn.qkv_proj.weight"), Some(gdn), &cfg));
        assert!(!is_fp8_eligible(&format!("model.layers.{gdn}.self_attn.o_proj.weight"), Some(gdn), &cfg));
        assert!(!is_fp8_eligible(&format!("model.layers.{gdn}.self_attn.f_a_proj.weight"), Some(gdn), &cfg));
        // MLA full-attn layer (idx 1 = DeepseekSparseAttention): main proj bf16
        // (e2e 2026-09-06 rollback: 44 small-matrix W8A8 gemvs at 0.6x + accept
        // 2.38->2.17 argmax flips were net-negative; revisit with the 6-matrix
        // mega gemv)
        let dsa = cfg.layer_types.iter().position(|t| *t == LayerType::DeepseekSparseAttention).unwrap();
        assert!(!is_fp8_eligible(&format!("model.layers.{dsa}.self_attn.q_a_proj.weight"), Some(dsa), &cfg));
        assert!(!is_fp8_eligible(&format!("model.layers.{dsa}.self_attn.kv_a_proj_with_mqa.weight"), Some(dsa), &cfg));
        assert!(!is_fp8_eligible(&format!("model.layers.{dsa}.self_attn.q_b_proj.weight"), Some(dsa), &cfg));
        assert!(!is_fp8_eligible(&format!("model.layers.{dsa}.self_attn.o_proj.weight"), Some(dsa), &cfg));
        // MLA indexer components -> bf16
        assert!(!is_fp8_eligible(&format!("model.layers.{dsa}.self_attn.indexer.wq_b.weight"), Some(dsa), &cfg));
        assert!(!is_fp8_eligible(&format!("model.layers.{dsa}.self_attn.indexer.weights_proj.weight"), Some(dsa), &cfg));
        assert!(!is_fp8_eligible(&format!("model.layers.{dsa}.self_attn.indexer.k_norm.weight"), Some(dsa), &cfg));
        // MoE router (mlp.gate / e_score_correction_bias) -> bf16
        assert!(!is_fp8_eligible(&format!("model.layers.{dsa}.mlp.gate.weight"), Some(dsa), &cfg));
        assert!(!is_fp8_eligible(&format!("model.layers.{dsa}.mlp.e_score_correction_bias"), Some(dsa), &cfg));
        // MoE experts -> fp8
        assert!(is_fp8_eligible(&format!("model.layers.{dsa}.mlp.experts.0.gate_proj.weight"), Some(dsa), &cfg));
        assert!(is_fp8_eligible(&format!("model.layers.{dsa}.mlp.experts.0.down_proj.weight"), Some(dsa), &cfg));
        // shared expert -> bf16 (checkpoint-native)
        assert!(!is_fp8_eligible(&format!("model.layers.{dsa}.mlp.shared_expert.gate_proj.weight"), Some(dsa), &cfg));
        // dense-MLP layer (idx 0 = Dense mlp): gate/up/down fp8
        let dense = cfg.mlp_types.iter().position(|m| *m == MlpType::Dense).unwrap();
        assert!(is_fp8_eligible(&format!("model.layers.{dense}.mlp.gate_proj.weight"), Some(dense), &cfg));
        assert!(is_fp8_eligible(&format!("model.layers.{dense}.mlp.down_proj.weight"), Some(dense), &cfg));
        // MTP/nextn layer (li == num_hidden_layers): treated as DSA — main proj fp8, indexer bf16
        assert!(is_fp8_eligible(&format!("model.layers.{n}.self_attn.q_a_proj.weight"), Some(n), &cfg));
        assert!(!is_fp8_eligible(&format!("model.layers.{n}.self_attn.indexer.wq_b.weight"), Some(n), &cfg));
    }
}
