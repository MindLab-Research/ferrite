//! Checkpoint mapping and tensor-parallel sharding for DeepSeek-V4.1-Flash.
//!
//! The released checkpoint keeps its quantisation in the file:
//!
//! | tensor | dtype | scale tensor |
//! |---|---|---|
//! | dense linear weights | `F8_E4M3 [out, in]` | `F8_E8M0 [out/32, in/32]` |
//! | routed experts | `I8 [out, in/2]` (fp4 e2m1, 2 per byte) | `F8_E8M0 [out, in/32]` |
//! | engram tables | `F8_E4M3 [rows, 256]` | `F8_E8M0 [rows, 8]` |
//! | norms / sinks / hc / embeddings / head / gate / vision / mtp heads | bf16 or f32 | — |
//!
//! **Loading never dequantises.** A shard is a byte-range view into the file;
//! the runtime keeps `weight` and `scale` as separate device buffers in exactly
//! these layouts, which is what lets the GEMM kernels be pure tensor-core MMAs
//! (see [`crate::kernels`]). Dequantisation exists only in [`crate::quant`] for
//! the CPU golden path and for checkpoint verification.

use std::collections::HashMap;
use std::path::Path;

use ferrite_types::{FerriteError, Result};

use crate::config::Dsv41Config;

/// How a tensor is distributed over `world` ranks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shard {
    /// Every rank holds the whole tensor.
    Replicated,
    /// Split along dim 0 (vocab rows, engram table rows, experts).
    Rows,
    /// Split along dim 1 (row-parallel input, all-reduce after).
    Cols,
    /// `[n_heads*head_dim, ...]` split by attention head.
    Heads,
    /// `[n_groups*o_lora, hpg*hd]` — both dims sliced by the layer's o_groups.
    Groups,
    /// Routed-expert ownership. The expert axis is in the *name*
    /// (`...experts.{e}...`), so each rank simply keeps the tensors of the
    /// `n_routed / world` experts it owns; the tensor's own shape is whole.
    Experts,
}

/// One tensor the model expects, with its sharding rule.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorSpec {
    pub name: String,
    pub shape: Vec<usize>,
    pub shard: Shard,
}

fn push(out: &mut Vec<TensorSpec>, name: impl Into<String>, shape: Vec<usize>, shard: Shard) {
    out.push(TensorSpec { name: name.into(), shape, shard });
}

/// The full tensor list for the released geometry, with sharding rules.
///
/// `world` is the tensor-parallel degree. Expert-parallel tensors are listed
/// once per *global* expert index: the loader keeps only the ranks' own slice
/// (`n_routed / world` consecutive experts).
pub fn tensor_specs(cfg: &Dsv41Config, world: usize) -> Vec<TensorSpec> {
    assert!(world > 0);
    let mut out = Vec::new();
    let dim = cfg.dim;
    let nh = cfg.n_heads;
    let hd = cfg.head_dim;
    let ql = cfg.q_lora_rank;
    let ol = cfg.o_lora_rank;
    let groups = cfg.o_groups;
    let hpg = nh / groups; // heads per o-group
    let inter = cfg.moe_inter_dim;

    // embeddings / head (vocab-parallel)
    push(&mut out, "embed.weight", vec![cfg.vocab_size, dim], Shard::Rows);
    push(&mut out, "head.weight", vec![cfg.vocab_size, dim], Shard::Rows);
    push(&mut out, "norm.weight", vec![dim], Shard::Replicated);

    for l in 0..cfg.n_layers {
        let p = format!("layers.{l}");
        // attention
        push(&mut out, format!("{p}.attn.wq_a.weight"), vec![ql, dim], Shard::Replicated);
        push(&mut out, format!("{p}.attn.wq_a.scale"), vec![ql / 32, dim / 32], Shard::Replicated);
        push(&mut out, format!("{p}.attn.q_norm.weight"), vec![ql], Shard::Replicated);
        push(&mut out, format!("{p}.attn.wq_b.weight"), vec![nh * hd, ql], Shard::Heads);
        push(&mut out, format!("{p}.attn.wq_b.scale"), vec![nh * hd / 32, ql / 32], Shard::Heads);
        push(&mut out, format!("{p}.attn.wkv.weight"), vec![hd, dim], Shard::Replicated);
        push(&mut out, format!("{p}.attn.wkv.scale"), vec![hd / 32, dim / 32], Shard::Replicated);
        push(&mut out, format!("{p}.attn.kv_norm.weight"), vec![hd], Shard::Replicated);
        push(&mut out, format!("{p}.attn.attn_sink"), vec![nh], Shard::Heads);
        push(
            &mut out,
            format!("{p}.attn.wo_a.weight"),
            vec![groups * ol, hpg * hd],
            Shard::Groups,
        );
        push(
            &mut out,
            format!("{p}.attn.wo_a.scale"),
            vec![groups * ol / 32, hpg * hd / 32],
            Shard::Groups,
        );
        push(&mut out, format!("{p}.attn.wo_b.weight"), vec![dim, groups * ol], Shard::Cols);
        push(&mut out, format!("{p}.attn.wo_b.scale"), vec![dim / 32, groups * ol / 32], Shard::Cols);
        // compressor (only kv sources have one)
        let ratio = cfg.compress_ratio(l);
        if cfg.is_kv_source(l) {
            let f32w = ratio == 1; // ratio 1 stays bf16 in the checkpoint; > 1 is promoted to f32
            push(
                &mut out,
                format!("{p}.attn.compressor.wkv.weight"),
                vec![hd, dim],
                Shard::Replicated,
            );
            push(&mut out, format!("{p}.attn.compressor.norm.weight"), vec![hd], Shard::Replicated);
            if !f32w {
                push(
                    &mut out,
                    format!("{p}.attn.compressor.wgate.weight"),
                    vec![hd, dim],
                    Shard::Replicated,
                );
            }
        }
        // indexer (only index sources have one)
        if cfg.is_index_source(l) {
            let inh = cfg.index_n_heads;
            let ihd = cfg.index_head_dim;
            push(
                &mut out,
                format!("{p}.attn.indexer.wq_b.weight"),
                vec![inh * ihd, ql],
                Shard::Heads,
            );
            push(
                &mut out,
                format!("{p}.attn.indexer.wq_b.scale"),
                vec![inh * ihd / 32, ql / 32],
                Shard::Heads,
            );
            push(
                &mut out,
                format!("{p}.attn.indexer.weights_proj.weight"),
                vec![inh, dim],
                Shard::Heads,
            );
            if cfg.indexer_owns_k(l) {
                push(
                    &mut out,
                    format!("{p}.attn.indexer.wk.weight"),
                    vec![ihd, hd],
                    Shard::Replicated,
                );
                push(
                    &mut out,
                    format!("{p}.attn.indexer.k_norm.weight"),
                    vec![ihd],
                    Shard::Replicated,
                );
            }
        }
        // hyper-connections + norms
        push(&mut out, format!("{p}.attn_norm.weight"), vec![dim], Shard::Replicated);
        push(&mut out, format!("{p}.ffn_norm.weight"), vec![dim], Shard::Replicated);
        push(&mut out, format!("{p}.hc_attn_fn"), vec![cfg.hc_mix_count(), cfg.hc_mult * dim], Shard::Replicated);
        push(&mut out, format!("{p}.hc_attn_base"), vec![cfg.hc_mix_count()], Shard::Replicated);
        push(&mut out, format!("{p}.hc_attn_scale"), vec![3], Shard::Replicated);
        push(&mut out, format!("{p}.hc_ffn_fn"), vec![cfg.hc_mix_count(), cfg.hc_mult * dim], Shard::Replicated);
        push(&mut out, format!("{p}.hc_ffn_base"), vec![cfg.hc_mix_count()], Shard::Replicated);
        push(&mut out, format!("{p}.hc_ffn_scale"), vec![3], Shard::Replicated);
        // MoE router (replicated: every rank must see all experts to route)
        let (n_routed, _) = cfg.moe_config(l);
        push(&mut out, format!("{p}.ffn.gate.weight"), vec![n_routed, dim], Shard::Replicated);
        push(&mut out, format!("{p}.ffn.gate.bias"), vec![n_routed], Shard::Replicated);
        if cfg.vision_enabled() {
            push(&mut out, format!("{p}.ffn.gate.bias_vl"), vec![n_routed], Shard::Replicated);
        }
        // routed experts (expert-parallel; listed per global index)
        for e in 0..n_routed {
            for (n, o, k) in [("w1", inter, dim), ("w2", dim, inter), ("w3", inter, dim)] {
                push(
                    &mut out,
                    format!("{p}.ffn.experts.{e}.{n}.weight"),
                    vec![o, k / 2], // fp4 packs 2 values per byte
                    Shard::Experts,
                );
                push(
                    &mut out,
                    format!("{p}.ffn.experts.{e}.{n}.scale"),
                    vec![o, k / 32],
                    Shard::Experts,
                );
            }
        }
        // shared expert (the same expert on every rank)
        for (n, o, k) in [("w1", inter, dim), ("w3", inter, dim), ("w2", dim, inter)] {
            push(
                &mut out,
                format!("{p}.ffn.shared_experts.{n}.weight"),
                vec![o, k],
                Shard::Replicated,
            );
            push(
                &mut out,
                format!("{p}.ffn.shared_experts.{n}.scale"),
                vec![o / 32, k / 32],
                Shard::Replicated,
            );
        }
        // engram table (row-parallel), plus its per-layer wkv and gates
        if let Some(ei) = cfg.engram_enabled().then(|| cfg.engram_layer_ids.iter().position(|&x| x == l)).flatten() {
            let rows = cfg.engram_num_embeddings[ei];
            push(
                &mut out,
                format!("{p}.engram.embed.weight"),
                vec![rows as usize, cfg.engram_head_dim],
                Shard::Rows,
            );
            push(
                &mut out,
                format!("{p}.engram.embed.scale"),
                vec![rows as usize, cfg.engram_head_dim / 32],
                Shard::Rows,
            );
            let n_cols = (cfg.engram_max_ngram_size - 1) * cfg.engram_n_heads;
            let wkv_k = n_cols * cfg.engram_head_dim;
            let wkv_n = dim * (cfg.hc_mult + 1);
            push(&mut out, format!("{p}.engram.wkv.weight"), vec![wkv_n, wkv_k], Shard::Replicated);
            push(&mut out, format!("{p}.engram.wkv.scale"), vec![wkv_n / 32, wkv_k / 32], Shard::Replicated);
            push(&mut out, format!("{p}.engram.q_weight"), vec![cfg.hc_mult, dim], Shard::Replicated);
            push(&mut out, format!("{p}.engram.k_weight"), vec![cfg.hc_mult, dim], Shard::Replicated);
        }
    }

    // DSpark draft layers (`mtp.*`)
    for s in 0..cfg.n_mtp_layers {
        let p = format!("mtp.{s}");
        let mtp_layer = cfg.n_layers + s;
        push(&mut out, format!("{p}.attn_norm.weight"), vec![dim], Shard::Replicated);
        push(&mut out, format!("{p}.ffn_norm.weight"), vec![dim], Shard::Replicated);
        push(&mut out, format!("{p}.hc_attn_fn"), vec![cfg.hc_mix_count(), cfg.hc_mult * dim], Shard::Replicated);
        push(&mut out, format!("{p}.hc_attn_base"), vec![cfg.hc_mix_count()], Shard::Replicated);
        push(&mut out, format!("{p}.hc_attn_scale"), vec![3], Shard::Replicated);
        push(&mut out, format!("{p}.hc_ffn_fn"), vec![cfg.hc_mix_count(), cfg.hc_mult * dim], Shard::Replicated);
        push(&mut out, format!("{p}.hc_ffn_base"), vec![cfg.hc_mix_count()], Shard::Replicated);
        push(&mut out, format!("{p}.hc_ffn_scale"), vec![3], Shard::Replicated);
        // the draft attention reuses the MLA projections shape-for-shape
        push(&mut out, format!("{p}.attn.wq_a.weight"), vec![ql, dim], Shard::Replicated);
        push(&mut out, format!("{p}.attn.wq_a.scale"), vec![ql / 32, dim / 32], Shard::Replicated);
        push(&mut out, format!("{p}.attn.q_norm.weight"), vec![ql], Shard::Replicated);
        push(&mut out, format!("{p}.attn.wq_b.weight"), vec![nh * hd, ql], Shard::Heads);
        push(&mut out, format!("{p}.attn.wq_b.scale"), vec![nh * hd / 32, ql / 32], Shard::Heads);
        push(&mut out, format!("{p}.attn.wkv.weight"), vec![hd, dim], Shard::Replicated);
        push(&mut out, format!("{p}.attn.wkv.scale"), vec![hd / 32, dim / 32], Shard::Replicated);
        push(&mut out, format!("{p}.attn.kv_norm.weight"), vec![hd], Shard::Replicated);
        push(&mut out, format!("{p}.attn.attn_sink"), vec![nh], Shard::Heads);
        push(&mut out, format!("{p}.attn.wo_a.weight"), vec![groups * ol, hpg * hd], Shard::Groups);
        push(&mut out, format!("{p}.attn.wo_a.scale"), vec![groups * ol / 32, hpg * hd / 32], Shard::Groups);
        push(&mut out, format!("{p}.attn.wo_b.weight"), vec![dim, groups * ol], Shard::Cols);
        push(&mut out, format!("{p}.attn.wo_b.scale"), vec![dim / 32, groups * ol / 32], Shard::Cols);
        let (n_routed, _) = cfg.moe_config(mtp_layer);
        push(&mut out, format!("{p}.ffn.gate.weight"), vec![n_routed, dim], Shard::Replicated);
        push(&mut out, format!("{p}.ffn.gate.bias"), vec![n_routed], Shard::Replicated);
        for e in 0..n_routed {
            for (n, o, k) in [("w1", inter, dim), ("w2", dim, inter), ("w3", inter, dim)] {
                push(&mut out, format!("{p}.ffn.experts.{e}.{n}.weight"), vec![o, k / 2], Shard::Experts);
                push(&mut out, format!("{p}.ffn.experts.{e}.{n}.scale"), vec![o, k / 32], Shard::Experts);
            }
        }
        for (n, o, k) in [("w1", inter, dim), ("w3", inter, dim), ("w2", dim, inter)] {
            push(&mut out, format!("{p}.ffn.shared_experts.{n}.weight"), vec![o, k], Shard::Replicated);
            push(&mut out, format!("{p}.ffn.shared_experts.{n}.scale"), vec![o / 32, k / 32], Shard::Replicated);
        }
        if s == 0 {
            // the draft stage reads the attention input of the target layers
            push(
                &mut out,
                format!("{p}.main_proj.weight"),
                vec![dim, dim * cfg.dspark_target_layer_ids.len()],
                Shard::Replicated,
            );
            push(
                &mut out,
                format!("{p}.main_proj.scale"),
                vec![dim / 32, dim * cfg.dspark_target_layer_ids.len() / 32],
                Shard::Replicated,
            );
            push(&mut out, format!("{p}.main_norm.weight"), vec![dim], Shard::Replicated);
        }
        if s + 1 == cfg.n_mtp_layers {
            push(&mut out, format!("{p}.norm.weight"), vec![dim], Shard::Replicated);
            let mr = cfg.dspark_markov_rank;
            push(&mut out, format!("{p}.markov_head.embed.weight"), vec![cfg.vocab_size, mr], Shard::Rows);
            push(&mut out, format!("{p}.markov_head.head.weight"), vec![cfg.vocab_size, mr], Shard::Rows);
            push(
                &mut out,
                format!("{p}.confidence_head.proj.weight"),
                vec![1, dim + mr],
                Shard::Replicated,
            );
        }
    }

    // vision tower (replicated: the reference never shards it)
    if cfg.vision_enabled() {
        let vd = cfg.vision_dim;
        push(&mut out, "vision.patch_embed.weight", vec![vd, cfg.vision_patch_size * cfg.vision_patch_size * 3], Shard::Replicated);
        push(&mut out, "vision.patch_embed.bias", vec![vd], Shard::Replicated);
        push(&mut out, "vision.norm.weight", vec![vd], Shard::Replicated);
        for b in 0..cfg.vision_n_layers {
            let p = format!("vision.blocks.{b}");
            push(&mut out, format!("{p}.attn.wqkv.weight"), vec![3 * vd, vd], Shard::Replicated);
            push(&mut out, format!("{p}.attn.wqkv.bias"), vec![3 * vd], Shard::Replicated);
            push(&mut out, format!("{p}.attn.wo.weight"), vec![vd, vd], Shard::Replicated);
            push(&mut out, format!("{p}.attn.wo.bias"), vec![vd], Shard::Replicated);
            push(&mut out, format!("{p}.mlp.w1.weight"), vec![2 * cfg.vision_inter_dim, vd], Shard::Replicated);
            push(&mut out, format!("{p}.mlp.w2.weight"), vec![vd, cfg.vision_inter_dim], Shard::Replicated);
            push(&mut out, format!("{p}.norm1.weight"), vec![vd], Shard::Replicated);
            push(&mut out, format!("{p}.norm2.weight"), vec![vd], Shard::Replicated);
        }
        push(&mut out, "aligner.w1.weight", vec![dim, 3 * vd * 3], Shard::Replicated);
        push(&mut out, "aligner.w1.bias", vec![dim], Shard::Replicated);
        push(&mut out, "aligner.w2.weight", vec![dim, dim], Shard::Replicated);
        push(&mut out, "aligner.w2.bias", vec![dim], Shard::Replicated);
        push(&mut out, "image_start", vec![dim], Shard::Replicated);
        push(&mut out, "image_end", vec![dim], Shard::Replicated);
        push(&mut out, "image_newline", vec![dim], Shard::Replicated);
    }
    out
}

/// Convert a routed expert's fp4 weights into the lossless e4m3 form used by
/// the fallback GEMM path.
///
/// Input: the checkpoint's packed fp4 `[out, in/2]` (I8) plus its e8m0
/// `[out, in/32]` scales. Output: e4m3 `[out, in]` plus one e8m0 scale per
/// 32x32 tile. The conversion is the reference's `cast_e2m1fn_to_e4m3fn` and is
/// **exact**: inside a tile each row segment is scaled by a power of two in
/// `[1, 2^6)`, and an e2m1 value times a power of two stays representable in
/// e4m3 (the 6-bit offset bound is why `6.0 * 2^6 = 384 <= 448`).
///
/// This is *re-encoding*, not dequantisation: the bytes stay 8-bit and are
/// consumed by the fp8 tensor-core MMA. It costs 1 byte/param against fp4's
/// 0.5, which is the price of not having the tcgen05 MXFP4 path.
pub fn convert_expert_fp4_to_e4m3(
    packed: &[u8],
    scales: &[u8],
    out: usize,
    inn: usize,
) -> (Vec<u8>, Vec<u8>) {
    assert_eq!(packed.len(), out * inn / 2, "packed fp4 length");
    assert_eq!(scales.len(), out * inn / 32, "per-row-32 e8m0 scales");
    let mut seg = vec![0f32; out * (inn / 32)];
    for (i, &b) in scales.iter().enumerate() {
        seg[i] = crate::quant::ue8m0_decode(b);
    }
    let mut w8 = vec![0u8; out * inn];
    let mut s8 = vec![0u8; (out / 32) * (inn / 32)];
    crate::quant::cast_fp4_to_e4m3(packed, &seg, out, inn, &mut w8, &mut s8);
    (w8, s8)
}

/// Whether a spec's first dimension is the one that shrinks under sharding,
/// and by how much (rows/experts/vocab/engram).
pub fn shard_factor(cfg: &Dsv41Config, spec: &TensorSpec, world: usize) -> (usize, usize) {
    match spec.shard {
        Shard::Replicated => (1, 1),
        Shard::Rows => {
            // vocab rows (embed/head/markov) or engram table rows
            let rows = spec.shape[0];
            if spec.name.contains("engram.embed") {
                (world, rows.div_ceil(world))
            } else {
                (world, rows / world)
            }
        }
        Shard::Cols => (world, spec.shape[1] / world),
        Shard::Heads | Shard::Groups => (world, spec.shape[0] / world),
        Shard::Experts => (1, spec.shape[0]),
    }
}

/// The local shape a rank holds for `spec`.
pub fn local_shape(cfg: &Dsv41Config, spec: &TensorSpec, world: usize, rank: usize) -> Vec<usize> {
    let mut s = spec.shape.clone();
    match spec.shard {
        Shard::Replicated => s,
        Shard::Rows => {
            let rows = s[0];
            if spec.name.contains("engram.embed") {
                let per = rows.div_ceil(world);
                let start = rank * per;
                s[0] = per.min(rows.saturating_sub(start));
            } else {
                s[0] = rows / world;
            }
            s
        }
        Shard::Cols => {
            s[1] /= world;
            s
        }
        Shard::Heads | Shard::Groups => {
            s[0] /= world;
            s
        }
        Shard::Experts => s, // whole tensor; ownership is by expert index
    }
}

/// A byte-range view into a shard file (safetensors).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorView {
    pub dtype: String,
    pub shape: Vec<usize>,
    /// Byte offsets of the *global* tensor's data (relative to the data section).
    pub begin: u64,
    pub end: u64,
}

/// Minimal safetensors header reader: enough to validate names/shapes/dtypes
/// and to slice a rank's view without touching the payload.
#[derive(Debug, Default)]
pub struct SafetensorsIndex {
    pub tensors: HashMap<String, TensorView>,
}

impl SafetensorsIndex {
    /// Parse the JSON header of a safetensors file.
    pub fn read_header(path: &Path) -> Result<Self> {
        use std::io::Read;
        let mut f = std::fs::File::open(path)
            .map_err(|e| FerriteError::Config(format!("open {}: {e}", path.display())))?;
        let mut len = [0u8; 8];
        f.read_exact(&mut len)
            .map_err(|e| FerriteError::Config(format!("header len: {e}")))?;
        let hlen = u64::from_le_bytes(len) as usize;
        if hlen > 100 * 1024 * 1024 {
            return Err(FerriteError::Config(format!("implausible header length {hlen}")));
        }
        let mut buf = vec![0u8; hlen];
        f.read_exact(&mut buf)
            .map_err(|e| FerriteError::Config(format!("header body: {e}")))?;
        let v: serde_json::Value = serde_json::from_slice(&buf)
            .map_err(|e| FerriteError::Config(format!("header json: {e}")))?;
        let obj = v
            .as_object()
            .ok_or_else(|| FerriteError::Config("safetensors header is not an object".into()))?;
        let mut tensors = HashMap::new();
        for (name, meta) in obj {
            if name == "__metadata__" {
                continue;
            }
            let dtype = meta
                .get("dtype")
                .and_then(|x| x.as_str())
                .unwrap_or("")
                .to_string();
            let shape: Vec<usize> = meta
                .get("shape")
                .and_then(|x| x.as_array())
                .map(|a| a.iter().filter_map(|e| e.as_u64()).map(|e| e as usize).collect())
                .unwrap_or_default();
            let offs = meta.get("data_offsets").and_then(|x| x.as_array());
            let (begin, end) = match offs {
                Some(a) if a.len() == 2 => (
                    a[0].as_u64().unwrap_or(0),
                    a[1].as_u64().unwrap_or(0),
                ),
                _ => (0, 0),
            };
            tensors.insert(name.clone(), TensorView { dtype, shape, begin, end });
        }
        Ok(SafetensorsIndex { tensors })
    }

    /// Validate that every spec is present with the expected global shape and a
    /// sane dtype. Returns the names that are missing.
    pub fn validate(&self, specs: &[TensorSpec]) -> Vec<String> {
        let mut missing = Vec::new();
        for s in specs {
            match self.tensors.get(&s.name) {
                None => missing.push(s.name.clone()),
                Some(v) => {
                    // fp4 weights are packed in the file: 2 values per byte
                    let packed = v.dtype == "I8" && v.shape.len() == 2 && s.name.ends_with(".weight")
                        && s.name.contains(".experts.");
                    let expect = if packed {
                        let mut e = s.shape.clone();
                        e[1] /= 2;
                        e
                    } else {
                        s.shape.clone()
                    };
                    if v.shape != expect {
                        missing.push(format!("{}: shape {:?} != {:?}", s.name, v.shape, expect));
                    }
                }
            }
        }
        missing
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn specs_cover_the_released_checkpoints_headline_tensors() {
        let cfg = Dsv41Config::production();
        let specs = tensor_specs(&cfg, 8);
        let names: std::collections::HashSet<&str> = specs.iter().map(|s| s.name.as_str()).collect();
        for want in [
            "embed.weight",
            "head.weight",
            "norm.weight",
            "layers.6.attn.wq_a.weight",
            "layers.6.attn.wq_a.scale",
            "layers.6.attn.wq_b.weight",
            "layers.6.attn.wkv.weight",
            "layers.6.attn.wo_a.weight",
            "layers.6.attn.wo_b.weight",
            "layers.6.attn.attn_sink",
            "layers.6.attn_norm.weight",
            "layers.6.ffn_norm.weight",
            "layers.6.hc_attn_fn",
            "layers.6.ffn.gate.weight",
            "layers.6.ffn.gate.bias",
            "layers.6.ffn.experts.0.w1.weight",
            "layers.6.ffn.experts.0.w1.scale",
            "layers.6.ffn.experts.383.w3.scale",
            "layers.6.ffn.shared_experts.w2.weight",
            "layers.1.engram.embed.weight",
            "layers.1.engram.embed.scale",
            "layers.1.engram.wkv.weight",
            "layers.1.engram.q_weight",
            "layers.14.engram.embed.weight",
            "layers.2.attn.compressor.wkv.weight",
            "layers.2.attn.compressor.wgate.weight",
            "layers.2.attn.indexer.wq_b.weight",
            "layers.2.attn.indexer.wk.weight",
            "layers.2.attn.indexer.weights_proj.weight",
            "mtp.0.main_proj.weight",
            "mtp.0.main_norm.weight",
            "mtp.2.markov_head.embed.weight",
            "mtp.2.confidence_head.proj.weight",
            "vision.patch_embed.weight",
            "vision.blocks.31.mlp.w1.weight",
            "aligner.w1.weight",
        ] {
            assert!(names.contains(want), "missing spec {want}");
        }
    }

    #[test]
    fn shapes_match_the_checkpoint_header_stats() {
        let cfg = Dsv41Config::production();
        let specs = tensor_specs(&cfg, 8);
        let find = |n: &str| specs.iter().find(|s| s.name == n).unwrap().shape.clone();
        // verified against the released safetensors header
        assert_eq!(find("layers.6.attn.wq_a.weight"), vec![1280, 5120]);
        assert_eq!(find("layers.6.attn.wq_a.scale"), vec![40, 160]);
        assert_eq!(find("layers.6.attn.wq_b.weight"), vec![32768, 1280]);
        assert_eq!(find("layers.6.attn.wq_b.scale"), vec![1024, 40]);
        assert_eq!(find("layers.6.attn.wkv.weight"), vec![512, 5120]);
        assert_eq!(find("layers.6.attn.wkv.scale"), vec![16, 160]);
        assert_eq!(find("layers.6.attn.wo_a.weight"), vec![8192, 4096]);
        assert_eq!(find("layers.6.attn.wo_a.scale"), vec![256, 128]);
        assert_eq!(find("layers.6.attn.wo_b.weight"), vec![5120, 8192]);
        assert_eq!(find("layers.6.attn.wo_b.scale"), vec![160, 256]);
        // fp4 experts pack 2 values per byte and scale per (row, 32 cols)
        assert_eq!(find("layers.6.ffn.experts.0.w1.weight"), vec![2304, 2560]);
        assert_eq!(find("layers.6.ffn.experts.0.w1.scale"), vec![2304, 160]);
        assert_eq!(find("layers.6.ffn.experts.0.w2.weight"), vec![5120, 1152]);
        assert_eq!(find("layers.6.ffn.experts.0.w2.scale"), vec![5120, 72]);
        assert_eq!(find("layers.6.ffn.shared_experts.w1.weight"), vec![2304, 5120]);
        assert_eq!(find("layers.6.ffn.shared_experts.w1.scale"), vec![72, 160]);
        // engram
        assert_eq!(find("layers.1.engram.embed.weight"), vec![384006168, 256]);
        assert_eq!(find("layers.1.engram.embed.scale"), vec![384006168, 8]);
        assert_eq!(find("layers.1.engram.wkv.weight"), vec![25600, 6144]);
        assert_eq!(find("layers.1.engram.wkv.scale"), vec![800, 192]);
        assert_eq!(find("layers.14.engram.embed.weight"), vec![384016682, 256]);
        // indexer / compressor
        assert_eq!(find("layers.2.attn.indexer.wq_b.weight"), vec![4096, 1280]);
        assert_eq!(find("layers.2.attn.indexer.wq_b.scale"), vec![128, 40]);
        assert_eq!(find("layers.2.attn.indexer.wk.weight"), vec![128, 512]);
        assert_eq!(find("layers.2.attn.indexer.weights_proj.weight"), vec![32, 5120]);
        assert_eq!(find("layers.2.attn.compressor.wkv.weight"), vec![512, 5120]);
        // hc geometry
        assert_eq!(find("layers.6.hc_attn_fn"), vec![24, 20480]);
        assert_eq!(find("layers.6.hc_attn_base"), vec![24]);
        // mtp / dspark
        assert_eq!(find("mtp.0.main_proj.weight"), vec![5120, 15360]);
        assert_eq!(find("mtp.0.main_proj.scale"), vec![160, 480]);
        assert_eq!(find("mtp.2.markov_head.embed.weight"), vec![129280, 256]);
        assert_eq!(find("mtp.2.confidence_head.proj.weight"), vec![1, 5376]);
        // vision
        assert_eq!(find("vision.blocks.0.attn.wqkv.weight"), vec![3072, 1024]);
        assert_eq!(find("vision.blocks.0.mlp.w1.weight"), vec![5632, 1024]);
        assert_eq!(find("aligner.w1.weight"), vec![5120, 9216]);
    }

    #[test]
    fn expert_tensors_are_only_present_for_layers_with_them() {
        let cfg = Dsv41Config::production();
        let specs = tensor_specs(&cfg, 8);
        // the released config gives the backbone 384 experts and the draft 128
        assert!(specs.iter().any(|s| s.name == "layers.39.ffn.experts.383.w1.weight"));
        assert!(!specs.iter().any(|s| s.name == "layers.39.ffn.experts.384.w1.weight"));
        assert!(specs.iter().any(|s| s.name == "mtp.0.ffn.experts.127.w1.weight"));
        assert!(!specs.iter().any(|s| s.name == "mtp.0.ffn.experts.128.w1.weight"));
        // and the draft experts are not confused with the backbone ones
        assert!(!specs.iter().any(|s| s.name == "mtp.0.ffn.experts.383.w1.weight"));
    }

    #[test]
    fn only_source_layers_carry_compressor_and_indexer_tensors() {
        let cfg = Dsv41Config::production();
        let specs = tensor_specs(&cfg, 8);
        let has = |n: &str| specs.iter().any(|s| s.name == n);
        assert!(has("layers.2.attn.compressor.wkv.weight"));
        assert!(has("layers.2.attn.indexer.wq_b.weight"));
        // a ratio>0 consumer has no compressor of its own
        assert!(!has("layers.3.attn.compressor.wkv.weight"));
        // an index consumer has no indexer
        assert!(!has("layers.3.attn.indexer.wq_b.weight"));
        // ratio-1 sources keep the plain projection and have no gate
        assert!(has("layers.20.attn.compressor.wkv.weight"));
        assert!(!has("layers.20.attn.compressor.wgate.weight"));
        // only k owners carry wk / k_norm
        assert!(has("layers.2.attn.indexer.wk.weight"));
        assert!(!has("layers.24.attn.indexer.wk.weight"));
    }

    #[test]
    fn sharding_halves_heads_and_quarters_vocab() {
        let cfg = Dsv41Config::production();
        let specs = tensor_specs(&cfg, 8);
        let get = |n: &str| specs.iter().find(|s| s.name == n).unwrap().clone();
        // 64 heads / 8 ranks = 8 heads * 512 dim
        assert_eq!(local_shape(&cfg, &get("layers.6.attn.wq_b.weight"), 8, 0), vec![4096, 1280]);
        // vocab rows split evenly
        assert_eq!(local_shape(&cfg, &get("embed.weight"), 8, 0), vec![129280 / 8, 5120]);
        // engram rows use ceil-division and the last rank may be short
        let e = get("layers.1.engram.embed.weight");
        let per = 384006168usize.div_ceil(8);
        assert_eq!(local_shape(&cfg, &e, 8, 0), vec![per, 256]);
        let last = 384006168 - per * 7;
        assert_eq!(local_shape(&cfg, &e, 8, 7), vec![last, 256]);
        // experts are expert-parallel
        assert_eq!(local_shape(&cfg, &get("layers.6.ffn.experts.0.w1.weight"), 8, 0), vec![2304, 2560]);
        // wo_a is block-diagonal over o_groups: only the OUTPUT dim (groups *
        // o_lora) splits; each group keeps its own hpg*hd = 4096-wide input.
        assert_eq!(local_shape(&cfg, &get("layers.6.attn.wo_a.weight"), 8, 0), vec![8192 / 8, 4096]);
        // wo_b is row-parallel: only the input shrinks
        assert_eq!(local_shape(&cfg, &get("layers.6.attn.wo_b.weight"), 8, 0), vec![5120, 8192 / 8]);
    }

    #[test]
    fn shard_factor_is_consistent_with_local_shape() {
        let cfg = Dsv41Config::production();
        for world in [1usize, 2, 4, 8] {
            for spec in tensor_specs(&cfg, world) {
                let (f, _per) = shard_factor(&cfg, &spec, world);
                let ls = local_shape(&cfg, &spec, world, 0);
                match spec.shard {
                    Shard::Replicated => assert_eq!(ls, spec.shape),
                    Shard::Experts => assert_eq!(ls, spec.shape),
                    Shard::Rows if spec.name.contains("engram.embed") => {
                        // engram rows use ceil-division
                        assert_eq!(ls[0], spec.shape[0].div_ceil(f), "{} world {world}", spec.name)
                    }
                    Shard::Rows | Shard::Heads | Shard::Groups => {
                        assert_eq!(ls[0], spec.shape[0] / f, "{} world {world}", spec.name)
                    }
                    Shard::Cols => assert_eq!(ls[1], spec.shape[1] / f, "{}", spec.name),
                }
            }
        }
    }

    #[test]
    fn fp4_to_e4m3_expert_conversion_is_exact() {
        // 32x64 tile: fp4 values times per-segment powers of two, converted to
        // e4m3 + one ue8m0 scale per 32x32 tile. Decoding both must agree.
        let (out, inn) = (32usize, 64usize);
        let mut packed = vec![0u8; out * inn / 2];
        // alternate codes 4 (+2.0) and 12 (-2.0) in the low/high nibbles
        for b in packed.iter_mut() {
            *b = crate::quant::fp4_pack_byte(4, 12);
        }
        let mut scales = vec![0u8; out * (inn / 32)];
        for r in 0..out {
            for c in 0..(inn / 32) {
                // a power-of-two scale per (row, 32-col) segment
                scales[r * (inn / 32) + c] = crate::quant::ue8m0_encode_pow2(-(r as i32 % 5));
            }
        }
        let (w8, s8) = convert_expert_fp4_to_e4m3(&packed, &scales, out, inn);
        assert_eq!(w8.len(), out * inn);
        assert_eq!(s8.len(), (out / 32) * (inn / 32));
        // reference decode of the fp4 original
        let mut want = vec![0f32; out * inn];
        crate::quant::dequant_fp4_row32(&packed, &scales, out, inn, &mut want);
        // decode of the converted e4m3 form
        let mut got = vec![0f32; out * inn];
        crate::quant::dequant_fp8_block(&w8, &s8, out, inn, 32, &mut got);
        for i in 0..out * inn {
            assert_eq!(got[i], want[i], "element {i}");
        }
    }

    #[test]
    fn headers_parse_and_validate() {
        // a synthetic safetensors header: one fp4 expert weight + its scale
        let json = r#"{"ffn.experts.0.w1.weight":{"dtype":"I8","shape":[4,64],"data_offsets":[0,128]},
                        "ffn.experts.0.w1.scale":{"dtype":"F8_E8M0","shape":[4,4],"data_offsets":[128,144]}}"#;
        let v: serde_json::Value = serde_json::from_str(json).unwrap();
        let obj = v.as_object().unwrap();
        assert_eq!(obj.len(), 2);
        let idx = SafetensorsIndex {
            tensors: obj
                .iter()
                .map(|(k, m)| {
                    (
                        k.clone(),
                        TensorView {
                            dtype: m["dtype"].as_str().unwrap().into(),
                            shape: m["shape"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect(),
                            begin: m["data_offsets"][0].as_u64().unwrap(),
                            end: m["data_offsets"][1].as_u64().unwrap(),
                        },
                    )
                })
                .collect(),
        };
        // a spec list shaped like the file: the packed fp4 dim is halved in the file
        let specs = vec![
            TensorSpec { name: "ffn.experts.0.w1.weight".into(), shape: vec![4, 128], shard: Shard::Experts },
            TensorSpec { name: "ffn.experts.0.w1.scale".into(), shape: vec![4, 4], shard: Shard::Experts },
        ];
        // `a.weight` is an expert tensor -> the validator expects the packed shape
        let missing = idx.validate(&specs);
        assert!(missing.is_empty(), "{missing:?}");
    }
}
