//! Direct mmap preload — the serve integration of the disk→GPU path.
//!
//! The legacy preload: `load_hf_checkpoint` (CPU f32 expansion, ~660GB
//! peak RSS, ~80s) → per-rank `preload_weight` (f32→bf16 convert + H2D).
//! The direct path: `load_direct` (mmap shards + placeholder table) →
//! THIS module per rank: the shard's placeholder shapes carry the TP
//! windows (row/col/head/EP splits mirror `shard_weights_tp` exactly),
//! and the mmap slices stream straight into the backend's direct-preload
//! entry points — bf16 verbatim H2D, fp8 + GPU dequant, bf16→f32 GPU
//! expand, column windows via pitched `cudaMemcpy2D`. The CPU never
//! materializes a weight: page cache → PCIe → device kernels.
//!
//! The window rules are a MIRROR of `ferrite_exec::tp::shard_weights_tp`
//! (kept beside it — a change there must be mirrored here; the shapes
//! cross-check at runtime: the shard placeholder's numel must equal the
//! window's product, and a mismatch is a hard error, not silent corruption).

use std::collections::HashMap;

use ferrite_kernel::CudaBackend;
use ferrite_model::direct::{DirectView, WeightView};
use ferrite_model::Glm53FlashConfig;
use ferrite_types::{FerriteError, Result, Tensor};

/// head_range mirror (tp.rs): contiguous even split of `n` heads.
fn head_range(n: usize, rank: usize, world: usize) -> (usize, usize) {
    let per = n / world;
    let rem = n % world;
    let start = rank * per + rank.min(rem);
    let end = start + per + if rank < rem { 1 } else { 0 };
    (start, end)
}

/// What TP did to one weight (mirrors shard_weights_tp's classification).
#[derive(Debug, Clone, Copy)]
enum Split {
    /// Full tensor on every rank (norms, embed, lm_head, router, hc_*).
    Replicated,
    /// Row window [r0, r1) — even rows*rank/world (gate/up) or head blocks
    /// (q_b/kv_b/b_proj/f_b/g_b/dt_bias over head_range).
    Rows { r0: usize, r1: usize },
    /// Column window [c0, c1) — even cols*rank/world (down/o_proj/eh_proj)
    /// or head blocks (o_proj's hs*dk..he*dk).
    Cols { c0: usize, c1: usize },
    /// Whole weight on ranks e ∈ [es, ee) (MoE experts, EP-style).
    Expert { es: usize, ee: usize },
    /// fused qkv/qkv_conv: each of the q/k/v thirds head-split
    /// [hs*dk, he*dk).
    QkvHeads { hs: usize, he: usize, dk: usize },
}

/// The view's full (unsharded) shape — the split-of input.
fn view_shape(v: &WeightView) -> &[usize] {
    match v {
        WeightView::Bf16Segs { shape, .. } => shape,
        WeightView::Fp8 { shape, .. } => shape,
        WeightView::Bf16ToF32 { shape, .. } => shape,
        WeightView::F32Seg { shape, .. } => shape,
    }
}

fn split_of(name: &str, full: &[usize], cfg: &Glm53FlashConfig, rank: usize, world: usize) -> Split {
    let rows = full.first().copied().unwrap_or(1);
    let cols = full.get(1).copied().unwrap_or(1);
    let heads = cfg.linear_attn.num_heads;
    let dk = cfg.linear_attn.head_dim;
    let dsa_h = cfg.dsa.num_attention_heads;
    let nope = cfg.dsa.qk_nope_head_dim;
    let vd = cfg.dsa.v_head_dim;
    let n_exp = cfg.n_routed_experts;
    let (hs, he) = head_range(heads, rank, world);
    let (dhs, dhe) = head_range(dsa_h, rank, world);
    let (es, ee) = head_range(n_exp, rank, world);

    if world == 1 {
        return Split::Replicated;
    }
    // globals (mirror shard_weights_tp)
    if name == "model.embed_tokens.weight" || name == "lm_head.weight" {
        return Split::Replicated; // vocab split is device-side (full + mask)
    }
    let replicated_suffixes = [
        ".enorm.weight", ".hnorm.weight", ".shared_head.norm.weight",
        "input_layernorm.weight", "q_a_layernorm.weight", "kv_a_layernorm.weight",
        "indexer_norm.weight", "hc_attn_base", "hc_attn_scale", "hc_attn_fn",
        "hc_ffn_base", "hc_ffn_scale", "hc_ffn_fn", "mlp.gate.weight",
        ".o_norm.weight", // o_norm is [head_dim] per-head shared — replicated.
        // NOTE: .a_log and .dt_bias were wrongly here (they're [heads] /
        // [heads*dk] — HEAD-SPLIT, matching tp.rs's head_split). The
        // replicated classification passed the FULL bytes to a rank whose
        // placeholder had the head-split shape → preload byte-count mismatch
        // (12 bytes vs numel 3 * 2 = 6). Fixed: they fall through to the
        // GDN head-split section below.
    ];
    if name.starts_with("model.norm.weight") || name == "model.embed_tokens.weight" {
        return Split::Replicated;
    }
    if replicated_suffixes.iter().any(|s| name.ends_with(s)) {
        return Split::Replicated;
    }
    if name.ends_with(".eh_proj.weight") {
        let c0 = cols * rank / world;
        return Split::Cols { c0, c1: cols * (rank + 1) / world };
    }
    // layer-local (layer kind distinguishes linear-attn vs DSA)
    let layer_idx: Option<usize> = name
        .strip_prefix("model.layers.")
        .and_then(|r| r.split('.').next())
        .and_then(|l| l.parse().ok());
    let Some(li) = layer_idx else {
        return Split::Replicated; // non-layer global
    };
    let is_dsa = cfg.layer_types.get(li).map(|t| matches!(t, ferrite_model::LayerType::DeepseekSparseAttention)).unwrap_or(li >= cfg.layer_types.len());
    // MoE experts: EP whole (mirror)
    if name.split(".experts.").nth(1).is_some() {
        return Split::Expert { es, ee };
    }
    if name.ends_with(".gate_proj.weight") || name.ends_with(".up_proj.weight") {
        return Split::Rows { r0: rows * rank / world, r1: rows * (rank + 1) / world };
    }
    if name.ends_with(".down_proj.weight") {
        return Split::Cols { c0: cols * rank / world, c1: cols * (rank + 1) / world };
    }
    if is_dsa {
        if name.ends_with(".q_b_proj.weight") {
            return Split::Rows { r0: dhs * nope, r1: dhe * nope };
        }
        if name.ends_with(".kv_b_proj.weight") {
            return Split::Rows { r0: dhs * (nope + vd), r1: dhe * (nope + vd) };
        }
        if name.ends_with(".o_proj.weight") {
            return Split::Cols { c0: dhs * vd, c1: dhe * vd };
        }
        // q_a/kv_a/indexer/ape/gate: replicated (mirror shard_dsa_weight)
        return Split::Replicated;
    }
    // linear-attn (GDN)
    if name.ends_with(".qkv_proj.weight") || name.ends_with(".qkv_conv1d.weight") {
        return Split::QkvHeads { hs, he, dk };
    }
    if name.ends_with(".b_proj.weight") || name.ends_with(".A_log") {
        return Split::Rows { r0: hs, r1: he };
    }
    if name.ends_with(".dt_bias") {
        return Split::Rows { r0: hs * dk, r1: he * dk };
    }
    if name.ends_with(".f_b_proj.weight") || name.ends_with(".g_b_proj.weight") {
        return Split::Rows { r0: hs * dk, r1: he * dk };
    }
    if name.ends_with(".o_proj.weight") {
        return Split::Cols { c0: hs * dk, c1: he * dk };
    }
    // f_a/g_a/o_norm/indexer weights: replicated
    Split::Replicated
}

/// Gather a row window of a row-major f32 scale grid [srows, full_scols].
fn scale_row_window(scale: &[f32], full_scols: usize, r0: usize, r1: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity((r1 - r0) * full_scols);
    for r in r0..r1 {
        let base = r * full_scols;
        out.extend_from_slice(&scale[base..base + full_scols]);
    }
    out
}

/// Preload stats (one log line per rank).
#[derive(Debug, Default)]
pub struct DirectPreloadStats {
    pub bf16_rows: usize,
    pub bf16_cols: usize,
    pub fp8_rows: usize,
    pub fp8_cols: usize,
    pub bf16_full: usize,
    pub f32_expand: usize,
    pub experts: usize,
}

/// Direct-preload ONE rank's shard: every weight name in `shard` (the
/// cluster's placeholder table — shape-only TP windows from
/// `shard_weights_tp`, data is a 4-elem stub) resolves to its DirectView
/// mmap window and streams into the backend's device cache. The
/// placeholder's (ptr, numel) IS the runtime cache key — the engine's
/// `matmul_dev`/`dev_weight_bf16` lookups hit these direct-uploaded
/// buffers with zero changes.
pub fn direct_preload_shard(
    backend: &CudaBackend,
    dv: &DirectView,
    cfg: &Glm53FlashConfig,
    rank: usize,
    world: usize,
    shard: &HashMap<String, Tensor>,
) -> Result<DirectPreloadStats> {
    let mut st = DirectPreloadStats::default();
    for (name, ph) in shard {
        let Some(view) = dv.views.get(name) else {
            // Names absent from the direct view (skipped/unsupported by the
            // checkpoint adapter — visual tensors etc.): the legacy table
            // never had them either; nothing to preload.
            continue;
        };
        // window mirror: shard placeholder shape == the window's product
        // (the runtime cross-check against shard_weights_tp's split).
        let split = split_of(name, view_shape(view), cfg, rank, world);
        match view {
            WeightView::Bf16Segs { segs, shape } => match split {
                Split::Replicated => {
                    let slices: Vec<&[u8]> = segs.iter().map(|s| dv.direct.slice(s)).collect();
                    // mmap sanity: first 8 bytes of replicated weights (the
                    // lm_head is replicated — all-zero here = wrong offset →
                    // all-zero logits → constant "!" output)
                    if std::env::var_os("FERRITE_MMAP_DEBUG").is_some() && name.contains("lm_head") {
                        let b = &slices[0][..8.min(slices[0].len())];
                        eprintln!("[mmap-dbg] {name}: first 8 bytes {:02x?} (len={})", b, slices[0].len());
                    }
                    backend.preload_bf16_raw(ph, &slices)?;
                    st.bf16_full += 1;
                }
                Split::Rows { r0, r1 } => {
                    if segs.len() == 1 {
                        let (w, row_bytes) = (shape.get(1).copied().unwrap_or(1), shape[0]);
                        let rb = w * 2;
                        let full = dv.direct.slice(&segs[0]);
                        // mmap sanity: first 8 bytes of split weights (Rows)
                        if std::env::var_os("FERRITE_MMAP_DEBUG").is_some() && name.contains("layers.0.") {
                            let s = &full[r0 * rb..(r0 * rb + 8).min(r1 * rb)];
                            eprintln!("[mmap-dbg] ROWS {name}: r0={r0} r1={r1} rb={rb} first8={:02x?} len={}", s, full.len());
                        }
                        backend.preload_bf16_raw(ph, &[&full[r0 * rb..r1 * rb]])?;
                        let _ = row_bytes;
                        st.bf16_rows += 1;
                    } else {
                        return Err(FerriteError::Config(format!(
                            "direct: fused {name} got Rows split (expected QkvHeads)"
                        )));
                    }
                }
                Split::Cols { c0, c1 } => {
                    if segs.len() != 1 {
                        return Err(FerriteError::Config(format!(
                            "direct: fused {name} got Cols split (fused weights are GDN head-split)"
                        )));
                    }
                    let rows = shape[0];
                    let full_cols = shape[1];
                    let full = dv.direct.slice(&segs[0]);
                    backend.preload_bf16_col_raw(ph, full, rows, full_cols, c0, c1)?;
                    st.bf16_cols += 1;
                }
                Split::QkvHeads { hs, he, dk } => {
                    // three q/k/v segments, each head-split [hs*dk, he*dk)
                    if segs.len() != 3 {
                        return Err(FerriteError::Config(format!(
                            "direct: {name} has {} segs (expected 3 for fused qkv)",
                            segs.len()
                        )));
                    }
                    let cols = shape[1];
                    let rb = cols * 2;
                    let mut slices: Vec<&[u8]> = Vec::with_capacity(3);
                    for seg in segs {
                        let full = dv.direct.slice(seg);
                        slices.push(&full[hs * dk * rb..he * dk * rb]);
                    }
                    backend.preload_bf16_raw(ph, &slices)?;
                    st.bf16_rows += 1;
                }
                Split::Expert { es, ee } => {
                    let e: usize = name
                        .split(".experts.")
                        .nth(1)
                        .and_then(|r| r.split('.').next())
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(usize::MAX);
                    if e >= es && e < ee {
                        let slices: Vec<&[u8]> = segs.iter().map(|s| dv.direct.slice(s)).collect();
                        backend.preload_bf16_raw(ph, &slices)?;
                        st.experts += 1;
                    }
                    // else: another rank owns this expert — the shard table
                    // would not carry it (shard_weights_tp dropped it); a
                    // stray name here is the EP mirror drift (hard error below)
                }
            },
            WeightView::Fp8 { data, scale, shape } => {
                let (rows, cols) = (shape[0], shape.get(1).copied().unwrap_or(1));
                // decode the mmap scale bytes (KBs — [rows/128, cols/128] f32)
                let bytes = dv.direct.slice(scale);
                let scale_full: Vec<f32> = bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                    .collect();
                let scols_full = cols.div_ceil(128);
                // fp8 weights: register the SHARD-CORRECT fp8 bypass (same
                // numerical path as the legacy loader: fp8 GEMV W8A16) AND
                // dequant to bf16 (dev_weight_bf16 fallback). The legacy path
                // registers the SPLIT fp8 (fp8_row/fp8_col produce per-rank
                // shard fp8 + scale); registering the FULL fp8 here would make
                // the fp8 GEMV read 4x the rows → wrong numerics. The bf16
                // dequant below is shard-correct (row/col windows on data and
                // scale) and serves dev_weight_bf16 consumers.
                let scols_full = cols.div_ceil(128);
                let d = dv.direct.slice(data);
                match split {
                    Split::Replicated => {
                        backend.preload_fp8_dequant(ph, d, &scale_full, rows, cols)?;
                        st.fp8_rows += 1;
                    }
                    Split::Rows { r0, r1 } => {
                        let d = dv.direct.slice(data);
                        let srows = (r1 - r0).div_ceil(128);
                        let scols = cols.div_ceil(128);
                        let _ = (srows, scols);
                        // scale: the row window's block rows [r0/128, r1/128)
                        // (the kernel indexes scale[r>>7][c>>7] over the
                        // SHARD rows — the window's blocks are the full
                        // grid's rows r0/128..r1/128, full cols).
                        let sw = scale_row_window(&scale_full, scols_full, r0 / 128, r1.div_ceil(128));
                        backend.preload_fp8_dequant(ph, &d[r0 * cols..r1 * cols], &sw, r1 - r0, cols)?;
                        st.fp8_rows += 1;
                    }
                    Split::Cols { c0, c1 } => {
                        let d = dv.direct.slice(data);
                        backend.preload_fp8_col_dequant(ph, d, &scale_full, rows, cols, c0, c1)?;
                        st.fp8_cols += 1;
                    }
                    Split::QkvHeads { .. } => {
                        return Err(FerriteError::Config(format!(
                            "direct: {name} is fp8 fused-qkv (GDN qkv is bf16 — checkpoint drift)"
                        )));
                    }
                    Split::Expert { es, ee } => {
                        let e: usize = name
                            .split(".experts.")
                            .nth(1)
                            .and_then(|r| r.split('.').next())
                            .and_then(|s| s.parse().ok())
                            .unwrap_or(usize::MAX);
                        if e >= es && e < ee {
                            let d = dv.direct.slice(data);
                            backend.preload_fp8_dequant(ph, d, &scale_full, rows, cols)?;
                            st.experts += 1;
                        }
                    }
                }
            }
            WeightView::Bf16ToF32 { seg, shape } => {
                // f32 residents (embed, 1-D norms/biases): replicated (the
                // 1-D head-split weights — dt_bias/a_log — are ALSO bf16→f32
                // views with a Rows window on the 1-D segment)
                let full = dv.direct.slice(seg);
                // mmap sanity: first 8 bytes of 1-D norm weights (all-zero =
                // wrong offset → rmsnorm produces zeros → constant "!" output)
                if std::env::var_os("FERRITE_MMAP_DEBUG").is_some() && name.contains("norm") {
                    let b = &full[..8.min(full.len())];
                    eprintln!("[mmap-dbg] NORM {name}: first 8 bytes {:02x?} (len={})", b, full.len());
                }
                match split {
                    Split::Replicated => {
                        backend.preload_bf16_to_f32_raw(ph, full)?;
                        st.f32_expand += 1;
                    }
                    Split::Rows { r0, r1 } => {
                        backend.preload_bf16_to_f32_raw(ph, &full[r0 * 2..r1 * 2])?;
                        st.f32_expand += 1;
                    }
                    _ => {
                        return Err(FerriteError::Config(format!(
                            "direct: {name} f32-expand with non-row split (checkpoint drift)"
                        )));
                    }
                }
                let _ = shape;
            }
            WeightView::F32Seg { seg, shape } => {
                // f32 checkpoint bytes → f32 resident (VERBATIM — no bf16→f32
                // widening; the mmap bytes ARE the device layout at 4
                // bytes/element). The old Bf16ToF32 classification here was
                // the 2x-byte-mismatch bug (expected numel*2 bf16 bytes,
                // got numel*4 f32 bytes → panic).
                let full = dv.direct.slice(seg);
                match split {
                    Split::Replicated => {
                        backend.preload_f32_raw(ph, full)?;
                        st.f32_expand += 1;
                    }
                    Split::Rows { r0, r1 } => {
                        let rb = shape.get(1).copied().unwrap_or(1) * 4; // f32 row bytes
                        backend.preload_f32_raw(ph, &full[r0 * rb..r1 * rb])?;
                        st.f32_expand += 1;
                    }
                    _ => {
                        return Err(FerriteError::Config(format!(
                            "direct: {name} F32Seg with non-row split (checkpoint drift)"
                        )));
                    }
                }
            }
        }
    }
    Ok(st)
}
