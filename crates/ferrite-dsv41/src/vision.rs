//! DeepSeek-V4.1-Flash vision tower — the CPU golden standard.
//!
//! Mirrors the reference multimodal path 1:1:
//!   * `inference/vision.py` — the 32-block ViT (2D RoPE, full bidirectional
//!     attention, RMSNorm, SwiGLU MLP), the 3x3 Aligner and the learned span
//!     embeddings;
//!   * `inference/model.py` — `encode_image` / `merge_image_embeddings`;
//!   * `inference/image_processor.py` — the resize plan, patch splitting,
//!     token-type sequences and the placeholder expansion of `prepare_vl_inputs`.
//!
//! Released shapes (`vision_config` of the bundled config):
//!   * ViT: hidden 1024, 16 heads (head_dim 64), 32 blocks, MLP inter 2816,
//!     patch 14 (`3*14*14 = 588`-wide flattened patches), rope_theta 10000.
//!     The 2D RoPE table has `head_dim/2 = 32` lanes per token: the first half
//!     encodes the row position, the second half the column position.
//!   * Aligner: `w1 [5120, 9216]` — `9216 = downsample_ratio^2 * vision_dim`
//!     (`3 * 1024 * 3`), i.e. one aligned row is the flatten of a 3x3 block of
//!     ViT tokens, then `gelu`, then `w2 [5120, 5120]`. The block flatten order
//!     is the `F.unfold` one, **channel-major**: `c*r*r + i*r + j`, with zero
//!     padding on the right/bottom edges when the patch grid is not divisible
//!     by the ratio.
//!   * Image span: `[IMAGE_START] + ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) *
//!     n_llm_h + [IMAGE_END]`, where `n_llm_* = ceil(n_vit_* / downsample_ratio)`.
//!
//! # Numerical policy (documented deviation)
//!
//! The reference runs the whole tower in bf16 (the tower's native dtype), with
//! f32 arithmetic only inside RMSNorm / RoPE / softmax. This golden keeps every
//! value in f32: checkpoint weights are assumed preconverted bf16->f32
//! (lossless), activations are not re-rounded to bf16 between ops. The CUDA
//! kernels (`kernels/cuda/dsv41_vision.cu`) run native bf16 MMA with f32
//! accumulation; align them with a bf16-sized tolerance (per-op relative error
//! up to ~2^-8, expected end-to-end Aligner divergence on the order of 1e-2
//! relative). [`bf16_round`] exposes round-to-nearest-even f32->bf16->f32 so
//! tests can emulate the storage type; [`prepare_vl_inputs`] applies it to the
//! normalized pixels exactly like the reference's `.to(torch.bfloat16)`.
//!
//! The pixel *plan* (patch/resize arithmetic, token types, placeholder
//! expansion) reproduces `image_processor.py` exactly, including its f64
//! operation order. The resample itself deviates: the reference uses PIL
//! (`ImageOps.pad`, **bicubic**, 127-gray margin), this module implements
//! bilinear + centered pad, so individual pixel values (and possibly a ±1 px
//! size on awkward ratios) may differ. In production, feed the buffer the
//! reference's PIL pass would produce; the ViT consumes pixels, not paths.

use crate::config::Dsv41Config;
use ferrite_types::{FerriteError, Result};

// ===========================================================================
// Token types (image_processor.py: TEXT, IMAGE_START, IMAGE, IMAGE_NEW_LINE,
// IMAGE_END = -1, 0, 1, 2, 3)
// ===========================================================================

/// Prompt position outside any image span.
pub const TEXT: i64 = -1;
/// First token of an image span — takes the `image_start` embedding.
pub const IMAGE_START: i64 = 0;
/// A regular image row — filled from the Aligner output in reading order.
pub const IMAGE: i64 = 1;
/// End-of-row token inside a span — takes the `image_newline` embedding.
pub const IMAGE_NEW_LINE: i64 = 2;
/// Last token of an image span — takes the `image_end` embedding.
pub const IMAGE_END: i64 = 3;

/// RMSNorm epsilon of the reference `vision.py::RMSNorm` (the model-wide
/// `norm_eps` of 1e-20 is **not** used inside the vision tower).
pub const VISION_NORM_EPS: f32 = 1e-6;

/// Round an f32 to the nearest bf16 value (round-to-nearest-even on the
/// truncated f32 significand) and return it as f32. Equivalent to
/// `tensor.to(torch.bfloat16).to(torch.float32)`.
pub fn bf16_round(x: f32) -> f32 {
    if !x.is_finite() {
        return x;
    }
    let bits = x.to_bits();
    let lsb = (bits >> 16) & 1;
    f32::from_bits(bits.wrapping_add(0x7fff + lsb) & 0xffff_0000)
}

// ===========================================================================
// Geometry
// ===========================================================================

/// ViT geometry (`vision.py::ViT` + the reference args).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VisionConfig {
    /// Hidden width (released: 1024).
    pub dim: usize,
    /// Attention heads (released: 16).
    pub n_heads: usize,
    /// SwiGLU inner width (released: 2816; `w1` outputs `2 * inter_dim`).
    pub inter_dim: usize,
    /// Transformer blocks (released: 32).
    pub n_layers: usize,
    /// Patch edge in pixels (released: 14).
    pub patch_size: usize,
    /// 2D RoPE base (released: 10000).
    pub rope_theta: f32,
}

impl VisionConfig {
    pub fn from_dsv41(c: &Dsv41Config) -> Self {
        Self {
            dim: c.vision_dim,
            n_heads: c.vision_n_heads,
            inter_dim: c.vision_inter_dim,
            n_layers: c.vision_n_layers,
            patch_size: c.vision_patch_size,
            rope_theta: c.vision_rope_theta,
        }
    }

    /// Per-head width (released: 64).
    pub fn head_dim(&self) -> usize {
        self.dim / self.n_heads
    }

    /// Lanes of the 2D RoPE table per token — the reference's `rope_dim`,
    /// half the head width (released: 32).
    pub fn rope_dim(&self) -> usize {
        self.head_dim() / 2
    }

    /// Flattened patch width fed to `patch_embed` (`3 * p^2`, released: 588).
    pub fn patch_vec(&self) -> usize {
        3 * self.patch_size * self.patch_size
    }
}

/// Aligner geometry (`vision.py::Aligner`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AlignerConfig {
    /// Spatial pooling ratio (released: 3).
    pub downsample_ratio: usize,
    /// ViT token width (released: 1024).
    pub vit_dim: usize,
    /// Output width — the LLM hidden size (released: 5120).
    pub out_dim: usize,
}

impl AlignerConfig {
    pub fn from_dsv41(c: &Dsv41Config) -> Self {
        Self {
            downsample_ratio: c.vision_downsample_ratio,
            vit_dim: c.vision_dim,
            out_dim: c.dim,
        }
    }

    /// Width of one pooled row: `downsample_ratio^2 * vit_dim` (released: 9216).
    pub fn in_dim(&self) -> usize {
        self.vit_dim * self.downsample_ratio * self.downsample_ratio
    }
}

// ===========================================================================
// Weights (row-major [out, in], named like the checkpoint — see weights.rs)
// ===========================================================================

/// One ViT block's weights. Checkpoint names: `vision.blocks.{b}.{norm1,attn.wqkv,
/// attn.wo,norm2,mlp.w1,mlp.w2}`; the MLP linears have **no bias** (reference
/// `nn.Linear(..., bias=False)`), attention and patch_embed do.
#[derive(Debug, Clone)]
pub struct VitBlockWeights {
    /// RMSNorm gain `[dim]`.
    pub norm1: Vec<f32>,
    /// QKV projection `[3*dim, dim]`.
    pub wqkv: Vec<f32>,
    /// QKV bias `[3*dim]`.
    pub bqkv: Vec<f32>,
    /// Output projection `[dim, dim]`.
    pub wo: Vec<f32>,
    /// Output bias `[dim]`.
    pub bo: Vec<f32>,
    /// RMSNorm gain `[dim]`.
    pub norm2: Vec<f32>,
    /// SwiGLU gate+up `[2*inter_dim, dim]` (gate rows first).
    pub w1: Vec<f32>,
    /// SwiGLU down `[dim, inter_dim]`.
    pub w2: Vec<f32>,
}

/// The ViT encoder's weights (`vision.py::ViT`).
#[derive(Debug, Clone)]
pub struct VitWeights {
    /// Patch projection `[dim, 3*p^2]`.
    pub patch_embed: Vec<f32>,
    /// Patch projection bias `[dim]`.
    pub patch_embed_bias: Vec<f32>,
    pub blocks: Vec<VitBlockWeights>,
    /// Final RMSNorm gain `[dim]`.
    pub norm: Vec<f32>,
}

impl VitWeights {
    /// Zero weights at the released shapes, RMSNorm gains at one (the
    /// reference `nn.RMSNorm`/`nn.Linear` init), attention/MLP projections
    /// zeroed — convenient for tests and loader scaffolding.
    pub fn zeros(cfg: &VisionConfig) -> Self {
        let d = cfg.dim;
        let blocks = (0..cfg.n_layers)
            .map(|_| VitBlockWeights {
                norm1: vec![1.0; d],
                wqkv: vec![0.0; 3 * d * d],
                bqkv: vec![0.0; 3 * d],
                wo: vec![0.0; d * d],
                bo: vec![0.0; d],
                norm2: vec![1.0; d],
                w1: vec![0.0; 2 * cfg.inter_dim * d],
                w2: vec![0.0; d * cfg.inter_dim],
            })
            .collect();
        Self {
            patch_embed: vec![0.0; d * cfg.patch_vec()],
            patch_embed_bias: vec![0.0; d],
            blocks,
            norm: vec![1.0; d],
        }
    }
}

/// The Aligner's weights (`aligner.w1/w2`, both with bias).
#[derive(Debug, Clone)]
pub struct AlignerWeights {
    /// `w1 [out_dim, r*r*vit_dim]` (released: `[5120, 9216]`).
    pub w1: Vec<f32>,
    /// `w1` bias `[out_dim]`.
    pub b1: Vec<f32>,
    /// `w2 [out_dim, out_dim]`.
    pub w2: Vec<f32>,
    /// `w2` bias `[out_dim]`.
    pub b2: Vec<f32>,
}

impl AlignerWeights {
    /// Zero weights at the released shapes (`w1` zeroed, biases zeroed).
    pub fn zeros(cfg: &AlignerConfig) -> Self {
        Self {
            w1: vec![0.0; cfg.out_dim * cfg.in_dim()],
            b1: vec![0.0; cfg.out_dim],
            w2: vec![0.0; cfg.out_dim * cfg.out_dim],
            b2: vec![0.0; cfg.out_dim],
        }
    }
}

// ===========================================================================
// Small numerical helpers (f32, mirroring the reference's op order)
// ===========================================================================

/// `y[r, :] = x[r, :] @ w^T + b`, `x [rows, k]`, `w [n, k]` row-major.
fn linear(x: &[f32], rows: usize, k: usize, w: &[f32], bias: Option<&[f32]>, n: usize) -> Vec<f32> {
    assert_eq!(x.len(), rows * k, "linear: x shape");
    assert_eq!(w.len(), n * k, "linear: w shape");
    let mut y = vec![0f32; rows * n];
    for r in 0..rows {
        let xr = &x[r * k..(r + 1) * k];
        for o in 0..n {
            let wk = &w[o * k..(o + 1) * k];
            let mut acc = 0f32;
            for i in 0..k {
                acc += xr[i] * wk[i];
            }
            y[r * n + o] = acc + bias.map_or(0.0, |b| b[o]);
        }
    }
    y
}

/// `vision.py::RMSNorm.forward`: `weight * (x * rsqrt(mean(x^2) + eps))`.
fn rms_norm(x: &[f32], rows: usize, dim: usize, weight: &[f32], eps: f32) -> Vec<f32> {
    assert_eq!(x.len(), rows * dim);
    let mut y = vec![0f32; rows * dim];
    for r in 0..rows {
        let row = &x[r * dim..(r + 1) * dim];
        let ms = row.iter().map(|v| v * v).sum::<f32>() / dim as f32;
        let inv = 1.0 / (ms + eps).sqrt();
        for c in 0..dim {
            y[r * dim + c] = weight[c] * (row[c] * inv);
        }
    }
    y
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// `erf` via Abramowitz-Stegun 7.1.26 (|abs error| <= 1.5e-7, f64 internally).
fn erf(x: f32) -> f32 {
    let x = x as f64;
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    (if x < 0.0 { -y } else { y }) as f32
}

/// Exact (erf-based) GELU — `F.gelu`'s default, used by the Aligner.
fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + erf(x * std::f32::consts::FRAC_1_SQRT_2))
}

fn softmax_in_place(x: &mut [f32]) {
    let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut s = 0f32;
    for v in x.iter_mut() {
        *v = (*v - m).exp();
        s += *v;
    }
    for v in x.iter_mut() {
        *v /= s;
    }
}

/// `vision.py::get_vision_cos_sin`: `[n_h * n_w, rope_dim]` cos/sin tables.
/// Row-major token order matches the patch grid; lanes `[0, rope_dim/2)` carry
/// the row position, lanes `[rope_dim/2, rope_dim)` the column position, with
/// `inv_freq[k] = theta^(-2k/rope_dim)`.
fn vision_rope_table(n_h: usize, n_w: usize, rope_dim: usize, theta: f32) -> (Vec<f32>, Vec<f32>) {
    let half = rope_dim / 2;
    let inv: Vec<f32> = (0..half)
        .map(|k| 1.0 / theta.powf(2.0 * k as f32 / rope_dim as f32))
        .collect();
    let n = n_h * n_w;
    let mut cos = vec![0f32; n * rope_dim];
    let mut sin = vec![0f32; n * rope_dim];
    for h in 0..n_h {
        for w in 0..n_w {
            let row = h * n_w + w;
            for k in 0..half {
                let a = h as f32 * inv[k];
                cos[row * rope_dim + k] = a.cos();
                sin[row * rope_dim + k] = a.sin();
                let b = w as f32 * inv[k];
                cos[row * rope_dim + half + k] = b.cos();
                sin[row * rope_dim + half + k] = b.sin();
            }
        }
    }
    (cos, sin)
}

/// `vision.py::apply_rotary` on one `[n, n_heads, head_dim]` tensor: the head
/// is split in half, lane `k` of the first half rotates against lane `k` of
/// the second half with table lane `k` of the token.
fn apply_rotary_2d(
    x: &mut [f32],
    n: usize,
    n_heads: usize,
    head_dim: usize,
    cos: &[f32],
    sin: &[f32],
) {
    let half = head_dim / 2;
    assert_eq!(cos.len(), n * half, "rope table shape");
    assert_eq!(sin.len(), n * half, "rope table shape");
    for t in 0..n {
        for h in 0..n_heads {
            let base = (t * n_heads + h) * head_dim;
            for k in 0..half {
                let x1 = x[base + k];
                let x2 = x[base + half + k];
                let c = cos[t * half + k];
                let s = sin[t * half + k];
                x[base + k] = x1 * c - x2 * s;
                x[base + half + k] = x2 * c + x1 * s;
            }
        }
    }
}

// ===========================================================================
// ViT
// ===========================================================================

/// `vision.py::Attention.forward` — full bidirectional attention inside one
/// image, scaled by `1/sqrt(head_dim)`, plus the output projection.
fn block_attention(
    x: &[f32],
    n: usize,
    cfg: &VisionConfig,
    cos: &[f32],
    sin: &[f32],
    w: &VitBlockWeights,
) -> Vec<f32> {
    let (dim, nh, hd) = (cfg.dim, cfg.n_heads, cfg.head_dim());
    let qkv = linear(x, n, dim, &w.wqkv, Some(&w.bqkv), 3 * dim);
    let mut q = vec![0f32; n * dim];
    let mut k = vec![0f32; n * dim];
    let mut v = vec![0f32; n * dim];
    for t in 0..n {
        let row = &qkv[t * 3 * dim..(t + 1) * 3 * dim];
        q[t * dim..(t + 1) * dim].copy_from_slice(&row[0..dim]);
        k[t * dim..(t + 1) * dim].copy_from_slice(&row[dim..2 * dim]);
        v[t * dim..(t + 1) * dim].copy_from_slice(&row[2 * dim..3 * dim]);
    }
    apply_rotary_2d(&mut q, n, nh, hd, cos, sin);
    apply_rotary_2d(&mut k, n, nh, hd, cos, sin);

    let scale = 1.0 / (hd as f32).sqrt();
    let mut out = vec![0f32; n * dim];
    let mut scores = vec![0f32; n];
    for h in 0..nh {
        for i in 0..n {
            for j in 0..n {
                let mut dot = 0f32;
                for d in 0..hd {
                    dot += q[(i * nh + h) * hd + d] * k[(j * nh + h) * hd + d];
                }
                scores[j] = dot * scale;
            }
            softmax_in_place(&mut scores);
            for d in 0..hd {
                let mut acc = 0f32;
                for j in 0..n {
                    acc += scores[j] * v[(j * nh + h) * hd + d];
                }
                out[(i * nh + h) * hd + d] = acc;
            }
        }
    }
    linear(&out, n, dim, &w.wo, Some(&w.bo), dim)
}

/// `vision.py::MLP.forward` — `w2(silu(gate) * up)`, no biases.
fn block_mlp(x: &[f32], n: usize, cfg: &VisionConfig, w: &VitBlockWeights) -> Vec<f32> {
    let (dim, inter) = (cfg.dim, cfg.inter_dim);
    let gu = linear(x, n, dim, &w.w1, None, 2 * inter);
    let mut act = vec![0f32; n * inter];
    for t in 0..n {
        for j in 0..inter {
            let g = gu[t * 2 * inter + j];
            let u = gu[t * 2 * inter + inter + j];
            act[t * inter + j] = silu(g) * u;
        }
    }
    linear(&act, n, inter, &w.w2, None, dim)
}

/// `vision.py::ViT.forward` — `patches [n_vit_h * n_vit_w, 3*p^2]` (in the
/// image processor's patch order) to `[n, dim]` after the final RMSNorm.
pub fn vit_forward(
    w: &VitWeights,
    cfg: &VisionConfig,
    patches: &[f32],
    n_vit_h: usize,
    n_vit_w: usize,
) -> Vec<f32> {
    let n = n_vit_h * n_vit_w;
    let dim = cfg.dim;
    assert_eq!(patches.len(), n * cfg.patch_vec(), "vit: patch shape");
    assert_eq!(w.blocks.len(), cfg.n_layers, "vit: block count");
    let mut x = linear(
        patches,
        n,
        cfg.patch_vec(),
        &w.patch_embed,
        Some(&w.patch_embed_bias),
        dim,
    );
    let (cos, sin) = vision_rope_table(n_vit_h, n_vit_w, cfg.rope_dim(), cfg.rope_theta);
    for blk in &w.blocks {
        let xn = rms_norm(&x, n, dim, &blk.norm1, VISION_NORM_EPS);
        let a = block_attention(&xn, n, cfg, &cos, &sin, blk);
        for i in 0..x.len() {
            x[i] += a[i];
        }
        let xn2 = rms_norm(&x, n, dim, &blk.norm2, VISION_NORM_EPS);
        let m = block_mlp(&xn2, n, cfg, blk);
        for i in 0..x.len() {
            x[i] += m[i];
        }
    }
    rms_norm(&x, n, dim, &w.norm, VISION_NORM_EPS)
}

// ===========================================================================
// Aligner
// ===========================================================================

/// The Aligner's `view/permute/pad/unfold` chain: a `[n_h, n_w]` grid of
/// `vit_dim`-vectors becomes `[n_llm_h * n_llm_w, r*r*vit_dim]`; each row is
/// the flatten of one 3x3 block in the `F.unfold` order — channel-major,
/// `c*r*r + i*r + j` — with zero padding at the right/bottom edges.
///
/// Returns `(n_llm_h, n_llm_w, packed)`.
fn downsample_pack(
    x: &[f32],
    n_h: usize,
    n_w: usize,
    r: usize,
    vit_dim: usize,
) -> (usize, usize, Vec<f32>) {
    assert_eq!(x.len(), n_h * n_w * vit_dim, "aligner: patch grid shape");
    let pad_h = (r - n_h % r) % r;
    let pad_w = (r - n_w % r) % r;
    let (hp, wp) = (n_h + pad_h, n_w + pad_w);
    let (n_llm_h, n_llm_w) = (hp / r, wp / r);
    let row_len = r * r * vit_dim;
    let mut out = vec![0f32; n_llm_h * n_llm_w * row_len];
    for oh in 0..n_llm_h {
        for ow in 0..n_llm_w {
            let l = oh * n_llm_w + ow;
            for c in 0..vit_dim {
                for i in 0..r {
                    for j in 0..r {
                        let rr = oh * r + i;
                        let cc = ow * r + j;
                        let v = if rr < n_h && cc < n_w {
                            x[(rr * n_w + cc) * vit_dim + c]
                        } else {
                            0.0
                        };
                        out[l * row_len + c * r * r + i * r + j] = v;
                    }
                }
            }
        }
    }
    (n_llm_h, n_llm_w, out)
}

/// `vision.py::Aligner.forward` — `x [n_h * n_w, vit_dim]` to
/// `[n_llm_h * n_llm_w, out_dim]` (`w2(gelu(w1(pack(x))))`).
pub fn aligner_forward(
    w: &AlignerWeights,
    cfg: &AlignerConfig,
    x: &[f32],
    n_h: usize,
    n_w: usize,
) -> Vec<f32> {
    let r = cfg.downsample_ratio;
    let (n_llm_h, n_llm_w, packed) = downsample_pack(x, n_h, n_w, r, cfg.vit_dim);
    let rows = n_llm_h * n_llm_w;
    let hidden = linear(&packed, rows, cfg.in_dim(), &w.w1, Some(&w.b1), cfg.out_dim);
    let act: Vec<f32> = hidden.iter().map(|&v| gelu(v)).collect();
    linear(&act, rows, cfg.out_dim, &w.w2, Some(&w.b2), cfg.out_dim)
}

// ===========================================================================
// Tower: encode + merge
// ===========================================================================

/// The complete vision tower: ViT encoder, Aligner and the three learned span
/// embeddings (`image_start` / `image_end` / `image_newline`, `[dim]` each).
#[derive(Debug, Clone)]
pub struct VisionTower {
    pub cfg: VisionConfig,
    pub aligner_cfg: AlignerConfig,
    pub vit: VitWeights,
    pub aligner: AlignerWeights,
    /// Learned embedding of the span's first token.
    pub image_start: Vec<f32>,
    /// Learned embedding of the span's last token.
    pub image_end: Vec<f32>,
    /// Learned embedding of each span row's last token.
    pub image_newline: Vec<f32>,
}

impl VisionTower {
    /// Zero-initialised weights at the released shapes (loader scaffolding and
    /// tests; the real tower is filled from the checkpoint).
    pub fn zeros(cfg: VisionConfig, aligner_cfg: AlignerConfig) -> Self {
        Self {
            vit: VitWeights::zeros(&cfg),
            aligner: AlignerWeights::zeros(&aligner_cfg),
            image_start: vec![0.0; aligner_cfg.out_dim],
            image_end: vec![0.0; aligner_cfg.out_dim],
            image_newline: vec![0.0; aligner_cfg.out_dim],
            cfg,
            aligner_cfg,
        }
    }

    /// `model.py::Transformer.encode_image` — ViT then Aligner; output
    /// `[n_llm_h * n_llm_w, out_dim]` in the span's reading order.
    pub fn encode_image(&self, patches: &[f32], n_vit_h: usize, n_vit_w: usize) -> Vec<f32> {
        let vit_out = vit_forward(&self.vit, &self.cfg, patches, n_vit_h, n_vit_w);
        aligner_forward(&self.aligner, &self.aligner_cfg, &vit_out, n_vit_h, n_vit_w)
    }

    /// `model.py::Transformer.merge_image_embeddings` for one sample:
    /// overwrite the `[start, start + types.len())` span of `h` (`[seq,
    /// out_dim]`, row-major) — `IMAGE` rows take the Aligner output in reading
    /// order, the delimiters take the learned embeddings. `TEXT` rows are left
    /// untouched.
    pub fn merge_image_embeddings(
        &self,
        h: &mut [f32],
        start: usize,
        types: &[i64],
        embeds: &[f32],
    ) {
        merge_image_embeddings(
            h,
            self.aligner_cfg.out_dim,
            start,
            types,
            embeds,
            &self.image_start,
            &self.image_end,
            &self.image_newline,
        );
    }
}

/// Free-function form of [`VisionTower::merge_image_embeddings`].
pub fn merge_image_embeddings(
    h: &mut [f32],
    dim: usize,
    start: usize,
    types: &[i64],
    embeds: &[f32],
    image_start: &[f32],
    image_end: &[f32],
    image_newline: &[f32],
) {
    let n_image = types.iter().filter(|&&t| t == IMAGE).count();
    assert!(embeds.len() >= n_image * dim, "merge: not enough aligner rows");
    assert!(start + types.len() <= h.len() / dim, "merge: span out of range");
    assert_eq!(image_start.len(), dim);
    assert_eq!(image_end.len(), dim);
    assert_eq!(image_newline.len(), dim);
    let mut cursor = 0usize;
    for (k, &t) in types.iter().enumerate() {
        let dst = &mut h[(start + k) * dim..(start + k + 1) * dim];
        match t {
            IMAGE_START => dst.copy_from_slice(image_start),
            IMAGE_END => dst.copy_from_slice(image_end),
            IMAGE_NEW_LINE => dst.copy_from_slice(image_newline),
            IMAGE => {
                dst.copy_from_slice(&embeds[cursor * dim..(cursor + 1) * dim]);
                cursor += 1;
            }
            // TEXT (and anything else) stays as-is.
            _ => {}
        }
    }
}

// ===========================================================================
// Image processing (image_processor.py)
// ===========================================================================

/// Preprocessing parameters — the `vision_*` half of the reference args.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImageProcConfig {
    /// Patch edge in pixels (released: 14).
    pub patch_size: usize,
    /// Aligner pooling ratio (released: 3).
    pub downsample_ratio: usize,
    /// Token cap per image (released: 1024).
    pub max_n_token: usize,
    /// Upscale floor in pixels (released: 544*544 = 295936).
    pub min_pixels: usize,
    /// Aspect clamp (`None` in the released config: `vision_max_wh_ratio: null`).
    /// Not parsed by [`Dsv41Config`]; this struct keeps the slot for parity.
    pub max_wh_ratio: Option<f64>,
}

impl ImageProcConfig {
    pub fn from_dsv41(c: &Dsv41Config) -> Self {
        Self {
            patch_size: c.vision_patch_size,
            downsample_ratio: c.vision_downsample_ratio,
            max_n_token: c.vision_max_n_token,
            min_pixels: c.vision_min_pixels,
            max_wh_ratio: None,
        }
    }
}

/// `image_processor.py::llm_grid` — the LLM token grid of a patch grid.
pub fn llm_grid(n_vit_h: usize, n_vit_w: usize, downsample_ratio: usize) -> (usize, usize) {
    (
        n_vit_h.div_ceil(downsample_ratio),
        n_vit_w.div_ceil(downsample_ratio),
    )
}

/// `image_processor.py::num_image_tokens`.
pub fn num_image_tokens(n_llm_h: usize, n_llm_w: usize) -> usize {
    n_llm_h * (n_llm_w + 1) + 2
}

/// `image_processor.py::solve_resize_ratio` — the largest aspect-preserving
/// pixel size whose token grid still fits `max_n_token`. Returns
/// `(height, width)` in pixels (patch multiples). f64 arithmetic in the
/// reference's operation order.
pub fn solve_resize_ratio(
    height: f64,
    width: f64,
    patch: usize,
    downsample_ratio: usize,
    max_n_token: usize,
) -> (usize, usize) {
    let ratio = height / width;
    let max_w_float = ((max_n_token as f64 - 2.0) / ratio + 0.25).sqrt() - 0.5;
    let max_h_float = max_w_float * ratio;
    let cell = (patch * downsample_ratio) as f64;
    if max_w_float < 1.0 {
        // very tall: collapse to a single column
        return (((max_n_token - 2) / 2) * patch * downsample_ratio, patch * downsample_ratio);
    }
    if max_h_float < 1.0 {
        // very wide: collapse to a single row
        return (patch * downsample_ratio, (max_n_token - 3) * patch * downsample_ratio);
    }
    let beta = (max_w_float.floor() * cell / width).min(max_h_float.floor() * cell / height);
    let h = (height * beta / patch as f64).floor() as usize * patch;
    let w = (width * beta / patch as f64).floor() as usize * patch;
    (h, w)
}

/// `image_processor.py::safe_resize` — shrink only when the patch-grid plan
/// exceeds the token cap. Returns `(n_llm_h, n_llm_w, best_height, best_width)`.
pub fn safe_resize(
    height: f64,
    width: f64,
    best_height: usize,
    best_width: usize,
    c: &ImageProcConfig,
) -> (usize, usize, usize, usize) {
    let p = c.patch_size;
    let r = c.downsample_ratio;
    let (mut n_llm_h, mut n_llm_w) = llm_grid(best_height / p, best_width / p, r);
    let (mut bh, mut bw) = (best_height, best_width);
    if num_image_tokens(n_llm_h, n_llm_w) > c.max_n_token {
        let (h2, w2) = solve_resize_ratio(height, width, p, r, c.max_n_token);
        bh = h2;
        bw = w2;
        let g = llm_grid(bh / p, bw / p, r);
        n_llm_h = g.0;
        n_llm_w = g.1;
        assert!(
            num_image_tokens(n_llm_h, n_llm_w) <= c.max_n_token,
            "safe_resize: solve_resize_ratio broke the token cap"
        );
    }
    (n_llm_h, n_llm_w, bh, bw)
}

/// `image_processor.py::plan_image_grid` — the resize plan for an image of the
/// given original size: `(n_llm_h, n_llm_w, best_height, best_width)`.
pub fn plan_image_grid(width: f64, height: f64, c: &ImageProcConfig) -> (usize, usize, usize, usize) {
    let p = c.patch_size;
    let (mut w, mut h) = (width, height);
    if let Some(mx) = c.max_wh_ratio {
        if w > h * mx {
            w = h * mx;
        }
    }
    if w * h > 0.0 && w * h < c.min_pixels as f64 {
        let ratio = (c.min_pixels as f64 / (w * h)).sqrt();
        w = (w * ratio).floor();
        h = (h * ratio).floor();
    }
    let best_w = (w / p as f64).ceil() as usize * p;
    let best_h = (h / p as f64).ceil() as usize * p;
    safe_resize(h, w, best_h, best_w, c)
}

/// A row-major RGB8 image — the pixel contract fed to the tower.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RgbImage {
    pub width: usize,
    pub height: usize,
    /// `3 * width * height` bytes, `(y * width + x) * 3 + c`.
    pub data: Vec<u8>,
}

impl RgbImage {
    pub fn new(width: usize, height: usize, data: Vec<u8>) -> Result<Self> {
        if data.len() != width * height * 3 {
            return Err(FerriteError::InvalidArg(format!(
                "rgb image: {} bytes for {width}x{height} rgb",
                data.len()
            )));
        }
        Ok(Self {
            width,
            height,
            data,
        })
    }
}

/// Bilinear RGB8 resample (row-major in/out), pixel-center mapping.
fn resize_rgb_bilinear(src: &[u8], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<u8> {
    let mut out = vec![0u8; dw * dh * 3];
    if dw == 0 || dh == 0 {
        return out;
    }
    for dy in 0..dh {
        let sy = (((dy as f64 + 0.5) * sh as f64 / dh as f64) - 0.5).max(0.0);
        let y0 = sy.floor() as usize;
        let y1 = (y0 + 1).min(sh - 1);
        let fy = (sy - y0 as f64) as f32;
        for dx in 0..dw {
            let sx = (((dx as f64 + 0.5) * sw as f64 / dw as f64) - 0.5).max(0.0);
            let x0 = sx.floor() as usize;
            let x1 = (x0 + 1).min(sw - 1);
            let fx = (sx - x0 as f64) as f32;
            for c in 0..3 {
                let p00 = src[(y0 * sw + x0) * 3 + c] as f32;
                let p01 = src[(y0 * sw + x1) * 3 + c] as f32;
                let p10 = src[(y1 * sw + x0) * 3 + c] as f32;
                let p11 = src[(y1 * sw + x1) * 3 + c] as f32;
                let v = p00 * (1.0 - fx) * (1.0 - fy)
                    + p01 * fx * (1.0 - fy)
                    + p10 * (1.0 - fx) * fy
                    + p11 * fx * fy;
                out[(dy * dw + dx) * 3 + c] = v.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}

/// Contain-resize to fit `best_w x best_h`, then center-pad with the
/// reference's 127-gray (`load_image`'s `ImageOps.pad` branch).
///
/// Deviation: the reference resamples with PIL's **bicubic**; this uses
/// bilinear. Size/layout follow the same contract, values carry resampling
/// error.
pub fn resize_pad_rgb(rgb: &[u8], w: usize, h: usize, best_w: usize, best_h: usize) -> Vec<u8> {
    assert_eq!(rgb.len(), w * h * 3, "resize: rgb shape");
    let scale = (best_w as f64 / w as f64).min(best_h as f64 / h as f64);
    let new_w = ((w as f64 * scale).round() as usize).clamp(1, best_w);
    let new_h = ((h as f64 * scale).round() as usize).clamp(1, best_h);
    let scaled = resize_rgb_bilinear(rgb, w, h, new_w, new_h);
    let mut out = vec![127u8; best_w * best_h * 3];
    let ox = (best_w - new_w) / 2;
    let oy = (best_h - new_h) / 2;
    for y in 0..new_h {
        let dst = ((oy + y) * best_w + ox) * 3;
        let src = y * new_w * 3;
        out[dst..dst + new_w * 3].copy_from_slice(&scaled[src..src + new_w * 3]);
    }
    out
}

/// `load_image`'s normalize + patchify: `v/255 -> (v - 0.5)/0.5`, then the
/// reference's reshape chain `[3, H, W] -> [3, nh, p, nw, p] -> [nh, nw, 3, p,
/// p] -> [nh*nw, 3*p*p]` — channel-major inside each patch (`c*p^2 + i*p + j`),
/// patches in row-major grid order. Output `[nh*nw, 3*p^2]` in f32 (the
/// reference then casts to bf16; see [`prepare_vl_inputs`]).
pub fn patchify_rgb(rgb: &[u8], width: usize, height: usize, patch: usize) -> Vec<f32> {
    assert_eq!(rgb.len(), width * height * 3, "patchify: rgb shape");
    assert!(width % patch == 0 && height % patch == 0, "patchify: size");
    let (nh, nw) = (height / patch, width / patch);
    let pp = patch * patch;
    let mut out = vec![0f32; nh * nw * 3 * pp];
    for ph in 0..nh {
        for pw in 0..nw {
            let base = (ph * nw + pw) * 3 * pp;
            for c in 0..3 {
                for i in 0..patch {
                    for j in 0..patch {
                        let y = ph * patch + i;
                        let x = pw * patch + j;
                        let v = rgb[(y * width + x) * 3 + c] as f32 / 255.0;
                        out[base + c * pp + i * patch + j] = (v - 0.5) / 0.5;
                    }
                }
            }
        }
    }
    out
}

/// `image_processor.py::image_token_types` — the default span layout,
/// `[IMAGE_START] + ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h + [IMAGE_END]`.
pub fn image_token_types(n_llm_h: usize, n_llm_w: usize) -> Vec<i64> {
    let mut types = Vec::with_capacity(num_image_tokens(n_llm_h, n_llm_w));
    types.push(IMAGE_START);
    for _ in 0..n_llm_h {
        types.extend(std::iter::repeat(IMAGE).take(n_llm_w));
        types.push(IMAGE_NEW_LINE);
    }
    types.push(IMAGE_END);
    types
}

/// One image expanded into its span — the reference's `ImageInput`.
#[derive(Debug, Clone)]
pub struct PreparedImage {
    /// First position of the span in the token sequence.
    pub start: usize,
    /// ViT patches `[n_vit_h * n_vit_w, 3*p^2]`, bf16-rounded like the reference.
    pub patches: Vec<f32>,
    /// Patch-grid height (pixels / patch_size).
    pub n_vit_h: usize,
    /// Patch-grid width (pixels / patch_size).
    pub n_vit_w: usize,
    /// One token type per span position (`images[i].types` in the reference).
    pub types: Vec<i64>,
}

/// `image_processor.py::prepare_vl_inputs`, minus the tokenizer: `prompt_tokens`
/// arrives already encoded, every `image_token_id` entry is expanded into its
/// image span. Returns `(tokens, token_types, image_inputs)`; `image_inputs`
/// is empty when the prompt has no images.
pub fn prepare_vl_inputs(
    prompt_tokens: &[u32],
    image_token_id: u32,
    images: &[RgbImage],
    c: &ImageProcConfig,
) -> Result<(Vec<u32>, Vec<i64>, Vec<PreparedImage>)> {
    let n_placeholders = prompt_tokens.iter().filter(|&&t| t == image_token_id).count();
    if n_placeholders != images.len() {
        return Err(FerriteError::InvalidArg(format!(
            "found {n_placeholders} image tokens but got {} images",
            images.len()
        )));
    }
    let mut tokens = Vec::new();
    let mut token_types = Vec::new();
    let mut prepared = Vec::new();
    let mut image_iter = images.iter();
    for &tok in prompt_tokens {
        if tok != image_token_id {
            tokens.push(tok);
            token_types.push(TEXT);
            continue;
        }
        let img = image_iter.next().expect("placeholder count checked above");
        let (n_llm_h, n_llm_w, best_h, best_w) =
            plan_image_grid(img.width as f64, img.height as f64, c);
        let resized = resize_pad_rgb(&img.data, img.width, img.height, best_w, best_h);
        // the reference casts the normalized pixels to bf16 before the ViT
        let patches: Vec<f32> = patchify_rgb(&resized, best_w, best_h, c.patch_size)
            .into_iter()
            .map(bf16_round)
            .collect();
        let types = image_token_types(n_llm_h, n_llm_w);
        prepared.push(PreparedImage {
            start: tokens.len(),
            patches,
            n_vit_h: best_h / c.patch_size,
            n_vit_w: best_w / c.patch_size,
            types: types.clone(),
        });
        tokens.extend(std::iter::repeat(image_token_id).take(types.len()));
        token_types.extend(types);
    }
    Ok((tokens, token_types, prepared))
}

// __TESTS_HERE__
