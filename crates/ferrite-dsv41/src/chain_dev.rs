//! GPU forward chain for DeepSeek-V4.1-Flash.
//!
//! Config-driven throughout: layer roles (`is_kv_source` / `is_index_source`),
//! per-layer expert counts, compression ratios, the engram layer set and the
//! draft layers all come from [`Dsv41Config`]; nothing here hard-codes the
//! production geometry.
//!
//! # Scope of this file, stated plainly
//! Implemented and exercised end to end on the GPU:
//!   * embedding + hyper-connection expansion,
//!   * the hyper-connection chain (`hc_mixes` -> `hc_collapse` -> block ->
//!     `hc_post`), with the reference's pre-mix threading (block N's attention
//!     mix collapses block N's FFN, the initial mix is the one-hot [1,0,0,0]),
//!   * the MLA window path: latent q/kv projections (fp8 tensor cores), q/kv
//!     RMSNorm, RoPE, the window ring, sparse attention over the selected
//!     positions, inverse RoPE and the block-diagonal grouped output projection,
//!   * MoE: bf16 gate GEMM, `noaux_tc` routing, per-expert MXFP4 tensor-core
//!     GEMMs and the fp8 shared expert.
//!
//! **Not in this file yet** (each is a separate increment, and none of them is
//! silently skipped — `RunOpts` reports which are active):
//!   * the compressor + indexer paths (attention is window-only here),
//!   * the engram write-back,
//!   * DSpark drafting.
//!
//! One token per `step`: the KV ring is per-sequence, so a batched prefill
//! would need per-row ring state. The kernels are all batched-ready; this
//! restriction is in the driver, not the maths.

use std::ffi::c_void;

use ferrite_types::{FerriteError, Result};

use std::sync::Arc;

use crate::config::{Dsv41Config, KvMode};
use crate::tp::Collective;
use crate::device::{DevBuf, Device};
use crate::load::{Dsv41DevWeights, LayerDev};
use crate::ops;

/// Runtime switches for isolating a stage during bring-up.
#[derive(Debug, Clone, Default)]
pub struct RunOpts {
    pub skip_experts: bool,
    pub skip_shared_expert: bool,
}

impl RunOpts {
    pub fn from_env() -> Self {
        let b = |k: &str| std::env::var(k).map(|v| v != "0").unwrap_or(false);
        RunOpts {
            skip_experts: b("DSV41_SKIP_EXPERTS"),
            skip_shared_expert: b("DSV41_SKIP_SHARED_EXPERT"),
        }
    }
}

/// Per-layer state that persists across steps.
struct LayerCache {
    /// window ring, `[window, head_dim]` f32 (the release stores it fp8; the
    /// ring is f32 in this increment — see the file header)
    ring: DevBuf,
    /// the selection for the current token: `[window + index_topk]` i32, -1 unused
    idxs: DevBuf,
    /// compressor carry state, `[ratio, head_dim]`, and the projections' outputs
    state_kv: DevBuf,
    state_score: DevBuf,
    kvp: DevBuf,
    scp: DevBuf,
    latent: DevBuf,
    out_rows: DevBuf,
    /// compressed rows published so far in this sequence
    compress_len: usize,
    /// pre-RoPE keys of those rows, `[max_compress, index_head_dim]`
    index_k: DevBuf,
}

struct Scratch {
    h: DevBuf,     // [hc*dim]
    h2: DevBuf,    // [hc*dim] (hc_post lands here, then copies back)
    x: DevBuf,     // [dim]
    xn: DevBuf,    // [dim]
    xq: DevBuf,    // [dim] fp8 e4m3
    xsc: DevBuf,   // [dim/32 + 8] f32 scales
    pre: DevBuf,   // [hc]
    post: DevBuf,  // [hc]
    comb: DevBuf,  // [hc*hc]
    qr: DevBuf,    // [q_lora]
    q: DevBuf,     // [nh*head_dim]
    kv: DevBuf,    // [head_dim]
    o: DevBuf,     // [nh*head_dim]
    wo: DevBuf,    // [o_lora]
    logits: DevBuf,
    ids: DevBuf,   // [1] i32
    // MoE
    scores: DevBuf,    // [n_experts] f32
    route_idx: DevBuf, // [topk] i32
    route_w: DevBuf,   // [topk] f32
    ex_in: DevBuf,     // [dim]
    ex_act: DevBuf,    // [2*inter]
    ex_out: DevBuf,    // [dim]
    hist: DevBuf,      // [n_experts] i32
    idx_q: DevBuf,     // [index_n_heads * index_head_dim]
    idx_k: DevBuf,     // [index_head_dim]
    idx_w: DevBuf,     // [index_n_heads]
    idx_lens: DevBuf,  // [1] i32
    // bf16 staging for the cuBLAS path
    bf16: DevBuf,
}

pub struct DevChain<'a> {
    pub dev: &'a Device,
    pub cfg: &'a Dsv41Config,
    pub w: &'a Dsv41DevWeights,
    pub opts: RunOpts,
    /// Tensor-parallel collective. `None` runs the model on one device; when
    /// present, the row-parallel sites reduce across ranks.
    pub comm: Option<Arc<Collective>>,
    layers: Vec<LayerCache>,
    s: Scratch,
    cos: DevBuf,
    sin: DevBuf,
}

fn fb(n: usize) -> usize {
    n * 4
}

impl<'a> DevChain<'a> {
    pub fn new(
        dev: &'a Device,
        cfg: &'a Dsv41Config,
        w: &'a Dsv41DevWeights,
        opts: RunOpts,
    ) -> Result<Self> {
        let dim = cfg.dim;
        let hc = cfg.hc_mult;
        let hd = cfg.head_dim;
        let nh = cfg.n_heads;
        let inter = cfg.moe_inter_dim;
        let n_exp = cfg.n_routed_experts.max(1);
        let topk = cfg.n_activated_experts.max(1);
        let ql = cfg.q_lora_rank;
        let bf16_cap = dim.max(ql).max(nh * hd).max(cfg.vocab_size);
        let bf16 = dev.alloc(bf16_cap * 2)?;

        let mut layers = Vec::with_capacity(cfg.n_layers + cfg.n_mtp_layers);
        for l in 0..cfg.n_layers + cfg.n_mtp_layers {
            let ratio = cfg.compress_ratio(l).max(1);
            let max_comp = cfg.max_seq_len / ratio + 2;
            layers.push(LayerCache {
                ring: dev.alloc(fb(cfg.window_size * hd))?,
                idxs: dev.alloc(fb(cfg.window_size + cfg.index_topk + 8).max(4))?,
                state_kv: dev.alloc(fb(ratio * hd))?,
                state_score: dev.alloc(fb(ratio * hd))?,
                kvp: dev.alloc(fb(hd))?,
                scp: dev.alloc(fb(hd))?,
                latent: dev.alloc(fb(hd))?,
                out_rows: dev.alloc(4)?,
                compress_len: 0,
                index_k: dev.alloc(fb(max_comp * cfg.index_head_dim.max(1)))?,
            });
        }

        let s = Scratch {
            h: dev.alloc(fb(hc * dim))?,
            h2: dev.alloc(fb(hc * dim))?,
            x: dev.alloc(fb(dim))?,
            xn: dev.alloc(fb(dim))?,
            // sized for the LARGEST activation quantised anywhere in the chain
            // (the attention output has n_heads*head_dim elements, well past
            // `dim` — sizing these by `dim` overflowed on the output projection)
            xq: dev.alloc(dim.max(nh * hd).max(cfg.o_lora_rank).max(inter))?,
            xsc: dev.alloc(fb(dim.max(nh * hd).max(cfg.o_lora_rank).max(inter) / 32 + 8))?,
            pre: dev.alloc(fb(hc))?,
            post: dev.alloc(fb(hc))?,
            comb: dev.alloc(fb(hc * hc))?,
            qr: dev.alloc(fb(ql))?,
            q: dev.alloc(fb(nh * hd))?,
            kv: dev.alloc(fb(hd))?,
            o: dev.alloc(fb(nh * hd))?,
            wo: dev.alloc(fb(cfg.n_groups_o_lora()))?,
            logits: dev.alloc(fb(cfg.vocab_size))?,
            ids: dev.alloc(4)?,
            scores: dev.alloc(fb(n_exp))?,
            route_idx: dev.alloc(fb(topk).max(4))?,
            route_w: dev.alloc(fb(topk).max(4))?,
            ex_in: dev.alloc(fb(dim))?,
            ex_act: dev.alloc(fb(2 * inter.max(dim)))?,
            ex_out: dev.alloc(fb(dim))?,
            hist: dev.alloc(fb(n_exp))?,
            idx_q: dev.alloc(fb(cfg.index_n_heads.max(1) * cfg.index_head_dim.max(1)))?,
            idx_k: dev.alloc(fb(cfg.index_head_dim.max(1)))?,
            idx_w: dev.alloc(fb(cfg.index_n_heads.max(1)))?,
            idx_lens: dev.alloc(4)?,
            bf16,
        };

        // RoPE tables covering the whole context.
        let table = cfg.max_seq_len.min(1 << 20);
        let half = cfg.rope_head_dim / 2;
        let cos = dev.alloc(fb(table * half))?;
        let sin = dev.alloc(fb(table * half))?;
        dev.rope_precompute(
            cos.ptr as *mut f32,
            sin.ptr as *mut f32,
            cfg.rope_head_dim as i32,
            table as i32,
            cfg.original_seq_len as i32,
            cfg.rope_theta,
            cfg.rope_factor,
            cfg.beta_fast,
            cfg.beta_slow,
        )?;
        let _ = bf16_cap;

        Ok(DevChain {
            dev,
            cfg,
            w,
            opts,
            comm: None,
            layers,
            s,
            cos,
            sin,
        })
    }

    pub fn reset(&mut self) -> Result<()> {
        for c in self.layers.iter_mut() {
            self.dev.zero(&c.ring)?;
            self.dev.zero(&c.state_kv)?;
            self.dev.zero(&c.state_score)?;
            c.compress_len = 0;
            // score_state is -inf except where filled; zeroing it would make an
            // empty slot look like a real (0-weight) entry, so seed it with -inf
            let neg = f32::NEG_INFINITY;
            let n = c.state_score.bytes / 4;
            let v = vec![neg; n];
            self.dev.upload_f32_at(c.state_score.ptr, 0, &v)?;
        }
        Ok(())
    }

    /// Quantise `src` (one row of `k` floats) to fp8 into `s.xq`/`s.xsc`.
    fn quant1(&self, src: *const f32, k: i32) -> Result<()> {
        self.dev
            .quant_fp8(src, self.s.xq.ptr as *mut u8, self.s.xsc.ptr as *mut f32, 1, k, 32, true)
    }

    /// fp8 dense linear for one row: `out[1, n_out] = a[1, k] @ w[n_out, k]^T`.
    fn lin(&self, a: *const f32, k: i32, w: &crate::load::DevTensor, ws: &crate::load::DevTensor, n_out: i32, out: *mut f32) -> Result<()> {
        self.quant1(a, k)?;
        self.dev.gemm_fp8_mx(
            self.s.xq.as_u8(),
            self.s.xsc.as_f32(),
            w.as_u8(),
            ws.as_u8(),
            std::ptr::null(),
            out,
            1,
            n_out,
            k,
        )
    }

    /// f32 linear for one row (see `gemm_f32`).
    fn lin_f32(&self, a: *const f32, k: i32, w: &crate::load::DevTensor, n_out: i32, out: *mut f32) -> Result<()> {
        self.dev
            .gemm_f32(a as *const c_void, w.ptr() as *const c_void, out, 1, n_out, k)
    }

    /// bf16 linear for one row (cuBLAS; the tensor is natively bf16).
    fn lin_bf16(&self, a: *const f32, k: i32, w: &crate::load::DevTensor, n_out: i32, out: *mut f32) -> Result<()> {
        self.dev.f32_to_bf16(a, self.s.bf16.ptr as *mut c_void, k as i64)?;
        self.dev
            .gemm_bf16(self.s.bf16.ptr as *const c_void, w.ptr() as *const c_void, out, 1, n_out, k)
    }

    /// One decode step. Returns the logits for the fed token.
    pub fn step(&mut self, token: u32, pos: usize) -> Result<Vec<f32>> {
        let cfg = self.cfg;
        let dim = cfg.dim;
        let hc = cfg.hc_mult;
        let ids = [token as i32];
        self.dev.upload_f32_at(self.s.ids.ptr, 0, unsafe {
            std::slice::from_raw_parts(ids.as_ptr() as *const f32, 1)
        })?;
        // embedding + hyper-connection expansion lands directly in `h`
        self.dev.embed_expand_dev(
            self.w.embed.as_ref().unwrap().ptr(),
            self.s.ids.as_i32(),
            self.s.h.ptr as *mut f32,
            1,
            dim as i32,
            hc as i32,
            cfg.vocab_size as i32,
        )?;

        // the initial collapse takes copy 0 of the stream
        let mut premix = vec![0f32; hc];
        premix[0] = 1.0;
        for layer in 0..cfg.n_layers {
            premix = self.layer(layer, pos, &premix)?;
        }
        self.upload_pre(&premix)?;
        self.dev.hc_collapse(
            self.s.h.ptr as *const f32,
            self.s.pre.as_f32(),
            self.s.x.ptr as *mut f32,
            1,
            hc as i32,
            dim as i32,
        )?;
        self.dev.rmsnorm(
            self.s.x.ptr as *const f32,
            self.w.norm.as_ref().unwrap().as_f32(),
            self.s.xn.ptr as *mut f32,
            1,
            dim as i32,
            cfg.norm_eps,
        )?;
        self.lin_bf16(
            self.s.xn.ptr as *const f32,
            dim as i32,
            self.w.head.as_ref().unwrap(),
            cfg.vocab_size as i32,
            self.s.logits.ptr as *mut f32,
        )?;
        self.stats("final logits", &self.s.logits, cfg.vocab_size)?;
        self.dev.sync()?;
        let mut out = vec![0f32; cfg.vocab_size];
        self.dev.download_f32(&self.s.logits, &mut out)?;
        Ok(out)
    }

    /// Tensor-parallel degree / this rank's index (1 / 0 without a collective).
    fn world(&self) -> usize {
        self.comm.as_ref().map(|c| c.world).unwrap_or(1)
    }
    fn rank(&self) -> usize {
        self.comm.as_ref().map(|c| c.rank).unwrap_or(0)
    }

    fn upload_pre(&self, v: &[f32]) -> Result<()> {
        self.dev.upload_f32_at(self.s.pre.ptr, 0, v)
    }

    fn dl(&self, src: *const f32, n: usize) -> Result<Vec<f32>> {
        let mut v = vec![0f32; n];
        let b = Device::view(src as *mut c_void, n * 4);
        self.dev.download_f32(&b, &mut v)?;
        Ok(v)
    }

    fn ul_i32(&self, dst: *mut c_void, v: &[i32]) -> Result<()> {
        self.dev.upload_f32_at(dst, 0, unsafe {
            std::slice::from_raw_parts(v.as_ptr() as *const f32, v.len())
        })
    }

    fn ul_f32(&self, dst: *mut c_void, v: &[f32]) -> Result<()> {
        self.dev.upload_f32_at(dst, 0, v)
    }

    /// One transformer layer. Returns the pre-mix the next block's first
    /// collapse must use (this block's *attention* mix).
    fn layer(&mut self, layer: usize, pos: usize, premix: &[f32]) -> Result<Vec<f32>> {
        let cfg = self.cfg;
        let dim = cfg.dim;
        let hc = cfg.hc_mult;
        let ld = &self.w.layers[layer];

        // ---------------- attention block ----------------
        self.dev.hc_mixes(
            self.s.h.ptr as *const f32,
            ld.hc_attn_fn.as_ref().unwrap().as_f32(),
            ld.hc_attn_scale.as_ref().unwrap().as_f32(),
            ld.hc_attn_base.as_ref().unwrap().as_f32(),
            self.s.pre.ptr as *mut f32,
            self.s.post.ptr as *mut f32,
            self.s.comb.ptr as *mut f32,
            1,
            (hc * dim) as i32,
            hc as i32,
            cfg.hc_sinkhorn_iters as i32,
            cfg.hc_eps,
        )?;
        let attn_pre = self.dl(self.s.pre.as_f32(), hc)?;
        self.upload_pre(premix)?;
        self.dev.hc_collapse(
            self.s.h.ptr as *const f32,
            self.s.pre.as_f32(),
            self.s.x.ptr as *mut f32,
            1,
            hc as i32,
            dim as i32,
        )?;
        self.dev.rmsnorm(
            self.s.x.ptr as *const f32,
            ld.attn_norm.as_ref().unwrap().as_f32(),
            self.s.xn.ptr as *mut f32,
            1,
            dim as i32,
            cfg.norm_eps,
        )?;
        self.attention(layer, pos)?;
        if layer == 0 {
            self.stats("L0 attn_out(o)", &self.s.o, dim)?;
        }
        self.dev.hc_post(
            self.s.o.ptr as *const f32,
            self.s.h.ptr as *const f32,
            self.s.post.as_f32(),
            self.s.comb.as_f32(),
            self.s.h2.ptr as *mut f32,
            1,
            hc as i32,
            dim as i32,
        )?;
        self.copy_h_back()?;

        // ---------------- FFN block ----------------
        self.dev.hc_mixes(
            self.s.h.ptr as *const f32,
            ld.hc_ffn_fn.as_ref().unwrap().as_f32(),
            ld.hc_ffn_scale.as_ref().unwrap().as_f32(),
            ld.hc_ffn_base.as_ref().unwrap().as_f32(),
            self.s.pre.ptr as *mut f32,
            self.s.post.ptr as *mut f32,
            self.s.comb.ptr as *mut f32,
            1,
            (hc * dim) as i32,
            hc as i32,
            cfg.hc_sinkhorn_iters as i32,
            cfg.hc_eps,
        )?;
        self.upload_pre(&attn_pre)?;
        self.dev.hc_collapse(
            self.s.h.ptr as *const f32,
            self.s.pre.as_f32(),
            self.s.x.ptr as *mut f32,
            1,
            hc as i32,
            dim as i32,
        )?;
        self.dev.rmsnorm(
            self.s.x.ptr as *const f32,
            ld.ffn_norm.as_ref().unwrap().as_f32(),
            self.s.xn.ptr as *mut f32,
            1,
            dim as i32,
            cfg.norm_eps,
        )?;
        self.moe(layer, ld)?;
        if layer == 0 {
            self.stats("L0 moe_out(o)", &self.s.o, dim)?;
        }
        self.dev.hc_post(
            self.s.o.ptr as *const f32,
            self.s.h.ptr as *const f32,
            self.s.post.as_f32(),
            self.s.comb.as_f32(),
            self.s.h2.ptr as *mut f32,
            1,
            hc as i32,
            dim as i32,
        )?;
        self.copy_h_back()?;
        Ok(attn_pre)
    }

    /// Diagnostic: report the magnitude of a stage's output. `DSV41_STATS=1`.
    /// Turns "the text is wrong" into "stage X is fine / stage Y exploded".
    fn stats(&self, label: &str, buf: &DevBuf, n: usize) -> Result<()> {
        if std::env::var("DSV41_STATS").map(|v| v != "0").unwrap_or(false) {
            self.dev.sync()?;
            let mut v = vec![0f32; n];
            let b = Device::view(buf.ptr, n * 4);
            self.dev.download_f32(&b, &mut v)?;
            let mut mx = f32::NEG_INFINITY;
            let mut mn = f32::INFINITY;
            let mut ss = 0f64;
            let mut nan = 0usize;
            for &x in &v {
                if x.is_nan() {
                    nan += 1;
                } else {
                    mx = mx.max(x);
                    mn = mn.min(x);
                    ss += (x as f64) * (x as f64);
                }
            }
            eprintln!(
                "[stats] {label:<26} rms={:9.4} min={:9.4} max={:9.4} nan={nan}",
                (ss / n as f64).sqrt(),
                mn,
                mx
            );
        }
        Ok(())
    }

    fn copy_h_back(&self) -> Result<()> {
        self.dev
            .memcpy_d2d(self.s.h.ptr, self.s.h2.ptr as *const c_void, self.s.h.bytes)
    }

    /// MLA window path + grouped output projection.
    fn attention(&mut self, layer: usize, pos: usize) -> Result<()> {
        let cfg = self.cfg;
        let dim = cfg.dim;
        let hd = cfg.head_dim;
        let nh = cfg.n_heads;
        let ql = cfg.q_lora_rank;
        let world = self.world();
        let rank = self.rank();
        // wq_b is ColumnParallel: this rank owns a contiguous block of heads
        let nlh = nh / world;
        let ld = &self.w.layers[layer];

        // queries: wq_a -> q_norm -> wq_b
        self.lin(
            self.s.xn.ptr as *const f32,
            dim as i32,
            ld.wq_a.as_ref().unwrap(),
            ld.wq_a_scale.as_ref().unwrap(),
            ql as i32,
            self.s.qr.ptr as *mut f32,
        )?;
        self.dev.rmsnorm(
            self.s.qr.ptr as *const f32,
            ld.q_norm.as_ref().unwrap().as_f32(),
            self.s.qr.ptr as *mut f32,
            1,
            ql as i32,
            cfg.norm_eps,
        )?;
        self.lin(
            self.s.qr.ptr as *const f32,
            ql as i32,
            ld.wq_b.as_ref().unwrap(),
            ld.wq_b_scale.as_ref().unwrap(),
            (nlh * hd) as i32,
            self.s.q.ptr as *mut f32,
        )?;
        // RoPE over the trailing `rope_head_dim` lanes of each head
        self.dev.apply_rope(
            self.s.q.ptr as *mut f32,
            self.cos.as_f32(),
            self.sin.as_f32(),
            nlh as i32,
            hd as i32,
            cfg.rope_head_dim as i32,
            (cfg.rope_head_dim / 2) as i32,
            pos as i32,
            1,
            false,
        )?;

        // window KV (single shared head)
        self.lin(
            self.s.xn.ptr as *const f32,
            dim as i32,
            ld.wkv.as_ref().unwrap(),
            ld.wkv_scale.as_ref().unwrap(),
            hd as i32,
            self.s.kv.ptr as *mut f32,
        )?;
        self.dev.rmsnorm(
            self.s.kv.ptr as *const f32,
            ld.kv_norm.as_ref().unwrap().as_f32(),
            self.s.kv.ptr as *mut f32,
            1,
            hd as i32,
            cfg.norm_eps,
        )?;
        self.dev.apply_rope(
            self.s.kv.ptr as *mut f32,
            self.cos.as_f32(),
            self.sin.as_f32(),
            1,
            hd as i32,
            cfg.rope_head_dim as i32,
            (cfg.rope_head_dim / 2) as i32,
            pos as i32,
            1,
            false,
        )?;

        let win = cfg.window_size;
        let slot = pos % win;
        // raw pointers rather than a live borrow: `compress` below needs
        // &mut self (it updates this layer's published count and buffers)
        // The release shares one KV store across a group of layers: the kv
        // source maintains it and its consumers read it. A consumer therefore
        // must not keep its own window ring (nothing would ever put the
        // compressed rows there), it reads the owner's — which also already
        // holds this step's token, since the owner runs earlier in the stack.
        let owner = self.kv_owner(layer);
        let ring_ptr = self.layers[owner].ring.ptr;
        let idxs_ptr = self.layers[owner].idxs.ptr;
        let cache = &self.layers[owner];
        let owns_kv = owner == layer;
        if owns_kv {
            self.dev.memcpy_d2d(
                (cache.ring.ptr as *mut u8).wrapping_add(slot * fb(hd)) as *mut c_void,
                self.s.kv.ptr as *const c_void,
                fb(hd),
            )?;
        }

        // selection: the window ring, oldest first (the ring index already
        // carries the ageing rotation), padded with -1
        // The window row is `win` entries in ring order with empty slots marked
        // -1, and `sparse_attn` skips negatives. The earlier revision took only
        // the LEADING `pos+1` entries — but those are the *high* slots, which
        // are exactly the invalid ones while `pos < window` — so every decode
        // step selected nothing and the attention output came out identically
        // zero (confirmed by DSV41_STATS: rms=0.0000 at pos>0 while pos=0 was
        // fine, since only then do the leading entries happen to be valid).
        let wsel = ops::window_topk_idxs(win, 1, 1, pos);
        let mut idx_host = vec![-1i32; win + cfg.index_topk];
        for (c, v) in wsel.iter().enumerate().take(win) {
            idx_host[c] = *v;
        }
        // Compressed KV. Only the kv sources run the compressor; every other
        // layer of the same group reads the latents they published, which is
        // why they all live in this layer's own copy of the sequence's rows.
        let mut comp_len = self.layers[layer].compress_len;
        if cfg.compress_ratio(layer) > 0 && cfg.is_kv_source(layer) {
            comp_len = self.compress(layer, pos)?;
        } else if cfg.compress_ratio(layer) > 0 {
            // a consumer inherits the count published by its source layer
            comp_len = self.source_compress_len(layer);
        }
        // Selection over the compressed rows. The release scores them with the
        // indexer and keeps `index_topk`; until the indexer is wired this takes
        // the most recent ones, which is a deliberate placeholder (it is a
        // superset-free pruning that at least makes the long-range rows
        // reachable — it is NOT the learned selection).
        let mut take_comp = comp_len.min(cfg.index_topk);
        if !owns_kv {
            // A consumer reads the owner's selection buffer, which the owner
            // filled earlier this step (including its learned top-k). Uploading
            // here would overwrite it with the placeholder — a consumer must
            // never write the shared buffer.
        } else if comp_len > 0 && cfg.is_index_source(layer) && cfg.indexer_owns_k(layer)
            && self.indexer(layer, pos, win, comp_len)? {
            // the kernel wrote `comp_len.min(index_topk)` entries at [win, ..)
            take_comp = comp_len.min(cfg.index_topk);
        } else {
            // No indexer on this owner yet: keep the most recent compressed rows.
            // NOT the learned selection — it only makes the long-range rows
            // reachable, and the config makes every compress owner an index
            // source, so in practice this branch is a safety net.
            let placeholder = comp_len.min(cfg.index_topk);
            for j in 0..placeholder {
                idx_host[win + j] = (win + comp_len - placeholder + j) as i32;
            }
            take_comp = placeholder;
            self.ul_i32(idxs_ptr, &idx_host)?;
        }
        let n_idx_cols = win + take_comp;

        self.dev.sparse_attn(
            self.s.q.as_f32(),
            ring_ptr as *const f32,
            ld.attn_sink.as_ref().unwrap().as_f32(),
            idxs_ptr as *const i32,
            self.s.o.ptr as *mut f32,
            1,
            1,
            nlh as i32,
            hd as i32,
            (win + comp_len) as i32,
            n_idx_cols.max(1) as i32,
            1.0 / (hd as f32).sqrt(),
        )?;
        self.dev.apply_rope(
            self.s.o.ptr as *mut f32,
            self.cos.as_f32(),
            self.sin.as_f32(),
            nlh as i32,
            hd as i32,
            cfg.rope_head_dim as i32,
            (cfg.rope_head_dim / 2) as i32,
            pos as i32,
            1,
            true,
        )?;

        // block-diagonal grouped output projection: group g owns rows
        // [g*o_lora, (g+1)*o_lora) against the head slice [g*hpg*hd, ...)
        let groups = cfg.o_groups;
        let hpg = nh / groups;
        // `o_lora_rank` is the PER-GROUP low-rank width (the reference's
        // wo_a weight is [n_groups * o_lora_rank, hpg*head_dim] viewed as
        // [n_groups, o_lora_rank, hpg*head_dim]), so a group's row block is
        // o_lora_rank tall — not o_lora_rank/groups.
        let olg = cfg.o_lora_rank;
        let nlg = groups / world; // wo_a is ColumnParallel: a block of groups each
        let k = hpg * hd;
        self.quant1(self.s.o.ptr as *const f32, (nlh * hd) as i32)?;
        for g in 0..nlg {
            // the rank's groups are a contiguous block, and its head block maps
            // onto exactly those groups (a group is hpg heads and olg output rows)
            let g_glob = rank * nlg + g; // global group this rank block covers
            let a = self.s.xq.as_u8().wrapping_add(g * k);
            let asc = self.s.xsc.as_f32().wrapping_add((g * k / 32) as usize);
            // weights and the output use GLOBAL group offsets (that is the
            // layout on disk and the layout the chain's `wo` buffer mirrors)
            let wp = ld
                .wo_a
                .as_ref()
                .unwrap()
                .as_u8()
                .wrapping_add(g_glob * olg * k);
            let wsp = ld
                .wo_a_scale
                .as_ref()
                .unwrap()
                .as_u8()
                .wrapping_add((g_glob * olg / 32) * (k / 32));
            self.dev.gemm_fp8_mx(
                a,
                asc,
                wp,
                wsp,
                std::ptr::null(),
                (self.s.wo.ptr as *mut f32).wrapping_add(g_glob * olg),
                1,
                olg as i32,
                k as i32,
            )?;
        }
        // wo_b is RowParallel: the input (groups*o_lora) is split, so this rank
        // reduces over its own slice and the ranks' partial sums are added.
        let ol_total = groups * cfg.o_lora_rank;
        let ol_local = ol_total / world;
        self.lin(
            (self.s.wo.ptr as *const f32).wrapping_add(rank * ol_local),
            ol_local as i32,
            ld.wo_b.as_ref().unwrap(),
            ld.wo_b_scale.as_ref().unwrap(),
            dim as i32,
            self.s.o.ptr as *mut f32,
        )?;
        if let Some(c) = self.comm.clone() {
            c.all_reduce_inplace(self.s.o.ptr as *mut std::ffi::c_void, fb(dim))?;
            c.end_round();
        }
        Ok(())
    }

    /// Indexer for one decode step: publish this layer's index key for the
    /// latent the compressor just produced, then score the published keys and
    /// let the kernel write the top-k straight into the selection buffer at
    /// `offset` (`window`, so it lands after the window block).
    /// Returns false when there was nothing to select from.
    fn indexer(&mut self, layer: usize, pos: usize, offset: usize, comp_len: usize) -> Result<bool> {
        let cfg = self.cfg;
        let dim = cfg.dim;
        let ql = cfg.q_lora_rank;
        let idx_nh = cfg.index_n_heads;
        let idx_hd = cfg.index_head_dim;
        let rd = cfg.rope_head_dim;
        let ratio = cfg.compress_ratio(layer).max(1);
        let ld = &self.w.layers[layer];
        let (Some(wq_b), Some(wq_b_s), Some(wk), Some(kn), Some(wp)) = (
            ld.idx_wq_b.as_ref(),
            ld.idx_wq_b_scale.as_ref(),
            ld.idx_wk.as_ref(),
            ld.idx_k_norm.as_ref(),
            ld.idx_weights.as_ref(),
        ) else {
            return Ok(false);
        };
        // the key for the group whose latent was just published (the latent
        // stands for its group's FIRST token, so RoPE uses that position)
        let group = self.layers[layer].compress_len.saturating_sub(1);
        self.lin_bf16(
            self.layers[layer].latent.ptr as *const f32,
            cfg.head_dim as i32,
            wk,
            idx_hd as i32,
            self.s.idx_k.ptr as *mut f32,
        )?;
        self.dev.rmsnorm(
            self.s.idx_k.ptr as *const f32,
            kn.as_f32(),
            self.s.idx_k.ptr as *mut f32,
            1,
            idx_hd as i32,
            cfg.norm_eps,
        )?;
        self.dev.apply_rope(
            self.s.idx_k.ptr as *mut f32,
            self.cos.as_f32(),
            self.sin.as_f32(),
            1,
            idx_hd as i32,
            rd as i32,
            (rd / 2) as i32,
            (group * ratio) as i32,
            1,
            false,
        )?;
        self.dev.memcpy_d2d(
            (self.layers[layer].index_k.ptr as *mut u8).wrapping_add(group * idx_hd * 4) as *mut c_void,
            self.s.idx_k.ptr as *const c_void,
            idx_hd * 4,
        )?;
        // the queries come from the q_lora stream
        self.lin(
            self.s.qr.ptr as *const f32,
            ql as i32,
            wq_b,
            wq_b_s,
            (idx_nh * idx_hd) as i32,
            self.s.idx_q.ptr as *mut f32,
        )?;
        self.dev.apply_rope(
            self.s.idx_q.ptr as *mut f32,
            self.cos.as_f32(),
            self.sin.as_f32(),
            idx_nh as i32,
            idx_hd as i32,
            rd as i32,
            (rd / 2) as i32,
            pos as i32,
            1,
            false,
        )?;
        // per-head weights; the reference folds softmax_scale * n_heads^-0.5 into
        // them, and our kernel applies softmax_scale * head_scale to the sum, so
        // the same factor can be passed there instead
        self.lin_bf16(
            self.s.xn.ptr as *const f32,
            dim as i32,
            wp,
            idx_nh as i32,
            self.s.idx_w.ptr as *mut f32,
        )?;
        let lens = [comp_len as i32];
        self.dev.upload_f32_at(self.s.idx_lens.ptr, 0, unsafe {
            std::slice::from_raw_parts(lens.as_ptr() as *const f32, 1)
        })?;
        let scale = 1.0f32 / (cfg.head_dim as f32).sqrt() / (idx_nh as f32).sqrt();
        self.dev.indexer_topk(
            self.s.idx_q.as_f32(),
            self.layers[layer].index_k.as_f32(),
            self.s.idx_w.as_f32(),
            std::ptr::null(),
            self.s.idx_lens.as_i32(),
            // the kernel writes `picked + offset` into out[row*cols + i], so
            // `out` points at the first compressed slot of the row
            (self.layers[layer].idxs.ptr as *mut i32).wrapping_add(offset),
            1,
            1,
            idx_nh as i32,
            idx_hd as i32,
            comp_len as i32,
            cfg.index_topk as i32,
            offset as i32,
            scale,
            1.0,
            false,
        )?;
        Ok(true)
    }

    /// Compressor for one decode step: the two projections in fp32 (the release
    /// promotes them), then the pooling half, then the latent lands in this
    /// layer's KV buffer at row `window + compress_len`. Returns the new count.
    fn compress(&mut self, layer: usize, pos: usize) -> Result<usize> {
        let cfg = self.cfg;
        let dim = cfg.dim;
        let hd = cfg.head_dim;
        let ratio = cfg.compress_ratio(layer).max(1);
        let ld = &self.w.layers[layer];
        let (Some(wkv), Some(norm)) = (ld.comp_wkv.as_ref(), ld.comp_norm.as_ref()) else {
            return Ok(self.layers[layer].compress_len);
        };
        let cache = &self.layers[layer];
        self.lin_f32(self.s.xn.ptr as *const f32, dim as i32, wkv, hd as i32, cache.kvp.ptr as *mut f32)?;
        if let Some(wg) = ld.comp_wgate.as_ref() {
            self.lin_f32(self.s.xn.ptr as *const f32, dim as i32, wg, hd as i32, cache.scp.ptr as *mut f32)?;
        } else {
            // ratio == 1: no gate; the pooling reduces to the plain projection
            self.dev.zero(&cache.scp)?;
        }
        self.dev.compressor_pool(
            cache.kvp.as_f32(),
            cache.scp.as_f32(),
            norm.as_f32(),
            cache.state_kv.ptr as *mut f32,
            cache.state_score.ptr as *mut f32,
            cache.latent.ptr as *mut f32,
            cache.out_rows.ptr as *mut i32,
            1,
            1,
            hd as i32,
            ratio as i32,
            pos as i32,
            cfg.norm_eps,
        )?;
        // the kernel reports on the device whether a latent came out this step
        let mut n = [0i32; 1];
        let b = Device::view(cache.out_rows.ptr, 4);
        self.dev.download_f32(&b, unsafe {
            std::slice::from_raw_parts_mut(n.as_mut_ptr() as *mut f32, 1)
        })?;
        let mut len = self.layers[layer].compress_len;
        if n[0] > 0 {
            let dst = (self.layers[layer].ring.ptr as *mut u8)
                .wrapping_add((self.cfg.window_size + len) * hd * 4);
            self.dev.memcpy_d2d(
                dst as *mut c_void,
                self.layers[layer].latent.ptr as *const c_void,
                hd * 4,
            )?;
            len += 1;
            self.layers[layer].compress_len = len;
        }
        Ok(len)
    }

    /// The layer whose KV store `layer` reads: itself, unless it is a consumer
    /// of a group whose compressed KV is maintained by the source above it.
    fn kv_owner(&self, layer: usize) -> usize {
        if self.cfg.kv_mode(layer) != KvMode::CompressConsumer {
            return layer;
        }
        for l in (0..layer).rev() {
            if self.cfg.kv_mode(l) == KvMode::CompressSource
                && self.cfg.compress_ratio(l) == self.cfg.compress_ratio(layer)
            {
                return l;
            }
        }
        layer
    }

    /// How many compressed rows the source layer of `layer` has published.
    /// Consumers share the source's cache, so they report the same count.
    fn source_compress_len(&self, layer: usize) -> usize {
        // the config lists compressors per layer; a consumer reads the most
        // recent source at or before it (the reference's kv_source mapping)
        for l in (0..=layer).rev() {
            if self.cfg.is_kv_source(l) && self.cfg.compress_ratio(l) == self.cfg.compress_ratio(layer) {
                return self.layers[l].compress_len;
            }
        }
        0
    }

    /// MoE: bf16 gate GEMM, `noaux_tc` routing, MXFP4 experts, fp8 shared expert.
    fn moe(&mut self, layer: usize, ld: &LayerDev) -> Result<()> {
        let cfg = self.cfg;
        let dim = cfg.dim;
        let inter = cfg.moe_inter_dim;
        // The expert matrices are sharded along `inter` (the reference cuts them
        // by inter/world), so every expert kernel must be sized by the LOCAL
        // width — passing the global 2304 made the gate/up kernel write twice
        // the rows the weight has, which is what faulted under TP.
        let inter_local = inter / self.world();
        let (n_routed, topk) = cfg.moe_config(layer);

        // gate: natively bf16, so a bf16 GEMM
        self.lin_bf16(
            self.s.xn.ptr as *const f32,
            dim as i32,
            ld.gate_w.as_ref().unwrap(),
            n_routed as i32,
            self.s.scores.ptr as *mut f32,
        )?;
        self.dev.route_topk(
            self.s.scores.as_f32(),
            ld.gate_bias.as_ref().map(|b| b.as_f32()).unwrap_or(std::ptr::null()),
            self.s.route_w.ptr as *mut f32,
            self.s.route_idx.ptr as *mut i32,
            self.s.hist.ptr as *mut i32,
            1,
            n_routed as i32,
            topk as i32,
            cfg.norm_topk_prob,
            cfg.route_scale,
            2, // sqrtsoftplus, per the checkpoint's routing
        )?;
        let mut idx = vec![0i32; topk];
        let mut wgt = vec![0f32; topk];
        self.dev.sync()?;
        self.dev.download_f32(&self.s.route_idx, unsafe {
            std::slice::from_raw_parts_mut(idx.as_mut_ptr() as *mut f32, topk)
        })?;
        self.dev.download_f32(&self.s.route_w, &mut wgt)?;

        // One token: every assignment shares the input row, so expert e's total
        // contribution is expert_e(x) * sum of its routing weights. Accumulating
        // the distinct experts in index order keeps the sum deterministic.
        self.dev.zero(&self.s.ex_out)?;
        let ne = ld.experts.len();
        // Expert-parallel: the loader hands this rank the CONTIGUOUS block
        // [rank*ne, (rank+1)*ne) of the global expert ids, so the global id must
        // be rebased before indexing the local array (the earlier version
        // indexed it directly, which silently used another rank's experts).
        let e_base = self.rank() * ne;
        let mut wsum = vec![0f32; ne.max(1)];
        for (slot, &e) in idx.iter().enumerate() {
            let e = e as usize;
            if e >= e_base && e < e_base + ne {
                wsum[e - e_base] += wgt[slot];
            }
        }
        if !self.opts.skip_experts {
            for e in 0..ne {
                if wsum[e] == 0.0 {
                    continue;
                }
                let ex = &ld.experts[e];
                // the input row is the same for every expert: quantise once per
                // expert (the fp4 kernel consumes it in place of a gather)
                self.dev.quant_fp4(
                    self.s.xn.ptr as *const f32,
                    self.s.xq.ptr as *mut u8,
                    self.s.xsc.ptr as *mut f32,
                    1,
                    dim as i32,
                    32,
                    true,
                )?;
                self.dev.expert_gate_up_fp4(
                    self.s.xq.as_u8(),
                    self.s.xsc.as_f32(),
                    ex.w1.as_u8(),
                    ex.w1_scale.as_u8(),
                    ex.w3.as_u8(),
                    ex.w3_scale.as_u8(),
                    self.s.ex_act.ptr as *mut f32,
                    1,
                    dim as i32,
                    inter_local as i32,
                    cfg.swiglu_limit,
                )?;
                self.dev
                    .swiglu_limit(self.s.ex_act.ptr as *mut f32, 1, inter_local as i32, cfg.swiglu_limit)?;
                self.ul_f32(self.s.ex_in.ptr, &[wsum[e]])?;
                self.dev.expert_down_fp4(
                    self.s.ex_act.ptr as *const f32,
                    ex.w2.as_u8(),
                    ex.w2_scale.as_u8(),
                    self.s.ex_in.as_f32(),
                    self.s.ex_out.ptr as *mut f32,
                    1,
                    dim as i32,
                    inter_local as i32,
                )?;
                // accumulate deterministically into o
                self.dev.add_inplace(&self.s.o, &self.s.ex_out, dim as i64)?;
            }
        }
        self.dev.memcpy_d2d(self.s.o.ptr, self.s.ex_out.ptr as *const c_void, fb(dim))?;

        // shared expert: fp8, every token. Its weights are replicated, so under
        // a collective exactly one rank may contribute it — otherwise the
        // all-reduce below would sum it `world` times.
        let shared_rank = self.comm.as_ref().map(|c| c.rank == 0).unwrap_or(true);
        if !self.opts.skip_shared_expert && shared_rank {
            if let (Some(w1), Some(w1s), Some(w3), Some(w3s), Some(w2), Some(w2s)) = (
                ld.shared_w1.as_ref(),
                ld.shared_w1_scale.as_ref(),
                ld.shared_w3.as_ref(),
                ld.shared_w3_scale.as_ref(),
                ld.shared_w2.as_ref(),
                ld.shared_w2_scale.as_ref(),
            ) {
                self.quant1(self.s.xn.ptr as *const f32, dim as i32)?;
                // gate and up land contiguously so `swiglu_limit` sees [gate|up]
                self.dev.gemm_fp8_mx(
                    self.s.xq.as_u8(),
                    self.s.xsc.as_f32(),
                    w1.as_u8(),
                    w1s.as_u8(),
                    std::ptr::null(),
                    self.s.ex_act.ptr as *mut f32,
                    1,
                    inter as i32,
                    dim as i32,
                )?;
                self.dev.gemm_fp8_mx(
                    self.s.xq.as_u8(),
                    self.s.xsc.as_f32(),
                    w3.as_u8(),
                    w3s.as_u8(),
                    std::ptr::null(),
                    (self.s.ex_act.ptr as *mut f32).wrapping_add(inter),
                    1,
                    inter as i32,
                    dim as i32,
                )?;
                self.dev
                    .swiglu_limit(self.s.ex_act.ptr as *mut f32, 1, inter as i32, cfg.swiglu_limit)?;
                self.quant1(self.s.ex_act.ptr as *const f32, inter as i32)?;
                self.dev.gemm_fp8_mx(
                    self.s.xq.as_u8(),
                    self.s.xsc.as_f32(),
                    w2.as_u8(),
                    w2s.as_u8(),
                    std::ptr::null(),
                    self.s.ex_out.ptr as *mut f32,
                    1,
                    dim as i32,
                    inter as i32,
                )?;
                self.dev.add_inplace(&self.s.o, &self.s.ex_out, dim as i64)?;
            }
        }
        // routed experts are expert-parallel, so each rank holds a partial sum
        if let Some(c) = self.comm.clone() {
            c.all_reduce_inplace(self.s.o.ptr as *mut std::ffi::c_void, fb(dim))?;
            c.end_round();
        }
        // the block output is the attention-branch accumulator `o`
        Ok(())
    }
}
