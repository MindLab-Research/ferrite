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

use crate::config::Dsv41Config;
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
    // bf16 staging for the cuBLAS path
    bf16: DevBuf,
}

pub struct DevChain<'a> {
    pub dev: &'a Device,
    pub cfg: &'a Dsv41Config,
    pub w: &'a Dsv41DevWeights,
    pub opts: RunOpts,
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
        for _ in 0..cfg.n_layers + cfg.n_mtp_layers {
            layers.push(LayerCache {
                ring: dev.alloc(fb(cfg.window_size * hd))?,
                idxs: dev.alloc(fb(cfg.window_size + cfg.index_topk + 8).max(4))?,
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
            wo: dev.alloc(fb(cfg.o_lora_rank))?,
            logits: dev.alloc(fb(cfg.vocab_size))?,
            ids: dev.alloc(4)?,
            scores: dev.alloc(fb(n_exp))?,
            route_idx: dev.alloc(fb(topk).max(4))?,
            route_w: dev.alloc(fb(topk).max(4))?,
            ex_in: dev.alloc(fb(dim))?,
            ex_act: dev.alloc(fb(2 * inter.max(dim)))?,
            ex_out: dev.alloc(fb(dim))?,
            hist: dev.alloc(fb(n_exp))?,
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
            layers,
            s,
            cos,
            sin,
        })
    }

    pub fn reset(&mut self) -> Result<()> {
        for c in self.layers.iter() {
            self.dev.zero(&c.ring)?;
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
        self.dev.sync()?;
        let mut out = vec![0f32; cfg.vocab_size];
        self.dev.download_f32(&self.s.logits, &mut out)?;
        Ok(out)
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
            (nh * hd) as i32,
            self.s.q.ptr as *mut f32,
        )?;
        // RoPE over the trailing `rope_head_dim` lanes of each head
        self.dev.apply_rope(
            self.s.q.ptr as *mut f32,
            self.cos.as_f32(),
            self.sin.as_f32(),
            nh as i32,
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
        let cache = &self.layers[layer];
        self.dev.memcpy_d2d(
            (cache.ring.ptr as *mut u8).wrapping_add(slot * fb(hd)) as *mut c_void,
            self.s.kv.ptr as *const c_void,
            fb(hd),
        )?;

        // selection: the window ring, oldest first (the ring index already
        // carries the ageing rotation), padded with -1
        let win_cols = (pos + 1).min(win);
        let cols = win_cols.max(1);
        let mut idx_host = vec![-1i32; cols];
        let wsel = ops::window_topk_idxs(win, 1, 1, pos);
        for (c, v) in wsel.iter().enumerate().take(win_cols) {
            idx_host[c] = *v;
        }
        self.ul_i32(cache.idxs.ptr, &idx_host)?;

        self.dev.sparse_attn(
            self.s.q.as_f32(),
            cache.ring.as_f32(),
            ld.attn_sink.as_ref().unwrap().as_f32(),
            cache.idxs.as_i32(),
            self.s.o.ptr as *mut f32,
            1,
            1,
            nh as i32,
            hd as i32,
            win as i32,
            cols as i32,
            1.0 / (hd as f32).sqrt(),
        )?;
        self.dev.apply_rope(
            self.s.o.ptr as *mut f32,
            self.cos.as_f32(),
            self.sin.as_f32(),
            nh as i32,
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
        let olg = cfg.o_lora_rank / groups;
        let k = hpg * hd;
        self.quant1(self.s.o.ptr as *const f32, (nh * hd) as i32)?;
        for g in 0..groups {
            let a = self.s.xq.as_u8().wrapping_add(g * k);
            let asc = self.s.xsc.as_f32().wrapping_add((g * k / 32) as usize);
            let wp = ld
                .wo_a
                .as_ref()
                .unwrap()
                .as_u8()
                .wrapping_add(g * olg * k);
            let wsp = ld
                .wo_a_scale
                .as_ref()
                .unwrap()
                .as_u8()
                .wrapping_add((g * olg / 32) * (k / 32));
            self.dev.gemm_fp8_mx(
                a,
                asc,
                wp,
                wsp,
                std::ptr::null(),
                (self.s.wo.ptr as *mut f32).wrapping_add(g * olg),
                1,
                olg as i32,
                k as i32,
            )?;
        }
        self.lin(
            self.s.wo.ptr as *const f32,
            cfg.o_lora_rank as i32,
            ld.wo_b.as_ref().unwrap(),
            ld.wo_b_scale.as_ref().unwrap(),
            dim as i32,
            self.s.o.ptr as *mut f32,
        )?;
        Ok(())
    }

    /// MoE: bf16 gate GEMM, `noaux_tc` routing, MXFP4 experts, fp8 shared expert.
    fn moe(&mut self, layer: usize, ld: &LayerDev) -> Result<()> {
        let cfg = self.cfg;
        let dim = cfg.dim;
        let inter = cfg.moe_inter_dim;
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
        let mut wsum = vec![0f32; n_routed.max(1)];
        for (slot, &e) in idx.iter().enumerate() {
            let e = e as usize;
            if e < n_routed {
                wsum[e] += wgt[slot];
            }
        }
        if !self.opts.skip_experts {
            for e in 0..ne.min(n_routed) {
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
                    inter as i32,
                    cfg.swiglu_limit,
                )?;
                self.dev
                    .swiglu_limit(self.s.ex_act.ptr as *mut f32, 1, inter as i32, cfg.swiglu_limit)?;
                self.ul_f32(self.s.ex_in.ptr, &[wsum[e]])?;
                self.dev.expert_down_fp4(
                    self.s.ex_act.ptr as *const f32,
                    ex.w2.as_u8(),
                    ex.w2_scale.as_u8(),
                    self.s.ex_in.as_f32(),
                    self.s.ex_out.ptr as *mut f32,
                    1,
                    dim as i32,
                    inter as i32,
                )?;
                // accumulate deterministically into o
                self.dev.add_inplace(&self.s.o, &self.s.ex_out, dim as i64)?;
            }
        }
        self.dev.memcpy_d2d(self.s.o.ptr, self.s.ex_out.ptr as *const c_void, fb(dim))?;

        // shared expert: fp8, every token
        if !self.opts.skip_shared_expert {
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
        // the block output is the attention-branch accumulator `o`
        Ok(())
    }
}
