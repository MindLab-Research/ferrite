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

use ferrite_types::Result;

use std::sync::Arc;

use crate::dsv41::config::{Dsv41Config, KvMode};
use crate::dsv41::tp::Collective;
use crate::dsv41::device::{DevBuf, Device};
use crate::dsv41::load::{Dsv41DevWeights, LayerDev};

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
    /// MoE scatter destination (written by the dispatch kernels; the host never
    /// reads it back, which is why rustc flags it).
    #[allow(dead_code)]
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
    /// The device position counter: the argmax (the step's last kernel)
    /// advances it, so every kernel during the step reads a stable current pos.
    pos_ctr: DevBuf, // [1] i32
    /// Per-layer compressed-KV counters, advanced by the compressor's commit
    /// kernel on the device (the host used to track compress_len and download
    /// `out_rows` to decide). Consumers read this instead of a launch argument.
    clen: DevBuf, // [n_layers] i32
    // MoE
    scores: DevBuf,    // [n_experts] f32
    route_idx: DevBuf, // [topk] i32
    route_w: DevBuf,   // [topk] f32
    ex_in: DevBuf,     // [dim]
    ex_act: DevBuf,    // [2*inter]
    ex_out: DevBuf,    // [dim]
    /// DSV41_MOE_BATCH only: the per-slot gate/up outputs, `[topk][2*inter]`.
    /// The sequential loop reused ONE ex_act per slot (overwrite); the batched
    /// gate/up writes every slot in one launch, so it needs disjoint slices.
    ex_act_b: DevBuf,
    /// DSV41_MOE_BATCH only: the per-slot down scratch, `[topk][dim]`. The
    /// batched down WRITES here (no cross-slot accumulation) and a fixed-order
    /// reduction sums the slots into `o` in the sequential order.
    ex_down_b: DevBuf,
    hist: DevBuf,      // [n_experts] i32
    idx_q: DevBuf,     // [index_n_heads * index_head_dim]
    idx_k: DevBuf,     // [index_head_dim]
    idx_w: DevBuf,     // [index_n_heads]
    // bf16 staging for the cuBLAS path
    bf16: DevBuf,
    // hc premix coefficients, kept DEVICE-resident and rotated between layers.
    // They used to round-trip through the host every layer (one download each for
    // attn_pre and ffn_pre, plus two uploads), and a download is a full device
    // sync — 45 layers x 2 syncs per step, each one draining the CPU/GPU
    // pipeline, which is where the ~7.5ms/layer went. Three slots, no aliasing.
    pre_a: DevBuf,
    /// The constant incoming premix [1,0,0,0], uploaded ONCE at reset; each step
    /// copies it into slot 0 with a 16-byte D2D (graph-capturable) instead of an H2D.
    premix_const: DevBuf,
    pre_b: DevBuf,
    pre_c: DevBuf,
    // ---- engram (n-gram memory write-back into the hc residual stream) ----
    /// hash ids for one token: `[n_engram_layers * n_hash_cols]` i64
    eng_ids: DevBuf,
    /// gathered table rows: `[n_hash_cols * engram_head_dim]` f32 (0 for rows
    /// another rank owns, so the collective sums one real row per column)
    eng_rows: DevBuf,
    /// the wkv projection's output: `[hc*dim + dim]` f32 (key then value)
    eng_kv: DevBuf,
    /// fp8 activation + per-32 scale for the wkv GEMM over `eng_rows`
    eng_xq: DevBuf,
    eng_xsc: DevBuf,
}

/// Device-resident state for the engram n-gram hash: the compressed-token map,
/// the per-layer multipliers and the (prime, offset) columns, all uploaded ONCE,
/// plus the cross-step token cache and the position counter (zeroed at reset).
struct EngDev {
    map: DevBuf,    // [vocab] i64
    cache: DevBuf,  // [max_seq] i64
    mults: DevBuf,  // [n_layers * 4] i64
    lms: DevBuf,    // [n_layers * n_cols] u64
    offs: DevBuf,   // [n_layers * n_cols] u64
    max_seq: usize,
}

/// DSV41_ENG_HOST=1 keeps the host hash + per-step upload (the A/B fallback).
fn eng_host() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("DSV41_ENG_HOST").map(|v| v != "0").unwrap_or(false))
}

/// DSV41_MOE_BATCH=1 collapses the routed-expert fp4 GEMV family from one
/// launch per (layer, top-k slot) to one launch per (layer, direction), which
/// is the MoE family's real lever (the per-call launch floor is ~3.05 us and
/// the inner loops are measured-exhausted — see docs/agent/perf-roadmap.md).
///
/// DEFAULT OFF: the batched path is a separate kernel set and the sequential
/// path is the live verified one. Read ONCE and cached (the house rule from
/// dsv41_glue.cu's g_hc_spread and eng_host() above — a per-call getenv is a
/// hot-path slip), and `"0"` means OFF even though it is "set".
fn moe_batch() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("DSV41_MOE_BATCH").map(|v| v != "0").unwrap_or(true))
}

fn build_eng_dev(
    dev: &Device,
    lay: &crate::dsv41::engram::EngramLayout,
    map: &crate::dsv41::engram::TokenMap,
    max_seq: usize,
) -> Result<EngDev> {
    let n_cols = lay.n_hash_cols();
    let nl = lay.layers.len();
    let d_map = dev.alloc(map.map.len() * 8)?;
    dev.upload_bytes_at(
        &d_map,
        unsafe { std::slice::from_raw_parts(map.map.as_ptr() as *const u8, map.map.len() * 8) },
    )?;
    let mut mults: Vec<i64> = Vec::with_capacity(nl * 4);
    for li in 0..nl {
        mults.extend_from_slice(lay.multipliers(li));
    }
    let mut lms: Vec<u64> = Vec::with_capacity(nl * n_cols);
    let mut offs: Vec<u64> = Vec::with_capacity(nl * n_cols);
    for li in 0..nl {
        for c in 0..n_cols {
            let (lm, off) = lay.layers[li].column(c);
            lms.push(lm);
            offs.push(off);
        }
    }
    let up = |v: &[u64]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>() };
    let d_mults = dev.alloc(mults.len() * 8)?;
    dev.upload_bytes_at(
        &d_mults,
        unsafe { std::slice::from_raw_parts(mults.as_ptr() as *const u8, mults.len() * 8) },
    )?;
    let d_lms = dev.alloc(lms.len() * 8)?;
    dev.upload_bytes_at(&d_lms, &up(&lms))?;
    let d_offs = dev.alloc(offs.len() * 8)?;
    dev.upload_bytes_at(&d_offs, &up(&offs))?;
    let d_cache = dev.alloc(max_seq * 8)?;
    dev.zero_at(d_cache.ptr, max_seq * 8)?;
    Ok(EngDev {
        map: d_map,
        cache: d_cache,
        mults: d_mults,
        lms: d_lms,
        offs: d_offs,
        max_seq,
    })
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
    /// the compressor's KV uses a DIFFERENT rope theta (160000 vs 10000);
    /// without separate tables every rope call uses the main theta
    cos_comp: DevBuf,
    sin_comp: DevBuf,
    // ---- engram ----
    /// Per-layer MoE-segment graphs (only with DSV41_GRAPH_MOE=1). The segment is
    /// everything up to the all-reduce; the AR stays host-issued.
    moe_graph: Vec<Option<*mut std::ffi::c_void>>,
    /// Armed from the second step on, once every kernel is warm.
    moe_graph_armed: bool,
    /// How many steps this chain has run (the first one warms the kernels).
    step_count: u32,
    /// Diagnostics for the MoE segment graphs.
    moe_graph_captures: u32,
    moe_graph_replays: u32,
    /// n-gram hash state (host side; the token cache spans prefill + decode)
    /// The whole-step CUDA graph (captured on the first DECODE step; see step_impl).
    step_graph: Option<*mut std::ffi::c_void>,
    /// Decode-path steps only: the capture must NOT happen during prefill, because
    /// the host's launch decisions (which branches, which kernel args) can differ
    /// between prefill and decode and a capture freezes them.
    decode_steps: u32,
    /// Device-side engram hash state (built lazily on the first step).
    eng_dev: Option<EngDev>,
    ngram: Option<crate::dsv41::engram::NgramHashState>,
    eng_layout: Option<crate::dsv41::engram::EngramLayout>,
    eng_map: Option<crate::dsv41::engram::TokenMap>,
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
        map: Option<crate::dsv41::engram::TokenMap>,
    ) -> Result<Self> {
        let dim = cfg.dim;
        let hc = cfg.hc_mult;
        let hd = cfg.head_dim;
        let nh = cfg.n_heads;
        let inter = cfg.moe_inter_dim;
        let n_exp = cfg.n_routed_experts.max(1);
        let topk = cfg.n_activated_experts.max(1);
        let ql = cfg.q_lora_rank;
        // engram sizing: (max_ngram_size - 1) * n_heads hash columns, one row
        // of `engram_head_dim` each, for the layers the config lists (1 and 14).
        let n_eng_layers = cfg.engram_layer_ids.len().max(1);
        let eng_cols =
            cfg.engram_max_ngram_size.saturating_sub(1).max(1) * cfg.engram_n_heads.max(1);
        let ehd = cfg.engram_head_dim.max(1);
        let bf16_cap = dim.max(ql).max(nh * hd).max(cfg.vocab_size);
        let bf16 = dev.alloc(bf16_cap * 2)?;

        // The model's max_position_embeddings (1M) sizes NOTHING at runtime:
        // index_k alone would be 256-512 MiB per layer (~16.5 GiB over 43
        // layers), and on a 4 GB host the driver's per-allocation bookkeeping
        // for that many mappings is what actually dies (a 256 MiB cudaMalloc
        // 'fails' with 182 GB free). Caches are sized for the positions this
        // run can actually reach — DSV41_MAX_POS, default 64k.
        let max_pos = cfg
            .max_seq_len
            .min(
                std::env::var("DSV41_MAX_POS")
                    .ok()
                    .and_then(|v| v.parse::<usize>().ok())
                    .unwrap_or(65536),
            );
        let mut layers = Vec::with_capacity(cfg.n_layers + cfg.n_mtp_layers);
        for l in 0..cfg.n_layers + cfg.n_mtp_layers {
            let ratio = cfg.compress_ratio(l).max(1);
            let max_comp = max_pos / ratio + 2;
            // No indexer bound constant lives here any more: the kernel walks the
            // compressed rows in fixed-size chunks, so its shared memory depends on
            // the chunk size and index_topk only - never on this per-step count. That
            // removes the whole "`*lens` past the cap silently loses candidates"
            // envelope the constant carried.
            // The KV buffer holds the window ring FOLLOWED by the compressed
            // latents: rows [0, window) are the ring, [window, window+max_comp)
            // are the compressor's output. The earlier allocation was window
            // rows only, so the compressor's memcpy_d2d wrote past the end —
            // silent corruption at 3 layers (adjacent allocation masked it),
            // SIGSEGV in the driver at 24+.
            layers.push(LayerCache {
                ring: dev.alloc(fb((cfg.window_size + max_comp) * hd))?,
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
            pos_ctr: dev.alloc(4)?,
            clen: dev.alloc(cfg.n_layers * 4)?,
            scores: dev.alloc(fb(n_exp))?,
            route_idx: dev.alloc(fb(topk).max(4))?,
            route_w: dev.alloc(fb(topk).max(4))?,
            ex_in: dev.alloc(fb(dim))?,
            ex_act: dev.alloc(fb(2 * inter.max(dim)))?,
            ex_out: dev.alloc(fb(dim))?,
            // DSV41_MOE_BATCH scratch (allocated unconditionally: it is a few
            // hundred KB and keeps the allocation graph static). Sized by the
            // FULL `inter`, which is >= the padded local width the kernels use.
            ex_act_b: dev.alloc(fb(topk.max(1) * 2 * inter))?,
            ex_down_b: dev.alloc(fb(topk.max(1) * dim))?,
            hist: dev.alloc(fb(n_exp))?,
            idx_q: dev.alloc(fb(cfg.index_n_heads.max(1) * cfg.index_head_dim.max(1)))?,
            idx_k: dev.alloc(fb(cfg.index_head_dim.max(1)))?,
            idx_w: dev.alloc(fb(cfg.index_n_heads.max(1)))?,
            bf16,
            pre_a: dev.alloc(fb(hc).max(8))?,
            pre_b: dev.alloc(fb(hc).max(8))?,
            pre_c: dev.alloc(fb(hc).max(8))?,
            premix_const: dev.alloc(fb(hc).max(8))?,
            eng_ids: dev.alloc(fb(eng_cols * n_eng_layers).max(8) * 2)?, // i64
            eng_rows: dev.alloc(fb(eng_cols * ehd).max(8))?,
            eng_kv: dev.alloc(fb((hc + 1) * dim))?,
            eng_xq: dev.alloc((eng_cols * ehd).max(8))?, // fp8 bytes
            eng_xsc: dev.alloc(fb((eng_cols * ehd).max(8) / 32 + 8))?,
        };

        // RoPE tables covering the whole context.
        let table = max_pos;
        let half = cfg.rope_head_dim / 2;
        let cos = dev.alloc(fb(table * half))?;
        let sin = dev.alloc(fb(table * half))?;
        let cos_comp = dev.alloc(fb(table * half))?;
        let sin_comp = dev.alloc(fb(table * half))?;
        // main rope (theta=10000) for the query and window KV
        dev.rope_precompute(
            cos.ptr as *mut f32, sin.ptr as *mut f32,
            cfg.rope_head_dim as i32, table as i32,
            cfg.original_seq_len as i32, cfg.rope_theta,
            cfg.rope_factor, cfg.beta_fast, cfg.beta_slow,
        )?;
        // compressor rope (theta=160000) for the compressed latent
        dev.rope_precompute(
            cos_comp.ptr as *mut f32, sin_comp.ptr as *mut f32,
            cfg.rope_head_dim as i32, table as i32,
            cfg.original_seq_len as i32, cfg.compress_rope_theta,
            cfg.rope_factor, cfg.beta_fast, cfg.beta_slow,
        )?;
        let _ = bf16_cap;

        // engram host-side state: the hash needs the compressed token map (a pure
        // function of the tokenizer, precomputed) and keeps a token cache that
        // spans prefill + decode.
        let eng_layout = crate::dsv41::engram::EngramLayout::from_config(cfg);
        let (ngram, eng_layout, eng_map) = match (eng_layout, map) {
            (Some(lay), Some(m)) => {
                let st = crate::dsv41::engram::NgramHashState::new(cfg, &m);
                (Some(st), Some(lay), Some(m))
            }
            _ => (None, None, None),
        };

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
            cos_comp,
            sin_comp,
            moe_graph: vec![None; cfg.n_layers],
            moe_graph_armed: false,
            step_count: 0,
            moe_graph_captures: 0,
            moe_graph_replays: 0,
            step_graph: None,
            decode_steps: 0,
            eng_dev: None,
            ngram,
            eng_layout,
            eng_map,
        })
    }

    pub fn reset(&mut self) -> Result<()> {
        // the incoming premix is the CONSTANT [1,0,0,0] every step; upload it once
        // here so the per-step refresh is a device-to-device copy (no H2D)
        {
            let mut pm = vec![0f32; self.cfg.hc_mult];
            pm[0] = 1.0;
            self.dev.upload_f32_at(self.s.premix_const.ptr, 0, &pm)?;
        }
        self.dev.zero_at(self.s.pos_ctr.ptr, 4)?;
        self.dev.zero_at(self.s.clen.ptr, self.cfg.n_layers * 4)?;
        // The capture must be re-armed PER REQUEST, and the captured graph must be
        // DROPPED per request. Two separate reasons, both measured:
        //  1. decode_steps carried over let the next request's PREFILL satisfy
        //     `decode_steps >= 1` and run through the decode graph (captured with
        //     decode-time host branch choices - the compressor's mode 2 instead of
        //     mode 1 at pos 0), which garbled the output.
        //  2. A graph bakes the DEVICE ADDRESSES of the buffers it recorded. That
        //     was safe while allocations came from the pool, whose whole contract
        //     is that a given size class returns the same address; the shared
        //     devrt allocator is byte-level and does not promise that, so reusing a
        //     graph from a previous request replays against addresses that may
        //     belong to something else now (measured: requests 1-3 fine, then
        //     one-step/empty outputs and faults). Dropping it costs a re-capture
        //     per request (a few ms) and makes the graph always match the state it
        //     was recorded from.
        if let Some(e) = self.step_graph.take() {
            // e is the graph EXEC (graph_instantiate's result); the captured
            // graph handle itself was already released right after instantiate.
            // graph_free destroys the exec when the second argument is non-null.
            self.dev.graph_free(std::ptr::null_mut(), e)?;
        }
        self.decode_steps = 0;
        if let Some(e) = self.eng_dev.as_ref() {
            self.dev.zero_at(e.cache.ptr, e.max_seq * 8)?;
        }
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
    fn lin(&self, a: *const f32, k: i32, w: &crate::dsv41::load::DevTensor, ws: &crate::dsv41::load::DevTensor, n_out: i32, out: *mut f32) -> Result<()> {
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

    /// f32 linear for one row. M=1 goes through our own GEMV: cuBLAS's GemmEx
    /// picked gemv2T at ~40 GFLOP/s for a single row (396us/call, 72 per step)
    /// while the weight read floors at ~112us.
    fn lin_f32(&self, a: *const f32, k: i32, w: &crate::dsv41::load::DevTensor, n_out: i32, out: *mut f32) -> Result<()> {
        if std::env::var("DSV41_CUBLAS_M1").map(|v| v != "0").unwrap_or(false) {
            return self
                .dev
                .gemm_f32(a as *const c_void, w.ptr() as *const c_void, out, 1, n_out, k);
        }
        self.dev.gemv_f32(w.ptr() as *const f32, a, out, n_out, k)
    }

    /// bf16 linear for one row. The bf16 path converts the *activation* to bf16
    /// and hands both to cuBLAS; our GEMV takes the activation in f32 and the
    /// weights natively bf16, which is both leaner and slightly more accurate
    /// (no activation rounding).
    fn lin_bf16(&self, a: *const f32, k: i32, w: &crate::dsv41::load::DevTensor, n_out: i32, out: *mut f32) -> Result<()> {
        if std::env::var("DSV41_CUBLAS_M1").map(|v| v != "0").unwrap_or(false) {
            self.dev.f32_to_bf16(a, self.s.bf16.ptr as *mut c_void, k as i64)?;
            return self
                .dev
                .gemm_bf16(self.s.bf16.ptr as *const c_void, w.ptr() as *const c_void, out, 1, n_out, k);
        }
        self.dev.gemv_bf16(w.ptr() as *const c_void, a, out, n_out, k)
    }

    /// Engram: n-gram memory write-back into the hc residual stream, applied
    /// BEFORE the block at the layers the config lists (1 and 14). Mirrors the
    /// reference's `Engram.forward`: gather the `n_cols` hash rows from the
    /// (row-sharded) table, project them with `wkv` into one key per hc copy
    /// plus a shared value, gate that value by the normalised dot of the stream
    /// against the key, and add it to every copy.
    fn engram_apply(&mut self, layer: usize, li: usize) -> Result<()> {
        let cfg = self.cfg;
        let rank = self.rank();
        let (dim, hc, ehd) = (cfg.dim, cfg.hc_mult, cfg.engram_head_dim);
        let n_cols = cfg.engram_max_ngram_size.saturating_sub(1) * cfg.engram_n_heads;
        let (table, tsc, wkv, wsc, qw, kw) = {
            let ld = &self.w.layers[layer];
            (
                ld.engram_embed.as_ref(),
                ld.engram_embed_scale.as_ref(),
                ld.engram_wkv.as_ref(),
                ld.engram_wkv_scale.as_ref(),
                ld.engram_q_weight.as_ref(),
                ld.engram_k_weight.as_ref(),
            )
        };
        let (Some(table), Some(tsc), Some(wkv), Some(wsc), Some(qw), Some(kw)) =
            (table, tsc, wkv, wsc, qw, kw)
        else {
            return Ok(());
        };
        // This rank's slice of the row-parallel table: convert.py shards
        // `ceil(rows / world)` rows and zero-pads the tail.
        let world = self.world().max(1);
        let global_rows = cfg.engram_num_embeddings.get(li).copied().unwrap_or(0) as usize;
        let per = global_rows.div_ceil(world);
        let ids = (self.s.eng_ids.ptr as *const i64).wrapping_add(li * n_cols);
        self.dev.engram_gather(
            table.ptr() as *const u8,
            tsc.ptr() as *const u8,
            ids,
            self.s.eng_rows.ptr as *mut f32,
            1,
            n_cols as i32,
            ehd as i32,
            (rank * per) as i64,
            per as i64,
        )?;
        // rows another rank owns arrived as 0, so the sum yields the real row
        if let Some(c) = self.comm.clone() {
            c.all_reduce_inplace(self.s.eng_rows.ptr as *mut std::ffi::c_void, fb(n_cols * ehd))?;
        }
        // kv = wkv(gathered): [(hc + 1) * dim] = [key(hc*dim), value(dim)]
        self.dev.quant_fp8(
            self.s.eng_rows.ptr as *const f32,
            self.s.eng_xq.ptr as *mut u8,
            self.s.eng_xsc.ptr as *mut f32,
            1,
            (n_cols * ehd) as i32,
            32,
            true,
        )?;
        self.dev.gemm_fp8_mx(
            self.s.eng_xq.ptr as *const u8,
            self.s.eng_xsc.ptr as *const f32,
            wkv.ptr() as *const u8,
            wsc.ptr() as *const u8,
            std::ptr::null(),
            self.s.eng_kv.ptr as *mut f32,
            1,
            ((hc + 1) * dim) as i32,
            (n_cols * ehd) as i32,
        )?;
        // gated write-back into h (in place)
        self.dev.engram_apply(
            self.s.h.ptr as *mut f32,
            self.s.eng_kv.ptr as *const f32,
            qw.ptr() as *const f32,
            kw.ptr() as *const f32,
            std::ptr::null(),
            1,
            hc as i32,
            dim as i32,
            cfg.norm_eps,
        )?;
        Ok(())
    }

    /// The MoE's all-reduce, issued OUTSIDE any captured segment: a CUDA graph
    /// cannot contain the host barrier that this path still uses, so the segment
    /// boundary sits exactly here.
    fn moe_reduce(&mut self) -> Result<()> {
        let dim = self.cfg.dim;
        // routed experts are expert-parallel, so each rank holds a partial sum
        if let Some(c) = self.comm.clone() {
            c.all_reduce_inplace(self.s.o.ptr as *mut std::ffi::c_void, fb(dim))?;
            c.end_round();
        }
        Ok(())
    }

    /// One decode step. Returns the logits for the fed token.
    pub fn step(&mut self, token: u32, pos: usize) -> Result<u32> {
        let ids = [token as i32];
        self.dev.upload_f32_at(self.s.ids.ptr, 0, unsafe {
            std::slice::from_raw_parts(ids.as_ptr() as *const f32, 1)
        })?;
        self.step_impl(token, pos)
    }

    /// Decode steady state: the token is ALREADY in s.ids on the device (the
    /// previous step's argmax wrote it), and the caller knows its value because
    /// that step returned it — the host value feeds only the n-gram hash, so this
    /// path does ZERO host-to-device traffic.
    pub fn step_dev(&mut self, token: u32, pos: usize) -> Result<u32> {
        self.decode_steps = self.decode_steps.wrapping_add(1);
        self.step_impl(token, pos)
    }

    /// The whole decode step as ONE CUDA graph (DSV41_GRAPH_STEP=1, the default):
    /// every per-step value now lives on the DEVICE - the position counter, the
    /// per-layer latent counters, the all-reduce epoch - so no launch argument
    /// changes from token to token and the capture is legal. The capture happens
    /// on the second step: the first warms every kernel, builds the lazy device
    /// state and sizes cublas' workspaces, all of which are illegal inside a
    /// capture. Capturing records WITHOUT executing, so the graph is launched
    /// straight afterwards to do this step's real work.
    fn step_impl(&mut self, token: u32, pos: usize) -> Result<u32> {
        let cfg = self.cfg;
        static GRAPH_STEP: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let want = *GRAPH_STEP.get_or_init(|| {
            // Things that must be OFF for a whole-step capture, each because it
            // makes a host round trip inside the recorded region:
            //   DSV41_ENG_HOST  - the host n-gram hash uploads the ids per step
            //   DSV41_STATS     - the per-layer probes download tensors
            // And one thing that must be ON: the all-reduce has to be the DEVICE
            // side AR v5, because a host barrier is not a CUDA call - it would not
            // be recorded, and the replayed graph would lose the inter-rank sync.
            // ar_v5() therefore also turns on with the graph (see tp.rs).
            let host_hash = std::env::var("DSV41_ENG_HOST").map(|v| v != "0").unwrap_or(false);
            let probes = std::env::var("DSV41_STATS").map(|v| v != "0").unwrap_or(false);
            // VERIFIED: a SINGLE request with the graph on is bit-identical to the
            // per-kernel path (DSV41_TOKTRACE compared step by step, no divergence),
            // so the captured operator set is right. With SEVERAL sequential
            // requests it is not: requests one to three (short, one step each, so
            // they never reach the capture) answer correctly, and the first long
            // request then dies with an illegal memory access on a rank that varies
            // run to run, after which the engine is sticky-faulted and every later
            // request is empty. The regression and the fault were chased through the
            // whole device layer (capture mode, streams, cublas stream, graph
            // wrappers, buffer accessors, the 52 launch wrappers - all byte-equal to
            // the pre-refactor file), through a per-request graph drop, and through a
            // rank rendezvous around the capture; none of them changed the outcome,
            // SO THE DEFAULT IS NOW THE GRAPH (2026-09-11): the cause WAS located - the indexer
            // sized its dynamic shared memory from a capture-frozen per-step value while
            // scanning the live counter - and fixed by decoupling the size from the count,
            // then verified with 12/12 answers and zero faults. DSV41_GRAPH_STEP=0 opts out.
            !host_hash
                && !probes
                && std::env::var("DSV41_GRAPH_STEP").map(|v| v != "0").unwrap_or(true)
        });
        // ONLY on the decode path: capturing during prefill froze the prefill
        // branches into the graph, so the decode replays took the wrong ones (the
        // observable symptom was output that looked like a plausible continuation of
        // something else - the model was being fed a mis-processed prompt).
        if want && self.decode_steps >= 1 {
            // Rendezvous with the peers BEFORE the branch. A capture only RECORDS
            // its all-reduce kernels, while the device-side AR has no host barrier
            // of its own (end_round returns early under ar_v5, since the publish
            // chain covers the normal case). Without this the ranks keep their
            // microsecond-level skew, and a peer that is EXECUTING its AR polls for
            // a stamp that a still-recording rank is only writing down, gives up,
            // and reads staging that was never published. Measured: with the graph
            // on, one request was bit-identical to the per-kernel path, yet the
            // fourth of six sequential requests died with an illegal access on a
            // rank that varied run to run - a race, not a deterministic fault.
            if let Some(c) = self.comm.as_ref() {
                c.host_barrier();
            }
            if let Some(e) = self.step_graph {
                self.dev.graph_launch(e)?;
            } else {
                self.dev.capture_begin()?;
                self.step_body(token, pos)?;
                let g = self.dev.capture_end()?;
                // The recording is finished; align again so no rank starts
                // replaying (and thus publishing) while a peer is still capturing.
                if let Some(c) = self.comm.as_ref() {
                    c.host_barrier();
                }
                let e = self.dev.graph_instantiate(g)?;
                self.dev.graph_free(g, std::ptr::null_mut())?;
                self.dev.graph_launch(e)?; // the capture did not execute
                self.step_graph = Some(e);
            }
        } else {
            self.step_body(token, pos)?;
        }
        self.step_count = self.step_count.wrapping_add(1);
        // the ONLY host read on the decode path: 4 bytes for EOS and printing
        self.dev.sync()?;
        let tok = self.dev.download_u32(self.s.ids.ptr)?;
        // Token trace for bisecting the graph's cumulative error. It runs OUTSIDE
        // the captured region on purpose (the token is already on the host here),
        // so DSV41_TOKTRACE=1 changes nothing about the capture and the two modes'
        // token sequences can be compared step by step.
        if std::env::var("DSV41_TOKTRACE").map(|v| v != "0").unwrap_or(false) {
            eprintln!(
                "[toktr] ds={} pos={} tok={}",
                self.decode_steps, self.step_count, tok
            );
        }
        if std::env::var("DSV41_TOP5").map(|v| v != "0").unwrap_or(false) {
            let mut lg = vec![0f32; cfg.vocab_size];
            let b = Device::view(self.s.logits.ptr, cfg.vocab_size * 4);
            self.dev.download_f32(&b, &mut lg)?;
            let mut top: Vec<(usize, f32)> = lg.iter().copied().enumerate().collect();
            top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            eprintln!("[top5] n={} top={:?}", lg.len(), &top[..5.min(top.len())]);
        }
        Ok(tok)
    }

    /// The step's kernels with NO host round trip in between: embedding through
    /// the argmax (which is also what advances the position counter). This is the
    /// region a graph captures; everything host-side lives in step_impl.
    fn step_body(&mut self, token: u32, pos: usize) -> Result<()> {
        let cfg = self.cfg;
        let dim = cfg.dim;
        let hc = cfg.hc_mult;
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

        // probe: h right after embedding + hc expansion
        if std::env::var("DSV41_HCDBG").map(|v| v != "0").unwrap_or(false) {
            let hv = self.dl(self.s.h.as_f32(), hc * dim)?;
            let r = (hv.iter().map(|v| v * v).sum::<f32>() / hv.len() as f32).sqrt();
            eprintln!("[mine] xin_rms={}", (r * 1e6).round() / 1e6);
        }

        // the initial collapse takes copy 0 of the stream: the constant [1,0,0,0]
        // lives in a device buffer uploaded once at reset; this 16-byte D2D is
        // graph-capturable where the old per-step H2D upload was not.
        self.dev.memcpy_d2d(
            self.s.pre_a.ptr,
            self.s.premix_const.ptr,
            hc * std::mem::size_of::<f32>(),
        )?;
        let mut premix_slot_idx = 0usize;
        // engram hashes for this token (all engram layers at once; the reference
        // computes them in one shot and indexes per layer). The state's token
        // cache spans prefill + decode, so this must be called every step.
        let mut eng_layer_of: Vec<(usize, usize)> = Vec::new();
        if let (Some(ng), Some(lay), Some(map)) =
            (self.ngram.as_mut(), self.eng_layout.as_ref(), self.eng_map.as_ref())
        {
            if eng_host() {
                let hs = ng.forward_row(lay, map, 0, &[token], pos, None);
                let n_cols = lay.n_hash_cols();
                if hs.len() >= lay.layers.len() * n_cols {
                    let mut bytes = Vec::with_capacity(hs.len() * 8);
                    for v in hs.iter() {
                        bytes.extend_from_slice(&v.to_le_bytes());
                    }
                    self.dev
                        .upload_bytes_at(&self.s.eng_ids, &bytes[..lay.layers.len() * n_cols * 8])?;
                }
            } else {
                // DEVICE hash: the token is read straight from s.ids (the buffer
                // the previous step's argmax wrote, or the prefill upload) and
                // the position counter lives on the device — no H2D, and the
                // call is graph-capturable. Bit-identical to the host reference
                // (the same serial arithmetic).
                if self.eng_dev.is_none() {
                    let e = build_eng_dev(&self.dev, lay, map, ng.max_seq)?;
                    self.eng_dev = Some(e);
                }
                let e = self.eng_dev.as_ref().unwrap();
                self.dev.engram_hash_step(
                    e.map.ptr as *const i64,
                    e.cache.ptr as *mut i64,
                    e.mults.ptr as *const i64,
                    e.lms.ptr as *const u64,
                    e.offs.ptr as *const u64,
                    self.s.eng_ids.ptr as *mut i64,
                    self.s.ids.as_i32(),
                    self.s.pos_ctr.ptr as *const i32,
                    map.map.len() as i64,
                    lay.layers.len() as i32,
                    lay.max_ngram_size as i32,
                    lay.n_heads as i32,
                    ng.pad_id,
                )?;
            }
            for l in 0..cfg.n_layers {
                if let Some(li) = lay.engram_index(l) {
                    eng_layer_of.push((l, li));
                }
            }
        }
        let mut t_attn = std::time::Duration::ZERO;
        let mut t_moe = std::time::Duration::ZERO;
        for layer in 0..cfg.n_layers {
            // the engram writes into the residual stream BEFORE the block runs
            if let Some(&(_, li)) = eng_layer_of.iter().find(|(l, _)| *l == layer) {
                self.engram_apply(layer, li)?;
            }
            let _ta = std::time::Instant::now();
            premix_slot_idx = self.layer(layer, pos, premix_slot_idx)?;
            let _el = _ta.elapsed();
            if std::env::var("DSV41_PHASE").map(|v| v != "0").unwrap_or(false) {
                let _ = (&mut t_attn, &mut t_moe);
                eprintln!("[phase] L{layer} layer={:?}", _el);
            }
            if std::env::var("DSV41_STATS").map(|v| v != "0").unwrap_or(false)
                && layer % std::env::var("DSV41_STATS_EVERY").ok().and_then(|v| v.parse().ok()).unwrap_or(5) == 0
            {
                self.stats(&format!("L{layer} h"), &self.s.h, hc * dim)?;
            }
        }
        self.dev.memcpy_d2d(
            self.s.pre_a.ptr,
            self.s.premix_const.ptr,
            hc * std::mem::size_of::<f32>(),
        )?;
        self.dev.hc_collapse(
            self.s.h.ptr as *const f32,
            self.premix_slot(1).as_f32(), // attn_pre stays on the device
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
        // The head keeps f32 activations × f32 weights (the checkpoint stores BF16;
        // it is widened losslessly at load). Casting the activation to bf16 here cost
        // ~3 bits on a 129280-way near-tie argmax — the reference keeps it in f32.
        self.lin_f32(
            self.s.xn.ptr as *const f32,
            dim as i32,
            self.w.head.as_ref().unwrap(),
            cfg.vocab_size as i32,
            self.s.logits.ptr as *mut f32,
        )?;
        self.stats("final logits", &self.s.logits, cfg.vocab_size)?;
        // Device-side argmax (the GLM HEAD_DEV pattern): the next token lands
        // straight in s.ids, which the next step's embedding reads — no 517 KB
        // full-vocab download, no O(vocab) host scan, and the token itself never
        // crosses to the host and back.
        self.dev.argmax(
            self.s.logits.ptr as *const f32,
            self.s.ids.ptr as *mut std::ffi::c_int,
            cfg.vocab_size as i32,
            self.s.pos_ctr.ptr as *mut std::ffi::c_int,
        )?;
        Ok(())
    }

    /// Tensor-parallel degree / this rank's index (1 / 0 without a collective).
    fn world(&self) -> usize {
        self.comm.as_ref().map(|c| c.world).unwrap_or(1)
    }
    fn rank(&self) -> usize {
        self.comm.as_ref().map(|c| c.rank).unwrap_or(0)
    }

    /// The three device-resident premix slots: 0 is the incoming premix the
    /// attention collapses with, 1 receives this layer's attn_pre, 2 receives the
    /// ffn_pre that the next layer uses.
    fn premix_slot(&self, i: usize) -> &DevBuf {
        match i % 3 {
            0 => &self.s.pre_a,
            1 => &self.s.pre_b,
            _ => &self.s.pre_c,
        }
    }


    fn dl(&self, src: *const f32, n: usize) -> Result<Vec<f32>> {
        let mut v = vec![0f32; n];
        let b = Device::view(src as *mut c_void, n * 4);
        self.dev.download_f32(&b, &mut v)?;
        Ok(v)
    }

    /// Host-upload helper (kept: the prefill and probe paths use this family).
    #[allow(dead_code)]
    fn ul_i32(&self, dst: *mut c_void, v: &[i32]) -> Result<()> {
        self.dev.upload_f32_at(dst, 0, unsafe {
            std::slice::from_raw_parts(v.as_ptr() as *const f32, v.len())
        })
    }

    #[allow(dead_code)]
    fn ul_f32(&self, dst: *mut c_void, v: &[f32]) -> Result<()> {
        self.dev.upload_f32_at(dst, 0, v)
    }

    /// One transformer layer. Returns the pre-mix the next block's first
    /// collapse must use (this block's *attention* mix).
    /// `pa` indexes the DEVICE-resident premix the attention collapses with; the
    /// return value indexes the one the next layer must use. Nothing here touches
    /// the host: the coefficients used to be downloaded and re-uploaded every
    /// layer, and a download is a full device sync.
/// Segment C fusion: hc_post written straight back onto the residual stream, which
/// drops the h2 staging buffer and its device-to-device copy. Read once, because
/// the hot path must never touch the environment per call.
fn fuse_c() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("DSV41_FUSE_C").map(|v| v != "0").unwrap_or(false))
}

    fn layer(&mut self, layer: usize, pos: usize, pa: usize) -> Result<usize> {
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
            self.premix_slot(1).ptr as *mut f32, // attn_pre
            self.s.post.ptr as *mut f32,
            self.s.comb.ptr as *mut f32,
            1,
            (hc * dim) as i32,
            hc as i32,
            cfg.hc_sinkhorn_iters as i32,
            cfg.hc_eps,
        )?;
        let _t_all = std::time::Instant::now();
        if layer == 0 && std::env::var("DSV41_HCDBG").map(|v| v != "0").unwrap_or(false) {
            let attn_pre = self.dl(self.premix_slot(1).as_f32(), hc)?;
            let po = self.dl(self.s.post.as_f32(), hc)?;
            let cb = self.dl(self.s.comb.as_f32(), hc * hc)?;
            eprintln!("[mine] L0 pre={attn_pre:?}");
            eprintln!("[mine] L0 post={po:?}");
            eprintln!("[mine] L0 comb={cb:?}");
            let rs: Vec<f32> = (0..hc)
                .map(|j| (0..hc).map(|k| cb[j * hc + k]).sum())
                .collect();
            eprintln!("[mine] L0 comb_rowsum={rs:?}");
        }
        self.dev.hc_collapse(
            self.s.h.ptr as *const f32,
            self.premix_slot(pa).as_f32(),
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
        if Self::fuse_c() {
            self.dev.hc_post_inplace(
                self.s.h.ptr as *mut f32,
                self.s.o.ptr as *const f32,
                self.s.post.as_f32(),
                self.s.comb.as_f32(),
                hc as i32,
                dim as i32,
            )?;
        } else {
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
        }

        if std::env::var("DSV41_PHASE").map(|v| v != "0").unwrap_or(false) {
            eprintln!("[phs] L{layer} attn={:?}", _t_all.elapsed());
        }
        let _t_moe = std::time::Instant::now();
        // ---------------- FFN block ----------------
        self.dev.hc_mixes(
            self.s.h.ptr as *const f32,
            ld.hc_ffn_fn.as_ref().unwrap().as_f32(),
            ld.hc_ffn_scale.as_ref().unwrap().as_f32(),
            ld.hc_ffn_base.as_ref().unwrap().as_f32(),
            self.premix_slot(2).ptr as *mut f32, // ffn_pre -> next layer
            self.s.post.ptr as *mut f32,
            self.s.comb.ptr as *mut f32,
            1,
            (hc * dim) as i32,
            hc as i32,
            cfg.hc_sinkhorn_iters as i32,
            cfg.hc_eps,
        )?;
        if std::env::var("DSV41_PHASE").map(|v| v != "0").unwrap_or(false) {
            eprintln!("[phs] L{layer} ffn={:?}", _t_moe.elapsed());
        }
        // the FFN collapses with THIS layer's attn_pre (slot 1), which stayed on
        // the device; the FFN's own pre (slot 2) is what the NEXT layer uses.
        self.dev.hc_collapse(
            self.s.h.ptr as *const f32,
            self.premix_slot(1).as_f32(),
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
        let _t_moeonly = std::time::Instant::now();
        // DSV41_GRAPH_MOE=1 captures the host-free part of the MoE (everything up
        // to the all-reduce) into one per-layer graph. The first step warms every
        // kernel; the capture happens on the next one, then it replays.
        if self.moe_graph_armed {
            if let Some(e) = self.moe_graph.get(layer).and_then(|x| *x) {
                self.moe_graph_replays = self.moe_graph_replays.wrapping_add(1);
                if self.moe_graph_replays == 1 {
                    eprintln!("[gmo] first MoE segment replay at L{layer} (captures={})",
                        self.moe_graph_captures);
                }
                self.dev.graph_launch(e)?;
            } else {
                self.dev.capture_begin()?;
                self.moe(layer, ld)?;
                let g = self.dev.capture_end()?;
                let e = self.dev.graph_instantiate(g)?;
                self.dev.graph_free(g, std::ptr::null_mut())?;
                self.moe_graph[layer] = Some(e);
                self.moe_graph_captures = self.moe_graph_captures.wrapping_add(1);
                if self.moe_graph_captures == 1 {
                    eprintln!("[gmo] MoE segment graph armed: first capture at L{layer}");
                }
            }
        } else {
            self.moe(layer, ld)?;
        }
        self.moe_reduce()?;
        if std::env::var("DSV41_PHASE").map(|v| v != "0").unwrap_or(false) {
            eprintln!("[phs] L{layer} moe={:?}", _t_moeonly.elapsed());
        }
        if Self::fuse_c() {
            self.dev.hc_post_inplace(
                self.s.h.ptr as *mut f32,
                self.s.o.ptr as *const f32,
                self.s.post.as_f32(),
                self.s.comb.as_f32(),
                hc as i32,
                dim as i32,
            )?;
        } else {
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
        }
        if std::env::var("DSV41_PHASE").map(|v| v != "0").unwrap_or(false) {
            eprintln!("[phs] L{layer} ffn_total={:?}", _t_moe.elapsed());
        }
        Ok(2) // slot 2 holds this layer's ffn_pre = the next layer's premix
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
        // ALL heads of this token are at the SAME position — step=0. The
        // earlier step=1 gave head i position pos+i (8 different positions for
        // 8 local heads), scrambling the positional encoding: every head's
        // RoPE rotated differently, so the attention scores were positionally
        // wrong. (The KV rope uses rows=1 so step is irrelevant there.)
        self.dev.apply_rope(
            self.s.q.ptr as *mut f32,
            self.cos.as_f32(),
            self.sin.as_f32(),
            nlh as i32,
            hd as i32,
            cfg.rope_head_dim as i32,
            (cfg.rope_head_dim / 2) as i32,
            self.s.pos_ctr.ptr as *const std::os::raw::c_int, 1, 0,
            0,
            false,
        )?;

        if layer == 0 && std::env::var("DSV41_HCDBG").map(|v| v != "0").unwrap_or(false) {
            let q = self.dl(self.s.q.as_f32(), nlh * hd)?;
            let r = (q.iter().map(|v| v * v).sum::<f32>() / q.len() as f32).sqrt();
            eprintln!("[mine] L0 q[0..4]={:?} q_full_rms={}", &q[..4], r);
        }
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
            self.s.pos_ctr.ptr as *const std::os::raw::c_int, 1, 0,
            1,
            false,
        )?;

        if layer == 0 && std::env::var("DSV41_HCDBG").map(|v| v != "0").unwrap_or(false) {
            let kv = self.dl(self.s.kv.as_f32(), hd)?;
            let r = (kv.iter().map(|v| v * v).sum::<f32>() / kv.len() as f32).sqrt();
            eprintln!("[mine] L0 kv[0..4]={:?} kv_rms={}", &kv[..4], r);
        }
        let win = cfg.window_size;
        // raw pointers rather than a live borrow: `compress` below needs
        // &mut self (it updates this layer's published count and buffers)
        // The release shares one KV store across a group of layers: the kv
        // source maintains it and its consumers read it. A consumer therefore
        // must not keep its own window ring (nothing would ever put the
        // compressed rows there), it reads the owner's — which also already
        // holds this step's token, since the owner runs earlier in the stack.
        // The window KV is PER-LAYER: the reference computes `_window_kv(x, ...)`
        // with each layer's own wkv and its own input, and only the *compressed*
        // KV plus the indexer are shared group-wide ("layers sharing a ratio also
        // share one compressed KV and one indexer"). Reading the owner's ring for
        // the window part fed every consumer layer the owner's kv — which is
        // exactly why layer 2 (the owner) matched the official while layer 3, the
        // first consumer, dropped ~15%. DSV41_RING_OWNER=1 restores the old
        // shared-ring behaviour for A/B.
        let owner = if std::env::var("DSV41_RING_OWNER").map(|v| v != "0").unwrap_or(false) {
            self.kv_owner(layer)
        } else {
            layer
        };
        let ring_ptr = self.layers[owner].ring.ptr;
        // index-source layers compute their OWN selection into their OWN buffer;
        // non-index layers read the owner's (shared) selection
        let idxs_ptr = if cfg.is_index_source(layer) {
            self.layers[layer].idxs.ptr
        } else {
            self.layers[owner].idxs.ptr
        };
        let cache = &self.layers[owner];
        let owns_kv = owner == layer;
        if owns_kv {
            // DEVICE-side slot: a host-computed destination address would be frozen
            // by the graph capture (slot = pos % win at capture time), so every
            // replay wrote the same ring row and the window went stale.
            self.dev.ring_append(
                cache.ring.ptr as *mut f32,
                self.s.kv.ptr as *const f32,
                self.s.pos_ctr.ptr as *const std::os::raw::c_int,
                win as i32,
                hd as i32,
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
        // The window indices are computed ON THE DEVICE from the position counter
        // (the decode branch of ops::window_topk_idxs, verbatim) - no host
        // compute and no per-layer H2D upload any more.
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
        // ALWAYS upload the window entries: the indexer overwrites the
        // compressed block on the device, but the window block [0, win) must be
        // fresh on every step. Only the placeholder branch uploaded them before,
        // so the index-source path read stale indices — the illegal memory
        // access in sparse_attn.
        self.dev
            .window_idxs(idxs_ptr as *mut i32, self.s.pos_ctr.ptr as *const i32, win as i32)?;
        if comp_len > 0 && cfg.is_index_source(layer) {
            // EVERY index-source layer runs its own indexer (into its own
            // buffer) — the reference creates one for each, and non-source
            // layers compute their own queries/selection from the keys the
            // kv-source published. Only the KEY PUBLISHING is the source's job.
            if self.indexer(layer, pos, win, comp_len)? {
            // the kernel wrote `comp_len.min(index_topk)` entries at [win, ..)
            }
        } else if !owns_kv && comp_len > 0 {
            // a non-index consumer reads the owner's selection, which the owner
            // (an index-source) filled earlier this step — do nothing
        } else if comp_len > 0 {
            // the owner has no indexer: recency placeholder (safety net)
            self.dev.comp_placeholder(
                idxs_ptr as *mut i32,
                (self.s.clen.ptr as *const std::os::raw::c_int).wrapping_add(owner),
                win as i32,
                cfg.index_topk as i32,
            )?;
        }

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
            (self.s.clen.ptr as *const std::os::raw::c_int).wrapping_add(owner),
            win as i32,
            cfg.index_topk as i32,
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
            self.s.pos_ctr.ptr as *const std::os::raw::c_int, 1, 0,
            0,
            true,
        )?;

        if layer == 0 && std::env::var("DSV41_HCDBG").map(|v| v != "0").unwrap_or(false) {
            let o = self.dl(self.s.o.as_f32(), nlh * hd)?;
            let r = (o.iter().map(|v| v * v).sum::<f32>() / o.len() as f32).sqrt();
            eprintln!("[mine] L0 sparse_o[0..4]={:?} sparse_rms={}", &o[..4], r);
        }
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
            // The weight tensor is ALREADY the rank's local slice (Shard::Groups
            // cut it at load time), so every offset must be LOCAL: group g of
            // this rank's block sits at local row g*olg. The earlier version
            // indexed with the GLOBAL group number (rank*nlg+g), which walks off
            // the end of the local buffer for any rank but 0 — the illegal
            // memory access in gemm_fp8_mx at tp=8.
            let a = self.s.xq.as_u8().wrapping_add(g * k);
            let asc = self.s.xsc.as_f32().wrapping_add((g * k / 32) as usize);
            let wp = ld.wo_a.as_ref().unwrap().as_u8().wrapping_add(g * olg * k);
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
        // wo_b is RowParallel: the input (groups*o_lora) is split, so this rank
        // reduces over its own slice and the ranks' partial sums are added.
        let ol_total = groups * cfg.o_lora_rank;
        let ol_local = ol_total / world;
        // this rank wrote its groups at local offsets [0, nlg*olg) = [0, ol_local)
        self.lin(
            self.s.wo.ptr as *const f32,
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
    fn indexer(&mut self, layer: usize, _pos: usize, offset: usize, comp_len: usize) -> Result<bool> {
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
        // Only kv-source layers publish index keys (they own the compressor's
        // latent). Non-source index layers compute their own queries and
        // selection from the keys their source already published.
        let owns_k = cfg.indexer_owns_k(layer);
        let group = if owns_k {
            self.layers[layer].compress_len.saturating_sub(1)
        } else {
            0
        };
        if owns_k {
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
            (self.s.clen.ptr as *const std::os::raw::c_int).wrapping_add(layer),
            ratio as i32,
            -(ratio as i32),
            1,
            false,
        )?;
        // DEVICE-derived destination: a host-computed group slot here is exactly
        // the frozen-address bug that made the window ring go stale under the
        // graph (the index key would land in the same group every replay).
        self.dev.index_k_publish(
            self.layers[layer].index_k.ptr as *mut f32,
            self.s.idx_k.ptr as *const f32,
            (self.s.clen.ptr as *const std::os::raw::c_int).wrapping_add(layer),
            idx_hd as i32,
        )?;
        } // end owns_k (key publishing only)
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
            self.s.pos_ctr.ptr as *const std::os::raw::c_int, 1, 0,
            0,
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
        // No upload: the indexer reads the length straight from the device
        // counter the compressor's commit kernel maintains (this was the last
        // per-step H2D inside the attention path).
        // The INDEXER's scale uses index_head_dim (128), not the attention head_dim
        // (512): the reference sets `self.softmax_scale = index_head_dim**-0.5`
        // for the Indexer and folds `n_heads**-0.5` in with the per-head weights.
        // Using head_dim made every index score 2x too small, so the top-k
        // selection picked the wrong compressed positions.
        let scale = 1.0f32 / (idx_hd as f32).sqrt() / (idx_nh as f32).sqrt();
        // keys live on the KV OWNER's buffer (the source layer that published
        // them); a non-source index layer's own index_k is empty
        let key_owner = self.kv_owner(layer);
        let idx_lens_ptr =
            (self.s.clen.ptr as *const std::os::raw::c_int).wrapping_add(key_owner);
        self.dev.indexer_topk(
            self.s.idx_q.as_f32(),
            self.layers[key_owner].index_k.as_f32(),
            self.s.idx_w.as_f32(),
            std::ptr::null(),
            idx_lens_ptr,
            // the kernel writes `picked + offset` into out[row*cols + i], so
            // `out` points at the first compressed slot of the row
            (self.layers[layer].idxs.ptr as *mut i32).wrapping_add(offset),
            1,
            1,
            idx_nh as i32,
            idx_hd as i32,
            // The LIVE compressed count again (what this passed before the constant
            // existed). It is only the FALLBACK now - the kernel prefers the device
            // counter it receives as `lens`, and its shared memory is sized from the
            // fixed chunk instead of from this value, so a graph replay stays exact
            // and no candidate can be lost to a frozen bound.
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
            self.s.pos_ctr.ptr as *const std::os::raw::c_int,
            cfg.norm_eps,
        )?;
        // FUSED COMMIT (device-side): reads out_rows on the device, ropes the
        // latent, stores it into the ring and advances this layer's device
        // counter. The old form downloaded out_rows (a sync D2H per layer per
        // step), branched on the host, roped, copied and bumped a host counter -
        // all impossible to capture in a graph.
        self.dev.compress_commit(
            cache.latent.as_f32(),
            self.cos_comp.as_f32(),
            self.sin_comp.as_f32(),
            cache.ring.ptr as *mut f32,
            cache.out_rows.ptr as *const std::os::raw::c_int,
            (self.s.clen.ptr as *mut std::os::raw::c_int).wrapping_add(layer),
            hd as i32,
            cfg.rope_head_dim as i32,
            (cfg.rope_head_dim / 2) as i32,
            cfg.window_size as i32,
            ratio as i32,
        )?;
        // The host keeps a MIRROR of the device counter using the SAME deterministic
        // rule the kernel applies ((pos + 1) % ratio == 0 commits one latent). The
        // host branches on this (whether to run the indexer, how many compressed
        // slots to expect) and the kernels read the device counter itself - so the
        // two agree by construction, without the download that used to be here.
        if (pos + 1) % ratio == 0 {
            self.layers[layer].compress_len += 1;
        }
        Ok(self.layers[layer].compress_len)
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
        // MoE is TP-split, NOT expert-parallel: every rank holds every expert,
        // and each expert's `inter` axis is cut by world. The slice is padded up
        // to the MMA K atom (64) with zeros by the loader, so the kernels are
        // sized by the padded local width.
        let inter_local = crate::dsv41::weights::padded_inter(inter / self.world());
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
        // DEVICE-side dispatch: the routing stays on the device and the expert
        // kernels read `ids[slot]` themselves. Both downloads here were blocking
        // cudaMemcpy calls (download_f32 uses the synchronous memcpy), i.e. a
        // per-layer host stall; they are gone, and the launch arguments no longer
        // depend on the routing (which is what a CUDA graph needs).
        let (idx, wgt) = (Vec::<i32>::new(), Vec::<f32>::new());

        // One token: every assignment shares the input row, so expert e's total
        // contribution is expert_e(x) * sum of its routing weights. Accumulating
        // the distinct experts in index order keeps the sum deterministic.
        // `expert_down_fp4` OVERWRITES its output (launch_mxf4 writes
        // out[row*n + col] = x), so the scratch AND the accumulator both start
        // from zero — `o` still held the attention output at this point, and the
        // memcpy that used to follow the loop replaced the finished sum with the
        // last expert's contribution alone.
        self.dev.zero(&self.s.ex_out)?;
        self.dev.zero(&self.s.o)?;
        // No expert parallelism: the local array holds ALL experts, so a global
        // routing id indexes it directly (with the EP scheme it had to be
        // rebased by rank*ne).
        let ne = ld.experts.len();
        let wsum = vec![0f32; ne.max(1)];
        if std::env::var("DSV41_MOEDBG").map(|v| v != "0").unwrap_or(false) {
            eprintln!("[mine] route idx={:?} wgt={:?}", &idx, &wgt);
        }
        // The experts' tensors are views into one per-layer pool with a uniform
        // per-expert stride, so the kernels can derive every pointer from a base
        // plus `ids[slot] * stride`. Taking the stride as the difference of two
        // experts' pointers keeps this correct for whatever layout the loader
        // chose.
        let (w1_base, w1_stride, w1s_base, w1s_stride, w3_base, w3_stride, w3s_base, w3s_stride) =
            if ne >= 2 {
                let (a, b) = (&ld.experts[0], &ld.experts[1]);
                let d = |x: *mut std::ffi::c_void, y: *mut std::ffi::c_void| {
                    (y as i64) - (x as i64)
                };
                (
                    a.w1.ptr() as *const u8, d(a.w1.ptr(), b.w1.ptr()),
                    a.w1_scale.ptr() as *const u8, d(a.w1_scale.ptr(), b.w1_scale.ptr()),
                    a.w3.ptr() as *const u8, d(a.w3.ptr(), b.w3.ptr()),
                    a.w3_scale.ptr() as *const u8, d(a.w3_scale.ptr(), b.w3_scale.ptr()),
                )
            } else {
                (std::ptr::null(), 0, std::ptr::null(), 0, std::ptr::null(), 0, std::ptr::null(), 0)
            };
        let (w2_base, w2_stride, w2s_base, w2s_stride) = if ne >= 2 {
            let (a, b) = (&ld.experts[0], &ld.experts[1]);
            let d = |x: *mut std::ffi::c_void, y: *mut std::ffi::c_void| {
                    (y as i64) - (x as i64)
                };
            (
                a.w2.ptr() as *const u8, d(a.w2.ptr(), b.w2.ptr()),
                a.w2_scale.ptr() as *const u8, d(a.w2_scale.ptr(), b.w2_scale.ptr()),
            )
        } else {
            (std::ptr::null(), 0, std::ptr::null(), 0)
        };
        if std::env::var("DSV41_MOEDBG").map(|v| v != "0").unwrap_or(false) {
            let n_active = wsum.iter().filter(|w| **w != 0.0).count();
            eprintln!(
                "[mine] L{layer} route idx={:?} active_experts={} of {} (topk={})",
                &idx, n_active, ne, topk
            );
        }
        if !self.opts.skip_experts {
            // The input row is identical for every expert, so quantise it ONCE
            // here instead of inside the loop: the fp4 path was re-quantising and
            // re-packing the same 5120-element row for each of the ~6 selected
            // experts, i.e. 6x the quant_fp4 + fp4_pack launches (two of the
            // per-expert small kernels nsys counts ~850 times).
            self.dev.quant_fp4(
                self.s.xn.ptr as *const f32,
                self.s.xq.ptr as *mut u8,
                self.s.xsc.ptr as *mut f32,
                1,
                dim as i32,
                32,
                true,
            )?;
            // Fixed 6-slot device-driven loop: the expert id comes from
            // route_idx on the device and the weights from route_w, so there is
            // no host round trip and the launch arguments are static.
            //
            // DSV41_MOE_BATCH=1 (default OFF) replaces the loop below with ONE
            // launch per direction (grid.y = slot). Requirements, all checked
            // here so an old .so or a degenerate layer falls back to the
            // verified sequential path instead of failing:
            //   * topk > 0 and ne >= 2 (the indirect weight-base scheme);
            //   * the batched symbols are actually present in the loaded .so.
            let batched = moe_batch()
                && topk > 0
                && ne >= 2
                && self.dev.supports_moe_batch();
            if batched {
                // Per-slot strides. `ex_act_b` holds [topk][2*inter_local] and
                // `ex_down_b` [topk][dim]; both slices are disjoint, which is
                // what the batched gate/up and down require (the sequential
                // loop instead reused one ex_act and accumulated into `o`).
                let act_slot = (2 * inter_local) as i64;
                let down_slot = dim as i64;
                let ids = self.s.route_idx.ptr as *const i32;
                self.dev.expert_gate_up_fp4_batched(
                    self.s.xq.as_u8(),
                    self.s.xsc.as_f32(),
                    self.s.ex_act_b.ptr as *mut f32,
                    act_slot,
                    1,
                    dim as i32,
                    inter_local as i32,
                    cfg.swiglu_limit,
                    topk as i32,
                    w1_base,
                    w1_stride,
                    w1s_base,
                    w1s_stride,
                    w3_base,
                    w3_stride,
                    w3s_base,
                    w3s_stride,
                    ids,
                )?;
                self.dev.swiglu_limit_batched(
                    self.s.ex_act_b.ptr as *mut f32,
                    1,
                    inter_local as i32,
                    cfg.swiglu_limit,
                    act_slot,
                    topk as i32,
                )?;
                // row_weight is PER SLOT here: route_w is [topk] and contiguous,
                // so the kernel reads route_w[slot] (rw_stride = 1) — the exact
                // scalar the sequential call passed as `route_w + slot`.
                self.dev.expert_down_fp4_batched(
                    self.s.ex_act_b.ptr as *const f32,
                    act_slot,
                    self.s.ex_down_b.ptr as *mut f32,
                    down_slot,
                    1,
                    dim as i32,
                    inter_local as i32,
                    self.s.route_w.ptr as *const f32,
                    1,
                    topk as i32,
                    w2_base,
                    w2_stride,
                    w2s_base,
                    w2s_stride,
                    ids,
                )?;
                // Fixed-order sum, slot 0 first: the SAME order the sequential
                // `o[row] += x` accumulation used (from the zeroed `o`), so the
                // result is bit-identical (fp addition is not associative).
                self.dev.moe_down_reduce(
                    self.s.ex_down_b.ptr as *const f32,
                    self.s.o.ptr as *mut f32,
                    dim as i32,
                    topk as i32,
                )?;
            } else {
                for slot in 0..topk {
                    let w = (self.s.route_w.ptr as *const f32).wrapping_add(slot);
                    let ids = self.s.route_idx.ptr as *const i32;
                    self.dev.expert_gate_up_fp4_indirect(
                        self.s.xq.as_u8(),
                        self.s.xsc.as_f32(),
                        self.s.ex_act.ptr as *mut f32,
                        1,
                        dim as i32,
                        inter_local as i32,
                        cfg.swiglu_limit,
                        w1_base,
                        w1_stride,
                        w1s_base,
                        w1s_stride,
                        w3_base,
                        w3_stride,
                        w3s_base,
                        w3s_stride,
                        ids,
                        slot as i32,
                    )?;
                    self.dev.swiglu_limit(
                        self.s.ex_act.ptr as *mut f32,
                        1,
                        inter_local as i32,
                        cfg.swiglu_limit,
                    )?;
                    self.dev.expert_down_fp4_indirect(
                        self.s.ex_act.ptr as *const f32,
                        self.s.o.ptr as *mut f32,
                        1,
                        dim as i32,
                        inter_local as i32,
                        w,
                        w2_base,
                        w2_stride,
                        w2s_base,
                        w2s_stride,
                        ids,
                        slot as i32,
                    )?;
                }
            }
        }

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
        // the block output is the attention-branch accumulator `o`
        Ok(())
    }
}
