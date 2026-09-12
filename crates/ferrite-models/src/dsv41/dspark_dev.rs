//! DSpark draft, device path: the `mtp.*` block-level draft of DeepSeek-V4.1.
//!
//! The host reference is [`crate::dsv41::dspark`] (CPU numerics) plus
//! [`crate::dsv41::chain::forward_spec`] (the call order). Both were written as
//! a CPU oracle; this module is the device twin, and it reuses the *main
//! chain's* kernels throughout ([`Device`]) because a draft block is isomorphic
//! to a backbone block: the checkpoint's `mtp.{s}.*` tensors are the same
//! shapes as a layer's, so the sequence is
//!
//! ```text
//!   forward_embed   main_proj(main_h) -> main_norm -> main_x  [1, dim]
//!                   embed[ids] hc-expanded                    [bs, hc, dim]
//!   per mtp block   hc_mixes -> collapse+norm -> DSparkAttention -> hc_post
//!                   hc_mixes -> collapse+norm -> MoE             -> hc_post
//!   forward_head    collapse -> norm -> head -> bs x markov head -> confidence
//! ```
//!
//! Only the draft *forward* exists (the reference ships no speculative loop).
//! The caller owns the orchestration: [`DsparkDev::note_target_hidden`] records
//! the target layers' attention input as the backbone walks past them, and
//! [`DsparkDev::draft_forward`] runs one draft block once the backbone has
//! sampled its next token.
//!
//! ## What this first cut deliberately does NOT do
//!
//! * No fusion gates, no side streams, no CUDA-graph-safe device counters. The
//!   per-step launch count is high on purpose: every call below is a straight
//!   transcription of the host reference onto an existing ABI, so the
//!   numerical questions and the scheduling questions stay separable.
//! * The grouping of the output projection loops one (group, row) pair at a
//!   time — see [`DsparkDev::draft_attention`].
//! * The head projection reads the whole (bf16) vocabulary head once per draft
//!   row — see [`DsparkDev::draft_head`].

use std::ffi::c_void;
use std::sync::Arc;

use ferrite_types::{FerriteError, Result};

use crate::dsv41::config::Dsv41Config;
use crate::dsv41::device::{DevBuf, Device};
use crate::dsv41::load::{Dsv41DevWeights, DevTensor, LayerDev};
use crate::dsv41::tp::Collective;
use crate::dsv41::unit_dump::{self, UnitDump};

/// Grid cap of `dsv41_dspark_markov_head`'s per-block partial scratch; must
/// match `DSPARK_MARKOV_MAX_BLOCKS` in `kernels/cuda/dsv41_glue.cu`.
const MARKOV_MAX_BLOCKS: usize = 2048;

/// The chain's target-hidden tap is a fixed `[DSPARK_TAP_SLOTS, dim]` buffer
/// (`DevChain`'s `dspark_tap`, filled by the `layer()` hook at the slot
/// `dspark_target_slot` names). It is stated here as well as there because
/// [`DsparkDev::import_tap`] copies that many slots in one go — a config
/// listing more targets than this would already overflow the CHAIN's buffer
/// inside `layer()`, not this one.
pub const DSPARK_TAP_SLOTS: usize = 3;

/// How many of the draft's sampled tokens one speculative step verifies.
/// `draft_forward` samples `dspark_block_size` (6) tokens into `ids[1..=bs]`;
/// the orchestration (`chain_dev.rs::dspark_shadow_step`) verifies the first
/// `DSPARK_DRAFTS` of them, and the accept chain's bonus token brings the
/// emitted block to `DSPARK_DRAFTS + 1` — exactly the configured block size.
/// Declared here rather than at the call site so the D2H accessor below and the
/// orchestration cannot drift apart about the width.
pub const DSPARK_DRAFTS: usize = 5;

/// `DSV41_DSPARK_TRACE=1` prints the draft's geometry once, for bring-up.
fn trace() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("DSV41_DSPARK_TRACE").map(|v| v != "0").unwrap_or(false))
}

/// The device draft. Holds the window rings (one per MTP block), every scratch
/// buffer the forward needs, and references to the shared weights/config.
pub struct DsparkDev<'a> {
    dev: &'a Device,
    cfg: &'a Dsv41Config,
    w: &'a Dsv41DevWeights,

    // ---- tensor parallelism ----
    /// TP degree / this rank's index (1 / 0 without a collective). The draft's
    /// ATTENTION tensors are replicated (every rank maps the GLOBAL head/group/
    /// vocab geometry), but its MoE is TP-split exactly like the backbone's, so
    /// the expert and shared-expert slices below are LOCAL and need an
    /// all-reduce to become the full block.
    world: usize,
    rank: usize,
    /// The collective the MoE's all-reduce runs on; `None` on a single device
    /// (and the reason a `world > 1` instance refuses to run without one).
    comm: Option<Arc<Collective>>,

    // ---- geometry ----
    dim: usize,
    hd: usize,
    nh: usize,
    ql: usize,
    /// per-group low-rank width (`o_lora_rank`)
    olg: usize,
    groups: usize,
    hpg: usize,
    /// `groups * o_lora_rank`
    ol_total: usize,
    win: usize,
    bs: usize,
    hc: usize,
    vocab: usize,
    mr: usize,
    n_target: usize,
    /// `inter / world` padded to the expert MMA K atom (the routed expert
    /// kernels are sized by it)
    inter_local: usize,
    /// The SHARED expert's local width: `inter / world` under DSV41_SHARED_TP
    /// (w1/w3 are `Shard::Rows`), the whole `inter` under the replicated
    /// rank-0-only layout. NOT padded — the loader slices the rows exactly, and
    /// the shared GEMM's `n` is a row count, not an MMA K.
    sh_il: usize,

    // ---- draft window rings, one per MTP block (`w.mtp[s]` has its own wkv) ----
    window: Vec<DevBuf>,

    // ---- forward_embed ----
    /// the recorded `h.mean(dim=hc)` of `dspark_target_layer_ids`, concatenated
    main_h: DevBuf,
    /// `main_norm(main_proj(main_h))`, `[1, dim]`
    main_x: DevBuf,
    /// the hc-expanded draft input, `[bs * hc, dim]`
    h: DevBuf,
    /// `hc_post`'s staging (it is not in-place for rows > 1)
    h_out: DevBuf,

    // ---- per-block scratch ----
    xn: DevBuf,
    qr: DevBuf,
    q: DevBuf,
    kv: DevBuf,
    /// the main stream's KV row for this step, `[hd]`
    mk: DevBuf,
    /// the window ring followed by the draft block, `[win + bs, hd]`
    all_kv: DevBuf,
    o: DevBuf,
    wo: DevBuf,
    xq: DevBuf,
    xsc: DevBuf,
    xq4: DevBuf,
    xsc4: DevBuf,
    pre_in: DevBuf,
    pre_attn: DevBuf,
    pre_ffn: DevBuf,
    post: DevBuf,
    comb: DevBuf,

    // ---- attention selection ----
    idxs: DevBuf,
    /// `[bs + 1]`: `ids[0]` is the backbone's token, the rest are the samples
    ids: DevBuf,
    /// the compressor-length stand-in `sparse_attn` reads: constant `bs`
    clen: DevBuf,
    /// one i32 the RoPE base is uploaded into
    pos_base: DevBuf,

    // ---- head ----
    collapse: DevBuf,
    normed: DevBuf,
    logits: DevBuf,
    confidence: DevBuf,
    mk_partial: DevBuf,
    mk_ctr: DevBuf,

    // ---- MoE ----
    scores: DevBuf,
    route_w: DevBuf,
    route_idx: DevBuf,
    ex_act: DevBuf,
    ex_act_b: DevBuf,
    moe_out: DevBuf,
    shared_out: DevBuf,

    /// `[hc]` of `1 / hc`, the mean the target-hidden recording collapses with
    pre_mean: DevBuf,
    /// `[bs * hc]` one-hot on copy 0, the first block's incoming premix
    premix_init: DevBuf,

    /// the main chain's RoPE tables (see [`DsparkDev::set_rope_tables`])
    cos: Option<*const f32>,
    sin: Option<*const f32>,

    /// `min(win, pos + 1)` the `idxs` buffer currently holds
    n_win_cached: usize,
    /// layer -> its slot in `main_h` (`None` for a layer that is not a target)
    target_slot: Vec<Option<usize>>,

    /// Golden per-unit capture (`DSV41_DSPARK_UNIT_DUMP=1`, see
    /// [`crate::dsv41::unit_dump`]). `Some` for exactly ONE forward — the first
    /// `draft_forward` with `pos > 0` on rank 0 — and `None` on every other
    /// call, so the per-dump check is one `Option` branch when the gate is off.
    unit: Option<UnitDump>,
}

fn need<'t>(t: &'t Option<DevTensor>, what: &str) -> Result<&'t DevTensor> {
    t.as_ref()
        .ok_or_else(|| FerriteError::Config(format!("dspark: {what} is not loaded")))
}

impl<'a> DsparkDev<'a> {
    /// `world`/`rank` are the tensor-parallel degree and this rank's index. The
    /// draft's attention runs on the GLOBAL geometry on every rank (its attention
    /// tensors are `Shard::Replicated`), but its MoE is TP-split like the
    /// backbone's, so `world` decides the LOCAL expert slices here — under TP8
    /// the checkpoint's `inter` (2304) is the global number and each rank holds
    /// `inter / world` (288, padded to 320 by the loader).
    pub fn new(
        dev: &'a Device,
        w: &'a Dsv41DevWeights,
        cfg: &'a Dsv41Config,
        world: usize,
        rank: usize,
    ) -> Result<Self> {
        let world = world.max(1);
        let dim = cfg.dim;
        let hd = cfg.head_dim;
        let nh = cfg.n_heads;
        let ql = cfg.q_lora_rank;
        let olg = cfg.o_lora_rank;
        let groups = cfg.o_groups;
        let hpg = nh / groups;
        let ol_total = groups * olg;
        let win = cfg.window_size;
        let bs = cfg.dspark_block_size;
        let hc = cfg.hc_mult;
        let vocab = cfg.vocab_size;
        let mr = cfg.dspark_markov_rank;
        let n_target = cfg.dspark_target_layer_ids.len();
        let inter = cfg.moe_inter_dim;
        // The routed experts are TP-split along `inter` (the loader cuts w1/w3's
        // rows and w2's columns by world, see weights.rs::load_expert_pool), so
        // every expert kernel below is sized by the PADDED LOCAL width.
        let inter_local = crate::dsv41::weights::padded_inter(inter / world);
        // The shared expert's w1/w3 are `Shard::Rows` only under
        // DSV41_SHARED_TP; otherwise they are replicated and rank 0 alone
        // computes them (weights.rs::shared_expert_tp).
        let stp = crate::dsv41::weights::shared_expert_tp();
        let sh_il = if stp { inter / world } else { inter };
        if bs == 0 || n_target == 0 || cfg.n_mtp_layers == 0 {
            return Err(FerriteError::Config(
                "dspark: the config has no draft (dspark_block_size / \
                 dspark_target_layer_ids / n_mtp_layers is empty)"
                    .into(),
            ));
        }
        let fb = |n: usize| n * 4;

        // The largest fp8 activation quantised anywhere below.
        let max_act = (n_target * dim)
            .max(bs * nh * hd)
            .max(bs * ol_total)
            .max(bs * dim)
            .max(bs * inter);

        let mut window = Vec::with_capacity(cfg.n_mtp_layers);
        for _ in 0..cfg.n_mtp_layers {
            let b = dev.alloc(fb(win * hd))?;
            dev.zero(&b)?;
            window.push(b);
        }

        let main_h = dev.alloc(fb(n_target * dim))?;
        dev.zero(&main_h)?;

        let s = DsparkDev {
            dev,
            cfg,
            w,
            world,
            rank,
            comm: None,
            dim,
            hd,
            nh,
            ql,
            olg,
            groups,
            hpg,
            ol_total,
            win,
            bs,
            hc,
            vocab,
            mr,
            n_target,
            inter_local,
            sh_il,
            window,
            main_h,
            main_x: dev.alloc(fb(dim))?,
            h: dev.alloc(fb(bs * hc * dim))?,
            h_out: dev.alloc(fb(bs * hc * dim))?,
            xn: dev.alloc(fb(bs * dim))?,
            qr: dev.alloc(fb(bs * ql))?,
            q: dev.alloc(fb(bs * nh * hd))?,
            kv: dev.alloc(fb(bs * hd))?,
            mk: dev.alloc(fb(hd))?,
            all_kv: dev.alloc(fb((win + bs) * hd))?,
            o: dev.alloc(fb(bs * nh * hd))?,
            wo: dev.alloc(fb(bs * ol_total))?,
            xq: dev.alloc(max_act.max(8))?,
            xsc: dev.alloc(fb(max_act / 32 + 8))?,
            xq4: dev.alloc((bs * dim / 2).max(8))?,
            xsc4: dev.alloc(fb(bs * dim / 32 + 8))?,
            pre_in: dev.alloc(fb(bs * hc))?,
            pre_attn: dev.alloc(fb(bs * hc))?,
            pre_ffn: dev.alloc(fb(bs * hc))?,
            post: dev.alloc(fb(bs * hc))?,
            comb: dev.alloc(fb(bs * hc * hc))?,
            idxs: dev.alloc(fb(bs * (win + bs)).max(4))?,
            ids: dev.alloc(fb(bs + 1).max(4))?,
            clen: dev.alloc(4)?,
            pos_base: dev.alloc(4)?,
            collapse: dev.alloc(fb(bs * dim))?,
            normed: dev.alloc(fb(bs * dim))?,
            logits: dev.alloc(fb(bs * vocab))?,
            confidence: dev.alloc(fb(bs).max(4))?,
            mk_partial: dev.alloc(MARKOV_MAX_BLOCKS * 8)?,
            mk_ctr: dev.alloc(4)?,
            scores: dev.alloc(fb(bs * cfg.n_routed_experts.max(1)))?,
            route_w: dev.alloc(fb(bs * cfg.n_activated_experts.max(1)).max(4))?,
            route_idx: dev.alloc(fb(bs * cfg.n_activated_experts.max(1)).max(4))?,
            // Shared by the routed SEQUENTIAL fallback ([bs][2*inter_local]) and
            // the shared expert ([2*sh_il]); under the replicated shared layout
            // (DSV41_SHARED_TP=0) `sh_il == inter` is the LARGER of the two, so
            // the size has to clear both.
            ex_act: dev.alloc(fb(bs * 2 * inter_local.max(sh_il)))?,
            ex_act_b: dev.alloc(fb(cfg.n_activated_experts.max(1) * bs * 2 * inter_local))?,
            moe_out: dev.alloc(fb(bs * dim))?,
            shared_out: dev.alloc(fb(bs * dim))?,
            pre_mean: dev.alloc(fb(hc))?,
            premix_init: dev.alloc(fb(bs * hc))?,
            cos: None,
            sin: None,
            n_win_cached: usize::MAX,
            target_slot: (0..cfg.n_layers + cfg.n_mtp_layers)
                .map(|l| cfg.dspark_target_layer_ids.iter().position(|&t| t == l))
                .collect(),
            unit: None,
        };

        // `clen` is the compressor-length stand-in `sparse_attn` reads; for the
        // draft the "compressed" block is the draft tokens themselves. The kernel
        // reads it as an `int`, so the four bytes must be the INTEGER's, not the
        // f32 the upload helper's name suggests.
        s.dev
            .upload_bytes_at(&s.clen, &(bs as i32).to_le_bytes())?;
        // The target-hidden mean and the first block's incoming premix are
        // constants of the geometry, so they are uploaded once here.
        let mean = vec![1.0f32 / hc as f32; hc];
        s.dev.upload_f32_at(s.pre_mean.ptr, 0, &mean)?;
        let mut pm = vec![0f32; bs * hc];
        for r in 0..bs {
            pm[r * hc] = 1.0;
        }
        s.dev.upload_f32_at(s.premix_init.ptr, 0, &pm)?;
        // The markov election counter must start at zero (cudaMalloc does not
        // zero) and self-resets inside the kernel from then on.
        s.dev.zero(&s.mk_ctr)?;

        if trace() {
            eprintln!(
                "[dspark] device draft: bs={bs} win={win} hc={hc} dim={dim} hd={hd} nh={nh} \
                 ql={ql} groups={groups} olg={olg} vocab={vocab} mr={mr} targets={n_target} \
                 world={world} rank={rank} inter={inter}/local={inter_local}/shared={sh_il} \
                 mtp={}",
                cfg.n_mtp_layers
            );
        }
        Ok(s)
    }

    /// Hand the draft the rank collective its MoE all-reduce runs on. Must be
    /// called before the first [`Self::draft_forward`] whenever `world > 1`;
    /// without it a multi-rank draft refuses to run rather than produce each
    /// rank's partial sum as if it were the whole block.
    pub fn set_comm(&mut self, c: Arc<Collective>) {
        self.comm = Some(c);
    }

    /// The main chain's RoPE tables (`cos`/`sin`, `[max_pos][rope_head_dim/2]`),
    /// built by `Device::rope_precompute` with the model's main theta. They are
    /// INJECTED rather than built here: a private copy would be another ~128 MiB
    /// on a 4 GiB host, and the draft's positions are a subset of the chain's.
    /// Must be called before the first [`Self::draft_forward`].
    pub fn set_rope_tables(&mut self, cos: *const f32, sin: *const f32) {
        self.cos = Some(cos);
        self.sin = Some(sin);
    }

    /// Copy the chain's target-hidden tap (`[3, dim]`, slot order =
    /// `dspark_target_layer_ids`) into this draft's `main_h` — one D2D, no host
    /// round trip.
    ///
    /// The device twin of [`Self::note_target_hidden`]: the main chain's
    /// `layer()` hook already collapses each target layer's attention input with
    /// the same `1/hc` mean (`DevChain`'s `dspark_tap`), so the orchestration
    /// hands the whole `[n_target, dim]` block over in one copy instead of
    /// calling back into the layer loop. `src` must span at least
    /// `n_target * dim` f32 in the SAME slot order the chain wrote them in
    /// (`Dsv41Config::dspark_target_slot`).
    pub fn import_tap(&mut self, src: *const f32) -> Result<()> {
        // The chain's tap is sized `DSPARK_TAP_SLOTS x dim`; a config with more
        // targets than that would have overflowed it inside `layer()` first.
        debug_assert!(
            self.n_target <= DSPARK_TAP_SLOTS,
            "dspark: the chain's tap holds {DSPARK_TAP_SLOTS} slots, but the config lists {} \
             target layers",
            self.n_target
        );
        self.dev.memcpy_d2d(
            self.main_h.ptr,
            src as *const c_void,
            self.n_target * self.dim * std::mem::size_of::<f32>(),
        )
    }

    /// Append a COMMITTED verify block's target hiddens to the window rings:
    /// rows `0..keep` of the block, written at the positions `pos_base + j`.
    ///
    /// This is the official `_update`'s `target_hidden_states[:, :accepted + 1]`
    /// append, and it is what keeps the draft's context WHOLE across a
    /// multi-token commit. The per-step path ([`Self::draft_forward`]) seeds
    /// exactly ONE row — the position it consumes — so before this existed every
    /// position a `k > 1` commit SKIPPED stayed a hole in the ring, and the
    /// draft's attention read across it (accept degrading with the number of
    /// multi-token steps).
    ///
    /// # Layout and row range (must stay in step with `dspark_commit`'s `keep`)
    ///
    /// `tap_r` is the chain's per-row tap, `[DSPARK_TAP_SLOTS][m][dim]` with slot
    /// `slot` (`Dsv41Config::dspark_target_slot`, 0/1/2 = the target layers in
    /// config order) and row `r` at `(slot * m + r) * dim`. Verify row `r` was
    /// forwarded at position `pos + 1 + r`, so row `j`'s hidden is the one for
    /// position `pos_base + j` with `pos_base = pos + 1`.
    ///
    /// `keep` is the accepted prefix length `k_acc` — the SAME value
    /// `dspark_commit` keeps in the ring, so the two cannot disagree: rows
    /// `0..keep` sit at `pos+1 ..= pos+keep`, which is exactly the ring prefix
    /// the commit preserved. Row `keep` (position `pos+keep+1` = the new
    /// `pos_ctr`) is deliberately NOT written: the verify fed it a REJECTED
    /// draft token, so its hidden is not the true one — that position's KV comes
    /// from the next step's `step_dev` and is seeded by the next
    /// [`Self::draft_forward`], exactly as for a plain decode step. Rows past
    /// `keep` are rejected and dropped.
    ///
    /// The rows are projected ONE at a time through the same
    /// [`Self::project_main_x`] + [`Self::seed_window`] pair the per-step seed
    /// uses, so a committed row and a per-step row are the same numbers.
    pub fn note_ctx_rows(
        &mut self,
        tap_r: *const f32,
        m: usize,
        keep: usize,
        pos_base: usize,
    ) -> Result<()> {
        if keep > m {
            return Err(FerriteError::Config(format!(
                "dspark: note_ctx_rows asked for {keep} ctx rows out of a {m}-row verify block"
            )));
        }
        if keep == 0 {
            return Ok(());
        }
        let dim = self.dim;
        let n_target = self.n_target;
        let row_bytes = dim * std::mem::size_of::<f32>();
        for j in 0..keep {
            // Row j's `forward_embed` input is the CONCATENATION of the target
            // layers' hidden[j]; the tap interleaves the layers (`m` rows apart),
            // so the `n_target * dim` block is rebuilt one `dim`-wide slice at a
            // time into the same `main_h` the per-step path fills.
            for slot in 0..n_target {
                let src = (tap_r as *const u8).wrapping_add((slot * m + j) * row_bytes);
                let dst = (self.main_h.ptr as *mut u8).wrapping_add(slot * row_bytes);
                self.dev
                    .memcpy_d2d(dst as *mut c_void, src as *const c_void, row_bytes)?;
            }
            self.project_main_x()?;
            for s in 0..self.cfg.n_mtp_layers {
                self.seed_window(s, pos_base + j)?;
            }
        }
        Ok(())
    }

    /// D2H of the drafted block: `draft_forward` samples into `ids[1..=bs]`
    /// (`ids[0]` is the backbone's token), and one speculative step consumes the
    /// first [`DSPARK_DRAFTS`] of those samples.
    ///
    /// One 20-byte device read, outside any capture. `download_u8` keeps it a
    /// plain byte copy, so no f32 reinterpretation is involved — the discipline
    /// `step_rows` already uses for its per-row argmax.
    pub fn drafts(&self) -> Result<[u32; DSPARK_DRAFTS]> {
        if self.bs < DSPARK_DRAFTS {
            return Err(FerriteError::Config(format!(
                "dspark: the draft block samples {} tokens; the speculative step verifies \
                 {DSPARK_DRAFTS} of them",
                self.bs
            )));
        }
        // `ids` is `[i32; bs + 1]`: skip the backbone's token at index 0.
        let b = Device::view(
            (self.ids.ptr as *mut i32).wrapping_add(1) as *mut c_void,
            DSPARK_DRAFTS * 4,
        );
        let mut bytes = [0u8; DSPARK_DRAFTS * 4];
        self.dev.download_u8(&b, &mut bytes)?;
        Ok(std::array::from_fn(|i| {
            u32::from_le_bytes([
                bytes[4 * i],
                bytes[4 * i + 1],
                bytes[4 * i + 2],
                bytes[4 * i + 3],
            ])
        }))
    }

    /// Record one target layer's attention input, `h.mean(dim=hc)` (dspark.rs's
    /// `forward_embed` reads the ATTENTION INPUT of layers 37/38/39, not the
    /// output). `h` is the layer's `[hc, dim]` residual stream on the device.
    ///
    /// A no-op for a layer that is not a `dspark_target_layer_ids` entry, so the
    /// caller can invoke it unconditionally from the layer loop.
    pub fn note_target_hidden(&mut self, layer: usize, h: *const f32) -> Result<()> {
        let Some(slot) = self.target_slot.get(layer).copied().flatten() else {
            return Ok(());
        };
        let dst = (self.main_h.ptr as *mut f32).wrapping_add(slot * self.dim);
        // hc_collapse IS the weighted sum over the hc copies; with pre = 1/hc it
        // is the mean the reference computes elementwise.
        self.dev.hc_collapse(
            h,
            self.pre_mean.ptr as *const f32,
            dst,
            1,
            self.hc as i32,
            self.dim as i32,
        )
    }

    /// `main_x = main_norm(main_proj(main_h))` — the `forward_embed` projection
    /// that turns the recorded target hiddens into the row the window KV is
    /// projected from.
    ///
    /// ONE path for both callers — [`Self::draft_forward`]'s per-step seed and
    /// [`Self::note_ctx_rows`]'s committed rows — so the two kinds of window row
    /// cannot drift apart numerically.
    fn project_main_x(&mut self) -> Result<()> {
        let cfg = self.cfg;
        let dim = self.dim;
        let k_main = self.n_target * self.dim;
        let main_proj = need(&self.w.main_proj, "mtp.0.main_proj.weight")?;
        let main_proj_scale = need(&self.w.main_proj_scale, "mtp.0.main_proj.scale")?;
        let main_norm = need(&self.w.main_norm, "mtp.0.main_norm.weight")?;
        self.quant1(self.main_h.ptr as *const f32, k_main)?;
        self.dev.gemm_fp8_mx(
            self.xq.as_u8(),
            self.xsc.as_f32(),
            main_proj.as_u8(),
            main_proj_scale.as_u8(),
            std::ptr::null(),
            self.main_x.ptr as *mut f32,
            1,
            dim as i32,
            k_main as i32,
        )?;
        self.dev.rmsnorm(
            self.main_x.ptr as *const f32,
            main_norm.as_f32(),
            self.main_x.ptr as *mut f32,
            1,
            dim as i32,
            cfg.norm_eps,
        )?;
        Ok(())
    }

    /// Run the whole draft: `t0` is the token the backbone just sampled, `pos`
    /// is its position. Fills `ids[1..=bs]` (the drafted block) and
    /// `confidence[0..bs]`; both stay on the device.
    pub fn draft_forward(&mut self, t0: u32, pos: usize) -> Result<()> {
        let (Some(_), Some(_)) = (self.cos, self.sin) else {
            return Err(FerriteError::Config(
                "dspark: set_rope_tables() must be called before draft_forward()".into(),
            ));
        };
        let cfg = self.cfg;
        let dim = self.dim;
        let bs = self.bs;
        let hc = self.hc;

        // ---- golden per-unit capture: arm ONCE, on the first decoded forward ----
        // `pos == 0` is the prefill window seed (it returns before the blocks), so
        // it is not a capturable forward; rank != 0 never writes (each TP rank is
        // its own process, all pointed at the same path).
        if self.unit.is_none()
            && self.rank == 0
            && pos > 0
            && unit_dump::enabled()
            && unit_dump::arm_once()
        {
            self.unit = Some(UnitDump::new());
        }

        // ---- golden input injection (`DSV41_DSPARK_UNIT_INJECT`) --------------
        // `import_tap` has already filled `main_h` with the LIVE serve's target
        // hiddens, so an armed injection OVERWRITES it with the reference
        // harness's own input, together with `ids[0]` and `pos`. It is gated on a
        // live capture so only the ONE dumped forward is touched — the rest of the
        // serve is untouched. `main_h` stays a device buffer: the override is one
        // H2D, not a host round trip.
        // The window RING is deliberately NOT injected: it is seeded by earlier
        // forwards, exactly like the reference's seed pass.
        let mut pos = pos;
        let mut t0 = t0;
        let mut injected = false;
        if self.unit.is_some() {
            if let Some(inj) = unit_dump::inject() {
                if inj.main_hidden.len() == self.n_target * dim {
                    self.dev.upload_f32_at(self.main_h.ptr, 0, &inj.main_hidden)?;
                    t0 = inj.token as u32;
                    if let Some(p) = inj.pos {
                        pos = p;
                    }
                    injected = true;
                } else {
                    eprintln!(
                        "[dspark] unit inject: main_hidden has {} values, expected {} \
                         (n_target {} x dim {dim}) — ignored",
                        inj.main_hidden.len(),
                        self.n_target * dim,
                        self.n_target
                    );
                }
            }
        }

        // ---- ids: the backbone's token first, the noise token for the rest ----
        // (dspark.rs::forward_embed; the noise row IS embed[noise_token_id])
        let noise = cfg.dspark_noise_token_id as i32;
        let mut ids = vec![noise; bs + 1];
        ids[0] = t0 as i32;
        self.upload_i32(&self.ids, &ids)?;

        // ---- forward_embed ----
        // main_x = main_norm(main_proj(concat(target hiddens)))
        self.project_main_x()?;
        self.dump_unit("main_x", self.main_x.ptr as *const f32, &[dim]);

        // h = hc_expand(embed[ids]) — the draft's residual stream starts as the
        // token embedding alone; the main stream enters through the window KV.
        need(&self.w.embed, "embed.weight")?;
        self.dev.embed_expand_dev(
            self.w.embed.as_ref().unwrap().ptr(),
            self.ids.as_i32(),
            self.h.ptr as *mut f32,
            bs as i32,
            dim as i32,
            hc as i32,
            self.vocab as i32,
        )?;
        self.dump_unit("embed", self.h.ptr as *const f32, &[bs, hc, dim]);

        // The window only seeds itself before the first decode step, exactly as
        // the reference does (`dspark_attention` with start_pos == 0).
        if pos == 0 {
            for s in 0..cfg.n_mtp_layers {
                self.seed_window(s, pos)?;
            }
            return Ok(());
        }

        self.ensure_idxs(pos)?;
        // pos_base is uploaded per RoPE call below; make the value that never
        // changes explicit.
        let eps = cfg.norm_eps;

        // ---- three draft blocks ----
        self.dev.memcpy_d2d(
            self.pre_in.ptr,
            self.premix_init.ptr,
            (bs * hc * 4) as usize,
        )?;
        for s in 0..cfg.n_mtp_layers {
            let ld = &self.w.mtp[s];
            // hc_mixes for the attention sub-block: writes THIS block's attn_pre
            // (slot 1), post and comb.
            self.hc_mixes(
                ld,
                false,
                self.pre_attn.ptr as *mut f32,
                self.post.ptr as *mut f32,
                self.comb.ptr as *mut f32,
            )?;
            // collapse with the INCOMING premix, then the attn norm
            let attn_norm = need(&ld.attn_norm, "mtp.*.attn_norm.weight")?;
            self.dev.hc_collapse(
                self.h.ptr as *const f32,
                self.pre_in.ptr as *const f32,
                self.xn.ptr as *mut f32,
                bs as i32,
                hc as i32,
                dim as i32,
            )?;
            // `h = hc_pre(x, pre_mix)`, the reference's `h(pre_mix)` — taken
            // BEFORE the in-place rmsnorm below, which is why it is recorded here.
            self.dump_unit_idx("h_premix_block", s, self.xn.ptr as *const f32, &[bs, dim]);
            self.dev.rmsnorm(
                self.xn.ptr as *const f32,
                attn_norm.as_f32(),
                self.xn.ptr as *mut f32,
                bs as i32,
                dim as i32,
                eps,
            )?;

            self.draft_attention(s, pos)?;
            // the attention block's own units: q/kv are the post-RoPE projections,
            // o is the module's output (after wo_b), all still live here.
            self.dump_unit_idx("q_block", s, self.q.ptr as *const f32, &[bs, self.nh, self.hd]);
            self.dump_unit_idx("kv_block", s, self.kv.ptr as *const f32, &[bs, self.hd]);
            self.dump_unit_idx("o_block", s, self.o.ptr as *const f32, &[bs, dim]);

            // residual: h = hc_post(o, post, comb)
            self.dev.hc_post(
                self.o.ptr as *const f32,
                self.h.ptr as *const f32,
                self.post.as_f32(),
                self.comb.as_f32(),
                self.h_out.ptr as *mut f32,
                bs as i32,
                hc as i32,
                dim as i32,
            )?;
            self.dev.memcpy_d2d(
                self.h.ptr,
                self.h_out.ptr,
                (bs * hc * dim * 4) as usize,
            )?;

            // ---- FFN sub-block ----
            self.hc_mixes(
                ld,
                true,
                self.pre_ffn.ptr as *mut f32,
                self.post.ptr as *mut f32,
                self.comb.ptr as *mut f32,
            )?;
            // the FFN collapses with THIS block's attn_pre, not the incoming one
            let ffn_norm = need(&ld.ffn_norm, "mtp.*.ffn_norm.weight")?;
            self.dev.hc_collapse_norm(
                self.h.ptr as *mut f32,
                self.pre_attn.ptr as *const f32,
                ffn_norm.as_f32(),
                self.xn.ptr as *mut f32,
                bs as i32,
                hc as i32,
                dim as i32,
                eps,
            )?;

            self.draft_moe(s, ld)?;
            // the MoE block's output, AFTER its all-reduce (full block on every
            // rank, not the rank's partial sum).
            self.dump_unit_idx("moe_out_block", s, self.moe_out.ptr as *const f32, &[bs, dim]);

            self.dev.hc_post(
                self.moe_out.ptr as *const f32,
                self.h.ptr as *const f32,
                self.post.as_f32(),
                self.comb.as_f32(),
                self.h_out.ptr as *mut f32,
                bs as i32,
                hc as i32,
                dim as i32,
            )?;
            self.dev.memcpy_d2d(
                self.h.ptr,
                self.h_out.ptr,
                (bs * hc * dim * 4) as usize,
            )?;
            // the block's residual stream once both sub-blocks are in, i.e. the
            // `h` the NEXT block (or `forward_head`) reads.
            self.dump_unit_idx("h_block", s, self.h.ptr as *const f32, &[bs, hc, dim]);

            // the NEXT block's incoming premix is this block's ffn pre
            self.dev
                .memcpy_d2d(self.pre_in.ptr, self.pre_ffn.ptr, (bs * hc * 4) as usize)?;
        }

        // ---- forward_head: collapse, norm, head, then the Markov sampler ----
        self.draft_head()?;

        // ---- serialise the capture (one file per armed forward) ----
        if let Some(u) = self.unit.take() {
            let meta = format!(
                "{{\"pos\":{pos},\"t0\":{t0},\"bs\":{bs},\"hc\":{hc},\"dim\":{dim},\"nh\":{},\
                 \"hd\":{},\"vocab\":{},\"mr\":{},\"n_target\":{},\"n_mtp\":{},\"world\":{},\
                 \"rank\":{},\"pid\":{},\"inject\":{injected}}}",
                self.nh,
                self.hd,
                self.vocab,
                self.mr,
                self.n_target,
                cfg.n_mtp_layers,
                self.world,
                self.rank,
                std::process::id(),
            );
            if let Err(e) = u.write(&meta) {
                eprintln!("[dspark] unit dump write failed: {e}");
            }
        }
        Ok(())
    }

    /// `hc_mixes` for one sub-block of one draft block. `ffn` selects the
    /// FFN's trio; `pre_out` receives the pre the NEXT sub-block collapses with.
    fn hc_mixes(
        &self,
        ld: &LayerDev,
        ffn: bool,
        pre_out: *mut f32,
        post: *mut f32,
        comb: *mut f32,
    ) -> Result<()> {
        let (f, sc, base) = if ffn {
            (
                need(&ld.hc_ffn_fn, "mtp.*.hc_ffn_fn")?,
                need(&ld.hc_ffn_scale, "mtp.*.hc_ffn_scale")?,
                need(&ld.hc_ffn_base, "mtp.*.hc_ffn_base")?,
            )
        } else {
            (
                need(&ld.hc_attn_fn, "mtp.*.hc_attn_fn")?,
                need(&ld.hc_attn_scale, "mtp.*.hc_attn_scale")?,
                need(&ld.hc_attn_base, "mtp.*.hc_attn_base")?,
            )
        };
        self.dev.hc_mixes(
            self.h.ptr as *const f32,
            f.as_f32(),
            sc.as_f32(),
            base.as_f32(),
            pre_out,
            post,
            comb,
            self.bs as i32,
            (self.hc * self.dim) as i32,
            self.hc as i32,
            self.cfg.hc_sinkhorn_iters as i32,
            self.cfg.hc_eps,
        )
    }

    /// `DSparkAttention` (dspark.rs::dspark_attention): the draft block's own
    /// q/k/v, the window ring seeded from the main stream, and every draft query
    /// additionally attending its own block.
    fn draft_attention(&mut self, s: usize, pos: usize) -> Result<()> {
        let cfg = self.cfg;
        let (dim, hd, nh, ql) = (self.dim, self.hd, self.nh, self.ql);
        let (bs, win, groups, hpg) = (self.bs, self.win, self.groups, self.hpg);
        let olg = self.olg;
        let ol_total = self.ol_total;
        let ld = &self.w.mtp[s];

        let wq_a = need(&ld.wq_a, "mtp.*.attn.wq_a.weight")?;
        let wq_a_s = need(&ld.wq_a_scale, "mtp.*.attn.wq_a.scale")?;
        let q_norm = need(&ld.q_norm, "mtp.*.attn.q_norm.weight")?;
        let wq_b = need(&ld.wq_b, "mtp.*.attn.wq_b.weight")?;
        let wq_b_s = need(&ld.wq_b_scale, "mtp.*.attn.wq_b.scale")?;
        let wkv = need(&ld.wkv, "mtp.*.attn.wkv.weight")?;
        let wkv_s = need(&ld.wkv_scale, "mtp.*.attn.wkv.scale")?;
        let kv_norm = need(&ld.kv_norm, "mtp.*.attn.kv_norm.weight")?;
        let attn_sink = need(&ld.attn_sink, "mtp.*.attn.attn_sink")?;

        // ---- the main stream's KV row goes into the ring ----
        // `pos` here is the ANCHOR's position; the main_x it projects is the
        // hidden of the anchor's PREDECESSOR (the just-consumed token — the
        // official source is hidden[anchor_pos - 1]), so both the ring slot
        // and the RoPE position are pos - 1. The historical call used pos
        // directly, which under the old (anchor == t0) timing happened to be
        // the same position; under the official timing it must shift back one.
        debug_assert!(pos > 0, "draft_forward: the anchor is never at pos 0");
        self.seed_window(s, pos - 1)?;

        // ---- q = wq_b(q_norm(wq_a(x))) with RoPE at the draft positions ----
        // D1 fix (audit-ffi-args): quantise ALL bs rows — the historical call
        // passed `dim` (ONE row) while the gemm below reads `bs × dim` bytes
        // and `bs·dim/32` scales, so rows 1..bs-1 consumed stale xq bytes and
        // the draft q/kv were silently garbage.
        self.quant1(self.xn.ptr as *const f32, bs * dim)?;
        self.dev.gemm_fp8_mx(
            self.xq.as_u8(),
            self.xsc.as_f32(),
            wq_a.as_u8(),
            wq_a_s.as_u8(),
            std::ptr::null(),
            self.qr.ptr as *mut f32,
            bs as i32,
            ql as i32,
            dim as i32,
        )?;
        self.dev.rmsnorm(
            self.qr.ptr as *const f32,
            q_norm.as_f32(),
            self.qr.ptr as *mut f32,
            bs as i32,
            ql as i32,
            cfg.norm_eps,
        )?;
        self.quant1(self.qr.ptr as *const f32, bs * ql)?;
        // wq_b is [nh * hd, ql]; one fp8 GEMM covers all bs draft rows.
        self.dev.gemm_fp8_mx(
            self.xq.as_u8(),
            self.xsc.as_f32(),
            wq_b.as_u8(),
            wq_b_s.as_u8(),
            std::ptr::null(),
            self.q.ptr as *mut f32,
            bs as i32,
            (nh * hd) as i32,
            ql as i32,
        )?;
        self.rope_queries(self.q.ptr as *mut f32, pos)?;

        // ---- kv = wkv(x), normed and roped like the backbone's window KV ----
        // D1 fix (audit-ffi-args): quantise ALL bs rows — one row left rows 1..bs-1
        // reading stale xq bytes into the gemm below.
        self.quant1(self.xn.ptr as *const f32, bs * dim)?;
        self.dev.gemm_fp8_mx(
            self.xq.as_u8(),
            self.xsc.as_f32(),
            wkv.as_u8(),
            wkv_s.as_u8(),
            std::ptr::null(),
            self.kv.ptr as *mut f32,
            bs as i32,
            hd as i32,
            dim as i32,
        )?;
        self.dev.rmsnorm(
            self.kv.ptr as *const f32,
            kv_norm.as_f32(),
            self.kv.ptr as *mut f32,
            bs as i32,
            hd as i32,
            cfg.norm_eps,
        )?;
        // The draft rows sit at pos + 1 + r, one row per step — the SAME
        // per-row positions as the queries (the sglang arbitration; the host
        // comment's `seqlen` is the sequence length, not the block size).
        self.rope_at(
            self.kv.ptr as *mut f32,
            bs as i32,
            hd as i32,
            1,
            pos as i32,
            1,
            false,
        )?;

        // ---- the ring's window rows, then the draft block, contiguous ----
        // COMPACT layout: only the n_win LIVE window rows are copied, so the
        // block's rows sit at [n_win, n_win+bs) and the kernel's derived
        // geometry is self-consistent on BOTH formulas: n = window + *clen =
        // n_win + bs (the true row stride) and topk = window + min(*clen,
        // index_topk) = n_win + bs (the true candidate count). The historical
        // copy took ALL win rows (stride win+bs) while the kernel derived
        // n_win+bs from the parameters — every kv read past row n_win landed
        // `win - n_win` rows off once pos < win-1, which is exactly the
        // "draft outputs unrelated garbage" signature.
        let n_win = win.min(pos);
        let wbytes = (n_win * hd * 4) as usize;
        self.dev
            .memcpy_d2d(self.all_kv.ptr, self.window[s].ptr, wbytes)?;
        self.dev.memcpy_d2d(
            (self.all_kv.ptr as *mut u8).wrapping_add(wbytes) as *mut c_void,
            self.kv.ptr,
            (bs * hd * 4) as usize,
        )?;
        // The anchor-KV seat: the window EXCLUDES the pos slot (n_win =
        // min(win, pos) — the ring's pos%win copy of this position stays
        // written for the NEXT round), so the anchor (t0) at pos has exactly
        // ONE kv in the candidates: the block's row 0, its EMBED-derived
        // projection — exactly the official block structure (DeepSpec's draft
        // blocks project their own inputs; the seat arbitration in the earlier
        // fix went one step further and swapped row 0's bytes for the
        // target-hidden projection mk, which is NOT what the official blocks
        // do — reverted).

        self.dev.sparse_attn(
            self.q.as_f32(),
            self.all_kv.ptr as *const f32,
            attn_sink.as_f32(),
            self.idxs.as_i32(),
            self.o.ptr as *mut f32,
            1,
            bs as i32,
            nh as i32,
            hd as i32,
            self.clen.ptr as *const std::os::raw::c_int,
            n_win as i32,
            bs as i32,
            (hd as f32).powf(-0.5),
        )?;
        // inverse RoPE, same per-query positions
        self.rope_queries_inv(self.o.ptr as *mut f32, pos)?;

        // ---- grouped low-rank output projection ----
        // The reference's `grp` reshape is the IDENTITY (o is already
        // [bs, groups, hpg*hd] row-major), so no permute is needed. wo_a is
        // [groups * olg, hpg * hd] and each row of a group uses THAT group's
        // weight block, so this is one small GEMV per (group, row) — the
        // activation stride between two rows of a group is nh*hd, which no
        // existing GEMM launcher can express. A fused grouped-output kernel (or
        // a group-major repack) is the obvious optimisation; see the module
        // header. The fp8 quantisation is shared by all of them because the
        // group blocks are whole 32-element quant blocks of the same row.
        self.quant1(self.o.ptr as *const f32, bs * nh * hd)?;
        let wo_a = need(&ld.wo_a, "mtp.*.attn.wo_a.weight")?;
        let wo_a_s = need(&ld.wo_a_scale, "mtp.*.attn.wo_a.scale")?;
        let k_grp = hpg * hd;
        for g in 0..groups {
            let wp = wo_a.as_u8().wrapping_add(g * olg * k_grp);
            let wsp = wo_a_s
                .as_u8()
                .wrapping_add((g * olg / 32) * (k_grp / 32));
            for r in 0..bs {
                let a = self
                    .xq
                    .as_u8()
                    .wrapping_add((r * nh + g * hpg) * hd);
                let asc = self
                    .xsc
                    .as_f32()
                    .wrapping_add(((r * nh + g * hpg) * hd / 32) as usize);
                let out = (self.wo.ptr as *mut f32)
                    .wrapping_add(r * ol_total + g * olg);
                self.dev.gemm_fp8_mx(
                    a,
                    asc,
                    wp,
                    wsp,
                    std::ptr::null(),
                    out,
                    1,
                    olg as i32,
                    k_grp as i32,
                )?;
            }
        }
        let wo_b = need(&ld.wo_b, "mtp.*.attn.wo_b.weight")?;
        let wo_b_s = need(&ld.wo_b_scale, "mtp.*.attn.wo_b.scale")?;
        self.quant1(self.wo.ptr as *const f32, bs * ol_total)?;
        self.dev.gemm_fp8_mx(
            self.xq.as_u8(),
            self.xsc.as_f32(),
            wo_b.as_u8(),
            wo_b_s.as_u8(),
            std::ptr::null(),
            self.o.ptr as *mut f32,
            bs as i32,
            dim as i32,
            ol_total as i32,
        )?;
        Ok(())
    }

    /// The MoE half of a draft block: the gate, the routed experts and the
    /// shared expert. `bs` rows at a time (the host runs `bs` rows through the
    /// same `moe_forward`).
    ///
    /// # TP geometry (this is a TP-split MoE, not a replicated one)
    ///
    /// The draft's ATTENTION is replicated (every rank holds the whole wq_b /
    /// wo_a / head and maps the GLOBAL head/group/vocab geometry), but its MoE is
    /// split exactly like the backbone's — the loader cuts each routed expert's
    /// `inter` axis by `world` and the shared expert's w1/w3 rows / w2 columns
    /// when `DSV41_SHARED_TP` is on. So the expert kernels here are sized by the
    /// LOCAL `inter_local` and the shared pair by `sh_il`, and every rank's
    /// `moe_out` is only a PARTIAL sum: the all-reduce at the tail (with
    /// `end_round`, the same pair `DevChain::moe_reduce` issues) is what makes
    /// the block identical on every rank. Without it each rank would carry
    /// `1/world` of the MoE — silently, which is worse than a fault.
    fn draft_moe(&mut self, s: usize, ld: &LayerDev) -> Result<()> {
        let cfg = self.cfg;
        let (dim, bs, world, rank) = (self.dim, self.bs, self.world, self.rank);
        let (inter_local, sh_il) = (self.inter_local, self.sh_il);
        let layer = cfg.n_layers + s;
        let (n_routed, topk) = cfg.moe_config(layer);
        let topk = topk.max(1);
        let n_routed = n_routed.max(1);
        // Which ranks contribute the shared expert (the backbone's rule verbatim):
        // all of them under DSV41_SHARED_TP, where each holds its own
        // `inter / world` slice; rank 0 alone under the replicated layout, where
        // a second contributor would make the all-reduce sum it `world` times.
        let stp = crate::dsv41::weights::shared_expert_tp();
        let shared_rank = if stp { true } else { rank == 0 };

        // ---- gate + route ----
        // The gate is bf16 and the GEMV is M=1, so `bs` rows are `bs` launches —
        // the same call the backbone's MoE makes.
        let gate_w = need(&ld.gate_w, "mtp.*.ffn.gate.weight")?;
        for r in 0..bs {
            self.dev.gemv_bf16(
                gate_w.ptr() as *const c_void,
                (self.xn.ptr as *const f32).wrapping_add(r * dim),
                (self.scores.ptr as *mut f32).wrapping_add(r * n_routed),
                n_routed as i32,
                dim as i32,
            )?;
        }
        let gate_bias = ld.gate_bias.as_ref().map(|b| b.as_f32());
        self.dev.route_topk(
            self.scores.as_f32(),
            gate_bias.unwrap_or(std::ptr::null()),
            self.route_w.ptr as *mut f32,
            self.route_idx.ptr as *mut i32,
            std::ptr::null_mut(),
            bs as i32,
            n_routed as i32,
            topk as i32,
            cfg.norm_topk_prob,
            cfg.route_scale,
            2, // sqrtsoftplus, per the checkpoint's routing
        )?;

        // The expert weights are fp4; the input row is quantised once for all
        // slots (the reference re-quantises it per expert, which is pure waste).
        self.dev.quant_fp4(
            self.xn.ptr as *const f32,
            self.xq4.ptr as *mut u8,
            self.xsc4.ptr as *mut f32,
            bs as i32,
            dim as i32,
            32,
            true,
        )?;

        let ne = ld.experts.len();
        if ne >= 2 {
            let (a, b) = (&ld.experts[0], &ld.experts[1]);
            let d = |x: *mut c_void, y: *mut c_void| (y as i64) - (x as i64);
            let strides = (
                a.w1.ptr() as *const u8,
                d(a.w1.ptr(), b.w1.ptr()),
                a.w1_scale.ptr() as *const u8,
                d(a.w1_scale.ptr(), b.w1_scale.ptr()),
                a.w3.ptr() as *const u8,
                d(a.w3.ptr(), b.w3.ptr()),
                a.w3_scale.ptr() as *const u8,
                d(a.w3_scale.ptr(), b.w3_scale.ptr()),
                a.w2.ptr() as *const u8,
                d(a.w2.ptr(), b.w2.ptr()),
                a.w2_scale.ptr() as *const u8,
                d(a.w2_scale.ptr(), b.w2_scale.ptr()),
            );
            let ids = self.route_idx.ptr as *const i32;
            if !ld.experts_ilv && self.dev.supports_moe_batch() {
                // The batched expert launchers' `rows` argument is validation
                // only - the grid is output-width based, so each call computes
                // ONE activation row (the finding recorded in
                // `DevChain::moe_rows`'s doc). Issue per row, with that row's
                // packed fp4 bytes, its `ids`/`route_w` slice and its own
                // `[row][slot][act_slot]` block - the same shape `moe_rows`
                // uses. (The historical call passed rows = bs and silently
                // computed row 0 only.)
                let act_slot = (2 * inter_local) as i64;
                let row_pitch = (topk * 2 * inter_local) as usize;
                for r in 0..bs {
                    self.dev.expert_gate_up_fp4_batched(
                        self.xq4.as_u8().wrapping_add(r * (dim / 2)),
                        self.xsc4.as_f32().wrapping_add(r * (dim / 32)),
                        (self.ex_act_b.ptr as *mut f32).wrapping_add(r * row_pitch),
                        act_slot,
                        1,
                        dim as i32,
                        inter_local as i32,
                        cfg.swiglu_limit,
                        topk as i32,
                        strides.0,
                        strides.1,
                        strides.2,
                        strides.3,
                        strides.4,
                        strides.5,
                        strides.6,
                        strides.7,
                        ids.wrapping_add(r * topk),
                        0,
                    )?;
                }
                for r in 0..bs {
                    self.dev.swiglu_limit_batched(
                        (self.ex_act_b.ptr as *mut f32).wrapping_add(r * row_pitch),
                        1,
                        inter_local as i32,
                        cfg.swiglu_limit,
                        act_slot,
                        topk as i32,
                    )?;
                }
                for r in 0..bs {
                    self.dev.expert_down_reduce_fp4_batched(
                        (self.ex_act_b.ptr as *const f32).wrapping_add(r * row_pitch),
                        act_slot,
                        (self.moe_out.ptr as *mut f32).wrapping_add(r * dim),
                        1,
                        dim as i32,
                        inter_local as i32,
                        (self.route_w.ptr as *const f32).wrapping_add(r * topk),
                        1,
                        topk as i32,
                        strides.8,
                        strides.9,
                        strides.10,
                        strides.11,
                        ids.wrapping_add(r * topk),
                    )?;
                }
            } else {
                self.dev.zero(&self.moe_out)?;
                for slot in 0..topk {
                    let w = (self.route_w.ptr as *const f32).wrapping_add(slot);
                    self.dev.expert_gate_up_fp4_indirect(
                        self.xq4.as_u8(),
                        self.xsc4.as_f32(),
                        self.ex_act.ptr as *mut f32,
                        bs as i32,
                        dim as i32,
                        inter_local as i32,
                        cfg.swiglu_limit,
                        strides.0,
                        strides.1,
                        strides.2,
                        strides.3,
                        strides.4,
                        strides.5,
                        strides.6,
                        strides.7,
                        ids,
                        slot as i32,
                    )?;
                    self.dev.swiglu_limit(
                        self.ex_act.ptr as *mut f32,
                        bs as i32,
                        inter_local as i32,
                        cfg.swiglu_limit,
                    )?;
                    self.dev.expert_down_fp4_indirect(
                        self.ex_act.ptr as *const f32,
                        self.moe_out.ptr as *mut f32,
                        bs as i32,
                        dim as i32,
                        inter_local as i32,
                        w,
                        strides.8,
                        strides.9,
                        strides.10,
                        strides.11,
                        ids,
                        slot as i32,
                    )?;
                }
            }
        } else {
            self.dev.zero(&self.moe_out)?;
        }

        // ---- shared expert (one row at a time: its w1/w3 pair shares the fp8
        // activation, and the swiglu output is a per-row [sh_il] block) ----
        // Only the rank(s) that actually contribute it run this half: all of them
        // under DSV41_SHARED_TP (each computing its own `inter / world` slice of
        // the Rows-sharded w1/w3), rank 0 alone under the replicated layout
        // (weights.rs::shared_expert_tp). `shared_rank == false` means the whole
        // block — including the merge — is skipped, so no stale `shared_out` can
        // leak into the all-reduce.
        let sh_w = if shared_rank {
            match (
                ld.shared_w1.as_ref(),
                ld.shared_w1_scale.as_ref(),
                ld.shared_w3.as_ref(),
                ld.shared_w3_scale.as_ref(),
                ld.shared_w2.as_ref(),
                ld.shared_w2_scale.as_ref(),
            ) {
                (Some(a), Some(b), Some(c), Some(d), Some(e), Some(f)) => Some((a, b, c, d, e, f)),
                _ => None,
            }
        } else {
            None
        };
        if let Some((w1, w1s, w3, w3s, w2, w2s)) = sh_w {
            self.quant1(self.xn.ptr as *const f32, bs * dim)?;
            for r in 0..bs {
                let a = self.xq.as_u8().wrapping_add(r * dim);
                let asc = self.xsc.as_f32().wrapping_add(r * dim / 32);
                // `sh_il` (inter/world under SHARED_TP, inter otherwise) is the
                // width THIS rank's w1/w3 actually have — asking for `inter_local`
                // here is what walked 8x past the local slice (`inter_local` is
                // 320 under TP8, the tensors hold 288 rows).
                self.dev.gemm_fp8_mx(
                    a,
                    asc,
                    w1.as_u8(),
                    w1s.as_u8(),
                    std::ptr::null(),
                    self.ex_act.ptr as *mut f32,
                    1,
                    sh_il as i32,
                    dim as i32,
                )?;
                self.dev.gemm_fp8_mx(
                    a,
                    asc,
                    w3.as_u8(),
                    w3s.as_u8(),
                    std::ptr::null(),
                    (self.ex_act.ptr as *mut f32).wrapping_add(sh_il),
                    1,
                    sh_il as i32,
                    dim as i32,
                )?;
                self.dev.swiglu_limit(
                    self.ex_act.ptr as *mut f32,
                    1,
                    sh_il as i32,
                    cfg.swiglu_limit,
                )?;
                self.quant1(self.ex_act.ptr as *const f32, sh_il)?;
                // w2 is Cols-sharded: [dim, sh_il] locally, so its reduction runs
                // over this rank's slice and the OUTPUT is a partial [dim].
                self.dev.gemm_fp8_mx(
                    self.xq.as_u8(),
                    self.xsc.as_f32(),
                    w2.as_u8(),
                    w2s.as_u8(),
                    std::ptr::null(),
                    (self.shared_out.ptr as *mut f32).wrapping_add(r * dim),
                    1,
                    dim as i32,
                    sh_il as i32,
                )?;
            }
            self.dev.add_inplace(
                &self.moe_out,
                &self.shared_out,
                (bs * dim) as i64,
            )?;
        }

        // ---- the MoE all-reduce ----
        // Both halves above are TP-split, so `moe_out` is a PARTIAL sum and the
        // full block is its sum over the ranks. The draft must produce the SAME
        // block on every rank (the verify path re-runs it under each rank's own
        // chain), so this is not optional — and `end_round` pairs with it exactly
        // as `DevChain::moe_reduce` does, releasing the round before the next
        // block's reduce reuses the staging slots.
        if world > 1 {
            let Some(c) = self.comm.as_ref() else {
                // Fail loudly: skipping the AR would look like a working draft
                // whose MoE is `1/world` of the truth, and the accept-length
                // statistics would silently collapse.
                return Err(FerriteError::Config(format!(
                    "dspark: world={world} but no collective is attached — the draft's MoE \
                     all-reduce cannot run (call DsparkDev::set_comm before draft_forward)"
                )));
            };
            c.all_reduce_inplace(
                self.moe_out.ptr as *mut c_void,
                bs * dim * std::mem::size_of::<f32>(),
            )?;
            c.end_round();
        }
        Ok(())
    }

    /// `forward_head`: the first block's collapse, the head, then the
    /// sequential Markov loop (`dsv41_dspark_markov_head`, one launch per step).
    fn draft_head(&mut self) -> Result<()> {
        let cfg = self.cfg;
        let (dim, bs, hc, vocab, mr) = (self.dim, self.bs, self.hc, self.vocab, self.mr);
        let norm = need(&self.w.dspark_norm, "mtp.last.norm.weight")?;
        let head = need(&self.w.head, "head.weight")?;
        let markov_embed = need(&self.w.markov_embed, "mtp.last.markov_head.embed.weight")?;
        let markov_head = need(&self.w.markov_head, "mtp.last.markov_head.head.weight")?;
        let confidence_proj = need(&self.w.confidence_proj, "mtp.last.confidence_head.proj.weight")?;

        // h = hc_pre(x, premix) then rmsnorm(dspark_norm)
        // The premix is the LAST BLOCK's returned pre_mix — its ffn_pre mix,
        // exactly what the official forward_head receives
        // (`h, pre_mix = layer(...); forward_head(h, pre_mix, ...)` — the same
        // convention the backbone's own final collapse uses, `h = layer.hc_pre(h,
        // pre_mix)`). The block loop leaves it in `pre_in`. (The intermediate
        // attn_pre variant was a misreading of the backbone's premix_slot(1);
        // the official source settles it.)
        self.dev.hc_collapse(
            self.h.ptr as *const f32,
            self.pre_in.ptr as *const f32,
            self.collapse.ptr as *mut f32,
            bs as i32,
            hc as i32,
            dim as i32,
        )?;
        self.dump_unit("collapse", self.collapse.ptr as *const f32, &[bs, dim]);
        self.dev.rmsnorm(
            self.collapse.ptr as *const f32,
            norm.as_f32(),
            self.normed.ptr as *mut f32,
            bs as i32,
            dim as i32,
            cfg.norm_eps,
        )?;
        self.dump_unit("normed", self.normed.ptr as *const f32, &[bs, dim]);

        // logits = head @ normed, one row at a time.
        //
        // ⚠️ COST: the checkpoint's head is bf16 [vocab, dim] = 1.29 GiB, and
        // the chain keeps the ACTIVATION in f32 (casting it to bf16 costs ~3
        // bits on a 129280-way near-tie argmax — see chain_dev's head site), so
        // each row is a separate f32-activation GEMV and the head is read `bs`
        // times: ~5 x 350 us. That single term is most of the draft budget. A
        // multi-row head kernel (m = bs, f32 activation, W reused across rows)
        // would remove the 5x; it does not exist yet.
        for r in 0..bs {
            match head.dtype.as_str() {
                "BF16" => self.dev.gemv_bf16(
                    head.ptr(),
                    (self.normed.ptr as *const f32).wrapping_add(r * dim),
                    (self.logits.ptr as *mut f32).wrapping_add(r * vocab),
                    vocab as i32,
                    dim as i32,
                )?,
                _ => self.dev.gemv_f32(
                    head.as_f32(),
                    (self.normed.ptr as *const f32).wrapping_add(r * dim),
                    (self.logits.ptr as *mut f32).wrapping_add(r * vocab),
                    vocab as i32,
                    dim as i32,
                )?,
            }
        }

        // The Markov loop. `ids[0]` is the backbone's token; each launch biases
        // `logits[step]`, samples `ids[step + 1]` and scores `confidence[step]`.
        //
        // The unit dump takes `logits_row0` BEFORE this loop: the Markov head
        // biases the rows in place, so after the loop `logits[0]` is the biased
        // row, not the head's raw output the reference records.
        self.dump_unit("logits_row0", self.logits.ptr as *const f32, &[vocab]);
        for step in 0..bs {
            self.dev.dspark_markov_head(
                self.logits.ptr as *mut f32,
                self.collapse.ptr as *const f32,
                markov_embed.as_f32(),
                markov_head.as_f32(),
                confidence_proj.as_f32(),
                self.ids.ptr as *mut i32,
                self.confidence.ptr as *mut f32,
                dim as i32,
                vocab as i32,
                mr as i32,
                step as i32,
                self.mk_partial.ptr as *mut u64,
                self.mk_ctr.ptr as *mut u32,
            )?;
        }
        // the drafted block the sampler produced: `[t0, d1..d_bs]`, the
        // reference's `output_ids`.
        self.dump_unit_i32("ids", self.ids.ptr as *const i32, &[bs + 1]);
        Ok(())
    }

    /// Write the main stream's KV row for the draft block into its window ring
    /// at `pos % win` (dspark.rs::dspark_attention, the `mk` half).
    fn seed_window(&mut self, s: usize, pos: usize) -> Result<()> {
        let cfg = self.cfg;
        let (dim, hd) = (self.dim, self.hd);
        let rd = cfg.rope_head_dim;
        let ld = &self.w.mtp[s];
        let wkv = need(&ld.wkv, "mtp.*.attn.wkv.weight")?;
        let wkv_s = need(&ld.wkv_scale, "mtp.*.attn.wkv.scale")?;
        let kv_norm = need(&ld.kv_norm, "mtp.*.attn.kv_norm.weight")?;

        self.quant1(self.main_x.ptr as *const f32, dim)?;
        self.dev.gemm_fp8_mx(
            self.xq.as_u8(),
            self.xsc.as_f32(),
            wkv.as_u8(),
            wkv_s.as_u8(),
            std::ptr::null(),
            self.mk.ptr as *mut f32,
            1,
            hd as i32,
            dim as i32,
        )?;
        self.dev.rmsnorm(
            self.mk.ptr as *const f32,
            kv_norm.as_f32(),
            self.mk.ptr as *mut f32,
            1,
            hd as i32,
            cfg.norm_eps,
        )?;
        self.rope_at(self.mk.ptr as *mut f32, 1, hd as i32, 1, pos as i32, 1, false)?;

        // ⚠️ The destination is a HOST-computed address (pos % win). That is
        // fine for a plain launch but frozen by a CUDA-graph capture; the
        // fix is `Device::ring_append`, which derives the slot from a device
        // counter (the draft has no such counter of its own yet).
        let slot = pos % self.win;
        let dst = (self.window[s].ptr as *mut f32).wrapping_add(slot * hd);
        debug_assert!(rd > 0);
        self.dev
            .memcpy_d2d(dst as *mut c_void, self.mk.ptr, hd * 4)
    }

    /// Forward RoPE for the `bs` draft queries: query `r` sits at
    /// `pos + 1 + r` — the NEXT position after the anchor, one row per step
    /// (sglang arbitration: `positions_2d = prefix_lens + arange(...)` with
    /// prefix_lens already counting the anchor, i.e. row r = anchor_pos + 1 + r.
    /// The historical `pos + bs + r` misread the host comment's `seqlen` as the
    /// block size — it is the SEQUENCE length; the two differ by bs-1 = 4
    /// positions, enough to scramble the whole attention).
    /// Every head of a query shares that position (the main chain's `step = 0`
    /// convention).
    fn rope_queries(&mut self, x: *mut f32, pos: usize) -> Result<()> {
        let (bs, nh) = (self.bs, self.nh);
        for r in 0..bs {
            self.rope_at(
                x.wrapping_add(r * nh * self.hd),
                nh as i32,
                self.hd as i32,
                0,
                pos as i32 + r as i32,
                1,
                false,
            )?;
        }
        Ok(())
    }

    /// The inverse RoPE over the attention output, same positions as
    /// [`Self::rope_queries`].
    fn rope_queries_inv(&mut self, x: *mut f32, pos: usize) -> Result<()> {
        let (bs, nh) = (self.bs, self.nh);
        for r in 0..bs {
            self.rope_at(
                x.wrapping_add(r * nh * self.hd),
                nh as i32,
                self.hd as i32,
                0,
                pos as i32 + r as i32,
                1,
                true,
            )?;
        }
        Ok(())
    }

    /// `apply_rope` with the position uploaded into `pos_base`.
    ///
    /// The kernel derives each row's position from a DEVICE counter
    /// (`*base * mul + off + row * step`), which is what makes the backbone's
    /// rope graph-capturable; `pos_base` is the draft's own little counter.
    /// ⚠️ The upload is a blocking H2D per call; a device-side draft position
    /// counter would remove both it and the capture hazard.
    #[allow(clippy::too_many_arguments)]
    fn rope_at(
        &mut self,
        x: *mut f32,
        rows: i32,
        row_len: i32,
        step: i32,
        base: i32,
        mul: i32,
        inverse: bool,
    ) -> Result<()> {
        let cfg = self.cfg;
        let rd = cfg.rope_head_dim;
        // The kernel reads `*base` as an `int`; upload the INTEGER's four bytes.
        self.dev
            .upload_bytes_at(&self.pos_base, &base.to_le_bytes())?;
        let cos = self.cos.unwrap();
        let sin = self.sin.unwrap();
        self.dev.apply_rope(
            x,
            cos,
            sin,
            rows,
            row_len,
            rd as i32,
            (rd / 2) as i32,
            self.pos_base.ptr as *const std::os::raw::c_int,
            mul,
            0,
            step,
            inverse,
        )
    }

    /// Rebuild `idxs` when the valid window length changes.
    ///
    /// `dspark_topk_idxs` (dspark.rs) is `[0, n_win) ++ [win, win + bs)` repeated
    /// over the batch and over every draft query; with `bs == 1` batch that is
    /// `bs` identical rows, so the whole matrix is built on the host and cached
    /// until `n_win` moves (it settles at `win` after the first `win` positions,
    /// so the steady-state decode path is H2D-free).
    fn ensure_idxs(&mut self, pos: usize) -> Result<()> {
        let (win, bs) = (self.win, self.bs);
        let n_win = win.min(pos);
        if self.n_win_cached == n_win {
            return Ok(());
        }
        // COMPACT layout: the block's rows sit at [n_win, n_win+bs) of
        // `all_kv` (only the live window rows are copied), NOT at [win, ..) —
        // the historical `win + i` indexed the full-width layout while the
        // buffer was compact, reading past every window row once pos < win-1.
        let mut row: Vec<i32> = Vec::with_capacity(n_win + bs);
        row.extend((0..n_win).map(|i| i as i32));
        row.extend((0..bs).map(|i| (n_win + i) as i32));
        let row_bytes: Vec<u8> = row.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut flat = Vec::with_capacity(row_bytes.len() * bs);
        for _ in 0..bs {
            flat.extend_from_slice(&row_bytes);
        }
        self.dev.upload_bytes_at(&self.idxs, &flat)?;
        self.n_win_cached = n_win;
        Ok(())
    }

    /// `Device::quant_fp8` of one contiguous row of `k` f32 into `xq`/`xsc`.
    fn quant1(&self, src: *const f32, k: usize) -> Result<()> {
        self.dev.quant_fp8(
            src,
            self.xq.ptr as *mut u8,
            self.xsc.ptr as *mut f32,
            1,
            k as i32,
            32,
            true,
        )
    }

    fn upload_i32(&self, dst: &DevBuf, v: &[i32]) -> Result<()> {
        let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        self.dev.upload_bytes_at(dst, &bytes)
    }

    // ---- golden per-unit dump (`DSV41_DSPARK_UNIT_DUMP=1`) ----------------
    //
    // Every call below is a no-op unless a capture is live, i.e. one branch on
    // `self.unit` — the gate itself is resolved once, at `draft_forward`'s top.
    // A failing D2H is LOGGED, never propagated: a debug dump must not answer an
    // error for a forward whose numbers are otherwise fine.

    /// Record one `f32` unit, `dims` naming its shape.
    #[inline]
    fn dump_unit(&mut self, name: &str, ptr: *const f32, dims: &[usize]) {
        if let Some(u) = self.unit.as_mut() {
            if let Err(e) = u.push_f32(self.dev, name, ptr, dims) {
                eprintln!("[dspark] unit dump {name}: {e}");
            }
        }
    }

    /// [`Self::dump_unit`] for a per-block unit: `{prefix}{idx}`. The name is
    /// only formatted when a capture is live, so the gate-off path allocates
    /// nothing.
    #[inline]
    fn dump_unit_idx(&mut self, prefix: &str, idx: usize, ptr: *const f32, dims: &[usize]) {
        if let Some(u) = self.unit.as_mut() {
            let name = format!("{prefix}{idx}");
            if let Err(e) = u.push_f32(self.dev, &name, ptr, dims) {
                eprintln!("[dspark] unit dump {name}: {e}");
            }
        }
    }

    /// Record one `i32` unit (the Markov sampler's `ids`).
    #[inline]
    fn dump_unit_i32(&mut self, name: &str, ptr: *const i32, dims: &[usize]) {
        if let Some(u) = self.unit.as_mut() {
            if let Err(e) = u.push_i32(self.dev, name, ptr, dims) {
                eprintln!("[dspark] unit dump {name}: {e}");
            }
        }
    }
}
