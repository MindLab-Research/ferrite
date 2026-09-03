//! ferrite-scheduler: PDAF (Prefill-Decode-Attention-FFN) separation.
//!
//! Compile-time static plan (layer op sequence derived once from the model
//! config — ferrite's specialisation philosophy) + a runtime router that
//! maps a `ScheduledBatch` onto execution domains:
//!
//! - **P** (prefill): chunked prompt processing, operates the linear-state
//!   recurrence in chunk form + DSA prefill attention.
//! - **D** (decode): one token per running seq, recurrent step + DSA
//!   sparse attention over the paged latent KV.
//! - **A/F**: the attention vs FFN operator boundary is exposed in the
//!   static plan (`OpKind`), so an engine can place them on separate
//!   executors (Q-First style); the CPU single-thread engine runs them in
//!   order, but the plan carries the affinity for the B300 deployment.
//!
//! Phase transfer events (P→D) model the KV/state move: locally a memcpy,
//! on B300 a Mooncake-style transfer — the engine interprets them.

use ferrite_model::{AttnKind, Glm53FlashConfig, MlpKind, build_layer_plans};

/// Execution phase (the P/D of PDAF).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Phase {
    Prefill,
    Decode,
}

/// Operator family of a scheduled layer step — the A/F of PDAF.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpKind {
    /// Attention-side compute: GatedDeltaNet recurrence or DSA attention
    /// (+ its indexer). Affinity: memory-bandwidth heavy.
    Attention(AttnKind),
    /// FFN-side compute: dense SwiGLU or MoE routing+experts.
    Ffn(MlpKind),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerOp {
    pub layer_idx: usize,
    pub op: OpKind,
    /// True when this op is the layer's attention half (runs before the
    /// FFN half of the same layer in the static order).
    pub is_first_half: bool,
}

/// The full static execution plan for one model pass — compile once,
/// run every step. Order: for each layer, attention op then FFN op.
#[derive(Debug, Clone)]
pub struct StaticPlan {
    pub ops: Vec<LayerOp>,
    pub num_layers: usize,
    pub num_linear: usize,
    pub num_dsa: usize,
    pub num_moe: usize,
    pub num_dense: usize,
}

impl StaticPlan {
    pub fn from_config(cfg: &Glm53FlashConfig) -> Self {
        let mut ops = Vec::with_capacity(cfg.num_hidden_layers * 2);
        let mut num_linear = 0;
        let mut num_dsa = 0;
        let mut num_moe = 0;
        let mut num_dense = 0;
        for plan in build_layer_plans(cfg) {
            match plan.attn {
                AttnKind::Linear => num_linear += 1,
                AttnKind::Dsa => num_dsa += 1,
            }
            match plan.mlp {
                MlpKind::Dense => num_dense += 1,
                MlpKind::Moe => num_moe += 1,
            }
            ops.push(LayerOp {
                layer_idx: plan.layer_idx,
                op: OpKind::Attention(plan.attn),
                is_first_half: true,
            });
            ops.push(LayerOp {
                layer_idx: plan.layer_idx,
                op: OpKind::Ffn(plan.mlp),
                is_first_half: false,
            });
        }
        StaticPlan {
            ops,
            num_layers: cfg.num_hidden_layers,
            num_linear,
            num_dsa,
            num_moe,
            num_dense,
        }
    }

    /// Ops of one phase in order. (Static plan is phase-agnostic; the same
    /// op list runs in both P and D — only the kernel flavours differ.)
    pub fn layer_ops(&self, layer_idx: usize) -> impl Iterator<Item = &LayerOp> {
        self.ops.iter().filter(move |o| o.layer_idx == layer_idx)
    }
}

/// A unit of work routed to one execution phase.
#[derive(Debug, Clone)]
pub struct PrefillWork {
    pub seq: u64,
    /// Number of prompt tokens in this chunk (the P executor consumes
    /// exactly this many from the prompt cursor).
    pub chunk_tokens: usize,
}

#[derive(Debug, Clone)]
pub struct DecodeWork {
    pub seq: u64,
    /// Token offset in the (prompt+output) context this step attends from.
    pub context_pos: usize,
}

/// A phase-transfer event: P finished the prompt of `seq`, its linear
/// states + DSA KV are ready to be consumed by the D executor.
///
/// `dst` carries the per-request destination metadata for the 2D reshard
/// (decode → prefill bootstrap, rides with the request so heterogeneous
/// decode groups can coexist): `(n_dcp, d)` selects the page filter
/// (`p mod n_dcp == d`, local slot `p div n_dcp`) the prefill side applies
/// during transfer. `None` = single-node / no DCP (identity reshard).
#[derive(Debug, Clone)]
pub struct TransferEvent {
    pub seq: u64,
    /// Per-request DCP destination (page-shard filter for the 2D reshard).
    pub dst: Option<ferrite_kv::shard::DstKvInfo>,
}

/// One engine step routed by phase — the output of the PDAF router.
#[derive(Debug, Clone, Default)]
pub struct PdafStep {
    pub prefill: Vec<PrefillWork>,
    pub decode: Vec<DecodeWork>,
    pub transfers: Vec<TransferEvent>,
}

/// Routes `ScheduledBatch` (from ferrite-batch) onto P/D work lists and
/// emits transfer events for sequences whose prefill completes this step.
pub struct PdafRouter {
    pub plan: StaticPlan,
}

impl PdafRouter {
    pub fn new(cfg: &Glm53FlashConfig) -> Self {
        PdafRouter { plan: StaticPlan::from_config(cfg) }
    }

    /// Route one scheduled batch. `prompt_len`/`prefilled`/`context_len`
    /// come from the batch scheduler's sequences; the caller supplies a
    /// closure to avoid a circular dep on ferrite-batch types.
    pub fn route(
        &self,
        batch: &ferrite_batch::ScheduledBatch,
        seq_info: &dyn Fn(u64) -> (usize, usize, usize), // (prompt_len, prefilled, context_len)
    ) -> PdafStep {
        let mut step = PdafStep::default();
        for &(seq, chunk) in &batch.prefill {
            let (prompt_len, prefilled, _) = seq_info(seq);
            let chunk_tokens = chunk.min(prompt_len - prefilled);
            step.prefill.push(PrefillWork { seq, chunk_tokens });
            if prefilled + chunk_tokens == prompt_len {
                step.transfers.push(TransferEvent { seq, dst: None });
            }
        }
        for &seq in &batch.decode {
            let (_, _, context_len) = seq_info(seq);
            step.decode.push(DecodeWork { seq, context_pos: context_len.saturating_sub(1) });
        }
        step
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrite_batch::BatchScheduler;
    use ferrite_model::Glm53FlashConfig;

    #[test]
    fn static_plan_test_config() {
        let cfg = Glm53FlashConfig::test_config();
        let plan = StaticPlan::from_config(&cfg);
        assert_eq!(plan.num_layers, 4);
        assert_eq!((plan.num_linear, plan.num_dsa), (2, 2));
        assert_eq!((plan.num_dense, plan.num_moe), (2, 2));
        assert_eq!(plan.ops.len(), 8, "4 layers x (attn+ffn)");
        // op order: layer0 attn, layer0 ffn, layer1 attn, ...
        assert_eq!(plan.ops[0].layer_idx, 0);
        assert!(plan.ops[0].is_first_half);
        assert!(!plan.ops[1].is_first_half);
        // attention half of layer 1 is DSA in the test config
        assert!(matches!(plan.ops[2].op, OpKind::Attention(AttnKind::Dsa)));
        // mlp of layer 2 is MoE
        assert!(matches!(plan.ops[5].op, OpKind::Ffn(MlpKind::Moe)));
    }

    #[test]
    fn static_plan_production() {
        let cfg = Glm53FlashConfig::production_config();
        let plan = StaticPlan::from_config(&cfg);
        assert_eq!((plan.num_linear, plan.num_dsa), (34, 11));
        assert_eq!((plan.num_dense, plan.num_moe), (3, 42));
        assert_eq!(plan.ops.len(), 90);
    }

    #[test]
    fn router_splits_phases_and_emits_transfer() {
        let cfg = Glm53FlashConfig::test_config();
        let router = PdafRouter::new(&cfg);
        let mut bs = BatchScheduler::new(4, 16);
        let a = bs.submit(vec![1, 2, 3, 4], 9, 4).unwrap();
        let b = bs.submit(vec![5, 6, 7, 8, 9, 10], 9, 4).unwrap();
        // snapshot helper: (prompt_len, prefilled, context_len) per seq id —
        // built before routing so the closure borrows the snapshot, not &bs.
        let snap = |bs: &BatchScheduler| -> std::collections::HashMap<u64, (usize, usize, usize)> {
            let mut m = std::collections::HashMap::new();
            for s in bs.running().chain(bs.finished()) {
                m.insert(s.id, (s.prompt.len(), s.prefilled, s.context_len()));
            }
            m
        };
        // step 1: both admitted, budget 16 covers both prompts fully
        let batch1 = bs.next_batch();
        let info1 = snap(&bs);
        let info = |id: u64| -> (usize, usize, usize) { *info1.get(&id).unwrap() };
        let step1 = router.route(&batch1, &info);
        assert_eq!(step1.prefill.len(), 2);
        assert!(step1.decode.is_empty());
        // both prompts fully prefilled this chunk -> 2 transfers
        assert_eq!(step1.transfers.len(), 2);
        for &(seq, chunk) in &batch1.prefill {
            bs.advance_prefill(seq, chunk).unwrap();
        }
        bs.post_step(&batch1);
        // step 2: decode both
        let batch2 = bs.next_batch();
        let info2 = snap(&bs);
        let info_b2 = |id: u64| -> (usize, usize, usize) { *info2.get(&id).unwrap() };
        let step2 = router.route(&batch2, &info_b2);
        assert!(step2.prefill.is_empty());
        assert_eq!(step2.decode.len(), 2);
        assert!(step2.transfers.is_empty());
        // context_pos points at the last token
        let da = step2.decode.iter().find(|d| d.seq == a).unwrap();
        assert_eq!(da.context_pos, 3, "prompt len 4 -> attending from idx 3");
        let db = step2.decode.iter().find(|d| d.seq == b).unwrap();
        assert_eq!(db.context_pos, 5);
        // partial chunk: seq with 10-token prompt and 4-token budget
        let mut bs2 = BatchScheduler::new(1, 4);
        let c = bs2.submit(vec![0; 10], 9, 4).unwrap();
        let b3 = bs2.next_batch();
        let info3 = snap(&bs2);
        let info_c = |id: u64| -> (usize, usize, usize) { *info3.get(&id).unwrap() };
        let step3 = router.route(&b3, &info_c);
        assert_eq!(step3.prefill[0].chunk_tokens, 4);
        assert!(step3.transfers.is_empty(), "prompt not done yet");
        let _ = c;
    }
}
