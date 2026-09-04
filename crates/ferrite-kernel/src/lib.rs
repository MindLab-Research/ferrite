//! ferrite-kernel: device-agnostic kernel interface + CPU reference backend.
//!
//! Design contract (the core of ferrite's compile-time specialisation):
//! - `KernelBackend` is a **generic bound, not a dyn trait**. The engine is
//!   `Engine<B: KernelBackend>` and every call site is monomorphised per
//!   backend — zero vtable/dispatch tax, the same shape SGLang pays for
//!   backend compat.
//! - Tensors are passed as `&Tensor` (inputs) + `&mut Tensor` (outputs,
//!   pre-allocated by the engine's buffer planner). Kernels never allocate.
//! - The CPU backend is the **numerical golden reference**: B300 (sm_100a)
//!   correctness is judged against it. Performance backends (CUDA) implement
//!   the same trait and must match within tolerance.
//!
//! GLM-5.3-Flash op set: GatedDeltaNet linear attention (chunkwise prefill /
//! recurrent decode), DSA sparse attention (indexer top-k + latent MLA),
//! dense + MoE FFN (sigmoid noaux-tc routing, SwiGLU with clamp), MHC
//! hyper-connections.

pub mod cpu;
#[cfg(feature = "cuda")]
pub mod cuda;
pub mod dcp;
pub mod graph;
#[cfg(feature = "cuda")]
pub mod nccl;
pub mod wyf;

pub use cpu::CpuBackend;
#[cfg(feature = "cuda")]
pub use cuda::CudaBackend;
pub use dcp::{lse_merge, sparse_attn_partial, split_pages_round_robin, PartialAttn};
pub use graph::{GraphCapable, OpRecord, OpTrace};

/// Shard index of the current fan_out worker thread (thread-local; 0 when
/// not inside fan_out). Probe dumps append _r{i} to isolate rank-sharded
/// outputs — 4 TP ranks writing the same probe file produced cross-rank
/// diffs that looked like 100% divergence (head-sharded outputs differ BY
/// DESIGN across ranks; the "divergence" was two different ranks' shards).
thread_local! {
    static SHARD_IDX: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Set the current fan_out worker's shard index (called by fan_out's spawn).
pub fn set_shard_idx(i: usize) {
    SHARD_IDX.with(|c| c.set(i));
}

/// Current shard index (rank-isolated probe dump paths).
pub fn shard_idx() -> usize {
    SHARD_IDX.with(|c| c.get())
}

use ferrite_types::{Result, Tensor};

/// Device-agnostic kernel interface for GLM-5.3-Flash inference.
///
/// Layout conventions:
/// - Matrices are row-major `[rows, cols]`.
/// - `w` in matmul is the weight layout `[out_features, in_features]`
///   (PyTorch `Linear.weight`), i.e. the op computes `x @ wᵀ`.
/// - Batch dim is the first dim (`[batch, ...]`); decode steps use batch
///   = tokens (`[n, ...]` where n may be 1).
pub trait KernelBackend: Send + Sync {
    // ------------------------------------------------------------------
    // Dense ops
    // ------------------------------------------------------------------

    /// `out[n, out_f] = x[n, in_f] @ w[out_f, in_f]ᵀ` (+ optional bias).
    fn matmul(&self, x: &Tensor, w: &Tensor, bias: Option<&Tensor>, out: &mut Tensor) -> Result<()>;

    /// RMSNorm over the last dim. `w: [dim]`.
    fn rmsnorm(&self, x: &Tensor, w: &Tensor, eps: f32, out: &mut Tensor) -> Result<()>;

    /// RMSNorm with an additive gate per element (linear-attn output norm).
    /// `x: [..., dim]`, `gate: [..., dim]` (sigmoid applied inside), `w: [dim]`.
    fn gated_rmsnorm(&self, x: &Tensor, gate: &Tensor, w: &Tensor, eps: f32, out: &mut Tensor)
        -> Result<()>;

    /// SwiGLU with GLM's clamp: `down(silu_n(clamp(gate)) * clamp(up))`,
    /// `gate_up: [n, 2*inter]` (gate first, then up), `limit` = swiglu_limit.
    fn swiglu_limited(&self, gate_up: &Tensor, limit: f32, out: &mut Tensor) -> Result<()>;

    // ------------------------------------------------------------------
    // Gated DeltaNet linear attention (34/45 layers)
    // ------------------------------------------------------------------

    /// Causal short conv over q/k/v fused channels.
    /// `x: [n, 3*proj]` (conv input = raw projected qkv), `w: [3*proj, conv]`,
    /// `state_in/state_out: [3*proj, conv-1]` (carried conv tail),
    /// `out: [n, 3*proj]`.
    fn causal_conv1d(
        &self,
        x: &Tensor,
        w: &Tensor,
        state_in: &Tensor,
        out: &mut Tensor,
        state_out: &mut Tensor,
    ) -> Result<()>;

    /// Single-step Gated DeltaNet recurrence (decode / chunkwise inner).
    /// Shapes (per call, all heads fused):
    /// - `q, k: [n, heads, dk]`, `v: [n, heads, dv]`  (dk == dv here)
    /// - `beta: [n, heads]` in (0,1)   (sigmoid of b_proj)
    /// - `gate: [n, heads, dk]`        (channel-wise forget gate, sigmoid of
    ///   the f_a→f_b projection — FLA standard per-channel decay)
    /// - `a_log: [heads]`              (per-head log decay rate)
    /// - `state_in/state_out: [heads, dk, dv]`
    /// - `out: [n, heads, dv]`
    ///
    /// Recurrence (per head, gated delta rule with channel-wise decay):
    /// ```text
    /// decay_i = exp(gate_t[h, i] * a_h)     // a_h = -exp(a_log_h) < 0
    /// S[i,:] = S[i,:] * decay_i             // per-channel row scaling
    /// S      = S - beta_t * k (kᵀ S)        // delta rule erasure
    /// S      = S + beta_t * k vᵀ            // write
    /// o_t    = qᵀ S
    /// ```
    fn gated_deltanet_step(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        beta: &Tensor,
        gate: &Tensor,
        a_log: &Tensor,
        state_in: &Tensor,
        out: &mut Tensor,
        state_out: &mut Tensor,
    ) -> Result<()>;

    /// Chunkwise prefill = loop over the chunk with the same recurrence
    /// (CPU reference: exact; CUDA backend implements the WYF-parallel form
    /// and must match this within tolerance).
    /// Same shapes as `gated_deltanet_step` with n = chunk tokens.
    fn gated_deltanet_chunk(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        beta: &Tensor,
        gate: &Tensor,
        a_log: &Tensor,
        state_in: &Tensor,
        out: &mut Tensor,
        state_out: &mut Tensor,
    ) -> Result<()>;

    // ------------------------------------------------------------------
    // DSA sparse attention (11/45 layers, nope-only MLA + indexer)
    // ------------------------------------------------------------------

    /// Indexer top-k (real GLM-5.3-Flash checkpoint semantics):
    /// `q_idx: [n, H*D]` per-head indexer queries (wq_b @ q_lora, H=index_n_heads),
    /// `k_idx: [t, D]` shared index keys (k_norm(wk @ x), D=index_head_dim),
    /// `w: [n, H]` per-head score weights (weights_proj @ x).
    /// score[i, j] = Σ_h w[i,h] · relu(q_idx[i,h,:] · k_idx[j,:]) / √D → topk over j.
    /// `ctx0`: causal guard — query row i may only select keys j <= ctx0 + i
    /// (prefill rows attend causally; decode rows n=1 see all t keys).
    /// `idx: [n, topk]` selected token indices (shared across heads).
    fn indexer_topk(
        &self,
        q_idx: &Tensor,
        k_idx: &Tensor,
        w: &Tensor,
        topk: usize,
        ctx0: usize,
        idx: &mut Tensor,
    ) -> Result<()>;

    /// Sparse MLA attention over selected tokens.
    /// `q: [n, heads, d_q]` (nope only for 5.3-Flash),
    /// `kv: [t, kv_lora]` latent cache,
    /// `w_up: per-head up-projection handled by caller as pre-absorbed
    /// q/k/v` — the CPU reference computes exact scores directly:
    /// `out[n, heads, v_dim]`.
    /// `idx: [n, topk]` from indexer_topk.
    fn sparse_mla_attn(
        &self,
        q: &Tensor,
        k_nope: &Tensor,
        v: &Tensor,
        idx: &Tensor,
        out: &mut Tensor,
    ) -> Result<()>;

    // ------------------------------------------------------------------
    // MoE (42/45 layers; sigmoid noaux-tc routing)
    // ------------------------------------------------------------------

    /// Router: sigmoid scores + noaux-tc bias + top-k + renormalise.
    /// `logits: [n, experts]`, `bias: [experts]` (the router e_score bias),
    /// returns `probs: [n, topk]` (already scaled by routed_scaling) and
    /// `ids: [n, topk]`.
    fn moe_route(
        &self,
        logits: &Tensor,
        bias: &Tensor,
        topk: usize,
        routed_scaling: f32,
        probs: &mut Tensor,
        ids: &mut Tensor,
    ) -> Result<()>;

    /// One expert FFN for a gathered batch (CPU reference: loop experts).
    /// `x: [m, hidden]` rows already routed to this expert, `gate_w/up_w/down_w`
    /// are `[inter, hidden] / [inter, hidden] / [hidden, inter]`.
    fn expert_ffn(
        &self,
        x: &Tensor,
        gate_w: &Tensor,
        up_w: &Tensor,
        down_w: &Tensor,
        swiglu_limit: f32,
        out: &mut Tensor,
    ) -> Result<()>;

    // ------------------------------------------------------------------
    // Sampling
    // ------------------------------------------------------------------

    /// Argmax over the last dim (greedy decode).
    fn argmax_lastdim(&self, logits: &Tensor, out: &mut Tensor) -> Result<()>;

    /// Softmax over the last dim (numerically stabilised).
    fn softmax_lastdim(&self, logits: &Tensor, out: &mut Tensor) -> Result<()>;

    // ------------------------------------------------------------------
    // MHC hyper-connections (sglang _mhc_pre/_mhc_post exact port; the
    // golden CPU math lives in ferrite-exec/src/mhc.rs — this is the
    // backend surface so the GPU path can run it in one kernel instead of
    // the per-token host loops)
    // ------------------------------------------------------------------

    /// MHC pre-step: `residual_flat [s, n*h]` + `fn_w [mix, n*h]` +
    /// `scale [3]` + `base [mix]` → `(li [s,h], post [s,n], comb [s,n,n])`.
    /// pre_i = sigmoid(mixes_i*s0 + base_i) + hc_eps; li = Σ pre_i·x_i;
    /// post = 2·sigmoid; comb = sinkhorn-normalised mix block.
    fn hc_pre(
        &self,
        residual_flat: &Tensor,
        fn_w: &Tensor,
        scale: &Tensor,
        base: &Tensor,
        rms_eps: f32,
        hc_eps: f32,
        sinkhorn_iters: usize,
    ) -> Result<(Tensor, Tensor, Tensor)>;

    /// MHC post-step: `out[t,i,j] = post[t,i]·x[t,j] + Σ_k comb[t,k,i]·res[t,k,j]`
    /// (note the comb transpose — transformers matmuls combᵀ·residual).
    /// `x [s,h]`, `residual [s,n,h]`, `post [s,n]`, `comb [s,n,n]` → `[s,n,h]`.
    fn hc_post(&self, x: &Tensor, residual: &Tensor, post: &Tensor, comb: &Tensor) -> Result<Tensor>;

    /// Downcast for the device-chain path (GDN/DSA device pipelines that
    /// compose ops at the DevBuf level — zero host round-trips in-layer).
    /// Returns the CudaBackend when this is one; None for every other
    /// backend (callers fall back to the Tensor-level ops).
    #[cfg(feature = "cuda")]
    fn as_cuda(&self) -> Option<&crate::cuda::CudaBackend> {
        None
    }
}
