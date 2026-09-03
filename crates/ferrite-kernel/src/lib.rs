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

    /// Indexer top-k: per query token, select `topk` KV tokens by
    /// `q_idx · k_idx` score (paged in the real impl; the CPU reference
    /// takes the full `[t, i_proj]` k cache and returns topk indices).
    /// `q_idx: [n, i_proj]`, `k_idx: [t, i_proj]` → `idx: [n, topk]`.
    fn indexer_topk(&self, q_idx: &Tensor, k_idx: &Tensor, topk: usize, idx: &mut Tensor) -> Result<()>;

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
}
