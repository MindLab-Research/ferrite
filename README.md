# ferrite

Rust-native inference engine for **GLM-5.3-Flash** (hybrid GatedDeltaNet linear attention + DSA sparse attention + dense/MoE FFN), built for PDAF-disaggregated serving with compile-time model specialisation.

**Design contract** (why it exists):

| SGLang-style tax | ferrite |
|---|---|
| runtime model dispatch (vtables, config lookups) | static layer plan compiled once from the model config |
| kernel backend = runtime trait objects | `Engine<B: KernelBackend>` generic monomorphisation — zero dispatch |
| PD/CP/EAGLE as patches on monolithic serving | PDAF phases (prefill/decode/attention/FFN) are first-class: `StaticPlan` op graph + `PdafRouter` phase routing + transfer events |
| CUDA graph as an afterthought | `GraphCapable` contract in the backend trait from day one; decode op-sequence stability is *tested* (graph bucketing preconditions) |

## Current state (perf-b1)

Decode (TP=4, B300 GPUs 4–7, GLM-5.3-Flash): non-MTP 49 tok/s → MTP speculative (draft=2/verify n=3) **64.8 tok/s** @200-step window (accept ≈2.4 tokens/step). One CUDA graph per step; A→B ping-pong recorded in-graph; single-kernel accept commit. See `AGENTS.md` for the full flag table, demo configs, profiling workflow (ncu; nsys 2024.2.3 does not work on CUDA 13.2/B300), and GPU test discipline.

## Layout

```
crates/
  ferrite-types     tensor/shape/dtype core (f32 golden storage)
  ferrite-model     GLM-5.3-Flash config → layer plan → weight layout;
                    safetensors loader (BF16/FP8→f32), 37,267-tensor layout
  ferrite-kernel    KernelBackend trait (monomorphised) + CpuBackend
                    (numerical golden standard) + GraphCapable (capture/
                    verify) + CudaBackend (--features cuda, FFI)
  ferrite-kv        dual-mode state: linear-attn fixed-size state slab +
                    GatedDeltaNet conv tails + DSA paged latent/indexer KV
  ferrite-batch     continuous batching (chunked prefill, admission)
  ferrite-scheduler PDAF: static op plan + phase router + transfer events
  ferrite-exec      Engine<B> — full forward (34 lin + 11 DSA + 3 dense +
                    42 MoE layers), MHC-residual approx, greedy decode
kernels/cuda/       ferrite_kernels.cu (sm_103a, B300) + build.sh
```

## Numerical contract

The CPU backend is the golden standard; every backend must match it:

- Gated DeltaNet recurrence (channel-wise gate, hand-derived golden test)
- chunked-prefill == single-chunk state (conv tails carry boundaries)
- DSA expanded K/V (absorbed-at-load MLA equivalence)
- MoE noaux-tc sigmoid routing
- op-sequence stability across decode steps (CUDA-graph precondition)

CPU tests: `cargo test` (74 workspace tests, all green).

## B300 build & validation (no GPU needed to *build*)

```bash
# 1. kernels (nvcc only — compile does not touch the GPU):
cd kernels/cuda && ./build.sh 103a          # → libferrite_kernels.so (sm_103a, B300)
# verified on 1102 (CUDA 13.0, B300): builds clean, 1.1 MB

# 2. rust workspace (CPU path — runs anywhere):
cargo test
cargo check -p ferrite-kernel --features cuda   # FFI layer compiles

# 3. GPU validation (needs the B300 when it frees up):
#    dlopen the .so, run the golden-diff harness against CpuBackend
```

## Runbook (B300 bring-up, when the card arrives)

1. `./build.sh 103a` on the B300 node (nvcc 13.2).
2. `CudaBackend::with_library("libferrite_kernels.so")`.
3. Golden-diff: run the same random-weight model through `CpuBackend` and
   `CudaBackend`, assert f32 tolerance 1e-5 op-by-op (the CPU tests are the
   oracle; port them to a `--features cuda` integration test).
4. Engine on GPU: `Engine::new(cfg, weights, cuda_backend)` — the PDAF
   loop is backend-agnostic.
5. CUDA graphs: `GraphRunner` capture→verify on CPU proves op-sequence
   stability; the CUDA backend maps the same contract onto
   `cuStreamBeginCapture`/`cuGraphInstantiate`/`cuGraphLaunch`.

## Status / TODO

- [x] CPU engine end-to-end (random weights, 4-layer test config): chunked
      prefill, decode, MoE routing, DSA top-k, linear attention, CUDA-graph
      op-sequence stability — 16 test targets green
- [x] CUDA kernels compiled for sm_103a (B300-verified, nvcc 13.2)
- [x] CudaBackend FFI skeleton (host↔device v1; golden-diff harness next)
- [x] MHC exact hyper-connections (mhc.rs — sglang-aligned hc_pre/hc_post
      with sinkhorn, 4-flow residual, hc_expand/contract lifecycle; 10/10)
- [x] Fused-weight aliases (apply_fused_aliases: fused_qkvbfg_a →
      qkv/b/f_a/g_a, fused_fg_b → f_b/g_b, fused_qkv_a_with_mqa → q_a/kv_a,
      gate_up → gate/up — real checkpoint loading)
- [x] Shard architecture skeleton (ferrite-kv::shard): compile-time
      ShardLayout + ReshardPlan for CP Layer-Split (prefill, div+remainder
      layer assignment) × DCP (decode, page filter `p mod n_dcp`, local
      slot `p div n_dcp`) 2D reshard; GLM-5.3-Flash hybrid routing — 34
      GatedDeltaNet layers state pass-through (fixed state, no page dim)
      + 11 DSA layers page-filtered; per-request DstKvInfo on
      TransferEvent (heterogeneous decode groups). Single-node identity
      plan today; multi-rank is the next seam.
- [x] WYF-parallel chunkwise GatedDeltaNet (Rust golden vs sequential
      recurrence + CUDA kernel, 32-token chunks, 32x fewer launches)
- [x] DCP attention merge infrastructure (ferrite-kernel::dcp): per-rank
      `sparse_attn_partial` (local softmax + max-shifted LSE) + `lse_merge`
      (stable log-sum-exp) + `split_pages_round_robin` — equivalence-tested
      (N-way partial merge == full attention for N=2/3/4; ±1000 score
      stability; empty shard LSE=-inf contributes 0; permutation invariant
      for EAGLE verify determinism). The CPU merge is the golden reference
      for the multi-rank (all-gather + LSE all-reduce) implementation.
- [x] PDAF distributed execution (ferrite-exec::distributed): CP LayerSplit
      prefill (owner-rank pools via cp_layer_range + engine state export),
      page-level 2D reshard (DSA: `p mod n_dcp` filter + slot compress,
      GatedDeltaNet: state pass-through replication), DCP decode (page
      shards + per-rank partial attention + LSE merge, global top-k over
      replicated k_idx, MHC 4-flow). End-to-end equivalence: CP=2/DCP=2
      and CP=4/DCP=4 decode == single-node Engine. 74 workspace tests green.
- [x] Tensor parallelism (ferrite-exec::tp): `shard_weights_tp` (GatedDeltaNet
      head split — qkv/conv/b/A_log/dt/o_norm/f_b/g_b head-sliced, o_proj
      column-split; DSA q_b/kv_b head-split, kv_a/indexer shared (MLA latent
      is head-agnostic); dense MLP row/col split; MoE expert-sliced EP-style
      with replicated router; shared-expert row-split; MHC/norms/embedding
      replicated) + `TpCluster` (layer-synchronous shards, attn/ffn partial →
      all-reduce at the NCCL boundary points, CPU-simulated collectives).
      Equivalence: TP=2 and TP=4 decode == TP=1 == stock Engine. MoE expert
      slicing is exact — each (token, expert) pair computed by exactly one
      rank, union = full routing.
- [x] NCCL bindings (ferrite-kernel::nccl, `--features cuda`): dlopen
      libnccl.so.2 (no link-time NCCL), `ncclCommInitAll` (single-process
      multi-GPU TP) + `ncclCommInitRank` (multi-rank bootstrap),
      AllReduce/AllGather/Broadcast/ReduceScatter on f32 device buffers,
      stream-async semantics matching the CPU simulation's sync points.
      The B300 data plane swaps these 1:1 for the simulated
      `all_reduce_sum`; RS+AG pair available for bandwidth-optimal residual.
- [x] Axis algebra for composable parallelism (ferrite-kv::axes): three
      attention/data axes — **Q** (prefill CP: query token segments, merge =
      row-gather, no compute), **Kv** (decode CP: KV pages, merge = stable
      LSE), **Head** (TP: weights slice, merge = concat for attention
      outputs / sum for o_proj-down_proj partials). **Phase-aware** module
      plan: the same DSA module shards Q on prefill and KV on decode; GDN's
      prefill token axis is a *pipeline* (WYF chunk state chain), not
      independent partials. Communication groups are per-axis sub-meshes
      (`group_along`). Module × phase × axis matrix is the single source of
      truth drivers consult for slicing and collectives.
- [x] 3D composability proven (ferrite-kernel::dcp):
      `q_kv_head_3d_merge_equals_full` — a (q_seg × kv_page × head) partial
      grid merges to *exactly* the full attention, in **both orders**
      (KV-LSE-then-head-concat == head-concat-then-KV-LSE: the axis merges
      commute). `kv_head_2d_merge_equals_full` covers the decode shape
      (DCP×TP). Primitives: `split_heads`, `split_q_segments`,
      `concat_heads` (+ existing `sparse_attn_partial`/`lse_merge`/
      `split_pages_round_robin`). Any (Cp × Dcp × Tp) topology reduces to
      the same three merge kinds — arbitrary meshes are expressible.
- [ ] GPU golden-diff + perf tuning (matmul tiling, WYF-parallel chunkwise
      GatedDeltaNet, fused SwiGLU, device-resident tensors) — iterates on
      the B300 once available
- [ ] Multi-GPU TP on device: TpCluster over CudaBackend shards + real
      NCCL collectives (comm init per device, all-reduce at the layer
      boundaries the simulation already exercises)
- [ ] Real checkpoint load (safetensors reader ready; BF16/FP8→f32 golden
      path, native-dtype path for GPU)
- [ ] Cross-node distributed: multi-rank KernelBackend (TP/CP/DCP/EP
      topology), 2D reshard transfer worker (Mooncake RDMA + page-filter
      at source) — shard.rs types are the seam, single-node identity plan
      validates the protocol today
- [ ] PDAF multi-process (currently in-process phases; the router/transfer
      events are the seam for the split-process deployment)
