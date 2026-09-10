# DeepSeek-V4.1-Flash — GPU bring-up status

Everything in this file was verified against the **real 48-shard checkpoint**
(`/opt/dlami/nvme/models/DeepSeek-V4.1-Flash`) on **b300-4**, not inferred.

## Where it stands

A single-GPU run of the chain works end to end on real weights:

```
[dsv41] weights loaded: 118.6 GiB in 86.7s        (3 layers + embed + head, engram tables skipped)
[dsv41] prefill 1 tokens in 0.12s
[dsv41] decode 6 tokens in 0.10s (58.2 tok/s)
--- generated ---
我的朋友你是谁/us921osererg
```

The leading tokens are **coherent Chinese** with only 3 of 40 layers and none of
the long-range components — i.e. embedding → hyper-connection → MLA → MoE →
head are numerically sound. The gibberish after a few tokens is attributable to
the parts that are deliberately not wired yet, listed below.

`DSV41_LAYERS=N` truncates the model for bring-up (`DSV41_STATS=1` prints
per-stage rms/min/max/NaN; `DSV41_DEBUG_SYNC=1` names the kernel that faults).

## Ground truth established against the checkpoint

Each of these contradicted an assumption and would have been a silent error:

| item | reality |
|---|---|
| tensor names | **no `model.` prefix** (`layers.6.attn.wq_a.weight`) |
| `attn.wo_a.weight` | `[8192, 4096]` fp8 **with `.scale`** — `[o_groups*o_lora, hpg*head_dim]`, per-group contiguous so the row stride equals `k`; per-group GEMMs run straight off this layout |
| `o_lora_rank` | **1024** (the reference dataclass default of 256 is wrong) |
| `ffn.gate.weight` | **BF16, no scale sidecar** → the gate GEMM is bf16 and routing is a separate step (`dsv41_route_topk`), not the fused fp8 kernel |
| `index_topk` | **512** |
| layer roles | `kv_source = [2,8,14,20]`, `index_source = [2,8,14,20,24,28,32,36]`, `compress_ratio = 0/0 then 2 (l2-19), 1 (l20-39)` |
| weight dtypes | norms / hc params / gate bias / embed are **BF16**; they feed f32 kernels, so they are widened losslessly at load (see below) |

## Bugs found and fixed on the way (each had a specific signature)

1. **`used[]` out-of-bounds in the router** — a flag array sized `[topk]` indexed
   by the *expert id* (≤ n_experts). Illegal memory access. Consumed experts are
   now marked by writing `-INFINITY` into the score array.
2. **fp8 activation scratch sized by `dim`** — the attention output projection
   quantises `n_heads*head_dim = 32768` elements, so the fp8 buffer and its
   per-32 scales ran past their allocations. Illegal memory access.
3. **bf16 bytes handed to f32 consumers** — `ffn.gate.bias`, `norm.weight`, the
   `hc_*` parameters and `embed.weight` are bf16; reading them as f32 doubles
   the length read. This corrupted values *and* ran off the end. Fixed by
   widening bf16→f32 at load — an **exact** conversion, not a dequantisation:
   the quantised weights (fp8+ue8m0, fp4+e8m0) are still uploaded verbatim.
   Only tensors a bf16 tensor-core GEMM consumes stay bf16.
4. **`ferrite_add_inplace` does not exist** — the symbol is `ferrite_add`. The
   full dlsym set (28 names) is now checked against the built object; the only
   absent entries are that one (fixed) and `dsv41_window_append` (optional, not
   implemented yet).
5. **The window selection took the invalid slots** — `window_topk_idxs` returns a
   full `window`-wide row in ring order (high slots first) with empty slots
   marked −1; the chain kept the *leading* `pos+1`, which for `pos < window` are
   exactly the invalid ones. `sparse_attn` then selected nothing and produced
   **exactly 0.0** at every decode step (`DSV41_STATS` showed rms=0.0000 while
   pos=0 was a healthy 0.1124). The whole row is now handed over; it skips
   negatives itself.

Two process lessons, both re-learned the hard way:

* **A stale `.so` invalidated a whole debugging session** — `bash build.sh 103a`
  must run after every sync, and the artifact timestamps must be checked against
  the source (`.so` was older than `dsv41_route.cu` while I chased a bug I had
  already fixed).
* **A test shape that cannot reach the bug is not a test** — the router passed at
  `n_experts=6` and only fell over at the production 384.

## Not yet wired (the reason the tail of the text degrades)

1. **Compressor + indexer** — attention is window-only (128 slots). The release
   compresses KV every `ratio` positions and selects `index_topk=512` of them;
   without it there is no long-range context at all.
2. **Engram write-back** — 189 GiB of n-gram tables, row-sharded per rank
   (`engram_part_rows`), gathered by `dsv41_engram_hash` + `dsv41_engram_gather`
   and applied by `dsv41_engram_apply`. Loaded weights are currently skipped.
3. **Tensor parallelism** — mandatory, not optional: 286 GiB of weights (engram
   excluded) against ~180 GiB per B300. The loader already slices by
   `(world, rank)`; the chain still needs the all-reduce points.
4. **DSpark drafting** — draft layers are loaded (`n_mtp_layers=3`) but unused.

## TP plan (the reference's parallel layout, to be mirrored)

* `wq_a`/`wo_a` column-parallel (output split), `wq_b`/`wo_b` row-parallel
  (input split) — the column→row *pair* needs no reduction in between, the row
  half needs one after it.
* MoE experts are expert-parallel → the routed sum and the shared expert both
  need an all-reduce.
* `head` is column-parallel over the vocabulary.
* The all-reduce can be built without NCCL: one process, one CUDA context per
  rank (8 devices), peer copies + a local sum kernel. AR payloads here are tiny
  (`dim` floats per site).

## Compressor / indexer — ground truth for wiring them up

**⚠️ Blocker found at the wiring step:** `dsv41_compressor`'s ABI takes fp8
weight pointers (`const uint8_t* wkv, const uint8_t* wkv_scale, ...`), but the
checkpoint's compressor weights are **BF16** and the reference runs the pooling
in **fp32**. The kernel as written cannot consume them. Two ways forward, both
small: (a) give `compressor_pool_kernel`/`compressor_state_kernel` a bf16 entry
(the state carry and the pooling themselves are dtype-agnostic — `k` is fp32 in
the state buffers already), or (b) keep the kernel for the GEMM-free part and
feed it activations computed by two `lin_bf16` calls in the chain. Do not "fix"
this by quantising the bf16 weights to fp8 — that is both lossy and contrary to
the standing rule that quantisation formats come from the checkpoint.

Established against the checkpoint and the reference's `Compressor`:

* **Compressor tensors exist only on the kv sources 🌐 `[2,8,14,20]`**; indexer
  tensors only on `[2,8,14,20,24,28,32,36]`. (The config derives both correctly;
  layer 6 has neither — the earlier "layer 6 has none" observation is the
  expected shape of the model, not a spec gap.)
* **Every compressor weight is BF16, not fp8** — `compressor.wkv.weight` is
  `[512, 5120]` BF16 on all four layers, `compressor.wgate.weight` exists only
  where `ratio > 1` (layers 2/8/14), and layer 20 (`ratio == 1`) has no gate.
  The reference *promotes* them to fp32 at runtime for the pooling; the stored
  values are bf16, so widening to f32 at load is lossless and reproduces that.
* Indexer: `wq_b` is **fp8 + scale** `[4096, 1280]`, while `wk` `[128, 512]`,
  `weights_proj` `[32, 5120]` and `k_norm` `[128]` are BF16.
* Reference semantics, per step:
  * `ratio == 1` → `latent = norm(wkv(x))`, one token per group;
  * `ratio > 1` → `kv = wkv(x)`, `score = wgate(x)`, the completed group of
    `ratio` tokens is pooled as `Σ kv_t · softmax_t(score)`, and an incomplete
    group waits in `kv_state`/`score_state` (shaped `[b, ratio, head_dim]`);
  * the latent is returned **pre-RoPE** — the indexer consumes the unrotated
    form and attention applies RoPE afterwards.
* Both halves already exist as kernels: `compressor_state_kernel` (stashes the
  trailing partial group for prefill, one slot per decode step) and
  `compressor_pool_kernel` (gated pooling) — read their bodies for the exact
  argument order. What is missing is the chain wiring: call them on kv sources,
  append the latent to the KV buffer at row `window + compress_len`, publish the
  key for the indexer, and on non-sources reuse the source layer's published
  selection (`is_index_source` / `is_kv_source` already say which is which).

## Indexer — exact reference semantics (for the wiring step)

`Indexer.forward(x, qr, latent, start_pos, offset)` — `x` is the normed attention
input, `qr` the q_lora stream, `latent` this layer's RoPE-free compressed latent
(`None` while its group is still filling), `offset` where the selection goes:

1. **Index keys** (only when `owns_k` and a latent came out): `k = k_norm(wk(latent))`,
   RoPE on the trailing `rope_head_dim` lanes using **the group's first token's
   position** (`group j → j*ratio`), then fp4 quantisation, then publish into the
   key cache at `start_pos // ratio`.
2. **Index queries**: `q = wq_b(qr)` (a ColumnParallel over the q_lora stream,
   `[4096, 1280]` fp8+scale in the checkpoint), RoPE with the *token's* positions,
   then fp4 quantisation.
3. `weights = weights_proj(x) * (softmax_scale * n_heads**-0.5)` — note it is
   scaled here, not inside the score.
4. `score = Σ_h relu(q_h · k) * w_h` over the published keys `[.., end_pos//ratio]`,
   and **an all-reduce over ranks** (TP note).
5. Visibility: `compress_lens = end_pos // ratio` (decode) — a block is visible
   once the query has passed its last token.
6. The top-k result is written at `offset` (the chain passes `window`, which is why
   the compressed entries land directly after the window block).

Deviation to keep in mind: the reference **fp4-quantises the index q/k**
(`fp4_act_quant`); our `dsv41_indexer_topk` computes them in f32. That is more
accurate but not bit-identical — it can flip which position is picked at the
margins, so it must be validated by the text, not by an equality check.

## Tensor parallel: implemented and verified

One process, one rank per thread. The CUDA runtime binds a device per thread, so
thread `r` calls `cudaSetDevice(r)` and everything it launches lands on device
`r`; each rank opens its own `Device`, loads its own slice (the loader already
slices by `world`/`rank`) and builds its own chain, with a `Barrier` keeping the
ranks in lockstep. Communication is peer copies plus a local reduction over the
staging slots **in rank order** — identical on every rank, no NCCL.

Verified: `--tp 4` and `--tp 8` produce the same leading token ids as a
single-GPU run (`[116169,122294,107153,30155,52695,1538]`).

### Sharding policy (what each rank holds)

| tensor | rule | why |
|---|---|---|
| routed experts | expert-parallel **and** cut along `inter` | 96% of the bytes; per-rank ~34 GiB at tp=8 |
| expert `w2` | cut along **columns** | it is `[dim, inter/2]`, so `inter` is its column dim |
| `wq_b` | ColumnParallel over heads | local heads = `n_heads/world`; the chain uses the local count for the q path, RoPE, `sparse_attn` and the inverse RoPE |
| `attn_sink` | per-head slice | matches the local head count |
| `wo_a` | ColumnParallel per group | rows `groups*o_lora/world`; a rank's contiguous row block **is** its group block, which is what keeps it aligned with the head block of `wq_b` |
| `wo_b` | RowParallel (input split) | each rank reduces over its `o_lora` slice; all-reduced after |
| compressor / indexer / `wq_a` / `wkv` / gate | replicated | plain `Linear` in the reference; the indexer's score would need an all-reduce before its top-k (score and top-k are one kernel today) |
| `embed` / `head` | replicated | the reference splits the vocabulary, which needs a gather — `embed_expand_dev_kernel` substitutes row 0 for an out-of-range id instead of skipping it, so a rank whose slice lacks the token would contribute a real (wrong) row |

### Five TP-only bugs (each invisible at tp=1)

1. `cudaDeviceEnablePeerAccess` needs the *peer's* context to exist; calling it at
   bind time fails silently and the first peer copy faults.
2. `cudaMemcpyPeerAsync`'s stream/direction rules did not fit one-thread-per-device;
   the synchronous form works and the payloads are tens of KB.
3. Expert `w2` carries the sharded `inter` axis on its **columns**; sharding it by
   rows sliced its output dim and the down GEMM walked off its weight.
4. `local_shape` reported expert/group tensors **unsliced** while the loader had
   sliced them — the metadata contradicted the bytes.
5. The expert kernels were handed the **global** `inter` width although their
   weights are `[inter/world, ...]`. (The first attempt at that edit silently did
   not apply; it is now verified by grepping the call sites.)

Process lessons re-learned: an async fault surfaces at the next checked call, so
the named op is often innocent — the peer copy was blamed for an expert-kernel
overrun. And a patch that does not match its anchor fails **silently**; grep the
call sites afterwards.

## Numerical parity against the reference — the method that works

The user's suggestion was decisive: **run the reference implementation's maths on
the same input and diff value-by-value**. One parity test found the root cause in
a single shot that hours of text-watching had not.

`tests/hc_parity.rs` runs `dsv41_hc_mixes` on the reference's deterministic input
and prints pre/post/comb; `/tmp/refcmp/hc_ref.py` computes the same numbers with
numpy straight from the checkpoint. Result after the fix: **7 significant digits
agree** (PRE `0.9425751` vs `0.94257504`, POST `0.031103252` vs
`3.1103250e-02`, COMB row0 `0.7791158` vs `7.7911586e-01`).

### The bug it found (root cause of input-independent output)

`hc_mixes` computed `ss = Σ x[c]²` and reduced it **within each warp only** — the
cross-warp step was missing, so at `blockDim=256` the sum covered 1/8 of the
elements and `inv = rsqrt(ss/hc_dim + eps)` was **sqrt(8) = 2.83x too large**.
Every hc coefficient was then wrong, which mis-mixed the entire residual stream
in every layer: the output had no relation to the input at any prompt length.
This is the same class as the GLM "8-warp reduce" bug already in this repo's
notes — **reductions must be checked for the cross-warp step**.

### Other reference mismatches found and fixed the same session

| item | was | now |
|---|---|---|
| `original_seq_len` | read a top-level key that does not exist → **0**, which DISABLES YaRN (`if original_seq_len > 0`) | reads `rope_scaling.original_max_position_embeddings` = **65536** ✓ |
| Sinkhorn order | ended with a ROW normalisation | reference order: softmax+eps → col → (iters-1)×(row, col), ending COL ✓ |
| compress rope | one table (theta=10000) for everything | separate table at `compress_rope_theta` = 160000 ✓ |
| compressed latent | stored in the KV ring **without RoPE** | rotated at the group's first position before the store ✓ |
| query RoPE | `step=1` → head i at position pos+i | `step=0` (all heads at the same position) ✓ |
| `layer()` return | the attention block's pre-mix | the FFN block's pre-mix (what the next layer collapses with) ✓ |
| indexer | only kv-source layers ran it | every index-source layer runs its own; keys read from the kv owner ✓ |
| window indices | uploaded only on the placeholder path | uploaded every step (the indexer path left them stale) ✓ |
| KV ring size | window rows only (compressor wrote past the end) | window + max_compress rows ✓ |
| wo_a offsets | GLOBAL group offsets into a LOCAL slice | local offsets ✓ |
| weight loading | ~92k individual `cudaMalloc`, per-tensor `File::open`, missing `h.begin` | one pooled allocation per layer, mmap'd shards, DMA (`cudaMemcpy`/`2D`), device-side bf16 widening ✓ |
| `bind_to` | enabled peer access at bind time (races context creation) | binds only; peer access enabled after all contexts exist ✓ |

## Remaining work

1. **One more numerical bug** (output still degenerate: `時刻_...` then blanks).
   The parity method is the way to find it: diff the **attention path** next
   (q/kv projections → sparse attention → inverse rope → wo_a/wo_b) against a
   numpy reference built from the same weights, exactly as was done for the hc
   chain. `sparse_attn`'s sink handling and the fp8 activation quantisation are
   the two least-verified pieces left.
2. **Engram write-back** (currently skipped; its absence changes quality but the
   user confirms it is optional, so it is not the cause of the degenerate text).
3. **Performance to 200 tok/s single-request (5 ms/step)** — not started; the
   correctness work was blocking. Where the time currently goes:
   * **~4 device-wide syncs per layer** (downloading hc coefficients, MoE routing
     indices/weights, the compressor's publish flag) ≈ 160-200 syncs/step. The hc
     coefficients can stay on the device (ping-pong two buffers — the kernel
     already takes device pointers), and the publish decision is a pure function
     of `pos % ratio` computable on the host. **This is the first thing to fix.**
   * the collective is synchronous (peer copy + device sync + 2 barriers per
     call, 2 calls per layer = 80/step).
   * MoE dispatch is host-driven: ~6 experts × 7 launches per layer.
   * no CUDA graph yet — ~60 launches/layer × 40 = 2400 launches/step.

## Verification recipes

```bash
# unit / contract tests (no GPU)
cargo test -p ferrite-dsv41

# real-weight loader contract (CPU only, reads the 48 shard headers)
DSV41_MODEL_DIR=/opt/dlami/nvme/models/DeepSeek-V4.1-Flash \
  cargo test -p ferrite-dsv41 --test real_checkpoint -- --nocapture

# real-weight fp8 GEMM against a CPU reference (1 GPU)
DSV41_MODEL_DIR=... DSV41_KERNELS=$PWD/kernels/cuda/libferrite_kernels.so \
  CUDA_VISIBLE_DEVICES=0 cargo test --release -p ferrite-dsv41 --test real_gemm -- --nocapture

# chain bring-up  (both artifacts must be rebuilt after any sync!)
cd kernels/cuda && bash build.sh 103a && cd ../.. && cargo build --release
DSV41_LAYERS=3 DSV41_STATS=1 DSV41_MODEL_DIR=... DSV41_KERNELS=$PWD/kernels/cuda/libferrite_kernels.so \
  CUDA_VISIBLE_DEVICES=0 ./target/release/dsv41-run --prompt "你好" --max-tokens 8
```

---

## 2026-09-11 会话：退化输出的根因定位（engram 是架构组件，不是可选附加）

### 本会话对拍/诊断的全部结论（均有数据）

1. **hc 链已与参考逐位一致（7 位有效数字）** ✓ — `hc_mixes` 的 `ss`(Σx²) 曾只做 warp 内归约、
   漏跨 warp → blockDim=256 时 `inv = rsqrt(ss/hc_dim+eps)` 偏大 √8=2.83× → 所有 pre/post/comb 错。
   （这是"输出与输入无关"的真根因，已修。）
2. **激活量化器与参考完全一致** ✓：`DSV41_TOP5` 诊断 + attn_parity 同输入对拍，
   XSC[0..4] 两边**逐位相同**（0.001953125 / 0.00048828125 …），XQ 字节仅 1 个码差
   （我 numpy 编码器的取整）。`fp8_block_size = 32` 与 config 一致 ✓。
3. **投影输出统计量吻合** ✓：`qr` rms 0.3999 vs 参考 0.3844（4%）；大元素吻合（-0.3162 vs -0.3165），
   小元素（|v|<0.2·rms）差异达 10-20% —— **属 fp8 量化噪声量级**，非结构性 bug。
4. **模型确实响应 prompt** ✓（三个 prompt → 三种不同输出）：`The capital of France is` →
   [3108,3108,3108]；`1+1=` → [19,31,19]（prompt 是 [19,13,19,31] → **复读**）；`你好` → [90133,65,3108]。
   → **token 3108 = `'...ĊĊ'`（字面 "..." + 两个换行）**，不是特殊 token（EOS=1、pad=2）。
5. **top-5 logits 每步都在变且分布合理**（step0: 11.44/11.03/10.76 …）→ logits 无 NaN、非陈旧值、
   head 路径（hc_collapse → rmsnorm → lin_bf16）**已施加最终 norm** ✓。
6. **残差流 rms 逐层 ×55 增长**（L0 0.0596 → L35 3.2882），L35 min/max=±16.7（5×rms 离群）。
   pre-norm 架构下增长本身可能正常，但**未与参考核对**。
7. **窗口/index 尺寸配置正确** ✓（`sliding_window=128` ✓ `index_topk=512` ✓）；
   `head.weight`/`embed.weight` 是 `Shard::Replicated`（weights.rs:83-90，非切分，属**性能项**非正确性项）。

### ★ 剩余退化的主因（本会话新定位）

`config.json` 的 `engram_layer_ids = [1, 14]`，`engram_num_embeddings = [384006168, 384016682]`，
`engram_max_ngram_size = 4 / n_heads = 8 / head_dim = 256`。

参考实现 `model.py:106` 的注释是 **"engram: n-gram hash lookups added into the residual stream
at a few layers"**，且 `model.py:1261-1262` 在**每层 forward 内**：
```python
for i, layer in enumerate(self.layers):
    if layer.engram is not None:      # layer 1 和 14
        ...                            # n-gram 哈希查表 → 加入残差流
```

→ **engram 是架构组件，不是可选附加** ✗（`h = layer.engram(h, ...)` 直接改写 hc 残差流）。

**精确状态（2026-09-11 逐处核实，非推测）**：
| 位置 | 事实 |
|---|---|
| `src/engram.rs` | `TokenMap` / `EngramLayout` / `NgramHashState` **已实现** ✓ |
| `src/ops.rs` | `ops::engram_forward`（CPU 版门控写回）**已实现** ✓ |
| `kernels/cuda/dsv41_glue.cu` | `dsv41_engram_apply`（门控写回 kernel）**已实现** ✓ |
| `kernels/cuda/dsv41_kernels.cu` | `dsv41_engram_hash` / `dsv41_engram_gather` **已实现** ✓ |
| `src/device.rs` | 三个 FFI wrapper **已实现** ✓ |
| `src/load.rs:186-191` | `LayerDev` 的 `engram_wkv`/`q_weight`/`k_weight` 字段**存在且已加载** ✓ |
| **`src/chain_dev.rs` 层循环** | **从不调用 engram** ✗✗（只有 `chain.rs:563-606` 的 CPU 链调 ✓） |
| **`bin/dsv41-run.rs:103,253`** | `skip_prefixes.push("engram.embed.")` → **189 GiB 表未加载** ✗✗ |
| `load.rs:236-237` 注释 | 自陈 "skip the 189 GiB engram tables **while the engram write-back is not yet wired**" ✗ |

→ **设备链（tp=8 跑的路径）里第 1/14 层的残差流完全缺 engram 写回** ✓ = 退化根因 ✓✓。
（CPU 链 `chain.rs` 有 engram 但它的 `row_off = 0` 且注释说"device path all-reduces afterwards"，
即 CPU 版只是参考实现，设备版本就该另写。）

**接线清单（下会话直接做，零件全在）**：
1. `dsv41-run.rs` 删掉两处 `skip_prefixes.push("engram.embed.")` → 表按 row-parallel 加载
   （`Shard::Rows`，`part_rows = ceil(384M/world)` ≈ 24 GB/rank ✓ 275 GB 装得下）
2. `chain_dev.rs::step()`：host 侧用 `NgramHashState::forward_row(...)` 算本步 token 的
   24 个 n-gram id → upload → `dev.engram_hash`（或直接用 host 结果）
3. 层循环里 layer∈{1,14} 时、**在 `self.layer()` 之前**：
   `dev.engram_gather(table, scale, ids, buf, rows=1, n_cols=24, head_dim=256, part_start, part_rows)`
   → **跨 rank all-reduce/AR**（只补非本 rank 的行；24 行 × 256 维 = 24 KB，AR 成本可忽略）
   → `wkv` GEMM（`[6144] × [6144,25600]`，权重已加载 ✓）
   → `dev.engram_apply(h, kv, q_weight, k_weight, null, 1, hc, dim, eps)`
4. 验证：`L1 h` 的 rms 应在 engram 后发生变化；出师表/Paris 文本应开始连贯。

### 下一步（按序）

1. **实现 engram 前向查表**（阻塞项）：n-gram hash（`engram.py` 的 NgramHashState：token_map →
   compressed id → 逐 lookback XOR 乘法 → `% primes[:, i-1]` → + offsets）→ 8 heads × 256 维查表 →
   加进第 1/14 层的残差流。**注意表是跨 rank 切分的**（model.py:108 注释：每 rank 分配
   `ceil(rows/world_size)`）→ 查表需要跨 rank 取行（all-to-all 或每 rank 本地行 + AR）。
2. 之后再做正式参考对拍（跑官方 PyTorch 前几层，比对逐层 h 统计）确认残差 ×55 增长是否正常。
3. 正确性达标后才启动性能工作（目标：单并发 200 tok/s 不开 MTP）——已识别的首批优化点：
   每层 4 次全设备同步（hc 系数/MoE 路由/compressor 标志）、80 次/步同步集合通信、
   主机侧 MoE 派发、无 CUDA graph（~2400 launch/步）。

### 本会话代码改动

- `tests/attn_parity.rs`：在**正确时点** dump 量化结果（第一次量化 + 就地 norm 前的 qr）。
- `bin/dsv41-run.rs`：`DSV41_TOP5=1` 时打印每步 top-4 logits（默认关）。

### engram 前向的完整语义（本会话抄录自 model.py:328-368，实现它只需这些）

**接法**（model.py:1261-1262，在**每层 forward 之前**）：
```python
for i, layer in enumerate(self.layers):
    if layer.engram is not None:                    # 仅 layer 1 / 14
        h = layer.engram(h, engram_hashes[:, :, layer.engram.layer_hash_index, :], engram_mask)
    h, pre_mix = layer(h, start_pos, pre_mix, image_mask)
```
`h` 此刻是 **hc 展开后的 [B, L, hc_mult, dim]** ✓ → engram 直接改写残差流 ✓。

**模块结构**：
```python
self.embed = ParallelEngramEmbedding(layout.num_embeddings[idx], layout.head_dim)   # 表 384M 行 × 256
n_hash_cols = (max_ngram_size - 1) * n_heads          # (4-1)*8 = 24
self.wkv = Linear(n_hash_cols * head_dim, dim * (hc_mult + 1))   # [24*256=6144 -> 5120*5=25600]
self.q_weight, self.k_weight = [hc_mult, dim] 参数（各 4×5120）
```

**forward（逐行等价实现即可）**：
```python
kv = self.wkv(self.embed(hash_ids).flatten(-2))      # [B,L,25600]
key, value = kv.split([hc_mult*dim, dim], -1)        # key [4,5120] per token, value [5120]
key = key.unflatten(-1, (hc, dim)); weight = q_weight * k_weight   # [4,5120]
h = x.float()
rstd = rsqrt(h.square().mean(-1) + eps) * rsqrt(key.square().mean(-1) + eps)   # 各自按 dim 归约
dot  = (h * weight * key).sum(-1) * rstd * dim**-0.5                           # [B,L,hc]
gate = sigmoid(copysign(sqrt(|dot|.clamp_min(1e-6)), dot))                     # 带符号 sqrt
return h + gate.unsqueeze(-1) * value.unsqueeze(-2)                            # value 广播到 hc 份
```

**n-gram 哈希**（engram.py:129-184，`NgramHashState.forward`）：
`token_map[input_ids]` → 写入按位置的 cache → 对 shift∈[0, max_ngram_size) 取 `cache[pos-shift]`
（缺失/越界用 `pad_id=2`）→ `tokens * multipliers` 逐 lookback **XOR** → `% primes[:, i-1]` → `+ offsets`，
输出 **[B, L, n_engram_layers, n_hash_cols=24]**。

**数据量与分布式**：`engram_num_embeddings=[384006168, 384016682]`，head_dim=256 →
两层表合计 **189 GiB**（≈256 维 × 4B 显存/行… 实为 bf16，每 rank 分 `ceil(rows/world)` = ~24 GB/rank ✓ 装得下）。
**查表跨 rank**：每 token 每层只有 24 行 × 256 维 = **24 KB** ✓ → 一次小 all-to-all 即可（数据量可忽略）。
`wkv` 是 25600×6144 = 157M 参数的稠密 fp8 ✓ 已在权重集里（`DSV41_SKIP_ENGRAM_WEIGHTS` 可跳过）。

---

## ★ 2026-09-11：官方 PyTorch 参考实现已可运行（最强验证工具）+ 逐级对拍结论

### 如何运行官方实现（已跑通，整套流程如下）

```bash
# 1) 隔离 venv（系统 torch 是 cpu-only，必须另装）
python3 -m venv /opt/dlami/nvme/dsv41_venv
/opt/dlami/nvme/dsv41_venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128
/opt/dlami/nvme/dsv41_venv/bin/pip install tilelang transformers safetensors sympy numpy tqdm pillow

# 2) 把 HF 检查点转成官方格式（名字映射/分片/engram 表补齐都由它做）
cd <ckpt>/inference && python3 convert.py --hf-ckpt-path <ckpt> \
  --save-path /opt/dlami/nvme/dsv41_mp8 --model-parallel 8 --expert-dtype fp4
# → model{0..7}-mp8.safetensors（每个 67 GB）

# 3) config：官方 ModelArgs 只认自己的字段名，HF config 的键名不同，必须映射
#    hidden_size→dim, moe_intermediate_size→moe_inter_dim, num_hidden_layers→n_layers,
#    num_nextn_predict_layers→n_mtp_layers, num_attention_heads→n_heads,
#    num_experts_per_tok→n_activated_experts, scoring_func→score_func,
#    routed_scaling_factor→route_scale, qk_rope_head_dim→rope_head_dim,
#    rms_norm_eps→norm_eps, sliding_window→window_size,
#    kv_source_layer_ids→kv_source_layers, index_source_layer_ids→index_source_layers,
#    candidate_source_layer_id→candidate_source_layer, engram_pad_token_id→engram_pad_id,
#    dspark_num_experts_per_tok→dspark_n_activated_experts,
#    rope_scaling{...}→original_seq_len/rope_factor/beta_fast/beta_slow,
#    + vision_config{num_hidden_layers→vision_n_layers, hidden_size→vision_dim, ...} + image_token_id
#    （**不映射 vision 会因 state_dict 里有 aligner.*/image_*/gate.bias_vl 而 strict 加载失败**）

# 4) 跑（贪心：generate.py 的 sample() 在 temperature==0 时就是 argmax）
/opt/dlami/nvme/dsv41_venv/bin/torchrun --nproc_per_node=8 generate.py \
  --ckpt-path /opt/dlami/nvme/dsv41_mp8 --config <mapped>.json \
  --input-file /tmp/prompt.txt --max-new-tokens 16 --temperature 0
```

**官方真值（golden，已确认可用）**：
```
Prompt: The capital of France is
Completion: The capital of France is **Paris**.        ← 官方贪心输出，连贯
```
裸 token 序列 `[671,6102,294,8760,344]` 下官方 top5 =
`[(11111, 20.329), (1613, 17.546), (4588, 16.698), (260, 16.568), (16, 16.458)]`，logits rms **2.8198**。

官方打印出的**权威配置**（与我的 config.rs 解析一致 ✓）：
`dim=5120 moe_inter_dim=2304 n_layers=40 n_heads=64 n_activated_experts=6 route_scale=1.5
swiglu_limit=10.0 q_lora_rank=1280 head_dim=512 rope_head_dim=64 norm_eps=1e-20 o_groups=8
o_lora_rank=1024 window_size=128 index_n_heads=32 index_head_dim=128 index_topk=512
candidate_source_layer=20 candidate_topk_blocks=2048 candidate_block_size=8 hc_mult=4
hc_sinkhorn_iters=20 hc_eps=1e-06 engram_layer_ids=[1,14] dspark_block_size=5
dspark_target_layer_ids=[37,38,39]`

### 逐级对拍结果（同 token 序列 `[671,…]`，官方记 pos0、我记第一步）

| 量 | 官方 | 我的 | 判定 |
|---|---|---|---|
| **embed 行 rms (token 671)** | 0.037195 | 0.037195 | **✓ 一致** |
| **L0 hc pre** | [5.9e-05, 3e-06, 1e-06, 0.993102] | [5.9085e-5, 3.48e-6, 1.44e-6, 0.9930845] | **✓ 一致** |
| **L0 hc post** | [0.000522, 0.000144, 0.0, 0.079789] | [0.00052252, 0.00014405, 1.78e-9, 0.0798377] | **✓ 一致** |
| **L0 hc comb** | [0.857548, 0.002101, 0.001035, 0.105167, …] | [0.8575853, 0.0020976, 0.001035, 0.1051372, …] | **✓ 一致** |
| **comb_rowsum** | [0.965851, 1.017994, 1.084354, 0.931798] | [0.9658552, 1.0180061, 1.0843558, 0.931779] | **✓ 一致** |
| **xn (attn_norm 输出)** | 0.020712 | 0.0207 | **✓ 一致** |
| **ffn_in** | 0.126523 | 0.1256 | **✓ 一致（0.7%）** |
| attn_out (o) | 0.774044 | 0.4606 | ✗ 1.68x（**探针口径存疑，见下**）|
| moe_out | 0.145160 | 0.0792 | ✗ 1.83x |
| h (L0) | 0.074201（**5 token 聚合**）| 0.0395（**单 token**）| 口径不同，不可比 |

**结论：hc 全链（pre/post/comb/sinkhorn）、embedding、attn_norm、ffn_in 都与官方一致或吻合到 1% 内** ✓✓。
**分歧集中在 attn_out / moe_out 的幅度** ✗。

**两个探针陷阱（都踩过，记录以免重犯）**：
1. `L0 attn_out(o)`：`attention()` 返回时 `s.o` 已是 **dim 宽**（wo_b 输出），我第一次按 `nh*hd` 读 →
   读到缓冲区尾部陈旧数据（0.1821 这个数是错的 ✗）。有效值是按 `dim` 读的 **0.4606**。
2. `L0 h`：官方一次 prefill 5 个 token、我逐 token → **rms 口径不同**（官方 0.0742 是 5 token 聚合、
   我 0.0395 是单 token）✗。要比就必须两边都取 pos0。

**下一步（用官方做神谕，逐级精确定位）**：
- 在官方 `Attention.forward` 里打印 **pos0 的 q/kv/wo_a 中间量**，与我的同位置数值逐元素对比；
  `ffn_in` 已吻合说明注意力输出的最终效果接近 ↔ `attn_out` 探针的 1.68x 很可能是探针口径而非真误差 ✗
  → 先用 pos0 的**逐元素**（不只看 rms）核对，再判定。
- `moe_out` 的 1.83x 优先查 **shared expert 是否被正确相加**（官方 `n_shared_experts=1`）。

---

## ★ 2026-09-11：用官方实现做神谕，逐级对拍修掉两个真 bug

### 修好的 bug（都有官方对照数据）

1. **`sparse_attn_kernel` 的 dot 只归约了第 0 个 warp** ✗✗
   `for (c = threadIdx.x; c < d; c += blockDim.x) dot += q[c]*k[c];` 按 blockDim.x 个线程分摊，
   但 `__shfl_xor_sync` 只覆盖 32 个 lane，且只把 warp 0 的部分和写进 `sdot` →
   blockDim=128、d=512 时 **score 只有真值的 1/4** → 单 KV 时 softmax 权重从 0.95 掉到 0.667
   → 整个注意力输出被均匀缩小 0.67 倍。
   **修**：加 `wpart[32]` 二级归约（跨 warp 求和）。
   **验证（逐元素，同 token 序列）**：sparse_o[1] 0.397 → **0.565**（官方 0.594）；
   注意力输出 o[1..3] = [-0.5707, 0.6732, 0.4596] vs 官方 [-0.59375, 0.660156, 0.457031] ✓。

2. **MoE 的专家求和被整段丢弃** ✗✗
   循环里 `add_inplace(&s.o, &s.ex_out, dim)` 逐个累加是正确的，但循环后
   `memcpy_d2d(s.o, s.ex_out)` 把累加结果**覆盖成最后一个专家的输出**；而且 `s.o` 从头到尾
   **没被清零**（里面还是注意力输出）。
   **修**：清零 `s.o` + 删掉覆盖用的 memcpy。
   **验证**：moe_out 0.0870 → **0.1037**（官方 pos0 0.145160）。

### 已与官方**逐位/逐一吻合**的量（同 token 序列 `[671,6102,294,8760,344]`）

| 量 | 官方 | 我的 |
|---|---|---|
| embed 行 rms (671) | 0.037195 | 0.037195 ✓ |
| hc pre / post / comb / rowsum | — | **全部吻合** ✓ |
| xn (attn_norm 输出) 逐元素 | [0.027344, 0.006134, -0.004578, -0.003403] | [0.0272978, 0.0061416, -0.0045922, -0.0034010] ✓ |
| q / kv 投影 + RoPE | — | ✓ 吻合（1-5%）|
| **attn_out (o)** | 0.774044 | **0.7806** ✓ |
| ffn_in | 0.126523 | 0.1265 ✓ |
| **final logits rms** | 2.819755 | **2.7983** ✓ |
| top5 | [(11111,20.33),(1613,17.55),(4588,16.70),(260,16.57),(16,16.46)] | 已出现 1613（官方第 2 名）✓ 但排序仍不同 ✗ |

### 剩余唯一已知差距：MoE 输出 1.4x 偏小（0.1037 vs 0.1452）

- 消融：仅 routed = 0.0456、仅 shared = 0.0462、完整 = 0.0476（h rms）→ **两路都存在且量级相当** ✗
  所以不是缺某一项，而是两路**共同环节**的系统性缩小。
- 已排除：gate/shared 权重分片（都是 Replicated ✓）、shared expert 的三权重命名与形状
  （checkpoint 就是 `.w1/.w2/.w3.` ✓）、swiglu_limit（gate 只夹上界、up 两侧夹 ✓ 与参考一致）、
  MLP 求和顺序、AR 语义（是**求和** ✓ 不是平均）。
- **下一步怀疑**：`expert_gate_up_fp4` / `expert_down_fp4`（mxf4）的**权重反量化 scale 语义**
  或 **激活 fp4 量化的 block 大小**（参考 `fp4_gemm_kernel` 的 `act_block_size` 默认 128，
  我用 32）—— 这是两路共用的环节 ✓。

### 探针纪律（踩过的坑，务必记住）

- 探针必须**同 token 序列、同位置（pos0）、同张量宽度**：
  ① 官方一次 prefill 全部 token、我逐 token → rms 口径不同 ✗；
  ② `attention()` 返回时 `s.o` 已是 `dim` 宽（wo_b 输出），按 `nh*hd` 读会读到陈旧尾部 ✗。
- 官方 `generate.py` 默认 `temperature=1.0`（采样）；**传 `--temperature 0` 才是 argmax 贪心**，
  才能与我的贪心逐 token 对齐 ✓。

### MoE 1.4x 的排查记录（已排除项，供下会话接力）

目标：官方 pos0 `moe_out=0.145160`，我 `0.1037`（修复 memcpy/清零后从 0.0870 升上来的）。

**已逐一排除**：
- `route_scale` 读取：`config.rs:272-273` 同时兼容 `route_scale` 与 `routed_scaling_factor` ✓（=1.5 ✓）
- `route_topk` 权重：与参考逐行一致 —— 用**无 bias** 的分数做权重、**带 bias** 的做选择，
  先 `w /= sum` 再 `* route_scale` ✓（`dsv41_route.cu:114-127`）
- `swiglu_limit`：gate 只夹上界、up 两侧夹、`silu(g)*u` ✓ 与参考 `Expert.forward` 完全一致
- gate / shared expert 的分片：都是 `Shard::Replicated` ✓（各 rank 路由一致，必需 ✓）
- shared expert 三权重命名：checkpoint 就是 `layers.N.ffn.shared_experts.{w1,w2,w3}.{weight,scale}` ✓，
  形状 `w1/w3=[inter,dim]`、`w2=[dim,inter]` ✓
- AR 语义：是**求和**（`tp.rs` 的 `all_reduce_inplace` = publish + `add_inplace_raw` 累加各 rank ✓），
  不是平均 ✓；shared expert 只在 rank 0 算 + SUM AR ⇒ 恰好计入一次 ✓
- `expert_down_fp4` 的写出：OVERWRITE（`launch_mxf4` 内核 `out[row*n+col] = x` ✓）

**专家内部量（layer0，我的 vs 官方）**：
| | 官方 | 我的 |
|---|---|---|
| x (专家输入) rms | 0.126262 | 0.1265（ffn_in ✓ 同输入）|
| gate rms | 0.304617 | 0.256–0.274（**~13% 偏小** ✗）|
| up rms | 0.296332 | 0.244–0.270（~13% 偏小 ✗）|
| swiglu rms | 0.070088 | 0.038–0.096（抖动大，均值接近 ✓）|

→ 差距不在"缺一项"，而在两路共用的环节，且表现为 ~13% 级而不是 1.4x 级，
   说明 1.4x 主要来自**多个专家的聚合**（`wsum[e]` 的 6 项求和）而非单专家。

**下会话最该做的两件事**：
1. 打印我的 `wsum[e]` 与官方的 `weights[idx, top]`（同一 token）逐项对比 —— 这是唯一还没直接
   量过的环节（路由**权重数值**，不是内核结构）。
2. 参考的 `fp4_gemm_kernel` 注释写明是 **"FP8 act x FP4 weight"**（激活 fp8、权重 fp4），
   而我 `quant_fp4` 把**激活也量化成 fp4** ✗ —— 数值上更粗（fp4 e2m1 只有 8 个幅值档）。
   若 1 无异常，改成"激活 fp8 × 权重 fp4"（参考的 `act_block_size` 默认 128，我用 32）。

### ✅ 路由已被证明与官方一致（本会话最强的一条正面证据）

同一 token（pos0）、同一层（L0），逐项对比：

```
我的：  idx=[277, 128, 155, 137, 251, 206]  wgt=[0.33745, 0.28548, 0.26174, 0.22107, 0.19490, 0.19937]
官方：  idx=[277, 128, 155, 137, 206, 251]  wgt=[0.33699, 0.28767, 0.26511, 0.21747, 0.20028, 0.19248]
```

→ **选中的 6 个专家完全相同** ✓（只有末两位顺序不同，对求和无影响 ✓）、
**权重逐一吻合到 1-2%** ✓✓ → `route_topk`（含 sqrtsoftplus、bias 只用于选择、先归一化后乘 route_scale）
与参考的 `Gate.forward` **行为一致** ✓。

**专家内部量**（我的专家 277 vs 官方前几次专家调用）：
| | 官方 | 我的 |
|---|---|---|
| gate rms | 0.3046 / 0.3700 / 0.2611 | 0.2776 / 0.3122 |
| up rms | 0.2963 / 0.3205 / 0.2643 | 0.2839 / 0.2829 |
| swiglu rms | 0.0701 / 0.0702 | 0.0524 / 0.0667 |

→ **同一量级** ✓（差异在 fp4/fp8 量化的误差范围）。

**因此 MoE 的 1.4x（0.1037 vs 0.1452）不在路由、不在专家内部量级** ✗。剩余最可能：
1. **专家权重的 fp4 路径**：参考 `fp4_gemm_kernel` 的 docstring 明写 **"FP8 act x FP4 weight"**
   —— 激活走 **fp8**、权重走 fp4；而我 `quant_fp4` 把**激活也量化成 fp4**
   （fp4 e2m1 每块只有 8 个幅值档，误差远大于 fp8 的 3 位尾数）；
   参考的 `act_block_size` 默认 **128**，我用 **32**。
2. shared expert 的最终幅度（其权重是 Replicated、只在 rank 0 计算 + SUM AR，结构已核对 ✓，
   但**数值**还没与官方逐项比过）。

**接手建议**：先按 1 把激活量化改成 fp8（block 128，与参考 `fp4_gemm_kernel` 一致），
这一步同时影响 routed 与 shared 两路，是最有可能一次性消掉这 1.4x 的改动。

### ★ 消融实验把 MoE 的 1.4x 精确定位到 **routed expert 路径**

同一 token、同一层，用两个开关分别关掉一支：

| 配置 | L0 `moe_out(o)` rms |
|---|---|
| `DSV41_SKIP_SHARED_EXPERT=1`（**仅 routed**）| **0.0560** |
| `DSV41_SKIP_EXPERTS=1`（**仅 shared**）| **0.0818** |
| 全量 | 0.1037 |
| 官方 pos0 | **0.145160** |

两支近似正交（√(0.0818²+0.0560²)=0.0991 ≈ 0.1037 ✓）。反推官方：
若其 shared ≈ 0.0818（结构已核对一致），则 **官方 routed ≈ √(0.1452²−0.0818²) = 0.1199**
→ **我的 routed 只有 0.0560，约 2.1x 偏小** ✗✗

**已排除（都有数据）**：路由（同 6 个专家、权重差 1-2% ✓）、专家内部量级
（gate/up/swiglu 与官方同量级 ✓）、shared expert（同名同形状同分片、量级正常 ✓）、
AR 语义（求和 ✓）、`expert_down_fp4` 的覆盖式写出（已知 ✓ 且已按此修正 ✓）。

**因此 bug 在"每个 routed 专家自身的输出/累加"** —— 最可疑：
- `expert_down_fp4` → `launch_mxf4(nullptr, nullptr, act, w2, w2_scale, w2, w2_scale, out,
  rows, dim, inter, -1, 2, 0.f, weight, true, s)` 里的 **`n_split=2`**（注意 gate_up 那一支
  的对应参数不同）—— 若它把 N 维切成 2 份而只算了一份，输出就会小一半；
- 或者 `wsum[e]`（每个专家的**权重和**）没有正确传到 down GEMM 的 `alpha/weight` 位置。

**接手第一步**：把 `expert_down_fp4` 的输出与"手工 numpy 复算（用官方走同一专家的权重+激活）"
对比，直接看 down 这一支是差 2x 还是差在 wsum。

### ★★ 官方消融给出最终定位：**shared 完全正确，routed 差 1.79x**

给官方 `MoE.forward` 加 `REF_ABLATE=routed|shared` 探针，取 layer0 **pos0**：

| 分支 | 官方 | 我的 | 判定 |
|---|---|---|---|
| **shared_only** | **0.080769** | **0.0818** | **✓ 1.3% 吻合** |
| **routed_only** | **0.100171** | **0.0560** | **✗ 1.79x 偏小** |
| full | 0.145146 | 0.1037 | ✗ |

→ **shared expert 的权重、fp8 GEMM、以及"仅 rank0 计算 + SUM AR"的接法全部正确** ✓✓
→ **路由（专家选择 + 权重数值）已证实与官方一致** ✓✓
→ **唯一错的环节 = 每个 routed 专家自身的输出** ✗

**已核对无误的相关代码**（都查过了，不是这些）：
- 专家分片：`weights.rs:215` `if n == "w2" { Shard::ExpertCols } else { Shard::ExpertRows }`
  —— w2 切 dim1（inter=K ✓）、w1/w3 切 dim0（inter=N ✓）✓
- 全部 384 专家都在本地（`e=277` 的调试能打印 ✓，`ne=384` ✓）
- `expert_down_fp4` → `launch_mxf4(..., n_total=dim, k=inter_local, b_split=-1, epi_mode=2,
  limit=0, row_weight=wsum[e], aq=true)` 的参数逐位映射正确 ✓；
  内核 `epi_mode==2` 就是 `x *= row_weight[row]` ✓
- `epi_mode==1` 的 gate/up 夹取（`col < b_split` 判 gate ✓）与我的 `swiglu_limit` 重复但等价 ✓

**下会话第一刀（最省事）**：把 `expert_down_fp4` 那一支的输出直接与"用官方同一专家的
w2/swiglu/权重做 numpy 复算"对比 —— 分三步各打印一次 rms：
① down 之前的 swiglu 向量；② down 之后的单专家向量；③ 累加 6 个之后的向量。
官方的对应量：单专家 routed 贡献可由 `routed_only/√6` 估 ≈ 0.041，6 个和 = 0.100171。
我的累加结果 0.0560 —— 看是"单个专家就小"还是"累加漏了专家"。
