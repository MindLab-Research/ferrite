# draft 链的设备级 CUDA 图（P3c）— 实现归档（2026-09-12）

`DSV41_DRAFT_GRAPH=1`（**默认 OFF**）。本文归档工部的实现：捕获范围、四个 host 依赖的
处置、barrier/D2H 的图外纪律、gate 的启用条件，以及**尚未在 GPU 上验证**的部分。

代码位置：
- `crates/ferrite-models/src/dsv41/dspark_dev.rs` —
  `draft_graph_want`（gate）/ `draft_forward`（prologue+派发）/ `draft_body`（被录制的
  kernel 序列）/ `draft_graph_arm` / `draft_capture` / `capture_draft` /
  `seed_window(.., slot_dev)` / `ensure_pos_dev` + `seed_slot_ptr` / `Drop`
- `crates/ferrite-models/src/dsv41/device.rs` — `Device::supports_ring_append`

---

## 0. 形态

```
draft_forward(t0, pos)                       ← 每次 draft 步调用（bs=5, n_mtp=3）
  PROLOGUE（host 侧，永不进图）
    ensure_pos_dev(pos)                      H2D: pos_base = [pos, pos+r…, pos-1]
    upload_i32(ids)                          H2D: ids[0]=t0，其余 noise
    if pos > 0 { ensure_idxs(pos) }           H2D（pos>=win 后成为 no-op）
  DISPATCH
    gate 未通过            → draft_body(pos, slot_dev=false)   直发（历史路径）
    gate 通过 & !dry_done  → draft_body(pos, slot_dev=true)    DRY（真实执行）
    gate 通过 & graph 存在 → host_barrier → graph_launch        REPLAY
    gate 通过 & 无 graph   → host_barrier → capture → end →
                             host_barrier → instantiate → launch（捕获不执行）
  尾部：unit_dump 落盘（host）
```

`draft_body` = `project_main_x` → `embed_expand_dev` → 3 × 块循环 → `draft_head`。
**DRY / CAPTURE / 直发三条路走同一个 `draft_body`**：录制必须是「正常 forward 会发的那串
kernel」，不能是它的变体，否则 A/B 无意义。

---

## 1. 捕获范围（哪些 kernel 进图）

进图（`draft_body` 全部设备工作）：

| 段 | kernel |
|---|---|
| forward_embed | `quant_fp8`(main_h) → `gemm_fp8_mx`(main_proj) → `rmsnorm`(main_norm) → `embed_expand_dev` |
| 每块 attn 半 | `hc_mixes` → `hc_collapse_norm`(a1) / `hc_collapse`+`rmsnorm` → `seed_window`(quant/gemm/rmsnorm/**ring_append**/rope) → `gemm_fp8_mx`(wq_a) → `rmsnorm`(q_norm) → `gemm_fp8_mx`(wq_b) → `apply_rope[_mrows]` → `gemm_fp8_mx`(wkv) → `rmsnorm`(kv_norm) → `apply_rope` → 2×`memcpy_d2d`(all_kv) → `sparse_attn` → `apply_rope[_mrows]`⁻¹ → `wo_a_grouped_fp8` → `quant_fp8` → `gemm_fp8_mx`(wo_b) |
| 每块 FFN 半 | `hc_post` → `hc_mixes` → `hc_collapse_norm` → `gate` GEMV(×bs 或 `gemv_bf16_v2_mrows`) → `route_topk` → `quant_fp4`/`quant_fp8` → `expert_gate_up_fp4_batched` → `swiglu_limit_batched` → `expert_down_reduce_fp4_batched` → 共享专家(w1/w3/swiglu/w2) → `add_inplace` → **`all_reduce_inplace`**(v5 设备侧) → `hc_post` |
| head | `hc_collapse` → `rmsnorm` → `head_gemv_bf16_mrows` → 5 ×(`dspark_markov_head` 或 `_sliced` + `argmax_key_pub`) |

**图外（永不录制）**：
- `drafts()` 的 D2H（`ids[1..=5]`）——设备读，独立方法，调用者在 `draft_forward` 之后调。
- 环绕捕获的两个 `host_barrier` 与每次 replay 前的一个（host barrier 不是 CUDA 调用，
  录不进去；见 §3）。
- `DSV41_DSPARK_UNIT_DUMP` 的 per-unit D2H、`DSV41_DSPARK_UNIT_INJECT` 的 H2D
  （gate 直接拒绝整个调试臂）。

---

## 2. 四个 host 依赖（账本 §P1 的 D1–D4）

| # | 原状（不图安全） | 处置 |
|---|---|---|
| **D1** | 每步阻塞 H2D `upload_i32(ids)` | 移到 prologue；图读同一地址 `self.ids` |
| **D2** | `seed_window` 目的地址为 host 算的 `window[s] + (pos%win)*hd`，`cudaMemcpyAsync` 节点会**冻结该地址** → 每次 replay 写同一环槽 | 改用 `dsv41_ring_append`：槽位由**设备计数器**在核内算；计数器是 `pos_base` 新增的 slot `bs+1`（= `pos-1`），prologue 的**单次** `ensure_pos_dev` H2D 顺带上传 |
| **D3** | `window → all_kv` 拷贝的 size 与其 `s0==0` 分支是 host 由 `win_rows(pos)` 算的 | gate 要求 `pos >= win` 且 `!seed_align()`：此时 `win_rows` 恒定 `(win, 0)`，size / 分支 / `sparse_attn` 的 `n_win` 全部恒定 |
| **D4** | `ensure_idxs` 的 H2D（n_win 变化时） | 移到 prologue（`pos>0`）；`pos>=win` 后 n_win 恒为 `win`，此后为 no-op，`idxs` 内容对每次 replay 相同 |

**RoPE 无需改动**：`ensure_pos_dev` 已把 base 放设备内存、核内解引用；所有 `rope_at` 的
`off` 都是相对 base 的**常数**（seed `-1`、query/inv `+r`、KV `0`），所以 prologue 每步刷新
`pos_base` 就够，位置随 replay 前进。

**其余 host 分支**（`draft_p3a/p3b`、`draft_head_fold`、`expert_act_e4m3`、`gateup_fuse`、
`supports_*`、`markov_head_geom`、`wo_a_grouped_fp8`/`gemv_bf16_v2_mrows`/`head_gemv_bf16_mrows`
的 `Ok(bool)`）全部由 config / `.so` 决定，跨 replay 恒定，可安全冻结。

---

## 3. barrier 与 D2H 的图外纪律

镜像 `DevChain::capture_verify` / `step_rows_sync`：

- **捕获**：`host_barrier` → `capture_begin` → `draft_body(.., true)` → `capture_end` →
  `host_barrier` → instantiate → launch。
  理由同 verify：capture 只 **RECORD** MoE 的 all-reduce，peer 可能正在 **EXECUTE** 自己的；
  录制方不发布 stamp。
- **replay**：`host_barrier` → `graph_launch`。
- **DRY**：不加 barrier（它是一次普通执行，与直发同级）。
- **D2H**：`drafts()` 是独立方法，天然在 `draft_forward` 之外（verify 的 argmax-D2H 同款）。
- **TP8 前提**：`comm.is_some()` 时 gate 要求 `ar_v5()` —— v5 是设备侧协议（epoch 在设备
  内存），能被录制；非 v5 走 host barrier，录不进去。

---

## 4. gate（`draft_graph_arm`）

任一不满足即走直发（逐位不变）：

1. `DSV41_DRAFT_GRAPH` 未开 / 已 latch 失败；
2. `!supports_ring_append()`（D2 修复的前提）；
3. `!supports_memset_async()`（顺序 MoE 臂会 zero `moe_out`，同步 memset 在 legacy stream）；
4. `comm.is_some() && !ar_v5()`；
5. **`pos < win`**（D3/D4 的稳定态；`win = window_size`）；
6. **`seed_align()` 开**（`s0` 每步旋转，拷贝变成两段变长）；
7. `unit_dump::enabled()` 或 `self.unit.is_some()`（调试臂 host 探测）。

⚠️ **启用时刻**：`pos >= win`（DSV41 配置 `window_size=128`）后**第二步**才 capture
（第一步 DRY）。短跑（< ~130 tok）不会上图 —— 这是 D3 的硬约束，不是可调参数。若要更早
上图，必须先把 `win_rows` / `all_kv` 拷贝改成设备派生（新 kernel，不在本次范围）。

---

## 5. launch 数

| | launch/step |
|---|---|
| gate OFF（历史） | ~150（P3a/P3b 全关时；开折叠后 ~70–90） |
| gate ON：replay | **1**（`cudaGraphLaunch`） |
| gate ON：prologue 仍在图外 | + 2 个阻塞 H2D（`pos_base`、`ids`）+ `ensure_idxs`（稳态 no-op） |

即「launch ~120 → 1」在 replay 后成立；图外的两个 H2D 是图能成立的**前提**（它们把每步
输入送到录制看过的同一地址），不是余量。

---

## 6. 失败回退

捕获/实例化/首发的任一失败 → 打印
`[draft_graph] capture FAILED (pos=…): … — the draft stays on the direct launches (latched)`
→ `graph_failed = true`（永久）→ 本步与之后每步都走直发。

**非对称 latch 的影响**：若只有单个 rank latch，它走直发、peer 走 replay。两者是**同一
串 kernel、同一批参数**（图只是发射方式的改变），所以数值一致，rank 间只有发射开销差异
（v5 协议对漂移容忍）。与 verify 的 latch 同风险等级，一并记录。

---

## 7. 生命周期

- 图烘焙的是地址：draft 自己的 buffer 由 `DsparkDev` 持有、**一生不重分配**（没有 per-request
  reset 钩子，与 chain 的 step/verify 图不同——那两个 per-request drop 是因为 chain 会重分配
  scratch）；权重 `w`、rope tables（`DevChain::new` 一次性 alloc、永不 free）、collective
  staging 同样进程级稳定。故图**跨 request 复用**。
- `DsparkDev` 新增 `Drop`：teardown 时 `graph_free(null, exec)`。

---

## 8. 尚未验证 / 待 GPU 会话

本任务只做「写代码 + `cargo check --workspace`（EXIT=0，无新增 warning）」。**未跑 GPU**。
上机时的验收顺序：

1. `DSV41_DRAFT_GRAPH=1` vs unset，**同一进程背靠背**比 `draft_ms`（serve 的 `[dspark]` 行）
   与 `[draft_graph] captured …` 行是否出现（**不出现 = gate 没生效**，见 §4 的 `pos>=win`
   与 `.so` 符号）。
2. 文本红线：出师表逐字 + 数字任务数数；`DSV41_DIFF_EAGER=1` 的 `[diff]` anchor 应每轮一致。
3. `k_acc` 直方图与 gate OFF 对比（accept 不应变化）。
4. 若 capture 失败：看 `[draft_graph] capture FAILED` 的 why（最可能是驱动拒绝某个 op），
   按 §6 回退即可，不影响正确性。
5. 数据并行臂（`DSV41_SEED_ALIGN=1`）下 gate 永久拒绝——这是有意的，见 §4.6。
