# D1 — 残差流 bf16 截断：边界清单与 GPU 验证手册

**状态**：代码已实现，仅 `cargo check --workspace` 通过（EXIT=0）。**GPU 验证未做**（本 session 禁 GPU）。
**gate**：`DSV41_BF16_TRUNCATE`（默认 OFF；`=1` 开，`"0"` 关——`chain_dev::bf16_truncate()` 的一次性 `OnceLock`）。
**改动文件**：`crates/ferrite-models/src/dsv41/{chain_dev.rs,device.rs}`。**不含 `.cu`**（复用已有 `dsv41_bf16_roundtrip`，无 ABI 变化 ⇒ 只需 `cargo build --release`，**不需要 `build.sh 103a`**）。

---

## 1. 判决依据（官方语义）

`ref_inference/generate.py:118` 的 `torch.set_default_dtype(torch.bfloat16)` 下，
官方**每个被 materialise 的激活**都是 bf16；下一条语句读的是那个**已舍入**的 row。
ferrite 全域 f32 ⇒ 每层比官方精确 ~1e-3，44 层链式放大 ⇒ near-tie argmax 翻转。

关键：**矩阵乘/归一的输出 dtype**才是边界，不是输入。

| 官方语句 | 返回 dtype | 位置 |
|---|---|---|
| `F.linear` with fp8/fp4 weight → `fp8_gemm`/`fp4_gemm` | `torch.get_default_dtype()` = **bf16** | `kernel.py:299,583` |
| `RMSNorm.forward` → `(self.weight * x).to(dtype)` | **bf16** | `model.py:288-293` |
| `hc_pre` → `y.to(x.dtype)` | **bf16** | `model.py:957-960` |
| `hc_post` → `y.type_as(x)` | **bf16** | `model.py:962-966` |
| `RowParallelLinear.forward` → `y.type_as(x)` | **bf16** | `model.py:271-278` |
| `Expert.forward` → `silu(gate)*up → .to(bf16)`，再 `w2(...)` | **bf16** | `model.py:841-849` |
| `MoE.forward` → `y.type_as(x)`（内部 f32 累加） | **bf16** | `model.py:889-904` |
| `hc_mixes`（权重 f32，`set_dtype(float32)`） | **f32 — 不是边界** | `model.py:946-956` |
| `Gate.forward`（`x.float()`, `weight.float()`） | **f32 — 不是边界** | `model.py:809-827` |
| `ParallelHead`（weight 显式 f32，`x.float()`） | **f32 logits，但输入 row 是 bf16** | `model.py:1008-1017` |

---

## 2. 修复前覆盖范围（现状）

`DSV41_BF16_TRUNCATE` 原有 **11 个 `bf16_snap` 调用点 + 1 个 kernel 内 `truncate` 参数**：

| # | 位置 | 边界 |
|---|---|---|
| 1 | `step_body` | 残差流诞生（embed + hc expand） |
| 2 | `engram_apply` | engram 写回 h（`(h + gate*value).to(x.dtype)`） |
| 3 | `layer` attn 段 | `attn_norm` 输出（`xn`） |
| 4 | `layer` attn 段 | attention 子层输出（wo_b + AR） |
| 5 | `layer` attn 段 | attn 侧 `hc_post` 残差更新 |
| 6 | `layer` ffn 段 | `ffn_norm` 输出（`xn`） |
| 7 | `layer` ffn 段 | MoE 子层输出 |
| 8 | `layer` ffn 段 | ffn 侧 `hc_post` 残差更新 |
| 9 | `attention` | `wq_a` / `wkv` 两个 GEMM 输出（`qr` / `kv`） |
| 10 | `attention` | `wq_b` 输出（rope 后） |
| — | `attention` | `sparse_attn` 输出（`s.o`，rope 后） |
| — | `attention` | `wo_a` 输出（`s.wo`） |
| K | `hc_collapse_norm` 的 `truncate` 实参 | `hc_pre` 的 collapse（norm 前 + 方差前） |

**缺口**：所有 *norm 的输出*（只 snap 了输入）、MoE 全部内部边界、compressor latent、
engram `wkv` GEMM 输出、末级 `hc_pre`/norm、以及**非融合** `hc_collapse` 分支。

---

## 3. 本批新增（15 个 round-trip 点 + 7 处 fusion decline）

全部只在 `bf16_truncate()` 为真时下发；OFF 时 `bf16_snap` 在 `n == 0 || !gate` 处直接 `return Ok(())`，
**一条指令都不发**，活跃路径逐位不变。

### 3.1 新增 round-trip

| # | 位置 | 缓冲 / 长度 | 对应官方语句 |
|---|---|---|---|
| N1 | `attention`（norm 之后） | `qr` / `ql` | `q_norm` 输出 = bf16（`model.py:770`） |
| N2 | `attention` kv（norm 后） | `kv` / `hd` **on `kv_stream`** | `kv_norm` 输出 = bf16（`model.py:703`） |
| N3 | `attention` kv（rope 后） | `kv` / `hd` **on `kv_stream`** | 原地 rope 作用于 bf16 ⇒ rope 输出也是 bf16 |
| N4 | `layer` attn 非融合分支 | `s.x` / `dim` | 非融合 `hc_pre` 的 collapse（`model.py:957-960`） |
| N5 | `layer` ffn 非融合分支 | `s.x` / `dim` | 同上 |
| N6 | `step_body` 非融合分支 | `s.x` / `dim` | 末级 `hc_pre` collapse |
| N7 | `step_body` 末级 norm | `s.xn` / `dim` | **head 的 norm 输出**（`model.py:1266`；⚠️ 见 §5） |
| N8 | `engram_apply` | `eng_kv` / `(hc+1)*dim` | `wkv` GEMM 输出（`model.py:355`，`key`/`value` 都 `.float()` 前已被舍入） |
| N9 | `compress_on` | `latent` / `hd` **on `s`** | compressor 的 latent = `norm(...)` = bf16（`model.py:485`） |
| N10 | `moe` batched gate/up 后 | `ex_act_b` / `topk*act_slot` | 路由 expert 的 `w1`/`w3` 输出 |
| N11 | `moe` batched swiglu 后 | `ex_act_b` / `topk*act_slot` | `silu(gate)*up → .to(bf16)` |
| N12 | `moe` batched down 后 | `ex_down_b` / `topk*dim` | 每个 expert 的 `w2` 输出（`y[idx] += expert(...)` 前） |
| N13 | `moe` 顺序（非 batched）gate/up 后 | `ex_act` / `2*inter_local` | 同上 |
| N14 | `moe` 顺序 swiglu 后 | `ex_act` / `inter_local` | 同上 |
| N15 | `moe` shared expert gate/up 后 | `ex_act` / `2*sh_il` | shared expert 的 `w1`/`w3` |
| N16 | `moe` shared expert swiglu 后 | `ex_act` / `sh_il` | shared expert 的 `silu*up` |
| N17 | `moe` shared expert w2 后 | `ex_out` / `dim` | shared expert 的 `w2` 输出 |

> N15/N16/N17 与 N13/N14 的 stream：写入方在侧流时用 `bf16_snap_on`（`device.rs` 新增
> `bf16_roundtrip_on`）——主流的 round-trip 会**越过生产者在跑**，舍入的是上一步的字节。
> 涉及三处侧流：`DUAL_CHAIN`(stream2, kv)、`COMPRESS_SIDE`(stream3, compressor)、
> `MOE_DUAL`(stream2, shared expert)。

### 3.2 fusion decline（**必须**，否则边界无处落地）

| gate | 位置 | 为什么必 decline |
|---|---|---|
| `norm_fuse` (`lin_rope_norm`) | `attention` | norm 在 GEMV prologue 内，归一化后的 row 从不落内存 ⇒ N1 无落点 |
| `qr_epi` (`rmsnorm_q` 的 fp8 epilogue) | `attention` | 在**未舍入**的 norm 输出上直接取 fp8，wq_b 会读脏 |
| `nr_fuse` (`rmsnorm_rope_on`) | `attention` | norm+rope 融合 ⇒ N2 无落点（`bf16` 必须落在 norm 与 rope 之间） |
| `compress_fuse` | `compress_on` | state+pool+norm+rope+commit 一次下发 ⇒ N9 无落点 |
| `gateup_fuse` (×2) | `moe` | 融合 epilogue 直接写 swiglu 结果 ⇒ N10/N11 无落点。**kernel 跟随**：`dsv41_experts_mxf4.cu` 把 `fuse` 绑死在 caller 的 `out_slot_stride == inter`，所以只改 Rust 侧 pitch 即可，无需改 `.cu` |
| `down_fuse` | `moe` | 融合体在一个 launch 内求和 ⇒ N12 无落点（缺每个 expert 一次舍入） |
| `moe_epi_add` (A5) | `moe` | w2 直接加进 `s.o`（f32），跳过 shared expert 输出的 bf16 舍入 ⇒ N17 无落点 |
| `swiglu_q` (A4, shared expert) | `moe` | 同一 launch 内出 swiglu 值 + 其 fp8 ⇒ N16 无落点 |

所有 decline 都是**逐位等价**（house 注释已论证），因此代价只有延迟，且只在 gate ON 的臂上。

### 3.3 明确**不动**的地方

- **verify / m-rows 链**（`moe_rows` / `attention_rows` / `layer_rows` / `hc_mixes_auto` 的 `false` 实参）：
  沿用既有纪律（`DSV41_BF16_TRUNCATE` 不许上 verify——050c7fd 破过零拉丁基线）。
- **DSpark draft 段**：`dspark_dev.rs` 已有自己的 `bf16_truncate()` 传参，本次未改。
- **`hc_mixes` 输出**：官方就是 f32（`set_dtype(torch.float32)`），**不是**边界，故意不加。
- **`Gate` 输出 / logits**：官方 f32，不加。

---

## 4. 已知残差（本批未修，需后续决定）

| # | 残差 | 说明 |
|---|---|---|
| R1 | **expert 路由权乘法次序** | 官方 `Expert.forward` 是 `silu*up → bf16 → (bf16 * w) → bf16 → w2`（`model.py:847-849`），即 `w` 乘在 bf16 激活上再舍入一次。ferrite 把 `w` 乘在 down **输出**上（`expert_down_*_fp4_*` 的 `route_w`）。数学等价、浮点不等价（~1e-3）。要修需把 `w` 提到 down 之前并对激活再 snap 一次 |
| R2 | **`wq_b` rope 前的舍入** | 融合的 `lin_rope`/`lin_rope_norm` 在 GEMV 内旋转，rope 前那次 bf16 舍入无法插入（`chain_dev.rs` 既有注释已记录）。当前只 snap rope **后** |
| R3 | **顺序 MoE 的 per-expert down 输出** | 该 kernel 直接累加进 `s.o`，没有 per-slot `[dim]` 行可舍入。非默认臂（`DSV41_MOE_BATCH` 默认 ON） |
| R4 | **indexer 的 GEMM 输出** | 官方 `idx_wq_b` / `weights_proj` / `wk` / `k_norm` 输出全 bf16（`model.py:527+`）。它们只影响 topk **选择**（整数），漂移极少翻；本次未动。若 A/B 显示 selection 有差，再补 |
| R5 | **MoE-TileLang / tcgen05 臂** | 它们有自己的 precision 路径（bf16 原生权重 / blockscaled）。N10/N11/N12 的 flat snap 对它们**同样生效**（都写 `[topk][act_slot]` raw 布局），但没做逐位论证 |
| R6 | **融合臂上的 fp8 发射** | 除已 decline 的 4 处外，其余 fp8 epilogue（`sparse_orope` / `o_q_epi` / B1 `wo_fuse`）**原先已在** `!bf16_truncate()` 下 decline（老代码），本次未动 |

---

## 5. ⚠️ 需要 GPU A/B 优先确认的一条

**N7（head 的 norm 输出）**。`step_body` 原有注释写「the reference keeps it in f32」——
**这是错的**：`ParallelHead.forward` 是 `F.linear(x.float(), weight)`，而传入的 `x` 是
`self.norm(h)`，`RMSNorm` 返回 bf16；`.float()` 只是 bf16→f32 的**无损**加宽（`model.py:1266,288-293,1014`）。
所以官方 argmax 的输入 row **是舍入过的**。

但同一条注释也记录了实测「casting it to bf16 cost ~3 bits on a 129280-way near-tie argmax」。
那是**精度**论证，不是**对齐**论证——两个目标相反。本批按「对齐官方」的红线加了 N7，
且只在 gate ON 时生效。**风险最高的一条**：它直接动 argmax 的输入。

---

## 6. GPU 验证手册（交给主 agent，subagent 不执行）

### 6.1 产物

**不需要 `build.sh 103a`**（无 `.cu` 改动）。只重编 Rust 侧：

```bash
cd ~/ferrite && source ~/.cargo/env && cargo build --release
```

### 6.2 全链零拉丁回归（gate ON 必须保持）

```bash
# 单轮制；serve 收尾一律 POST /shutdown
DSV41_BF16_TRUNCATE=1 <标准 dspark 环境> ./target/release/dsv41-run --serve ...
# 探针 1：数字任务数数（1..100，逐行）——对重复/错位最灵敏
# 探针 2：出师表背诵（LEN 应 = 146）
```

判定：数字 1..100 逐行无重复、出师表 LEN=146 且无拉丁/乱码。

### 6.3 A/B（OFF vs ON），同会话背靠背

| 臂 | env | 期望 |
|---|---|---|
| A | 不设 `DSV41_BF16_TRUNCATE`（默认 OFF） | **逐位 = 今日 head**（回归零点） |
| B | `DSV41_BF16_TRUNCATE=1` | 与官方 PyTorch 逐层偏差显著下降；argmax 翻转减少 |

**归因用逐层对照**（这是本任务的判决指标，不是文本）：

```bash
DSV41_BF16_TRUNCATE=1 DSV41_LAYER_DUMP=1 ...   # [layer] l=N norm=... 逐层残差流范数
# 与 ref_inference 的官方逐层 dump 对齐比较：
#   目标：逐层相对偏差从 2.4-2.9% 降到 <1%（bf16 量化噪声量级）
```

### 6.4 单变量拆分（N7 风险隔离）

先只验 N7：把 head 的 norm snap 临时注释掉再跑一次 B 臂，看零拉丁与逐层偏差是否变化。
若 N7 破坏零拉丁而其余边界都正常 ⇒ 报回用户仲裁（精度 vs 对齐的取舍）。

### 6.5 性能（gate ON 的代价，仅记录，非验收）

新增 launch/步 ≈ 17 个 round-trip + 8 处 decline 带来的额外 launch。
用 `[dspark] steps=` 的 p50 对比 A/B 两臂（**不许吞吐反推**）。

---

## 7. 验收状态

| 项 | 状态 |
|---|---|
| `cargo check --workspace` | ✅ EXIT=0 |
| CPU 单测 `cargo test -p ferrite-models --lib` | ✅ 97 passed / 0 failed / 2 ignored |
| GPU 单测/parity 测试 | ⛔ 本地无 `libcudart.so` / `libferrite_kernels.so`，**HEAD 同源同failing**（环境限制，与本次改动无关） |
| GPU e2e / 逐层精度对照 | ⏳ **未做**（禁 GPU）→ §6 手册 |
