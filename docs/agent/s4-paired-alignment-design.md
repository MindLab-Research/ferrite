# S4 — paired KV act_quant 对齐设计（draft ↔ backbone ring）

> 工部 · 2026-09-12 · **只读 + 设计，未改动任何源码、未执行 GPU 命令；本文件为唯一产出**
> 基准 revision：`06861aa`（`crates/ferrite-models/src/dsv41/chain_dev.rs` 14520 行 /
> `dspark_dev.rs` 3760 行）。所有代码事实给出 `文件:行号`；推断项显式标注。
> 上游输入：`accept-first-strategy.md §1.1/§1.4/§3-S4`、`accept-gap-1214-to-3.md §2.5/§5`。

---

## 0. TL;DR（五条）

1. **【代码级·关键修正】官方 ring 存的不是「fp8 字节」，而是「e4m3 网格上的值」。**
   `ref_inference/model.py:663-668` 的 `window_kv_cache` 用 `torch.zeros(...)` 建缓冲，
   在 `generate.py:118` 的 `torch.set_default_dtype(torch.bfloat16)` 之下 ⇒ 容器是 **bf16**；
   而写入前调用 `act_quant(..., inplace=True)`（`model.py:707` / `1042` / `1062`），
   kernel 语义是 `Cast(bf16, Cast(f32, Cast(fp8, clamp(x/s))) * s)`（`kernel.py:81-86`）
   ⇒ **值是 e4m3 网格值（block=32、power-of-2 的 ue8m0 scale），容器 bf16**。
   `chain_dev.rs:62` 的「the release stores it fp8」把**精度**说成了**容器**。

2. **【数值级·好消息】因此 S4 不需要改 ring 的 dtype，也不需要改任何读出路径。**
   e4m3 只有 3 位尾数，乘 power-of-2 scale 仍是 ≤4 位有效位 —— 精确可表于 bf16 **与** f32。
   所以「把值吸附到 e4m3 网格、仍存 f32」与官方「吸附后存 bf16」**是同一个实数**。
   → 读出侧（`sparse_attn` / `sparse_attn_orope` / draft `sparse_attn`，全部吃 `*const f32`）
   **零改动**，这正是把 S4 从"大手术"降成"写入侧 4 个点"的原因。

3. **【改动面】4 个写入点，一个共享 gate。** backbone 单行（`s.kv`）、backbone 验证 m 行
   （`s.kv_r`）、draft 种子（`seed_window` 的 `mk`）、draft 块行（`draft_attention` 的 `kv`）。
   两侧**读同一个 `pub(crate) fn ring_format()`**（放在 `chain_dev.rs`，`dspark_dev.rs` 调用），
   「两侧同动」由**构造**保证，而不是靠纪律。

4. **【必须先解决的分歧】block size：官方参考是 32，ferrite 自己的注释写 128。**
   `model.py:27 fp8_block_size = 32` 并原样传进三处 `act_quant`；但
   `quant.rs:10` / `ops.rs:868` / `dsv41_kernels.cu:123` 都声称「window KV: block 128」。
   错一个就是整片 scale 粒度错。**实现前必须用 golden 的
   `stage{s}.attn.quant.<tag>_pre/_post` 对钉**（见 §6-U1）—— 那对张量是官方 `act_quant` 的
   输入/输出原样快照，能直接判定 block 与 scale 语义。

5. **【风险】这不是一个纯 accept 杠杆：backbone ring 是交付文本的 KV。**
   量化它会改变**已提交 token 流**（不是只改 draft）。红线（零拉丁 + 出师表逐字 +
   DIFF_EAGER `[diff]`）必须**先**过。历史上 `DSV41_BF16_TRUNCATE` 就打破过零拉丁
   （`chain_dev.rs:11791` 附近注释自述），先例在案。

---

## 1. 官方 KV 存储语义（把参照物钉死）

### 1.1 backbone：`_window_kv`

`ref_inference/model.py:700-720`：

```python
def _window_kv(self, x, freqs_cis, start_pos):
    """... The K stays fp8, quantized over the whole post-RoPE vector, RoPE tail included."""
    kv = self.kv_norm(self.wkv(x))
    apply_rotary_emb(kv[..., -self.rope_head_dim:], freqs_cis)
    act_quant(kv, fp8_block_size, scale_fmt, scale_dtype, True)   # ← line 707
    ...
    self.window_kv_cache[:bsz, start_pos % win] = kv.squeeze(1)   # ← 存「吸附后的值」
```

### 1.2 draft：`DSparkAttention`

`model.py:1032-1074`：

```python
main_kv = self.kv_norm(self.wkv(main_x))
apply_rotary_emb(main_kv[..., -rd:], main_freqs_cis)
act_quant(main_kv, fp8_block_size, scale_fmt, scale_dtype, True)  # ← 1042，种子行
...
kv = self.kv_norm(self.wkv(x))
apply_rotary_emb(kv[..., -rd:], freqs_cis)
act_quant(kv, fp8_block_size, scale_fmt, scale_dtype, True)        # ← 1062，块行
...
self.window_kv_cache[:bsz, start_pos % win] = main_kv.squeeze(1)   # ← 1065
kv = torch.cat([self.window_kv_cache[:bsz], kv], dim=1)            # ← 1066
o = sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)
```

**两处 KV 都被 `act_quant`**：种子行（来自 `main_x`）与块行（来自 `x`）。这与 ferrite 的
`seed_window`（种子）+ `draft_attention`（块行）两个写入点一一对应。

### 1.3 `act_quant(inplace=True)` 的精确语义

`ref_inference/kernel.py:41-95`（逐行读过）：

| 项 | 值 | 出处 |
|---|---|---|
| 容器 dtype | `out_dtype = in_dtype if inplace else out_dtype` | `kernel.py:52` |
| amax 下限 | `amax = max(amax, 1e-4)` | `kernel.py:76` |
| scale | `round_scale ? 2^ceil(log2(amax/448)) : amax/448` | `kernel.py:78-80` |
| 值 | `Cast(out_dtype, Cast(f32, Cast(fp8, clamp(x/s, ±448))) * s)` | `kernel.py:83-86` |
| 调用参数 | `block_size=fp8_block_size=32`，`scale_fmt="ue8m0"`（⇒ `round_scale=True`），`scale_dtype=e8m0` | `model.py:27/29/30/707` |

即：**e4m3 网格吸附 + 每 32 元素一个 power-of-2 scale，量化后反量化回容器 dtype**。
RoPE 尾段一并量化（「whole post-RoPE vector, RoPE tail included」）。

### 1.4 「容器 bf16 vs f32」为什么在数值上无关

- e4m3 值 = `m·2^e`，`m` 有 3 位尾数；乘 power-of-2 的 `s` 后仍是 ≤4 位有效位。
- bf16 有 8 位有效位、f32 有 24 位 ⇒ 两者都**精确**表示该值。
- `kernel.py:83` 的最后一跳 `Cast(bf16, ·)` 因此是恒等。
⇒ **官方存储值 ≡ `e4m3_decode(e4m3_encode(clamp(x/s))) · s`（f32 精确）**。

**结论：ferrite 保持 f32 ring，只把值吸附到 e4m3 网格，就与官方逐位等价。**
（`chain_dev.rs:62` 的「the release stores it fp8」若指的是生产 release 存 fp8 字节 + scale，
读侧反量化后得到的是同一批实数 —— 两种读法数值一致，所以不影响设计。）

---

## 2. ferrite 现状：三条写入路径都是「原始 f32，无吸附」

### 2.1 backbone ring 的缓冲与所有权

| 项 | 事实 | 出处 |
|---|---|---|
| 缓冲 | `LayerCache.ring: DevBuf`，注释明写 f32 | `chain_dev.rs:61-64` |
| 尺寸 | `(window_size + max_comp) * hd` ⇒ 行 `[0,win)` 是窗口行、`[win, win+clen)` 是**压缩行** | `chain_dev.rs:2904` |
| 所有权 | 默认 `owner = layer`（每层自持）；`DSV41_RING_OWNER=1` 才是旧的组共享 | `chain_dev.rs:12690` 附近（`ring_owner_shared()`） |
| 读出 | `sparse_attn` / `sparse_attn_orope` 吃 `*const f32` | `device.rs:2484-2513` / `2525+` |

### 2.2 四个写入点（ALTER 清单的骨架）

| # | 侧 | 位置 | 生产的 KV | 紧邻的写入调用 |
|---|---|---|---|---|
| **W1** | backbone 单行 decode（含 prefill，prefill 逐 token 走 `step`） | `chain_dev.rs:12703-12737` 做 norm+rope，`12749-12751` dual join，`12758` 起进 ring 段 | `s.kv`（1×hd） | `ring_win_fuse_ph`/`ring_win_fuse`（`12823`）/ `ring_append`（`12867`） |
| **W2** | backbone 验证 m 行（`step_rows_inner`） | `chain_dev.rs:9018-9025` `norm_rows(kv_r)`，`9026-9040` `apply_rope(kv_r)` | `s.kv_r`（m×hd） | 逐行 `ring_append`（`9210`） |
| **W3** | draft 种子（`seed_window`） | `dspark_dev.rs:3278`，`3307` `rope_at(mk)` | `self.mk`（1×hd） | `ring_append`（`3319`）/ `memcpy_d2d`（`3329-3331`） |
| **W4** | draft 块行（`draft_attention`） | `dspark_dev.rs:1767`，`1954` `kv_pos`，`1955-1963` `rope_at(kv)` | `self.kv`（bs×hd） | `memcpy_d2d` 进 `all_kv`（`2008-2012`） |

补充事实：

- **W3 同时覆盖 `note_ctx_rows`**：它按行调 `seed_window`（`chain_dev.rs:7141/7375/7588/7978`
  的调用点），所以提交路径自动被覆盖，不需要第 5 个点。
- **W4 的窗口行无需再处理**：`draft_attention` 把 `window[s]` 的窗口行 memcpy 进 `all_kv`
  （`dspark_dev.rs:1989-2005`），而那些行是 W3 写的 —— W3 吸附了，读出来的就是吸附值。
- **prefill 自动覆盖**：`prefill_chain` 是逐 token `chain.step`（`ferrite-dsv41/src/serve.rs:758-765`），
  走 W1 的同一条代码路径。

### 2.3 读者清单（证明「读出零改动」）

| 读者 | 参数类型 | 读什么 | S4 是否需要动 |
|---|---|---|---|
| `sparse_attn` | `kv: *const f32`（`device.rs:2494`） | backbone ring / draft `all_kv` | **否**（值已吸附，容器仍 f32） |
| `sparse_attn_orope` | 同上（`device.rs:2535`） | 同上 | **否** |
| draft `sparse_attn` | 同上（`dspark_dev.rs` 调用处） | `self.all_kv` | **否** |
| `kv_snapshot`/`kv_restore` | 裸字节 | ring | **否**（字节拷贝，格式无关） |
| `dspark_snapshot`/`rollback` | 裸字节 | ring 槽 | **否** |
| `compress_commit` | `latent` → ring `[win, ·)` | **压缩行** | **否**（见 §5-2，另议） |

---

## 3. 设计：paired alignment

### 3.1 核心决策

> **不换容器（ring 继续 f32），只把写入的 KV 行吸附到官方网格；
> 吸附动作封装成一个共享 helper，由唯一一个 gate 驱动，两侧共用。**

理由（三条，按重要性）：

1. **数值上已经等价**（§1.4）：官方存 bf16-of-grid，ferrite 存 f32-of-grid，是同一个实数。
   换容器只会在读出侧引入 dequant 与 dtype 适配，收益为零、风险为正。
2. **「两侧同动」可以被构造保证**：所有写入点调用同一个
   `pub(crate) fn ring_format()` + 同一个 `snap_kv(...)` helper。
   想只动一侧在结构上就做不到 —— 这正是 accept-first-strategy §1.1 那条警告的解药。
3. **最小改动**：零 kernel ABI 变更（读出侧不动）、零 graph 结构变更（只是多一个节点）。

### 3.2 Gate 设计

```rust
// chain_dev.rs（模块级，pub(crate)；与 bf16_truncate() 同款，OneLock 只读一次）
pub(crate) enum RingFormat { Off, Bf16, Fp8 }

/// DSV41_RING_FORMAT=bf16|fp8   （unset / "0" / "off" / "f32" ⇒ Off）
///
/// ⚠️ 一个 gate 同时驱动 backbone 与 draft 两侧的写入点。没有 per-side 开关，
/// 所以「只改一侧」不能通过配置发生 —— 这是 S4 的结构性前提（见 §5-1）。
pub(crate) fn ring_format() -> RingFormat { /* OnceLock */ }
```

语义：

| 值 | 动作 | 对齐到的目标 | 备注 |
|---|---|---|---|
| unset / `0` / `off` / `f32` | 什么都不做 | 现状（f32 原始值） | **默认**；必须与今天逐位一致 |
| `bf16` | `x ← bf16(x)`（就地） | 官方**容器 dtype** | 中间的隔离臂：只测 dtype，不测 fp8 网格 |
| `fp8` | `x ← e4m3_decode(e4m3_encode(clamp(x/2^k)))·2^k`（就地） | **官方值域**（即 `act_quant(inplace=True)`） | 真正的 S4 臂 |

> 命名说明：官方 = 「bf16 容器 + fp8 网格值」。所以 `fp8` 臂**已经蕴含** bf16 可表性
> （grid 值精确可表于 bf16），`bf16` 臂是一个更弱、用来做归因的中间臂。
> 若实测 `bf16` 臂就有收益而 `fp8` 不额外收益 → 说明起作用的只是 dtype 而不是网格。

### 3.3 共享 helper 与四个调用点

```rust
impl DevChain {
    /// 把 KV 行吸附到当前 ring format。`rows` 行、每行 `cols` 个 f32。
    /// Off ⇒ 零开销直接返回；bf16 ⇒ dsv41_bf16_roundtrip；fp8 ⇒ dsv41_act_quant_rt。
    fn snap_kv(&self, ptr: *mut f32, rows: i32, cols: i32) -> Result<()> { ... }
}
```

| 点 | 插入位置（HEAD `06861aa`） | 调用 |
|---|---|---|
| W1 | `chain_dev.rs:12751`（dual join 之后）与 `12758`（`let win`）之间 | `self.snap_kv(self.s.kv.ptr as *mut f32, 1, hd as i32)?;` |
| W2 | `chain_dev.rs:9040`（`apply_rope(kv_r)` 之后）与逐行循环之前 | `self.snap_kv(self.s.kv_r.ptr as *mut f32, m as i32, hd as i32)?;` |
| W3 | `dspark_dev.rs:3307`（`rope_at(mk)`）之后、`3319` 之前 | `self.snap_kv(self.mk.ptr as *mut f32, 1, hd as i32)?;` |
| W4 | `dspark_dev.rs:1963`（`rope_at(kv)`）之后、`2008`（memcpy 进 all_kv）之前 | `self.snap_kv(self.kv.ptr as *mut f32, bs as i32, hd as i32)?;` |

**顺序约束（照官方抄）：rope 之后、写入之前。** 官方是
`apply_rotary_emb → act_quant → 存入 ring`（`model.py:706-707/1041-1042/1061-1062`），
ferrite 的 `rmsnorm_rope`/`apply_rope` 都在写入之前，所以插入位天然满足。

**为什么按「源缓冲区」而不是按「ring 缓冲」吸附**：ring 一个缓冲里同时住着**窗口行**与
**压缩行**（`[0,win)` vs `[win, win+clen)`，`chain_dev.rs:2904`），而官方的压缩 KV 用的是
**另一种量化器**（fp4 / block16 / e4m3 scale，`model.py:760`）。对整块 ring 做吸附会错误地
用窗口行的格式去量化压缩行。按写入源吸附天然只覆盖窗口行。

### 3.4 需要的量化原语

| 臂 | 现成？ | 说明 |
|---|---|---|
| `bf16` | ✅ **已存在** | `Device::bf16_roundtrip`（`device.rs:1309`）+ `dsv41_bf16_roundtrip_kernel`（`dsv41_glue.cu:2126`）。零新代码。 |
| `fp8` | ❌ **需新增 1 个 kernel** | 现有 `dsv41_quant_fp8`（`dsv41_kernels.cu:3516`）只把 e4m3 **字节**写进另一个缓冲，不做就地反量化；全仓无「fp8 就地往返」kernel（`grep roundtrip/dequant` 只命中 bf16 与 fp4 残差）。 |

**新增 kernel 规格**（`kernels/cuda/dsv41_glue.cu`，紧挨 `dsv41_bf16_roundtrip`）：

```c
// 就地 act_quant(inplace=True)：x[i] = e4m3_decode(e4m3_encode(clamp(x[i]/s))) * s
// s 由每 (row, block) 的 amax 决定；round_scale=true ⇒ s = 2^ceil(log2(amax/448))。
// 数值契约：amax 归约树 / fast_round_scale / ±448 夹取 / e4m3 编码
//          必须与 quant_kernel<0>（dsv41_kernels.cu:126）逐项相同，
//          这样它等价于「quant_fp8 到 scratch + dequant 回写」但只有 1 个 launch。
extern "C" int dsv41_act_quant_rt(float* x, int rows, int cols, int block,
                                  int round_scale, cudaStream_t s);
```

配套接线（house style）：

1. `kernels.rs`：`extern "C" { pub fn dsv41_act_quant_rt(...) -> i32; }`
2. `device.rs`：`Kernels` struct 加 `act_quant_rt: Option<unsafe extern "C" fn(*mut f32, c_int, c_int, c_int, c_int, CuStream) -> c_int>`，
   `kernels()` 里 `ko!(rt, "dsv41_act_quant_rt")`；加 `Device::act_quant_rt(...) -> Result<bool>`。
3. **符号缺失时返回 `Ok(false)` 并打**一次**提示**（不要像 `bf16_roundtrip` 那样静默 no-op：
   静默会让整臂退化成 off 而 A/B 读成「无变化」，见 §5-3）。

### 3.5 成本

| 项 | 估算 | 说明 |
|---|---|---|
| 额外 launch | backbone 单行 +40/step；验证 +40/verify；draft +3~4 | W1/W2 是每层一个 |
| 时间 | ~+90~130us/step（按 ~2.5us/launch） | 在 10ms 目标上约 1%；33ms 懒臂上 <0.5% |
| 优化路径（仅当 S4 打赢才做） | 折进 `rmsnorm_rope` / `apply_rope` 的 epilogue | 先例：`apply_rope_q`（fuse rope + fp8 发射）、`sparse_attn_orope` phase 3 |

---

## 4. A/B 方案

### 4.1 固定基底（与 1.214 臂逐位相同，只动 `DSV41_RING_FORMAT`）

沿用 `scripts/s2_ab_matrix.sh` 头部记录的口径：

```
DSV41_SPEC=1 DSV41_DSPARK=1 DSV41_SIDS_WRITEBACK=1
DSV41_EXPERT_ACT_E4M3=1 DSV41_SH_EXP_MROWS=1 DSV41_DRAFT_P3A=1
DSV41_TIMING=1 DSV41_LAZY_VERIFY=1 DSV41_VERIFY_GRAPH=1
DSV41_BF16_TRUNCATE=1                     # 零拉丁红线，不可动
DSV41_TAP_INPUT=1 DSV41_DRAFT_BF16_DOMAIN=1   # P0-3 + P1-5 = 1.214 的两个杠杆
DSV41_SEED_POS 未设                        # S1 已判据：SEED_POS 降分
```

### 4.2 三臂矩阵

| 臂 | `DSV41_RING_FORMAT` | 读作 | 预期 |
|---|---|---|---|
| **A** | unset（Off） | 参考基线（必须复现 1.214，逐位） | — |
| **B** | `bf16` | 只对齐容器 dtype | ±0~0.2 |
| **C** | `fp8` | 对齐官方值域（真正的 S4） | **+0.2 ~ +0.6**（accept-first-strategy §3-S4 的区间） |

判定：`B-A`、`C-A`。**红线优先于 accept**：零拉丁 + 出师表逐字 + `DSV41_DIFF_EAGER` 的 `[diff]`。

### 4.3 可选诊断臂（把 §1.1 的警告从「教条」变成「数据」）

新增一个**仅供诊断**的子开关 `DSV41_RING_FORMAT_SIDE=both|backbone|draft`（默认 `both`）：

| 臂 | 设定 | 预期 | 价值 |
|---|---|---|---|
| **D** | `fp8` + `SIDE=draft` | **降分** | 直接验证「单侧 = 去相关 = 降 accept」 |
| **E** | `fp8` + `SIDE=backbone` | **降分** | 同上，并分离两侧贡献 |

若 D/E 不降反升 → §1.1 的整套「必须成对」推理被证伪，S4 的整体策略要重排。
此开关**不得**作为生产配置；实现上放在同一个 `ring_format()` 旁边的独立 OnceLock 即可。

### 4.4 实现前的零 GPU 验证（强烈建议先做，省掉一轮 serve）

1. **kernel 位精确性**：用 golden 的 `stage{s}.attn.quant.<tag>_pre → _post`（官方 `act_quant`
   的输入/输出原样快照，`/tmp/unit_golden.py` 的 `aq_wrap`）作为 oracle：
   拿 `_pre` 喂新 kernel，要求与 `_post` **逐位相等**。这一步同时**钉死 block size 与
   `round_scale` 语义**（见 U1），是全设计里最便宜、最强的一步。
2. **draft 侧整体口径**：`DSV41_DSPARK_UNIT_DUMP=1` + `DSV41_DSPARK_UNIT_INJECT=1`，
   对照 golden：吸附后 `kv_block` 应当对上 `quant.draft_kv_post`，
   `attn.sparse.kv` 应当对上 golden 的 `ring_before + draft rows`。
3. **红线快检**：`cargo test -p ferrite-models --lib dspark_parity -- --ignored`（device 自比，
   两侧同 gate ⇒ 不受影响，可确认没引入几何 bug）。

---

## 5. 风险评估

### 5-1 【最高】两侧漂移（accept-first-strategy 的核心警告）

- **风险**：S4 的前提是「draft KV 与 backbone/verify KV 必须在同一数值域」。
  只动一侧 ⇒ 两条数值路径去相关 ⇒ accept 塌（先例：`BF16_TRUNCATE` 单侧 1.080→0.820，
  `+DRAFT_BF16_DOMAIN` 双侧回到 1.214）。
- **缓解（结构性）**：唯一 gate + 唯一 helper；`dspark_dev.rs` 通过
  `crate::dsv41::chain_dev::ring_format()` 取格式（先例：`bf16_truncate`
  在 `dspark_dev.rs:2149/2953` 就是这么用的）。**没有** per-side 生产开关。
- **残留风险**：四个调用点里漏掉一个（尤其 W2 m-row 与 W4 块行），效果等同于单侧。
  **自检**：`fp8` 臂下用 unit-dump 对 `quant.*_post` 逐点核对，四个点各自可见。

### 5-2 【高】范围蔓延：环内存着三种 KV，只有窗口行该动

- ring = 窗口行 `[0,win)` + 压缩行 `[win, win+clen)`（`chain_dev.rs:2904`）。
- 官方窗口行 = e4m3/block32（`model.py:707`）；官方压缩行 = **fp4/block16/e4m3-scale**
  （`model.py:760`，另一个量化器）；indexer key 又是第三种。
- **draft 没有压缩行**（`DSparkAttention` 断言 `compress_ratio == 0`，`model.py:1034`；
  `get_dspark_topk_idxs` 只给 window + block，`model.py:1021-1029`）。
- ⇒ 压缩行**不在** S4 的「成对」范围内：动它只改 verify 不改 draft，按 §1.1 的规则是单侧动作，
  需要自己的 A/B（记为 **S4b**，独立 gate，本设计不做）。
- **缓解**：§3.3 的「按写入源吸附」天然把范围锁在窗口行。

### 5-3 【高】stale `.so` 静默降级成 off

- 若新符号不在 `.so` 里，`Ok(false)` 让它变成 no-op ⇒ 整臂 == A 臂 ⇒ A/B 读成「无变化」，
  进而误判「KV 域不是杠杆」。
- **缓解**：符号缺失时**一次性显式提示**（先例 `chain_dev.rs` 的
  `act_e4m3_skipped_note()`）；并在启动时打印解析出的 arm（`[ring-format] fp8`），
  让 A/B 日志自带臂名（对应 U1「arm 名必须记录」的教训）。

### 5-4 【高】红线：backbone ring 是交付文本的 KV

- W1/W2 的吸附改变 backbone 自己的 attention → logits → **已提交 token 流**。
  这不是「只影响 accept」的杠杆。
- **反方论据**：官方也吸附，所以对齐应当让 ferrite **更接近**官方文本；若当前 f32 已能给出
  零拉丁，吸附大概率仍能（官方就是在吸附下产出正确文本）。
- **缓解**：三臂每臂都跑红线；任一红线破 ⇒ 该臂的 accept 数字无效（`s2_ab_matrix.sh` 的既定判据）。
- **先例**：`DSV41_BF16_TRUNCATE` 首次打破零拉丁（`chain_dev.rs:11791` 附近自述）——
  dtype/值域改动**确实**能翻近 tie 的 argmax。

### 5-5 【中】block size 与 amax 下限的位精确性

- block：官方参考 `model.py:27` 是 32，实际传入三处 `act_quant`；ferrite 注释声称 128
  （`quant.rs:10` / `ops.rs:868` / `dsv41_kernels.cu:123`）。**必须钉死**（U1）。
- amax 下限：参考 `kernel.py:76` 是 `amax = max(amax, 1e-4)`；ferrite 的 `quant_kernel` 是对
  **scale** 取 `fmaxf(..., 1e-30)`（`dsv41_kernels.cu:144`）。两者仅在 `amax < 1e-4` 的块上分叉
  （KV 经 RMSNorm 后 amax ~O(1)，实际几乎不可能触发）。**建议新 kernel 照抄参考**（含 1e-4 下限），
  这样它对 golden 是位精确的；同一条也顺便解释了为何不能直接复用 `quant_fp8` 的现成写出。

### 5-6 【中】快照 / 前缀缓存 / CUDA graph

- `kv_snapshot`/`kv_restore`、`dspark_snapshot`/`rollback`、`kv_snapshot` 的 ring 段：
  都是**裸字节拷贝** ⇒ 格式无关；吸附在写入前完成，快照自然携带吸附值。**不需要改**。
- 跨进程前缀缓存不存在（`KvCache` 是进程内），所以不存在「A 臂快照被 C 臂恢复」。
- graph capture：新 kernel 是固定 shape 的普通 launch，capture-safe；gate 在启动时读一次
  （OnceLock），捕获期不会变。**注意**：verify graph 的 `DRY→capture` 路径里，
  新 launch 必须在 eager 与 captured 两条路上都出现（否则捕获后的 replay 少一个节点）。

### 5-7 【低-中】性能

- 见 §3.5：+~90-130us/step。若 S4 打赢，用 §3.5 的融合路径回收。
- **注意**：accept 与 step 是乘数关系，S4 的验收标准是 **accept**；步时只记录、不设门限。

### 5-8 【低】诊断口径的变化（会「看起来像」指标变差）

- W3/W4 就地在 `self.kv` 上吸附 ⇒ 现有 `dump_unit_idx("kv_block", ...)`
  （`dspark_dev.rs:1413`）从「post-rope 未量化」变成「post-quant」。
  对照脚本应改比 golden 的 `quant.draft_kv_post`，而不是 `rope.draft_kv_rope`。
  不然会用错基准，读成「unit diff 变大」的假回归。

---

## 6. 实现前必须钉死的未知（U 项）

| # | 未知 | 影响 | 取证方式（零/低成本） |
|---|---|---|---|
| **U1** | 官方 window KV 的 `act_quant` **block = 32 还是 128**？（`model.py:27` vs ferrite 的三处注释） | scale 粒度；错了整片错 | 用 golden 的 `quant.<tag>_pre → _post` 反推：32 与 128 各跑一遍，位精确命中即答案 |
| **U2** | 新 kernel 的 amax 下限是否照抄参考的 `1e-4`（而非 ferrite 的 scale 下限 `1e-30`） | 仅极小块；决定能否宣称「对 golden 位精确」 | U1 的同一对照 |
| **U3** | `DSV41_RING_FORMAT` 与 `Bf16` 是否与 `DSV41_DRAFT_BF16_DOMAIN` / `DRAFT_ATTN_BF16` **幂等/冲突** | 臂的可解释性 | 代码层面确认：它们分别动 `o`/`wo`/`xn`/`normed`，与 KV 缓冲不相交 ⇒ 正交；但 A/B 必须在同一基底下跑 |
| **U4** | m-row 与 single-row 在吸附后是否仍满足 verify parity（`step_rows(truth)[r] == single-row argmax`） | 若破 ⇒ 是几何 bug 不是建模问题 | `dspark_parity --ignored`（device 自比，两侧同 gate） |
| **U5** | S4 打赢后，压缩行（S4b，fp4/block16）要不要跟 | 决定后续路线 | 先不做；S4 的结论出来再排 |

---

## 7. 交付清单（若批准实现）

**必改（5 文件 + 1 kernel）**

1. `kernels/cuda/dsv41_glue.cu` — 新增 `dsv41_act_quant_rt`（紧挨 `dsv41_bf16_roundtrip`，`glue.cu:2126`）。
2. `crates/ferrite-models/src/dsv41/kernels.rs` — extern 声明。
3. `crates/ferrite-models/src/dsv41/device.rs` — struct 字段 + `ko!` + `Device::act_quant_rt`。
4. `crates/ferrite-models/src/dsv41/chain_dev.rs` — `RingFormat` 枚举 + `ring_format()` + `snap_kv()`；W1（~12751）、W2（~9040）两处调用。
5. `crates/ferrite-models/src/dsv41/dspark_dev.rs` — W3（~3307）、W4（~1963）两处调用（`ring_format` / `snap_kv` 从 `chain_dev` 取）。

**外加（非必需）**

6. `DSV41_RING_FORMAT_SIDE`（§4.3 诊断臂）——同一处 OnceLock，默认 `both`。

**不改**：任何读出 kernel、ring 缓冲 dtype、`compress_commit`、快照/恢复、host oracle
（`dspark.rs` / `chain.rs`；它们不参与 serve A/B，若做 device-vs-host 对照再另议）。

**测试**

- 新 kernel 对 golden `quant.*_pre/_post` 的位精确对照（§4.4-1）。
- `cargo test -p ferrite-models --lib dspark_parity -- --ignored`（§4.4-3）。
- 三臂 A/B（§4.2）+ 红线。

---

## 8. 一句话交付

> **官方 ring 存的不是 fp8 字节，而是「吸附到 e4m3 网格、容器 bf16」的值；
> 而 grid 值精确可表于 f32，所以 ferrite 不必换容器、更不必改读出 ——
> S4 退化成「在 4 个写入点上，用同一个 gate 驱动的同一个 helper，把 KV 行吸附到官方网格」。**
> **收益与风险同源：backbone ring 是交付文本的 KV，所以这既是 accept 杠杆，也会动红线。**
> **实现前只有一件事必须先钉死：`block` 是 32 还是 128（`model.py:27` 说 32，ferrite 自己的三处
> 注释说 128）—— 用 golden 的 `quant.<tag>_pre/_post` 一次对钉，顺带钉死 amax 下限。**

---

*工部 · 只读设计，未改动任何源码、未执行 GPU 命令；本文件为唯一产出。*
*代码事实均给出 `文件:行号`（基准 `06861aa`）；推断项已标注。*
