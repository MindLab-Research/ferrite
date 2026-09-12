# sglang DSpark 的实际 benchmark 分解数据

> 缘起：用户挑战既有口径——「accept ~5 / 13ms 步时」是**循环推导**（13ms = 5 ÷ 383.7 tok/s），
> 唯一硬数据是 383.7 tok/s。本文档去一手来源把分解数据钉死，并给出对我们策略的修正。
> 调研时间 2026-09-12，工作目录 `/home/smith/src/ferrite`。
>
> **一句话结论：博客里确实有非循环的实测分量——`target verify = 7.3 ms`。
> 它与 383.7 tok/s 一起，给出了 accept 的独立下界 τ ≥ 2.80 tok/step，并把
> 「13 ms 步时」从循环假设升级为自洽推导。同时纠正一处口径错误：DSV4 DSpark 是
> γ=5 / 6 token 步，不是 γ=7 / 8 token 步。**

---

## 0. TL;DR（先看这个）

| 问题 | 答案 | 证据强度 |
|---|---|---|
| sglang DSpark 实际步时 | **13.03 ms/步**（= 5 ÷ 383.7）；其中 **verify 7.3 ms（实测）** + 非 verify ≈ **5.7 ms（反推）** | verify 7.3ms = **一手实测**；13.03 = 推导（但被 7.3ms 佐证） |
| sglang DSpark 实际 accept | **τ ≈ 5 tok/步（含 bonus）⇒ mean-k ≈ 4 个 draft 被接受**（高接受度工作负载） | 博客自报 ~5 + GH200 复现 ~3.5–5.05 |
| draft 配置 | **γ = 5（block size 5）**，`--speculative-num-draft-tokens 6`；3 层 MoE + mHC + SWA128 + Markov head | 论文 §5.1 + sglang 源码 + GH200 启动命令（三方一致） |
| 非 MTP 纯 decode 基线 | B300 无公开绝对值；**GH200 独立复现：92.9 tok/s / 10.76 ms/token**（8K code prompt） | 独立第三方实测 |
| 硬件/配置 | **B300 × 1 节点 TP8，B=1**，DeepSeek-V4-Pro-DSpark，`RAGGED_VERIFY_MODE=compact` + ZOS overlap | LMSYS 博客 + release note |

**对我们的净影响：accept 与步时不是「谁优先」的问题，而是乘积约束 `τ / step ≥ 0.4 tok/ms`。
但 accept 决定「物理可达性」，步时决定「能否兑现」——见 §4。**

---

## 1. 硬数据（一手来源，按可信度排序）

### 1.1 【S 级】LMSYS 官方博客 —— 383.7 tok/s 的原始出处

`https://www.lmsys.org/blog/2026-07-06-dspark-sglang`（SGLang 集成 DSpark，PR #30261）

原句（§Performance optimizations and ZOS）：

> Together they reach **383.7 tok/s at accept length ~5** at batch size 1 on
> **DeepSeek-V4-Pro, TP=8, B300**.
>
> We rewrote the clusters of tiny ops as fused Triton kernels … In one example
> profile, **things outside the target verify shrinks by 1.7 ms, against a 7.3 ms
> verify.**

**⇒ 这是我们之前没拿到的关键：`7.3 ms` 是实测的 target verify 分量，不是从吞吐反推的。**

同时澄清一个歧义：`things outside the target verify shrinks by 1.7 ms` 语法上是
「**减少了** 1.7 ms」还是「**就是** 1.7 ms」无法只靠这句话判定。用算术排除：

- 读法 A（非 verify = 1.7 ms 绝对值）⇒ 步时 = 9.0 ms ⇒ τ = 383.7 × 0.009 = **3.45**，
  与博客自报 "accept length ~5" **矛盾** ⇒ 排除。
- 读法 B（非 verify 被优化掉 1.7 ms）⇒ 步时 = 5/0.3837 = **13.03 ms** ⇒ 非 verify ≈
  13.03 − 7.3 = **5.7 ms**，即优化前 ≈ 7.4 ms ⇒ 与 "accept ~5" **自洽**。

**⇒ 采用读法 B。这也是本文档最重要的一条：`accept ~5` 与实测 `verify 7.3 ms` 互相印证，
不是循环假设。**

其它可用的博客事实：

| 事实 | 值 |
|---|---|
| Figure 4 工作负载窗口（block size 6） | gsm8k **5.24** / arena-hard **3.78** / poetry **2.91** tok |
| window vs ceiling 利用率 | 0.88 – 0.97 |
| 步级分布 | gsm8k ~55% 的步填满 window 6；poetry ~80% 的步 ≤ 3 |
| 高并发 dynamic trim vs fixed budget | ~**+20%** 吞吐 |
| ZOS（overlap scheduler） | 比关掉时 **~1.5× tighter**（无 bubble） |
| 融合 kernel 收益 | 非 verify 部分 −1.7 ms |
| 调度成本模型 | `T(bs,K) = bias + alpha(bs) + theta(M), M = bs+K` |
| B=1 时 trimming 收益 | **≈ 0**（两条臂打平；trim 只在高并发挣钱） |

⚠️ 硬件归属提醒：Figure 1 / 3 / 6 是 **H200 × DP4（V4-Flash）**；Figure 5 与 383.7 tok/s
是 **B300 TP8（V4-Pro）**。7.3 ms 那句与 383.7 同段，判为 B300 TP8 B=1 的 profile，
但博客未逐字写明该 profile 的硬件——**置信度中高，非 100%**。

### 1.2 【S 级】sglang 源码（main 分支，实测抓取）

`python/sglang/srt/speculative/dspark_components/dspark_config.py`

```python
:21  DEFAULT_DSPARK_GAMMA = 7              # ← 仅「缺省 fallback」，不是 DSV4 的实际值
:26  DSV4_DRAFT_ATTENTION_BACKEND = "dsv4"
:51  def dspark_gamma_from_num_draft_tokens(num_draft_tokens: int) -> int:
:52      gamma = int(num_draft_tokens) - 1     # gamma = num_draft_tokens - 1
      ...
      "DSpark speculative_num_draft_tokens must be >= 2 (= gamma + 1)"
```

配套事实（issue #31018 / GH200 复现）：

- checkpoint `config.json` 里 `dspark_block_size = 5`；`--speculative-dspark-block-size`
  可以覆盖它（issue #31018 的例子用 `7`），不一致时只 **warn 不报错**。
- **DSV4 DSpark goal：block_size 5 ⇒ γ=5 ⇒ `--speculative-num-draft-tokens 6`。**

**⇒ 修正既有文档的错误口径**：`accept-first-strategy.md §2.1` 写的
「`DEFAULT_DSPARK_GAMMA = 7`（:17），gamma+1 = **8** token/步」是把**fallback 常量**当成了
实际配置，且行号也偏（实际 :21）。**sglang 在 DSV4 上是 6 token/步，与 ferrite 的
`dspark_block_size == 5`（`config.rs:594`）块长完全一致。**

### 1.3 【S 级】DSpark 论文（arXiv 2607.05147）

生产部署（§5.1）：

> The parallel backbone comprises **three MoE layers** with mHC and a **sliding window
> attention of 128**. We configure the **maximum block size to γ = 5** and utilize the
> **Markov head** for sequential modeling.

离线实验（§4.1）：block size **7**、draft **5 层**；Eagle3 TTT horizon 7 / 1 层；DFlash 5 层。
（**离线 7 与生产 5 是两个不同配置，别混用。**）

口径定义（§4.1 脚注 4，关键）：

> For clarity, unless otherwise stated, **all reported metrics for accepted length and
> acceptance rate include the target-generated bonus token.**

目标函数（Eq.1）：`L = (T_draft + T_verify) / τ`。

生产对比（§5.4，全部 vs **MTP-1 基线 = 静态 2 token verify**）：

| 引擎 | SLA 锚点 | DSpark 相对 MTP-1 |
|---|---|---|
| V4-Flash | 80 tok/s/user | 聚合吞吐 **+51%** |
| V4-Flash | 120 tok/s/user | nominal **+661%**（基线已到边界，作者自己说别当倍率看） |
| V4-Flash | **matched throughput** | per-user **+60% ~ +85%** |
| V4-Pro | 35 tok/s/user | 聚合吞吐 **+52%** |
| V4-Pro | 50 tok/s/user | nominal **+406%** |
| V4-Pro | **matched capacity** | per-user **+57% ~ +78%** |

调度行为（§5.4）：< 200（Flash）/ < 150（Pro）并发时，verify budget 从 MTP-1 的**静态 2**
扩到 **~4–6 token/request**；并发上去后动态收紧。

draft 长度代价（§4.3.2）：draft 长度 4→16 只给全轮延迟加 **0.2%–1.3%**（因为 target verify 主导）。

### 1.4 【A 级】GH200 独立复现（dnhkng，第三方）

`https://dnhkng.github.io/posts/gh200-benchmarking-part-4-dsv4-released`
硬件：2× GH200（Hopper 96GB，NVLink 桥仅 ~58 GB/s）。**不是 B300，但这是最干净的独立分解数据。**

**vLLM DSpark k-sweep**（8K code prompt，code review，2048 out，3 次中位数）：

| k (num_speculative_tokens) | Median TG | Median TPOT | 备注 |
|---|---|---|---|
| **0（无 spec）** | **92.9 tok/s** | **10.76 ms** | **纯 decode 基线** |
| 4 | — | — | **非法**（< block size 5） |
| 5 | 268.2 tok/s | 3.73 ms | |
| **6** | **275.9 tok/s** | **3.62 ms** | **最佳**（= 2.97× no-spec） |
| 7 | 262.2 tok/s | 3.81 ms | 回落 |
| 8 | 216.3 tok/s | 4.62 ms | 回归 |
| 10 | 223.9 tok/s | 4.47 ms | 回归 |

SGLang（修完 loader bug 后）：code review **308.6–317.0 tok/s**；story chat **178.5–191.4 tok/s**。
并发曲线（c1→c8）：263.0 / 223.2 / 177.8（agg 630.3） / 122.1（agg 598.4）tok/s。

**两个可直接引用的实测**：

1. **loader bug 的 accept 影响**：shared expert 没 remap 时 story decode 115.7 → 176.0 tok/s，
   code-review acceptance **从 ~1.3–1.8 提到 ~3.5–5.05 accepted tokens**。
   ⇒ 「accept ~5」在 sglang 侧是**修完 bug 才能拿到**的，不是白送的。
2. **accept 强依赖工作负载**：作者原话「V4 Flash 的 DSpark head 是针对 Code 微调的」
   ⇒ code 高、prose 低。这与 Figure 4 的窗口 5.24 / 3.78 / 2.91 一致。

### 1.5 【B 级】其它二手汇总（数字与上面一致，仅作交叉印证）

- SGLang release note / LMSYS X：「383.7 tok/s at accept length ~5 on DeepSeek-V4-Pro,
  **TP8 on B300 (bs=1)**；`--speculative-dspark-block-size` 调 block」。
- alphasignal / dreaming.press：明确提醒 383.7 是 **best-case, low-batch, latency-bound**
  「ceiling, not the number you'll see under load」。

---

## 2. 非循环推导链（本文档的核心产出）

### 2.1 从 383.7 tok/s + 7.3 ms verify 能得到什么

verify 在关键路径上 ⇒ **step ≥ 7.3 ms** 是硬下界。于是：

```
τ ≥ 383.7 tok/s × 0.0073 s = 2.80 tok/step          ← accept 的独立下界（非循环）
```

再叠上博客自报的 accept ~5：

```
step = 5 / 383.7 = 13.03 ms
非 verify = 13.03 − 7.3 = 5.73 ms   （优化前 ≈ 7.4 ms，与 1.7ms 的削减自洽）
```

**⇒ 「13 ms 步时」不再是「从吞吐反推的空假设」：它是 `accept ~5` + 实测 `verify 7.3ms`
的联合推论，且反过来被 5.7ms 这个量级合理的非 verify 开销佐证。**

### 2.2 与 ferrite 的对照（同口径）

| 量 | sglang（B300 TP8 B=1，V4-Pro） | ferrite（DSV41，B=1） | 比值 |
|---|---|---|---|
| 步时 | 13.03 ms | **22.56 ms**（lazy verify，无截断） | **1.73×** |
| 其中 verify | **7.3 ms（实测）** | 22.56 ms（全部） | 3.09× |
| accept（tok/step，含 bonus） | ~5.0 | **2.214**（mean-k 1.214） | **2.26×** |
| 吞吐 | 383.7 tok/s | **98.1 tok/s** | **3.91×** |

自洽性检查：`1.73 × 2.26 = 3.91 = 383.7 / 98.1` ✅ ——两条独立路径的数对上了，
说明 §2.1 的读法 B 是自洽的（读法 A 会在这一列崩掉）。

### 2.3 口径修正：块长其实一样，缺口全在 per-token 命中率

因为 sglang 也是 **block 5**（见 §1.2），两边块长相同：

| 实现 | block | τ（含 bonus） | mean accepted m=τ−1 | 几何 p（截断在 5） |
|---|---|---|---|---|
| sglang DSV4 | 5 | ~5.0 | **~4.0** | **p ≈ 0.93** |
| ferrite DSV41 | 5 | 2.214 | **1.214** | **p ≈ 0.56** |

（p 由 `Σ_{k=1..5} p^k = m` 反解；p=0.56 ⇒ 1.20 ✅，p=0.93 ⇒ 4.04 ✅）

- per-token 比：**0.93 / 0.56 ≈ 1.66×**（既有文档算的 ~1.5× 方向正确）。
- 但**截断的非线性**把 1.66× 放大成 3.3×（因为 p=0.93 时已经接近打满 5 个槽）。

**⇒ 两个修正**：
1. 既有文档「**一半的 4× 缺口是块长**」**不成立**——块长两边都是 5，缺口 100% 来自 per-token p。
2. 既有文档的「复刻 sglang 的 p ⇒ 上限 **mean-k 3.16**」**偏悲观**：
   在 block 5 上复刻 p=0.93 的几何上限是 **mean-k ≈ 4.0 / τ ≈ 5.0**（3.16 是按错误的 γ=8 算的）。

---

## 3. 五个问题的直接答案

1. **实际步时**：13.03 ms/步（V4-Pro/B300/TP8/B=1）。**实测分量：target verify 7.3 ms；
   非 verify ≈ 5.7 ms（反推）。** B=1 时 trimming 无收益（trim 是高并发杠杆）。
2. **实际 accept**：τ ≈ 5 tok/步（含 bonus），mean-k ≈ 4，per-token p ≈ 0.93；**但在
   `frontier_prompt.txt`（16 道 GSM8K 拼接）上测得，属高接受度工作负载**；同引擎
   poetry 只有 ~2.91。GH200 独立复现给出 3.5–5.05（code）与 ~1.3–1.8（修 bug 前）。
3. **draft 配置**：**γ = 5**（block size 5，`num_draft_tokens 6`）；draft = **3 层 MoE**
   + mHC + **SWA 128** + **Markov head**（生产）。离线论文用 γ=7 / 5 层。
   `DEFAULT_DSPARK_GAMMA = 7` 只是 fallback，**不要当成 DSV4 的实际配置**。
4. **非 MTP 纯 decode 基线**：B300 上 LMSYS 只给了曲线没给数；**可引用的硬数是 GH200 的
   92.9 tok/s = 10.76 ms/token**（同 prompt 下 DSpark k=6 是它的 2.97×）。
   ferrite 自己的纯 decode 是 6.15 ms = 162.6 tok/s（⚠️ 硬件/模型/文本都不同，不可直接对比）。
5. **硬件与配置**：单节点 **B300 × TP8，B=1**，DeepSeek-V4-Pro-DSpark，
   `SGLANG_RAGGED_VERIFY_MODE=compact` + ZOS overlap + 全 CUDA graph +
   `SGLANG_DSV4_FP4_EXPERTS=1`；`--mem-fraction-static 0.82`、`--cuda-graph-max-bs 4`。

---

## 4. 对 ferrite 策略的影响：accept vs 步时

### 4.1 把目标写成乘积约束

400 tok/s ⟺ `τ / step ≥ 0.4 tok/ms`，其中 `τ = mean-k + 1`。展开成「给定 accept，步时上限」：

| mean-k（accepted drafts） | τ（tok/step） | 400 tok/s 允许的最大步时 |
|---|---|---|
| **1.214（ferrite 现状）** | 2.214 | **5.54 ms** |
| 1.92 | 2.92 | 7.30 ms ← 恰好 = sglang 的 verify 实测值 |
| 2.60 | 3.60 | 9.00 ms ← 既有文档的「L5 floor 8–9ms」 |
| 3.16（旧文档的乐观上限） | 4.16 | 10.40 ms |
| **4.00（复刻 sglang 的 p）** | 5.00 | 12.50 ms |
| 5.00（block 5 打满） | 6.00 | 15.00 ms |

### 4.2 单轴都不够（都不是「优先」，是「都必需」）

| 只动一轴 | 极限 | 结果 |
|---|---|---|
| 只提 accept，步时停在 22.56 ms | τ 最大 6.0（block 5 打满） | **265.9 tok/s** ❌ |
| 只降步时，accept 停在 2.214 | 步时压到 7.3 ms（= sglang 的 verify 实测） | **303.3 tok/s** ❌ |
| 双动（sglang 的点） | τ=5.0，步时 13.03 ms | 383.7 tok/s ✅ |

**⇒ 「accept 优先」的说法需要精确化：**

- **accept 决定物理可达性**：`τ < 2.92` 时，即使步时压到 sglang 的 verify 实测下界 7.3 ms，
  也拿不到 400（303 tok/s 封顶）。ferrite 现在 τ=2.214，**在「步时无限好」的假设下都到不了 400**。
  这条论证现在有**实测锚点**（7.3 ms），不再是估计。
- **步时决定能否兑现**：accept 拿到 4.0 后，步时必须 ≤ 12.5 ms；停在 22.56 ms 只有 177 tok/s。
- **结论**：accept 是**门槛（necessary）**，步时是**兑现（sufficient 的一半）**。
  正确的表述是「**先过 accept 门槛（mean-k ≥ 1.92），再拼步时**」，而不是「只做 accept」。

### 4.3 对既有计划的具体回改建议

| 既有结论（docs/agent） | 本研究后的修正 |
|---|---|
| `accept-first-strategy.md §2.1`：sglang γ=7 / 8 token 步；块长占一半缺口 | ❌ **错**。sglang = γ5 / 6 token 步；块长两边相同，缺口全在 per-token p（1.66×，被截断放大到 3.3×） |
| `accept-first-strategy.md`：复刻 p ⇒ 上限 **mean-k 3.16** | ⚠️ 偏悲观。同 block 5 上复刻 p≈0.93 的上限是 **mean-k ≈ 4.0 / τ ≈ 5.0** |
| S1–S5 路径（1.214 → 1.95/3.10） | ✅ 方向对，但**终点要抬到 mean-k ≈ 4** 才够 400（S5 的 3.10 仍差一口气） |
| `dd62986`：400 在 accept 1.214 下物理不可能 | ✅ **被独立证实**（新锚点：verify 7.3 ms ⇒ τ 下界 2.80） |
| `dd62986`：lazy beats batched / 10× 更少 kernel / 4× per-row | ✅ 与 sglang 的 7.3ms/6 行 ≈ 1.22 ms/row vs ferrite 22.56/6 ≈ 3.76 ms/row（**3.1×**）量级一致 |
| 步时侧杠杆（融合 kernel、单 graph、overlap） | 🆕 sglang 的实测收益是**非 verify −1.7 ms + ZOS ~1.5× tighter**，正好是 ferrite 步时侧的两个可对标项 |

### 4.4 新增的、必须知道的 caveat

1. **383.7 是高接受度工作负载的数**（frontier_prompt = 16 × GSM8K）。任何拿它跟
   ferrite 在自己中文/长文上测的 accept 1.214 做的对比，**必须同文本**，否则口径不可比。
2. **B=1 时 sglang 的动态截断不贡献任何收益**（博客明说两臂打平）。所以 383.7 来自
   「**深 draft（3 层 MoE）+ 高 p + 融合 kernel + ZOS**」，不是来自 confidence scheduler。
   **ferrite 要复刻的是这四件事，不是 scheduler。**
3. **7.3 ms 那句的硬件未逐字写明**（同段 = B300 TP8 V4-Pro，置信度中高）。
   若它其实是 H200 的数，则 sglang 真实 verify 更快、ferrite 的缺口更大——**方向不变，幅度更差**。
4. **accept ~5 依赖 loader 正确**：GH200 上 shared expert 漏 remap 直接把 accept 打到 1.3–1.8。
   ferrite 的 accept 1.214 处在**同一个量级** ⇒ 值得优先自查「我们是否也有一个类似的
   silent loader/mapping 缺口」，这与 `accept-first-strategy.md §2.3` 的结构层怀疑一致。

---

## 5. 来源清单

| # | 来源 | 关键贡献 | 可信度 |
|---|---|---|---|
| 1 | `lmsys.org/blog/2026-07-06-dspark-sglang` | 383.7 tok/s；**verify 7.3 ms**；Figure 4 窗口 | S（一手） |
| 2 | sglang `main`:`dspark_components/dspark_config.py` | `DEFAULT_DSPARK_GAMMA=7`（:21，fallback）；`gamma = num_draft_tokens-1`（:51-52） | S（一手源码） |
| 3 | arXiv 2607.05147v1（DSpark 论文） | γ=5、3 层 MoE、SWA128、τ 含 bonus、MTP-1 静态 2、+51%/+60-85% | S（一手） |
| 4 | `dnhkng.github.io/.../gh200-benchmarking-part-4-dsv4-released` | no-spec 92.9 tok/s；k-sweep；loader bug 的 accept 影响；并发曲线 | A（独立第三方） |
| 5 | sglang issue #31018 | checkpoint `dspark_block_size=5` vs `--speculative-dspark-block-size 7` 的冲突 | A |
| 6 | SGLang release note / LMSYS X / alphasignal / dreaming.press | 交叉印证 B300 TP8 B=1 与「ceiling not under-load」 | B（二手） |

---

## 6. 未解 / 需要进一步确认

1. **7.3 ms 的硬件归属**（B300 vs H200）——决定缺口是 3.1× 还是更大。
2. **sglang 侧是否有 B300 的 `SGLANG_ENABLE_METRICS_DEVICE_TIMER` 公开数字**（可复现的
   step_gpu_ms）。博客的 SPS profiler 命令给了：`SGLANG_DSPARK_ENABLE_SPS_RECORD=1`
   + `SGLANG_SIMULATE_ACC_LEN=1.0` + `python3 -m sglang.benchmark.dspark_sps_profiler all`。
3. **ferrite 的 accept 1.214 是在哪份文本上测的**——必须与 sglang 的 GSM8K prompt 同口径才能下结论。
4. **sglang 的 3 层 MoE draft 的每步成本**（论文说 draft 侧是「fixed cost，低接受度时不可回收」）
   ——这是 ferrite 判断「要不要加深 draft」的前置数据。
