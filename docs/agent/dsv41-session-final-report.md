# DSV4.1-Flash 会话最终报告（定稿）

> **状态：已定稿**（2026-09-12 00:20，HEAD `8b32bcf`）。本文件是会话的权威汇总，
> 数字全部来自本会话已验证的 serve A/B 或隔离复现器；未验证项已显式标注。
>
> **会话终态**：**13.28 → 6.26ms（+112.2%），75.3 → 159.7 tok/s**。
> v16（gate 修复后）定案 **6.26ms / 159.7 tok/s**，四段文本全对、`faults=0`。
> **通往 200+ 的唯一数量级路径已确认：swapAB**（预估 ~238 tok/s），实施中。

---

## 0. 一句话结论

会话把单请求 decode 步时从 13.28ms 砍到 **6.26ms**（160 tok/s），并**彻底解开**了
v13→v15 连续三轮的 6.6ms 谜团——根因**不是**"热点 kernel 代码存在性"（residue-hunt 的误诊），
而是 commit `006bd0c` 的 `FileReplace` **改错了同名变量**：
把 K-split（已验证 **−0.33ms** 收益）误关、把 PDEPTH pipeline（**+0.04ms** 回归）误开，
净 **+0.37ms 隐藏回归**。这是一条**代码卫生 > 性能分析**的教训。

---

## 1. 优化链全景表（13.28 → 最终）

| # | 优化 | p50 ms | 累计收益 | commit | 验证 |
|---|---|---|---|---|---|
| 0 | 会话基线 | 13.28 | 75.3 tok/s | — | — |
| 1 | shared expert TP 切分 | 11.85 | +10.8% | `9cddfb8` | serve A/B，四段全对 |
| 2 | sparse 3-deep key-split + e4m3 LUT + 融合系列 | 9.38 | +29.4% | `710c107`、`54f470b`、`9e4e3df`、`2d7eead` | serve A/B |
| 3 | gate v2 + gateup MLP unroll + down revert | 8.60 | +35.2% | `65acfb1`、`73e4a50`、`5013268` | serve A/B |
| 4 | hc tail split + swiglu_q | 8.23 | +38.0% | `7d623ad` | serve A/B |
| 5 | allv2（K-split + P1 a32 + B+C + dead slots + launch_bounds） | 6.84 | +48.5% | `f019237`、`a6c4ed3`、`d76a183`、`c12ff60` | serve A/B |
| 6 | cp.async 权重先行 + warps8 + EARLY 侧流 | 6.66 | +50.5% | `f6aa8c8`、`a7600db`、`9558491` | serve A/B |
| 7 | down 回归修复 + dots-LATE merge | 6.48 | +51.4% | `96f169b`、`1b265a8` | serve A/B |
| 8 | AR store 并行化（grid.y=world） | 6.47 | +51.5% | `f911abd` | serve A/B |
| 9 | P4 act-cpasync（默认 ON） | **6.24** | **+112.8%** | `fdd70d4` | serve A/B，160.3 tok/s |
| — | *v10–v13 回归区（见 §4）* | *6.31→6.63* | *回退* | — | — |
| 16 | **gate 错配修复**（ksplit=2 / pipeline=1） | **6.26** | **+112.2%** | `4bed9f6` | serve A/B，159.7 tok/s ✓ |

> v9（P4 act-cpasync，6.23–6.24ms / 160.5 tok/s）是会话中的最优读数；v16 回到 6.26ms
> （差 0.03ms 在噪声内），即"基线恢复"。**"最终值"取 6.23–6.26ms / 160–161 tok/s。**

---

## 2. 落地优化清单（全部 serve A/B 验证）

| # | 优化 | 机制 | 来源 |
|---|---|---|---|
| 1 | **shared expert TP 切分** | 共享专家从"仅 rank0 串行"改为所有 rank 各切 `inter/world` | `9cddfb8` |
| 2 | **sparse 3-deep 预取 × key-split** | 同一 kernel 加 key 分块，grid=(C,b·m,h)，per-slot 78→10.7ns | `710c107` |
| 3 | **e4m3 LUT** | fp8 解码从位运算改 256-entry smem LUT | `54f470b`/`9e4e3df` |
| 4 | **gate v2** | MoE gate 的 bf16 gemv（n=384，延迟受限 3% 峰值带宽）向量化 + K-split | `65acfb1` |
| 5 | **gateup MLP unroll** | `#pragma unroll 2→4` + uint2 宽加载 | `73e4a50` |
| 6 | **hc tail split** | tail 拆成 EARLY（主流）/LATE（侧流）两半，侧流 fork/join | round-41 |
| 7 | **allv2（P1 a32 dead-slot + B+C）** | a32 物化直写 `s_af`（消 `s_a` 中转，smem 48512→43392B，4→5 blk/SM） | `f019237`/`a6c4ed3` |
| 8 | **cp.async 权重先行 + warps8 + EARLY 侧流** | prologue 与权重读重叠；n≥2048 时 warps=8 减半 block | `f6aa8c8`/`a7600db` |
| 9 | **down 回归修复 + dots-LATE merge** | 去 `__launch_bounds__(256,4)`（4→6 blk/SM）+ 恢复 ILV；dots 与 LATE 合一个 launch | `96f169b`/`1b265a8` |
| 10 | **AR store 并行化** | `grid.y=world`（5→40 blocks，3%→25% SM），每 block 只写一个 peer | `f911abd` |
| 11 | **P4 act-cpasync（默认 ON）** | 激活 staging 用 cp.async，消除 a32 物化的串行 LDG | `fdd70d4` |
| 12 | **COMPRESS_FUSE 3→1** | decode compressor 的 state+pool+commit 合一 launch，逐字节不变 | `595437d` |
| 13 | **gate 错配修复** | 恢复 K-split=2（−0.33ms）、PDEPTH pipeline=1（去 +0.04ms） | `4bed9f6` ✓ |

**旁证**：`175462d` 的 fold 物理清理与 `3570524` 的 hot-kernel-restore 本身也是"落地"——
它们不是为兑现收益，而是**修复缺陷 gate、消除 ABI 风险、恢复 np1 形态**（见 §4）。

---

## 3. 失败关停清单（全部有机制级解释）

| # | 优化 | 预期 | 实际 | 根因 | 处置 |
|---|---|---|---|---|---|
| 1 | **PDEPTH pipeline**（2/5） | −0.48ms | **+0.04ms** | 占用率损失 > 延迟隐藏（42.8KB smem → 2 blocks/SM）；隔离时长 kernel 掩盖占用率 | 默认 `GATEUP_PIPELINE=1`（OFF） |
| 2 | **w2 L2 prewarm** | −0.25ms | **+0.04ms** | warmer 与 gateup 尾部争 SM（PDL 重叠变争抢） | `W2_PREWARM=0`（OFF） |
| 3 | **quant fold** | −0.072ms | **+0.39ms**（与 swiglu 合计） | fork_ev 是 **kernel 级**——EARLY 加工作 = 加到 main 关键路径；且 gate 缺陷使 fold 一直生效 | 代码物理删除（`175462d`） |
| 4 | **swiglu fold** | −0.068ms | ↑ 同上 | w2 prologue 的 swiglu 在 GEMV 上下文比独立 kernel 贵（寄存器/布局） | 代码物理删除 |
| 5 | **AR stamp fold** | −0.15ms | **29.5s/step** | 单调 counter 在图 replay 下机制坏掉（poll 自旋等永远不来的 stamp） | 代码物理删除（`175462d`） |
| 6 | **sparse-merge 选举折叠** | −0.05~0.10ms | **中性**（v14a=v14b=6.61ms） | 真中性；但其代码存在性被误判为 +0.34ms 回归源 | 整体回退（`3570524`） |
| 7 | **ringwin 全折叠（RW_FOLD）** | −0.02~0.05ms | 未上机 | 未验证即进树；给热点 `rmsnorm_rope_kernel` 加 6 尾参 | 整体回退（`3570524`） |
| 8 | **a32-vec4** | −0.13ms（隔离） | **中性** | P4 的 cp.async 已隐藏 staging，vec4 在其上增益≈0 | 保留（无害） |
| 9 | **AR reduce grid** | −0.08~0.16ms | **中性** | 8µs 主导项是 stamp/poll 的 NVLink 往返（~6µs），网格形状只影响 1–2µs | 保留 |
| 10 | **wo-pair 两段核** | −0.13ms | **+0.46ms** | 共享 smem 池按 max(k)=5120 分配 → phase2 占用率 96→20 warp/SM（4.8× 崩塌） | `WO_PAIR` 保持 OFF |
| 11 | **sh-pair** | −0.26ms | 未上机 | 与 wo-pair 同根因且更极端（k 比 8:1） | 审查后放弃 |
| 12 | **M>1 expert batching** | — | 价值 = 0 | 单请求无第二个 token；层内 6 专家已在一个 launch | 关闭 |
| 13 | **expert MMA 化** | 绕过指令地板 | 前提不成立 | sm_103a 无 M=8/16 的 fp4 MMA（tcgen05 mxf4 的 M 钉死 128）；M=128 masked 实测 0.2% 峰值 | 关闭 |
| 14 | **Stage C persistent 段核** | −1.0~1.5ms | 最小版仅 0.1~0.2ms | smem 池按 max(k) 分配 → 占用率崩塌 + `grid.sync` 与整步图捕获不兼容 + 段内近全串行 | 关闭 |
| 15 | **cross-layer pipe** | −0.3~0.5ms | −0.05~0.15ms | 80% 被路由依赖挡死（L+1 专家取哪 6/384 由 gate(xn) 决定） | 关闭 |
| 16 | **bf16-lut（a32 物化的 bf16 LUT）** | −0.1ms | **分析否证（未落地）** | 位序错配：`*(bf162*)(lut + b4)` 取 `LUT[c0]`/`LUT[c0+1]`，而 `o.y` 要 `LUT[c1]` ⇒ 49.8% 元素错 | 关闭（勿再提案） |
| 17 | **cvt 解码路线** | 省 smem gather | **分析否证（未落地）** | 转换管 16 results/clk/SM < LDS 32 值/clk/SM；且丢位一致 | 关闭（见 §5） |

---

## 4. v10 → v16 完整侦探链（本会话最有价值的知识）

> 这段是本会话的核心：一条 **0.37ms 的隐藏回归**如何在 5 个 commit 里被反复误诊，
> 最终从"性能问题"变成"代码卫生问题"。

### 4.1 v10（07:00）：三项组合回归 +0.08ms

| 臂 | p50 | tok/s | 判定 |
|---|---|---|---|
| v10（PDEPTH=5 + w2warm + bf16cpasync） | 6.31ms | 158.5 | **+0.08 回归** |

四段全对，faults=0。**operand-supply 理论在 serve 中未兑现**（第三次隔离→生产失效）。

### 4.2 v10 bisect（07:30）：三项各自 +0.04 / +0.04 / 中性

| 臂 | p50 | 判定 |
|---|---|---|
| v10np1（pipeline OFF） | 6.27ms | pipeline 贡献 +0.04 |
| v10np2（PDEPTH=2） | 6.30ms | 浅 pipeline 也 +0.03 |
| v10nw（w2warm OFF） | 6.27ms | warm 贡献 +0.04 |
| bf16cpasync | — | 中性（两臂一致推出） |

**处置**：PDEPTH 默认 → 1（OFF）、W2_PREWARM 默认 → 0（OFF）、bf16 保持 ON。
> ⚠️ **这次"翻默认值"的提交 `006bd0c` 就是后续所有谜团的种子**——见 §4.7。

### 4.3 v11 崩溃（08:00）：部分提交的 FFI 错位（709）

`git add -A` 的 docs 提交扫入了 quant-early-fold subagent 的**部分实施**：
新 `.cu`（`hc_mixes_tail_kernel` 多 `xq4/xsc4`）+ 旧 Rust（旧 FFI）= 参数错位 = **cuda error 709**。
修复 `106db01` 补上 Rust 侧。**这是第 4 次部分提交事故，第一次真正崩溃**。

### 4.4 v12（08:30）：三个 fold 全部失败

| 臂 | p50 | 判定 |
|---|---|---|
| v12（quant fold + swiglu fold ON） | 6.59–6.62ms | **+0.36~0.39 回归** |
| v12sf（+ stamp fold ON） | **29.5s/step** | **灾难性失败（~4700× 慢）** |

- **quant fold**：fork_ev 在 EARLY kernel **完成时**记录 ⇒ fp4 直出 +1.8µs × 80 fronts = +0.14ms，
  省的 quant launch 只有 −0.072ms ⇒ 结构性净负。
- **stamp fold**：单调 arrival counter 在图 replay 下机制坏掉（poll 等永不到来的 stamp）。
- 处置：QUANT_FOLD / SWIGLU_FOLD / AR_STAMP_FOLD 全部默认 OFF。

### 4.5 v13（09:30）：fold 代码全 OFF，回归仍在

| 臂 | p50 | 判定 |
|---|---|---|
| v13（fold 代码全 OFF + compress-fuse ON） | 6.63ms | +0.40 回归 |
| v13nc（COMPRESS_FUSE=0） | 6.62ms | compress-fuse 不是源 |
| v13nq（全 gate 显式 OFF） | 6.60ms | gate 无关——初判"代码存在性"是源 |

**寄存器假设证伪**：remote `nvcc -Xptxas -v` 显示 pre-fold 与当前都 40 registers（无膨胀）；
hc_mixes_tail 在 decode 只有 1 个 block——占用率不是变量。

**fold-correctness-audit（10:00）**：quant-fold 的 **gate 有缺陷**——`=0` 时 `layer()` 仍传非空 `xq4`
（`chain_dev.rs`），kernel 只看 `xq4 != nullptr` ⇒ **fp4 直出照做**。所以 v13 ≈ fold 一直生效。

**`175462d`**：fold 代码物理清理（847 删除 / 14 文件），修复 gate 缺陷 + 消除 ABI 风险。

### 4.6 v14（11:00）：清理无效！——侦探链的分水岭

| 臂 | p50 | 判定 |
|---|---|---|
| v14（fold 清理 + sparse-merge + compress-fuse） | **6.60ms** | **清理未恢复**（回归仍在） |
| v14a（`SPARSE_MERGE_FOLD=0`） | 6.61ms | — |
| v14b（`SPARSE_MERGE_FOLD=1`） | 6.61ms | sparse-merge **真中性** |

**residue-hunt（假设 A）**：+0.34ms = "热点 kernel 编译产物重量"——
归因于 `sparse_attn_split_kernel` 的 **+12 运行时参数 + 选举代码**（gate OFF 无法编译期消除），
以及 `rmsnorm_rope_kernel` 的 6 尾参同款风险。
**sparse-gate-analysis**：gate 有效（选举不执行）但"代码存在性"有害。

**`3570524`（hot-kernel-restore）**：据此回退 **499 删除**，两个 kernel 恢复 np1（`d546139`）形态；
`c4220bb` 修复误删的 `rmsnorm_rope` launch 行（构建修复）。

> ⚠️ **假设 A 是误诊**——v15 未能恢复（见下）。

### 4.7 v15（00:12）：错误默认值仍 6.59ms → gate-hygiene-audit 揭穿真相

| 臂 | p50 | tok/s | 配置 |
|---|---|---|---|
| v15 | 6.59ms | 151.7 | ksplit=1（**误关**）+ pipeline=2（**误开**） |

**真相（gate-hygiene-audit）**：commit `006bd0c` 的 `FileReplace` 以 `int v = 2` 作 `old_string`，
**命中了错误的同名变量**：
- 打中 `dsv41_gateup_ksplit`（`dsv41_experts_mxf4.cu:869`，**已验证 −0.33ms 收益**）→ 被改成 `1`（关闭）
- 而本该改的 `dsv41_gateup_pipeline`（`:965`，**+0.04ms 回归**）原样停在 `2`（开启）

净效果 = **−0.33ms 收益丢失 + 0.04ms 回归保留 = +0.37ms 隐藏回归**，
这就是 v13/v14/v15 全部 6.6ms 的原因。**"热点 kernel 代码存在性"理论是误诊。**

**`4bed9f6`**：恢复 `dsv41_gateup_ksplit` 默认 = 2、`dsv41_gateup_pipeline` 默认 = 1。

### 4.8 v16（00:20）：基线恢复，谜团彻底解开 ✓

| 臂 | p50 | tok/s | 配置 |
|---|---|---|---|
| v15（错误默认） | 6.59ms | 151.7 | ksplit=1 + pipeline=2 |
| **v16（gate 修复）** | **6.26ms** | **159.7** | ksplit=2 + pipeline=1 |

**恢复 0.33ms**（K-split 的验证收益）。四段全对，`faults=0`。

### 4.9 侦探链的元教训

1. **两次"代码存在性"假设都失败了**（v13 的 fold 代码 → 实为 gate 缺陷；v14 的 hot kernel 代码 → 实为 gate 错配）。
   两次真根因都是 **gate 相关**。
2. **"改默认值"是高风险操作**：`FileReplace` 用变量声明（如 `int v = 2`）作锚点时，
   同文件里若有多个同形声明，会静默改错对象。改 gate 默认值必须**按函数名/上下文锚定**，
   改完**读回确认**。
3. **hot-kernel-restore（499 删除）是错诊下的过度清理**——但清理本身无害
   （sparse-merge 确实中性、ringwin 未验证，都不值得保留），所以**不需要回滚**。

---

## 5. lut-floor-research 定案（突破 LUT-gather 地板的研究）

**背景**：200 tok/s 判定指出剩余 ~1.05ms 必须突破 **LUT gather 地板**
（gemm a32 物化 1.55µs/call × 246 + expert gateup LUT-gather）。

| 结论 | 内容 |
|---|---|
| **业界无"激活解码层"** | FlashInfer / DeepGEMM 等对 M=1 decode 没有任何把 LUT 解码当卖点的 kernel——本仓库的 e4m3 LUT 物化是自创路径，业界走 MMA 直吃 fp8/fp4。 |
| **swapAB 是唯一的数量级路径（2×）** | `mma.m16n8k32` A/B 对调：权重做 A（M=输出行）、激活做 B（N=token）。利用率 **1/16 → 1/8**（16 行 M 只 1 token 有效 → 8 列 N 只 1 token 有效）。布局**零转置**（权重已 row-major `[n,k]`、激活 k 连续 = col-major B 的列 0），per-32-block scale 方案**已存在**（dense kernel `:320-329`）。a32 物化 + LUT 解码**整体删除**。目标 ~1.1–1.5µs/call；预估步时 4.2ms ≈ **238 tok/s**。工作量：新 kernel ~150 行 + launcher + Rust dispatch + parity test（fuse 族迁移另计）。 |
| **a32 的 416x（实为 640x）块级冗余** | a32 物化是**每 block 重算**：416 blocks（n=3328, warps=8）× k=5120 = 2.13M 次解码，而该 token 只有 5120 个唯一值 ⇒ **416× 冗余**。生产大 shape（n=5120, warps=8）为 **640 blocks ⇒ 640× 冗余**。这才是 1.55µs/call 的真身。 |
| **cvt 解码被否证** | 用 `cvt.rn.f16x2.e4m3x2` 替代 smem gather：转换管吞吐 **16 results/clk/SM**（Table 4 "All other type conversions"）< shared memory 的 **32 值/clk/SM**（LDS.32）；且每元素还需 `cvt.f32.f16` 展回 f32 ⇒ ≤10.7 元素/clk/SM，**严格更慢**；位一致也无法保持（scale 重结合改舍入序）。 |

**bf16-LUT 同族否证**：`s_lut` 由 256×f32 改 256×bf16 的提案**物理不成立**——
`*(bf162*)(lut + (b4&0xFF))` 取到 `LUT[c0]`/`LUT[c0+1]`，而 `o.y` 要的是 `LUT[c1]`（码是任意字节，无相邻关系）；
随机枚举实测 **49.8% 元素错**，直接破坏位一致。且 LDS 不减半、smem 512B 不跨驻留边界。**勿再提案。**

---

## 6. row-stationary 定案（a32 冗余的另一种修法）

**提案**：把 a32 解码改成 row-stationary——在寄存器里复用解码后的激活，消掉每 block 的重算（§5 的 416×/640× 冗余）。

| 维度 | 结论 |
|---|---|
| 可行性 | **可行**（数学上正确，布局不需重排） |
| 收益 | **< 1µs/call** —— 相比 swapAB 的 ~1.5µs/call 目标低一个量级 |
| 阻碍 1 | **寄存器 64 墙**：`gemm_fp8_gemv_kernel` 的 register cap（launch_bounds）只有 64；row-stationary 要把整行/多行解码结果常驻寄存器，直接撞墙 |
| 阻碍 2 | **epilogue 契约**：kernel 的融合 epilogue（B1 fp8 行发射 / rope / AR-v5 / xq 发射）要求特定的行布局与 thread 映射；row-stationary 的重排会破坏这些契约 |
| 判定 | **性价比低于 swapAB** —— 同为"消 a32 冗余"，swapAB 收益更高、且顺带删掉 LUT 解码；row-stationary 收益 <1µs 却要动 epilogue 契约，风险/收益比差 |

**结论：不做 row-stationary；a32 冗余的正确修法是 swapAB（§5）。**

---

## 7. 方法论铁律（7 条 + gate 卫生）

1. **隔离探针只用于淘汰，正向收益必须 serve A/B。**
   隔离探针无法模拟 serve 的三个条件：**SM 争抢（侧流并行）、L2 竞争、占用率敏感**。
   已确认 5 次隔离→生产失效：a32-vec4 → AR grid → PDEPTH pipeline → w2 warm → 隔离版 cp.async。
2. **fork_ev 是 kernel 级事件，不是 block 级。**
   `fork_ev` 在 EARLY kernel **完成时**记录，所以给 gating kernel 加任何工作 = 加到 main 的关键路径。
   小 kernel 合并的收益必须 **> gate 语义的代价**（省 launch 的收益 < 加到 gate kernel 的代价时是净负）。
3. **失败实验的代码立即物理删除，不留 gated-off。**
   gated-off 的死代码也可能影响 ABI 与代码布局；且"gate 可探测 ≠ gate 生效"。
   **gate 必须验证"OFF 时是否真的回退"。**
4. **FFI 边界（.cu ↔ Rust）是原子性单位，必须同一 commit。**
   `git add -A` 在 subagent 并行工作时是危险的——docs 提交会扫入进行中的实施。
   docs 提交用 `git add <specific-files>`。
5. **给热点 kernel 加运行时参数/分支 = 编译产物变重 = 回归风险（模板或独立 kernel）。**
   ⚠️ **修正**：v13/v14 的 +0.34ms 曾被归为此条，但 gate-hygiene-audit 证明真因是 **gate 错配**（§4.7）。
   此条作为**设计偏好**保留（数量少、编译期可消除的参数更安全），但**不再作为那 0.34ms 的解释**。
6. **小 kernel 合并、folding 的收益分析必须考虑 gate/fork 语义**（见 2）。三条 fold（quant/swiglu/stamp）全败。
7. **位一致是硬约束。** 本仓库所有优化必须通过 fingerprint / 四段文本 / 逐位一致性验证。
   bf16 LUT 的 **49.8% 元素错**不可接受（§5）。

**（附加，最决定性）gate 卫生**：**改 gate 默认值时，`FileReplace` 必须按函数名/上下文锚定，
改完读回确认。** 用 `int v = 2` 这类裸变量声明作锚点会静默改错同名对象——
这一个错误造成了 v13→v15 连续三轮、~0.37ms 的误诊与 499 行无谓清理。

---

## 8. 关键地板清单（当前架构不可逾越）

| 地板 | 值 | 依据 |
|---|---|---|
| **gemm a32 物化** | 1.55µs/call（640× 块级冗余的 LUT gather + consume） | consume 地板 4.45µs（LDS 延迟受限）+ launch 0.71µs ≈ 5.2µs；生产 mean ~7.5–8µs → 246 次 ≈ 1.85–2.0ms。**swapAB 可删掉此地板**（§5）。 |
| **expert gateup LUT-gather** | — | cp.async 无效已证 5 次；IPC 0.8/4，80% issue 槽停等（操作数供给受限，非 FMA 吞吐） |
| **expert down L1TEX** | — | 8 项理论优化全部失败（k-split/uint4/加 blockDim/降 rows…） |
| **AR NVLink 协议** | ~8µs/call（往返 ~6µs 是 stamp/poll） | 网格形状只影响 1–2µs；AR 总量 ~0.65ms |
| **CUDA graph node dispatch** | **0.411µs/node** | `kernels/cuda/graph_bench.cu` 实测；1800 节点 ≈ 0.74ms |

**已过时的旧地板**：roadmap 的 "5.3ms 真实工作地板" 写于 operand-supply 发现与侧流优化系列之前，不再成立。

---

## 9. 200 tok/s 判定与前行路径

| 量 | 值 |
|---|---|
| 会话最优（v16，已验证） | **6.26ms / 159.7 tok/s**（v9 为 6.23ms / 160.5） |
| 目标 | 5.00ms / 200 tok/s |
| Gap | **~1.26ms** |

**判定：在"逐 kernel 抠 SIMT 地板"的框架下不可达；但 swapAB 打开了新框架。**

- 全部**已识别的小优化路径**（节点削减 / gemv_bf16 / w2 warm / down cp.async / epilogue folding）
  若全部兑现 → ~6.05ms ≈ 165 tok/s，**仍达不到 200**。
- **唯一数量级路径 = swapAB**：gemv 2.33ms → ~0.3ms（删掉 LUT 解码 + 640× 冗余），
  步时预估 **~4.2ms ≈ 238 tok/s** ✓。实施中（新 kernel + launcher + Rust dispatch + parity）。
- 剩余研究级项：expert 侧 fp4 的 tcgen05 swapAB（需 `kind::f8f6f4` 混精，属另一条线）。

**结论一句话**：SIMT 抠法已到顶（~6.0ms / 166 tok/s）；**突破 200 靠 swapAB 换范式**，而非再抠现有 kernel。

---

## 附：本报告的口径与来源

- serve A/B：`scripts/dsv41_serve_ab.sh`，同二进制背靠背，判据 = 四段文本（Paris/Tokyo/1+1=/静夜思/出师表）
  逐字 + `faults=0` + p50。
- 相关文档：`crates/ferrite-dsv41/STATUS.md`（逐 commit 记录）、`docs/agent/dsv41-methodology.md`（方法论手册）、
  `docs/agent/perf-roadmap.md`（gemv/a32/swapAB 分析）、`docs/agent/dsv41-nsys-v14-plan.md`（nsys 分解）、
  `docs/agent/roadmap-200-tokps.md`（执行计划）。
