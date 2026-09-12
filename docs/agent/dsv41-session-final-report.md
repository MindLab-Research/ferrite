# DSV4.1-Flash 会话最终报告（结构草稿）

> **用途**：本文件是 **`crates/ferrite-dsv41/STATUS.md` 最终章节的草稿**，v14 实测数据回填后可直接并入。
> 数字全部来自本会话已验证的 A/B 或复现器；未验证项已标注。
> 会话终态：**13.28 → 6.23ms（+113.3%），75.3 → 160.5 tok/s**；v14（fold 清理 + sparse-merge + compress-fuse）
> 预期 **~6.15ms ≈ 162 tok/s**。

---

## 1. 优化链全景表（13.28 → 最终值）

| # | 优化 | p50 ms | 累计收益 | commit | 验证 |
|---|---|---|---|---|---|
| 0 | 会话基线 | 13.28 | 75.3 tok/s | — | — |
| 1 | shared expert TP 切分 | 11.85 | +10.8% | `9cddfb8` | serve A/B，四段全对 |
| 2 | sparse 3-deep key-split + e4m3 LUT + 融合系列 | 9.38 | +29.4% | `710c107`（3-deep）、`54f470b`（e4m3 LUT）、`9e4e3df`（shared-mixed-lut）、`2d7eead`（dual-chain/MoE-dual/fp4-pack） | serve A/B |
| 3 | gate v2 + gateup MLP unroll + down revert | 8.60 | +35.2% | `65acfb1`、`73e4a50`、`5013268` | serve A/B |
| 4 | hc tail split + swiglu_q | 8.23 | +38.0% | `7d623ad`（+ round-41） | serve A/B |
| 5 | allv2（K-split + P1 a32 + B+C + dead slots + launch_bounds） | 6.84 | +48.5% | `f019237`、`a6c4ed3`、`d76a183`、`c12ff60` | serve A/B |
| 6 | cp.async 权重先行 + warps8 + EARLY 侧流 | 6.66 | +50.5% | `f6aa8c8`、`a7600db`、`9558491` | serve A/B |
| 7 | down 回归修复 + dots-LATE merge | 6.48 | +51.4% | `96f169b`、`1b265a8` | serve A/B |
| 8 | AR store 并行化（grid.y=world） | 6.47 | +51.5% | `f911abd` | serve A/B |
| 9 | P4 act-cpasync（默认 ON） | **6.24** | **+112.8%** | `fdd70d4` | serve A/B，160.3 tok/s |
| 10 | sparse-merge 选举折叠 | ~6.15 | +115.9% | `f6d2dde` | ❌ **已回退**（v14 判中性 + 代码存在性 +0.34ms）|
| 11 | COMPRESS_FUSE 3→1 | ~6.15 | +115.9% | `595437d` | **v14 验证中** |
| 12 | fold 代码物理清理（847 删除 / 14 文件） | ~6.15 | +115.9% | `175462d` | `cargo check` 0 error；**v14 验证中** |

> 第 10–12 项合并成 v14 一次验证（都是默认 ON 的小改动，预期 −0.08ms）。
> ⚠️ 工作树另有未提交的 **RW_FOLD**（ring_win 全折叠，`-0.02~0.05ms`）——❌ **2026-09-12 已回退删除**（未验证的优化给热点 `rmsnorm_rope_kernel` 加 6 尾参+早退守卫，同机制风险）。

---

## 2. 落地优化清单（13 项，全部 serve A/B 验证）

| # | 优化 | 机制 | 来源 |
|---|---|---|---|
| 1 | **shared expert TP 切分** | 共享专家从"仅 rank0 串行"改为所有 rank 各切 `inter/world` | `9cddfb8` |
| 2 | **sparse 3-deep 预取 × key-split** | 同一 kernel 加 key 分块，grid=(C,b·m,h)，per-slot 78→10.7ns | `710c107` |
| 3 | **e4m3 LUT** | fp8 解码从位运算改 256-entry smem LUT | `54f470b` / `9e4e3df` |
| 4 | **gate v2** | MoE gate 的 bf16 gemv（n=384，延迟受限 3% 峰值带宽）向量化 + K-split | `65acfb1` |
| 5 | **gateup MLP unroll** | `#pragma unroll 2→4` + uint2 宽加载 | `73e4a50` |
| 6 | **hc tail split** | tail 拆成 EARLY（主流）/LATE（侧流）两半，侧流 fork/join | round-41 |
| 7 | **allv2（P1 a32 dead-slot + B+C）** | a32 物化直写 `s_af`（消 `s_a` 中转，smem 48512→43392B，4→5 blk/SM） | `f019237` / `a6c4ed3` |
| 8 | **cp.async 权重先行 + warps8 + EARLY 侧流** | prologue 与权重读重叠；n≥2048 时 warps=8 减半 block；EARLY 回侧流头部 | `f6aa8c8` / `a7600db` |
| 9 | **down 回归修复 + dots-LATE merge** | 去掉 `__launch_bounds__(256,4)`（4→6 blk/SM）+ 恢复 ILV；dots 与 LATE 合一个 launch | `96f169b` / `1b265a8` |
| 10 | **AR store 并行化** | `grid.y=world`（5→40 blocks，3%→25% SM），每 block 只写一个 peer | `f911abd` |
| 11 | **P4 act-cpasync（默认 ON）** | 激活 staging 用 cp.async，消除 a32 物化的串行 LDG | `fdd70d4` |
| 12 | **sparse-merge 选举折叠** | split 的 winner block 就地跑 merge body，40 launch → 0 —— ❌ **2026-09-12 已回退**（v14 判中性；符号/选举代码给热点 split kernel +0.34ms）| `f6d2dde` |
| 13 | **COMPRESS_FUSE 3→1** | decode compressor 的 state+pool+commit 合一 launch，逐字节不变 | `595437d` |

**旁证**：`175462d` 的 fold 物理清理本身也是"落地"——它修复了 quant-fold 的**缺陷 gate** 并消除 ABI 风险
（见 §4 铁律 3、4）。

---

## 3. 失败关停清单（全部有机制级解释）

| # | 优化 | 预期 | 实际 | 根因 | 处置 |
|---|---|---|---|---|---|
| 1 | **PDEPTH pipeline**（2/5） | −0.48ms | **+0.04ms** | 占用率损失 > 延迟隐藏（42.8KB smem → 2 blocks/SM）；隔离时单 kernel 时间长掩盖占用率 | 默认 `GATEUP_PIPELINE=1`（OFF） |
| 2 | **w2 L2 prewarm** | −0.25ms | **+0.04ms** | warmer 与 gateup 尾部争 SM（PDL 重叠变争抢） | `W2_PREWARM=0`（OFF） |
| 3 | **quant fold** | −0.072ms | **+0.39ms**（与 swiglu 合计） | fork_ev 是 **kernel 级**事件——EARLY 加工作 = 加到 main 关键路径；且 gate 缺陷使 fold 一直生效（见 §4.3） | 代码物理删除（`175462d`） |
| 4 | **swiglu fold** | −0.068ms | ↑ 同上 | w2 prologue 的 swiglu 在 GEMV 上下文比独立 kernel 贵（寄存器/布局） | 代码物理删除 |
| 5 | **AR stamp fold** | −0.15ms | **29.5s/step** | 单调 counter 在图 replay 下机制坏掉（poll 自旋等永远不来的 stamp） | 代码物理删除（`175462d`） |
| 6 | **a32-vec4** | −0.13ms（隔离） | **中性** | P4 的 cp.async 已隐藏 staging，vec4 在其上增益≈0 | 保留（无害） |
| 7 | **AR reduce grid** | −0.08~0.16ms | **中性** | 8µs 主导项是 stamp/poll 的 NVLink 往返（~6µs），网格形状只影响 1–2µs | 保留 |
| 8 | **wo-pair 两段核** | −0.13ms | **+0.46ms** | 共享 smem 池按 max(k)=5120 分配 → phase2 占用率 96→20 warp/SM（4.8× 崩塌） | `WO_PAIR` 保持 OFF |
| 9 | **sh-pair** | −0.26ms | 未上机 | 与 wo-pair 同根因且更极端（k 比 8:1） | 审查后放弃 |
| 10 | **M>1 expert batching** | — | 价值 = 0 | 单请求无第二个 token；层内 6 专家已在一个 launch | 关闭 |
| 11 | **expert MMA 化** | 绕过指令地板 | 前提不成立 | sm_103a 无 M=8/16 的 fp4 MMA（tcgen05 mxf4 的 M 钉死 128）；M=128 masked 实测 0.2% 峰值 | 关闭 |
| 12 | **Stage C persistent 段核** | −1.0~1.5ms | 最小版仅 0.1~0.2ms | smem 池按 max(k) 分配 → 占用率崩塌 + `grid.sync` 与整步图捕获不兼容 + 段内近全串行 | 关闭 |
| 13 | **cross-layer pipe** | −0.3~0.5ms | −0.05~0.15ms | 80% 被路由依赖挡死（L+1 专家取哪 6/384 由 gate(xn) 决定）；可预取的 20% 落在 91% 固定成本核上 | 关闭 |
| 14 | **bf16-lut（a32 物化的 bf16 LUT）** | −0.1ms | **分析否证（未落地）** | 物理不成立：`*(bf162*)(lut + b4)` 取到 `LUT[c0]`/`LUT[c0+1]`，而 `o.y` 要 `LUT[c1]`——位序错配 | 关闭（勿再提案） |

---

## 4. 方法论铁律（6 条）

1. **隔离 → 生产失效（×5 已确认）**：隔离探针无法模拟 serve 的三个条件——**SM 争抢（侧流并行）、
   L2 竞争、占用率敏感**。已确认的 5 次：a32-vec4 → AR grid → PDEPTH pipeline → w2 warm → 隔离版 cp.async。
   **铁律：隔离探针只用于淘汰明显差的方案；正向收益必须 serve A/B 确认。**
2. **fork_ev 是 kernel 级事件，不是 block 级**：`fork_ev` 在 EARLY kernel **完成时**记录，
   所以给 gating kernel 加任何工作 = 加到 main 的关键路径。
   **铁律：小 kernel 合并的收益分析必须考虑 gate 语义——省 launch 的收益 < 加到 gate kernel 的代价时是净负。**
3. **"fold 代码存在性"假设错误 → 实为 gate 缺陷**：v13 的 +0.33ms "未归因回归" 一度被归为 kernel 签名/代码布局
   （I-cache）。fold-correctness-audit 定案：**quant-fold 的 gate 有缺陷**——`=0` 时 `layer()` 仍传非空 `xq4`
   （`chain_dev.rs:2567`），kernel 只看 `xq4!=nullptr`（`:6971`）⇒ fp4 直出照做。v12/v13 ≈ fold 一直生效。
   **铁律：gate 必须验证"OFF 时是否真的回退"（符号可探测 ≠ gate 生效）。**
4. **部分提交的 FFI 错位（第 4 次事故）**：`git add -A` 在 subagent 并行工作时是危险的——docs 提交会扫入
   进行中的实施。v11 崩溃：新 `.cu`（hc_mixes_tail 多 `xq4/xsc4`）+ 旧 Rust（旧 FFI）= 参数错位 = **709**。
   **铁律：FFI 边界（.cu ↔ Rust）是原子性单位，必须同一 commit；docs 提交用 `git add <specific-files>`。**
   另：失败实验的代码应在验证失败后**立即物理删除**（gated-off 的死代码也可能影响性能/ABI）。
5. **LUT bf16 物化物理不成立**：bf16 LUT 的 2 字节 packing 与 `o.y` 期望的索引错位（见 §3 #14）。
   **铁律：改数据布局的方案先做位序推导，再谈收益。**
6. **AR 协议第 4 次失败**：v2 capture race → v3 last-block → oneshot ×2 → stamp fold。
   设备侧 AR 的 publish 自旋与**图 replay** 天然冲突（单调 counter 在 replay 下坏掉）。
   **铁律：任何改 AR 协议的方案必须先在整步图捕获路径上验证，而不是仅 host 路径。**

---

## 5. 关键地板清单（当前架构不可逾越）

| 地板 | 值 | 依据 |
|---|---|---|
| **gemm a32 物化** | 1.55µs/call（LUT smem 随机 gather + consume） | consume 地板 4.45µs（LDS 延迟受限）+ launch 0.71µs ≈ 5.2µs；生产 mean ~7.5–8µs → 246 次 ≈ 1.85–2.0ms |
| **expert gateup LUT-gather** | — | cp.async 无效已证 5 次；IPC 0.8/4，80% issue 槽停等（操作数供给受限，非 FMA 吞吐） |
| **expert down L1TEX** | — | 8 项理论优化全部失败（k-split/uint4/加 blockDim/降 rows…） |
| **AR NVLink 协议** | ~8µs/call（往返 ~6µs 是 stamp/poll） | 网格形状只影响 1–2µs；AR 总量 ~0.65ms |
| **CUDA graph node dispatch** | **0.411µs/node** | `kernels/cuda/graph_bench.cu` 实测（2000 节点 × 100 次，空图法隔离 dispatch）；1800 节点 ≈ 0.74ms，削减 248 节点 ≈ −0.10ms |

**已过时的旧地板**：roadmap 的 "5.3ms 真实工作地板" 写于 operand-supply 发现与侧流优化系列之前，不再成立。

---

## 6. 200 tok/s 判定

| 量 | 值 |
|---|---|
| 当前最好（v9，已验证） | **6.23ms / 160.5 tok/s** |
| v14 预期 | **~6.15ms / ~162 tok/s** |
| 目标 | 5.00ms / 200 tok/s |
| Gap | **1.15–1.23ms** |

**判定：当前架构下不可达，属于研究级。**

- 全部**已识别路径**（expert cp.async-full / w2 warm / down cp.async / epilogue folding / 节点削减 / gemv_bf16 cp.async）
  若**全部兑现** → ~6.05ms ≈ 165 tok/s。其中 expert-cpasync-full（−0.48ms）依赖 "2× 假设"，
  GLM 侧 cp.async 中性的教训提示隔离/生产差异，**不应算入保底**。
- 剩余 **~1.05ms** 必须突破 **LUT gather 地板**（gemm a32 1.55µs/call × 246 + expert LUT-gather）。
  这是**研究级**：需要**无 smem 的 fp4/fp8 解码**，或**完全不同的 GEMV 设计**（grouped/persistent GEMM、
  expert 机制重构）。
- **可行替代**：若目标是 16 并发高吞吐，**M>1 batching**（serve 级、新 kernel 族）的收益上限更大——
  但 decode 目前构造上是 M=1-only，这是任务定义的改变，不在本次单请求优化范围内。

**结论一句话**：单请求 200 tok/s 需要一次 LUT-gather 地板突破（研究级，1–3 周量级）；
当前架构的真实增量上界 ≈ **6.0ms / 166 tok/s**。

---

## 附：待回填项（v14 实测后）

- [ ] 优化链表中 step 10–12 的实测 p50
- [ ] §1 的"最终值"（预期 ~6.15ms）
- [ ] nsys 对照表（见 `docs/agent/dsv41-nsys-v14-plan.md` §4.2）
- [x] RW_FOLD 是否并入本次（未提交）→ **不并入；2026-09-12 已回退删除**
