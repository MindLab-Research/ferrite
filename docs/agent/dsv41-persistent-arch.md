# DSV4.1 persistent / mega-kernel 架构 — Stage C 设计

**一句话**：把每层的 ~17.5 次 launch 收敛成 **3 个段级 persistent 核**（段 A/B/C，块内顺序走完段内阶段链、中间量留 smem），AR 折进段边界；图节点从 ~700 降到 **40×3 = 120**，残差从 ~1.15ms 降到 ~0.2ms。

**前置事实（同二进制实测，2026-09-11）**：当前 10.16ms / 98.4 tok/s；融合路径已验证 9.44ms / 105.9 tok/s（数值 bug 修复中）；gap-analysis：`gemm_fp8_gemv` 2.18ms（91% 是固定项）、残差 1.15ms（~700 节点 × ~1.5µs ramp-down）。

**⚠️ 事实纠正**：DSV4.1 = **40 层**（`crates/ferrite-models/configs/dsv41_flash.json` 的 `num_hidden_layers=40`；45 层是 GLM-5.3 的层数，见 `ferrite-unified-arch.md:45`）。故段核总数 = 40×3 = **120**，不是 135。

---

## 0. "persistent" 的粒度（问题 1）

| 粒度 | 形态 | 判定 |
|---|---|---|
| L0 整模型一核 | 一个 grid-resident kernel 吃下整个 decode step | ✗ **不可能**：每层 2 处 AR 是**跨 rank 物理边界**（`chain_dev.rs:794` `moe_reduce`、`chain_dev.rs:1961` attn 侧），跨 rank 集合通信进不了核内（hc-merge 已证：单核 + ticket 自旋 = **+3.2ms**）；且权重必须流自 HBM |
| **L1 段级 persistent（选定）** | 一个 kernel 跑一层的一段，块内顺序走完段内阶段链 | ✓ 段边界正好落在 AR；段内中间量留 smem；每段 ~65-85µs（254µs/层 ÷ 3） |
| L2 层内 cooperative | 段内核用 `grid.sync()` 消除相位栅栏 | 可选增强（需 cooperative launch，与图捕获/PDL 兼容性待验） |

**结论**：persistent = **"segment-persistent"**，不是 model-persistent。严格"每层一核"不存在（AR 是硬边界）⇒ 可行的 persistent 单元 = **每段一核**。

## 1. 每层 3 段的段内核（问题 2）

段划分承 `dsv41-layer-fusion.md §1`，与代码调用序一致：

```
段A: hc_mixes → hc_collapse → rmsnorm → attention(wq_a/q_norm/wq_b/rope/wkv/kv_norm/rope/
       ring_append/window_idxs/compress/indexer/sparse_attn/rope⁻¹/quant1/wo_a/wo_b)
  ── AR#1 ──
段B: hc_post → copy_h_back → hc_mixes → hc_collapse → rmsnorm →
       MoE(gate/route_topk/quant_fp4/gate_up/swiglu/down/reduce + 共享专家)
  ── AR#2 ──
段C: hc_post → copy_h_back
```

**段内核 = 相位机（phase machine）**：block 内顺序执行阶段，相位间 `__syncthreads()` + smem 交接。阶段的工作映射随算子族变：

| 阶段族 | 映射 | 依据 |
|---|---|---|
| hc 族（mixes/collapse/post） | 沿 hc_dim 分 tile，**T=64 → 320 blocks** | smem 给上界、并行度给下界，取 T=64（`dsv41-layer-fusion.md §3`：148 SM × 2 波） |
| GEMM/GEMV 族 | 沿输出行 grid-stride | M=1 退化为 GEMV，复用 `gemv_fp8`/`gemv_bf16` 的 warp-per-row 范式（`dsv41_glue.cu:329`） |
| sparse_attn | 沿 heads（nlh=8） | 复用 `sparse_attn_warp` 的 3 深预取 |
| MoE | 沿 inter_local / 专家 slot | batched gate_up/down + **升序 slot** reduce（数值契约） |

**smem 布局（段 A）**：激活行 `[dim]f32` = 20KB（或 fp8 5KB）；块内中间量按 tile：T 列 × 4B × ~4 活跃缓冲 ≈ 16T B；hc 的 `wpart[32]` 归约；合计 ≪ 227KB/block（opt-in 上限见 `dsv41_kernels.cu:1991`）。
**跨 block 点**（必须复用既有模式，不另创）：hc 的 ss/comb 归约（`dsv41_kernels.cu:1002` 的 warp shuffle → `wpart` → 跨 warp）；需全局归约处用"设备全局 partials + is_last 选举"（`dsv41_kernels.cu:2709`）。⚠️ **split 必须 = 1**——split=8 会改变部分和顺序 ⇒ 非逐位等价。

## 2. AR 与段边界（问题 3）

AR v5 = `store`（纯逐元素拷 partial → peer staging，`ferrite_kernels.cu:8117`）+ `pubred`（stamp + 自旋 + 归约，`ferrite_kernels.cu:8140`）**两个 launch**。每层 2 AR ⇒ 现 4 节点/层、160 节点/步。融合分两级：

- **安全级（保留 AR 节点）**：`store` 折进段的**最后写者** epilogue（段 A = wo_b 的 gemm；段 B = `add_inplace`/`moe_down_reduce`）；`pubred` 仍独立 launch，其自旋窗口用 **PDL**（`ferrite_kernels.cu:748` 的 `pdl_or_plain`，`FERRITE_PDL=1`）与下一段重叠。节点 = 120 + 80。
- **激进级（零 AR 节点）**：store 折进段 X 末写者并在**同一 epilogue stamp**；pubred 折进段 X+1 的**段首**（全块 head poll → reduce → 推进 epoch）。节点 = 120。⚠️ 这是 v5 协议改动（stamp 从 pubred 移到 store），列 P5。

**hc-merge 铁律复用**：绝不把跨 rank 同步塞进单核。pubred 的自旋必须"**全 block 等同一绝对 epoch**"（`ferrite_kernels.cu:8150`）——若某尾块单独长自旋 ⇒ 扣 SM 当人质 ⇒ +3.2ms 回归。段内藏自旋只允许**段首全块栅栏**形态，不允许尾块独等。

## 3. 激活常驻（问题 4）

| 范围 | 介质 | 依据 |
|---|---|---|
| 段内中间量 | **smem** | dim=5120 f32 = 20KB；4 块 80KB，可行 |
| 跨段激活 | **global（L2 常驻）** | 段输出即下段输入（`s.h`/`s.xn`/`s.o`，各 20-80KB），L2 60MB 容得下 |
| 权重 | HBM（不变） | DRAM 地板，persistent 不改 |

⚠️ 勘误：20KB 激活**本来就在 L2**（`STATUS.md:2085`）⇒ "落 global 再读"的成本不是 DRAM 带宽，而是**节点过渡的固定成本**（每 GEMM ~11.6µs × 91%）。persistent 消的是这个，不是带宽。

## 4. CUDA graph 兼容（问题 5）

- 段内核仍是**普通 launch**（每段一节点）⇒ 图可捕获，节点 700 → 120（+ AR）。
- 可捕获性硬约束：核内**禁止** `cudaMalloc`/`cudaStreamSynchronize`/host 交互；所有分配在预热期完成（`dsv41_kernels.cu` 的 per-device 预热 scratch 是既有范式）。`step_body` 默认在图捕获区内（`chain_dev.rs:830` `step_impl`）。
- 残差账：700 × 1.5µs = 1.05ms → 120 × 1.5µs = 0.18ms ⇒ **−0.87ms**。
- PDL 增量在**段尾 ramp-down**（图内启动已 ~0.2-0.3µs/节点）。

## 5. 数值一致性（问题 6）

hc-merge 教训的推广：**融合不是拼装，是精确的相位重排**。三条硬性律：

1. **同编译单元**：融合核与基线核若分处 `dsv41_kernels.cu` / `ferrite_kernels.cu`，两边独立决定 FMA 收缩 ⇒ **1 ULP** ⇒ 确定性文本变化（`hc_post_parity.rs` 实测 2.98e-8 = 2^-25）；要么同 CU，要么显式 `__fmaf_rn` 固定。
2. **归约顺序不变**：K 循环逐指令照抄；down 的升序 slot 累加；AR 的升序 rank 求和（v5 已保证 1-ulp 一致）；hc 的 split=1。
3. **逐位 parity 门禁**：每个融合核配一个 `*_parity.rs`（同设备、确定性输入、逐位比），先于 serve A/B。

---

## 6. 分阶段实施计划

| 阶段 | 内容 | 预期收益 | 风险 |
|---|---|---|---|
| **P0（前置）** | 统一融合 env 两侧默认（`.cu` `g_fuse` return 0 + Rust `unwrap_or(false)`）；`DSV41_AR_STORE_FUSE` 逐位验后翻 ON | −0.08ms + 铺路 | 低 |
| **P1** | 段 C 融合：`hc_post` + `copy_h_back` → `hc_post_inplace`（省 80KB D2D ×2/层） | −0.15~0.25ms | 低（parity 既有） |
| **P1b（已实施，env 默认关）** | `hc_post_inplace` 再折进**产生它 `x` 的那个 AR** 的 pubred epilogue（`DSV41_HCPOST_EPI=1`，核 `ferrite_p2p_ar_v5_hcpost`）：2 个 site/层 = 80~90 节点。这是"相邻两核合一"的第一步，也是段核的第一个可运行原型 | −0.15ms | 中（跨 CU 位级：epilogue 用显式 `__fmul_rn`/`__fmaf_rn`，须过 `ar_hcpost_parity.rs` + 同二进制 token 逐字 A/B） |
| **P1c（已实施，env 默认关）** | **段核「相位机」机制原型**：`hc_pre` 的 dots+tail+collapse 合成**单块相位机**（核 `hc_pre_persist_kernel`，grid=(rows,)、block=1024、`DSV41_HC_PERSIST=1`，导出 `dsv41_hc_front_persist`）。无 ticket、无自旋——块内 4 个 phase 各一个 `__syncthreads`；点的 lane 分配/归约分组逐句照抄 ⇒ **逐位等价**（`hc_persist_parity.rs` 门禁）。**这是"段核 = 相位机"的第一个可运行证据，机制可复用** | 待测（**预期非收益**，见右） | ⚠️ **smem 放得下，并行度放不下**：两 launch 版把 x+1 个权重行(160KB)放进 24 个 block ⇒ 24-SM 并行（531GB/s，7.4µs）；单块装不下 24 个权重行(1.9MiB) ⇒ 24 行点积挤在 **1 个 SM**、权重走 global。**合并省 1 次 launch，但牺牲点积的块并行度 ⇒ 大概率持平或更慢，纯属机制+parity 原型**。真正的段核须用 §1 的「沿 hc_dim 分 tile，T=64 → 320 blocks」多块形态，不是 grid=(1,) |
| **P2** | 段 A 融合：hc 链 + attention 投影链一核，中间量留 smem；−6~8 launch/层 | −0.6~1.0ms | 中（hc 归约 / split=1） |
| **P3** | 段 B 融合：MoE cooperative（gate→route→quant→gate_up→swiglu→down→reduce 一核，中间量留 smem） | −0.4~0.7ms | 中（slot 定序） |
| **P4** | 跨层流水：ffn mixes 挪到 L+1 的 attn 段内下发，hc 的 ⟨B⟩ 半藏进 AR poll 窗口（PDL） | −0.3~0.5ms | 中 |
| **P5** | 零 AR 节点：store+stamp 折进段末、pubred 折进下段首（v5 协议改动） | −0.15ms + 40 节点 | **高** |
| **P6（Stage D，开放）** | model-persistent：权重分片驻留 smem + cooperative grid sync | 不确定 | 极高，不承诺 |

每阶段独立 env gate、默认 OFF；验收 = 同二进制背靠背 A/B + 人眼四段文本 + parity 测试。

## 7. 关键风险与回退

1. **跨 rank 同步进核（hc-merge 复现）**——最高风险。回退：AR 永远保留独立 launch / PDL 重叠；任何"省一个 kernel"的方案先证明不产生尾块长自旋。
2. **数值 1 ULP**——FMA 收缩 / 归约顺序。回退：每融合一个 env gate（默认 OFF），parity 不过不翻。
3. **图捕获失败**（分配/host 交互漏进捕获区）——回退：`DSV41_GRAPH_STEP=0` 逐 kernel 路径仍在。
4. **段内核寄存器/占用崩**（多阶段塞一核 ⇒ reg 压力）——回退：段再切细（A→A1/A2），仍远少于 17.5 launch/层。
5. **收益递减**：persistent 只回收 ~2.5-3ms 开销；**5ms 需另攻"真实工作"**（见 §9）。

## 8. 与统一架构的关系

persistent 核是 `ferrite-kernel` 的「**模型描述 → 图节点**」编译器的**输出形态**（`ferrite-unified-arch.md §2`）：`LayerDesc{kernels: &[KernelId], weights, shard}` 在**构造期**决定每层编译成 3 个段核还是 N 个独立核；引擎 = 图构建器，段 = 图节点，AR = 节点边界。即 persistent **不是新的运行时**，而是同一编译器的**更高融合档位**——DSV4.1 的 40×3 段与 GLM 的 45 层 linear/DSA 混合走同一条 `layer_descs → 构造期分配 → 捕获` 路径（`ferrite-unified-arch.md §3` Phase 3 是它的前置）。

## 9. 回收预算与诚实判定

**可回收（persistent 目标）**：残差 1.15（→0.2）+ `gemm_fp8_gemv` 固定项 ≈1.98（段融合消其 launch+往返，部分回收）+ MoE 往返 + 跨层流水 ≈ **−2.5~3.0ms** ⇒ 落点 **~7.2-7.7ms（130-139 tok/s）**。

**不可回收（真实工作地板）**：`expert_gemv_fp4` ~1.78（L1TEX 管道地板，内层杠杆全阴性）+ hc 链 ~1.59 + `gemv_bf16` ~0.74 + AR ~0.66 + sparse ~0.32 ≈ **5.3ms**。

⇒ **persistent 单独到不了 5ms**（5.3 > 5）。5ms 需 persistent **叠加**对专家核机制（指令地板）与 hc 链的再攻；而 **16 并发 1600 tok/s 的原目标，batching 的杠杆更大**（延迟受限已证：8→16 seq 只 +7%，见 `ferrite-unified-arch.md §4`）。

---
_事实来源：`chain_dev.rs:794/1013/1314/1961`；`ferrite_kernels.cu:748/590/8117/8140`（AR 与 PDL）；`dsv41_kernels.cu:1002/1991/2709`（hc 归约 / smem opt-in）；`configs/dsv41_flash.json`（40 层）；`STATUS.md`（gap-analysis、第 14/18-21 轮）；`dsv41-layer-fusion.md`（段设计、T=64）。_
