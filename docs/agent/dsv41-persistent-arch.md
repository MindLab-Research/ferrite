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
| hc 族（mixes/collapse/post） | 逐元素算子沿 hc_dim 分 tile，**T=64 → 320 blocks**；**归约型 dots 只能沿 K(=hc_dim) 切 chunk** | smem 给上界、并行度给下界，取 T=64（`dsv41-layer-fusion.md §3`：148 SM × 2 波）。⚠️ **勘误（P1d 实测推导）**：tile 只对**逐元素**输出有效（collapse/post 的每个输出互相独立，切 tile 不改任何单个输出的归约顺序 ⇒ 逐位保持）；**dots 是 24 个标量、每个都是整条 hc_dim 的归约**，切 tile 只会把部分和打碎成 320×24 份，真正需要的是 **K-split**（每块做 K/split 的部分点积 + ck 升序合并）。24 个输出 ⇒ M 方向已用满（24 块），**K 是唯一剩下的轴** |
| GEMM/GEMV 族 | 沿输出行 grid-stride | M=1 退化为 GEMV，复用 `gemv_fp8`/`gemv_bf16` 的 warp-per-row 范式（`dsv41_glue.cu:329`） |
| sparse_attn | 沿 heads（nlh=8） | 复用 `sparse_attn_warp` 的 3 深预取 |
| MoE | 沿 inter_local / 专家 slot | batched gate_up/down + **升序 slot** reduce（数值契约） |

**smem 布局（段 A）**：激活行 `[dim]f32` = 20KB（或 fp8 5KB）；块内中间量按 tile：T 列 × 4B × ~4 活跃缓冲 ≈ 16T B；hc 的 `wpart[32]` 归约；合计 ≪ 227KB/block（opt-in 上限见 `dsv41_kernels.cu:1991`）。
**跨 block 点**（必须复用既有模式，不另创）：hc 的 ss/comb 归约（`dsv41_kernels.cu:1002` 的 warp shuffle → `wpart` → 跨 warp）；需全局归约处用"设备全局 partials + is_last 选举"（`dsv41_kernels.cu:2709`）。⚠️ **split 必须 = 1**——split=8 会改变部分和顺序 ⇒ 非逐位等价。

## 2. AR 与段边界（问题 3）

AR v5 = `store`（纯逐元素拷 partial → peer staging，`ferrite_kernels.cu:8317`）+ `pubred`（stamp + 自旋 + 归约，`ferrite_kernels.cu:8340`）**两个 launch**。每层 2 AR ⇒ 现 4 节点/层、160 节点/步。融合分两级：

- **安全级（保留 AR 节点）**：`store` 折进段的**最后写者** epilogue（段 A = wo_b 的 gemm；段 B = `add_inplace`/`moe_down_reduce`）；`pubred` 仍独立 launch，其自旋窗口用 **PDL**（`ferrite_kernels.cu:748` 的 `pdl_or_plain`，`FERRITE_PDL=1`）与下一段重叠。节点 = 120 + 80。
- **激进级（零 AR 节点）**：store 折进段 X 末写者并在**同一 epilogue stamp**；pubred 折进段 X+1 的**段首**（全块 head poll → reduce → 推进 epoch）。节点 = 120。⚠️ 这是 v5 协议改动（stamp 从 pubred 移到 store），列 P5。

**hc-merge 铁律复用**：绝不把跨 rank 同步塞进单核。pubred 的自旋必须"**全 block 等同一绝对 epoch**"（`ferrite_kernels.cu:8370`）——若某尾块单独长自旋 ⇒ 扣 SM 当人质 ⇒ +3.2ms 回归。段内藏自旋只允许**段首全块栅栏**形态，不允许尾块独等。
**stamp 的并行化边界（2026-09-11）**：stamp 已由 thread 0 串行 → `thread r 写 peer r`（8 个不相交远端地址，跨 slot 顺序无关，因为每个 peer 只盯自己那一格），但 `__threadfence_system()` + `*epoch = e+1` 仍必须在 **block 0 的 `__syncthreads()` 之后**才有 thread 0 发出——顺序保证靠"barrier join 所有 stamp"，不靠单线程串行。任何把 stamp 折进 store epilogue（P5 激进级）的改造必须保留这一"先齐备、再 fence、再推进 epoch"的形状。

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
| **P1b（已实施，env 默认 ON，2026-09-11 起）** | `hc_post_inplace` 再折进**产生它 `x` 的那个 AR** 的 pubred epilogue（`DSV41_HCPOST_EPI`，核 `ferrite_p2p_ar_v5_hcpost`）：2 个 site/层 = 80~90 节点。这是"相邻两核合一"的第一步，也是段核的第一个可运行原型。**2026-09-11 互斥解除**：原先与 tail split 互斥（split 分支要求 `!hcpost_epi()`），根因是 fold 在 AR 内消费 `post`/`comb`，早于 `layer` 里那个 join 的位置；现在 `ar_hc_post_fold` 自己在 fused AR 之前 wait `join_ev`（`chain_dev.rs:1383-1391`），两者可共存 | −0.15ms（第 31 轮实测「中性」，见 `STATUS.md`） | 中（跨 CU 位级：epilogue 用显式 `__fmul_rn`/`__fmaf_rn`，须过 `ar_hcpost_parity.rs` + 同二进制 token 逐字 A/B；`DSV41_HCPOST_EPI=0` 为 A/B 臂） |
| **P1c（已实施，env 默认关）** | **段核「相位机」机制原型**：`hc_pre` 的 dots+tail+collapse 合成**单块相位机**（核 `hc_pre_persist_kernel`，grid=(rows,)、block=1024、`DSV41_HC_PERSIST=1`，导出 `dsv41_hc_front_persist`）。无 ticket、无自旋——块内 4 个 phase 各一个 `__syncthreads`；点的 lane 分配/归约分组逐句照抄 ⇒ **逐位等价**（`hc_persist_parity.rs` 门禁）。**这是"段核 = 相位机"的第一个可运行证据，机制可复用** | 待测（**预期非收益**，见右） | ⚠️ **smem 放得下，并行度放不下**：两 launch 版把 x+1 个权重行(160KB)放进 24 个 block ⇒ 24-SM 并行（531GB/s，7.4µs）；单块装不下 24 个权重行(1.9MiB) ⇒ 24 行点积挤在 **1 个 SM**、权重走 global。**合并省 1 次 launch，但牺牲点积的块并行度 ⇒ 大概率持平或更慢，纯属机制+parity 原型**。真正的段核须用 §1 的「沿 hc_dim 分 tile，T=64 → 320 blocks」多块形态，不是 grid=(1,) |
| **P1d（已实施，env 默认关）** | **段核多块形态**：`hc_pre` 仍为**一次 launch**，但 dots 沿 **K(=hc_dim) 切成 `split` 块**（核 `hc_pre_persist_mb_kernel`，grid=`(mix*split+1, rows)`、block=1024、`DSV41_HC_PERSIST_MB=1`，导出 `dsv41_hc_front_persist_mb`）。角色三合一：`bid<mix*split` 为一个 (投影行, K chunk) 做部分点积写 `g_hc_part[r][m][ck]`；`bid==mix*split` 是**并行** collapse 块（不依赖 dots）；**最后一个 publish 的 dot 块**被选举跑 tail（`atomicAdd` 计数，**无 ticket、无自旋**）。`split=8` ⇒ 192 个 dot 块。⚠️ **`split>1` 非逐位**（K 部分和按 ck 升序合并 ≠ 单 warp 树和），**`split=1` 逐位等价**于两 launch ⇒ 作为 parity 目标 | **实测第 37 轮：+3.3ms/步 = +41µs/次 × 80（11.89 vs 8.61ms，四段文本全对、0 fault）⇒ 回归，默认 OFF** | ⚠️「回归」实测已排除的假设（explore 复核，见下「P1d 回归归因」）|

### P1d 回归归因（explore 复核 2026-09-11，纯代码分析）

- ❌ **假设 3（计数器跨 replay 不重置）：证伪**。若 `g_hc_mb_done` 累积，`prev == ndot-1` 永不可达 ⇒ 无块被选举 ⇒ tail 从不执行 ⇒ `pre/post/comb` 保持 .bss 零值 ⇒ 文本必为乱码。实测四段逐字正确 ⇒ 复位确实生效。同一自复位纪律在生产里的 `gv2_route_epilogue`（`ferrite_kernels.cu:2749-2798`）已记载 graph-safe。
- ❌ **假设 1/2（fence / atomicAdd 争用）：量级不够且与规模无关**。每次 launch 只有 193 次 `__threadfence` + 192 次**同地址** `atomicAdd`；fence 是 per-thread、跨 148 SM 并行，单地址 atomic 的 L2 RMW 吞吐 ~ns 级 ⇒ 临界路径增量 ≤1~2µs，与 41µs 差 20 倍。旁证：`hc_front_kernel` 只有 **24** 个 dot 块（24 fence/atomic）却同样 +3.2ms —— 若 fence/atomic 是主因，193 块版应坏 ~8 倍。
- ✅ **最可能的结构性根因（本核内部）**：**tail 的串行 L2 依赖链被拉长 + 只能排在整格 drain 之后**。① hcpm 的 tail 无条件走 `ss_in == 0` 分支（`dsv41_kernels.cu:4333`），把两 launch 版默认 ON 的「ss 由 dots 的分块结果给出」优化（`g_hc_ss`，`hc_mixes_tail_kernel:3461-3469`）丢了 ⇒ tail 必须**整行重读 20480 float**；② `mixes` 循环按 `ck` 读 8 个 `g_hc_part` 槽（`split` 是运行期参数、循环不可展开）⇒ 又是 8× 的 L2 往返；③ tail 由「最后一个 publish 的块」执行，即必然在**整格 drain 期间**跑这条 ~6.5µs 的纯串行链（`STATUS.md:5210`：tail 临界路径 = warp0 的串行链），而两 launch 版是把 tail 作为独立 launch 丢在**已排空**的 GPU 上。
- ⚠️ **次因（需实测确认）**：193 块 × 1024 线程 + 三合一（dot/collapse/tail）的合并帧 ⇒ 寄存器数决定 blocks/SM；**smem 不是限制**（20KB×2=41.6KB ≪ 228KB），**regs 才是**（≤32 才 2 blocks/SM）。若 >32 ⇒ 1 blocks/SM ⇒ 193 块/148 SM = **1.3 波**，即 `down-vec-320` 已记录过的「regs → blocks/SM → wave」回归模式（`STATUS.md:5875-5901`）。复核手段：`cudaOccupancyMaxActiveBlocksPerMultiprocessor`（项目已有 dv320 工具链）|
- ⚠️ **另一个必须先排除的「假回归」通道**：`hc_mixes_auto`（`chain_dev.rs:1565-1641`）的回退链是 `if mb { mb() } else if persist { persist() } else { hc_front() }`——**`mb` 返回 Ok(false) 时不会重试 `hc_front`**，而是直接掉到 legacy `hc_mixes` + 调用方另跑一次 collapse_norm（≈4 launch/次）。而 `hc_front_persist_mb` 把 launcher 侧任何失败都以 `(int)e` 返回，`cudaErrorInvalidValue == 1` 正好被 Rust 当「kernel 拒绝」吞掉（`device.rs:2632`）⇒ **臂可能量到的是 legacy 路径，而不是 mb 核**。定位前必须先用 nsys 确认 hcpm 臂里出现的核名/时长。|
- ⚠️ **潜伏越界（`hc>4` 时）**：`__shared__ float wpart[32]`（:4225 / 3458）而 `nwarp = ss_stride>>5 = mix`；launcher 只校验 `mix ≤ 64`，**未校验 `mix ≤ 32`**。`hc=5` ⇒ `mix=35` ⇒ `wpart[32..34]` 越界写（静默污染相邻 smem）。当前 `hc=4`（mix=24）不触发。建议在 `hc_front` / `hc_front_persist_mb` 两处都加 `if (mix > 32) return InvalidValue;`|
- 🚫 **重设计约束**：`cudaLaunchCooperativeKernel` 与图捕获不兼容（`STATUS.md:4114`）⇒ **段内跨块 grid 同步在图里不可用**；「一次 launch 装一整段」必须靠**无同步的角色分解**或 PDL，不能靠选举式自同步。|
| **P2** | 段 A 融合：hc 链 + attention 投影链一核，中间量留 smem；−6~8 launch/层 | −0.6~1.0ms | 中（hc 归约 / split=1） |
| **P3** | 段 B 融合：MoE cooperative（gate→route→quant→gate_up→swiglu→down→reduce 一核，中间量留 smem） | −0.4~0.7ms | 中（slot 定序） |
| **P4** | 跨层流水：ffn mixes 挪到 L+1 的 attn 段内下发，hc 的 ⟨B⟩ 半藏进 AR poll 窗口（PDL） | −0.3~0.5ms | 中 |
| **P5** | 零 AR 节点：store+stamp 折进段末、pubred 折进下段首（v5 协议改动） | −0.15ms + 40 节点 | **高** |
| **P6（Stage D，开放）** | model-persistent：权重分片驻留 smem + cooperative grid sync | 不确定 | 极高，不承诺 |

每阶段独立 env gate、默认 OFF；验收 = 同二进制背靠背 A/B + 人眼四段文本 + parity 测试。

### P1e 候选（hc_pre dots 融进前驱 hc_post）：可行性复核（explore 2026-09-11，纯代码分析）

**命题**：dots 的输入 `x` 就是前驱 `hc_post` 刚写出的残差 `s.h`（`hc_mixes_auto` 传 `x = s.h`、`hc_dim = hc*dim = 20480`；`hc_post_inplace` 原位写 `res[i*h + j]`，域完全一致）⇒ 二者是同一条 Producer/Consumer 链，dots 是 hc_post 的自然延伸。**这一点成立**。但**载体（pubred AR epilogue）选错了**，三条硬阻塞：

1. **`p2p_ar_pubred_v5_hcpost` 现在在默认路径上**（2026-09-11 起）：`DSV41_HCPOST_EPI` **默认 ON**（`chain_dev.rs:2033`）。它此前与**默认 ON** 的 tail split（第 41 轮实测 −0.20ms）**互斥**——`hc_mixes_auto` 的 split 分支要求 `!Self::hcpost_epi()`；该互斥已解除：根因是 fold 在 AR 内消费 `post`/`comb`（早于 `layer` 中位于 AR **之后**的 join），修复方式是让 `ar_hc_post_fold` 在 fused AR 之前自行 `hc_tail_join()`（`chain_dev.rs:1383-1391`），即把 join 挪到「LATE 输出的第一个消费者」之前。默认配置现在同时跑 split 与 fold。
2. **跨 TU 设备符号**：`g_hc_part` 是 `dsv41_kernels.cu:3632` 的 `__device__` 全局，而 `p2p_ar_pubred_v5_hcpost_kernel` 在 `ferrite_kernels.cu:8523`；`build.sh` **无 `-rdc=true`** ⇒ 跨 TU 设备符号不可见（hc_post 数学当初正为此在 `ferrite_kernels.cu:8467` **逐句复制**而非共享）。要在 pubred 里写 `g_hc_part`，只能额外把 partial 缓冲当 kernel 参数从 Rust 传指针进来，并让读端（tail）也改走该指针。
3. **grid 并行度**：pubred grid = `ceil(n/1024)=5` block，但活跃线程只有 `n4 = 1280`（≈**1.25 block**），其余 3840 线程在 reduce/epilogue 全程空转。24 行点积的权重流是 **1.92MB**，现在由 **24 block/24 SM** 拉（531GB/s、7.4µs）。融进 pubred 等于把同样字节压到 ~2 个 SM ⇒ 正是 P1c/P1d 记录过的「单块/少块装不下 24 个权重行」形态，回归风险高。（**2026-09-11 部分消解**：三个 AR v5 launcher 已改成 `threads=256, blocks=ceil(n4/256)`，n=5120 时是 5 个满块 × 256 线程摊在 5 个 SM，不再有 3840 空转线程。但 5 个 SM 对 24 行点积的 1.92MB 权重流仍不够 —— 本条阻塞的**结论不变**：dots 不该折进 pubred。）

**非默认路径下的次优载体是 `dsv41_hc_post_inplace`**（kernel `dsv41_kernels.cu:4459`，launcher `:4496`）：与 `g_hc_part`/`hc_mixes_tail_kernel` **同 TU**（无跨 TU 问题）；线程所有权与 pubred epilogue 同构（一线程 4 列 × 全部 hc 行），且与 tail split 兼容。⚠️ 它**不在默认路径上**：`DSV41_HCPOST_EPI` 默认 ON（`hcpost_epi()`，`chain_dev.rs:2077`）时 `layer` 在 `chain_dev.rs:2216`（attn 侧）/ `:2345`（MoE 侧）**跳过**它，只有 `DSV41_HCPOST_EPI=0` 才回到它（`fuse_c()` 默认 true，`chain_dev.rs:2056` 是选择 in-place / h2-staging 的闸）。

**若要做，需一并解决（按序）**：① `hc_mixes_tail_kernel` 的 `mixes` 只读 `g_hc_part[r][m][0]`（`:4100`）——**不 sum ck**，K-split 必须给它加 split-sum（或新变体）；② 前端要 **tail-only 入口**（跳过 dots launch：`hc_front:4687` / `hc_front_split:4781` 各一个）；③ **ss 非逐位**：`ss_in=1` 的 partial 由 `hc_mix_dots_kernel:3982` 按 `m*32+lane` 残差类生成、tail 读 `[r][tid][1]`（`:4081`），该分组在列所有权下无法逐位复现 ⇒ 须容差门禁 + 同二进制 A/B；④ **engram 层（1、14）必须排除 MoE 侧融合**：`engram_apply`（`chain_dev.rs:1230`，调用点 `:1572`）在 block 前**原位改写 `s.h`**，切断了「上一层 MoE hc_post → 本层 attn hc_pre」的直连。

**收益口径修正**：`hc_mix_dots` 是 **80 次/步**（2/层 × 40 层，`STATUS.md:3981`），不是 40 次。融合只省**图节点**（~1.5µs/节点 × 80 ≈ **0.12ms**），点积计算本身（7.0µs × 80 = 0.56ms）仍要付 ⇒ **融合后点积计算绝不能变慢**，否则收益被吞掉。

**新方向复核（dots 融进「前一段最后一个 kernel」的 epilogue，2026-09-11，纯代码分析）：结论不变 —— 结构性可行但净负，不值得做。**

- **段序列：新方向落回同一个载体。** 默认路径（hcpost fold ON）下 attention 段的最后一个 kernel **就是** `p2p_ar_pubred_v5_hcpost`（`chain_dev.rs:2207` 调用 `attention()`，其 `:3154-3174` 的 AR 即收尾）。到 FFN 侧 dots 之间**只有一次 `hc_tail_join` 事件等待、无其它 kernel**（`:2211`；fold 命中时 `:2216` 的 `hc_post_inplace` 被跳过）。唯一例外是 engram 层（1、14）：`engram_apply`（调用点 `chain_dev.rs:1677`）在段前**原位改写 `s.h`**，必须排除。
- **grid 不匹配其实可以精确解决**：残差布局是 `res[i*dim + j]`（`dsv41_kernels.cu:4471`，行距 = `dim`），dots 的 K 空间 = 拉平的 `hc*dim = 20480`（**不是 5120**）。pubred epilogue 每线程恰好拥有 4 列 × 全部 hc 行 = **16 个 K 元素**，`1280 线程 × 16 = 20480` ⇒ **逐元素精确划分**；且 `ar5_hc_post_col4`（`ferrite_kernels.cu:8495`）已在寄存器里走完全部 hc 行 ⇒ x 零额外读取。
- **归约有正解**：24 行 × ck 个块 partial。**原子加会破坏逐位与复现性**；正解是让 LATE 的 `mixes` 按 ck 升序合并（`hc_front_kernel:4320` 已有先例，`g_hc_part[r][m][ck]` 的 ck 维是 8，`dsv41_kernels.cu:4241/4253`）。但这改变部分和顺序 ⇒ 容差门禁 + 同二进制 A/B（与 §1「split 必须 = 1」冲突）。跨 TU（`g_hc_part` 在 `dsv41_kernels.cu`、pubred 在 `ferrite_kernels.cu`，无 `-rdc`）须把 partial 缓冲当参数传指针。
- **真正的硬阻塞仍是并行度**：pubred grid = `ceil(n4/256) = 5` block × 256 线程（`ferrite_kernels.cu:8625`）压在 **5 个 SM**；dots 现在 **24 block/24 SM、7.4µs**（3.93MB/7.4µs = 531GB/s ≈ **22GB/s/SM**，是**延迟地板**而非 DRAM 地板）。把 1.92MB 权重流压到 5 个 SM ⇒ 估算 ~17µs，而 dots 在关键路径上（LATE 等它）⇒ **≈ +10µs/层 × 80 = +0.8ms**，只换回 0.12ms 节点 ⇒ **净亏**。放宽 grid 也救不了：ck 维上限 8 ⇒ K-split 最多 8 块，8 × 22GB/s 也只有 176GB/s（≈11µs），仍慢于 7.4µs。

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

## 10. 注意力投影链的下一段（explore 2026-09-11，纯代码分析，未上机）

**关键几何发现（本节基石）**：`sparse_attn_pf_kernel`（`dsv41_kernels.cu:558`，launcher `:3271`，grid=(b*m, h)、block=128）与 `apply_rope_kernel`（`:1209`，o-rope 经 `dsv41_apply_rope_q:3537`，rows=nlh、block=128）的 **block 归属逐位相同**：一个 block 吃一个 head 的完整 d 行。head 宽（hd=512）是 32 的倍数 ⇒ fp8 发射的 per-32-block 索引在 head 边界对齐，head-local 计算 == 全局 flat 计算（逐位）。

⇒ **sparse_attn + o-rope(inverse) + fp8 发射 三合一 = 几何零变化的融合**，唯一新增的是 sparse 写 out 后的一次 `__syncthreads()`。省 40 launch/步（o-rope 40 次）。是「fork/join 不改 grid 形态 + 单点 epilogue」两条成功模式的直接继续。

**为什么「sparse 的 o 直接进 wo_a 的 dot」不行**：wo_a 的 k = hpg*hd = 4096 = nlh*hd = 8 个 head = **8 个不同 block**（`chain_dev.rs:2798-2808`）⇒ 跨 block 依赖 ⇒ 触犯 hcpm/hc-merge 铁律（`cudaLaunchCooperativeKernel` 与图捕获不兼容，`STATUS.md:4114`）。

**为什么 wo_a→wo_b 链式核不行**（复核 gemv-call-pair-wo，`STATUS.md:5973`）：wo_b 的 k = ol_local = 1024 **恰是 wo_a 的完整 n** ⇒ consumer 必须等 producer 全部 drain，链式核内部即跨块栅栏（选举已被证伪 +3.3ms）。异 k（4096 vs 1024）也不能共享 mx2 的同一份 staging。**只有 B1 epilogue 落地**（`gemm_fp8_gemv_kernel:2113` 的 xq/xsc 尾参 + 跨 warp amax epilogue）。

**方向 6「权重串联」否决（explore 2026-09-11，纯代码分析）**：`wo_b∘wo_a` 之间**无非线性**
（`ref_inference/model.py:785-788` 是 einsum→Linear，无 norm/激活；o-rope 在 wo_a **之前**，
`chain_dev.rs:2966-3049`；默认 `DSV41_WOB_F32` 路径 wo_b 直读 f32 `s.wo`，中间**已无** fp8 量化）
⇒ 数学上可以把 `wo_eff = wo_b_local @ wo_a_local` 在加载期预计算成 **单矩阵 [dim=5120, k=4096]**
（= 21M 参数 / 21MB fp8）。**但 FLOPs 不是「不变」而是 ×2.22**：现值
`wo_a 1024×4096 (4.19M) + wo_b 5120×1024 (5.24M) = 9.44M` MACs/rank，串联后 `5120×4096 = 20.97M`。
低秩因子化（o_lora=1024）本来就是省 FLOPs 的手段，串联等于把它取消。按实测 per-call
（wo_b 5.24MB ≈ 9.5µs，即 ~0.55 TFLOP/s 的占用率/延迟地板）线性外推，串联核 ≈38µs vs 现值
17µs ⇒ **+21µs/层 × 40 = +0.84ms**，最多只回收 40 次 launch（≤0.34ms）⇒ **净 +0.5ms**。
另外 ① wo_b 是 RowParallel，**AR#1 不会被消掉**；② 串联必须把训练好的 fp8 权重先反量化再乘、
再重新量化到 32×32 ue8m0 ⇒ **新增一个精度源**（不是任务书假设的 1e-6，而是 ~1e-3 量级、
且必然破坏 parity 逐位契约）；③ wo_a 的 `nlg=1`、wo_b 的 k=`ol_local`=1024 都极小，中间量只有
4KB，不存在可回收的「中间流量」。**结论：不可行，勿试。**

**真正「persistent」杠杆 = PDL 串链（已实施 2026-09-11，未上机验证）**：`pdl_or_plain`（`ferrite_kernels.cu:725-765`，`cudaLaunchAttributeProgrammaticStreamSerialization`）已存在且在 GDN/DSA 投影族验证过 capture。现在 `dsv41_kernels.cu` 里有了自己的副本 **`dsv41_pdl_or_plain`**（gate `DSV41_PDL`，**默认 ON**，`=0` 回退；launcher 用 `cudaLaunchKernelEx` 发射），覆盖注意力投影链 consumer 端的 **8 个 launch 点**：

| consumer kernel | launcher | 入口 sync |
|---|---|---|
| `gemm_fp8_gemv_kernel` | `dsv41_gemm_fp8_mx`（M=1 分支） | `dsv41_kernels.cu:2578` |
| 同上 | `dsv41_gemm_fp8_mx_rope` | 同上 |
| 同上 | `dsv41_gemm_fp8_mx_rope_norm` | 同上 |
| 同上 | `dsv41_gemm_fp8_mx2_rope` | 同上 |
| 同上 | `dsv41_gemm_fp8_mx_add` | 同上 |
| 同上 | `dsv41_gemm_fp8_mx_f32`（wo_b） | 同上 |
| 同上 | `dsv41_gemm_fp8_mx2` | 同上 |
| `sparse_attn_pf_kernel` | `dsv41_sparse_attn`（pf 分支） | `dsv41_kernels.cu:574` |

**未覆盖（刻意）**：M>1 的 `gemm_fp8_kernel` tile 路径、`sparse_attn_split/merge/warp` 三个 A/B 变体——它们仍走 plain launch。

**关键修正（与本节早期假设不同）**：「把 prologue 藏进 producer ramp-down」对这两个 kernel **headroom 很小**——它们的 prologue 主体（gemv 的 activation staging、sparse 的 q-row staging / `*clen`）**本身就读 producer 的输出**，必须在 sync 之后；真正不依赖 producer 的只有 gemv 的 256 项 e4m3 LUT（每线程 1 次迭代）和指针设置，hoist 收益 < 结构化改写的风险（gemv 的权重行 cp.async 在 row loop 内，其 commit/wait 配对承载比特一致性）。因此 sync 放在 kernel 入口（与 `gemv_bf16_v2_kernel:2887` 既有先例一致），**回收的是节点过渡/launch 开销，不是 prologue 的算术**。真正的收益量级必须在图上 A/B。

**风险线**：`DSV41_PDL` 默认 ON ⇒ rebuild 后所有 DSV41 运行即生效。GLM 路径上 PDL 曾测为**中性**（且当时只覆盖 4 个 launcher），所以**上线前必须先做 `DSV41_PDL=0/1` 的图 A/B**；`=0` 是回退臂。host 侧 gate 不做 arch 判断、device 侧 sync 有 `__CUDA_ARCH__ >= 900` 守卫，故本文件必须按 sm_90+ 编译（build.sh 默认 100a）；不支持的设备上 attribute 会让 launch 显式报错，不会静默。

**expert 链的 PDL 扩展（已实施 2026-09-11，未上机验证）**：同一模式延伸到 `quant_fp4 → gateup → down_reduce`。`dsv41_experts_mxf4.cu` 是独立 TU，因此带**自己的副本 `dsv41_experts_pdl_or_plain`**（`dsv41_experts_mxf4.cu:747-774`），gate 复用 `DSV41_PDL`（默认 ON、`=0` 回退），语义与另两个副本逐条相同。覆盖的 consumer 是 **3 个 launch 点 / 2 个 kernel**：

| consumer kernel | launcher | 入口 sync |
|---|---|---|
| `expert_gemv_fp4_batched_kernel<ILV>` (gate/up) | `dsv41_expert_gate_up_fp4_batched` (`:1659/1666`) | `dsv41_experts_mxf4.cu:846` |
| `expert_gemv_fp4_batched_kernel<ILV>` (down) | `dsv41_expert_down_fp4_batched` (`:1691`) | 同上（同一 kernel，staging 源不同） |
| `expert_gemv_fp4_down_reduce_kernel<STAGED>` | `dsv41_expert_down_reduce_fp4_batched` (`:1756/1761`) | `dsv41_experts_mxf4.cu:1250` |

**刻意未覆盖**：`*_indirect` 顺序 per-slot 入口（fallback 臂）；`quant_fp4_fused_kernel`（见下）。
**producer 关系（已核实）**：gateup 的 producer = `quant_fp4_fused_kernel`（写 `a`/`a_scale`）；down / down_reduce 的 producer = gateup launch（写 swiglu 后的 `act_base`）。`ids`（`route_idx`）与 `row_weight`（`route_w`）由 **router 在 producer 之前的若干个 kernel** 写，PDL secondary 释放时已 flush，因此在 sync 之前读它们是安全的 —— 这正是被 hoist 的指针工作。
**与 attention 链不同的 headroom**：这里真的有可 hoist 的**非平凡** prologue —— 256 项 e2m1 LUT —— 它已从 staging 之后**移到 sync 之前**（只写本 CTA 的 smem，由既有 `__syncthreads()` 发布，与 staging 的 smem 区间不重叠 ⇒ 顺序中立、逐位不变）。所以 expert 链回收的是 **节点过渡 + LUT/指针 prologue 之和**，比 attention 链仅回收节点过渡略多。
**为什么 `quant_fp4_fused_kernel` 不加 PDL**：① 它对紧邻 producer（`gemv_bf16_route` 写的 `scores`/`route_idx`/`route_w`）**没有任何数据依赖**（只读 `xn`），本可完全重叠；但那样会削弱 `route_idx` 对 gateup 的**传递可见性** —— gateup 的 sync 只保证 quant 的写可见，不保证 quant 没等过的 route 的写可见。② 若在 quant 入口加 sync 则安全（恢复全序）、收益仅节点间隙，但 `dsv41_quant_fp4` 是**共用工具**（attention 链每层 6+ 次 `quant1` 也走它），加 attribute 会越出 pdl-chain-impl 逐点审计过的范围。需要时为 expert 链单开一个带 attr 的入口更合适。
**验证**：`DSV41_PDL=0/1` 图 A/B；`DSV41_EXPERT_ILV=1` 与 `=0` 各跑一次（ILV/FUSE 组合的逐位契约见 `dsv41-kernel-inventory-v3.md`）。本机**无 nvcc**，`.cu` 未编译（仅做括号平衡 + 变参展开的 host 桩测试，见下）。

**骨架（三合一核）**：

```cpp
// grid=(b*m, h), block=128 —— 与 sparse_attn_pf / apply_rope 逐位同几何
__global__ void sparse_attn_orope_kernel(const float* q, const float* kv, const float* sink,
                                         const int32_t* idxs, float* out, int b, int m, int h, int d,
                                         const int* clen, int win, int index_topk, float scale,
                                         const float* cos, const float* sin, const int* base,
                                         int rope_rd, int half, int mul, int off,
                                         uint8_t* xq, float* xsc) {
  // phase 1: sparse_attn_pf 主体逐句照抄（qv staging / 三深预取 / online softmax）
  //          唯一改动：out 行留在 smem（512 f32 = 2KB/block，128 线程 x 4 lane）
  __syncthreads();                       // 新增的唯一栅栏
  // phase 2: inverse rope，仅本 head 的 [d-rope_rd, d)，pair(2i,2i+1)
  //          表达式取自 apply_rope_kernel:1218-1224（inverse=true）
  // phase 3: fp8 发射，per-32 block；索引 = (row*h+hh)*d + c，512%32==0 ⇒ 与 flat 版逐位同
  //          公式取自 apply_rope_kernel:1230-1246（fast_round_scale + clamp + e4m3）
}
```

**验收**：`DSV41_SPARSE_OROPE=0` 回退旧双 launch；`sparse_orope_parity.rs` 逐位比（同设备、确定性输入）。

### 10.1 实现落地（2026-09-11，未上机编译、未跑 parity）

已按上骨架落地，**新增而非改写**（不改 `dsv41_sparse_attn` / `apply_rope_kernel` 的既有行为）：

| 位置 | 内容 |
|------|------|
| `dsv41_kernels.cu:1282` | `sparse_attn_orope_kernel`：phase 1 逐句照抄 `sparse_attn_pf_kernel`；phase 2 = `apply_rope_kernel:1218-1224`；phase 3 = `:1230-1246` |
| `dsv41_kernels.cu:3702` | `dsv41_sparse_attn_orope` launcher（新函数，未改 `dsv41_sparse_attn`）。**decline 哨兵 = 1/2/3**：1=形状不可用；2=plain 调用不会选 pf（`DSV41_ATTN_SEQ` / key-split / `DSV41_ATTN_PF=0`）；3=发射形状不精确。**刻意避开哨兵 1**（与 `cudaErrorInvalidValue` 冲突，见 `dsv41_apply_rope_q` 的 legacy 陷阱） |
| `device.rs:172/590/1594` | `sparse_attn_orope` 可选符号（`ko!`）+ struct `Option` 字段 + wrapper（`Ok(false)` = 回退） |
| `chain_dev.rs:565/2766` | `sparse_orope()` env gate（`DSV41_SPARSE_OROPE`，默认 ON）；调用点先试融合，`!s_orope` 才跑 `sparse_attn` + `apply_rope_q` + `quant1` |

**两处对骨架的修正**：

1. **smem 预算是 2 KB 而不是 16 KB**。§10 正文写对了（`512 f32 = 2KB/block`），但任务书写成 `nlh*hd*4B = 16KB`——那是 grid 全体的 o 行总量，不是单 block 的。grid=(b*m,h) 下 `gridDim.y == h`，一个 block 恰好一个 head，`sh_row[512]` 足够。总静态 smem = 8448 B（`sh_smax/sh_se/sh_acc` 不变）+ 2048 B = **10.5 KB**，远在 48 KB 默认线内，**无需 `cudaFuncSetAttribute`**。
2. **新增两个栅栏而非一个**：phase 2（rope）读其它 lane 写的 `sh_row` 列 → 需要 phase1→phase2 栅栏；phase 3（全局 store + fp8 发射）读 rope 改过的列 → 需要 phase2→phase3 栅栏。加原 merge 栅栏与 hh 循环末栅栏，共 4 个/head，全在 block 内。

**bit-identical 的关键等价**：per-head 发射的 flat block 索引 = `((row*h+hh)*d)/32 + blk`；因 `d % 32 == 0`，head 起点必落在 32-block 边界，故它 == flat 版 `xsc` 的同一下标。**若 `hd % 32 != 0` 此等价失效** —— launcher 的 `(d & 31) != 0 → return 1` 为此硬门。

**phase 3 直接写 roped 值到 global**（而非写未 rope 的中间值再原地旋转）：旧双 launch 的净效果同样是 `s.o` 最终 = roped；`sparse_attn` 与 `apply_rope` 之间无人读 `s.o`，中间态不可观测。

**未验证项（上机前必做）**：① 本机无 nvcc/GPU，`.cu` **未编译**（仅括号平衡自检 + 逐行静态复核）；② `--use_fast_math` 下 rope 的 `x0*cc - x1*ss` 收缩需 parity 逐位确认；③ 建议 `DSV41_SPARSE_OROPE=0` 与 ON 各跑一次逐步 RMS 对照。
