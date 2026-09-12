# 下一会话交接文档（NEXT-SESSION-HANDOVER）

> 上一会话终态：**13.28ms → 6.15ms（+115.4%），75.3 → 162.6 tok/s**（v24a，serve 验证）。
> HEAD `b7dce61`，工作树干净。**本文件是"打开就能干活"的入口**；细节一律回指下表的来源文档。
> 读完本文件 + `docs/agent/expert-tcgen05-plan.md` 即可开工。

---

## 1. 首要任务（唯一存活的路径）：expert tcgen05 fp4 swapAB

**目标**：expert 段 2.0ms/步 → ~1.0ms/步（步时 ~5.15ms ≈ 194 tok/s）。
**工时**：4-5 人日，四阶段。**计划书（必读）**：`docs/agent/expert-tcgen05-plan.md`（含 Phase 0/1 实施状态 §1b/§1c、6 条协议、风险清单）。

### 1.1 为什么是这条路
- expert 段 2.00ms = 步时 32%，是 **DRAM-bound**：0.9GB/步 ÷ 8TB/s = **0.11ms 地板**，当前 **18×** 差距。
- 现路径用硬件固定 M=128 tile 跑 M=1 的**单行激活**，**128× 冗余**（实测 16.8GB/s = 0.2% 带宽）。
- swapAB 把**权重复**当 M 操作数：gateup M=3840=30 个满 128 tile、down M=5120=40 tile，权重侧**零冗余**。
- ⚠️ **但只换映射不改 staging 仍是 0.2% 地板**——`mxf4_gemm_kernel:348-421` 是 LDG→STS→MMA 单级，每级付满 DRAM 延迟。所以新 kernel 必须带 **kRing TMA 环**。

### 1.2 已完成
| 阶段 | 交付 | 位置 | 状态 |
|---|---|---|---|
| **Phase 0** | 真实量化 parity 探针 | `kernels/cuda/tests_tcgen05_mxf8f6f4_1x.cu`（**1565** 行） | **host 侧数值闭环 ✓；GPU parity 待跑** |
| **Phase 1** | gateup swapAB kernel **骨架（A 臂 mxf8f6f4）** | `kernels/cuda/dsv41_experts_mxf4.cu` 尾部（第 2686 行起，**713** 行） | 编译验证 ✓（101 regs / 0 spill / 46080B static smem），**GPU parity 待跑** |
| **Phase 1（默认臂）** | gateup swapAB kernel **骨架（B 臂 mxf4）** | 同文件尾部、`namespace tc5::mxf4`（第 3400 行起，**~640** 行） | 编译验证 ✓（**92 regs / 0 spill / 40960B static smem**，kRing=8），**GPU parity 待跑** |
| **Phase 1 默认 MMA** | 🔴 **`kind::mxf4`**（两侧 packed，E2M1×E2M1，`scale_vec::2X`） | 计划书 §1e；骨架注释内（~30 行可切） | **B 臂 GPU 已验 EXACT**；A 臂 `mxf8f6f4`/e4m3 降为备选 |

> 🔴 **2026-09-12 修订**：Phase 1 默认 = `kind::mxf4`（§1e 定案）。理由：**唯一已在 GPU 数值验证过**的
> fp4 tcgen05 形式 + 与现生产 e2m1 基线逐位同源 + 操作数 smem 3.06x（ring 3→7-8）。
> ⚠️ 订正流传"定案"的两处**与代码不符**（详见 `expert-tcgen05-plan.md` §1e.1）：
> 参考实现的 fp4-weight `linear()` 激活是 **FP8 e4m3**（`model.py:187` + `kernel.py:111,490,511`），
> **不是** fp4 e2m1——"激活本来就是 e2m1"说的是**现生产 ferrite**；故 mxf4 相对参考**不是零损失**，
> 而是沿用既有偏差。另：「smem 减半 ⇒ 更多 CTA/SM」不成立（TMEM 256 列/CTA 钉死），真收益是 ring 深度。

**Phase 0 关键结论（本地已闭环，无 GPU）**：
- 已抓到并修掉一个**真 bug**：e4m3 编码器把 `efield>=15` 当溢出 ⇒ [256,448] 十倍程被压到 448。**这类 bug 从 GPU 侧不可见**（golden 用同一份字节）——**codec 自检必须留在套件里，每次运行都跑**。
- K≤256 时 f32 累加与 double golden **逐位相同**；K=5120 时 max rel 2.5e-05 ⇒ GPU 上 "tcgen05 vs golden" 应报 ~1e-6..1e-5，判据 5e-2 有两个量级余量。
- **判据只看 norm 列**（l1/l2）：逐元素 p50≈0.12 / max 极大是随机点积近零输出的除零效应，看逐元素会误判。
- 残留：①输入仍是合成 f32，**未接真实 checkpoint 字节**（接法见 §1b）；②smem staging 上限 `PH0_MAXBLK=8`（K=256），真实 K 需 chunked（属 Phase 1）；③PERBLK 装不下真实 K，**只能消歧不能上产**。
- 构建验证（4 种全过）：`nvcc -gencode arch=compute_103a,code=sm_103a` / `-DPROBE_ASM_ONLY=1` / `-DPROBE_RAW_LAYOUT=1` / `g++ -DPH0_HOST_ONLY`。

**Phase 1 骨架的 4 个新发现（实施前必读，超出原设计）**：
1. **PACKED SF 是被迫的，不是选择**：PERBLK 需 4 列/块 ⇒ 160 块 = 640 列 > 512 TMEM 列，装不下。⇒ PACKED 挂 = 整个 Phase 1 布局假设挂。SF 寻址：`SFA col = sfa_col + 4*(b>>2), a_sf_id = b&3`；`SFB col = sfb_col + (b>>2), b_sf_id = b&3`（契约 `dim % 128 == 0`）。
2. **mxf8f6f4 的 fp4 操作数必须 UNPACKED（1 元素/字节）**，checkpoint 是 2 元素/字节 ⇒ 每个 ring slot 需**两块 buffer**（TMA 落的 raw 打包区 + MMA 读的 unpack 区）+ 一次 1:2 展开。**这是本 kernel 最大 smem 开销，也是"TMA 直写操作数"不可能的原因。** ⚠️ 这是 **A 臂（备选）的固有成本** —— §1e 已把默认切到 B 臂，B 臂**无此开销**。
   - **🔴 默认演化（§1e）**：`kind::mxf4`（两侧打包）⇒ 去掉**展开**段（permute 段仍在，见注）、操作数 smem **per-slot 13312→4352 B（3.06x）**、用**唯一已在 GPU 验证过**的 `tests_tcgen05_mxf4.cu` 那条 MMA。⚠️ 精度上它是**相对参考**的 e4m3→e2m1 降级（**不是"零损失"**；零增量风险只相对**现生产** e2m1 基线，见 §1e.1/§1e.3）；2X 粒度需配对 checkpoint 的 per-32 scale（现有 kernel 已在做）。**切换约 30 行。** ⚠️ smem 减半**不会**翻 occupancy（TMEM 256 列/CTA 才是绑定点）；真收益是 kRing 3→7-8。
   - **✅ 已落地（§1f）**：B 臂骨架已写进同一 TU（`namespace tc5::mxf4`，gated `DSV41_TCGEN05_GATEUP_MXF4_SKELETON`），kRing=8 / 92 regs / 0 spill / 40960 B static smem。它比原估的"30 行"大，因为**没有 raw buffer 后 permute 从"smem→smem 展开"变成"16 B chunk 直落 canonical unit 的 TMA 目的地址"**（每 stage 258 条 bulk copy），这一处是结构差异而非常量替换。
3. **静态 smem 装得下**（kPackK=64/kRing=3 时 45 KiB）⇒ 无需 `cudaFuncSetAttribute`，**图捕获安全**。更深 ring 才需动态 smem + **init 期一次性** attribute（**捕获内禁用**）。
4. **occupancy 是硬约束**：grid=(30, slots)，每 CTA 256/512 TMEM 列 ⇒ ≤2 CTA/SM。**slots=1 只有 30 CTA（≈15 SM）⇒ 隔离微基准必须跑 slots=8（240 CTA）**，否则测的是延迟不是带宽。若 slots=8 仍离地板远 ⇒ **K-split**（更多 CTA 覆盖同权重，需确定性升序 reduce）。

**骨架待填**：GPU parity（复用 Phase 0 harness + 真实 checkpoint 字节）、`kPackK`/`kRing` 调优、2D tensor TMA（1 条指令替代 129 条）、K-split reduce、Phase 2 的 pool/ids 间接寻址 + Rust FFI。骨架内已标 `TODO-1..5` + `K-SPLIT`。

**骨架编译验证命令（无需 GPU）**：
```bash
# A 臂（备选，mxf8f6f4 / 1X / e4m3）
nvcc -gencode arch=compute_103a,code=sm_103a -O3 -std=c++17 \
     -DDSV41_TCGEN05_GATEUP_SKELETON=1 -c kernels/cuda/dsv41_experts_mxf4.cu -o /tmp/t5.o
# ptxas -v 必须：0 spill + tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X

# B 臂（默认，mxf4 / 2X / e2m1）
nvcc -gencode arch=compute_103a,code=sm_103a -O3 -std=c++17 \
     -DDSV41_TCGEN05_GATEUP_MXF4_SKELETON=1 -c kernels/cuda/dsv41_experts_mxf4.cu -o /tmp/t5m4.o -Xptxas -v
# 已验：92 regs / 0 spill / 40960 B static smem + tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X
# 4 种宏组合（无宏 / A / B / A+B）全 0 error，两种臂可同时 -D（tc5::mxf4 是嵌套命名空间）。
```
A 臂入口 `tc5::expert_tcgen05_gateup_kernel` / launcher `tc5::tc5_launch_gateup` / FFI `dsv41_expert_tcgen05_gate_up`（env `DSV41_EXPERT_TCGEN05`）。
B 臂入口 `tc5::mxf4::expert_tcgen05_gateup_mxf4_kernel` / launcher `tc5::mxf4::m4_launch_gateup` / FFI `dsv41_expert_tcgen05_gate_up_mxf4`（env `DSV41_EXPERT_TCGEN05_MXF4`）。
两者门禁均为**函数内 static const 只读一次**，默认 OFF。`build.sh` **不定义这两个宏** ⇒ 都不进 `.so`。

### 1.3 期望管理（**必须提前对齐，否则会重演 swapAB 的浪费**）
来自 `dsv41-session-final-report.md` §9.2 的 perf-model 定案：
- **天花板**：即使 tcgen05 100% 完美，步时也只到 ~4.65–5.15ms ≈ **194–215 tok/s**（仅 0.5ms 场景才真正越过 200）。
- **叠加 serve-translation 折扣后现实预期 = 170–175 tok/s**（乐观 60% 折扣 180 / 中性 80% 折扣 174 / 悲观同 swapAB 162.6）。
- **200 tok/s 需要三者同时**：tcgen05 + gemv 减半（**无路径**）+ hc/AR 再砍（**已在地板**）⇒ **研究级，不是工程目标**。
- **止损门**：若隔离微基准达标但 **serve A/B 中性（Δ<0.05ms）⇒ 立即关闭路径、默认 OFF**，不再投变体矩阵（勿重演 v17→v21 四变体全中性）。

---

## 2. 验证基线（唯一认可的口径）

- **终态基线**：**6.15ms = 162.6 tok/s**（v24a）；次优 6.17ms/162.1（v17a）。同配置读数漂移 ±0.09ms，属噪声。
- **serve A/B 驱动**：`scripts/dsv41_serve_ab.sh <tag> [VAR=VALUE ...]`（同二进制背靠背跑两臂）。
  - 判据 = **四段文本逐字相同**（`The capital of France is` / `请背诵《静夜思》` / `1+1=` / `请背诵《出师表》开头`）+ **`faults=0`** + **p50 下降**。计时只认日志里的 `[dsv41] step pos=N` 行，**禁段平均**。
  - 关键 env：`DSV41_MODEL_DIR`（默认 `/opt/dlami/nvme/models/DeepSeek-V4.1-Flash`）、`DSV41_PORT`（默认 8090）、`ARCH`（默认 103a；build.sh 默认 100a 已过时）。
  - **构建顺序（唯一可行）**：`build.sh ARCH`（重写 `.build_id`）→ `touch crates/ferrite-kernel/build.rs`（强制 rerun）→ `cargo build --release`。**单独 `cargo build` 永远修不好**（cargo 增量 + build.rs 戳记）。脚本已内置自愈，但需 nvcc。
  - 缺 `.so`/binary 时先跑：`PHASES=0 scripts/dsv41_recovery_verify.sh`。
- **生产构型**（做隔离实验前必须一致）：gate 已 ON = `DSV41_GATEUP_KSPLIT=2`（−0.33ms）、`DSV41_GATEUP_PIPELINE=1`（旧深度 2/5 已删）、P4 `DSV41_GATEUP_CPASYNC` ON、`DSV41_EXPERT_ILV` ON、`DSV41_DOWN_FUSE` ON；**W2 prewarm OFF**（+0.04ms 回归）。

---

## 3. 铁律（违反 = 数据无效/回归）

**8 条方法论**（`dsv41-session-final-report.md` §7.1）：
1. **隔离探针只用于淘汰；正向收益必须 serve A/B。** 隔离无法复现 serve 的 4 个条件：SM 争抢（侧流并行）、L2 竞争、占用率敏感、graph replay。本会话 **7 次**隔离→生产失效。
2. **`fork_ev` 是 kernel 级事件，不是 block 级**——给 gating kernel 加任何工作 = 加到 main 关键路径。小 kernel 合并收益必须 > gate 语义代价。
3. **失败实验代码立即物理删除**，不留 gated-off；**gate 必须验证"OFF 时是否真的回退"**。
4. **FFI 边界（.cu ↔ Rust）是原子性单位，必须同一 commit。** `git add -A` 在并行 subagent 下危险（docs 提交扫入进行中实施 = v11 的 709 崩溃）；docs 提交用 `git add <specific-files>`。
5. **给热点 kernel 加运行时参数/分支 = 编译产物变重 = 回归风险**（用模板或独立 kernel）。
6. **小 kernel 合并/folding 的收益分析必须考虑 gate/fork 语义**（quant/swiglu/stamp 三条 fold 全败）。
7. **位一致是硬约束**（fingerprint / 四段文本 / 逐位一致性）。
8. **gate 卫生（最决定性）**：改门禁默认值时 `FileReplace` 必须按**函数名/上下文锚定**，改完**读回确认**。用 `int v = 2` 裸声明作锚点会静默改错同名对象——正是它造成 **v13→v15 连续三轮 ~0.37ms 误诊 + 499 行无谓清理**（`006bd0c` 的 FileReplace 把 K-split 误关、PDEPTH pipeline 误开）。

**serve-faithful 隔离协议（6 条，做隔离实验必须全满足，否则只能当淘汰信号）**：
① 生产构型 ② 图模式 capture+replay（禁 direct-launch 计时）③ 侧流干扰注入 ④ L2 污染（每轮换缓冲）⑤ 关键路径计费（按 `fork_ev`）⑥ 判据分层。

**三分类的教训**（§7.2，7 次失效）：测量有偏 ×3 / serve 条件改变 ×3 / **系统差异 ×1（swapAB，最贵）**。系统性差异权重：**SM 争抢 > L2 竞争 > graph replay > 构型漂移**。

---

## 4. 已关闭的路径（**勿重试**）

- **swapAB（gemv 版）**：`gemm_fp8_swapab_kernel` 族 ×5 变体（全量 / 形状分发 / memset 消除 / TMA bulk staging / ring 几何 KStep=64）**serve 全矩阵中性/回归**，**v24 正式关闭**（§3.5）。隔离 1.76–1.94× 在 serve **完全不兑现**。代码保留（默认 OFF）仅供架构变化后重测，**不得再计入路线图**。
- **DL K-chunk**（`hc_dots_late_kchunk_kernel`，`DSV41_HC_DL_KCHUNK`）：v22 serve **略负**（6.25 vs 6.17）。默认 OFF。
- **DOTS_T** = 256/512：中性，保持默认 128。
- **PDEPTH pipeline**（深度 2/5）：**+0.04ms**，深度 >1 的实例已物理删除。
- **w2 L2 prewarm**：**+0.04ms**，OFF。
- **Stage C persistent 段核**：占用率崩塌 + `grid.sync` 与图捕获不兼容，关闭。
- **cross-layer pipe**：仅 −0.05~0.15ms（80% 被路由依赖挡死），关闭。
- **M>1 expert batching**：**价值 = 0**（单请求无第二个 token；层内 6 专家已在一个 launch）。
- **expert MMA 化（旧框架）**：前提不成立——**sm_103a 无 M=8/16 的 fp4 MMA**，M=128 masked 实测 0.2%。
- 另见 §3 表 13 条失败项；bf16-lut / cvt 解码路线已被**分析否证**（勿再提案）。
- **M 侧固定成本认知**：gemv 2.7ms 是 SIMT **compute-bound**（isolated 11µs 不变），a32-vec4/P4/swapAB×5 全无效 ⇒ **gemv 确认无路径**。

---

## 5. 关键文件位置

| 用途 | 路径 |
|---|---|
| **本会话全部定案（权威流水）** | `crates/ferrite-dsv41/STATUS.md`（7936 行） |
| **会话终报（最权威汇总）** | `docs/agent/dsv41-session-final-report.md` |
| **技术细节/证据链** | `docs/agent/perf-roadmap.md`（2147 行） |
| **首要任务计划书** | `docs/agent/expert-tcgen05-plan.md` |
| Phase 0 探针 | `kernels/cuda/tests_tcgen05_mxf8f6f4_1x.cu` |
| Phase 1 骨架（文件尾部） | `kernels/cuda/dsv41_experts_mxf4.cu`（2686–3398） |
| serve A/B 驱动 | `scripts/dsv41_serve_ab.sh` |
| 构建恢复 | `scripts/dsv41_recovery_verify.sh` / `kernels/cuda/build.sh` |
| kernel 清单/架构 | `docs/agent/dsv41-kernel-inventory-v3.md`、`ferrite-unified-arch.md` |

---

## 6. 下一会话第一步（建议顺序）

1. 读本文件 + `expert-tcgen05-plan.md`（**§1e 定案最重要**，其次 §1c）。
2. **Phase 0 GPU 验收（顺序已按 §1e 改判）**：
   a. **B 臂（默认）**：`nvcc -gencode arch=compute_103a,code=sm_103a -O3 -std=c++17 -o t tests_tcgen05_mxf4.cu && ./t` ⇒ 期望 `maxdiff=0.000e+00`（7 shapes，EXACT）。这是唯一已有 GPU 证据的路径，工作量最小。
   b. **A 臂（备选，可选）**：`tests_tcgen05_mxf8f6f4_1x.cu` 上机跑全绿（含 codec 自检 + PACKED/PERBLK 判决）。**parity 不过不进入该臂的 Phase 1**。
3. Phase 1：**默认按 B 臂切骨架**（~30 行：idesc/LBO-SBO/ring/去 unpack），填骨架 TODO；**先跑 `slots=8` 隔离微基准**判是否达 DRAM 带宽（否则先做 K-split）。
4. 隔离达标后**直接进 Phase 3 serve A/B**（两次独立启动，臂 = `DSV41_EXPERT_TCGEN05=0/1`）；**中性即止损关闭**。
