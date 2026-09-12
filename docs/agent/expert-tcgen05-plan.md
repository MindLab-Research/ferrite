# expert tcgen05 fp4 swapAB 实施计划（下一会话）

> 前置：本会话 `tcgen05-probe` 已用 `kernels/cuda/tests_tcgen05_mxf8f6f4_1x.cu`
> 在 `-gencode arch=compute_103a,code=sm_103a` 下 ptxas 验收通过
> `tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X`，操作数布局确认：
> A=fp4 smem descriptor，B=e4m3，D=TMEM，SFA/SFB=TMEM e8m0。使用（不重证）。
> 目标：expert 2.0ms/步 → ~1.0ms/步 ⇒ 步时 ~3.0ms ≈ 300+ tok/s（**唯一路径**，4-5 人日）。
> 代码基线：`kernels/cuda/dsv41_experts_mxf4.cu`（现有 `mxf4_gemm_kernel` / `expert_gemv_fp4*`）。

## 0. 为什么 expert 与 gemv 的 swapAB 不同（避免重蹈 serve 中性）

| | gemv swapAB（已关闭） | expert tcgen05（本项目） |
|---|---|---|
| 权重读取 | 延迟绑（并行度不足，10.8x 差） | **DRAM 绑**：0.9GB/步 ÷ 8TB/s = 0.11ms 地板，当前 2.0ms = **18x** 差距 |
| M=128 tile | n 侧列复制的固定成本 | **权重侧满填**：gateup 3840/128=30 tile、down 5120/128=40 tile |
| 结论 | 隔离 1.76-1.94x，serve 中性 | **有真实 DRAM 缺口**，但仍有翻译风险 ⇒ 必须走 6 条协议 |

**swapAB 语义**：`D[M=权重行, N=激活列] = A[权重 M×K] × B[激活 K×N]`。
M 侧 3840/5120 整除 128（满 tile），N 侧 decode=1 token → 补到 **N=8**（mxf8f6f4 最小合法 N），
仅激活侧 8x 冗余（可忽略，权重侧零冗余）。
- gateup：A=W1W3[3840, dim=5120]，B=act[5120, 8]，K=5120（160 个 32-块 ✓）
- down：A=W2[5120, inter=1920]，B=act[1920, 8]，K=1920（60 个 32-块 ✓）
- scale：K 粒度 32 e8m0，与 checkpoint per-(row,k/32) **零转换**复用。

## 1. Phase 0（0.5 天）探针扩展：真实权重 + 数值 parity

**目的**：把「ptxas 通过」升级为「数值正确」，在 GPU 上闭环。
1. 复用 `tests_tcgen05_mxf8f6f4_1x.cu` 脚手架，把 PROBE_K 从 32 扩到真实 K（5120/1920 分块循环）。
2. 输入改用**真实 checkpoint 权重**（某专家 W1/W3/W2 的 fp4 e2m1 + e8m0 per-32）；
   激活用同一 x，量化到 **e4m3（block 32, e8m0）**（注意：现 expert 路径量化到 fp4 e2m1；
   此处改 fp8 精度更高，数值风险降低）。
3. 判据（对齐 `tests_tcgen05_mxf4.cu` 口径）：
   - `max|diff| / max|ref| < 5e-2`（vs 当前 SIMT `expert_gemv_fp4` / mxf4 GEMV）
   - 每元素相对误差 p50/p90/p99/max < 5e-2（分母下限 1e-3·max|ref|）
   - CPU golden（同字节块缩放和）两路都 < 5e-2（防共享布局 bug）
   - `argmax` 一致 + 报 `max|diff| vs top1-top2 margin`
4. **验收**：GPU 上 `nvcc -gencode arch=compute_103a,code=sm_103a && ./t` 全绿。
   **若 parity 不过，不进入 Phase 1**（布局/scale 映射的 bug 必须在此拦下）。

## 1b. Phase 0 实施状态（2026-09-12 代码已落地；GPU 待跑）

`kernels/cuda/tests_tcgen05_mxf8f6f4_1x.cu` 已由"随机码布局探针"扩展为**真实量化 parity 套件**
（1565 行，原探针完整保留，可 `-DPROBE_RAW_LAYOUT=1` 单独跑）：

- **默认 `main`** = codec 自检 → 随机码布局探针（case 0，精确算术）→ 4 个真实量化 case
  （K=32/128/256/256；N(0,1)/outlier/lognormal 组合）。每个 case 跑**两种 SF 布局假设**：
  `PACKED`（4 块/字 `[SF(b0..b3)]`、sf_id=b%4、字列按 32 行组步进 —— 即 `dsv41_experts_mxf4.cu`
  已数值验证的 2X 字布局搬到 1X）与 `PERBLK`（1 块/字、byte 0、sf_id=0，逐块等价于已验证的单块探针）。
  nblk=1 时两者按构造逐位相同（套件只跑一次并说明）。⇒ **自带判决**：PACKED 挂而 PERBLK 过
  ⇒ SF 字节选择模型错，不是 MMA / smem descriptor / TMEM 映射错。
- **四方参照**：(a) 反量化 CPU golden（double）＝判据对象；(b) 未量化 f32 ＝量化损失（信息项）；
  (c) GPU f32 SIMT 反量化 GEMV ＝现 expert 路径的算术形状（`expert_gemv_fp4` 读打包 fp4 激活，
  不能直接调用，故镜像其算术）；(d) f32 `fmaf` 逐块累加 ＝预测 tcgen05-vs-golden 的量级。
- **判据**：max rel err < 5e-2（含 p50/p90/p99）、norm 口径（max/max、l2、l1）、
  argmax（M 向 / N 向）、min top1-top2 margin 与 diff/margin 比。
- **本地已闭环（无 GPU）**：4 种构建全过（`-gencode arch=compute_103a,code=sm_103a` /
  `-DPROBE_ASM_ONLY=1` / `-DPROBE_RAW_LAYOUT=1` / `g++ -DPH0_HOST_ONLY`）；`ptxas -v`：
  `ph0_parity_kernel` 77 regs / 34828 B smem / **0 spill**，PTX 内 5 条
  `tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X`。host-only 模式可本地复现全部数值：
  - **codec 自检（暴力最近码，每次运行都跑）**：e2m1 超出 0、e4m3 超出 0、e8m0 往返 OK。
    **已抓到并修掉一个真 bug**：e4m3 编码器原先把 `efield >= 15` 当溢出钳位 ⇒ 整个 [256,448]
    十倍程被压到 448（超出最近码 192），量化损失被虚高约 2 倍。**这类 bug 从 GPU 侧不可见**
    （golden 用的是同一份字节），所以自检必须留在套件里。
  - **(d) 预测**：K≤256 时 f32 累加与 double golden **逐位相同**（e2m1×e4m3 积有限位精确，
    32 项部分和 < 24 bit）；K=5120 时 max rel 2.5e-05 / l2 3.2e-08 ⇒ GPU 上
    "tcgen05 vs golden" 应报 ~1e-6..1e-5，判据 5e-2 有两个量级余量。
  - **(a) 量化损失（信息项，不是判据）**：合成 N(0,1) 权重按 checkpoint 口径
    （per-32 amax → `2^ceil(log2(amax/6))`，与生产 `fast_round_scale6` 同源）量化后
    **l1 rel ≈ 0.11-0.16、l2 ≈ 0.11-0.15**；逐元素 p50≈0.12、max 极大 —— 后者是随机点积
    近零输出的除零效应，**判据必须只看 norm 列**，否则会误判。格式本身会翻掉 1-2/8 的
    列 argmax（128 行的 top1-top2 margin 相对行范数只有 1e-3 量级）——真实模型不比这更宽松，
    这也是 §5 止损线要盯 argmax margin 的原因。
- **残留（不影响下一会话首次 GPU 验收）**：
  1. 输入仍是合成 f32（按 checkpoint 尺度规则量化），**未接真实 checkpoint 字节**；
     接法：把 `ph0::build_case` 的 W/A 换成读盘字节（fp4 + e8m0 布局）即可，其余全复用；
  2. smem 单级 staging 上限 `PH0_MAXBLK=8`（K=256）；真实 K=5120/1920 需 chunked staging 或
     动态 smem + `cudaFuncSetAttribute`（**图捕获内禁用**），属 Phase 1；
  3. `PERBLK` 需 4 TMEM 列/块 ⇒ 真实 K（160 块 = 640 列 > 512）装不下，只能用于消歧，不能上生产；
  4. N=8 的 8 行都填独立随机激活（比"1 token + 7 行零"覆盖更强；行间独立，真实零填充同代码路径）。

## 1c. Phase 1 骨架已落地（2026-09-12 代码已写；GPU parity 待跑）

落点：`kernels/cuda/dsv41_experts_mxf4.cu` **文件尾部**，`namespace tc5`，整块由
`#ifdef DSV41_TCGEN05_GATEUP_SKELETON` 包住 —— `build.sh` 不定义该宏，所以**不进 .so**；
单独编译验证：`nvcc -gencode arch=compute_103a,code=sm_103a -DDSV41_TCGEN05_GATEUP_SKELETON=1 -c`。

- 入口：`tc5::expert_tcgen05_gateup_kernel`（M=128 tile / N=8 / K_STEP=32 / kPackK=64 / kRing=3 /
  128 线程），launcher `tc5::tc5_launch_gateup` + `extern "C" dsv41_expert_tcgen05_gate_up`
  （env `DSV41_EXPERT_TCGEN05` 门禁，函数内 `static const` 读一次，默认 OFF）。
- **本地已闭环（无 GPU）**：sm_103a 编译过；ptxas `-v` = **101 regs / 0 spill / 46080 B static smem**；
  PTX 内含 `tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X`、
  `cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes`、
  `mbarrier.arrive.expect_tx`、`tcgen05.st.sync.aligned.32x32b.x1.b32`（x1 拼写 ptxas 接受）。
  默认构建（不定义宏）不受影响，仍 0 error。
- **骨架里 4 个新结论（超出 §2 原设计，实施时必须知道）**：
  1. **PACKED SF 在真实 K 下是被迫的、不是选择**：PERBLK 需 4 列/块 ⇒ 160 块 = 640 列 > 512 TMEM 列，
     装不下。所以 §1b 里 PACKED 假设的 GPU 判定从"两种布局哪个对"升级为**唯一可上线路径**，
     PACKED 挂 = 整个 Phase 1 布局假设挂（这是 §1c 最重要的一条）。
  2. **mxf8f6f4 的 fp4 操作数必须 UNPACKED（1 元素/字节）**，而 checkpoint 是 2 元素/字节 ⇒
     每个 ring slot 需要两块 buffer（TMA 落的 raw 打包区 + MMA 读的 unpack 操作数），
     并在中间做一次 1:2 展开。这就是本 kernel 最大的 smem 开销，也是"TMA 直接写操作数"不可能的原因。
     **Alternative B（已记为备选，未实现）**：改 `kind::mxf4`（两侧都打包）⇒ 无 unpack 段、操作数 smem 减半、
     MMA 是 `tests_tcgen05_mxf4.cu` 已数值验证的那条；代价是激活格式 e4m3→e2m1（放弃 Phase 0 买的精度余量），
     且 2X 粒度需把 checkpoint 的 per-32 scale 配对（现 kernel 已在做）。切换约 30 行。
  3. **静态 smem 装得下**（kPackK=64/kRing=3 时 45 KiB）⇒ 不需要 `cudaFuncSetAttribute`，
     **图捕获安全**；更深的 ring（更多 in-flight 字节）才需要动态 smem + init 期一次性的 attribute 设置。
  4. **occupancy 是硬约束**：grid=(30, slots) 且每 CTA 256/512 TMEM 列 ⇒ ≤2 CTA/SM；
     slots=1 时只有 30 CTA（≈15 SM）——**隔离微基准必须跑 slots=8（240 CTA）**，否则测的是延迟不是带宽。
     若 slots=8 仍离地板远，下一步是 **K-split**（同一份权重换更多 CTA 数），需配一个
     确定性升序 reduce（与 §3 down 的 ascending-slot 契约同款）。kernel 的 ring 不用改。
- **仍待填（骨架里已标 TODO-1..5 + K-SPLIT）**：真实 K 的 GPU parity（复用 Phase 0 harness + 真实
  checkpoint 字节）、`kPackK`/`kRing` 调优基准、2D tensor TMA（每 stage 1 条指令替代 129 条）、
  K-split reduce、Phase 2 的 pool/ids 间接寻址 + Rust FFI。

## 2. Phase 1（1-2 天）gateup kernel

落点：`kernels/cuda/dsv41_experts_mxf4.cu` 新增 swapAB gateup kernel（保留旧 kernel）。
**实际命名见 §1c**：`tc5::expert_tcgen05_gateup_kernel`（原计划名 `mxf8f6f4_swapab_gemm_kernel` 未采用，
骨架已按最终签名写好）。
- **A 侧（权重）**：fp4 smem descriptor。**必须补多级异步 staging**（TMA bulk / cp.async）——
  现 `mxf4_gemm_kernel:348-421` 全程 LDG→STS、无 cp.async/TMA（STATUS:5038 的 16.8GB/s 0.2%
  带宽地板根因）。swapAB 只解决 M 钉死，**不补 staging 仍是 0.2% 地板**。
- **B 侧（激活）**：e4m3 8×K staged；N=8（col0=token，其余 0）。
- **D**：TMEM 累加器；epilogue 用 `tcgen05.ld` 取回，套用现有 epi_mode（gate/up clamp `limit`，
  down 的 row_weight，`epi_mode==3` 累加）。
- **scale**：SFA（权重 per-row per-32 e8m0）、SFB（激活 per-32）经 `tcgen05.st` 写 TMEM；
  idesc 用 `make_idesc_mxf8f6f4(a_fmt=E2M1=5, b_fmt=E4M3=0, sf_id...)`。
- 形状契约沿用：launcher 只校验 `k % 32 == 0`；`n % 128 == 0` 需新增（M 侧 tile）。
- **验收**：隔离微基准 + Phase 0 同款 parity（真实权重）。

## 3. Phase 2（0.5-1 天）down kernel + launcher + Rust FFI

- down：同 kernel 模板，K=1920、M=5120，套 `expert_gemv_fp4_down_reduce_kernel` 的
  **ascending-slot 定点序**（fp 加法不结合，顺序即契约）或产 scratch 走 `dsv41_moe_down_reduce`。
- launcher：新增 `DSV41_EXPERT_TCGEN05` 门禁（**OnceLock 读一次**，否则破坏 graph capture），
  默认 OFF；旧 GEMV 路径保留为 fallback 与 A/B 基准。M=1 时**不能**再走 128x 冗余的旧 `mxf4_gemm_kernel`
  （见 `launch_mxf4:2109` 的注释史）。
- Rust FFI：`crates/ferrite-models/src/dsv41/kernels.rs` 新增符号 + `device.rs` 包装
  （参照 `expert_gate_up_fp4_batched` / `expert_down_reduce_fp4_batched`），
  `chain_dev.rs` 的 `moe_batch()` 分支按门禁派发；batched 形状（grid.y=slot）必须保持 disjoint 输出。
- **验收**：`cargo check` + 符号在 `.so` 中存在（`supports_*` 版本探测，旧 .so 回退）。

## 4. Phase 3（0.5 天）serve A/B + 数值 parity + 四段文本

- 走 `scripts/dsv41_swapab_text_parity.sh` 同款：**两次独立启动**（gate 每进程只读一次），
  臂 = `DSV41_EXPERT_TCGEN05=0 / =1`。
- 判据（硬线，全部满足）：
  1. 四段文本（Paris / 静夜思 / 1+1=2 / 出师表）**正文逐字相同**（首 token 可翻，仅当正文相同）；
  2. `faults=0`；3. p50 下降（预期 6.17 → ~5.2ms，~190+ tok/s；若真达 3.0ms 则 ~330）。
- 单测：新增 `expert_tcgen05_parity.rs`（CPU golden + vs SIMT + argmax margin），无 checkpoint 可跑。

## 5. 风险缓解（按 6 条 serve-faithful 协议）

STATUS:7713 定案的 6 条，隔离评估必须全部满足，否则数据无效：
1. **生产构型**（含已 ON 的前置优化：ILV / W2 prewarm / DOWN_FUSE / gateup_fuse）；
2. **图模式 capture+replay** 计时（禁 direct-launch 计时）；
3. **侧流干扰注入**（复现 4 流并行的 SM 争抢）；
4. **L2 污染**（每轮换缓冲，禁驻留红利）；
5. **关键路径计费**（gate/fork kernel 按 `fork_ev` 语义）；
6. **判据分层**：隔离只做**淘汰**；**正向收益一律 serve A/B + 四段文本 + faults=0**。

**额外的 expert 专属风险**：
- ⚠️ **翻译风险最高的一类（第 7 类系统差异）**：swapAB 已在此处失败（隔离 1.76-1.94x → serve 中性）。
  **止损门**：若隔离微基准达标但 serve A/B 中性（Δ<0.05ms），**立即关闭路径、默认 OFF**，
  不再投入变体矩阵（勿重演 v17→v21 四变体全中性的浪费）。
- ⚠️ **fast-math 重结合**：`--use_fast_math` 默认 ON（build.sh:42），任何多点累加必须
  `__fadd_rn/__fmul_rn/fmaf` 钉死，否则 ~1ULP/层漂移、40 层后模型坍缩（已发生过）。
- ⚠️ **图捕获禁区**：门禁不可在 kernel 内/env 每调用读；`cudaFuncSetAttribute`/动态 smem
  在捕获内会 err 900/901（STATUS:916）。TMA descriptor 必须在捕获外构造。
- ⚠️ **err 716 / 16B 对齐**：smem descriptor 的 LBO/SBO 与 cp.async 需 16B 对齐；写错静默错值。
- ⚠️ **N=8 浪费**：仅激活侧，可接受；但别误把 topk 个专家塞进 N（A/权重不同，不可共享）。

## 6. 里程碑

| 里程碑 | 交付 | 工时 | 通过判据 |
|---|---|---|---|
| M0 | 探针扩展 | 0.5d | GPU parity 全绿 |
| M1 | gateup tcgen05 kernel | 1-2d | 隔离微基准 + parity |
| M2 | down + launcher + FFI | 0.5-1d | cargo check + 符号探测 |
| M3 | serve A/B | 0.5d | 四段文本 + faults=0 + p50 |
| **Gate** | 去留决策 | — | serve 中性 ⇒ 关闭默认 OFF；否则转默认 ON |

## 7. 建议改动点（按顺序）

1. `kernels/cuda/tests_tcgen05_mxf8f6f4_1x.cu` —— Phase 0 扩展（真实权重 + parity）。
2. `kernels/cuda/dsv41_experts_mxf4.cu` —— 新增 swapAB gateup/down kernel + launcher 门禁。
3. `kernels/cuda/build.sh` —— 无需改（新 kernel 同 TU，自动进 `.so` + build_id 哈希）。
4. `crates/ferrite-models/src/dsv41/kernels.rs` / `device.rs` —— FFI + 版本探测。
5. `crates/ferrite-models/src/dsv41/chain_dev.rs` —— `moe_batch()` 派发。
6. `crates/ferrite-dsv41/tests/expert_tcgen05_parity.rs` —— 新增单测。
