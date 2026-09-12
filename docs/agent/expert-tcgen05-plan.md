# expert tcgen05 fp4 swapAB 实施计划（下一会话）

> 前置：本会话 `tcgen05-probe` 已用 `kernels/cuda/tests_tcgen05_mxf8f6f4_1x.cu`
> 在 `-gencode arch=compute_103a,code=sm_103a` 下 ptxas 验收通过
> `tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X`，操作数布局确认：
> A=fp4 smem descriptor，B=e4m3，D=TMEM，SFA/SFB=TMEM e8m0。使用（不重证）。
> 目标：expert 2.0ms/步 → ~1.0ms/步 ⇒ 步时 ~3.0ms ≈ 300+ tok/s（**唯一路径**，4-5 人日）。
> 代码基线：`kernels/cuda/dsv41_experts_mxf4.cu`（现有 `mxf4_gemm_kernel` / `expert_gemv_fp4*`）。

> 🔴 **2026-09-12 修订（本文件当前口径，覆盖以下各节原本的 mxf8f6f4 假设）**
> **Phase 1 默认 = `kind::mxf4`**（A/B 两侧 packed，E2M1×E2M1，ue8m0，`scale_vec::2X`）；
> **`mxf8f6f4`（`scale_vec::1X`，e4m3 激活）降为备选臂**。依据与代价见新增 **§1e**。
>
> ⚠️ 本轮流传的"定案"里有两处**与代码不符**，本文件按代码订正（逐条 `file:line` 见 §1e）：
> ① 参考实现的 fp4-weight `linear()` 激活是 **FP8 e4m3，不是 fp4 e2m1**
>    （`ref_inference/model.py:187` 调的 `act_quant` 产出 `float8_e4m3fn`；`kernel.py:111,490,511`）。
>    "激活本来就是 e2m1"说的是**现生产 ferrite**，不是参考。
> ② 因此 mxf4 相对参考**不是零损失**，而是"沿用既有 e2m1 偏差"；"零"只成立于
>    **相对现生产 162.6 tok/s 基线**（同一套 `dsv41_quant_fp4`）。另："smem 减半 ⇒ 更多 CTA/SM"
>    不成立——occupancy 由 TMEM 256 列/CTA 钉死，真收益是 **ring 深度**（§1d/§1e）。

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
     **Alternative B**：改 `kind::mxf4`（两侧都打包）⇒ 无 unpack 段、操作数 smem 减半、
     MMA 是 `tests_tcgen05_mxf4.cu` 已数值验证的那条；代价是激活格式 e4m3→e2m1（放弃 Phase 0 买的精度余量），
     且 2X 粒度需把 checkpoint 的 per-32 scale 配对（现 kernel 已在做）。切换约 30 行。
     🔴 **§1e 已把 Alternative B 升为 Phase 1 默认路径**：本条描述的 UNPACKED 双 buffer 开销
     是 **mxf8f6f4 臂（现为备选）的固有成本**，不再是默认路径的成本。
  3. **静态 smem 装得下**（kPackK=64/kRing=3 时 45 KiB）⇒ 不需要 `cudaFuncSetAttribute`，
     **图捕获安全**；更深的 ring（更多 in-flight 字节）才需要动态 smem + init 期一次性的 attribute 设置。
  4. **occupancy 是硬约束**：grid=(30, slots) 且每 CTA 256/512 TMEM 列 ⇒ ≤2 CTA/SM；
     slots=1 时只有 30 CTA（≈15 SM）——**隔离微基准必须跑 slots=8（240 CTA）**，否则测的是延迟不是带宽。
     若 slots=8 仍离地板远，下一步是 **K-split**（同一份权重换更多 CTA 数），需配一个
     确定性升序 reduce（与 §3 down 的 ascending-slot 契约同款）。kernel 的 ring 不用改。
- **仍待填（骨架里已标 TODO-1..5 + K-SPLIT）**：真实 K 的 GPU parity（复用 Phase 0 harness + 真实
  checkpoint 字节）、`kPackK`/`kRing` 调优基准、2D tensor TMA（每 stage 1 条指令替代 129 条）、
  K-split reduce、Phase 2 的 pool/ids 间接寻址 + Rust FFI。

## 1d. Alternative B（`kind::mxf4`）量化分析（2026-09-12，纯代码分析，无 GPU）

**数值判定（激活格式检查）**
- **参考实现的 expert 激活是 e4m3**，不是 fp4：`ref_inference/model.py:186-198` 的 `linear()`
  对 fp4 权重走 `act_quant(x, fp8_block_size=32)`，`kernel.py:537-540` 再把 fp4 权重 upcast 成
  FP8 做 fp8 MMA。⇒ Phase 0 选 e4m3 是**对齐参考**，不是自创。
- **但现生产 ferrite 是 e2m1 激活**：`chain_dev.rs:4217` `quant_fp4`（e2m1, block 32）→
  `expert_gate_up_fp4_batched`（`kind::mxf4`，E2M1×E2M1，`kernels.rs:110-127` /
  `dsv41_experts_mxf4.cu:13-18,256-273`）。⇒ **Alternative B 的激活格式与今天 162.6 tok/s 的
  serve 基线逐位同源（同一个 `dsv41_quant_fp4`、同一 block-32 e8m0）⇒ 零格式增量风险**；
  相对参考是"沿用既有偏差"（现状只被四段**短**提示验证过，长文残余风险非零）。
- e2m1 码表是 **8 个幅值** `{0,±0.5,±1,±1.5,±2,±3,±4,±6}`（`quant.rs::FP4_TABLE`、
  `tests_tcgen05_mxf4.cu:26`），不是 4 个。最坏相对误差 **20%**（4/6 中点），不是 25-50%；
  e8m0 幂次 block 缩放后块内绝对误差 ≤ amax/6；§1b 实测合成数据 l1/l2 rel ≈ 0.11-0.16。
- e2m1×e2m1 积 + 2^e scale 在 fp32 **精确** ⇒ `tests_tcgen05_mxf4.cu` GPU 上
  `maxdiff=0.000e+00`（7 shapes，`crates/ferrite-dsv41/README.md:94-101`）。**Alt B 用的 MMA 是
  唯一已在 GPU 数值验证过的形式**；1X 的 PACKED SF 布局（§1c #1）至今没有 GPU parity。

**性能账（kPackK=64/kRing=3）**
- 每 slot 操作数：Alt A `a_raw 4096 + a_op 8192 + b_raw 512 + b_op 512 = 13312 B`
  → Alt B `4096 + 256 = 4352 B`（**3.06x**，不是 2x）。静态 smem 46.4 KiB → ~19.5 KiB
  （`sf_stage` 6400 B 不变）。
- ⚠️ **"smem 减半 ⇒ occupancy 翻倍"不成立**：绑定点是 **TMEM 256 列/CTA**
  （`dsv41_experts_mxf4.cu:2834,3141-3143`）⇒ 仍 2 CTA/SM。要翻 occupancy 必须把 SF 改成
  **per-slot staging**（照生产 kernel `:437-514` 按 K stage 复用；1X/2X 的 SFA 列密度相同：
  每 K=128 用 4 列，两个 atom 共享一个 32-bit 字，`SFA_ID=0/2`）。
- 真收益是 **ring 深度**：同一个 48 KiB 窗口 kRing 3 → 7-8 ⇒ in-flight 8 KiB → 28-32 KiB/CTA
  ⇒ 240 CTA 从 1.9 MB 到 **6.7 MB**，逼近 8 TB/s × ~700 ns ≈ 5.6 MB 的带宽-延迟积
  （骨架 DEPTH NOTE）。这才是打 0.2% 地板的杠杆。
- ⚠️ 更正"无 unpack 段"：**permute 段落删不掉**。canonical UMMA 布局
  `unit16(m,kb)=(m%8)+8kb+16(m/8)`（`:249-254`）与 raw packed 行主序（chunk=2m+kb）不兼容，
  描述符的 (m%8) stride 固定为 1（CUTLASS `((8,n),(2,1)):((1,SBO),LBO)`）。Alt B 只把
  `16B in→32B out + byte_perm` 减成 `16B in→16B out`。⇒ "TMA 直写操作数"对 **1D bulk 仍不成立**
  （TODO-3 的 2D tensormap 只解决 issue storm，不解决布局）。

**推荐路径**：B 是"首个绿灯风险最低"的臂（唯一 GPU 验证过的 MMA + 与现状格式同源），A 的 1X PACKED
SF 布局未验证且 §1c #1 明确"PACKED 挂 = Phase 1 全挂"。做法：同一 skeleton 用编译期宏切两臂
（idesc 格式 / LBO-SBO / ring 尺寸 / 去掉 `tc5_unpack_a`，~30 行），共享 ring+TMA+occupancy 机制；
Phase 0 harness 把激活量化换成 e2m1（1 行）即可出 B 的 parity；**终门仍是 serve 四段文本**。

## 1e. 定案（2026-09-12 修订）：Phase 1 默认 = `kind::mxf4`

**结论**：Phase 1 的 gateup/down swapAB kernel **默认用 `kind::mxf4`**（A/B 两侧 packed fp4 e2m1，
`scale_vec::2X`，ue8m0）；**`mxf8f6f4`（`scale_vec::1X`，e4m3 激活）为备选臂**。两臂用编译期宏切换
（idesc 格式 / LBO-SBO / ring 尺寸 / 去掉 unpack），共享 ring+TMA+occupancy 机制（§1d 末段）。

### 1e.1 订正：流传"定案"中与代码不符的表述

| 流传说法 | 代码事实 | 出处 |
|---|---|---|
| 参考实现的 linear 激活就是 fp4 e2m1 | **是 FP8 e4m3**。fp4 权重分支走 `act_quant(x, fp8_block_size=32)`，`act_quant` 分配的输出是 `torch.float8_e4m3fn`；`fp4_gemm` 的 A 形参即 FP8，docstring 写明 "FP8 act x FP4 weight" | `ref_inference/model.py:186-195`；`ref_inference/kernel.py:111, 490-493, 511` |
| mxf4「零数值损失」 | 相对**参考**是有损（e4m3→e2m1，即放弃 Phase 0 买的精度余量）；"零"仅成立于**相对现生产 162.6 tok/s 基线**——生产本来就是 e2m1/block-32/ue8m0，且是同一个 `dsv41_quant_fp4` | `kernels.rs:110-127`；`chain_dev.rs:4217`；`dsv41_experts_mxf4.cu:14` |
| smem 减半 ⇒ 更多 CTA/SM | **不成立**。绑定点是 **TMEM 256 列/CTA** ⇒ 仍 ≤2 CTA/SM；真收益是 **ring 深度 kRing 3→7-8** | §1d；`dsv41_experts_mxf4.cu:2834,3141-3143` |
| 无 unpack 段 | **部分成立**：1:2 **展开**段可去掉，但 **permute 段删不掉**（canonical UMMA 布局 `unit16(m,kb)=(m%8)+8kb+16(m/8)` 与 raw packed 行主序不兼容，描述符 (m%8) stride 固定为 1） | `dsv41_experts_mxf4.cu:249-254`；§1d |

> 注：参考实现里确实存在 fp4 激活，但在 **attention 路径**（`model.py:546,552` 的 `fp4_act_quant` 用于
> q/k），**不是 expert/linear 路径**。expert 的 `linear()` 走的是 e4m3（上表第 1 行）。

### 1e.2 mxf4 成为默认的（可验证的）理由 —— 不是"零损失"

1. **唯一已在 GPU 上数值验证过的 fp4 tcgen05 形式**：`tests_tcgen05_mxf4.cu` 报
   `maxdiff = 0.000e+00`（EXACT，7 shapes，含两个 ABI 入口 gate/up 与 down），
   `crates/ferrite-dsv41/README.md:94-101`。而 A 臂（mxf8f6f4）依赖的 **1X PACKED SF 布局
   （§1c #1）至今没有 GPU parity**。
2. **与现生产 serve 基线逐位同源**：激活 = `dsv41_quant_fp4` 的 e2m1/block-32/ue8m0，
   ⇒ serve A/B 的变量被干净地隔离成**单一变量"staging（kRing TMA）"**，而不是"格式 + staging"两个。
3. **操作数 smem 3.06x**（每 slot 13312 → 4352 B，§1d）⇒ 静态 46.4 → ~19.5 KiB，同一 48 KiB 窗口
   kRing 3 → 7-8 ⇒ 240 CTA 的 in-flight 1.9 MB → **6.7 MB**，逼近 8 TB/s × ~700 ns 的带宽-延迟积。
   **这才是打 0.2% 地板的杠杆**（第 2 条把这条杠杆的读数变干净）。
4. **少一段代码、少一类失败模式**：无 1:2 展开 ⇒ 无 raw/operand 双 buffer（§1c #2 的固有成本只属 A 臂）。

### 1e.3 代价（诚实列出）

- 相对**参考**是精度降级（e4m3 → e2m1），属"沿用既有偏差"而非新增；生产仅被**四段短文本**验过，
  **长文残余风险非零** ⇒ 终门仍是 §4 的四段文本 + `faults=0`。
- mxf4 atom 是 **K=64 / `scale_vec::2X`**（scale 粒度 64），checkpoint 的 per-32 scale 需**两两配对**；
  现 `mxf4_gemm_kernel` 已在做（`dsv41_experts_mxf4.cu:430`），但**这是必须保住的契约**。
- A 臂仍是"对齐参考"的那条：若 B 臂在 serve 四段文本上翻车（长文数值），**回退 A 臂并补跑其 Phase 0 GPU parity**。

### 1e.4 两臂与 Phase 0 harness 的对应

| 臂 | Phase 0 harness | 激活 | MMA | 状态 |
|---|---|---|---|---|
| **B（默认）** | `tests_tcgen05_mxf4.cu` | e2m1 | `kind::mxf4.block_scale.scale_vec::2X` | **GPU 已验 EXACT** |
| A（备选） | `tests_tcgen05_mxf8f6f4_1x.cu` | e4m3 | `kind::mxf8f6f4.block_scale.scale_vec::1X` | 本会话 host 闭环；**GPU parity 待跑** |

⇒ Phase 0 的**第一件事**从"A 臂 GPU parity"改判为"**确认 B 臂的 `tests_tcgen05_mxf4.cu` 在当前
checkpoint 尺度规则下仍全绿**"（工作量更小，且是唯一已有 GPU 证据的路径）。

## 1f. mxf4 臂骨架已落地（2026-09-12；GPU parity 待跑）

落点：**同一文件同一 TU**，`kernels/cuda/dsv41_experts_mxf4.cu` 尾部（mxf8f6f4 块之后），
`namespace tc5::mxf4`，整块由 `#ifdef DSV41_TCGEN05_GATEUP_MXF4_SKELETON` 包住 —— `build.sh` 不定义该宏，
**不进 .so**。为 `namespace tc5::mxf4` 嵌套而非另起命名空间：两臂常量同名，嵌套保证 `-D` 两个宏同时开
也能编译（已验，0 error）。

- 入口：`tc5::mxf4::expert_tcgen05_gateup_mxf4_kernel`（M=128 / N=8 / **kKStep=64** / kPackK=64 /
  **kRing=8** / kNStep=1 / 128 线程）；launcher `tc5::mxf4::m4_launch_gateup` +
  `extern "C" dsv41_expert_tcgen05_gate_up_mxf4`（env `DSV41_EXPERT_TCGEN05_MXF4` 门禁，
  函数内 `static const` 读一次，默认 OFF）。
- **本地已闭环（无 GPU）**：sm_103a 编译过；ptxas `-v` = **92 regs / 0 spill / 40960 B static smem**；
  PTX 含 `tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X`、`cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes`、
  `mbarrier.arrive.expect_tx`、`tcgen05.st.sync.aligned.32x32b.x1.b32`。
  4 种宏组合（无宏 / A / B / A+B）全 0 error；默认构建不受影响。
- 编译验证命令：`nvcc -gencode arch=compute_103a,code=sm_103a -O3 -std=c++17 \
  -DDSV41_TCGEN05_GATEUP_MXF4_SKELETON=1 -c kernels/cuda/dsv41_experts_mxf4.cu -o /tmp/t5m4.o -Xptxas -v`
- **两臂的差异被压缩到 4 处**（其余结构逐行同形，serve A/B 因此只隔离"staging 深度"一个变量）：
  1. 操作数格式与 smem：PACKED 2/byte、**无 raw buffer**、无 1:2 展开；slot 13312 → **4352 B**
     （a_op 4096 + b_op 256）；kRing 3 → 8（40 KiB / 48 KiB 窗口）。
  2. MMA + idesc：`kind::mxf4...scale_vec::2X`，`a_format = b_format = 1`（MXF4Format::E2M1）。
     N=8/sf_id=0 的 idesc = **0x08820480**（A 臂的对偶断言是 0x08820280，只差两个 format 字段）。
  3. K 原子 32 → 64；2X 下 32-bit SF 字由**两个 atom 共享**（bytes[0,1]=偶 atom, bytes[2,3]=奇 atom），
     idesc 的 SFA_ID/SFB_ID 从 `b & 3` 变为 `2 * (a & 1)`。
  4. 激活指针语义：`[dim]` e4m3 → `[dim/2]` packed e2m1。
- **不变的（重点）**：SF 列密度与 TMEM 预算完全一致（SFA 每 K=128 用 4 列、SFB 每 pair 1 列 ⇒
  8+160+40 = 208 ≤ 256），所以 **occupancy 不变（仍 ≤2 CTA/SM）**；ring/mbarrier/epilogue/launcher
  契约同形。⇒ 3.06x 的收益**只**兑现在 ring 深度（in-flight 8 KiB → 28 KiB/CTA，240 CTA 达 6.7 MB）。
- **permute 段的真实形态（订正本文档早前的措辞）**：mxf4 臂**没有** raw staging buffer，
  "permute" 由 `m4_off(row, kb) = 16*((row & 7) + 8*kb + 16*(row >> 3))` 把每个 **16 字节**
  raw chunk 直接 TMA 到它的 canonical unit —— 因为 packed fp4 的 raw chunk 与 canonical unit 都是
  16 B，**两者同尺寸**，所以布局变换退化成"换目的地址"，这是 4352 B/slot 的来源。
  代价：每 stage 的 TMA issue 数从 129（A 臂，32 B/行）涨到 **258（16 B/块）**；`[TODO-3]` 的 2D
  tensor 形式（8 行 × 16 B 的 box → 一条指令填一个 canonical core matrix，32 条/stage）是解药。
  ⚠️ 16 B 对齐因此成为**硬契约**：`wp + row*(dim/2) + k0/2 + 16*kb` 与 `act + k0/2 + 16*kb`
  都必须 16 B 对齐（`dim % 128 == 0` ⇒ `dim/2 % 64 == 0`）。
- **2X 配对是正确性契约**：一个 32-bit SF 字跨 4 个 block = 128 K 元素 ⇒ launcher 硬拒 `dim % 128 != 0`。
- 骨架内仍留 `TODO-3`（2D tensor TMA）与 `K-SPLIT`（occupancy 是绑定点，见 §1c #4）。

## 2. Phase 1（1-2 天）gateup kernel

落点：`kernels/cuda/dsv41_experts_mxf4.cu` 新增 swapAB gateup kernel（保留旧 kernel）。
**实际命名见 §1c**：`tc5::expert_tcgen05_gateup_kernel`（原计划名 `mxf8f6f4_swapab_gemm_kernel` 未采用，
骨架已按最终签名写好）。
- **A/B 两侧都 packed**（🔴 §1e 定案：默认 `kind::mxf4`）；
  **必须补多级异步 staging**（TMA bulk / cp.async）——
  现 `mxf4_gemm_kernel:348-421` 全程 LDG→STS、无 cp.async/TMA（STATUS:5038 的 16.8GB/s 0.2%
  带宽地板根因）。swapAB 只解决 M 钉死，**不补 staging 仍是 0.2% 地板**。
- **B 侧（激活）**：fp4 e2m1 8×K staged（与现生产 `dsv41_quant_fp4` 同源）；N=8（col0=token，其余 0；
  `kind::mxf4` 的 N 合法域为 [8,256] step 8，见 `dsv41_experts_mxf4.cu:14-18`）。
- **D**：TMEM 累加器；epilogue 用 `tcgen05.ld` 取回，套用现有 epi_mode（gate/up clamp `limit`，
  down 的 row_weight，`epi_mode==3` 累加）。
- **scale**：SFA（权重 per-row per-32 e8m0）、SFB（激活 per-32）经 `tcgen05.st` 写 TMEM；
  idesc 用 `make_idesc_mxf4(n_dim, a_sf_id, b_sf_id)`（两侧 `E2M1`，`scale_format=UE8M0`，
  见 `dsv41_experts_mxf4.cu:256-273`）。mxf4 为 **`scale_vec::2X`（粒度 64）** ⇒ 每 K=64 atom
  吃两个 per-32 scale 字（`dsv41_experts_mxf4.cu:430` 已是此契约）。
- 形状契约沿用：launcher 只校验 `k % 32 == 0`；`n % 128 == 0` 需新增（M 侧 tile）。
- **验收**：隔离微基准 + Phase 0 同款 parity（真实权重）。

### 2b. Phase 1.5（2026-09-12）**3 个 ABI 缺口已修复，serve 派发已接通**（默认 OFF）

已落地（`cargo check -p ferrite-models` 通过）：
- `kernels.rs:171` `dsv41_expert_tcgen05_gate_up_mxf4`（**18 参数**，见下）；
- `device.rs` `Kernels.expert_tcgen05_gate_up_mxf4`（**`ko!` 可选符号**：`build.sh` 不定义
  `DSV41_TCGEN05_GATEUP_MXF4_SKELETON` ⇒ 现成 `.so` 无此符号，用 `km!` 会让 `Device::open`
  对所有用户失败）+ 包装 `Device::expert_tcgen05_gate_up_mxf4`（`rc==0` ⇒ `Ok(false)`；
  **形状/对齐被拒时返回非零 ⇒ 硬错误**，调用方必须先自检契约）+ 探测
  `supports_expert_tcgen05_mxf4()`；
- `chain_dev.rs:307` `expert_tcgen05_mxf4()` OnceLock 门禁（镜像 `.cu` 的 `e[0]=='1'`，
  默认 OFF），并在 `moe()` 的 routed 分支放**一次性告警**（`tcgen05_mxf4_skipped_note`）——
  只在该门禁被 arm 但派发被拒（.so 无符号 / ILV / 无 batched 路径）时触发，
  避免"门禁 ON 但实际跑旧路径"的测量偏差（本仓 #1 失效模式）。

✅ **3 个 ABI 缺口的最终修复（2026-09-12，设计见 commit f9b3f9c）**：
1. **权重布局 + expert 间接寻址（缺口 1+3 合并）**：kernel 参数从"单 `w`/`w_scale` +
   `w_base/ws_base` 间接对"改为**四组 base/stride**（`w1/w1s/w3/w3s`），行寻址
   `row < split ? w1p[row] : w3p[row - split]`（`dsv41_experts_mxf4.cu:3803-3813` 的 `issue`，
   参考生产已验证的 `mxf4_gemm_kernel:626-632` 的 `b`/`b_hi` + `b_split`）；SF prologue 同样
   32 行一组双指针（`:3865-3874`）。`ids != nullptr` 时四个指针按 `base + ids[slot]*stride`
   逐 slot 推导；`ids == nullptr` 时四个 base **就是**直接指针（parity harness 形式），
   因此不再需要单独的直接指针参数。
2. **激活 scale（缺口 2）**：B-scale prologue（`:3892-3906`）改为读 4 个 f32 →
   `f_pow2_to_ue8m0`（`:132`，旧 SIMT 路径同款）→ 拼 32-bit SF 字。`dsv41_quant_fp4` 的 ABI
   与旧 SIMT 路径**零改动**；`round_scale=true` 时该转换无损（scale 本身就是 2^n）。
3. launcher（`:4037`）新契约：`dim % 128 == 0`、`rows = 2*inter`、`split % 32 == 0`，
   以及**四组 base/stride 的 16B 对齐硬校验**（含 `w3`——双指针后它同样过 TMA；
   错值不 fault，所以必须显式拒绝）。`chain_dev.rs:4308+` 按门禁派发，`ran_tc` 为真时
   **强制关闭** gateup+swiglu 融合（tcgen05 epilogue 只做 clamp，输出 unfused `[2*inter]`），
   因此后续 `swiglu_limit_batched` 与 down 的 `act_slot = 2*inter_local` 都与旧 unfused 路径一致。

**验证**：`cargo check -p ferrite-models` ✅；`nvcc -arch=sm_103a -DDSV41_TCGEN05_GATEUP_MXF4_SKELETON=1`
编译 ✅（0 error，`expert_tcgen05_gateup_mxf4_kernel` 0 spill / 92 reg / 40960 B smem，
与 SMEM 注释一致）；harness（`#include dsv41_experts_mxf4.cu`）编译 ✅。
**未做**：GPU parity（需 B300）与 serve A/B——见 §4/§5。

⚠️ **仍存在的风险**：`w3` 现在与 `w1` 同样受 16B 契约约束，而 feed 它的 loader 平面偏移
（`load.rs:650-665`）尚未被独立验证过对齐（launcher 会拒绝而不是静默错值）；`split` 落在
128 行 tile 中间的情形（`inter % 128 != 0`）已由**逐行**选择覆盖，但尚无 parity 用例。

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
| M0 | 探针：**先确认 B 臂 `tests_tcgen05_mxf4.cu` 全绿**；再扩 A 臂探针 | 0.5d | GPU parity 全绿 |
| M1 | gateup tcgen05 kernel（**默认 `kind::mxf4`**） | 1-2d | 隔离微基准 + parity |
| M2 | down + launcher + FFI | 0.5-1d | cargo check + 符号探测 |
| M3 | serve A/B | 0.5d | 四段文本 + faults=0 + p50 |
| **Gate** | 去留决策 | — | serve 中性 ⇒ 关闭默认 OFF；否则转默认 ON |

## 7. 建议改动点（按顺序）

1. `kernels/cuda/tests_tcgen05_mxf4.cu` —— **B 臂（默认）探针**，先确认在当前 checkpoint 尺度规则下全绿。
   备选臂保留 `kernels/cuda/tests_tcgen05_mxf8f6f4_1x.cu` —— Phase 0 扩展（真实权重 + parity）。
2. `kernels/cuda/dsv41_experts_mxf4.cu` —— 新增 swapAB gateup/down kernel + launcher 门禁。
3. `kernels/cuda/build.sh` —— 无需改（新 kernel 同 TU，自动进 `.so` + build_id 哈希）。
4. `crates/ferrite-models/src/dsv41/kernels.rs` / `device.rs` —— FFI + 版本探测。
5. `crates/ferrite-models/src/dsv41/chain_dev.rs` —— `moe_batch()` 派发。
6. `crates/ferrite-dsv41/tests/expert_tcgen05_parity.rs` —— 新增单测。
