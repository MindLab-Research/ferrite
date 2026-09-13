# 【索引】本文件已 1300+ 行 / §1–§49 —— 先读这几节

> **最新结论（2026-09-14 深夜）**：fp4 操作数的真实 smem 语义已**定谳**（§47）：
> **packed 数据（2 元素/字节）放在 16 B 容器里、硬件只读每槽前 8 B** ⇒ 每行 footprint 128 B、每 stage 16384 B。
> 写公式 `hw_pack_sw128()`（§47），**描述符/递进/idesc 全部保持官方原值**；实测 **relerr = 0 精确 PASS**。
> 此前 §29("必须 packed")与 §43("官方是 unpacked")**不是矛盾**：§47 调和（container footprint vs 容器内数据）。

## 阅读路径
| 想知道什么 | 读哪节 |
|---|---|
| **最终结论与修法** | **§47**（语义定谳 + 写公式 + 实测对照表）→ §48（足迹核验）→ §49（nibble/尺寸三方一致性） |
| 官方权威参照（本机已 PASS 的最小参考） | **§43**（权威约定提取表 + 生成码路径）→ §44（逐项核对：参数面已排除） |
| 6 个已修的真 bug（idesc/布局/K递进/4 处栅栏） | §10–§12 附近（另见 AGENTS.md 的一行摘要） |
| **诊断/运维踩过的坑（先看这些，能省整个窗口）** | §16（NUMCHECK 需关图）、§18（微基准不可信）、§22（编译检查要在仓库目录编 shim）、§25（build-id 不匹配拒启）、**§35（5 个图门都要关）**、§36（`git apply --3way` 假成功）、§42（诊断臂 vs 性能臂口径） |
| 已被**排除**的怀疑（别再重查） | §44/§45/§46（参数面、SF 行对应与角色、mbarrier parity） |
| 精度对齐（用户硬性要求） | §14/§21（两处不对齐 + 具体修法）、§39（验收方案：DBG 五点回读） |
| 性能与 400 目标 | §39（packed 顺带红利）、`mtp-verify-amortization-model.md`、perf 系列文档 |
| 别的编码路线（保留作对照，均 FAIL） | §30–§34/§38/§40/§41（packed 几何候选 B/D/E 与自洽性分析） |
| **修复的验证链**（实现 ↔ 公式 ↔ 硬件行为） | §48（足迹算术）、§49（nibble 序 + 子 tile 尺寸三方一致）、**§54（我方实现的独立算术核验）** |
| **精度对齐的后续** | §50（投影层 fp8 也是 block 32 ✓）、§53（补丁 OFF 路径逐行安全核实）、§55（验收判据机械化 `wq_check.py`） |
| **性能与 400 的量化现实** | **§52**（8.87ms 口径裁决 = plain decode 的 `eager(1)`；440/400 算术**紧**：只靠摊薄 ⇒ ~10.8ms > 9.8ms 门槛 ⇒ **必须靠 tilelang/mma 把 verify 压到低于自私单行**） |
| **TileLang 路径复活** | §57（err700 很可能是**错误闩存**、非本 kernel；官方 TMA 自带 16 B 容器展开 ⇒ fp4 部分**无需改**；真风险 = m>1 的三段流水线，需 GPU 验） |
| **e2e 仍错的排查（主线）** | §56（F 回合：kernel 已精确但 e2e 仍错）+§65（定谳：**我的仪器化回归 M1 让 BS 臂整体失效** + `goto` 跳探针 M2） |
| **接线缺陷与审计** | §62（**swapAB 输出打包错位：修复前 512/640 位置错**）、§65（六项映射 OK）、§69（**差异法十项全 SAME** + D3/D4/D5 已修） |
| **精度：实现与验证** | §61（8 缺陷）、§66（四项语义交叉确证）、§70（合入台账 + 合并方法教训）、**§71（单测 12/12 全绿）**、§72（关闭 engram/vision + 剩余累加序项） |
| **精度：配置与转正** | §63/§64（出货脚本未开 `DSV41_ROUTED_DOWN_QUANT`）、§58（DBG 也受图捕获守卫）、`~/promote_precision.sh` |
| **`[NC]` 仪器为何拿不到数** | **§73**（整步图/MoE 图的 capture 录制期覆盖了 BS 调用；⇒ 判定改用 `~/wq_check.py` + 差异测试 `~/bs_vs_old.sh`） |
| **`[NC]`/DBG 仪器的可达性** | **§73**（图录制期必然捕获）、**§79**（五门 DBG 是"天然延迟型"⇒ 推广窗口可直跑） |
| **精度五门的验证与安全** | §75（I3 顺序语义闭环）、§76（五门无半挂）、§78（OFF 路径逐门核实）、§80（两项数值件主 agent 亲验） |
| **累加序（最后一项）** | **§77**（10 项 CPU 即判无害 / 4 项抽检 / 3 项需 GPU；**风险唯一落点 = head logits**） |
| **head logits（C 档风险落点）** | **§81**（真差异是**输入边界**而非序差：边界项 1.95e-2 vs 序差 1.2e-4，差 162~560 倍；我方默认"精度偏高"）+ §82（三处调用点、外层是融合门） |
| **流程事故（必读）** | **§83**（合并引入重复定义 ⇒ **完整构建失败**，单文件检查漏掉 ⇒ 那一轮 e2e 全废；含普查确认唯一）+ AGENTS.md 的「合并纪律」 |
| **下一步入口** | 顶部「下一步」+ 本文件末节的 TODO 快照 |

## 三个"别再犯"的方法论
1. **不要凭"某一族配置全失败"下硬件语义结论** —— 必须有 (a) 正向存在一个**精确 PASS** 的配置，
   且 (b) **原始 smem 偏移级的直接观测**（§47 的做法）。
2. **`const` 用例对 K-block 递进是盲的**；定标必须用**稠密随机** + `sfprobe`（§47 备注）。
3. **文本相似度不是判据**（"更像计数"≠ 更接近正确）；判据是 `[NC]` 数值或精确 parity（§40/§47）。

---

# fp4 MoE BS illegal memory access 完整排查记录（2026-09-14）

> 状态：排查中（no-TL-proj 判别实验在跑）。单元测试已证明 **BS 臂完全正常**（m=1/5/6 全 PASS），crash 是模型执行上下文的交互问题。

## 症状

- `moe_bs_up_tl_kernel`（TileLang 生成的 blockscaled fp4 MoE up-GEMM）在 B300 sm_103a 上运行时 **illegal memory access**（cuda 700）
- 8/8 ranks 全部 crash
- ARMED 回执正常打印（INIT 成功），crash 发生在 kernel 执行时
- MMA skip 测试（跳过 MMA launch）**不 crash** → crash 在 MMA kernel 内

## 与 JIT 的对照（关键证据）

| 维度 | JIT（不 crash） | AOT（crash） |
|------|-----------------|--------------|
| 源码 | `~/.tilelang/cache/.../device_kernel.cu` | `moe_bs_up_tl.cu`（diff 仅差手加的 relinquish asm） |
| 模板 | TileLang 安装版 | vendored `tilelang_inc/`（diff 逐文件一致） |
| dtype 枚举 | 14 (16U4_ALIGN16B) | 同（离线验证 cuda.h:3754） |
| W 布局 | 纯布局 gstride[1]=819,200 | 块布局 gstride[1]=2,641,920 |
| descriptor 创建 | TileLang host 代码（TVM FFI） | 手写 shim（cuTensorMapEncodeTiled） |
| 编译 flags | `-std=c++20 -w -lineinfo --shared -lcuda` | `-O3 -shared -Xcompiler -fPIC [-Xcompiler -fPIC] -std=c++20` + 选择性 fast-math |

## 已否定假设（8 项 + 三路 audit）

1. **relinquish_alloc_permit 缺失**（illegal-instr-5 发现）
   - 修复（d25d5fe + gen 脚本自动注入 9594610）——**保留**（ISA 合规）
   - 但 **不是根因**：修复后仍 crash；JIT 无它也不 crash（fp4 原型同形跑通）
   
2. **idesc ki-bits 损坏**（illegal-instr-2 误报，illegal-instr-3 推翻）
   - `(ki << 4)` = b_sf_id、`(ki << 29)` = a_sf_id——SF 字节选择，故意设计
   - 教训：位域解读必须用仓库权威位表（dsv41_experts_mxf4.cu:58-71），不能用记忆中的 CUTLASS 布局

3. **--use_fast_math**（NO_FAST_MATH A/B 测试）
   - 两种配置都 crash；只改变错误类型（illegal instruction ↔ illegal memory access）
   - 保留选择性 fast-math build.sh（tilelang_gen 无 fast math）作为防御措施

4. **w3 指针偏移错误**（w3-ptr-audit 否定）
   - w1 = pool + 0、w3 = pool + 870,400 ✓、w_stride = 2,641,920 ✓（实测）
   - SFW1/SFW3 分段连续基址 ✓

5. **Eid 越界**（eid-init-audit 否定）
   - 每 rank 驻留全部 384 expert（TP-split by inter，非 EP）
   - Eid 值域 [0, 384) 全合法；空段 eid=0
   - "每 rank 48 expert" 是 FERRITE_MOE_EP 路径的误植

6. **descriptor 参数不匹配**（desc-param-diff 否定）
   - 6 个 descriptor 逐参数对比：除有意的 gstride[1] 差异外**全部匹配**
   - dtype 14、box、swizzle 128B、l2 128B、elementStrides 全一致
   - SFW 的 51,200 stride 与池布局核实一致

7. **内存边界越界**（手工验证）
   - W1 TMA reach: 383×2,641,920 + 819,200 = 1,012,674,560 < pool 1,014,497,280 ✓（margin ~2MB）
   - A: 23,587,839 < 23,592,960 ✓；C: 11,796,476 < 11,796,480 ✓
   - SFA bulk copy: 737,279 < 737,280 ✓（恰好）
   - shared memory: 全在 kSmem 内；TMEM: 160 < 512 列

8. **JIT vs AOT 源码差异**（diff 验证否定）
   - device_kernel.cu vs moe_bs_up_tl.cu：仅差手加的 relinquish asm
   - vendored 模板 vs 安装模板：逐文件一致

## 完整 Kernel 分析（2026-09-14 下午——173 行逐行审读）

**Warp 分工**：
- Warp 0 (0-31): TMA producer（A, W1, W3, SFA, SFW1, SFW3 → smem）
- Warp 1 (32-63): MMA consumer（tcgen05_cp SF→TMEM + tcgen05.mma + commit）
- Warp 2 (64-95): SF transpose（smem 内 SF 数据转置 → sf_full 信号）
- Warp 3 (96-127): idle（只参与 barrier）

**Barrier 结构**（全部验证自洽）：
- `loaded[3]` (init=32): warp 0 arrive → warp 1/2 wait
- `sf_full[3]` (init=32): warp 2 arrive → warp 1 wait
- `consumed[3]` (init=1): MMA commit arrive → warp 0 wait（防 stage 覆写）
- `tmem_full[1]` (init=1): 最后 MMA commit → 全部 wait（epilogue 前）

**SF 转置验证**（tcgen05_sf_warp_transpose）：
- 4×32 uint32 块内转置，XOR swizzle 避 bank conflict
- 读写 index 均在 [0, 128) 内 ✓
- SFW chunk = SFW1[0,64) + SFW3[64,128)——cp 复制全部 128 到 TMEM sfa_data+4

**已验证正确的项**：
- 所有 smem 偏移和大小
- 所有 mbarrier init/arrive/wait 计数
- TMEM 分配（128+32=160 < 512）和读取（128 列恰好）
- tcgen05_ld 的 128 f32/thread × 128 threads = C 的 128×128
- epilogue 的 swizzled C_sh 写入
- TMA store 的 4×32 列分片

**结论**：静态分析已穷尽——kernel 结构在 m=1 和 m=6 之间没有区别。crash 必须用 compute-sanitizer 定位。

## ROOT CAUSE 分析（2026-09-14 下午更新）

**变量拆分测试结果（用户方法论指令）**：
- **crash 不是数据依赖的——是结构性的！**
- ZERO_SF：❌ 仍 crash（SF 数据不是触发器）
- ZERO_A：❌ 仍 crash（激活数据不是触发器）
- ZERO_EID：❌ 仍 crash（expert ID 不是触发器）
- ZERO-DIAG 消息确认代码执行

**SYNC-DIAG 带捕获守卫的最终判决**：
- `.scale_vec::1X` + `"memory"` clobber 实验：**打破 eager 路径**（之前 MMA OK → 现在 ALL MMA illegal instruction）——**已撤销（4a28cdc）**
- 手写验证代码的 `.scale_vec::1X` 适用于它的探针数据布局，但不适用于我们的 TileLang 生产布局

**真实错误类型**（SYNC-DIAG 揭示）：
- MMA kernel 的错误是 **"illegal instruction"**（不是 "illegal memory access"）
- 之前看到的 "illegal memory access" 是 context poisoning 的二级效应
- eager 路径（m=1）MMA 正常，spec 路径（m=6 verify）MMA illegal instruction

**结构性触发的剩余候选**（判别实验在跑）：
1. SWALLOW_STEP 机制 → no-swallow 测试
2. 流水线深度（40 k-iterations）→ k-limit=1 测试

**关键待解问题**：什么导致 MMA 在 spec 路径（m=6 / CUDA graph capture）下遇到 illegal instruction？
- 可能：idesc 的 sf_id 字段在真实 SF 数据下的行为
- 可能：CUDA graph capture/replay 的执行上下文差异
- 可能：TMEM 布局在 m=6 下的越界
- 需求：compute-sanitizer 精确定位（~/sanitizer_run.sh 已部署）

## 诊断工具（已部署远端）

| 工具 | 位置 | 用途 |
|------|------|------|
| safe_test.sh | ~/safe_test.sh | fail-fast 构建+测试（不用旧二进制） |
| moe_fp4_sync_test.sh | ~/moe_fp4_sync_test.sh | SYNC-DIAG 启用版测试 |
| diag_chain.sh | ~/diag_chain.sh | 诊断决策树（读 DIAG 输出决定下一步） |
| sanitizer_run.sh | ~/sanitizer_run.sh | compute-sanitizer（精确 crash 位置） |
| full_verify.sh | ~/full_verify.sh | crash 修复后的完整验证序列 |
| num_verify.py | ~/tl_bs/num_verify.py | 数值验证（两种 nibble 顺序） |

## 方法论教训

1. **五路会审**（用户指令：5 个相同 prompt subagent）产出 3 个严重发现但只有 1 个真 bug（relinquish）——多角度独立审查的价值在于覆盖面，不在于命中率
2. **JIT 隔离**是最有效的判别实验：直接区分"kernel 设计问题"vs"编译/接线问题"
3. **静态分析有极限**：当所有边界/指针/参数都验证正确时，需要 runtime 诊断
4. **旧日志陷阱**：build 失败时测试不跑，旧日志残留会误导判断——必须 `rm -f` 旧日志 + fail-fast

## 代码级二分策略（2026-09-14 晚——用户指令：直接重写）

**二分步骤 1：memory clobber only**（0b67ea9）
- 只加 `"memory"` clobber 到 MMA 模板（不加 `.scale_vec::1X`）
- 测试中——如果修复，编译器重排是根因

**二分步骤 2：手写 kernel**（8fb3d8a）
- 完全重写 MMA kernel（`moe_bs_handwritten.cu`）
- 基于验证过的 tcgen05 原语（tests_tcgen05_mxf8f6f4_1x.cu）
- 顺序执行（无 TMA/mbarrier pipeline）
- 直接 global load（不用 TMA descriptor）
- `__syncthreads()` 同步（不用 mbarrier）
- Gate: `DSV41_MOE_BS_HANDWRITTEN=1`
- 测试脚本: `~/test_handwritten.sh`
- 如果手写版工作而 TileLang 不工作 → TileLang pipeline 结构是根因

**二分步骤 3（备用）：DEV 入口表复制**
- 让 DEV 入口 D2D 复制调用方表到 shim scratch
- 测试是否调用方表指针有问题

## 完整代码审查（2026-09-14 晚——用户警告"低级失误"后的全面检查）

**moe_bs_weights 返回值验证** ✓：
- `w.0 = a.w1.ptr()` — expert 0 的 W1（gate 权重面基址）
- `w.1 = a.w3.ptr()` — expert 0 的 W3（up 权重面基址）**不是** expert 1 的 W1
- `w.2 = s1.ptr()` — SF_W1 pool 基址（segregated pool 第一段起点）
- `w.3 = t1.ptr()` — SF_W3 pool 基址（segregated pool 第二段起点）
- `w.4 = w_stride` — 相邻 expert 的 block stride（实测 2,641,920）

**shim 参数映射** ✓：
- `w1` → W1 TMA descriptor base（expert 0 的 W1）
- `w3` → W3 TMA descriptor base（expert 0 的 W3）
- `sfw1` → SFW1 TMA descriptor base（SF_W1 pool）
- `sfw3` → SFW3 TMA descriptor base（SF_W3 pool）
- `w_stride` → TMA gstride[1]（expert stride）

**SFW pool 布局验证** ✓：
- Segregated: [E×plane WSF1][E×plane WSF3]
- SFW1 at pool base + 0；SFW3 at pool base + 19,660,800
- SF plane = sf_words × NP = 40 × 320 = 12,800 uint32 = 51,200 bytes
- TMA gdim=(12800, 384), gstride[0]=51200（每 expert），box=(64, 1)

**ex_act_r 输出缓冲验证** ✓：
- 分配：VERIFY_ROWS × topk × 2 × max(inter, dim) = 6×6×2×5120 = 368,640 floats
- scatter 最大写：(5×6+5)×640+639 = 23,039 floats（远小于分配）

**结论：无低级错误。代码路径完整正确。**

## 🎉 根因定谳（2026-09-14 晚——手写 kernel 二分成功）

**手写 kernel 测试结果**：0 illegal errors（vs TileLang 的 10 errors）
- HANDWRITTEN gate 激活确认（日志有消息）
- **TileLang kernel 的 pipeline 结构（TMA + mbarrier + 3-stage）是 m>1 crash 的根因**
- tcgen05 MMA 指令、数据加载、SF 处理全部正确

**第一次测试的 cuda error 1**：手写 kernel 缺少 `cudaFuncSetAttribute`（smem 65536 > 48KB 默认限制）——已修复（ced1fb9）

**含义**：
- TileLang 的 TMA+mbarrier 流水线在 m>1（verify）场景下有某种竞态或状态问题
- 手写版的顺序执行（直接 load + __syncthreads）完全安全
- 性能代价：无 TMA 硬件加速、无流水线重叠——比 TileLang 慢但可用

**后续**：
1. 验证手写 kernel 输出质量（修复 SetAttribute 后）
2. 如果输出正确：用 DSV41_MOE_BS_HANDWRITTEN=1 跑 push400
3. 后续优化：给手写版加 TMA 和简单双缓冲（不学 TileLang 的复杂 3-stage）

## Core Matrix 布局修复（2026-09-14 晚——输出乱码的根因）

**问题**：手写 kernel 0 errors 但输出乱码（数字无序，mean-k=2.000）
**根因**：UMMA smem descriptor 期望 **core matrix 布局**（8行×16B 原子），不是 row-major

**Core Matrix 布局公式**：
```
addr(m, k) = (m/8)*1024 + (k/16)*128 + (m%8)*16 + (k%16)
```
- 8行×16B = 128B 一个 core matrix（原子）
- K-blocks per row: 128/16 = 8
- M-blocks: 128/8 = 16
- lbo = 16B（原子内行距）= 1（16B 单位）
- sbo = 1024B（原子间距）= 64（16B 单位）
- layout = 0（无 swizzle，自然 core matrix 顺序）

**为什么 TileLang 不需要手动做**：TMA 硬件自动写 core matrix 布局
**手写 kernel 必须**：在 global→smem 加载时手动转换为 core matrix 布局

**修复**（8471fc0 + 6408f11）：
- A tile: `A_sh[(m>>3)*1024 + (kk>>4)*128 + (m&7)*16 + (kk&15)] = val`
- B tile: 同样公式（W1 前 64 行 + W3 后 64 行）
- Descriptor: layout 从 2 改为 0

## cp.async 双缓冲优化设计（下一步性能优化）

**当前性能**：verify=34.45ms（顺序执行，无重叠）
**优化后预期**：verify≈22ms（load 完全隐藏在 compute 后）
**400 tok/s 需要**：step≤9.8ms——需要更激进的优化

**双缓冲流程**：
```
预加载 iteration 0 → buf0 (cp.async)
for k in 0..40:
    cp.async 加载 k+1 → buf1
    __pipeline_wait_prior(1)  // 等 k 的数据
    __syncthreads()
    计算 k（从 buf0）
    swap(buf0, buf1)
```

**smem 布局（双缓冲，~98KB）**：
- buf0: A[0,16K) B[16K,32K) SF[32K,34K)
- buf1: A[35K,51K) B[51K,67K) SF[67K,69K)
- mbar: [34K, 34K+8)
- C staging: 覆盖 buf1（所有 MMA 完成后写入，此时 buf1 不再需要）

## 🎯 K-block Descriptor Advance 根因（2026-09-14 深夜——数学确认）

**根因**：手写 kernel 的 K-block descriptor 前进量用了 TileLang 的 `ki*32`，但 core matrix 布局需要 `ki*16`。

**数学分析**：
- Core matrix 布局：每个 K-block（32 元素）跨 2 个 K-atom
- 每个 K-atom = 8行 × 16B = 128B
- K-block ki 起始位置 = ki × 2 × 128B = ki × 256B
- Descriptor 单位 = 16B
- 正确前进量 = ki × 256/16 = **ki × 16**

**旧值 ki×32 的影响**：
| ki | 正确位置 | 旧值读取位置 | 结果 |
|----|---------|------------|------|
| 0 | 0B | 0B | ✓ 正确 |
| 1 | 256B | 512B | ✗ 读 K-block 2！|
| 2 | 512B | 1024B | ✗ 读下一 M-atom！|
| 3 | 768B | 1536B | ✗ 完全错误 |

**75% 的 K-blocks 读错误数据 → 垃圾输出**

**为什么合成数据测试没发现**：
- 合成数据（全 0x22 权重）中，读错位置 = 读对位置（所有字节相同）
- bit-exact 验证通过因为错数据 = 对数据
- 真实数据（变化权重）中，错位置读错数据 → 垃圾

**修复链**：
1. TileLang m>1 crash → 手写 kernel（验证过的原语）
2. 输出乱码 → core matrix 布局修复（8行×16B 原子格式）
3. 仍乱码 → K-block advance 修复（ki*32 → ki*16）
4. （测试中）正确输出？

**验证**：gather verify MATCH（数据流正确）+ kernel bit-exact（合成数据自洽但错）

## lbo 修复（2026-09-14 深夜——K-block fix 不够，lbo 也错了）

**发现**：K-block advance fix (ki*16) 单独不够——仍乱码。lbo 也需要修复。

**CUTLASS 文档**：
- LBO (Leading Byte Offset) = "core matrices 间 K 方向的距离"
- Core matrix = 8行 × 16B = 128B
- LBO = 128 bytes = **8 units**（不是 1 unit = 16 bytes）

**修复**：lbo 1→8（commit 45aa679）

**完整的 descriptor 参数**（修复后）：
| 参数 | 值 | 含义 |
|------|-----|------|
| start_address | A_sh/B_sh | tile 起始 |
| lbo | 8 (128B) | K-atom 间距 |
| sbo | 64 (1024B) | M-atom 间距 |
| layout | 0 | 无 swizzle |
| K-block advance | ki*16 (256B) | K-block ki 起始位置 |

**与 TileLang 的差异**：
- TileLang: lbo=1, ki*32（TMA + swizzle 布局）
- 手写: lbo=8, ki*16（core matrix + layout=0）
- 两者不可互换！

---

## §10 手写 kernel 乱码的三个真 bug（2026-09-14，逐行对照 TileLang 生成码定位）

全部通过**逐行对照** `kernels/cuda/tilelang_gen/moe_bs_up_tl.cu`（TileLang 生成的 MoE BS 内核）
与 `kernels/cuda/tilelang_inc/tl_templates/cuda/common.h` 得出，并用 subagent 对权威位表交叉验证。

### Bug 1（真·已修）：`hw_make_idesc` 的 `b_sf_id` 漏了 `<< 4`
- 权威位表（`dsv41_experts_mxf4.cu:58-67` + `tests_tcgen05_mxf8f6f4_1x.cu:837-840`）：
  `[4,6) b_sf_id`、`[0,2) sparse_id2`、`[2,3) sparse`、`[29,31) a_sf_id`
- 我方（修前）：`d |= (uint32_t)(sf_id & 3);`（**无移位**）→ `b_sf_id` 落到 `[0,2)`，
  同时把 `[0,2) sparse_id2` 污染成 `ki`
- **后果**：ki=1/2/3 时 B 侧 scale 字节选择子恒为 0（B 权重全用 block-0 的 scale）
  **且** dense MMA 被塞了非零 sparse_id2（UB）
- ki=0 时两处都退化为 0 ⇒ **合成数据 / 只看第一个 K-block 的测试完全掩盖它**
- 修法：`d |= (uint32_t)(sf_id & 3) << 4;`

### Bug 2（真·已修）：smem 写入布局与 descriptor 的 swizzle 声明不匹配
- TileLang：`initialize_tcgen05_descriptor(desc, A_sh, 1, 64, 0, 0, **2**)`，
  第 7 参 = `layout_type_ = 2` = **SWIZZLE_128B**；TMA 用 `CU_TENSOR_MAP_SWIZZLE_128B` 写入
- 我方（修前）：`layout=0`（SWIZZLE_NONE）却写"core matrix 序" —— 与 SW128 的 smem 内容不符
- SW128 的正确写入（CUTLASS `swizzle<3,4,3>`：bit[4,7) ^= bit[7,10)）：
  `addr(r,c) = (r/8)*1024 + (r%8)*128 + (((c/16) ^ (r%8))*16) + (c%16)`

### Bug 3（真·已修）：K-block descriptor 递进单位误判 8×
- `Tcgen05SMemDescriptor::operator+`（`common.h:768`）做的是
  `ret.reg32_[0] += uint32_t(offset) >> 4` —— **offset 单位是字节，右移 4 转成 16B 单位**
- 故 TileLang 的 `desc_a + (ki * 32)` = start_address **+ki*2 单位 = +ki*32 字节**
- 我方（修前）直接对 64 位裸值 `+ (ki*16)` = **+ki*16 单位 = +ki*256 字节 ⇒ 8 倍过大**
- 修法：`a_desc_base + (uint64_t)(ki * 2)`

### 交叉参考：仓库自带已验证的 canonical 布局（`dsv41_experts_mxf4.cu`，四臂一致）
- 公式：`unit16(m,kb) = (m%8) + 8*kb + 16*(m/8)`（16B 单位）
  → **LBO = 8 单位 (128B)**、**SBO = 16 单位 (256B)**、`layout_type = 0`
- 与 TileLang（SW128 + lbo=1/sbo=64）是**两套不同的自洽方案**，参数不可互换
- idesc 朝向差异：`tests_tcgen05_mxf8f6f4_1x.cu` 用 **a=E2M1/b=E4M3（swapAB）**，
  TileLang 与我方用 **a=E4M3/b=E2M1**

### 诊断工具（已入 shim，one-shot，默认开）
- `[moe-bs][GATHER-DIAG]`：g_a（gather 产物）vs xq4（原始）前 8 字节 → 已验证 **✅ MATCH**
- `[moe-bs][MMA-DIAG]`：g_c（MMA 输出）前 16 个 f32 → 用于**跨臂数值对比**（TileLang 臂 vs 手写臂）

## §11 新增工具与实验矩阵（2026-09-14）

### 运行期开关（全部默认 OFF，`cudaMemcpyToSymbol` 于 INIT 期设置）
| env | 作用 | 默认 |
|---|---|---|
| `DSV41_MOE_BS_SCALEVEC1X=1` | MMA 用 `.scale_vec::1X` 后缀（仓库已验证 e4m3 臂的拼写） | 0（无后缀，TileLang 拼写） |
| `DSV41_MOE_BS_CANON=1` | smem 布局切到仓库 canonical interleave（lbo=8/sbo=16/layout=0，递进 ki*4096B） | 0（TileLang SW128：lbo=1/sbo=64/layout=2，递进 ki*32B） |
| `DSV41_MOE_BS_NUMCHECK=1` | **主机参考对拍**：对第一个 assignment 用显式 e4m3/e2m1/ue8m0 解码器在 double 下算 gate[0..15]，与 MMA 的 g_c[0..15] 比，打印 ref/mma/worst-rel/MATCH | OFF |

### 关键事实：TileLang 臂不可用作参照
`~/moe_tl.log`（TL 臂 eager 测试）显示 **8/8 rank `cuda error 700`（illegal address）发生在配置探针阶段**
（`dsv41_bf16_roundtrip` / `dsv41_gemm_fp8_mx_f32`）——即 TL 臂在 INIT/首次调用就 fault，
错误被闩存到后续无关调用上。故 TL 臂的"空串输出"不是数值退化，而是**根本没跑起来**。
⇒ 从 TileLang 生成码抄来的参数**不能视为已证正确**，必须用 NUMCHECK 做绝对对拍。

### 已逐字对齐 TileLang 的指令层（对照结果）
| 项 | 结论 |
|---|---|
| MMA asm + 谓词 | 逐字相同（`tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale [%0],%1,%2,%3,[%5],[%6],p`） |
| idesc | 修 `<<4` 后 = `144708608 \| ki<<29 \| ki<<4` ✓ |
| SF→TMEM 复制 asm | 逐字相同（`tcgen05.cp.cta_group::1.32x128b.warpx4`） |
| SF TMEM 地址 | 都是 base+0（SFA）/ base+4（SFB）；TMEM 分配都是 C=128 列 / SF=32 列 ✓ |
| TMEM→寄存器 | 都是 `tcgen05_ld_32dp32bNx<128,false>` ✓ |
| A/B descriptor | 同参数（lbo=1/sbo=64/layout=2）、K-block 递进同为 ki*32 字节 ✓ |

**仍未证正确的两项**：① smem 写入的 SW128 公式（我从 CUTLASS `swizzle<3,4,3>` 推导：
`addr(r,c)=(r/8)*1024+(r%8)*128+(((c/16)^(r%8))*16)+(c%16)`，等价于 `r*128 + ((c/16)^(r%8))*16 + (c%16)`）；
② `.scale_vec` 后缀（省略时的 PTX 默认值未知）。

### 精度对齐核对（用户指令：fp4/fp8 必须与官方完全一致）
- 激活：`dsv41_quant_fp8(x, y, s, m, dim, 32, /*round_scale=*/true)` —— block=32、**2 的幂 ceil 标度**，
  其 `fast_round_scale`（`dsv41_kernels.cu:113-119`）与官方 `fast_log2_ceil`+`fast_pow2`（`kernel.py:22-37`）**逐位同构** ✓
- 权重：e2m1 + ue8m0 per-32 ✓；官方 `fp4_gemm` 是「FP4 无损 cast 成 E4M3 后跑 FP8×FP8」，
  而 e2m1 的 8 个幅值 {0.5,1,1.5,2,3,4,6} 都能被 e4m3 精确表示 ⇒ 混合 e4m3×e2m1 MMA 与官方 FP8×FP8 **数学等价**
- latent：`xn_r` bf16_snap ✓（官方 FFN 输入是 bf16）

### 实验矩阵（`~/matrix_test.sh`，单次构建跑 6 臂）
`SELECT * FROM {SW128, canonical} × {no-suffix, scale_vec::1X}` + TileLang 臂 + 老路径基线；
每臂打印 e2e 计数输出、ERR 计数、step、MMA-DIAG(g_c)、GATHER-DIAG、NUMCHECK 三条行。

## §12 内存模型/栅栏缺失（第 5、6 个真 bug，2026-09-14）

手写 kernel 与 TileLang 生成码在**同步/可见性**上还差两类栅栏。两者都**在常数/合成数据下完全不可见**（陈旧 smem 内容 ≡ 当前内容），只在真实变化数据下爆乱码——这正是"数值单测通过但模型乱码"的原因。

### Bug 5（真·已修）：缺 `fence.proxy.async`
- **机理**：`A_sh/B_sh/SFA_sh/SFB_sh` 由**泛型** store 写入；`tcgen05.mma` 与 `tcgen05.cp` 通过 **async proxy** 读 smem。
  PTX 内存模型要求泛型写对 async proxy 可见，必须 `fence.proxy.async.shared::cta`。
- **证据**：TileLang 生成码在 consumer 路径里有 `tl::fence_proxy_async()`（`moe_bs_up_tl.cu:138,160`）；
  我方原 kernel 只在 k-loop **外**有 1 处 fence（`moe_bs_handwritten.cu:229`，为 mbarrier init）。
- **修法**：在 SF 转置之后、warp-1 的 cp/MMA 之前插入 `fence.proxy.async.shared::cta`（全线程执行）。
- **后果若缺**：MMA 可能读到**上一轮 k-iteration 或上一次 launch 的陈旧 smem**。

### Bug 6（真·已修）：缺 `tcgen05.fence::before/after_thread_sync`
- **机理**：tcgen05 是异步操作，其完成与其他线程的同步（`__syncthreads`）之间需要
  `tcgen05.fence::before_thread_sync`（屏障前）+ `tcgen05.fence::after_thread_sync`（屏障后）。
- **证据**：TileLang 在 epilogue 与 k-loop 的每个同步点都成对插入
  （`tcgen05_before_thread_sync()` / `__syncthreads()` / `tcgen05_after_thread_sync()`，
  见 `moe_bs_up_tl.cu:128-131,145-149,156-159`）；我方原来只有裸 `__syncthreads()`。
- **修法**：epilogue（TMEM 读前后）与 k-loop（mbarrier wait 之后）两处都补成对栅栏。

### 已逐字核对一致的指令层（本次新增）
| 项 | 结论 |
|---|---|
| `tcgen05.commit...mbarrier::arrive::one` | 逐字相同（`tcgen_05.h:110` vs `moe_bs_handwritten.cu:82`） |
| `tcgen05_sf_warp_transpose` | 逐字相同（含 `__syncwarp()`；我方仅用 1 个 warp，TileLang 用 3 个 warp 冗余执行——后者有竞态隐患，我方更干净） |
| `make_sf_smem_desc` | 相同（start>>4、SBO=8、version=1、LBO=0、layout=0） |
| descriptor 位域 | start[0,14)、lbo[16,30)、sbo[32,46)、version[46,48)=1、base_offset[49,52)、lbo_mode[52]、layout[61,64)；**编码枚举：SWIZZLE_NONE=0、SWIZZLE_128B=2、SWIZZLE_64B=4、SWIZZLE_32B=6**（`common.h:750-757`）⇒ 我方 layout=2 = SWIZZLE_128B 判断正确 |
| smem 区间 | A[0,16384) B[16384,32768) SFA[32768,33280) SFB[33280,33792) mbar[33792,33800) C[0,65536)（C 与 A/B/SF/mbar 重叠，仅在全部 MMA 完成后写入，安全） |
| TMEM 生命周期 | alloc(128/32) → relinquish → dealloc；与 TileLang 同 |

### 六条数据通路一致性核对（本次）
1. A 行距：kernel `HW_K=5120` ≡ gather `abytes=kDim=5120` ✓
2. 激活 SF：gather 写 `sfa[g*M + seg*128 + r]`，kernel 读 `SFA[k*(SEGCAP*128) + seg*128 + i]` ✓（`SEGCAP*128 = M`）
3. 权重 SF：池 `[E][40][320]`，kernel 读 `SFW1[e*40*320 + k*320 + n_tile*64 + i]` ≡ pack_wsf 的 `dst[g*rows + row]` ✓
4. ue8m0 转换：gather 的 `f_pow2_to_ue8m0` 对 2 的幂 s=2^e 返回 e+127 ✓（解码 2^(b-127) 一致）
5. B 行映射：W1 行 `n_tile*64+row` → B_sh 行 `row`；W3 行 `n_tile*64+row` → B_sh 行 `64+row`；scatter 列映射 `c<64 → gate(n_tile*64+c)`、`c>=64 → up(n_tile*64+c-64)` ✓
6. SF 字节序：gather `byte j = 第 j 个 K-block`，MMA `sf_id=ki` 选 byte ki，descriptor 递进到 K-atom ki ✓
7. **scatter 的列布局**：`tl_moe_bs_scatter_kernel`（`moe_bs_shim.cu`）把 C 的第 `col` 列映射为
   `n = (j<64) ? (bx*64+j) : (320 + bx*64 + (j-64))`（`bx = col/128`、`j = col%128`）⇒ **gate 落 [0,320)、up 落 [320,640)**，
   与手写 kernel 的 epilogue 列布局**完全一致**，也与下游 swiglu 的 `row[i]`（gate）/`row[inter+i]`（up）读法一致 ✓。
   `dst = (idx/split)*out_pitch + (idx%split)*nup`（up 路径 split=topk、out_pitch=topk*2*inter、nup=2*inter）
   = 该 assignment 的槽内偏移 ✓。**该项排除**。（注：NUMCHECK 只验 scatter **之前**的 g_c，所以专门查了这一环。）

## §13 微基准定谳：非 swapAB 朝向算不出正确乘积（第 7 个真 bug 级发现）

**工具**：`kernels/cuda/tests_bs_mma_micro.cu`（独立单 CTA 微基准，M=N=K=128 = 生产一个 k-iteration 同构；
数据逐元素变化、scale 逐 K-block 不同；参考为双精度、用**解码后**的值；反自欺：cp 前先毒化 TMEM 的 SF 列）。
编译必须 `nvcc -gencode arch=compute_103a,code=sm_103a`（`-arch=sm_103a` 被该 nvcc 静默降级成 sm_103，ptxas 全拒 tcgen05）。

**结果（8 组合 = {SW128, canonical} × {无后缀, `.scale_vec::1X`} × {生产 SF 路径 cp, 已验证 SF 路径 st}）**：
- **全部 FAIL**：`relerr ≈ 92`、`bad = 16384/16384`（每个元素都错）、`bad_rows=128/128`、`bad_cols=128/128`
- **case 4 与 case 5（cp vs st 两条 SF 通路）数值逐位相同**（如 mma 都是 140.095703 / 743.810547 / 11.093750 …）
  ⇒ **SF 通路无辜**（两条完全不同的 SF 投递方式给出同一结果），错误在操作数朝向/布局本身
- 参考实现已自查：`sae[m][j]=((m*7+j*5)%7)-3`（指数落进对应字节）、ref 用解码值逐 K-block 乘 `2^sae·2^sbe`、
  编码非精确元素计数 `enc_mismatch` 期望 0 —— 参考无误 ⇒ **FAIL 是真实的**

**推论**：`(a_format=0 E4M3, b_format=5 E2M1)` 这个朝向在 sm_103a 上**不产生正确的 A×B 乘积**。
与 §12/§10 的旁证一致：树内**唯一 GPU 已验证**的 mxf8f6f4 配置（`tests_tcgen05_mxf8f6f4_1x.cu` 的 Phase-0 探针、
`tc5::e4`）全部使用 **swapAB（A=权重 e2m1，B=激活 e4m3）**；而用非 swapAB 的两个实现都失败
（TileLang 生成码 err 700@INIT；手写 kernel 乱码）。

**已实现的门控**：`DSV41_MOE_BS_SWAPAB=1`（A_sh 存权重、B_sh 存激活、SFA/SFB 角色对调、
idesc 用 a_fmt=5/b_fmt=0、epilogue 因 M=输出通道/N=token 而转置落盘）。
**交叉验算**：swapAB 下 idesc(ki=0) = 144,704,128 = 已验证探针的 0x08820280(142,738,048)
+ n_dim 差（N=8→128 即 15<<17 = 1,966,080）—— 完全吻合 ✓

## §14 精度对齐待办（用户硬性要求：fp4/fp8 与官方**完全**对齐，不能高也不能低）

审计（subagent official-op-parity，逐算子 file:line 对照）结论：**已对齐**与**未对齐**各若干。

### 已对齐 ✅
| 项 | 证据 |
|---|---|
| 激活量化 `dsv41_quant_fp8(block=32, round_scale=true)` | `fast_round_scale`（`dsv41_kernels.cu:113-119`）与官方 `fast_log2_ceil`+`fast_pow2`（`kernel.py:22-37`）逐位同构；调用点 `chain_dev.rs:17145-17154` 传 `true` |
| 权重 scale 来源 | 双方都是 checkpoint 的 e8m0 字节**原样搬运**（`load.rs:1066-1073` "pure byte permutation … bit-exact and lossless"；`convert.py:180` fp4 只是 `.view()`） |
| scale 形状/序 | `[out, in//32] u8` 行主序、**byte 0 = 最低 K-block**、数值 `2^(byte-127)` —— 双方一致 |
| 官方 fp4_gemm 的语义 | B 由 FP4 **无损** cast 成 E4M3 再跑 FP8×FP8；因 e2m1 的 8 个幅值 {0.5,1,1.5,2,3,4,6} 都是 e4m3 精确可表示 ⇒ 与我们的 e4m3×e2m1 混合 MMA **数学等价** |
| shared expert 路径 | `swiglu_limit_on → bf16_snap_on → quant1_on`（`chain_dev.rs:22020-22030`）≡ 官方 `silu*up → .to(bf16) → act_quant` ✓ |

### 本轮新增验证 ✅（逐算子对拍继续）
| 项 | 结论 |
|---|---|
| **swiglu（clamp + silu）** | **逐式一致**：`dsv41_glue.cu:1543-1560` 的 `swiglu_limit_batched_kernel` 做 `g=fminf(g,limit)`（gate 只 clamp 上限 ✓）、`u=fminf(fmaxf(u,-limit),limit)`（up 双侧 clamp ✓）、`(g/(1+expf(-g)))*u`（silu ✓）——与官方 `model.py:845-849` 的 `up=clamp(up,-lim,+lim)` / `gate=clamp(gate,max=+lim)` / `F.silu(gate)*up` **完全同构**。唯一差别是 `F.silu` 与 `g/(1+exp(-g))` 的浮点运算次序（≤1 ulp），以及该 TU 编译时带 fast-math（`expf` 精度 ~2 ulp）——量级可忽略 |
| 激活量化输出 | 与官方**逐字节一致**（subagent official-numeric-parity 的 Q2 实测） |
| **fp4 (e2m1) 解码表** | **逐值一致**：`kFp4Table`（`dsv41_kernels.cu:108-111`）与官方 `convert.py:13-15` 的 `FP4_TABLE` 16 个码点完全相同（`0, .5, 1, 1.5, 2, 3, 4, 6` 及其负值）|
| **fp4→e4m3 无损性** | 成立（e2m1 的 8 个幅值 `{0,.5,1,1.5,2,3,4,6}` 都是 e4m3 精确可表示 ⇒ 官方"cast 到 fp8 再跑 FP8×FP8"与我们"e4m3×e2m1 混合 MMA"**数学等价**）|
| **全零块的标度下限** | **数值无害**：官方 `max(amax,1e-4)`（`kernel.py:76`）vs 我方 `max(scale,1e-30)`（`dsv41_kernels.cu:156`）——全零块的**值**恒为 0，乘积恒 0，标度不会被当作除数使用 ⇒ 无影响（仅字面差异）|

### 未对齐 ⚠️（**待修，且必须等 BS 臂正确后再动**——一次只改一个变量）
| # | 官方写法 | 我方写法 | 影响 |
|---|---|---|---|
| 1 | `x = weights * x` 在 **w2 之前**（`model.py:849`，即 `w2(quant_bf16(w ⊙ silu(g)·u))`） | `x *= row_weight[slot]` 在 **w2 的 epilogue**（4 条 down 臂全是：`dsv41_experts_mxf4.cu:842-852 / 2522 / 2885-2897 / 6873`） | 量化发生在加权前后不同 ⇒ 舍入路径不同 |
| 2 | w2 输入是 **e4m3（block 32, ue8m0）**（`model.py:851 → Linear.forward → act_quant`） | **routed** 路径的 down **直接吃 f32**（`chain_dev.rs:21759-21790` 传 `ex_act_b as *const f32`；`dsv41_experts_mxf4.cu:873-875` 的 `a_f32 != nullptr` 分支）⇒ 我方精度**偏高** | 相对误差 ≤0.2%（我方）vs 典型 1-2%（官方） |

**修法方向（已修正——上一版此处说错了）**：`dsv41_expert_down_reduce_fp4_batched` 的签名是
`(act_base f32, act_stride, out, rows, dim, inter, row_weight f32, rw_stride, slots, w2_base u8, w2_stride, w2s_base u8, w2s_stride, ids, stream)`
（`dsv41_experts_mxf4.cu:3833-3837`）—— 尾部的两个 `*const u8` 是 **w2 的权重指针**，**不是**激活量化对；
**该 launcher 没有任何 e4m3 激活入参**，只吃 `act_base`（f32）。
⇒ 要对齐官方，需要**换用支持 AQ（量化激活）模式的内核**：`expert_gemv_fp4_kernel` 本身就有
`a`（e4m3）+ `a_scale` + `a_f32` 三个入参、由 `aq` 选择路径（`dsv41_experts_mxf4.cu:751`、`:440` 的
`a_f32 // [rows, k] f32 (AQ=true)`），即 M=1 的 gate/up 走的就是 AQ 模式。
⚠️ **历史警告**：把这条 GEMV 扩展到 down 路径**曾被试过并把模型搞坏**（`dsv41_experts_mxf4.cu:3268-3273`：
"was tried and CORRUPTED the model — one prompt returned all zeros and others hit an illegal memory access — so it is
reverted until the down call's exact arguments are worked out"）。所以这一步必须**单独、谨慎**做，先把
down 的 `rows/dim/inter/act_stride/slots/ids` 的确切语义与 `epi_mode` 组合钉死。
顺序仍必须与官方一致：**先加权（route_w）→ 再 bf16 → 再 e4m3 量化**（不是先 bf16 再加权）。
代价：routed 路径多 1 次 quantify（40→80 launch/step 量级），µs 级，对 400 tok/s 目标可忽略。
⚠️ 注意顺序必须与官方一致：**先加权、再 bf16、再 e4m3**（不是先 bf16 再加权）。
⚠️ 该改动会改变现有"老路径"的数值（其当前输出是完美的）——按用户指令，对齐官方才是正确行为，但必须在 BS 臂正确性钉死之后单独验证。

## §15 当前状态与下一步（2026-09-14 深夜，供接手）

### 已钉死的结论
1. 手写 kernel 的乱码有 **6 个已修的真 bug**：idesc `b_sf_id` 漏 `<<4`（§10）、smem 布局与 descriptor 的 swizzle 声明不匹配（§10）、K-block 递进单位误判 8×（§10）、缺 `fence.proxy.async`（§12）、缺 `tcgen05.fence::before/after_thread_sync` 三处（§12）。
2. **微基准定谳（§13）：`(a=E4M3, b=E2M1)` 朝向算不出正确乘积**——8 组合全 FAIL、每个元素都错、且两条完全不同的 SF 投递路径给出逐位相同的结果 ⇒ SF 通路无辜、朝向有罪。
3. 树内**唯一 GPU 已验证**的 mxf8f6f4 配置全部是 **swapAB**（§13），其 idesc 与我的 swapAB idesc 交叉验算吻合。
4. 精度对齐：激活量化/scale 来源/形状与字节序**均已对齐**官方；**两处未对齐**（§14）待修：路由权重在 w2 之后乘、routed down 输入是 f32（官方是 e4m3 量化 ⇒ 我方精度偏高）。

### 已落地的开关（全部默认 OFF）
| env | 作用 |
|---|---|
| `DSV41_MOE_BS_HANDWRITTEN=1` | 走手写 kernel（否则 TileLang 臂——注意 TileLang 臂在 INIT 就 err 700，不可用） |
| `DSV41_MOE_BS_CANON=1` | smem 布局切 canonical interleave（lbo=8/sbo=16/layout=0，递进 ki*4096B） |
| `DSV41_MOE_BS_SCALEVEC1X=1` | MMA 加 `.scale_vec::1X` 后缀 |
| `DSV41_MOE_BS_SWAPAB=1` | A=权重(e2m1)/B=激活(e4m3)，idesc a_fmt=5/b_fmt=0，epilogue 转置 |
| `DSV41_MOE_BS_NUMCHECK=1` | 主机参考对拍（对第一个 assignment 的多个 (row,col) probe 点，seg 0/1/2，比 g_c） |

### 远端现成工具
| 脚本/文件 | 用途 |
|---|---|
| `~/arm_run.sh <name> [ENV...]` | 通用单臂 e2e（打印开关回执/输出/ERR/step/GATHER-DIAG/[NC] 对拍） |
| `~/matrix2_test.sh` | swapAB 2×2 矩阵（S1..S4） |
| `~/next_round.sh` | 一个 GPU 窗口 = 16 组合微基准 + swapAB 矩阵 |
| `~/verify_correct.sh <port> <label>` | 正确性验收（1..100 前 61 行 + 拉丁探针 + step p50） |
| `~/ferrite/kernels/cuda/tests_bs_mma_micro.cu` | 16 组合微基准（838 行；**必须** `-gencode arch=compute_103a,code=sm_103a`） |

### 下一步（按序）
1. 跑 `~/next_round.sh` → 拿 16 组合微基准的 PASS/FAIL 与 swapAB 矩阵的 e2e/[NC] 数据。
2. 若 swapAB 的某组合 PASS：用该组合（swapAB + 对应布局 + 对应 scale_vec）做单臂 e2e，`verify_correct.sh` 验收，然后全 gate 回归 + `~/push400_hw_test.sh` 测真实 p50。
3. 若 swapAB 也全 FAIL：看微基准的 `bad_rows/bad_cols` 分布定位到 M/N/K 哪一维；重点查 K-block 递进与 SF 的 sf_id 语义（`.scale_vec::1X` 下 sf_id=ki 是否真的对应第 ki 个 K-block）。
4. 正确性钉死后：做 §14 的两处精度对齐（单独验证，一次一个变量）。

## §16 诊断工具的触发陷阱（重要，别再踩）

### NUMCHECK 为何"不打印"
`NUMCHECK`（§11 的 `[NC]` 对拍）插在 shim 的 **device-tables 入口**（`dsv41_moe_tilelang_gate_up_bs_dev`，
Rust 侧 rows=1 的单行解码走这条）。但：
- 单行解码步**全部被 whole-step CUDA graph 捕获**（`DSV41_GRAPH_STEP` 默认 ON，`chain_dev.rs:7649-7689`）；
- shim 对**捕获期**的调用一律 `return 2`（tcgen05 illegal-instruction 规避），NUMCHECK 的 `cudaStreamIsCapturing == none` 守卫因此永不通过；
- 多行 prefill 走的是**另一个入口**（host-tables，未插 NUMCHECK）。
⇒ 现象：`.so` 里有 `[NC]` 字符串（`strings … | grep "WORST rel"` 命中）却**从不打印**。

**零代码改动的解法**：跑 e2e 时带 **`DSV41_GRAPH_STEP=0`**（同时可加 `DSV41_VERIFY_GRAPH=0`）——
单行步不再被捕获 ⇒ BS 臂**全程运行**（不再回退 GEMV）+ NUMCHECK 触发。命令形如：
`bash ~/arm_run.sh <name> DSV41_MOE_BS_SWAPAB=1 DSV41_MOE_BS_CANON=1 DSV41_GRAPH_STEP=0`

### "DECLINE during graph capture" 不代表该次运行没测 BS 臂
`[moe-bs] DECLINE during graph capture …` 与 Rust 侧的 `ARMED but skipped … this run measures the OLD path`
都是**一次性**日志：它们只说明**捕获期的那部分调用**回退了 GEMV。非捕获调用（prefill、捕获预热）
仍然走 BS 臂 ⇒ **prefill 的 KV 被污染**，解码期即便回退 GEMV 也照样输出乱码。
这就是为什么 A1/A2/B1/B2 与 swapAB 各臂的输出差异是**真实信号**：
各臂日志都含 `DECLINE=1 ARMED_SKIPPED=1`（已核对 `~/arm_*.log`），但输出从随机乱码（非 swapAB）
到"5 4 3 2 1 6 5 4"（swapAB+canonical，递减连续数字）——**仍是可比较的**。

## §17 SF 放置的独立推导（关闭"SF 嫌疑"这条线）

shim 差分审计把 **smem-SF 通路（`tcgen05_sf_warp_transpose` + `tcgen05.cp`）** 列为嫌疑榜首，
理由：它在树内**没有已验证先例**（`tests_tcgen05_mxf8f6f4_1x.cu` 的 Phase-0 探针用的是
`tcgen05.st.32x32b.x4`，即寄存器→TMEM，而非 smem→TMEM 的 `cp`）。

**主 agent 的独立推导（把整条链算穿）**：
1. 转置前 `SFA_sh[r]` = 第 r 行的 SF 词（4 字节，byte j = 该行第 j 个 K-block 的标度）。
2. `hw_sf_transpose`：`values[i] = smem[(i ^ (lane>>3))*32 + lane]`，随后
   `smem[lane*4 + (i ^ (lane>>3))] = values[i]`。
   ⇒ 转置后 `smem[lane*4 + i]` = 原 `smem[k*32 + lane]`，其中 `k = i ^ (lane>>3)`。
3. `tcgen05.cp … 32x128b.warpx4`：把每 lane 的 4 个 u32（16 B）搬到 TMEM 的 4 个连续列。
4. 于是：原第 r 行的 SF 词 → TMEM **lane = r%32、column = SF_base + (r/32)**、byte = K-block。
5. 与已验证探针的约定（"SF word 位置 = lane m%32、column SF_base + m/32、byte sf_id"）
   **逐项吻合** ⇒ 我们的 SF 放置与已验证形式一致。
6. 另：微基准的 8 组合实测中 **cp 与 st 两条通路给出逐位相同的结果** ⇒ 投递机制等价。

**结论**：SF 的**放置**与**投递机制**都不构成嫌疑；smem-SF 无先例只是"差分法盲区"，不是缺陷。

### 一处仍需 GPU 解释的现象
shim 的 capture 期 decline 注释记载：tcgen05 路径在 **CUDA graph replay** 下曾报过
"illegal instruction"（直接 launch 同 kernel 却正常）。根因未定。已修的两个内存模型栅栏
（`fence.proxy.async`、`tcgen05.fence::before/after_thread_sync`）**正是** replay 与直接 launch
行为差异的常见来源 ⇒ 下一次带 graph 的 e2e 若不再出现该 decline，即为此前的栅栏缺失所致。

## §18 ⚠️ 两个 harness 自身的可信度问题（重大更正）

### (a) 16 组合微基准的 FAIL **不作数**
它报出：**换 smem 布局（canonical ↔ SW128，写公式与 descriptor 成对换）、换 SF 通道（`tcgen05.cp` ↔ `tcgen05.st`）、
换 A/B 朝向（`A=E4M3/B=E2M1` ↔ swapAB），D 的误差**逐位相同**（`max|diff|=2.2651e+03`、`relerr=9.2376e+01`、`bad=16384/16384`）。
**这在物理上不可能**：两套写公式给 smem 的内容必然不同，配套 descriptor 也不同，MMA 读到的地址则必然不同 ⇒ D 必须不同。
最简解释：**harness 自己的 D 读回/累加器复位在 case 之间没有真正生效**（例如 case 2 读到的仍是 case 1 的 TMEM），
于是"全 16 组合 FAIL 且逐位一致"只是同一个陈旧 D 被比了 16 次。
⚠️ 因此 §13 基于该 harness 得出的"非 swapAB 朝向算不出正确乘积"**不能作为定论**（它可能只是同一份陈旧输出）。
⇒ 判据必须换成**独立于该 harness** 的证据：a) e2e 文本（swapAB 近正确 vs 非 swapAB 乱码，见下）；
b) 全新、最小化的**冲激响应**探针（`kernels/cuda/tests_bs_impulse.cu`，由 subagent `bs-impulse-probe` 编写并实跑）。

### (b) `e2e` 的信号**仍然有效**（与 harness 无关）
同一台机器、同一份模型，仅改朝向开关：
- 非 swapAB（A1/A2/B1/B2）：随机乱码（"3600 720 探头…"、"7 8 9 10 11 12 17 18 24 27…"）
- **swapAB + canonical：`5 4 3 2 1 6 5 4`（递减连续数字）**；swapAB + SW128：`1 (0) 2 (0kie) 3 (0va)…`
⇒ 朝向确实改变了 MoE 数值（近正确 vs 随机），**swapAB 明显更接近正确**。

### (c) in-tree "已验证原语"的前提需要重新验证
`kernels/cuda/tests_tcgen05_mxf8f6f4_1x.cu`（Phase-0 探针，文件注释自称 round-trip VERIFIED）
在**本机也跑不过它自己的金标准**（subagent official-numeric-parity 实测）。而全树其余实现都是抄它的原语。
⇒ 要么注释过期、要么环境（nvcc 13.2 / ptxas / driver / sm_103a）变了。
**它是全树最省时的锚点**：先让它在本机通过，再谈其它。另：官方 TileLang 路径在同一台卡上能复现
float64 金标准到 bf16 精度（3.8e-3）⇒ 硬件与官方路径没有问题，问题只在我们的 block-scale 取数/SF 部件。

### 已确认为**无需改动**的一项
激活量化的输出与官方**逐字节一致**（Q2 对拍）；唯一可讨论的是全零块标度的下限语义：
官方 `max(amax, 1e-4)`（`kernel.py:76`）vs 我们 `max(scale, 1e-30)`（`dsv41_kernels.cu:156`）——仅在字面要求时才需对齐。

## §19 SF 字节序假设 + `DSV41_MOE_BS_SFREV` 门控（当前头号嫌疑）

**症状回顾**：e2e 上 swapAB 已把输出从**随机乱码**推到**递减的连续数字**（`5 4 3 2 1 6 5 4`），
说明朝向是主因、且**还剩一个较小的系统性误差**。而"误差在**所有**布局/朝向/SF 通道下表现一致"
这一签名，指向一个**所有组合共有**的部件。

**假设**：硬件读 SF 词时是 **MSB-first**（即 byte j 实际携带 K-block `3-j` 的标度），
而我们两侧都按 **LSB-first** 打包（`pack_wsf`（权重侧，装载期）与 gather（激活侧，运行期）都是
`byte j = 第 j 个 K-block`，见 §12 的表）。若成立，则每个 K-atom 都被乘了**别的 block** 的标度。

**为什么这个假设很有解释力**：它**与朝向、布局、SF 投递方式全都无关** ⇒ 能解释"误差在所有组合下形态相同"；
而它又不能解释"swapAB 明显好于非 swapAB" ⇒ 说明它是**叠加在朝向之上的第二个错**（两层错正好对应
"从乱码 → 递减数字"与"递减数字 → 完全正确"两步）。

**修法只有一处**（很优雅）：`hw_make_idesc` 的 `a_sf_id/b_sf_id` 取 `3-ki` 而不是 `ki` ——
因为两侧的词都是 `byte j = K-block j`，所以选 byte `3-ki` 会**同时**修正权重侧与激活侧。
已落地为门控 **`DSV41_MOE_BS_SFREV`**（默认 OFF，见 `moe_bs_handwritten.cu` 的 idesc 构造处注释）。

**决定性实验**（脚本已就绪：`~/sfrev_round.sh`）：
| 臂 | env | 目的 |
|---|---|---|
| A | `SWAPAB=1 CANON=1 GRAPH_STEP=0` | 拿 in-situ `[NC]` 数值（关图后 NUMCHECK 才触发，见 §16） |
| B | A + `SFREV=1` | SF 字节序假设 |
| C | B + `SCALEVEC1X=1` | 后缀与字节序的交互 |

判读：若 B/C 的 `[NC]` 从 `MISMATCH` 变 `ALL MATCH`（或文本变正确）⇒ 字节序就是残余根因。
若三条臂 `[NC]` 都 MISMATCH 且数值相近 ⇒ 换别的共有部件（descriptor 参数 / D 的 TMEM 行列映射）。

**另一条独立证据线**（subagent `bs-impulse-probe`，正在跑）：用**冲激响应** + **"只改第 j 个 scale 字节"
扫描**直接测出「A 侧 byte j → 生效的 K-block」与「B 侧 byte j → 生效的 K-block」两张表，
并自证 MMA 真的发射过（TMEM 预毒化 + 数据敏感性检查）。两条线互为印证。

## §20 中间结论：补栅栏**不能**修复"16 例逐位相同"

subagent `bs-impulse-probe` 的第一件事是给 micro harness 打了一个"补栅栏版"
（`tests_bs_mma_micro_fenced.cu`：把 §12 那三处 tcgen05 栅栏按 TileLang 的位置补齐），
重跑 ⇒ **仍然全错且 16 例逐位相同**。
⇒ **栅栏不是"逐位相同"的原因**（我原先的"harness 缺栅栏"假设被否证）。
⇒ 剩下的解释只有两类：
 (a) **MMA 根本没发射/没完成**（那么 D 读回的是从未被写过的 TMEM，16 次自然完全相同）——
     探针正在加"TMEM 预毒化 + 数据敏感性 + mbar 等待计时"三条自证；
 (b) MMA 发射了，但**读到的数据与布局无关**（这需要解释"两套写公式给出不同 smem 内容"为何不影响结果）——
     只有 (a) 或"读回恒为常量"能做到。

⇒ 在探针给出"MMA 确实发射过"的自证之前，**任何**基于 harness 的布局/朝向结论都不成立（§18 已述）。

## §21 精度对齐的具体实现路径（已查明的可行方案，待 BS 臂正确后执行）

§14 记录了两处不对齐。补查后，**路径已经清晰**：

### 关键事实
- `quant1_on(src, k, stream)` 处理的是**单行** `k` 个元素 → `s.xq`（`[dim]` e4m3）+ `s.xsc`（`[dim/32+8]` f32）
  （`chain_dev.rs:201-202` 的分配注释、`:5867`/`:5876` 的定义）⇒ **不能直接用于 routed 的 `[rows*topk][2*inter]` 矩阵**。
- 但**激活侧量化本身已经是批量的**：`dsv41_quant_fp8(x, y, s, m, dim, block=32, round_scale=true)`
  处理 `m` 行 × `dim` 列（这正是 BS 臂 gate/up 输入在用的那个）⇒ 对 routed down 只需**一次发射**：
  `dsv41_quant_fp8(ex_act_b, xq_dn, xsc_dn, rows*topk, 2*inter, 32, true)`。
- 障碍在下游：`dsv41_expert_down_reduce_fp4_batched` 的签名**只吃 f32**
  （`dsv41_experts_mxf4.cu:3833-3837`）⇒ 需要给它加 **AQ 支持**（e4m3 字节 + 标度两条入参），
  或改成调用已有 AQ 模式的 `expert_gemv_fp4_kernel`（M=1 的 gate/up 正走这条，
  `:751` + `:440` 的 `a_f32 // [rows,k] f32 (AQ=true)`）。
  ⚠️ 历史警告不变：把那条 GEMV 扩到 down 路径**曾把模型搞坏**（`:3268-3273`）⇒
  必须先把 down 的 `rows/dim/inter/act_stride/slots/ids/epi_mode` 语义与 epilogue 的组合钉死，再动。

### 顺序（必须与官方一致）
`silu(gate)*up` → **×route_w[slot]（per-slot 标量，官方在 w2 之前乘）** → `bf16` → `e4m3(block 32, ue8m0 ceil)` → down GEMM。
我们当前是 `... → bf16 → f32 → down GEMM → ×route_w`（在 epilogue）⇒ 两处都要挪/改。

### 代价
routed 路径多 1 次批量量化 + 1 次 per-slot 加权（可融合进量化前的 pass），约 +1~2 launch/step，µs 级 —— 对 400 tok/s 目标可忽略。

### 执行纪律
**等 BS 臂正确性钉死之后再动**（一次只改一个变量：先让模型输出正确，再改精度路径并单独验证）。

## §22 编译检查的正确做法（避免虚惊 + 避免浪费 GPU 窗口）

**事实**：`kernels/cuda/tilelang_gen/moe_bs_handwritten.cu` **不是独立 TU** —— 它被
`moe_bs_shim.cu:154` 用 `#include "moe_bs_handwritten.cu"` **文本包含**（因此它自己只 include
`<cuda.h>/<cstdint>/<cstdio>`，不 include TileLang 头；`build.sh` 里也**没有**它的单独规则）。
⇒ 单独 `nvcc -c moe_bs_handwritten.cu` **必然**报
`error: expected a ";"`（`tl::tcgen05_ld_32dp32bNx` 未声明），这是**假警报**，不是代码坏了。

**正确的一文件 compile-only 检查**（CPU only，可在没有 GPU 窗口时做）：
```bash
ssh ubuntu@43.202.208.136 'cd ~/ferrite/kernels/cuda/tilelang_gen && \
  nvcc -c moe_bs_shim.cu -o /tmp/shim_check.o -gencode arch=compute_103a,code=sm_103a \
       -O2 -std=c++17 -I. -I../tilelang_inc 2>&1 | tail -6; echo RC=$?'
```
（`SHIM_RC=0` = 手写 kernel + shim 全部编译通过。）

**本轮实测**：该检查在 `DSV41_MOE_BS_SFREV`（§19）落地后**通过** ⇒ 下一个 GPU 窗口不会因编译错误浪费。

## §23 【设计级事实】官方 `fp4_gemm` **不用**硬件 block_scale —— 这决定了我们的排查定位

读官方源码（`ref_inference/kernel.py:520-557`）确认，官方 fp4 GEMM 的主循环是：

```python
for k in T.Pipelined(K_iters, num_stages=2):          # block_K = 32
    T.copy(A[...k*block_K], A_shared)                  # e4m3 激活
    T.copy(B[...k*block_K], B_fp4_shared)              # fp4 权重
    for i, j in T.Parallel(block_N, block_K):
        B_shared[i, j] = T.Cast(FP8, T.Cast(FP32, B_fp4_shared[i, j]))   # fp4 -> e4m3（无损）
    for i in T.Parallel(block_N):
        scale_b_frag[i] = T.Cast(FP32, scales_b[bx*block_N + i, k])       # per-(n, K-block)
    for i in T.Parallel(block_M):
        scale_a_frag[i] = T.Cast(FP32, scales_a[by*block_M + i, k // n_sub])
    T.gemm(A_shared, B_shared, C_local, transpose_B=True)                 # ← 普通 FP8×FP8 MMA
    for i, j in T.Parallel(block_M, block_N):
        C_local_accum[i, j] += C_local[i, j] * scale_a_frag[i] * scale_b_frag[j]   # ← 标度在累加里显式乘
    T.clear(C_local)
```

**三条推论**：
1. 官方的数值 = `Σ_k (a·b) · 2^(sa[m][k/32]) · 2^(sb[n][k/32])`，**标度是在 f32 累加器里显式应用**的，
   用的是普通 MMA（**没有** `kind::mxf8f6f4.block_scale`、没有 TMEM SF 操作数）。
   数学上与我们用硬件 block_scale 想做的事**等价**——但**实现路径完全不同**。
2. ⇒ **硬件 block_scale 这条路在树内没有任何可用参考**：
   PH0 探针（唯一的"验证过"来源）在**本机跑不过它自己的金标准**（§18）；
   `tc5::e4`/`mxf4` 两臂在仓里也未被 GPU parity 证实。所以"SF 语义按我们的假设"这一点
   **从来没有被任何在本机通过的东西证明过**。
3. ⇒ 我们的残余误差（"哪个 K-block 的标度被用上"这一族问题）**正好落在没有参考的那一环**上，
   与 §19 的 SF 字节序假设同源。**这解释了为什么 16 组合微基准（含 PH0 风格的原语）全都失败。**

**因此排查的正确形态**（两条独立判据）：
- **A. 硬件路径是否正确**：冲激/字节映射探针（`bs-impulse-probe`，§20/§19）+ in-situ `[NC]` 数值（§16）——
  直接问"给定输入，硬件算出来的数对不对"。
- **B. 用官方当 oracle**：官方 `fp4_gemm` 在**同一台机同一样本**上能复现 float64 金标准到 bf16 精度（3.8e-3），
  所以它本身就是一个**可用的 oracle**；把同一批字节喂进我们的 kernel、逐元素比官方输出，
  看误差**在哪个维度上成规律**（行/列/K-block），比继续试配置快。

**性能上的权衡（备忘）**：若最终判定硬件 block_scale 在本机不可用，退路是**按官方的方式做**
（每 K-block 一次普通 MMA + f32 累加器里显式乘标度）。代价是每 K-block 需要独立累加（160 个 block），
要么用多个 TMEM 累加器 + epilogue 里做 rank-1 缩放（4 个累加器/128 列可行，但跨 40 个 k-iteration
的标度不同 ⇒ 需要每轮 TMEM 读回缩放 ⇒ 昂贵），要么接受更大开销。**优先把硬件路径搞对**。

## §24 决策树（两条独立探针的结论 → 立即动作）

### 探针 A：`bs-impulse-probe`（冲激响应 + SF 字节扫描 + MMA 存活自证）
| 结论 | 立即动作 |
|---|---|
| **MMA 根本没发射**（TMEM 预毒化后 D 不变） | 探针自身无效 ⇒ 改用 in-situ `[NC]` 路线（§16，`DSR41_GRAPH_STEP=0`）；我们的 kernel 的 MMA **确实在跑**（证据：e2e 输出随朝向/数据改变）⇒ 只修探针 |
| **A 侧 byte j ↔ K-block (3-j)** | **SF 字节序反了** ⇒ 直接用已落地的 `DSV41_MOE_BS_SFREV=1`（§19，一行 idesc 改动）跑 `~/sfrev_round.sh` |
| **byte j ↔ K-block j**（与我们假设一致） | SF 字节序无罪 ⇒ 转向 descriptor/取数约定：比探针给出的 (m0,k0)→(m,n) 表与 canonical 公式的预期，改 `hw_smem_idx` 或 descriptor 的 lbo/sbo/layout |
| **(m0,k0)→(m,n) 完全符合我们的公式** | 我们 kernel 的取数**没错** ⇒ 残余误差在别处（SF 之外的路径：TMEM D 的列顺序、epilogue 转置、或权重池寻址）⇒ 用 in-situ `[NC]` 与"官方 oracle"逐元素比 |

### 探针 B：`tl-blockscale-anchor`（TileLang 最小硬件 block-scale 参考）
| 结论 | 立即动作 |
|---|---|
| **跑通**（本机第一个可用参考） | 拿它 dump 的生成 CUDA 提取**权威约定**（smem 写公式 / descriptor lbo,sbo,layout / idesc 全字段 / SF 投递方式与 TMEM 列偏移 / K-block 递进量）⇒ 逐项改我们的 kernel（这是最省时的路径） |
| **跑不通**（与 PH0/moe_bs_up_tl 一样失败） | 硬件 block_scale 在本机/本驱动**不可用** ⇒ 执行 §23 的退路：按官方方式（普通 FP8 MMA + f32 累加器显式乘标度）重做 gate/up，性能代价需重新评估（但正确性优先） |

### 共同前提
- 任何 in-situ 结论都必须带 **`DSV41_GRAPH_STEP=0`**（否则 shim 在捕获期 decline，BS 臂只在非捕获调用里跑）。
- 我们的 kernel 的 MMA **确实在发射**（e2e 输出随朝向/数据变化）；因此"探针里 MMA 不发射"是**探针侧**问题，不推翻 e2e 结论。

## §25 运维陷阱：build-id 不匹配 ⇒ serve 拒绝启动（开 GPU 窗口前必查）

**现象**：起了 serve 但 18 秒就退出、日志里**一条 `[moe-bs]` 都没有**（健康检查拿不到 200）。
**日志原文**（`~/armrun_*.log`）：
```
[single-flight] engine fault: config error: kernel build-id mismatch — REFUSING TO START
(so and binary must be the same build): .so build_id=…-dirty+cu5dc9942ebee25707
vs binary build_id=…-dirty+cuf683f43bb06c66f5. Rebuild both from the same checkout:
  `cd kernels/cuda && bash build.sh 103a && cd ../.. && cargo build --release`
```
**根因**：`.so` 与二进制各自记录的 **kernel 源码哈希不同**，运行期防线（三道 runtime 防线之一）直接拒绝启动。
本次的具体成因：我在**预编译尚未结束**时就开了 GPU 窗口，而且期间 subagent 往 `kernels/cuda/` **新建了文件**
（`tests_bs_impulse.cu` 等）⇒ 两次构建看到的树不一致（注意 build_id 里带 `-dirty`，未跟踪文件也算进哈希）。

**纪律（新增）**：
1. **开 e2e 窗口前先确认 `.so` 与 binary 的 mtime 都已就绪**（`stat -c %y`），别在预编译 in-flight 时开窗口。
2. **双产物必须背靠背、在同一条命令里产出**，中间**不允许**任何 subagent 往 `kernels/cuda/` 或 `crates/` 写文件；
   多 subagent 并发时，**先让写代码/写测试的 subagent 收尾**再重编，或把它们的产物放在 `/tmp`/`~/` 而非仓库里。
3. `arm_run.sh` 的 `SERVE_FAILED` 分支要用 `tail` 显示日志尾部（本次 `tail -26` 恰好截掉了关键行，是靠事后
   手动 `tail ~/armrun_A_ng.log` 才看到 build-id 那行）——**判读 e2e 失败时永远先看 serve 日志尾部**。

## §26 idesc 位域全貌核实 + 两处独立交叉验算（已确认无误）

`moe_bs_handwritten.cu` 的 `hw_make_idesc(m, n, a_fmt, b_fmt, sf_id)` 完整构造：

```c
d |= (sf_id & 3) << 4;        // b_sf_id   [4,6)    ← §10 修的那个漏 <<4 的 bug
d |= (a_fmt & 7) << 7;        // a_format  [7,10)   0 = E4M3
d |= (b_fmt & 7) << 10;       // b_format  [10,13)  5 = E2M1
d |= ((n >> 3) & 63) << 17;   // n_dim     [17,23)  N/8
d |= 1u << 23;                // scale_format = UE8M0
d |= ((m >> 4) & 31) << 24;   // m_dim     [24,29)  M/16
d |= (sf_id & 3) << 29;       // a_sf_id   [29,31)
```

**关键**：`a_sf_id` 与 `b_sf_id` 是**两个独立字段，但都取自同一个 `sf_id` 形参** ⇒ §19 的 SFREV 一行改动
（`sf_id = 3-ki`）**同时覆盖 A、B 两侧**（这正是它优雅的原因）。

**两处独立交叉验算（均吻合，可复算）**：
| 配置 | 我们的值 | 独立来源 | 结论 |
|---|---|---|---|
| swapAB（a_fmt=5,b_fmt=0,M=N=128,sf=0） | `0x08A00280` = 144,704,128 | 树内已验证探针 `0x08820280`（M=128,N=8）+ `n_dim` 差（N=8→128 ⇒ `15<<17` = 1,966,080） | **逐位吻合** ✓ |
| 非 swapAB（a_fmt=0,b_fmt=5,0x0A01400 结构） | `0x08A01400` | TileLang 生成码常量（`moe_bs_up_tl.cu`） | **逐位吻合** ✓ |

## §27 ⚠️ 自我更正：早先"A 朝向更好"的对比**混了两个变量**，必须重做受控实验

**混淆**：
| 臂组 | 构建 | 含栅栏 | 朝向 |
|---|---|---|---|
| A1/A2/B1/B2（`~/matrix_test.sh`，78e58770） | **039235b** | 只有 `fence.proxy.async`（缺 3 处 tcgen05 thread-sync 栅栏） | 非 swapAB |
| S1..S4（`~/matrix2_test.sh`）、swapab_canon | **8637ff9** | **完整**（4 处都补齐） | swapAB |

⇒ "swapAB 明显更好"的结论**同时混了"朝向"与"补齐了 tcgen05 栅栏"两个变量**，**不成立**。
更不利的证据：B1（**非 swapAB**、canonical、部分栅栏）当时给出的文本是**升序**的
`7 8 9 10 11 12 17 18 24 27 29 30 31 33 40 42 47 51 53`，从"从 1 数到 10"这个任务看，
**升序比 swapAB 那一臂的降序 `5 4 3 2 1 6 5 4` 更接近正确**。

**因此必须以 `[NC]` 数值为主判据重做受控实验**（文本只是辅助）：
`~/orient_controlled.sh`（四条臂，**同一份新构建** + `DSV41_GRAPH_STEP=0` 让 NUMCHECK 生效）：
| 臂 | env | 控制什么 |
|---|---|---|
| N1 | `CANON=1` | 非 swapAB 基线（完整栅栏） |
| N2 | `CANON=1 SWAPAB=1` | 只变朝向 |
| N3 | `CANON=1 SWAPAB=1 SFREV=1` | 再变 SF 字节序 |
| N4 | `SWAPAB=1` | 只变布局（SW128 vs canonical） |

判读：比较四臂的 `[NC] WORST rel=` 数值——**数值最小的那一臂才是"更接近正确"**，
不要去比文本"像不像在计数"（那会被模型的退化模式误导）。

## §28 `[NC]` 数值的定量指纹（可检验的预测）

NUMCHECK 报的 `rel = |ref - mma| / max(|ref|, 1e-6)` 的**量级**本身就是判据，可以对号入座：

| 观测到的 rel 量级 | 指向 | 依据 |
|---|---|---|
| ~1e0 – 1e2（即误差 ~100% – 10000%） | **标度取错块**（SF 的字节序/块映射错） | K-block 标度是 2 的幂、范围约 2^-3..2^3（我们造的数据）/ 实际 checkpoint 更宽 ⇒ 用错块的标度会让每项差 2^k 倍，累加后相对误差落在 10^0–10^2 量级 |
| ~1e-1 | 部分项错（例如只有某些 K-block 或某些行/列错） | 结构性的部分错 |
| ~1e-2 – 1e-3 | 基本正确（e4m3 量化级舍入） | 与官方"复现 float64 金标准到 bf16 精度 3.8e-3"同量级 |
| ~2 或 ~0.5（整倍） | **标度被乘了两次 / 漏乘一次** | 对称的整倍偏差 |

**已知参考点**：不可信的 16 组合微基准当时报 `relerr = 9.2376e+01`（≈9200%）——落在"标度取错块"那一档，
与 §19 的 SF 字节序假设**相容**（注意：该 harness 的绝对值不可信，但量级仍可作为旁证）。

**下一步**：拿到 in-situ `[NC]` 的 `WORST rel=` 后，先按本表定档，再决定是查 SF（§19）还是查别的共有部件。

# 🎯🎯 §29 根因定谳：硬件把 fp4(E2M1) 操作数按 **PACKED（2 元素/字节）** 读，我们按 unpacked（1 元素/字节）写

**这是整条排查链的终点。** 由 subagent `bs-impulse-probe` 用冲激响应 + 字节敏感性实验实证（非推导）。

## 决定性证据（三条互相独立）
| 观测 | B 字节 = `0x02`（低 nibble=2=1.0，高 nibble=0） | B 字节 = `0x22`（两个 nibble 都=1.0） |
|---|---|---|
| A 侧 128 点 **k 扫描** | **64/128 响应**，且**恰好是全部偶数 k** | **128/128 响应** |
| **稠密 parity**（金标准 128） | D = **64**（正好 1/2） | D = **128，relerr=0，精确 PASS** |
| `sfprobe`（金标准 2720） | D = **1360**（正好 1/2） | — |

**机制**：硬件把 B 的**每个字节劈成两个 K 元素**（低 nibble = 偶 k、高 nibble = 奇 k）。
我们的 unpacked staging 里字节 `0x02` 的高 nibble 恒为 0 ⇒ **所有奇数 K 元素恒为 0、K 覆盖腰斩** ⇒
稠密 GEMM 只有一半 K 参与 ⇒ D 恰好是金标准的 1/2、逐元素全错；而单冲激（值 1.0，落在偶 k）依然精确
⇒ 这正是"**16 组合逐位相同且全错**"的形态（16 个组合没有一个改动 B 的打包方式）。

**独立旁证（官方尺寸自洽）**：`moe_bs_up_tl.cu:88-93` 的 W1/W3 子 tile 各 **8192 B**；
官方 idesc = `0x08A01400`（`n_dim=16` ⇒ N=128）⇒ 128 行 × **64 B** = 8192 B ✓ = **packed fp4**（K=128×4bit）；
若 unpacked（128 B/行）则需 16384 B ✗。

## 其它一切**都已验证正确**（勿再怀疑）
- **36 例冲激 `(m0,k0)` 扫描全部命中**（`m0 ∈ {0,1,7,8,31,32,63,64,127} × k0 ∈ {0,32,64,96}`）：
  响应恰好是 **row m0 全 128 列、值精确 1.0000、其余全 0** ⇒ M/N/K 三维的原子与步长**零错位**。
- **原始物理字节偏移扫描**：A 的 smem 读法与 canonical 公式**逐字节一致**。
- **descriptor 字段语义**：`sbo=16`(256 B) → 128/128 行响应；`lbo=8`(128 B) → 64/128；K 递进 4096 B 使 4 个 K atom 全部命中。
- **存活自证**：`tcgen05.st` 把 D 的 128 列毒化成本行 `1000+row`、**不发射 MMA** → 16384/16384 逐元素读回正确
  ⇒ TMEM 读回与 lane/column 映射可用；有 MMA 时毒化被完全覆盖 ⇒ **MMA 确实发射并写 D**；`mbar_wait_cycles ≈ 434–486` ⇒ 等待真的阻塞。
- **栅栏不是根因**：补齐三处 tcgen05 栅栏后 16 组合输出**逐位不变**；`-DIMP_NO_FENCES` 下探针同样 PASS。

## PH0 探针的 FAIL 也同源（"验证过的原语"前提正式推翻）
`tests_tcgen05_mxf8f6f4_1x.cu` 本机 FAIL（`max|diff|=36448.6`，`[FAIL] layout/scale mismatch`，
case 2/4 还出现粘性 `misaligned address`）。它是 swapAB 朝向（fp4 在 A 侧），其 A tile 是
`128 行 × 32 B = 4096 B` 的 **unpacked** staging ⇒ 同样被硬件按 packed 读 ⇒
**它不是独立反例，而是同一缺陷的镜像**。

## 修法（待收尾的那一步）
把 fp4 操作数在 smem 里按 **packed（2 元素/字节，64 B/行 for K=128）** staging，
并相应改 descriptor（LBO/SBO）与 K-block 递进。探针已量到硬约束：
**一个 `scale_vec::1X` K-block（32 元素）= B 侧 16 字节** ⇒ B 的 K-block 递进应是 **128 B（8 units）** 而非 4096 B，
且 row-group 需容纳 4 个 chunk。探针试了 v1（16 B-chunk 家族 + 递进 `(kb/2)*4096+(kb%2)*128`）：
`relerr/|ref|` 从 1.49 降到 **0.46**、`D[0][0]` 已正确，但部分行仍差一个 K-block（`D[0][n]=96/112` vs 128）。
⇒ **权威 ground truth 应从官方 TMA 规格取**：`moe_bs_shim.cu:370-405` 的 W1 spec（box/swizzle/dtype，
`dtype 14 = 16U4_ALIGN16B`）或 `moe_bs_up_tl.cu:88-123` 的 TMA 描述符。

**注意**：修好之后，**朝向（swapAB）与 SF 字节序（SFREV）都要用受控实验重新判定**（§27），
因为之前所有基于"全错"文本的结论都是在 B 打包错的条件下得到的。

## §30 packed 几何的独立推导（对 §29 修法的预测，待 subagent 实测校验）

**已知锚点**（都是树内的硬事实）：
1. 根因（§29）：硬件按 **packed（2 元素/字节）** 读 fp4 操作数，低 nibble = 偶 k。
2. 官方 W1/W3 子 tile = **8192 B** = 128 行 × **64 B**（K=128 packed）⇒ **无 padding 的稠密 64 B/行**。
3. 官方给 **W 操作数**的 descriptor（`moe_bs_up_tl.cu:115`，由前一个 subagent 摘出）：
   **`lbo=1、sbo=64、layout_type=2`（SWIZZLE_128B）**。
4. 探针实测的硬约束：一个 `scale_vec::1X` K-block（32 元素）= B 侧 **16 字节**。

**推导**：
- 16 B = 32 个 packed 元素 = **正好一个 K-block** ⇒ **`lbo=1`（16 B）就是 K-block 的步长**，即
  **连续 K-block 在 smem 里是紧邻的**（row-major packed，行内 4 个 16 B chunk）。
  ⇒ **K-block 的 descriptor 递进量 = 16 B = 1 unit**（不是 e4m3 那种 32 B，也不是我们之前用的 4096 B）。
- `sbo=64`（1024 B）：在"2 行占一个 128 B swizzle span"的形态下，8 个 span = **16 行** ⇒
  **SBO 的语义仍是"行组步长"，但 packed 形态下每个行组是 16 行**（e4m3 形态下是 8 行）。
- 由此的**预测**：
  | 参数 | 预测值 |
  |---|---|
  | smem 行字节数（K=128） | **64 B** |
  | LBO | **1 unit（16 B）** |
  | SBO | **64 units（1024 B）** |
  | layout_type | **2（SWIZZLE_128B）** |
  | K-block 递进 | **ki × 16 B（ki × 1 unit）** |
  | 行内 chunk 置换 | 应在 4 个 16 B chunk 内按行号 XOR（而非 e4m3 的 8 chunk 全 span XOR），因为 64 B 行只占半个 128 B span |
- **e4m3 侧（A 操作数）不变**：它仍是 1 B/元素、128 B/行，`lbo=1/sbo=64/layout=2`、递进 32 B ✓（§26 已核对）。

**校验方式**：`bs-packed-geometry` subagent 的实测（要求 `const` 稠密 parity **relerr=0** 且 `sweep1d k`
**128/128**）若与本预测一致 ⇒ 直接落地；若不一致 ⇒ 以实测为准，并回头修正本节的推导（把差异原因记下来）。

## §31 packed 修复的实施清单（几何定标后机械落地）

**改动面**：只在 `kernels/cuda/tilelang_gen/moe_bs_handwritten.cu`（+ 若需要新门控则加 shim 的 setter）。

**新增门控**：`DSV41_MOE_BS_PACKED`（默认先 OFF，跑通受控对比后再考虑转正）。
**注意**：swapAB 下 fp4 操作数在 **A_sh**、否则在 **B_sh**；e4m3 操作数路径**完全不动**。
即门控只影响"fp4 那一个操作数"的 (a) staging、(b) descriptor 参数、(c) K-block 递进。

### 1) staging（fp4 操作数）
```c
// 现在（错）：把一个 packed 字节拆成 2 个 1-byte 元素写成 unpacked
//   (g_swapab ? A_sh : B_sh)[hw_smem_idx(row, k0, g_canon)] = packed & 0xF;
//   (g_swapab ? A_sh : B_sh)[hw_smem_idx(row, k1, g_canon)] = packed >> 4;
// 改成（对）：**原字节一次写入**（2 元素/字节），地址用 packed 几何
if (g_packed) {
    (g_swapab ? A_sh : B_sh)[hw_pack_idx(row, col)] = packed;   // col = packed 字节下标 0..63
} else { /* 保留旧路径，供受控 A/B 对比 */ ... }
```
- 行字节数：**K/2 = 64 B**（K=128）—— tile 从 16384 B 降到 **8192 B** ✓（与官方 W 子 tile 一致）。
- `hw_pack_idx(row, col)` 的确切 swizzle 形式**以 `bs-packed-geometry` 的实测为准**；
  §30 的预测是"16 B chunk 在行内按行号 XOR"。

### 2) descriptor 参数（fp4 操作数）
预期（§30 推导）：**`lbo=1（16 B）、sbo=64（1024 B）、layout_type=2（SWIZZLE_128B）`** ——
即与 e4m3 侧的 `lbo=1, sbo=64, layout=2` **数值相同**（这也与官方给 W 的 descriptor 一致）。
⚠️ 若实测给出不同的 lbo/sbo，以实测为准。

### 3) K-block 递进（fp4 操作数）
**`ki × 16 B`（ki × 1 unit）** ---- 因为一个 K-block（32 元素）在 packed 形态下就是 16 B（= LBO ✓）。
（e4m3 侧仍是 `ki × 32 B` = `ki × 2 units`，不动。）

### 4) 不改的部分（已核对）
- `hw_make_idesc` 的 `b_format`/`a_format` **不变**（`bs-impulse-probe` 的 PASS 用的就是现有 idesc `0x08A01400`）；
- e4m3 操作数的 staging / descriptor / 递进不变；
- SF 的打包与投递（`pack_wsf` / gather / transpose / `tcgen05.cp`）不变；
- epilogue / scatter 不变。

### 5) 验证顺序（**每次只改一个变量**）
1. 先只改 fp4 staging + descriptor + 递进（门控 ON），**朝向与 SF 字节序都保持现状**；
2. 用 `~/arm_run.sh` + **`DSV41_GRAPH_STEP=0`** 拿 in-situ `[NC]`：期望 `WORST rel` 从"1e0–1e2 档"掉到 **≤1e-2**（§28 的指纹表）；
3. `[NC]` 达标后再用 `~/verify_correct.sh` 验文本（1..100 前 61 行）——**文本只是辅助判据**；
4. **然后**才用 `~/orient_controlled.sh` 重新受控判定朝向（SWAPAB）与 SF 字节序（SFREV）——
   因为之前所有"全错"的文本结论都是在 B 打包错的条件下得到的（§27）。

## §32 两条新知识（编译检查环境 + packed 几何的候选 B/C）

### (a) 编译检查必须在**仓库目录**里做
把 shim/kernel 拷到 `/tmp/xxx` 再 `nvcc -c` 会报一堆**假**错误
（`identifier "tl_bs_init" is undefined` / `kBm` / `kSegCap` / `kMovThreads` undefined），
因为仓库目录里还有 shim 需要的其它头/宏。**正确做法**（本次实测 `PK_RC=0`，产物 164 KB）：
```bash
ssh ubuntu@43.202.208.136 'cd ~/ferrite && git fetch -q origin && \
  git checkout -q origin/main -- kernels/cuda/tilelang_gen/ && \
  cd kernels/cuda/tilelang_gen && \
  nvcc -c moe_bs_shim.cu -o /tmp/pk_shim.o -gencode arch=compute_103a,code=sm_103a \
       -O2 -std=c++17 -I. -I../tilelang_inc 2>&1 | grep -E "error" | head -6; echo RC=${PIPESTATUS[0]}'
```
（§22 记过"要编 shim 而不是编 kernel"；本节补上"要在仓库目录里编"。）

### (b) packed 几何：候选 B / C（有原理，待实测定标）
从三条实测硬事实反推：① 硬件 2 元素/字节；② 一个 `scale_vec::1X` K-block（32 元素）= **16 B**；
③ 官方 W 子 tile = **8192 B = 128 行 × 64 B**。
再由 UMMA 的"core matrix 原子 = 8 行 × 16 B"（packed 下 16 B 恰好 = 一个 K-block）得：

| 候选 | 写公式（`kb = k/32` = K-block 下标） | lbo | sbo | layout | K-block 递进 |
|---|---|---|---|---|---|
| **B**（主推） | `byte(row,kb) = (row%8)*16 + kb*128 + (row/8)*512` | **8**（128 B） | **32**（512 B） | 0（SWIZZLE_NONE） | `ki*128 B` = `ki*8` |
| **C**（备选） | 同 B | 8 | 32 | **1**（SWIZZLE_128B_BASE32B） | 同 B |
| ~~§30 的猜测~~ | 行内 16 B chunk 按行号 XOR | 1 | 64 | 2 | `ki*16 B` | ← 已被本节的推导取代（但仍保留为对照） |

判据同前：`const` 稠密 parity `max|D-ref| = 0` 且 `sweep1d k` 128/128。

## §33 图关掉后的两臂结果（无 `[NC]`，输出退化为制表符）

**`DSV41_GRAPH_STEP=0` 确实改变了行为**（证明图捕获确实挡着 BS 臂的一部分调用）：
- 两臂输出都变成 **`'\n\t\t\t\t…'`（纯制表符）**，比之前"递减数字"更退化 ⇒
  BS 臂现在在**更多调用**里生效，而它的 **B 打包是错的**（§29）⇒ 模型更快崩坏。
- `step pos=62: 375.09ms / 227.21ms`（2.7–4.4 tok/s）——图关掉后自然慢，**不可用于性能判断**。
- **`[NC]` 仍未打印**（两臂都没有）⇒ 已加一次性入口标记
  （`[NC] entered (capture/guard check next; stream=…)`）以区分"没进到那里"与"进了但静默早退"
  （后者最可能是 `tl_bs_numcheck` 里的某个 `cudaMemcpy` 失败）。**下一次构建会带上这个标记。**

**判读**：这一轮的价值 = 证明"图捕获挡调用"这一环确实存在（§16），且**在 B 打包修好之前，
任何文本结论都不可信**（现在文本已退化为纯制表符）。⇒ 优先级回到 **packed 修复**（§29–§32），
`[NC]` 数值只作为修好后的定量确认手段。

## §34 packed 几何的**必要条件**已核验（两套候选都是合法排布）

对 §32 的两套候选写公式做了纯算术核验：
| 公式 | 双射 | 覆盖 | 最大 offset |
|---|---|---|---|
| geom0（§30 猜测）：`(row>>3)*512 + (row&7)*64 + (((col>>4)^(row&3))<<4) + (col&15)` | ✓ | 8192/8192 | 8191 |
| geom1（候选 B）：`(row&7)*16 + (col>>4)*128 + (row>>3)*512 + (col&15)` | ✓ | 8192/8192 | 8191 |

⇒ 两者都是 128×64 字节的**合法双射排布**、都恰好占 **8192 B**（与官方 W 子 tile 一致）。
**区别只在 descriptor（lbo/sbo/layout）与 K-block 递进语义** —— 这正是要靠实测定标的那部分。
⇒ 已做成运行期可切换（`DSV41_MOE_BS_PACKGEOM`，见 §32），并在
`~/packed_matrix.sh` 里准备了 packed 版的受控矩阵（geom × 朝向 × SF 字节序，共享同一份构建、
`[NC]` 为主判据、文本仅作参考）。

## §35 `[NC]` 不打印的真正原因（补充 §16）：**per-op 图捕获**也必须关

树内一共有 **5 个图开关**（grep 结果）：`DSV41_GRAPH_STEP`（整步）、以及
`FERRITE_GRAPH`、`FERRITE_GRAPH_LAYER`、`FERRITE_GRAPH_MOE`、`FERRITE_GRAPH_MID`、`FERRITE_GRAPH_DSA`（per-op）。
**只关 `DSV41_GRAPH_STEP` 不够**：per-op 捕获仍然会把 BS shim 的那次调用**套在捕获里** ⇒
shim 按既有设计 `return 2`（decline），同时 NUMCHECK 的 `cudaStreamIsCapturing == none` 守卫也过不去 ⇒
`[NC]` 永不打印（这正是 §33 观察到的现象）。

**诊断/探针臂必须带全套**：
```
FERRITE_GRAPH=0 FERRITE_GRAPH_LAYER=0 FERRITE_GRAPH_MOE=0 FERRITE_GRAPH_MID=0 FERRITE_GRAPH_DSA=0
DSV41_GRAPH_STEP=0
```
（已写进 `~/packed_matrix.sh` 的 `COMMON`。另：`arm_run.sh` 已加"失败时打印决定性日志行"，
下次若有 build-id/decline 一类问题会直接看到。）

## §36 运维坑：`git apply --3way` 在 linked worktree 场景"报告成功却没落地"

**现象**：在隔离 worktree `/tmp/prec-align` 里 `git diff HEAD -- crates kernels > /tmp/prec_code.patch`，
回到主工作树执行 `git apply --3way /tmp/prec_code.patch`，它逐文件打印
`Applied patch to 'crates/.../chain_dev.rs' cleanly.` ⇒ 看起来成功；
但 `git diff --stat` 为空、`git diff --cached --stat` 为空、工作树里 `grep` 不到新符号
（`routed_down_prep` / `DSV41_ROUTED_DOWN_QUANT` 全 0）⇒ **实际什么都没改**。
（`--3way` 隐含 `--index`，在 linked worktree 上的行为与预期不符；不再深究。）

**确定性替代（本次采用，已验证）**：直接把 worktree 里**已打好补丁的文件内容**拷回主工作树——
```bash
cd /home/smith/src/ferrite
for f in crates/ferrite-models/src/dsv41/chain_dev.rs \
         crates/ferrite-models/src/dsv41/device.rs \
         crates/ferrite-models/src/dsv41/kernels.rs \
         kernels/cuda/dsv41_glue.cu; do cp /tmp/prec-align/$f $f; done
grep -c routed_down_prep kernels/cuda/dsv41_glue.cu crates/ferrite-models/src/dsv41/device.rs   # 3 / 8 ✓
git diff --stat | tail -6    # 4 files changed, 797 insertions(+), 8 deletions(-) ✓
```
**纪律**：跨 worktree 搬代码时，**必须**用 `grep -c <新符号>` + `git diff --stat` **双向确认落地**，
不要相信 apply 的"cleanly"字样。

## §37 历史语义旁证（印证根因**在权重侧**）

shim 里能力符号 `dsv41_moe_bs_act_e4m3_cap()` 的注释（`moe_bs_shim.cu` §8 导出符号 3）记载：

> "D2 修复把 `xq4` 的**语义**从「packed fp4 半字节（dim/2 B/行）」改成「e4m3（dim B/行）」，
>  但 **C ABI 的形状没变**（还是 `const uint8_t*` + 同样的形参序）⇒ 旧 `.so` 会**静默**把
>  5120 B 的行当 2560 B 的 fp4 读（错值，不是报错）。"

⇒ **历史上激活侧**曾是 fp4-packed 半字节（`dim/2` B/行，**正是 packed**！），后来被改成 e4m3（`dim` B/行）。
即：**"packed 半字节"在这条代码线里本来就是既有语义**，只是用在了激活侧；
而**权重侧的 fp4（W1/W3）一直是 fp4**，我们却把它按 **unpacked（1 字节/元素）** staging 到 smem
——这正是 §29 定谳的那处错，且与"历史上 packed 语义确实存在"互相印证。

**教训**：当 C ABI 的形状不变而语义变过（如上面 D2 那次），**必须靠能力符号/断言钉死语义**，
否则就是"静默错值面"。我们对 W 侧的 staging 正是踩在同类面上：**形状对了（都是 u8 指针），
语义错了（packed vs unpacked）**。

## §38 候选 B 的**最强原理依据**：它是树内 canonical 结构的 packed 泛化

树内自己的 blockscaled 实现（`dsv41_experts_mxf4.cu:4247-4284`）给出 canonical fp4 操作数布局：
```
constexpr int kLboBytes = 128;  // 8 x 16 B: K-chunk stride of the K=32 atom
constexpr int kSboBytes = 256;  // 16 x 16 B: 8-row-group stride
// a_op 是 canonical UMMA Major-K SWIZZLE_NONE 操作数布局，每个 K_STEP 一个 16-byte-chunk-atom：
//     unit16(m, kb) = (m % 8) + 8*kb + 16*(m / 8)      kb in {0,1}
// 即 LBO = 128 B（两个 K-chunk 之间）、SBO = 256 B（8 行组之间）
```
**⚠️ 关键**：它同处注释写明 `A = 128 rows x 32 B **unpacked** fp4 = 4096 B`（每 16 B chunk = 16 个元素）
⇒ **它也是 unpacked**，所以它同样会被 packed 硬件读错（再次印证 §18/§29"树内无可用参考"）。

**把该结构泛化到 packed（16 B chunk = 32 个 packed 元素 = 一整个 K-block）**：
- 每个 K-block（32 元素）恰占 **16 B** ✓（与探针实测"一个 `scale_vec::1X` K-block = 16 B"**完全一致**）；
- 一行 K=128 = 4 个 K-block = 4 个 chunk = 64 B ✓（与官方 W 子 tile 8192 B = 128×64 一致 ✓）；
- 一个 8 行组 = 8 行 × 4 chunk × 16 B = **512 B** ✓ ⇒ **SBO = 32 units**；
- 相邻 K-block 相距 **128 B** ⇒ **LBO = 8 units**（与树内 `kLboBytes = 128` **同值**！因为"K-chunk 步长"在
  两种形态下都等于 128 B，只是 chunk 的**含义**从"16 个元素"变成"32 个元素"）。

⇒ **候选 B = 把树内 canonical 公式按 packed 语义重算的必然结果**：
`byte(row, kb) = (row%8)*16 + kb*128 + (row/8)*512`，`lbo=8, sbo=32, layout_type=0, 递进 ki*128 B`。
（对比：候选 D 走的是"TMA swizzle（SWIZZLE_64B）"路线，与 canonical interleave 是两套不同编码；
两者各自内部自洽，**由实测决定硬件接受哪一套**。§34 已验两者都是 8192 B 的双射。）

**结论**：**候选 B 的优先级应高于 D/geom0**（它是"树内唯一 canonical 结构的正确泛化"，
而 geom0/D 都含我推测的 swizzle 假设）。已同步给 `bs-packed-geometry` 以便其仪器优先扫 B。

## §39 packed 修复的两个副产品（验收方案 + 性能红利）

### (a) 精度门控的验收方案（已就绪，等 BS 臂正确后执行）
补丁自带的 DBG 回读（`DSV41_ROUTED_DOWN_QUANT=1 DSV41_ROUTED_DOWN_QUANT_DBG=1`）会打印**同一 (row, slot) 的 32 个元素**
的 5 组值：① 加权前 ② 加权后 ③ bf16 取整后 ④ e8m0 标度字节 ⑤ 量化→反量化后；
并同时打印**主机侧按官方语义独立算的同一 5 组值**与逐元素差。
**判据**：①–⑤ 逐元素差应为 0（除 ±1 ulp 舍入）。
**然后**（同样的 env）跑 `~/verify_correct.sh <port> <label>`：1..100 前 61 行 + 拉丁探针 + step p50，
要求"不能重复、不能乱码"（用户红线）且文本与 **gate OFF** 的对照一致（官方语义下输出应几乎不变，
因为差异只有 0.2% vs 1-2% 的量化误差）。

### (b) 性能红利（顺带）
packed staging 后，fp4（权重）操作数的 smem tile 从 **16384 B 降到 8192 B**（每行 128 B → 64 B），
而权重侧是每 k-iteration 每 n_tile 都要重新装载的那个面 ⇒ **权重装载的 smem 写入流量减半**
（40 k-iter × 5 n_tile × 128 行）。这是"修正确"顺带带来的收益，不计入 §10 的性能预期也应当出现。
⚠️ 注意：**几何必须与写公式成对**（B 配 canonical、D 配 SWIZZLE_64B），混配必然错——
这也是 §34 强调"两者各自内部自洽"的原因。

## §40 packed 候选的实测证据链（截至 E 回合前）

全部在**同一份构建 + 5 个图门全关**（⇒ BS 臂在每次调用都生效）下取得，prompt 为"请从 1 数到 10，每个数字单独一行"：

| 臂 | 几何 | 朝向 | 输出（前 60 字符） | 判读 |
|---|---|---|---|---|
| （打包修复前，unpacked） | — | swapAB+canon | `'\n\t\t\t\t\t\t…'`（**纯制表符**） | 完全退化 |
| P0 | geom0（§30 猜测：lbo=1/sbo=64/layout=2） | swapAB+canon | `' OR "OW  或者是: \t (  + \t:   +   :    \t. 在 \t:'` | **混合结构化文本**（有引号/中文）⇒ packed 生效 |
| P1 | **候选 B**（canonical 泛化：lbo=8/sbo=32/layout=0） | swapAB+canon | `'\n( \t \t \t,  \t, 6, 6, 7, 7, 7, 7, 7, 7, 7,'` | **出现数字**（6/7 重复）⇒ 目前最接近"计数" |

**趋势**：unpacked（纯制表符）→ geom0（混合文本）→ **候选 B（数字）** ⇒ 每一级都在向"正确"靠近，
说明 **packed staging 这个方向是对的**（§29），剩下的只是**几何配对**（写公式 ↔ descriptor ↔ 递进）。

**E 回合（running，`3adc5499`）**：E1 = 候选 E（**朴素行主序** `row*64+col` + lbo=1/sbo=32/layout=0，
依据官方 TMA 实参 §32/§39），E2 = 同 E 但**不 swapAB**（§27 要求重新受控判定朝向），
E3 = 候选 D（SWIZZLE_64B）。本轮同时应首次拿到 **`[NC]` 数值**（5 图门已全关 + 6 处静默早退已改为打印 ABORT）。

## §41 候选 B 与 E 都是"自洽编码"——差别在硬件对 SWIZZLE_NONE 的真实期望

对两套候选做了一次严格的自洽性核验（写公式 ↔ LBO/SBO ↔ 递进 必须互相印证）：

| 候选 | 16 B chunk 的位置 | 推出的 LBO | 推出的 SBO | 自洽 |
|---|---|---|---|---|
| **B**（canonical 泛化） | `(row%8)*16 + kb*128 + (row/8)*512`，kb = col/16 | 128 B = **8 units**（chunk 间距） | 512 B = **32 units**（8 行组 = 8×4 chunk×16 B） | ✓ |
| **E**（朴素行主序） | `row*64 + kb*16` | 16 B = **1 unit** | 512 B = **32 units**（8 行 × 64 B） | ✓ |

两者**内部都自洽**（行组都恰好 512 B，无重叠；§34 已验双射），所以**单看数学无法判别**。
判别点在于**硬件对 `layout_type=0`（SWIZZLE_NONE）的真实期望**：

- 树内 canonical 实现（`dsv41_experts_mxf4.cu:4247-4284`）用的是**"8 行 × 16 B 原子"**结构
  （`unit16(m,kb) = (m%8) + 8*kb + 16*(m/8)`，即**行与行之间 16 B**、chunk 之间 128 B、行组 256 B），
  那是**每行 2 个 chunk**（unpacked，K=32 原子 = 2×16 元素）的形态 ⇒ 行组 = 8×2×16 = **256 B**（SBO=16 ✓）。
- **packed 下每个 K=32 原子只占 1 个 chunk** ⇒ 行组的 chunk 数从 2 变 4 ⇒ 行组 512 B（SBO=32 ✓candidate B）。
- **E 则是另一条完全不同的编码路线**（TMA 朴素 64 B 行，chunk 在行内紧邻、chunk 间距 16 B ⇒ LBO=1）。

⇒ **必须靠实测**：B 已在 P1 给出"数字"（最接近），E 正在 E 回合里测（`3adc5499`）。
⚠️ **纪律**：`PACKGEOM` 的取值**只能整组使用**（写公式 + lbo + sbo + layout + 递进是一组），
任何混配都会静默错值——这也是为什么把它们做成一个开关而不是四个。

## §42 两个臂脚本的分工（避免再用错口径判读）

| 脚本 | 图 | 用途 | step 时间可用？ |
|---|---|---|---|
| `~/arm_run.sh` | **5 个图门全关** | 数值/正确性诊断（`[NC]` 才能打印；BS 臂每次调用都生效） | ❌ 约慢 10×，**不可当性能数** |
| `~/arm_run_fast.sh` | 保持默认（图 ON） | **性能测量**（真实 p50） | ✓ 可用 |
| `~/verify_correct.sh <port> <label>` | 默认 | 正确性验收（1..100 前 61 行 + 拉丁探针 + step p50） | ✓ |
| `~/push400_hw_test.sh` | 默认 | 全 gate 回归 + 400 目标压测 | ✓ |

**纪律**：看到 `step pos=… 191ms/375ms` 这种数字，先确认是不是 `arm_run.sh`（诊断口径）——
它比真实值慢约一个数量级。性能结论一律来自 `arm_run_fast.sh` / `verify_correct.sh` / `push400_hw_test.sh`。

# ⚠️⚠️ §43 【重大更正】§29 的"必须 packed"结论**是错的** —— 权威 PASS 配置是 **unpacked + SW128**

`tl-blockscale-anchor` subagent 在本机（B300/sm_103a/GPU7）用 **TileLang 0.1.14 + 官方 blockscaled 入口**
跑出了一个**通过 float64 金标准**的最小参考实现（**两种朝向都 PASS**）：

```
[a8b4] max|C|=31230.1  max|C-gold|=0.00390625  rel = 1.251e-07  -> PASS
[a4b8] max|C|=32977.7  max|C-gold|=0.00439453  rel = 1.333e-07  -> PASS
```
其触发入口是 `T.tcgen05_gemm_blockscaled(...)`（**不是** `T.gemm`；`T.gemm` 没有 scale 参数），
`sf_a_granularity_k = sf_b_granularity_k = 32`。

## 从**已 PASS 的生成源码**里提取的权威约定（逐项）
| 项 | 权威值 |
|---|---|
| **fp4 操作数 smem 形态** | **`T.float4_e2m1_unpacked` —— 1 值/字节**（TMA 把全局 packed **解包**：全局 8192 B/行组 → smem **16384 B**） |
| fp4 操作数物理布局 | **SW128（SWIZZLE_128B）**，与 e4m3 操作数**同一套公式**：`addr(r,c)=(r/8)*1024+(r%8)*128+(((c/16)^(r%8))*16)+(c%16)` |
| descriptor | **LBO=1（16 B）、SBO=64（1024 B）、base=0、lbo_mode=0、layout_type=2** —— **两种操作数相同** |
| K-block(32) 递进 | `desc + ki*32` B（`reg32_[0] += bytes>>4`）⇒ **+32 B** |
| stage 递进 | 每 stage **+16384 B**（= 一个操作数 stage 的字节数） |
| idesc | `144708608 | (ki<<29) | (ki<<4)`；a8b4 = **0x08A01400**、a4b8 = **0x08A00280** ⇒ **与我们逐位相同** ✓ |
| SF 投递 | `T.tcgen05_cp_warpx4(SFA_sh,...)` + `T.tcgen05_sf_warp_transpose(...)` + `T.fence_proxy_async()` ⇒ **与我们相同** ✓ |
| **明确警告** | **"smem operand 必须 `float4_e2m1_unpacked`（1 值/字节）；packed `float4_e2m1fn` 能编译能跑但静默错值"** |

## 更正与推论
1. **§29 的"硬件按 packed(2 元素/字节) 读"结论错误**：权威 PASS 用的是 **1 值/字节（unpacked）**。
   我那个冲激探针的"0x02 只剩一半"很可能是**探针自己**在 packed 假设下**只写了 64 B/行**
   （而硬件读 128 B/行）⇒ 上半行未被写入 ⇒ 恰好一半响应。**探针的结论不可再作为判据。**
2. **`DSV41_MOE_BS_PACKED` 门控方向是错的**（packed 会让数值静默错）⇒ 默认保持 OFF 是**正确的**，
   **不要**把它转正（§29–§42 中依赖 packed 的推理全部作废；§38/§41 的推导作为"另一套编码"分析仍有参考价值）。
3. **我们的 SW128 路径（`g_canon=0`）参数与权威逐项一致**（写公式 ✓、LBO=1/SBO=64/layout=2 ✓、
   K 递进 32 B ✓、idesc ✓、SF 投递 ✓）⇒ **真正的残余差异不在这些参数里**，
   必须在别处找：候选面 = ①我们的 **sf_id 语义**（§19 的 SFREV 假设）、②**SF 的 word 打包顺序**、
   ③**A/B 谁持有 fp4**（朝向）、④**`enable_d`/clear_accum 的时机**（我方只在最首个子 MMA 清零一次；
   官方是 `clear_accum=(k==0)` **每 stage** 清零）、⑤**我们的 B 行映射**（W1→0..63 / W3→64..127 是否与官方一致）。
4. **最省时的下一步**：把官方的**最小参考**（`/home/smith/tl_bs_min_generated.cu`，已 PASS）
   当作 **oracle**，与我们的 kernel 在**同一批输入**上逐元素对拍 ⇒ 直接定位差异环节；
   或用它做**逐项消融**（改我们的一处参数看 relerr 是否掉到 0）。

## §44 与官方 PASS 参考的逐项核对结果：**参数面全部一致** ⇒ 缺陷在别处

读完官方已 PASS 的最小内核（`/home/smith/tl_bs_min_generated.cu`）与测试（`/home/smith/tl_bs_min.py`）后，
把"我们 vs 官方"逐项对齐，**没有一项不同**：

| 面 | 官方 PASS | 我们 | 一致？ |
|---|---|---|---|
| fp4 操作数 smem 形态 | 1 B/元素（16384 B/stage，128 B/行，TMA 展开） | unpacked（`packed&0xF`/`packed>>4` 写相邻两字节） | ✓ |
| 布局公式 | SW128 `(r/8)*1024+(r%8)*128+(((c/16)^(r%8))*16)+(c%16)` | 同 | ✓ |
| descriptor | A/B 都 `initialize_tcgen05_descriptor(…, 1, 64, 0, 0, 2)` | 同 | ✓ |
| K-block 递进 | `desc + ki*32` | 同 | ✓ |
| stage 递进 | `increase_descriptor_offset(desc, k*16384)`（**多 stage 才需要**；我们是单缓冲原地重写 ⇒ 无需） | 不适用 | ✓ |
| idesc | `144708608 | (ki<<29) | (ki<<4)` | 逐位相同 | ✓ |
| **`enable_d` 时机** | `((0 < ki) ? 1 : ((k == 0) ? 0 : 1))` = **只在最首个子 MMA 清零** | 同（kk=0,ki=0 清零一次） | ✓ |
| SF 投递 | `tcgen05_cp_warpx4` + `sf_warp_transpose` + 3-warp 部分同步 + `fence_proxy_async` | 同（我们也是先 transpose 再 cp） | ✓ |
| **SF 字节序** | `w[:,0::4]` 进 LSB ⇒ **byte 0 = 组内最低 K-block** | 同（`pack_wsf`/gather 都是 byte j = 第 j 个 K-block） | ✓ |
| SF 词组布局 | group-major `[K/128][R]` | 同（W: `[E][40][row]`；激活: `[40][SEG][row]`） | ✓ |
| SF 粒度 | `gran = 32`（1 字节覆盖 32 K，1 u32 = 4 字节 = 128 K） | 同 | ✓ |
| sf_id | `ki` | 同 | ✓ |

⇒ **参数面已排除**。剩余可能（全部在"我们自己的实现细节"里）：
1. **A/B 两个 tile 的"内容装配"**：我们加载的是哪些行/列（激活行 = `seg*128+m` ✓；W 行 = `n_tile*64+row`（W1）/`64+row`（W3）；
   W 的行距 2560 B、K 迭代步进 64 B）——**需要与"我们实际喂的 W 张量布局"再核对一次**（尤其 `w_stride` 与 `2560` 的取值来源）。
2. **SF *内容* 与行的对应**：我们 `gather` 写 `sfa[g*M+seg*128+r]`、kernel 读 `SFA[k*(SEGCAP*128)+seg*128+i]`；
   权重侧 `SFW1[e*(40*320)+k*320+n_tile*64+i]`——**K 组的 `k` 与 tile 的 `k` 是否同步**（我们逐 k 迭代重写 tile ⇒ 必须同步）。
3. **朝向**（U 回合正在测）与 **M/N 角色**。
4. 一处**非常容易静默错**的量：我们 `SFREV` 与 packed 门控都默认 OFF ✓，但 **`DSV41_MOE_BS_SCALEVEC1X`** 也默认 OFF——
   官方生成码里 **没有** `.scale_vec::1X` 后缀 ⇒ **保持 OFF 是对的** ✓（这一点之前有过反复，记录以免再动）。

**下一步（已交 subagent）**：拿官方 oracle 做**同输入逐元素对拍**，把差异按行/列/K-block 打成分布，
直接指向上面 1–3 中的哪一个。

## §45 排除：SF 的行对应与角色（两种朝向都正确）

`moe_bs_handwritten.cu:362-376, 402-403` 实测：
```c
// 激活 SF（128 行 = 128 个 token 行）
(g_swapab ? SFB_sh : SFA_sh)[i] = SFA[k*(HW_SEGCAP*HW_BM) + seg*HW_BM + i];
// 权重 SF：W1 -> B 行 [0,64)，W3 -> B 行 [64,128)（与权重行 m = HW_NH + row 对齐 ✓）
(g_swapab ? SFA_sh : SFB_sh)[i]          = SFW1[e*(40*HW_NP) + k*HW_NP + n_tile*HW_NH + i];
(g_swapab ? SFA_sh : SFB_sh)[HW_NH + i]  = SFW3[e*(40*HW_NP) + k*HW_NP + n_tile*HW_NH + i];
...
hw_sf_transpose(SFA_sh); hw_sf_transpose(SFB_sh);
hw_tc_cp(hw_make_sf_desc(SFA_sh), SF_tmem + 0);   // A 操作数的 SF
hw_tc_cp(hw_make_sf_desc(SFB_sh), SF_tmem + 4);   // B 操作数的 SF
```
**结论**：
1. **W3 的 SF 确实落在 B 行 64..127** ✓（不是误写在 `[i]`）——§44 里点名的这个候选**排除**；
2. **SF 的角色切换正确**：swapAB 下权重 SF 走 SFA（A 操作数 = 权重 ✓）、激活 SF 走 SFB（B 操作数 = 激活 ✓）；
3. 激活 SF 的 gather 布局 `[k][seg][row]`（步长 `SEGCAP*BM = 4608` 词）与 kernel 的读法**一致** ✓。

⇒ §44 的候选 ② 里"W3 行错位"这一支也被排除，剩余集中在
①A/B tile 的**内容装配**（激活行 / W 行距 / `w_stride` 的实际取值）与 ③朝向/M-N 角色（U 回合在测）。

## §46 排除：MMA 完成的 mbarrier parity（正确）

`moe_bs_handwritten.cu:286-293, 472-481` 实测：
- init：`mbarrier.init.shared::cta.b64 [mbar], 1`（1 次到达）+ `fence.mbarrier_init.release.cluster` ✓（且在第一道 barrier **之前** ✓）
- 每轮 k：`if (warp==1) hw_tc_commit(mma_bar)`（一次 commit = 一次到达 ✓）
- 全体线程：`phase = k & 1` + `mbar_wait(mma_bar, phase)` ✓
⇒ 每轮翻转一次相位、与到达次数**一一对应** ⇒ **parity 语义正确** ✓（不是"瞬过"型竞态）。

**至此 §44–§46 已排除**：参数面（布局/描述符/idesc/K 递进/enable_d/SF 投递/字节序/词组/粒度）、
**SF 行对应与角色**、**mbarrier parity**。剩余最可疑的三处（全部在"我们自己的装配细节"）：
① **A/B tile 的内容装配**（激活行取自哪张表、W 行距 2560 与 `w_stride` 的**实际取值来源**、
`n_tile*64` 与 W1/W3 半区的对应）；
② **K 组 `k` 与逐迭代重写 tile 的同步**（我们 k = 0..39，每轮原地重写；SF 的 `k` 必须与 tile 的 `k` 同源）；
③ **朝向**（U 回合在测；但注意官方 a8b4/a4b8 **两朝向都 PASS** ⇒ 朝向差异必然来自我们自己的 swapAB 实现）。

# 🎯🎯 §47 【定谳】fp4 操作数的真实 smem 语义：**packed 数据 + 16 B 容器只用前 8 B**

由 subagent `bs-packed-geometry` 在硬件上**精确定标**（`max|D-ref| = 0`、`relerr = 0`，不是"更小"）得出，
并**同时调和了 §29（packed）与 §43（unpacked）两派看似矛盾的结论**——它们是同一个布局的两面。

## 真实语义（三条实测事实）
1. **数据是 packed**：硬件每个字节吃 **2 个 4-bit 元素**（**低 nibble = 偶 k、高 nibble = 奇 k**，
   由 nibble 全随机的稠密 `random` 用例**精确 PASS** 独立证实）。
2. **但放在 16 B 容器里、只用前 8 B**：TMA dtype `16U4_ALIGN16B` 的字面含义 =
   **16 个 4-bit 元素 = 8 B 数据 / 16 B 容器**。实测：把单字节写到 fp4 操作数的**原始 smem 偏移**上扫 0..1023，
   **SW128（`lbo=1,sbo=64,layout=2`）与 SWIZZLE_NONE（`lbo=8,sbo=16`）两族描述符都只读每个 16 B 槽的 `0-7` 字节**，
   `8-15` **从不被读**。
3. ⇒ 一行 64 个 packed 字节折成 **8 个容器 = 128 B footprint**；128 行 = **16384 B** ——
   **与官方每 stage 的 fp4 smem 尺寸完全一致**，也解释了官方 `ki*32` 对 A/B **都用 32 B**：
   一个 MMA 消费 **2 个槽**（A：2×16 B = 32 个 e4m3；B：2×8 B = 16 B = 32 个 packed fp4），K 对两边都是 32，自洽。

## 正确配置（推荐：**保持官方描述符不变，只改写公式**）
```c
// p = 行内 packed 字节下标 [0,64)；c = 16 B 容器下标 [0,8)
__device__ __forceinline__ int hw_pack_sw128(int row, int p) {
    const int c = (p >> 3) & 7;
    return (row >> 3) * 1024 + (row & 7) * 128 + (((c ^ (row & 7)) & 7) << 4) + (p & 7);
}
```
- **描述符保持 `lbo=1(16 B) / sbo=64(1024 B) / layout_type=2(SW128)`**（= 官方原值，**不改**）
- **K-block 递进保持 `ki*32 B`**（= 官方原值，**不改**）；idesc 不动；smem 占用也不变（仍 16384 B/操作数）
- staging：把**源 packed 字节原样**写到 `hw_pack_sw128(row, p)` ✓

**备选（V-canonical）**：写公式 `(p>>4)*4096 + (row>>3)*256 + (row&7)*16 + (((p>>3)&1)*128) + (p&7)`，
描述符 `lbo=8(128 B)/sbo=16(256 B)/layout_type=0`，递进 `ki*4096 B` —— 同样**精确 PASS**。

## 实测对照（同一仪器，稠密 random）
| 写入方式 | relerr | 判据 |
|---|---|---|
| **hw_pack_sw128 + 官方描述符** | **0** | **PASS** ✓（推荐） |
| **hw_pack_canon + canonical 描述符** | **0** | **PASS** ✓ |
| unpacked（即我们此前的写法） | 1.491 | FAIL |
| v1（`hw_pack_idx`） | 1.031 | FAIL |
| 候选 D（`lbo=1/sbo=32/layout=4`） | 0.4569 | FAIL |
| 稠密行 `row*64+col` | — | FAIL |
其它判据：`const`（BSB=0x02 与 0x22）、`sfprobe`（期望 2720）、`random_sf` 全部 `relerr≈0`；
`sweep1d k/m/n` 各 **128/128**；冲激 A/B 各 **36/36**。

## ⚠️ 方法论教训（重要）
- **`const` 对 K-block 递进是盲的**（均匀数据下 `BSADV` 从 8 到 256 单位都 PASS）⇒
  定标**必须**用稠密 `random` + `sfprobe` 才能锁定递进与 SF 字节序。
- **旧的"sweep1d k 应 128/128"在 unpacked 写入下永远是 64/128**：奇数 kk 的元素落到 16 B 槽的**后半**
  （不被读）且与空的高 nibble 配对 ⇒ 该现象曾被误读为"A 侧 parity 问题"，**实为 B 槽布局问题**（已写入 §29 的错误链条）。
- ⇒ **教训**：仅凭"某一族配置全都失败"就下"硬件语义是 X"的结论是危险的；
  需要 **(a) 正向存在一个精确 PASS 的配置**（本轮做到了：relerr=0）+ **(b) 原始 smem 偏移级的直接观测**（本轮做了）
  才能定谳。

## 主 agent 已落地的修复
`kernels/cuda/tilelang_gen/moe_bs_handwritten.cu`：新增 `hw_pack_sw128()`；
`g_packed` 分支的 W1/W3 staging 改用它（源字节原样写入）；packed 分支的描述符固定为
`hw_make_desc(..., 1, 64, 2)`、递进固定 `ki*2`（= 32 B）。门控 `DSV41_MOE_BS_PACKED=1` 开启即用此布局。

## §48 修复的足迹核验（纯算术）
`hw_pack_sw128(row,p)` 的最大值 = `(127>>3)*1024 + (127&7)*128 + (((7^7)&7)<<4) + 7`
= `15*1024 + 7*128 + 0 + 7` = **16375 < 16384** ✓——
即恰好装满**一个操作数 stage**（16384 B），与描述符 `sbo=64`（1024 B/8 行组 × 16 组）完全吻合，
且不越界进入 SF 区（A=0/SFA=32768/SFB=33280）✓。
⇒ 修复**不改变 smem 占用**（仍 16384 B/操作数），只是把数据摆到硬件真正会读的那 8 B/槽上。

## §49 修复的三方一致性交叉验证（nibble 序 + 子 tile 尺寸）

修复正确性依赖两个约定，二者**必须同时成立**，现逐项核对：

### (a) nibble 序：源字节"低 nibble = 偶 k"
- **官方 PASS 证据**：`bs-packed-geometry` 用 **nibble 全随机**的稠密 `random` 用例精确 PASS（relerr=0）
  ⇒ 硬件确实"低 nibble = 偶 k、高 nibble = 奇 k" ✓
- **我们的装载约定**：加载期的 fp4 打包（`fp4_pack_kernel`）为 `packed = (lo & 0xF) | (hi << 4)`，`lo` = 元素 `2i`
  ⇒ **低 nibble = 偶元素** ✓（与上面一致 ✓）
⇒ 我们的**源字节可以原样**写到 `hw_pack_sw128(row,p)`（$p$ = 行内 packed 字节下标）✓ **无需再拆装 nibble**。

### (b) 子 tile 尺寸：64 行 → 8192 B footprint
- 官方 `moe_bs_up_tl.cu:88-93` 的 **W1/W3 子 tile 各 8192 B** ✓
- §47 的容器语义给：**一行 footprint = 8 个 16 B 容器 = 128 B**
  ⇒ W1 子 tile = **64 行 × 128 B = 8192 B** ✓✓ **吻合**
- 我们 kernel 的 B tile = 128 行（W1 64 + W3 64）⇒ footprint = **16384 B** ✓（= §48 的核验值 ✓）

⇒ (a)(b) 同时成立 ⇒ **修复的写公式与官方描述符配套，且不需要任何额外的 nibble 变换** ✓。
**⚠️ 反过来说**：任何"把 packed 字节拆成两个 1-byte 元素"的写法（我们此前的 unpacked staging）
都会让硬件只看到一半的 K —— 这正是 §29 那个"0x02 只剩 1/2"现象的真正来源 ✓（也因此它**不是**探针假象）。

## §50 精度对齐再补一项：**投影层 fp8 的量化 block 也是 32**（与我方一致）

此前有一个未核对的疑点：官方 `act_quant` 的**函数默认形参**是 `block_size=128`
（`ref_inference/kernel.py:41`），而我方 `dsv41_quant_fp8` 用 **block 32** ⇒ 疑似不对齐。

**核对结果：一致，无缺口。** 官方 `model.py:27-30` 明确写死：
```python
fp8_block_size = 32   # one fp8 scale per 32x32 weight block / 32 activations
fp4_block_size = 32   # one fp4 scale per 32 elements along K
scale_fmt = "ue8m0"
scale_dtype = torch.float8_e8m0fnu
```
且模型里所有 `act_quant(...)` 调用传的都是 `fp8_block_size`（= 32）⇒ **官方 fp8 与 fp4 都是 block 32**，
与我方 `dsv41_quant_fp8(block=32, round_scale=1)`（`dsv41_kernels.cu`）**逐项一致** ✓
（也与 §14 里 subagent 实测"激活量化输出与官方**逐字节一致**"互相印证 ✓）。

**教训**：读官方实现时**不要用函数默认形参**当结论，要看**模型实际传入的值**
（`kernel.py:41` 的 `block_size=128` 默认值曾一度让我怀疑存在缺口）。

## §51 F 回合（修复版 `hw_pack_sw128`）的判读标准 —— 结果一到即可瞬时判定

F 回合 = 修复落地后的两条 e2e 臂（`~/arm_run.sh`，5 图门全关）：
`F1 = PACKED=1 SWAPAB=1`（fp4 在 A 侧）与 `F2 = PACKED=1`（fp4 在 B 侧）。

| 判据 | 期望（修复成功） | 若不符 ⇒ 指向 |
|---|---|---|
| **文本** | 前 61 行为 `1,2,…,61`（严格递增） | 仍乱 ⇒ 打包修复未生效**或**残留接线问题（用 `[NC]` 定位） |
| **`[NC] WORST rel=`** | **≤ 1e-2**（量化级舍入；§28 的指纹表） | 仍 1e0–1e2 ⇒ 标度/取数仍有系统性错；先看 `[NC] seg=… ` 逐点值哪个 col/行差 |
| **`[NC] entered`** | 必须出现（5 图门已关 ⇒ 非捕获调用能进） | 不出现 ⇒ 回到 §35/§16；有 `ABORT` 行 ⇒ 看是哪一次 memcpy |
| **ERR_COUNT** | 0 | 非 0 ⇒ 先看 serve 日志尾部（§25） |
| **两朝向** | F1 与 F2 **都应正确**（官方 a8b4/a4b8 两朝向都 PASS ⇒ 朝向不应是正确性前提） | 只有一个对 ⇒ 我们自己 swapAB 实现里的那条分支有 bug（SF 角色/转置/epilogue 之一） |
| **step 时间** | **不看**（图关掉后 ~10× 慢，非性能数；§42） | — |

**修复成功后的紧接着三步（顺序不可颠倒）**：
1. **把 packed 布局转正为默认**（否则所有既有测试脚本仍走 measured-wrong 的 unpacked 路径！）：
   在 `moe_bs_handwritten.cu` 里让 `g_packed` 初值为 1，unpacked 路径留 env 逃生门（遵守"退化默认关"的规则反向应用）。
2. **正确性验收**：`~/verify_correct.sh`（1..100 前 61 行 + 拉丁探针 + step p50）+ EAGER 对照 + 无重复/无乱码
   （新脚本 `~/wq_check.py` 由 subagent `wq-accept-checker` 交付后接入）。
3. **全 gate 回归 + push400**：`~/push400_hw_test.sh`（真实 p50 与红线）。

## §52 性能路线图（subagent `perf-roadmap-400`）的三条关键结论 + 一条口径裁决

### (1) 【主 agent 裁决】`step ≈ 8.87ms / ~112 tok/s` 是 **plain m=1 decode 步时**，不是 MTP 步时 ✓
证据：本次会话所有 e2e 臂都**没有**开 `DSV41_SPEC`（spec/MTP 路径未激活），而 `[dsv41] step pos=` 行
正是 plain decode 的步时口径。roadmap 独立地做了同一判断（它指出"MTP step 的现有口径是 28.5–31.3ms"，
而 8.87ms 与 `roadmap-200-tokps.md §2` 的"9.0ms → ~112 tok/s"完全吻合）。
⇒ **因此 8.87ms 就是性能模型里的 `eager(1)` 项**（即用户口中的 "eager 6.3ms" 的当前值）。

### (2) 【最重要量化发现】440/400 的算术**是紧的**：只"照抄 eager 优化 + 摊薄"**不够**
按 `mtp-verify-amortization-model §1`：`step ≈ verify(m) + draft + commit ≈ eager(1)+ε + 1.5~2 + 0.4`。
取 `eager(1) = 8.87ms` ⇒ **即使摊薄 100% 兑现，step ≈ 10.8ms > 9.8ms 门槛**。
⇒ **必须靠 tilelang/mma 把 `verify` 压到 *低于* "自私单行"的水平**（或把 draft 折进 verify 的第 0 行）。
这正说明 **MoE BS（fp4 blockscaled / tcgen05）臂是 400 的必经之路**——它把 routed experts 的 8.30ms 压下去，
而 routed 族是 breakdown 里第二大项（§52 表 A：8.30ms / 22.2%，且字节不可压、只能靠核效率）。

### (3) 三个正确性阻塞项互相独立（都是 400 的头部杠杆）
① **MoE BS fp4 blockscaled 臂**（本文件）② **TileLang 投影臂**（`tl-garbage-verdict`：5 条根因已定位，
修复 `98f50e8` 已推，**双挂 e2e 未重编**）③ **MoE grouped SIMT A′**（`g1-moe-expert-union-verdict §3`：
零布局、逐位等价、**未上机**）。⇒ 正确性一解决，这三条可并行推进。

### 附带纪律（roadmap 提醒，与本文件 §42 同类）
**票面必须用 nsys 占比算，不能用"每层节省 × 40"这类代数**：实测五 gate 只兑现 −0.6ms，
而票面远高于此（best1 −1.14 / best2 −2.98 / MPAR 二连败）。已判死清单见 roadmap（MPAR、⑤a L2 直读、
proj-mma、p3lite+ALIGN、GROUPED 系列、launch 税、M-tile 调参、bf16 dequant 显存、MROWS_FOLD_R/AR_STORE_FUSE/AR_SINGLE_POLL）
——**勿再投入**。

## §53 精度补丁的 OFF 路径安全性 —— 已逐行核实 ✓

`git show 16d9953 -- crates/ferrite-models/src/dsv41/chain_dev.rs | grep '^-'` 显示被删除/改写的行**只有三处**：
```
rw_base,                                        ×2   ← down 调用的 row_weight 实参
self.bf16_snap(self.s.ex_act_b.ptr as *mut f32, topk*act_slot as usize)?;   ← bf16 边界调用
self.s.route_w.ptr as *const f32,               ×2   ← 路由权重实参
```
⇒ 补丁只改写了**这三个调用点的实参/调用**，并把它们包进 `if <gate> { … } else { 原样 }` 分支。
**门控 OFF 时**：`rw_base` 照传、`route_w.ptr` 照传、`bf16_snap` 照跑 ⇒ **与改动前逐字节等价** ✓
（与 subagent 的论证 + `cargo check --workspace` 通过互相印证 ✓）。

**顺带说明该补丁要修的正是这三处**（对应 §14/§21 的两处不对齐 + 其载体）：
① 路由权重必须**在 bf16/e4m3 边界之前**乘（官方 `model.py:849`）⇒ 门控 ON 时由新内核在融合步里施加，
   调用方不再传 `row_weight`（避免乘两次）；
② down 的输入必须是 **e4m3(block32) 量化后再反量化**的值（复现官方 w2 看到的操作数）；
③ 因此原有的 `bf16_snap` 在门控 ON 时**必须跳过**（否则会对加权前的值多舍入一次，与官方单次舍入不符）。

## §54 我方 `hw_pack_sw128` 实现的独立算术核验（不依赖 GPU，纯算术）✓

从源码抽出实现体后与 subagent **实测 relerr=0** 的权威公式逐值比对（全部 128×64 = 8192 个 `(row,p)` 组合）：

| 核验项 | 结果 |
|---|---|
| 与权威公式逐值一致 | **0 处不一致** ✓ |
| 区间内双射 | ✓（8192 个取值互异） |
| 取值范围 | `[0, 16375]` ⊂ `[0, 16384)` ✓ |
| 被使用的 16 B 槽数 | **1024**（= 128 行 × 8 容器）✓ |
| **越过槽内前 8 B 的槽数** | **0** ✓ ← 与硬件实测"只读每槽前 8 B"完全吻合 |

⇒ 修复的**代码实现与实测通过的规范严格一致**，且结构性质（8 B/槽、恰好铺满一个操作数 stage）
已由独立算术证明，不依赖任何 GPU 测量 ✓（即使 subagent 的仪器有偏，这一步也能保证实现无笔误）。

## §55 验收判据已机械化：`wq_check.py`（subagent `wq-accept-checker` 交付，主 agent 已实跑验证）

**路径**：`/home/smith/wq_check.py`（主 agent 已 scp 到远端 `~/wq_check.py`）。
**本地与远端 `--selftest` 均 PASS（12/12 用例）** ✓（主 agent 实跑，非仅看报告）。

### 用法
```bash
# 单文本判定
python3 wq_check.py --text-file <f> [--expect-count 61] [--eager-file <f2>]
# 多份 arm 日志 → 结论表
python3 wq_check.py --log ~/armrun_F1.log ~/armrun_F2.log [--eager-file <f>]
# 自测（12 用例：3 种事故形态 + 正常计数 + EAGER 同现/不一致 + 反转义 + 截断降级）
python3 wq_check.py --selftest
```
退出码：`0 = PASS/WARN`、`1 = 有 FAIL`。

### 它机械化的四条判据（正是用户红线）
| 判据 | 实现要点 | FAIL 阈值（可调，见脚本 `TH` 字典） |
|---|---|---|
| **重复** | 最长重复子串（≥3 次）+ 重复密度 + 单字符占比 + 单数字占比 + 最常见 token 占比 | 任一子信号命中 |
| **乱码** | 制表符/控制符占比、不可打印字符、可读字符占比、字母数字 CJK 占比 | 比例越界 |
| **计数** | 逐行抽整数（容忍 `1.`/`- 1`/`**1**`/反引号），要求严格 `1..N`；**不足 N 只降级 WARN** | 出现乱序/跳号 ⇒ FAIL 并报 `first_bad_line` |
| **EAGER 对照** | 比较两者的整数序列公共前缀与逐字一致性 | **两者退化一致 ⇒ `exculpated`（模型行为，判 PASS）** ✓ |

⇒ 这条把文档里"**退化与 EAGER 一致 = 干净**"的判读规则**做成了机械判据**，不再靠人眼判断 ✓。
**验收流程**（§51 第 2 步）改为：`arm_run` → `wq_check.py --log ... [--eager-file ...]` → `VERDICT` 决定是否继续。

## §56 F 回合结论（修复后 e2e 仍不正确）+ `[NC]` 仪器的分级探针

### F 回合实测（同一构建、5 图门全关、`swapAB` 两朝向）
| 臂 | 配置 | 输出 | 判读 |
|---|---|---|---|
| F1 | `PACKED=1 SWAPAB=1`（canon=0 ⇒ SW128） | `'\n-\t-\t\t\t----------------------------- .\t\t-'` | **仍不正确**（短线/制表符混合） |
| F2 | `PACKED=1`（canon=0） | `'\n\t\t\t\n\t\t…'` | **仍不正确**（纯制表符） |

**两条关键事实**：
1. **kernel 语义已精确**（§47 的 relerr=0 是实测）但 **e2e 仍错** ⇒ 缺陷在**接线/取数**，
   不在 MMA 的 smem 语义里。（且注意：修复**确实改变了输出**——从修复前的"引号+中文混合"
   变成现在的"短线混合"，说明改动生效了，只是还有第二个缺陷。）
2. **`[NC]` 仍然 not-entered** ——即便：(a) 5 个图门全关；(b) launcher 确认
   `env $BASE $COMMON "$@" … ferrite-serve` 把 `DSV41_MOE_BS_NUMCHECK=1` 传进了进程；
   (c) `ARMED gate_up_bs (**device tables**)` 证明走的是含 NUMCHECK 的那个入口；
   (d) NUMCHECK 块位于主路径上、在 sync/mma diag 之后。
   ⇒ **必然有某个提前 `return` 在它之前命中。**

### 本轮的仪器化（一次构建即可定位）
- `[NC-TRACE] dev entry ENTERED (NUMCHECK env=?/1)`：放在 dev 入口函数体**第一行** ⇒
  区分"入口根本没被调用"与"调用了但提前返回"。
- dev 入口内**全部 9 处提前 `return`** 都被包上一次性打印
  `[NC-TRACE] dev EARLY-RETURN at shim line <L> -><expr>;` ⇒ 无论命中哪个守卫都能直接报出行号。
- 编译检查通过（`PK9_RC=0`）。

### 已排除（本轮静态核查，勿重查）
- `w_stride` **不是**硬编码：Rust 从实际指针测量（`w3 - w1`，注释明写 "block layout, not NP*K/2"）
  并有断言校验（`chain_dev.rs:17029-17089`）✓
- **W 的 nibble 序**：官方 fp4 打包用 torch 的 sub-byte 约定（`convert.py:181` 的
  `.view(torch.float4_e2m1fn_x2)`）+ `kernel.py:131-181` 的 `fp4_quant_kernel`（`fp4_max=6.0`，
  `T.clamp(x/s, ±6)` → `T.Cast(FP4, …)`）⇒ **低 nibble = 偶元素**，与我们"源字节原样写入"一致 ✓
- **A 侧取数**：`GATHER-DIAG` 证实 `g_a[0..15] == xq4[0..15]` ✓（至少首 16 元素）
- **容器与 K-block 的一致性**（自洽推演）：一个 16 B 容器 = 8 packed 字节 = **16 个元素** ⇒
  32 元素的 K-block = **2 个容器** = 32 B ⇒ 与描述符 `ki*32 B` 递进**完全自洽** ✓；
  一行 64 packed 字节 = 8 容器 = 128 元素 ✓ ⇒ footprint 128 B ✓（与 §47/§48 一致 ✓）

## §57 TileLang 路径复活可行性（subagent `tl-path-revival`）+ 一条重要更正

### 更正：err700 很可能**不是这个 kernel 产生的**（错误归属/latching）
它的首要嫌疑是**错误闩存**：本文件 §(配置探针段) 已记录该 700 出现在**配置探针阶段**
（`dsv41_bf16_roundtrip` / `dsv41_gemm_fp8_mx_f32`）被闩存到后续无关调用上；
而 SYNC-DIAG 揭穿后指出本 kernel 的真实错误类型是 **illegal instruction**，
"illegal memory access" 是 context poisoning 的**二级效应**。
⇒ 之前把 700 当成"本 kernel 的越界"这条推断**要收回**。

### 描述符逐字段对比：官方生成物 vs 已 PASS 的最小参考 —— **无一项不一致**
（它先解出 TVM FFI 的参数序：`[0]desc_ptr [1]dtype [2]rank [3]global_addr` 随后
`gdim[rank] / gstride[rank] / box[rank] / estride[rank]` 再 4 项 `ilv, swz, l2, oob`，
并用 4 个独立样本交叉验算自洽。）核心对照：

| 项 | 官方生成物 | 已 PASS 参考 | 一致 |
|---|---|---|---|
| fp4 dtype | **14 = 16U4_ALIGN16B** | 14 | ✅ |
| fp4 `box[0]` 单位 | **128（4-bit 元素 = 64 B 数据）** | 128（4-bit 元素） | ✅ |
| fp4 行距 | 2560 B = K/2 | 128 B = K/2 | ✅ |
| fp4 swizzle | **SWIZZLE_128B** | 同 | ✅ |
| fp4 smem 足迹 | 64 行 × **128 B/行** = 8192（W1 槽） | 128 行 × 128 B/行 = 16384 | ✅ |
| **fp4 `expect_transaction`** | **4096 = 64×64 B 数据** | 8192 = 128×64 B 数据 | ✅ **都是"槽的一半"** |
| A dtype/box | u8：128×128 | 同 | ✅ |
| SF | SFA 1-D bulk 512 B；**SFW1/SFW3 走 2-D 描述符，box=(64,1)** | 同 | ✅ |

⇒ **§47 的语义在 TL 路径下无需任何改动**：官方 TMA 自己就做 16 B 容器展开
（`expect_transaction` 只报"数据字节"= 槽的一半，正是这件事的直接证据）。
⇒ 结论：**err700 不在描述符层**；TL 路径真正的剩余风险是 **3-stage TMA+mbarrier 流水线在 m>1 下的行为**，
这条**只能 GPU 验证**（留给主 agent，且与当前 e2e 缺陷是两件事）。

## §58 前置检查：精度 DBG 回读**同样带图捕获守卫** ⇒ 验收必须用"图全关"的臂

主 agent 实读代码发现（`crates/ferrite-models/src/dsv41/chain_dev.rs:19770`）：
```
"[routed-down-quant] DSV41_ROUTED_DOWN_QUANT_DBG is set but a capture is in …"
```
⇒ 与 `[NC]` 同一类陷阱（§16/§35/§42）：**若该 launch 处在 CUDA-graph capture 内，DBG 五点回读会被静默跳过**。

**因此 §39 的验收命令必须走 `~/arm_run.sh`（它已内置 5 个图门全关）**，正确写法：
```bash
bash ~/arm_run.sh RQ1 DSV41_ROUTED_DOWN_QUANT=1 DSV41_ROUTED_DOWN_QUANT_DBG=1
grep -E "routed-down-quant" ~/armrun_RQ1.log | head -20     # 5 组值 + 主机参考 + 逐元素差
```
**判据**：①–⑤ 逐元素差为 0（±1 ulp）；随后同 env 跑 `~/verify_correct.sh` 做文本红线（配合 `~/wq_check.py`）。
⚠️ 性能数字仍必须来自 **无 DBG** 的快速臂（`~/arm_run_fast.sh`）——DBG 本身会 D2H 回读、污染计时。

## §59 排除：权重 SF 的平面尺寸三方一致（字节 ↔ 词 ↔ TMA 步长）

`chain_dev.rs:moe_bs_weights` 的断言与 kernel/官方三处必须同源，实测**一致** ✓：

| 侧 | 表达式 | 值 |
|---|---|---|
| Rust 断言（**字节**） | `moe_bs_sf_words(dim) * inter_local * 4` = `(5120/128) * 320 * 4` | **51200 B** |
| 官方 TMA 描述符 | `SFW1/SFW3` 的 `gstride[1]`（见 §57 的对照表） | **51200 B** ✓ |
| kernel 索引（**词**） | `SFW1[e*(40*HW_NP) + k*HW_NP + n_tile*HW_NH + i]` ⇒ 平面 = `40*320` | **12800 词 = 51200 B** ✓ |

⇒ 权重 SF 的**每专家平面尺寸、K 组步长（HW_NP=320 词 = 一组的 inter 行）、行索引**三处自洽 ✓
（`s_stride`/`t_stride` 也在 Rust 侧被断言等于该值，不满足则不 arm——而臂确实 arm 了 ✓）。

## §60 排除：pad 段的 `eid` 取 0 ⇒ 不存在"越界 expert id"

假设：手写 kernel 只取 `blockIdx.y` 作 seg（`moe_bs_handwritten.cu:267`，**自身无 `seg >= nseg` 守卫**），
若 `seg ≥ nseg` 的块读到越界 expert id，就会让权重指针出界 ⇒ 非法访问 ⇒ 后续 launch 报错 ⇒
入口在 `if (e != cudaSuccess) return` 处提前返回（同时解释 `[NC]` 不打印 + e2e 输出错）。

**证伪**：shim 的 ABI 文档明写（`moe_bs_shim.cu:94-96`）：
```
counts[SEG_CAP] : i32 -- 该段的 live 行数（1..2；**pad 段 0**）
eid[SEG_CAP]    : i32 -- 该段的 expert id（**pad 段任意，取 0**）
```
⇒ pad 段的 `eid = 0` 是**合法** id（expert 0 在池内）⇒ 不存在越界权重指针 ⇒ **该假设作废** ✓
（pad 段算出的行不会被使用：scatter 侧按 `counts[seg]==0` 早退 ✓）。
⇒ **`[NC]` 不进入的原因仍待 NC-TRACE 逐行探针回答**（T2 窗口运行中；9 处提前 return 都已带行号打印）。

## §61 【精度】全算子审计结论（subagent `precision-audit-full`）：**8 个独立缺陷**

> 用户硬性要求："精度必须和官方 pytorch 实现完全对齐，fp4 fp8，**不能高也不能低**"。
> 本次审计把范围扩到**全算子**（此前 ~85%），逐项带双侧 file:line。
> 结论：**ALIGNED 22 项 / MISALIGNED 10 项（去重 8 个独立缺陷）/ NEEDS-GPU 5 项 / UNKNOWN 3 项**。

### 三项"缺失量化"（我方**精度偏高**，与官方不一致）—— 新发现
| 项 | 官方 | 我方 | 后果 |
|---|---|---|---|
| **A2 窗口 KV** | `model.py:707`（主干）/`:1042`/`:1062`（DSpark）：`kv_norm → rope → act_quant(kv, block=32, "ue8m0", inplace=True)` ⇒ 入 ring **前**把 bf16 行吸附到 e4m3/block32/幂次网格 | `chain_dev.rs:62-64` 注释自述 "ring is f32"；`:20197-20247` 只做 `rmsnorm→rope→bf16_snap`；`:20328-20379` 的 `ring_win_fused_kernel` 是**纯 f32 拷贝**；全仓无就地 `act_quant` | **精度偏高** ✗（仓库内已自认：`accept-first-strategy.md:67`、`s4 §3.4`） |
| **A3 压缩 KV latent** | `model.py:758-760`：`fp4_act_quant(latent, **block=16**, True, scale_dtype=**e4m3**)`；`kernel.py:159-166` 的 **e4m3-标度分支**：`amax=max(amax, 6*2^-9)`、`s = Cast(f32, Cast(e4m3, amax/6))`（**不是幂次！**）、byte=`fp4(clamp(x/s,±6))`、inplace 写回反量化值 | `chain_dev.rs:21102-21203` `compress_on`：只 `bf16_snap_on(latent)` → commit；`glue.cu:1185-1220` 只 rope + f32 存储。**全仓没有 block=16 的 fp4 量化，也没有 e4m3-标度分支**（`quant_kernel` 只有幂次标度） | **缺整个量化器分支** ✗ |
| **A4 indexer q/k** | `model.py:546`(k)/`:552`(q)：`fp4_act_quant(·, block=32, True)`，默认 e8m0 幂次标度（`kernel.py:165-166`），inplace 写回反量化值 | `chain_dev.rs:20892-20929`(k)、`:20945-21014`(q)：**全程 f32，无 fp4 往返** | **精度偏高** ✗ |

### 其余 MISALIGNED（原报告另有 5 项，含 RoPE 系数、累加序、PV 操作数、一处位置相位（非精度）、
以及"已知两处"的**默认 OFF 态**——即在出货默认臂下那两处**仍未对齐**）。
### NEEDS-GPU 5 项（其中最关键：**出货臂的 env 组合无法从源码判定**，决定上述若干项是否真实成立）。

**⇒ 行动**：新开三条实现线（各自隔离 worktree、默认 OFF、带 DBG 探针、CPU 可 `cargo check` 验证），
分别补齐 A2 / A3 / A4 三个缺失量化；GPU 验收与 env 裁决由主 agent 独占执行。

## §62 🎯【已修】swapAB 输出打包缺陷：修复前 **640 个输出里有 512 个位置错**（80%）

### 缺陷（已修，`moe_bs_handwritten.cu` epilogue）
scatter（`moe_bs_shim.cu:583-586`）按 **每个 n_tile 128 列** 解读中间 scratch：
```c
bx = col / 128;  j = col - bx*128;
n = (j < 64) ? (bx*64 + j) : (320 + bx*64 + (j - 64));   // 前 64 列 = gate、后 64 列 = up
```
- **非 swapAB 分支**：`col = n_tile*128 + c`（c = W 行，0..63 = W1/gate、64..127 = W3/up）⇒ **正好匹配** ✓
- **swapAB 分支（缺陷）**：`col = n_tile*64 + r`（gate 全打包进 [0,320)、up 全进 [320,640)）
  ⇒ 与 scatter 的解读**不符** ✗

### 纯算术核验（不依赖 GPU）
| | 覆盖列数 | scatter 能否还原出正确 n |
|---|---|---|
| **修复后**（`col = n_tile*128 + r`） | **640**（恰铺满 [0,640)） | **640/640 全部正确**（0 处不匹配）✓ |
| **修复前**（gate/up 各打包 320） | 640 | **仅 128/640 正确 ⇒ 512 个位置错** ✗ |

⇒ 这是一个**决定性的接线缺陷**（80% 输出错位），完全解释了"kernel 语义已精确、e2e 输出仍是结构化错文本"
这一现象。修复 = 让 swapAB 分支改用与非 swapAB **相同的列公式**，只交换行/列角色
（`r` = 权重行、`c` = token），即：
```c
C[(int64_t)(seg * HW_BM + c) * HW_NUP + n_tile * HW_BN + r] = C_sh[r * HW_BN + c];
```
### 注意（下一个待查项）
**非 swapAB 分支经同一核验是"匹配 scatter"的** ⇒ 因此 **F2 臂（非 swapAB）仍然错，必是另一个缺陷**，
且该缺陷**在两条朝向共用**。待 `bs-wiring-audit` / `old-new-diff-audit` 两条审计线报告。

## §63 【精度·配置缺口】出货臂**没有开** `DSV41_ROUTED_DOWN_QUANT` ⇒ 两处修复默认不生效

审计的 NEEDS-GPU N1 指出"出货臂的 env 组合无法从源码判定"，主 agent 直接查了脚本（CPU 可判定）：
- `~/push400_hw_test.sh` 的 env 列表里**没有** `DSV41_ROUTED_DOWN_QUANT`（也没有 `..._DBG`）；
- `~/verify_correct.sh` 同样没有；
- 补丁本身默认 **OFF**（`chain_dev.rs:1236-1245`）。

⇒ **在出货/回归的实际配置下，§14/§21 的两处精度修复（路由权重时机、routed-down 输入量化）并未生效**
⇒ 相对官方仍**不对齐**（我方精度偏高）。这与用户的硬性要求直接冲突。

**行动顺序（不可颠倒）**：
1. **先上机验证**补丁本身正确（§58 的命令：`arm_run RQ1 DSV41_ROUTED_DOWN_QUANT=1 DSV41_ROUTED_DOWN_QUANT_DBG=1`
   ⇒ 五点回读与主机参考逐元素差 0）；
2. 验证通过后**把 `DSV41_ROUTED_DOWN_QUANT=1` 加进出货/回归脚本**（`push400_hw_test.sh`、`verify_correct.sh`），
   并跑 `wq_check.py` 确认文本红线不回退（官方语义下输出应几乎不变）；
3. 一并把 A2/A3/A4 三条缺失量化（§61）的实现按同样流程转正。

**注意**：这三个门控**都会改变数值**（把"精度偏高"降到官方水平）⇒ 每次转正都必须走
"DBG 对拍 → 文本红线 → 全 gate 回归"三步，且**逐项单独转正**（一次一个变量）。

## §64 【精度·收口清单】出货配置 vs "对齐所需门控"

主 agent 直接查脚本（CPU 可判定，不依赖 GPU），把三处合起来得到完整结论：

| 精度项 | 对齐是否需要专门门控 | 出货脚本（`push400_hw_test.sh` / `verify_correct.sh`）现状 | 结论 |
|---|---|---|---|
| MoE 专家激活走 e4m3（官方 fp4 权重配 e4m3 激活） | 需要 `DSV41_EXPERT_ACT_E4M3=1` | **已开** ✓ | 对齐 ✓ |
| hc_pre/hc_post 的 bf16 边界、head 输入边界 | 需要 `DSV41_BF16_TRUNCATE=1` | **已开** ✓ | 对齐 ✓ |
| **路由权重时机 + routed-down 输入量化**（§14/§21 的两处修复） | 需要 `DSV41_ROUTED_DOWN_QUANT=1` | **未开** ✗（且补丁默认 OFF） | **不对齐（我方精度偏高）** ✗ |
| **A2 窗口 KV 就地量化往返**（§61） | 需新门控（如 `DSV41_WINDOW_KV_QUANT`） | 尚未实现 | **不对齐（精度偏高）** ✗ |
| **A3 压缩 latent 的 fp4(block16, e4m3 标度) 往返** | 需新门控（如 `DSV41_COMPRESS_LATENT_QUANT`） | 尚未实现 | **不对齐** ✗ |
| **A4 indexer q/k 的 fp4(block32, e8m0) 往返** | 需新门控（如 `DSV41_INDEXER_FP4_RT`） | 尚未实现 | **不对齐（精度偏高）** ✗ |

⇒ **要满足用户"精度不能高也不能低"，必须转正 4 项**（1 项已有补丁 + 3 项新实现）。
**转正流程**（已完成脚本化：`~/promote_precision.sh "<GATE=1> [DBG=1]"`）：
① DBG 五点回读与主机参考**逐元素差 0** → ② `wq_check.py` 文本红线 PASS → ③ 快速臂无回归 →
④ 加进出货脚本 → ⑤ 全 gate 回归（`push400_hw_test.sh`）。**逐项转正、一次一个变量。**

## §65 【定谳】bs-wiring-audit：两个确定性缺陷（其中一个是**我自己造成的回归**）+ 六项映射 OK

### M1 —— ⚠️ 我的仪器化回归（**已彻底回退**）
上一轮为定位 `[NC]` 而加的 `[NC-TRACE]` 仪器化，把**独立的 `return X;` 行**改写成
"打印块 + `return X;`"两行 ⇒ 原先是 `if (条件)\n return 2;` 的结构，变成
`if (条件)\n {打印}\n return 2;` ⇒ **`return` 掉到 `if` 外面、变成无条件返回**（9 处守卫全废）。
后果：**dev 入口在第一个守卫就无条件 `return 2`** ⇒ **BS 臂整体失效**，任何 e2e 都**静默回落到老路径**
——正是本项目 #1 测量偏置陷阱（"跑的不是你以为的那条路"）。
⇒ **推论（重要）**：**F / P / U / E 各回合的文本判据全部建立在"BS 臂已生效"的假设上，该假设不成立** ⇒
那些文本结论（如"包打包后文本变得更像计数"）**必须作废重判**。
⇒ **教训（已回退）**：**仪器化只能"加"，绝不能重写控制流**；且改完**必须验证语义**（编译通过 ≠ 行为不变）。

### M2 —— `goto` 跳过探针（**已修**）
`HANDWRITTEN=1` 分支在 `moe_bs_shim.cu` 用 `goto scatter_launch;` 直达 scatter，
**跨过**了 MMA-DIAG 与 NUMCHECK 两个探针块 ⇒ 手写路径上 `[NC]` **结构上不可能打印**
（不是"某个提前 return"，而是 `goto` 直接跳过）。我 §56 的"(d) NUMCHECK 位于主路径上"**前提是错的**。
⇒ 已在该 `goto` **之前**接上同一套（带非捕获守卫 + 同样 D2H 拷贝的）NUMCHECK 探针，并加一次性入口标记。
⇒ **教训**：判断"可达性"必须**沿控制流走**（尤其 `goto`），不能只看代码位置。

### 六项映射审计结论：**全部 OK**
入口实参（`xq4/xsc4/out/w1/w3/sfw1/sfw3/eid/order/counts/nseg/w_stride/rows/dim/inter/topk`）
**ABI 同序同型、常量与冻结几何逐项对齐**（SEG_CAP36/BM128/BK128/NH64/NP320/N_UP640/DIM5120/K_ITER40/sf_words40/smem166912）；
A 行距 = `dim`（e4m3 1 B/value）、W 行距 = `K/2 = 2560`（§59 已验）；SF/eid/order/epilogue/act_slot 逐条 OK。

### 审计点名的两处代码质量项（本轮一并修）
1. **注释陷阱**：`moe_bs_shim.cu:746/885`、`device.rs:8127/887` 把 `xq4` 写成 `[rows*topk][dim]`，
   而**代码与生产者都是 `[rows][dim]`**（激活按**行**量化）。这种误导性注释会诱导人"按注释改代码"⇒ 必须改对。
2. **跨语言不对称契约**：shim **不检查** `topk*rows ≤ SEG_CAP`（只在 Rust 侧
   `chain_dev.rs:16732` 的 `n_assign > TILELANG_SEG_CAP → Ok(false)` 把关）⇒ 建议 shim 加一条断言
   （当前生产形状 (6,6)=36 恰取等，不触发）。

## §66 两项精度语义的**主 agent 交叉确证**（发给 subagent 的 brief 已核实）

为避免 subagent 误读最难的那两处，主 agent 亲自读了官方原文：

### A2（窗口 KV，fp8 就地往返）—— `ref_inference/kernel.py:70-92`
```python
amax_local[i] = T.max(amax_local[i], 1e-4)                     # 下限 1e-4
if round_scale:  s_local[i] = fast_round_scale(amax_local[i], fp8_max_inv)   # 幂次
else:            s_local[i] = amax_local[i] * fp8_max_inv                    # 非幂次（本处不用）
if inplace:      y = Cast(out_dtype, Cast(compute_dtype,
                     Cast(FP8, clamp(x/s, fp8_min, fp8_max))) * s)           # ← 写回**反量化值**
```
⇒ 与 §61/发给 A2 的 brief **逐项一致** ✓（block=32、clamp ±448、下限 1e-4、round_scale=True、写回反量化值）。

### A3（压缩 KV latent，fp4 + **e4m3 标度**）—— `ref_inference/kernel.py:152-166`
```python
if scale_dtype == FP8:      # "Training's compressed KV: keep even an all-zero group's scale nonzero."
    amax_local[i] = T.max(amax_local[i], 6 * (2**-9))
    s_local[i] = T.Cast(compute_dtype, T.Cast(FP8, amax_local[i] / fp4_max))   # ← **非幂次**（e4m3 舍入）
else:
    amax_local[i] = T.max(amax_local[i], 6 * (2**-126))
    s_local[i] = fast_round_scale(amax_local[i], fp4_max_inv)                   # 幂次（A4 用这条）
```
⇒ 与 §61/发给 A3 的 brief **逐项一致** ✓；且**注释原文直接点名是 "compressed KV"**，
独立印证 A3 的目标就是压缩 latent（block=16，`model.py:760`）✓；A4（indexer）走 `else` 的**幂次**分支 ✓。

## §67 排除：RoPE 系数与 YaRN（出货配置下惰性）

审计列了一项"RoPE 系数"。主 agent 直接对照两侧常量来源：
- 官方 `ref_inference/model.py:87-90`：`compress_rope_theta = 40000.0`、`original_seq_len = 0`、
  `rope_theta = 10000.0`、`rope_factor = 40`。
- 我方 `crates/ferrite-models/src/dsv41/config.rs:63-65`：同名字段（`compress_rope_theta` 默认 40000.0 ✓），
  传入点 `chain_dev.rs:5302/5309` 传 `cfg.rope_theta` / `cfg.compress_rope_theta` / `cfg.original_seq_len`。
⇒ **两侧同源（同一份 config.json）**，且 **`original_seq_len = 0`** ⇒ 官方的 YaRN 外推分支
（`precompute_freqs_cis(..., original_seq_len, base, factor, ...)`，`model.py:369`）**惰性**、
`rope_factor` 不参与计算 ⇒ **出货配置下该项对齐** ✓（若要改 `original_seq_len` 才需重新核对，
这一条已记入"NEEDS-GPU/需配置确认"清单）。

## §68 【精度·新开线】I3：attention PV 的**概率操作数 bf16 舍入**（我方精度偏高）

审计 §61 的"PV 操作数"一项，取回原文后的确切语义：
- 官方 `ref_inference/kernel.py:339` 与 `:377-380`：在线 softmax 的 `acc_s`（= exp 后的**概率**）先
  **`acc_s_cast = acc_s.to(BF16)`**，然后 `T.gemm(acc_s_cast, kv_shared, acc_o)`
  ⇒ **概率被舍到 bf16 后才进 PV 乘加**（`acc_o` 累加器仍是 f32；KV 侧另有 bf16 容器 + e4m3 网格 = A2 那条线）。
- 我方（`ops.rs:347-388`、`kernels.cu:1611-1705` 一带）`my_acc[i] = my_acc[i]*corr + e*kb[i]` **全程 f32**
  ⇒ **我方精度偏高** ✗。

**已开实现线 `prec-i3-pv-bf16-p`**（隔离 worktree、门控 `DSV41_ATTN_PBf16` 默认 OFF、带 DBG 五点回读、
要求列出**全部** PV 乘加点——含 spec/verify 复用路径，**半挂配置是设计内非法**）。
⚠️ **关键待确认（决定实现正确性）**：官方是"**先**舍 bf16 概率**再**乘 KV"，以及在线 softmax 的
`corr` rescale 作用在**累积器**还是**概率**上——若作用在累积器，则只有"当前块的概率"需要 bf16 舍入。
已要求该 subagent 读代码确认并在报告中写明顺序。

## §69 【差异法审计】新路径 vs 旧路径（per-slot GEMV，**输出已知正确**）——**十项取数对照全部 SAME**

方法：以**旧路径为地面真值**（它的输出是完美的），逐项比对新路径每一处取数索引。
准绳位置：`dsv41_experts_mxf4.cu:1342-1344`（gate/up pair body）、`:3523-3719`（launcher）、
`:2258/3734`（down）；新路径 = shim 的 gather/scatter + `moe_bs_handwritten.cu` 的 loader/epilogue。

| # | 对照项 | 结论 |
|---|---|---|
| 1 | W1/W3 → gate/up 划分（通道 = 面内行号） | **SAME** |
| 2 | W 的行距（`kbytes = K/2 = 2560`） | **SAME** |
| 3 | expert 步长（两侧都实测 `w1` 指针差，**都不是 `NP*K/2`**） | **SAME** |
| 4 | W 的 K 方向/起点/步长 + **低 nibble = 偶 K** | **SAME** |
| 5 | 激活矩阵形态（e4m3，1 B/value，**per-activation-row**，`src_row = idx/topk`） | **SAME** |
| 6 | 激活 SF（SFA：行距 160，byte c = K-block `4g+c`，`sf_id = ki`） | **SAME** |
| 7 | 权重 SF（SFW1/SFW3：行距 `align16(k/32)=160`，平面 40×320 词） | **SAME** |
| 8 | eid / order / counts（每 assignment 的专家/行/槽三元组同源） | **SAME** |
| 9 | 输出布局（`dst = idx*640` 的代数恒等 + 门/up 列映射双射覆盖 [0,640)） | **SAME** |
| 10 | clamp/silu 分工（旧在 pair 写盘前 clamp；新由下游 `swiglu_limit_batched` 施加） | **DIFFER（无害）** |

⇒ **取数层已排除**：新路径索引与"输出正确"的旧路径**逐项等价**。

### 它给出的 5 处小改动（已落 D3/D4/D5，其余记录）
| # | 项 | 处理 |
|---|---|---|
| **D3** | scatter 的 `live` 未做与 gather 相同的 `min(kBm)` clamp ⇒ counts 被写坏时越界读写（静默错值而非崩溃） | **已修**（两侧同式）✓ |
| **D4** | `dsv41_moe_bs_debug_gather` 把 **`eid` 当 `nseg`** 传 ⇒ 该诊断仪器只填前 `eid[0]` 段、**任何用它做的 parity 结论假通过/假失败** | **已修**（先上行 nseg 到 `g_nseg` 再传）✓ |
| **D5** | `moe_bs_weights` 测了 `u_stride`（W3 的专家间距）**却丢弃**，kernel 对 W3 复用 `w_stride`（今天成立但属**隐式契约**） | **已修**（arm 条件加 `u_stride != w_stride`）✓ |
| D1 | 激活标度强制走 ue8m0：`quant_fp8` 在 `amax==0` 时把标度抬成 `1e-30`（非幂次）被折成 `2^-100`；**该 32-K 块内激活全零 ⇒ 数值无害**，但暴露"bs 臂正确性依赖 `round_scale=true` 的幂次输出"这一隐式耦合 | **暂不改**（改 `dsv41_kernels.cu:156` 会动共享量化器数值 ⇒ 必须走转正流程）；已在文档与下面记录 |
| D2 | 中间态 `ex_act` 旧路径 clamp、新路径不 clamp（**净结果同**，但中间态字节不同）⇒ 任何在 gate/up 与 swiglu 之间读 `ex_act` 的环节会看到未 clamp 值 | **记录**（若要逐位可比需在 scatter 加同一 clamp，代价是 ABI 加形参） |

## §70 精度合入台账（主树现状）+ 合并方法教训

| 门控（env） | 语义 | 默认 | 状态 |
|---|---|---|---|
| `DSV41_ROUTED_DOWN_QUANT` | 路由权重时机 + routed-down 输入量化（§14/§21） | OFF | **已在主树** ✓（更早合入） |
| `DSV41_WINDOW_KV_QUANT` (+`_DBG`) | 窗口 KV 的 fp8(block32, e8m0) 就地量化往返（A2） | OFF | **已合入主树** ✓ |
| `DSV41_INDEXER_FP4_RT` (+`_DBG`) | indexer q/k 的 fp4(block32, 幂次标度) 往返（A4） | OFF | **已合入主树** ✓ |
| `DSV41_COMPRESS_LATENT_QUANT` (+`_DBG`) | 压缩 KV latent 的 fp4(**block16** + **e4m3 非幂次**标度) 往返（A3） | OFF | 实现中（`prec-a3-latent-quant`） |
| `DSV41_ATTN_P_BF16` (+`_DBG`) | attention PV 的**概率操作数 bf16 舍入**（I3） | OFF | 实现中（`prec-i3-pv-bf16-p`） |

### 合并方法（**教训**：不要整文件拷贝 worktree）
- ❌ **整文件拷贝**：worktree 基于较早提交，且可能与我后续的修改**同文件**（如 A2 与我的 D5 都改 `chain_dev.rs`）
  ⇒ 会**静默覆盖**我的修复（D5 就差点被覆盖）⇒ **禁止**。
- ✅ **导出 worktree 的未提交 diff，再平铺应用**（本次成功路径）：
  1. `cd /tmp/prec-XX && git diff HEAD -- crates kernels > /tmp/xx.patch`（**排除文档**避免与我大量编辑冲突）
  2. `git apply --check` 预检 → `git apply`（无冲突时）
  3. 冲突时（本次 A4 与已合入的 A2 在同一文件尾部新增、上下文偏移）改用
     **`patch -p1 -F3 --no-backup-if-mismatch`**（纯新增函数的偏移可容忍，`fuzz 1` 成功）；并确认**无 `.rej`/`.orig` 残留**。
  4. **必须双向确认落地**：`grep -c <新符号>`（两侧非 0）+ `git diff --stat`（与 subagent 报告一致）
     + **检查我此前的修复仍在**（本次专项 `grep -c "u_stride != w_stride"` = 1 ✓）。
  5. 验收：本地 `cargo check --workspace` + 远端**单文件**编译冒烟（`dsv41_glue.cu`、`moe_bs_shim.cu`）。
- ⚠️ **注意**：`git merge <worktree-branch>` **无效**——subagent 的改动通常是**未提交**的，
  分支尖端仍指向基线提交（本次 `git merge prec-a2` 返回 "Already up to date"）。

## §71 【精度·独立 CPU 验证】已合入的两门单测 **12/12 全绿**（主 agent 实跑）

```
cargo test -p ferrite-models --lib win_kv_quant   → 6 passed; 0 failed
cargo test -p ferrite-models --lib routed_down_prep → 6 passed; 0 failed
```
关键用例（都是"对照独立参考实现"型，不是自证）：
- `win_kv_quant_tests::cpu_reference_matches_the_python_reference_{floor,all_ones,spike_and_tails,large_spike}` ✓
  ⇒ Rust 参考实现与**复现官方 `act_quant` 语义的 Python 参考**逐值一致；
- `win_kv_quant_tests::cpu_reference_scale_is_a_power_of_two` ✓（守住 §D1 依赖的那条契约）；
- `routed_down_prep_tests::cpu_reference_matches_an_independent_python_implementation` ✓、
  `cpu_reference_applies_route_weight_before_bf16` ✓（正是官方 `model.py:849` 的顺序）、
  `bf16_rn_is_round_to_nearest_even` ✓、`cpu_reference_uses_the_reference_amax_floor` ✓。

⇒ 这两门的**语义正确性**在 CPU 上已被独立参考背书；剩下的是 GPU 上的**端到端**对拍（§58 的 DBG 五点回读 + 文本红线）。

## §72 关闭审计的两处 UNKNOWN + 明确剩余的最后一项

### (a) engram 数值路径 —— **关闭**
官方 `ref_inference/engram.py` 全文**没有任何量化调用**（对 `act_quant|fp4_act_quant|quant|float()|bf16|half()`
的 grep 零命中）⇒ engram 的数值只经过**常规量化线性层**（`model.py:181-207` 的 `linear()`，即 A1）
与 n-gram 查表/加和 ⇒ **A1 已对齐即 engram 对齐** ✓（我方 `engram_proj_mrows`/`engram_gather` 用的是同一套
`dsv41_quant_fp8(block32, round_scale)` ✓）。

### (b) vision —— **不适用（N/A）**
本战役是**纯文本推理**，不加载 visual 分支 ⇒ vision 的精度面不在范围（`vision_rope_theta` 等仅存在于 config）✓。

### (c) 剩余最后一项：**累加序（≤ulp 级）**
审计里若干项标为"ALIGNED（乘法结合顺序不同 ⇒ ≤ulp）"或"MISALIGNED（累加序）"——
即数值**顺序**差异（torch 的归约树 vs 我们的 warp/分块归约）。这类差异是 **≤1 ulp × 项数**量级，
要"完全对齐"必须复刻官方逐位归约顺序（代价高）。
**处理**：按用户"完全对齐"的要求，这类项**必须由 GPU 量化其实际影响**（属 NEEDS-GPU N3），
在逐门转正时用 `wq_check.py` + `[NC]`/DBG 对拍观察是否出现可观测漂移；若只是 ≤ulp 抖动则记录为
"顺序差、量级 ≤ulp"并保留（不强求逐位）。

## §73 `[NC]` 探针被挡住**最终根因**：整步图/MoE 图的 **capture 录制期覆盖了 BS 调用**

决定性窗口（`5f5721cd`）的输出预览首次出现：
```
[moe-bs][DIAG] Eid[0..36): 0 0 0 0 0 0 0 0  [NC] entered (handwritten path; capture guard next)
```
⇒ ① **M2 修复生效**（手写路径的探针已可达 ✓）；② 探针随后**停在捕获守卫** ⇒ 该调用当时**确实处在 capture 内**。

**根因（主 agent 查证）**：本仓唯一的 `cudaStreamBeginCapture` 入口是
`ferrite-kernel/src/cuda.rs:graph_capture_begin`（置位 capturing 标志），其**调用点**在
`crates/ferrite-models/src/dsv41/chain_dev.rs:8470`（step 图发射路径）、`:9524`（`capture_verify`）、
`:20074`（**per-layer MoE 图** `self.moe_graph` 的录制/重放路径）。
⇒ **图的录制是"把整个 step（含 MoE）录一遍"**，所以录制期间任何在该 stream 上发起的调用——
包括我们的 BS 探针——都会看到 `cudaStreamIsCapturing != None`。
`DSV41_GRAPH_STEP=0` / `FERRITE_GRAPH*`=0 只能影响**是否使用**图，**不能避免"首次录制"这一瞬间**。

**结论与处置**：
- `[NC]` 五点回读是**诊断增强**，不是判定必要条件 ⇒ **不再为它投入**。
- **判定改用已具备且机械化的手段**：`~/wq_check.py --log`（重复/乱码/1..61 计数/EAGER 豁免）+ `[NC] entered` 标记
  + BS 臂自身的 GATHER-DIAG/Eid DIAG 行；性能用 `~/arm_run_fast.sh`。
- 若要真正拿到设备侧数值，正确做法是**把对拍整体搬到设备内**（参考值与待测值都在 device 上比，
  只把"结论计数"带回 host）——记为**可选后续**，不与当前主线争资源。

## §74 规划洞察：精度转正**不被 BS 臂阻塞**，可与 MoE 正确性解耦推进

四处精度改动的作用阶段与 fp4 MoE BS 臂（gate/up 的 tcgen05 路径）**互不相交**：
| 门控 | 作用阶段 |
|---|---|
| `DSV41_ROUTED_DOWN_QUANT` | MoE 的 **down**（w2）投影，且是"路由权重/输入量化"层面，与 gate/up 的 MMA 无关 |
| `DSV41_WINDOW_KV_QUANT` (A2) | attention 的 **窗口 KV**（ring），在注意力之前 |
| `DSV41_COMPRESS_LATENT_QUANT` (A3) | **压缩 KV latent**（compressor），在压缩投影处 |
| `DSV41_INDEXER_FP4_RT` (A4) | **indexer** 的 q/k |
| `DSV41_ATTN_P_BF16` (I3) | attention 的 **PV**（概率操作数） |

⇒ **它们的验收（DBG 对拍 + 文本红线 + 全 gate 回归）只需 GPU 窗口，不依赖 BS 臂是否已正确**。
⇒ 因此窗口分配上可以**并行推进两条线**：一条修 BS 臂正确性，另一条逐门转正精度。
（唯一注意：两者都会改变文本 ⇒ 做**文本红线**时最好**分窗口**，避免归因混淆；DBG 对拍则完全独立。）

## §75 I3 的**顺序语义已复核确认**（§68 留的待确认项已闭环）+ 合并状态

§68 把"bf16 舍入与在线 softmax 的 `corr` rescale 谁作用在谁身上"列为**决定实现正确性**的待确认项。
实现线（`prec-i3-pv-bf16-p`）逐行复核官方后确认：
- **bf16 舍入发生在 `exp` 之后、乘 KV 之前** ✓（即 `acc_s_cast = acc_s.to(BF16)` 在 `T.gemm(acc_s_cast, kv)` 之前）；
- **在线 softmax 的 `corr`（scores scale）作用在累积器 `acc_o` 上** ✓
  ⇒ 因此**只有"当前块/当前 slot 的概率"需要 bf16 舍入**（累积器本身保持 f32）；
- 与 §66 的交叉确证一致（官方 `acc_o` 是 f32 累加器 ✓）。

**合并状态**：I3 的 patch（`chain_dev.rs` +466、`device.rs` +28、`ops.rs` +55/−34、`dsv41_kernels.cu` +331/−34）
已按 §70 方法应用（含删除行 ⇒ 用 `patch -p1 -F3`，**无 `.rej`/`.orig` 残留**），
落地确认：新符号 19 + 13 处 ✓、**此前四处修复全部仍在**（D5=1 / A2=8 / A3=9 / A4=17）✓；
门控 `DSV41_ATTN_P_BF16` 默认 **OFF**。验收（cargo + 全库单测 + 内核单文件）在跑。

## §76 精度五门的"半挂"覆盖性核查（设计内非法 ⇒ 必须逐门确认）

**纪律**：接线契约要求"要么全部改、要么明确说明哪些没改以及为什么"——半挂配置是**设计内非法**。
主 agent 逐门实读调用点（不只看 subagent 报告）：

| 门 | 需覆盖的路径 | 实读结果 |
|---|---|---|
| **A2** `DSV41_WINDOW_KV_QUANT` | 融合路径（`ring_win_fuse`）+ 非融合路径（`ring_append`） | ✓ **一处插入即覆盖两者**：量化作用在**行**上、位于两条 ring 内核之前（`chain_dev.rs:21020-21038` 自带论证："ring_win_fuse()/DSV41_RING_WIN_FUSE=0 因此不改变该臂"） |
| **A3** `DSV41_COMPRESS_LATENT_QUANT` | 双链的两条路径 | ✓ 两个调用点 `chain_dev.rs:13050` 与 `:22545` |
| **A4** `DSV41_INDEXER_FP4_RT` | indexer 的 **k** 与 **q** | ✓ 包装器 `indexer_fp4_rt_launch`（`:22125`）的两个调用点：`:22242` 传 `b"k\0"`、`:22345` 传 q 侧 |
| **I3** `DSV41_ATTN_P_BF16` | **全部** PV 乘加点（含 spec/verify 复用路径） | 由 `prec-i3-pv-bf16-p` 交付清单给出（报告含调用点清单）；合并后已做符号/旧修复确认 |
| `DSV41_ROUTED_DOWN_QUANT` | routed down 的乘法点 + 调用方不再传 `row_weight` | ✓ §53 已逐行核实（唯一三处改动） |

⇒ 五门均**无半挂**（或已按契约说明）。**转正时仍须逐门单独验证**（§64 的流程）。

## §77 【精度·最后一项闭环】累加序（≤ulp）穷举审计：**10 项可 CPU 关闭，风险唯一落点是 head logits**

审计（`accum-order-audit`，纯静态 + 纯 CPU f32 仿真标定）的核心论断（全文最重要的一句）：

> **官方每个算子末端都有一个 bf16（或 fp8）量化边界** —— `RMSNorm → (w*x).to(dtype)`、
> `Expert → x.to(dtype)`、`MoE → y.type_as(x)`、attention 的 `o` 是 BF16 tensor、`hc_pre/hc_post → .to(x.dtype)`。
> ≤1e-6 的 f32 序差被这些边界吞掉后，只表现为"**每行 0.3~1 个元素移动 1 个量化 ulp**"。
> **唯一的例外是 head 的 logits：它是 f32、没有边界、末端直接 argmax** ⇒ 全部风险的落点。

**量级标定（可复算）**：`u = 2^-24`；串行 n 项 `|δ| ≲ (n-1)u`；平衡树深度 d `|δ| ≲ d·u`；
量化边界 ulp：bf16 = 2^-9（值∈[1,2)）、e4m3 = 2^-4。实测两种结构差 ≈ **5e-7（median，~8 ulp）**
⇒ 代入 bf16 边界：`5e-7/1.95e-3 = 2.6e-4` × 5120 元素 = **每行约 1.3 个元素偏 1 个 bf16 ulp**。

| 分档 | 项数 | 处置 |
|---|---|---|
| **A. CPU 即可判定无害（立刻关闭）** | **10** | 下游有量化边界 + 末端非 argmax/比较 |
| **B. 量级已可判无害，建议 1 次 GPU 抽检** | 4 | 分母链/跨 rank 域；风险在"边界是否真的生效"这一**配置假设**（与 §64 的出货 env 清单同源） |
| **C. 必须 GPU 量化** | **3** | 无量化边界 + 末端是 argmax/topk：**head logits**、indexer topk、draft/verify 程序差（直接影响 accept） |
| 其中**可廉价对齐** | 2 | #5 MoE down 的 slot→expert-id 求和排列；#13 swiglu 的 silu 形式 |

⇒ **行动**：C 档 3 项纳入 GPU 验证清单（与精度五门的转正同窗口执行即可）；
B 档 4 项在转正时用 `wq_check.py` 观察是否出现可观测漂移；
**A 档 10 项即刻关闭**（不再投入）。另有一个对拍前提值得记住：
`ref_inference/generate.py:118` 设 `torch.set_default_dtype(torch.bfloat16)`（参考实现全程 bf16 默认）
且 `:108` 的 `world_size` **由启动环境决定** ⇒ 对拍臂的 world size 会影响 AR 项是否成立。

## §78 五门的 OFF 路径安全性已逐门核实（两类实现方式）

| 门 | 实现方式 | OFF 路径的保证 |
|---|---|---|
| `DSV41_WINDOW_KV_QUANT` (A2) | **host 侧门控**：`window_kv_quant_on()` 在 gate 关时**在任何发射之前返回**（`chain_dev.rs:21035-21037` 自带论证） | 不调用 ⇒ **逐字节等价** ✓（更强保证） |
| `DSV41_COMPRESS_LATENT_QUANT` (A3) | 同上（`:13048/22540` 的 `if compress_latent_quant()` 包裹） | 不调用 ⇒ 等价 ✓ |
| `DSV41_INDEXER_FP4_RT` (A4) | 同上（`if idx_fp4_rt { … }`） | 不调用 ⇒ 等价 ✓ |
| `DSV41_ROUTED_DOWN_QUANT` | host 侧 + 每调用点分支（§53 逐行核实） | 三处调用点原样传参 ⇒ 等价 ✓ |
| **`DSV41_ATTN_P_BF16` (I3)** | **内核侧参数**：`dsv41_pv_prod(e, kv, p_bf16)` 在 `p_bf16==0` 时返回 `e * kv`，调用方表达式 `acc*corr + <product>` **逐字未改**（`dsv41_kernels.cu:996-1011`） | 同一表达式树（编译器可见原始 `acc*corr + e*kv`）⇒ 等价 ✓ |

⇒ 前四门是"**不调用**"型（最强保证）；只有 I3 必须**改既有内核**，故必须做内核侧的 OFF 等价论证——已核实 ✓。
**转正时的不变量**：任何一门转正后，其余门的 OFF 路径**不得**被牵连改动（逐门单独转正即保证这点）。

## §79 【去风险】五门的 DBG 回读是"天然延迟型"，推广窗口不会被图捕获吃掉

§58 曾记录"DBG 也带图捕获守卫"（现已逐门实读确认），并担心重演 `[NC]` 那种"永不再试"的静默失败。
**实读结论：不会**。五门的 DBG 块形式统一为（以 A3 为例，`chain_dev.rs:21023`）：
```rust
let dbg = if dbg_arm && compress_latent_quant_dbg() && !self.dev.capturing() {
    Some(self.dev.alloc(LAT_Q_DBG_FLOATS * 4)?)      // 捕获时既不分配
} else {
    None                                             // …也不消耗任何一次性标志
};
```
⇒ **捕获时只是本轮不跑**，标志不消费 ⇒ 后续任一**非捕获调用**（且 `dbg_arm` 成立，如"本步有 group 完成"）
会**自动补跑并打印** ✓。这与 `[NC]` 的原缺陷不同（它原先"先置 `done=true` 再拷贝" ⇒ 一旦捕获就永不再试）——
`[NC]` 已按此模式改为延迟执行（§73/§78）。
⇒ **推广窗口（`~/promote_all.sh`）可直接跑**：DBG 会在首次非捕获的合格调用上输出；
若某个门在整轮都没打印，应按 §58/§73 的顺序排查（先确认 `dbg_arm` 条件是否成立，再确认是否全程被捕获）。

## §80 本轮两项"数值件"的主 agent 亲自核验（不依赖 subagent 报告）

### (a) 激活 SF 的幂次→ue8m0 转换：正确 ✓
`moe_bs_shim.cu:489-495`：
```c
__device__ __forceinline__ uint8_t tl_bs_f_pow2_to_ue8m0(float s) {
    if (!(s > 0.f)) return 0;                                  // 非正 → 0 字节（"无标度"）
    int e = (int)((__float_as_uint(s) >> 23) & 0xFFu) - 127;    // f32 指数域
    if (e < -127) e = -127;
    if (e > 127)  e = 127;                                     // 钳到 e8m0 值域
    return (uint8_t)(e + 127);                                 // 加偏置
}
```
与 e8m0 定义 `2^(b-127)` **一致**：`e=0 → byte 127 → 1.0` ✓、`e=-127 → byte 0 → 2^-127` ✓、
`e=127 → byte 254` ✓。且与 `dsv41_experts_mxf4.cu:287` 的既有实现**逐字同源**（注释自述）✓。

### (b) 权重 SF 的平面尺寸：三方一致 ✓（详见 §59）
Rust 断言（字节）= 官方 TMA `gstride[1]` = kernel 的词步长 × 4 ✓。

⇒ 这两项都是"不经 subagent、主 agent 自己读出来的"结论，与 §44–§46/§59 同属"参数面已排除"的证据链。

## §81 【精度·重要】head logits 的真差异**不在序差，而在输入边界**（含量化敏感度）

subagent `head-logits-verify-prep` 的纯 CPU 交付（含 f32 仿真标定），三条关键结论：

### (1) 量级：输入边界项比序差项大 **162~560 倍**
| 差异项 | Δ（等效 logit 偏离） | 含义 |
|---|---|---|
| 归约序差（我方 `gemv_bf16_kernel` 的 lane 串行 + 蝶形树 vs 官方 cuBLAS 树） | ~1.2e-4（归一化 ~0.4 ulp，与 §77 一致） | 每 ~**2.8e4** token 翻一次 top-1 |
| **head 输入的 bf16 边界**（官方 `RMSNorm.forward` → `(w*x).to(bf16)`；我方 `bf16_snap` **被 `DSV41_BF16_TRUNCATE` 门住、默认 OFF**） | **~1.95e-2** | 每 ~**174** token 翻一次 top-1 |
⇒ **我方 head 在默认臂下"精度偏高"**（输入未吸附到 bf16 网格）——正是用户红线里"高"的那一半。
敏感度模型：`P(flip) ≈ Δ/β`，`β = σ_logit/√(2 ln V) ≈ 3.09`（V=129280、σ_logit=15）。
反向：要让**序差**把 top-1 翻掉，需要输入扰动 `|δx|∞ ≈ 1.4e-7`，比 bf16 边界自带的舍入（7.7e-5）**小 550 倍**。

### (2) 这是 C 档里**唯一可廉价对齐**的一项，但需**专用门控**
现状：`DSV41_BF16_TRUNCATE` 是"**整批 17 个边界一起开**"（`chain_dev.rs:748-750`），而 head 输入只是其中一处
（`chain_dev.rs:8999 bf16_snap(s.xn, dim)`）。
⇒ 要逐项转正（§64 的纪律），应把 **head 输入边界**（以及 `hc_pre collapse` 那处 `:8962-8972`）拆成**专用门控**，
以便单独验证与开启。（出货脚本已开整批门 ⇒ 出货配置下**该项其实已对齐** ✓，但默认臂未对齐。）

### (3) 反直觉副产品（anti-ulp）：我方 head 归约**本身**偏离真值更多
我方 `gemv_bf16_kernel`（lane 32 串行 + 蝶形 `off=16,8,4,2,1`）对"正确舍入真值"的误差：
**74 ulp**；朴素串行 43 ulp；平衡对树 **27 ulp**。
⇒ **"复刻顺序"在此不是精度折中而是精度损失**；若要动 head 归约，方向应是**平衡树**（且必须单独门控 + 单独验证）。

### 附带结论（无需动作）
- head 权重：官方 `model.py:1006` 保留 **fp32**（注释明写 checkpoint 是 bf16），我方 `head.weight` 按 **bf16 驻留**、
  kernel 内 `__bfloat162float` 无损加宽 ⇒ **数学同值** ✓（累加序不同 ⇒ ≤ulp）。
- 词表切分（`DSV41_HEAD_SLICE` 默认 ON）**不改数值** ✓；argmax 平局规则不同（官方 CUDA argmax 未定义 vs 我方取最低索引）
  ⇒ 只在精确平局处不同（测度零）✓。

## §82 §81 的落地备注：head 输入边界在**三处**，且外层门控是"融合门"而非专用门

实读（`chain_dev.rs`）：
- **三处** `self.bf16_snap(self.s.xn.ptr as *mut f32, dim)?;`：`8999`（head 主路径）、`20242`、`20376`（verify/spec 路径）。
  ⇒ 若要为 head 输入边界做**专用门控**，必须**同时覆盖三处**（半挂配置是设计内非法）。
- `bf16_truncate()` 在本文件里的使用形态是**融合门的负项**（`!bf16_truncate()`），例如
  `20915/20945`（norm 融合）、`21209`（`gemm_fp8_norm` 融合）、`22461`（compress 融合）、`23480`（down 融合）
  ⇒ 即：**该门同时承担两个角色**——① 17 个算子的 bf16 边界（含 head 输入）② 若干融合路径的开关。
  **这解释了为什么"拆专用门"不能只改一处**：head 边界不在某个 `if bf16_truncate()` 里，而是散布在三处
  且与融合门交织。
- **约定瑕疵（仅记录，不改）**：`bf16_truncate()` 用 `.map(|v| v != "0")`（空串/垃圾值也会**开启**），
  与新增五门的 `.starts_with('1')` 不一致。**但它是既有门，改动会改变现有用户的行为** ⇒ 按"退化默认关"的
  精神**保持原状**，只在此记录。

**处置**：head 输入边界的专用门控属**可选优化**（其唯一收益是"逐项转正"的粒度；而出货配置已开整批门 ⇒
**该项在出货配置下本就对齐** ✓）。故**不为此投入**，除非后续需要单独 A/B 该边界。

## §83 【流程事故 + 修复】精度合并引入**重复函数定义** ⇒ 完整构建失败；单文件检查漏掉了它

**现象**：T3 回合的输出显示 `1 error detected in the compilation of "./dsv41_glue.cu"` + `build failed`
⇒ 该轮二进制**陈旧**（臂跑的是旧 `.so`）⇒ **该轮结果作废**。

**根因**（用与 build.sh **相同**的标志复现才看到）：
```
dsv41_glue.cu(3281): error: function "<unnamed>::glue_e2m1_encode" has already been defined
                                      (previous definition at line 3001)
```
⇒ **A3 的合并**加了带 `fminf(fabsf(v), 6.0f)` clamp 的 `glue_e2m1_encode`，
**A2/A4 的合并**又加了同名（"假定调用方已 clamp"）的版本 ⇒ 同一匿名命名空间内**重复定义**。

**修法**：**保留带 clamp 的那一版**（官方的 `fp4_quant_kernel` 正是 `T.clamp(x/s, ±6)` 之后再 `Cast(FP4,…)`
⇒ clamp 属官方语义），删除无 clamp 的重复版；其调用方改用前者。
**复验**：用 build.sh 的真实标志（`-O3 --use_fast_math`）编译 ⇒ **0 error**，产出 1.26 MB 目标文件 ✓。

**教训（重要，已写进本次流程）**：
1. **合并后必须用项目自身的构建验收**（`bash build.sh 103a`），**不能只靠单文件编译**——
   我的单文件检查当时报 RC=0，却没暴露这个重复定义（标志/上下文不同）。
2. **多个 subagent 往同一个大文件（`dsv41_glue.cu`）加"同名小工具函数"是重复定义的高危模式**——
   以后给 subagent 的 brief 里应要求"新增辅助函数必须带唯一前缀（如 `a2_`/`a3_`/`i3_`）"。
3. **看到 `build failed` 时，那一轮的任何 e2e 结果都不能用**（二进制陈旧 = 跑的不是你以为的代码，
   与本项目 #1 测量偏置陷阱同类）。

**§83 补记（主 agent 已做的普查）**：为确认"重复定义"是否还有同类，对四个被合并的 `.cu` 做了一次扫描
（`dsv41_glue.cu` / `dsv41_kernels.cu` / `tilelang_gen/moe_bs_shim.cu` / `tilelang_gen/moe_bs_handwritten.cu`
里所有 `__device__`/`__global__`/`static` 函数定义名）：
**共 73 个定义、重复名 0** ⇒ `glue_e2m1_encode` 是**唯一**的合并碰撞，已被清除 ✓。

## §84 合并后"集成卫生"三项普查（主 agent 亲自做，全部通过）

多次把 subagent 的改动合进同一批大文件后，除"数值正确性"外还有三类**与语义无关但会静默毁掉一轮**的风险。
本轮逐项普查结论：

| 普查项 | 方法 | 结果 |
|---|---|---|
| **重复函数定义**（本次真出过事，§83） | 扫四个被合并 TU 里所有 `__device__`/`__global__`/`static` 定义名 | 73 个定义、**重复名 0** ✓（`glue_e2m1_encode` 是唯一一处，已清） |
| **env 名冲突**（一个门误开另一个门） | 扫全仓 `DSV41_[A-Z0-9_]+` 引用 | 222 个引用；**新增五门名字唯一** ✓；仅 4 个 env 被多处读取，均为"定义点 + 使用点"的正常分离 ✓ |
| **launcher 越界守卫**（新内核的 grid/block 与传入尺寸不匹配 ⇒ 静默越界） | 逐门读 `extern "C"` 入口的前置校验 | A2 `dsv41_win_kv_quant_rt`：校验 `kv/cols>0/block∈(0,256]且%32==0/cols%block==0` ✓；A3 `dsv41_compress_latent_fp4`：校验 `x/rows/hd>0且%16==0/ld` ✓；A4 `dsv41_indexer_fp4_rt`：校验 `x/rows,cols>0/cols%32==0` ✓；I3 的门判定与 `ops.rs` 镜像一致（`[0]=='1'` ≡ `starts_with('1')`）✓ |

⇒ 三类风险**均已排除**。**纪律**：每次多线合并后都应跑这三项普查（成本几分钟，能避免整轮 GPU 窗口作废）。

## §85 【状态快照】本 session 末（供下个 session 30 秒接上）

### 已完成且已验证
| 面 | 状态 |
|---|---|
| **fp4 smem 语义** | 定谳（§47）：packed 数据 + 16 B 容器只用前 8 B；写公式 `hw_pack_sw128`（**默认已转正**）；描述符/递进/idesc 保持官方原值；独立算术核验（§54）与硬件 relerr=0 实证 |
| **BS 臂接线** | 修了两个真缺陷：swapAB epilogue 输出打包（§62，修复前 640 个输出里 512 个位置错）；`HANDWRITTEN` 分支 `goto` 跳过 `[NC]` 探针（§65 M2）；我的仪器化回归已回退（§65 M1） |
| **差异法审计** | 新路径 vs 旧路径（输出正确）**十项取数全 SAME**（§69）；D3/D4/D5 已修 |
| **精度五门** | `DSV41_ROUTED_DOWN_QUANT` / `DSV41_WINDOW_KV_QUANT`(A2) / `DSV41_COMPRESS_LATENT_QUANT`(A3) / `DSV41_INDEXER_FP4_RT`(A4) / `DSV41_ATTN_P_BF16`(I3) —— **全部已合入主树、默认 OFF**；单测 **116 passed**；OFF 路径逐门核实（§78）；无半挂（§76）；集成卫生三项普查通过（§84） |
| **累加序审计** | 17 项分档（§77）：10 项 CPU 即判无害 / 4 项抽检 / 3 项需 GPU；**风险唯一落点 = head logits**（§81：真差异是**输入边界**，比序差大 162~560 倍） |
| **合并纪律** | 已写进 `AGENTS.md`：用项目自身构建验收、`pipefail` 传播退出码、helper 唯一前缀、worktree diff 平铺应用 + 三项确认（§83 事故复盘） |

### 待办（按优先级）
1. **有效构建后的决定性回合**（§83 修好后重开）：看两朝向的**机械文本判据**（`~/wq_check.py`）
   —— 正确则进 2；不正确则用 `~/bs_vs_old.sh`（差异测试）+ 延迟 `[NC]` 探针定位。
2. **`~/endgame.sh`**：全 gate 回归（`push400_hw_test.sh`，已 tee 落盘）+ 真实 p50。
3. **`~/promote_all.sh`**：五门逐项转正（每门：DBG 对拍 → 文本红线 → 快速臂无回归），
   转正后把对应 env **加进出货脚本**（`push400_hw_test.sh`/`verify_correct.sh`）。
4. **采纳"可廉价对齐"两项**（`cheap-align-two` 正在做：#5 MoE down 求和排列 / #13 swiglu silu 形式）。
5. **启用 `cp.async` 门**（`bs-cpasync-pipeline` 正在做；smem 预算已核算：双缓冲 67584 B ≪ 166912 B ✓）。
6. **C 档剩余两项**（indexer topk / draft-verify）按 `cgrade-verify-prep` 的手册上机抽检。

### 环境与脚本（远端 `~/`）
`arm_run.sh`（5 图门全关 + OUT/ERR/STEP 回写日志）、`arm_run_fast.sh`（图 ON，唯一可作性能 p50）、
`wq_check.py`、`bs_vs_old.sh`、`decisive2.sh`、`endgame.sh`、`promote_all.sh`、`promote_precision.sh`、`packed_matrix.sh`。
**编译检查**：必须在**仓库目录**编 shim（kernel 被 shim `#include`）；`.cu` 变则**双产物背靠背重编**。

## §86 工具与验证方案补充（cp.async 门 + 合并验收脚本）

### (a) `cp.async` 双缓冲门的**验证方案**（`DSV41_MOE_BS_CPASYNC`，等实现交付后执行）
它属于**性能门**，与精度门不同，判据必须**两条都过**：
1. **数值逐字节一致**：同一次会话内 `DSV41_MOE_BS_CPASYNC=0` 与 `=1` 两臂跑同一 prompt，
   用 `~/wq_check.py --log A --eager-file <B 的 OUT>` 判"同现"（等价 ⇒ SAME）；
   **更硬的判据**：`DSV41_MOE_BS_NUMCHECK=1` 的 `[NC]` 数值在两臂应**完全一致**（延迟探针在重放期命中）。
2. **性能**：两臂背靠背（同会话、同构建）取 `~/arm_run_fast.sh` 的 `[dsv41] step pos=` **p50**（图 ON ✓），
   预期 verify 34.5ms → ~22ms（交接文档的量化目标）；**禁止用图关掉的诊断臂数字**（§42）。
3. smem 预算**已核算**（§85/§84 附）：单 stage 装载 33792 B、双缓冲 67584 B ≪ `kSmem=166912` ✓
   —— 但 **C staging 的别名区间**必须在实现里重新确认（epilogue 之前不得再读 A/B）。

### (b) `~/merge_worktree.sh`（新工具：把"合并纪律"自动化）
```bash
bash ~/merge_worktree.sh <worktree-dir> <新符号> [必须仍存在的符号...]
```
它按 AGENTS.md 的合并纪律逐步执行：导出未提交 diff（排除 docs）→ 预检 → `git apply`（冲突则 `patch -p1 -F3`）
→ **三项确认**（新符号计数 / `diff --stat` / 传入的"必须仍存在"符号）→ 残留检查（`.rej/.orig`）
→ **重复定义普查**（§83 的教训，只扫被改动的 `.cu`）→ 提示用 **build.sh 真实标志**编译。
**不自动提交**（先看输出）。三次合并（cheap-align / cpasync / cgrade）都用它。

## §87 【判据澄清】精度门转正时"文本与开门前逐字一致"是**错误**判据

五道精度门的作用是**把我们的数值降到官方水平**（我方原本"精度偏高"）。因此：
- **门开 vs 门关的输出必然可能不同** ✓（例如 A2 让整条窗口 KV 活在 e4m3 网格上、I3 让 PV 的概率被舍到 bf16）
  ⇒ **不能用"逐字一致"当通过条件** ✗（那会把**正确**的转正判成失败）。
- **正确判据**（两条都要）：
  1. **DBG 对拍为 0**：该门自带的一次性回读与**主机侧独立参考**逐元素差 0（±1 ulp 可容忍）
     —— 这证明**我们忠实复刻了官方语义**；
  2. **红线不破**：`~/wq_check.py` 在转正臂上 PASS（不重复、不乱码、计数前 61 行有效）；
     **EAGER 对照**（同 prompt、关门基线）用于区分"模型行为"与"我们的 bug"——
     即：门开后输出变化是**允许**的，但不能退化成复读/乱码。
  3. 附带：**吞吐不回归**（`~/arm_run_fast.sh` 的 p50 与开门前同一会话背靠背比较）。
- **反面例子（勿重犯）**：把"输出变了"当成门有 bug；或把"输出没变"当成门生效了
  ——后者尤其危险（门可能压根没接上，见 §65 M2 的 `goto` 跳过与 §83 的陈旧二进制）。

### 与"性能门"的区别（对照 §86）
- **精度门**：**允许**输出变化（只要 DBG 为 0 + 红线不破）；判据是**数值忠实度**。
- **性能门**（如 `DSV41_MOE_BS_CPASYNC`）：**必须**数值逐字节/逐位一致；判据是**等价性 + 时间**。
两类门的判据**不可混用**。

## §88 BS 臂诊断预案（结果到手即可按序执行，不临场找工具）

现有可复用仪器（均已核实存在）：
| 工具/门 | 位置 | 用途 |
|---|---|---|
| `~/wq_check.py --log A [--eager-file B]` | 远端 `~/` | **机械红线**：重复/乱码/1..61 计数/EAGER 豁免 |
| `~/bs_vs_old.sh` | 远端 `~/` | **差异测试**：BS 臂 vs 旧 GEMV 路径（后者输出已知正确 = 地面真值） |
| `DSV41_DIFF_EAGER=1` | `chain_dev.rs:6730 diff_eager_probe` | 逐轮把 emitted 的每个 token 重放为**单行 forward**，报**第一个 mismatch 的 index 与绝对位置** |
| `DSV41_MOE_BS_NUMCHECK=1` | shim 的延迟探针（§73/§79） | `[NC]` 五点回读 vs 主机独立参考（**需重放期命中**） |
| `DSV41_DSPARK_DEBUG=1` / `DSV41_ACC_HISTOGRAM=1` / `DSV41_ORACLE_TAP=1` | dspark/acc_hist | spec 路径的逐轮 trace 与 accept 直方图（用于 C 档第 2 项） |

**判定顺序（文本判据到手后）**：
1. `wq_check` **PASS** ⇒ 进 `~/endgame.sh`（全 gate 回归 + 真实 p50），随后逐门转正（§87 判据）。
2. **FAIL** ⇒ 先 `wq_check --eager-file`（BS 关掉的那一臂）看是否**退化与 EAGER 一致**（一致 ⇒ 模型行为，不算 bug）；
   不一致 ⇒ 依次：
   a. `~/bs_vs_old.sh`（差异测试，最快指出"新路径 vs 旧路径"的偏离）；
   b. `DSV41_DIFF_EAGER=1`（给出**第一个 mismatch 的绝对位置**，把问题钉到某一轮/某一 token）；
   c. `DSV41_MOE_BS_NUMCHECK=1` + 延迟探针（若能在重放期命中 ⇒ 直接给出**哪个 (row, col) 的数值与参考差多少**）。
3. 每次只改**一个变量**并重跑；**每次改动后必须用 `bash build.sh 103a` 走真实构建**（§83：单文件检查会漏）。

## §89 两条筹备线的合并处置（一成一败）+ 三条流程教训

### (a) `cheap-align-two`（`DSV41_SEQ_ALIGN`，累加序 #5/#13）—— **回滚，待修正重交**
合并后**完整构建失败**，两处都是**结构性**错误（不是标志差异）：
1. `dsv41_experts_mxf4.cu:3811`：把 **Rust 门函数名 `seq_align`** 当内核实参传进 `.cu`
   （`.cu` 看不到 Rust 符号）⇒ 必须**穿过 C ABI 新增形参**，由 Rust 调用点传入。
2. `dsv41_glue.cu:244`：注释的 `//` 丢了 ⇒ `} The f32 this kernel writes back` 变成语法错误。
**处置**：把 5 个文件恢复到合并前（**不做 `git revert`、不改写历史**，改动完整保留在 worktree
`/tmp/cheap-align`），并**已开 `cheap-align-redo` 线**带上这两条错误 + "必须在真实标志下自验"的要求。

### (b) `bs-cpasync-pipeline`（`DSV41_MOE_BS_CPASYNC`，cp.async 双缓冲）—— **已落地并权威复验**
在 origin/main 上用真实口径编译（`tilelang_gen/*` 用 `-O2`）⇒ **0 error** ✓（1.9 KB 目标文件）。
它是以一个 `TEMP` 临时提交的形式落地的（见下教训 2），补丁本身经权威复验无问题。

### 三条流程教训（已写进 `scripts/merge_worktree.sh` 与 AGENTS.md）
1. **`moe_bs_handwritten.cu` 不能单独编译**（被 `moe_bs_shim.cu` `#include`；单编必假报错）。
   我的合并工具最初单编它 ⇒ **误报 4 个 error、错误拒收了一个好补丁**。已修：这类文件改判为**编 shim**。
2. **不要用"临时提交 + push"做远端编译门**：会污染 origin/main，且脚本若在 push 后、本地回滚前被打断，
   就会**留下游荡的 `TEMP` 提交**（本次真实发生）。已改为**临时目录 scp + 原地编译**，历史零扰动。
3. **验证顺序不可颠倒**：符号检查通过 ≠ 能编译。本次我先提交后验证，导致"坏提交进了主干"再回滚。
   工具已改为**先过真实编译门、再允许提交**（失败自动恢复工作树）。

## §90 【更正后的结论】"OUT 为空"与 AR v5 停滞：**非 nsys 专属，且是间歇性**

**现象**：连续数轮（F/T/U/W 之前的几轮）`wq_check` 报 `NO-OUT-LINE`，`arm_run` 打印的 `OUT:` 为空。

**根因**（主 agent 读 serve 日志尾部发现）：
```
[ar5-hang] rank=2 site=0 peer=6 need=2 cur=1 spins>5000000 TIMEOUT -> PARK     ← 刷屏
[ar5-hang] rank=5 site=0 peer=6 need=2 cur=1 spins>5000000 TIMEOUT -> PARK
```
⇒ serve 卡在 **all-reduce v5** 路径上（等待 peer 6、自旋超时后 PARK）⇒ **completions 请求永不返回**
⇒ `curl` 拿到空 body ⇒ 那一轮**根本没有文本可判** ✗。

**这与 BS 臂的正确性无关**——是**诊断臂的 env 缺了 AGENTS.md 明写必带的死锁规避配方**：
`DSV41_AR_V5=0` + `NCCL_NVLS_ENABLE=0` + `env -u FERRITE_P2P`（AGENTS.md「测量与工具纪律」第 3 条：
"死锁规避必带：`DSV41_AR_V5=0 DSV41_GRAPH_STEP=0` + `env -u FERRITE_P2P` + `NCCL_NVLS_ENABLE=0`"）。

**修复**（已落地 `~/arm_run.sh`）：
1. `COMMON` 增加 `DSV41_AR_V5=0 NCCL_NVLS_ENABLE=0`；
2. 发射行改为 `setsid nohup env -u FERRITE_P2P $BASE $COMMON …`；
3. 顺带加固请求：捕获 HTTP 码 + **有界重试 6 次** + 失败时把 `http=`/`body_head=` 写进日志
   （原先 `curl -s` 静默吞错 ⇒ 空 body 无迹可寻）。

**教训（流程级，比本次 bug 更值钱）**：
- **"无文本"必须先怀疑工具/环境，而不是模型**。本轮差点把"OUT 为空"误读成"BS 臂没修好"。
- **诊断臂的 env 配方应当从文档里"抄全"**（AGENTS.md 把死锁规避三件套写在测量纪律里，
  而 `arm_run.sh` 只带了 `FERRITE_GRAPH*` 那五个 ⇒ 漏了 AR 相关三件套）。
- 已给 `wq_check` 的提示补上"arm_run 的 OUT 只到 stdout"的历史坑；本次再加一条：
  **看到 OUT 为空先 `grep -c ar5-hang`**。

### §90 更正（用户质询 + 主 agent 取证，2026-09-14）
用户质疑："只有开 nsys 才会死锁吧" ⇒ 主 agent 直接统计各轮日志：

| 日志 | `ar5-hang` 行数 | nsys 参与 | `step pos` 行数 |
|---|---|---|---|
| `armrun_U1.log` | **56** | **0（无 nsys）** | **0（完全没跑起来）** |
| `armrun_F1.log` | 0 | 0 | 48（正常） |
| `armrun_P1.log` | 0 | 0 | 48（正常） |

**据此更正两处**：
1. **该停滞不是 nsys 专属** ✓（U1 的 56 条 hang 完全无 nsys 参与）。
   AGENTS.md 里那条配方确实写在 **nsys 语境**（"nsys 多跑…死锁规避必带…"），但现象本身不限于 nsys。
2. **但它也不是"缺 `DSV41_AR_V5=0` 就必然发生"** ✗——**F1/P1 同样没带那三件套却完全正常**（48 个 step 行）
   ⇒ **间歇性**。因此把"缺配方"称为 OUT 为空的**根因**是**过强**的表述，本节此前那样写是我的错。
   ⇒ 正确表述：**U1 那一轮的 OUT 为空，是 AR v5 停滞（间歇）导致的"完全没跑起来"**；
   而 F/T/P 等轮**确实产出了文本**（只是文本本身不对）——两者是不同的失败模式，不能混为一谈。

**处置（降级为缓解而非根修）**：`arm_run.sh` 现已带 `DSV41_AR_V5=0 NCCL_NVLS_ENABLE=0` +
`env -u FERRITE_P2P`（文档化的规避），**仅用于诊断轮**（它改变 AR 形态 ⇒ 性能数字仍必须来自不带它的轮次，
AGENTS.md 已明写"nsys 轮只看 kernel 相对倍数（AR 形态已变），吞吐数字必须来自非 nsys 轮"）。
请求侧加固（HTTP 码 + 有界重试 + 失败原因入日志）保留 ✓——它让"空 body"不再无迹可寻。

**方法论教训（比结论更值钱）**：**现象的相关性不等于因果**。我因"日志里刷屏 ar5-hang + OUT 为空"
就下了"根因"判断，而**同一时间窗内的对照组（F1/P1，同样缺该 env）却完全正常**——
**下结论前必须先找对照组**（本次由用户质询促成，感谢）。

## §91 项目对 ar5-hang 的既有定论（与 §90 的关系）+ 诊断臂的图门已显式钉住

用户质询后主 agent 检索项目文档，发现**已有系统性结论**（`docs/agent/400-final-frontier-analysis.md`）：
- `:27`：**"'batched 被 ar5-hang 阻塞'这句话必须精确为'batched + CUDA graph 被阻塞'"**；
- `:28`：**"SWALLOW **nograph**（`SWALLOW_STEP=1` + `VERIFY_GRAPH=0`）已实测 **0 ar5-hang + 零拉丁**"**；
- `:351`：**"只是'batched+graph'被 ar5-hang 阻塞——nograph 已 0 hang + 零拉丁"**。

⇒ **与本轮取证一致**：该停滞属"**batched + 图**"这一**窄配置**，`nograph` 实测 0 hang ✓。
⇒ **但 U1 那一轮尚未被完全解释**：我的诊断臂当时**已把 5 个图门全关**，而
`DSV41_VERIFY_GRAPH` / `DSV41_DRAFT_GRAPH` **默认本就是 OFF**（`chain_dev.rs:4233`、`dspark_dev.rs:186`）——
按文档它**不该** hang。可能的原因留待后续（例如 `DSV41_MOE_BS_*` 臂自身在首次请求时的某条路径、
或该轮的构建/时序差异）⇒ **不硬下结论**（§90 的教训）。

**本轮据此做的两件事（都是"不依赖假设"的稳妥化）**：
1. **把两个 DSpark 图门显式钉进 `arm_run.sh` 的 `GRAPH_OFF`**（`DSV41_VERIFY_GRAPH=0 DSV41_DRAFT_GRAPH=0`）——
   今天它们默认 OFF，但**默认值可能变**；诊断臂不应把"是否 nograph"留给运气。
2. `DSV41_AR_V5=0` 的缓解**保留但定位明确**：它是**改变 AR 形态**的钝器 ⇒ **只用于诊断轮**，
   性能数字必须来自不带它的轮次（AGENTS.md 已明写同一纪律）。

## §92 🎯【已确认的挂死路径】kernel 的 MMA 等待是**无界自旋** ⇒ MMA 不到达就永久挂死

主 agent 读 `kernels/cuda/tilelang_gen/moe_bs_handwritten.cu` **第 781 行附近**（k 循环末尾）实测原文：

```c
        if (lane == 0) {
            hw_tc_commit(mma_bar);            // 发 commit：全部 MMA 完成时 mbarrier 到达
        }
    }

    // (6) 所有线程等待 MMA 完成（mbarrier wait + syncthreads）
    if (tid == 0) {
        const uint32_t phase = k & 1;         // "phase 0 for first use, then alternating"
        asm volatile(
            "{
	.reg .pred P;
	"
            "WAIT:
	"
            "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;
	"
            "@!P bra WAIT;
	}"              // ← **无条件回跳**：P 永假 ⇒ 永久自旋
            :: "r"((uint32_t)__cvta_generic_to_shared(mma_bar)), "r"(phase));
    }
    __syncthreads();                          // ← 其余 127 线程在此等 tid 0
```

**机制（与线上事故逐条吻合）**：
- 若 MMA 因任何原因**没有 arrive**（被跳过、commit 次数与 wait 不匹配、`phase` 与实际相位错位、
  或某条 early `return` 让 MMA 未发射），**tid 0 就永久自旋**；
- 其余 127 个线程**堵在 `__syncthreads()`** ⇒ 整个 block 挂死 ⇒ 内核永不返回 ⇒ **serve 永不返回**；
- 其它 rank 在 all-reduce 里等这个 rank ⇒ 日志刷
  `[ar5-hang] rank=… peer=… need=… cur=… spins>5000000 TIMEOUT -> PARK`，且整轮 **0 个 step**（§90 的现象）。
⇒ 这与"间歇性"（只在 MMA 未到达/相位错位时触发）和"对照组正常"（F1/P1 没踩到）**都吻合** ✓。

**处置**：已开 `bs-wait-hang-proof` 线，要求把它改成**有界等待**（超限则打印诊断并返回 ⇒ 把"永久挂死"
变成"明确报错"），并顺带审计文件内其它等待点（`cp.async.wait_group`、`__syncthreads` 等由硬件/构造保证返回）。
**纪律**："无界变有界"是**安全属性**，可无条件生效（正常路径第一次 `try_wait` 即成功 ⇒ 行为不变）；
诊断打印用门控（`DSV41_MOE_BS_WAITDBG=1`，默认 OFF）。

**方法论**：这条路径是**读代码读出来的**（不是靠日志猜的）——与 §90 的教训互为镜像：
**先用代码确定"是否存在会永不返回的路径"，再用日志佐证现象**。

## §93 挂死审计（双文件）+ 一条方法论要点：**asm 级自旋对源码扫描不可见**

对两个文件做了"永不返回路径"审计：

| 文件 | `__syncthreads` | 潜在无界循环（C 级扫描） | `return` | 备注 |
|---|---|---|---|---|
| `tilelang_gen/moe_bs_shim.cu`（gather/scatter/pack_wsf） | **0** | 0 | 86 | 无屏障 ⇒ 无"提前 return 卡住屏障"风险 ✓ |
| `tilelang_gen/moe_bs_handwritten.cu`（kernel） | 12 | **0（但见下）** | 11 | 见方法论要点 |

**已排除**：
- **"提前 `return` 卡住 `__syncthreads`"**：kernel 体内的 `return` 全部位于**屏障之后**或**全体线程一致**的分支
  ⇒ 不存在"部分线程退出、其余线程永久堵在屏障"的构造 ✓（shim 侧更简单：0 个屏障 ✓）。

**⚠️ 方法论要点（本轮真正的收获）**：C 级扫描报"0 个无界循环"，**却漏掉了 §92 那个真正的挂死点**——
因为它是写在 `asm volatile("WAIT: … @!P bra WAIT;")` 里的**汇编级无条件回跳**，对任何源码级
`while/for/do` 正则**完全不可见** ✗。
⇒ **审计"是否可能永不返回"时，必须把 `asm volatile` 里的标签/跳转单独过一遍**
（本文件里 `asm` 共 20+ 处，`bra`/标签只在 §92 那一处，其余都是单指令无回跳 ✓）。
⇒ 这与 §90 的教训是同一枚硬币的两面：**现象的相关性**（§90）与**扫描的完备性**（本节）都会骗人，
唯一可靠的是**逐处读实现 + 用对照组佐证现象**。

## §94 §92 自旋的**判据本身是正确的** ⇒ 挂死只可能来自"MMA 真的未到达"

逐项核对 §92 那个等待的**配对与相位**：
| 项 | 实测 | 结论 |
|---|---|---|
| mbarrier 到达计数 | `mbarrier.init.shared::cta.b64 [mbar], 1` | 1 ✓ |
| commit 次数 | `hw_tc_commit(` 全文件 2 处 = **1 个定义 + 1 个调用点**（k 循环内，每轮一次） | 与 wait 一一配对 ✓ |
| wait 次数 | 同一处（k 循环内，每轮一次，`if (tid == 0)`） | ✓ |
| 相位判据 | `phase = k & 1`（k=0→0、1→1、2→0…，与"首次相位 0、其后交替"一致） | **正确** ✓ |
| 初始化可见性 | `fence.mbarrier_init.release.cluster` + `__syncthreads()` | ✓ |

⇒ **判据没错** ⇒ 自旋**只会在 MMA 真的没有 arrive 时发生**（例如 MMA 未发射、发射后出错、
或 commit 与实际 MMA 不匹配）。这一点很重要：**有界等待落地后，挂死会变成一份可读诊断**
（`k`、期望相位、mbarrier 状态），从而**暴露 MMA 未到达的真实原因**——而不是像现在这样只表现为 serve 卡死 +
其它 rank 刷 `ar5-hang`。**这是把"间歇挂死"变成"可定位故障"的关键一步。**

## §95 挂死与"非法指令"是同一枚硬币（据 §92/§94 的结构性推理）

读 MMA 发射段（`moe_bs_handwritten.cu:748-782`）：
```c
for (int ki = 0; ki < 4; ++ki) {
    ...
    if (lane == 0) {            // 单线程发射
        /* tcgen05.mma ... */
    }
}
if (lane == 0) { hw_tc_commit(mma_bar); }     // 紧随其后，同一 warp
```
⇒ **发射与 commit 结构上严格配对**（每轮 k 各 4 次 MMA + 1 次 commit），**不存在"跳过 MMA 却照常等待"** ✓
（§94 已确认相位与配对也正确）⇒ §94 的"MMA 未到达"只剩一种来源：**MMA 执行本身出错**。

而本项目**已独立知道**：本 kernel 的真实错误类型是 **illegal instruction**（"illegal memory access" 是
context poisoning 的二级效应，见 §57 的更正 + 早前的 SYNC-DIAG 结论）：
> **MMA 非法指令 ⇒ 不 arrive ⇒ `try_wait` 永假 ⇒ tid 0 无界自旋、其余线程堵在 `__syncthreads` ⇒
> 内核永不返回 ⇒ serve 卡死 ⇒ 其它 rank 在 all-reduce 里超时刷 `ar5-hang`。**

⇒ **两个此前的谜题（"间歇挂死"与"illegal instruction"）应视为同一故障的两个侧面**，
而**有界等待 + 诊断**（§92 的修复）正是让这个故障**可定位**的关键：
它会把"卡死"变成"打印 `k`/期望相位/mbarrier 状态后返回"，从此外层能拿到真实 CUDA 错误
（`cudaGetLastError` 的 illegal instruction），把间歇性变成可复现。

## §96 W 回合是**判别实验**：它的结果直接区分"AR 形态问题"与"我们 kernel 的自旋"

W 回合（`83c08750`）的配置特征：
- **已带 AR 缓解**（`DSV41_AR_V5=0` + `NCCL_NVLS_ENABLE=0` + `env -u FERRITE_P2P`）⇒ AR 形态已改变；
- **已带请求加固**（HTTP 码 + 有界重试 + 失败原因入日志）；
- **§92 的无界自旋仍在**（`bs-wait-hang-proof` 尚未交付）。

⇒ **判读表**（无论哪种结果都有信息量）：
| W 回合结果 | 结论 |
|---|---|
| 两臂产出**文本**（`wq_check` 能判） | AR 停滞是**可绕过的环境问题** ✓ ⇒ 回到主线：用文本判 BS 臂正确性（§88 预案）；同时 `[NC]` 应能命中 |
| **仍挂**（日志刷 `ar5-hang`、0 个 step） | 改 AR 形态**没用** ⇒ 停滞来自**我们这一侧**（最可能是 §92 的无界自旋：MMA 未到达）⇒ 立刻上 `bs-wait-hang-proof` 的有界等待 + 诊断 |
| 挂法与之前不同（如报明确 CUDA 错误） | 说明请求加固/日志改动已经让故障**可见** ⇒ 直接按那个错误定位 |

⇒ 因此 W 回合**不需要"正确"才有价值**：它是把"间歇挂死"二值化的一次实验。
（已同时把 `DSV41_VERIFY_GRAPH=0 DSV41_DRAFT_GRAPH=0` 钉进 `arm_run.sh` 的 `GRAPH_OFF`，见 §91。）

## §97 【基准】SGLang 官方博客（DeepSeek-V4.1-Flash kernel 优化）与我们进度的对照

来源：`https://www.sglang.io/blog/deepseek-v4.1-flash-kernel-optimization`
（"from 35 to 873 tokens/s"，**4× GB300、BS=1、attention TP4、MoE TP4**、random 4k/1k、**模拟 accept 5.5**）

### 16 步阶梯（他们自己的编号与实测 tok/s）
| # | 步骤 | tok/s |
|---|---|---|
| 1 | 首个可跑版本 | 35.2 |
| 2 | **MXFP8 GEMM（修量化块/scale 布局不匹配，直接吃到 Blackwell MXFP8 kernel）** | **117.8** |
| 3 | RoPE + FP4 融合 | 133.5 |
| 4 | mHC 按输入行选 tile | 141.1 |
| 5 | Reduce + Sinkhorn 融合 | 146.5 |
| 6 | **跨层共享 scratch（不再每层重建请求索引/缓冲）** | 148.4 |
| 7 | C2 pooling 融合 | 152.1 |
| 8 | mHC 统计与 attention/MoE **重叠（双流）** | 186.4 |
| 9 | 快路径默认开 | 186.6 |
| 10 | GEMV / norm / Engram gate | 203.3 |
| **11** | **DSpark（spec）打开** | **558.2** ← 单步最大跃升（×3） |
| 12 | verify / MoE 融合与重叠 | 718.8 |
| 13 | 小批量投影 / mHC 融合 | 761.7 |
| 14 | indexer 后处理 / Q-RoPE / WO-A 量化融合 | 802.4 |
| 15 | C2 verify 压缩融合（L2/L8/L14 的 pair-pool+norm+rope+quant+KV写 合一） | 853.5 |
| 16 | **MoE 从 EP4 改 TP4 + padding**（inter 576→640） | **873.6**（+2.22%） |

### 三条**直接适用于我们**的结论（重点）
1. **第 2 步是本战役的镜子**：他们 35→118 靠的是"**量化块/scale 布局与后端期望不匹配 ⇒ 走了慢速回退 GEMM**"，
   并明确总结：**"bringing up a new model 时，确认一个 GEMM 实际 dispatch 到哪个 kernel，通常比调 tile 更值钱"** ✓
   ⇒ 这正是我们 fp8/fp4 **scale 布局**（`SF` 行 pitch、容器语义 §47）工作的同一类杠杆，且是**最大单项** ✓。
2. **第 11 步证明 spec/MTP 是最大杠杆**（186→558）✓ ⇒ 我们"先把 BS 臂搞对、再做 verify 融合"的方向正确 ✓；
   他们的**第 12 步（verify/MoE 融合）** 正是我们 BS 臂（tcgen05 fp4 MoE）要吃的部分 ✓。
3. **测试口径应对齐**：他们用 **random 4k/1k + 模拟 accept 5.5** 作为可复现基准，
   并用"accept 长度由 server 配置控制、每版同输入"来保证可比 ✓
   ⇒ **我们的 push400/对比也应该统一到同一口径**（否则无法与 sglang 数字对齐）✓。

### 与我们现状的对应（粗略）
| 他们的步骤 | 我们的对应物 | 状态 |
|---|---|---|
| 2 MXFP8 布局 | fp4/fp8 的 scale/容器布局（§47 定谳、§83 构建） | **本轮主攻** ✓ |
| 3/7/14 融合 | 我们的 RoPE-quant 融合、compressor、indexer 融合（多条已落地） | 部分 ✓ |
| 6 跨层共享 scratch | 我们的跨层 scratch/常驻 buffer | 已做 ✓ |
| 8/12 重叠（双流） | 我们的 mHC/verify 重叠门（`DSV41_*_FUSE`/`*_MROWS` 系列） | 部分 ✓ |
| **11 DSpark** | 我们的 `DSV41_SPEC`/`DSPARK` 路径 | **已实现，正在修正确性** ✓ |
| 12 verify/MoE 融合 | **BS 臂（tcgen05 fp4 MoE）+ cp.async 双缓冲** | **本轮主攻** ✓ |
| 16 MoE TP4+padding | 我们是单机 TP8（拓扑不同，需按我们自己的 `TP` 重新评估） | 待议 |

⇒ **行动**：(a) 把 `push400_hw_test.sh` 的口径对齐到 **random 4k/1k + 模拟 accept 5.5**（可复现、可对比）；
(b) 继续按"**先正确性、再融合/重叠、最后 MoE kernel**"的顺序推进（与他们的阶梯一致）；
(c) 记住他们的 A/B 方法：**同轮背靠背、一次一个变量、accept 长度固定**。

## §98 【标定】与 SGLang 的可比性折算（"击败 sglang"必须先看清尺度）

博客成绩的硬件是 **4× GB300（attention TP4 + MoE TP4）**，我们是**单机 8× B300 TP8** ⇒
**只有折算到"每 GPU 吞吐"才可比**：

| 配置 | tok/s | GPU | **per-GPU tok/s** | 备注 |
|---|---|---|---|---|
| SGLang 首个可跑版本 | 35.2 | 4 | 8.8 | plain decode |
| SGLang 修好 MXFP8 GEMM（第 2 步） | 117.8 | 4 | 29.4 | plain decode |
| SGLang 前 10 步全开 | 203.3 | 4 | **50.8** | plain decode |
| SGLang DSpark 打开（第 11 步） | 558.2 | 4 | 139.6 | spec，**模拟** accept 5.5 |
| SGLang 全 16 步 | 873.6 | 4 | **218.4** | spec，**模拟** accept 5.5 |
| **ferrite 现状（本战役实测）** | **112.0** | 8 | **14.0** | plain decode，**无 spec** |
| 用户目标 | 400.0 | 8 | **50.0** | spec（目标态） |

**结论（必须诚实面对）**：
1. **plain decode 每 GPU：SGLang 50.8 vs 我们 14.0 ⇒ 落后 3.6×** ✗；
   更重要的是：SGLang 的 plain 50.8 相当于我们**目标态**（400/8 = 50）——
   也就是说**我们现在还没到"SGLang 第 10 步"的水平**，而他们后面还有 6 步（其中第 11 步单步 ×3）。
2. **我们的 400 目标只相当于 SGLang 全优化态的 23%**（50 vs 218.4，每 GPU）⇒
   若要"**击败 sglang**"，400 是**必经的中间里程碑**，不是终点。
3. **三条不可忽略的限定**（否则对比失真）：
   - 他们的 spec 数字用**模拟 accept 5.5**，我们当前是**实测 accept ~2.24** ⇒ 不能直接比；
     要可比，必须**固定 accept 口径**（要么都用模拟、要么都报实测 + accept 值）。
   - 他们 attention/MoE 都是 **TP4、4 卡**；我们**单机 TP8** ⇒ 通信域、EP/TP 划分、padding 策略都不同。
   - 他们的 plain 203.3 是**前 10 步全开**；我们 plain 112 且 **BS 臂正确性尚未闭环**。

**行动含义（按杠杆排序）**：
① **先把 BS 臂正确性闭环**（挂死 §92 的有界等待 + 诊断 ⇒ 才能拿到有效文本判据）；
② **GEMM dispatch 覆盖审计**（博客的**第一条**教训：确认实际 dispatch 到哪个 kernel 比调 tile 更值钱，
   他们靠这一条拿到 35→118 = **3.3×**）——已开线；
③ 继续按博客阶梯：**融合 → 重叠 → spec → verify/MoE 融合 → 小批量投影 → indexer 后处理 → MoE kernel**；
④ 每一步都用**同轮背靠背 A/B** + **固定 accept 口径** 测量（`~/bench_protocol.sh` 已按博客口径落地）。

## §99 【用户裁决·判定标准更正】sglang 的 acc 5.5 是**模拟的** ⇒ 我们的真实口径目标 = **acc 2.2 下 ~450 tok/s**

用户原话（2026-09-14）：
> **"注意 sglang 的 acc LEN 5.5 是模拟的，不代表真的能有这么高。我们 2.2 如果能有 450 就算可以击败他了"**

**这条更正了 §98 的对标框架**（连同更正我此前的推论）：
- 博客的 spec 数字（558.2 / 873.6）是在 **"simulated accept length of 5.5"** 与
  **"The server configuration controls the accept length"** 下测的 ⇒ 那是**把 accept 人为钉在 5.5 去测速度**，
  **不是**他们的真实接受率 ⇒ **不能当作"我们要达到的 accept 水平"** ✗。
- ⇒ §98 里"**我们的 400 目标仅为其全优化态（218.4/GPU）的 23%**"这句话，是用**口径不同的两个数**得出的
  ✗ —— 我们的 2.24 是**实测真实 accept**，他们的 5.5 是**模拟设定**。二者**不可直接比**，
  我此前那样比是**不严谨**的（记此更正）。

**正确的判定标准（用户给出）**：
| 口径 | 我们 | SGLang |
|---|---|---|
| accept 长度 | **实测 ≈ 2.24** | **模拟 5.5**（非真实） |
| 判定 | **在 acc ≈ 2.2 下达到 ~450 tok/s ⇒ 算击败** | — |

⇒ **我们要打的不是"acc 5.5"，而是"在真实 acc 2.2 的条件下把 step 压到 ~5ms 级"**
（`tok/s ≈ acc / step` ⇒ 450 tok/s @ acc 2.2 ⇒ **step ≈ 4.9 ms**）。
⇒ 这把 400/450 的路线重新聚焦为**纯粹的 step 削减问题**（与 `docs/agent/mtp-verify-amortization-model.md` 一致），
而**accept 的提升（2.2 → 3+）是另一条可选杠杆**，不是击败 sglang 的必要条件 ✓。

**对已开的 `accept-2p2-vs-5p5` 线的更正**：已发消息要求它**不要**把 5.5 当缺陷目标，
改为回答三件事：① `k_acc` 与"每步接受 token 数"是否同一口径；② 在不改 draft 质量的前提下，
spec 机制本身有无被浪费的接受机会（block 偏小、过早 commit、mask 误挡）；③ **收益换算**：
acc 2.2→3.0/3.5 各对应多少 tok/s（**并明确"提 acc"与"压 step"哪个更划算**）。

## §100 【GEMM dispatch 审计结论】快慢分裂在哪里（对应博客第一条教训）

来源：subagent `gemm-dispatch-audit`（纯 CPU 只读，逐调用点追 env 门 + 形状启发式 + `supports_*()`，
并以 `~/push400_hw_test.sh` 的**逐字 env 列表**为"出货配置"基准）。

### 结论 1：稠密投影族**已经还清**博客第一条教训 ✓
出货配置把 **wq_a/wkv/wq_b/wo_a/wo_b（verify+eager）、routed gate/up、共享专家、head** 全部推到了
TileLang/tcgen05 快内核（`DSV41_GEMM_TILELANG=1` + `_EAGER=1`、`DSV41_MOE_TILELANG_BS=1`、
`DSV41_SH_EXP_TILELANG=1`、`DSV41_HEAD_TILELANG=1`）✓ ⇒ SGLang 那条"FP8 布局不匹配→慢回退"
在本项目稠密族上**不存在** ✓。

### 结论 2：**最大结构性缺口 = MoE down 方向没有 blockscaled/tcgen05** ✗
- **gate/up**：走 tcgen05 block-scaled MMA（快）✓
- **down**：仍走 SIMT `expert_gemv_fp4_down_reduce_kernel` —— **v3 profile 1.00 ms/步、占 10.3%**，
  且是**占用率受限**的核 ⇒ "快慢分裂最刺眼的一条" ✓
⇒ **这是对标博客第 12 步（verify/MoE 融合）时最大的一块肉**，且与我们 BS 臂的工作**同源**
（同一套 fp4 blockscaled 语义，§47 的容器布局可直接复用）✓。

### 结论 3：**draft（MTP）侧整体没进 TileLang/tcgen05** ✗
`draft_moe` 的 routed gate/up/down **全是 SIMT fp4**；`draft_attention` 的四个投影走
`gemm_fp8_mx(m=bs)` 的 **16-row TILE MMA program**，既不是 verify 的 mrows program、也不是 TileLang
⇒ 而博客第 11 步（DSpark）恰是最大单步杠杆（×3）⇒ **draft 侧的 kernel 化是仅次于 down 的杠杆** ✓。

### 结论 4：若干"内核就绪、只差一个 env"
compressor 投影 mrows、`ATTN_PROJ_ALIGN` 等**代码已在**，出货脚本未开 ⇒ **先验证再开**（§87 判据）✓。

### 其它值得注意的中速项
- `wo_b`（eager）：`DSV41_WOB_F32` **默认 ON** ⇒ 走 SIMT f32 GEMV，**跳过 fp8 往返**（非 tensor core）⇒ 中等；
- `idx_wq_b`：TileLang wq_b 在该站 **decline**（`out_stride == n`，`chain_dev.rs:7305-7306`）⇒ 落 SIMT `gemm_fp8_mx_rope`；
- `comp_wkv/comp_wgate`（eager）：`lin_f32_on` → SIMT f32 GEMV。

## §101 有界等待补丁的形状审读 + **AR 侧代码的独立佐证**（与 §95 互为印证）

主 agent 审读 `bs-wait` worktree 的实现（落地前"写前先读"）：
- 新增 `whp_mbar_probe(bar, phase)`：把 `mbarrier.try_wait.parity` 的结果**返回给 C**（`selp` 取值），
  **不再用汇编级 `@!P bra WAIT` 回跳** ⇒ 自旋由 C 侧计数与上限控制 ✓；
- 超限调用 `whp_mma_timeout_report(...)` 打印诊断（block/k/phase/warpid 状态）✓，冗长转储由
  `DSV41_MOE_BS_WAITDBG=1` 门控（默认 OFF）✓；
- 辅助函数带 **`whp_` 唯一前缀** ✓（遵守 AGENTS.md 的合并纪律，避免同名重复定义）；
- 注释明确"**正常路径不变**"：稳态下首次 `try_wait` 即返回真 ⇒ 每 K 轮只多一次比较+分支 ✓。

### 🎯 独立佐证（价值等同一次实验）
该 subagent 读 AR 侧代码时发现：**`[ar5-hang] … spins>5000000 TIMEOUT -> PARK` 这行本身是"有界"的**
（自旋到 5e6 就 PARK ✓）⇒ 它的注释直接写下结论：**"那行是症状：某个 rank 的 kernel 从未返回、stream 从未推进"** ✓
⇒ **与主 agent 从 kernel 侧读出的 §92/§95 结论完全一致**（两条独立路径、同一结论）：
> 我们的 kernel 在无界自旋里卡死 ⇒ 其余 rank 在 all-reduce 里等到超时并 PARK ⇒ 刷 `ar5-hang`。

⇒ 这使 §95 的"挂死与非法指令是同一枚硬币"从**推理**升级为**双侧印证** ✓：
一侧是 kernel 的无界自旋（读 kernel 源码），另一侧是 AR 的有界 PARK（读 AR 源码）。
**有界等待落地后**，前者会变成一份可读诊断，从而把间歇故障变成可复现的报错 ✓。

## §102 作战路线（据 §97-§100 收束；判定标准见 §99）

**判定标准**：**真实 acc ≈ 2.2 下达到 ~450 tok/s = 击败 SGLang**（他们的 5.5 是模拟值 ⇒ 不追它）。
`tok/s ≈ acc / step` ⇒ **450 @ acc 2.2 ⇒ step ≈ 4.9 ms** ⇒ **这是一道纯 step 削减题**。

| 序 | 动作 | 为什么是这一条 | 状态 |
|---|---|---|---|
| **1** | **BS 臂正确性闭环**：有界等待（补丁已交付/合并中）→ 重编 → 重跑 → 拿到**有效文本**或**可读诊断** | step 削减的一切前提；且它把"间歇挂死"变成可复现故障（§92/§95/§101） | 合并中 |
| **2** | **拿 `[NC]` 数值 / wq_check 判据** ⇒ 正确则 `~/endgame.sh`（全 gate 回归 + 真实 p50） | 判据是 §88 的预案；`grep -c ar5-hang` 先查（§90） | 待 1 |
| **3** | **MoE down 的 blockscaled/tcgen05**（已开线） | §100 最大结构性缺口：**1.00 ms/步、10.3%**，且与 gate/up 共用已定谳的 fp4 语义 | 线在跑 |
| **4** | **draft(MTP) 侧 kernel 化**（全 SIMT fp4 → tcgen05；四投影用 mrows program 而非 16-row TILE） | 博客第 11 步是最大单步杠杆（×3）；draft 侧目前完全没进快内核体系 | 待 x3 起 |
| **5** | **精度五门逐项转正**（`~/promote_all.sh`；每门 DBG 对拍→红线→无回归→加进出货脚本） | 用户硬性要求"不能高也不能低"；且**不被 BS 臂阻塞**（§74） | 可并行 |
| **6** | **启用"内核就绪、只差 env"**（compressor 投影 mrows、`ATTN_PROJ_ALIGN` 等） | §100 结论 4：零新代码的低风险收益 | 可并行 |
| **7** | **口径对齐**：`~/bench_protocol.sh`（random 4k/1k、固定输出 1024、BS=1、greedy）+ 固定 accept 口径 | 否则无法与 SGLang 数字对话（§97/§99） | 脚本已就绪 |

**纪律**（贯穿）：同轮背靠背 A/B、**一次一个变量**、判据用 `wq_check`/`[NC]`/p50（**不用**图关掉的诊断臂数字）、
每次改动后**走项目自身构建**（§83）、合并用 `~/merge_worktree.sh`（§84/§89）。

## §103 "内核就绪、只差一个 env"的实情核实（§100 结论 4 的落地）

主 agent 逐项核对（代码定义点 + `~/push400_hw_test.sh` 实际是否开启）：

| env | 代码 | 出货脚本 | 结论 |
|---|---|---|---|
| `DSV41_COMPRESSOR_PROJ_MROWS` | `device.rs:7842` | 以 `DSV41_COMPRESSOR_MROWS=1` 开启 ✓ | 已生效 |
| `DSV41_ATTN_MROWS_ROPE_NORM` | `device.rs:6496` | `=1` ✓ | 已生效 |
| `DSV41_GEMM_TILELANG` / `DSV41_MOE_TILELANG_BS` | — | `=1` ✓ | 已生效 |
| **`DSV41_ATTN_PROJ_ALIGN`** | **`dspark_dev.rs:134` 有完整文档** | **计数 = 0 ⇒ 未开** ✗ | **唯一"就绪未开"项** ✓ |

⇒ **行动（已更正）**：⚠️ 主 agent 起初把它当**性能门**是**错的** ✗ —— 读其自带文档（`dspark_dev.rs:134-160`）后确认：
**"Why this is a NUMERICAL fix, not a perf one."** 原因：`dsv41_gemm_fp8_mx` 按 `m` 分派
（`m==1` 走 SIMT warp-per-row GEMV；**`m>1` 走 16-row TILE MMA**），而 **draft 恰好在 `m = bs = 5`**
⇒ draft 的四个 attention 投影落在 **TILE 程序**上，其 k-walk 与归约树**跨 tile 共享** ⇒
与 verify 的 `proj_mrows`（复现 `m==1` 的 consume 表达式、升序 kb 走法、per-row `shfl_xor` 树）
**求和结构不同（不是 ulp 级）** ✗。而官方 `model.py` 里 **draft 块与 verify 链对这些权重调用的是同一个 `F.linear`**
⇒ 两侧**必须同程序** ✓。
⇒ 因此它必须按**精度门**（§87 的第一类判据）转正：**开/关数值允许变化**，判据是
**红线不破 + 与 EAGER/对照的一致性**（而不是逐字节一致 ✗）。

### 🎯 由此得到一条**直指最大杠杆**的线索（已同步给 `accept-2p2-vs-5p5` 线）
该门修的正是 **spec 路径上 "draft 侧程序 ≠ verify 侧程序"** 的不一致；而 **accept 长度取决于 draft 与目标的一致性**
⇒ **这种程序不一致会压低 accept** ⇒ **`DSV41_ATTN_PROJ_ALIGN=1` 很可能直接把 accept 从 2.24 抬上去**，
且文档明确写着"**with it ON ... the accept comparison stops measuring the kernel mismatch**" ✓
⇒ 这是"零新代码、现成一门、直击最大杠杆"的候选，应**优先上机验证**（在 spec 臂上，看 accept 与 `[acc-hist]`）。
**注意**：它作用在 **draft（MTP）侧的 attention 投影**（`dspark_dev.rs`），因此
**只在 spec 臂上才有意义**（plain decode 不经过）⇒ 验证必须在 `DSV41_SPEC=1` 的臂上做，
且要与 `DSV41_DIFF_EAGER`/accept 长度一起看（对齐若改动了程序，accept 可能变）。

## §104 两个"就绪未开"的门都在 **spec 路径**上，但**判据不同**（勿混用）

| env | 代码文档 | 性质 | 判据（§87 的两类） |
|---|---|---|---|
| `DSV41_ATTN_PROJ_ALIGN` | `dspark_dev.rs:134-160` | **数值修复**：draft 四个 attention 投影原走 `gemm_fp8_mx(m=bs=5)` 的 **16-row TILE MMA**，与 verify 的 `proj_mrows` **求和结构不同**；官方两侧用**同一个 `F.linear`** ⇒ 必须同程序 | **精度门**：允许数值变化；判据 = 红线不破 + 与 EAGER/对照一致 + `[acc-hist]`/accept 观察（**不是**逐字节一致） |
| `DSV41_DRAFT_MOE_MROWS` | `dspark_dev.rs:112-130` | **发射形状 A/B**：`rows` 是 launcher 的**第三维 `blockIdx.z`** ⇒ per-row 形式每 stage 每 MTP 块要付 **`bs` 次 kernel launch**；mrows 形式一次调用内部派生每行指针。其文档明写"**两条臂是同一批 kernel、同一批实参** ⇒ A/B 只是发射形状的比较" | **性能门**：**必须逐位一致**（其文档自述 ROW INDEPENDENCE 论证：`rows = m` 的第 r 行 == 该行的 `rows = 1` 发射，行间不共享输出/累加器/smem staging）+ 同轮背靠背 p50 |

⇒ **行动**：两者都在**出货脚本里为 0**（`DSV41_SPEC=1`/`DSV41_DSPARK=1` 已开 ✓，与此二者无关）⇒
   都属于"零新代码、只差一个开关"的收益，且**都作用在 spec 路径**（博客最大单步杠杆所在）✓
   ⇒ 上机验证时**先各测一条**（一次一个变量）：`ATTN_PROJ_ALIGN` 看 accept/红线；`DRAFT_MOE_MROWS` 看 p50 与逐位一致。

## §105 spec 侧两门的 A/B 实验已脚本化（`~/spec_gates_test.sh`）+ **一条关键隔离设计**

`~/spec_gates_test.sh` 三臂、**一次一个变量**（同会话背靠背）：
`SG0_base`（`DSV41_SPEC=1 DSPARK=1` + `DSV41_ACC_HISTOGRAM=1`）→ `SG1_align`（+`DSV41_ATTN_PROJ_ALIGN=1`）
→ `SG2_mrows`（+`DSV41_DRAFT_MOE_MROWS=1`）；每臂报 accept/`[acc-hist-summary]`、p50、`wq_check` 红线。

### ⚠️ 关键隔离设计（没有它，判据会被污染）
**每一臂都显式关闭 fp4 MoE BS 臂**（`DSV41_MOE_TILELANG_BS=0 DSV41_MOE_DOWN_BS=0`）
⇒ MoE 走**已知正确的旧 per-slot GEMV 路径**。
**理由**：本轮同时在进行"BS 臂是否正确"的另一条战线。若 spec 门实验带着 BS 臂跑，
一旦文本异常，就**无法区分**"这个 spec 门有害"与"BS 臂还没对"——两个变量混在一起，
结论不可用（本战役已多次踩到"半挂/混挂"的坑）。
（`arm_run` 是 `env $BASE $COMMON "$@"` ⇒ **后面的 `$@` 覆盖 COMMON** ⇒ 该覆盖可行 ✓。）

**判读**：
- `SG1` vs `SG0`：accept 若上升且红线不破 ⇒ **draft/verify 程序不一致确实是 accept 被压在 ~2.2 的成因之一**
  ⇒ 按 §87 的**精度门**规则转正；
- `SG2` vs `SG0`：必须**逐位一致**（同 kernel 同实参，只是发射形状不同）⇒ 若一致且 p50 改善 ⇒ 按**性能门**规则转正。

## §106 【口径更正 + 目标重算】accept 计量差 1；450 tok/s ⇒ step ≈ 7.2 ms（不是 4.9 ms）

来源：subagent `accept-2p2-vs-5p5`（纯 CPU 只读，已在收到用户两次澄清后重框）。**它纠正了主 agent §99 的算术。**

### 🔴 更正一：**accept 计量口径差 1**（我们被先天低估 1/3）
- 我方 `mean-k = 2.240`（`serve.rs:663-691`）计的是**被接受的 draft token 数，不含 bonus**；
- SGLang 的 "accept length" **含 bonus**：硬证据是 **block size 5 而 accept = 5.5 > 5** ——
  若不含 bonus，上限只有 5，5.5 在算术上**不可能**；再叠论文脚注"accepted length … include the
  target-generated bonus token" ✓。
- ⇒ **同口径下我们的数是 `tok/step = mean-k + 1 = 3.24`**，**不是 2.24** ✓。
  **拿 2.24 去比 5.5 是"少算一个 token 再比"，先天低估我们约 1/3** ✗（我 §98/§99 就是这么比的，记此更正）。

### ✅ 更正二：目标 step 重算
`tok/s = tok/step ÷ step = (mean-k + 1) / step` ⇒ **450 tok/s @ tok/step = 3.24 ⇒ step ≈ 7.2 ms**
（**不是** §99 写的 4.9 ms —— 我那里误用了 acc 2.2 而非 tok/step 3.24）⇒ **目标比原先设想的宽松** ✓。

### 🎯 结论：**450 是 step 问题，不是 accept 问题**（与用户判断一致）
现况 step ≈ 32.5 ms（draft 3.87 + verify 28.17 + commit 0.47）下：
| 动作 | 吞吐 |
|---|---|
| 现状（tok/step 3.24 @ 32.5 ms） | ~100 tok/s |
| 把 acc(mean-k) 提到 3.0 / 3.5（step 不动） | ~123 / ~138 tok/s（+23% / +38%） |
| **step 压到 10 ms（acc 不动）** | **~324 tok/s（+224%）** |
⇒ **压 step 的杠杆比提 acc 大 2.7 倍以上** ⇒ 主线必须继续压在 **verify/step 削减**上（与 §102 一致）。

### 结构与块数：**与官方逐项相同**（故不是差异来源 ✓）
`DSPARK_DRAFTS = 5`（编译期常量，**无 env**，是 accept 宽度的**权威**；`dspark_block_size` 须 =5，`>5` 只会多采样仍只验 5）、
`VERIFY_ROWS = 6`（含静态断言 `== DRAFTS + 1`）、config `dspark_block_size = 5`、**3 层 MTP**
（`dspark_target_layer_ids = [37,38,39]`）、**Markov head 参与 draft 生成**、draft 采样 = argmax（与主链贪心一致）、
接受判据 = 最长公共前缀、**bonus token 存在**（`emitted = [next] ++ verify_out[..k_acc]`）。

### ⚠️ 发现一个**文档陷阱**（数据卫生）
`docs/agent/r0-r1-accept-diagnosis-manual.md:80` 那条带 `p1=0.7919 / hist={…} / oracle rate=0.868` 的
`[acc-hist-summary]` 行，**该文件 line 83 明写"数字是格式示意，不是测量值"** ⇒ **不得作为实测引用** ✗。
**当前栈的 accept 直方图从未实测过**（`acc_hist.rs:16-18` 自述）⇒ 硬证据只有两条：
① `mean-k = 2.240`（多文档一致）；② **数字任务实测 `k_acc = 5 5 5 5 5`（打满块长上限）**
⇒ **机制上没有把 accept 压在 5 以下的结构缺陷** ✓。

### 从这条线得到的**可执行项**
1. **`DSV41_ACC_HISTOGRAM=1` 必须真跑一次**（当前栈从未测过）⇒ 拿到真实 `k_acc` 分布，
   才能判断 2.24 是"draft 质量的合理值"还是"有条件被浪费"。
2. `DSV41_ATTN_PROJ_ALIGN` 那条线索（§103）它已纳入分析（draft 走 TILE 程序、verify 走 mrows ⇒ 同一权重两侧不同程序）
   ⇒ 与 §105 的实验脚本配套。
3. 报告口径**统一改成 `tok/step`**（含 bonus）对外比较；`mean-k` 保留为内部量。

## §107 【验证优先】单窗口全验证脚本 `~/verify_all.sh` + 验证次序（回应用户关切）

用户明确要求**优先把验证跑起来**，并指出"验证没跑通一直迭代不是个事情"。为此把**全部待验证项**
收敛成**一条命令、一个窗口**（避免每个判据各开一个窗口、也避免"跑不通还继续迭代"）：

```
~/verify_all.sh
  0. 背靠背重编（带 pipefail ⇒ 构建失败不会伪装成成功，§83）
  1. BS 臂正确性判据：X1（swapAB）+ X2（默认）→ wq_check 机械红线
     + 延迟 [NC] 五点回读（若命中）+ ar5-hang 计数 + 有界等待的超时诊断（§88/§90）
  2. 五道精度门的 DBG 对拍 + 红线（promote_all.sh，§64/§87）
  3. 博客口径基准（random 4k/1k、固定输出 1024、纯文本提示）（§97）
```
**设计要点**：各段**互不中断**（某臂失败只记进 `FAILED` 汇总，不放弃其它判据）；
每臂串行（`arm_run` 自己起 serve 且会 pkill ⇒ 不能并行）。

### 验证次序（为什么是这个顺序）
1. **BS 臂文本判据**优先：它是 §102 路线的前提；且现在的臂已是**带回全部修复**的版本
   （有界等待 + AR 缓解 + 请求加固 + 看门狗 + 图门全关）⇒ 要么给**有效文本**，要么给**可读诊断**。
2. **五道精度门**（与 BS 臂**解耦**，§74）⇒ 可在同一窗口紧跟着跑，直接回应"精度不能高也不能低"。
3. **博客口径基准** ⇒ 拿到**可对外比较**的 `tok/step`（**含 bonus**，§106）与 tok/s。

### 至此已修掉的、曾让验证"跑不通"的**全部**拦路项
| # | 拦路项 | 修复 | 章节 |
|---|---|---|---|
| 1 | 我的仪器化回归让 BS 臂**静默整体失效** | 回退 + 11 处 `return` 结构核验 | §65 |
| 2 | swapAB epilogue 输出打包错位（512/640 位置错） | 修 + 纯算术核验 640/640 | §62 |
| 3 | `glue_e2m1_encode` 重复定义 ⇒ **完整构建失败**、臂跑陈旧二进制 | 删重复定义 + 真实标志复验 | §83 |
| 4 | 臂挂在**无界 mbar 自旋**（→ 其它 rank 刷 `ar5-hang`、整轮 0 step） | 有界等待 + 超时诊断 | §92/§101 |
| 5 | `[NC]` 探针被 `goto` 跳过 | 接到 `goto` 之前 | §65 |
| 6 | `[NC]`/DBG 被图录制期捕获挡住 | 延迟到重放期（天然延迟型） | §73/§79 |
| 7 | `OUT` 为空（`curl -s` 静默吞错 + AR v5 停滞） | HTTP 码 + 有界重试 + 失败原因入日志 + AR 缓解 | §90 |
| 8 | 挂死会一直占窗口 | `arm_run` 加**后台看门狗**（日志停滞 ~120s 即杀 serve，快速失败） | §. |
