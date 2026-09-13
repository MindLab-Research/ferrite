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
| **e2e 仍错的排查（当前主线）** | §56（F 回合：kernel 已精确但 e2e 仍错 ⇒ 缺陷在接线/取数；`[NC]` 被某个提前 return 挡住）+ 顶部「下一步」 |

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
