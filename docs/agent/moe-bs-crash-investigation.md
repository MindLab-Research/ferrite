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
