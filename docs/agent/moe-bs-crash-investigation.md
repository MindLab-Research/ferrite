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
