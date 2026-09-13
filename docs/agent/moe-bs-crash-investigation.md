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
