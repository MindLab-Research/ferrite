# fp4 MoE BS illegal memory access 完整排查记录（2026-09-14）

> 状态：排查中（DIAG 测试在跑）。本文记录到当前为止的全部排查过程、已否定假设、和剩余嫌疑。
> JIT 隔离测试已证明 **kernel 设计本身无问题**（不 crash），问题在 AOT 编译/接线链路。

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
