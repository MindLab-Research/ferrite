# tcgen05 "rank 7 misaligned" 判词 — rank 7 叙事作废，先修观测再谈修复

> 来源：ministry-justice / tcgen05-rank7-rootcause（2026-09-12）。
> 审查对象：为什么 tcgen05 misaligned "总是 rank 7" 且 4 轮修复全失败。
> **本文档作废** `session-final-handover.md:227` 的「rank 7 分片边界天然不 16B 对齐」结论及第 3/4 轮修复的方向依据。

## 0. 一句话判词

4 轮修复全失败的**结构性原因**：它们全部作用在**不含 `rank` 的量**上——均匀分片 `base_r = r·per` 在数学上根本产生不出 `{7}` 这个不对齐集合；"总是 rank 7"最可能来自 `serve.rs:250` 丢弃非首个 Err 造成的**上报竞态**。**先修观测，再谈修复**——当前证据链不足以支撑任何布局改动。

## 1. [严重] serve.rs 只保留第一个 rank 的 Err（上报竞态）

`crates/ferrite-dsv41/src/serve.rs:243-264`：`if fault.is_none() { fault = Some(...) }` —— 后续 rank 的错误全丢。misaligned poison 该 rank 的 CUDA context，只在下一次 `Device::sync()`（`devrt.rs:1073`）报出；8 rank lockstep 同步 ⇒ **必然几乎同时**返回 Err ⇒ "只有 rank 7 出错"从未被验证，实现上不可证伪。历史打印过 rank 5、6、7 三个不同值（`dspark-correctness-chain.md:1852`、`batch-reverification-plan.md:22/187`）= 竞态产物特征。

## 2. [严重] "rank 7 基址不 16B 对齐" 数学上不可能

分片规则 = 均匀截断 `per = D/world; base_r = r·per`（`load.rs:416-431,546-561`、`weights.rs:425-444,539-590`）。若 rank 1..6 base 对齐（per ≡ 0 mod 16）则 rank 7 必对齐；不对齐集合永远是残数类：

| per mod 16 | 不对齐 rank 集合 |
|---|---|
| 8 | {1,3,5,7} |
| 4 | {1,2,3,5,6,7} |
| 2 | {1,2,3,4,5,6,7} |
| 0 | ∅ |

**永远不可能是 {7}。** 且 tcgen05 A 侧操作数地址 = `pool + e·block + poff[k] + row·pitch + j·16`，没有任何一项含 rank（`block/poff` 是 loader 一次性布局，8 rank 相同；Rust 侧传入的 `w1_base/w1_stride` 与 rank 无关，`chain_dev.rs:12963-12970,13220-13227,16322-16327`）。⇒ A 侧整族（含 cp.async.bulk 源）**构造上 rank 对称**；第 1-4 轮改动对"只有 rank 7"不可能有影响。

## 3. [一般] check_bulk_geometry 漏查 w2 SF 行 pitch — "8 rank 全错"第一嫌疑

`weights.rs:470-501` 只查 dim%32、(dim/32)%16、inter_local%16、padded_inter(inter_local)%16。漏掉：**w2 的 SF 平面每 row 只有 `padded_inter(inter_local)/32 = 320/32 = 10` 字节**（9 真 + 1 补，`local_shape:567-586`），`10 % 16 ≠ 0`。rank 对称 ⇒ 不解释"只有 rank 7"，但是门没覆盖的唯一确定非 16B 行 pitch ⇒ **"8 rank 都错"情形的第一嫌疑**（down 路径 SF 用 LDG.128/uint4 读时 err 716 或静默错位）。

## 4. 真正 rank-specific 的对象只有两个（都不在 tcgen05 A 侧）

- **engram.embed**（`load.rs:418-424`、`weights.rs:545-551`）：唯一非均匀切分 `div_ceil`（"the last rank may be short"）。384006168/8=48000771 恰好整除 ⇒ 当前无害，但是唯一让 rank 7 拿不同长度切片的地方。
- **collective staging**（`tp.rs:377-384,545-549`）：rank 7 的槽在最高地址，`reduced[world-1]` guard 紧贴其后 ⇒ 任何越界写只命中它（SWALLOW OOB 修复已证此类事故真实发生过）。

## 5. production 形状 D%8 逐个核算 — 无任何维度有余数

dim=5120, inter=2304, nh=64, hd=512, ql=1280, ol=1024, groups=8, hpg=8, n_routed=384, vocab=129280, K_ATOM=64：所有被分片维度（含 engram rows、vocab）被 8 整除。⇒ "某维度 D%8≠0 让 rank 7 拿余数"前提不成立。真正要记的是**补齐**引入的不对称：`padded_inter(288)=320`（+11%）、`padded_inter(288)/32=10`。

## 6. 其他缺陷

- **[建议] alloc 16B 契约只在 debug_assert**（`devrt.rs:1112-1117`）：release 无保护；cudaMalloc 若返回非 16B 指针 → 整族静默 err 716。应改总是检查 + Err。
- **[建议] launcher 门"四缺三"**：`tc5::mxf4` 漏 `act`；`e4x` grouped 漏 `bh_base/bhs_base/bh_stride/bhs_stride`；`dsv41_gemm_fp8_swapab` 一个没查。rank 对称但会让"臂静默 decline"被误读成"臂没生效"（`device.rs:5784` 的 `rc==0 → Ok(false)` 正是本轮被坑的机制）。

## 7. 三条解释路径 + 判定实验

| 路径 | 机制 | 可能性 | 判定实验 |
|---|---|---|---|
| **A** | rank 对称布局缺陷 + 上报竞态（8 rank 全错，"rank 7"=谁先到 channel） | **最高** | 改 serve.rs 全量打印 Err 后跑一次：8 行全报 err 716 ⇒ A 成立，rank 7 叙事整体作废 |
| **B** | 单 rank 成因但对象不是分片 base：engram div_ceil 短尾 / collective staging 高地址槽 / DevBuf view 偏移 | 中 | `DSV41_ALIGN_AUDIT=1` 装载期逐 rank 打印 base&15/stride%16/pitch%16/nsf%16 + `compute-sanitizer --destroy-on-device-error kernel` 枚举全部 fault（默认 context 模式第一个 fault 后毁 context，只会看到 1 个 misaligned——**这正是 4 轮被误导的机制**） |
| **C** | 只有 rank 7 真在跑 tcgen05 臂（rank 0 角色分叉） | 最低（与 run_ranks 同构执行冲突） | launch 计数（不是 misaligned==0——"0 misaligned"是空洞证据，design §6 V5） |

## 8. 修复方向

**不要做（已数学否证）**：rank 7 scalar/byte-fallback（bulk 16B 不可字节化）；继续在"rank 7 base 不对齐"上调权重 padding 口径。

**应该做**：
1. **第 0 步（观测修复，进行中 — ministry-works/tcgen05-observation-fix）**：serve.rs 全量 Err 收集；`DSV41_ALIGN_STRICT=1`（默认 warn）+ check_bulk_geometry 补 w2 SF 行 pitch；devrt alloc 契约响应亮化；launcher 门补齐；`DSV41_ALIGN_AUDIT=1` 装载期审计。
2. **若 8 rank 全错（路径 A）**：修布局不变量——`padded_inter(inter/world)/32 % 16 == 0`（即 inter/world 需被 512 整除）。**唯一根修，数值不变只改地址**。
3. **若 1-2 rank 错（路径 B）**：查 engram.embed div_ceil 短尾视图边界 + collective staging `reduced[world-1]` guard。
## 9. 第 5 轮判定实验可执行序列（观测修复 = 3cd7258 + A0 site 分流 = 95d7083 之后）

**远端前置**：`git fetch && git reset --hard origin/main && cd kernels/cuda && bash build.sh 103a && cd ~/ferrite && cargo build --release`（动了 .cu ⇒ 双产物必做）。

**tcgen05 臂 gate 前提链**（权威出处 `tcgen05-retest-after-guardfix.md §4.1`；两个 `starts_with('1')` 门不能写 `=true`/`=on`；`DSV41_NO_GEMV_FP4` 必须不存在）：
```
DSV41_SPEC=1 DSV41_DSPARK=1 DSV41_EXPERT_ACT_E4M3=1 DSV41_EXPERT_TCGEN05_E4M3=1
DSV41_EXPERT_GROUPED=1 DSV41_GATEUP_FUSE=0 DSV41_EXPERT_ILV=0 DSV41_MOE_BATCH=1
CUDA_VISIBLE_DEVICES=0..7, LD_LIBRARY_PATH=$HOME/ferrite/kernels/cuda, --tp 8 --port 8712
```

**步骤 0（装载审计，秒级）**：上 gate 链 + `DSV41_ALIGN_AUDIT=1` 启动，grep `^\[align\]`：
- 预期（若路径 A 成立）：每层 `w2.scale … pitch=10 pitch%16=10 row1&15=10` 且 **8 rank 完全一致**（rank 对称 ⇒ 解释"8 rank 全错"）
- `worst_base&15 != 0` ⇒ 专家位移破环；只有某 rank 不同 ⇒ 真 rank-local

**步骤 1（复现归因，单轮制）**：上 gate 链 + `CUDA_LAUNCH_BLOCKING=1 DSV41_VERIFY_GRAPH=0 DSV41_GRAPH_MOE=0`（不用 nsys；判据是日志行），/health 后发"你好" max_tokens=20，读 stderr：
- `[tp] step failed on 8/8 ranks` + 8 行 Err 文本逐字相同（misaligned/716）⇒ **路径 A**：w2 SF 行 pitch 布局根修（`padded_inter(inter/world)/32 % 16 == 0`，即 inter/world 被 512 整除；数值不变只改地址）
- `1/8 ranks` 单行 Err ⇒ **路径 B**：查 engram.embed div_ceil 短尾视图边界 + collective staging `reduced[world-1]` guard + DevBuf::view bare wrapping_add
- `N/8` + `cuda error 1 (invalid argument)` + `dsv41_expert_tcgen05_gate_up_*` ⇒ **路径 C**：门触发（对照 .cu 门覆盖矩阵，看哪个参数被拒）
- `[align] … base misaligned by N B` 行出现 ⇒ 门已抓到布局事故（这行就是证据）
- **关键**：改后同一故障稳定打 8 行——8 行里每行的 rank 号只是 ack 到达顺序，**判断 rank-specific 唯一标准是"是否只有一行"**

**步骤 2（A0 探针，顺带同会话）**：SWALLOW 全 gate 集（`swallow-full-gate-config.md §4` 逐字手写，禁裸跑 batched_400_v2.sh；V5_LEDGER=0）+ `DSV41_AR_PROBE=1 DSV41_AR_TIMEOUT_TRAP=1`，计数任务 max_tokens≥120（探针每 512 轮/site 才打印，84 轮/步 ⇒ ≥7 步才出第一行）。读 `[ar-probe] rank= site= n= avg_spin=`（单位 SM 周期，1µs≈1800cyc）：
- ATTN/MOE 各 40 轮/步稳态（新分流的 verify AR）；site=1 与 site=0 各 ≥1 行 × 8 rank、n≥512 才可判
- `avg_spin≈31k cyc(17.3µs)` ⇒ 账本成立，靶子 A2c rank 负载均衡（先看 per-rank 是否"一超多零"）
- `≫31k` ⇒ nsys 自旋放大为主，AR 方向预算重估；`≪31k(≈9k)` ⇒ AR 无肉，转 MoE(17.4%)/投影(15.1%)/hc_dots
- 辅助：`avg_stamp`~0.2-0.5µs（大 ⇒ store 路径贵=A1a 靶）；`avg_epi` lazy 4-6µs / SWALLOW 设计 8-15µs（>15 ⇒ T3 fold/reduce 向量化）
- 收尾一律 `POST /shutdown`

## 10. 第 5 轮判定实验结果（2026-09-12 22:15 执行，✅已定谳）

**配置**：观测修复（3cd7258）+ A0 site 分流（95d7083）双产物重编后；gate 链 = §9 前提链 + `DSV41_ALIGN_AUDIT=1 CUDA_LAUNCH_BLOCKING=1 VERIFY_GRAPH=0 GRAPH_MOE=0`；证据 `~/tc5_round5_EVIDENCE.log`（2443 行）。

**步骤 0（装载审计）**：
- `bulk-geometry WARN` ×8——**8 rank 逐字相同**（rank 对称确认）
- 2416 行 `[align]` 审计中**唯一 violation = w2.scale**：`L42 r0 e0 w2.scale base&15=0 worst_base&15=0 stride%16=0 pitch=10 pitch%16=10 row1&15=10`，其余 plane（w1/w1.scale/w3/w3.scale，pitch=2560/160）全部 %16=0
- 汇总行：`1 16B violation(s) in the expert pool — the tcgen05/bulk operands CANNOT be addressed: w2.scale: row pitch 10 B is not a multiple of 16 (row 1 starts 10 B off the grid) — every row past the first of this plane is unaddressable by the bulk/uint4 paths`

**步骤 1（"你好" 20 tok 触发）**：
```
[tp] rank 0 (device 0) err: config error: dsv41_expert_tcgen05_gate_up_e4m3: cuda error 716
[tp] rank 3 / 1 / 5 / 6 / 7 / 4 / 2 (device 同号) err: （逐字相同）
[tp] step failed on 8/8 ranks
```

**判决：路径 A 确凿。**
1. **8/8 ranks 全部 cuda error 716，文本逐字相同** ⇒ 破坏 rank 无关；"只有 rank 7"正式作废（`serve.rs` 只留首个 Err 的上报竞态，如 §1 所判）。
2. **唯一 violation = w2 SF 平面行 pitch 10 字节**（`padded_inter(288)/32=10`；row1&15=10 ⇒ 每行都偏离 16B 网格）⇒ §3 第一嫌疑证实。
3. **根因定谳**：tcgen05 misaligned = w2 scale 平面行 pitch 10B 的 rank 对称布局缺陷；4 轮修复全部打在"rank 7 分片边界"上 = 方向完全错误。

**根修口径**（未实施，属 L4-3/L4-4 收尾）：使 SF 行 pitch 满足 16B ⇒ `padded_inter(inter/world) % 512 == 0`（即 inter/world 需被 512 整除；当前 288 → padded 320 → 10B）。需要 scale 平面物理行 stride 与逻辑 `k/32` 解耦（行间 padding + 内核 scale 索引 stride 参数化），数值不变（pad 区不读）。TMA bulk 的 16B 硬对齐使 byte-fallback 不可行（§1.2 已证）。

**修复后复验口径**：`DSV41_ALIGN_AUDIT=1` 装载期 `pool geometry OK`（0 violation）→ `DSV41_ALIGN_STRICT=1` 装载通过 → tcgen05 臂 8/8 → 0/8 Err → 计数前 61 行 + 零拉丁红线。
