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
4. `FERRITE_ALIGN_STRICT=1` 进 CI，让新破环在 CI 炸而不是 191ms 处。
