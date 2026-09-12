# draft accept 提升路径（accept 是 400 的真正乘数）

> 刑部 · 2026-09-12 · 只读分析
> 基线：mean-k ≈ 0.68~0.83 · verify 37.31ms · draft 4.9ms

## 0. TL;DR（六条）

1. **accept 的最大剩余杠杆不是精度，是「draft 与 verify 不在同一数值域」**——主链的 e4m3 双趟修复**没有覆盖 draft**：`draft_moe` 的 routed expert 激活仍是 e2m1（隔离测量差距 **8.4×**）。
2. **`DSV41_SEED_ALIGN` 一个 gate 压着三项 accept 修正**（相位 + 窗口行集/序 + 6 行索引对齐链），默认 OFF；`SWALLOW_STEP` 走同一条路径 ⇒ **SWALLOW 同时吃这三项修正 + −4.55ms**。
3. **tap 采集点存在三方不一致**：ferrite/sglang 采「层输出」，ref_inference 采「层输入」——整整差一层。parity 查不到（inject 覆盖了 live 值）。
4. **mean-k ≥ 3 结构性不可达**（均匀命中率需 p ≈ 0.83；tail 是模型固有熵）。400 应改写为 `mean tok/step 2.4~2.6 @ ≤8ms ⇒ 300~325 tok/s`。
5. 3 个 mtp block + markov 5 步**不是瓶颈**（与官方逐字同构）；markov 的 5 步不接收 hidden（只接 token embedding），瓶颈全部在 row 的 hidden 质量。
6. "0.83 → 3" 的 3.6× 缺口里至少 **1.6× 是口径误差**（直方图 mean-k = 0.677 vs 文档 0.833，差恰好 36/232）。真正的数值缺口是 **~1.3~1.5×**。

## 3. 剩余 accept 压制因素

### F1 ——【严重·确定性】draft 的 routed expert 激活仍是 e2m1，e4m3 修复未覆盖 draft
- draft：`dspark_dev.rs:1582` 单趟 e2m1，无任何 e4m3 分支
- 主链有两份 e4m3 双趟（`moe()` 和 `moe_rows()`），都判 `expert_act_e4m3()`
- 后果：draft 的 hidden 与 verify 的 hidden 系统性分离（e2m1 rel-L2 1.215e-1 vs e4m3×2 1.447e-2 = **8.4×**）
- 修复：把 `moe()` 的双趟原样镜像到 `draft_moe()`（Rust-only）+ ILV 兼容（见 F1 前置）

### F2 ——【严重·待仲裁】tap 采集点：层输出 vs 层输入
- ferrite（live）采层输出（`chain_dev.rs:9217`，层末 hc_collapse）
- sglang 也采层输出（`deepseek_v4.py:2362-2369`）
- **ref_inference 采层输入**（`model.py:1264-1266`，在 layer() 之前，注释"reads the attention input"）
- ⚠️ parity 查不出（inject 覆盖了 live 值）
- 需要 30 分钟判定实验：用层输入做一次 golden 对照

### F3 —— SEED_ALIGN 压着的三项修正（默认 OFF）
- 相位：draft_forward(next, pos+1) → seed 落 pos ✓ 与 tap 同位
- 窗口行集/序：回绕按位置序展开
- 6 行索引对齐链：spec_accept(.., true)
- SWALLOW 走同一路径 → 开 SWALLOW = 同时吃三项 + −4.55ms

### F4 —— verify 剩 1 mismatch（0.4%）不是 accept 杠杆

## 5. 建议的实验序列（按 accept × 把握 排序）
1. **draft 的 e4m3**（F1）——最大单项（8.4× 精度差消除）
2. **SWALLOW/SEED_ALIGN**（F3）——一次吃三项 + 性能 −4.55ms
3. **tap 判定实验**（F2）——30 分钟，可能改变整个 accept 链
4. **口径修正**（F4）——确认 mean-k 的真实值
