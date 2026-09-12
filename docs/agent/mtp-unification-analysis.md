# MTP 的两套实现：GLM（mtp_step）vs DSV41（dspark_spec_step）—— 逐项对比与归一 seam

**日期**: 2026-09-12（Wave 6 深度统一的输入）
**动机**: 用户目标第一条是"架构完全没重复逻辑"。**MTP 是当前最大的重复面**：两套 draft/verify/accept/commit 各自演化，各有各的正确性事故史。

## 一、共同骨架（两者都是"投机解码的四段"）

```
draft(n=1..8) → verify(块) → accept(最长前缀 k) → commit(前进 k)
```

## 二、逐项对比（读码）

| 环节 | GLM（`ferrite-exec/src/tp.rs` 的 `mtp_step`） | DSV41（`ferrite-models/src/dsv41/chain_dev.rs` 的 `dspark_spec_step`） |
|---|---|---|
| **draft 步数** | `nd = N-1`（`FERRITE_MTP_N`，1..=8）——**N-UNIFIED**（N=1 即普通 decode） | 固定 5（`DSPARK_DRAFTS`，checkpoint 的 block_size=5） |
| **draft 的 hidden 来源** | **图内 ping-pong**（draft i 的 `hprev` = draft i-1 的 `h_out`；draft 0 = 已提交的 `hprev`）——`ferrite_mtp_commit` 里做 `hprev <- hf_v[k-1]` | **`import_tap`**（host 发起的 D2D：主链收集层 37/38/39 的 hc 均值 → `dspark_tap`） |
| **verify 块** | **`[t_last, d1..d_nd]`（含 anchor 行）**，n_v = 1+nd 行 ——`mega_v` 图的一次 replay | **`[d1..d5]`（不含 anchor）**，5 行 ——`step_rows` 一次 batched forward |
| **verify 的输入注入** | `graph_run_ids`（**device embed**：ids 直进图，首节点 `embed_expand_dev`）——省 host embed + 576KB staging | `ul_i32(ids_r)` + `ul_i32(pos_rows)`（**图外 H2D 刷新**，图内读） |
| **accept 链** | `k=1; while drafts[k-1] == out[k-1] { k+=1 }`——**索引对齐**（因为块含 anchor，行 i 的 argmax 与 drafts[i] 同位置） | `if drafts[0]==next { k=1; while drafts[k]==verify_out[k-1] { k+=1 } }`——**错位一位**（因为块不含 anchor，行 j 的 argmax 是 pos+2+j） |
| **commit** | **1 个 kernel**（`ferrite_mtp_commit`：B_k→A 的 ping-pong + `hprev` 更新） | **host 镜像 + 多 kernel**（`dspark_commit`：rollback_keep + compress_replay + 计数推进） |
| **图化** | **✓ 全图**（`mega_v{seq}` 为 verify、`mega_d{seq}_{i}` 为 draft） | **部分**：`DSV41_VERIFY_GRAPH=1`（已实现、默认 OFF）+ 裸链的 6232 launch/步 |
| **状态回退** | `dsa_host_rollback`（pinned 记账）+ 图内 ping-pong | `dspark_snapshot`/`dspark_rollback_keep`（ring 槽快照 + compressor 状态 + clen） |

## 三、归一 seam（提案，与 Wave 1 的 `StepEngine` 呼应）

**吸收两边的优点**（共同的四段 + 一个 trait）：

```rust
/// 投机解码的一个步（GLM 的 mtp_step 与 DSV41 的 dspark_spec_step 的共同抽象）。
pub trait SpecStep {
    /// 1. 起草 nd 个 token（含各自的 hidden 链）
    fn draft(&mut self, anchor: u32, pos: usize, nd: usize)
        -> Result<Vec<u32>>;
    /// 2. 一次 verify 块：[anchor, d1..d_nd]（**含 anchor**——统一到 GLM 的布局，
    ///    DSV41 的"吞主链步"正是这个方向），返回每行的 argmax
    fn verify(&mut self, block: &[u32], pos: usize) -> Result<Vec<u32>>;
    /// 3. 最长前缀 accept（**索引对齐**的统一链——上面那张表说明"错位一位"只是
    ///    "块不含 anchor"的副产品，不是不同的数学）
    fn accept(drafts: &[u32], verify_out: &[u32]) -> usize {
        let mut k = 0;
        while k < drafts.len() && k < verify_out.len() && drafts[k] == verify_out[k] { k += 1; }
        k
    }
    /// 4. 前进 k（含状态提交/回退）
    fn commit(&mut self, pos: usize, k: usize) -> Result<()>;
}
```

**归一的收益（按价值）**:
1. **DSV41 得到 GLM 的 verify 图化**（6232 launch/步 → 图内 node dispatch）——**这就是 400 tok/s 的性能主项**
2. **DSV41 得到 GLM 的 device-embed 注入**（`ul_i32` 已是图外刷新 ✓，但 host 的 embed 可省）
3. **GLM 得到 DSV41 的"块不含 anchor"的省行**？——**不**：GLM 的含 anchor 是它的 commit 设计（`hprev` 从 verify 行取）——**统一到含 anchor 更有优势**（它让 accept 链索引对齐，且 anchor 行的 forward 顺便产出 tap——**DSV41 的"吞主链步"正是这条**）
4. **accept 链单一实现**（消除两套链——**当前两套各自正确但极易被改错**：本会话的 B1（块位置）+ 错位实验都是在这条链上踩的）
5. **commit 的 1-kernel 形态**（GLM 的 `ferrite_mtp_commit`）可以吸收 DSV41 的 compressor 语义（**需要 kernel 侧支持 DSV41 的状态集**）

## 四、实施顺序（建议）

1. **DSV41 的 verify 图化 + 多行化**（正在做——`proj-mrows-redo`/moe 已接）→ **先拿到 400 的性能**
2. **DSV41 的"吞主链步"**（6 行含 anchor——**同时**把 accept 链切成索引对齐的统一形式，见 `dspark-swallow-step-diff.md` 的修正版）
3. **抽 `SpecStep` trait**（两套实现各自 impl，**行为不变**的纯重构——用 `dspark_parity` + GLM 的 MTP 文本作回归）
4. **commit 的 1-kernel 化**（把 GLM 的 `ferrite_mtp_commit` 泛化——最后做，风险最高）
