# 吞主链步 —— 精确改动清单（实施草稿，2026-09-12）

**目标**：spec 模式每步省掉独立的主链 `step_dev`（6.15ms），改为 verify 的 6 行 `[anchor, d1..d5]` 承担 anchor 的 forward（它同时产出 tap）。净收益 ≈ −4.5ms/步（verify 多一行 ≈ +1.6ms）。
**前置**：`proj-mrows-redo` 的多行投影落地（同一文件），且 accept 链修复的验证通过（draft@pos + 旧链）。

## 1. `chain_dev.rs` — 新字段 `spec_primed`

```rust
// ChainBufs（或 DevChain，与其它 per-request 标志同位）
/// Spec 模式的引导标志：首轮走旧路径（step_dev 提供 tap+next），之后吞掉主链步。
/// `reset()` 清零（新请求重新引导）。
spec_primed: bool,
```
- `reset()`：`self.spec_primed = false;`
- **不要**放进 `KvSnapshot`（与 dspark 显式互斥，见 serve.rs:491 的 gate）。

## 2. `dspark_spec_step` 的两分支

```rust
pub fn dspark_spec_step(&mut self, dspark: &mut DsparkDev, token: u32, pos: usize)
    -> Result<DsparkSpecReport>
{
    // ... 现有的 armed 检查 / pos_ctr 校验 ...

    if !self.spec_primed {
        self.spec_primed = true;
        // ---- 首轮：现状不变（step_dev → tap → draft_forward(token, pos) → 5 行 verify
        //      → accept(drafts[0]==next, drafts[j]==verify_out[j-1]) → commit(keep=k_acc)）
        //      —— 但**新增 tap 的跨轮传递**（见 §4）
        ...
        return Ok(rep);
    }

    // ---- 后续轮：吞主链步 ----
    // 输入：token = 上轮 verify 的 bonus（已在上一轮的 verify 里被 forward，KV 已 append）
    //       pos   = token 的位置（== pos_ctr）
    //       tap   = 上一轮 verify 的 anchor 行（行 0）的层输出（§4 已拷进 dspark_tap）

    // 1. draft（与现状同一个调用，位置实参 = pos）
    dspark.import_tap(self.s.dspark_tap.ptr as *const f32)?;
    dspark.draft_forward(token, pos)?;
    let drafts = dspark.drafts()?;

    // 2. snapshot（m = DSPARK_DRAFTS + 1 = 6）
    let m6 = DSPARK_DRAFTS + 1;
    let host_mirrors = self.dspark_snapshot(pos_ctr, m6)?;

    // 3. verify 6 行 [anchor, d1..d5]
    let mut rows_in: Vec<u32> = Vec::with_capacity(m6);
    rows_in.push(token);                 // 行 0 = anchor（它的 argmax = pos+1 的 token = 旧路径的 next）
    rows_in.extend_from_slice(&drafts);  // 行 1..5
    self.spec_capture = true;
    let rows = self.step_rows(&rows_in)?;   // 行 j @ pos+j（pos_base = pos_ctr = pos）
    self.spec_capture = false;

    // 4. accept（6 行块的链 = 索引对齐，与 GLM 的 MTP 同构——见 tp.rs:1993 的
    //    `while drafts[k-1] == out[k-1]`）：verify 行 i 的输入是 [t0, d1..d5][i]、
    //    位置 pos+i，所以行 i 的 argmax = pos+1+i 的预测；而 drafts[i] 是"对 pos+1+i
    //    的提案"（draft 块 @ pos，行 i @ pos+i）——**两者同索引对齐**：
    //      drafts[0]（pos+1 的提案） vs verify_out[0]（行 0 = t0 @ pos 的 argmax = pos+1）
    //      drafts[1]（pos+2 的提案） vs verify_out[1]（行 1 = d1 @ pos+1 的 argmax = pos+2）
    //      ...
    //    （对照：5 行块不含 anchor，行 j 的 argmax 是 pos+2+j，所以那条链是
    //      drafts[0]==next + drafts[j]==verify_out[j-1]——**不同布局、各自自洽**。）
    let verify_out = rows;                   // len = 6
    let mut k_acc = 0usize;                  // = 接受的 draft 数（0..=5）
    while k_acc < DSPARK_DRAFTS && drafts[k_acc] == verify_out[k_acc] {
        k_acc += 1;
    }
    // emitted = [verify_out[0], .., verify_out[k_acc]]
    //   = [pos+1 的 token（= 旧路径的 next，确定值）, pos+2 .. pos+1+k_acc]
    //   —— 共 k_acc+1 个（k_acc=0 时仅 verify_out[0]，与旧路径的 emitted=[next] 相同）
    let mut emitted = Vec::with_capacity(k_acc + 1);
    emitted.extend_from_slice(&verify_out[..=k_acc]);

    // 5. commit：保留的行 = 0..=k_acc（即 pos..pos+k_acc），pos_ctr = pos + k_acc + 1
    self.dspark_commit(pos_ctr, m6, k_acc, &host_mirrors)?;   // keep 的语义核对见 §3

    // 6. tap 的跨轮传递（§4）
    self.copy_verify_row0_tap()?;

    Ok(DsparkSpecReport { next: verify_out[0], drafts, verify_out: [...], k_acc, emitted, .. })
}
```

## 3. `dspark_commit` / `dspark_rollback_keep` 的 `keep` 语义核对（**最需要小心的一处**）

现状（5 行块）：`keep = k_acc` = 保留 ring 的行 `0..k_acc`（位置 `pos+1..pos+k_acc`），`pos_ctr = pos + k_acc + 1`。
新（6 行块 [anchor,d1..d5] @ `pos..pos+5`）：
- 保留的行 = `0..=k_acc`（**k_acc+1 行**：anchor + 前 k_acc 个 draft）——位置 `pos..pos+k_acc`
- `pos_ctr = pos + k_acc + 1`
- **⇒ 同一个函数可以用**，只要调用侧的 `keep` 传 `k_acc`（**语义 = 保留 0..keep** 行）——**核对 `dspark_commit` 的实现**：它按 `keep` 保留 `0..keep` 还是 `0..=keep`？**5 行块的现状是"保留 0..k_acc 行 + pos_ctr=pos+k_acc+1"，而 6 行块要"保留 0..=k_acc 行"** ⇒ `keep` 传 `k_acc` 时二者相同（都是保留 k_acc 行？）——**必须逐行读 `dspark_commit`/`rollback_keep`/`compress_replay` 的行基址与循环边界**（`pos_base` 是 pos 还是 pos+1 的差异就在这里）。
- **tap 的位置**：`note_ctx_rows(..., m, k_acc, pos + 1)` 的 `pos+1` 基址也要相应改成 `pos`（6 行块的行 0 在 pos）。

## 4. tap 的跨轮传递（3 次 D2D）

`dspark_tap_r` 的布局 = `[slot][row][dim]`（slot stride = `VERIFY_ROWS * dim`）；行 0（anchor）的 3 个 slot：
```rust
for slot in 0..DSPARK_TAP_SLOTS {
    self.dev.memcpy_d2d(
        (self.s.dspark_tap.ptr as *mut f32).wrapping_add(slot * dim) as *mut c_void,
        (self.s.dspark_tap_r.ptr as *const f32).wrapping_add(slot * VERIFY_ROWS * dim) as *const c_void,
        dim * 4,
    )?;
}
```
- **首轮**也要做（把 step_dev 的 tap 传给第二轮？——**不**：首轮的 tap 由 `step_dev` 写 `dspark_tap`（现有路径 ✓），**第二轮**的 tap 才需要从 verify 的行 0 拷）——**所以 §4 只在"后续轮"的结尾做**，且**首轮的结尾不需要**（下轮是后续轮，读的是 verify 的拷入？——**不**：后续轮的 draft 读 `dspark_tap`——它的来源 = **上一轮**的 verify 行 0（首轮的 verify 也跑！）——**所以首轮的结尾也要拷**（首轮的 verify 的行 0 是 d1 的行——**hmm**：首轮 verify 的 5 行 [d1..d5] 里**没有 anchor 行**——**所以首轮的 tap 只能来自 step_dev ✓**（已写）+ **首轮 verify 的行 0（d1）的 tap 是"d1 的层输出"——下一轮（后续轮）的 anchor = 首轮 verify 的 bonus（= verify_out[k_acc-1]，**不是 d1**）——**所以后续轮的 tap 需要对"bonus 行"的 tap**——**首轮的 verify 行 j 里 j = k_acc-1 是 bonus 行**——**tap 的拷贝应该从"bonus 行"取**！

**⚠️ 这是一个需要精确处理的设计点**：新路径的 anchor = 上轮的 bonus——**上轮的 bonus 行在 verify 的块里的下标是 k_acc（或 k_acc-1，取决于 emitted 的组成）**——**tap 必须从那一行拷**（不是恒定的行 0）。**实施时先把这个下标关系在纸上推一遍**（emitted 的最后一个是 bonus；它在 verify 块里的行号 = ？），再写拷贝。

## 5. 验收
- **同会话 A/B**：`DSV41_SWALLOW_STEP=0/1`（给这个改造一个默认 OFF 的 gate，先并跑对照）
- **正确性**：出师表逐字 + 无相邻重复（`has_double_char`）+ `dspark_parity` 的 verify 行级对照
- **性能**：`[dspark] steps` 的 `verify=` 应 +1.6ms（6 行 vs 5 行）、**步时 −4.5ms**（主链步消失）——用 `DSV41_TIMING` 的 `[tick] total` 交叉验证
