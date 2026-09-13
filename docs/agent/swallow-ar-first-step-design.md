# SWALLOW 步时优化的第一步设计 —— AR 优先：把「36%」的账先算清，再动手

> 工部 · 2026-09-12 · **只读勘察 + 本文件（唯一产出）**。未执行 GPU 命令、未改动任何源码。
> 输入：`docs/agent/swallow-unlocked-shpair-m6-throughput-plan.md` · `swallow-full-gate-config.md` ·
> `ar-l4l5-optimization-design.md` · `ar-step1-a2b-a0-implementation.md` · `ar-step2-a1a-moe-store-implementation.md` ·
> `lazy-verify-optimization-path.md` · `swallow-nsys-batched-analysis-framework.md` · `dspark-correctness-chain.md`（尾部 nsys 表）。
> 读码：`kernels/cuda/ferrite_kernels.cu` · `crates/ferrite-models/src/dsv41/{chain_dev.rs,tp.rs,device.rs}` · `scripts/nsys_wave1.sh`。
> 代码基线 = 工作树 HEAD `ba88df1`（`chain_dev.rs` 有未提交改动，行号可能漂移 ⇒ 一律以**函数名 + gate 名**为准）。
> **口径纪律**：每条 μs/ms 标来源（**实测** / **账本** / **设计** / **代数** / **nsys**）。

---

## 0. 先读九条（前四条**修正任务前提**，必须上报尚书省）

1. **❗「AR 36% = 10~11.2ms/步」这个乘法是错的。正确的每步 AR 是 `84 轮 × 78.3μs = 6.58ms`（23.5% 的 28ms 步时）。**
   SWALLOW 每步的 v5 轮数是**结构常数 84**（`chain_dev.rs:1894`：
   「the round ledger puts `legacy == aligned == 165` and **`swallowed == 84`** rounds/step」；
   `ferrite_kernels.cu:9212` 同述，missing 81 = `step_dev` 的 40 层 ×2 + head 的 argmax 交换）。
   924 实例 ÷ 84 = **11 步**（与计数测试「61/72 位」一致，~12 步）。
   ⇒ 36% 是 **kernel-sum 窗口内的占比**，不是步时占比。自洽核对：`72.3/0.36 = 200.8ms` 可见总量，
   除以 11 步 = **18.3ms kernel/步**（28ms 墙钟 − 18.3 = ~10ms 是 gap/宿主/prefill）。
   **AR 的真实票面 = 6.58ms/步（36% of 18.3ms kernel）**，不是 11.2ms。【nsys 表 + 代数】

2. **❗那条 #1 的 kernel 身份**与**该 run 自己声明的配置**互相矛盾——不查清不能投 AR。
   表里写 `pubred_v5_hcpost`，run 配置写 `AR_V5=0`（`dspark-correctness-chain.md:5970`「+ AR_V5=0（nccl）」）。
   但 `ar_v5() = GRAPH_STEP ∥ AR_V5`（`tp.rs:1027-1047`）；两腿全 0 ⇒ `ar_v5()==false` ⇒
   `all_reduce_inplace` 走 **host-barrier 伪路径**（`ar_store` / `ar_stamp` / `ar_reduce` 三核 + `barrier.wait()`），
   **根本没有 `p2p_ar_pubred_v5_*` 上场**。⇒ 二者只能有一个是真的：
   要么 run 其实 `AR_V5=1`（表名对、配置记录错），要么表名是人工归纳（不同 kernel 被并成一行）。
   **这是本设计的第一条硬前提：AR 的「身份」必须先钉死。**【读码 + 文档矛盾】

3. **❗SWALLOW 稳态**不可能**跑单行 `hcpost`。** SWALLOW 从第 2 轮起不走 `step_dev`/`layer()`（单行），
   只走 `layer_rows()`/`moe_rows()`（`chain_dev.rs:10261`/`:12888`）。而 `_hcpost` 单行折叠的调用点在
   `layer()`（`:15371`）与 `moe()`（`:14357`）。稳态的两条 verify AR 是：
   * `layer_rows` → `ar_hc_post_fold_rows()`（**:11679**，需 `VERIFY_AR_FOLD ∧ FUSE_C ∧ HC_VERIFY_FUSE` 三开门）
     **decline 则** `all_reduce_inplace()`（**:11686**，`fb(m*dim)`）；
   * `moe_rows` → 同一 fold（**:13514**）decline 则 `all_reduce_inplace()`（**:13523**，`fb(mdim)`）。
   ⇒ 若 nsys 的 gate 串按权威脚本（`HC_VERIFY_FUSE` 在 FORBIDDEN 里）⇒ **两条都 decline** ⇒
   84 轮**全部**落到 `ferrite_p2p_ar_v5`（plain，`ferrite_kernels.cu:9343`）⇒ 真正的 #1 名应是
   **`p2p_ar_pubred_v5_kernel`**，不是 `_hcpost`。**表里的名字要么错、要么那一轮真的开了三开门。**【读码】

4. **❗「78.3μs 里传输只有 0.6μs ⇒ 77.7μs 全是等待」的算法漏了一半。**
   `0.6μs = 122,880B ÷ 200GB/s` 只算了**一份 payload**（m·dim = 6×5120）。但每轮的远程写是
   **8 peer × 120KB = 960KB**（`p2p_ar_store_v5_kernel`，`gridDim.y = world`），加上 reduce 读
   8×120KB、fold 的 4 读 4 写/列 —— 每轮约 **2~3MB 的 L2/HBM 流量**（设计口径 5~15μs，不是 0.6μs）。
   而且 **store 是独立的 kernel 行**（`p2p_ar_store_v5_kernel`），它**不在任务给的前 6 行里**
   ⇒ 「等待占 77.7μs」这个结论**没有证据**，必须先把 `store / stamp / wait / reduce+fold` 四段分开。
   【读码 + 算术】

5. **SWALLOW 每轮的 poller 数是 lazy 的 6 倍：960 vs 160（新发现，零代码可 A/B）。**
   poller 数 = `grid blocks × world`，`blocks = ceil(n4/64)`（`ferrite_kernels.cu:9255-9263`，threads=64）。
   * lazy decode：n = 5,120 ⇒ n4 = 1,280 ⇒ **20 blocks ⇒ 160 pollers/轮**；
   * **SWALLOW verify：n = m·dim = 6×5,120 = 30,720 ⇒ n4 = 7,680 ⇒ 120 blocks ⇒ 960 pollers/轮**。
   960 个线程每 ~100ns 轮询**同样 8 个** stamp 字（`ar5_wait_round` 的 OFF 臂，`:9113-9125`）。
   A4 单块臂（`DSV41_AR_SINGLE_POLL=1`）把它压到 **8**（`:9069-9107`：block 0 轮询 + 设备内广播字 `epoch[1]`）。
   A4 在 lazy 上端到端**中性**（82.9→82.6 tok/s）；**中性不构成反证**——lazy 是 160→8，SWALLOW 是 960→8，
   且 SWALLOW 每轮数据量 6×，poll 占比先验高得多。**这是第一步里最便宜、先验最高的 A/B。**
   安全前提已读码确认：A4 的广播字 `epoch+1` 与 `V5_LEDGER` 的 canary（`ctr_at+8/16/32/48`）**不重叠**
   （`tp.rs:315-334` 明写「epoch at `ctr_at`, A4's flag at `ctr_at + 4`, canaries start at `+8`」）。【读码】

6. **A0 探针在 SWALLOW 上「site 分流」失效——这是第一步必须补的一个小件。**
   探针按**入口符号**分 site（`ferrite_kernels.cu:8979-8983`）：MOE=0 / ATTN=1 / VERIFY=2 / OTHER=3。
   但 SWALLOW 稳态的两条 verify AR 都走 `all_reduce_inplace` ⇒ `ferrite_p2p_ar_v5` ⇒ **site=3(OTHER)**
   （注释自己承认：「KNOWN LIMIT: …plus every site the split cannot tell apart」`:8974-8978`）。
   ⇒ 现状下探针只会打印**一个 lumped 的 OTHER 桶**，**答不了「哪一半在等」**（设计 §4-A0 的核心问题）。
   **修法（0.5 人日，两个新 `extern "C"` 符号，只改 site 标签，不改协议）**：见 §1.5。
   另外 `_hcpost_add` 缺失时 `_hcpost` 会**同时**服务 ATTN 与 MoE（site 会翻倍）——三证里必须查该符号。【读码】

7. **nsys 对 v5 的自旋有已知放大，绝对 μs 不可直接编预算。**
   `scripts/nsys_wave1.sh:33-40` 自陈：「its publish kernel SPINS on peer stamps, and under nsys's per-node tracing
   that spin is amplified **~300x (measured: 240s of wall clock for 69 steps)**」，并因此把 nsys 一律 pin 成
   `AR_V5=0 + GRAPH_STEP=0`（host-barrier 伪路径）。⇒ 对 AR：**nsys 只能用来「数实例 / 数符号 / 定身份」，
   不能用来读 μs/占比**。唯一的绝对量工具是 **device 侧 A0 探针**（`clock64()`，无 host 读，capture-safe），
   且**探针轮禁止与 nsys 同跑**（`ferrite_kernels.cu:8936-8959`）。【脚本自陈 + 读码】

8. **两个口径差 5.1ms/步 = 第一步要钉死的全部不确定性。**
   | 口径 | 每轮 | 每步（×84） | 占 28ms | 来源 |
   |---|---:|---:|---:|---|
   | 账本（v5 协议地板） | **17.3μs** | **1.45ms** | 5.2% | `lazy-verify-optimization-path.md:98`（AR 1.40ms/步 ÷ 80 轮） |
   | nsys（本表） | **78.3μs** | **6.58ms** | 23.5% | `dspark-correctness-chain.md:5974` |
   差 **5.1ms/步**——**比 S1 计划要省的「AR −3~5ms」还大**。在这个差被 device 侧数据切开之前，
   任何「AR 优化能省 X ms」都是猜。**这就是第一步的全部理由。**【账本 vs nsys】

9. **第一步的产出必须是「判定 + 一个已上场的零代码 A/B」，不是一次性投 2-3 人日。**
   S1 计划（A2b+A0+A1a）**代码已落地**（`ar-step1`/`ar-step2` 两文档，`cargo check` 过），但**一次 GPU 验证都没有**：
   * A2b（超时=响亮失败）：`ferrite_kernels.cu:9010-9054`（`AR5_TIMEOUT_SPINS` / PARK / TRAP）；
   * A0（探针）：`:8965-9008`（site/rank/seg），env `DSV41_AR_PROBE`（`:9290`，默认 OFF）；
   * A1a（store 折进 producer）：MoE 侧 `ar_store_fuse_moe`（`chain_dev.rs:2230`）+ `ar_hc_post_fold_after_store`（`:5332`）
     + 三个 store-less 入口（`ferrite_p2p_ar_pubred_v5{,_moe,_hcpost}`，`:9403/:9425/:9652`）。
   **它们全是 OFF 的**（`DSV41_AR_STORE_FUSE` 默认 OFF，`ar_store_fuse()` @ `:2203`）。⇒ 顺序是「先验证+定谳，再上第二件」。【读码】

---

## 1. Step A（**本次会话**）—— 身份 + 三段账 + 两个零代码 A/B

> 目的：**一次会话**同时回答 (i) AR 是真是假（工作 or 等待）；(ii) 该走哪条优化分支；
> (iii) 两个「已实现/零代码」的项是否立刻兑现。**不新增任何算术假设。**
> 姿态：**一臂一进程**（gate 用 `OnceLock` 读一次）；**一 prompt 一 serve**；**交错 A B C D A B C D** 抵消热漂。

### 1.1 先做符号与进程自检（**不上 GPU 也要做**，防「设了没生效」的幻影门）

```bash
SO=kernels/cuda/libferrite_kernels.so        # 本机不存在 ⇒ 须在远端（build.sh 103a）
# ① .so 是否带 A1a / A4 / A0 / A2b 的全部符号
nm -D $SO | grep -cE 'ferrite_p2p_ar_pubred_v5$|ferrite_p2p_ar_pubred_v5_moe|ferrite_p2p_ar_pubred_v5_hcpost'
nm -D $SO | grep -cE 'ferrite_add_store|dsv41_moe_down_reduce_st|ferrite_p2p_ar_v5_hcpost_add'
# ② 进程真读到的 env（本仓 #1 陷阱）
tr '\0' '\n' < /proc/$(pgrep -x ferrite-serve|head -1)/environ | grep -E 'DSV41_AR_|DSV41_SWALLOW|DSV41_V5_LEDGER' | sort
```
判据：`ferrite_p2p_ar_v5_hcpost_add` **必须存在**——否则 `_hcpost` 会同时服务 ATTN 与 MoE，探针 site 翻倍（§0-6）。

### 1.2 四臂矩阵（同一 binary + 同一 `.so`）

`COMMON` = `swallow-full-gate-config.md §4` 的**全 gate**（含 `SPEC/DSPARK/SIDS_WRITEBACK/SWALLOW_STEP/
SWALLOW_EPOCH_PAD/VERIFY_GRAPH/TAP_INPUT/DRAFT_BF16_DOMAIN/MARKOV/FORK/RING_WIN/SH_PAIR_M/Wave1/DRAFT_*`），
**逐字手写**（不裸跑 `batched_400_v2.sh`：`:154` 仍是 `SWALLOW_EPOCH_PAD` 常数 pad，与 `t1t4` P4 的 over-pad 结论冲突）。
**禁止**：`LAZY_VERIFY` / `SEED_ALIGN`（抢臂，`swallow-full-gate-config §2`）。**`V5_LEDGER=0`**（10 个 D2H 同步点/步）。

| 臂 | 增量 | 读什么 | 回答 |
|---|---|---|---|
| **A**（参照） | 无 | `steady_median`、`[dspark] steps=` | 步时基准（同会话可比口径） |
| **B**（三段账） | `DSV41_AR_PROBE=1` | `[ar-probe]` 的 `avg_stamp/avg_spin/avg_epi` × site × **rank** | **AR = 工作还是等待（唯一决定性数据）** |
| **C**（惊群） | `=B` + `DSV41_AR_SINGLE_POLL=1` | 步时 + `ar5-hang` | 960→8 pollers 是否兑现 |
| **D**（A1a） | `=C` + `DSV41_AR_STORE_FUSE=1` | 步时 + 逐 token 一致 + store 实例数 | 2 发/轮 → 1 发是否兑现 |

* B/C/D 都带 `DSV41_AR_TIMEOUT_TRAP=1`（A2b 的 TRAP 模式，`ferrite_kernels.cu:9028-9032`）——A/B 轮里一次挂死就能确定性地失败，而不是等 1800s pool watchdog。
* prompt：**计数 `请从 1 数到 200…`，MAXTOK≥120**。约束：探针每 512 轮/site 才打印一行（`:9002` `if ((n & 511ull) == 0ull)`），84 轮/步 ⇒ **需 ≥7 步**才出第一行，≥13 步才出两行。
* nsys：**只在 D 臂之后单独补一次「计数轮」**（`--trace=cuda --cuda-graph-trace=node`），**只为数实例**：`p2p_ar_store_v5_kernel` 的 Instances（A1a 前后对比）、以及 #1 行的**完整 kernel 名**（§0-2/§0-3 的身份判定）。**该轮的 μs/占比一律不读。**

### 1.3 读什么（探针三段的含义与对照线）

`[ar-probe] rank=%d site=%d n=.. avg_spin=.. max=.. avg_stamp=.. avg_epi=.. max_epi=..`（单位 = SM 周期；B300 ~1.8GHz ⇒ **1μs ≈ 1800 cyc**）

| 段 | 语义（`ferrite_kernels.cu:8936-8959`） | 对照线 |
|---|---|---|
| `avg_stamp` | kernel 入口 → 本 rank 的 8 发 `atomicExch_system` + 屏障 + epoch 写完 | ~0.2~0.5μs（≈ 1k cyc）。**大 ⇒ store/发射路径贵（A1a 的靶子）** |
| `avg_spin` | 本 rank stamp 发布 → **最慢 peer** 到达 | 账本 17.3μs ≈ **31k cyc**；工作地板 ~5μs ≈ 9k cyc |
| `avg_epi` | 等待结束 → 本块 reduce + fold 退役（**工作地板**） | lazy 4~6μs；**SWALLOW 6× payload ⇒ 设计 8~15μs（≈ 15~27k cyc）** |
| `max_epi` | 最慢块的 epi | 与 `avg_epi` 拉开 ⇒ 尾波/占用 |

**rank 列的读法（决定 A2c 是否成立）**：
* 一个 rank 的 `avg_spin ≈ 0`、其余 ≈ 落后量 ⇒ **「最后到达者」是瓶颈** ⇒ **A2c 负重平衡**；
* 所有 rank 的 `avg_spin` 都大且彼此接近 ⇒ **同步本身**（协议/惊群/发射序列化）⇒ A4 → A1a → A2d → PDL。

### 1.4 判据（决策树，**合取**）

| 分支 | 探针读数 | 定性 | 下一步（§2） |
|---|---|---|---|
| **T1** | `avg_spin ≫ 31k cyc` 且 per-rank **不对称** | 某 rank 系统性晚到 | **A2c** 负重平衡（§2.1） |
| **T2** | `avg_spin ≫ 31k cyc` 且各 rank **均匀** | 同步/协议本身 | **A4 → A1a → A2d → PDL**（§2.2） |
| **T3** | `avg_spin` 小（≈ 地板）但 **`avg_epi ≫ 15μs`** | **AR 是工作绑定**（6× payload + 6 行 fold） | **不碰协议**；转 reduce/fold 的 kernel 优化（§2.3） |
| **T4** | `avg_spin ≈ 地板`（<5~8μs）**且** `avg_epi` 正常 | **nsys 的 36% 是放大伪影** | **停止 AR 投入**；转 MoE(17.4%)/投影(15.1%)/mrows（§2.4） |

**通过线**：`[ar-probe]` 每个 site 至少 1 行 × 8 个 rank；`0 ar5-hang`；计数红线绿；
`ar5-hang == 0` 且 `k_acc`/计数序列逐位=参照臂。
**缺证据（探针没出、或只在某个 rank 出）⇒ 不得下结论，exit 2。**

### 1.5 第一步里唯一的小改：**探针 site 分流**（0.5 人日，可先写不占 GPU）

现状：SWALLOW 的 84 轮全落 `AR5_SITE_OTHER`（§0-6）。改法（照 A1a 的「加符号不改 ABI」先例，`ferrite_kernels.cu:9419-9439`）：

* 新增两个 `extern "C"`：`ferrite_p2p_ar_v5_attn` / `ferrite_p2p_ar_v5_moe` —— 与 `ferrite_p2p_ar_v5`（`:9306`）
  **逐字同体**，只把 `p2p_ar_pubred_v5_kernel` 的 site 实参从 `AR5_SITE_OTHER` 换成 `AR5_SITE_ATTN` / `AR5_SITE_MOE`；
* `device.rs`：`Kernels` 增两字段 + 两个 `supports_*`；`tp.rs`：`all_reduce_inplace` 增一个 `site` 形参（或按调用点分两个包装
  `all_reduce_inplace_attn/_moe`），`chain_dev.rs` 的 `layer_rows`（`:11686`）传 ATTN、`moe_rows`（`:13523`）传 MOE；
* **零协议风险**：不改 store/stamp/epoch/轮数，只改一个编译期标签；`.so` 缺符号 ⇒ 回落 `all_reduce_inplace`（老路径），
  探针退回 lumped（不静默丢 AR）。
* **✅ 已实施（2026-09-12，ar-a0-site-split）**：`ferrite_p2p_ar_v5_attn` / `ferrite_p2p_ar_v5_moe`（`ferrite_kernels.cu:9544/9565`）+ `device.rs::ArV5Site`/`p2p_ar_v5_site`（符号缺失自动回落 plain 入口，AR 绝不丢）+ `tp.rs::all_reduce_inplace_attn/_moe` + `chain_dev.rs` 两处接线（`layer_rows` verify AR → ATTN、`moe_rows` verify AR → MOE）。`cargo check --workspace --all-targets` EXIT=0。site 判读表与 GPU 运行手册见 subagent 报告（ATTN/MOE 各 40 轮/步稳态；`avg_spin ≈31k cyc ⇒ 账本成立靶子 A2c；≫31k ⇒ nsys 放大为主；≪31k ⇒ AR 无肉`）。**动了 .cu ⇒ 远端双产物重编 + `nm -D` 符号三证后再跑探针**。
* 若要更省事：也可以~~先不写这个~~（**已写完**），直接用 `avg_epi + avg_spin` 的**总量**走 T1~T4 的分支（分数会略粗，但 T3/T4 的判定不受影响）。

> ⚠️ 该改动**不进吞吐轮**（吞吐轮用 A 臂）；它是**诊断件**，与 §1.2 的 B 臂同轮跑。

---

## 2. Step B —— 四条分支的具体设计（按 Step A 的判定选一条）

### 2.1 T1：rank 负重不平衡（A2c）——SWALLOW 的三个可疑源

判据：某 rank `avg_spin ≈ 0`，其余 ≈ 落后量。落点（**改的是工作分配，不是 AR**）：

| 源 | 机制 | 代码落点 |
|---|---|---|
| **共享专家所有权** | `shared_here = sh_w.is_some() && sh_w2_ok`——只有部分 rank 跑共享专家那一份 GEMV | `chain_dev.rs:2230-2260`（`ar_store_fuse_moe` 的判据）、`moe()`/`moe_rows()` 的进入条件 |
| **engram AR + gather** | `eng_rows_r` 每层一发 AR + **per-row gather 循环**（6 次 launch），表按 rank 分片 ⇒ 片大的 rank 晚 | `:12489`（AR）、`:12477-12488`（per-row gather） |
| **rank 0 的 head/argmax** | `argmax_xchg_v5*` 的 owner + head GEMV 常落 rank 0 | `:5189` 邻域、head 段 |

预期：落后 δ/层 ⇒ **每步 84δ**（δ=5μs ⇒ 0.42ms/步；δ=20μs ⇒ 1.7ms/步）。成本 1~3 人日，风险低（不碰协议）。

### 2.2 T2：同步/协议（均匀 spin）——按「成本×风险」排的四件

| 序 | 项 | 改动 | 预期 | 成本 | 风险 |
|---|---|---|---|---|---|
| **T2-1** | `DSV41_AR_SINGLE_POLL=1` | **零代码**（已实现，`:9281`） | 960→8 pollers；若惊群为主，**−0.3~3ms/步** | 0 | 低（须红线复验） |
| **T2-2** | `DSV41_AR_STORE_FUSE=1`（A1a） | **零代码**（已实现，`:9644-9669`） | 每轮 2 发→1 发，84 轮 ⇒ **−0.1~0.2ms/步** | 0（须重编 `.so`） | 中（last-writer 论证 + 双产物逐 token 一致） |
| **T2-3** | **A2d：MoE AR 纳入捕获段** | `moe_reduce` 的 AR 现在在段外（`chain_dev.rs:5365-5368`「a CUDA graph cannot contain the host barrier」）；`ar_v5()` 下已无 host barrier ⇒ 可入图 | **−0.2~0.5ms/步**（消 host jitter，40 轮的锁步化） | 3~5 人日 | 中（capture 合法性：段内不得有 cuda 主机调用/分配） |
| **T2-4** | **A1b：PDL（prologue 重叠）** | 后继 launch 带 `programmaticStreamSerializationAllowed`，入口 `cudaGridDependencySynchronize()`（`:8409` 已有 mode 2/3 实验骨架） | −1~3μs/轮 ⇒ **−0.1~0.25ms/步** | 2~3 人日 | 中（capture 下 PDL 属性存活须先跑模式 3 实测） |

**A2d 在 SWALLOW 下的前置（必须读码确认，不是推断）**：SWALLOW 的 verify 段 `capture_begin/end` 的**边界**
与 `layer_rows`/`moe_rows` 的 AR 调用点的相对位置（`chain_dev.rs:5607-5609` 是 decode `step_body` 的捕获；
SWALLOW 的块捕获在 `swallow_step()` 一线）。若 SWALLOW 的 84 轮**已在图内** ⇒ A2d 对它**无效**，直接跳过。

### 2.3 T3：AR 是工作绑定（`avg_epi` 大）——不碰协议

SWALLOW 每轮的真工作是 lazy 的 **6 倍**（payload 120KB vs 20KB），且 `_hcpost_rows` 的 fold 是
**每列 4 读 4 写 × m 行** 的复读模型（`:9468-9497` 的 `ar5_hc_post_col4` 逐字展开）。
若 `avg_epi` 主导，靶子按收益排：
1. **fold 的复读**（`hc_res` 6 行 × 4 读 → 寄存器化/块内共享）；
2. **reduce 的 rank 循环**（每线程 8 次 16B L2 load、1 次 out 写、**120 blocks × 64 线程**铺满 120 个 SM —— 并行度已足，剩的是 L2 延迟）；
3. **store 的 16B/线程模式**（61,440 线程各 1 发 16B 远程 store，**延迟绑定**）⇒ 每线程 2~4 发向量化。
> 判定门槛：`avg_epi > 15μs`（>27k cyc）才走这支。

### 2.4 T4：AR 无肉（**最应该被认真对待的结果**）

若 `avg_spin ≈ 地板`，则 nsys 的 36% 是**自旋放大 + 可见分母塌缩**的产物（§0-7），
AR 的真实票面 = 账本 1.45ms/步（5%）⇒ **AR 的天花板（工作地板 5μs/轮 × 84 = 0.42ms）只剩 ~1ms/步**。
此时应立即**停止 AR 投入**，把 S1~S4 的预算重排到：
**MoE 17.4%（tcgen05，−2ms 设计）→ 投影 15.1%（mrows<6>/L4-1，−1~1.4ms）→ hc_dots 6.7% → B6**。
（注意：`swallow-nsys-batched-analysis-framework §0-3` 已经预告过这一支；`nsys-clean-stack-91 §行263`
甚至把 AR 直接列为「不是 L4/L5 目标」。）

---

## 3. ROI 排序（**按 SWALLOW 的兑现期望，不按设计票面**）

| 排名 | 项 | 预期（SWALLOW） | 代码 | 人日 | 风险 | 前置 |
|---:|---|---|---:|---:|---:|---|
| **R1** | **Step A：探针三段账 + 身份钉死** | 决定 R2~R6 的**全部**预算（±5.1ms） | 0（除 §1.5 的 0.5 人日 site 分流） | 0.5 | **0** | — |
| **R2** | **T2-1 `AR_SINGLE_POLL`** | **−0.3~3ms**（960→8 pollers） | **0** | 0 | 低 | 探针 `avg_spin` 大 |
| **R3** | T2-2 `AR_STORE_FUSE`（A1a） | −0.1~0.2ms | **0**（已实现） | 0（+1 次重编） | 中 | 逐 token 一致 |
| **R4** | T1 A2c 负重平衡 | −0.4~1ms | 1~3 | 低 | 探针 per-rank 不对称 |
| **R5** | T2-3 A2d（MoE AR 入图） | −0.2~0.5ms | 3~5 | 中 | 各 rank 均匀 + 段边界确认 |
| **R6** | T3 fold/reduce 向量化 | −0.3~1ms（**仅当 `avg_epi` 主导**） | 2~4 | 中 | `avg_epi > 15μs` |
| **R7** | T2-4 A1b PDL | −0.1~0.25ms | 2~3 | 中 | 模式 3 实测 |
| — | T4：转 MoE/投影 | 见 §2.4 | — | — | — | 探针判 T4 |

**一句话**：R1 的期望收益（把 5.1ms/步的不确定性变成确定性）**大于 R2~R7 的全部票面之和**。

---

## 4. 每项的验证方法（缺一不算完成）

| 项 | 最小证据集 |
|---|---|
| **R1 探针** | (a) `[ar-probe]` 每 site ≥1 行 × 8 rank，`n ≥ 512`；(b) `/proc/environ` 回读 `DSV41_AR_PROBE=1`；(c) **该轮无 nsys**；(d) 计数红线 + `ar5-hang=0` |
| **R1 身份** | `nm -D` 符号表 + nsys **计数轮**的**完整 kernel 名**（`p2p_ar_pubred_v5_kernel` vs `_hcpost` vs `_hcpost_rows`）+ `Instances/步 ≈ 84`（= 40 attn + 40 MoE + ~4 engram） |
| **R2 A4** | 步时 `steady_median` 位移 ≥0.5ms 才算兑现；`ar5-hang=0`；红线绿；`[ar-probe]` 的 `avg_spin` 同时下降（自证机理，防「快了但不是因为 poll」） |
| **R3 A1a** | ① 新 `.so` + `AR_STORE_FUSE=0` vs 旧 `.so` ⇒ **逐 token 一致**（重编译漂移）；② 新 `.so` 上 `=1` vs `=0` ⇒ 逐 token 一致；③ nsys 计数轮 `p2p_ar_store_v5_kernel` **Instances 下降**（MoE site：非 `shared_here` 的 rank 每层 −1） |
| **R4 A2c** | `[ar-probe]` 的 per-rank `avg_spin` 从「一超多零」变平；步时位移 ≥0.3ms；红线绿 |
| **R5 A2d** | `[verify_graph] captured` 出现 + 图回放 vs 直发逐 token 一致；`ar5-hang=0` ×3 run |
| **R6 fold/reduce** | 位级：改动前后 `k_acc` 逐位不变 + 计数前 61 行一致；nsys **计数轮** 该 kernel 的 Instances 不变（证明是同体优化） |

---

## 5. 陷阱清单（本条路径专属）

1. **轮数不可变**：任何 AR 改动必须保持 **84 轮/步**。`SWALLOW_EPOCH_PAD` 的存在正是因为「不同臂轮数不同 ⇒ epoch rift」
   （`chain_dev.rs:1887-1905`、`ferrite_kernels.cu:9212-9230`）。A1a/A4/A2d 都**不改轮数**（store 只是提前、poll 只是收敛、
   发射点只是入图）——**改动后必须用 `[dspark] steps=` 的 epoch 增量复核 ≈165**（84 真 + 81 pad）。
2. **探针与 nsys 不可同跑**；**探针只在 block 0 计时**（`ferrite_kernels.cu:9127`、`:9555`），所以 `avg_spin` 是
   「block 0 的 8 个 poller 的最大值」——读它时不要当成「全 grid 的最大值」。
3. **`avg_spin` 的 512 轮打印窗口**：MAXTOK 太小 ⇒ 探针一行都不出（假失败）。
4. **A4 的广播字与 ledger canary 不重叠**（§0-5），但**吞吐轮 `V5_LEDGER` 仍必须 0**（10 D2H/步）。
5. **`DSV41_AR_STORE_FUSE=1` 的 A1a 覆盖缺口**：`_hcpost_rows`（多行折叠）**没有 store-less 孪生**
   （只有 `ferrite_p2p_ar_pubred_v5_hcpost` 单行版，`:9652`）⇒ 若 `VERIFY_AR_FOLD=1`（⇒ `_hcpost_rows` 上场），
   verify 的 AR **仍是 2 发**。要在 SWALLOW 上吃满 A1a，得补 `ferrite_p2p_ar_pubred_v5_hcpost_rows`（A1a 第二步）。
   同理 **ADD_EPI/A5 覆盖的 rank 不携带 store**（`ar-step2 §7` 已声明）。
6. **`_hcpost` 的双职**：`.so` 缺 `ferrite_p2p_ar_v5_hcpost_add` 时 `_hcpost` 会同时服务 ATTN 与 MoE
   （`ferrite_kernels.cu:8974-8978`）⇒ site=1 计数翻倍。三证之一必须查该符号。
7. **口径三件套**：每次比较必须标 `arm + rounds/step + timer`（serve 墙钟 / `[dspark] steps=` / nsys），
   禁止跨会话跨栈比——`56.6 tok/s` 就是一次口径混用的产物（`swallow-unlocked-shpair-m6 §1`）。
8. **计数 prompt 是唯一分母**：`400 = 6 / 步时`；出师表（accept 1.214）下 400 物理不可达。

---

## 6. 一页纸交付

1. **第一步 = 把 AR 的账算清，不是立刻优化**：一次会话（4 臂 × 计数 prompt × 3 run）拿
   **`avg_stamp` / `avg_spin` / `avg_epi`（× site × rank）**——这是唯一能切开
   **账本 1.45ms/步 vs nsys 6.58ms/步（差 5.1ms）** 的工具，而那个差**比 S1 要省的还大**。
2. **同时必须钉死 AR 的「身份」**：`AR_V5=0` 与表里的 `pubred_v5_hcpost` **互相矛盾**（`AR_V5=0` 根本不上场 v5）；
   SWALLOW 稳态也**不该**有单行 `_hcpost`。身份不明 ⇒ 不投钱。
3. **同一会话里收两个零代码项**：`DSV41_AR_SINGLE_POLL=1`（**960→8 pollers/轮**，lazy 中性不构成反证，**先验最高**）
   与 `DSV41_AR_STORE_FUSE=1`（A1a，2 发→1 发）。这是「第一步」里唯一**确定能拿到**的东西。
4. **四条分支**：T1 负重不平衡（A2c）／T2 同步协议（A4→A1a→A2d→PDL）／T3 工作绑定（fold/reduce 向量化，不碰协议）／
   **T4 AR 无肉**（⇒ 停 AR，转 MoE 17.4% + 投影 15.1%）。
5. **纪律**：AR 改动**不得改轮数**（84 真 + 81 pad = 165）；探针轮**禁 nsys**；吞吐轮**禁 V5_LEDGER**；
   每次 A/B 必须逐 token 一致 + 计数红线 + `ar5-hang=0`。

## 7. A0 探针执行结果（2026-09-12 22:30 执行，✅已判决）

**配置**：site 分流（95d7083）双产物重编后；SWALLOW 63.8 生产栈（SWALLOW_STEP+EPOCH_PAD+VERIFY_GRAPH+b2+b3+Wave1 全开+HC_VERIFY_FUSE/FRONT_ROWS/AR_FOLD 三开门）+ `DSV41_AR_PROBE=1 AR_TIMEOUT_TRAP=1`；计数 200 tok（198 生成，50 步，mean-k=1.340，verify=28.17ms/步）；证据 `~/ar0_probe_EVIDENCE.log`（55 行探针，多 rank 并发打印有交错——用正则级提取解析 30 条完整记录）。

**稳态判读（site 分桶，取每 rank 最大 n——冷启动已摊薄）**：

| site | 是谁 | n 范围 | 8-rank avg_spin | 判读 |
|---|---|---|---|---|
| **2 VERIFY** | `_hcpost_rows` attn fold AR（63.8 栈三开门 ⇒ attn verify AR 走此桶） | 4096-7168 | **6179 cyc ≈ 3.4µs/轮** | **稳态真值** |
| 1 ATTN | 图捕获阶段 layer()（捕获完成后 n 不再增长） | 512 | 40874 cyc | 冷启动污染，排除 |
| 3 OTHER | engram 等 | 512 | 12102 cyc | 小样本 |
| 0 MOE | moe_rows verify AR（新标签） | 交错丢失 | （n=512 批约 4-12k cyc） | 量级同 site=2 |

**判决：`avg_spin ≪ 31k cyc（17.3µs）` ⇒ §6-4 的 T4 分支成立——AR 已无肉。**
- nsys 账本（78.3µs/轮、6.58ms/步、36% kernel-sum）被 device 侧真值否定：**真实 AR 等待 ≈ 3-7µs/轮，84 轮 ≈ 0.3-0.6ms/步 ≈ 28ms 步时的 ~2%**
- nsys 的 36% 是自旋放大假象（pubred 等 peer 的时间在 nsys 注入开销下级联放大 ~20×；探针 max 的 32M cyc 离群证实捕获期冷启动轮的存在）
- **AR 四分支 T1-T4 的终局 = T4**：T1（A2c 负载均衡）/T2（协议）/T3（fold 向量化）全部失去靶子——它们都建立在"等待是真"上
- **§6-2/§6-3 的零代码项（SINGLE_POLL/STORE_FUSE）正式失去先验**（R1 已 GPU 判死 7×，A1a 已判死 8×，现在连"理论收益"也不存在）
- **SWALLOW 28ms 步时的真实构成**：AR ≈ 0.5ms（此测量结论保留有效）。~~肉在真实计算 kernel = L4/L5 的对象；L4/L5 是 400 唯一路径~~ **⛔ 2026-09-13 订正**：此推论作废——28.17ms 本身是 verify 未摊薄的病（应为 eager+ε ≈ 7ms），主战场是逐 kernel 找 ~6× 未摊薄项，见 `mtp-verify-amortization-model.md`（400 = step 8ms + acc 2-3）。

**后续纪律**：AR 方向**全关**（R1/R2/R3/A1a/A2c/T1-T3）；任何新优化提案若引用 nsys 的 AR µs 数，必须先过 A0 探针复测；探针轮禁 nsys、吞吐轮禁 V5_LEDGER、AR 改动不得改轮数（165）的纪律不变。

---

*工部 · 只读勘察 + 本文件（唯一产出）；未执行 GPU 命令、未改动任何源码。*
*每条 μs/ms 已标来源（实测／账本／设计／代数／nsys）；与任务前提冲突处给出 file:line 依据（§0-1~§0-4）。*
*代码基线 HEAD `ba88df1`；行号漂移处以函数名 + gate 名为准。*
*§7 为 2026-09-12 深夜主 agent 执行 A0 后追加（基线已推进到 95d7083）。*
