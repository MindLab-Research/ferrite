# A1a（AR Step 2 store fold）修复方案设计 —— 修 vs 弃

> 工部 · 2026-09-12 · **只读勘察 + 本文件（唯一产出）**。未改任何源码，未执行任何 GPU 命令。
> 对象：`docs/agent/ar-step2-a1a-moe-store-implementation.md`（实现）、
> `docs/agent/ar-step2-a1a-moe-store-gpu-verification.md`（验证设计）、
> `docs/agent/dspark-correctness-chain.md` §6266-6278（退化判定）。
> ⚠️ **任务里提到的 `docs/agent/ar-step2-regression-rootcause.md` 在工作树、`git log --all`、
> 悬空对象、reflog 里都不存在**（已穷举：`find -iname '*rootcause*'`、`git log --all -- <path>`、
> `git fsck --lost-found`）。因此本文件以**现场读码 + correctness-chain 最新段**为输入，
> 自行完成根因定位。行号均在当前树（`31a341d` 前后）现场核对，**注意有 peer 正在改 `chain_dev.rs`，
> 行号会漂移，引用以函数名为准**。

---

## 0. 结论（先给决策）

**弃（deprecate A1a 的 store fold），并重定向 AR 优化。**

三条理由，全部有代码证据（§2）：

1. **A1a 的 MoE 载体在默认/生产配置下不可达** —— 即 `+665` 行在实测配置里**一个 launch 都没省**
   （routed 载体被 `DSV41_DOWN_FUSE=1` 挡在 `else` 分支；shared 载体被 `DSV41_ADD_EPI=1` 挡在
   `add_epi_ready()`）。⇒ **实测到的退化不可能出自 A1a 的 MoE 代码。**
2. **同一把门 `DSV41_AR_STORE_FUSE` 同时打开 attn 折**，而 attn 折是**树内既有的、
   文档已记录为坏的特性**（`chain_dev.rs` 的注释：round 19 破四文本）。⇒ 实测退化 = attn 折。
3. 门还**静默切换 wo_b 的数值路径**（关掉 `DSV41_WOB_F32`），这是与 store 无关的数值变更，
   本身就违反项目"gate ON vs OFF 逐字节一致"的纪律。

**收益上限**（设计自述）：`−0.08~0.16 ms/步`，是 400 阶梯里最小的一项（mrows/hc/tcgen05 是它的
10~30 倍）。**成本**：要让 MoE 半真正生效，必须给**融合 kernel** `expert_down_reduce_fp4_batched`
新加一个 store epilogue（新 kernel 改动 + 位级测试），再叠加 attn 折的未知根因调试 ——
**超过 2 人日硬止损**。

---

## 1. 现场事实（决策的前提，逐条核对）

| # | 事实 | 证据 |
|---|---|---|
| F1 | **只有一把门**：`DSV41_AR_STORE_FUSE`。`ar_store_fuse()`（attn）与 `ar_store_fuse_moe()`（MoE）**共用**它 | `chain_dev.rs::ar_store_fuse` / `::ar_store_fuse_moe`（后者首行即 `ar_store_fuse() && …`） |
| F2 | 代码里**不存在** `DSV41_AR_ST_ATTN` / `DSV41_AR_ST_MOE`（`stage-b-execution.md` 里写过，但全仓 grep 无符号） | `grep -rn AR_ST_ATTN\|AR_ST_MOE crates/ = 0` |
| F3 | attn 折**是树内既有**（A1a 之前就在），且注释明确记录它坏过 | `chain_dev.rs` `attention()` 内：*"round 19 showed the fused path breaks the four texts even with GATEUP/DOWN_FUSE off … DSV41_AR_STORE_FUSE=1 re-enables"* |
| F4 | **routed MoE 载体只在 `!down_fuse()` 分支** | 见 §2.2 代码 |
| F5 | **shared MoE 载体只在 `!add_epi_ready()`**，而 `add_epi()`/`hcpost_epi()`/`fuse_c()` 三个默认全 ON | `chain_dev.rs::add_epi/hcpost_epi/fuse_c` 均 `unwrap_or(true)`；`::add_epi_ready()` |
| F6 | `down_fuse()` 默认 **ON**（`.unwrap_or(true)`；`dsv41-layer-fusion.md` 说"默认 OFF"是**旧注释**） | `chain_dev.rs::down_fuse` |
| F7 | SWALLOW 全 gate 配置（`swallow-full-gate-config.md` §4）**没有** `DSV41_DOWN_FUSE` / `DSV41_ADD_EPI` / `DSV41_WOB_F32` 任何一行 ⇒ 全部取默认 | 该文件全文 grep 无这三个名字 |
| F8 | `DSV41_WOB_F32` 默认 ON，**且被 store 折强制关掉** | `chain_dev.rs` `attention()`：`if !wo_paired && !wo_fused && !ar_store_fused && Self::wob_f32() && …` |

---

## 2. 根因定位

### 2.1 归因错误的根源：一把门 = 两个折

```
DSV41_AR_STORE_FUSE
      ├── ar_store_fuse()      → attention() 的 wo_b store 折（树内既有，round 19 已判坏）
      └── ar_store_fuse_moe()  → A1a 新加的 MoE 载体（本次 +665 行）
```

`A1a` 的验收（`AR_FUSE=1`）把**两个折一起**打开。因此：

> **实测的"数值退化"不能作为 A1a 有 bug 的证据** —— 这是验证设计文档 §0.4-1 自己写下的警告
> （"门 ON 若数值失败，不能直接断定是 MoE 侧（attn 折在 round 19 有过失败史）"），
> 而 correctness-chain 的判定（§6275 "A1a 有根本性数值 bug"）**跳过了这一步**。

### 2.2 ★决定性发现：A1a 的 MoE 载体在默认配置里**根本不可达**

routed 载体（`dsv41_moe_down_reduce_st`）的实际调用点，**整个包在 `else` 里**：

```rust
// chain_dev.rs :: moe()  （约 :16669）
if down_fuse() && self.dev.supports_down_fuse() {
    self.dev.expert_down_reduce_fp4_batched(... out = s.o ...);   // 融合 down+reduce
} else {
    self.dev.expert_down_fp4_batched(... out = s.ex_down_b ...);
    if ar_carry && !shared_here {                                  // ← A1a 载体在这里
        ar_carried = self.dev.moe_down_reduce_ar(...);             //   （约 :16718）
    }
    if !ar_carried { self.dev.moe_down_reduce(...); }
}
```

`down_fuse()` **默认 ON**（F6）⇒ 默认配置永远走上面那条分支，`expert_down_reduce_fp4_batched`
是 `s.o` 的最后写者、而它**没有** store epilogue ⇒ `ar_carried` 恒 `false`。

shared 载体（`ferrite_add_store`）同理：
```rust
if self.add_epi_ready() {          // ADD_EPI=1 + hcpost_epi=1 + fuse_c=1 ⇒ true（符号在）
    self.moe_add_in[layer] = Some(self.s.ex_out.ptr as *const f32);   // 延迟进 AR
} else if ar_carry {
    ar_carried = self.dev.add_inplace_ar(...);                        // ← A1a 载体（约 :16946/:16983）
} else { self.dev.add_inplace(...); }
```
默认 `ADD_EPI=1` ⇒ merge 被折进 AR 自己的 store epilogue ⇒ **没有可挂载的 producer**。

> **⇒ 在 SWALLOW 全 gate（以及任何默认）配置下，A1a 的两个 MoE 载体一次都不发。**
> 实测的 regression **不可能**由它们产生。

（旁证：验证设计 §3.1 预期"store 计数 80→40/步，MoE 折贡献 40×7" —— 这个预期**在默认
`DOWN_FUSE=1` 下不成立**，是对 reachability 的误判。要让它成立，必须显式 `DSV41_DOWN_FUSE=0`。）

### 2.3 那么数值是谁坏的：attn 折 + 一个"隐式数值耦合"

**（a）attn 折本身是既有坏特性**（F3）。它在 `attention()` 里把 wo_b 的 AR 改成 pubred-only，
依赖 gemv epilogue 把 partial 写进 peer 槽。round 19 已经判过它"破四文本"，
以 `DSV41_AR_STORE_FUSE` 默认 OFF 挂起 —— A1a 的验收把门打开，等于**复活了一个已知坏路径**。

**（b）★门还偷偷改了 wo_b 的算术**（F8）：

```rust
// attention() 里 wo_b 的激活，优先级 2 是 RAW-f32 GEMV（DSV41_WOB_F32 默认 ON）
if !wo_paired && !wo_fused && !ar_store_fused && Self::wob_f32() && … {
    wb_f32 = self.dev.gemm_fp8_mx_f32(...);          // 读原始 f32 s.wo
}
if !wo_paired && !wb_f32 {
    let (wb_q, wb_sc) = … quant1(self.s.wo) …;       // ← ar_store_fused 时被迫走这里
    if ar_store_fused { self.dev.gemm_fp8_mx_ar(...); }  //   fp8（e4m3 量化激活）GEMV
    else { self.gemm_fp8_mx_or_swap(...); }
}
```

`ar_store_fused=1` 会**强制关掉 `wb_f32`**，把 wo_b 的激活从"原始 f32"换成"quant1 后的 fp8"。
两者**不是逐位可交换的算术**（e4m3 block-32 量化 vs 原始 f32）。所以：

> **即使 store 折本身完全正确，`AR_FUSE=1` 也会改变 wo_b 的输出。**
> 这条耦合就足以让"gate ON vs OFF 逐字节一致"失败，并且与 store 无关。

### 2.4 SWALLOW 8× 的性能症状

**它不是独立的性能 bug，最可能是数值破坏的下游效应。** 依据：

| 观察 | 推论 |
|---|---|
| pubred 轮数不变（折只搬 store，不加/减 round） | AR 的同步结构没变，不会有 8× 的额外自旋 |
| `ar5-hang == 0`（A2b 超时会 park/挂死，不是慢） | 不是 epoch rift / 超时路径 |
| 折**减少**一个 launch，理论上更快 | 8× 不可能来自"少了 store kernel" |
| SWALLOW 吞吐 ∝ 每步 accept（spec decode） | 数值坏 → draft/verify 分歧 → accept 塌缩 → 吞吐成倍掉 |

⇒ 机制：**数值坏（§2.3）→ accept 塌缩 + 步内串行化 → 7.1 tok/s**。
lazy 不用 spec decode，所以只看到"数值坏、性能中性（90.9≈91.1）"。
**两者是同一个根因的两种表现**，不是两个 bug。

> ⚠️ 残余不确定：§2.2 的前提是实测 run 用了默认 `DOWN_FUSE=1`/`ADD_EPI=1`。SWALLOW 全 gate
> 配置（F7）确实不含这两个变量，但那次 `AR_FUSE=1` 是**手工命令**（全仓 grep 无脚本引用
> `AR_STORE_FUSE`）。**验证第一步必须用 `/proc/<pid>/environ` 实读确认**（项目的 #1 测量陷阱）。
> 若那次显式设了 `DOWN_FUSE=0`，则 MoE 载体是活的，§2.2 的"不可达"结论要改成
> "**默认不可达**"，但 §2.1/§2.3（门耦合、attn 既有坏、wo_b 路径切换）依然成立。

---

## 3. 修复方向 1/2/3 的逐条回答

### 3.1 "找出 store fold 的数值 bug"

**没有一个"store fold 的数值 bug"需要修** —— 实测数值破坏来自：

- **(M1)** attn 折（树内既有、round 19 已判坏）被同一把门复活；
- **(M2)** 门把 wo_b 从 `WOB_F32` 强制切到 fp8 量化路径（隐式数值耦合）；
- **(M3)** 即使 MoE 载体被激活，它的**写值与独立 store 逐位一致**（`add_kernel` 就是把
  `x+y` 的同一个寄存器值写两份；`moe_down_reduce_kernel` 是把定稿后的 `acc` 写两份），
  地址公式也与 `p2p_ar_store_v5_kernel` 逐字相同 —— **代码层面找不到数值差异**。
  换言之 A1a 的 MoE 半"照图施工"是对的，问题是它**不生效**（§2.2）。

### 3.2 "找出 SWALLOW 的性能 bug"

**没有独立的性能 bug。** 8× 是 accept 塌缩（§2.4）。若验证第一步证明那次 run 里 MoE 载体是活的，
那么**唯一**可疑的性能点是载体 epilogue 的**串行 peer 写**：

```cuda
// p2p_ar_v5_store_elem（ferrite_kernels.cu）/ dsv41_ar5_store_elem（dsv41_experts_mxf4.cu）
__device__ __forceinline__ void p2p_ar_v5_store_elem(
    float* const* __restrict__ staging_tbl, size_t base, int world, size_t i, float v) {
    #pragma unroll 4
    for (int rr = 0; rr < world; rr++) staging_tbl[rr][base + i] = v;   // 每个线程串行写 8 个 peer
}
```

对照 `p2p_ar_store_v5_kernel`：它把 peer 维放进 `gridDim.y`（"Before, ONE thread wrote all
`world` peer slots serially … 5.9us measured … Moving the peer loop to blockIdx.y … leaves
exactly ONE remote store per thread"）。**载体把这条已经修掉的串行远程写又请了回来**：
- peer 串行（少 8× 并行度）；
- **标量 4B** 远程写（独立 kernel 用的是 `float4`）；
- 而且它现在位于 **producer 的关键路径上**（不再与后续 kernel 重叠）。

这条最坏情况下是"每轮 store 从 ~5.9us 退化回串行档 × 80 轮/步"的量级，值得在 §3.2 归因实验里
用 `DSV41_DOWN_FUSE=0` 显式打开载体后单独量一次。

### 3.3 "修复或者放弃"

**放弃。** 见 §0/§4。收益上限 `−0.08~0.16ms/步`（设计自述），而要让 MoE 半真正生效必须
**给融合 kernel 加新 epilogue**（超预算），attn 半则是"既有坏特性 + 未知根因"。2 人日硬止损
下，把预算投给 mrows/hc/tcgen05 是明确更优解。

---

## 4. 弃案：具体改动（≤0.5 人日，可立即执行）

> 原则：**保留代码（历史证据），但把门变成"响亮不可用"**，并让未来的归因不再踩同一个坑。

### 4.1 `DSV41_AR_STORE_FUSE` 标记 deprecated

1. **`chain_dev.rs::ar_store_fuse()`**：读 env 时若被设为 `1`（含任何非 `"0"`），
   **一次性 epilogue 打印**（`eprintln!` 一次，非热路径）：
   ```
   [ar-store-fuse] DSV41_AR_STORE_FUSE is DEPRECATED (2026-09-12): it enables BOTH the attn
   fold (known-broken, round 19 — see ar-step2-a1a-fix-design.md §2.3) and the A1a MoE fold,
   whose carriers are UNREACHABLE under the default DSV41_DOWN_FUSE=1 / DSV41_ADD_EPI=1.
   It is retained for historical A/B only; do not use it in any 400-ladder arm.
   ```
   保留返回值语义不变（默认 OFF；`=1` 仍能打开，供复现/考古）。
2. **文档**：`AGENTS.md` 的 gate 表、`stage-b-execution.md` 第 2 项、`swallow-full-gate-config.md`
   各加一行"DEPRECATED — 见 `ar-step2-a1a-fix-design.md`"。
3. **验收/优化计划**：从 400 ladder 里删除 "AR fix (A1a)" 这一项，替换为 §5 的新方向。

### 4.2 清理"死门"引用（可选，0.2 人日）

`stage-b-execution.md` 提到的 `DSV41_AR_ST_ATTN` / `DSV41_AR_ST_MOE` **不存在**。两条路：
- **路线甲（推荐）**：不新增门，只把文档改成"单一门、且已 deprecated"，避免后人再去找这两个 env。
- **路线乙**：真的把门拆成两个（`DSV41_AR_ST_ATTN` / `DSV41_AR_ST_MOE`，默认 OFF，
  `DSV41_AR_STORE_FUSE=1` 作为"两个都开"的别名）。**这是任何后续 A1a 归因实验的前置**
  （§6 的隔离实验 A/B 需要它），但按弃案它只是"为将来留接口"，可选。

### 4.3 删除 / 隔离 MoE 载体的死代码（可选，0.3 人日，**建议做**）

现状：`+665` 行里大部分在默认配置**永不执行**（§2.2）。项目已有一条明确教训
（`e337b1e`："failed experiment code should be removed immediately, not left gated-off in the tree"）。
建议二选一：
- **保守**：保留，但在 `ar_store_fuse_moe()` 里加一个**编译期/启动期可达性断言**：
  当 `ar_store_fuse()` 为真**且** `down_fuse()` 为真**且** `add_epi_ready()` 为真时，
  打印一条 `[ar-store-fuse] the MoE carriers are UNREACHABLE in this configuration`
  （即"门开了但没生效"——正是这次踩的坑）。
- **激进**：删掉两个载体入口与其 Rust 接线，只留 attn 半（attn 半也默认 OFF）。

---

## 5. 弃案之后：新的 AR 优化方向（重定向）

AR 的真实预算（校准后口径）是 **84 轮 × 78.3µs ≈ 6.58ms/步**（36% 的 kernel-sum 占比）。
store fold 只能省 `~0.1ms`，**杠杆不在"更便宜的 store"**，而在下面几项：

| 序 | 方向 | 机制 | 预期 | 风险 |
|---|---|---|---|---|
| **R1** | **A4 单块轮询 + 广播**（树内已有 `DSV41_AR_SINGLE_POLL`，默认 OFF） | v5 让**每个 block** 都轮询所有 peer（160 个 poller）；A4 回到"block 0 轮询 + 一个广播字"，**160 → 8 个 poller** | 直接砍 `spin` 段（A0 探针的能量就在这） | 低（代码已在树内，只差 A/B） |
| **R2** | **减轮数：attn+MoE 两次 AR 合并为每层 1 轮** | 当前每层 2 轮（attn/MoE 各 1）× 40 层 = 80 轮；若把同一层的两个 partial 拼进**一次** AR（payload 2×dim，但**同步次数减半**） | 省 ~40 轮/步（≈ −3ms 的同步面） | 中高（payload 变大、slot 要扩容、数值顺序需重证） |
| **R3** | **A1b PDL**（programmatic dependent launch） | 让 pubred 的 poll 与前置工作**重叠**（`cudaGridDependencySynchronize` 之后才读 *epoch） | 隐藏一部分 spin | 中（图捕获 + 依赖语义） |
| **R4** | **异步化**：把 AR 变成"依赖驱动"而非"步内同步点" | 让下一层独立工作（norm/gate/engram 前端）在 AR 未回时先跑 | 取决于依赖图 | 高 |

**推荐顺序：R1 → R2 → R3**（R1 零风险且已在树内；R2 是唯一能带来 ms 级收益的拓扑改动）。

---

## 6. 验证方法（每一步都可执行；GPU 部分由有卡侧执行）

### 6.0 先钉死"那次 run 到底开了什么"（**零 GPU 成本，必须先做**）

```bash
# 若进程还在（或重跑一次）：实读进程 env —— 项目的 #1 测量陷阱
tr '\0' '\n' < /proc/$(pgrep -x ferrite-serve | head -1)/environ | grep -E '^DSV41_' | sort
# 必须回答：DSV41_DOWN_FUSE / DSV41_ADD_EPI / DSV41_WOB_F32 各是什么？
#   DOWN_FUSE 缺省(ON) 且 ADD_EPI 缺省(ON)  ⇒ §2.2 成立：A1a 载体不可达（期望结果）
#   DOWN_FUSE=0 或 ADD_EPI=0                ⇒ MoE 载体是活的，走 §6.2 的 B 线
```

### 6.1 ★归因实验（零代码改动）：旧 `.so` + `AR_FUSE=1` 是否同样退化

`stage-b-execution.md` 的旧基线 = `3ea879f`（`d88cb41` 的父提交）：**有 attn 折、没有 A1a**。
按 `ar-step2-a1a-moe-store-gpu-verification.md` §0.2 的双产物纪律建旧 pair（旧 `.so` 必须配旧
binary，否则 build-id 三重校验 REFUSING TO START）：

```bash
VERIFY=/tmp/a1a-attrib; OLD_REV=3ea879f
git -C /home/smith/src/ferrite worktree add $VERIFY/tree-old $OLD_REV
cd $VERIFY/tree-old/kernels/cuda && bash build.sh 103a
cd $VERIFY/tree-old && touch crates/ferrite-kernel/build.rs \
  && CARGO_TARGET_DIR=$VERIFY/target-old cargo build --release
# 符号审计：A1a 的 4 个符号一个都不该有
nm -D --defined-only $VERIFY/tree-old/kernels/cuda/libferrite_kernels.so \
  | grep -cE 'ferrite_add_store|dsv41_moe_down_reduce_st|ferrite_p2p_ar_pubred_v5_moe|ferrite_p2p_ar_pubred_v5_hcpost'   # 期望 0
```

然后**同一 config、同一 binary**，只差门：

| arm | .so | 门 | 期望判读 |
|---|---|---|---|
| **A0** | old | `AR_STORE_FUSE=0` | 基线（lazy 91.1 / SWALLOW 58.3，前 61 行 ✓） |
| **A1** | old | `AR_STORE_FUSE=1` | **只开 attn 折** |
| **B0/B1** | new | `0` / `1` | 复现本次实测 |

**判读规则（决定性）**：
- 若 **A1 ≈ B1**（都退化：lazy 前 61 行 ✗ / SWALLOW 掉到 ~7 tok/s）
  ⇒ **退化 = attn 折**，A1a 的 MoE 半无责。**弃案成立**，且"因果链"文档要订正归属。
- 若 **A1 正常、B1 退化** ⇒ MoE 半真的有 bug（此时 §2.2 的 reachability 前提必被打破，
  说明那次 run 设了 `DOWN_FUSE=0`/`ADD_EPI=0`）⇒ 转 §6.2 B 线。

### 6.2 B 线（仅当 §6.1 证明 MoE 半分有责）：**显式打开载体**再二分

```bash
# 强制让 routed/shared 载体成为 last writer（复现那次 run 的可能配置）
DSV41_AR_STORE_FUSE=1 DSV41_DOWN_FUSE=0 DSV41_ADD_EPI=0 DSV41_WOB_F32=0 ...
```
- 先做 **wo_b 路径对照**：`WOB_F32=1`（默认，会被门关）vs `WOB_F32=0`（门开时的实际路径）。
  若两者文本就不同 ⇒ 证据是 **M2 隐式耦合**，与 store 无关。
- 再做 **store 正确性的位级判读**（不需要文本）：`nsys --report cuda_gpu_kern_sum` 读
  `p2p_ar_store_v5_kernel` 计数（只读计数、**绝不读 ms**：v5 自旋在 nsys 下放大 ~300×）。
  预期（`DOWN_FUSE=0 && ADD_EPI=0`，rank 1..7 routed + rank 0 shared）：
  store 计数应降到 **0/轮**；若没降 ⇒ 门/符号/reachability 有问题，A/B 作废。
- 再做 **payload 位级**：`DSV41_DOWN_FUSE=0 AR_FUSE=1` vs `DSV41_DOWN_FUSE=0 AR_FUSE=0`
  的 `[toktr]` md5（`DSV41_TOKTRACE=1`，段外打印、安全）。不同 ⇒ MoE 载体写值/地址有差异，
  此时才去查 `dsv41_ar5_slot_base` / `p2p_ar_v5_store_elem` 的逐字一致性。

### 6.3 弃案的验收（不碰 GPU 语义，只要"门响亮不可用"）

```bash
cargo check --workspace            # EXIT=0
cargo test --workspace             # 既有 1 个失败（shard_factor…）与本次无关，需与改动前对照
# 启动一次 serve，确认 deprecation 一行打印且只在 env=1 时出现：
DSV41_AR_STORE_FUSE=1 ... ferrite-serve ... 2>&1 | grep -c 'ar-store-fuse.*DEPRECATED'   # 期望 1
DSV41_AR_STORE_FUSE=0 ... ferrite-serve ... 2>&1 | grep -c 'DEPRECATED'                  # 期望 0（零开销）
```

### 6.4 新方向 R1 的验收（若采纳 §5 R1）

R1 = 打开树内已有的 `DSV41_AR_SINGLE_POLL`（A4）。判据用 **A0 探针**（`DSV41_AR_PROBE=1`）：
`[ar-probe]` 行的 `avg_spin` 应显著下降、`avg_epi` 不变；文本逐字节不变（A4 是纯同步优化）。
注意 nsys 只读计数、不读 ms。

---

## 7. 本设计**未**覆盖 / 遗留

- **`ar-step2-regression-rootcause.md` 缺失**：本文件是它的替代；建议把本文件的 §2 回填进
  `dspark-correctness-chain.md` §6266-6278，**订正"根因 = A1a"的归属**。
- **`DSV41_GRAPH_MOE` 是死路径**（无赋值点）——属既有发现，建议单独立项清理。
- **`moe_batch()` 默认值在文档里是错的**（`dsv41-layer-fusion.md` 说默认 OFF，代码是
  `unwrap_or(true)` = ON）——同类"文档 vs 代码"漂移会让下一次 reachability 判断再踩坑，
  建议做一次 gate 默认值审计。
- **§2.4 的 accept 塌缩**只是机制假设（未实测）；§6.1 的 A1 arm 如果拿到 accept 数据会更硬。

*工部 · 本文件为唯一产出；命令中模型目录 / nsys / nvcc 路径以节点实际为准。*
