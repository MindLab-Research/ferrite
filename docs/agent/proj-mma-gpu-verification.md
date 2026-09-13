# PROJ-MMA 接线（第一轮 ks=1）— GPU 验证手册

> 上游设计：`docs/agent/tensorcore-proj-design.md`（§2 数值等价、§3 实施框架、§4 收益估算、§7 并入时机）。
> 载体代码：`kernels/cuda/dsv41_proj_mma_skel.cu`（516 行，独立 TU）+ 本次接线
> （`kernels/cuda/build.sh` / `crates/ferrite-models/src/dsv41/device.rs` /
> `crates/ferrite-models/src/dsv41/chain_dev.rs`）。
> **本文是验证手册，不是实测结论**：本任务禁止 GPU/e2e，**没有任何 e2e 数据**；
> 唯一的实测是远端 `nvcc` compile-only。本文的「票面」数字全部来自设计 §4 的**估算**，
> 已逐条标注出处。
> 日期：2026-09-13。

---

## 0. 一句话

投影族（`proj_mrows` = wq_a / wkv / wq_b / wo_b + indexer 的 wq_b）多了一个**张量化程序**
`gemm_fp8_mrows_mma_kernel<M>`；门 `DSV41_PROJ_MMA=1`（默认 OFF）只是一个**符号选择**，
真正的判活靠**双门禁**（step_ms **AND** mean-k）。**这不是逐位臂**——它与 SIMT mrows
不逐位（设计 §2.2 已证不可达），所以**禁止**用 byte-compare 对 SIMT 判活。

---

## 1. 本次接线清单（diff）

| 文件 | 改动 | 验收 |
|---|---|---|
| `kernels/cuda/build.sh` | PROJ-MMA TU 并入 `SRCS`（`dsv41_proj_mma_skel.cu`）⇒ 每个 `.so` 都带符号 `dsv41_gemm_fp8_mrows_mma` | 远端 `bash build.sh 103a` EXIT=0 + `nm -D` 见符号（§3） |
| `crates/ferrite-models/src/dsv41/device.rs` | 新 FFI 项 `gemm_fp8_mrows_mma`（ABI 14 参）+ `Device::gemm_fp8_mrows_mma`（与 `gemm_fp8_mrows` 同形）+ `supports_gemm_fp8_mrows_mma` | `cargo check --workspace --all-targets` EXIT=0 |
| `crates/ferrite-models/src/dsv41/chain_dev.rs` | ① gate `DevChain::proj_mma()`（严格 `== "1"`，默认 OFF）② `proj_mrows` 三分支选路 **① MMA > ② MPAR > ③ legacy**（① 检查在 `swapab()` 拒绝**之前**）③ receipt `proj_mma_declined_note()` | 同上门禁 |

**为什么 ① 必须在 `swapab()` 拒绝之前**：`proj_mrows` 原来在 `Self::swapab()` 为真时直接
`Ok(false)`（投影退回逐行 `gemm_fp8_mx_or_swap`）。(b′) 臂要**成对**开
`DSV41_PROJ_MMA=1` + `DSV41_SWAPAB=1`（设计 §6 风险 #1），若拒绝在前，配对开法下 mma 分支
**永不可达**——正是本树反复踩的「armed but inert」幻影门。三分支顺序即设计 §3.3 的
`① PROJ_MMA > ② MPAR（逐位）> ③ legacy（逐位）`；②/③ 都在 C 入口
`dsv41_gemm_fp8_mrows` 内部，所以 Rust 侧只需「mma 符号 or C 入口」。

---

## 2. 门禁与回执（判活前先看这三行）

### 2.1 门（env）

| 名字 | 取值 | 语义 |
|---|---|---|
| `DSV41_PROJ_MMA` | **严格 `"1"`**（其余/unset = OFF） | 允许走张量化程序 |
| `DSV41_PROJ_MMA_KS` | 正整数 | **显式 ks**（sweep 用）。**第一轮必须 `=1`**（见 §2.3） |
| `DSV41_SWAPAB` | `"1"` | (b′) 成对开法（§6.2） |

### 2.2 回执（`stderr`，每进程一次）

| 行 | 出处 | 含义 |
|---|---|---|
| `[proj-mma] ARMED m=.. n=.. k=.. ks=.. -> grid=.. warps, block=32, smem=..` | C launcher 首次 **ARMED** 发射 | 真的走了 mma，且给出解析后的几何 |
| `[proj-mma] ARMED but DECLINED: ...` | Rust `proj_mma_declined_note()` | **armed 但没走 mma**（.so 无符号 / 形状拒 / ks>1 无 scratch）⇒ 这一臂**不是** PROJ-MMA 测量 |
| `[mrows-mpar] ARMED ...` | C launcher（已有） | ② MPAR 臂。**① 与 ② 互斥**：① 命中时这一行不出现 |

> **判读纪律**：只有 `[proj-mma] ARMED`（而不是 DECLINED）出现，这一臂才算 arm 上了。
> 只有 DECLINED ⇒ 立即弃臂，不要解释（本树 #1 测量陷阱）。

### 2.3 第一轮为什么必须 `DSV41_PROJ_MMA_KS=1`

`ks` 由 **C 侧** `proj_mma_ks_for(n, k, sms, override)` 决定（只吃 `n,k`，**绝不吃 `m`**——
这是 (b′) 的唯一前提）。但 Rust 侧**本轮还没有** `[ks][M][n]` partial scratch：

- `Device::gemm_fp8_mrows_mma` 传 `partial = null`、`ctr = null`、`pmma_n = n`；
- C 侧 `if (ks > 1 && (partial == nullptr || ctr == nullptr)) return 2;`
  ⇒ **任何 ks>1 都 decline 回 legacy**，且**不碰 scratch**（零越界风险）。

自动 ks 在生产形状上都是 >1（wkv n=512→ks=32、wq_a n=1280→ks=8、wq_b n=4096→ks=4、
wo_b n=5120→ks=2），所以第一轮**必须显式 `DSV41_PROJ_MMA_KS=1`**。
→ 本轮拿到的是**符号与数值结论**（程序对不对、mean-k 塌不塌），**不是**性能结论
（ks=1 时 wo_b 只有 320 个 warp，小 n 档更少；性能要等 scratch widening 后按 ks 规则顶满）。

---

## 3. 前置：双产物同源重建（唯一可行顺序）

`.cu` 已改（TU 并入）⇒ **`build.sh` 与 `cargo build` 必须成对、且顺序固定**
（否则 `.build_id` 不匹配，进程在 dlopen 处拒启）：

```bash
cd /home/smith/src/ferrite
(cd kernels/cuda && bash build.sh 103a)          # 写 .build_id + 产出 .so
touch crates/ferrite-kernel/build.rs             # 强制 build.rs 重跑（cargo 增量会跳过）
cargo build --release
# 证据
nm -D kernels/cuda/libferrite_kernels.so | grep -c dsv41_gemm_fp8_mrows_mma   # 期望 1
cat kernels/cuda/.build_id
strings target/release/dsv41-run | grep -cF "$(cat kernels/cuda/.build_id)"    # 期望 >=1
```

---

## 4. 臂设计（**一臂一进程**）

| 臂 | env | 目的 |
|---|---|---|
| **A0**（基线） | `DSV41_TIMING=1` | mean-k / step_ms / verify_ms 的对照（clean 基线见 §6.1） |
| **AP**（PROJ-MMA） | `DSV41_TIMING=1 DSV41_PROJ_MMA=1 DSV41_PROJ_MMA_KS=1` | ① 只验「mma 程序接上了、数值没崩」 |
| **AP′**（(b′)，本轮**尽力**） | `… DSV41_PROJ_MMA=1 DSV41_PROJ_MMA_KS=1 DSV41_SWAPAB=1` | ① + m=1 侧也走张量核（见 §6.2 的**诚实边界**） |

驱动：`scripts/dsv41_serve_ab.sh <tag> [VAR=VALUE ...]`（四 prompt、`DSV41_TIMING=1`、
逐 step `[dsv41] step pos=N: X ms` 是唯一接受的计时基准，段均值不算）。
**每个臂前先 `grep '[proj-mma]' /tmp/ab_<tag>.log`**（§2.2）。

---

## 5. 门禁一：step_ms（票面 −3.5ms，来自设计 §4，**非实测**）

设计 §4 的中央值（**估算**）：投影族 ≈ 21% × verify 24.5ms ⇒ **−3.5ms**
（verify **24.5 → 21.0ms**），区间 **−2.5 … −4.5ms**。三个不相交分量：
E1 M 折叠变免费（−2.5）／E2 单行成本坍缩（−0.3）／E3 (b′) 下 m=1 同源（−0.7）。

**判据**（读 `[dspark] steps=... mean-k=... draft=... verify=... commit=...` 的 `verify=` 字段
与 `[dsv41] step pos=` 的 p50）：

| 观察 | 判读 |
|---|---|
| verify 降幅 **≥ 2.5ms** | 票面兑现，进入 §6.2 / §7 的下一步 |
| 降幅 0 … 2.5ms | 部分兑现——**但先看 ks**：ks=1 的网格远小于规则值（§2.3），性能结论**本轮不成立**，只记符号 |
| 降幅 < 0（变慢） | 与 ks=1 的小网格一致的可能；**不要**因此判死 mma 路，性能判活留给 scratch widening 后的 ks 规则臂 |
| **> −1ms 反而改善** 且 §6.2 mean-k 也掉 | 直接弃臂 |

⚠️ **禁止**在 §6.2（mean-k）过关前用性能数下结论（MPAR 的教训：先看符号，再看幅度）。

---

## 6. 门禁二：mean-k（**换程序后 acc 必须重验**）

### 6.1 基线

| 口径 | 值 | 出处 |
|---|---|---|
| clean 基线（S1 fix-on，SWALLOW accept-optimal 栈） | **mean-k = 2.240** | `verify-amortization-lesion-audit.md` §10.10（commit `6937d77`） |
| 塌陷基线（S1 污染） | 1.34 / 1.38（fix-off） | 同上（WOB 形状：非逐位投影 ⇒ accept 崩） |
| 接受带 | **2 – 3** | `400-*` 系列一致口径 |

**判据**：AP/AP′ 臂的 mean-k **必须落在 2–3 带**。

- mean-k ≥ 2.0 且 step_ms 同向 ⇒ 该臂**数值过关**（可用）；
- **mean-k 掉出 2–3（尤其回到 1.2–1.4）⇒ 直接弃臂**，不解释、不看性能——
  这正是设计 §2.3 判的「非逐位改动**充分危险**、无法静态排除」的实测形态
  （WOB 先例：`DSV41_VERIFY_WOB_MROWS_F32` 让它掉到 0.75–0.92）。

⚠️ 为什么**必须**重验：mma 换的是**程序**。设计 §2.4.1 的逐位等价
（`row r of an M-row launch ≡ row r of the M=1 launch`）**只在 mma 程序内部**成立；
它与**旧 SIMT 程序**的跨程序差是 ~几 ULP 级（设计 §2.1 D1/D2/D3），
near-tie argmax 对 1 ULP 敏感 ⇒ **acc 只能实测**。

### 6.2 AP′（(b′)）的诚实边界——**本轮未能完全满足**

设计 §2.4.2/§6.1：(b′) 要求 **m=1 与 m=2..8 走同一个 mma 程序**。本轮实现只覆盖
**verify 侧的 `proj_mrows`**（m ≤ VERIFY_ROWS，含 m=1 的 verify 调用）。

- 开 `DSV41_SWAPAB=1` 让 EAGER/draft 的 m=1 也走**张量核**，但走的是
  `gemm_fp8_swapab_kernel`——**与 mma 程序不是同一个程序**。
- ⇒ 严格意义的 (b′)（同源）本轮**不成立**；AP′ 的形态更接近设计 §2.3 的
  **(a)**（跨程序），风险类型是 WOB。
- **因此 AP′ 的成绩只能用双门禁读，不能当 (b′) 的结论**。真正的 (b′)
  需要把 EAGER/draft 的 m=1 投影也改路由到 `gemm_fp8_mrows_mma`（= 设计 §7 的下一刀）。

---

## 7. 内部契约（**唯一可做的 byte-compare**，先过它再看 e2e）

按设计 §2.4.1，mma 程序**自己内部**逐位等价，与 ks 无关（前提：**ks 是 (n,k) 的函数**）：

```
对每个 m ∈ 1..8、每个 ks ∈ {1}：
  m 行 launch 的第 r 行  ==  m=1 launch 的第 r 行      （逐位）
```

- 载体：`kernels/cuda/tests_dsv41_gemm_mrows.cu` 形态的 micro bench
  （`dsv41_gemm_fp8_mrows_mma` 直调，`ks` 固定，读回做 `memcmp`/`to_bits` 比较）。
- **先过这一条**（它证明程序自洽），**再**做 §4 的 e2e 臂；
- **禁止**拿它当「与 SIMT 等价」的证据（设计 §2.2：不可能）。

---

## 8. 与 SIMT 的关系：只测容差 + 文本指纹

参考 `crates/ferrite-dsv41/tests/swapab_parity.rs` 的判据（设计 §2.4.2 先例）：
**容差 + 文本指纹**（DIFF_EAGER / 四 prompt 输出 / 数字位置），**不要** byte-compare。
预期量级：~几 ULP（1e-7 相对），**但 near-tie 会翻**——这是 §6.2 存在的原因。

---

## 9. nsys 判据（投影族 kernel **换名**）

| 项 | legacy 臂 | PROJ-MMA 臂 |
|---|---|---|
| kernel 名 | `gemm_fp8_mrows_kernel<M>`（MPAR 时 `gemm_fp8_mrows_mp_kernel<M>`） | **`gemm_fp8_mrows_mma_kernel<M>`** |
| grid | `ceil(n/nwarps) * ng` | **`(n/16) * ks`** |
| block | `nwarps*32` | **32（=1 warp）** |
| 每 step 调用数 | 每投影 1 发 | 同（每投影 1 发，名字/几何变） |

- **判活**：nsys 里出现 `gemm_fp8_mrows_mma_kernel`，且 `M` 等于该步的行数（verify m）；
  `GridX = (n/16)*ks`（n=5120、ks=1 ⇒ 320）。
- **判死/幻影**：只有 `[proj-mma] ARMED` 却没有该 kernel 名 ⇒ 不可能（名字是唯一的），
  优先怀疑 nsys 过滤/符号截断。
- 附带（micro bench only，非本轮）：`sm__throughput` 应从 ~13% 抬起、
  `smsp__warp_issue_stalled_short_scoreboard` 应降、
  `l1tex__data_pipe_lsu_wavefronts_mem_shared` 应显著降（**LUT 消失**——mma 程序不建 256 项表）。

---

## 10. 回滚

`unset DSV41_PROJ_MMA`（或 `=0`）⇒ `proj_mma()` 恒 false ⇒ `proj_mrows` 走
`dsv41_gemm_fp8_mrows`（②/③ 原路），**逐字节回 legacy**。`.so` 带符号本身**不改变任何行为**
（C 侧 `g_proj_mma` 默认 0，`dsv41_gemm_fp8_mrows_mma` 立即 `return 2`）。

---

## 11. 本轮**范围外**（诚实标注，别误读成失败）

| # | 项 | 状态 |
|---|---|---|
| 1 | `[ks][M][n]` partial scratch（Rust `chain_dev.rs` sizing） | ❌ 未做 ⇒ ks>1 全 decline（§2.3）。需要时再改（设计 §3.4 第二刀） |
| 2 | 严格 (b′)（EAGER/draft m=1 也走 mma 程序） | ❌ 未做（§6.2）——跨程序，风险类型仍是 WOB |
| 3 | 性能判活（ks 规则顶满网格） | ❌ 本轮不可得（ks 被钉在 1） |
| 4 | epilogue store 合并度（4 次分散 STG）、激活暂存 vs 直读 | ❌ 设计 §6 的 follow-up 旋钮 |
| 5 | PDL：mma launcher 用**普通** `<<<>>>` 发射（骨架原样），而 legacy/swapAB 走 `dsv41_pdl_or_plain` | ⚠️ 只是**少一次重叠**，**不是**正确性问题（普通发射 = 正常流序）；列为此后旋钮 |
| 6 | `ks` 与 `m` 无关（(b′) 唯一前提） | ✅ **已硬写进 C 签名**（`proj_mma_ks_for(n,k,sms,override)` 只吃 n,k） |

---

## 12. 附：命令速查

```bash
# 0) 编译级验收（本任务已做，见交付报告）
(cd kernels/cuda && bash build.sh 103a) && cargo check --workspace --all-targets
nm -D kernels/cuda/libferrite_kernels.so | grep dsv41_gemm_fp8_mrows_mma

# 1) 双产物同源（§3）
(cd kernels/cuda && bash build.sh 103a) && touch crates/ferrite-kernel/build.rs && cargo build --release

# 2) A0 基线
scripts/dsv41_serve_ab.sh a0 DSV41_TIMING=1

# 3) AP 臂（第一轮：必须 KS=1）
scripts/dsv41_serve_ab.sh ap DSV41_TIMING=1 DSV41_PROJ_MMA=1 DSV41_PROJ_MMA_KS=1

# 4) 先看回执，再看双门禁（§2.2 / §5 / §6）
grep -E '\[proj-mma\]|\[mrows-mpar\]' /tmp/ab_ap.log
grep -E '\[dspark\] steps=|\[dsv41\] step pos=' /tmp/ab_ap.log | tail

# 5) nsys（投影族换名，§9）
nsys profile -o /tmp/ap --force-overwrite=true --trace=cuda ./target/release/dsv41-run --serve ...
nsys stats --report cuda_gpu_kern_sum /tmp/ap.nsys-rep | grep -E 'mrows'
```
