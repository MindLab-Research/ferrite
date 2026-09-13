# head fold 的 v2 argmax 翻转 —— 根因判决 + v1-order 修复 + GPU 验证手册

> 工部 · 2026-09-13 · **未执行 GPU/e2e**（任务禁止；远端 nvcc 13.2 `sm_103a` compile-only 见 §6）。
> 触发：`draft-gate-diff` 的判词「v2 折已被实测翻 argmax 33%↔9%」。
> 上游判决：`chain_dev.rs` 的 `verify_head_fold` 文档（33%/9% 的实测口径）、
> `docs/agent/dspark-correctness-chain.md` §「draft head v2 fold vs verify head v1 的 program 不匹配」、
> `docs/agent/accept-ceiling-analysis.md` §2。
> 改动面：**2 文件**（`chain_dev.rs` 接线 + `dsv41_glue.cu` 注释），**零 kernel 体改动**。

---

## 0. 结论先行（六条，含两条对任务前提的修正）

1. **翻转的根源不是 argmax 的比较顺序、也不是多行合并**：折叠 kernel
   （`head_gemv_bf16_mrows_kernel`，`dsv41_glue.cu:864`）**体内没有任何 argmax**，
   且**已经是每行独立累加**（`float acc[M]`，`acc[r]` 之间从不相加，C4）。翻转来自
   **两个不同 GEMV「程序」的舍入差**：v1（逐行程序）vs v2（折叠程序）的
   **行内累加链分组不同**。见 §2。
2. **归约树不是差异点**（任务假设的另一半也否掉）：v1 的 `__shfl_xor_sync(off=16,8,4,2,1)`
   与 v2 的 `__shfl_down_sync(off=16,8,4,2,1)` 对 **lane 0 的最终值是两个完全相同的
   二叉归约树、相同的配对顺序**（xor 是蝶形、down 是树型，但 lane 0 的求和树同构）。
   差异 100% 在行内链上。见 §2.3。
3. **「让 v2 的 argmax 与 v1 逐位」在数学上不可实现**：v1 每 lane 是 160 步
   **严格串行** `acc += w*x`，v2 是 20 步 × 8 元素（两个 4 项 FMA 簇）。串行链无法被
   任何向量化重结合逐位复现 ⇒ **不存在「v2 参数重排后与 v1 逐位」的改动**。见 §3。
4. **唯一逐位等价的折叠 = 用 v1 自己的程序折叠**，即
   `dsv41_gemv_bf16_v1_mrows`（`dsv41_glue.cu:442` / launcher `:1624`）——它的 row r 是
   `gemv_bf16_kernel` 体的**逐字转写**。**draft head 早已用上它**（commit `e9fcf2a`，
   `dspark_dev.rs:3590`，gate `DSV41_DRAFT_HEAD_FOLD` 默认 ON）——**draft 侧无需改动**。
5. **本次修复的是唯一残留的可达 v2 折**：verify 的 **UNSLICED** 臂
   `chain_dev.rs:8000` 原本 arm 的就是 v2 折（`head_gemv_bf16_mrows`）。
   已改为 **v1-order 折**（`head_gemv_bf16_v1_mrows`）⇒ v2 折在**全仓范围内不再有任何
   head 调用点**（`grep` 见 §4.2），而该 gate 的收益（1262 MB head 读一次而不是 m 次、
   m 发变 1 发）完整保留。见 §4。
6. **票面必须分两段报**，不要把两件事混成一件：
   | 轴 | 状态 |
   |---|---|
   | **正确性**（argmax 一致性 / 文本 verbatim） | ✅ 本修复**构造性消除**（不再有 v2 程序跑 head） |
   | **accept（mean-k）** | ➖ 已知**中性**：`DRAFT_HEAD_FOLD v1` 实测 1.214 不变（`dspark-correctness-chain.md:1736`）。**不要指望本修复抬 accept** |

---

## 1. 事实基础：三个 head 调用点各跑哪个程序（先钉死）

`Device::gemv_bf16`（`device.rs:4612`）是**唯一的 head GEMV 入口**，它按
`gemv_bf16_v2_wanted(n)` 分派：

```rust
// device.rs:7929 / 7951-7956
const GEMV_V2_MAX_N: i32 = 2048;
fn gemv_bf16_v2_wanted(n: i32) -> bool { n > 0 && n < GEMV_V2_MAX_N && DSV41_GEMV_V2 != "0" }
```

| head 调用点 | `n` | 分派到的程序 |
|---|---|---|
| eager 主链 head（`chain_dev.rs:6840` / `:6848` / `:6915`） | `seg`=16160 或 `vocab`=129280 | **v1** `dsv41_gemv_bf16` → `gemv_bf16_kernel` |
| verify head（`chain_dev.rs:8047` / `:8097`） | `seg` 或 `vocab_size` | **v1** 同上 |
| verify UNSLICED 折（`chain_dev.rs:8000`，**本次改点**） | `vocab_size` | 原 **v2** 折 → 现 **v1-order 折** |
| draft head 折（`dspark_dev.rs:3590`） | `lg_pitch` = `seg` 或 `vocab` | **v1-order 折**（`e9fcf2a` 已修） |
| draft head per-row 回退（`dspark_dev.rs:3604`） | 同上 | **v1** |

**结论**：head 的 `n` 永远 ≥ 16160 ≫ 2048 ⇒ **head 的 per-row 程序恒为 v1**，v2/`nt`
（`gv2_wpr` 在 out_f ≥ 16384 时给 WPR=1 这件事）**对 head 永远不可达**。这正是 v2 折的
致命前提错误：它按 `gv2_wpr` 推「head 会拿到 v2/`nt`」，但 `gemv_bf16_v2_wanted` 这道
**host 侧闸**在它前面，head 从来拿不到。

---

## 2. 翻转根源（file:line + 累加链分析）

### 2.1 v1 程序（head 真正跑的）

`gemv_bf16_kernel`，`kernels/cuda/dsv41_glue.cu:370-395`：

```c
for (int row = blockIdx.x*nwarp + wid; row < n; row += gridDim.x*nwarp) {
    float acc = 0.f;
    for (int c = lane; c < k; c += 32) acc += __bfloat162float(wr[c]) * x[c];   // :389
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(~0u, acc, off); // :390-392
    if (lane == 0) out[row] = acc;                                                // :393
}
```

- lane→元素映射：`c = lane, lane+32, lane+64, …`（跨步 32）；
- `k = 5120` ⇒ **每 lane 160 个元素，160 步严格串行单元素累加**（每一步一次乘加）。

### 2.2 v2 程序（v2 折转写的那个）

`gemv_bf16_nt_kernel<NT, WPR>`，`kernels/cuda/ferrite_kernels.cu:3455-3536`，WPR==1 体：

```c
int kper = ((in_f + 1 - 1)/1 + 7) & ~7;      // = in_f（in_f%8==0）
int k0 = 0, k1 = min(kper, in_f);
for (int k = k0 + lane*8; k + 7 < k1; k += 32*8) {          // :3477 步长 256
    uint4 wv = *(const uint4*)(wr + k);                     // :3478 一次 8 个 bf16
    … f0..f3 = __bfloat1622float2(w2[0..3]);                // :3480-3483
    acc[t] += xa.x*f0.x + xa.y*f0.y + xa.z*f1.x + xa.w*f1.y;  // :3489  4 项 FMA 簇
    acc[t] += xb.x*f2.x + xb.y*f2.y + xb.z*f3.x + xb.w*f3.y;  // :3490  4 项 FMA 簇
}
```

- lane→元素映射：`c = lane*8, lane*8+256, …`（**连续 8 个**元素一组）；
- `k = 5120` ⇒ **每 lane 仍是 160 个元素，但分组为 20 步 × 8 元素**，每步先把 8 项
  折成两个 4 项 FMA 簇再进 `acc`。

**v2 折**（`head_gemv_bf16_mrows_kernel`，`dsv41_glue.cu:864-925`）是 2.2 的**逐字转写**
（`for (int c = lane*8; c + 7 < k; c += 32*8)`，同一表达式同一位置），所以
「v2 折 == v2 单行程序」是**逐位成立**的（`kernels/cuda/tests_dsv41_head_mrows.cu` 断言）。

### 2.3 唯一的差异点：行内累加链的分组（不是归约树）

| 维度 | v1（`gemv_bf16_kernel`） | v2（`gemv_bf16_nt_kernel<*,1>`） | 是否改变结果 |
|---|---|---|---|
| lane→元素映射 | `c = lane; c += 32` | `c = lane*8; c += 256` | 顺序变 → **是** |
| 每 lane 元素数 | 160 | 160 | 同 |
| 行内分组 | **160 × 1 项**（严格串行） | **20 × 8 项**（2×4 项 FMA 簇） | **是（根源）** |
| 归约树 | `__shfl_xor_sync off=16,8,4,2,1` | `__shfl_down_sync off=16,8,4,2,1` | **否**（见下） |
| 行间 | 一 lane 一 row | `acc[M]` 每 row 独立，**跨行永不相加** | 同（C4 已满足） |
| 尾处理 | `c < k` 标量覆盖任意 k | `k%8 != 0` 由 launcher 拒绝 | — |

**归约树等价性的逐位论证**（否掉「归约树改变」这一半假设）：两者都是 `off = 16,8,4,2,1`
的二叉归约，且 lane 0 的求和树同构：

```
off=16: lane0 += a16 ;  lane8 += a24 ;  lane4 += a20 ; …
off=8 : lane0 += (a8+a24) ;  lane4 += (a12+a28) ; …
off=4 : lane0 += (a4+a20+a12+a28) ; …
off=2 : lane0 += ((a2+a18)+(a10+a26)) + ((a6+a22)+(a14+a30)) ; …
off=1 : lane0 += （奇数 lane 半树）
```

xor 是蝶形（每个 lane 都得到全和）、down 是树型（只有 lane 0 得到全和），但**结果只从
lane 0 取**（`if (lane == 0)`），而 lane 0 在这两种指令下的**配对对象和配对顺序完全一致**。
⇒ 树贡献 **0 差异**；差异全部来自 §2.3 的行内分组。

### 2.4 一句话根源

> **v2 折的 row r 与 v1 逐行**在 5120 项点积上**用了不同的结合顺序**（160 步串行 vs
> 20×8 项分簇）⇒ f32 舍入分叉 ⇒ 129280 路 vocab 的**近 tie argmax 在漂移位置翻转**
> ⇒ `verify_out[0] == next` 从 9% 升到 33%（回显/复读）。

**实测口径提醒**（诚实边界）：33%↔9% 是 **serve 级**观测量，量级远大于纯 ulp 噪声，说明
受影响位置**系统性地落在近 tie 区**（复读吸引子）。而 accept 侧 A/B **中性**
（1.214 不变，`dspark-correctness-chain.md:1736`）。所以本修复的定位是
**「消除 draft/verify 的程序不一致（正确性/parity）」**，不是「抬 accept 的性能杠杆」。

---

## 3. 不可能性：为什么不能「把 v2 修成与 v1 逐位」

要「让 v2 的 argmax 与 v1 逐行逐位」，必须让 v2 的 **GEMV 输出**与 v1 逐位。而：

- v1 的每 lane 值是一个 **160 步的严格串行 f32 链**：
  `(((0 + t₀) + t₁) + t₂) + … + t₁₅₉`，共有 160 次独立舍入；
- v2 的每 lane 值是 **20 步**，每步把 8 项先在**子表达式内**求和后再进 `acc`
  （`acc += <4 项簇>` 两次），舍入点与结合结构都不同。

f32 加法不满足结合律 ⇒ **只要 v2 保留「每步 8 元素」的向量化载入，就不可能逐位等于 v1**；
而若把 v2 改成「每步 1 元素、跨步 32」，它**就不再是 v2**，只是 v1 的复制。
⇒ 结论：**「修 v2」这条路不存在**；正确做法是在折叠处**换用 v1 的程序**（§4）。

---

## 4. 修复 diff

### 4.1 改动清单

| 文件 | 改动 |
|---|---|
| `crates/ferrite-models/src/dsv41/chain_dev.rs` | ① `:7997` 的 UNSLICED 折调用点：`head_gemv_bf16_mrows` → **`head_gemv_bf16_v1_mrows`**（ABI 完全相同，零参数变更）+ 注释改写；② `verify_head_fold()` 文档改写（记录 v2 程序从 head 退役 + 不可能性论证）；③ `verify_head_mrows` / `verify_head_mrows_note` / `verify_head_geom` 三处仍称「v2 折是 `verify_head_fold` 的」的陈述同步修正（含一处本已过时的「两折互斥」描述）；④ `:8000` 上方「GEMV stays PER ROW on purpose」段的改点说明 |
| `kernels/cuda/dsv41_glue.cu` | `head_gemv_bf16_mrows_kernel` 头注释：把**错误的**「PRODUCTION single-row head GEMV」改为「**v2** single-row GEMV」并加 ⚠️⚠️ 段（v2 程序无 head 用户；head 要用 `dsv41_gemv_bf16_v1_mrows`）。**零 kernel 体改动** |
| `dspark_dev.rs` / 其它 kernel / device.rs | **未碰**（draft head 已正确） |

### 4.2 修复后的可达性（`grep` 收据）

```
$ grep -rn "head_gemv_bf16_mrows(" crates/ kernels/ | grep -v ://
crates/ferrite-models/src/dsv41/device.rs:4780      ← Rust wrapper 定义
kernels/cuda/dsv41_glue.cu:1770                     ← C 入口定义
kernels/cuda/tests_dsv41_head_mrows.cu:196, :346    ← 仅测试
```

⇒ **v2 折在全仓范围内已无任何生产调用点**（只剩定义 + 测试）。v1-order 折的三个调用点
（`dspark_dev.rs:3590`、`chain_dev.rs:8000`(新)、`chain_dev.rs:8078`）都是逐位安全的。

### 4.3 为什么改这一处而不是删 v2 折

- v2 折的 kernel / C 入口 / Rust wrapper / `tests_dsv41_head_mrows.cu` **保持原样**：
  它们是「v2 程序」自己的、自洽的工件（v2 折 == v2 单行，逐位可证），在
  `n < GEMV_V2_MAX_N` 的形状上仍是正确工具（MoE gate 的
  `ferrite_gemv_bf16_v2_mrows`，`ferrite_kernels.cu:3372`，就用同一个 v2 体）。
  删掉它超出本任务范围且会拆掉那条线的测试基线。
- 只把**head 的调用点**换成 v1 程序：收益（head 读一次 / m 发变 1 发）不变，
  数值变成逐位等价 ⇒ 这是「修复后 gate 可安全启用」的可实现形式。

---

## 5. 逐位论证（v1-order 折 == v1 逐行）

`gemv_bf16_v1_mrows_kernel<M>`（`dsv41_glue.cu:442-475`）对 `head_gemv_bf16_mrows_kernel`
的 5 条差异逐条归零：

| # | v2 折 | v1-order 折 | 论证 |
|---|---|---|---|
| C1 lane→k | `c = lane*8; c += 256` | `c = lane; c += 32`（v1 逐字） | 同 v1 |
| C2 行内链 | 20 × (2×4 项 FMA 簇) | `acc[r] += wv * x[r*k+c]` ×160（v1 逐字，**含**「无 `#pragma unroll`」这一选择） | 同 v1 |
| C3 归约 | `__shfl_down_sync` | `__shfl_xor_sync(off=16..1)`（v1 逐字） | 同 v1（§2.3 已证两树同构，此处更直接同指令） |
| C4 行间 | `acc[r]` 独立 | `acc[r]` 独立 | 都不跨行相加 |
| C5 K-split | 无（WPR==1） | **无**（单 warp 全 k） | 都不存在 partial fold |

唯一被 m-折允许改变的，是**权重再读**：`wv = __bfloat162float(wr[c])` 被提到 r 循环外
（per-c 值，同字节同解码，复用 M 次）。解码值不变 ⇒ 任何 row 的算术不变。
行→warp 映射与 grid **不在契约内**（row 之间独立，C4）⇒ 哪个 warp 算哪行不影响值。

`tests_dsv41_draft_parity.cu` 的 **HEAD-FOLD-v1** 臂（`:921-959`）正是这条主张的 in-process
收据：`dsv41_gemv_bf16_v1_mrows(w, x, out, m=bs, n, k)` vs `m` 次 `dsv41_gemv_bf16(...)`，
**整缓冲 raw f32 位比较**（memcmp/uint32，±0 与 NaN payload 都算差异）。

---

## 6. 编译 / 检查结果（本机）

| 项 | 命令 | 结果 |
|---|---|---|
| Rust | `cargo check -p ferrite-models` | **EXIT=0**（仅仓库既有 warning：`weights.rs:668` unused `cfg`、`chain_dev.rs:292` 未读字段等，均非本次引入） |
| Kernel 编译 | 远端 nvcc 13.2 `-gencode arch=compute_103a,code=sm_103a -O3` compile-only | 见下方「远端 nvcc 收据」 |
| 测试二进制 | `tests_dsv41_draft_parity.cu`（含 dsv41_glue.cu + ferrite_kernels.cu + dsv41_experts_mxf4.cu）、`tests_dsv41_head_mrows.cu`（含 ferrite_kernels.cu） | link 见下方收据 |

> 远端收据（`bash scripts/dsv41_compile_check.sh` 同款；本机无 nvcc、无 GPU）：

```
$ ssh ubuntu@43.202.208.136 'nvcc --version | tail -1'
Cuda compilation tools, release 13.2, V13.2.51

$ nvcc -gencode arch=compute_103a,code=sm_103a -O3 -std=c++17 -c dsv41_glue.cu -o dsv41_glue.o
GLUE_RC=0                       # 无 error（dsv41_glue.cu 只改了注释）

$ nvcc -gencode arch=compute_103a,code=sm_103a -O3 --use_fast_math -std=c++17 \
       -o t_draft_parity tests_dsv41_draft_parity.cu dsv41_glue.cu ferrite_kernels.cu dsv41_experts_mxf4.cu
PARITY_RC=0                     # parity 套件（HEAD-FOLD-v1 臂所在 TU）link 通过

$ nvcc -gencode arch=compute_103a,code=sm_103a -O3 --use_fast_math -std=c++17 \
       -o t_head_mrows tests_dsv41_head_mrows.cu ferrite_kernels.cu
HEADMROWS_RC=0                  # v2 程序族的自洽收据套件 link 通过
```

仅有仓库既有的 `warning #1308-D: taking the address of a temporary`
（`ferrite_kernels.cu:10722`，fp8 解码路径，非本次改动引入）。

---

## 7. GPU 验证手册（**双门禁**：parity 门 + 性能/accept 门）

> 三道门**必须全绿**才允许把本改动当作「已验收」。禁止在无 receipt 的情况下宣称通过。
> 建议顺序：G1（零成本、几分钟）→ G2（A/B 两次 serve）→ G3（跨 rank，最贵）。

### G1 · parity 门（kernel 级，in-process 位比较）——**必跑**

```bash
# 1) 编译 parity 套件（需 nvcc，无需 GPU；也可用 scripts/verify_mrows.sh 的远端路径）
CU=~/.local/lib/python3.10/site-packages/nvidia/cu13
$CU/bin/nvcc -I$CU/include -L$CU/lib -gencode arch=compute_103a,code=sm_103a \
     -O3 --use_fast_math -std=c++17 \
     -o /tmp/t_draft_parity kernels/cuda/tests_dsv41_draft_parity.cu \
     kernels/cuda/dsv41_glue.cu kernels/cuda/ferrite_kernels.cu kernels/cuda/dsv41_experts_mxf4.cu

# 2) 跑（需 1 张空闲 GPU；峰值 ~250 MB）
CUDA_VISIBLE_DEVICES=<free> /tmp/t_draft_parity          # 全部臂
# CUDA_VISIBLE_DEVICES=<free> /tmp/t_draft_parity --quick  # 跳过 moe/head/k2 三个重臂
```

**判据（必须逐条核对，不能只看 exit code）**：

| 检查 | 期望 | 说明 |
|---|---|---|
| `[CTL/norm-tree] OK` | ✅ | 控制臂：证明编译环境/归约树没变（**它红 ⇒ 后面全不可信**） |
| `[HEAD-FOLD-v1] OK  v1_mrows == 5 x gemv_bf16 (m=5 n=4096 k=5120)` | ✅ | **本次修复的核心收据**：v1-order 折逐位等于 5 次 per-row |
| `unwritten == 0`（head 臂内部） | ✅ | 全缓冲覆盖（NaN 哨兵测「从没写过」） |
| exit code | `0` | 至少一个臂 diff ⇒ 1，并打印首个差异的 `r/c/位型` |
| 若某臂 `SKIP` | 记录原因 | 例：`DSV41_GEMV_FP8_MODE < 3` ⇒ l4/K2 臂跳过（与本修复无关，但要在报告里写明覆盖率） |

> ⚠️ **本套件刻意不测 v2 折**（`tests_dsv41_draft_parity.cu:917-919` 明确写了
> 「v2 order is a KNOWN numerical change … it is the v2 that flips argmaxes」）。
> 本次修复后 v2 折**无 head 调用点**，所以「不测」是正确的覆盖状态；
> 若有人想给 v2 折加臂，正确形态是**期望 DIFFER 的对照臂**（记录差异而非失败），
> 而不是期望 OK 的 parity 臂。v2 折 == v2 单行程序的逐位收据在
> `tests_dsv41_head_mrows.cu`（`--quick` 跑 WPR==1 边界形状）。

### G2 · 性能 + accept 双门禁（serve 级 A/B）——**必跑**

**目的**：证明 (a) 折的**收益**仍在（`draft=` 时间），且 (b) 折**没有再动数值**
（`mean-k` 与文本 verbatim）。

```bash
# 两臂必须同机、同 prompt、同 seed、同轮数（本项目 #1 陷阱是「armed but inert」）
COMMON="CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
        DSV41_SPEC=1 DSV41_DSPARK=1 DSV41_SIDS_WRITEBACK=1 \
        DSV41_TIMING=1 DSV41_DSPARK_DEBUG=1 \
        DSV41_LAZY_VERIFY=1 DSV41_VERIFY_GRAPH=1"

# 臂 A（参考）：unsliced verify head + 折 OFF
env $COMMON DSV41_VERIFY_HEAD_SLICED=0 DSV41_VERIFY_HEAD_FOLD=0  <serve cmd>  2>&1 | tee /tmp/A.log
# 臂 B（TEST）：unsliced verify head + **本次改的 v1-order 折 ON**
env $COMMON DSV41_VERIFY_HEAD_SLICED=0 DSV41_VERIFY_HEAD_FOLD=1  <serve cmd>  2>&1 | tee /tmp/B.log
# 臂 C（可选，draft head 侧回归）：sliced（默认臂）FOLD=0 vs 默认 ON
env $COMMON DSV41_DRAFT_HEAD_FOLD=0 <serve cmd> 2>&1 | tee /tmp/C0.log
env $COMMON                          <serve cmd> 2>&1 | tee /tmp/C1.log
```

**readout（每臂都要，逐条打勾）**：

| # | 项 | 位置/命令 | 判据 |
|---|---|---|---|
| 1 | **gate 真在** | `tr '\0' '\n' < /proc/$PID/environ \| grep DSV41_VERIFY_HEAD_FOLD` | 臂 B 必须**可见** `=1`（否则测的是旧路径） |
| 2 | **符号真在**（跑之前） | `nm -D --defined-only kernels/cuda/libdsv41.so \| grep dsv41_gemv_bf16_v1_mrows` | **必须在**；缺 ⇒ `Ok(false)` ⇒ 臂 B 静默退回 per-row |
| 3 | **性能门（主判据）** | `[dspark] steps=N … draft=X.XXms verify=… commit=…` | 臂 B 的 `draft=` 应 ≈ 臂 A（head 在 verify 侧，draft 不变）；**verify** 侧应出现折的收益（一次 head 读 vs m 次，量级 ~−0.7ms/step 的对照量） |
| 4 | **accept 门（数值门）** | `[spec e2e] mean-k=` | 臂 B 必须落在**臂 A 的噪声带内**（本修复是逐位等价 ⇒ 差值应在 run-to-run 抖动内，而不是系统性偏移） |
| 5 | **文本 verbatim** | 出师表 300 tok × N≥3，md5 + `双字` 计数 | 臂 B 的 md5 与臂 A 相同、`双字=0`、无拉丁残片 |
| 6 | **回显计数（33%↔9% 的直读）** | `verify_out[0] == next` 的行占比（serve 侧统计） | 臂 B 应回到**低值**（历史 FOLD=0 为 9% 量级），**不是 33%** — 这是本修复的直接靶心 |
| 7 | 无死锁 / 无故障 | `grep -c 'ar5-hang'`；`… faults=` | =0 且正常退出 |

> **判读规则（先钉死再看数）**：
> - (3) 时间有收益 **且** (4)(5)(6) 全绿 ⇒ **修复成功**，gate 可进默认候选。
> - (3) 无收益（`draft=`/verify 都不动） ⇒ 折的 traffic 收益被别处掩盖：**不是失败**，
>   记录为「正确但性能中性」，gate 保持 OFF。
> - (4) 或 (5) 红 ⇒ **立即回退**（`git revert` 本次改动），说明 v1-order 折与
>   per-row 有未预期的差异，按 `dspark-correctness-chain.md` 的判定表处理。
> - (6) 仍是 33% ⇒ 回显**不是** head 折造成的：回到 `accept-ceiling-analysis.md` 的其他假设
>   （结构性对齐 / 输入配对），本任务到此为止。

### G3 · 跨 rank 门（可选，最贵）——**只在 G1/G2 全绿后跑**

`DSV41_VERIFY_HEAD_SLICED=1` 的 `argmax_sliced_rows` 交换（1 个 v5 round 批 m 行）与
`DSV41_VERIFY_HEAD_MROWS` 的历史组合曾触发 `ar5-hang`（`dspark-correctness-chain.md`
§「SWALLOW_STEP 的 AR v5 hang」）。跑法：

```bash
bash kernels/cuda/…  # 见 kernels/cuda/tests_dsv41_argmax_rows.cu（scripts/verify_mrows.sh --test argmax）
```

判据：`rows = 1/3/6` 的 epoch 增量**恒为 1**；每行 == 生产全词表 `dsv41_argmax_sliced`；
`grep -c 'ar5-hang'` = 0。**本改动不触碰 argmax 族**，此门是回归闸而非本修复的靶心。

---

## 8. 遗留与风险（诚实边界）

1. **`tests_dsv41_head_mrows.cu` 的头注释仍称「the production single-row program is NOT
   v1's … it is `gemv_bf16_nt_kernel`'s per-token body at WPR == 1」**——按 §1 的
   `gemv_bf16_v2_wanted` 闸，这句话对 head 形状**不成立**（head 的 `n` ≥ 16160）。
   该文件的 arm 语义（mrows 折 == v2 单行 + == nt 批量）**仍然正确且有用**（它给的是
   「v2 程序族内部自洽」的收据），只有「production」这个定语误导。
   **未改**（测试文件属 tester 域，避免冲突）；建议由 tester 把 s/production/v2/。
2. **`head_gemv_bf16_mrows` 的两个 ABI 完全相同的孪生入口**
   （`device.rs:4819` 自己写了「a caller must know which program it is claiming parity with」）
   是本次事故的结构性成因：换错一个函数名 = 静默换数值。本修复把调用点收敛到 v1，但
   **同名同 ABI 的双胞胎仍在**。若要根治，可在 C 入口加「v2 程序只接受 `n < 2048`」的域闸
   （拒绝时返回 `cudaErrorNotSupported`，Rust 侧 `Ok(false)` → 调用方退回 per-row）；
   本次**未做**（超出任务范围，且会让 `DSV41_VERIFY_HEAD_FOLD` 变成结构性死 gate）。
3. **accept 不会因此变好**（已实测中性，`dspark-correctness-chain.md:1736`）。
   本修复的收益是**正确性/一致性**：draft、verify、eager 三条 head 路径从此跑**同一个程序**，
   近 tie argmax 不再有程序级漂移。
4. **本机无 nvcc、无 GPU**：§6 的远端 nvcc 只到 compile-only；G1-G3 均需在有空闲 GPU 的
   机器上执行（本任务明确禁止 GPU/e2e）。

---

## 9. 参考

| 主题 | 位置 |
|---|---|
| v1 逐行程序 | `kernels/cuda/dsv41_glue.cu:370-395`（`gemv_bf16_kernel`） |
| v1-order 折（**本修复用的程序**） | `kernels/cuda/dsv41_glue.cu:442-475` + launcher `:1624` |
| v2/`nt` 程序（WPR==1 体） | `kernels/cuda/ferrite_kernels.cu:3455-3536`；单行版 `:3235-3298` |
| v2 折（**已无 head 调用点**） | `kernels/cuda/dsv41_glue.cu:864` + launcher `:1770` |
| v2 分派闸 | `crates/ferrite-models/src/dsv41/device.rs:7929` / `:7951` / `:4612` |
| draft head 的 v1 修复（历史） | commit `e9fcf2a`；`dspark_dev.rs:3588-3600` |
| parity 套件 HEAD-FOLD-v1 臂 | `kernels/cuda/tests_dsv41_draft_parity.cu:921-959` |
| v2 程序族自洽收据 | `kernels/cuda/tests_dsv41_head_mrows.cu` |
| 33%↔9% 的原始口径 | `chain_dev.rs` `verify_head_fold`（改写前）；`docs/agent/dspark-correctness-chain.md` |
| accept 中性实测 | `docs/agent/dspark-correctness-chain.md:1736` |
| 「这操作符把 save/restore…」（无关）/ 总设计 | `docs/agent/verify-family-fusion.md` §7 / `[H1]` |
