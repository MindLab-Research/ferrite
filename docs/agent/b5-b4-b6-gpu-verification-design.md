# B5 / B4 / B6（mrows Phase B）GPU 验证设计

> 工部（ministry-works）· 2026-09-12 · **只读 + 设计**。未执行 GPU 命令、未改动任何源码（本文件是唯一产出）。
> 代码基线：HEAD `2bafc60`（`crates/ferrite-models/src/dsv41/{chain_dev.rs,device.rs,dspark_dev.rs}`、
> `kernels/cuda/ferrite_kernels.cu`、`kernels/cuda/dsv41_kernels.cu`、`scripts/batched_400_v2.sh`）。
> 上位文档：`docs/agent/mrows-swallow-batched-implementation-design.md`（§3/§7/§8）、
> `docs/agent/b6-mrows-f32-design.md`（§5/§6）。

---

## 0. 判词（先读这五条）

1. **三个 gate 名全部确认，且三个都是「新增核」的 opt-in 臂**（默认 OFF，严格 `== "1"`）。
   与任务表给出的两个 `?` 对应：B5 = `DSV41_GATE_MROWS_ROUTE`、B4 = `DSV41_RMSNORM_ROPE_MROWS`、
   B6 = `DSV41_VERIFY_WOB_MROWS_F32`（B6 的 gate 名**同时**驱动 verify 与 draft 两个站点）。
2. **`scripts/batched_400_v2.sh` 有两个必须修的缺口**，否则「臂 1（b2+b3 基线）」根本无法起、「臂 3（B4）」
   会违反它自己写明的红线（详见 §2）。这是本次设计里**最需要上报**的两点。
3. **B4 的 A/B 必须把 `DSV41_VERIFY_FORK=1` 纳入臂内**（该 gate 默认 OFF，且**从未在 SWALLOW 上测过**）。
   因此「+B4」这一臂天然带进一个「FORK」变量 ⇒ 4 臂线性阶梯**不成立**，需要一个 FORK 控制臂（§3）。
4. **预期值要下修**：任务表的 B5 `-0.3~0.6ms` / B4 `-0.2~0.4ms` 是**旧口径**。
   `GATE_MROWS`（＝fold gate）与 `NORM_MROWS` 已在基线矩阵里，B5/B4 各只剩 **−1 发/层 = −40 发/步
   = −0.13ms @3.3µs**（设计 §附-4 已记录）。B6 的 `-0.6~1.0ms` 与设计一致（−0.79ms @3.3µs）。
5. **B5 的 nsys「核名」证据不可用**：它实例化的是与 `DSV41_GATE_MROWS` **同一个** `gemv_bf16_nt_kernel<NT,WPR>`
   ⇒ 设备核名与 plain fold **逐字相同**，「上了场」的唯一硬证变成 **`dsv41_route_topk` 计数下降**（§4.3）。
   B4/B6 有独立核名，不受此影响。

---

## 1. Gate 名确认（读码，非文档转述）

| 项 | gate 名 | 读点 | 判据 | 默认 | 前置依赖 |
|---|---|---|---|---|---|
| **B5** | `DSV41_GATE_MROWS_ROUTE` | `chain_dev.rs:1362-1367`（`gate_mrows_route()`） | 严格 `== "1"` | **OFF** | **必须** `row_fold_gate()` 为真：`DSV41_ROW_FOLD_GATE` ≠ `"0"` **或** `DSV41_GATE_MROWS` ≠ `"0"`（`chain_dev.rs:1337-1343`）。**基线矩阵已含 `DSV41_GATE_MROWS=1`** ⇒ 增量只需 `DSV41_GATE_MROWS_ROUTE=1` |
| **B4** | `DSV41_RMSNORM_ROPE_MROWS` | `chain_dev.rs:1311-1316`（`rmsnorm_rope_mrows()`） | 严格 `== "1"` | **OFF** | **必须**同臂 `DSV41_VERIFY_FORK` 为真（`chain_dev.rs:1173-1176`，`!= "0"`，unset⇒**false**）。替代的两发是 FORK 的 `*_on` 侧链（`chain_dev.rs:1309`、`11038-11040`），只有 FORK=1 才走到那条 stream |
| **B6** | `DSV41_VERIFY_WOB_MROWS_F32` | verify：`chain_dev.rs:4981-4986`（`Self::wob_mrows_f32()`）<br>draft：`dspark_dev.rs:173-184`（同名 fn） | 严格 `== "1"` | **OFF** | 无 gate 依赖；运行期还需 `dev.supports_gemm_fp8_mrows_f32()`（符号存在）**且** `mrows`（shape）为真，否则静默 `Ok(false)` 回落 |

**接线点（供 A/B 归因）**

| 项 | 接线点 | 替换掉什么 |
|---|---|---|
| B5 | `moe_rows` 的 `gate_folded` 块（`chain_dev.rs:13115-13155`）→ `Device::gemv_bf16_v2_mrows_route` | `gemv_bf16_v2_mrows`(1) + `dsv41_route_topk`(1) → **1 发/层** |
| B4 | `attention_rows` 的 kv 半链（`chain_dev.rs:11032-11060`）→ `Device::rmsnorm_rope_mrows(..., kv_stream)` | `norm_rows_on`(1) + `apply_rope_on`(1) → **1 发/层** |
| B6 | verify：`attention_rows`（`chain_dev.rs:11738-11755`）→ `Device::gemm_fp8_mrows_f32(a_stride=ol_total)`；draft：`draft_attn_out`（`dspark_dev.rs:2194-2211`）→ 同核（`a_stride=ol_total`） | verify `m×quant_fp8` + `proj_mrows` → **1 发/层**；draft 省掉 `quant1(wo)` 那一发 |

> **一处易错**：B5 的调用点写成 `if row_fold_gate() { if gate_mrows_route() {…} }`，所以
> **只设 `DSV41_GATE_MROWS_ROUTE=1` 而把 fold gate 关掉（`DSV41_GATE_MROWS=0`）会整段不执行**——
> 这正是本仓 #1 陷阱「gate 设了但路径没变」。基线矩阵已带 `GATE_MROWS=1`，别在臂里覆盖它。

---

## 2. 脚本缺口（**必须上报 / 修**）

`scripts/batched_400_v2.sh` 在 HEAD 上有两个 opt-in arm，但**都表达不了本次要跑的臂**：

### 缺口 A — `B400_MROWS_A` 是单值，无法表达「b2+b3」

```bash
# scripts/batched_400_v2.sh:274-281（现状）
MROWS_A="${B400_MROWS_A:-}"
case "$MROWS_A" in
    "") ;;
    b2|B2) GATES_ONLINE… DSV41_ATTN_MROWS_ROPE_NORM=1 ;;
    b3|B3) …          DSV41_ATTN_MROWS2=1 ;;
    b1|B1) …          DSV41_VERIFY_ROPE_MROWS=1 ;;
    *) echo "FATAL: … not a Phase A arm (b1|b2|b3)"; exit 2 ;;
esac
```

任务要求「臂 1 = SWALLOW + mrows **b2+b3**」（63.8 基线），但 63.8 是
`DSV41_ATTN_MROWS_ROPE_NORM=1` **且** `DSV41_ATTN_MROWS2=1`（`dspark-correctness-chain.md:4699`），
而 `case` 只接受单值、多值会 `exit 2`。⇒ **必须加一个组合 arm**：

```bash
# 建议补丁（工部不改码，仅给出需应用的 diff）
    bc|B2B3) GATES_ONELINE="$GATES_ONELINE DSV41_ATTN_MROWS_ROPE_NORM=1 DSV41_ATTN_MROWS2=1" ;;
```

### 缺口 B — `B400_B4` 漏了它自己写明的 `DSV41_VERIFY_FORK=1`

```bash
# scripts/batched_400_v2.sh:392-395（现状）
B4_ARM="${B400_B4:-0}"
if [ "$B4_ARM" = 1 ]; then
    GATES_ONELINE="$GATES_ONELINE DSV41_RMSNORM_ROPE_MROWS=1"
fi
```

而同文件 `:380-382` 的注释明写：

> ⚠️ A/B MUST run with `DSV41_VERIFY_FORK` in the SAME arm: … only a FORK=1 run exercises the stream
> the new kernel must ride.

`verify_fork()` 默认 **OFF**（unset⇒false），所以这个 arm 跑起来**恰恰不满足自己的红线**：
它只测了「B4 在主流上融两发」（数值对，但 stream 路径没被碰）。⇒ 修正：

```bash
    GATES_ONELINE="$GATES_ONELINE DSV41_VERIFY_WOB_MROWS_F32=1"   # 这是 B6（:352-356）
    # …
    GATES_ONELINE="$GATES_ONELINE DSV41_RMSNORM_ROPE_MROWS=1 DSV41_VERIFY_FORK=1"   # B4 修正
```

并建议**新增一个 FORK-only arm**（做控制，见 §3）：

```bash
B400_FORK="${B400_FORK:-0}"
if [ "$B400_FORK" = 1 ]; then
    GATES_ONELINE="$GATES_ONELINE DSV41_VERIFY_FORK=1"
fi
```

### 缺口 C（无害，记录）— `B400_B5` 重复设 `DSV41_GATE_MROWS`

`B400_B5=1` 追加 `DSV41_GATE_MROWS=1 DSV41_GATE_MROWS_ROUTE=1`（`:388-391`）。`GATE_MROWS` 已在基线
矩阵（`:158`）里，重复设置**幂等无害**，保留即可（它保证即使有人 `=0` export 也回到 ON）。

---

## 3. 臂设计（一臂一 serve，交错 A/B）

**关键约束**：B4 的臂必须带 `DSV41_VERIFY_FORK=1`（§1/§2B），而 FORK 默认 OFF 且未在 SWALLOW 上测过
⇒ 若要「干净归因」，FORK 必须**在同一阶梯里被控制**。给出两套方案：

### 方案 A（**推荐**，5 个 serve）——带 FORK 控制臂，归因最干净

| 臂 | tag | = 基线矩阵 + 下列 env | Δ 归因 | 期望 |
|---|---|---|---|---|
| **A0** | `a0_base` | `DSV41_ATTN_MROWS_ROPE_NORM=1 DSV41_ATTN_MROWS2=1` | — | **63.8**（复现基线） |
| **A1** | `a1_b5` | A0 + `DSV41_GATE_MROWS_ROUTE=1` | **B5 = A1−A0** | −0.13ms（@3.3µs） |
| **A2** | `a2_fork` | A1 + `DSV41_VERIFY_FORK=1` | FORK（控制，非本次目标） | ~中性 |
| **A3** | `a3_b4` | A2 + `DSV41_RMSNORM_ROPE_MROWS=1` | **B4 = A3−A2** | −0.13ms |
| **A4** | `a4_b6` | A3 + `DSV41_VERIFY_WOB_MROWS_F32=1` | **B6 = A4−A3** | −0.79ms |

### 方案 B（备选，4 个 serve）——FORK 常开，牺牲 63.8 复现

| 臂 | tag | env | Δ |
|---|---|---|---|
| B0 | `b0` | b2+b3 **+ `DSV41_VERIFY_FORK=1`** | 新基线（≈63.8+FORK） |
| B1 | `b1` | B0 + `DSV41_GATE_MROWS_ROUTE=1` | B5 |
| B2 | `b2` | B1 + `DSV41_RMSNORM_ROPE_MROWS=1` | B4 |
| B3 | `b3` | B2 + `DSV41_VERIFY_WOB_MROWS_F32=1` | B6 |

> 选 **A**：B5 的 delta（A1−A0）不受 FORK 影响，B4/B6 也各自有直接对照；且顺带量出 FORK 在 SWALLOW 上的
> 真实代价/收益（这是 400 路线的一个未知量，handover 里 FORK 只在 lazy 上有 ~+1–2% 的旧猜测）。
> 若 GPU 时间只够 4 臂，退 **B**，但 A0 的 63.8 复现就没了，**且不能声称 B4 的 stream 红线已验证**。

### 逐臂命令（应用 §2 补丁后）

```bash
# 臂 A0 —— 复现 63.8 基线（counting prompt，见 §4.5）
PROMPT='请从 1 数到 100，每个数字单独占一行，只输出数字本身，不要任何解释。' MAXTOK=80 \
  B400_MROWS_A=bc bash scripts/batched_400_v2.sh

# 臂 A1 —— +B5
PROMPT='请从 1 数到 100，…' MAXTOK=80 \
  B400_MROWS_A=bc B400_B5=1 bash scripts/batched_400_v2.sh

# 臂 A2 —— +FORK（控制）
… B400_MROWS_A=bc B400_B5=1 B400_FORK=1 bash scripts/batched_400_v2.sh

# 臂 A3 —— +B4
… B400_MROWS_A=bc B400_B5=1 B400_FORK=1 B400_B4=1 bash scripts/batched_400_v2.sh

# 臂 A4 —— +B6（全 mrows 栈）
… B400_MROWS_A=bc B400_B5=1 B400_FORK=1 B400_B4=1 B400_B6=1 bash scripts/batched_400_v2.sh
```

> **每臂必须各跑两遍 prompt**：`counting`（量吞吐 + 前 61 行）与 `出师表`（零拉丁红线）。
> 脚本一次只有一 prompt、且 `[dspark] steps=` 累加器跨请求不清零 ⇒ 换 prompt 必须换 serve（重跑该臂）。
> `出师表` 用默认 `PROMPT`、`MAXTOK=1000`；`counting` 用上面 80 tok 的短跑。
> 交错纪律：`A0 A1 A0 A1 …` 至少两轮取中位，抵消热漂（设计 §5）。

---

## 4. 每臂验证判据

### 4.1 V1 — env 实读（防「设了没生效」，本仓 #1 陷阱）
脚本已做：`$LOGDIR/<tag>.env` 是 `/proc/<pid>/environ` 的 `DSV41_*` dump（`:741-743`）。逐臂核对本臂新增的
gate 恰好在列，且 `FORBIDDEN`（`DSV41_LAZY_VERIFY`）不在列。
```bash
grep -E 'GATE_MROWS_ROUTE|RMSNORM_ROPE_MROWS|VERIFY_WOB_MROWS_F32|VERIFY_FORK|ATTN_MROWS2|ATTN_MROWS_ROPE_NORM' \
     /tmp/batched_400_v2/<tag>.env
```

### 4.2 V2 — 符号存在性（**build 后、serve 前**，一次覆盖三核）
三个新 C 入口都编进**同一个** `libferrite_kernels.so`（`build.sh:14-17`：`ferrite_kernels.cu` +
`dsv41_kernels.cu` + `dsv41_route.cu` … 同一 `-o`）：
```bash
SO=kernels/cuda/libferrite_kernels.so
for s in ferrite_gemv_bf16_v2_mrows_route dsv41_rmsnorm_rope_mrows dsv41_gemm_fp8_mrows_f32; do
    printf '%-40s %s\n' "$s" "$(nm -D $SO | grep -c "$s")"   # 期望均 ≥1
done
```
`=0` ⇒ 该臂**静默 inert**（`Ok(false)` 回落），A/B 会读成「无效果」——即 phantom-gate 陷阱。

### 4.3 V3 — 上场证据（nsys sum 表；**这是唯一硬证**）

| 项 | 设备核名 | 判据 | 备注 |
|---|---|---|---|
| **B5** | ⚠️ **`gemv_bf16_nt_kernel<NT,WPR>`** | **`dsv41_route_topk` Instances/步 比 A0 **少 40**（一层的 moe_rows 那一路消失）。若 draft 侧也调 route_topk（`dspark_dev.rs:2345`），则**不是归零**而是「-40」——以此为准 | **核名与 plain fold 相同**（`ferrite_kernels.cu:3676-3682`/`:3753-3761`），**核名不可区分**；route_topk 计数是唯一硬证。次级信号：融合发 `smem≈out_f*2*4+topk*4`（n=384,topk=6 ⇒ 3096B）vs plain fold 的 `smem=0` |
| **B4** | `dsv41_rmsnorm_rope_mrows_kernel` | 出现该核名；且 kv 侧 `apply_rope*` Instances/步 **-40**（40→0，若 kv rope 只在 verify 走） | FORK=1 臂下同在 `kv_stream` 侧链上 |
| **B6** | `gemm_fp8_mrows_f32_kernel<M>` | 出现该核名；且 verify wo_b 的 `dsv41_quant_fp8`（或 `quant1`）调用数**归零** | draft 侧同时消失 1 发 `quant1(wo)` |

> 采集：`nsys profile -o <tag> … ferrite-serve …` + `nsys stats --report cuda_gpu_kern_sum`（或既有
> `swallow-nsys-batched-analysis-framework` 的汇总脚本）。**decline 是静默的**：没有上述计数变化就不能说「已上场」。

### 4.4 V4 — 步时
`steady_median`（丢前 20 步，`STEADY_SKIP=20`）为主判据，`verify_ms`（`[dspark]` 行）为辅。
双门（设计 §5）：保守止损 `|Δ| ≥ 0.8ms`（否则判 instruction-bound，转下一项）；期望窗
`|Δ| ∈ [launch 账×60%, launch 账]`。
- **A1−A0（B5）**：期望 ≈ −0.13ms；**不要把任务表的 −0.3~0.6ms 当门**（旧口径，见 §0-4）。
- **A3−A2（B4）**：期望 ≈ −0.13ms。
- **A4−A3（B6）**：期望 ≈ −0.79ms（@3.3µs）～−1.49ms（@6.2µs），**这是三项里最大**。

### 4.5 V5 — 正确性红线（每臂、每 prompt 都要）
1. **counting，前 61 行**：`PROMPT`=「请从 1 数到 100…」、`MAXTOK=80`。脚本写的 `run.txt` 逐行核对：
   ```bash
   python3 - <<'PY'
   L=[l.strip() for l in open('/tmp/batched_400_v2/run.txt',encoding='utf-8',errors='ignore') if l.strip()]
   print('lines=',len(L),'first_bad=',next((i+1 for i,l in enumerate(L[:61]) if l!=str(i+1)),None))
   PY
   ```
   期望 `first_bad=None` 且前 61 行恰为 `1..61`。
   ⚠️ **counting 跑时脚本自身会报 `REDLINE FAIL: missing 先帝创业未半`（rc=1）——这是预期的**（红线脚本是给出师表设计的），
   不要当成失败；counting 的判据是上面这段 `first_bad`。
2. **出师表，零拉丁**：`MAXTOK=1000` 默认 prompt。判据 = `latin==0` **且** `dbl==0` **且** `先帝创业未半=yes`
   **且** 非空（脚本 `:866-877` 已自动判定并 rc=1/0）。
3. **无 panic / 无 hang**：`grep -ciE 'panic|CUDA error|illegal memory|misaligned' <tag>.log` == 0；
   请求必须在 `REQ_TIMEOUT=1800s` 内返回（stderr/`resp_err` 为空）。

### 4.6 V6 — 逐位 / 直方图（B5、B4 适用）
B5/B4 声称**逐位等价**（设计 §8）⇒ 要求 `k_acc` 序列/直方图 `hist0..6` 与对照臂一致
（脚本已写 `<tag>.metrics` 的 `hist0..6`/`kacc_mean`），且 **`verify_graph` 的 shapes/captures/replays 不劣化**。
B6 **非位等价**（跳过 fp8 往返，严格更准）⇒ V6 对 B6 只要求 `k_acc` **直方图 mode 不降**，改由 V5 + V3 的 quant 计数判据兜。

---

## 5. 双产物重编（**.cu 变了 ⇒ 必须全链路重编**）

**为什么必须**：B5 动了 `kernels/cuda/ferrite_kernels.cu`；B4/B6 动了 `kernels/cuda/dsv41_kernels.cu`；
且三者都动了 Rust FFI/gate 接线（`device.rs` + `chain_dev.rs` + `dspark_dev.rs`）。
只重编 `.so` 或只 `cargo build` 任意一个，都会得到**不同源的产物对**（本仓的 `.build_id` 门会拦，但拦之前先别踩）。

**唯一正确顺序**（`batched_400_v2.sh` 在 `BUILD=1`（默认）时已自动做，并在 build 后**证明同源**）：
```bash
# 1) .so：把 ferrite_kernels.cu + dsv41_kernels.cu + dsv41_route.cu + … 编成一个 .so
bash kernels/cuda/build.sh 103a           # 成功判据是日志里的 "built … for sm_103a"
# 2) 二进制：.build_id 由 build.rs 烘进二进制 ⇒ 必须先 touch build.rs 再 cargo（否则增量不重跑 build.rs）
touch crates/ferrite-kernel/build.rs && cargo build --release
# 3) 同源证明（脚本 :465-488 已做）：strings target/release/ferrite-serve | grep -F "$(cat kernels/cuda/.build_id)"
```
脚本内的实现见 `:436-488`（`build_so.log`/`build_bin.log` + `.build_id` 嵌入检查 `PAIR_OK`）。
**任何 `--no-build` 的臂只允许在「刚刚 build 过的同一 HEAD、且未改过源码」时使用**，否则 A/B 测的是旧路径。

> 注：`kernels/cuda/build.sh` 最后一句是 `[ ${#SKELETON_FLAGS[@]} -gt 0 ] && echo …`，无 skeleton flags 时
> 会在 `set -e` 下 **exit 1 即使编译成功**——判据用日志行 `built …`，不要用 rc（脚本 `:440-448` 已注明）。

---

## 6. parity 测试（可选，**建议 serve 之前先跑**，几分钟、不占模型）

三个 standalone 位等价套件都是 nvcc 编译、**单卡**运行，与 serve 无关，可插在重编之后、起 serve 之前。
（**须先确认 `nvcc` 可用 + 至少一张空卡**；三套件都对 raw-bit 整块比对 + qNaN sentinel。）

```bash
# B5 —— 融合 vs gemv_bf16_v2_mrows + dsv41_route_topk（scores/weights/indices 三块逐位）
nvcc -gencode arch=compute_103a,code=sm_103a -O3 --use_fast_math -std=c++17 \
  -o /tmp/t_mrows_route kernels/cuda/tests_mrows_route.cu \
     kernels/cuda/ferrite_kernels.cu kernels/cuda/dsv41_route.cu
# B4 —— 融合 vs dsv41_rmsnorm_rows + dsv41_apply_rope（含非 0 pos_base / rope_off / inverse=1）
nvcc -gencode arch=compute_103a,code=sm_103a -O3 --use_fast_math -std=c++17 \
  -o /tmp/t_rmsnorm_rope_mrows kernels/cuda/tests_rmsnorm_rope_mrows.cu kernels/cuda/dsv41_kernels.cu
# B6 —— m 行核第 r 行 vs M=1 dsv41_gemm_fp8_mx_f32 第 r 行（逐位）+ decline 表 + 形状矩阵
nvcc -gencode arch=compute_103a,code=sm_103a -O3 --use_fast_math -std=c++17 \
  -o /tmp/t_gemm_mrows_f32 kernels/cuda/tests_dsv41_gemm_mrows_f32.cu kernels/cuda/dsv41_kernels.cu

CUDA_VISIBLE_DEVICES=<free> /tmp/t_mrows_route
CUDA_VISIBLE_DEVICES=<free> /tmp/t_rmsnorm_rope_mrows
CUDA_VISIBLE_DEVICES=<free> /tmp/t_gemm_mrows_f32            # 或 --quick（小形状）
DSV41_GEMV_A32=1 CUDA_VISIBLE_DEVICES=<free> /tmp/t_gemm_mrows_f32   # B6 arm 5，第二半（必须也绿）
```

**注意 B5 套件的编译不同于其它**：它把 `ferrite_kernels.cu` 与 `dsv41_route.cu` 作为**额外 TU** 链接
（被测入口在别的 TU 里）——照 `tests_mrows_route.cu:36-42` 的头注。B4/B6 只链 `dsv41_kernels.cu`。

---

## 7. 执行清单（工部不做，交调用者跑）

1. `cargo check --workspace` + `bash -n scripts/batched_400_v2.sh`（改脚本后）——做零成本预检。
2. **应用 §2 补丁**（组合 arm `bc` + B4 补 `VERIFY_FORK` + `B400_FORK` 控制臂）。
3. `bash kernels/cuda/build.sh 103a` + `touch crates/ferrite-kernel/build.rs && cargo build --release`
   + **§4.2 符号检查**（三符号 ≥1）。**失败即停**（否则全程 phantom-gate）。
4. **§6 parity 三套件**（可选但推荐，先于 serve）。
5. 起臂：§3 方案 A 的 A0→A4，每臂 **counting + 出师表 两个 serve**，交错至少两轮。
6. 每臂收 `V1–V6`（env / nsys 计数 / steady_median / 红线 / k_acc 直方图 / panic）。
7. 判定（**逐项独立**，不打包）：
   - B5 兑现 ⇐ A1−A0 落在期望窗 **且** `route_topk` -40/步 **且** 红线过。
   - B4 兑现 ⇐ A3−A2 落在期望窗 **且** 核名出现 + kv rope -40/步 **且** 红线过（FORK=1 臂）。
   - B6 兑现 ⇐ A4−A3 落在期望窗 **且** 核名出现 + wo_b quant 归零 **且** 红线过 + `k_acc` mode 不降。
   - 任一项 `Δ>0`（变慢）⇒ 先查 §1 各接线的陷阱（B5 的 fold gate 是否被关、B4 的 `kv_stream`、B6 的 `a_stride`），
     再怀疑测量。

---

*工部 · 本文件为**设计**产出；本段无 GPU 命令执行、未改动任何源码。gate 名、依赖、接线点、脚本缺口均以 HEAD `2bafc60` 的代码为准，非文档转述。*
