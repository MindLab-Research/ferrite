# C4 head A/B 手册 —— H3 `DSV41_HEAD_TILELANG` / H1 `DSV41_HEAD_MTILE`

> 工部 · 2026-09-13 · **纯代码分析，本机 `cargo check --workspace --all-targets` EXIT=0**；
> 全部 GPU 操作（AOT 生成 / build.sh / serve / nsys）留给主 agent。
> 依据：post-fp4-roadmap 的 C4 行（全表最高 ROI）、`docs/agent/mtile-woa-head-design.md`
> （H1 的完整设计 + §5 GPU 手册）、`docs/agent/tilelang-attn-head.md`（H3 原型）、
> `kernels/cuda/tilelang_gen/head_bf16_shim.cu`（H3 shim）。
> 交付物：H1 的 1 处 Rust 接线（§2）+ 3 臂双门禁命令清单（§4）。

---

## §0 一句话

C4 的两个臂今天都**不需要新的 C/kernel 工作**：H3（TileLang M-in-tile，2.93× 但欠
2.3e-2 数值债）早已三件套齐备，只差 AOT 生成物进 `.so` 并由 acc 门背书；H1
（`gemv_bf16_v1_mtile_kernel`，**位级相同、零数值债**）kernel + launcher 早已在
`.cu` 里，缺的只是 Rust 侧一个调用点的 arming gate —— 本交付补这一处（§2）。
H3 是「赢的赌注」，H1 是「不输的兜底」。

---

## §1 H3 就绪确认（逐件 file:line）

| 件 | 位置 | 状态 |
|---|---|---|
| gate `head_tilelang()`（strict `== "1"`） | `crates/ferrite-models/src/dsv41/chain_dev.rs:2442` | ✅ 就位 |
| 一次性 decline 报告 `head_tilelang_note(m, seg)` | `chain_dev.rs:2466` | ✅ 就位（`OnceLock`） |
| `Kernels::head_bf16_tilelang` 字段 | `device.rs:687` | ✅ 就位 |
| `ko!(rt, "dsv41_head_bf16_tilelang")` 注册 | `device.rs:2213` | ✅ 就位（可选符号） |
| device wrapper `Device::head_bf16_tilelang`（`rc == 2` ⇒ `Ok(false)` 回退） | `device.rs:5492` | ✅ 就位 |
| `supports_head_tilelang()`（.so 有无符号） | `device.rs:5523` | ✅ 就位 |
| **SLICED 调用点**（最高优先级，先于 fold/逐行） | `chain_dev.rs:9119-9128` | ✅ 就位 |
| **UNSLICED 调用点**（刻意保留：避免结构死门） | `chain_dev.rs:9201-9210` | ✅ 就位 |
| C 导出 `dsv41_head_bf16_tilelang` | `kernels/cuda/tilelang_gen/head_bf16_shim.cu:234` | ✅ 就位 |
| 形状门（n==16160 && k==5120 && m∈1..=8 && 16B 对齐） | `head_bf16_shim.cu:236-243` | ✅ 就位 |
| capture guard（P0-2 红线：capturing 且 INIT 未完 ⇒ `return 2`） | `head_bf16_shim.cu:250-253` | ✅ 就位 |
| INIT（`cudaFuncSetAttribute` + 常驻 scratch，瞬态 OOM 不闩死） | `head_bf16_shim.cu:164-201` | ✅ 就位 |
| cast（f32→bf16 staging，行 ≥ m 写 0） | `head_bf16_shim.cu:219` | ✅ 就位 —— **数值债的载体** |
| 活性回执 `[head-tilelang] ARMED …` | `head_bf16_shim.cu:262-270` | ✅ 就位 |

⇒ **结论：H3 的 Rust 侧 0 代码需求成立**，`DSV41_HEAD_TILELANG=1` 即可 arm
（`chain_dev.rs:9119`），无需任何改动。

### 1.1 ⚠️ 但 H3 有一个**非代码**前置（主 agent 必须先做）

`head_bf16_shim.cu` 的全部内容（含导出符号）都在 `__has_include` 守卫内
（`head_bf16_shim.cu:102-108`），而本机树上**生成物缺席**：

```
kernels/cuda/tilelang_gen/head_partial_tl.cu   不存在
kernels/cuda/tilelang_gen/head_reduce_tl.cu    不存在
kernels/cuda/tilelang_gen/head_tl_config.txt   不存在
```

生成物缺席 ⇒ 该 TU 编译成空单元 ⇒ `.so` 里**没有** `dsv41_head_bf16_tilelang`
⇒ `Device::head_bf16_tilelang` 拿不到符号、返回 `Ok(false)` ⇒ 逐行/fold 应声，
**而 Rust 会打 `head_tilelang_note` 警告**（这正是设计好的「绝不静默测老路」）。
⇒ **Arm 2 的启动前检查**（§4.0）是硬前置：先 `gen_head_aot.py` AOT 生成 +
`build.sh` 重建 `.so`，再 `nm -D` 确认符号存在。**没确认就跑 = 用一整轮 serve 测了
老路径。**

---

## §2 H1 接线（本交付的唯一代码改动）

### 2.1 C 入口存在性确认（**不用改 .cu**）

| 件 | 位置 | 说明 |
|---|---|---|
| kernel `gemv_bf16_v1_mtile_kernel<M, BN_MAX=4>` | `dsv41_glue.cu:551-602` | v1 序 verbatim（含**刻意不加** `#pragma unroll`）、`acc[r][nn]` 独立链、同 `shfl_xor` 树、无 K-split、无跨行合并 |
| launcher `dsv41_gemv_bf16_v1_mrows` | `dsv41_glue.cu:1666` | **同一个导出符号**，内部按 env 分派 |
| `DSV41_HEAD_MTILE` / `_BN` 读取（`atoi != 0` / 1..4 默认 2） | `dsv41_glue.cu:1691-1703` | 函数内 `static`（**一进程一值**） |
| armed 分支（`grid = ceil(n/(nwarp*bn))`, `nwarp=8`） | `dsv41_glue.cu:1704-1730` | 优先级 **MTILE > ①HEAD_ACT_F32VEC > v1_mrows**，被遮蔽的臂在回执里点名 |
| 活性回执 `[head-mtile] ARMED m=… n=… k=… bn=… nwarp=… -> grid=…` | `dsv41_glue.cu:1712-1717` | **A/B 是否有效的唯一证据** |

⇒ H1 **不新增 extern "C" 符号、不改 ABI**（与 `mtile-woa-head-design.md` §5.1
一致）⇒ **device.rs 无需任何改动**：`gemv_bf16_v1_mrows` 字段（`device.rs:659`）、
`ko!` 注册（`device.rs:2212`）、wrapper `Device::head_gemv_bf16_v1_mrows`
（`device.rs:5442`）**早已就位**。任务书里的「device.rs：`Option<fn>` +
`ko!()` + wrapper」在本案是**空操作**：重复注册同一个符号只会是噪声。

### 2.2 Rust 接线（`crates/ferrite-models/src/dsv41/chain_dev.rs`，+129/-6）

```
+ fn verify_head_mtile() -> bool            // :2532  读 DSV41_HEAD_MTILE，strict == "1"
+ fn head_mtile_note(arm_sliced, head_bf16, performed, m)   // :2558  一次性三态报告
  SLICED 调用点   :9133-9136   let armed_mrows / let armed_mtile
                              mrows = !tl_ok && (armed_mrows || armed_mtile) && dev.head_gemv_bf16_v1_mrows(...)
  SLICED note     :9157-9162   armed_mrows → verify_head_mrows_note(...)
                              armed_mtile → head_mtile_note(true, true, false, m)
  UNSLICED 调用点 :9211-9215   同上（`(armed_mrows || armed_mtile)`）
  UNSLICED note   :9231-9236   armed_mtile → head_mtile_note(false, head.dtype == "BF16", mrows, m)
```

**为什么必须有这个 gate（而不是只靠 .cu 自己的 env）。** arm 的载体是 fold 的调用
点，而那个调用点今天由 `DSV41_VERIFY_HEAD_MROWS` 把守 ⇒ 只导出
`DSV41_HEAD_MTILE` 时 launcher **根本到不了**，整轮 serve 会静默测逐行 head（本项目
#1 测量偏置陷阱，也是 `verify_head_mrows` 被修过的同一个坑）。本 gate 就是那个
调用点的独立 arming 开关。

**语义纪律（三条，都在 docstring 里写明）。**

1. **无数值债**：只换「哪个 warp 算哪个元素」；k 走序 verbatim、`xv[r]` hoist 读同一
   地址同一字节、每元素一棵同序 `shfl_xor` 树、无 K-split ⇒ 逐位相同。**A/B 里
   `mean-k` 应完全相等——任何 delta 说明实现有 bug，不是噪声。**
2. **strict `== "1"`**（照 `DSV41_VERIFY_HEAD_MROWS` / `DSV41_HEAD_TILELANG` 的惯例：
   空值/拼错不得 arm 一个实验）。⚠️ C 侧读的是 `atoi != 0`，所以 `"1"` 是两侧
   **唯一约定值**；其它非零拼写只能经**另一个** gate 的调用（`MROWS` / `FOLD`）
   到达 mtile kernel —— A/B 契约只用 `=1`。
3. **一进程一值**：`.cu` 的 gate 是函数内 static ⇒ `DSV41_HEAD_MTILE_BN` 的 sweep
   必须**一值一 serve**（不能同进程扫）。

### 2.3 第三条路径（无需改动的既有行为，供判读）

UNSLICED 几何下 `verify_head_fold()`（`chain_dev.rs:2359`，默认 ON）走的**也是**
`head_gemv_bf16_v1_mrows`（`chain_dev.rs:9082`）⇒ 只要 `DSV41_HEAD_MTILE=1`，那条
路径的 C 分派就已经是 mtile kernel（本交付未动它，也不需动）。含义：
**未切片臂下 H1 是默认活的**；只有当 `DSV41_VERIFY_HEAD_FOLD=0` 时才需要 §2.2 的
第二处 arming —— 那是本次为「gate 在任何几何下都不是部分静默门」补的对称性。

---

## §3 编译验证

| 项 | 命令 | 结果 |
|---|---|---|
| Rust（本机，含新增 gate/note/两处调用点） | `cargo check -p ferrite-models --all-targets` | ✅ **EXIT=0**（仅既存 `dead_code` warning） |
| Rust 全仓 | `cargo check --workspace --all-targets` | ✅ **EXIT=0** |
| CUDA `.cu` | **本交付未改任何 `.cu`** | 不需要重编（H1 的 kernel/launcher 已在 HEAD 的 `.so` 里） |

> `.so` 与新二进制的兼容性：H1 **无新符号** ⇒ 旧 `.so` 足够；H3 **需要新符号**
> （§1.1）⇒ 必须重建 `.so`。

---

## §4 A/B 命令清单（主 agent 串行执行）

> **纪律**：一次一个 serve（8 卡上两个 serve 给的不是噪声而是无意义数字）；一臂一
> prompt（`[dspark]` / `[acc-hist]` 是**进程级**累加器）；同栈、同 prompt、背靠背；
> 收尾一律 `POST /shutdown` + `pkill -9 -x ferrite-serve`。

### 4.0 启动前检查（三证，缺一跑出来的数字不算数）

```bash
ssh -o BatchMode=yes ubuntu@43.202.208.136 'bash -s' <<'REMOTE'
set -u
cd ~/ferrite || exit 9
# ① 双产物同源
./target/release/ferrite-serve --version 2>/dev/null | head -2
cat kernels/cuda/libferrite_kernels.so.build_id 2>/dev/null | head -1
# ② H1 的 kernel 在 .so 里（H1 无新符号，但确认 launcher 是新版）
nm -D kernels/cuda/libferrite_kernels.so | grep -c dsv41_gemv_bf16_v1_mrows
# ③ H3 的符号在不在（Arm 2 的硬前置；=0 就先跑 gen_head_aot.py + build.sh 103a）
nm -D kernels/cuda/libferrite_kernels.so | grep -c dsv41_head_bf16_tilelang
ls -la kernels/cuda/tilelang_gen/head_partial_tl.cu kernels/cuda/tilelang_gen/head_reduce_tl.cu 2>&1
REMOTE
```

**判读**：②≥1；③=0 ⇒ **Arm 2 无意义**（会打 `warning: DSV41_HEAD_TILELANG=1 but
the TileLang head did NOT run` 并静默测老路），必须先 AOT + `bash kernels/cuda/build.sh 103a`
（生成/编译是主 agent 的 GPU/整编职责）。

### 4.1 一臂一 serve 的模板（`ARM=none|h3|h1|h1bn1|mrows`）

以下是**一条 ssh 命令**把整段喂给节点执行（本地不留半截状态）。`$EXTRA` 是本臂的
gate；其余 env 是基线栈（照 `docs/agent/r0-r1-accept-diagnosis-manual.md` §6.1，
即 SWALLOW + `DSV41_TIMING=1` + `DSV41_ACC_HISTOGRAM=1` 的当前栈，mean-k 基线 2.240）。

```bash
ARM=h1          # none | h3 | h1 | h1bn1 | mrows
case "$ARM" in
  none)  EXTRA="" ;;
  h3)    EXTRA="DSV41_HEAD_TILELANG=1" ;;                              # 欠 acc 门
  h1)    EXTRA="DSV41_HEAD_MTILE=1" ;;                                 # bn 默认 2
  h1bn1) EXTRA="DSV41_HEAD_MTILE=1 DSV41_HEAD_MTILE_BN=1" ;;           # 几何对照（无复用）
  mrows) EXTRA="DSV41_VERIFY_HEAD_MROWS=1" ;;                          # fold 单独臂（位级同）
esac

ssh -o BatchMode=yes ubuntu@43.202.208.136 "ARM=$ARM EXTRA='$EXTRA' bash -s" <<'REMOTE'
set -u
cd ~/ferrite || exit 9
LOG=/tmp/c4head_${ARM}.log; OUT=/tmp/c4head_${ARM}.json; PORT=8320
pkill -9 -x ferrite-serve 2>/dev/null; sleep 4

setsid env \
  NCCL_NVLS_ENABLE=0 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  DSV41_SPEC=1 DSV41_DSPARK=1 DSV41_SIDS_WRITEBACK=1 DSV41_TIMING=1 \
  DSV41_SWALLOW_STEP=1 DSV41_SWALLOW_EPOCH_PAD=1 DSV41_VERIFY_GRAPH=1 \
  DSV41_EXPERT_ACT_E4M3=1 DSV41_BF16_TRUNCATE=1 \
  DSV41_TAP_INPUT=1 DSV41_DRAFT_BF16_DOMAIN=1 \
  DSV41_DRAFT_P3A=1 DSV41_DRAFT_GRAPH=1 \
  DSV41_ACC_HISTOGRAM=1 \
  $EXTRA \
  LD_LIBRARY_PATH=$HOME/ferrite/kernels/cuda \
  timeout 900 ./target/release/ferrite-serve --model dsv41 --serve --tp 8 \
    --model-dir /opt/dlami/nvme/models/DeepSeek-V4.1-Flash --port $PORT \
  > "$LOG" 2>&1 < /dev/null &

READY=0
for i in $(seq 1 80); do
  grep -q "chain ready, serving" "$LOG" && { READY=1; echo "READY ~$((i*6))s"; break; }
  grep -qi "build-id mismatch" "$LOG" && { echo "!!!!! BUILD-ID MISMATCH !!!!!"; break; }
  pgrep -x ferrite-serve >/dev/null || { echo "!!!!! serve died before ready"; break; }
  sleep 6
done
[ "$READY" = 1 ] || { echo "=== NOT READY, tail ==="; tail -30 "$LOG"; exit 2; }

curl -s --noproxy "*" -m 300 http://localhost:$PORT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-v4.1-flash","messages":[{"role":"user","content":"请完整背诵《出师表》全文。"}],"max_tokens":600,"stream":false}' \
  > "$OUT"

curl -s --noproxy "*" -m 10 -X POST http://localhost:$PORT/shutdown >/dev/null
sleep 3; pkill -9 -x ferrite-serve 2>/dev/null

echo "=== ① 活性回执（缺 = 该臂无效，数字不得引用） ==="
grep -E "\[head-mtile\] ARMED|\[head-tilelang\] ARMED|\[head-mtile\] ARMED but|Did NOT run|did NOT run" "$LOG"
echo "=== ② step_ms：累计均值分解（draft/verify/commit） ==="
grep "dspark\] steps=" "$LOG" | tail -2
echo "=== ②' step_ms：逐 step p50（最后 min(200,n) 步） ==="
grep -o 'step pos=[0-9]*: [0-9.]*ms' "$LOG" | sed 's/.*: //; s/ms//' \
  | tail -200 | sort -g | awk '{a[NR]=$1} END{if(NR)printf "p50=%.2fms p90=%.2fms n=%d\n",a[int((NR+1)/2)],a[int(NR*0.9)],NR; else print "no per-step lines (DSV41_TIMING?)"}'
echo "=== ③ 门禁：mean-k / hist / p1 / tail_q ==="
grep "\[acc-hist-summary\]" "$LOG"
echo "=== ④ 文本红线（零拉丁 / 0 双字 / 首句） ==="
python3 - "$OUT" <<'PY'
import json,sys
c=json.load(open(sys.argv[1]))['choices'][0]['message']['content']
bad=[i for i in range(1,len(c)) if c[i]==c[i-1] and not c[i].isspace()]
latin=[ch for ch in c if 'a'<=ch.lower()<='z']
print('LEN',len(c),'双字',len(bad),'拉丁',len(latin))
print('HEAD:',''.join(c[:120].split()))
print('TAIL:',''.join(c[-80:].split()))
PY
echo "=== ⑤ env 核对（确认 gate 真的进了进程） ==="
grep -c "PACING" "$LOG" >/dev/null 2>&1 || true
REMOTE
```

**建议的串行臂序（每臂一条上面的命令，改 `ARM=`）**

| # | ARM | gate | 角色 |
|---|---|---|---|
| 1 | `none` | — | 基线：逐行 head（m 次 launch，每行一遍 slice） |
| 2 | `h3` | `DSV41_HEAD_TILELANG=1` | **M-in-tile 赌注**（原型 m=6 绝对值 2.93×），欠 acc 门 |
| 3 | `h1` | `DSV41_HEAD_MTILE=1`（bn=2） | **位级兜底**（fold + N-tile），`mean-k` 必须与 Arm 1 **逐位相等** |
| 4 | `h1bn1` | `+ DSV41_HEAD_MTILE_BN=1` | 几何对照：只换几何无复用 ⇒ 把「fold 的 m→1 收益」与「tile 复用收益」拆开 |
| 5 | `mrows` | `DSV41_VERIFY_HEAD_MROWS=1` | *可选*：v1 fold 单独臂（位级同、已是成熟臂）——Arm 3 vs 5 = 纯粹的 tile 效应 |

> `h1`/`h1bn1`/`mrows` 都走**同一个** launcher ⇒ 三个臂的差值是干净的：Arm1→Arm5
> 是 fold（m 遍 slice → 1 遍），Arm5→Arm3 是 N-tile（激活读 `n*M` → `n*M/bn`）。
> Arm 4 是「几何变了但没复用」的阴性对照（应 ≈ Arm 5）。

### 4.2 双门禁判据（每臂都要）

| 门 | 读什么 | 判据 |
|---|---|---|
| **门 1：step_ms** | `[dspark] steps=` 的 `verify=`（+ `draft=` / `commit=`）与逐 step **p50**（AGENTS.md 测量纪律 1：**禁止吞吐反推**） | H1：`verify` 不掉（位级同 ⇒ 只有指令/流量差）；H3：`verify` 应显著下降（原型 2.93× ⇒ 折算 step 级别收益按 §`mtile-woa-head-design` §5 口径）。**符号先看，幅度后看** |
| **门 2：accept** | `[acc-hist-summary]` 的 `mean-k` / `hist={…}` / `p1` / `tail_q` | H1：**`hist` 与 `mean-k` 必须与 Arm 1 逐位相等**（任何 delta = 实现 bug，不是噪声）；H3：这是**唯一能还数值债的门** —— 2.3e-2 ≫ 1e-3 翻面阈值，若 `mean-k` 掉或 `hist` 尾部塌陷 ⇒ H3 **判决失败**（micro bench 证明不了它，只有这一门能） |
| **前置：回执** | `[head-mtile] ARMED …`（H1）/ `[head-tilelang] ARMED …`（H3） | **没打印 = 没走新程序，该臂整轮作废**（本项目的一号测量偏置陷阱）。H3 还要看 `head_tilelang_note` 的警告；H1 还要看 `head_mtile_note` 的警告 |
| **红线** | 响应文本：零拉丁 / 0 相邻双字 / 首句「先帝创业未半」 | 任一破 ⇒ 该臂的数字不得引用 |

### 4.3 p50 窗口的诚实边界

`[dspark] steps=` 每 **50** step 才打一行且是**进程级累计均值**；逐 step 的
`[dsv41] step pos=` 才是 p50 的来源。300 tok 在 mean-k≈2.24 下只 ~90 步 ⇒ 窗口不足
200；本清单用 `max_tokens=600` 把稳态窗口拉到 ~190 步（≈200），仍然**报告 n**。
若主 agent 沿用 300 tok 的口径，请把 §4.1 的 `max_tokens=600` 改回 300 并接受
`n≈90`（p50 仍可用，只是窗口短）。

### 4.4 回滚

```bash
unset DSV41_HEAD_MTILE DSV41_HEAD_MTILE_BN DSV41_HEAD_TILELANG DSV41_VERIFY_HEAD_MROWS
```

`DSV41_HEAD_MTILE` 未设 ⇒ 本交付的两处 `armed_mtile` 都是 false ⇒ 调用点回到
`verify_head_mrows` 单独把守 ⇒ **逐字节回到接线前的行为**。

---

## §5 风险与坦白

1. **H3 的 AOT 前置是本清单唯一的硬阻塞**（§1.1）。生成物不在树里 ⇒ Arm 2 会以
   「armed but no symbol」收场。**主 agent 必须在 4.0 里确认 ③ 非 0**。
2. **H1 的性能押注是 issue-bound**：head 的 `n*k*2` = 165 MB（seg=16160 @ world=8）
   权重流是 DRAM 侧支配项，本臂**不动**它。若 GPU 显示纯权重-DRAM-bound ⇒ 本臂是
   wash —— **上报，不调 bn 掩盖**（`mtile-woa-head-design` §7.2）。
3. **H1 的 CLI 契约**：Rust strict `== "1"`、C `atoi != 0`，`"1"` 是唯一约定值（§2.2
   第 2 条）。这是**刻意的**，但如果主 agent 想用 `DSV41_HEAD_MTILE=2` 之类验证
   clamp 行为，那条路径**不会**经本 gate 生效。
4. **未做 GPU 验证**（任务禁 GPU）：H1 的逐位论证是**读码论证**（kernel header 的
   C1-C6 + `mtile-woa-head-design` §3），不是实测。§4 的 Arm 4/5 就是为把它变成一个
   **可观测的**断言（`hist` 逐位相等）而设的。
5. **文档口径**：H3 的 2.93× 是原型（`tilelang-attn-head.md`）数字，不是本栈数字；
   本手册不把它当既定收益，只当假设。
