# tcgen05 冒烟重测的设计（确保 kernel 真的被调用）

> 工部 · 2026-09-12 · **只读调查 + 本文件（唯一产出）**。未执行任何 GPU 命令、未改动任何源码。
> 代码基线：工作树 HEAD（`git log` 顶端含 `8dafabd` 双产物重编 + `845506c/ab64f61` ar5-hang 修复）。
> 现场核对文件：`crates/ferrite-models/src/dsv41/{chain_dev.rs,load.rs,weights.rs,device.rs,kernels.rs}`、
> `kernels/cuda/dsv41_experts_mxf4.cu`、`crates/ferrite-dsv41/src/serve.rs`、
> `crates/ferrite-dsv41/src/bin/dsv41-run.rs`、`scripts/{tcgen05_smoke.sh,tcgen05_bench.sh}`、
> `docs/agent/{tcgen05-e4m3-grouped-expectation.md,dspark-correctness-chain.md}`。

---

## 0. 结论摘要（先看这个）

**「0 misaligned」之所以空洞，不是「判据太弱」，而是「这个项目里根本不存在“kernel 跑成功”的正证据」。**

三条代码级事实构成空洞的结构性原因：

1. **成功是静默的。** 全链路只有 **decline 告警**（`tcgen05_e4m3_skipped_note` /
   `tcgen05_e4m3_ext_skipped_note` / `expert_grouped_skipped_note`，`chain_dev.rs:772/789/920`）
   和 **失败**（`self.kerr(rc, …)`）会打日志；一个成功执行的 `dsv41_expert_gemm_e4m3_grouped`
   **不打任何日志**。`.cu` 侧同样：`atomicAdd` 计数 0 处、launch 打印 0 处（整个
   `dsv41_experts_mxf4.cu` 只有 `:1069` 一条 PDEPTH 的 `fprintf`）。所以在日志里 grep
   「tcgen05 相关行」**当前必然为空**——这不是「kernel 没跑」，是「跑没跑都不说」。
2. **`/health` 不证明引擎 load 过。** `TpRankPool::ensure_loaded()`（`serve.rs:204`）只在
   `StepEngine::prefill`（`serve.rs:267`）里被调用，而 HTTP listener 先绑定、ranks 在后台
   load。所以 `scripts/tcgen05_smoke.sh:281` 的「health OK」只证明**端口活着**，对
   权重加载、引擎初始化、kernel dispatch **零信息量**。这正好解释了
   「34cf4c74：serve 启动 → 请求 → 结果不明」为什么会被读成「没崩就是好」。
3. **`DSV41_EXPERT_TCGEN05_E4M3` 一次武装两个未验证 kernel，而且 GPU 首触是单行那个。**
   同门名下：
   - **prefill / eager decode 的 `moe()`**（单行）→ `dsv41_expert_tcgen05_gate_up_e4m3`
     （swapAB，`tc5::e4`，`chain_dev.rs:14279-14310`）；
   - **spec decode 的 `moe_rows()`**（verify 多行）→ `dsv41_expert_gemm_e4m3_grouped`
     （grouped masked tile，`tc5::e4x`，`chain_dev.rs:10929`）。

   prefill 是**逐 token 单行 forward**（`dsv41-run.rs:389-391`），所以**第一个上 GPU 的
   tcgen05 kernel 是 swapAB 臂，不是 grouped 臂**。「冒烟崩了 → 归咎 grouped kernel」这条路
   从一开始就是错的。

> 因此本设计的核心不是「换个判据」，而是：**先造正证据（第 2 节），再按 kernel 隔离（第 3 节）。**

---

## 1. 关键调查（逐条答复任务里的 4 个问题）

### 1.1 `weights.rs` 的加载逻辑：ILV=0 时权重怎么加载？

**答案：ILV=0 就是 checkpoint 的原生布局，完全支持；交错是 ferrite 自己的 load-time 变换，不是 checkpoint 属性。**

- `load_expert_pool`（`load.rs:603-734`）按名字读 checkpoint 的 **6 个独立平面**：
  `w1.weight / w1.scale / w3.weight / w3.scale / w2.weight / w2.scale`（`load.rs:612-614`）。
- **`ilv == false`（第 706-715 行）**：对 6 个平面各做一次 `dma_plan`，落到
  `poff[k]`（顺序累积，`:657-664`）——即 `[w1][w1.scale][w3][w3.scale][w2][w2.scale]`。
  这是 **ILV 引入之前的历史基线布局**，也就是 checkpoint 的原样。
- **`ilv == true`（第 683-705 行）**：把 w1/w3 先 DMA 进一个 scratch，再调
  `interleave_gateup_fp4` 生成「一个区域装两个平面」的交错区
  （`gate[0..8] + up[0..8] + gate[8..16] + …`，`load.rs:594-602`），
  并让 w3 的 view **别名** w1 的区间（`:720-724`）。
- `ilv_ok()`（`load.rs:767-778`）是**唯一**的布局决策点，且它把 `gateup_fuse()`
  作为合取项 ⇒ **`DSV41_GATEUP_FUSE=0` 本身就强制 `ilv=false`**，
  `DSV41_EXPERT_ILV=0` 是 belt-and-braces（expectation §1.3 已记录，本次复核成立）。
- 代价：ILV 实测仅 **−0.09ms**（`stage-b-execution.md:63`），所以 ILV=0 ≈ +0.09ms。

**结论：不存在「checkpoint 只有交错权重」的问题。ILV=0 是原始布局，安全、可用、且是 grouped
臂的硬前置**（`chain_dev.rs:10884` 对 `ld.experts_ilv` 直接 decline）。

### 1.2 `moe_experts_grouped_gate_up` 的调用条件——什么情况下走 tcgen05 路径？

调用点 `chain_dev.rs:11191`（`moe_rows` 内），实参门链在 `chain_dev.rs:10835-10891`：

| # | 条件 | 代码位置 | arm 取值 | 不满足的后果 |
|---|---|---|---|---|
| 1 | `expert_tcgen05_e4m3()` | `:10850` | `=1` ✓ | decline + `expert_grouped_skipped_note` |
| 2 | `e4m3`（`expert_act_e4m3() && supports_expert_act_e4m3()`） | `:10859` | `=1` ✓ | decline（激活是 fp4 nibble） |
| 3 | `!gateup_fused` | `:10867` | `GATEUP_FUSE=0` ✓ | decline（e4x epilogue 只 clamp） |
| 4 | `dim % 64 == 0 && 2*inter_local % 64 == 0` | `:10876` | 5120 / 4608(TP1) / 640(TP8) ✓ | decline |
| 5 | `!ld.experts_ilv` | `:10884` | `ILV=0` ✓ | decline |
| 6 | `supports_expert_gemm_e4m3_grouped()` | `:10892` | 需 `nm -D` 有符号 | decline |
| 7 | `ld.experts.len() >= 2` | `:10900` | ✓ | decline |

**并且在调用者一侧还有一层**（`moe_rows`，`chain_dev.rs:11174-11194`）：
- `moe_route_grouped(...)` 必须先返回 `true`（`DSV41_EXPERT_GROUPED=1` + `dsv41_route_*` 三符号
  齐备 + 容量 guard 通过，`:10717-10795`），否则 `_grouped == false`，`grp_gu` 短路边为 false；
- `grp_gu == true` 时**跳过** `expert_gate_up_fp4_batched`（`:11298`），即 SIMT 回退不跑。

**关键旁证：`e4x_tile = false` 是硬编码的**（`chain_dev.rs:11244`）。
所以 `dsv41_expert_gemm_e4m3_ext`（dense M=128 tile 臂）在 `moe_rows` 里**永远不发**，
只会打一条 `tcgen05_e4m3_ext_skipped_note`（`:11245-11260`）。
⇒ **在 verify 路径上，`DSV41_EXPERT_TCGEN05_E4M3=1` 唯一能落地的 e4x kernel 就是 grouped masked 那个。**
但这条 decline 告警**每进程只打一次**（`OnceLock`），而 grouped 臂若先成功，same gate 的这条
告警也会照打——**「看到 ext decline 告警」不能推出「grouped 也 decline」**。这是判读陷阱。

### 1.3 serve 日志里 tcgen05 的启动行——有 arm/disarm 信息吗？

**没有。当前一个都没有。**

- Rust 侧：只有 3 条 one-shot **decline** 告警 + `tcgen05_e4m3_skipped_note` /
  `tcgen05_e4m3_ext_skipped_note` / `tcgen05_mxf4_skipped_note`（`chain_dev.rs:772/789/807`）。
- `.cu` 侧：`dsv41_expert_gemm_e4m3_grouped` 的 wrapper（`:6120-6139`）**只**在 gate OFF 时
  `return 0`，成功时 `return (int)rc` 且 `(void)cudaGetLastError()`——**零日志、零计数**。
- `e4x_launch_gemm_grouped`（`:6072-6104`）有完整的 shape 契约检查（`k % kAtomK`、
  `n_total % kNTile`、`epi_mode ∈ {0,1}`、16B 对齐、B stride 16B 倍数），
  **失败返回 `cudaErrorInvalidValue`，成功静默**。注意这个返回值会经
  `dsv41_expert_gemm_e4m3_grouped` 返回给 Rust，`self.kerr(rc, …)` 会把它变成错误
  ——所以「shape 不满足」是**响的**，「形状满足但数值错」是**静的**。
- 现有 arm 文档里的 `grouped` / `e4x_launch` / `arm` 这些词，全是**注释里的措辞**，
  不是运行时可 grep 的日志行。**按日志 grep 确认执行，今天做不到。**

### 1.4 权重布局问题（ILV=0 需要非交错 checkpoint？）

见 1.1：**这不是问题。** 非交错就是 checkpoint 原样。真正需要确认的是另一件事——
`moe_experts_grouped_gate_up` 同时传 `w1_base/w1_stride` **和** `w3_base/w3_stride`
（`:10929-10952`，`b_split = inter_local`），所以 **`ilv=false` 下两个平面必须是真实的独立区域**
（ILV=1 时 w3 是别名，grouped 臂读它会读错——这正是第 5 条 decline 的由来）。
ILV=0 天然满足。

### 1.5 补充调查：为什么「serve 起了」和「kernel 跑了」之间隔着很长一段路

引擎启动到 tcgen05 kernel 执行的完整路径（每一段都可能 fault，且都发生在 kernel 之前）：

```
listener bind (/health OK)                         ← 冒烟脚本只测到这一层
  └─ TpRankPool::new → 8 个 rank 线程                 serve.rs:151-200
       └─ rank_loop: Loader::load()                   serve.rs:373
            └─ load_expert_pool: dma_plan × 6×384×40   load.rs:680-731  ← ILV=0 走这
            └─ DevChain::new + reset() (zero/upload)  dsv41-run.rs:381-383
       └─ 首次 prefill → chain.step(tk,i) 逐 token    dsv41-run.rs:389-391
            └─ step_impl → moe() → **swapAB tcgen05**  chain_dev.rs:14292   ← GPU 首触
       └─ decode: SPEC=1 → dspark_spec_step → step_rows(m=6-7)
            └─ layer_rows → moe_rows → grouped tcgen05 chain_dev.rs:10929   ← 目标 kernel
```

「crash 在 init/fault 路径」在这张图里有**至少 5 个候选位置**，
而 `f15ecd37` 的 `rank 5: sync: misaligned address` **只知道是某次 sync 报的**
（异步 kernel fault 是 sticky 的，会在**下一个** sync 处浮出来）。
文档 `dspark-correctness-chain.md:2404` 的追因认为是
`expert_gemv_fp4_batched_kernel:1673/1678` 的对齐守卫被 `pair_body=false` 绕过
——**那是 SIMT 回退 kernel，不是 tcgen05 kernel**。这恰好印证第 0 节的第 3 条：
**之前的「tcgen05 冒烟」测到的很可能是别的 kernel。**

---

## 2. 设计要点一：造正证据（本设计的核心）

### 2.1 证据分级

| 级别 | 内容 | 当前状态 |
|---|---|---|
| **L0** | 无 decline 告警 | 已有（但**只是 absence of evidence**） |
| **L1** | gate 前提成立（env 在 `/proc/<pid>/environ`、符号在 `.so`、布局 plain） | 已有（脚本 stage 1/2 部分覆盖） |
| **L2** | **launch 发生**（host 侧） | ❌ **完全缺失** |
| **L3** | **kernel body 执行**（device 侧） | ❌ **完全缺失** |
| **L4** | 数值可接受（红线） | 已有（stage 3） |

**重测的最低门槛 = L2 有正证据。** 推荐 L2+L3 双证据（互相校验）。

### 2.2 L2 的三种实现（按侵入度排序）

**方案 A（零改动，金标准）：nsys。**
```bash
nsys profile -t cuda --stats=true -o /tmp/tc5_smoke \
  ./target/release/ferrite-serve ...   # 同现有脚本的 env
# 证据：
nsys stats --report cuda_gpu_kern_sum /tmp/tc5_smoke.nsys-rep \
  | grep -E 'e4m3_gemm_grouped_kernel|e4m3_gemm_kernel'
```
- `e4m3_gemm_grouped_kernel`（`dsv41_experts_mxf4.cu:5831`）= **grouped 臂**（本次目标）；
- `expert_tcgen05_gateup_e4_kernel`（`:4821`）= **swapAB 单行臂**
  （`dsv41_expert_tcgen05_gate_up_e4m3` → `e4_launch_gateup`，`:5122/5191`）；
- `e4m3_gemm_kernel`（`:5444`）= dense ext 臂（`e4x_launch_gemm`）。⚠️ 该臂在 `moe_rows` 里
  **恒不发**（`e4x_tile = false`，`chain_dev.rs:11244`），所以它在 verify 路径上的调用数
  应当**恒为 0**；如果 nsys 里看到它 > 0，说明有人在别的路径上把它打开了——那是异常，要查。
- **优点**：不碰源码；profiler 记录每一次 launch，**kernel 没跑就是没有行**——不可能空洞。
- **缺点**：node 上要有 nsys；profile 会拖慢（对正确性冒烟无害，对计时有害，所以**别在同一轮读数**）。

**方案 B（约 10 行 `.cu`，推荐）：env-gated 一次性 launch 打印。**
在 `dsv41_expert_gemm_e4m3_grouped` 的 wrapper（`:6120`）里、`e4x_launch_gemm_grouped` 之后加：
```cuda
static const int trace = [] { const char* e = getenv("DSV41_TCGEN05_TRACE");
                              return (e && e[0] == '1') ? 1 : 0; }();
static int n_trace = 0;
if (trace && n_trace < 4) {  // 前 4 次，避免污染日志
    ++n_trace;
    fprintf(stderr, "[tcgen05] grouped e4x launch #%d: grid=(%d,%d,%d) n_total=%d k=%d "
                    "b_split=%d epi=%d rc=%d\n",
            n_trace, (int)(n_total / 64), (int)((m_cap + 127) / 128), n_assign,
            n_total, k, b_split, epi_mode, (int)rc);
}
```
- `.cu` 已经有 `#include <cstdio>`（`:94`）与 `fprintf` 先例（`:1069`），风格一致。
- **优点**：证据落在**冒烟脚本本来就在 grep 的那个 serve 日志**里；`grid` 与 `rc` 一起打印，
  `grid.y == ceil(m_cap/128)`、`grid.z == n_assign = m*topk` 可反查形状是否合理。
- **注意**：这是 **L2**（launcher 跑了），不等于 kernel body 跑了。
  所以 **A 或 C 用来交叉验证**。

**方案 C（device 侧计数，L3）：**
给 `e4m3_gemm_grouped_kernel` 加一个 `unsigned long long* launches` 形参，CTA(0,0,0) 的
`threadIdx.x == 0` 做一次 `atomicAdd`；host 在请求末尾 D2H 读回并打印。
- **优点**：证明**kernel body 真的在 GPU 上执行过**，且顺带给出实际 CTA 数。
- **缺点**：改了 ABI（加形参）⇒ Rust 侧 FFI 要同步（`kernels.rs`/`device.rs`），
  属于「方案外的源码改动」，**需尚书省批准后再做**。建议 **A+B 先跑，C 只在 A/B 仍不足以定案时才上**。

### 2.3 失败隔离（把「fault 出在哪次 launch」钉死）

这一步几乎免费，但能直接终结本次争议：

1. **`CUDA_LAUNCH_BLOCKING=1`**（第一轮**必须**开）。异步 kernel fault 默认 sticky，
   在下一次 sync 才报——`f15ecd37` 的「rank 5: sync: misaligned」就是这么丢掉出处的。
   开了之后，报错就归到**真正 fault 的那次 launch**。
2. **`compute-sanitizer --tool memcheck --launch-timeout 120`**（第一轮**建议**开）。
   直接给出 kernel 名 + 出错指令 + 地址，把「哪个 kernel、哪个偏移不对齐」一次问清。
   慢（10-100x），但这是**一次性首触**，值。
3. **`DSV41_VERIFY_GRAPH=0` / `DSV41_GRAPH_MOE=0` 显式写死**。
   两者默认都 OFF（`verify_graph_want()` `chain_dev.rs:1916`；`moe_graph_armed` `:12476`），
   但显式关掉可以去掉「capture 只记录不执行 / capture FAILED 静默降级到 direct」这类
   会改变执行路径的变量。**首触不需要任何图。**

### 2.4 三轮定案（把「引擎能跑」与「kernel 能跑」分开）

空洞的另一个来源：把「请求返回了」当成「kernel 跑了」。用三轮把它拆开：

| 轮 | 配置 | 这一轮要回答的问题 | 判定 |
|---|---|---|---|
| **R0 空白轮** | 基线 env（无任何 tcgen05 门），同一 prompt | 引擎在本树、本机**能不能正常起 + 出正确文本**？ | 必须 PASS，否则**一切归因无效**（这也是之前缺的一轮） |
| **R1 layout 轮** | 只 `GATEUP_FUSE=0 EXPERT_ILV=0`（无 tcgen05 门） | 门税（≈+0.4~0.5ms）+ plain 布局的 SIMT 路径**能不能跑通**？ | 必须 PASS。**R1 崩 = arm 的必付税本身有毒，与 tcgen05 无关**（`f15ecd37` 的 misaligned 就高度疑似这一档） |
| **R2 arm 轮** | R1 + `TCGEN05_E4M3=1 [GROUPED=1]` | kernel 是否被调用、是否崩、数值是否可接受 | 见 §4 判定表 |

**R1 是本次设计新增的关键一轮**，它把 `f15ecd37` 那类「SIMT 守卫绕过型 misaligned」
从 tcgen05 的账上剥离出去。没有 R1，R2 的任何失败都归因不清。

---

## 3. 设计要点二：按 kernel 隔离（把两个未验证 kernel 分开测）

`DSV41_EXPERT_TCGEN05_E4M3` 一门两 kernel（§0-3）。设计上必须**先单独测简单的那个**。

### 3.1 Phase 0a：swapAB 单行臂（`tc5::e4`）——单 GPU，无需 serve，无需 TP

**现成 harness：`target/release/dsv41-run --tp 1`**。
- 它 prefill 逐 token（`dsv41-run.rs:389-391`）、decode 走 `step_dev`（`:406`）
  ——**两段都是单行 `moe()`**，所以整条路径只碰 swapAB 臂，**完全不碰 grouped 臂**。
- 输出 `[dsv41] DECODE … tok/s` 与逐 token 文本；`--tp` 默认就是 1（`:52`）。
- 建议 env：
  ```
  CUDA_VISIBLE_DEVICES=0 DSV41_LAUNCH_BLOCKING=... \
  CUDA_LAUNCH_BLOCKING=1 DSV41_TCGEN05_TRACE=1 \
  DSV41_EXPERT_ACT_E4M3=1 DSV41_EXPERT_TCGEN05_E4M3=1 \
  DSV41_GATEUP_FUSE=0 DSV41_EXPERT_ILV=0 DSV41_MOE_BATCH=1 \
  DSV41_MODEL_DIR=... DSV41_KERNELS=.../libferrite_kernels.so \
  ./target/release/dsv41-run --tp 1 --prompt "你好" --max-tokens 32
  ```
- **PASS 判据**：进程 exit 0 + 文本可读（无拉丁/无相邻重复）+ 日志有 launch 证据
  （方案 B 的 `[tcgen05]` 行，或 nsys 里出现 **dense 臂 `e4m3_gemm_kernel`** 的调用）。
  ⚠️ swapAB 臂的符号是 `dsv41_expert_tcgen05_gate_up_e4m3`（不是 grouped 那个），
  nsys 里对应 `e4m3_gemm_kernel`（`e4x` 的 dense 形态，`:5444`）。
- **为什么先它**：单 token、swapAB、无 mask、无 group table、无 scatter——**变量最少**。

### 3.2 Phase 0b：grouped 臂（`tc5::e4x` masked tile）——单 GPU，直调 `step_rows`

serve 走 grouped 臂**必然先经过 prefill 的 swapAB**（§0-3），所以 serve 不能做「grouped 隔离」。
最干净的隔离是**直接调公开的 verify API**：

- `DevChain::step_rows(&mut self, toks: &[u32]) -> Result<Vec<u32>>`
  （`chain_dev.rs:5190`，`pub`）→ `layer_rows`（`:8729`）→ `moe_rows`（`:8925`）
  ——**没有 prefill、没有 draft、没有 serve、没有 TP collective**。
- 落地形态：新增 `crates/ferrite-dsv41/tests/real_grouped_tcgen05.rs`，
  照 `tests/real_gemm.rs` 的模板（`DSV41_MODEL_DIR` 缺席即 skip、`CUDA_VISIBLE_DEVICES=0`、
  单 GPU），构造 `Device` + `Loader::load(cfg, 1, 0)` + `DevChain::new` + `reset`，
  然后 `chain.step_rows(&[tok0, tok1, tok2, tok3, tok4, tok5])`（m=6 = VERIFY_ROWS）。
- **PASS 判据**：
  1. `step_rows` 返回 `Ok`（不 fault）；
  2. 日志/nsys 出现 **`e4m3_gemm_grouped_kernel`**（`:5831`）——**这是本设计的 L2/L3 正证据**；
  3. 无任何 `expert_grouped_skipped_note`（`chain_dev.rs:920`）。
- **成本**：单卡 + 只加载一层专家即可（若加载全模型太重，可只构造一层 MoE 的 weights——
  这是实现细节，属「方案外」，需尚书省确认范围）。
- **收益**：**这是唯一能在「引擎未起 / TP 未通」的情况下证明 grouped kernel 被执行的测法。**

### 3.3 只有当 0a + 0b 都 PASS，才做 serve 级（TP8）重测

serve 级 R2 的顺序仍是：`R0 → R1 → R2`（§2.4），且 R2 里必须**同时**看到两个 kernel 的
launch 证据，才能证明「prefill 的 swapAB 与 verify 的 grouped 都跑过」。

---

## 4. 设计要点三：单 GPU vs TP8

**结论：tcgen05 在单 GPU 上没有架构性障碍；而且 TP1 恰好消掉唯一已观测到的失败模式。测 TP1。**

依据：

1. **`tcgen05.mma` 是 per-CTA/per-SM 指令**，与 TP 无关；TP 只改变权重分片与集合通信。
2. **已观测的 misaligned 是 TP 分片特异**：`dspark-correctness-chain.md:2404` 归因为
   「TP 分片让部分 rank 的 w3 view 落在非 8B 对齐处 ⇒ rank 5/6 的 misaligned」。
   **TP1 只有一个分片、基址就是 pool 基址，这一整类问题不存在。**
3. **形状契约在 TP1 / TP8 都成立**（`e4x_launch_gemm_grouped` 要求 `k % 64 == 0`、
   `n_total % 64 == 0`；Rust 侧额外要求 `dim % 64 == 0`、`2*inter_local % 64 == 0`）：

   | world | `inter_local = padded_inter(2304/world)` | `k = dim` | `n_total = 2*inter_local` | `k%64` | `n_total%64` |
   |---|---|---|---|---|---|
   | 1 | 2304 | 5120 | 4608 | 0 ✓ | 0 ✓ |
   | 8 | 320（实测口径，`tcgen05_bench.sh` 头注） | 5120 | 640 | 0 ✓ | 0 ✓ |

4. **TP1 不改变 grouped 路由的容量**：`grp_n_experts` / `grp_topk_max` / `grp_m_cap`
   （`chain_dev.rs:2904/2913/2924`）只依赖 `cfg`，与 `world` 无关。
5. **`VERIFY_ROWS = 6`**（`chain_dev.rs:84`）与 `grid.y = ceil(m_cap/128) = 1`
   （`m_cap = 6*6 = 36`）在 TP1/TP8 相同。

⚠️ **但 TP1 有个不同的风险**：`moe_rows` 里的 AR（all-reduce）在 TP1 下走
`world == 1` 的分支（`self.comm` 可能为 `None`，见 `chain_dev.rs` 里大量
`self.comm.as_ref()` 的可选处理）。所以**TP1 PASS 不能推出「TP8 的集合通信没问题」**——
它只推出「**kernel 本身没问题**」。这正是我们要的隔离：**先证 kernel，再证 TP。**

---

## 5. 设计要点四：gate 前提核对（逐门附读取语义）

arm env（`scripts/tcgen05_smoke.sh:116-117` 已有，此处补齐语义与陷阱）：

| 门 | 读取语义 | 代码 | arm 值 | ⚠️ 陷阱 |
|---|---|---|---|---|
| `DSV41_EXPERT_ACT_E4M3` | `v != "0"` | `chain_dev.rs:856` | `=1` | 关它 = 换激活格式，**引入第二个变量**；二分时**最后**关 |
| `DSV41_EXPERT_TCGEN05_E4M3` | **严格 `starts_with('1')`** | `:756` | `=1` | `=true`/`=on`/`=0` **都不 arm** |
| `DSV41_EXPERT_GROUPED` | **严格 `starts_with('1')`** | `:903` | `=1` | 同上；grouped 臂的必要门 |
| `DSV41_GATEUP_FUSE` | `v != "0"`，**默认 true** | `:1891` | `=0` | **必须显式 `=0`**；它单独就蕴含 `ilv=false` |
| `DSV41_EXPERT_ILV` | `v != "0"`，**默认 true** | `weights.rs:487` | `=0` | 必须显式 `=0`；对 grouped 臂是**硬前置** |
| `DSV41_MOE_BATCH` | `v != "0"`，**默认 true** | `:686` | 不设（默认 ON） | 两个臂都要 batched 分支；别关 |
| `DSV41_SPEC` / `DSV41_DSPARK` | — | — | `=1` | **grouped 臂需要**（否则 decode 走单行 `moe()`） |
| `DSV41_NO_GEMV_FP4` | **bare `getenv`** | `:696` | **必须不存在** | 设任何值都会把 rows==1 送进 tcgen05 GEMM，并经 `ilv_ok` 强制 plain 布局 |
| `DSV41_VERIFY_GRAPH` | `v != "0"`，默认 OFF | `:1916` | `=0`（显式） | 图会引入「记录不执行 / 静默降级 direct」 |
| `DSV41_GRAPH_MOE` | — | `:12476` | `=0`（显式） | 同上 |
| `DSV41_SKIP_ENGRAM_WEIGHTS` | — | `dsv41-run.rs` 头注 | `=1`（可选） | 省 189GiB 加载时间，与 kernel 无关 |

**所有门都是 `OnceLock` 读一次**（house rule），所以 env 必须在**进程 spawn 时**就位；
脚本已用 `/proc/<pid>/environ` 回读（`tcgen05_smoke.sh:289`）——这一层要保留并扩展。

---

## 6. 判读清单（每轮跑完必做）

### 6.1 判定表

| 观测 | 解释 | 动作 |
|---|---|---|
| R0 就崩 | 引擎/权重加载问题，**与 tcgen05 无关** | 修引擎；**本轮作废** |
| R1 崩（misaligned/fault） | **SIMT 回退 + plain 布局** 的对齐问题（`f15ecd37` 疑似此档） | 修 SIMT 守卫；tcgen05 无责 |
| R2 崩 + 有 `[tcgen05]` launch 行 + `CUDA_LAUNCH_BLOCKING=1` 指向该 kernel | **kernel 内真的有对齐/寻址错** | compute-sanitizer 定位；kernel 级修复 |
| R2 崩 + **无** launch 行 | kernel **没被调用** | 查 decline 告警 / gate / 符号；**不要**报「kernel 崩了」 |
| R2 不崩 + 无 launch 行 | **空洞！**（旧测试的失败模式） | 判定 FAIL：arm 没生效 |
| R2 不崩 + 有 launch 行 + 文本可读 + 首 N 字与基线一致 | 真 PASS（冒烟门） | 进入性能 A/B |
| R2 不崩 + 有 launch 行 + 首字即分叉 | 数值错（`[OPEN]` 两条：dense idesc format code / fp4 packed-vs-unpacked） | 不 ship；按 expectation §7 第 4 步二分 |

### 6.2 必须记录的证据（缺一不能下结论）

1. `/proc/<pid>/environ` 的逐门 dump（已有）；
2. `nm -D $SO` 的 5 个符号（已有：`dsv41_expert_act_e4m3_cap`、
   `dsv41_expert_gemm_e4m3_grouped`、`dsv41_route_group`、`dsv41_route_gather_rows`、
   `dsv41_route_scatter_rows`）；
3. **launch 证据**（方案 A 的 nsys kernel 表，或方案 B 的 `[tcgen05]` 行）← **本设计新增，必录**；
4. **归属证据**：`[gmo]` / `[phs]` 是否出现（`chain_dev.rs:12480/12396/12436`，判定窗口内
   是否跑过单行 `moe()`）；
5. 原始响应 JSON + serve 日志 tail（解析失败时**必须**抓原始体，不能只记 `resp_err`）；
6. `[verify_graph]` 三态。
7. **R0/R1/R2 三轮的同一 prompt 文本对比**（不能用「与历史基线逐字节相同」做判据，
   见 expectation §4.2-1：MMA+折叠 vs SIMT fma 链的舍入次序不同）。

---

## 7. 执行顺序（一页）

```
Phase 0（单 GPU，无 serve，无 TP）
  0a  dsv41-run --tp 1 + swapAB arm gates        → 证 swapAB 臂（单行，变量最少）
      证据: exit 0 + 文本可读 + launch 证据
  0b  新增 tests/real_grouped_tcgen05.rs
        Device + Loader(world=1,rank=0) + DevChain + reset + step_rows(6 tokens)
      env: ACT_E4M3=1 TCGEN05_E4M3=1 GROUPED=1 GATEUP_FUSE=0 ILV=0
           CUDA_LAUNCH_BLOCKING=1 (首轮) VERIFY_GRAPH=0 GRAPH_MOE=0
      → 证 grouped 臂（**唯一干净的隔离**）
      证据: Ok 返回 + nsys/日志出现 e4m3_gemm_grouped_kernel + 无 decline 告警

Phase 1（TP8 serve，仅当 0a+0b 都 PASS）
  R0  基线 env                                     → 引擎自证
  R1  GATEUP_FUSE=0 ILV=0（无 tcgen05 门）          → 门税/plain 布局自证
  R2  R1 + TCGEN05_E4M3=1 GROUPED=1                → 目标
      证据: nsys 同时看到 e4m3_gemm_kernel(swapAB, prefill)
            与 e4m3_gemm_grouped_kernel(verify)
  首轮一律: CUDA_LAUNCH_BLOCKING=1 + compute-sanitizer memcheck
  ⚠️ 计时与 profilng 不要同轮（nsys/sanitizer 会污染 ms 读数）

Phase 2（只有冒烟 PASS 才做）
  性能 A/B（scripts/batched_400_v2.sh 或 tcgen05_bench.sh 的 T1 微基准）
```

---

## 8. 需要上报/批准的「方案外」项

本设计**全部是测试设计**，但落地时有 3 项需要尚书省决策：

1. **方案 B 的 `.cu` 改动**（约 10 行，env-gated launch 打印，`DSV41_TCGEN05_TRACE=1`）。
   属于源码改动 —— 请批准后再动。
2. **方案 C 的 ABI 扩展**（launch counter 形参，需同步 `kernels.rs` + `device.rs` FFI）。
   建议**先不上**，A+B 应已足够。
3. **`tests/real_grouped_tcgen05.rs`** 的范围：要不要加载全模型（慢），
   还是只构造一层 MoE 权重（快，但要写隔离的加载脚手架）。
4. 若 R1 确实复现 misaligned，则**SIMT 回退的守卫修复**是独立的一件工程（`f15ecd37` 那一档），
   应单独立项，不要和 tcgen05 混在一个 verdict 里。

---

## 9. 一句话总结

**空洞的根不是判据弱，是「成功没有正证据」+「一次实验混了两个 kernel + 三种布局税」。**
所以重测必须做三件事：**(1) 造正证据**（nsys 或 env-gated launch 打印，让「跑没跑」可 grep）；
**(2) 分离变量**（R0 基线 / R1 门税 / R2 arm，再加 swapAB 与 grouped 两个 kernel 分开测）；
**(3) 单 GPU 先行**（TP1 消掉唯一已观测的 misaligned 类别，且 `dsv41-run --tp 1` 与
`step_rows()` 分别给了两个 kernel 现成/近现成的隔离 harness）。**「0 misaligned」必须被替换成
「launch 计数 > 0 且无 fault」——否则下一次重测还会是空洞的。**

---

*工部 · 只读调查 + 本文件（唯一产出）；未执行任何 GPU 命令、未改动任何源码。*
*所有行号均对工作树 HEAD 现场核对；`grid.y`、`m_cap`、`inter_local` 等推算项已标注口径。*
