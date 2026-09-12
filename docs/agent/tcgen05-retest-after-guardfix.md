# tcgen05 重测方案（对齐守卫修复之后）——让 kernel 真的被调用、且可被证明

> 工部 · 2026-09-12 · **只读调查 + 本文件（唯一产出）**。未执行任何 GPU 命令、未改动任何源码。
> 基线：工作树 HEAD `69487d7`；守卫修复 = commit `ed720d6`（`kernels/cuda/dsv41_experts_mxf4.cu`）。
> 现场核对：`crates/ferrite-models/src/dsv41/{chain_dev.rs,load.rs,weights.rs,device.rs}`、
> `kernels/cuda/{dsv41_experts_mxf4.cu,dsv41_route.cu,build.sh}`、
> `crates/ferrite-dsv41/src/{serve.rs,bin/dsv41-run.rs}`、
> `scripts/tcgen05_smoke.sh`、`crates/ferrite-dsv41/tests/real_gemm.rs`。
> **所有行号均对当前工作树现场核对；推算项已标注口径。**

---

## 0. 结论摘要（先看这个）

**这次重测必须先纠正一个前提错误：`ld_uint2_a8` 守卫保护的是“配对体（pair body）”，而冒烟臂（GATEUP_FUSE=0 + ILV=0）跑的却是“分体体（split body）”——两者不是同一段代码。**

三条代码级事实：

1. **守卫在 `expert_gemv_fp4_batched_kernel` 里，不在 tcgen05 kernel 里。** `ed720d6` 只改了
   `dsv41_experts_mxf4.cu` 的 SIMT 回退 kernel（`:1669-1710`、`:1746-1756`）。tcgen05 的
   `e4m3_gemm_grouped_kernel` / `e4m3_gemm_kernel` 一行未动 —— **对齐守卫修复对 tcgen05 kernel
   本身零覆盖**；它只是把“上一次 crash 的真凶候选”从账上划掉。

2. **⚠️ 关键：冒烟臂的配置根本不会执行被守卫的那几行。**
   `pair_body = ((fuse_swiglu != 0) || ILV) && (b_split > 0)`（`:1405`）。
   冒烟臂是 `DSV41_GATEUP_FUSE=0`（⇒ `fuse=0`，launcher `:2666-2667` 由
   `out_slot_stride == inter` 绑定，`act_slot = 2*inter` 使其为假）+ `DSV41_EXPERT_ILV=0`（⇒ `ILV=0`），
   于是 **`pair_body = false`** ⇒ 走 **split body**（`vec==2`，权重读是 **4 字节 `uint32`**，
   `:1873/:1874`），**永远不会碰到 `:1677/:1706/:1708/:1754` 的 `uint2` 读**。
   修复 commit 的说明其实自己就点破了这一点：“the arm actually runs the UNFUSED body with
   uint32 (4B) reads — if it still crashes, the next suspects are :1943/:1944 (vec==2) and :2015”。

   ⇒ **“守卫已修复”不能推出“冒烟臂的 crash 会消失”。** 上一版 `tcgen05-smoke-redesign.md` §2.4
   的 R1（GATEUP_FUSE=0 ILV=0）如果复现 misaligned，**与这次守卫修复无关**，需要另立修复项
   （见 §5 与 §6.4）。

3. **成功仍然静默。** 全链路正证据依旧缺失：Rust 侧只有 4 条 one-shot **decline** 告警 +
   `self.kerr(rc, …)`；`.cu` 侧 `dsv41_expert_gemm_e4m3_grouped`（`:6120-6139`）成功时
   `return (int)rc`，**零打印、零计数**。⇒ **“0 misaligned 且没崩”在没有 launch 证据时依然是空洞。**

**所以本方案的核心是两件事：(1) 造正证据（L2/L3）；(2) 把“守卫覆盖的路径”与“冒烟臂跑的路径”分别写进矩阵，别混。**

---

## 1. `moe_experts_grouped_gate_up` 的调用条件（逐门 + 行号）

**定义**：`chain_dev.rs:11114`（`fn moe_experts_grouped_gate_up`）。
**唯一调用点**：`chain_dev.rs:11470-11473`（在 `moe_rows` 内）：

```rust
let grp_gu = _grouped
    && self.moe_experts_grouped_gate_up(m, topk, n_routed, dim, inter_local, e4m3, gateup_fused, ld)?;
```

### 1.1 调用者一侧（`moe_rows`，`chain_dev.rs:11266`）

| # | 条件 | 位置 | 不满足的后果 |
|---|---|---|---|
| 0 | `_grouped = self.moe_route_grouped(...)` 为 `true` | `:11454` | `grp_gu` 短路边为 false，SIMT 回退照跑 |
| 0a | `DSV41_EXPERT_GROUPED` **严格 `starts_with('1')`** + `dsv41_route_*` 三符号齐 + 容量 guard | `:10989`（`moe_route_grouped`） | 同上 |

`grp_gu == true` 时 **跳过** `expert_gate_up_fp4_batched`（`:11577` 的 `if !grp_gu`）——这是
“armed gate 绝不能静默测旧路径”的硬约束。

### 1.2 被调者一侧（方法内 7 道门，`chain_dev.rs:11129-11186`）

| # | 条件 | 位置 | arm 取值 | 备注 |
|---|---|---|---|---|
| 1 | `expert_tcgen05_e4m3()` | `:11129` | `DSV41_EXPERT_TCGEN05_E4M3=1` | **严格 `starts_with('1')`**（`weights.rs:756-763`） |
| 2 | `e4m3` | `:11138` | `DSV41_EXPERT_ACT_E4M3=1` + 符号在 | 关它 = 换激活格式（多一个变量） |
| 3 | `!gateup_fused` | `:11146` | `DSV41_GATEUP_FUSE=0` | e4x epilogue 只 clamp，不融合 swiglu |
| 4 | `dim % 64 == 0 && 2*inter_local % 64 == 0` | `:11155` | TP1: 5120 / 4608 ✓；TP8: 5120 / 640 ✓ | 形状契约 |
| 5 | `!ld.experts_ilv` | `:11163` | `DSV41_EXPERT_ILV=0` | grouped 臂**硬前置** |
| 6 | `supports_expert_gemm_e4m3_grouped()` | `:11171` | `.so` 有符号 | `device.rs:5262` |
| 7 | `ld.experts.len() >= 2` | `:11179` | ✓ | 用于推导 per-expert stride |

**旁证（判读陷阱）**：`e4x_tile = false` 是**硬编码**（`chain_dev.rs:11523`），所以 dense
`dsv41_expert_gemm_e4m3_ext` 在 `moe_rows` 里**永远不发**，只会打一条
`tcgen05_e4m3_ext_skipped_note`（`:11524-11539`）。**这条告警与 grouped 是否成功无关**，
不能用来推断 grouped 也 decline（都是 `OnceLock`，one-shot）。

**结论**：`DSV41_EXPERT_TCGEN05_E4M3=1` 一门武装**两个**未验证 kernel：
- prefill / eager decode 的 `moe()`（单行）→ swapAB `dsv41_expert_tcgen05_gate_up_e4m3`（`:14571`）；
- spec verify 的 `moe_rows()`（多行）→ grouped `dsv41_expert_gemm_e4m3_grouped`（`:11208`）。
**ATTRIBUTION 必须按 kernel 分开**（见 §3）。

---

## 2. ILV=0 的权重加载：**完全可用，且就是 checkpoint 的原生布局**

**答复：不存在“checkpoint 只有交错版”的问题。交错是 ferrite 自己的 load-time 变换，不是 checkpoint 属性。**

- `load_expert_pool`（`load.rs:603-748`）按名字读 checkpoint 的 **6 个独立平面**：
  `w1.weight / w1.scale / w3.weight / w3.scale / w2.weight / w2.scale`（`:612-614`）。
- **`ilv == false`**（`:706-715`）：对 6 个平面各做一次 `dma_plan`，落到 `poff[k]`（顺序累积，
  `:657-664`）——即 `[w1][w1.scale][w3][w3.scale][w2][w2.scale]`。**这是 ILV 引入之前的历史基线
  布局 = checkpoint 原样。**
- **`ilv == true`**（`:683-705`）：把 w1/w3 先 DMA 进 scratch，再 `interleave_gateup_fp4` 生成
  “一个区域装两个平面”的交错区；w3 的 view **别名** w1 区间（`:720-724`）。
- `ilv_ok()`（`load.rs:767-778`）是**唯一**布局决策点，且把 `gateup_fuse()` 作为**合取项**
  ⇒ **`DSV41_GATEUP_FUSE=0` 本身就强制 `ilv=false`**；`DSV41_EXPERT_ILV=0` 是 belt-and-braces
  （`weights.rs:487-490` 的 `gateup_ilv()` 只是其中一项）。
- 代码代价：ILV 实测仅 −0.09ms，故 ILV=0 ≈ **+0.09ms**。

**同时要确认的（真问题）**：grouped 臂同时传 `w1_base/w1_stride` **和** `w3_base/w3_stride`
（`chain_dev.rs:11192-11232`，`b_split = inter_local`），所以 **`ilv=false` 下两个平面必须是真实
独立区域**（ILV=1 时 w3 是别名，grouped 臂读它会读错——这正是第 5 条 decline 的由来）。ILV=0 天然满足。

---

## 3. `ld_uint2_a8` 守卫修复的位置核对（**含反直觉发现**）

### 3.1 修复内容（commit `ed720d6`，+36/−6）

- 新增 `ld_uint2_a8`（`:170-176`），8 字节版 `ld_uint4_a16`；对齐则明文 load，否则 byte-wise
  `__builtin_memcpy`（字节**完全相同**，只去掉 fault）。
- 应用到 **`expert_gemv_fp4_batched_kernel`（SIMT 回退）** 的 **配对体** 直读点：
  `:1672`（uint4/ILV）、`:1677`（uint2/!ILV）、`:1703`、`:1706`、`:1708`、`:1754`。
- 同一 commit 里还有 SH_PAIR 的编译修复（与本主题无关）。

### 3.2 反直觉发现：**冒烟臂不执行这些行**

```
pair_body = ((fuse_swiglu != 0) || ILV) && (b_split > 0)     //  :1405
```

| 配置 | `fuse` | `ILV` | `pair_body` | 实跑体 | 权重读 |
|---|---|---|---|---|---|
| **冒烟臂**（`GATEUP_FUSE=0`, `ILV=0`） | 0 | 0 | **false** | **split body** | `uint32` 4B（`:1873/:1874`）/ `uint16`（`:1944`） |
| 生产默认（`GATEUP_FUSE=1`, `ILV=1`） | 1 | 1 | true | pair body | `uint4` 16B（`:1672`）← **本次修复的受益者** |
| `GATEUP_FUSE=1`, `ILV=0` | 1 | 0 | true | pair body | `uint2` 8B（`:1677/:1706/:1708/:1754`）← **本次修复的受益者** |

launcher 侧的 `fuse` 绑定：`:2666-2667` `fuse = (g_fuse && g_expert_fp4_mode == 2 &&
(dim % 512) == 0 && out_slot_stride == inter)`；`moe_rows` 在 `gateup_fused == false` 时传
`act_slot = 2*inter_local`（`chain_dev.rs:11615-11619` 口径），故 `fuse = 0`。

**⇒ 本修复覆盖的是“ILV=1 或 GATEUP_FUSE=1”的配置（即生产默认臂），而不是冒烟臂。**
冒烟臂那条 split body 的 4B 读本身是天然对齐安全的（`brow = bb + r*kbytes`，
`kbytes = dim/2 = 2560`，`(g<<8) + (lane<<3)` 均为 8 的倍数）。

**这条必须进测试矩阵**：只跑 `GATEUP_FUSE=0/ILV=0` 无法验证这次修复；要验证修复必须**额外**跑
`GATEUP_FUSE=1/ILV=1`（或 `GATEUP_FUSE=1/ILV=0`）这两个“配对体”配置。

---

## 4. 正确重测方案（命令 + 预期 + 诊断）

### 4.1 前置共识

1. **harness 事实**：
   - `dsv41-run --tp 1` 走 `chain.step()` + `chain.step_dev()`（`dsv41-run.rs:189/210`），**不构造
     DsparkDev、不调 `step_rows`** ⇒ **只能测 swapAB 臂，永远到不了 grouped 臂**。
   - grouped 臂只在 `moe_rows` 里，`moe_rows` 只被 `layer_rows`→`step_rows` 调用 ⇒ **只有
     serve（`DSV41_SPEC=1 DSV41_DSPARK=1`）或直接调 `DevChain::step_rows()`（`chain_dev.rs:5275`，`pub`）
     能到**。零改动路径 = serve。
   - `.so` 只在 GPU 节点上（本地树无 `*.so`，无 `target/release/{ferrite-serve,dsv41-run}`）；
     节点默认 `ubuntu@43.202.208.136`，仓库 `~/ferrite`（`tcgen05_smoke.sh:90/159-160`）。
2. **gate 前提链（arm）**：
   `DSV41_EXPERT_ACT_E4M3=1 DSV41_EXPERT_TCGEN05_E4M3=1 DSV41_EXPERT_GROUPED=1
    DSV41_GATEUP_FUSE=0 DSV41_EXPERT_ILV=0`
   \+ `DSV41_SPEC=1 DSV41_DSPARK=1`（否则 decode 走单行 `moe()`，武装的是**另一个** kernel）
   \+ **`DSV41_NO_GEMV_FP4` 必须不存在**（bare getenv，`:696-699`）。
   两个 `starts_with('1')` 的门**不能写 `=true`/`=on`**（写成别的值等于 OFF）。

### 4.2 Phase 0（无 GPU）：symbol + 双产物 precheck

```bash
bash scripts/tcgen05_smoke.sh --dry-run        # stage 0 可达性 + stage 1 五符号
```
**预期**：`dsv41_expert_act_e4m3_cap` / `dsv41_expert_gemm_e4m3_grouped` / `dsv41_route_group` /
`dsv41_route_gather_rows` / `dsv41_route_scatter_rows` **五个全 PASS**（exit 0）。
任一缺失 ⇒ **停**，先在节点上按双产物纪律重编：`bash build.sh 103a` + `touch
crates/ferrite-kernel/build.rs && cargo build --release`。**一个 GPU 都不该花。**

### 4.3 Phase 1（GPU）：逐轮隔离，每轮**必须**带正证据

**在每组命令前统一设**（首轮必须）：
```bash
export CUDA_LAUNCH_BLOCKING=1        # 异步 fault 默认 sticky，会在“下一次 sync”才报 → 必须开
export DSV41_VERIFY_GRAPH=0 DSV41_GRAPH_MOE=0   # 显式关图，去掉“只记录不执行/静默降级”
```

#### R0 —— 基线（无任何 tcgen05 门，`GATEUP_FUSE`/`ILV` 都不设）
**回答**：引擎在本树、本机能不能起 + 出正确文本。
```bash
ssh ubuntu@43.202.208.136 'cd ~/ferrite && pkill -9 -x ferrite-serve; sleep 8; true'
ssh ubuntu@43.202.208.136 'cd ~/ferrite && nohup env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  LD_LIBRARY_PATH=$HOME/ferrite/kernels/cuda \
  DSV41_KERNELS=$HOME/ferrite/kernels/cuda/libferrite_kernels.so \
  CUDA_LAUNCH_BLOCKING=1 DSV41_VERIFY_GRAPH=0 DSV41_GRAPH_MOE=0 \
  DSV41_SPEC=1 DSV41_DSPARK=1 DSV41_EXPERT_ACT_E4M3=1 DSV41_TIMING=1 \
  ./target/release/ferrite-serve --model dsv41 --serve --tp 8 \
  --model-dir /opt/dlami/nvme/models/DeepSeek-V4.1-Flash --port 8699 \
  > ~/tc5_r0.log 2>&1 &'
# 等 /health → 发请求（出师表 100 tok）→ 取文本
```
**PASS**：进程活着、文本可读（零拉丁、无相邻重复）、`has_kaishen=yes`。
**R0 就崩 ⇒ 引擎/权重加载问题，与 tcgen05 无关，后续所有归因作废。**

#### R1 —— 布局税 + 守卫覆盖矩阵（**无 tcgen05 门**）
这是**本方案新增的关键一轮**，必须跑**三个变体**（因为 §3.2）：

| 变体 | env | 走哪个 kernel / 哪个 body | 守不守 |
|---|---|---|---|
| **R1a** | `GATEUP_FUSE=0 ILV=0` | SIMT **split body**（4B 读） | **不受本次修复影响** |
| **R1b** | `GATEUP_FUSE=1 ILV=0` | SIMT **pair body**（uint2 读） | **本次修复的直接受益者** |
| **R1c** | `GATEUP_FUSE=1 ILV=1`（生产默认） | SIMT **pair body**（uint4 读） | **本次修复的直接受益者** |

```bash
# 例：R1a（把 ARM 门去掉，只留布局门）
ssh ubuntu@43.202.208.136 'cd ~/ferrite && nohup env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  LD_LIBRARY_PATH=$HOME/ferrite/kernels/cuda \
  DSV41_KERNELS=$HOME/ferrite/kernels/cuda/libferrite_kernels.so \
  CUDA_LAUNCH_BLOCKING=1 DSV41_VERIFY_GRAPH=0 DSV41_GRAPH_MOE=0 \
  DSV41_SPEC=1 DSV41_DSPARK=1 DSV41_EXPERT_ACT_E4M3=1 DSV41_TIMING=1 \
  DSV41_GATEUP_FUSE=0 DSV41_EXPERT_ILV=0 \
  ./target/release/ferrite-serve --model dsv41 --serve --tp 8 \
  --model-dir /opt/dlami/nvme/models/DeepSeek-V4.1-Flash --port 8699 > ~/tc5_r1a.log 2>&1 &'
```
**判读**：
- **R1a 崩**（misaligned/fault）⇒ SIMT split body 的对齐问题，**与本次修复无关**（修复打在 pair body），
  按 `CUDA_LAUNCH_BLOCKING` 报出的 kernel 名另立修复项。**这一档若不修，R2 的失败归因不清。**
- **R1b / R1c 崩** ⇒ 守卫修复没盖住（下一批嫌疑点：`vec==3` 的 `uint16`（`:1944`）、
  `s_act` 的 `float4`（`:1946`）等），按 fault 归属继续查。
- **R1 全 PASS** ⇒ 门税 + SIMT 回退自证，进入 R2。

#### R2 —— arm 轮（**目标**）：分组臂 + nsys 正证据
```bash
ssh ubuntu@43.202.208.136 'cd ~/ferrite && pkill -9 -x ferrite-serve; sleep 8; true'
ssh ubuntu@43.202.208.136 'cd ~/ferrite && nohup env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  LD_LIBRARY_PATH=$HOME/ferrite/kernels/cuda \
  DSV41_KERNELS=$HOME/ferrite/kernels/cuda/libferrite_kernels.so \
  CUDA_LAUNCH_BLOCKING=1 DSV41_VERIFY_GRAPH=0 DSV41_GRAPH_MOE=0 DSV41_TIMING=1 \
  DSV41_SPEC=1 DSV41_DSPARK=1 DSV41_EXPERT_ACT_E4M3=1 DSV41_MOE_BATCH=1 \
  DSV41_EXPERT_TCGEN05_E4M3=1 DSV41_EXPERT_GROUPED=1 \
  DSV41_GATEUP_FUSE=0 DSV41_EXPERT_ILV=0 \
  nsys profile -t cuda --stats=false -f true -o /tmp/tc5_r2 \
  ./target/release/ferrite-serve --model dsv41 --serve --tp 8 \
  --model-dir /opt/dlami/nvme/models/DeepSeek-V4.1-Flash --port 8699 \
  > ~/tc5_r2.log 2>&1 &'
# 等 /health → 发"你好" 20 tok（抓 crash）→ 再发"出师表" 100 tok（数值）
# 收证据后：
ssh ubuntu@43.202.208.136 'nsys stats --report cuda_gpu_kern_sum /tmp/tc5_r2.nsys-rep'
```

**R2 的 PASS 判据（必须同时满足，缺一 = 空洞）**：
1. **L2/L3 正证据**：nsys 的 kernel 表里 **`e4m3_gemm_grouped_kernel` 调用数 > 0**
   （这是 grouped 臂的真身，`:5831`）。**仅当 prefill 也跑了，才应同时看到 `e4m3_gemm_kernel`
   （swapAB 臂，`:5444`）—— 两个 kernel 分开计数，这就是 attribution。**
   - ⚠️ `e4m3_gemm_kernel` 在 verify 路径上**恒为 0**（`e4x_tile=false`，`:11523`）；它若出现，
     说明 prefill 的 swapAB 也跑了，属正常（prefill 必经），但要分清是 prefill 还是 verify 调用的。
2. **无 decline 告警**（`tcgen05_smoke.sh:313-314` 的 4 条 exact substring 一条都不出现）。
3. **无 fault**：进程活着（`/health` OK）+ 日志无 `illegal|fault|CUDA error|panic|abort`。
4. **数值**：首 10 字与 R0/基线一致（MMA 与 SIMT 的 f32 求和次序不同，只比前缀，**不比全字节**）。

### 4.4 正证据的三种实现（按侵入度排序）

| 方案 | 做法 | 侵入 | 结论 |
|---|---|---|---|
| **A（推荐，金标准）** | R2 里 `nsys profile`，`grep cuda_gpu_kern_sum` 数 `e4m3_gemm_grouped_kernel` | **零改动** | kernel 没跑就是**没有行**，不可能空洞 |
| **B** | `.cu` wrapper（`:6120`）加 env-gated 一次性 launch 打印（`DSV41_TCGEN05_TRACE=1`） | ~10 行源码，**需尚书省批准** | 证据落在冒烟脚本本就 grep 的 serve 日志里；是 **L2**（launcher 跑了），需 A/C 交叉 |
| **C** | `e4m3_gemm_grouped_kernel` 加 launch counter 形参 + D2H 读回 | 改 ABI，需同步 `kernels.rs`/`device.rs` FFI | **不上**，A+B 足够 |

### 4.5 失败隔离（几乎免费，直接终结“fault 出在哪”）

1. **`CUDA_LAUNCH_BLOCKING=1`**（首轮**必须**）：让报错归到**真正 fault 的那次 launch**。
2. **`compute-sanitizer --tool memcheck --launch-timeout 120 ./target/release/ferrite-serve …`**
   （首轮**建议**）：直接给 kernel 名 + 出错指令 + 地址（慢 10-100x，但这是一次性首触，值）。
3. **`DSV41_VERIFY_GRAPH=0 DSV41_GRAPH_MOE=0`** 显式写死：去掉 capture 带来的执行路径变量。

---

## 5. 判读表（每轮跑完必做）

| 观测 | 解释 | 动作 |
|---|---|---|
| R0 就崩 | 引擎/权重加载/集合通信问题，**与 tcgen05 无关** | 修引擎；本轮作废 |
| **R1a 崩** | SIMT **split body** 的对齐问题（**本次修复不覆盖**） | **另立修复项**，别记在 tcgen05 账上 |
| **R1b/R1c 崩** | 守卫修复未盖全（下一批嫌疑：`:1944` uint16 / `:1946` float4） | 按 `CUDA_LAUNCH_BLOCKING` 报出的 kernel 继续查 |
| R2 崩 + **有** launch 证据 + `CUDA_LAUNCH_BLOCKING` 指向该 kernel | tcgen05 kernel 内真有对齐/寻址错 | compute-sanitizer 定位；kernel 级修复 |
| R2 崩 + **无** launch 证据 | kernel **没被调用** | 查 decline 告警 / gate / 符号；**不要**报“kernel 崩了” |
| **R2 不崩 + 无 launch 证据** | **空洞！**（旧测试的失败模式） | 判 FAIL：arm 没生效 |
| R2 不崩 + 有 launch 证据 + 文本可读 + 首 10 字同基线 | 真 PASS（冒烟门） | 进性能 A/B |
| R2 不崩 + 有 launch 证据 + 首字即分叉 | 数值错（两条 `[OPEN]`：dense idesc format code / fp4 packed-vs-unpacked） | 不 ship；按 expectation §7 第 4 步二分 |

---

## 6. 如果 kernel 没被调用：诊断方法（按顺序）

1. **decline 告警**（最直接）：
   ```bash
   ssh ubuntu@43.202.208.136 "grep -E '^warning: DSV41_EXPERT_' ~/tc5_r2.log"
   ```
   4 条 one-shot decline 各带**原因**，直接指到 §1.2 的哪一道门：
   `DSV41_EXPERT_GROUPED is set, but …` / `…TCGEN05_E4M3 is set, but the routed MoE still dispatches`
   / `…stays on the batched` / `DSV41_EXPERT_ACT_E4M3 is set, but the routed experts still run the`。
2. **env 是否真的进了进程**（不是只进了 shell）：
   ```bash
   ssh ubuntu@43.202.208.136 "tr '\0' '\n' < /proc/\$(pgrep -x ferrite-serve | head -1)/environ | grep -E '^DSV41_' | sort"
   ```
   逐门核对 §4.1-2；注意 `=true`/`=on` 对两个 `starts_with('1')` 的门等于 OFF。
3. **符号是否在 .so**：
   ```bash
   ssh ubuntu@43.202.208.136 "nm -D --defined-only \$HOME/ferrite/kernels/cuda/libferrite_kernels.so | grep -E 'expert_gemm_e4m3_grouped|route_group|expert_tcgen05_gate_up_e4m3'"
   ```
   缺 ⇒ 重编（`build.sh` 的 e4m3 skeleton **默认 ON**）。
4. **双产物一致性**：`cat kernels/cuda/.build_id` 与 Rust 侧期望一致（不一致进程**会拒绝启动**）。
5. **nsys kernel 表反查“到底跑了什么”**：
   ```bash
   ssh ubuntu@43.202.208.136 'nsys stats --report cuda_gpu_kern_sum /tmp/tc5_r2.nsys-rep | head -40'
   ```
   `e4m3_gemm_grouped_kernel` 计数为 0 而 `expert_gemv_fp4_batched_kernel` 计数 > 0
   ⇒ **arm 没生效，SIMT 回退在答题**（这就是空洞的指纹）。
6. **确认真的走到了 verify 路径**：日志里 `[gmo]`/`[phs]` 出现 = 判定窗口内跑过单行 `moe()`；
   若全程只有 `[gmo]`、没有 verify 的多行痕迹，说明 spec/dspark 没武装，grouped 臂根本没入口。
7. **请求是否真的到达引擎**：`/health` OK **不证明**引擎 load 过——`ensure_loaded()` 只在首次
   prefill 里调（`serve.rs:204/267`）。要确认 load 完成看日志的 `[dsv41] tp pool ready: N ranks loaded`
   （`serve.rs:222`）。缺这行 ⇒ 请求还没真正进引擎。

---

## 7. 执行顺序（一页）

```
Phase 0  无 GPU
  bash scripts/tcgen05_smoke.sh --dry-run        → 五符号齐 + 双产物一致
Phase 1  GPU（每轮 CUDA_LAUNCH_BLOCKING=1 + VERIFY_GRAPH=0/GRAPH_MOE=0）
  R0   基线（无 tcgen05 门）                      → 引擎自证（必须 PASS，否则归因无效）
  R1a  GATEUP_FUSE=0 ILV=0（无 tcgen05）          → split body（本次修复不覆盖）
  R1b  GATEUP_FUSE=1 ILV=0（无 tcgen05）          → pair body/uint2（修复受益者）
  R1c  GATEUP_FUSE=1 ILV=1（无 tcgen05）          → pair body/uint4（修复受益者＝生产默认）
  R2   R1a + TCGEN05_E4M3=1 GROUPED=1 + nsys      → 目标；证据 = grouped kernel 计数>0
首轮一律: CUDA_LAUNCH_BLOCKING=1 (+ compute-sanitizer memcheck)
⚠️ 计时与 profiling 不要同轮（nsys/sanitizer 污染 ms 读数）
Phase 2  只有冒烟 PASS 才做性能 A/B（scripts/batched_400_v2.sh / tcgen05_bench.sh）
```

**一句话**：`ld_uint2_a8` 修复的是 **pair body**（ILV=1 或 GATEUP_FUSE=1 时才执行），
而冒烟臂跑的 **split body 不受它保护**；再加上“成功静默”，所以这次重测**必须**
先跑 R1a/R1b/R1c 把“哪条 body 在跑、守没守住”钉死，再在 R2 上用 **nsys 的
`e4m3_gemm_grouped_kernel` 计数 > 0** 作为唯一有效的 PASS 门槛。

---

## 8. 需要上报/批准的「方案外」项

1. **方案 B 的 `.cu` 改动**（~10 行 env-gated launch 打印）——需批准后再动。
2. **`crates/ferrite-dsv41/tests/real_grouped_tcgen05.rs`**（直接调 `DevChain::step_rows()` 的
   单卡隔离 harness）：范围要定（加载全模型慢 / 只构造一层 MoE 权重要写脚手架）。
3. **若 R1a 复现 misaligned** ⇒ SIMT **split body** 的守卫修复是**独立一件工程**，单独立项，
   不要与 tcgen05 混在一个 verdict 里。
4. **若 R1b/R1c 复现** ⇒ 说明本次修复的覆盖不全，需按 `CUDA_LAUNCH_BLOCKING` 报出的点补守卫。

---

*工部 · 只读调查 + 本文件（唯一产出）；未执行任何 GPU 命令、未改动任何源码。*
*`pair_body` 判定、`fuse` 绑定、`vec==2` 读宽等均已对工作树 HEAD 现场核对；
新提议的 `DSV41_TCGEN05_TRACE`、`tests/real_grouped_tcgen05.rs` 明确标注为“待批准”。*
