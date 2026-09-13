# `dsv41_expert_tcgen05_gate_up_e4m3` err 716 —— 根因与阻断点判决（重启版）

> 工部 · 2026-09-13 · **只读调查 + 一处 .cu 修复**。未执行任何 GPU/e2e；远端 b300-2 `nvcc -c` compile-only 通过。
> 现场核对：`kernels/cuda/dsv41_experts_mxf4.cu`（HEAD 4593c8e + 本次修复）、
> `crates/ferrite-models/src/dsv41/{chain_dev.rs,device.rs,load.rs,weights.rs}`、
> `crates/ferrite-dsv41/src/serve.rs`、`kernels/cuda/tests_tcgen05_misalign_repro.cu`。
> 上游：`docs/agent/tcgen05-rank7-verdict.md` §8-§11、`docs/agent/verify-amortization-lesion-audit.md` §10。

---

## 0. 一句话判词

`DSV41_EXPERT_TCGEN05_E4M3` 是**一个名字**，却武装**两个互相独立的 arm**：

| 消费者 | 符号 | 什么时候跑 |
|---|---|---|
| 单行 swapAB e4m3（`tc5::e4`） | `dsv41_expert_tcgen05_gate_up_e4m3` | **prefill 的每一层**（`prefill_chain` → `chain.step` → `layer()` → `moe()`）|
| grouped masked e4m3（`tc5::e4x`） | `dsv41_expert_gemm_e4m3_grouped` | **verify**（`moe_rows` → `moe_experts_grouped_gate_up`）|

ticket 要量的 grouped arm 只在 verify 里跑；而**同一个门先把单行 swapAB arm 在 prefill 里武装了**。那个 arm **从未在任何 GPU 上执行过**（`tcgen05-e4m3-grouped-expectation.md` §9："从未在任何 GPU 上执行过"），它的第一次真实执行就是 prefill —— 于是 prefill 死在 err 716，`moe_rows` 一次都没走到。

**⇒ "MoE grouped -1.5~2.5ms 被 716 卡住"的真实结构：不是 grouped arm 有 716，而是它的兄弟 arm 在它前面把 prefill 打死了。**
修复 = 把两个 arm 的门拆开（本文件 §4）。

---

## 1. 调用链更正（先修一处流传的误诊）

线索里写的「`moe_experts_grouped_gate_up`（:14600-14700）→ `dsv41_expert_tcgen05_gate_up_e4m3`」**不成立**。
现场核对：

- `moe_experts_grouped_gate_up`（`chain_dev.rs:14308`，在 **`moe_rows`** 内调用）→
  `self.dev.expert_gemm_e4m3_grouped(...)`（:14402）→ `device.rs:6492` →
  **`dsv41_expert_gemm_e4m3_grouped`**（`.cu:6912`，`e4m3_gemm_grouped_kernel`）。
- `dsv41_expert_tcgen05_gate_up_e4m3` 在本仓库**只有唯一一个调用点**：
  `chain_dev.rs:18157`，位于 **`fn moe()`（:17738）** 内 —— 单行 backbone 路径。
  该入口的 `.cu` 侧是 :5927 的 `extern "C"` → `e4_launch_gateup`（:5874）→
  `expert_tcgen05_gateup_e4_kernel`（:5567）。

`moe()` 只在单行路径被调用（`chain_dev.rs:16327/:16339`，函数 `layer()` :16081 ← `step_body()` :6496 ←
`step()` :6349 ← `prefill_chain()` `serve.rs:876-883`：**每个 prompt token 一次 forward**）。
⇒ **err 716 发生在 PREFILL**，在任何 verify / grouped 调用之前。

## 2. 为什么 716 不可能来自「被门检查过的那组基址」

这是本次调查最硬的一条判据（值级）：

`e4_launch_gateup` 的门（`.cu:5897-5904`）检查 `act/w1/w1s/w3/w3s` 的**基址与 per-expert 步长**，
不通过时 `return cudaErrorInvalidValue`（=**1**）。观测到的是 **716**。
⇒ **被门覆盖的那组指针没有任何违反**（否则日志会是 `cuda error 1`，Rust 侧表现为「arm 静默 decline」）。

生产值（`cfg.dim=5120`、`moe_inter_dim=2304`、`world=8`、`topk=6`）：

| 量 | 值 | %16 | 依据 |
|---|---|---|---|
| `inter_local = padded_inter(2304/8=288)` | **320** | 0 | `weights.rs:452`（K_ATOM=64）|
| `rows = 2*inter_local` / `split` | 640 / 320 | 0 | `chain_dev.rs:17753`、`.cu:5886/5890` |
| `kbytes = dim>>1`（权重行距）| 2560 | **0** | `.cu:5602`，与 ALIGN_AUDIT `pitch=2560` 一致 |
| `nsf = dim>>5`（scale 行距）| 160 | **0** | `.cu:5603`，与 ALIGN_AUDIT `pitch=160` 一致 |
| `pk0 = g*(kPackK/2)` / `st*16` / `16*kb` | 16 的倍数 | 0 | `.cu:5644/5657/5666` |
| A 侧 TMA src | `w1p/w3p + row*kbytes + pk0` | **0** | 基址与步长过门 |
| B 侧 TMA src | `act + g*kPackK + 16*kb` | **0** | `act` 过门（`xq4` cudaMalloc） |
| SF 序言 src | `w1sp/w3sp + rr*nsf + i*16` | — | `ld_uint4_a16`（`.cu:5711`，字节相等，永不 fault） |
| smem 目的侧 | `a_raw/a_op/b_op/sf_stage` 偏移 + `e4_off ≡ 0 (mod 16)` | **0** | `.cu:5379-5388` 的 static_assert |

未过门、且随后的读点：`act_scale`（f32，4B 读）、`out_s[row]`（f32 写）、`ids[slot]`（i32 读）——
三者的宿主缓冲 `xsc4` / `ex_act_b` / `route_idx` 全部来自 `dev.alloc`（cudaMalloc，≥256B 对齐），
**构造上不可能不齐**。

⇒ **该 kernel 的 global/smem 操作数寻址在冻结形状下是 16B-clean 的**；
残下的 fault 只可能在「门看不见、也无法字节化」的那一类：tcgen05 的 **smem descriptor / TMEM SF 地址 / mbarrier**。
这一类**没有 host 侧判据**，只能靠 `compute-sanitizer` 报 SASS 指令定位（§5）。

## 3. 判决：阻断点已定，kernel 内的具体指令待 sanitizer

1. **阻断点（确凿）**：单行 swapAB arm 与 grouped arm 共用一个 env 名，
   而单行 arm 在 prefill 里先跑并 fault。⇒ grouped ticket **结构上无法被测量**。
2. **716 的宿主 kernel（确凿）**：`expert_tcgen05_gateup_e4_kernel`（`.cu:5567`），
   调用点 `chain_dev.rs:18157`。8/8 rank 同文本 = rank 对称（形状/步长类），与「prefill 固定形状」一致。
3. **716 的具体违反点（未定）**：§2 已排除全部被门覆盖的操作数与全部内部偏移；
   剩余候选只有 tcgen05 侧三类（descriptor / SF TMEM / mbarrier），
   它们的地址在 host 侧不可见，**本次 compile-only 调查无法再缩小**。§5 给出一步到位的定位手册。
4. **与 SF 根修（`w2.scale` pitch 10→16）的关系**：无关，且此前已判
   （`tcgen05-rank7-verdict.md` §10 的"唯一 violation = w2.scale"对 gate/up 是误诊 ——
   gate/up 的 B 侧是 `w1/w3`，scale 行距 `dim/32 = 160` **本来就是 16B 倍数**）。

## 4. 修复（本次交付，`.cu` 单侧，默认行为=证明过的路径）

**一处改动**：`kernels/cuda/dsv41_experts_mxf4.cu` 的
`dsv41_expert_tcgen05_gate_up_e4m3`（:5927）入口，在共享门之后加一道**专属门**
`DSV41_EXPERT_TCGEN05_E4M3_SWAPAB`（默认 OFF，严格 `starts_with('1')`，进程内读一次）：

- **共享名 `DSV41_EXPERT_TCGEN05_E4M3` 现在只武装 grouped consumer**
  （`dsv41_expert_gemm_e4m3_grouped` 的 `enabled` lambda 自己 AND `DSV41_EXPERT_GROUPED`，未改）。
- 本符号在专属门 OFF 时 `return 0` —— 这是该入口**文档化的 fallback 契约**
  （`.cu:5917-5918`："Returns 0 (and does nothing) while disabled, so the caller can call it
  unconditionally and keep the proven GEMV path as the fallback"）⇒ 单行路径落到**证明过的 SIMT GEMV**，
  与 pre-tcgen05 行为逐位相同。
- 降级**响亮**：一次性 stderr note（armed-gate 不许静默测旧路径的纪律）。

**为什么不改 Rust 镜像**：`chain_dev.rs::expert_tcgen05_e4m3()` 同时是 **grouped arm 的门**
（`chain_dev.rs:14323`：`if !expert_tcgen05_e4m3() { ... return Ok(false) }`）。
改它会把 grouped arm 一起关掉 —— 正是要避免的。
单行侧的"Rust 认为 arm 在跑 / `.so` 返回 0"这一处**故意的不一致**：`ran_tc = Ok(false)`
走的就是上面那条 fallback 分支，并有 `.so` 侧 note 可读，因此不构成"armed 却静默测旧路径"。

**数值影响**：零。被测量的 arm（grouped）一行未动；单行路径回落到既有 SIMT GEMV。

**编译**：`cargo check --workspace --all-targets` EXIT=0；
远端 `nvcc -O3 -std=c++17 -c -gencode arch=compute_103a,code=sm_103a`
（`-DDSV41_TCGEN05_GATEUP_E4M3_SKELETON=1 -DDSV41_TCGEN05_GATEUP_MXF4_SKELETON=1`）**EXIT=0**。

## 5. GPU 验证手册

**A. 主验收（grouped arm 解锁，双门禁）**

```bash
# 双产物：动了 .cu ⇒ 必须两边都重编
git fetch && git reset --hard origin/main && cd kernels/cuda && bash build.sh 103a
cd ~/ferrite && cargo build --release

# gate 链（权威出处 tcgen05-retest-after-guardfix.md §4.1；两个 starts_with('1') 的门不能写 =true）
DSV41_SPEC=1 DSV41_DSPARK=1 DSV41_EXPERT_ACT_E4M3=1 DSV41_EXPERT_TCGEN05_E4M3=1 \
DSV41_EXPERT_GROUPED=1 DSV41_EXPERT_GROUPED_DOWN=1 DSV41_GATEUP_FUSE=0 DSV41_EXPERT_ILV=0 \
DSV41_MOE_BATCH=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
LD_LIBRARY_PATH=$HOME/ferrite/kernels/cuda \
./target/release/ferrite-serve --model dsv41 --serve --tp 8 --port 8712
```

判据（缺一即空洞）：
1. **`[arm] ... dsv41_expert_tcgen05_gate_up_e4m3 (swapAB e4m3 ...) is OPT-IN and OFF`** 一行在
   —— 证明新门进了进程（env 回读纪律），且**不再有 `716`**。
2. `[tp] ... err:` **0 行**（8/8 rank 无 Err）；`/health` OK；无 `illegal|fault|CUDA error|panic`。
3. **正证据**：`nsys profile -t cuda --stats=false` + `nsys stats --report cuda_gpu_kern_sum`
   里 **`e4m3_gemm_grouped_kernel` 调用数 > 0**（该 arm 成功时零打印，正证据只能来自 kernel 计数）；
   同时 **`expert_tcgen05_gateup_e4_kernel` 调用数 = 0**（新门生效的直接证据）。
4. 门禁双门：`step_ms`（[dspark] 分解）**AND** `mean-k`（基线 2.240）。
   票面：verify **−1.5 ~ −2.5ms**、mean-k 2.24 不降。
5. 文本红线：计数前 61 行 + 零拉丁（`scripts/tcgen05_smoke.sh` STAGE 2/3 已实现）。
6. **逃生门**：把 `DSV41_EXPERT_TCGEN05_E4M3_SWAPAB=1` 加上，716 应**立刻复现**
   —— 这是"716 确实属于单行 arm"的反证实验（1 次进程，秒级判定）。

**B. 定位 716 的具体指令（可选，独立 harness，比 serve 便宜两个数量级）**

`kernels/cuda/tests_tcgen05_misalign_repro.cu` 已就绪，但它的 e4 case 用的是 `inter=2048`；
**要用真实生产形状**（`dim=5120, inter=320, slots=6`，即 `split=320/rows=640`）：

```bash
nvcc -gencode arch=compute_103a,code=sm_103a -O2 -std=c++17 \
     -DDSV41_TCGEN05_GATEUP_MXF4_SKELETON=1 -DDSV41_TCGEN05_GATEUP_E4M3_SKELETON=1 \
     -o /tmp/t_misalign kernels/cuda/tests_tcgen05_misalign_repro.cu
# 一 case 一进程（fault 毒化 context，串跑会造假 rc）
CUDA_VISIBLE_DEVICES=<free> compute-sanitizer --tool memcheck --launch-timeout 0 \
     --destroy-on-device-error kernel --show-backtrace device --log-file /tmp/san.txt \
     /tmp/t_misalign --case <e4/direct/prod-shape 的编号，需先把 inter 改成 320>
```
报告读法 → 修法：
- `cp.async.bulk`（`CPASYNC…BULK`）⇒ **A/B 侧 TMA 源或 smem 目的的 16B**（§2 已排除基址，
  则剩下的是 smem 目的或 `e4_off` —— 改 smem 摆位或加 static_assert 钉死）；
- `LDG.E.128/E.64` ⇒ 补 `ld_uint4_a16` / `ld_uint2_a8` 系；
- `STS`/descriptor ⇒ smem 侧摆位；
- `MMA`/TMEM ⇒ **SF TMEM 列地址**（`sfa_col + 4*(b>>2)`，`.cu:5800`）或 descriptor 的
  `start = (smem_base>>4) & 0x3FFF`（`.cu:5409`）—— 这一类目前**无 host 判据**。

## 6. 遗留 / 需要上报

1. **编辑区越界（照实上报）**：任务书允许我改 `chain_dev.rs` 的 grouped 传参区
   （:14600-14700）。本次修复**不需要**改 Rust（见 §4），因此 `chain_dev.rs` **零改动** ——
   唯一的 diff 在允许的 `kernels/cuda/dsv41_experts_mxf4.cu` 内。无冲突。
2. **A 侧 TMA 复制粒度为 16 B**（`.cu:5656`）：不是对齐 bug，但一笔 16 B 的
   `cp.async.bulk` 只覆盖半个 32 B DRAM sector；若日后要调 K 原子，这里是第一个该看的效率点。
3. **`tc5::mxf4`（e2m1 swapAB）的 `act_scale` 读法可疑**（`.cu:4276` 把 f32 尺度当 packed e8m0
   `uint32` 读），而 e4 arm 是 f32 读后 `f_pow2_to_ue8m0` 转换 —— 两个 swapAB arm 的
   activation-scale ABI **不一致**。本条与 716 无关（不对齐），但是一个**静默错值**候选，
   登记备查（不属于本次范围）。
