# 1b（mrows 激活 staging → cp.async16）状态清算 + GPU 验证手册

> 工部 · 2026-09-13 · **未执行 GPU/e2e**；本机 `cargo check --workspace --all-targets` + 远端
> nvcc 13.2 `sm_103a` **编译检查**（无 GPU 工作）。
> 设计：`l4-mgrid-first-step-design.md §4`；批次：`l4l5-next-batch-implementation-plan.md §2 W-N1 N1-1`。

---

## 0. 结论先行（两条）

1. **该项的代码早已落树，不需要"实施"** —— 2026-09-12 两次提交：
   `ce74891`（1b 本体：激活行 staging 从标量改为 `dsv41_cp_async16`）+ `d0706ff`
   （编译修复：把 host-scope `g_mrows_act_cpasync` 改成 `mrows_act_cpasync_host()` + `int act_cp16` 尾参）。
   现场读码四条腿全齐：kernel 分支、launcher 取值、parity 轴、lazy A/B 臂。
2. **"激活"缺的是另外两件（本次补上）**：
   - **活性证据 = 0**：该臂无自有符号、且**不改 grid** ⇒ kernel 名/发数/GridX 分布与 OFF 完全相同，
     `/proc/<pid>/environ` 只能证明"变量进了进程"，永远不能证明"分支被走到"。
   - **生产（batched）路 A/B 臂 = 0**：`scripts/batched_400_v2.sh`（63.8 tok/s 的那条路）有
     tcgen05 / MROWS_A(b1|b2|b3) / HC / B6 / B5 / B4 七个 opt-in 臂，**独缺 1b**。

---

## 1. 设计要点摘录（`l4-mgrid-first-step-design.md §4`）

| 项 | 内容 |
|---|---|
| 非对称 | **同一个 kernel** 里两条 staging 走两套规则：权重行 `cp.async16`（16 B/lane/issue，`:5452`），激活行标量（1 B/lane/issue，原 `:5249`）⇒ **16× 指令差**。kernel 自己在权重侧把这条账写死（`:5420-5430`「SIXTEEN times the instructions」），**却只修了权重侧**。`DSV41_GEMV_ACT_CPASYNC`（P4）只接了 m=1 的 gemv，没接 mrows。 |
| 指令账（wkv, 128 thr, k=5120, m=6） | 标量 6×5120/128 = **240 次/线程 ×2 指令 = 480 warp-instr**（且**完全暴露在 `__syncthreads()` 之前**）→ cp.async16 = **15 warp-instr**，与权重 cp.async 一起在 barrier 下重叠。 |
| 对齐前提 | launcher 已拒绝 `k & 31`；`s_a = s_as + fold_r*nb_k`（`nb_k*4 = 640`，16B 倍数）⇒ 合法。devices 侧仍以 `dsv41_f4_ok(a) && dsv41_f4_ok(s_a)` **运行时复核**，不满足则回落标量（避免 err 716）。 |
| 数值 | **纯拷贝**：同字节 → 同 slot（`s_a[r*k+i]`），consume 路径 / K 走序 / `acc[r] += av*wv` 链 / `shfl_xor` 树**一字不动** ⇒ **逐位等价**（与权重侧 staging 修复同一论证）。 |
| 预期 | batched −0.3 ~ −0.5 ms/步（设计，指令账）；族 15.1% / ~4.2 ms ⇒ 回收 12~24%。把握**最高**（「隔壁修过这边漏了」，非外推）。 |
| 止损 | 中性 ⇒ **该核不是 staging 受限** ⇒ 重估 instruction-bound 归因（上报），不投变体矩阵。 |

---

## 2. 落树现状（读码，逐条给 `file:line`）

| 腿 | 位置 | 状态 |
|---|---|---|
| kernel 分支 + 对齐守卫 + 标量回落 | `kernels/cuda/dsv41_kernels.cu:5387`（`a16`）、`:5394-5404`（cp16/标量两支）、`:5409`（retire 点） | ✅ |
| 尾参 `int act_cp16` | 同上 `:5339` | ✅ |
| gate（**每次 launch** 读 getenv —— 刻意的，为让 parity 在**同进程**扫两臂） | `:3932` `mrows_act_cpasync_host()` | ✅ |
| launcher 取值 + 8 个特化传参 | `:5615`、`:5688-5695` | ✅ |
| parity 轴（同进程 {unset, "1"} × 5 形状，含 m=1/lazy） | `kernels/cuda/tests_dsv41_gemm_mrows.cu` `mr_case_cp16_axis` | ✅ |
| lazy(m=1) A/B 臂 | `scripts/lazy_l45_ab.sh` 臂 `1b`（`:6`/`:125`） | ✅ |
| **batched 生产 A/B 臂** | — | ❌ **本次补** |
| **活性回执** | — | ❌ **本次补** |
| `tp.rs` | 与本项**无关**：`:325-353` 是 V5 ledger canary/guard 偏移，不是 staging 布局 | n/a |

**已知未修的同类（不在本批次，别顺手做）**：`wo_a_grouped_gemv_kernel` 的同类 `cp.async16` 修复
单独立项于 `projection-family-optimization §3-R1`（设计 §9-3 明文划在范围外）。

---

## 3. 本次改动（2 文件 / +75 行，零数值）

### 3.1 `kernels/cuda/dsv41_kernels.cu:5649-5685`（+37，纯 host）
`dsv41_gemm_fp8_mrows` 里加**一次性活性回执**：门 armed 时，在**首个 armed launch** 打印
```
[mrows-act-cp16] ARMED m=6 n=512 k=5120 fold_r=6 nwarps=4 smem=56064 -> activation staging = cp.async16 (16B per lane per issue)
```
或（该形状被 16B 守卫拒绝时）
```
... -> activation staging = scalar loop (the 16B guard DECLINED this shape: the arm is INERT here ...)
```
* **补的正是 plan §5.2 判定 leg ① 的洞**：无符号 + 无 grid 变化 ⇒ 这个 ceil 之前**无法证伪"armed but inert"**（本仓 #1 陷阱 R6 的形态）。现在`decline` 也打印。
* 只印**一次/进程**（`static int reported`；沿用 `:6790` `[align]` 的 static-counter 风格），gate unset（出厂默认）时**什么都不印** ⇒ 默认路径 stdout/stderr 逐字节不变。
* **host code，取不到动态 smem 地址** ⇒ smem 侧用放 `s_a` 的同一算式 `nwarps*k + 256*4 + fold_r*nb_k*4` 判 16B，device 侧 `dsv41_f4_ok(s_a)` 仍是权威。
* 若日志**无回执**而 `.env` 显示 armed ⇒ mrows launcher 没走到这里（更早 decline：`mode<3` / `NO_GEMV_FP8` / 形状拒绝），或该路根本不调它 —— 这本身是有用信息。

### 3.2 `scripts/batched_400_v2.sh:88-90, 400-434`（+38，opt-in 臂）
新增 `B400_1B=1` → `DSV41_MROWS_ACT_CPASYNC=1`，默认 unset ⇒ `$GATES_ONELINE` 与出厂配置逐字节相同。
注释里写死了：活性证据（回执 + parity）、逐位等价判定（**不许**按 red-line 臂判）、以及
"该核在本路是活的"（`proj_mrows` 的 verify wq_a/wkv/wq_b @ `m <= VERIFY_ROWS=6` + 共享专家 w1/w3/w2 都调它）。

> ⚠️ 脚本不在本任务的源码白名单内（白名单：`dsv41_kernels.cu` / `tp.rs` / 如必须的 `chain_dev.rs`），
> 但它是"激活"这件事的载体（plan §6 把 W-N1 的臂就是记在 `scripts/batched_400_v2.sh`），
> 且与 peer 的冲突面为零（peer 在 `load.rs`/`weights.rs`/`dsv41_experts_mxf4.cu`）。
> 如需回退：删 `:400-434` 与 `:88-90` 两段即可。

---

## 4. 验证记录（0 GPU）

| 门 | 结果 |
|---|---|
| `cargo check --workspace --all-targets` | **EXIT=0** |
| 远端 nvcc 13.2 编译检查（`-gencode arch=compute_103a,code=sm_103a -O3 -std=c++17 -c dsv41_kernels.cu`） | **无 error**，`.o` 产出 5.79 MB；仅两条**既有** `#177-D` unused（`:108` `e2m1_to_f`、`:7682` `k1max`，非本次改动） |
| `cargo test -p ferrite-dsv41` | 唯一失败 = `tests/ar_hcpost_parity.rs` 的 `dlopen(libcudart.so) failed`（本机无 CUDA runtime，**环境性、与本次改动无关**，本次未动任何 Rust 路径） |
| `bash -n scripts/batched_400_v2.sh` | OK |

---

## 5. GPU 验证手册（给拿卡的人，照抄执行）

### 5.1 门与臂
```
门名   : DSV41_MROWS_ACT_CPASYNC     （1 = cp.async16 激活行 staging；unset/0 = 今天的标量，出厂默认）

batched（生产路，63.8 tok/s 那条）
  ARM0 : bash scripts/batched_400_v2.sh
  ARM1b: B400_1B=1 bash scripts/batched_400_v2.sh
lazy（m=1）备选
  ARMS="base 1b" bash scripts/lazy_l45_ab.sh
```
**同臂硬要求**：设计 `§7.3` 要求 `DSV41_SH_PAIR_M=1` 同臂（否则 sh w1/w3 的 mrows 靶子还在，混淆归因）——
lazy 脚本的 `BASE_ENV` 已含它；**batched 矩阵里没有**，故 batched A/B 需在两臂同时加上
`DSV41_SH_PAIR_M=1`，或至少明确记录"未加"并在结论里扣掉该混淆。
`DSV41_MROWS_FOLD_R`（1a）**不要**同臂——1a 与 1b 正交，且 batched 上 `auto` 已修成恒等（ng=1）。

### 5.2 三段判据（缺一不算完成）
| 段 | 判据 |
|---|---|
| ① 活性 | `grep '\[mrows-act-cp16\]' $LOGDIR/<tag>.log` ⇒ 必须出现 **`ARMED ... = cp.async16`**。出现 `scalar loop ... DECLINED` =**该形状臂是 inert**（别当收益读）；**完全无回执** = launcher 没走到（更早 decline）。旁证：`kernels/cuda/tests_dsv41_gemm_mrows.cu` 的 `mr_case_cp16_axis`（同进程双臂 vs 同一 m=1 参照）。**注意**：kernel 名/发数/GridX 在 on/off 两臂**相同**，`cuda_gpu_kern_sum` 在这里**不可用**。 |
| ② 正确性 | **逐位臂**（staging 是纯拷贝）⇒ 必须 token 级一致：四段文本逐字 + 计数数字顺序 + 出师表零拉丁 + `faults=0` + `ar5-hang=0`。batched 脆弱判决：**先看 accept/k_emit**——吞吐 = `k_emit / C(6)`，任何破 accept 的旋钮是 6~8× 灾难，`tok/s 没掉` **不构成数值证据**。 |
| ③ 收益 | **同会话背靠背** `steady_median` 位移（`STEADY_SKIP=20`），**一 gate 一变一 commit**；nsys **不读绝对 ms**（v5 publish 自旋被放大 ~300×）。 |
| ④ 生效 | `$LOGDIR/<tag>.env`（`/proc/<pid>/environ` 实读）必须见 `DSV41_MROWS_ACT_CPASYNC=1`。 |

### 5.3 票面与止损
| | 值 |
|---|---|
| 预期（batched，指令账） | **−0.3 ~ −0.5 ms/步**（族 4.2 ms 的 12~24%）；lazy 下按 `×k_emit` 折算**不入预算** |
| 把握 | **本批最高**（姊妹核同款修复已在树） |
| 止损 | `|Δsteady_median| < 0.8ms`（lazy 脚本的噪声地板）且回执已证明 `cp.async16` 生效 ⇒ **判该核非 staging 受限**，记录并转下一项，**不投变体矩阵** |
| 反向证据（写进账） | `{SH_EXP_MROWS, VERIFY_GRAPH, VERIFY_ROPE_MROWS, DRAFT_P3A}` 全开只 −1.21ms（预期 −24）⇒ instruction-bound + 低占用的墙，符号只能由 A/B 定 |

---

## 6. 数值红线声明

**本项不改变数值顺序**（fp 非结合面为零）：新旧两支把**同样的字节**写进**同一个 slot**，
consume/K-walk/归约树/`bias` 加法一字未动；16B 守卫不满足时**回落标量**而非改变语义。
⇒ **不需要** gate ON/OFF 逐字节比对之外的容忍度口径，判定按**逐位臂**做。
本次改动（活性回执 + 脚本臂）**全程在 host 侧，未触碰任何 kernel 指令** ⇒ 对数值**零影响**。
