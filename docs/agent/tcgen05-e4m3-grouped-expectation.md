# tcgen05 e4m3 GROUPED 臂：预期收益与风险（只读分析）

> 工部 · 2026-09-12 · **只读分析 + 本文件（唯一产出）**。未执行任何 GPU 命令、未改动任何源码。
> 代码基线：工作树 HEAD `2bd7212`（tcgen05-unblock 的 verdict + script arm）。
> 输入：`chain_dev.rs` / `dsv41_experts_mxf4.cu`（tc5::e4x）/ `dsv41_route.cu` / `load.rs` /
> `device.rs` 现场核对，以及 `routed-expert-residual.md`、`verify-ms-breakdown.md`、
> `grouped-routing-design.md`、`expert-tcgen05-plan.md`、`batched-400-v2-prediction.md`、
> `final-400-config.md`、`STATUS.md`、`scripts/batched_400_v2.sh`、`scripts/tcgen05_bench.sh`。

---

## 0. 结论摘要（先看这个）

**任务前提里有两处与现场不符，必须先钉死，它们直接改变收益的量级。**

| 任务前提 | 现场事实 | 出处 |
|---|---|---|
| 「routed experts 8.3ms → 1.5ms（−6.8ms）」 | **−6.8ms 是 swapAB 全 routed（gate/up **＋** down）的目标数**；本 arm **只替换 gate/up**，down 仍是 SIMT（无 tcgen05 down kernel） | `routed-expert-residual.md` §2.3（gateup 4.82 / down 3.48）；`moe_experts_grouped_gate_up` 只发 gate/up；`tcgen05_bench.sh` 头注 "NO tcgen05 down kernel exists yet" |
| 「−6.8ms 的落点 1.0~1.5ms」 | 那个落点来自 `tc5::mxf4/mxf8f6f4` **swapAB + kRing=8 TMA ring** 的设计（`expert-tcgen05-plan.md` §1d/§1e）；本 arm 是**另一个 kernel**（`tc5::e4x` dense masked tile，**无 cp.async / 无 TMA / 单缓冲**） | `dsv41_experts_mxf4.cu:5256-6094`（e4x 全区间只有 `fence.proxy.async`，0 条 `cp.async`）；`kStageAtoms=2` 单缓冲 |

**修正后的预期**（推导见 §5）：

- **本 arm 的收益上限 = 被替换的 gate/up 那一半**。m=5 口径 gateup 4.82ms、down 3.48ms；m=6（SWALLOW）口径 routed ≈ 9.96ms、gateup ≈ 5.78ms。
- **乐观**（e4x 像 swapAB 计划那样高效，gateup → ~1.0ms）：verify −3.8ms；**现实**（无流水线、M=128 固定、~100x 张量核过量计算）：**−1.0 ~ −2.8ms**；**pessimistic/decline**：+0.5ms（回退路径比基线慢，见 §2/§3）。
- **`verify 35 → 28ms / step 32ms / 63 tok/s` 不可由本 arm 单独达成**。按乐观修正：verify ≈ 30~33ms、step ≈ 34~37ms、@accept 1.022 ≈ 55~59 tok/s、@accept 3 ≈ 110~118 tok/s。
- **最大风险不是性能，是正确性**：该 kernel **从未在任何 GPU 上执行过**，且源码自带两个 `[OPEN]` 未决项（dense idesc 的 format code、f8f6f4 的 fp4 操作数是 packed 还是 unpacked）。二者任一错 → 静默错值（拉丁/乱码红线）或非法指令。

---

## 1. 现场核对：这个 arm 实际会跑什么

### 1.1 gate 链（逐条核对，全部成立）

| 门 | 读取方式 | arm 里的值 | 代码位置 |
|---|---|---|---|
| `DSV41_EXPERT_ACT_E4M3` | `v != "0"`，默认 false | `=1`（**已在基线矩阵里**） | `chain_dev.rs:856` |
| `DSV41_EXPERT_TCGEN05_E4M3` | `starts_with('1')`，默认 false | `=1`（arm 加） | `chain_dev.rs:756` |
| `DSV41_EXPERT_GROUPED` | `starts_with('1')`，默认 false | `=1`（arm 加） | `chain_dev.rs:903` |
| `DSV41_GATEUP_FUSE` | `v != "0"`，**默认 true** | `=0`（arm 加） | `chain_dev.rs:1891` |
| `DSV41_EXPERT_ILV` | `v != "0"`，默认 true | `=0`（arm 加） | `weights.rs:487` |

### 1.2 派发链（moe_rows，verify 多行路径）

```
moe_rows (chain_dev.rs:10758)
  ├─ quant_fp8（e4m3 激活，行= m）                      :10918
  ├─ moe_route_grouped  → route_group + route_gather_rows  :10946 / :10479-10565
  ├─ moe_experts_grouped_gate_up → dsv41_expert_gemm_e4m3_grouped + route_scatter_rows :10962 / :10606
  │    要求：expert_tcgen05_e4m3() ✓、e4m3 ✓、!gateup_fused ✓、dim%64==0 ✓、
  │          2*inter%64==0 ✓、!ld.experts_ilv ✓、symbol ✓、experts≥2 ✓       :10621-10690
  ├─ swiglu_limit_batched（因为 !gateup_fused）          :11100
  └─ down：expert_down_reduce_fp4_batched（**仍是 SIMT**） :11118
```

- **只有 gate/up 被 tcgen05 接管**；down 3.48ms 原样保留。
- `e4x_tile = false`（`:11015`）：dense-tile e4x 臂在 `moe_rows` 形状下**永远不发**，只会打一条 decline 告警。所以本 arm 在 verify 里跑的**只有 grouped masked kernel**。

### 1.3 一个必须说清的耦合：`DSV41_EXPERT_ILV=0` 是**冗余**的

`load.rs:767-778` 的 `ilv_ok()`：

```rust
gateup_ilv() && moe_batch() && gateup_fuse() && expert_fp4_mode()==2
  && !no_gemv_fp4() && dim%512==0 && supports_{moe_batch,gateup_fuse,expert_ilv}
```

`gateup_fuse()` 是**合取项**。所以 **`DSV41_GATEUP_FUSE=0` 单独就足以让 `ilv_ok()` 为 false ⇒ 权重按 plain 布局加载**，`DSV41_EXPERT_ILV=0` 只是 belt-and-braces（无害、且防未来 `ilv_ok` 被改动）。

⇒ **「五件套」实际是 4 个独立门 + 1 个蕴含门**。含义有二：
1. 不能做「ILV=0 但 GATEUP_FUSE=1」的实验（loader 会强制 ILV=0）；
2. 反过来，只要 arm 里 GATEUP_FUSE=0，ILV 一定是 0，不会出现混合布局。

### 1.4 还有一处**未被 arm 说明覆盖的副作用**：`DSV41_EXPERT_TCGEN05_E4M3=1` 同时武装了另一个 kernel

`DSV41_EXPERT_TCGEN05_E4M3` 是 **e4m3 tcgen05 家族共用的运行时门**。除了 grouped 臂（`moe_rows`），它还门控：

- `tc5::e4x` dense-tile 臂（`dsv41_expert_gemm_e4m3_ext`）——在 `moe_rows` 里恒 decline（`e4x_tile=false`），**无害**；
- **`tc5::e4` swapAB 臂（`dsv41_expert_tcgen05_gate_up_e4m3`）**——在**单行 `moe()`**（`chain_dev.rs:13981-13995`）里，条件是 `e4m3 && expert_tcgen05_e4m3() && symbol && !ilv`，arm 下**全部成立**。

也即：**如果 serve 在一次请求里跑到 `moe()`（eager 路径），那个同样从未在 GPU 上跑过的 swapAB e4m3 kernel 也会被执行**。本次测量的是 verify（`moe_rows`），但 `moe()` 是否在窗口内被调用必须现场确认（判据见 §8.3）——若在，则一次 arm 同时把**两个**未验证 kernel 推上 GPU，归因难度倍增。

---

## 2. ILV=0 的影响

### 2.1 其他路径是否还工作

| 路径 | ILV=0 下的行为 | 结论 |
|---|---|---|
| SIMT 回退（`expert_gate_up_fp4_batched`）| 模板实例 `<false,1>` **已编译进 .so**（`dsv41_experts_mxf4.cu:2664-2680` 的 launcher 按运行时 `ilv` 选实例），plain 布局走「逐行分别读 gate/up」的 split 臂，`pair_body = fuse‖ilv = false`，`ksplit` 强制 1 | ✅ 可用（这就是 ILV 之前的原始布局） |
| `moe()`（单行 eager）| 同上，`ilv=0` 传下去；`ilv && !batched` 的报错分支不触发（`:13787-13792`）| ✅ |
| draft / MTP（`dspark_dev.rs::draft_moe`）| 同一 `ilv_ok` 决策，同一个 launcher 参数 | ✅ |
| **SH_EXP（共享专家）** | 共享专家权重是**独立张量**（`ffn.shared_experts.w1/w3.weight`，`load.rs:859-864` 直接 `take!` 加载，**不走 `load_expert_pool`**），ILV 只作用 routed pool；且 SH_EXP 走 fp8（`gemm_fp8_mx2` / `gemm_fp8_mrows` / `sh_pair`），与 ILV/GATEUP_FUSE 无关 | ✅ **完全不受影响** |
| down（w2）| ILV 只交错 w1/w3；w2 从不交错 | ✅ |

### 2.2 代价

- ILV 的实测收益是 **−0.09ms**（`stage-b-execution.md:63` 的验证口径），所以 ILV=0 的回退代价 ≈ **+0.09ms**，量级可忽略。
- 加载期少掉 15744 次 `dsv41_interleave_gateup_fp4`（一次性 ~34.6ms，`stage-b-execution.md:57` 的 nsys 陷阱）——不影响稳态步时。
- **对 grouped arm 本身**：`kind::f8f6f4` 读 plain w1/w3 平面，ILV=1 会让它读错字节（`ld.experts_ilv` 是硬 decline 条件，`:10655`），所以 ILV=0 是**必要前置**而不是优化。

### 2.3 结论

ILV=0 **安全**：SIMT 回退可用、SH_EXP 不受影响、代价 ~+0.09ms。唯一要注意的是它**不是独立变量**（§1.3）。

---

## 3. GATEUP_FUSE=0 的影响

### 3.1 机制（单一真值源：调用方的 slot pitch）

launcher 里 `fuse` **不是**只看 env，而是绑定调用方给的 `out_slot_stride`（`dsv41_experts_mxf4.cu:2614-2624`）：

```
out_slot_stride == inter    → fuse=1（swiglu'd [inter] epilogue）
out_slot_stride == 2*inter  → fuse=0（raw gate|up [2*inter]，随后单独的 swiglu）
```

Rust 侧 `act_slot = if gateup_fused { inter_local } else { 2*inter_local }`（`chain_dev.rs:10898-10906`），两侧由 pitch 结构性地绑死。**GATEUP_FUSE=0 ⇒ raw pair + 单独 `swiglu_limit_batched`**。

### 3.2 对非 tcgen05 路径的影响

| 路径 | 影响 |
|---|---|
| routed batched gate/up（verify / eager / **draft**）| 全部退回**未融合**：多一次 `swiglu_limit_batched` 启动、`ex_act` 写出从 `inter` 变 `2*inter`。`STATUS.md:5493` 估 **−0.4ms**（融合开启的收益），round 22-23 实测 gateup+down 合计 −0.51ms（`STATUS.md:5551-5557`）⇒ **回退代价 ≈ +0.35~0.4ms** |
| **SH_EXP** | **不受影响**：共享专家的融合门是 `sh_exp_mx2()` / `sh_exp_fused()`（`chain_dev.rs:1234/1366`），**不读 `gateup_fuse()`**（唯一的 `gateup_fuse()` 读者是 `:10898`、`:14035`、`:14128`、`dspark_dev.rs:2300`、`load.rs:771`，全是 routed/draft MoE） |
| down | `down_fuse()` 独立（`chain_dev.rs:941`），arm **没有**关它 ⇒ down 仍是融合的 down+reduce |
| ILV | **被连带关闭**（§1.3） |

### 3.3 为什么它是硬前置

grouped e4x 的 epilogue **只做 clamp，不做 swiglu**（`epi_mode 1`，`dsv41_experts_mxf4.cu:5995-6000`），而且 `b_split` 只能表达 `gate|up` 两段 ⇒ 融合形状它是**结构性不支持**，`moe_experts_grouped_gate_up` 里 `if gateup_fused { decline }`（`:10638`）。所以 GATEUP_FUSE=1 时 arm 根本不进入。

数值上：`swiglu_limit_kernel` / `swiglu_limit_batched_kernel` 与融合 epilogue 是**逐项同形**（`g = fminf(g,limit); u = fminf(fmaxf(u,-limit),limit); out = g/(1+expf(-g))*u`，`dsv41_glue.cu:173-188 / 1129-1146`），并且 e4x 的 `epi_mode 1` 是**同一个 clamp** ⇒ 双重 clamp 幂等，**数值与融合路径一致**。

### 3.4 结论

GATEUP_FUSE=0 **数值安全、SH_EXP 无涉**，但代价是**全局 ~+0.4ms**，且这个代价在 arm 失败回退时会**全额计入**——它使得「arm 没生效」看起来像**性能退化**而不是「无变化」。

---

## 4. 数值域安全

### 4.1 已经成立的部分

1. **gather / scatter 是纯搬运**：`dsv41_route.cu:330-404` 全是 `uint4` / `float4` / 逐字节拷贝，零算术、零重量化；`gather_src=-1 → 写 0 行`、`perm_map=-1 → 写 0 行`（`:337-341 / 388-393`）。作者自己的判据是 `memcmp` 逐字节相同（`grouped-routing-design.md` §5.4 / §7-B）。
2. **masked kernel 的每行 K 序与 dense e4x 臂一致**：同一 atom 升序走、同两条 `kind::f8f6f4` MMA/atom、同一个 `C_accum += C_local*sa*sb` 折叠点（`dsv41_experts_mxf4.cu:5762-5779`）。masked 行的 A/scale 在 smem 里是 0、累加器不回写，**不可能污染活输出单元**（`:5904-5916, 5987-6002`）。
3. **scale 外提**在数值上是精确的：`sa`（激活，`fast_round_scale` → 2 的幂，`dsv41_kernels.cu:113-119`）与 `sb`（权重 e8m0）都是 2 的幂 ⇒ `sa*sb` 精确、无舍入、无重结合风险（`dsv41_experts_mxf4.cu` 的 NUMERIC DOMAIN §2）。
4. **布局置换本身不进数值**：`counts/starts/perm/gather_src` 只决定「哪些行共享一次 launch」。
5. **确定性**：`route_group_kernel` 的赋值扫描是**单线程串行**（`:269-313`），无 atomic、不依赖 block 调度 ⇒ CUDA graph 重放得到逐位相同的排列。

### 4.2 **尚未成立**的四条（这才是关键）

1. **「与逐行版一致」指的是与 *dense e4x 臂* 一致，不是与被替换的 *SIMT GEMV* 一致。** 本 arm 替换的是 `expert_gate_up_fp4_batched`（SIMT，LUT 解码 + fma 链），而 e4x 是张量核 MMA + 每 32 块一次寄存器折叠。两者的 f32 累加次序**不同** ⇒ **不是逐位相等**。这就是「红线（拉丁/双字/缺句）必须重验」的原因，也意味着**不能要求四段文本与基线逐字节相同**（与 `ILV=1 vs 0` 那次 `run_ilv_case` 的 EXACT 判据不同）。
2. **dense + 非 block-scaled 的 idesc format code 是猜的**：源码自己写 `[OPEN] the non-block-scaled idesc format codes ... This arm takes a_format = E4M3 = 0 and b_format = E2M1 = 5 from MXF8F6F4Format — the only enum this repo has decoded`，且「非 block-scaled 形式把 bits [4,6) 花在 **D type** 上，而不是 `b_sf_id`」——这个解码**从未在 GPU 上跑过**。
3. **f8f6f4 家族里 fp4 操作数的 packed/unpacked 假设是猜的**：源码写 `[OPEN] ... the fp4 operand of the f8f6f4 family is the UNPACKED ("unpacksmem") form ... The alternative reading (packed fp4, 16 bytes per row) would halve B's smem but invent an SBO=128B descriptor this repo has no precedent for — it is one numeric parity run away either way`。e4x 的 B staging 确实按「1 element/byte」展开（`e4x_expand`，`:5882-5901`）。**若硬件实际要 packed，整个 B 侧全错。**
4. **该 MMA 形式在本仓库没有任何 GPU 先例**：唯一在 GPU 上数值验证过的 tcgen05 是 `kind::mxf4.block_scale.scale_vec::2X`（`tests_tcgen05_mxf4.cu`，7 shapes `maxdiff=0.000e+00`）。`tests_tcgen05_mxf8f6f4_1x.cu`（同一 1X 家族）**GPU parity 待跑**，而 e4x 用的 dense `kind::f8f6f4`（无 block_scale）又是第三种形式，**连 host 侧 parity 套件都没有**（`kernels/cuda/tests_*.cu` 里没有任何文件引用 `e4x` / `gemm_e4m3_grouped`）。

### 4.3 数值域结论

- 「gather/scatter 纯搬运 + masked K 序与 dense 臂一致」这两条**成立，但它们只保证 grouped 臂自身自洽**，不保证首次 GPU 执行能出对的值。
- **真正的数值风险集中在 MMA 的两条 `[OPEN]` 上**（idesc format code / fp4 packed-or-not），这两条只能靠一次 GPU parity 或一次 serve 红线来裁决。
- 建议：把「四段文本 + `faults=0`」当作**唯一判据**，不要用「与基线文本逐字节相同」做判据——新 GEMM 引擎的舍入差异是预期内的。

---

## 5. 预期收益（修正后）

### 5.1 收益上限：只有 gate/up 那一半

`routed-expert-residual.md` §2.3 的实测拆分（m=5，verify 37.31ms 基线下）：

```
gateup  4.82 ms  (200 行 × 24.1µs)
down    3.48 ms  (200 行 × 17.4µs)
────────────────────────────
routed  8.30 ms   ← verify-ms-breakdown.md 主表 #1
```

- 本 arm 替换的是 **gateup 的一半**（`moe_experts_grouped_gate_up` 一次 launch 产出 gate|up 的 `[2*inter]`）。
- **down 没有 tcgen05 版本**（`tcgen05_bench.sh` 头注、`expert-tcgen05-plan.md` gap ①）⇒ 3.48ms 原样保留。
- ⇒ **本 arm 的收益上限 = 4.82ms**（gateup → 0 的理想情况），而不是 8.30ms / 6.8ms。
- m=6（SWALLOW，verify 实际行数）口径下：routed ≈ 9.96ms、**gateup ≈ 5.78ms**（= 4.82 × 6/5，与 `batched-400-v2-prediction.md` 的 SWALLOW +1.6ms 相符）。

### 5.2 e4x grouped kernel 的成本结构（为什么不敢按 1.0~1.5ms 计）

核对 `dsv41_experts_mxf4.cu:5780-6002` 后，这个 kernel 有**三个与 1.0~1.5ms 落点假设不符**的地方：

1. **M=128 是固定的，mask 省不掉张量核工作量。** `1-CTA kind::f8f6f4` 的 MMA 固定 `kMTile=128`；每个 expert 只占 `counts[e] ≈ 1~3` 行（m=6/topk=6 → 36 个 assignment 摊到 ~34 个活 expert），`grid.y = ceil(m_cap/128) = 1` ⇒ **每个 expert-tile 都跑满 128 行的 MMA**。
   - 每 step 的张量核 MAC ≈ 40 层 × ~34 expert × (n_total/kNTile = 640/64 = 10) × (128×64×5120) ≈ **570 GMAC**；
   - 有用 MAC ≈ 40 层 × 36 assignment × (640×5120) ≈ **4.7 GMAC**；
   - **过量 ≈ 120×**。`grouped-routing-design.md` §6 说 masked kernel「36 行真实数据只付 36 行的 MMA」——**与实现不符**（实现仍付 128 行）。所以 **−6.8ms 的赌注全部压在「张量核比 SIMT 解码路径快 ≥ 120×」上**：要落在 ~1.0ms，需要 ~1 PFLOP/s 的有效吞吐；落在 3ms 就已是净亏。
2. **没有软件流水线。** `kStageAtoms=2`、`s.a/s.b` 单缓冲、每 stage `fence.proxy.async` + `__syncthreads()`，然后 2 个 atom 各自 `MMA → tc_commit → mbar_wait → 折叠`，**load 与 MMA 完全不重叠**。整个 K 循环 = `5120/128 = 40` 个 stage × 2 个串行化的 commit/wait。`routed-expert-residual.md` §3(b) 说上一次 tcgen05 负结果的机制之一是「**无 cp.async**」——**这个 kernel 同样没有 cp.async / TMA / kRing**（e4x 全区间只有 `fence.proxy.async`，0 条 `cp.async`）。
3. **每 32 块一次的 TMEM 回读 + 寄存器折叠**（`scale 外提` 的固有代价）：per thread per atom = 8×`tc_ld_x16` + 128 FMA，80 atom/CTA ⇒ ~0.5ms/step 的纯 SIMT FMA + 大量 TMEM 事务。

**结论**：这个 e4x 臂是**正确性优先的骨架**（源码自称 SKELETON），它的形状与 `expert-tcgen05-plan.md` 里那个「kRing=8 TMA ring / in-flight 28KB/CTA / 240 CTA」的设计**不是同一个东西**。**不能把 swapAB 计划的 1.0~1.5ms 直接搬过来**。

### 5.3 收益分解（三档）

| 档 | 假设 | gateup 落点 | verify 变化 | 说明 |
|---|---|---|---|---|
| **乐观** | e4x 张量核效率 ~1 PFLOP/s、staging 不暴露 | ~1.0ms | **−3.8ms** | 需要有 GPU 微基准背书，目前**没有** |
| **现实** | 120x 过量 + 无流水线，落在 SIMT 与理想之间 | ~2.0~3.5ms | **−1.3 ~ −2.8ms** | 也扣掉 +0.4ms（unfused）与额外 launch |
| **回退** | grouped 因任一前置 decline / 符号缺失 | 走 SIMT（unfused+plain） | **+0.4~+0.5ms** | **比基线慢**，因为 arm 强制 GATEUP_FUSE=0 + ILV=0 |

### 5.4 步时与吞吐（按任务给的公式 `tok/s = (mean_k+1) × 1000/step_ms`）

基线锚（`batched-400-v2-prediction.md`）：`verify(m=6) ≈ 34~37ms`、`step ≈ 35~42ms`、`mean_k ≈ 1.022`。

| 场景 | verify | step ≈ verify+draft(3.6~4.9)+commit(0.2) | tok/s @k_acc 1.022 | tok/s @k_acc 3 |
|---|---|---|---|---|
| 任务前提（−6.8ms） | 35 → **28** | **32** | **63** | **125** |
| **乐观修正（−3.8ms）** | 35 → **31.2** | **35.0~36.3** | **55.7~57.8** | **110~114** |
| **现实修正（−1.3~−2.8ms）** | 35 → **32.2~33.7** | **36.0~37.8** | **53.5~56.2** | **106~111** |
| 回退 | 35 → **35.4** | **39.2~40.5** | **49.9~51.6** | **99~102** |

**⇒ 任务里「verify ~28ms / step ~32ms / 63 tok/s」这一档只有「e4x 达到 swapAB 计划的效率」时才成立；按已实现 kernel 的成本结构，更可能是 31~34ms / 55~58 tok/s。**

---

## 6. 风险清单（按严重度）

| # | 风险 | 机制 | 失败表现 | 可观测判据 | 等级 |
|---|---|---|---|---|---|
| **R1** | **MMA 的 `[OPEN]` 两条（dense idesc format code；f8f6f4 的 fp4 packed/unpacked）** | 解码猜测错 | 静默错值 → 拉丁/乱码；或非法指令/挂起 | 四段文本红线；serve health timeout | 🔴 **最高** |
| **R2** | **kernel 从未在任何 GPU 上执行过**（dense 与 grouped 都无 harness；`tests_tcgen05_mxf8f6f4_1x.cu` 的 GPU parity 未跑） | — | 第一次真实执行就是 serve，无隔离微基准 | 无（只能靠 serve） | 🔴 |
| **R3** | **收益上限被 down 锁死** | 只替换 gate/up（4.82/5.78ms），down 3.48ms 不动 | 即使 gateup 归零也只有 −4.8~−5.8ms | 对比 verify 变化量是否 < −4.8ms | 🟠 |
| **R4** | **M=128 固定 ⇒ ~120× 张量核过量** | mask 只省 A 的读与 epilogue 写，不减 MMA/折叠/TMEM 回读 | 落点远差于 1.5ms，甚至慢于 SIMT | 只靠 serve 读数（无隔离微基准） | 🟠 |
| **R5** | **无 cp.async / TMA / 流水线（单缓冲 + 每 atom 一次 commit/wait）** | staging 延迟全暴露，40 stage 串行 | 落点受每 stage 固定延迟支配 | nsys 按 kernel 名看 duration | 🟠 |
| **R6** | **arm 把「回退」变成「退化」** | GATEUP_FUSE=0（+0.4ms）+ ILV=0（+0.09ms）全局生效 | arm 未生效时读数 = 基线 +0.5ms，易被误读为「kernel 慢」 | `expert_grouped_skipped_note` / `tcgen05_e4m3_ext_skipped_note` 一次性告警 | 🟠 |
| **R7** | **ARM 同时武装单行 `moe()` 的 swapAB e4m3 kernel**（同门 `DSV41_EXPERT_TCGEN05_E4M3`） | 若 serve 窗口内跑到 `moe()`，两个未验证 kernel 一起上 GPU | 归因困难 / 额外红线风险 | 日志里查 `[gmo]` / `[phs]`；或先确认 verify 是否独立于 `moe()` | 🟠 |
| **R8** | **数值不是逐位一致（换了 GEMM 引擎）** | MMA+折叠 vs SIMT fma 链 | 四段文本可能与基线不同（**预期内**） | 只用红线判定，不要用逐字节比对 | 🟡 |
| **R9** | **CUDA graph 捕获（VERIFY_GRAPH=1）** | 新增 4 个 launch/layer（route_group/gather/gemm/scatter）：capture 失败会掉进设计内的静默降级 | step +（图失效的 ~1.5ms）且读数口径变化 | `[verify_graph] captured` vs `capture FAILED` | 🟡 |
| **R10** | **`.env` 泄漏防护不含这 5 个门** | 脚本只 FORBIDDEN 3 个门；shell export 会静默改路径 | 基线轮被污染 | 每轮 `<tag>.env` 必须逐门核对（基线：这 5 个应**不存在**） | 🟡 |
| **R11** | **grouped buffer 容量是零余量** | `grp_xq = 36×dim`、`grp_xsc = 36×dim/32 (+8)`，gather 恰好写满 | 任何 m/topk 变化都会越界（现已被 capacity guard 挡住） | `moe_route_grouped` 的 capacity decline 告警 | ⚪ |

---

## 7. 建议的测试顺序

前提：`batched_400_v2.sh` 每次调用只跑**一个 arm**（`run_case run "$PORT"`，flock 串行）。所以排序就是「调用顺序」。

### 第 0 步（零 GPU 成本）
```bash
bash scripts/batched_400_v2.sh --dry-run
```
确认打印的 gate 行里含 `B400_TCGEN05_E4M3_GROUPED` 那一串；确认 build 链（`build.sh 103a` + `touch build.rs` + `cargo build --release`）不变。**注意 dry-run 不做 build/rsync。**

并在节点上做一次**符号预检**（比一次 20 分钟的 serve 便宜得多）：
```bash
ssh $NODE 'nm -D ~/ferrite/kernels/cuda/libferrite_kernels.so | grep -E \
  "dsv41_expert_gemm_e4m3_grouped|dsv41_route_(group|gather_rows|scatter_rows)"'
```
四个符号缺任何一个 → arm 必然 decline（`supports_route_group()` 要求三个 route 符号齐备），**不要跑 arm**。

### 第 1 步：**基线轮（arm OFF）**
```bash
bash scripts/batched_400_v2.sh
```
作用：(a) 确认重建后的 pair 同源、服务健康；(b) 拿到**同一次 rebuild** 的 verify/step 参考；(c) 验证 §1.3 的耦合——基线轮的 `.env` 里**不应出现**这 5 个门中的任何一个（`grep -E 'TCGEN05|GROUPED|EXPERT_ILV|GATEUP_FUSE' <tag>.env` 应为空）。

### 第 2 步：**layout 对照轮（推荐，但脚本当前不支持）**
目标配置：**只** `DSV41_GATEUP_FUSE=0 DSV41_EXPERT_ILV=0`，**不**设 `TCGEN05_E4M3`/`GROUPED`。
- 它测的正是「arm 的必付税」（§3.4 的 ~+0.4~0.5ms），也是 arm 轮真正的比较基准。
- 现状：脚本的 `TC5_GATES` 是整组加的，没法只加这两个 ⇒ 需要**手工起一个 serve** 或给脚本加一个 knob（属源码改动，本次只读未做）。
- **如果第 3 步读数出现「无变化或轻微退化」，这一轮是唯一能把「kernel 慢」与「门税」分开的实验**；不做它，任何非乐观读数都归因不清。

### 第 3 步：**tcgen05 grouped 轮**
```bash
B400_TCGEN05_E4M3_GROUPED=1 bash scripts/batched_400_v2.sh
```

**为什么把 arm 放在基线之后而不是之前**：arm 需要同一棵树的 rebuild 才可信（脚本已保证），而基线轮是零额外成本的对照；但**如果时间很紧，可以在第 1 步之前插入一次「短 prompt 冒烟」**（`MAXTOK=100~200`），用来在 2 分钟内抓「非法指令 / serve 起不来 / 拉丁红线」，避免用一次完整 1000-token 轮去撞一个必然崩的 kernel。

**不建议把 arm 当作第一个 GPU 接触点**：这个 kernel 的 GPU 首触应该是**隔离 parity**（本仓库没有 e4x 的 harness，`tcgen05_bench.sh` 只覆盖 swapAB mxf4 入口），做不到时就至少先冒烟。

### 第 4 步（条件触发）
- **若红线破**（拉丁/双字/缺句）：按 §8.2 的最小回退二分——`GROUPED=0`（保留 `TCGEN05_E4M3=1`）→ `TCGEN05_E4M3=0`。注意 `GROUPED=0` 时全 e4m3 tcgen05 家族一起消失（含单行 swapAB 臂），所以它能同时排除 R7。
- **若读数在「基线」与「基线+0.5ms」之间摆动**：做第 2 步的对照轮，否则结论不可下。
- **若读数确有 −2ms 级改善**：再考虑补一个「gateup 微基准」（新 harness）去回答「离 1.0ms 还差多远」，以及是否值得为 down 补一个 tcgen05 臂（那才是 −6.8ms 的另一半）。

---

## 8. 判读清单（每轮跑完必须做的）

### 8.1 arm 是否真的生效（本项目 #1 陷阱：armed 却测了旧路径）
- 在 serve 日志里 grep 三条一次性告警，**任何一条出现 = arm 没生效**：
  - `expert_grouped_skipped_note`: `"DSV41_EXPERT_GROUPED is set, but the routed MoE still runs the proven per-(row, slot) launches"`（`chain_dev.rs:920`）
  - `tcgen05_e4m3_ext_skipped_note`: `"…the dense-tile tc5::e4x arm declined…"`（`:789`）
  - `tcgen05_e4m3_skipped_note`: `"…the routed MoE still dispatches the gate/up to the proven GEMV/GEMM"`（`:772`）
- `<tag>.env` 里必须**逐门**看到：`DSV41_EXPERT_ACT_E4M3=1`、`DSV41_EXPERT_TCGEN05_E4M3=1`、`DSV41_EXPERT_GROUPED=1`、`DSV41_GATEUP_FUSE=0`、`DSV41_EXPERT_ILV=0`；且 **`DSV41_EXPERT_ILV` 缺失也行**（§1.3：GATEUP_FUSE=0 已蕴含 ILV=0），但 `DSV41_EXPERT_ILV=1` 出现就是配置错误。

### 8.2 红线（唯一正确性判据）
- `latin=0`、`dbl=0`、`has_kaishen`、`chars` 与 `rounds`（~450 轮）—— 脚本的 `metrics` 已给。
- **不要**用「四段文本与基线逐字节相同」做判据（§4.2 第 1 条）。
- 破红线时的最小二分：`GROUPED=0` → `TCGEN05_E4M3=0`（**不要**先关 E4M3，那是激活格式，会引入第二个变量）。

### 8.3 归因三件套
1. `verify_ms` / `steady_median` 相对**第 1 步基线**的位移（不是相对历史基线）；
2. `[verify_graph]` 三态（`captured` / `capture FAILED` / 无行）——图没接上时读数口径变了；
3. **`[gmo]` / `[phs]`**：确认窗口内是否触发过 `moe()`（R7 的 swapAB e4x 臂）。

### 8.4 预期落点表（读数的三种解释）

| 读数（相对同次 rebuild 基线） | 解释 | 动作 |
|---|---|---|
| `steady_median` 改善 **≥3ms** | kernel 生效且高效 | 记录；考虑补 gateup 微基准 + 论证 down 臂 |
| 改善 **1~3ms** | kernel 生效但受 §5.2 的成本结构限制 | 看 nsys kernel duration；不要急着调参 |
| **无变化** | 要么 decline（查 §8.1 告警），要么被 +0.5ms 门税抵消 | 先查告警；再做 §7 第 2 步对照轮 |
| **退化 0.3~0.7ms** | 高概率 = **回退**（门税），不是 kernel 慢 | 查 §8.1 告警；若告警在，则 kernel 根本没跑 |
| **红线破** | §4.2 的 `[OPEN]` 命中（最可能）或 R7 | §7 第 4 步二分 |

---

## 9. 一句话总结

**这个 arm 值得跑，但它的名号（−6.8ms）配错了 kernel**：−6.8ms 属于「swapAB + kRing TMA、替换 gateup **和** down」的另一套设计；
本 arm 是 `tc5::e4x` 的 dense masked tile，**只替换 gate/up（上限 4.8~5.8ms）、无异步 staging、M=128 固定带来 ~120× 张量核过量、且从未在任何 GPU 上执行过**。
所以正确的预期是 **−1 ~ −2.8ms（乐观 −3.8ms）**，并按「先基线、再门税对照、后 arm」的顺序跑，把「armed 却测旧路径」和「回退被读成退化」这两个陷阱先堵住。

---

*工部 · 只读分析 + 本文件（唯一产出）；未执行任何 GPU 命令、未改动任何源码。*
*所有 ms/launch 数均标注来源；源自不同 kernel 设计的推算项（§5.2/§5.3）已显式标注为「非实测」。*
