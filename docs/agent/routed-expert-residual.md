# routed experts 残差 +7.10ms 的可收空间（户部 · 资源与性能）

> 目标问题：verify 里 **routed experts 8.30ms**、带宽下限 **0.408ms**、launch 1.20ms
> ⇒ **残差 +7.10ms（85%）**，达成带宽 **377 GB/s = HBM 峰值的 4.9%**。**这 7.10ms 能收多少？**
>
> 方法：只读代码 + shape 精确推导 + 仓库内实测单价折算（`STATUS.md` 的 nsys/微基准数）。
> **本机无 GPU ⇒ 不跑 nsys/ncu；所有 ms 标了来源，未实测项在 §7 列明。**
>
> 户部 · 2026-09-12 · HEAD `fdc1c15` · 基线 `DSV41_TIMING` verify=37.31ms（m=5）

---

## 【分析范围】

| 文件 | 取用内容 |
|---|---|
| `crates/ferrite-models/src/dsv41/chain_dev.rs:7991-8330` | `moe_rows`：routed 的 4 个 launch 站点 + 各 env 门（`GATEUP_FUSE`/`DOWN_FUSE`/`EXPERT_FP4_MODE`/`EXPERT_ACT_E4M3`） |
| `kernels/cuda/dsv41_experts_mxf4.cu` | `expert_gemv_fp4_batched_kernel`（:1130-1830）、`expert_gemv_fp4_down_reduce_kernel`（:1857-2260）、两个 launcher（:2401/:2527/:2572）、tcgen05 骨架（`tc5`/`tc5::mxf4`，`:3800+`，被宏关住不进 .so） |
| `crates/ferrite-models/src/dsv41/config.rs:502` + `configs/dsv41_flash.json` | production 真实参数（`production_shapes()` 断言逐条核对） |
| `crates/ferrite-models/src/dsv41/weights.rs:446-540` | `K_ATOM=64` / `padded_inter` / TP8 后 `inter_local = padded(2304/8) = 320` |
| `crates/ferrite-dsv41/STATUS.md` | 唯一的时间来源：:3585（nsys 每步每卡）、:5026（gateup 443GB/s + IPC 0.8/4）、:6959-6982（down 17.2/23.8µs 回归归因）、:6607（地板 vs 实测）、:715-724（mode 2/3/4 微基准） |
| `docs/agent/verify-ms-breakdown.md` / `verify-calc-floor.md` | 上一轮账本（本文件的输入口径，冲突处已在 §6 标注） |

**口径常量**：HBM 峰值 7.672 TB/s = 7672 MB/ms；B300 148 SM × 128 FMA/SM/clk × 1.865 GHz = 35.3 TMAC/s；issue 率 148×4×1.865e9 ≈ 1.10e12 instr/s。

---

## 0. 先说一个口径冲突（会影响下面所有数字，必须先钉死）

任务给的形状是 `k=dim=7168 → n=inter_local=256`、`topk=8`、**"3133MB 已是权重一次读"**。但仓库里的 production 配置是：

| 量 | 仓库实际（`config.rs::production_shapes()` 断言 + json） | 任务给的 |
|---|---|---|
| dim | **5120** | 7168 |
| moe_inter / TP8 → inter_local | 2304 / `padded(288)` = **320** | 256 |
| topk（`num_experts_per_tok`） | **6** | 8 |
| n_routed | **384** | — |

**3133.4MB 与 8.30ms 只有在仓库配置下才自洽**：

```
assignments = 40 层 × m 行 × topk = 40 × 5 × 6 = 1200
每 assignment 权重 = 2.6112 MB   (§1 推导)
1200 × 2.6112 MB = 3133.4 MB    ✓ 与 verify-calc-floor §2 逐位一致
200 行 × 24.1µs (gateup med) + 200 行 × 17.4µs (down med) = 8.30 ms  ✓ 与实测一致
```

按任务形状（dim 7168 / il 256 / topk 8 / m 5）算：每 assignment 2.9245MB × 1600 = **4679MB**，不是 3133MB（差 1.49×）。
⇒ **本文件以仓库 production 配置（5120/320/6/384）为主口径**，任务形状的数全在 §5 给对照列。
（`dsv41_experts_mxf4.cu:1863` 那句 "production dim = 7168" 是**微基准期的历史注释**，与今天的 `cfg.dim=5120` 不符；
`STATUS.md:5898` 的 "生产形状 dim=7168, k=320" 同源。**这是一处该修的注释/口径漂移**，建议单独确认。）

---

## 1. 每 assignment 的真实工作量（第一手推导）

一个 assignment = **(一个激活行, 一个被路由到的专家)**。每 assignment 恒为两次 GEMV：

| 项 | GEMM 形状 | MACs | FLOPs | **权重字节** | 激活字节 |
|---|---|---:|---:|---:|---:|
| **gate/up**（w1‖w3 融合） | k=dim=5120 → n=2·inter_local=640 | 3,276,800 | 6.55 M | **1.7408 MB**<br>w1+w3 = 2×(320×(5120/2)=819,200 + 320×(5120/32)=51,200) | 2720 B<br>fp4 packed 2560 + e8m0 scale 160 |
| **down**（w2） | k=inter_local=320 → n=dim=5120 | 1,638,400 | 3.28 M | **0.8704 MB**<br>5120×(320/2)=819,200 + 5120×(320/32)=51,200 | 1280 B<br>f32（swiglu 后的激活） |
| **合计** | | **4,915,200** | **9.83 M** | **2.6112 MB** | 4000 B |

- **算术强度 AI = 3.76 FLOP/byte** ⇒ 比 B300 的 fp4 机器平衡点（~数百）低两个数量级：**它天生是访存/解码题，不是算力题**。
- 每步（1200 assignments）：**5.898 GMAC = 11.80 GFLOP**，**3133.4 MB 权重**（= 6.267 **G 个 fp4 值**）。
- 三层地板：**字节 0.408ms** < 启动 1.20ms（400×3µs 口径）；**fp32-SIMT 算力地板仅 0.17ms**。
  实测 8.30ms ⇒ 字节地板的 **20.3×**、算力地板的 **49.7×**。
  ⇒ **§0 的结论不变：既不是带宽、也不是算力，两者都解释不了这 7.10ms。**

---

## 2. 377 GB/s 的成因：不是"每 block 工作量太小"，是**解码指令流 + 低占用**

### 2.1 先排除两个候选

**(a) "tile 太小 / SM 空转"——不成立（与 verify head 那次不同）**
gateup 的 grid = `(n_total/warps, slots, rows)` = `(640/8=80, 6, rows)`；每 CTA 8 warp × 32 = 256 线程，
**每 warp 独占一整行输出**（k=5120 的完整点积）。m=5 时 grid = **2400 CTA**，每 CTA 只有 22.5KB smem ⇒
**有 2400 个 CTA 可调度**，不存在"每 block 工作量太小"的填不进去问题。down 更极端：grid = `(dim/8=640, 1, rows)` = **3200 CTA**。
⇒ **SM 不缺活干；缺的是"每个 warp 的指令效率"。**

**(b) "launch 固定延迟"——也不成立（而且仓库已经做过这个实验）**
`moe_rows` 今天**已经**把 m 行折进 `grid.z`（`dsv41_experts_mxf4.cu:2487-2493`，"rows == 1 degenerates to the previous launch exactly"），
所以 routed 的每层 launch 数是 **2~3 发**（`quant_fp4` + `gate_up_batched` + `down_reduce_batched`），
不是 400 发。nsys 侧证：**`expert_gemv_fp4_batched` 105 次/步/卡、Med 83.1µs、合计 8.98ms**（`STATUS:3585`）。

> **关键推论**：把 400 发（逐行）变成 ~100 发（行批）**并没有让 8.30ms 变小**，
> 只是把每发从 20.8µs 变成 83.1µs（5 行摊薄后 16.6µs/行，**−25%**，这是行批唯一的收益）。
> ⇒ **+7.10ms 里"每发固定延迟"的成分已被这个实验证伪；残差在 kernel 内部。**

### 2.2 真正的成因：每 weight value ~2.5 条 L1TEX 指令，且只有 2 CTA/SM 来藏延迟

`dsv41_experts_mxf4.cu:946` 是仓库自己的 ncu 结论，直接引用：

> *"the measured symptom is an expert gateup at **22.2us/call, 443 GB/s and IPC 0.8/4**, i.e. the issue slots are
> **80 percent stalled on operand supply**, not on arithmetic."*（同 `STATUS:5026`：480 块/148SM = 3.24 块/SM）

而每 value 的指令数来自解码形态（`:1155` 的 vec==2 tail）：
**1 LDG.U8 权重 + 1 LDG.U8 scale + 1 LDS.64 LUT + 2 LDS.32 激活 = 5 条 / 2 个 value = 2.5 条/value**。

把两个数乘起来（这是 377GB/s 的算式）：

```
权重 value 数      = 3133.4 MB × 2 value/B        = 6.267 G value
L1TEX 指令数       = 6.267 G × 2.5                = 15.7 G 条
L1TEX 吞吐（148SM × 4 quadrant × 1.865GHz）       = 1.10 e12 条/s
⇒ 指令地板         = 14.2 ms
实测               = 8.30 ms  ⇒ 等效 ~1.46 条/value（~2× 模型内）
```

⇒ **377 GB/s 不是"带宽没打满"，而是"每读一个 fp4 值要付 2.5 条 L1TEX 指令"**：
核在**解码**上花掉了绝大部分 issue 槽（IPC 0.8/4 = 只有 20% 的槽在干活），
而这个解码量 ∝ **权重 value 数（6.267G）**，与字节宽度无关。
**这就是"字节不可压 ⇒ 机械解不存在"的机理**：压字节不减少 value 数（fp4 已是 2 value/B），
要减的是**每 value 的指令数**或**把解码整段删掉**。

### 2.3 两个半边的限流点完全不同（这是路径选择的依据）

| | gateup（占 4.82ms） | down（占 3.48ms） |
|---|---|---|
| smem | `dim*4 (20 KB s_act) + 256*8 (LUT) = 22.5 KB` | `slots*inter*4 (7.7KB) + 2 KB = 9.7 KB` |
| 限流 | **smem：48KB / 22.5KB = 2 CTA/SM**（16 warp = 64 warn 上限的 **25%**）；launcher **没有** `cudaFuncSetAttribute`，天花板就是 48KB | **寄存器：mode 2 = 40 regs → 6 blocks/SM**（`__launch_bounds__(256,6)`，"1.01 waves"）；mode 3 = 56 regs → 4 blocks/SM → **1.51 waves = 生产 +38% 回归**（`:1857-1894`） |
| 每 CTA 的 L1TEX 构成 | 权重 LDG + **s_act 的 LDS 重读**（每 warp 读满 k=5120 → 1280 条 LDS.128） | 每 warp 读自己 slot 的 320 float 激活（**80 条 LDS.128**）vs 权重仅 160 B（~10 条 LDG.128）⇒ **激活 LDS 指令是权重 LDG 的 ~8×** |

⇒ **gateup 的 20KB `s_act`（占其 smem 的 89%）就是它 2 CTA/SM 的根因**；
**down 的 8:1 激活 LDS 重读**（而不是 DRAM 字节）是它的地板距离。两者都是**指令/占用**问题，不是带宽问题。

---

## 3. 三条削减路径：量化 + 工作量

> **评估口径**：以实测 8.30ms 为基；收益全部按"改后落点 ms"给；
> 「把握」按仓库已有的实测/微基准证据强度标（高=有实测，中=有代码，低=纯外推）。

### 3.0 路径收益总表（先看这个）

| 路径 | 削的是什么 | 落点（ms） | 预期收益 | 把握 | 工作量 |
|---|---|---:|---:|---|---|
| **(a) assignment 合并 / expert 分组** | 权重**唯一读**（去除重复专家的重读） | 8.30 → **8.00** | **−0.30ms**（3.7%） | **高**（组合数学精确） | 中（计数排序 + 分组 kernel 的 rows/N 重排） |
| **(b) tcgen05 `mxf4` swapAB routed path** | **整段 SIMT 解码删掉**（张量核直读 packed fp4 + e8m0） | 8.30 → **1.0~1.5** | **−6.8 ~ −7.3ms** | **中**（骨架已落地、MMA 形式 GPU 已验 EXACT；但 dispatch 不可达 + 无 TMA ring） | **高**（1-3 周 GPU 迭代，研究级） |
| **(c) 激活 fp4/f32 → fp8 e4m3** | **s_act 的 20KB → 5KB（gateup 占用 2→6 CTA/SM）+ down 的激活 LDS 指令 ÷4** | 8.30 → **5.0~6.3** | **−2.0 ~ −3.3ms** | **中**（e4m3 残差原语已落地，见 `expert_act_e4m3`；但 s_act 的类型切换是新改动） | 中（kernel 变体 + 量化口径对齐） |
| (d) *[仓库已测臂]* down 的 4-value 解码 @40 regs | down 的 L1TEX 指令/value ÷2（2.5→1.25） | 8.30 → **7.95** | −0.35ms | **中**（微基准 0.90× 已测，但 40-reg 版未在 serve 跑过） | 低-中（压寄存器） |

**合并结论：可收空间 ≈ −6.8 ~ −7.3ms（路径 b），把 8.30 打到 ~1.0-1.5ms = HBM 地板的 2.5-3.7×；
若只做 (a)+(c)+(d)，能收到 ~7.9ms，残差从 7.10 降到 6.7ms —— 即"不看张量核，最多收 5%"。**

---

### (a) assignment 合并：**收益由 `n_routed/topk` 唯一决定，production 下只有 3.7%**

**机制**：合并只对"**同一个专家被多行/多 slot 命中**"有效——合并后该专家的 2.6112MB 只读一次，
且**只有合并后 N>1 才能把每 value 的解码摊到多行上**。所以去重率就是收益上限：

```
去重因子 = (m × topk) / E[unique]      E[unique] = n_routed·(1 − (1 − 1/n_routed)^(m·topk))
```

| n_routed | topk | m | draws | E[unique] | 权重字节节省 | 倍数 |
|---:|---:|---:|---:|---:|---:|---:|
| **384（production）** | **6** | 5 | 30 | 28.89 | **3.7%** | ×1.04 |
| 128（MTP 层 `moe_config(41)`） | 3 | 5 | 15 | 14.21 | 5.3% | ×1.06 |
| 256 | 8 | 5 | 40 | 37.10 | 7.3% | ×1.08 |
| 64 | 8 | 5 | 40 | 29.91 | 25.2% | ×1.34 |
| **8** | **8** | 5 | 40 | **7.96** | **80.1%** | **×5.02** |

- **任务里"40 个 assignment 里有 ~40/8=5 个重复专家"只在 `n_routed ≈ topk` 时成立**。
  production 是 **384 选 6**，两行命中同一专家的概率只有 9%，跨 5 行总共 ~1 对重复 ⇒ **只有 3.7%**。
- 落点：权重 3133.4 → 3021MB。但**这 112MB 省的是"无效重读"，而核本来就不是带宽受限的**
  ⇒ 只兑现为 **112MB / 377GB/s ≈ 0.30ms**（不是 112MB/7.6TB/s 的 0.015ms —— 这点很重要：
  **在低效核里省字节，只能按低效核的带宽兑现**）。
- **与 expert-major（GLM 负结果）的区别**：GLM 那次是把 **token 排序**（重排激活、换 L2 访问序），
  本路径是把 **`ids` 相同的 assignment 归并后把"行数"喂给 batched kernel 的 `rows`/N 维**。
  机制上确实不同，**但收益上限仍被上表钉死**：production 下两者都是 ≤4% 级别。
- **前置**：`route_topk` 后加一次 counting sort（`ferrite_kernels.cu:4903` 已有 `moe_sort_assignments`
  ——"expert-major Phase 1: counting sort"），再由 host 把 unique 数与行索引传给 kernel。
- **工作量**：中（1 个排序 kernel 已存在 + `rows`/`ids` 语义改造 + 逐位一致性复验）。
- **把握：高（组合数学），收益：低（−0.30ms）。** ⇒ **不建议单独做，只作为 (b) 的附属。**

### (b) tcgen05 / `mxf4` routed path：**唯一能真正删掉解码的路，但 dispatch 不可达**

**机制**：SIMT 路径的 2.5 条指令/value 是因为**每个 fp4 值都要经过 LUT 解码再喂 FMA**。
tcgen05 `kind::mxf4.block_scale.scale_vec::2X` 让张量核**直读 packed fp4 + e8m0 scale**，
解码与乘加一起在 TMEM/MMA 里完成 ⇒ **每 value 的 SIMT 指令数 → ~0**，成本回到纯字节 + TMA issue。

**为什么它天然适配 verify**：swapAB 语义 `D[M=权重行, N=激活列]`，M 侧满 tile（gateup 640、down 5120 都整除 128），
**N 侧最小合法 8**，而 verify 恰好有 **m=5 行**（pad 到 8）——**主链 decode（1 token）才是 N 侧浪费 7/8 的那个**。
⇒ verify 是这条路径**最划算的落地场景**（`expert-tcgen05-plan.md` 把 N=1→8 的 8× 激活冗余算成"可忽略"）。

**落点外推**（按 `expert-tcgen05-plan.md §1d/§1e` 的自算 + 字节地板）：
```
字节地板                        0.408 ms
ring 深度 3→8、in-flight 8→28 KB/CTA、240 CTA = 6.7 MB
  ≈ HBM 带宽-延迟积（8TB/s × ~700ns = 5.6MB）  ⇒ 有资格打满
乐观落点 1.0 ~ 1.5 ms（= HBM 的 2.5~3.7×）      ⇒ 残差 7.10 → 0.6~1.1 ms
```

**当前状态（代码级核对）**：
- 两臂骨架**都已落地**但 `build.sh` **不定义宏** ⇒ **不进 .so**，`DSV41_EXPERT_TCGEN05[_MXF4]` 门禁后的函数体在设备上**根本不存在** ⇒ 任务说的"dispatch 不可达"**确认**。
- mxf4 臂（默认臂）：`tc5::mxf4::expert_tcgen05_gateup_mxf4_kernel`，M=128/N=8/kKStep=64/kRing=8/128 线程，
  ptxas **92 regs / 0 spill / 40KB static smem**，PTX 内含 `tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X` + `cp.async.bulk` + mbarrier。
- 其 MMA 形式（`tests_tcgen05_mxf4.cu`）是**唯一在 GPU 上数值验证过的 fp4 tcgen05**（`maxdiff = 0.000e+00`，7 shapes）。
- **入场券 vs 缺口**：① 只有 gateup 骨架，**down 的 swapAB + fused asc-slot reduce 未写**；
  ② **240 CTA 的 grid 是 `(30, slots)`，slots=1 只有 30 CTA ≈ 15 SM** ⇒ 必须 slots=8（这与 verify 的 topk=6 冲突，要么 pad、要么 K-split）；
  ③ K-split reduce 未写；④ pool/ids 间接寻址 + Rust FFI 未接；⑤ 没有 2D tensor TMA（每 stage 258 条 TMA issue）。
- **风险**：本仓库 tcgen05 已负过一次（`STATUS:6607` 的 16.8 GB/s / 0.2% 地板）；
  但**那次负的原因是"未换向 + 无 cp.async"**（M=5/N=1 的死方向 + `:341-358` 的 LDG→STS），
  **两条今天都有解**（swapAB + kRing TMA ring）⇒ 本次不是同一堵墙。
- **工作量**：**高**，研究级 1-3 周 GPU 迭代（layout/descriptor、激活侧 e8m0 量化、epilogue 的按专家加权累加、
  K 序变化后的 parity 复验）。**前置门：先跑单层 microbench 打 22.2µs（gateup）/17.2µs（down），打不过不进集成。**

### (c) fp4/f32 → fp8 e4m3 激活：**不是省 DRAM 字节（激活只占 0.13%），是省 smem 与 LDS 指令**

任务把这条标成"对权重流无影响"——**对 DRAM 确实无影响，但它的真收益在另外两处**：

| 半边 | 现状 | 换 fp8 后 | 收益 |
|---|---|---|---|
| **gateup** | `s_act = dim*4 = 20 KB`（f32 反量化激活，**占 smem 的 89%**）→ 48KB/22.5KB = **2 CTA/SM** | `s_act = dim = 5 KB` → smem 7KB → **6 CTA/SM**（+2KB LUT） | 常驻 warp 16 → 48（25%→75%），直接打 §2.2 的"80% issue 槽停等" |
| **down** | 每 warp 读 320 float 激活 = **80 条 LDS.128**，而权重只有 ~10 条 LDG.128 ⇒ **激活 LDS 指令是权重的 ~8×** | 320 B fp8 = **20 条 LDS.128** | down 的 L1TEX 指令 ~90 → ~30/CTA（**−67%**） |
| 权重流 | 3133.4 MB | 3133.4 MB（不变） | **0** |

⇒ 这条路径的真实收益 **不是"激活字节 ÷4"（那只有 4MB/步 = 0.01ms）而是"占用 ×3" + "down 的 LDS 指令 ÷4"**。
**前提已就位**：`expert_act_e4m3()` 门（`chain_dev.rs:698`）+ `dsv41_sub_dequant_fp4` 残差原语（"e4m3 修复"）已落地，
只是默认 OFF 且当前口径是"两趟 e2m1×2"（`:210-220`，为修 round-18 的 `[inter]` vs `[2*inter]` 契约）。
**落点估算**：gateup 4.82ms × (0.5~0.65) ≈ 2.4~3.1ms；down 3.48ms × (0.65~0.8) ≈ 2.3~2.8ms
⇒ 合计 **5.0~6.3ms**（−2.0 ~ −3.3ms）。
**工作量**：中（s_act 类型 + LUT 解码目标类型 + 与 `sub_dequant` 两趟口径对齐 + 逐位/精度复验）。
**把握：中**（占用的算术是硬的；但 down 的 −67% 指令数只是静态计数，**没有实测** ⇒ 标 §7-V3）。

### (d) 附：down 的 4-value 解码 @ 40 regs —— 仓库里最便宜的一枪

`dsv41_experts_mxf4.cu:715-724` 的隔离微基准（**生产形状 `dim=7168, k=320, 896 blocks, 256 thr, sm_103a`**）：

```
mode 2, 40 regs, 6 blocks/SM, 1.01 waves : 1.00 (baseline)
mode 3, 40 regs, 6 blocks/SM, 1.01 waves : 0.90   (+__launch_bounds__(256,6))
mode 3, 56 regs, 4 blocks/SM, 1.51 waves : 0.87   <- 最快，但生产 +38%（1.51 waves 悬崖）
mode 4, 40 regs, 6 blocks/SM : 1.03 | mode 4, 62 regs, 4 blocks/SM : 0.97
```

- 生产回归的**根因已定位**（`:1863-1894`）：56 regs ⇒ 4 blocks/SM ⇒ **1.51 waves**（640 块 / 592 常驻），
  而 **6 blocks/SM 的 0.90 臂从未在 serve 跑过**——因为 mode 3 在真实 kernel 里压不到 40 regs。
- ⇒ **动作**：把 4-value 解码的寄存器压力压到 40（LUT 索引拆算、切分内层、去掉同时活的 4 个 float2），
  目标 **down 17.2 → ~15.5µs**（0.90×）⇒ **−0.35ms**。
- 工作量**低-中**、把握**中**（微基准已测；差"40 regs 可行"这一步）。**这是全表性价比最高的一枪。**

---

## 4. 决策建议（户部口径：先便宜后贵）

1. **(d) 先打**（0.90× 臂在 40 regs 落地）：−0.35ms，低风险、有微基准背书。
2. **(c) 紧接着**（激活改 fp8/e4m3，同时吃 gateup 占用 ×3 与 down 的 LDS ÷4）：−2.0~3.3ms，中风险。
   注意这与 `expert_act_e4m3` 的**精度口径**是同一件事，**一次改到位**，别分两次 A/B。
3. **(b) 并行立项**（tcgen05 mxf4 routed path）：**唯一能把残差从 7.1 打到 <1.2ms 的路**，
   但它是研究级（1-3 周）且**必须先过单层 microbench 门（22.2/17.2µs）**。建议把 verify 当第一落地场景
   （N=8 天然匹配 m=5），并把 §3(b) 列的 5 个缺口排成里程碑。
4. **(a) 只作为 (b) 的附属**：production 下 3.7% = −0.30ms，单体不值一次 A/B；
   但 (b) 落地时它变成必需（张量核要按 expert 分组喂 tile），**顺手做**。
5. **不要再做的**：任何"压权重字节"的尝试（fp4 已 2 value/B；去重 3.7%）；
   单独把 `DSV41_GATEUP_PIPELINE` 深度调深（**已实测 2/5 各 +0.04ms**，`:963`）。

**一句话**：这 7.10ms **不是"分散的小浪费"，是一件事**——**6.267G 个 fp4 值每个要 2.5 条 SIMT 指令去解码，
且在 2 CTA/SM（gateup）/ 6 blocks/SM（down）的占用下藏不住延迟**。
删掉解码（b）能收 −6.8ms；不删解码、只提占用与指令宽度（c+d）能收 −2.4~3.7ms；只去重（a）收 −0.3ms。

---

## 5. 任务形状（dim=7168 / il=256 / topk=8 / m=5）的对照列

| 项 | 仓库 production（5120/320/6/384） | 任务形状（7168/256/8） |
|---|---:|---:|
| 每 assignment：gate/up | 3.28 MMAC / 1.7408 MB | 3.67 MMAC / 1.9497 MB |
| 每 assignment：down | 1.64 MMAC / 0.8704 MB | 1.84 MMAC / 0.9748 MB |
| 每 assignment 合计 | 9.83 MFLOP / **2.6112 MB** | 11.01 MFLOP / **2.9245 MB** |
| 每步 assignments | 1200 | 1600 |
| 每步权重字节 | **3133.4 MB** ✓（与账本一致） | **4679 MB**（≠ 任务的 3133MB） |
| 去重因子（n_routed 未知） | 3.7%（384） | 25%（64）/ 80%（8）/ 7.3%（256） |
| 结论 | 本文件主口径 | 若真换成该形状，**3133MB 口径需重算，且去重因子取决于 n_routed** |

> **需要确认的一条**：任务里的 `dim=7168 / inter_local=256 / topk=8` 是从哪来的？
> 若确实是新模型/新配置，**§1 的表要整表重算**，且 `n_routed` 必须一起给（它决定 (a) 的 3.7% vs 80%）。

---

## 6. 与前一轮账本的差异（口径冲突，不改结论方向）

| 处 | `verify-ms-breakdown.md` | 本文件 | 影响 |
|---|---|---|---|
| routed launch 数 | **400**（40L × 5 行 × 2） | **~100**（`grid.z=rows` 已批行，nsys 实测 `expert_gemv_fp4_batched` 105 次/步/卡） | 400 是**逐行口径的记账**（200×24.1 + 200×17.4 = 8.30ms ✓ 同样自洽）；但"launch 半 1.20ms"对 routed 是**虚账**——实际 submit 只有 ~0.3ms，**残差比 7.10ms 更集中在核内** |
| 残差性质 | "核效率（每发 20.7µs）" | "**每 value 2.5 条 L1TEX 指令 + 低占用**" | 定性一致；**定量上"每发固定延迟"应剔除**——行批实验已证它不解释时间 |
| 路径 C 的预期 | "tcgen05 → 8.30 → 1.5~2.5ms，把握低-中" | 1.0~1.5ms，**机制从"换核"细化为"swapAB + kRing TMA ring"**，且有"那次负结果的真实原因是未换向+无 cp.async"的订正 | 更乐观一点点，且**风险来源被重新定位**（不是 tcgen05 本身，是 staging） |

---

## 7. 待验证（无实测支撑，必须 nsys/ncu 复核）

| # | 断言 | 为什么需要验证 |
|---|---|---|
| **V1** | routed = 8.30ms 且 launch 数 ~100（不是 400） | 用 `DSV41_TIMING` + nsys 按 kernel 名聚合，**数 `expert_gemv_fp4_batched` / `down_reduce` 的 instances/步/卡**；`STATUS:3585` 的 105 是"39 层/8 rank 折算"口径，需在本步型上复核 |
| **V2** | "2.5 条 L1TEX 指令/value" 是 377GB/s 的主因 | 静态计数（源码级）成立；**实测需 ncu**：`Issue Slots Busy`、`L1TEX throughput`、`Warp Occupancy`。两半边的 `IPC`/`stall reason` 要分开取 |
| **V3** | 激活改 fp8 后 gateup 占用 2→6 CTA/SM、down L1TEX 指令 ÷4 | 占用是 smem 算术（硬）；**"指令 ÷4 ⇒ 时间 ÷2" 是静态外推**，未实测 |
| **V4** | (a) 去重 3.7% ⇒ −0.30ms | 组合数学硬；**"省字节按 377GB/s 兑现"是假设**（若核是纯指令受限，省字节可能**一分不省**——那 (a) 就归零） |
| **V5** | (b) 落点 1.0~1.5ms | 依赖 `kRing=8` 的 in-flight 估计与"slots=8 时 240 CTA 逼近带宽-延迟积"，**纯外推**；且 verify 的 topk=6 与骨架的 slots=8 需先对齐 |
| **V6** | down 的"激活 LDS : 权重 LDG ≈ 8:1" | 按 warp 读 320 float（80×LDS.128）vs 160 B 权重（~10×LDG.128）静态推导；**未含 LUT/LDS.64 与 bank conflict**，需 ncu 的 `smsp__inst_executed_op_shared_ld` 核对 |
| **V7** | `dsv41_experts_mxf4.cu:1863` 的 "production dim=7168" 与 `cfg.dim=5120` 冲突 | 注释/口径漂移，影响所有基于"生产形状"的微基准外推（mode 2/3/4 那张表就是按 7168/896 blocks 测的，**而生产是 5120/640 blocks**）⇒ **该微基准的 0.90/0.87 能否外推到生产形状本身就要复核** |

---

*户部 · 只读分析：本轮唯一动作是读代码/文档 + 算术推导，未执行任何 GPU 命令、未改动任何源码。*
*本文件为唯一产出；所有 ms 均标了来源，未实测项已集中在 §7。*
