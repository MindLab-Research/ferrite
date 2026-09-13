# G1 — MoE expert 并集去重：路线 A/B 判决 + 路线 A′ 实施（2026-09-13）

> 工部 · 2026-09-13 · **禁止 GPU/e2e**（本文件的所有代码结论都是静态 + compile-only）。
> 现场核对：`kernels/cuda/dsv41_experts_mxf4.cu`、`dvs41_route.cu`、
> `crates/ferrite-models/src/dsv41/{chain_dev.rs,device.rs}`。
> 上游：`docs/agent/sglang-verify-model.md`（G1 票面）、`tcgen05-716-e4m3-confound-verdict.md`
> §4-§5（拆门 + 门链 + GPU 手册）、`tcgen05-e4m3-grouped-expectation.md`（e4x 的形态风险，已应验）。

---

## 0. 一句话判词

**路线 B 死（主 agent GPU 实测 +16ms）；路线 A 的"字面版"（只排序、喂 batched kernel 的 rows 维）
在数学上不是去重杠杆** —— `rows` 是 `blockIdx.z`，一个 (row, slot) 一个 CTA，排序只改执行顺序；
**真正的去重是把「同一 expert 的多条 assignment」放进同一个 CTA 里循环，并把该 expert 的权重块
stage 一次**。这不是 tcgen05 专属机制：它就是一个 SIMT 循环。本次交付 = 这个循环的 gate/up 版
（`expert_gemv_fp4_gate_up_grouped_kernel`），**零布局开销、零 ABI、零 Rust 改动、逐位等价**。

---

## 1. 路线 B 判死（主 agent 实测，本文件只做机理归因）

实测（commit `0eb5464`）：0/8 Err ✓（拆门修复生效，prefill 不再被单行 swapAB arm 打死）**但
verify = 40.43ms vs 基线 24.5ms（+16ms）**，mean-k 2.140 保持、数值健康。

两个成本来源在代码里都可指认，且**没有一个是"tcgen05 有 bug"**：

| 成本 | 机理 | 出处 |
|---|---|---|
| **布局开销（3 个中继 kernel/层）** | `route_group` 是**单 block、thread 0 串行**：384 次 expert 循环 + `m*topk` 内层（36 次 × 6 行）⇒ 每层几百个串行步；再叠 `route_gather_rows`（n_assign 个 block）+ `route_scatter_rows`（n_assign 个 block）。这三个 kernel **不计算任何数**，纯粹为 e4x 准备/回收布局 | `dsv41_route.cu:234-315`（`route_group_kernel`，`if (threadIdx.x == 0)` 内两个大循环）、`:447`、`:459` |
| **e4x 的形态浪费** | `e4m3_gemm_grouped_kernel` 的 M 是硬件固定的 128（`kMTile`），而 verify 形状下**每 expert 只有 1-3 行**（`m<=6, topk=6`）⇒ 每个 (expert, m-tile) 跑满 128 行 MMA，**有效行 ~1/128**；grid = `(n_total/64) × 1 × n_assign(36)` 里绝大多数 CTA 在 `*n_active` 处立刻退出 | `.cu:6657`（`e4m3_gemm_grouped_kernel`）、`.cu:6930-6932`（grid）、`tcgen05-e4m3-grouped-expectation.md §0`（"M=128 固定、~100x 张量核过量计算"**实测应验**） |

**结论：grouped 的方向没错，错的是它的两个既有实现**（布局中继 + 128 行 MMA）。
`route_gather` 布局**不是**去重的必需品：去重只需要"同一 expert 的 assignment 由同一个 CTA 依次处理"
—— 这件事可以在**已有 `ids` 表上原地选举**（本文件的路线 A′），一个 kernel 都不多加。

---

## 2. 路线 A 字面版（纯排序 + 喂 rows 维）= 不是杠杆（判死）

任务书/sparse 票面的表述是"把 36 个 (row,slot) 按 expert 排序 → 同 expert 的行连续喂
`expert_gemv_fp4_batched_kernel` 的 rows 维（每 expert 一次 launch）"。**这条做不到"权重读 36→|active|"**，
理由是恒等式级（值级）的：

1. `expert_gemv_fp4_batched_kernel` 的 `rows` 是**grid 第三维**，不是 CTA 内循环：
   `const int arow = (int)blockIdx.z;`（`.cu:1385`）→ `slot0 = arow*gridDim.y` →
   `b_use = b_base + e*b_stride`（`.cu:1394-1395`）。
2. kernel 自带的 ROW INDEPENDENCE 注（`.cu:1379-1384`）："rows 之间不共享输出/累加器/smem staging
   —— `arow` 只移动 base pointer"。⇒ **(row, slot) 的排序只改变"哪个 CTA 在什么时候读同一段权重"**，
   每个 assignment 仍然把该 expert 的 `grid.x × rows_per_cta` 行权重从 gmem 读一遍：
   `n_assign × 2·inter × k/2` 字节不变。
3. 因此"每 expert 一次 launch / 段合并 launch"**等价于把同一段权重在时间上挪到一起** ⇒
   只可能改变 L2 命中时序，不改变指令数、不改变 DRAM 唯一字节数（当前 36 个 assignment 的 CTA
   本来就并存在同一波里，L2 共享已经发生）。

⇒ **票面"权重读 36 → |active|"要求的是 CTA 内复用**，不是调度顺序。

---

## 3. 路线 A′（本次交付）：expert 并集 gate/up，SIMT，零布局

### 3.1 机制（一句话）

blockIdx.z = **FLAT assignment 索引** `aa = arow*slots + slot`；CTA 在 kernel 内对 `ids[0..aa)`
做**expert 选举**（只有第一次出现的那个 CTA 干活），干活的 CTA 把该 expert 的**权重行块 stage 一次**，
然后**依次处理该 expert 的每一条 assignment**：每条 assignment 只需 stage 它自己那行激活，
再对 CTA 名下的输出行做点积。

权重字节恒等式（这就是 G1 要的那一条）：

```
per-(row, slot) launch : n_assign × (2·inter 行 × k/2 字节)
本 kernel              : |active| × (2·inter 行 × k/2 字节)     ← 每个 (expert, 行块) 只 stage 一次
```

`grid.x = 2·inter/kGtRows` 个 tile **恰好覆盖两个平面各一次**（gate 半边读 w1/w1s，up 半边读
w3/w3s，tile 不跨 `b_split`），所以"每个 expert 的 gate+up 权重恰好从 gmem 读一遍"是**构造性**的，
不依赖 L2 运气。激活字节不变（每条 assignment 的激活行仍是每个行块读一次，与 batched staging 同量）。

### 3.2 逐位等价论证（值级，逐条）

| # | 量 | batched split arm | 本 kernel | 等价性 |
|---|---|---|---|---|
| 1 | 点积 | `expert_gemv_fp4_batched_kernel` split body（`.cu:1975-2132`，vec 0/1/2/3 + 尾循环 + shuffle 树） | `dsv41_down_row_dot`（`.cu:2644-2763`） | **逐分支同一份代码**：已逐行核对 vec==2（1985-2041 vs 2651-2701）、vec==3（2042-2084 vs 2702-2728）、vec==1（2085-2108 vs 2729-2751）、vec==0（2109-2117 vs 2752-2760）；`vec` 参数 = 调用方的 `g_expert_fp4_mode`，两处同一个值 |
| 2 | 激活 staging | e4m3：`s_act[g*16+q] = e4m3_to_f(pb[q]) * asc`（`.cu:1587-1601`）；fp4：`nb32` 路径（`:1602-1616`）；尾路径（`:1624-1633`） | 同三支，**逐字复制** | 同一浮点表达式、同一 `>>5` scale 索引、同一 nibble 顺序 ⇒ `s_act` 逐位相同 |
| 3 | 权重字节 | `brow = b_use + r*kbytes`，读 `+ (g<<8) + (lane<<3)`（vec2） | smem 副本，读 `s_w + rl*wrs + (g<<8) + (lane<<3)` | 字节拷贝（`ld_uint4_a16` 16B 粒度），偏移同一表达式 ⇒ 同值；`wrs = align16(kbytes)` 保证 8B/4B 对齐前提不变（down twin 已用同一手法） |
| 4 | scale 字节 | `srow = bb_s + r*b_sf_pitch`，读 `srow[j>>5]` | smem 副本，同一表达式 | 同上（物理 pitch `b_sf_pitch` 由调用方传入，与 batched 的 `dsv41_sf_pitch(dim)` 同值） |
| 5 | 归属行 | `row < b_split` → w1，否则 w3 且 `r = row - b_split` | 同一 `hi/rshift` 推导，且 `inter % kGtRows == 0` 保证一个 tile 不跨界 | 同值 |
| 6 | epilogue | `epi_mode==1` 的 clamp（`row<b_split` 只上限，否则双限），`out[row] = x` | 同，写 `out[aa*out_slot_stride + row]` | 目标单元 = batched 的 `out + (slot0+slot)*out_slot_stride`（同一个 flat 索引 aa）⇒ 同一个 cell |
| 7 | 独立性 | ROW INDEPENDENCE：无共享输出/累加器/staging | 同（每 (assignment, 输出行) 的 cell 互斥；s_act 由两次 `__syncthreads()` 保护） | 无论谁先跑，结果同 bit |

⇒ **排序/换 CTA 只改"谁算哪一格、什么时候算"，不改任何一次舍入**；与已交付的 down twin
（`expert_gemv_fp4_down_batched_grouped_kernel`，`.cu:2774`，其 header 就是同一套论证）同构。

### 3.3 交付 diff（全部在 `kernels/cuda/dsv41_experts_mxf4.cu`）

| 位置 | 内容 |
|---|---|
| `.cu:2988` | `kGtRows = 8`、`kGtMaxAssign = 64`（与 down twin 的 `kGdRows/kGdMaxAssign` 同构） |
| `.cu:3000` | `gt_smem_layout`：smem 布局**一处定义**（kernel 与 launcher 共用，防越界） |
| `.cu:3021` | `expert_gemv_fp4_gate_up_grouped_kernel`：本次新 kernel（选举 + 权重 stage 一次 + assignment 循环 + split-body 点积/收尾） |
| `.cu:3197` | `gg_declined_note`：armed-but-declined **响亮一次**（本 repo 的头号测量陷阱纪律） |
| `.cu:3225` | `g_grouped_gateup` 门：`DSV41_EXPERT_GROUPED_GATEUP`，**故意不 AND `DSV41_EXPERT_GROUPED`**（见下） |
| `.cu:3642-3670` | 在**既有入口** `dsv41_expert_gate_up_fp4_batched` 内派发（**无新符号、无 ABI、Rust 零改动**；decline 时回退到原路径） |

**为什么门不与 `DSV41_EXPERT_GROUPED` 相 AND**（与 down twin 的约定不同，是**新数据驱动的修正**）：
down twin 消费"grouped 栈"的语义，所以绑在它上面；本 kernel **不消费任何布局**（选举在 `ids` 上原地做），
若为了 arm 它而打开 `GROUPED=1`，会顺带打开 `route_group/gather/scatter` 这套**纯开销且无消费者**的中继
（正是路线 B 死因之一）⇒ 两者解耦：`GROUPED` 管布局与 e4x/down 消费者，`GROUPED_GATEUP` 管本 kernel；
同时开也合法（那时布局对另外两个消费者是活的）。

### 3.4 编译/资源证据（远端 b300-2，nvcc 13.2，compile-only，无 GPU）

```
nvcc -O3 -std=c++17 -c -gencode arch=compute_103a,code=sm_103a \
     -DDSV41_TCGEN05_GATEUP_E4M3_SKELETON=1 -DDSV41_TCGEN05_GATEUP_MXF4_SKELETON=1 \
     dsv41_experts_mxf4.cu -o /tmp/g1/t5.o
RC=0
[本 kernel] expert_gemv_fp4_gate_up_grouped_kernel: 63 registers, 0 spill stores/loads,
            32 B stack, 1 barrier
```

`cargo check --workspace --all-targets` = EXIT 0（Rust 侧零改动，仅为纪律复跑）。
smem = `gt_smem_layout(5120)` = `align16(8*160) + 8*align16(2560) + 5120*4 + 2048 = 44288 B` < 48KB
（不需要 `cudaFuncSetAttribute`，避免"按当前 device 一次性 opt-in 会漏掉其他 7 个 rank"的老问题）。

---

## 4. GPU 验证手册（主 agent 执行）

### 4.1 双产物

```bash
cd kernels/cuda && bash build.sh 103a          # 动了 .cu ⇒ 必须重编 .so
cd ~/ferrite && cargo build --release
```

### 4.2 A/B（隔离本 kernel：**不加** `DSV41_EXPERT_GROUPED`，也就没有任何布局中继）

```bash
# 参考臂（REF）：split body（FUSE=0 + ILV=0），本 kernel OFF
DSV41_SPEC=1 DSV41_DSPARK=1 DSV41_EXPERT_ACT_E4M3=1 DSV41_GATEUP_FUSE=0 \
DSV41_EXPERT_ILV=0 DSV41_MOE_BATCH=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
LD_LIBRARY_PATH=$HOME/ferrite/kernels/cuda \
./target/release/ferrite-serve --model dsv41 --serve --tp 8 --port 8712

# 实验臂（G1-A）：REF + 本 kernel
#   DSV41_EXPERT_GROUPED_GATEUP=1
```

判据（缺一即空洞）：

1. **零 decline notice**：stderr 里**不得**出现
   `DSV41_EXPERT_GROUPED_GATEUP armed but the grouped gate/up kernel declined`（出现即臂无效，
   它的 `why` 会指名原因：pair body / dim%32 / inter%kGtRows / rows*slots 越界 / smem>48KB）。
2. `[tp] ... err:` **0 行**（8/8 rank）；`/health` OK；无 `illegal|fault|CUDA error|panic`。
3. **正证据**：`nsys profile -t cuda` → `nsys stats --report cuda_gpu_kern_sum` 里
   **`expert_gemv_fp4_gate_up_grouped_kernel` 调用数 > 0**，且
   **`expert_gemv_fp4_batched_kernel` 的调用数应下降到 ~0**（gate/up 方向被接管；down 方向仍是
   `expert_gemv_fp4_batched_kernel<false,1>` 或 grouped down，取决于 `GROUPED_DOWN`）。
4. **双门禁**：`step_ms`（[dspark] 分解）**AND** `mean-k`（基线 2.240）。数值健康 + 文本红线
   （计数前 61 行 + 零拉丁，`scripts/tcgen05_smoke.sh` STAGE 2/3）。
5. **口径纪律**：本臂改的是 gate/up 的**执行组织**，不改配置矩阵；`FUSE=0/ILV=0` 必须**三臂同值**，
   否则测的是 FUSE 的代价而不是去重。若要与生产（FUSE=1）对比，那是另一个变量（见 §5.2）。

### 4.3 读数与预期

- 权重侧 DRAM/L2 字节 **−44%**（36 → |active|，|active| 实测 18-24）；gate/up 占总 verify ~8.8%
  ⇒ 若 gate/up 是带宽/访存受限，期望 verify **−1.5 ~ −2.5ms**（票面）。
- **反号风险（必须先看）**：expert 并集让一个 CTA 做 A 条 assignment（A ∈ 1..6），
  **负载不均**（wave 时间 = max over CTA）；且如果 split body 实际是 **FMA-issue 受限**
  （见 `.cu:1996-2001` 的注释："issue slots are ~80 percent stalled on the FMA port"）
  而不是 load 受限（pair body 的 `:1728-1733` 说 "~94% of cycles stalled on the K loads"），
  那么去重省下的是 load 而不是 fma ⇒ 收益会**远小于**票面。**这两个假设的判决只能来自这次 A/B**。

---

## 5. 遗留 / 下一步（按优先级）

### 5.1 已定位的两个反号风险（A/B 若为负，按此归因）

1. **负载不均**：把 `kGtRows` 从 8 降到 4 会加倍 CTA 数（更多并行、更小 tile）—— 一个 A/B 旋钮；
2. **瓶颈归属**：若 split body 是 FMA 受限，收益会被吞掉一半以上。

### 5.2 pair body（FUSE=1，**生产默认形态**）的版本 —— 设计已备，未实施

本 kernel 只服务 split body（FUSE=0）。生产/verify 默认是 **fused pair body**，而审计的
"MoE 成本 ∝ n_assign、711/722 GB/s 有效带宽"**正是 pair body 的口径**（load 受限 ⇒ 去重理论收益最大），
所以 pair body 版才是收益上限所在。它**同构可行**，代价是第二个 kernel：

- `kGtPairRows = 4`（pair body 每行要同时 hold gate+up 两个平面 ⇒ tile 字节翻倍）：
  smem = `4×align16(2·kbytes=5120) + 4×2·ksc + k·4 + 2KB = 20480+1280+20480+2048 = 44288 B` ✓ 同尺寸；
- 权重 tile 布局一处两用：**plain** 时 gate 放 `[0,kbytes)`、up 放 `[kbytes,2kbytes)`（读 `q` / `kbytes+q`）；
  **ILV** 时把该行的交错 16B 粒度原样拷入（读 `2q` 的 `uint4`）—— 由 `template<bool ILV>` 选择偏移；
- 内层循环 = pair body（`.cu:1696-1962`）**逐字复制**，只把 `g_row/q`、`u_row/q` 换成 smem 副本；
  **保留 ksplit**（`g_begin/g_end` 切片 + `s_ks` 跨半合并，`__syncthreads()` 在 assignment 循环内是
  block-uniform 的，因为 `ids` 读是一致的）；**不实现 P4 prefetch**（pf 只改字节来源不改值，pf 路径
  与直读路径同值 ⇒ 不影响逐位）；
- CTA 形状：`blockDim = (kGtPairRows*ksplit)*32`，row 映射 `row_local = warp/ksplit`（与 batched 同式），
  只改"哪个 CTA 覆盖哪些行"，不改 ksplit 的求和顺序 ⇒ 逐位。

### 5.3 down 方向

`expert_gemv_fp4_down_batched_grouped_kernel`（`.cu:2774`）+ `DSV41_EXPERT_GROUPED_DOWN`（**需同时开
`DSV41_EXPERT_GROUPED`**）**已交付**：down 的 6% 已经有去重臂，A/B 时把 `GROUPED_DOWN` 一起开才有完整口径
（注意：那时布局中继会被打开 ⇒ 需单独核对它的开销是否被 down 的收益覆盖）。

---

## 6. MoE 最终口径（回答"36 sweep 是不是 ferrite 架构的下限"）

**不是架构下限。** 分三层：

1. **36 sweep 是"每 (row,slot) 一个 CTA"这个实现的恒等式**，不是算法的下界。下界是
   `|active|`（实测 18-24）次权重流过 —— 即 MoE 理论上还能再省 **33-50% 的 gate/up 权重字节**。
2. **它像结构性下界的唯一原因，是既有两条实现都赔本**：
   - route-gather 布局（3 个纯中继 kernel，其中一个是单 block 串行）+ **128 行 MMA 的过量计算**
     ⇒ 实测 **+16ms**（路线 B，已判死）；
   - 而"只排序不循环"（路线 A 字面版）**不改指令数**（§2 已证）。
3. **本文件交付的第三条路**（路线 A′）把去重放回它该在的位置：**一个 SIMT 循环 + 一次 stage**，
   没有布局、没有张量核、没有新 ABI。它是这一层唯一还活着的实现；但它只覆盖 split body，
   pair body（生产形态）的版本见 §5.2。**MoE 的真正下界（`|active|` sweep）在 ferrite 的架构里
   是可达的，代价是一个循环，不是一次架构改动。**

---

*本文件由工部撰写；结论均为静态 + compile-only 级，唯一未决项（收益符号与量级）由 §4 的 A/B 判定。*
