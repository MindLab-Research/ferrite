# DSV4.1 优化方法论 — 可复用的成功模式与反模式

> **本文件是一份"怎么做"的手册，不是"做了什么"的流水账。**
> 事实来源：`crates/ferrite-dsv41/STATUS.md`（6632 行）、本会话（2026-09-11）40+ 次尝试的提交历史
> （`git log --all`）、以及同期的 `docs/agent/{ferrite-unified-arch,dsv41-layer-fusion,dsv41-kernel-inventory-v3}.md`。
> 每条模式都带**案例引用**（commit / file:line / env gate），便于验证与复用。
>
> **本会话快照**：13.28 → 8.23ms **已验证**（round 41，四段文本逐字对、faults=0，75.3 → 121.5 tok/s，+61.4%）；
> 其后累积改动（runtime smem ceiling + PDL 统一 OFF + serve_ab SIGPIPE fix）已于 **c056c165** 远端复验通过：
> one-shot OK（`1+1=` → `2`，ids `[20,1]`）、serve 四段文本逐字对、faults=0，**p50=6.90ms = 144.9 tok/s**
> （p10/p90 = 6.81/6.99，98 steps，单并发短上下文口径）。
> ⚠️ 早期 3.56ms / 220 tok/s 只是**投影而非实测**：本次实测未达。200 tok/s（5ms/step）目标仍未达成，
> 需继续按 §0 的「回收开销 → 改执行形态」路径推进。尝试总数 40+，其中成功约 20、中性约 10、明确否决约 8。

---

## 0. 一条总纲：这个仓库的性能工作 = 「回收开销」+「不改数学」

单步时间被两样东西吃掉：**真实工作**（GEMV 的 L1TEX/指令地板、expert fp4 的每 warp in-flight 字节、
AR 的 NVLink 协议）与**开销**（~1156 个 kernel 图节点 ≈ 2.0ms、launch 间隙、冗余 global 往返、
grid 失衡导致的空转 SM）。架构级评估（`STATUS.md:6547`）的结论是：
**kernel 级优化已触到"开销回收"上界（8.23 → ~5ms 是回收上界），再往下必须改变执行形态**
（专家核 GEMM 化 / M>1 batching）。

因此本会话所有成功模式都共享一个性质：**它们重排"发射"而非重写"计算"**。
这带来三个可验证的收益：
1. **天然位级一致**（kernel 与操作数逐字不变）⇒ 正确性风险低；
2. **天然可回退**（每项一个 env gate，默认开、`=0` 回退）；
3. **天然跨模型**（GLM 侧复用同一套设备层原语即可，见 §7）。

---

## 1. 成功模式 A：fork/join 侧流（本会话最大方法论收获，6 连胜）

### 模式

把一层里两条**互不读写**的子链拆到不同 CUDA stream，用
`cudaEventRecord` + `cudaStreamWaitEvent` 做 fork/join。三条硬性质（缺一不可）：

1. **不改 grid 形态**——只改发射流，kernel 与操作数逐字不变 ⇒ **位级一致**；
2. **无跨块 / 跨 rank 同步**——不引入任何核内栅栏或自旋（对比 §2.1 的反面教材）；
3. **纯事件驱动 + 图可捕获**——`cudaEventDisableTiming` 事件在捕获区内合法，
   `fork_ev`/`join_ev` 可在图内按程序序复用。

### 6 个胜利（全部 default ON，各有独立 env gate 可回退）

| # | 名称 | 拆出的两条链 | fork 点 | join 点 | gate / 案例 |
|---|---|---|---|---|---|
| 1 | **hc tail split** | EARLY（collapse+rmsnorm+fp8 ≈1.7µs）vs LATE（ss+mixes+sinkhorn+comb ≈10.7µs） | `hc_front_split` 内（C 侧） | 主流 hc_post 前 | `DSV41_HC_TAIL_SPLIT`（`chain_dev.rs:2238`）；round 41 实测仅 −0.20ms（理论 −0.86，缺口见 §2.2 的调度坑） |
| 2 | **EARLY-on-side** | EARLY 移到侧流头部与 dots 并发（dots 4.9µs > EARLY 1.7µs） | side 链头（`in_ev` 之后） | `early_ev` | `e152f47`，−0.14ms，bit-exact |
| 3 | **dots-on-side** | dots 也上侧流（只写 `g_hc_part`，唯一读者是 LATE） | side（EARLY 之后） | 流内顺序即 happens-before ⇒ **删掉 `fork_ev`** | `806ec7a`，关键路径 −0.256ms（确定性值，非区间） |
| 4 | **attention dual-chain** | q 链（norm/lin_rope+wq_b+rope ≈13.5µs，关键路径）vs kv 链（rmsnorm_rope ≈10.6µs，填充） | `lin2` 之后 | kv 链之后、`ring_win_fuse` **之前** | `DSV41_DUAL_CHAIN`（`chain_dev.rs:312`）；`2d7eead` |
| 5 | **MoE dual** | routed experts（≈42µs）vs shared expert（≈22µs） | `moe()` 顶部（`sh_w` 后、gate 前） | routed 之后 join，再 `add_inplace(&s.o,&s.ex_out)` | `DSV41_MOE_DUAL`（`chain_dev.rs:365`）；`2d7eead` |
| 6 | **compress side** | kv-source 层（2/8/14/20）的 4 个 compress launch（≈30µs） | `lin2` 之后（与 dual_chain 同点） | `window_idxs` 之后、**indexer 之前** | `DSV41_COMPRESS_SIDE`（`chain_dev.rs:338`）；`ec439e3`，需**新开 `side_stream3`** |

### 基础设施（共享，已在 `ferrite-kernel`，GLM 可直接用）

`devrt.rs` 的 `create_side_stream`（`:144`，`cudaStreamCreateWithPriority` + `cudaStreamNonBlocking`）、
`side_stream()`（`:765`）/`side_stream2()`（`:790`）/`side_stream3()`（`:840`）、
`record_event`（`:862`）、`stream_wait_event`（`:873`）、事件对 `fork/join`（`:796`/`:803`）、
`in/early`（`:812`/`:819`）、`fork2/join2`（`:826`/`:833`）、`fork3/join3`（`:847`/`:855`）、
`SidePrio` 解析（`parse_side_prio`，`:126`）。

### 必须遵守的三条规则（从 6 个实现里提炼）

- ⚠️ **规则 1：fork 点必须在 mainstream 上游的写之后。** side 链若读主流上游写的缓冲
  （`s.h`/`pre_collapse`），必须补一条 `main→side` 的 `in_ev` 边；否则该 kernel 会变成图 ROOT、
  抢在生产者之前读到陈旧值（整步 capture 时图内无隐式顺序）。
- ⚠️ **规则 2：join 点 = 最早消费者之前，不是直觉位置。** compress 的 join 在 `indexer` 之前
  （连 `sparse_attn` 之前都不够）；attention dual-chain 的 join 在 `ring_append` 之前
  （不是 `sparse_attn`——ring append 立即读 `s.kv`）。**先找到真正的首个消费者，再定 join。**
- ⚠️ **规则 3：时间窗重叠的链必须各占一条 side stream。** kv 链与 compress 都落在
  `lin2`→消费者之间，共用会把 10.6µs 串到 30µs 后面；hc LATE 在 `lin2` 之前就已 fork，
  也不能与 attention dual-chain 共用 `side_stream2`。**当前三类窗口 = 三条流（+主），是几何决定的最小配置。**

### 配套：多流优先级（只对"图回放 + 节点 READY"有效）

机制是 `cudaGraphInstantiateFlagUseNodePriority`（`devrt.rs:1391`，flags 在 `:636` 构造；
当且仅当**任一**侧流优先级非默认才开启，`:633` 的 `DSV41_GRAPH_NODE_PRIORITY` 是总开关）。

- `DSV41_HC_TAIL_PRIO`（默认 greatest）——**疑似过度分配**（39µs 余量下无收益，反抢投影 wave 的 SM）。
- `DSV41_DUAL_PRIO`（默认 default）——两条链都在主流自己的串行路径上，压主流不划算。
- `DSV41_COMPRESS_PRIO`（默认 greatest）——三条侧流里唯一真正 gate 住 attention 的链。
- **唯一证据**：stderr 的 `[hc_tail]/[dual_chain]/[compress_side] side stream priority = N`；
  否则优先级可能没进 capture。

### ⚠️ 已知缺口：fork/join 保证正确性，但不保证并发调度（round 41/42）

tail split 理论上限 −0.86ms，实测只兑现 −0.20ms。根因两条并存（`STATUS.md:6287`）：
1. side stream 用 `cudaStreamCreate` ⇒ 默认优先级，1-block 的 LATE 节点与主流上千投影块同优先级抢 SM；
2. **更关键**：`graph_instantiate` 传 `cudaGraphInstantiate(..., 0)` ⇒ 图回放让所有节点跑在
   launch stream 的优先级上，**per-node 优先级被静默忽略**（只有传 `...FlagUseNodePriority`(=8) 才启用）。

**方法论教训**：侧流的收益 = `min(被隐藏的工作, 重叠窗口) − 调度开销`。
当"缺口 ≈ 图节点开销本身"时（审计：1355 节点 ≈ 2.0ms，~1.5µs/节点；split 每次多发
1 kernel + 2 event 节点 ≈ 0.3-0.5ms/步），应该**减少节点数，而不是继续调优先级**。

---

## 2. 成功模式 B：生产者直出融合（T1 模式）

### 模式

把 producer 的**尾段**（rmsnorm / quant / rope / epilogue）搬进 consumer 的 **prologue 或 epilogue**，
让中间量不写回 global 再读。这是「每层 3 段」融合的**最小可运行单元**。

### 已落地的实例（全部 `DSV41_*` gate，多数 default ON）

| 实例 | 融合点 | 消掉的 launch | 关键约束 / 等价性 |
|---|---|---|---|
| `DSV41_NORM_FUSE` | `rmsnorm_q + fp8 encode` 搬进 `wq_b` 的 gemv **prologue** | 40 次 `rmsnorm_q`（0.13ms） | 强制 32 warps/block（归约树要与 1024 线程逐位对齐）；mode 强制 4；**`qr` 保持 RAW** |
| `DSV41_OROPE_Q` | o-rope 后**追加一趟**整行量化（`dsv41_apply_rope_q`） | 40 次 `quant1` | 必须 barrier 之后的第二趟（rope 循环只碰每 head 尾部 `rope_head_dim` 列；量化块是整行 `nlh*hd`） |
| `DSV41_ROPE_FUSE` | q/idx_q rope 折进 GEMV epilogue | 40(q) + ~7(idx_q) | 强制 32 warps/block、grid=n/32；launcher 校验形状，不过返回 decline |
| `DSV41_WOB_F32` | wo_b 直读 f32 激活（`dsv41_gemm_fp8_mx_f32`） | 每步 40 次 `quant1` | **只改数据路径、不改 grid 形态**；非逐位（跳过量化往返、精度更高）⇒ 必须 parity |
| `DSV41_SWIGLU_Q` | swiglu 直出 `(xq,xsc)` | `quant_kernel` 40 次 | 逐位（`tests_dsv41_glue.cu` 的 swiglu_q 用例逐项核对） |
| `DSV41_GATEUP_FUSE` / `DSV41_DOWN_FUSE` | expert 出口 2·inter→inter + swiglu epilogue；down+reduce 合一 | `swiglu_limit_batched` / `moe_down_reduce` | 逐位（K 循环照抄 / `__fadd_rn`/`__fmul_rn` 显式分开 / slot 升序累加） |
| `DSV41_MOE_EPI_ADD` | `add_inplace` 折进 lane-0 epilogue | 80 节点 | 结合律不变 ⇒ 逐位 |
| `DSV41_COMP_PLACEHOLDER_FUSE` | placeholder 折进 `ring_win_fuse` epilogue（`5ba96bb`） | 30 次 launch/节点 | `clen==nullptr` 时与基础版逐字节相同 |
| engram wkv f32 直读 | 复用 `dsv41_gemm_fp8_mx_f32` | 2 次/步 `quant_fp8` | 读的是 **AR 之后**的 f32 行（AR 前发射 fp8 违反 `fp8(Σ) ≠ Σ fp8`） |

### 两条铁律

- ⚠️ **铁律 1：融合不能改变 grid 形态。** B1 把量化做进 wo_a epilogue，被迫 32 warps/block
  ⇒ grid=n/32、148→32 活跃 SM，实测 **+0.24ms**（见 §3.2）。正确形态是「只改数据路径」——
  block 形状与普通 GEMV 相同，无 SM 利用率损失。
- ⚠️ **铁律 2：跳过量化往返 = 精度更高但非逐位。** 凡是"直读 f32 / 跳过 quantize→dequantize"
  的融合，上机必须做一次 parity（同二进制 A/B 的 text/fingerprint 对比），不能只在无 GPU 环境做 `cargo check`。

---

## 3. 成功模式 C：占用率审计（把「估算」变「实测」）

### 案例：a32 的 20KB smem → blocks/SM 从 8 掉到 4

**对象**：`gemm_fp8_gemv_kernel` 的 M=1 路径带一张 block 级预解码激活表 `s_af`（"a32"，`k×f32` = 20KB @ k=5120），
把 smem 从 ~28KB 抬到 ~47.4KB（mode 4 的 `gsmem` ≈ 48384B）⇒ **blocks/SM 4 → 8**。
独立门 `DSV41_GEMV_A32`（`dsv41_kernels.cu:2580` 的 `g_gemv_a32`，`:2585` 的 `dsv41_gemv_a32_bytes`），
`=0` 时丢弃 a32、把 decode+scale 内联回消费循环（**逐位等价**，smem 少 20KB）。

**命名陷阱**：`DSV41_GEMV_FP8_MODE=3` **不是** a32 开关（mode 3 也物化 `s_af`）。
mode 3 vs 4 隔离的是 **k 字节的激活 staging**，不是 **4k 字节的 a32 表**（`STATUS.md:6408`）。

### 方法论（可复用）

- 用 `cudaOccupancyMaxActiveBlocksPerMultiprocessor` + `ptxas -v` 把
  「smem 预算 → blocks/SM → wave」从**估算**变成**实测**。工程里已有探针：
  `scripts/dsv41_a32_bench.cu`（经 `dsv41_gemv_gsmem`/`dsv41_gemv_occupancy` 两个 host 探针打印
  每 shape 的 smem/blocks-per-SM/µs + 指纹）+ `scripts/dsv41_a32_bench.sh` + `scripts/dsv41_recovery_verify.sh`。
- ⚠️ **smem 余量极紧**：mode 4 只剩 49152 − 48384 = **768 B**。任何给 mode 4 加 smem 的改动
  都会越过 48 KB 门槛而触发 `cudaFuncSetAttribute`，必须一并复核 launcher。
- ⚠️ **占用率假设必须被实测证伪/证实，不能跨核外推。** 两个方向都踩过：
  - `01291b2` 的 nv8 曾以「+14 寄存器 ⇒ 6→4 blocks/SM ⇒ 1.5 wave ⇒ +49%」定案（`STATUS.md:5876`）；
  - 但随后的 4 值隔离微基准（`9fe0766`）是**全场最快**，直接**推翻"40 regs 是单 wave 红线"**——
    "寄存器分配是 per-function 的，正确的问题是哪种宽度对上现有 schedule，而不是怎么压到 40 以下"。
  ⇒ **占用率解释只在被测的那个 kernel/形状内成立。**
- **a32 的收益是在 n=256/1024/1664 的探针上测的（−6/−8/−13%），从未在生产 k=5120 复测**。
  这是一个**二元项**（成功 −0.74ms / 失败 +0.74ms），恢复后必须**单变量先测**（见 §4.5 反模式 E）。

### 同类的"grid 太小 = 延迟病"（可复用的诊断句式）

这条审计思路不限于 a32，凡"grid 极小 + 单线程/长串行扫描"的核都是同一类病：

| 案例 | 现象 | 修复 | 收益 |
|---|---|---|---|
| `engram_apply` | grid=4 block（2.7% SM） | float4 体 + blockDim 128→256 | 16.36 → **4.72µs**（3.5x） |
| `engram_hash_step` | **1 线程**串行走 48 列 | 一列一线程 + grid-stride | 15.64 → **3.37µs**（4.6x） |
| `indexer_topk` score 循环 | 每候选 1 线程 × nh(32) 头 × hd(128) 串行 | warp-per-candidate（lane h = head h，shuffle gather 保 h 升序） | −0.38ms |
| MoE gate（n=384） | 延迟受限、3% 峰值带宽（17.2µs） | dispatch 到 `gemv_bf16_v2`（uint4 + K-split WPR=4） | −0.46ms（预测兑现 92%） |
| AR v5 reduce | `ceil(n/1024)×1024` ⇒ block0 干 80%、block2-4 全空 | `threads=256` + `blocks=ceil(n4/threads)` | −0.12~0.16ms（见 §4.4） |

**诊断句式**：先问「是数据量（带宽），还是并行度（延迟）」——`engram`/`gate` 的 3% 带宽、
`indexer` 的单 CU 串行、AR reduce 的单 SM 堆载，全都是**延迟病**，加宽并行度即可。

---

## 4. 成功模式 D：平凡修复大收益（launch / grid 配置审计）

### 4.4 案例：AR v5 的 grid 失衡（一行修复 −0.12ms）

**问题**（`ferrite-unified-arch.md:161`）：AR v5 三个 launcher
（`ferrite_p2p_ar_v5`:`ferrite_kernels.cu:8427` / `ferrite_p2p_ar_pubred_v5`:`:8495` /
`ferrite_p2p_ar_v5_hcpost`:`:8652`）旧映射是 `ceil(n/1024) × 1024 线程`，
`i4 = blockIdx*1024 + tid, step = 5120`。n=5120 ⇒ n4=1280 ⇒
**block0 干 80%、block1 干 20%、block2-4 全空**，8192 次 load 压在单 SM。

**修复**：`threads = 256`（`world > 256` 时回退 1024——stamp/poll 要求 `blockDim.x >= world`）
+ `blocks = ceil(n4/threads)` ⇒ n=5120 得 **5 个满块 × 256 线程 = 1280 线程**，摊在 5 个 SM。
**逐位等价**（i4 所有权与升序 rank 求和序未变、无跨线程归约）。store kernel 同映射同改。

**方法论**：这是「**grid 失衡 ≠ 协议成本**」的教科书。AR 的 0.66ms 是 NVLink 协议地板
（store 的 160KB 远程写 + stamp 传播）；reduce 那 1.5-2µs 是**本地读 + grid 失衡**，不是 NVLink 项。

### 同类的"launch 配置审计"收益

- **`__launch_bounds__` / `ptxas -v` 先量再改**：`down` 的 nv8 回归（`40 regs → 54 regs → 6→4 blocks/SM`）
  与 gateup CSE 的 "-12%" 都说明**源码更少 ⇒ 实测更慢**是常态，改前先量寄存器数与 blocks/SM。
- **`moe_batch=false` 误关 ⇒ +560 launch/步 = +8.7ms**（`f3b1be1` 连带 gate OFF 的事故）
  ⇒ **批量/融合类 gate 的默认值必须与 kernel 侧一致**，两侧分裂（`.cu return 1` vs Rust `unwrap_or(false)`）
  就是 round-18 乱码与 safe3 退化的真根因。

---

## 5. 成功模式 E：代码审查抓真 bug（GPU 测试测不出来的那些）

### 案例 1：indexer 的 kidx 池缓存 bug（`eb9b13a`）

池缓存 8 个 entry，**小于** `nsm`(12)；线性扫描 top-k 循环因此**有时整个跳过 fused epilogue**，
产出垃圾分数（2.2e-37 而非 ~1e-5）。修复 = 把 epilogue 提出循环（5 行）。
提交信息原话：**"a REAL correctness bug that would have corrupted the verification -
caught by code review, not by the GPU tests."**

为什么 GPU 测试测不出：短上下文由 128 槽窗口主导，压缩路径影响小（`STATUS.md:3008`）；
再加上"暴露条件 = 缓存 entry 数 < nsm"是一个**结构性**条件，不是随机数据能覆盖的。

### 案例 2：把「宿主算出的地址被烤进图」当作一个 bug 类（`STATUS.md:2996`）

**审计方法**（受用户 GLM 经验启发）：逐处检查步内所有 `memcpy_*` 的源/目的地址与
kernel 参数里的指针运算，凡**每步会变**却参与图捕获的，都是冻结点。本会话抓到三处：

| # | 位置 | 性质 | 状态 |
|---|---|---|---|
| 1 | KV 环追加 `memcpy_d2d(ring + (pos%win)*hd, kv)` | 目的地址每步变 | 已修 + 隔离验证（`ring_append`） |
| 2 | index_k 发布 `memcpy_d2d(index_k + (compress_len-1)*idx_hd, …)` | 同上 | 已修（新 `dsv41_index_k_publish`，组号由 `*clen[owner]` 在设备上算）+ 隔离验证 OK |
| 3 | `indexer_topk(..., comp_len, offset)` | `comp_len` 每步值作参数 | 已修（kernel 改读设备 `lens`；仅 `.cu` ⇒ 零 Rust 冲突） |

**⚠️ 反向教训**：第 3 项的 `offset` **曾被误判为冻结点**，读调用点后发现实参是 `win`（静态模型维度）
⇒ **无需修**。**判据要看代码，不看名字**——凡"每步值"必须读调用方实际传了什么才能判定。

**为什么 GPU 五段文本测试会掩盖它**：短上下文被 128 槽窗口掩盖，压缩路径影响小，文本仍逐字对。

### 可复用的"审查优先"清单

1. **每步变化的地址/计数/形状，只要进了捕获区就是 bug**（图会把它冻结）。
2. **两侧默认值必须一致**（`.cu` 的 `return 1` ↔ Rust 的 `unwrap_or(...)`）。
3. **decline 码的语义撞车**（见 §6.3）。
4. **FFI 参数顺序是手写转录**（见 §6.4）。
5. **隔离复现器模板**：`/tmp/{hc,sa,ikp}_repro.cu` = 直接链 `libferrite_kernels.so` + 预热
   + 循环断言 + **空 kernel 地板对照**。⚠️ 模板坑：返回 `int` 的 FFI **不能**用 `cudaError_t` 包装宏。

---

## 6. 反模式（已证伪，勿重试）

### 6.1 跨块同步原语 —— 图捕获下是灾难

| 尝试 | 机制 | 结果 |
|---|---|---|
| **hcpm**（`DSV41_HC_PERSIST_MB`） | 192 个 dot 块每块 publish 后 `__threadfence()` + `atomicAdd` 选举 | **+3.3ms**（`STATUS.md:6076`） |
| **hc-merge（B'）** | 单 kernel + ticket 自旋（`m<mix` 做 dot→ticket；`m==mix` 自旋等 ticket==mix） | **+3.2ms**，"SM 人质"（`STATUS.md:6119`，`ferrite-unified-arch.md:76`） |

**hcpm 的真根因**（`hcpm-regression-analysis`，`STATUS.md:6145`）：
❌ fence/atomic/counter 全部证伪（量化 ≤2µs，与 41µs 差 20 倍；旁证：`hc_front_kernel` 只有 24 块却同样 +3.2ms）；
✅ **主因是 tail 的纯串行 L2 依赖链（ss→mixes→sinkhorn→comb）被塞进 dots 的同一 grid**
——tail 只能等最后一次 publish，必然与整格 drain 串行，无法与任何工作重叠。
✅ 次因三条：丢了 `ss_in=1` 优化、`g_hc_part` 读取 8x 放大、三合一帧 >32 regs → 1 block/SM → 1.3 波。

**规则**：**不要为「少一个 kernel」把跨块/跨 rank 同步塞进核内。**
AR 是两处物理边界（每层 20KB），严格「每层 1 算子」不存在，可行形态是**每层 3 段**。
要重叠跨段依赖，用 **PDL（§2.2 引用）** 或 **侧流（§1）**，不是 `cudaLaunchCooperativeKernel`
（与图捕获不兼容）。

### 6.2 改变 grid 形态 —— 已调优的 grid 不能动

**B1**（wo_a epilogue 直出 fp8）把量化做进 wo_a 的 epilogue，被迫 **32 warps/block**
（32 连续行 = 一个量化块）⇒ grid = `n/32`、**148 → 32 活跃 SM** ⇒ 实测 **+0.24ms**
（`STATUS.md:5977`、`:6506`）。

**规则**：融合只改**数据路径**，block 形状与普通 GEMV 相同（`g_gemv_warps` 默认 4），
无 SM 利用率损失。（这正是 `WOB_F32` 明确标注"为什么不会重蹈 B1"的原因。）

### 6.3 decline 码 1 —— 与 `cudaErrorInvalidValue` 撞车

`cudaErrorInvalidValue == 1`，而多个融合 launcher 用 `return 1` 表示"形状不支持，请回退"。两个方向都翻过车（`2d7eead`、`dsv41-layer-fusion.md §7`）：

1. **误判为回退**：C 侧真实错误恰好是 1 时，Rust 的 `rc == 1` 把它当优雅 decline 静默吞掉；
   launcher 若不清 sticky，错误还会**泄漏到下一个 launcher** 并让 serve 崩（r42 的实际机制：
   `SetAttribute` 失败 → sticky 存活 → 下一个 `quant_fp8` 的 kerr 报错 → serve crash）。
2. **误判为硬错**：Rust 写 `rc == 2` 而 C 侧仍 `return 1` ⇒ 真 decline 被当错误抛出 ⇒ 融合路径永远不生效。

**约定（新符号起）**：**decline 哨兵一律用 `2`，永不用 `1`**（2 在实际路径上远不可能被
`cudaError_t` 真实返回，1 是最常见的 InvalidValue）。C 侧 `return 2;` ↔ Rust 侧 `if rc == 2 { Ok(false) }`。
全量审计表见 `dsv41-layer-fusion.md §7.1`。

**配套纪律（r44，`§7.2`）**：任何**非 `cudaGetLastError()` 来源**的 CUDA 调用
（`cudaEventRecord`、`cudaStreamWaitEvent`、`cudaFuncSetAttribute`、`cudaMalloc` …）
在返回错误码前都必须先 `(void)cudaGetLastError()` 清 sticky，否则错误会泄漏给下一个 launcher
造成**归因错位**（`dsv41_hc_front_split` 的 3 处事件失败曾让错误错误归到 `quant_fp8` 上）。

### 6.4 FFI 参数顺序手写 —— 静默失败多轮

`DSV41_ROPE_FUSE` 曾因把 `CuStream` 放在**第 9 个参数位**（照抄带 C++ 默认尾参的 `dsv41_gemm_fp8_mx`），
而 C 侧 `dsv41_gemm_fp8_mx_rope` 没有可选尾参、stream 是**最后一个**参数 ⇒ C 读到的形参整体错位一格：
`rope_rd <- 真 rope_inverse(0)` ⇒ `rope_rd <= 0` 成立 → 返回 1（decline）→ Rust `Ok(false)` →
**调用点静默回退**。

**静默回退的识别特征**（记住这个组合）：
「gate 默认 ON + `supports_*()` 符号探测为真 + kernel 侧实现正确，但 nsys 里旧 kernel 仍是 88/步，
且那一轮的 A/B 恰好"中性"（既没省 launch 也没加 epilogue 开销）」。

**规则**：`Option<unsafe extern "C" fn(...)>` 的参数顺序是本仓库主流约定「**stream 放最后**」
（只有带 C++ 默认尾参的旧符号例外）；新符号的 FFI 类型必须**逐参对照 C 原型**，编译器不会校验。

### 6.5 假设不实测 —— 本仓库最贵的一类错误

| 案例 | 假设 | 实测结果 |
|---|---|---|
| **a32** | "20KB smem 把占用率压到 25%，关掉它 → 8 blocks/SM → gemv 5-6µs" | 收益只在 n=256/1024/1664 探针上测过；**生产 k=5120 从未复测**（二元项 −0.74ms） |
| **"40 regs 红线"** | "40 regs 是单 wave 红线，超过就退化" | 被 4 值微基准**推翻**：56 regs / 4 blocks/SM 的版本反而**最快**（`9fe0766`） |
| **hc_mixes = 步时间 49%** | nsys 单核占比 | 两次自我更正（`STATUS.md:2223`/`:2254`）：口径错误（含 prefill/首步），**只有同二进制背靠背 A/B 成立** |
| **多卡 nsys 绝对值** | 单次耗时可直接比较 | **不可信**，只有排序可参考（`STATUS.md:2402`） |
| **发射数是瓶颈** | 段图跑通就能提速 | 段图已跑通（正确）但**不带来提速**（`STATUS.md:1756`） |

**规则**：
- **同二进制背靠背 A/B + 四段文本逐字 + faults=0** 是唯一判据（跨版本比较必须双产物重编）。
- 微基准必须**生产形状** + **隔离复现** + **空 kernel 地板对照**。
- 「变快」永远不能用「少算/降精度」换（禁止 `git revert` 当"优化"）。
- **微基准本身也会 flaky**（`STATUS.md:1717`）：先确认它稳定，再拿结论。
- **朴素求和会重复计账**：PDL/fp4-pack/sparse-o-rope/hc_post 同属"launch 消除"，**不可线性叠加**；
  四条侧流共享同一侧流预算，**也不可线性叠加**（`STATUS.md:6474`）。

---

## 7. 跨模型复用（GLM 侧）

**一句话结论**：DSV41 会话最大的**结构性**收获不是某个 kernel，而是
**「层内并行不一定要动 kernel，只要动发射流」**——它把「融合」从「重写计算」降级为「重排发射」，
因而天然位级一致、天然可回退、天然跨模型。**GLM 的缺口不在能力，而在设备层未收敛。**

| 基建 | DSV41 | GLM | 复用路径 |
|---|---|---|---|
| fork/join 侧流原语 | ✓ `devrt` 全套 + 三流六链 default ON | ✗ **完全没有**（`cuda.rs` 里无 `side_stream`） | **最高价值迁移项**：GLM 层链在 `ferrite-exec/tp.rs`，发射走 link-time `cuda.rs`，而 `devrt` 是 dlopen 的另一套设备层（Phase 4 前是两个世界）。GLM 的 hc 链（MHC）与 MoE shared expert 是最明显候选窗口 |
| 多流优先级 + 图节点优先级 | ✓ 三 gate + `UseNodePriority` | ✗ | 机制在 `devrt.rs`；GLM 图实例化走 `cuda.rs`，未接该 flag。随 Phase 4 做 |
| PDL 串链 | ✓ default ON，11 个 launch 点 | ⚠️ `pdl_or_plain` default OFF、覆盖窄 | 可扩到 MoE（quant→gateup→down）、hc、投影族；⚠️ 每个新 TU 要带自己的 `pdl_or_plain` 副本（file-static 不能跨 TU） |
| 生产者直出融合 | ✓ 十余处 | ⚠️ 部分（`gdn_step_v2`、`hc_pre big-fuse`、`gv2_route_epilogue`） | 抽象 = 「把 producer 尾段搬进 consumer 首/尾，且不改 grid 形态」 |
| AR v5 | ✓ 共享实现 | ✓ 蓝本 | **已统一**；grid 修复（§4.4）两模型共享同一组 launcher |
| decline 码 = 2 | ✓ 全量审计 | n/a | 写成**跨模型规范**；非 `cudaGetLastError()` 调用一律清 sticky |
| a32 / 占用率审计工具 | ✓ `dsv41_a32_bench.cu` + 2 host 探针 + 脚本 | ⚠️ 手动 | 探针/脚本可泛化为模型无关工具 |
| 段级 persistent（3 段核） | 🔬 设计完成，P1d 回归，默认 OFF | ✗ | `LayerDesc` 描述化的共同产物 |

---

## 8. 判据与纪律（每次改动照做）

1. **一批改动一个 gate，默认开、`=0` 可回退**（§1-§4 的每一项都遵守）。
2. **同二进制背靠背 A/B**，判据 = 四段文本逐字 + `faults=0` + p50。
3. **禁止** `git revert` 当优化；禁止"变快 = 少算"；禁止跨版本拿 A 树 `.so` 配 B 树二进制。
4. **远端命令前必须先同步**（否则测的是旧码）；`.so` 时间戳/`build_id` 必须新于源码
   （陈旧 `.so` 曾让整个调试会话作废）。
5. **微基准门禁**：改动前先跑生产形状的单核微基准，**打不过现形态就不进集成**
   （本仓库"in-serve 试错每次烧 ~1h"）。
6. **跳过量化往返的融合必须 parity**（非逐位）。
7. **优先级真的进了 capture 的唯一证据**是 stderr 的 `side stream priority = N` 行。
8. **判据看代码不看名字**：凡"每步值"必须读调用方实际传了什么（`indexer_topk` 的 `offset` 教训）。

---

## 9. 工具与资产索引（可直接复用）

| 资产 | 用途 |
|---|---|
| `scripts/dsv41_a32_bench.{cu,sh}` | a32/占用率隔离微基准（smem + blocks/SM + µs + 指纹） |
| `scripts/dsv41_recovery_verify.sh` | 驱动恢复后的哨兵 → base → a32 A/B 一键序列；phase 0 强制同源构建顺序：`rm -f` 陈旧 `.so`+`.build_id` → `build.sh` → `cargo build`，并校验 binary 内嵌 build_id（`.so`/`.build_id` 均 gitignored/untracked，`git clean` 不会清） |
| `scripts/dsv41_serve_ab.sh <tag> [ENV]` | 同二进制背靠背 A/B（逐步直打 p50 + faults）；启动前预检 `.so`/binary/`.build_id` 同源，不匹配即 fail-fast |
| `scripts/dsv41_gemv_bench.cu` / `dsv41_indexer_bench.cu` | GEMV shape 扫描 / indexer n_pos 扫描（带选择指纹） |
| `scripts/dsv41_profile.sh` | nsys 采集 + 差分口径（1-token 差分得 decode-only 分解） |
| `/tmp/{hc,sa,ikp}_repro.cu`（模板） | 隔离复现器：直接链 `.so` + 预热 + 循环断言 + 空 kernel 地板 |
| `devrt.rs` 侧流/事件/图原语 | §1 的共享基础设施 |
| `POST_RECOVERY_COMMANDS.md` | 驱动 wedge 后的恢复步骤 |

---

_本文件与 `ferrite-unified-arch.md §4`（同一批模式的架构版描述）、`dsv41-layer-fusion.md §6/§7`
（融合清单 + decline 审计）、`dsv41-kernel-inventory-v3.md`（口径/占用率/AR 交叉验证）互为引用；
本文件侧重**方法论**（怎么做、为什么有效、怎么不踩坑），不重复它们的 kernel 明细。_
