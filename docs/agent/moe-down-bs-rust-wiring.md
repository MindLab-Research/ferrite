# MoE down blockscaled 臂 —— Rust 侧接线设计（**已备好，未落盘**）

> 状态：**设计已备**；Rust 代码**一行未改**（等 GPU 对拍 §4 通过后再接）。
> 上位：`docs/agent/moe-down-bs-design.md`（kernel 设计/对拍）、`docs/agent/tilelang-moe-bs-wiring.md`（gate/up 同族臂的接线先例）。
> 门控：**`DSV41_MOE_DOWN_BS`（默认 OFF）** —— 与 `moe_bs_dn_shim.cu:251` 读的**同一个 env**。
> 前提：`kernels/cuda/tests_dn_bs_parity.cu` 编译 0 error 且 §4.2 的 GPU 对拍通过。

---

## §0 一句话

down 臂的接线**比 gate/up 臂简单一档**：它**不需要装载期任何新池**（直读既有 `w2` packed fp4 池 + 既有
`w2_scale` pad 面），**不需要 e4m3 预量化激活**（shim 自己把 f32 激活量化成 e4m3），
**不用 TMA**。所以 Rust 侧只加：**1 个门 + 1 个能力探针 + 1 个 FFI 包装 + 2 处 branch**，
**`load.rs` 零改动**。

---

## §1 门控与插入层

### 1.1 门

| 项 | 内容 |
|---|---|
| 门名 | `DSV41_MOE_DOWN_BS`，`=1` 才 arm，默认 OFF（与 shim `moe_bs_dn_shim.cu:251` 同一变量） |
| Rust accessor | `pub fn moe_down_bs() -> bool`，加在 `crates/ferrite-models/src/dsv41/weights.rs`，紧邻既有 `moe_tilelang_bs()`（`weights.rs:511`） |
| 读法 | `OnceLock` 每进程读一次（与 `sf_stride_pad()` `weights.rs:488` / `moe_tilelang_bs()` `weights.rs:511` 完全同构） |

> 为什么仍放 `weights.rs`：gate/up 臂是因为 `load.rs` 也要读它才放这里；down 臂没有装载期依赖，
> 但**所有 MoE arm 门集中在一处**能避免"两次定义漂移"，且将来若要在装载期校验 w2_scale 布局也顺手。

⚠️ shim 是**每次调用**读 env（`moe_bs_dn_shim.cu:251`），Rust 门是每进程读一次。两者不一致时
只可能更保守（Rust ON + env off ⇒ shim `return 2` ⇒ 回落老路径），不会更激进 —— 但要在文档里写明
**Rust 门才是分派开关**。

### 1.2 能力探针（旧 `.so` 必须让门保持 OFF 并出声）

照 `supports_down_fuse()`（`device.rs:2918`）/ `supports_moe_bs_act_e4m3()`（`device.rs:8424`）的模式加：

| 文件 | 改动 |
|---|---|
| `crates/ferrite-models/src/dsv41/device.rs` 结构体 | 1 个字段 `moe_bs_down_cap: Option<unsafe extern "C" fn() -> c_int>`（照 `expert_act_e4m3_cap` `device.rs:1388`） |
| 同文件 `ksym` 注册表 | `moe_bs_down_cap: ko!(rt, "dsv41_moe_bs_down_cap")`（照 `device.rs:2377`） |
| 同文件探针 | `pub fn supports_moe_bs_down(&self) -> bool { self.kernels.moe_bs_down_cap.is_some() }`（照 `device.rs:2918`） |
| 同文件 FFI 包装 | `pub fn moe_bs_down_dev(...) -> Result<bool>`，**`rc == 2 → Ok(false)`**，其余非零 `self.kerr(...)`（照 `moe_tilelang_down_bf16_dev` `device.rs:8171`，含 `self.stream` 收尾） |

用**能力符号**（`moe_bs_dn_shim.cu:329` 的 `dsv41_moe_bs_down_cap`）而不是"探 `_dev` 符号是否存在"，
是因为本臂将来会有 ABI 不变但语义变的修订（与 `dsv41_moe_bs_act_e4m3_cap` 同样的理由，
`moe_bs_shim.cu` §8 的先例）。

### 1.3 插入点（file:line）

| # | 站点 | 位置 | 说明 |
|---|---|---|---|
| (a) | `moe_rows`（verify，m>1） | `crates/ferrite-models/src/dsv41/chain_dev.rs:18863` **之前** | 该点已是"`tl_dn_done` ? 复用 reduce : 走 fused / 非 fused"的三岔口 |
| (b) | eager（rows=1） | `crates/ferrite-models/src/dsv41/chain_dev.rs:23824` **之前** | 该点已是"`tl_dn_done` ? 复用 reduce : 走 fused / 非 fused"的三岔口 |

**不要**接 `chain_dev.rs:23931` 的 sequential 非 batch 路径（`DSV41_MOE_BATCH=0`）：它的 down 是
"直写 `s.o`"的形态，没有 per-slot partial，接进来要另写归约 —— 不在本臂定义域内，ready gate 直接拒。

**结构建议**（两处同构，照 `tl_dn_done` 的写法 `chain_dev.rs:18791-18834`）：

```rust
// 在 fused 分支判定之前
let bs_dn_done = /* 见 §2 的 ready gate */ self.moe_down_bs_ready(ld, m, topk, dim, inter_local, n_routed)
    && self.moe_down_bs_launch(...)?;      // Ok(false) = shim/表 declined
// fused 分支的条件补 `&& !bs_dn_done`
if bs_dn_done || tl_dn_done {
    // 复用既有 per-row ascending-slot 归约（含 ar_carry 分支），逐字不动：
    //   moe_rows : chain_dev.rs:18809-18825 的那段
    //   eager    : chain_dev.rs:23890-23929 的那段
}
```

关键：**down 臂写到"per-assignment partial 缓冲"**（`ex_down_r` / `ex_down_b`），
即非 fused `expert_down_fp4_batched` 写的那块，于是既有的 `moe_down_reduce_seq_or_plain`
（`chain_dev.rs:1329`）与 `moe_down_reduce_ar`（eager，`:23905`）**一行都不用改**。
数值契约（fp 加法不结合 ⇒ 必须 ascending-slot 定序）因此天然保持。

---

## §2 形参从哪来（18 参逐个对齐）

shim 签名（`moe_bs_dn_shim.cu:244-248`）与 Rust 侧实参来源：

| # | shim 形参 | Rust 实参（moe_rows / eager） | 来源与生命周期 |
|---|---|---|---|
| 1 | `act_base` | `self.s.ex_act_r.ptr` / `self.s.ex_act_b.ptr` | 上游 gate/up 本步刚写；`[rows][slots][act_slot]` f32，只读前 `inter` 个 float。**与 fused SIMT down 同一指针**（`chain_dev.rs:18872` / `:23865`） |
| 2 | `act_stride` | `act_slot as i64` | 与 fused SIMT down 第 2 实参**同一个变量**（`:18873` / `:23866`）：unfused=RAW `2*inter`，fused swiglu=`inter` |
| 3 | `out` | `self.s.ex_down_r.ptr` / `self.s.ex_down_b.ptr` | `[rows*topk][dim]` per-assignment partial，**OVERWRITE**；与 `expert_down_fp4_batched` 写点逐格相同（`:18898` / `:23867`）。⚠️ **不是** `moe_out_r`（那是 fused 的终值缓冲） |
| 4 | `w2_base` | `w2_base` | `ld.experts[0].w2`（`:18340` / `:23169`），**与 fused SIMT down 同源同变量** |
| 5 | `w2_stride` | `w2_stride` | 实测 `w2[1]-w2[0]` 的 per-expert 字节距（`:18342` / `:23172`），同样**直接复用 fused 的实参** |
| 6 | `w2s_base` | `w2s_base` | `ld.experts[0].w2_scale`（= e8m0 pad 面），`sf_pitch_plane` 已把行距 pad 到 16 |
| 7 | `w2s_stride` | `w2s_stride` | 实测 `w2_scale[1]-w2_scale[0]` |
| 8 | `eid_dev` | `self.s.bs_eid.ptr as *const i32` | `moe_bs_tables_dev`（`chain_dev.rs:17744`）产出的 DEVICE 表 |
| 9 | `order_dev` | `self.s.bs_order.ptr` | 同上，**BM=128** 行距（`TILELANG_BS_BM`，`chain_dev.rs:826`） |
| 10 | `counts_dev` | `self.s.bs_counts.ptr` | 同上 |
| 11 | `nseg_dev` | `self.s.bs_nseg.ptr` | 同上（DEVICE 指针，shim 自己做 D2D 读，无 H2D/D2H ⇒ capture 安全） |
| 12 | `route_w` | `rw_eff` | `:18862` / `:23819`。`DSV41_ROUTED_DOWN_QUANT` arm 时是 `nullptr`（rw 已在激活里，禁止二次乘） |
| 13 | `rw_stride` | `1i64` | 平坦 `[rows*slots]`；shim 在 `route_w!=null && rw_stride!=1` 时 decline（`moe_bs_dn_shim.cu:286`） |
| 14 | `rows` | `m as i32` / `1` | 与 fused 同值 |
| 15 | `dim` | `dim as i32` | = 5120（生产），shim 要求 `%128==0 && <=8192` |
| 16 | `inter` | `inter_local as i32` | = 320，shim 硬要求 `inter == DN_K == 320`（`moe_bs_dn_shim.cu:266`） |
| 17 | `topk` | `topk as i32` | 1..=6；`rows*topk <= DN_SEGCAP(36)` |
| 18 | `stream` | `self.stream`（包装器内加） | 与 fused 同 stream ⇒ 与 capture / 主流同序 |

**段表（8-11）的生命周期 —— 这是唯一需要新逻辑的地方**：

* 表由 `moe_bs_tables_dev(m, topk, n_routed)`（`chain_dev.rs:17744`）产出，走
  `dsv41_route_group` + `dsv41_moe_align_from_group` **DEVICE 链**（无 D2H/H2D ⇒ capture 安全，`:17727-17743`）。
* 它消费的是**本步**的 `route_idx_r`（`chain_dev.rs:17687`），与 gate/up 臂**同一份 routing**。
* **gate/up BS 臂已经在同一步建过这张表**（`chain_dev.rs:18541` 的 `moe_bs_tables_dev`，moe_rows；
  `:23428`，eager）⇒ 若上层 gate/up 也走了 BS 臂，down 侧**可直接复用 `bs_*`，不必重建**
  （建议用一个"本步已建"标志）。
* 若 gate/up 走的是别的臂（SIMT / bf16 TileLang），down 侧**必须自己调一次**
  `moe_bs_tables_dev`。该函数幂等（`route_group` 幂等，`:17739` 明写）且 stream-ordered：
  即便与 gate/up 臂各建一次，第二次的 `route_group`/`moe_align_from_group` 一定排在
  上游 MMA 核**读完之后**（同 stream 串行），不会 clobber 上游正在读的表。代价是 2 个小核，可忽略。
* `n_routed`：两处都是既有的 `n_routed`（moe_rows）/ `ne`（eager），与 gate/up 臂同一个实参。

**不需要的东西（对比 gate/up 臂）**

| gate/up BS 臂需要 | down 臂 | 原因 |
|---|---|---|
| 装载期 packed SF 池 `wsf1/wsf3`（`load.rs:1066` 的 pack 块） | **不需要** | down 直读 `w2_scale` pad 面（`inter=320 ⇒ 10 列 → pitch 16`） |
| `DSV41_EXPERT_ACT_E4M3` 预量化激活 `xq4/xsc4` | **不需要** | shim 的 `dn_qgather_kernel`（`moe_bs_dn_shim.cu:312`）自己把 f32 激活量化成 e4m3 + 组 SF |
| 实测 block `w_stride`（TMA `gstride[1]`） | **不需要** | 无 TMA（`moe_bs_dn_shim.cu:61-63`），w2 行距 = `inter/2` 固定 |
| `DSV41_MOE_BF16_DEQUANT` | **不需要** | 直吃原生 fp4 |

⇒ **`load.rs` 零改动、开关 OFF 时零新增显存。**

---

## §3 OFF 等价性论证（门关必须逐字节等价）

三层，逐层可证：

1. **Rust 门关 ⇒ 那段代码根本不执行**。新分支的守卫是 `moe_down_bs() && supports_moe_bs_down() && ...`，
   为假时**不发生 FFI 调用、不建段表、不发射任何 kernel**。传给既有 fused / 非 fused 分支的所有
   实参（`rw_eff`、`ex_down_*`、`w2_base`、`w2_stride`、`act_slot`、`ids_base` …）与今天**同一批指针、同一个值**。
   ⇒ 生成的 launch 序列与现有二进制逐指令相同 ⇒ 逐字节等价。

2. **即使进入了分支也写不进任何东西**。`shim` 的每一个 `return 2` 都在任何发射/`cudaMalloc` **之前**
   （`moe_bs_dn_shim.cu:252 / 260 / 264 / 268 / 272 / 276 / 280 / 284`），Rust 包装器把 `rc==2` 映成
   `Ok(false)`（照 `device.rs:8196`）⇒ 回落老路径，读者看到的还是老路径写的字节。
   **可执行证据**：harness 的 arm D（`tests_dn_bs_parity.cu:196-205`）就是这条断言的机器可检查版本 ——
   `rc=2` 且 `1234.5` 哨兵 `memcmp` 未动。

3. **`.so` 层面**：OFF 不要求任何新符号存在（分支不进）。旧 `.so` 里没有 `dsv41_moe_bs_down_cap`
   ⇒ `ko!` 返回 `None` ⇒ `supports_moe_bs_down()==false` ⇒ 分支跳过，并按项目惯例打一条
   "ARMED but skipped" 一次性告警（照 `moe_bs_skipped_note` `chain_dev.rs:18118`），
   **绝不静默回落**（本项目 #1 测量偏置陷阱）。

4. **数值契约不变**：down 臂写的是与 SIMT 非 fused 路径**逐格相同**的 `[row*topk+slot][dim]`
   partial 布局 ⇒ ascending-slot 归约核与调用序列不变。⇒ **门开时唯一的数值差就是 e4m3 输入量化**
   （算法差，不是实现差；设计文档 §5 / harness arm A vs arm B 已把它与 fp 序差分开定标）。

---

## §4 GPU 验收清单（交给主 agent 执行；**必须先 §4.2 通过再动 Rust**）

### 4.1 双产物编译（远端）

```bash
cd ~/ferrite && git fetch -q origin && git reset -q --hard origin/main
cd kernels/cuda && bash build.sh 103a 2>&1 | tee /tmp/dnbs_build.log; echo BUILD_RC=${PIPESTATUS[0]}   # 必须 0
cd ~/ferrite && source ~/.cargo/env && cargo build --release
nm -D kernels/cuda/libferrite_kernels.so | grep -E 'dsv41_moe_bs_down_(dev|cap)'   # 两个符号都在
```
**判据**：`BUILD_RC=0` 且两符号存在（缺符号 ⇒ 后面所有 e2e 作废）。

### 4.2 kernel 级对拍（**本步是"接线"的前置门**）

```bash
cd ~/ferrite/kernels/cuda
nvcc -gencode arch=compute_103a,code=sm_103a -O2 -std=c++17 \
     -I tilelang_gen -I . -I tilelang_inc \
     -o /tmp/t_dnbs tests_dn_bs_parity.cu            # 本地 CPU 已验证 0 error（见 §7 备注）
CUDA_VISIBLE_DEVICES=<free> /tmp/t_dnbs --bench | tee /tmp/dnbs_parity.log
DSV41_MOE_DOWN_BS=1 DSV41_MOE_DOWN_BS_RWOP=1 CUDA_VISIBLE_DEVICES=<free> /tmp/t_dnbs --quick
```
**判据（按序，任一不过就停，不要接线）**：
1. `[arm D] gate OFF -> rc=2, out untouched=1`
2. `arm B ... max_rel < 1e-4`（期望 ~1e-5）
3. `arm A` 的 `max/rms_ref` ∈ [1e-2, 5e-2]（CPU 预测 2.47%）；远大于 5e-2 ⇒ 是 kernel 错，先查 arm B
4. `--bench` 的 `speedup` 只作**上界**（L2-hot），不作验收依据

### 4.3 OFF 等价性 e2e（接完线后，门关）

```bash
~/arm_run.sh DNBS_off DSV41_MOE_DOWN_BS=0     # 与接线前的出货配置背靠背、同会话
~/.xbin/verify_correct.sh <port> DNBS_off     # 若脚本有
```
**判据**：
* 生成文本 / token 序列与**接线前**逐位一致（最强者：比 token id 或末位 logits）；
* 日志里**不得**出现 `[moe-bs] ARMED but ...`；
* step p50 与接线前**同分布**（新增的只是一次布尔判断）。
> 接线本身在 OFF 下应当"零观测差异"。任何 p50 漂移都说明分支守卫写错了位置（例如提前算了段表）。

### 4.4 门开 vs 门关（一次一个变量）

```bash
~/arm_run.sh DNBS_on DSV41_MOE_DOWN_BS=1
~/.xbin/verify_correct.sh <port> DNBS_on
```
**判据**：
* 性能门：`[dsv41] step pos=` 的 **p50**（同会话背靠背）；nsys 里
  `expert_gemv_fp4_down_reduce_kernel` 应消失、`moe_bs_dn_kernel` 出现；
* 精度门（**不要找逐字节一致**）：文本红线（不重复、不乱码、1..61 数字、拉丁探针）+
  `DSV41_DIFF_EAGER=1` 的 anchor 与 EAGER 一致 + accept 不下降；
* 归因：任何异常用 `DSV41_MOE_DOWN_BS=0` 单独关掉本臂复现。

**⚠️ 第一轮 e2e 请把 `DSV41_MOE_TILELANG_BS` 保持 OFF**：down 臂与 up 臂**正交**
（它只要求上游把 f32 的 `ex_act_*` 按 `act_slot` 写好，SIMT up 也满足），
所以可以单变量地量出 down 臂自己的收益。两个门同时开会把两个变量混在一起（本战役已踩过）。

---

## §5 风险与不确定点（按风险排序）

1. **M/N/K 角色与 gate/up 相反**：down 是 `M=128`（段内 assignment 行）、`N=dim=5120`（40 个 N-tile）、
   `K=inter=320`（**3 个 K-span，末 span 只 2 个 K-block**）。gate/up 是 `K=5120`（40 迭代）。
   ⇒ 唯一没在硬件上单独验证过的几何是**末 span 的描述符/SF 递进**（设计文档 §8.2 #1）。
   这正是 §4.2 arm B 是硬门的原因。
2. **`out` 必须 16 B 对齐**（shim `moe_bs_dn_shim.cu:282`）。`ex_down_r/ex_down_b` 是 `cudaMalloc` 出来的
   `DevBuf`（≥256 B 对齐）⇒ 满足；但**绝不能**传行偏移后的指针，也**绝不能**误传 `moe_out_r`
   （那是 fused 的终值缓冲，语义不同，会静默改掉 AR carrier 的写者）。
3. **`w2s` 行距假设**：shim 在 `moe_bs_dn_shim.cu:294-296` **无条件**按 `(inter>>5 + 15) & ~15 = 16`
   算 pitch，即**假定 `DSV41_SF_STRIDE_PAD` 为 ON**。而 SIMT 核是 `dsv41_sf_pitch()`
   （`dsv41_experts_mxf4.cu:153-168`）**运行期读 env**，两种布局都对。
   ⇒ `DSV41_SF_STRIDE_PAD=0` 时本臂会**每行多读 6 字节 ⇒ 静默错误答案**。
   **缓解（必须在 ready gate 里做）**：`&& crate::dsv41::weights::sf_stride_pad()`。
   （更彻底：给 shim 加一个 `w2s_pitch` 形参，但那会改 ABI，留待对拍后决定。）
4. **`DSV41_ROUTED_DOWN_QUANT` × `DSV41_MOE_DOWN_BS_RWOP` 互斥**：两者都把 rw 乘进**操作数**。
   Rust 在 `q_on` 时传 `route_w = nullptr`；若此时 shim 的 RWOP 又为 1，它会拿 null 的 route_w 去乘
   ⇒ 未定义。**缓解**：ready gate 拒 `routed_down_quant() && moe_down_bs_rwop()`
   （需要给 RWOP 也加一个 `weights.rs` accessor，与 `moe_down_bs()` 并列）；
   并打一条显式告警，指向"两者互斥"这条契约（`moe_bs_dn_shim.cu:39-41`）。
5. **INIT 与 CUDA graph capture**：`dn_bs_init()` 是**首次 arm 调用时惰性执行**的
   （`moe_bs_dn_shim.cu:298`），里面有 `cudaMalloc` / `cudaFuncSetAttribute` ——
   在 capture 内调用会让 `cudaStreamEndCapture` 失败（shim 自己的硬约束 #2，`:64-65`）。
   gate/up BS 臂靠"首个 eager 步先 warm"侥幸过关，down 臂必须**显式**处理：
   ready gate 里加 `&& (!self.dev.capturing() || INIT 已成功)`（或镜像 up 臂的既有做法）。
   ⚠️ 上线前务必确认 capture 的首步序列会先跑一次非 capture 的 arm 调用。
6. **段表复用/重建的次序**：见 §2 末。当前 stream-ordered 使"重建"安全；但若将来 up 臂改到别的
   stream，就变成真 race ⇒ 建议直接用"本步已建则复用"的标志，别依赖 stream 序。
7. **`act_slot` 传错**：fused swiglu 时 pitch=`inter`，非 fused 时=`2*inter`。shim 每个 slot 只读前
   `inter` 个 float，两种都合法 —— 但前提是**传的是上游臂真实写的那个 pitch**。照抄 fused 调用的
   实参即可（`:18873` / `:23866`），不要重新推导。
8. **收益下限是"打平"**：本臂不减权重流量（仍 per-assignment 重复读 29.5 MB/层，见设计文档 §8.1）。
   ⇒ p50 不比预期更差要有心理准备；真正的流量杠杆是 `DSV41_EXPERT_GROUPED_DOWN`（正交）。

---

## §6 与 gate/up 臂的逐项对照（改动清单）

| 项 | gate/up BS 臂 | down 臂 | 备注 |
|---|---|---|---|
| `weights.rs` 门 | `moe_tilelang_bs()` | **+`moe_down_bs()`**（+`moe_down_bs_rwop()`，见 §5.4） | 同构 |
| `device.rs` 字段/注册/探针/包装 | 已有（`ko!` `:2394`） | **+4 处**（照 `:1388/:2377/:2918/:8171`） | 同构 |
| `load.rs` | gate/up 需建 packed SF 池 | **零改动** | down 直读既有池 |
| 激活前置 | 需 `DSV41_EXPERT_ACT_E4M3` + `xq4/xsc4` | **不需要**（shim 内部量化） | — |
| 段表 | `moe_bs_tables_dev` | **同一个**（可复用） | BM=128 |
| `chain_dev.rs` 插入 | 2 处（`:18539` / `:23428`） | **2 处**（`:18863` 前 / `:23824` 前） | 结构同构 |
| 归约 | 复用既有 `moe_down_reduce*` | **同一个**，不改 | 数值契约不变 |

**改动规模估计**：`weights.rs` ~20 行、`device.rs` ~70 行、`chain_dev.rs` ~90 行（2 处 × ready+launch+接归约）。
**`load.rs` 0 行，kernel 侧 0 行。**

---

## §7 附：harness 编译修复记录（2026-09-13，CPU）

`tests_dn_bs_parity.cu` 原报 4 error（2 个调用点 × "too few arguments" + "cudaStream_t 传给 int"），
**根因**：`dsv41_expert_down_reduce_fp4_batched` 在 `DSV41_SEQ_ALIGN`（#5）合并后于
`kernels/cuda/dsv41_experts_mxf4.cu:3966` 新增了 `int seq_align` 形参（在 `cudaStream_t` 之前）。
harness 写于该 ABI 变更之前，**共 4 个调用点**（`:280`、`:308`、`:345`、`:357`）都少这一个实参；
先前只补了前两处（`:280` / `:308`），**`--bench` 块里的后两处被漏掉** ⇒ 它们才是 347/359 行报错的来源
（实参错位到最后一个位置，于是 `st` 落进了 `int seq_align`，`stream` 缺失）。

**修法（最小）**：在 `:346` 与 `:358` 各补 `0,`（含义：该臂 OFF，与参考的历史形态一致），
语义不动任何 kernel。

**证据（本机纯 CPU，nvcc 13.3，无 GPU）**：

```bash
cd kernels/cuda/tilelang_gen
nvcc -c ../tests_dn_bs_parity.cu -o /tmp/dnp.o -gencode arch=compute_103a,code=sm_103a \
     -O2 -std=c++17 -I. -I.. -I../tilelang_inc -I<CUDA_INC>
# EXIT=0；/tmp/dnp.o = 891112 B（4 error → 0 error）
```
> ⚠️ 本机 `nvcc` 不在 PATH（在 `/tmp/nvccx/nvidia/cu13/bin/nvcc`），且该 wheel 的 `include/` 只有
> `fatbinary_section.h` ⇒ 需要 `-I<homedir>/.local/lib/python3.10/site-packages/nvidia/cu13/include`
> 才能找到 `cuda_runtime.h`。这是**机器环境**问题，与 harness 无关；
> 远端（真 CUDA toolkit）按 §4.2 的命令即可。
> ⚠️ 必须在 `tilelang_gen/` 下编：harness 的 `#include "../dsv41_experts_mxf4.cu"`
> （`tests_dn_bs_parity.cu:50`）按**当前工作目录**解析，从别处会找不到该文件。
