# DSV4 并入共享 engine 并删除 `ferrite-dsv41`（执行规格）

**用户令**：「改成用同一个 engine，改好后彻底删掉 dsv4 crate」。
目标终态：**一个 engine、两个模型**（GLM-5.3-Flash / DeepSeek-V4.1-Flash）。模型侧只保留
**算子 + 层链 + 权重布局**；其余全部是共享代码。

## 一、现状与问题（本会话已核实）

| 层 | GLM | DSV4 | 处置 |
|---|---|---|---|
| 算子 | `kernels/cuda/ferrite_kernels.cu` | `kernels/cuda/dsv41_*.cu` | **保留两套** ✓（模型 ops 不同，用户认可）|
| HTTP/serve | `ferrite-http` | 已并入 `ferrite-http` ✓（本会话完成）| ✓ 已收敛 |
| Device/FFI/流/分配/图捕获 | `ferrite-kernel/src/cuda.rs`（已有 `cuStreamBeginCapture` 等全套）| `ferrite-dsv41/src/device.rs` **又写了一套** ✗ | 换共享 ✗ |
| 集合通信 AR | `ferrite_p2p_ar_v5`（ferrite_kernels.cu）| `dsv41_ar_v5_*`（dsv41_glue.cu）**协议同源、实现两遍** ✗ | **已换共享 ✓**（Phase 2：DSV41 调共享 entry，删自家三 kernel；`ferrite_kernels.cu` **零改动** ✓）|
| engine 契约 | `ferrite-exec`（tp.rs 的 mega-graph 链、StepEngine）| `dsv41-run.rs` 里的 `TpRankPool` 自己实现 StepEngine ✓ | 保留形状、挪进共享 |
| 模型定义 | 散布在 `ferrite-exec` | `dsv41/src/{config,load,chain_dev,weights,engram,quant}.rs` | 移入 `ferrite-models/src/dsv41/` |

## 二、终态目录（建议）

```
crates/ferrite-kernel/     Device/FFI/流/分配/CUDA graph（共享，两模型共用）
crates/ferrite-exec/       engine 契约 + TP 集群 + 图链（共享）
crates/ferrite-models/     ← 新增：模型定义
  src/glm53/               （GLM 的 config/load/层链；从 ferrite-exec 抽或保留原位）
  src/dsv41/               config.rs / load.rs / chain.rs / engram.rs / quant.rs（把 dsv41 crate 里
                           的模型逻辑搬过来，device/tp/graph 部分丢弃）
crates/ferrite-http/       共享 serve 栈（已就绪 ✓）
crates/ferrite-serve/      唯一二进制：--model {glm53,dsv41}
kernels/cuda/              两套 .cu（各自 host wrapper 同目录）
```
**删除**：`crates/ferrite-dsv41`（整个 crate）✓。

## 三、执行顺序（每步都能编译 + 能跑，随时可停）

1. **映射/清单**（半日）：逐个模块标注 [共享化] / [模型保留]，并列出共享侧缺什么（缺则以通用方式补进
   `ferrite-kernel`，**不许在模型侧 fork** ✗）。
2. **Device 换共享**（1 日）：把 `dsv41/src/device.rs` 的调用改为 `ferrite-kernel` 的等价接口。
   先做薄适配层（同名转发）以保证可编译，再逐步删转发直连。**验收**：四段文本 + 长文正确 ✓。
3. **AR 换共享**（1 日）：DSV4 用共享的 AR 实现（协议相同 ⇒ 只需参数化 staging 布局/slot 大小/
   world ✓）。**验收**：四段 + 长文 + 与旧实现同二进制 A/B（应逐位一致 ✓）。
4. **图捕获换共享**（半日）：删 `dsv41/device.rs` 里我本会话加的 5 个符号绑定 ✗，改用
   `ferrite-kernel` 的捕获机制 ✓。**注意**：`ring_append`/`compress_commit`/`window_idxs` 这些
   **设备侧位置派生**是 DSV4 的模型逻辑 ✓（保留在 dsv41 侧 ✓），但**捕获/回放原语**必须共享 ✓。
5. **模型定义搬家**（1-2 日）：`config/load/chain_dev/engram/quant` → `ferrite-models/src/dsv41/`；
   把 `TpRankPool` 的 engine 适配（含 **look-ahead 批命令** ✓）挪进共享的 engine 侧 ✓。
6. **单一二进制**（半日）：`ferrite-serve --model dsv41`（config 选择模型 ✓；chat frame/stop set
   按模型注入 ✓，`ferrite-http` 已有 `Seg::Id`/`StopSpec`/`ChatFrame` 通用机制 ✓）。
7. **删除** `crates/ferrite-dsv41` ✓ + workspace 清理 ✓ + 文档更新 ✓。
8. **终验收**：GLM 与 DSV4 两条路径各跑一次（四段 + 长文 + `/v1/stats` 计时 ✓），性能不低于迁移前 ✓。

## 四、硬约束（沿用本会话纪律）

- **正确性红线**：每步之后复验四段文本（亲自读 ✓）；任何改动先编译、再 `build.sh`（改 `.cu` 必重编 `.so` ✓）。
- **禁止 fork 共享栈**：缺能力 ⇒ 以**通用**方式补进 `ferrite-kernel`/`ferrite-http`/`ferrite-exec` ✓。
- **禁止 git revert**：退化就改成默认关闭的 env 开关或就地修正 ✓。
- **性能论证只用同二进制背靠背 A/B** ✓；不用多卡 nsys 的单次耗时 ✗。
- **kill 用 `pgrep -x`**（`-f` 会匹配 ssh 自身 ⇒ 自杀 exit 255 ✗）。

## 五、本会话已为此铺好的路（可直接用 ✓）

- HTTP/serve/engine 契约：`ferrite-http`（`ServeEngine`/`SingleFlight`/`Seg::Id`/`StopSpec`/`ChatFrame`）✓
- DSV4 侧：三处「设备侧位置派生」kernel（`window_idxs` / `compress_commit` / `ring_append`）✓；
  **图重放的坑**（宿主算地址被烤进图 ✗）已定位并修复 ✓ —— 迁移时必须原样保留这个修法 ✓。
- AR v5 协议（GLM 版为蓝本）已实现并验证 ✓；DSV4 的 staging 布局参数（world/slot 字节）已知 ✓。

## Phase 0/1 执行结果（映射 + Device 换共享，本次）

### 交付 1：模块映射表（13 个模块 → 共享化 / 保留 + 共享侧缺什么）

| 模块（`dsv41/src`） | 行数 | 归属 | 依据 / 缺什么 |
|---|---|---|---|
| `device.rs` | 2037 → **1416** | **共享化 ✓（本阶段完成）** | 通用设备操作已转发到共享 `ferrite-kernel::devrt`；只留 `Kernels` 表 + 52 个 kernel 启动封装 |
| `tp.rs` | 506 | **共享化 ✓（Phase 2 完成）** | `Collective::all_reduce_inplace` 调共享 `ferrite_p2p_ar_v5`；DSV41 staging 本已是共享布局 ⇒ **零参数化**；kernel 只留一份 ✓ |
| `chain_dev.rs` | 1819 | **拆分** | 设备编排设备态（位置计数器/图分支/premix D2D → 共享）+ **层链逻辑** ✓（→ 模型）|
| `kernels.rs` | 434 | **模型保留** | DSV4 kernel ABI 声明（随模型走）|
| `chain.rs` / `ops.rs` | 684 / 1060 | **模型保留** | 模型层链 / CPU golden 基准 |
| `config.rs` / `weights.rs` / `load.rs` | 596 / 841 / 883 | **模型保留** | DSV4 checkpoint 布局、TP 分片、engram 表 |
| `quant.rs` / `engram.rs` / `dspark.rs` / `vision.rs` | 528 / 451 / 335 / 1402 | **模型保留** | 量化辅助 / n-gram / draft / 视觉塔 |
| `bin/dsv41-run.rs` | 975 | **拆除（Phase 5/6）** | `TpRankPool` engine 适配（含 look-ahead 批命令）→ 共享 engine，再收敛单一二进制 |

**共享侧缺的能力（→ 已补 / 待补）**：

| 缺什么 | 状态 |
|---|---|
| 字节级、**不池化**的设备分配 + 原始 H2D/D2H/D2D/2D/peer 拷贝 + memset | **已补** → `devrt::DevRuntime` ✓ |
| 显式 CUDA-graph 捕获/实例化/回放的**原语**（返回裸句柄、可指定捕获模式）| **已补** → `devrt` ✓（默认 **Relaxed(2)**，见 Phase 3 差异）|
| 通用 kernel 符号解析（在已加载 `.so` 上 `dlsym` 任意名字）| **已补** → `kernel_sym` / `kernel_sym_opt` ✓ |
| 通用 cuBLAS f32/bf16 GEMM（裸指针，不涉及 Tensor）| **已补** → `gemm_f32` / `gemm_bf16` ✓ |
| AR v5 的**多形态参数化**（staging 表形态、slot/stride、epoch 位置、`out` 直写）| **已解 ✓**（Phase 2：四个 gap **全在 DSV41 侧**收敛 ⇒ 共享 entry **零参数化**、`ferrite_kernels.cu` **零改动**）|
| engine 契约（`StepEngine`/批命令/look-ahead）的统一入口 | **待补（Phase 5）** |
| `cuda::CudaBackend`（GLM）与 `devrt` 两个设备层**收敛成一个** | **待办（Phase 4）** |

### 交付 2：Device 的签名级对照（dsv41 → 共享）

| dsv41 `Device::` | 共享等价物（`ferrite_kernel::devrt`） | 处置 |
|---|---|---|
| `open(so)` | `DevRuntime::open(so)` + `kernel_sym*` 建表 | 转发（建表留模型侧 ✓）|
| `stream` / `device_id` / `device_count` / `bind_to` | `DevRuntime::{stream,device_id,device_count,bind_to}` | 转发 |
| `alloc` / `free` / `mem_free` / `view` | `DevRuntime::{alloc,free,mem_free}` + `DevBuf::view` | 转发 |
| `upload` / `upload_f32` / `upload_f32_at` / `upload_bytes_at` / `upload_from` / `upload_from_2d` | 同名 `DevRuntime::*` | 转发 |
| `download_f32` / `download_u8` / `download_u32` | 同名 | 转发 |
| `zero` / `zero_at` / `memcpy_d2d` / `memcpy_peer` / `sync` / `dev_sync` / `enable_peer_access` | 同名 | 转发 |
| `gemm_f32` / `gemm_bf16` | `gemm_f32` / `gemm_bf16` | 转发（原 cuBLAS 代码移入 ✓）|
| `capture_begin` / `capture_end` / `graph_instantiate` / `graph_launch` / `graph_free` | 同名（**捕获原语已在共享侧 ✓**）| 转发 |
| `kerr` | `DevRuntime::kerr` | 转发 |
| `add_inplace` / `add_inplace_raw` | — | **模型保留**（绑定 `ferrite_add` 符号）|
| 其余 52 个 kernel 启动封装 | — | **模型保留**（逐字未改 ✓）|

### 交付 3：验证结果与下一步

- 全绿：`cargo check -p ferrite-kernel`、`-p ferrite-kernel --features cuda`（GLM 路径不受影响）、
  `-p ferrite-dsv41`、`cargo build -p ferrite-dsv41`（**链接通过** ⇒ dlopen 符号可解析）、`cargo check --workspace`。
- `-p ferrite-dsv41 --all-targets` 有一个**既有**错误：`chain.rs` 测试模块用了 `KvMode` 未 import
  （aed0785 起即存在，非本次引入 ✗）。
- **未做 e2e**（用户硬约束：只允许 `cargo check`/`cargo build`）⇒ 行为等价由「保留方法逐字对比 + 签名一致」
  保证，待 GPU 侧四段文本复验。
- 校验脚本确认：保留的 52 个 kernel 封装 + `Kernels` 结构体**逐字一致**；唯一 body 变化是 `free`
  的日志前缀（`[dsv41]` → `[ferrite]`）。

**Phase 2（AR）的坑（本次新发现）**：
- DSV4 是 **三次发射**（store/publish/reduce ✗），GLM 是**单入口融合**（`ferrite_p2p_ar_v5` ✓）⇒ DSV4 的
  `Collective` 必须改成**一次调用**，否则 epoch 推进语义不同（DSV4 在 reduce 末块推进 ✓）。
- DSV4 的 epoch 放在 **staging 尾部 `ctr_at`** ✓，GLM 是**独立参数** ✓ ⇒ 参数化时必须保留其一并换算偏移。
- `publish` 自旋 ⇒ 剖析时用 NCCL 模式（去掉 P2P/AR v5 的 env ✓）。

**Phase 3（图）的坑（本次新发现）**：
- `devrt::capture_begin` 现用 **Relaxed(2)**（= DSV4 现有行为 ✓）；GLM 的 `ferrite_graph_begin` 用
  **ThreadLocal(1)** ⇒ 两套捕获入口**模式不同**，Phase 4 收敛时**必须二选一**（Relaxed 是超集、对 GLM 无害
  ⇒ 建议统一到 Relaxed ✓）。
- `devrt`（dlopen、指针级）与 `cuda.rs`（link-time、Tensor 级）是**两个设备层** ⇒ Phase 4 应把 `CudaBackend`
  的裸指针部分下沉到 `devrt`，避免长期双份。

## 附：清告警的流程教训（本会话自伤 5 次，务必避免）

**现象**：按 `cargo check` 的告警**批量套用**建议（`replace(..., 1)` / 简单正则 ✗），连续 5 次改错位置：
`rank`（删了正在用的那处 ✗）、`ctr2`（重命名了在用的 ✓/✗）、`best`/`mi`（去掉了真正需要的 `mut` ✗）、
`tensor_specs` 的 `cfg` 参数（函数体在用 ✗）—— 每次都靠编译错误才发现 ✗。

**根因**：告警给的 `file:line` 与我用的**模式**匹配到的**不是同一处** ✗（同一 crate 里同名变量/参数很多 ✓）。

**正确流程**（已按此修完 ✓）：
1. 逐条处理：**按诊断的精确 `file:line` 读该行原文** ✓，确认它就是要改的那处 ✓；
2. 用**唯一上下文锚点**（含前后各 1-2 行 ✓），禁用 `replace(..., 1)` 盲改 ✗；
3. **每改 1-2 处立刻 `cargo check`** ✓（不要攒一批 ✗）；
4. 加 `#[allow(dead_code)]` 时**必须写一句为什么保留** ✓（否则下一个人不敢删 ✓）。

**顺带确认的边界**：`cargo check -p X` 的输出**含 X 的依赖 crate 的告警** ✗ ⇒ 统计某 crate 自身告警时必须
按 `grep "<crate>/src"` 过滤 ✓（本会话一开始把依赖的 63 条误算到 ferrite-serve 头上 ✓）。

## Phase 2 的精确输入（已读双方代码，可直接执行 ✓）

**共享侧（GLM）已有的 AR —— 单入口、融合式**：
```c
// crates/ferrite-kernel/src/cuda.rs:262 绑定的 extern "C"
ferrite_p2p_ar_v5(partial, staging_tbl /*f32**/, ready_tbl /*u32**/, epoch /*u32*/,
                  staging_local, ready_local, out, n, world, my_rank, stride, s)
```
表以 **`T*[]`**（“我这份 + 每个对端那份”的本地视图数组 ✓）传入；一次调用内含 store → publish → reduce ✓。

**DSV4 侧现状（三个 kernel + Rust 三次发射 ✗）**：
`ar_v5_store_kernel` / `ar_v5_publish_kernel` / `ar_v5_reduce_kernel`
（`kernels/cuda/dsv41_glue.cu:257/272/292` ✓），表以 **`u64[]`**（对端地址数组 ✓）传入，
epoch 放在 staging 尾部（`ctr_at` ✓），reduce 直写调用方缓冲 ✓，最后一块推进 epoch ✓。

**合并方案**：
1. **保留一份实现**：`ferrite_p2p_ar_v5`（共享侧 ✓）。若 DSV4 需要额外参数，只做**通用**扩展
   （例如 `out != partial` 的直写形态 ✓ —— DSV4 的 reduce 直写调用方缓冲，需要 `out` 语义 ✓，GLM 已有该参数 ✓）。
2. **DSV4 侧**：`Collective::{all_reduce_inplace, end_round}` 改成薄调用（传自己的 staging 表 + epoch ✓），
   删除 `dsv41_ar_v5_*` launcher 与那三个 kernel ✓。
3. **布局映射**（两者的协议同源，只是表的形式不同 ✓）：
   `peer_slots[u64[]]` ↔ `staging_tbl[f32*[]]`；`peer_stamps[u64[]]` ↔ `ready_tbl[u32*[]]`；
   DSV4 的 `epoch@ctr_at` ↔ GLM 的 `epoch` 参数；奇偶半区 `epoch & 1` ✓ 一致。
   DSV4 每层的 slot 字节（`bytes` ✓）↔ `stride`；AR 长度 ↔ `n` ✓。
4. **验收**：`cargo test -p ferrite-dsv41 --test ar_micro`（world=4/8 ✓ 与主机参考逐轮比对 ✓）
   + 四段文本 ✓（**求和顺序必须一致**：两者都是按 rank 升序 ✓ ⇒ 应逐位一致 ✓）。

⚠ 执行时注意（本会话实测的坑）：
- **AR 的 publish 会自旋** ⇒ 剖析时必须用 NCCL 模式（去掉 P2P/AR v5 的 env ✓）。
- **图捕获期**：epoch 由 **reduce 的最后一块**推进 ✓ ⇒ 只被 replay 推进 ✓（不能有 dry-run 参与 ✓）。
- **测试旋钮必须 static 缓存** ✗（每次 AR 调用都 `getenv` 是热路径罪 ✓）。

## Phase 2 执行结果（AR 换共享，已完成 ✓）

**方向**：DSV41 采用共享布局、调共享 `ferrite_p2p_ar_v5`（PREFERRED），**`ferrite_kernels.cu` 零改动** ✓
（GLM 路径因此天然逐位不变 ✓ —— 没有新增参数，也没有动 GLM 的调用点/launcher 签名）。

**为什么零参数化就能通**：DSV41 的 staging 本就已经是共享布局 ✓ ——
`[2][world][bytes]` 奇偶半区在偏移 0、`[world]` ready 行在 `stamps_at`、设备 epoch 在 `ctr_at`；
共享 entry 收的就是 `staging_local`（本 rank 半区基址）+ `ready_local`（本 rank 行）
+ 两个指针表（对端 staging 基址 / 对端 ready 行基址）+ 一个**设备** epoch。⇒ 四个 gap 全在 **DSV41 侧**闭合：

| 签名的 gap | 处置方向 | 怎么合的 |
|---|---|---|
| staging 表形态（GLM `float*[]` vs DSV41 `u64[]`）| **DSV41 侧** ✓ | 64 位下二者都是「8 字节设备地址」，DSV41 直接把 `peer_slots`/`peer_stamps` 以 `*const *mut f32` / `*const *mut u32` 传入（`tp.rs:290-291`）|
| slot·stride | **DSV41 侧** ✓ | DSV41 的 `bytes/4`（每槽 f32 数）就是共享的 `stride`；奇偶偏移公式两者一致（`((e&1)*world + rank)*stride`）|
| epoch 位置 | **DSV41 侧** ✓ | DSV41 把 `staging + ctr_at` 当共享的 `epoch` 入参（`tp.rs:287`）；仍在**设备内存**、内核运行时读 ⇒ 图可回放 ✓ |
| `out` 直写 | **已存在** ✓ | GLM 的 v5 本就有 `out` 参数；DSV41 传 `out = partial = buf`（内核边界保证 store 已读完 partial）|

**改动**（file-by-file）：
- `crates/ferrite-dsv41/src/tp.rs`：`all_reduce_inplace` 的 ar_v5 分支由「三次发射（store/publish/reduce）」改为**一次** `self.dev.p2p_ar_v5(...)`；`use std::ffi::{c_int, c_uint}`。
- `crates/ferrite-dsv41/src/device.rs`：`Kernels` 增 `p2p_ar_v5`（`ko!(rt, "ferrite_p2p_ar_v5")`，GLM 复用区）；新增转发方法 `Device::p2p_ar_v5`；**删除** `ar_v5_store/ar_v5_publish/ar_v5_reduce` 三个字段、解析与 wrapper。
- `kernels/cuda/dsv41_glue.cu`：**删除** `ar_v5_store_kernel`/`ar_v5_publish_kernel`/`ar_v5_reduce_kernel` 与 `dsv41_ar_v5_{store,publish,reduce}` 三个 extern "C" launcher。
- `crates/ferrite-dsv41/src/kernels.rs`：**无改动**（AR launcher 的 ABI 原本就内联在 `device.rs` 的 fn 指针类型里，不在 kernels.rs）。

**保留的语义**（未变 ✓）：`DSV41_AR_V5=0` → 仍旧走 `all_reduce_inplace_inner` 的 host-barrier（`end_round` 的 `SpinBarrier`）；
epoch 仍在设备内存（图可回放）；归约仍按 **rank 升序**（逐位一致）。

**⚠ 关键差异（换共享后引入，需 GPU 复验）**：共享 `p2p_ar_store_v5_kernel` 用 **float4** 写 staging
（`reinterpret_cast<float4*>`）⇒ 要求 `stride % 4 == 0` **且** `partial`/`out` 16 字节对齐；DSV41 旧 kernel 是
标量、无此约束。DSV41 的 `bytes` 均为 `hc_dim*4` / `vocab*4` / `n*4`（16 的倍数），理论满足，但**必须** e2e 复验。
另：共享 pubred 由 **block 0** 推进 epoch（DSV41 旧实现由 reduce 最后一块推进）——同为「每次调用恰好 +1、设备侧」✓。

**验证状态**：`cargo check --workspace` **0 error** ✓；`ar_micro` **编译通过** ✓（`cargo test --no-run`）。
⚠ **本机无 GPU、无 nvcc** ⇒ `ar_micro` **未实跑**、`.cu` **未重编**：需在远端 `bash kernels/cuda/build.sh 103a`
重编 `.so`（改了 `dsv41_glue.cu` ⇒ BUILD_ID/CU_HASH 变化，Rust 侧会拒绝加载旧 `.so`），再跑
`DSV41_KERNELS=... cargo test --release -p ferrite-dsv41 --test ar_micro -- --nocapture`。

## Phase 3 的精确输入（图原语合并，已读双方 ✓）

**共享侧已有**（`kernels/cuda/ferrite_kernels.cu:7022+`，绑定在 `ferrite-kernel/src/cuda.rs` ✓）：
```c
ferrite_graph_begin(s)        -> cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal)
ferrite_graph_end(s, &g)      -> cudaStreamEndCapture
ferrite_graph_instantiate(&e, g) -> cudaGraphInstantiate(e, g, 0)
ferrite_graph_launch(e, s)    -> cudaGraphLaunch
ferrite_graph_destroy_exec(e) -> cudaGraphExecDestroy
```

**⚠ 关键差异（本会话实测）**：共享侧用 **`ThreadLocal`** ✓；DSV4 是**单进程 8 个 rank 线程** ✓
（每个 rank 一条流、但**共享 CUDA 上下文的 legacy stream** ✗）⇒ 用 `Global` 会被**其它 rank 的同步
API 调用**作废（实测 `cudaErrorStreamCaptureUnjoined` 901 / instantiate 900 ✗）⇒ DSV4 必须用
**`Relaxed`** ✓（最宽松：只拒绝捕获线程自己的非法调用 ✓）。
**建议**：把**模式作为参数**（或共享默认改 **`Relaxed`** ✓ —— 它是 `ThreadLocal` 的超集，对 GLM 无害 ✓）。

**必须随共享 API 一起"继承"的 4 个捕获纪律**（本会话逐个踩过并修好 ✓，写进共享实现的注释 ✓）：
1. **捕获区内禁止任何同步流 API** ✗：`cudaMemset`/`cudaMemcpy`（会跑在 legacy stream ✗ —— 报
   "operation would make the legacy stream depend on a capturing blocking stream" ✓）⇒ 一律用
   `cudaMemsetAsync`/`cudaMemcpyAsync` + 自己的流 ✓。
2. **捕获区内禁止分配** ✗：`cudaMalloc`（err 900 ✓）⇒ 预分配；若确需在捕获期分配，用
   `cudaMallocAsync` + 自己的流 ✓（并从 stream 池取 ✓）。
3. **先热身再捕获** ✓：第一步跑真实路径（预热 kernel/建懒态/给 cuBLAS 定 workspace ✓），
   第二步才捕获 ✓；**捕获只记录不执行** ⇒ 捕完**立刻 launch 一次**以完成本步 ✓。
4. **冻结点审计**（最隐蔽 ✗）：任何**以宿主计算值作为 kernel 参数或 `cudaMemcpy` 源/目的地址**
   的调用都会被烤死 ✗ —— 本会话的真凶就是 `memcpy_d2d(ring + (pos%win)*hd, ...)` ✗
   ⇒ 必须改成**设备侧派生**（新 kernel 内用 `*pos_ctr` 算 ✓：`window_idxs`/`compress_commit`/
   `ring_append` 三个 DSV4 kernel 就是为此而生 ✓，**保留在模型侧** ✓）。
   判据：症状与生成长度相关、拐点与某个宿主计数周期吻合（本例 ratio=4 ✓）⇒ 直指该路径 ✓。

## Phase 4-6 的具体清单（模块归属已按代码头部注释逐个判定 ✓）

**`crates/ferrite-dsv41/src/` 13 个模块的归属**（行数为证 ✓）：
| 模块 | 行数 | 归属 | 依据 |
|---|---|---|---|
| `chain.rs` | 684 | **模型** ✓ | 头部注释："The model chain. Layer order is the reference's, which is *not* the usual one" ✓ |
| `ops.rs` | 1060 | **模型** ✓ | "CPU reference implementations … the numerical golden standard" ✓（保留为对拍基准 ✓）|
| `chain_dev.rs` | 1819 | **拆分** ✗ | 设备编排设备态（位置计数器/图分支/premix D2D ✗ → 共享）+ **层链逻辑** ✓（attention/compress/moe/indexer/hc ✓ → 模型 ✓）|
| `load.rs` / `weights.rs` | 883 / 841 | **模型** ✓ | DSV4 checkpoint 布局 + fp4/fp8 分片 + 引擎的 engram 表 ✓ |
| `config.rs` | 596 | **模型** ✓ | HF `config.json`（嵌套 `text_config`）与官方键名映射 ✓ |
| `quant.rs` / `engram.rs` | 528 / 451 | **模型** ✓ | 量化辅助 / n-gram 哈希（含本会话的设备化 kernel 对接 ✓）|
| `vision.rs` | 1402 | **模型** ✓ | DSV4 的视觉塔 ✓（迁移时一并带走 ✓）|
| `dspark.rs` | 335 | **模型** ✓ | draft/投机相关（**注意：用户的 MTP 禁令与此模块的启用条件需在迁移时确认 ✓**）|
| `kernels.rs` | 434 | **模型** ✓ | "Kernel ABI … `dsv41_kernels.cu` implements exactly these extern C" ✓（随模型走 ✓）|
| `device.rs` | 1416 | **共享** ✗ | 换成 `ferrite-kernel` 等价接口（Phase 1 ✓）|
| `tp.rs` | 506 | **共享** ✗ | `Collective`/`SpinBarrier` 换成共享实现（Phase 2 ✓）|

**Phase 5（单一二进制）**：`ferrite-serve --model {glm53,dsv41}` ✓ —— 只按**数据**分叉：
`StopSpec`（停词集 ✓）/`ChatFrame`（chat 模板 ✓）/`EngineDriver` 实现 ✓ —— 机制全部在共享栈 ✓
（`ferrite-http` 的 `ServeEngine`/`SingleFlight`/`Seg::Id`/`StopSpec`/`ChatFrame` ✓ 本会话已通用化 ✓）。

**Phase 6（删除核对表）**：
1. `crates/ferrite-dsv41/` 整目录 ✗ → workspace `members` 同步 ✓；
2. **ABI 边界核对** ✓：`dsv41_*.cu` 的 `extern "C"` 符号与 `kernels.rs` 的声明一一对应 ✓
   （迁移后 ksel 仍需能 dlopen 到同一 `.so` ✓ —— `build.sh` 已把 dsv41 的四个 TU 一起编译 ✓）；
3. **环境变量核对** ✓：`DSV41_*`（MODEL_DIR/KERNELS/GRAPH_STEP/TIMING/AR_V5/ENG_HOST/… ✓）
   在共享侧仍要生效 ✓ —— 其中 `DSV41_AR_V5`/`DSV41_GRAPH_STEP` 的语义会随 Phase 2/3 变化 ✓
   （开图即需要设备侧 AR ✓）；
4. **文档**：把 `crates/ferrite-dsv41/STATUS.md` 的知识**合并进**共享文档 ✓
   （本会话的全部根因与纪律 ✓），然后随 crate 一起删除 ✓；
5. **终验收**：GLM 与 DSV4 各一次（四段+长文亲自读 ✓ + `/v1/stats` ✓ + 稳态 step time ✓）。

### 捕获纪律（第 5 条，Phase 0/1 实测新增 ✓）

**图里烤的是设备地址 ✗ —— 池的"地址稳定"是承重契约 ✓。**

Phase 0/1（`device.rs` → 共享 `devrt` 的字节级不池化分配）落地后复验，出现**回归** ✗：
请求 1-3 正确 ✓，此后 `1 steps`/空输出 ✗ + 2 个 fault ✗；**同一二进制关图**
（`DSV41_GRAPH_STEP=0`）**全部正确** ✓（Paris / 《静夜思》整首 / 长文通顺 ✓，0 fault ✓）。

定位：我此前"图可跨请求复用"的前提是"**所有 launch 参数都已设备化**" ✓ —— 但**漏了
图里同时烤死了它录制的缓冲区地址** ✗。旧池的契约恰是"同 size class 返回同一地址"✓，
所以跨请求复用成立 ✓；新分配器是字节级 ✓ 不作此承诺 ⇒ 复用上一请求的图 = 对着**可能已属于
别人的地址**重放 ✗。

处置：`reset()` **丢弃 graph exec** ✓（`graph_free(null, exec)` ✓ —— 捕获的 graph 句柄在
instantiate 后已释放 ✓，故第二参数选 `cudaGraphExecDestroy` ✓），下一请求重新捕获（几 ms ✓）。
**更长远的正解**（迁移时一并做 ✓）：把 step 期的缓冲区改为**构造期一次性分配** ✓
（地址在整个进程内稳定 ✓）⇒ 图可安全保留 ✓。
