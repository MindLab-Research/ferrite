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
| 集合通信 AR | `ferrite_p2p_ar_v5`（ferrite_kernels.cu）| `dsv41_ar_v5_*`（dsv41_glue.cu）**协议同源、实现两遍** ✗ | 换共享 + 参数化 staging ✗ |
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
