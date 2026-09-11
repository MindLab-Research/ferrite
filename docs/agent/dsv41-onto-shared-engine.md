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
