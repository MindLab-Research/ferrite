# T 臂乱码根因审判（两份审计合成判决，2026-09-13 深夜）

## 判决摘要

verify-only TileLang（`DSV41_GEMM_TILELANG=1`）e2e 从 line 2 起乱码的根因，经两份独立审计（tl-garbage-diagnosis + tl-parity-vs-old）交叉确认：

### 已排除（代码级证据，勿重查）

| 嫌疑 | 判决 | 证据 |
|---|---|---|
| reduce 无 m-predicate | **证伪（主 agent 误读）** | `wkv_reduce_tl.cu:31` 的 `if ((i*8+(threadIdx.x>>5)) < m)` 包住 `:75` 的 `store_global_256`；`i` 只循环 0..1（不是 0..15），行号 = `i*8+warp` |
| 部署产物陈旧 | **证伪** | D5 sha256：deployed = frozen（`13630c20...` reduce / `63ae91de...` partial），逐位一致 |
| WSC 布局不匹配 | **证伪** | 双侧都是 32×32 块 `[N/32, K/32]`（生成 kernel `T.Tensor((N//32,K//32))` = ferrite `vec![hd/32, dim/32]` = [16,160]） |
| 确定性写越界 | **证伪** | 全部写要么带 m-predicate、要么地址由构造保证界内（`out` 行 stride 与分配 pitch 逐形状核对一致） |
| A 激活读越界 | **证伪** | partial kernel `:57` 的 `if (tid>>3 < m)` 包住 A_sh staging，`:59-62` else 分支行≥m 写 0 |

### 根因（按致乱码嫌疑排序）

**#1 [最高·确定性·已修] xsc/xsc_r 的 MPAD floor 算术 no-op（tl-parity-vs-old 缺陷 #1）**

```rust
// 错误（c43c078 版本）：
xsc: dev.alloc(fb((dim.max(nh*hd).max(cfg.o_lora_rank).max(inter))  // = 32768
    .max(16 * dim / 32)                                           // = 2560 → no-op!
    / 32 + 8))?                                                   // = 1032 f32
```

- `.max(2560)` 放在 `/32` **之前**，被外层 32768 吞掉 → `/32` 后只给 1032 f32
- 生成 kernel 读 ASC 到 index 2559（16 行 × 160 pitch）→ **越界 1528 f32 = 6112B**
- **修复（98f50e8）**：`.max()` 移到 `/32` 之后（元素域）：`max(X/32, 16*dim/32) + 8 = 2568`
- **同修 xsc_r**：`max(VERIFY_ROWS*max_dim/32, 16*max_dim/32) + 8 = 16392`（wo_a G8 需 16×1024=16384）

**#2 [高·结构性] phase-2 四形状零 GPU 实测同 gate 上线（tl-parity-vs-old 缺陷 #3）**

- wq_a/wq_b/wo_b/wo_a 四形状的 shim/生成物全部通过 `DSV41_GEMM_TILELANG` 同 gate 上线
- `PROVENANCE.md:352`：**"本阶段未跑 GPU…CPU 侧 compile-only 全绿"**
- verify parity 已证只覆盖 wkv（phase-1 台架）；phase-2 的 parity 台架在 §8.4 是 ⏳ 未执行
- **一次请求内同时跑 5 个新 kernel（4 个从未上 GPU）——任一错即污染**
- **修复**：二分（临时只留 wkv → 逐形状放回）

**#3 [中·设计盲区] out_stride 的 "row 0 免疫" 陷阱（tl-parity-vs-old 缺陷 #4）**

- m=1 的老 per-row 基线**不使用 out_stride**（行 0 恒在 offset 0）
- 调用侧传错 out_stride 时：row 0 正确、row≥1 整行错位
- verify 的 wq_b 传 `nlh*hd` 而真行距是 `nh*hd`——只有多行 kernel 才暴露
- **修复**：多行入口加 debug 断言（或首次调用回执核对）

**#4 [中·概率性] g_part 进程级单例竞写（tl-garbage R1 + tl-parity #5）**

- `wkv_shim.cu:95` 的 `static float* g_part` 是进程级单例（`ks*MPAD*N` 一块）
- TP8 的 ranks-are-threads 模型下，同层两个 rank 的调用**时间重叠** → 竞写 P → 静默数值损坏
- **修复**：per-rank 切片（`g_part + rank * stride`）或 caller-provided buffer

**#5 [设计内非法] 半挂配置（tl-garbage R2）**

- `chain_dev.rs:6211-6222` 契约原文：**"the arm is only coherent when eager AND verify both take it"**
- `DSV41_GEMM_TILELANG=1` 单挂（verify TileLang + eager 老 kernel）= proj-mma 判死的同族形态
- **修复**：重跑必须双挂 `DSV41_GEMM_TILELANG=1 DSV41_GEMM_TILELANG_EAGER=1`

## 修复落地

| 修复 | commit | 状态 |
|---|---|---|
| xsc/xsc_r 算术域修正 | `98f50e8` | ✅ 已推 |
| moe_bs 守卫（build 堵死） | `29033a5` | ✅ 已推 |
| 双挂 e2e 重跑 | — | ⏳ 等第六次重编 |

## 关键教训

1. **`.max()` 的域（dimension）比存在本身更重要**——放在 `/32` 前面是 no-op，移到后面才是真 floor
2. **parity 微基准的输入必须与生产布局同构**——hash 输入 + 自分配缓冲验证的是"自洽"而非"同构"
3. **"两个谓词是一对"**——A 有 m-predicate 而 ASC 没有，这是隐性耦合（reduce 的谓词兜住了下游）
4. **m=1 基线对 stride 类缺陷免疫**——out_stride/a_stride/scale pitch 的错误只有多行才暴露
5. **半挂配置是设计内非法**——接线契约明文规定双侧同换
