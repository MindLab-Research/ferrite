# `kernels/tilelang/vendor/` — TileLang 源码补丁的 vendoring 目录

> 工部 · 2026-09-13。上位：`docs/agent/tilelang-moe-bs-wiring.md`（fp4 MoE 臂接线）、
> `docs/agent/tcgen05-blockscaled-proto.md`（原型）。

本目录是 ferrite 对 **TileLang 自身缺陷**的正式修复记录。定位与
`kernels/cuda/tilelang_inc/`（vendored 的 C++ 头，编译期 `-I` 用）并列：
这里是**Python 侧源码补丁 + 幂等施补器**，改的是**库的源码**，不是运行期注入。

---

## 1. 为什么是「源码补丁」而不是 monkey-patch

`T.tcgen05_gemm_blockscaled()` 在 TileLang **0.1.14 漏写 `ann["is_tcgen05"] = 1`**
（`T.tcgen05_gemm()` 有）。后果：`cuda::Gemm::SelectInst` 的 `isTcgen05_` 分支永不
命中，而 SFA/SFB region 已定义 ⇒ 落进「这一定是 SM120 的 NVF4 `mma.sync` 路径」分支
并硬失败：

```
InternalError: T.mma_gemm_blockscaled() requires an SM120 CUDA target,
    but got target={..., "arch":"sm_103a"}
```

原型（`kernels/tilelang/moe_bs_proto.py`）当年的绕过是**运行时重新 `exec` 函数体 + 注入
那一行注解**。用户裁决：**不允许任何 hack**。区别是实质性的：

| | 运行时 monkey-patch | 本目录的源码补丁 |
|---|---|---|
| 生效范围 | 只有做了注入的那个进程/那次 import | 任何 import 该库的进程 |
| 与 `tilelang.compile` 的交互 | 依赖导入顺序；`_gemm_op` 与 `T.` 两份引用要同时换 | 无 |
| 可审计 | 「以为打上了其实没打」是常态 | 文件 sha256 + AST 复核 |
| 上游可接收 | 否 | 是（就是一份 unified diff） |

## 2. 内容

| 文件 | 性质 | 说明 |
|---|---|---|
| `tilelang-0.1.14/tcgen05-blockscaled-is_tcgen05.patch` | **补丁（唯一真相）** | 对 v0.1.14 `tilelang/language/gemm_op.py` 的 unified diff + 出处/基准 sha256/上游状态 |
| `apply_tilelang_patch.py` | 施补器 | 定位包 → **AST 判定** → `patch -p1`（sha 匹配时）或锚点插入（退化路径）→ **AST 复核**。幂等。 |

**基准**（补丁的上下文基准，改动前必须核对）：

| 项 | 值 |
|---|---|
| 目标文件 | `tilelang/language/gemm_op.py` |
| 版本 | v0.1.14 tag（= 远端 B300 上装的 0.1.14） |
| pristine sha256 | `c775813d5635e39a81df4b45a074f890895032f54d8b696b94b016d50e6426fa` |
| pristine 行数 | 627 |
| 打完后 sha256 | `9736412fa6191d021e093c7de47487c51d02971a68c9b5bc11224d2cffe016a7` |
| 打完行数 | 635（+8 行：注释 6 行 + 注解 1 行 + 空行 1 行） |
| 上游状态 | **截至 `main` 仍未修**（`tcgen05_gemm()` 有、`tcgen05_gemm_blockscaled()` 没有）——2026-09-13 逐一比对 raw.githubusercontent.com |

> 「`main` 也没修」这条不是可选项：它决定了本补丁是**我们自己的 vendored 修复**，
> 而不是「等上游发版」。升级 TileLang 时必须重跑施补器的 `--check-only`。

## 3. 施补（唯一入口）

```bash
# 检查（不改文件；退出码 1 = 未修复）
python3 kernels/tilelang/vendor/apply_tilelang_patch.py --check-only

# 施补（幂等：已修就什么都不做）
python3 kernels/tilelang/vendor/apply_tilelang_patch.py

# 用**拥有 tilelang 的那个解释器**（远端 B300 上是 dsv41_venv）
/opt/dlami/nvme/dsv41_venv/bin/python kernels/tilelang/vendor/apply_tilelang_patch.py

# 包不在默认解释器里 / 想指定包目录
python3 kernels/tilelang/vendor/apply_tilelang_patch.py \
    --tilelang-path ~/.local/lib/python3.12/site-packages/tilelang
```

**判据**（每一次都必须看到这两行）：

```
[vendor] tcgen05_gemm_blockscaled.is_tcgen05 : fixed
[vendor] verified         : tcgen05_gemm_blockscaled() now sets ann["is_tcgen05"] = 1
```

施补器**只认 AST**，不认 grep：`is_tcgen05` 在隔壁 `tcgen05_gemm()` 里也有，
全文搜索会把「隔壁函数有」误判成「这个函数有」——那正是这个 bug 的形状。

### 3.1 退化路径（源文件不是 pristine 时）

若 sha256 ≠ 基准（换过小版本、或本树已被别的补丁动过），施补器**跳过 `patch -p1`**、
改用锚点插入，并把锚点唯一性与「锚点确实落在 `tcgen05_gemm_blockscaled()` 函数体内」
两条都验过才写。**这条路径每次都打 warning** —— 出现 warning 就意味着 vendoring
基准要更新（顺手把新 sha 记进 §2 表）。

## 4. 与 AOT 生成脚本的关系（不允许再出现 monkey-patch）

`kernels/tilelang/gen_moe_bs_aot.py` **不再**注入任何东西。它开头调用的是**校验**：

```python
verify_blockscaled_fix()   # 未修 ⇒ 直接 RuntimeError，附施补命令
```

即：**AOT 生成的前提是库已经被修好**。这条顺序是硬的 —— 生成脚本在远端跑，
施补也必须在远端先跑（远端有自己的 site-packages）。

---

## 5. 主 agent 执行清单：fp4 MoE AOT 生成（远端 GPU）

> ⚠️ **本清单是给主 agent 的**（工部禁止远端 GPU 操作，见 `AGENTS.md` GPU 纪律铁律）。
> ⚠️ 本清单**不含任何 bench**：`gen_moe_bs_aot.py` 只做 lowering + codegen + dump 源码，
> 不 launch、不碰 GPU 计算。真跑数值验证见 `docs/agent/tilelang-moe-bs-wiring.md §6`。

### STEP 0 — 施补（在远端、用拥有 tilelang 的解释器）

```bash
# 本地 → 远端：三个文件（施补器 + 补丁 + AOT 生成器）
scp -r kernels/tilelang/vendor ubuntu@43.202.208.136:~/tl_bs/vendor
scp kernels/tilelang/gen_moe_bs_aot.py ubuntu@43.202.208.136:~/tl_bs/

# 施补 + 校验（幂等，可重复执行）
ssh ubuntu@43.202.208.136 \
  'cd ~/tl_bs && /opt/dlami/nvme/dsv41_venv/bin/python vendor/apply_tilelang_patch.py'
# 判据：出现 "tcgen05_gemm_blockscaled.is_tcgen05 : fixed" + "[vendor] verified"

# 复核（应 exit 0）
ssh ubuntu@43.202.208.136 \
  'cd ~/tl_bs && /opt/dlami/nvme/dsv41_venv/bin/python vendor/apply_tilelang_patch.py --check-only'
```

> 若 STEP 0 的 sha256 与 §2 表不符（远端装的可能不是 tag 内容）⇒ 看有没有 warning；
> 有 warning 也照样会验 `verified` 那行，**只要 AST 复核过就成立**。

### STEP 1 — 生成 AOT（BM=64 目标档；`--bm 128` 是原型认证回退）

```bash
ssh ubuntu@43.202.208.136 \
  'cd ~/tl_bs && mkdir -p aot_gen && /opt/dlami/nvme/dsv41_venv/bin/python \
     gen_moe_bs_aot.py aot_gen --bm 64'
# 产出：
#   aot_gen/moe_bs_up_tl.cu        <- device dump（shim #include 的那个）
#   aot_gen/moe_bs_up_tl_host.cu   <- TileLang 自己的 host launcher = CUtensorMap 权威配方
#   aot_gen/moe_bs_up_tl.banner    <- banner 行（含 raw sha256 / raw bytes）
#   aot_gen/moe_bs_tl_config.txt   <- 冻结几何 + device 签名 + 布局契约
# 判据：stdout 里出现 "[vendor] tilelang fix verified" 与 config 头的
#       "is_tcgen05_fix=verified-in-source ..."（**不是** patched=1）
```

### STEP 2 — 回传

```bash
scp ubuntu@43.202.208.136:'~/tl_bs/aot_gen/*' kernels/cuda/tilelang_gen/
# 注意：PROVENANCE.md 约定 "banner 之外逐字节等于 dump"，banner 由 _banner() 打进
# moe_bs_up_tl.banner；仓内 .cu 的落盘形态 = banner + raw（照其他 *_tl.cu 的先例）
```

### STEP 3 — 编译 + 符号自检（CPU，无需 GPU）

```bash
cd ~/ferrite && ./kernels/cuda/build.sh 103a
nm -D libferrite_kernels.so | grep -E "dsv41_moe_tilelang_gate_up_bs|dsv41_moe_bs_pack_wsf"
```

⚠️ **这一条是本臂的接线前提**：`moe_bs_shim.cu` 用
`#if __has_include("moe_bs_up_tl.cu")` 决定 `FERRITE_MOE_BS_TL_MISSING`。**产物到位前
该 TU 编译成一个空翻译单元、两个符号都不存在**，`supports_moe_tilelang_bs()` 为
false ⇒ `DSV41_MOE_TILELANG_BS=1` 会（正确地）被拒绝并报「.so 缺符号」。
见 §6。

### STEP 4 — 数值/性能验证

不在本清单内：见 `docs/agent/tilelang-moe-bs-wiring.md §6`（空 B300、无 co-tenant、
`DSV41_MOE_TILELANG_BS=1` vs `=0` 的逐元素与 tok/s A/B）。

---

## 6. 落地检查：`moe_bs_shim.cu` 的 `__has_include` 守卫

| 项 | 结论 |
|---|---|
| 守卫写法 | `kernels/cuda/tilelang_gen/moe_bs_shim.cu:121` `#if __has_include("moe_bs_up_tl.cu")` |
| 能否命中新产物 | ✅ 能。`#include "..."` 先搜**包含者所在目录**，`moe_bs_up_tl.cu` 与 shim 同在 `tilelang_gen/`，不需要额外 `-I`（`build.sh:178` 只有 `-I tilelang_inc`，够用） |
| 产物到位后 | `FERRITE_MOE_BS_TL_MISSING` **不再定义** ⇒ 走 `#ifndef` 之后的全部实体（编译期形参自检 + 两个导出符号） |
| 产物缺失时 | 定义 `FERRITE_MOE_BS_TL_MISSING` ⇒ 整个 TU 退化成空，`build.sh` 的 `tilelang_gen/*_shim.cu` 通配不会因为缺 include 而炸 |
| 运行期门 | `device.rs::supports_moe_tilelang_bs()`（`.so` 里两个符号都在）——**build-vs-runtime 分离**，与其余各臂同构 |

⇒ **守卫无需改动**。它已经能接住新产物；STEP 3 的 `nm -D` 就是「接住了」的证据。

---

## 7. 接线审计（2026-09-13）：fp4 臂**尚未接进执行链**

> 本节是**审计结论**，不是改动。工部按「照图施工」执行：任务书要求「确认
> `chain_dev.rs` 的 `DSV41_MOE_TILELANG_BS` gate 指向 `moe_bs_shim` 的 **device 入口**，
> 与 bf16 臂同构」——实测**两侧都不成立**：gate 不存在，device 入口也不存在。
> 这是**方案与代码的冲突**，按工部规则**上报**，不自行发明接口。

### 7.1 实测（`grep` 可复现）

| 断言 | 命令 | 实测 |
|---|---|---|
| `chain_dev.rs` 里有 BS gate | `grep -c moe_bs crates/ferrite-models/src/dsv41/chain_dev.rs` | **0** |
| 有 `moe_tilelang_gate_up_bs(...)` 调用点 | `grep -rn "moe_tilelang_gate_up_bs(" crates/` | **无**（只有 `device.rs` 的字段/包装器定义） |
| `.so` 里有 device-table 入口 | `grep -n 'extern "C"' kernels/cuda/tilelang_gen/moe_bs_shim.cu` | 只有 `dsv41_moe_tilelang_gate_up_bs`（**HOST 表**）+ `dsv41_moe_bs_pack_wsf`；**没有** `_dev` 孪生体 |

⇒ 现状：`DSV41_MOE_TILELANG_BS=1` **只会**在 `load.rs` 里建 packed-SF 池
（`load.rs:815` 的 `want_bs`），**没有任何运行期分派**。AOT 产物即使生成、编译进
`.so`，也**不会被调用**。这正是本项目 #1 测量偏置陷阱的镜像形态（不是「测成了老路」，
而是「根本没有路」）。

### 7.2 两处缺口的具体形态

**(A) shim 侧：缺 device-table 入口。** 现有入口的表是 HOST 数组
（`eid/order/counts` 走 `cudaMemcpyAsync(HostToDevice)`），并且**无条件**在 capture 中
decline（`moe_bs_shim.cu:603-607`）。bf16 臂的对照实现是
`dsv41_moe_tilelang_gate_up_bf16_dev`（`moe_bf16_shim.cu:495`）：

| 维度 | bf16 `_dev`（已落地） | BS 臂（缺） |
|---|---|---|
| 表指针 | `const int*`，**DEVICE** | 现在只有 HOST |
| `nseg` | `const int* nseg_dev`（mover 里 `seg >= *nseg_dev`） | 现在是 `int nseg` 形参 |
| capture 守卫 | **状态门**：只在 INIT 未完成（`g_a_up == nullptr`）时 decline | **无条件 decline**（永远进不了 graph） |
| 表生产 | `dsv41_moe_align_from_group`（device） | 同上可用，但 BS 需要 **`bm = 64`** 的那一套 |

⚠️ **`bm` 不一致是硬约束**：`tl_moe_bs_gather_kernel` / `_scatter_kernel` 的索引是
`order[seg * kBm + r]`，`kBm = 64`；而 `chain_dev.rs` 现有的 `tl_order` scratch 是
`TILELANG_SEG_CAP * TILELANG_BM`（**BM = 16**，给 bf16 臂用的）。⇒ BS 臂需要**独立的
一套 `bm=64` 表**（`36*64*4 = 9216 B` + 两张 144 B + 4 B ≈ **9.5 KB**），
由**第二次** `dsv41_moe_align_from_group(..., bm = 64)` 填充（该 kernel 的 `bm` 是形参，
生产者已经具备）。

**(B) `chain_dev.rs` 侧：缺整条分派。** 需要的改动与
`moe_tilelang_ready` / `moe_tilelang_tables_dev` 同构（照
`docs/agent/tilelang-moe-bs-wiring.md §5.3`，但把「host 表」换成「device 表」）：

1. `moe_bs_ready(...)` —— 就绪门：`weights::moe_tilelang_bs()` ∧
   `supports_moe_tilelang_bs()` ∧ `!ld.experts_ilv` ∧ 冻结形状（384/5120/320/topk∈[1,6]/m∈[1,6]）
   ∧ packed-SF 池在。**注意**：与 §5.3 的草案不同，**不再需要 `!capturing()`** ——
   设备表 + 状态门守卫下 capture 合法（这正是 bf16 臂在 commit `0818d78` 里去掉那一项的
   同一个理由）。
2. `moe_bs_weights(ld, dim, inter_local)` —— 4 个基址（`w1/w3.w1`、`sfw1/sfw3`）+
   `w3-w1`、`sf3-sf1` 的步距校验（参差池 ⇒ 算术索引会静默错值 ⇒ 返回 `None`）。
3. device 表：新增 `bs_order/bs_counts/bs_eid/bs_nseg` scratch（≈9.5 KB）+ 一次
   `route_group` + `align_from_group(bm=64)`（可复用 `grp_*` 中间量）。
4. 分派：`moe_rows` 的点位 + eager 点位各一处，**并且**下游
   `if !grp_gu && !tl_gu {` 必须变成 `&& !bs_gu` —— 否则本臂的结果会被兜底 launch 覆盖
   （bf16 臂踩过的原坑）。

### 7.3 上报

⇒ **(A) 与 (B) 需要用户/尚书省裁决后才动手**：这不只是「接线」，它给 shim 增加了
一个新的**导出契约**（表的 DEVICE 化 + `nseg` 形态变化），并需要 `chain_dev.rs` 的
新 scratch 与两处分派。工部的规则是**不自行调整方案**（`AGENTS.md` §工部原则 2）。

**在当前状态下，主 agent 执行 §5 的 STEP 0-3 是安全的**（只产出源码 + 编译进 `.so`），
但**必须知道**：`DSV41_MOE_TILELANG_BS=1` 在那之前**不会有任何效果**，端到端
fp4 收益（wiring §0 的预期）在 (A)(B) 落地前**不可测**。

