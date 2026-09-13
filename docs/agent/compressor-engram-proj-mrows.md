# compressor / engram 投影 mrows 化（2026-09-13 实施记录 + GPU A/B 手册）

> 病灶来源：`verify-amortization-lesion-audit.md` §0 第三行「compressor/engram 投影黑洞（无 gate）：
> ~3.5-5ms / 新 kernel / subagent 实施中」。权威模型：`mtp-verify-amortization-model.md`。
> 本文是实施交付：改动清单、逐位等价性论证、验证结果、上机 A/B 手册。

## 1. 病灶与修法

| # | 病灶 | 位置 | 原状 | 修法 | gate |
|---|---|---|---|---|---|
| 1 | compressor 投影 per-row | `chain_dev.rs::compress_proj_rows`（审计 :12357-12377） | 每 row 2 发 `lin_f32_on`（`comp_wkv`→`kvp_r`, `comp_wgate`→`scp_r`），m=6 ⇒ 12 发/compress-source 层，权重 `[128, 5120]`=2.6MB f32 重读 6 遍 | 每权重 1 发 `dsv41_gemv_f32_mrows`（m 行折进一次权重流） | `DSV41_COMPRESSOR_PROJ_MROWS=1`（默认 OFF） |
| 2 | engram 投影 per-row | `chain_dev.rs::engram_apply_rows`（审计 :12676-12689） | 每 row 1 发 `gemm_fp8_mx_on(wkv)`，重读 `engram_wkv`；且 `gemm_fp8_mx` 按 `m` 切换程序（m==1 SIMT GEMV vs m>1 16-row tile MMA）——"SEVERE" 判词 | 1 发 `dsv41_gemm_fp8_mrows`（**m==1 程序本身**的多行形式，C1-C6 逐位等价） | `DSV41_ENGRAM_PROJ_MROWS=1`（默认 OFF） |
| 3 | engram_gather per-row | `chain_dev.rs::engram_apply_rows`（审计 :12632-12645） | 每 row 1 发 `engram_gather`（m=6 ⇒ 12 发，2 个 engram 层 × 6 行） | 1 发 `dsv41_engram_gather_rows`（新增 `id_stride` 参数解 per-token 语义） | `DSV41_ENGRAM_GATHER_MROWS=1`（默认 OFF） |

**为什么 2 不是「违反 SEVERE 判词」**：判词说的是 `gemm_fp8_mx` **按 m 派发不同程序**是病，
不是「m 折行」是病。正确折行就是 verify 其它投影已经在用的 `dsv41_gemm_fp8_mrows`——它是
**m == 1 程序本身**的权重驻留多行形式。

## 2. 改动清单（file:line）

### CUDA（`kernels/cuda/dsv41_kernels.cu`）
- `:930` `engram_gather_kernel` 增加 `int id_stride`（hash_ids 的 per-token 行距，元素数）。
- `:9325` `dsv41_engram_gather`：调用点传 `id_stride = n_cols`（**旧寻址逐元素不变**）。
- `:9349` 新增 `dsv41_engram_gather_rows`（`rows` 折行 + `id_stride`；返回 0/2，`[engram-gather-rows]` 一次性回执）。
- `:6041` `gemv_f32_v2_mrows_kernel<M>`（**前一 subagent 遗留，经复核保留**）= `gemv_f32_v2_kernel<WPR>` 的逐语句转写（见 §3-A）。
- `:6119` `dsv41_gemv_f32_mrows` 入口（同上前一 subagent 遗留，复核保留）：m∉1..8 / `k%4` / v2 escape hatch / `n≥2048` → 返回 2（decline），`[gemv-f32-mrows]` 一次性回执。

### Rust（`crates/ferrite-models/src/dsv41/`）
- `device.rs:305` / `:1588` / `:3072`：`engram_gather_rows` 的符号声明 + `ko!` 绑定 + `Device::engram_gather_rows`（`Result<bool>`，decline ⇒ `Ok(false)`）。
- `device.rs:776` / `:1626` / `:5964` / `:5992`：`gemv_f32_mrows` 的符号声明 + `ko!` 绑定 + `Device::gemv_f32_mrows` + `supports_gemv_f32_mrows`。
  ⚠️ 这两条 `device.rs` 改动是**方案外的最小加法**：前一 subagent 把 compressor 臂写成"ARMED but INERT"
  （gate 无 C 绑定 ⇒ 永远回落），那正是本仓 #1 记录的"幻影 gate"陷阱。键绑定是让 gate 真正生效的唯一路径。
- `chain_dev.rs:1291/1328/1358/1391/1402`：三个 gate（`engram_proj_mrows` / `compressor_proj_mrows` / `engram_gather_mrows`）+ 三个一次性 decline 回执。
- `chain_dev.rs:12573-12610`：compressor 投影接线（fold 两权重；decline ⇒ 整条 per-row 循环 + 回执）。
- `chain_dev.rs:12963-12985`：engram 投影接线（单发 mrows；decline ⇒ per-row 循环 + 回执）。
- `chain_dev.rs:12885-12912`：engram_gather 接线（单发 rows；decline ⇒ per-row 循环 + 回执）。

## 3. 数值等价性论证

### A. compressor 投影（`dsv41_gemv_f32_mrows`）
per-row 参照 = `lin_f32_on` → `Device::gemv_f32_on` → **`dsv41_gemv_f32_v2`**
（`n = head_dim = 128 < GEMV_F32_V2_MAX_N = 2048`，`DSV41_GEMV_F32_V2` 默认 ON）。
转写点对点（`dsv41_glue.cu:1346-1416`）：

| v2 | mrows 版 | 一致？ |
|---|---|---|
| `kper = ((k+WPR-1)/WPR + 3) & ~3`；`k0 = kw*kper`；`k1 = min(k0+kper,k)` | 同式（WPR 由模板参数变运行参数，只进整数算术与 fold 上界） | ✅ |
| `for (c = k0+lane*4; c+3 < k1; c += 128)` `float4` 走 | 同式，`#pragma unroll 2` 同 | ✅ |
| `acc = __fmaf_rn(wv.x,xv.x,acc)` x,y,z,w 顺序 | 同序、每行**独立**累加器 `acc[r]`，无跨行合并 | ✅ |
| `__shfl_down_sync(0xffffffff, acc, off)` off=16,8,4,2,1 | 同树，每行各跑一次 | ✅ |
| `part[warp]` + `sum += part[(warp/WPR)*WPR + j]` j 升序 | `part[r*8+warp]`，j 升序同 | ✅ |
| `out[row] = ...` at lane 0 / kw==0 | `out[r*n + row]` 同点 | ✅ |
| `wpr = n>=16384?1 : n>=4096?2 : n>=1024?4 : 8`；`rpb = 8/wpr`；grid `ceil(n/rpb)`；block 256 | 同式同 grid | ✅ |

**唯一变化**：`wv` 的 `float4` 被提到 r 循环外（同一地址同一字节，复用 m 次）。权重字节是只读常量，
复用不可能改变任何一行；行间无重组。⇒ **行 r 与"自己那一发 m==1"逐位相同**。
decline 路径（v2 关闭 / `n≥2048` / `k%4` / m∉1..8）保持 per-row 循环，字节不变。

⚠️ 注意：转写的是 **v2**（K-split + smem fold，与 v1 的 fold 顺序差 ~1e-6 f32）。
若误转写 v1 就会复现 engram 那个"按 m 派发不同程序"的 SEVERE 类缺陷——所以入口在
`DSV41_GEMV_F32_V2=0` 或 `n≥2048` 时 **decline**。

### B. engram 投影（`dsv41_gemm_fp8_mrows`）
per-row 参照 = `dsv41_gemm_fp8_mx` at **m==1**（`DSV41_NO_GEMV_FP8` 未设）→ `gemm_fp8_gemv_kernel`。
`gemm_fp8_mrows_kernel<M>` 头部 C1–C6（`dsv41_kernels.cu:5280-5333`）逐条对 m==1 程序：
- C1 同一 K 走序（`kb` 升序、`j = kb*32+lane`，staged 保序形式；launcher 要求 `g_gemv_fp8_mode ≥ 3`）；
- C2 同操作数同字节（权重行 fp8 字节 + ue8m0 行；激活 `s_lut[s_a[]] * s_as[]`，与 gemv 的物化形式按构造逐位同）；
- C3 同 `shfl_xor` 树，每 (warp, r) 一次；C4 无跨行重组；C5 无 K-split/smem fold；C6 累加保持单条串行 `acc[r] +=`（`#pragma unroll 32` 源形）。
⇒ 每行是该行 m==1 发射的逐位复刻，且**这正是 batched / lazy 两臂共享的程序**（修的是 dispatch 不一致）。

### C. engram gather（`dsv41_engram_gather_rows`）
per token 语义 = hash id 是 per-token 的，且 `eng_ids_r` 布局是 `[row][engram layer][col]`
⇒ 同一 engram 层的 m 行相距 `n_eng * n_cols`（生产 `n_eng = 2`，`engram_layer_ids = [1,14]`）。
新增的 `id_stride` 就是这个行距：`id = hash_ids[(row_id/n_cols)*id_stride + row_id%n_cols]`。
- 逐元素：`row r` 读的 id = `(r*n_eng + li)*n_cols` 处的 id（与 per-row 调用 base 指针给出的完全相同）；
  写槽 `out[(row*n_cols + col)*head_dim + j]` 与 per-row 调用 `out + r*n_cols*ehd` 相同（调用方 `eng_rows_r` 行连续）。
- ⇒ 折行版 = m 次 per-row 调用的**逐元素拼接**；不批量的原因（"per-token 语义不可批量"）被 `id_stride` 消解。
- 代价收益：**仅省 launch 开销**（m=6、2 个 engram 层 ⇒ 12 发 → 2 发，约 −10 次 launch ≈ 0.03–0.05ms）。
  这是三臂里最小的一个；不是带宽问题（每次仅 `n_cols * engram_head_dim` 个元素）。

## 4. 验证结果

| 项 | 结果 |
|---|---|
| `cargo check --workspace --all-targets` | ✅ EXIT=0（无新 warning） |
| `cargo test -p ferrite-models --lib` | ✅ 92 passed / 0 failed / 2 ignored |
| 远端 nvcc compile-only：**HEAD + 本次全部 hunks**（隔离掉并发 sparse-attn 改动） | ✅ 无 error，`our.o` 6.13 MB |
| 新 kernel 寄存器 | `gemv_f32_v2_mrows_kernel<1..8>` = 32/34/40/48/48/54/54/46 regs，smem = M×32 B；`engram_gather_kernel` = 24 regs |
| 既有 warning | 仅 `#177-D`（`e2m1_to_f`、`k1max`）—— 与本改动无关 |

⚠️ **开工时的仓库状态**：工作树里有另一个 agent 正在并发改 `dsv41_kernels.cu` 的
**W2-MROWS-TP sparse-attn `row_pitch`** 一族（`sparse_attn_{,warp,pf,split,merge,orope}_kernel`
+ `dsv41_sparse_attn{,_orope}` 启动器）。截至 2026-09-13 00:47 该半成品**尚不能编译**：
`dsv41_kernels.cu(1188/1551/2106) 'rp' is undefined` + `(9682/9713) too few arguments`。
**这 5 个 error 全部属于那一族改动，与本交付无关**（本交付区域 nvcc 无 error，见上表隔离编译）。
未提交、未回退、未触碰它们。

## 5. GPU A/B 手册（上机时照做）

### 5.1 前置
同一次 serve 内背靠背，**只看 `[dspark] steps=` 的 draft/verify/commit 真实分解**（禁止吞吐反推）；
e2e 一律 background 模式；nsys 轮要 `DSV41_AR_V5=0 DSV41_GRAPH_STEP=0` + `env -u FERRITE_P2P` + `NCCL_NVLS_ENABLE=0` + 5 分钟 SIGINT 硬帽。

### 5.2 三个臂（各自独立 A/B，先 OFF 后 ON）
| 臂 | gate | 回执（stderr） | 期望 |
|---|---|---|---|
| B-CP (compressor 投影) | `DSV41_COMPRESSOR_PROJ_MROWS=1` | `[gemv-f32-mrows] ARMED m=.. n=128 k=5120 WPR=8 -> ONE launch over m rows` | verify −? ms |
| B-EG (engram 投影) | `DSV41_ENGRAM_PROJ_MROWS=1` | （`dsv41_gemm_fp8_mrows` 自身回执） | verify −? ms |
| B-GT (engram gather) | `DSV41_ENGRAM_GATHER_MROWS=1` | `[engram-gather-rows] ARMED rows=.. n_cols=.. hd=.. id_stride=.. -> ONE launch` | verify −0.03~0.05ms |
| 反例（验证 gate 真的没被 silently 旁路） | 上表 gate 开着但出现 `[compressor-proj-mrows] ARMED but DECLINED` / `[engram-proj-mrows] ... DECLINED` / `[engram-gather-mrows] ... DECLINED` | 说明落回 per-row 循环，**本轮数字不可当 fold 测量** |

**票面**：三臂合计 ~3.5-5ms / verify step（审计 §0 第三行）。compressor 投影是主体
（40 层量级 × 6 行 × 2 权重 × 2.6MB 重读），engram 两臂是小头。

### 5.3 逐字节判据（每臂必须过）
同一 serve 内 `gate=0` 与 `gate=1` 两轮，同 prompt、同 seed、同步数：
1. **逐字节**：两轮的输出 token 序列（以及 `DSV41_DSPARK_DUMP=1` 的 layer dump）**完全相同**；
   `faults=0`。
2. 若 dump 里有 `eng_kv_r` / `kvp_r` / `scp_r` 通道：这些中间张量也应与 OFF 轮**逐位相同**
   （A/B 用的是同一份程序，理论上 0 差；出现非 0 即等价性论证被证伪，必须回头查）。
3. `step_ms`：ON 轮应**下降**且 acc 不变（acc 变了说明数值被打坏）。
4. 反例回执：确认 stderr 里**没有** `DECLINED` 字样（否则等于没测）。

### 5.4 组合建议
- 先单独 B-CP（最深、票面最大）；再 B-EG（同文件、同类）；B-GT 因其收益在噪声内，
  建议与 B-CP 一起开关并只在 nsys 里看 gather launch 数（12 → 2）作为证据，不单独追 step_ms。
- 三臂与 B1 快赢臂（`ATTN_MROWS` / `VERIFY_ROPE_MROWS` / `WOB_MROWS_F32` / `GATE_MROWS_ROUTE` /
  `MROWS_ACT_CPASYNC`）无耦合：分别测，不要一次全开（否则无法归因）。
