# Stage B 执行清单 — 从 9.38ms 起（可立即开工）

**口径**：B=1 decode 稳态 p50，同二进制背靠背 A/B；判据 = 四段文本逐字 + `faults=0` + p50。
基线 = 9.38ms / 106.6 tok/s（第 27 轮，默认配置 MIX_GATE=OFF + GATEUP/DOWN_FUSE=ON + HEAD_SLICE + sparse 3-deep + T1 + A32/LUT + P1/P2 + env 缓存）。
行号已逐个读码复核（HEAD `c24dff4` + 工作区改动）。

## 1. Top-6 执行顺序（把握×收益）

**排序分 = 把握 × 收益中点**，同一分内先做"补丁已就绪/零风险"的。

| 序 | 项 | 主 kernel / 行号 | 预期 | 把握 | 分 | 实施步骤 | 验证口径 |
|---|---|---|---|---|---|---|---|
| **1** | **down-vec-320** | `dsv41_experts_mxf4.cu:1153-1176`（新增 256 值/组）、尾循环 `:1178`；核 `:1027`，launcher `:1488-1502` | **−0.15~0.20** | 高 0.9 | **0.157** | 代码已在工作区（未提交）。①`bash kernels/cuda/build.sh` ②老 .so vs 新 .so 在**同为 `DSV41_DOWN_FUSE=1`** 下比 token 流 | `DSV41_DOWN_FUSE=1`（必须 ON 才走融合核）。位级：新/旧 .so token 逐字节相同。⚠️ 与 `expert_gemv_fp4_batched_kernel:871-923`（k=inter 同 320）**已不再逐位相同**，须单独确认那侧走的是 nv2 主循环 |
| **2** | **AR store 融合**（attn 40 + moe 40） | store `ferrite_kernels.cu:8117`、pubred `:8140`；Rust 已就位 `chain_dev.rs:2041-2082`；补丁 `~/.xbot/users/web-4/workspace/ar-fuse-store/` | **−0.08~0.16**（−80 节点） | 中高 0.7 | **0.084** | ①树内先单独复测 `DSV41_AR_STORE_FUSE=1`（第 19 轮失败与 GATEUP_FUSE 数值 bug 耦合，该 bug 已在 `c599f6c` 修复）②通过后 `git apply -p1` 打 moe 侧补丁（`add_inplace`/`moe_down_reduce` 也是载体） | `DSV41_AR_ST=0` 全关；单点 `DSV41_AR_ST_ATTN=0` / `DSV41_AR_ST_MOE=0`。位级两道：新 .so `AR_ST=0` vs 旧 .so 必须一致（防 GEMV 重编译漂移）；node 计数 `p2p_ar_store_v5` 82→2 |
| **3** | **quant 生产者直出 T2** | `dsv41_kernels.cu:86` kernel、launcher `:1625`；T1 已做 `chain_dev.rs:700/1443/1544`；**T2 已落地（`db29175`）**：`dsv41_kernels.cu:2731 rmsnorm_q_kernel` + `:2768 dsv41_rmsnorm_q`（qr 侧）、`chain_dev.rs:567/2439/2573`（MoE 侧）| **−0.13~0.28** | 中 0.5 | **0.103** | 剩余 ~126 次/步，可摘的 80 次集中在 4 个 site：`:1995`(wo_a，即 `o`)、`:2043`(wo_b，即 `wo`)、`:2701`(shared w1/w3)、`:710`(lin)。**已实施 40 次**：`:710` 的 qr 路（rmsnorm epilogue 直出，`DSV41_QR_EPI`）+ `:2701`（fp4 换独立 scratch，T1 flag 得以存活）。**未实施**：`:1995`/`:2043`（生产者是 sparse_attn/apply_rope 与 wo_a 的 gemv epilogue，absmax 需整行 ⇒ 各自要动 #1 核或两个 producer，风险高） | 位级：scale 是同一 absmax 的同一舍入 ⇒ 应逐位；不逐位则立刻回退（`DSV41_QR_EPI=0`）；MoE 侧是纯 Rust 的 scratch 拆分，数值不可变。⚠️ T2 的 MoE 一半依赖 T1 flag：`2381` 的 quant1(xn) 现在只在 `MIX_GATE=1` 时才跑（否则它白吃掉 T1 flag）|
| **4** | **hc_post_inplace 折进 AR#2 pubred** | `dsv41_kernels.cu:2938/2975`，调用点 `chain_dev.rs:1490/1613`；折入目标 `ferrite_kernels.cu:8140` | **−0.15**（−80 节点） | 中 0.6 | **0.09** | ⚠️ **不是折进 w2 gemv**——w2 在 AR#2 之前，`s.o` 未定稿。`hc_post` 的 `x` 是 AR 之后的 `s.o` ⇒ 唯一合法载体是 **pubred epilogue**（`moe_reduce` `chain_dev.rs:888→892`） | 新 env（如 `DSV41_HCPOST_EPI`）。逐位：`hc_post_inplace` 用 `__fmaf_rn` 且升序 k，跨 CU 必须显式 `__fmaf_rn`（1 ULP = 文本变化）。⚠️ `res==out` 别名：pubred 的 rank-slice 网格与"单线程独占 4 列"约束冲突，需先做列分区 |
| **5** | **cross-layer pipe P4** | `dsv41_kernels.cu:3143` tail、`:3048` dots；AR 核 `:8117/:8140`；设计 `dsv41-persistent-arch.md:91` | **−0.30~0.50** | 低 0.3 | **0.12** | front 拆 ⟨A⟩collapse_norm 留原位（MoE 依赖 xn）+ ⟨B⟩mixes dots+tail 推迟到 L+1，与 AR 合 launch（AR 只 5×1024 线程，poll 窗口上百 SM 空转） | 新 env 分段开（`DSV41_XPIPE`、`DSV41_XPIPE_DOTS`）。⚠️ AR 核 `step = gridDim.x*blockDim.x` 会因新增 block 错位 ⇒ 必须 `blockIdx.x < ar_blocks` 分区；post/comb 缓冲双缓冲 |
| **6** | **gateup s_act CSE** | `dsv41_experts_mxf4.cu:768-852`（gate 链 `:788-807`，up 链 `:819-838`，读同一 `s_act[j..j+15]`） | **0~0.15** | 低 0.4 | **0.03** | 先 `cuobjdump -sass` 确认 nvcc **未**做 CSE；未做则把 16 个 s_act 值显式 load 进寄存器供两链复用（L1TEX op 42→26，−38%） | 无 env（源级恒等变换）。位级：`s_act[]` 是同一 smem、同序 fma ⇒ 逐位；p50 A/B 必须同二进制前后编译 |

**合计**：9.38 → **8.7ms**（1–4 兑现，−0.51~0.69）；1–6 全兑现 → **8.6ms**；再加 xn-megafuse（−0.25~0.5，`dsv41-xn-megakernel-design.md`）→ **~8.2ms**。
**不可忽略的隐藏项**：~700 launch × ~1.5µs ≈ **0.9ms 节点尾延迟**。本清单的 #2 + #4 已摘掉 160 个节点（≈ −0.24ms），#5 再摘 ~40 个；根治仍须 Stage C persistent（700→120 节点，`dsv41-persistent-arch.md`）。

## 2. 不做清单（明确否决 / ROI 不足）

| 项 | 原预期 | 判定 |
|---|---|---|
| **hc sinkhorn 藏进 collapse** | −0.21 | ❌ **实测否决**：可藏窗口 = collapse P1 **0.46µs** ≪ sinkhorn **6.5µs**，收益 0.018ms（`STATUS.md` 第 5198+ 行，`/tmp/tail_phase_probe.cu`）。tail 的临界路径就是 warp0 串行链，同核内无等长窗口 |
| **f32→bf16 cast 消除** | −0.21 | ❌ **口径不符，对本基线贡献为 0**：唯一来源是 `ferrite-kernel/src/cuda.rs:2113 gemm_cublas`，只用在 `matmul_dev` 的 `n==16` 分支（`:2480`）；dsv41 单序列链走 `lin_bf16`（`chain_dev.rs:791`）→ `cublas_m1()` **默认 false** → `gemv_bf16`（核内转换，无 cast）。profile 里确实无 `f32_to_bf16`。该优化已在 `2026-09-11 cast 消除落地` 以 `FERRITE_GEMM3`（`cuda.rs:2260/2461`，默认 ON）落地，余下只是 b300 上跑 `FERRITE_GEMM3=0` 对照确认 **m=16 路径**，与 9.38ms 无关 |
| **route_topk 深度优化** | −0.06 | ❌ 5.2µs/次已是噪声地板（P1 只删了 `hist` 死码，固定成本未降）；不值得投入 |
| **xn-megafuse「5 族一 launch」** | −1.0 | ❌ 前提已被否决（`s.xn` 是复用缓冲，两簇被 AR#1 隔开）；且 B 簇"再省一遍 LUT/a32"的空间已被 `9e4e3df` 补齐后消失。**只剩 2-launch 版（−0.25~0.5，真收益是 idx_wp 那个 8-block 小 slot）**，列入 #6 之后 |
| **4-deep sparse prefetch** | −0.03~0.05 | ⚠️ ROI 不足：N=4 恰好整除（零 tail）是唯一实益，但 0.34ms 的核再省 10-15% 只有 ~0.04ms；排在 #5/#6 之后。**6-deep 不做**（余 4，tail 变长） |
| **MoE cooperative 段核 / 段 A 融合** | −0.4~0.7 / −0.6~1.0 | ⚠️ 收益最大但属 Stage C 结构性改动（同编译单元 + 数值契约），**不在 Stage B 范围**；Stage B 先拿满增量收益再上 |
| 批量 sed 翻默认值、给 sparse/quant 加 blockDim、拆 split=8、MTP | — | ❌ 历史阴性/明令禁止（`roadmap-200-tokps.md` §6） |
