# draft 链 P3-lite —— attn 半的段级融合（可实施级设计）

> 上游口径：`docs/agent/mtp-verify-amortization-model.md`（唯一权威性能模型）、
> `docs/agent/draft-1ms-design.md` §3 P3、`docs/agent/draft-perf-ledger.md`（289–292 launch / 2.66 GB）。
> 段内核先例：`docs/agent/draft-p3-fusion.md`（P3a/P3b/P3c 分期）、
> `docs/agent/mrows-swallow-batched-implementation-design.md` §3（**B4/B6/K2 的核头论证**）、
> `docs/agent/verify-amortization-lesion-audit.md` §6–§7（P3B 的实测与二分）。
> 本机 **无 GPU**：所有 ms 为推算并标 `(推)`；launch 计数与 file:line 为**代码精确值**。
> 仓库状态：工作树 `1d8b953` + peer（mrows-mpar）未提交改动。
> 生产几何：dim 5120 / hd 512 / nh 64 / q_lora 1280 / o_groups 8 / win 128 / hc 4 /
> vocab 129280 / mr 256 / n_mtp 3 / bs 5 / TP8。

---

## 0. 结论（TL;DR）

1. **attn 半的严格清单 = 32 个节点/block**（29 kernel + 3 `memcpy`），本报告 §1 逐条给 `file:line`。
   任务书的「24 发」= 其中 **draft_attention 本体**（去掉 seed 链 4 kernel + 3 memcpy，
   `sparse_attn` 按 1 个调用点计）= 24 ✓ —— 口径已对齐，见 §1.3。
2. **P3-lite（形态 I，无网格栅栏）逐位等价可达 24 → 11**；任务书的 **6-9 需要额外一步：
   启用 `gemm_fp8_mrows_f32`（B6）f32 直读族 —— 而 B6 正是二分判"非逐位"的那一支**。
   ⇒ **6-9 与"全逐位"在当前仓库里不可同时成立**；本报告给出两者的分离点（§2.6）。
3. **五段里有三段是"零新核"**（K2 / B4 / orope 都已在 HEAD + `device.rs` 已接线）：
   **段 C（−1/block）、段 D（−6/block）、段 B（−7/block）**。段 A（hc 前端）与段 E 需要新核（§2.5/§2.6）。
4. **段 D 有一个静默错答案陷阱**：`sparse_attn_orope` 的 **split 臂**（draft 的 `b*m=5 ≤ kAttnMaxBM=8`
   ⇒ **必走 split 臂**）的 merge 核 **没有 `mm*row_step` 项**（`dsv41_kernels.cu:1812`），
   而 orope 非 split 核有（`:2352`）。⇒ 一次 `m=bs` 调用会让 row 1..4 全部 rope 在 `rope_pos`
   ——**5 行里 4 行位置错**，且**不报错**。必须先给 merge 核补上该项（`row_step == 0` 时逐位不变）。
5. **P3B 的漏洞不是"算术错"，是"论证的许可范围"错**（§3）：
   （a）b1 的"partial attempt is harmless by construction"**证伪** —— `swiglu_limit_q` 覆写的是
   `xq` 里那份**在两个臂之外只量化一次**的 `xn`（`dspark_dev.rs:2708`），fallback 会读到被覆写的输入；
   （b）b3 的"bit-identical **PROVIDED** the per-row path would take v2"是一个**条件式**，
   而同一家族里 `head_gemv_bf16_mrows` 的**同款条件式已被实测证伪**
   （`dspark_dev.rs:3133-3140`：header 声称位级同一，实测 33% echo ⇒ ~1e-3 ⇒ near-tie argmax 翻）。
   MoE gate 的下游是 `route_topk`（128 路离散 top-3），**放大器比 argmax 更硬**（翻一个 expert = O(1)）。
   （c）二分从未把 P3B 隔离成单变量（`lesion-audit §7` 第 4 刀"待出"）。
   ⇒ 本设计立 5 条硬规则（§3.3），其中 **R1 同程序律** 与 **R5 一折一门** 直接堵死上述三洞。
6. **实施**：本报告已落地 **段 C + 段 D**（§5 的 diff），新增门 `DSV41_DRAFT_P3LITE`（默认 OFF）
   + `DSV41_P3LITE_{SEED_NORM_ROPE,KV_NORM_ROPE,ATTN_OROPE}` 三个**单变量**开关。
   段 B 需要 `DSV41_ATTN_PROJ_ALIGN` 做前置（R1），段 A/段 E 需要新核 + `device.rs` 接线（peer 区，未碰）。

---

## 1. 现状：attn 半的完整调用序列与依赖图

### 1.1 逐发清单（`file:line` = 当前工作树）

记号：`K` = kernel，`M` = `memcpy`/`cudaMemcpyAsync` 节点，`†` = 有 gate 的分支。

| # | 类别 | 调用 | 代码位置 | 形状/语义 | 依赖（读→写） |
|---|---|---|---|---|---|
| **seed 链**（`seed_window(s, seed_pos, slot_dev)`，调用点 `dspark_dev.rs:1846`） | | | | |
| 1 | K | `quant1(main_x)` | `:3341` | `[dim]` f32 → fp8+sc | `main_x` → `xq/xsc` |
| 2 | K | `gemm_fp8_mx(wkv)` | `:3342` | `m=1, n=hd, k=dim` | `xq/xsc` → `mk` |
| 3 | K | `rmsnorm(mk, kv_norm)` | `:3353` | `n=1, dim=hd` | `mk` → `mk` |
| 4 | K | `apply_rope(mk)`（`rope_at :3361`） | `rope_at :3586` | `rows=1,row_len=hd,step=1,pos=seed_pos` | `mk` → `mk` |
| 5 | M | `ring_append` / `memcpy_d2d(→slot)` | `:3373` / `:3385` | `hd` f32 | `mk` → `window[s]` |
| **attention 本体**（`draft_attention(s,pos,slot_dev)` `:1791`） | | | | |
| 6 | K | `quant1(xn)` | `:1853` | `bs*dim` f32 → fp8 | `xn` → `xq/xsc` |
| 7 | K | `gemm_fp8_mx(wq_a)` †`proj_attn_mrows` | `:1872` / `:1862` | `m=bs, n=ql, k=dim` | `xq/xsc` → `qr` |
| 8 | K | `rmsnorm(qr, q_norm)` | `:1884` | `n=bs, dim=ql` | `qr` → `qr` |
| 9 | K | `quant1(qr)` | `:1892` | `bs*ql` | `qr` → `xq/xsc` |
| 10 | K | `gemm_fp8_mx(wq_b)` †`proj_attn_mrows` | `:1908` / `:1898` | `m=bs, n=nh*hd, k=ql` | `xq/xsc` → `q` |
| 11 | K | `apply_rope(q)` **×bs**（`rope_queries :1925`） | `:3405`（`rope_at :3586`） | 每行 `rows=nh,row_len=hd,step=0,pos=rope_pos+r` | `q` → `q` |
| 12 | K | `quant1(xn)` | `:1930` | `bs*dim`（D1：全 bs 行） | `xn` → `xq/xsc` |
| 13 | K | `gemm_fp8_mx(wkv)` †`proj_attn_mrows` | `:1945` / `:1934` | `m=bs, n=hd, k=dim` | `xq/xsc` → `kv` |
| 14 | K | `rmsnorm(kv, kv_norm)` | `:1957` | `n=bs, dim=hd` | `kv` → `kv` |
| 15 | K | `apply_rope(kv)`（`rope_at :1979`） | `:3586` | `rows=bs,row_len=hd,step=1,pos=kv_pos` | `kv` → `kv` |
| 16 | M | `memcpy_d2d(window→all_kv)` | `:2012` / `:2018`+`:2025` | `n_win*hd*4`（wrap 时两段） | `window[s]` → `all_kv` |
| 17 | M | `memcpy_d2d(kv→all_kv)` | `:2032` | `bs*hd*4`，dst=`all_kv+wbytes` | `kv` → `all_kv` |
| 18 | K×2 | `sparse_attn`（split + merge） | `:2055`（launcher `device.rs`；impl `dsv41_kernels.cu:9671`） | `b=1,m=bs,h=nh,d=hd,window=n_win,index_topk=bs,idx_stride=0` | `q,all_kv,sink,idxs,clen` → `o` |
| 19 | K | `apply_rope_inv(o)` **×bs**（`rope_queries_inv :2080`） | `:3468`（`rope_at :3586`） | 每行 `rows=nh,row_len=hd,step=0,pos=rope_pos+r,inverse` | `o` → `o` |
| 20 | K | `quant1(o)` | `:2129` | `bs*nh*hd` | `o` → `xq/xsc` |
| 21 | K | `wo_a_grouped_fp8` | `:2133` | `grid=(n/8, groups)`, `rows=bs, k=4096, out_stride=ol_total` | `xq/xsc,wo_a` → `wo` |
| 22 | K | `quant1(wo)` ‡`gemm_fp8_mrows_f32` | `:2217` / `:2201` | `bs*ol_total`（B6 门默认 OFF） | `wo` → `xq/xsc` |
| 23 | K | `gemm_fp8_mx(wo_b)` †`proj_attn_mrows` | `:2234` / `:2224` | `m=bs, n=dim, k=ol_total` | `xq/xsc` → `o` |
| ‡ | K | `bf16_roundtrip(o)` / `(wo)` / `(o)` | `:2127` / `:2192` / `:2255` | `DSV41_DRAFT_BF16_DOMAIN` / `_ATTN_BF16` **默认 OFF** | — |

**合计 32 节点**：kernel = 29（含 `sparse_attn` 的 2 发与两处 `×bs` 的 rope = 5+5），memcpy = 3
（`:2016` 的 wrap 情形是 2 段 ⇒ 33）。与 `draft-perf-ledger.md` §1.2(b) 的「32 条/block」逐条一致 ✓。

### 1.2 依赖图（严格串行；唯一的扇出在 `quant1` 与 rope）

```
                      [前块 h / premix]  (块间串行，见 draft-p3-fusion §0.2)
                              │
 主链 x → quant1(main_x) ──► gemm(wkv) ──► rmsnorm(mk) ──► rope(mk) ──► ring_append ──┐
   (seed)      ①               ②              ③              ④            ⑤           │
                                                                                       ▼
 xn ──► quant1 ⑥ ──► gemm(wq_a) ⑦ ──► rmsnorm(qr) ⑧ ──► quant1(qr) ⑨ ──► gemm(wq_b) ⑩ ──► rope(q)×5 ⑪
 └──────────┐                                                                                    │
            ▼                                                                                    ▼
      quant1 ⑫ ──► gemm(wkv) ⑬ ──► rmsnorm(kv) ⑭ ──► rope(kv) ⑮ ──────────────┐        sparse_attn ⑱ ◄─ window[s]
                 (与 seed ② 同一 wkv 权重，不同 m/位置)                        ▼        (split ⑱a → merge ⑱b)
                                                     all_kv = [window ; kv]  ◄─ ⑯⑰                │
                                                                                                   ▼
                                          rope_inv(o)×5 ⑲ ──► quant1(o) ⑳ ──► wo_a_grouped ㉑        │
                                                                                    │                │
                                                                                    ▼                ▼
                                                                    quant1(wo) ㉒ ──► gemm(wo_b) ㉓ ──► o  → 块 B 的 hc_post
```

**读出的三个结构事实**（决定段边界）：
1. **扇出点只有两个**：`quant1(xn)` ⑫ 被 wq_a ⑦ 与 wkv ⑬ 共用（同一份 fp8 输入）；
   `rope` 与 `quant` 都是行局部（逐行零共享）。
2. **两处跨块耦合**（决定"能不能合进一个 launch"）：`hc_mixes` 的整行归约（`hc_dim=20480`）
   与 `rmsnorm` 的行归约；`sparse_attn` 的 key-split（`g_attn_part` 跨块）。其余全是行局部。
3. **`sparse_attn` 的两发是同一 `g_attn_part` 的 producer/consumer**（`split` → `merge`），
   它们已经是"一个调用点两次发射"，且 `orope` 版本把 merge 变成了 rope+fp8 的 epilogue。

### 1.3 与任务书「24 发」的口径对齐

| 口径 | 数 | 说明 |
|---|---|---|
| draft_attention + seed_window 全节点 | 32 | 29 K + 3 M（§1.1） |
| − seed 链（①–⑤） | 27 | seed 属于"进 ring 的主链 KV"，本报告的**段 C** 覆盖它 |
| − 3 个 memcpy（⑤⑯⑰） | 24 | `memcpy` 是 copy 节点，单独由 `ring_append` / 定长 win 行处理 |
| − `sparse_attn` 两发计 1 个调用点 | 23→ | 任务书口径把 split/merge 记 1 条 |
| **= 任务书「attn 半 24 发」** | **24** | 本节即该集合的权威 `file:line` 清单 |

---

## 2. 段融合方案

### 2.0 术语与两条形态（沿用 `draft-p3-fusion.md` §3.0）

| | 形态 I（无栅栏） | 形态 II（段内核 = 相位机 + 网格栅栏） |
|---|---|---|
| 相位间 | launch 边界 | 软件网格栅栏（全块到达） |
| 前置 | 相邻相位"所有权一致"或用已有融合核 | grid ≤ 常驻容量 + 栅栏自复位 |
| 风险 | **低**（全是已验证核的组合） | **高**（`dsv41-persistent-arch.md §6`：P1d +3.3ms / hc-merge +3.2ms） |
| 本报告 | **P3-lite 只做形态 I** | 不做（标注为后续项） |

**P3-lite 的定义**：只用"**同程序**的相邻 launch 合并"（含仓库已有的融合核），
**不引入网格栅栏、不换 contraction 程序、不折 AR**。

### 2.1 段 A —— attn 侧 hc 前端（`hc_mixes` + `hc_collapse_norm`）【新核】

**现状**（3 发，`draft_body`，不在上面 24 内）：`hc_mixes(attn)` `:1357`
→ `hc_collapse` `:1422` + `rmsnorm` `:1448`（P3a a1 已可折成 `hc_collapse_norm` `:1410`）。

**融合可行性（关键约束：两条相位要的 blockDim 不同）**：
- `dsv41_hc_mixes` 的 launch 是 `<<<rows, mix*32=768, smem>>>`，`hc_mixes_kernel` 的 dot 相位
  是 `for (c = lo + threadIdx.x; c < hi; c += blockDim.x)`（`dsv41_kernels.cu:2432`）——
  **`blockDim.x` 直接进求和顺序**（`c += 768` vs `c += 1024` 是两条不同的加法链）。
- `dsv41_hc_collapse_norm` 的 launch 是 `<<<rows, 1024>>>`，其 tree 用 `blockDim.x >> 5 = 32` 个
  warp 部分和（`:10950-10957`）——**换成 768 线程 ⇒ 24 个部分和 ⇒ 1 ULP 级改动**。

**结论（本报告的设计）**：新核 `dsv41_draft_hc_front` 取 `<<<rows, 1024>>>`，
两个相位各自**保留自己的逻辑宽度**：

```
phase 1（hc_mixes）：只让 threadIdx.x < 768 的线程参与 dot 相位，
                    并把 `c += blockDim.x` 写成显式 `c += 768`
                    ⇒ lane→c 映射与 blockDim=768 的基线逐位相同；
                    shfl_xor 树、wpart[32] 跨 warp 折、sinkhorn 单 warp 寄存器
                    —— 三段全部逐句照抄 hc_mixes_kernel:2462-2541。
phase 2（collapse_norm）：1024 线程全参与，body 逐句照抄
                    dsv41_hc_collapse_norm_kernel:10939-10963（fmaf 链 + shfl_down 树 + 升序跨 warp 和）。
相位间：__syncthreads()（只定序，不改值）。
smem：hc_mixes 的 `mixes[mix]`+`cm[hc*hc]`（24+16 f32）+ collapse 的 `red[32]` → 独立槽位，总 < 300 B。
```

- 发数：**3 → 1**（P3a a1 开时 2 → 1）。
- 逐位论证：**每个相位与它所替代的核在同一条加法链上**（列举：dot 的 lane→c 映射 / `shfl_xor`
  树 / 跨 warp 折序 / sinkhorn 迭代序 / collapse 的 `fmaf` 升序 hc / rmsnorm 的 `shfl_down` 树 +
  threadIdx.x==0 的 `red[i]` 升序折），**逐句 pin，不依赖编译器收缩**。
- 票面：**−2 发/block × 3 = −6 发/步**，`(推)` −0.06~−0.10ms。
- ⚠️ **需要新核 ⇒ 需要 `device.rs` 的 launcher + 符号探测（peer 区，本次未碰）**。

### 2.2 段 B —— q 投影链（`wq_a` + `q_norm` + `wq_b` + rope）【零新核：K2】

**现状**（`:1853..1925`）：`quant1(xn)` ⑥ → `gemm(wq_a)` ⑦ → `rmsnorm(qr)` ⑧ → `quant1(qr)` ⑨
→ `gemm(wq_b)` ⑩ → `rope(q)×5` ⑪ = **10 发**。

**融合方案**：`dsv41_gemm_fp8_mrows_rope_norm`（**K2**，`dsv41_kernels.cu:6934`；
`device.rs::gemm_fp8_mrows_rope_norm` 已接线）把 `norm_rows(qr) + quant_rows(qr) + proj_mrows(wq_b)
+ apply_rope_mrows(q)` **四段并成一发**（核头 `:6676-6690` 逐段给出参照核）。

**⚠️ R1 同程序律（段 B 的前置）**：K2 的第 2 段是 `gemm_fp8_mrows_kernel<M>`
（SIMT、权重驻留、`shfl_xor` 树），而 draft 现状的 `gemm_fp8_mx@m=bs` 走的是 **16 行 TILE MMA**
（`dspark_dev.rs:1854-1860` 明文：*"gemm_fp8_mx at m = bs = 5 is the 16-row TILE program, while the
verify's wq_a runs proj_mrows → a DIFFERENT summation"*）。
⇒ **K2 单独开启 = 换程序 = P3B 那一类改动**，不是纯 launch 融合。
✅ 正解：**段 B = `DSV41_ATTN_PROJ_ALIGN` ∘ K2**（`attn_proj_align()` 已存在，`:1861`）。
该门的存在理由正是"official calls the same `F.linear` on both sides"，所以程序统一 +
融合是**同一个正确性主张**，必须同门开关。段 B 的落地形式：

```
quant1(xn) ⑥                                   (1 发，保留：f32 直读族=B6，见 §2.6)
gemm_fp8_mrows(wq_a)  = proj_attn_mrows ⑦'      (1 发，ATTN_PROJ_ALIGN 的程序)
K2: [rmsnorm(qr) + quant1(qr) + gemm_fp8_mrows(wq_b) + rope_mrows(q)]  = 1 发
```
- 发数：**10 → 3**（若再接受 B6 的 f32 直读，→ 2）。
- 逐位论证：K2 的核头逐段 pin（1a/1b/1c/2/3），且 **block=1024/32 warps 是"载重"的**
  （`:6692-6704`：norm 的 tree 由 blockDim 定，mrows 的 256 线程会换求和顺序）。
  段 3 的两个 `__syncthreads()` 也是载重的（`:6706-6720`）。
- **`pos_rows` 是唯一位置源**（`:6673`：no `pos_ctr`/`mul`/`off`/`step`）——draft 的
  `ensure_pos_dev` 已经把 `pos_rows[r] = pos + r` 上传（`:3520-3522`）✓ 与
  `rope_queries` 的 `pos + r` 完全同源。
- 票面：**−7 发/block × 3 = −21 发/步**，`(推)` −0.20~−0.30ms。**五段里最大的一刀。**
- 落地代价：需与 `ATTN_PROJ_ALIGN` 同轮 A/B（两个门一起看，否则测的是程序差）。

### 2.3 段 C —— kv 链 + seed 链（`wkv` + `kv_norm` + rope）【零新核：B4/`rmsnorm_rope`】

**现状**：seed ①②③④（`quant1, gemm, rmsnorm, rope`）+ kv ⑫⑬⑭⑮（同构）= 8 发（含 2 个 `quant1`）。

**B4 模式**（`dsv41_rmsnorm_rope` / `dsv41_rmsnorm_rope_mrows`）把 `rmsnorm + apply_rope` 并成 1 发。
**B4 在 verify 侧被列为"弃用四 gate"之一 —— 它的逐位论证哪里破了？**（任务书点名）
→ 见 §3.4。**本报告的判定：B4 的核内论证是干净的（逐句照抄，相位 2 无归约 ⇒ 线程映射不影响值），
它在 verify 侧被弃用的原因是"票面/交互"而不是"逐位"**：
(i) 票面被 `NORM_MROWS` 削半（`mrows-swallow-batched-design §7-4`：只剩 −1 发/层 = −0.13ms）；
(ii) **`VERIFY_FORK` 的 stream 耦合**——它替代的两发是 `*_on`（side-stream）版本，
新核必须**接 `kv_stream`**，否则"静默把 kv 半链拖回主流"，抵消 FORK 的收益（核头 `:10469-10472`）。
⇒ 教训：**融合核的 stream 参数是一等公民**，draft 侧不涉及 FORK，但要写进签名与注释（段 C 的落地即如此）。

**draft 侧的落地**（本报告已实施，见 §5）：
```
seed: rmsnorm(mk) ③ + apply_rope(mk) ④  →  dsv41_rmsnorm_rope(n=1, dim=hd, rope_len=rd,
                                             base=pos_base, mul=1, off=seed_pos-pos_dev, step=1)
kv:   rmsnorm(kv) ⑭ + apply_rope(kv) ⑮  →  dsv41_rmsnorm_rope(n=bs, dim=hd, rope_len=rd,
                                             base=pos_base, mul=1, off=kv_pos-pos_dev, step=1)
```
- **逐位论证（指令级）**：
  ① norm 相位：`rmsnorm_rope_kernel`（`:2408-2423`）与 `ferrite_rmsnorm`（`ferrite_kernels.cu:329`）
     是**同一条式**：`i += blockDim.x` 步进、`__shfl_down_sync` 树、`red[32]`、
     `threadIdx.x==0` 的 `red[i]` **升序**折、`rsqrtf(t/dim+eps)`、`or_[i] = xr[i]*inv*w[i]`。
     两者 **launch 都是 `<<<n, 1024>>>`** ⇒ 树宽相同（32 个 warp 部分和）。
     （`ferrite_rmsnorm` 是**另一个 TU**（`ferrite_kernels.cu`）——这正是 §2.7 那条红线的适用处；
     但 R2 的 parity 套件已实测 `rmsnorm_rows == ferrite_rmsnorm`（`tests_dsv41_r2_parity.cu` 的
     T4.norm-ferrite），`rmsnorm_rope_kernel` 与 `rmsnorm_rows` 同体 ⇒ 传递性成立。）
  ② rope 相位：`t = (*base)*mul + off + row_i*step`（`:2426`），rope 区 =
     `or_ + (dim - rope_len)`（`:2427`，**尾部** `rope_len` 列）。
     `apply_rope` 的区 = `x + r*row_len + (row_len - dim_rope)`（B4 核头 `:10448-10450`）
     = 尾部 `2*half = rope_len` ✓ 同区。旋转式 `x0*c - x1*s` / `x0*s + x1*c` 同。
     **t 的整数**：seed `pos_dev + (seed_pos - pos_dev) + 0*1 = seed_pos` ✓（`rope_at` 的 off 同）；
     kv 第 r 行 `pos_dev + (kv_pos - pos_dev) + r = kv_pos + r` ✓。
     **相位 2 无归约、每元素只写一次 ⇒ 线程映射不可能移动值**（同 B4 核头 `:10462-10464`）。
- 发数：**seed 4 → 3，kv 4 → 3**（各 −1）；合计 **8 → 6**，即 24 集合内 **−1**。
- 票面：−2 发/block × 3 = **−6 发/步**，`(推)` −0.06~−0.10ms。
- **回退契约**：`rmsnorm_rope` 无"decline"哨兵（`rmsnorm_rope_kernel` 只在 `n<=0||dim<=0` 时
  返回 `cudaErrorInvalidValue`，`rope_len<=0` 时**跳过 rope 只做 norm**）⇒ 必须**先探符号**
  （`Device::rmsnorm_rope` 在 `.so` 缺符号时 `Ok(false)`）**再决定发射**，否则旧 `.so` 上会
  **只做 norm 不做 rope**（静默错答案）。本报告的接线按此写（§5.3）。

### 2.4 段 D —— `sparse_attn` + `rope_inv(o)` + `quant1(o)`【零新核：orope + 一个载重修正】

**现状**：`sparse_attn` ⑱（split+merge = 2 发）+ `rope_inv(o)×5` ⑲ + `quant1(o)` ⑳ = **8 发**。
`dsv41_sparse_attn_orope`（`:9851`）把 merge 变成 rope+fp8 的 epilogue：
- split 臂（`:9831-9840`）：`sparse_attn_split_kernel` + `sparse_attn_merge_kernel(..., cos, sin,
  base, rope_rd, half, mul, off, step, inverse, xq, xsc, row_pitch)`；
- 非 split 臂（`:9843-9846`）：`sparse_attn_orope_kernel(..., row_step, row_pitch)`。

**发数：8 → 2**（一次调用 = split+merge 两发）。**−6/block = −18 发/步**，`(推)` −0.18~−0.25ms。

#### ⚠️ 段 D 的静默错答案陷阱（本报告最重要的单点发现）

rope 位置在两发里的公式**不一样**：

| 核 | `tt` 公式 | 位置 |
|---|---|---|
| `sparse_attn_merge_kernel`（split 臂） | `(*base)*mul + off + hh*step` | `dsv41_kernels.cu:1812` |
| `sparse_attn_orope_kernel`（非 split 臂） | `(*base)*mul + off + hh*step + **mm*row_step**` | `:2352` |

其中 `hh = blockIdx.y`（**head** 下标），`mm = row % m`（**行**下标）。
draft 的 `b=1, m=bs=5` ⇒ `b*m = 5 ≤ kAttnMaxBM = 8`（`:1498`）⇒ **split 臂必被选中**
（`:9705` 的分支条件）⇒ **`row_step` 被静默忽略** ⇒ merge 会给 row 0..4 全部算
`tt = rope_pos`，而 draft 要的是 `rope_pos + r` ⇒ **5 行里 4 行的 o-rope 相位错**。
这不是崩溃，是"看起来合理的错输出"——与 `44f4956` 的静默错 token 同类。

**修正（本报告已实施，§5.2）**：给 `sparse_attn_merge_kernel` 加形参 `int row_step`，
`tt` 加 `+ mm * row_step`（`mm = row % m`）。**`row_step == 0` 时与该核的旧公式逐位相同** ⇒
所有既有调用者（plain `dsv41_sparse_attn` 的 split 臂）传 0，**零影响**；orope 的 split 臂
传调用方的 `row_step`。修正后两臂的 `tt` 同式。

#### 段 D 的逐位论证

1. **split 相位**：与 plain 路径**是同一个 `sparse_attn_split_kernel`、同一组实参**
   （唯一差异是 `row_pitch`，draft 传 0；orope impl 的选择器与 plain launcher 同源同序，
   `:9821-9825` 明文"the two must agree or the fused path declines shapes it could have carried"）。
2. **merge 相位 / rope**：`sh_row[]` 在 smem，rope 区 = `sh_row + (d - rope_rd)`（尾部 rd，
   `:42-53`），与 `apply_rope(o + r*nh*hd, rows=nh, row_len=hd, dim=rd)` 的**尾部 rd** 同区 ✓；
   `t = pos_dev + (rope_pos - pos_dev) + hh*0 + mm*1 = rope_pos + mm` ✓
   （draft 的 `rope_queries_inv` 每行是 `t = rope_pos + r`，且 `step=0` 保证同一行的
   `nh` 个头同相位 ✓ 对应 orope 的 `hh*step` 项为 0）。
   旋转式与 `(inverse ? -1 : 1)` 的处理与 `apply_rope_kernel` 同（`:50-52`）。
3. **fp8 发射**：layout `xbase = ((size_t)row * h + hh) * d`（`:63`）⇒ `[b*m, h, d]` 连续
   = draft 的 `quant1(o, bs*nh*hd)` 的 `[bs, nh*hd]` 布局 ✓（`xq/xsc` 是调用方的 compact 块）。
   `quant_kernel<0>` 的 32 lane `shfl_xor` amax 是同一段代码（核头 `:56-60` 声明 PHASE 3 verbatim）。
4. **位置锚**：`base = pos_base`（slot 0 = `pos_dev`），`mul=1`，`off = rope_pos - pos_dev`，
   `step = 0`，`row_step = 1`，`inverse = true`，`clen_rows = null`（`cl = *clen = bs`），
   `idx_stride = 0`（复现历史 `topk` pitch）。

> **R1 检查**：段 D 的两发与 plain 路径**同程序**（同一批 kernel 符号），无 contraction 程序更换 ✓。
> **R4 检查**：`o` 的下游是 `wo_a` → `wo_b` → `hc_post` → … → head **argmax** ⇒ 段 D 的输出喂离散决策，
> 必须按 R4 拿**实测回执**（k_acc 序列），不能只凭上面 4 条源码论证（§6）。

### 2.5 段 E —— `wo_a` + `wo_b`【pair】

**现状**：`wo_a_grouped_fp8` ㉑ + `quant1(wo)` ㉒ + `gemm_fp8_mx(wo_b)` ㉓ = **3 发**。
- `wo_a_grouped` 已是 weight-stationary 的 1 发覆盖 8 group（`dsv41_kernels.cu:7216`，核头 C1-C6）。
- 唯一的可折项是 ㉒：`quant1(wo)` → 交给 `gemm_fp8_mx_f32` / `gemm_fp8_mrows_f32` 直读 f32。
  **B6 门已在（`wob_mrows_f32`，`:2200`）且默认 OFF**——**§2.6 说明为什么不能默认它**。
- ⇒ **段 E 在"全逐位"约束下 3 → 3（无收益）**；接受 B6 才 3 → 2。

### 2.6 6-9 的分离点：B6（`gemm_fp8_mrows_f32`）与"全逐位"不可兼得

把 §2.1–§2.5 加总（只数 24 集合内的 kernel，`sparse_attn` 按 1 个调用点计）：

| 方案 | 段 B | 段 C | 段 D | 段 E | 合计 | 逐位？ |
|---|---|---|---|---|---|---|
| 现状 | 10 | 4 | 8 | 3 | **25** | — |
| **P3-lite 全逐位**（本报告） | **3** | **3** | **2** | 3 | **11** | ✅ 逐位（R1/R2/R3 全过） |
| + B6 f32 直读 | 2 | 2 | 2 | 2 | **8** | ❌ B6 非逐位（`lesion-audit §6`：断崖头号嫌疑） |

- **11 已经能拿到票面**：−14 发/block × 3 = −42 发/步；按账本 16.8µs/发 − 段内真实工作，
  `(推)` **−0.5 ~ −0.75ms**（段 B 的 −21 发是大头）⇒ draft 3.87 → **3.1-3.4ms**；
  与已落地的 P0/P3a/图叠加后才落到任务书说的 1.4-1.9ms。
- **6-9 需要 B6**：B6 把 `quant1 + gemm_fp8_mx(m=1)` 换成 f32 直读的 mrows GEMV。
  它是一个**不同的量化程序**（核内直接对 f32 做 32-block amax，而不是读 `quant1` 写下的
  bytes+scale），所以**不可能逐位**——`lesion-audit §6` 把它列为 accept 断崖的头号嫌疑。
  ⇒ **决策建议：P3-lite 先交付 11 发（全逐位）并测 `[dspark] draft=`；只有在 11 发仍未达标、
  且 B6 拿到独立的双门禁回执（`draft_ms` 降 AND `mean-k` 不掉）时才上 6-9。**

### 2.7 smem 预算与三条硬性律（逐段适用）

| 段 | 核 | smem | 备注 |
|---|---|---|---|
| A | `dsv41_draft_hc_front`（新） | `mix*4 + hc*hc*4 + 32*4` ≈ 288 B（静态） | 两相位独立槽位 |
| B | K2 `gemm_fp8_mrows_rope_norm` | `32*(k) + table + M*(k/32)*4 + M*k/1 + …`，launcher 走 `dsv41_smem_ceiling` 动态属性；k=ql=1280 ⇒ < 48 KB | 核头 `:6744-6745` |
| C | `rmsnorm_rope` | `32*4` B | `red[32]` |
| D | orope/merge | `sh_row[512]` = 2 KB + `g_attn_part`（设备全局，非 smem） | `d <= 512` 由 launcher 保证 |
| E | `wo_a_grouped_fp8` | `nwarps*k + 256*4 + (k/32)*4 + k` ⇒ k=4096 时 **< 48 KB**（否则 decline） | `:7237-7239` |

**三条硬性律（`dsv41-layer-fusion.md` / `persistent-arch.md §5`）逐段复核**：
1. **同编译单元或显式 `__fmaf_rn`**：段 A/B/C/D 的核**全部在 `dsv41_kernels.cu`**，
   基线核（`hc_mixes`/`hc_collapse_norm`/`rmsnorm`/`apply_rope`/`sparse_attn`/`gemm_fp8_mx`）
   **也在同一个 TU** ⇒ 收缩决定一致 ✓。**唯一的跨 TU 项是 `ferrite_rmsnorm`**
   （`ferrite_kernels.cu`）——段 C 依赖 `rmsnorm_rope_kernel` 与它同式，故 **§6 的 parity 门禁必须
   包含"`rmsnorm_rope` vs `ferrite_rmsnorm`+`apply_rope`"这一对比**（R2 的 T4 已有同族先例）。
2. **归约顺序逐指令照抄**：段 A 的 dot 是"单条 `acc` 链 + `shfl_xor` 树"（**禁止** ACC4 的四累加器
   分支——它 `:2514` 自认"not bit-identical"）；collapse 的 `fmaf(pre[i], x[i*dim+c], acc)` 升序 i；
   norm 的 `shfl_down` 树 + 升序跨 warp。
3. **`split = 1`**：段 A 内的 `hc` 归约不做 K-split（`hc_dim=20480` 一行一树）；
   `DSV41_HC_MIXES_SPREAD` **不进段内核**（`:10717-10727` 明文：spread 的答案与单块不同）。

---

## 3. P3B 的漏洞分析（引以为鉴）

### 3.1 事实基线

| 项 | 内容 | 出处 |
|---|---|---|
| P3B 的四折 | b1 共享专家 mrows / b2 routed mrows / b3 gate mrows / b4 共享专家 epilogue add | `dspark_dev.rs:403-478` |
| 实测 | draft 3.87 → **3.44ms（−0.43ms）** ✓ 票面兑现一半 | `lesion-audit §6` |
| 代价 | `mean-k 1.34 → 0.75-0.92`，断崖从 line 62 **提前到 line 52** | 同上 |
| 二分 | 刀1 = B1 全量（含 P3B）→ 52；刀2 = −B6 → 52（B6 排除）；刀3 = −B6 −SF → 52（SF 排除）；**刀4 = "待出"** | `lesion-audit §7` |

⇒ **P3B 从未被单变量隔离**：它是在一个 4 路并行的臂里被连带判掉的（判词"P3B 已被二分判弃用"
是**工程裁决**，不是单变量实证）。这本身是本报告 §3.3 规则 R5 的由来。

### 3.2 三个洞（按可证性排序）

#### 洞 ①（可证，代码级）：b1 的"partial attempt is harmless by construction"**是假的**

`dspark_dev.rs:2710-2714` 的注释原文：
> *"Ok(false) = gate off, a shape the kernels decline, or a stale .so — the per-row loop follows
> and rewrites every buffer this attempt touched, so a partial attempt is harmless by construction
> (only `moe_out` is not scratch, and it is written by the merge `add_inplace` alone)."*

**反例**：`shared_expert_mrows`（`:2919-3008`）的第 3 步
`swiglu_limit_q(act, bs, sh_il, limit, self.xq.ptr, self.xsc.ptr)` 把**全部 bs 行的 fp8
写进 `self.xq` 的偏移 0**。而 `self.xq` 里那一份 `xn` 的量化是**在两个臂之外、只做一次**的
（`:2708` `self.quant1(self.xn.ptr, bs*dim)`）。于是：
- 若第 1、2 步成功、**第 3 步的 `swiglu_limit_q` 成功而后面的 `gemm_fp8_mrows(w2)` 返回
  `Ok(false)`**（形状/mode 级 decline 是可达的：`gemm_fp8_mrows` 的 `fold_r`/`a32`/`act_cp16`
  与 smem 属性都在 launcher 里判），则 `shared_expert_mrows` 返回 `Ok(false)`；
- fallback 的 per-row 循环读 `a = self.xq + r*dim`（`:2733`）——**row 0 的输入已被
  `swiglu_limit_q` 覆写成 swiglu 的输出**，而 fallback **不会重新量化 `xn`**（它在 `:2708`，
  在臂之外）。
⇒ **fallback 不是参考路径，而是"读被污染输入"的第三条路径。** 这正是任务书猜的那一类
（"累积器状态/缓冲区状态"）。

#### 洞 ②（结构性，家族级）：b3 的"bit-identical **PROVIDED**…"是条件式，且同款条件式**已被实测证伪**

`dspark_dev.rs:2311-2321`：
> *"row r is bit-identical to the per-row call it replaces, **PROVIDED** the per-row path would
> take v2; the wrapper enforces that (`gemv_bf16_v2_wanted(n_routed)` + the symbol)"*

**同家族的反例是写在仓库里的**（`dspark_dev.rs:3125-3140`）：
> *"the earlier v2 fold (`head_gemv_bf16_mrows`) was a NUMERICAL change: **its header claims
> bit-identity, but the verify's head measured it as false** (`verify_head_fold`: folded gives
> `verify_out[0] == next` on 33% of rows vs 9% for the per-row `gemv_bf16`), because it pairs the
> fma/decode differently ⇒ ~1e-3 on the logits ⇒ a near-tie argmax can flip."*

⇒ **"核头声称位级同一"在本仓库里已被实测推翻过至少一次**。而 b3 的下游比 argmax 更硬：
`gate` 的 scores → `route_topk`（`:2345`）是一个 **128 路离散 top-3 选择**。
任何 ULP 级差异在"128 路近邻"里翻一个 expert 的概率远高于 129280 路里翻一个 argmax，
而**翻一个 expert ⇒ MoE 输出 O(1) 变**（不是 1 ULP）——这就是"accept 掉"的放大链。
**论证的漏洞**：把"两个核是同一 program"当成**读源码可得**的结论，
而没有在**消费者粒度**（`route_idx` 序列）上取证。

#### 洞 ③（图捕获边界，放大器）：臂的决定发生在**捕获期**，decline 在 replay 不可见

`draft_moe` 在 `draft_body` 内，而 `draft_body` 是 `DSV41_DRAFT_GRAPH` 的**捕获区**
（`dspark_dev.rs:1286`、`draft_capture:1725`）。b1 的 `if !(p3b.sh_exp_mrows && …)` 是**host 分支**
⇒ 它在**捕获时**被解析、把选中的那一条序列表进图里；**replay 不会再判**。
于是洞 ① 的"半融合 fallback"一旦发生在捕获期，就被**永久写进图**——
而 replay 期没有任何一行日志能区分"融合臂"与"半融合 fallback 臂"。
⇒ 与 `chain_dev.rs:3964-3979` 的 *"print, do not silently degrade"* 纪律**正面冲突**。

### 3.3 P3-lite 的 5 条硬规则（每条对应上面的洞）

| # | 规则 | 堵住 | 落地形式 |
|---|---|---|---|
| **R1** | **同程序律**：只能合并**同一个 contraction 程序**的相邻 launch。禁止把 `gemm_fp8_mx@m=bs`（16 行 TILE MMA）与 `gemm_fp8_mrows`（SIMT GEMV）配对 | 洞 ② | 段 B 必须与 `DSV41_ATTN_PROJ_ALIGN` 同门；段 A/C/D 复核"同一批符号" |
| **R2** | **launch 级 all-or-nothing**：融合核**一次 launch 接管整条相位链**，绝不"试发 4 发、第 4 发失败再回退" | 洞 ① | 段 C/D 用**单次调用**；`rmsnorm_rope` 先探符号再发（§2.3） |
| **R3** | **回退不得读被污染输入**：fallback 读到的每个缓冲区，必须**不由被尝试的融合写过** | 洞 ① | 段 C/D 的融合**不写任何 `xn`/`xq` 的输入区**（只写 `mk`/`kv`/`o`/`xq(o)` 自身） |
| **R4** | **离散消费者律**：输出喂 argmax / top-k router 的折，必须拿**消费者粒度**的实测回执（`k_acc` 序列 / `route_idx` 序列），不接受纯源码论证 | 洞 ② | §6 的双门禁 + parity（段 D 必过） |
| **R5** | **一折一门**：每个折一个独立 env，能单独开关以做**单变量二分** | 洞 ③ 与判词方法学 | `DSV41_P3LITE_{SEED_NORM_ROPE,KV_NORM_ROPE,ATTN_OROPE}` |

### 3.4 B4（`rmsnorm_rope_mrows`）在 verify 侧被弃用的原因 —— 复核

任务书要求"注意 B4 在 verify 侧也是弃用四 gate 之一！它的逐位论证哪里破了？"。
**本报告的复核结论：B4 的逐位论证没有破；它被弃用的原因是票面 + stream 交互。**

- 逐位论证（核头 `:10452-10467`）：phase 1 = `dsv41_rmsnorm_rows_kernel` 逐句、phase 2 =
  `apply_rope_kernel` 的旋转逐句、**唯一的添加是一个 `__syncthreads()`**（只定序不改值）
  —— 本报告复核该核体（`:10489-10532`）**与论证一致**（无重结合、无跨行、无 K-split）。
- 真正的弃用理由（`mrows-swallow-batched-design` §7-4、§3 B4 行）：
  1. **票面被 `NORM_MROWS` 削半**：`norm_rows` 与 `apply_rope` 本来**各自已经是"m 行单发"**，
     B4 只是"两发并一发" = **−1 发/层 = −0.13ms**（不是 lazy 口径的"逐行 × m"）；
  2. **`VERIFY_FORK` 的 stream**：它替代的是 `norm_rows_on` / `apply_rope_on`（side-stream），
     新核必须接 `kv_stream`，否则静默拖回主流（核头 `:10469-10472`）。
⇒ **对 P3-lite 的可迁移教训有两条**：(i) 段 C 的票面同样只是 −1/链，别按"×5 行"编票；
(ii) **stream 是签名的一等公民**（draft 侧无 FORK，但要显式写明"用 main stream"）。

---

## 4. 实施优先级排序（节省 × 等价置信度 ÷ 风险）

| 序 | 段 | 节省 | 等价置信度 | 风险 | 需要 | 判决 |
|---|---|---|---|---|---|---|
| **1** | **段 D orope** | 8→2（**−6/block**） | **高**（同程序 split 核 + 尾部 rd rope + 同 layout fp8 发射；核头有 4 段参照） | 中（**必须先修 merge 的 `row_step`**；peer 正在改同族核） | 零新核 | **✅ 已实施** |
| **2** | **段 C rmsnorm_rope** | 8→6（**−2/block**） | **高**（同式 tree + 无归约的 rope；B4 已论证） | 低（需探符号，否则 `rope_len<=0` 静默只做 norm） | 零新核 | **✅ 已实施** |
| **3** | 段 B K2 | 10→3（**−7/block**） | **中**（前置：必须换到 `ATTN_PROJ_ALIGN` 的程序，R1） | 中-高（与 `ATTN_PROJ_ALIGN` 是一个共同主张，不能单独 A/B 段 B） | 零新核 | 设计就绪，待与 ALIGN 同轮 |
| **4** | 段 A hc 前端 | 3→1（−2/block） | 高（双相位逐句 + 逻辑宽度 768 的显式化） | 中（新核） | **`device.rs` 接线（peer 区）** | 设计就绪 |
| **5** | 段 E wo_b f32 | 3→2（−1/block） | **低**（B6 非逐位） | 高（accept 断崖头号嫌疑） | 零新核（门已在） | **不建议**（除非双门禁回执） |

**排序理由**：1/2 都是"**同一批 kernel 符号、行局部、无归约**"的折 —— 逐位论证最短、且**零新核**
（`device.rs` 免动，避开 peer 区）；3 的票面最大但**带一个程序统一的前置主张**，必须同轮 A/B；
5 的等价置信度最低而风险最高，直接排除。

---

## 5. 实施记录（本报告已落地的部分）

**门（全部默认 OFF，house rule）**：

| env | 折 | 默认 |
|---|---|---|
| `DSV41_DRAFT_P3LITE` | 总开关（三个 item 的默认值） | OFF |
| `DSV41_P3LITE_SEED_NORM_ROPE` | 段 C：seed 的 `rmsnorm(mk)+rope(mk)` → 1 发 | 跟总开关 |
| `DSV41_P3LITE_KV_NORM_ROPE` | 段 C：kv 的 `rmsnorm(kv)+rope(kv)` → 1 发 | 跟总开关 |
| `DSV41_P3LITE_ATTN_OROPE` | 段 D：`sparse_attn + rope_inv(o)×bs + quant1(o)` → 1 次调用（2 发） | 跟总开关 |

每项可用 `=0` 单独关掉（**R5：单变量二分**）。

### 5.1 改动清单

| 文件 | 改动 |
|---|---|
| `kernels/cuda/dsv41_kernels.cu` | ① `sparse_attn_merge_kernel` 加形参 `int row_step`，`tt` 加 `+ mm*row_step`（`mm = row % m`）；② 两个调用点：plain `dsv41_sparse_attn` 的 split 臂传 **0**（逐位不变），orope 的 split 臂传调用方的 `row_step` |
| `crates/ferrite-models/src/dsv41/dspark_dev.rs` | ① `DraftP3Lite` + `draft_p3lite()`（`:386` 附近，与 `draft_p3a`/`draft_p3b` 同构）；② `seed_window` 的 norm+rope 折；③ `draft_attention` 的 kv norm+rope 折；④ `draft_attention` 的 orope 折 |

**未碰**：`chain_dev.rs` / `device.rs` / `dsv41_experts_mxf4.cu` / `load.rs` / `weights.rs` / `tp.rs`（peer 区）。
**符号**：段 C/D 用的 `dsv41_rmsnorm_rope`、`dsv41_sparse_attn_orope`（+ `_rp`）**都已在 `device.rs` 接线并有符号探测**（`kernels.rmsnorm_rope` / `sparse_attn_orope`），故无需新 FFI。

### 5.2 段 D 的 merge 修正为什么是安全的

`row_step == 0` ⇒ `tt` 的算式与旧版**逐位相同**（`+ 0` 是 f32 整数的恒等），
且 `sparse_attn_merge_kernel` 的**唯一既有调用者**是 plain `dsv41_sparse_attn` 的 split 臂
（本报告给它传 0）。⇒ **OFF 臂（plain 路径）逐位不变**，是本设计的回退契约。

### 5.3 回退契约（每条折都不改变 OFF 臂）

1. **段 C**：`Device::rmsnorm_rope` 在**符号缺失**时返回 `Ok(false)` → 接线**先问再发**
   （`if p3lite.kv_norm_rope { fused = dev.rmsnorm_rope(...)?; }`），`fused == false` 时走原来的
   `rmsnorm + rope_at` 两发 —— 与今天逐位相同。**绝不"发了再说"**（否则旧 `.so` 上
   `rope_len` 语义会让它只做 norm）。
2. **段 D**：`sparse_attn_orope` 在**符号缺失 / 形状 decline（哨兵 1..3）**时 `Ok(false)` →
   接线回退到 `sparse_attn + rope_queries_inv + quant1(o)` 三发。**回退不读被污染缓冲**
   （R3）：orope 只写 `o` 与 `xq(o)/xsc`，而 fallback 的第一发 `sparse_attn` **重写 `o`**、
   `rope_inv` 再重写 `o`、`quant1(o)` 重写 `xq/xsc` ⇒ **fallback 是完整的第三条路径**，
   不依赖任何被污染的中途状态 ✓（与 P3B 洞 ① 的区别就在这里）。
3. **交叉污染检查**：段 C 写 `mk`/`kv`；段 D 写 `o`/`xq`/`xsc`。段 D 的 `xq/xsc` 写**发生在**
   段 C 之后（seed → q → kv → attn → o），且段 C 的两个输入（`mk`/`kv`）**不是** `xq/xsc`
   ⇒ 两折的缓冲集合**不相交** ✓。

---

## 6. 验证手册（双门禁）

**双门禁（`lesion-audit §8` 的纪律，本轮必须照办）**：每个臂同时报告
`[dspark] steps=` 的 `draft=` 字段 **AND** `mean-k`（A0 基线 **1.34**）。
`draft_ms` 降 **且** `mean-k` 不掉（≥1.34）才算过；`mean-k` 掉 = 数值回归，**立即弃用该门**。

### 6.0 本机（无 GPU）可做的全部验证

```
1. cargo check --workspace --all-targets            # EXIT=0（本次已跑，见下）
2. 远端 nvcc compile-only（.cu 语法 + 寄存器/smem 报告）
3. 符号三证：nm -D $SO | grep -c dsv41_sparse_attn_orope / dsv41_rmsnorm_rope
   + .so 与 Rust 同时重编（段 D 改了 merge 核的**形参表**！见下 ⚠️）
```

> ⚠️ **ABI 警告**：给 `sparse_attn_merge_kernel` 加形参改的是**设备端 kernel 的形参表**
> （内部符号，不是 `extern "C"` 入口），但 **`.so` 必须与 Rust 侧同轮重建**；
> `dsv41_sparse_attn` / `dsv41_sparse_attn_orope` 的 **C ABI 不变** ⇒ Rust 侧无需改签名，
> 只需保证用的是同一个 `.so`。

### 6.1 parity（先于 serve A/B，R4）

| 测试 | 对比 | 判据 |
|---|---|---|
| `p3lite_seed_parity` | `rmsnorm_rope(n=1)` vs `ferrite_rmsnorm` + `apply_rope` | **逐位**（f32 bits 相同） |
| `p3lite_kv_parity` | `rmsnorm_rope(n=5)` vs `rmsnorm`(n=5) + `apply_rope`(rows=5,step=1) | **逐位** |
| `p3lite_orope_parity` | `sparse_attn_orope(m=5,row_step=1)` vs `sparse_attn` + `apply_rope×5` + `quant_fp8` | `o` 逐位 + `xq/xsc` 字节逐位 |
| `p3lite_merge_rowstep_parity` | merge 核 `row_step=0` vs **HEAD 版 merge 核** | **逐位**（回归门） |
| 边界 | `m=1`（退化）/ `b*m=8`（`kAttnMaxBM` 边界）/ `d=512`（smem 上限）/ `rope_rd=512` | 不崩 + 逐位 |

### 6.2 serve A/B 序列（一臂一进程，200 tok）

```
基线（全 OFF）        → draft=?, mean-k=1.34        # 同会话背靠背
+ ATTN_OROPE         → draft 应降 ~0.18-0.25ms(推), mean-k ≥ 1.34   ← 段 D 单变量
+ KV_NORM_ROPE       → draft 应降 ~0.06ms(推),     mean-k ≥ 1.34   ← 段 C 单变量
+ SEED_NORM_ROPE     → 同上
+ 三者全开            → 票面 −0.5~0.75ms(推)
```
**每条都必须看五段全文**（不只 k_acc 数字）：`lesion-audit §7` 的教训是"断崖漂移"
（line 62 → 52）比 mean-k 更能暴露数值问题。

### 6.3 若上段 B（K2）

```
基线 ALIGN_ONLY（DSV41_ATTN_PROJ_ALIGN=1）   → 记录 draft / mean-k / 全文
+ K2（P3LITE_Q_ROPENORM=1）                 → 必须与 ALIGN_ONLY 同轮对比
```
⚠️ **绝不可把"段 B 单独开"的结果当成 K2 的收益**（R1：它同时换了两条链的 contraction 程序）。

---

## 7. 未决 / 需上裁

1. **6-9 与"全逐位"的取舍**（§2.6）：6-9 必须启用 B6（`gemm_fp8_mrows_f32`，二分判非逐位）。
   建议：**先交付 11 发（全逐位）实测**，用实测决定是否值得为 −3 发承担 accept 风险。
2. **段 B 与 `ATTN_PROJ_ALIGN` 的合并主张**：段 B 的 −7 发与"把 draft 的 4 个 attention 投影
   统一到 verify 的程序"是**同一个正确性主张**。需上裁：是否接受把 `ATTN_PROJ_ALIGN` 从
   "程序对齐 A/B 臂"升格为"P3-lite 的组成部分"（这将改变两个门的 A/B 编排）。
3. **段 A 需要 `device.rs`**（新核 launcher + 符号探测），该文件当前在 peer 手上 ⇒ 需要一个
   合并窗口或由 peer 代接线。
4. **段 D 与 peer 的同族核改动**：peer（mrows-mpar）正在改 `sparse_attn_split/merge/orope`
   （`row_pitch` 一族）。本报告的 `row_step` 修正与 `row_pitch` 是**正交的两个形参**，
   但落在同一个函数签名上 ⇒ **合并时以"两者都保留"为准**（`row_pitch` 管 `q`/`out` 的行距，
   `row_step` 管 rope 的**位置步长**）。

---

*工部 · 基于 2026-09-13 仓库工作树（`1d8b953` + peer 未提交改动）。*
*launch 计数与 `file:line` 为代码精确值；ms 为推算并标「(推)」；本机无 GPU，未经 e2e。*
