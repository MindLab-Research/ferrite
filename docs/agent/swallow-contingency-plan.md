# SWALLOW 成功 / 失败 contingency 计划（engram slot 修复后）

> 工部 · 2026-09-12 · **只读分析 + 本文件（唯一产出）**。未执行 GPU 命令、未改动任何源码。
> **触发前提**：engram gather slot 修复已提交（HEAD `249c1b2`，代码在 `1877eb6`；slot
> `122880 → 147456 B`，两个 `Collective::new` 生产点都补了 engram 项）。
> 输入（现场核对）：`crates/ferrite-dsv41/src/serve.rs`、`crates/ferrite-dsv41/src/bin/dsv41-run.rs`、
> `crates/ferrite-models/src/dsv41/{tp.rs,chain_dev.rs,config.rs}`、
> `crates/ferrite-models/configs/dsv41_flash.json`、`scripts/{batched_400_v2.sh,sh_pair_ab.sh}`、
> `ab_shmrows.sh`、账本 `swallow-unlocked-next-plan` / `swallow-1token-fix-design` /
> `oob-fix-result-analysis-framework` / `dspark-correctness-chain` / `sh-pair-template-m-design` /
> `lazy-batched-gate` / `final-400-battle` / `batched-400-v2-remaining-roi`。
> 本文件职责：**给出「成功怎么接」「失败怎么查」两套可执行路径 + 400 的可达判定**。

---

## 0. 一页纸判决（先读这里）

**0.1 这次测试的主假设（必须先说明白）**

前四轮把「epoch 54 / 静默损坏」追到了 **一次 OOB 写清零 staging**（canary `0xdeadbeef → 0`，
`epoch 999 → 54` 重数，全 rank 均匀）。`check_payload` 抓到的那笔载荷
`147456 B > slot 122880 B` 是 **engram 多行 gather**（`engram_apply_rows`，
`VERIFY_ROWS × n_cols × engram_head_dim = 6×24×256 = 36864 f32`）——**合法需求**，
不是宽度 bug。修复 = 把 slot 抬到 147456 B。

⇒ **本次测试要验证的因果链是一条**：
`slot 不足 → OOB 写 → (a) staging 被清零 (b) AR 结果损坏 → EOS 提前 / 只出 1 token`。
若这条链成立，**两个症状应该同时消失**。

**0.2 关键提醒：旧 batched 数据全部不可信，lazy 91.1 可信**

- 该载荷**一直在**：只要 verify 块是 m=6（`VERIFY_ROWS=6`，编译期常量），
  `engram_apply_rows` 每轮都发 `m*n_cols*ehd`。**所有历史 batched 跑都带这个越界**。
- lazy 路径 `m=1` ⇒ 载荷 `n_cols*ehd = 6144 f32 = 24576 B` ⇒ **不越界**。
  ⇒ **干净栈的 91.1 tok/s（lazy）是干净的基线，可以继续当锚；历史 batched 的性能/accept/文本
  数据一律作废，不得引用。**

**0.3 两个症状 → 三个判定出口**

| 出口 | 签名 | 处置 |
|---|---|---|
| **✅ 绿** | 无 panic + `LEN > 60` + ledger 有 `arm=swallowed` note 且 `k_emit ≥ 1` + CANARY/GUARD=0 + 文本零拉丁 | 走 **§3 成功路径** |
| **🟡 黄** | 生成正常但**变慢/内容坏** | 走 **§4.4 / §4.5** |
| **🔴 红** | 新 panic（payload 又超） / 仍 1-token / hang | 走 **§4.1 / §4.2 / §4.3** |

**0.4 400 的判定（一句话）**
400 只在 **counting 口径**（accept≈5 ⇒ 6 tok/step）下可达，且要求
`SWALLOW + mrows 族 + SH_PAIR + tcgen05 + B6` **四项同时足额**（步时 ≤14.5ms）。
按历史 60% 兑现率，现实落点是 **~18–20ms ⇒ 300–340 tok/s**。详见 §5。

---

## 1. 事实基础（code-verified，本文件的地基）

### 1.1 修复本身

```rust
// serve.rs:394-414 / dsv41-run.rs:354-367（两个生产点，同一形状）
let eng_cols = cfg.engram_max_ngram_size.saturating_sub(1) * cfg.engram_n_heads;  // (4-1)*8 = 24
let eng_rows = VERIFY_ROWS * eng_cols * cfg.engram_head_dim;                        // 6*24*256 = 36864
let ar_bytes = (hc_dim.max(VERIFY_ROWS * cfg.dim).max(eng_rows)) * 4;               // 36864*4 = 147456
```

- `check_payload`（`tp.rs:596-603`）是 **`assert!`（panic，不是 `Err`）**：`len <= self.bytes`。
- 与调用点的一致性：`engram_apply_rows`（`chain_dev.rs:12347`）里
  `n_cols = cfg.engram_max_ngram_size.saturating_sub(1) * cfg.engram_n_heads`、
  `ehd = cfg.engram_head_dim`，AR 长度 `fb(m * n_cols * ehd)`（`:12397-12400`）。
  **两处推导逐字相同 ⇒ 修复随 config 自动跟随**（换 `engram_max_ngram_size`/`n_heads` 不会漏）。
- 配置实测（`dsv41_flash.json` text_config）：`hidden_size=5120`、`hc_mult=4` ⇒
  归一化 `dim=5120`、`hc_dim=20480`；`engram_max_ngram_size=4`、`engram_n_heads=8`、
  `engram_head_dim=256`、`num_hidden_layers=40`。

### 1.2 ⚠️ 全量 AR 载荷审计（§4.1 的地基——我亲自枚举了所有调用点）

`grep -n "all_reduce" chain_dev.rs` 得到 **11 个调用点**，逐点算载荷：

| 行 | 表达式 | f32 | bytes | ≤ 147456？ |
|---|---|---:|---:|:---:|
| `:5160` | `n_cols*ehd`（单行 engram） | 6144 | 24 576 | ✓ |
| `:5398` | `dim` | 5120 | 20 480 | ✓ |
| `:11592` | `m*dim`（attn out, mrows） | 30720 | 122 880 | ✓ |
| **`:12397`** | **`m*n_cols*ehd`（engram rows）** | **36864** | **147 456** | **= 边界（0 余量）** |
| `:13431` | `mdim = m*dim`（MoE out） | 30720 | 122 880 | ✓ |
| `:15411/:15425` | `dim`（head/argmax 侧） | 5120 | 20 480 | ✓ |
| `:5270/:5283`（hcpost） | `len == hc_h == dim` | 5120 | 20 480 | ✓ |
| `:5336`（hcpost_rows） | `len = m*dim`，门 `len/4 == rows*hc_h` | 30720 | 122 880 | ✓ |
| `:5387`（add 融合） | `dim` | 5120 | 20 480 | ✓ |

**结论**：`147456` 是**全树最大载荷**，且**恰好等于新 slot** ⇒
1. 修复后不会再有第二个「已知」越界；
2. **零余量**是本次修复的固有特征（见 §1.3-R2）。

### 1.3 修复的副作用（成功路径也要盯的三件事）

| # | 副作用 | 判定 |
|---|---|---|
| **R1** | `ctr_at` 随 `bytes` 移动（`tp.rs:377-384`：`stamps_at=2*world*bytes` → `reduced_at` → `ctr_at=reduced_at+world*4+8`）。**全是派生量，无硬编码偏移** ⇒ canary/guard 语义不变、`V5_LEDGER_CANARY_OFFS=[8,16,32,48]` 仍落在 `ctr_at` 的 64 B 尾里 | ✅ 低风险；但**canary 字节偏移绝对值变了**——任何「手动 dump staging 某地址」的旧脚本要重算 |
| **R2** | **零余量**：`147456 == 147456`。任何把 m 从 6 推大的改动、或非 5120/24/256 的 config、或**新增一条比 engram 更宽的 m-row AR**，都会立刻重新 panic | ⚠️ **已知暴露面**（见 §4.1） |
| **R3** | 每 rank staging 从 `2*8*122880 ≈ 1.97 MB` → `2*8*147456 ≈ 2.36 MB`（+20%）。8 rank 合计 +0.4 MB | ✅ 可忽略 |

### 1.4 pad 两臂（成功路径里第一个「免费收益」候选）

- `DSV41_SWALLOW_EPOCH_PAD`（D2 常量 pad）：吞掉 `step_dev` 后**每轮补发固定 `2*n_layers+1 = 81` 个空 round**
  （`chain_dev.rs:1887-1936`、`:8726-8734`）。
- `DSV41_SWALLOW_DYNAMIC_PAD`（11-B 共识）：每步 `1×D2H + 1×RankMax rendezvous`，**只给落后的 rank 补**
  （`:9842-9857`；注释自称「tens of µs against a ~6ms step」）。
- **两者都是为 OOB 造成的 desync 打的补丁**。OOB 修好后它们可能**不再必要**——这是一个
  零代码、一次 A/B 就能拿回来的收益（§3.2）。⚠️ 注意 `batched_400_v2.sh:154` 把
  `DSV41_SWALLOW_EPOCH_PAD=1` 摆在矩阵里且注释「NOT optional」——**那条注释写在 OOB 修复之前，
  必须重新验证，不能当教条**。

### 1.5 判据纪律（沿用仓内既有约定，违反则整张表作废）

1. **三个计时器不可混用**：serve 墙钟（`step pos=N: Y ms`，用户两次判不可信）/
   `[dspark] steps=… verify=`（含 host barrier+D2H）/ **nsys per-kernel GPU 时间（唯一纯模型时间）**。
2. **每条 ms 必须标 臂 + m**（lazy=m1 / batched=m6），否则等于没测。
3. **吞吐测量必须 `DSV41_V5_LEDGER=0`**（ledger 每步多一次 D2H；`batched_400_v2.sh:161-168` 已有该开关）。
4. **k_acc 逐位对照只在同 prompt 同 seed 下有效**；换 prompt 比 k_acc = 无效判据。
5. **FORBIDDEN（会静默换路径）**：`DSV41_LAZY_VERIFY`、`DSV41_HC_VERIFY_FUSE`、`DSV41_HC_FRONT_ROWS`
   与 `DSV41_SWALLOW_STEP` 不可同时 armed（`batched_400_v2.sh:175-177`；lazy 的 v5 足迹
   `3+81*k_emit` 与常量 pad **不可能**对齐）。

---

## 2. 判据：什么叫「SWALLOW 成功」

**五条硬门（缺一不算），后两条是「解锁 batched」的门。**

| 门 | 判据 | 取证方式 | 失败则走 |
|---|---|---|---|
| **G0** | **无 panic**：日志无 `collective payload … > slot …` | `grep -n "collective payload" $LOG` 必须空 | §4.1 |
| **G1** | **正常生成**：`LEN > 60` ∧ `finish_reason ∈ {stop, length}` 且 length 时 `LEN ≈ max_new` | `/v1/chat/completions` 响应 + `[dsv41]` 行 | §4.2 |
| **G2** | **臂真的跑了**：ledger 出现 `arm=swallowed` 的 **note** 且 `k_emit ≥ 1`（`[v5-ledger-note]` / `[dsv41] emitted=`） | ledger 行（**只在取证轮开** `DSV41_V5_LEDGER=1`） | §4.2 |
| **G3** | **观测干净**：`CANARY=0 ∧ GUARD=0 ∧ ar5-hang=0` | `grep -c "v5-ledger-CANARY"` / `…-GUARD` / `ar5-hang` | §4.3 |
| **G4** | **内容正确**：零拉丁 + 无复读 + 语义连贯（出师表/计数各一段） | 人读 + 字符集统计 | §4.4 |
| **G5** | **抗竞态**：**3 次独立运行 3/3 无 hang**，其中一次长跑跨过历史 hang 出现的步数 | 3× run | §4.3 |
| **G6** | **accept 不变**：同 prompt 同 seed 下 k_acc 序列与 lazy 逐位相同 | 出师表 lazy 序列 `4 0 0 0 3 0 1 1 0 0 0 1 2 0 0 5 0 0 0 1` | §4.4 |

> ⚠️ **第一条纪律**：`LEN=1` **不是**「生成 1 个 token」，是**fault 签名**
> （`single_flight.rs:157-198` 的 `fail()` 直接 retire + `driver.rs:337-350` 的 `Length`）。
> 报结果时必须 **LEN + finish_reason + `spec step err` / `did not answer` 三样一起报**，
> 只报 `LEN=1` 一律退回重测（`swallow-1token-fix-design §5` 末）。

---

## 3. 成功路径（G0–G6 全绿之后）

### 3.1 P0 —— 基线测量：SWALLOW(+pad) vs lazy 91.1

**目的**：把 S0→S1 那一格从「设计口径」变成「实测」。

| 臂 | env（除 base 栈外） | 说明 |
|---|---|---|
| A（对照） | `DSV41_LAZY_VERIFY=1`（**不要**带 SWALLOW_STEP） | 复现 91.1 |
| B | `DSV41_SWALLOW_STEP=1 DSV41_SWALLOW_DYNAMIC_PAD=1` | 本次主臂 |
| C | `DSV41_SWALLOW_STEP=1 DSV41_SWALLOW_EPOCH_PAD=1` | 若 B 有 hang 才启用 |

- base 栈：`DSV41_SPEC=1 DSV41_DSPARK=1 DSV41_SIDS_WRITEBACK=1 DSV41_EXPERT_ACT_E4M3=1 DSV41_BF16_TRUNCATE=1 DSV41_VERIFY_GRAPH=1 DSV41_TIMING=1 DSV41_DSPARK_DEBUG=1`
  （照 `batched_400_v2.sh`，**去掉** `V5_LEDGER`）。
- **按任务分桶测**（三档 accept 决定谁赢）：
  | 任务 | 实测 accept | 稳定臂 | 阈值（`lazy iff (1+mean_k) < B/c`，B≈28ms、c=6.15 ⇒ 3.55） |
  |---|---|---|---|
  | 计数 | 5.0 | **batched** | 6 tok/step |
  | 出师表 | 1.21 | **lazy** | 2.21 tok/step |
  | 对话 | 0.96 | **lazy** | 1.96 tok/step |
- **必测两组数**：① nsys per-kernel（口径唯一可信）② `[dspark] steps=… verify=`；
  serve 墙钟只作旁证。
- **预期**：SWALLOW 净 **−4.55ms**（主链 −6.15 + anchor 行 +1.6，`chain_dev.rs:6000-6015`）。
  若起点按内部口径 31–33ms ⇒ **26.5–28.5ms**。
- **止损**：位移 < −2ms ⇒ 先查 anchor 行的 +1.6ms 是否被重复计；位移为正 ⇒ 立即停止叠加
  后面的 gates，先归因（本仓 #1 陷阱：一次叠多个因子）。

### 3.2 P0.5 —— pad 必要性 A/B（**本次最便宜的一格，别跳过**）

OOB 修好后，两个 pad 都是「防 OOB 造成 desync」的补丁。**逐个撤掉测**：

| 子项 | A/B | 判据 | 预期 |
|---|---|---|---|
| **P0.5-a** | `SWALLOW_STEP=1`，**无任何 pad** ⇒ vs 带 `DYNAMIC_PAD` | G0/G3/G5 仍绿？步时？ | 若绿：**DYNAMIC_PAD 的 `1×D2H + 1×rendezvous/步` 可以退掉**（注：D2H 是**同步点**，真实代价可能远大于注释自称的 tens of µs） |
| **P0.5-b** | `SWALLOW_STEP=1 + EPOCH_PAD=1` ⇒ vs 无敌 pad | 同上 | 常量 pad **每轮补 81 个空 round**，代价远大于 a ⇒ 若 a 绿则 b 直接淘汰 |

**⚠️ 这一格的产出直接改 §3.7 的 400 账**：如果 pad 能退，等于白拿回一笔（且把
`batched_400_v2.sh` 里「PAD is NOT optional」那条过期注释改掉）。

### 3.3 P1 —— m=6 mrows 族逐个 A/B（零代码）

**前提**：batched 常开（P0 已定）。**一次只上一个 gate**，每个都验 **G6（k_acc 逐位不变）**。

| 顺序 | gate | 位置 | 预期 | 备注 |
|---|---|---|---:|---|
| 1 | `DSV41_GATE_MROWS`（= `DSV41_ROW_FOLD_GATE`） | `row_fold_gate()` `:1282-1305` | −2.75ms | 设计口径，未单测 |
| 2 | `DSV41_INDEXER_MROWS` | gate `:1249` | −1.0~1.5ms | 代码已就位 |
| 3 | `DSV41_VERIFY_ROPE_MROWS` | gate `:1210` | −0.53ms | launch 账 |
| 4 | **`DSV41_VERIFY_HEAD_MROWS`** | gate `:1621`；调用点 `:5935-6007` | −0.7~0.9ms | **历史 ar5-hang 的那个组合 ⇒ 最后单独上，单独成轮** |

- **为什么这批只在 batched 存在**：它们全是「m 行折成 1 发」，lazy `m=1` ⇒ 折核退化成自身 ⇒ 零节省。
  旁证：`{SH_EXP+GRAPH+ROPE+P3A}` 全开在 lazy 下只有 −1.21ms（预期 −24）。
- **止损门**：任一 gate 位移 < 预期的 40% ⇒ **停 P1，转 P2**（ms 账更大）；**不要**在同一轮里
  叠 gate 找感觉（`SH_EXP_MROWS` 两次零收益是同一机理的先例）。

### 3.4 P2 —— SH_PAIR M=6（ms 账最大单项，但要先修 parity）

**先做一件零 GPU 成本的事**：把 parity 失败清单的完整 case 表打出来
（`kernels/cuda/tests_dsv41_sh_exp_mrows.cu:318-386` 五支判据），按
`(fold_r, n1, k1, n2, limit, epi_add, with_act)` 分类。已报只有 4/36 条且只覆盖 (a)(d) 两支。

**两条已知指纹（都不是数值噪声）**：
- **F1 只差符号位**（`0xf9 vs 0x79`、`0x00 vs 0x80`，异或恒 `0x80`）⇒ 归约顺序**改不了符号位** ⇒
  先前「amax 树归约顺序」的假设很可能是错的；用「打印 aq 前 8 字节两臂 + `aqsc` r=0」区分
  量化器符号路径 / 行基址 stride bug。
- **F2 整缓冲 sentinel**（`sent == m·n1` 恰好）⇒ kernel 对这条形状 **decline / 没跑**，不是漏写行。

**A/B 四臂**：`SH_PAIR_M_FOLD ∈ {1,2,6}` + `SH_PAIR_M=1` 开关；输赢用
**raw f32 bits memcmp（非容差）** 做 parity 硬门。
**预期**：−4.9~7.9ms（设计口径，来自 shared expert 10.4ms 的族账）。
**止损**：诊断树走完仍无单一根因 ⇒ 冻结 `SH_PAIR_M` 默认 OFF，把这 −5ms 从阶梯划掉
（阶梯落点掉一格到 ~19ms）。

### 3.5 P3 —— tcgen05 重测（**不是 −6.8ms，是 −1.0~3.8ms**）

次序：符号预检 → 冒烟 → 门税对照 → grouped。`≥3ms` 改善算足额，**中性即止损**。
红线：`tc5::e4x` 从未上过 GPU、两个 `[OPEN]`；冒烟出拉丁/非法指令 ⇒ 立即关路径，
**不投变体矩阵**（勿重演 v17→v21 四变体全中性）。

### 3.6 P4 —— B6 `dsv41_gemm_fp8_mrows_f32`（B 类公共祖先）

判据：「m 行核第 r 行 == M=1 f32 GEMV 第 r 行」逐位。**B6 单项 −0.66~1.5ms**
（−200~240 发/步的第一性计数）；**B1–B6 全族 −2.8~4.9ms / 8~12 人日**。
⚠️ 不要把全族的数挂在 B6 单项头上（旧账的错误来源之一）。

### 3.7 400 组合（S3–S5 的验收跑）

**必须一次跑「全开组合」**，但**归因仍靠前面的单因子轮**：

```
counting 任务（accept≈5 ⇒ 6 tok/step）
DSV41_SPEC=1 DSV41_DSPARK=1 DSV41_SIDS_WRITEBACK=1 DSV41_EXPERT_ACT_E4M3=1
DSV41_BF16_TRUNCATE=1 DSV41_VERIFY_GRAPH=1
DSV41_SWALLOW_STEP=1                       （pad 按 P0.5 结论决定）
DSV41_GATE_MROWS=1 DSV41_INDEXER_MROWS=1 DSV41_VERIFY_ROPE_MROWS=1
DSV41_VERIFY_HEAD_MROWS=1                  （最后一次加）
DSV41_SH_PAIR_M=1 [DSV41_SH_PAIR_M_FOLD=?] （仅在 parity 100% 后）
[DSV41_EXPERT_TCGEN05_E4M3=1 DSV41_EXPERT_GROUPED=1 DSV41_GATEUP_FUSE=0 DSV41_EXPERT_ILV=0]
DSV41_TIMING=1                             （不加 V5_LEDGER）
```

**验收**：一次跑同时满足 **G0–G6 + 步时 ≤ 14.5ms**（= 400 tok/s @ 6 tok/step）。

---

## 4. 失败路径 contingency

### 4.1 F1 —— 新 panic（`check_payload` 又抓到越界）

**症状**：日志出现 `collective payload N > slot 147456`（全 rank）。

**诊断（15 分钟内可判）**：

```bash
grep -n "collective payload" "$LOG" | head        # 拿到 N
# 用 §1.2 的表把 N 对上号；N/4 = 载荷 f32 数
python3 -c "print($N/4)"                          # 36864=engram, 30720=m*dim, 5120=dim, 6144=单行engram
```

**判定规则（关键：区分「合法需求」与「调用点 bug」）**：

| 情形 | 判据 | 处置 |
|---|---|---|
| **N 是某个已知 m-row 载荷的整数倍放大**（如 `7×24×256`、`m=7` 的 `m*dim`） | N / (n_cols*ehd) 或 N/dim 是**整数且 > VERIFY_ROWS** | **先查是谁把 m 推过 6**：`VERIFY_ROWS` 是**编译期常量**（`chain_dev.rs:84`，且有 `const_assert!(VERIFY_ROWS == DSPARK_DRAFTS+1)`），所以 m>6 只能是**新的 m-row 载荷**或 **config 变更**。→ 合法需求**继续抬 slot**（并同步 `dsv41-run.rs` 第二个构造点） |
| **N 与 config 推导不符** | 用 cfg 重算 `VERIFY_ROWS*eng_cols*ehd*4` ≠ 147456 | **config/推导不一致 = 修复漏了参数** → 修 `eng_cols` 表达式，**不要**抬 slot |
| **N 是非 m×dim 形状**（整除不了 dim/ehd） | — | **调用点 bug**（striding/长度算错）→ 修调用点，**不要**抬 slot |

**兜底**：若判定不了，先 `compute-sanitizer --tool memcheck` 抓真实越界地址，
再回上表。**禁止**「无脑把 slot 抬到 1MB」——那会把 §1.3-R2 的暴露面变成静默区。

**二次防线（已知会在 OOB 时响的东西）**：guard（`ctr_at-8`）/ 4 个 canary（`ctr_at+8/16/32/48`）。
注意**加大 slot 后 canary 的绝对地址变了**（§1.3-R1）——若你手上有按旧地址 dump 的脚本，
先重算地址再判读。

### 4.2 F2 —— 仍然 1-token（`finish_reason=length`）

**这一支大概率已被 engram 修复消掉**，但必须按「它有第二根因」准备，因为
`a418f8c` 那轮**同时**报了「OOB 四项全过」与「LEN=1」——两者可能同源、也可能并行。

**第 0 步（一步分流，只读日志）**：

```bash
grep -n "collective payload" "$LOG"                # 先排除 F1
grep -n "spec step err at pos" "$LOG"              # A：arm 返回 Err（serve.rs:611-614）
grep -n "POISONING the whole request" "$LOG"       # A 的伴生行
grep -n "a rank did not answer" "$LOG"             # B：hang→超时（serve.rs:255-258）
grep -nE "pos=15" "$LOG" | head                    # 两条 pos=15 之间 ≥1800s ⇒ B
```

| 命中 | 根因 | 处置 |
|---|---|---|
| `spec step err at pos 15` | **A：第一次 swallowed 轮返回 `Err`** | 按错误文本对号入座走 **E1–E6**（见下表）；**先看 E3** |
| 只有 `did not answer` + 时间缺口 | **B：hang（非 `[ar5-hang]` 型）** | 先上 **B1/B2/B3** 取证（缩短超时 + 带 pos 的报错 + 四相位 trace），再按 hang 点修 |
| 两者都无 | 观测不足 | 先补 §4.2 的 exit-tag 打印（`dspark_spec_step` 的 swallowed 分支 `.map_err(|e| eprintln!("[swallow-ERR] pos={pos} err={e}"))?`），重跑 |

**E1–E6（A 类子表，按「最可能先命中」排序；全部来自 `swallow-1token-fix-design §2.A`）**：

| 子项 | `Err` 入口 | 触发条件 | 修复成本 |
|---|---|---|---|
| **E1** | `inv_ids`（`:8854`） | `DSV41_INV_CHECK=1` ∧ `SIDS_WRITEBACK` 关 ⇒ swallowed 臂 `s.ids` 未回写 ⇒ `seen != emitted.last()` 必失败 | **极小**（把两门同生共死，~5 行） |
| **E2** | `step_rows`（`:8761`） | m=6 的 `m > VERIFY_ROWS` / graph capture-replay slot 分支错 | **中**（`VERIFY_ROWS` 已=6 则 0 成本，仅补断言） |
| **E3** | `dspark_commit`（`:8795`）→ `compress_replay` | **`k_emit ≥ 1` 恒成立而 legacy 的 `k_acc` 可为 0** ⇒ swallowed **每轮**走 replay；`pos_base` 语义（注释写 "`pos+1`"，swallowed 传 `pos`！）**最可疑** | 小~中（改 1 个实参 / 补语义分支） |
| **E4** | `carry_kept_tap(…, k_emit)`（`:8798`） | `k_emit > VERIFY_ROWS` 越界（release 下 `debug_assert` 不拦） | **极小**（`clamp(1..=VERIFY_ROWS)` + release `assert`） |
| **E5** | `note_ctx_rows`/`import_tap`/`draft_forward`（`:8742/8747/8796`） | tap 靠 carry 而非 `step_dev`（新时序）⇒ 首轮 tap 可能半写 | **中**（两臂同字段 trace diff） |
| **E6** | `spec_accept` 返回 0（`:8782`） | `k_emit=0` ⇒ `k_acc = k_emit-1` **usize 下溢 panic** | **极小**（`k_emit.max(1)` + 边界断言） |

**验收（A/B 类共同）**：ledger 出现 `arm=swallowed` 的 **note** 且 `k_emit ≥ 1`；
`LEN > 60`；`finish_reason ∈ {stop, length(max_new)}`。**不许只报 LEN=1。**

### 4.3 F3 —— ar5-hang 回归 / 卡死

**症状**：`[ar5-hang]` 行出现，或一步卡住 → 1800s 超时 → `did not answer`。

**为什么「一次 0 hang」不算修好**：历史 hang 概率序列是 `1-2 → 22 → 3`（竞态）⇒
**必须 3 次独立运行 + 一次长跑**（跨过历史 hang 出现的步数）才叫修好。

**增大 slot 后新增的怀疑面**（本次特有）：
1. `ctr_at`/`stamps_at`/`reduced_at` 的**绝对地址变了**（§1.3-R1）。v5 store 的
   `stride = bytes/4`（`tp.rs:676`）也随之变 ⇒ **任何对 `stride` 有隐含假设的 kernel/脚本要复核**。
2. `dev.zero_at(staging.ptr, ctr_at + 64)`（`tp.rs:400`）覆盖范围变大 ⇒ 初始化时间略增（可忽略），
   但**若某处按旧 `ctr_at` 计算过偏移，就会清零错位置**——这正是 11 号修复前的 bug 形状。

**处置顺序**：
1. 立刻 `DSV41_SWALLOW_STEP=0` 回退（不要在 hang 未复现前叠 P1 的 gate，否则失去归因）。
2. 复现时抓**最后一个 phase**（B3 的四相位 trace：snapshot / draft / verify / commit）。
3. 记录为已知缺陷；**`VERIFY_HEAD_MROWS` 单独成轮，绝不与其它项混**。

### 4.4 F4 —— 生成正常（G1 通过）但内容坏

**症状**：拉丁字符 / 复读 / EOS 提前 / 计数任务数字自锁。

**已知机理（按概率）**：
1. **EOS 提前**：前四轮的 EOS 提前被归因于「损坏的 AR 结果 → 错误 attention → 错误 logits」。
   若已修但仍有 ⇒ 查 `spec_accept` / `carry` 的 k_emit 与 `emitted` 是否同源（E6）。
2. **拉丁/乱码**：仓内已有结论——`opa`/`anao` 是**词表里的合法 token**（id 41291 / 83514），
   ⇒ **模型真的输出了它们**，不是显示层。⇒ 是**数值路径**问题（bf16 truncate / 激活精度），
   不是采样器。处置：关 `DSV41_BF16_TRUNCATE` / `DSV41_TAP_BF16` 做对照（两臂四次文本）。
3. **计数自锁**：`inv_ids`（`s.ids != emitted.last()`）的经典签名 ⇒ 走 E1。

**判据**：lazy vs batched **同 prompt 同 seed** 的 k_acc 逐位对照（G6）；两次文本对照
（batched 应至少不差于 lazy）。

### 4.5 F5 —— 生成正常但没变快（或变慢）

| 可能 | 判据 | 处置 |
|---|---|---|
| 臂没真的切到 batched | `grep -E "verify_graph.*_m1|_m6"`（`VERIFY_GRAPH_SLOTS=2` 两槽，**两行都可能出现**） | 按请求分桶，不要假设整场只有一个臂 |
| 计时器被骗 | 三方对照（nsys / `[dspark] steps=` / 墙钟） | 以 nsys 为准 |
| 叠加的 gate 全中性 | 每个 gate 单独 A/B 的位移 < 40% | 停 P1，转 P2（ms 账更大） |
| accept 太低（对话/出师表） | `mean_k < 3.55` | **不是 bug**：该任务本就该走 lazy；400 只在 counting 口径成立 |

---

## 5. 400 的最终判定

**5.1 步时阶梯（口径已统一；起点用 P0 实测重钉）**

| # | 阶段 | 增量 | 累计步时 | 依据强度 |
|---|---|---:|---:|---|
| **S0** | Wave 1（lazy 现状） | — | **31–33ms**（内部口径；外部墙钟 25.17ms 不可信） | ⚠️ P0 重钉 |
| **S1** | + SWALLOW | −4.55 | **26.5–28.5ms** | 设计口径，**本次测试验** |
| **S2** | + mrows 族（m=6） | −4.5~5.8 | **21–24ms** | 设计口径（族级旁证偏负面：−1.21 vs 预期 −24） |
| **S3** | + SH_PAIR `template<M>` | −4.9~7.9 | **13–19ms** | 设计口径；**当前 parity 36 failed** |
| **S4** | + tcgen05（gate/up only） | −1.0~3.8 | **9.2–18ms** | 修正口径（**非 −6.8ms**：down 无 tcgen05 核） |
| **S5** | + B6 | −0.66~1.5 | **7.7–17.3ms** | 第一性计数（−200~240 发/步） |
| S6 | + L4 | −5~8 | ~4–13ms | **仓内零实测背书**，16~21 人日（属越过 400 之后） |

**5.2 兑现率折算（counting，6 tok/step）**

| 兑现率 | S2 | S3 | S4 | S5 | 落点 |
|---|---:|---:|---:|---:|---|
| 100%（设计口径） | 22 | 15 | 13.5 | 12.5 | 400 ✓ |
| **60%（历史兑现率）** | 24 | 20 | 18.5 | 17.8 | **337 tok/s**（差 16%） |
| 40% | 25.5 | 23 | 22 | 21.5 | 279 tok/s |

**5.3 400 可达的充要条件（判定表）**

| 条件 | 门槛 | 当前状态 |
|---|---|---|
| ① 口径 | **counting**（accept≈5 ⇒ 6 tok/step） | 出师表/对话**物理不可达**（需 ≤5.5/5ms，低于 L5 地板 8–9ms） |
| ② SWALLOW 解锁 batched | G0–G6 全绿 | **本次测试验** |
| ③ 步时 | **≤14.5ms**（S4→S5 之间是临界点 15ms） | 需 S3/S4/S5 **同时足额** |
| ④ 最短可行集 | `SWALLOW + mrows + SH_PAIR + tcgen05 + B6`（P1+P2+P3+P4），**不含 L4/L5** | SH_PAIR parity 未修；tcgen05 未上过 GPU |
| ⑤ 副产品 | pad 能否退（P0.5） | 待测，可能白拿一笔 |

**5.4 判决**
1. **400 的现实票面是「步时 ~18–20ms、counting 300–340 tok/s」**；
   400 要求四项**同时足额**——仓史上前所未有（从未把设计口径全兑现）。
2. **纯步时路线的最短可行集不含 L4/L5**；L4 是唯一能把「中等 accept（2.5–3）」也拉进 400 的层，
   但 16–21 人日、零实测背书、v17→v21 全中性的历史警告「只动一个因子无效」。
3. **先做 P0.5（pad 必要性）再排后面**：它最便宜、且直接影响 ③ 的分母。

---

## 6. 执行顺序 & 门（一张图）

```
本次 SWALLOW 测（engram 修复后）
        │
        ├─ G0 失败（新 panic）───────────────▶ §4.1 F1：载荷对号 → 补 slot / 修调用点
        ├─ G1 失败（仍 1-token）─────────────▶ §4.2 F2：A/B 分流 → E1..E6 / B1..B4
        │      └─ G3/G5 失败（hang）─────────▶ §4.3 F3：回退 SWALLOW_STEP=0 + 相位 trace
        ├─ G4 失败（内容坏）─────────────────▶ §4.4 F4：EOS / 拉丁 / 自锁 三分支
        └─ G0–G6 全绿 ───────────────────────▶ §3 成功路径
                  │
                  ├─ P0   基线（SWALLOW+pad vs lazy 91.1，分任务桶）
                  ├─ P0.5 pad 必要性（**先做**，可能白拿一笔）
                  ├─ P0-c hang 抗性 3×run + 长跑
                  ├─ P1   mrows 族逐个 A/B（HEAD_MROWS 最后单独）
                  ├─ P2   SH_PAIR parity 修复 → M=6（并行支线）
                  ├─ P3   tcgen05（并行支线，红线即止损）
                  ├─ P4   B6
                  └─ P5   400 组合验收跑（counting，≤14.5ms）
```

**关键路径（唯一串行主干）**：`P0 → P0.5 → P0-c → P1 → P3 → 400 组合`。
SH_PAIR（P2）与 B6（P4）是**并行支线**，不占关键路径的 GPU 位。

---

## 7. 速查表

| 对象 | 位置 |
|---|---|
| engram slot 修复（两个生产点） | `crates/ferrite-dsv41/src/serve.rs:394-414`、`.../bin/dsv41-run.rs:354-367` |
| `check_payload`（assert，panic） | `crates/ferrite-models/src/dsv41/tp.rs:596-603` |
| staging 布局（`stamps_at`/`reduced_at`/`ctr_at`/guard/canary） | `tp.rs:306-440` |
| `engram_apply_rows`（载荷来源） | `chain_dev.rs:12347`，AR 在 `:12397-12400` |
| `VERIFY_ROWS`（编译期常量 + const_assert） | `chain_dev.rs:84-98` |
| `SWALLOW_STEP` 分派 / gate | `chain_dev.rs:8165-8187` / `:2697` |
| `swallowed` 臂本体 | `chain_dev.rs:8708-8866` |
| DYNAMIC_PAD / EPOCH_PAD | `:9842` / `:9766`、`:1913`、`:8726` |
| ledger 门 / 失败签名 | `serve.rs:611-614`（Err）、`:255-258`（did not answer）、`:110`（1800s 超时） |
| `Length` 的来路（fault 签名） | `ferrite-http/src/driver.rs:337-350`、`single_flight.rs:157-198` |
| mrows gates | `chain_dev.rs:1210`(rope) `:1249`(indexer) `:1305`(gate) `:1621`(head) |
| SH_PAIR 门 / 核 / parity 套件 | `chain_dev.rs:1470-1486`；`dsv41_kernels.cu:6933`；`kernels/cuda/tests_dsv41_sh_exp_mrows.cu:318-386` |
| 全矩阵 / SH_PAIR A/B 脚本 | `scripts/batched_400_v2.sh`、`scripts/sh_pair_ab.sh`、`ab_shmrows.sh`、`scripts/nsys_wave1.sh` |
| FORBIDDEN（与 SWALLOW 不共戴天） | `DSV41_LAZY_VERIFY`、`DSV41_HC_VERIFY_FUSE`、`DSV41_HC_FRONT_ROWS` |

---

*工部 · 只读分析 + 本文件（唯一产出）；未执行 GPU 命令、未改动任何源码。*
*所有 ms 数均标注来源与口径（launch 账 / ms 账 / 设计口径 / 实测）；对任务前提的修正已显式给出：*
*① 历史 batched 数据全部作废（载荷一直在）；② tcgen05 不是 −6.8ms；③ B6 单项不是全族数；*
*④ 「PAD is NOT optional」是 OOB 修复前的注释，必须重新验证。*
