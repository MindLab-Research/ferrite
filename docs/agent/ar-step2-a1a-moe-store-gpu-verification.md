# AR Step 2（A1a MoE store fold）的 GPU 验证设计

> 工部 · 2026-09-12 · **只读勘察 + 本文件（唯一产出）**。未改动任何源码，未执行任何 GPU 命令。
> 对象：`docs/agent/ar-step2-a1a-moe-store-implementation.md`（A1a：把 MoE AR 的 staging 拷贝
> 从独立 `p2p_ar_store_v5_kernel` 挪进 payload 最后写者的 epilogue）。
> 现场核对：`kernels/cuda/{ferrite_kernels.cu,dsv41_experts_mxf4.cu,build.sh}`、
> `crates/ferrite-models/src/dsv41/{device.rs,tp.rs,chain_dev.rs}`、
> `crates/ferrite-kernel/{build.rs,src/{cuda.rs,devrt.rs}}`、`crates/ferrite-dsv41/src/serve.rs`、
> `scripts/{dsv41_serve_ab.sh,sh_pair_ab.sh,l49_ab.sh,nsys_wave1.sh}`。
> **所有行号均对当前工作树现场核对。**

---

## 0. 前提与基线（先钉死，再谈三项）

### 0.1 A1a 的两个提交与"旧"基线

A1a 不是单个提交：

| 提交 | 内容 |
|---|---|
| `d88cb41` | `ferrite_kernels.cu`（+141：slot 助手 / `add_kernel` 尾参 / `ferrite_add_store` / `_pubred_v5_moe` / `_pubred_v5_hcpost`）、`dsv41_experts_mxf4.cu`（+56：`dsv41_moe_down_reduce_st`）、`device.rs`（+80：符号装载 + wrapper） |
| `4e21d2a` | `chain_dev.rs`（+223：`ar_store_fuse_moe` / `moe() -> Result<bool>` / `moe_reduce(carried)` / 折叠孪生）、`tp.rs`（+89）、`device.rs`（+156） |

⇒ **旧基线 = `3ea879f`**（`d88cb41` 的父提交，已现场核对：该 revision 的 `.cu` 里
`ferrite_add_store` / `p2p_ar_v5_slot_base` / `p2p_ar_pubred_v5_moe` / `dsv41_moe_down_reduce_st`
**计数全为 0**；`chain_dev.rs` 有 `ar_store_fuse()`（attn 侧）但**没有** `ar_store_fuse_moe`）。
即：旧树 = "attn 侧已在树内、MoE 侧尚缺"的准确状态。

### 0.2 双产物纪律（不可绕过）

- `.so` 的 build stamp = `git rev + sha256(.cu) + flags`（`build.sh:117-127`）；
  binary 在 `cargo build` 时由 `ferrite-kernel/build.rs` 把同一串烧进 `FERRITE_BUILD_ID`。
- 进程启动时 `cuda.rs:2088-2140` / `devrt.rs:292-320` 三重校验：**dlopen 的 `.so`、`LD_LIBRARY_PATH`
  链接到的镜像、binary 内嵌 id 三者必须一致**，否则 **REFUSING TO START**。
- ⇒ **旧 `.so` 不能配新 binary**。要跑"旧产物"，必须同时构建**旧 binary**（旧 pair）。
- 唯一可用顺序（`dsv41_serve_ab.sh:34-40`）：`build.sh 103a`（写 `.build_id`）→
  `touch crates/ferrite-kernel/build.rs`（强制 build.rs 重跑）→ `cargo build --release`。

### 0.3 判据口径（项目红线，全部沿用）

| 探针 | prompt / 口径 | 通过判据 |
|---|---|---|
| **P1 计数** | `请从1数到200，每个数字单独一行。`，`temperature=0`，`max_tokens=1000`，`stream=false` | **仅前 61 行**：非空行 `lines[0..61] == ["1".."61"]`（`batch-reverification-plan.md:45`） |
| **P2 逐字节** | 同上 + `请完整背诵《出师表》全文，从先帝创业未半而中道崩殂开始。` | **全量响应 `content` 字节 + `[toktr]` token 序列 md5 完全一致** |
| **P5 hang** | 全程 | `grep -c ar5-hang == 0` |
| **P6 faults** | 全程 | `grep -cE 'illegal\|fault' == 0` |
| **P7 计数** | nsys `cuda_gpu_kern_sum` | 见 §3 |
| **P8 确定性** | 同 arm 重跑 | 两次 md5 相同（否则字节级 A/B 无意义） |

> **P1 只对前 61 行有效**是项目最重要的一条口径（`session-final-handover.md:14,32`）：
> line 62 起 "重置到 12" 是 **base 模型的自然退化**，不是引擎 bug。但**逐字节（P2）比的是全量**——
> 因为两条 arm 若真的位级同源，退化段也必须一致。

### 0.4 门与配置的现场事实（设计必须建立在正确的事实上）

1. **只有一个门**：`DSV41_AR_STORE_FUSE`（`chain_dev.rs:2202-2206`，默认 OFF）。
   `ar_store_fuse()`（attn）与 `ar_store_fuse_moe()`（MoE，`:2230-2236`）**共用它**。
   ⇒ **开这个门 = 同时开 attn 折 + MoE 折**。
   `docs/agent/stage-b-execution.md:14` 里写的 `DSV41_AR_ST_ATTN=0` / `DSV41_AR_ST_MOE=0`
   **在代码里不存在**（全仓 grep 无 `AR_ST_ATTN|AR_ST_MOE|AR_STORE_ATTN|AR_STORE_MOE`）。
   ⚠️ **归因后果**：门 ON 若数值失败，**不能直接断定是 MoE 侧**（attn 折在 round 19 有过失败史）。
   §3.2 给出**用旧/新 `.so` 做 MoE 半归因**的绕法。
2. **载体按 rank 分**（`ar_store_fuse_moe` 的注释 `:2207-2229`、实现 `:16530` 附近）：
   - **不跑共享专家**的 rank（默认复制布局 = rank 1..7，`shared_rank = comm.rank == 0`，`:15934-15938`）：
     routed batched down-reduce 是 `s.o` 最后写者 ⇒ `moe_down_reduce_ar` 携带 ⇒ **store 消失**。
   - **跑共享专家**的 rank（rank 0）：`ADD_EPI` 默认 ON（`:1508-1511`）、`hcpost_epi`/`fuse_c`
     默认 ON（`:14165-14172`）⇒ `add_epi_ready()==true` ⇒ merge 被折进 AR 自己的 store epilogue
     ⇒ **没有可挂载的 producer ⇒ 该 rank 保持两发 AR（store 仍在）**。
   ⇒ 默认配置下 **MoE 折只在 rank 1..7 生效**（与实施报告 §2.5 "少做的地方" 一致）。
3. **`DSV41_GRAPH_MOE` 是死路径**：`moe_graph_armed` 全仓只有 `false` 初始化（`:3905`）与读取
   （`:14480`），**没有任何赋 `true` 的点**；`git grep DSV41_GRAPH_MOE crates/` 只命中注释。
   （旁证：`docs/agent/dsv41-layer-fusion.md:365` 已判定 "`DSV41_GRAPH_MOE` 是死路径 ✗"。）
   ⇒ **捕获验证必须用 `DSV41_GRAPH_STEP`（整步图，默认 ON），不能依赖 `DSV41_GRAPH_MOE`**。
4. **AR v5 是 fold 的前提**：`ar_v5() = DSV41_GRAPH_STEP!=0 || DSV41_AR_V5!=0`（`tp.rs:1224-1245`
   在 `graph||env` 语义下默认 true）。而 `ar_store_fuse_moe` 要求 `comm.uses_v5()==true`
   （`uses_v5() == ar_v5()`，`tp.rs:559-561`）。
   ⇒ **计数/探针 pass 想同时"v5 开 + 无图"**：`DSV41_GRAPH_STEP=0 DSV41_AR_V5=1`
   （两腿是 `||`，env 腿照样点亮 v5）——**这是本设计的关键技巧**，见 §3.1。
5. **捕获兼容性**：`ar_carry` 是进程常量（env + 符号），`moe_reduce(layer, carried)` 的 publish+reduce
   在段外（`:14474-14479, 14480-14505`），回放时表述不变。

---

## 1. 验证一：双产物重编 + gate OFF 逐字节一致（"重编译漂移"）

**要回答的问题**：`add_kernel` / `moe_down_reduce_kernel` 加了 5 个**默认尾参**、又新增了独立入口
（`ferrite_add_store` / `dsv41_moe_down_reduce_st`）——重编译是否改变了 **旧调用点** 的代码生成
（`build.sh:33-41` 记载的 fast-math 重结合陷阱），以及新 Rust 在 gate OFF 时是否真的"一个字都不发"。

**设计要点**：因为 §0.2 的 build-id 门，必须比较**两个完整 pair**（旧 `.so`+旧 binary、新 `.so`+新 binary），
两边都 gate OFF。

### 1.1 步骤 A：建旧 pair（3ea879f）

```bash
VERIFY=/tmp/a1a-verify
OLD_REV=3ea879f                     # d88cb41 的父提交；已核对无 A1a 符号
mkdir -p $VERIFY/{old,new,log,out}

# 独立 worktree，别碰主树
git -C /home/smith/src/ferrite worktree add $VERIFY/tree-old $OLD_REV

cd $VERIFY/tree-old/kernels/cuda && bash build.sh 103a          # 写 .build_id
cd $VERIFY/tree-old && touch crates/ferrite-kernel/build.rs \
  && CARGO_TARGET_DIR=$VERIFY/target-old cargo build --release  # 烧 stamp

cp $VERIFY/tree-old/kernels/cuda/libferrite_kernels.so  $VERIFY/old/
cp $VERIFY/tree-old/kernels/cuda/.build_id              $VERIFY/old/
cp $VERIFY/target-old/release/ferrite-serve             $VERIFY/old/

# 证明"旧"确实是旧（四个 A1a 符号一个都不该有）
nm -D --defined-only $VERIFY/old/libferrite_kernels.so \
  | grep -cE 'ferrite_add_store|dsv41_moe_down_reduce_st|ferrite_p2p_ar_pubred_v5_moe|ferrite_p2p_ar_pubred_v5_hcpost'
# 期望：0
```

### 1.2 步骤 B：建新 pair（HEAD）

```bash
cd /home/smith/src/ferrite/kernels/cuda && bash build.sh 103a
cd /home/smith/src/ferrite && touch crates/ferrite-kernel/build.rs && cargo build --release

cp /home/smith/src/ferrite/kernels/cuda/libferrite_kernels.so  $VERIFY/new/
cp /home/smith/src/ferrite/kernels/cuda/.build_id              $VERIFY/new/
cp /home/smith/src/ferrite/target/release/ferrite-serve         $VERIFY/new/

# 符号审计：四个都必须在，且 ABI 未被破坏（device.rs 按名绑定）
nm -D --defined-only $VERIFY/new/libferrite_kernels.so | grep -E \
  'ferrite_add_store$|dsv41_moe_down_reduce_st$|ferrite_p2p_ar_pubred_v5_moe$|ferrite_p2p_ar_pubred_v5_hcpost$'
# 期望：4 行
cat $VERIFY/old/.build_id $VERIFY/new/.build_id     # 两者必须不同（同源则 A/B 无意义）
```

### 1.3 步骤 C：两侧都 gate OFF 各跑一次 + 确定性对照

一个 arm = 一个 serve（gate 是进程常量），串行、跑完即拆。

```bash
run_arm() {   # run_arm <tag> <libdir> <bindir>  <extra env...>
  local tag=$1 libdir=$2 bindir=$3; shift 3
  local port=$(( 8400 + RANDOM % 100 ))
  local log=$VERIFY/log/arm_$tag.log
  pkill -9 -x ferrite-serve 2>/dev/null; sleep 8
  ( cd /home/smith/src/ferrite && nohup env \
      CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
      LD_LIBRARY_PATH=$libdir \
      DSV41_TIMING=1 DSV41_AR_V5=1 \
      "$@" \
      $bindir/ferrite-serve --model dsv41 --serve --tp 8 \
        --model-dir /opt/dlami/nvme/models/DeepSeek-V4.1-Flash \
        --lib $libdir/libferrite_kernels.so --port $port \
      >$log 2>&1 & )
  for i in $(seq 1 60); do sleep 5; grep -q "chain ready, serving" $log && break; done
  grep -q "chain ready, serving" $log || { echo "FATAL $tag: not ready"; tail -20 $log; return 2; }

  # 实读进程 env —— 项目 #1 测量偏差陷阱（gate 没进进程）
  tr '\0' '\n' < /proc/$(pgrep -x ferrite-serve | head -1)/environ \
    | grep -E '^DSV41_' | sort > $VERIFY/log/arm_$tag.env

  for P in "请从1数到200，每个数字单独一行。" \
           "请完整背诵《出师表》全文，从先帝创业未半而中道崩殂开始。"; do
    local name=$([ "${P:0:2}" = "请从" ] && echo count || echo shizhen)
    curl -s --noproxy '*' -m 600 -X POST http://localhost:$port/v1/chat/completions \
      -H 'Content-Type: application/json' \
      -d "{\"model\":\"dsv41\",\"messages\":[{\"role\":\"user\",\"content\":\"$P\"}],\"max_tokens\":1000,\"temperature\":0,\"stream\":false}" \
      > $VERIFY/out/${tag}.${name}.json
  done

  grep -E '^\[toktr\]' $log | sed 's/^\[toktr\] //' > $VERIFY/out/${tag}.toktr
  curl -s -m 10 -X POST http://localhost:$port/shutdown >/dev/null 2>&1
  sleep 5; pkill -9 -x ferrite-serve 2>/dev/null; sleep 8
}

# O1 = 旧 pair gate OFF；N0 = 新 pair gate OFF；N0b = 新 pair gate OFF 重跑（P8 确定性对照）
run_arm O1  $VERIFY/old $VERIFY/old  DSV41_AR_STORE_FUSE=0 DSV41_TOKTRACE=1
run_arm N0  $VERIFY/new $VERIFY/new  DSV41_AR_STORE_FUSE=0 DSV41_TOKTRACE=1
run_arm N0b $VERIFY/new $VERIFY/new  DSV41_AR_STORE_FUSE=0 DSV41_TOKTRACE=1
```

### 1.4 判据（P1+P2+P5+P6+P8）

```bash
python3 - $VERIFY/out <<'PY'
import json, hashlib, sys, pathlib
d = pathlib.Path(sys.argv[1])
def content(tag, name):
    j = json.load(open(d/f"{tag}.{name}.json"))
    return j["choices"][0]["message"]["content"]
def md5s(s): return hashlib.md5(s.encode()).hexdigest()

for tag in ("O1","N0","N0b"):
    c = content(tag, "count")
    nz = [l.strip() for l in c.splitlines() if l.strip()]
    ok = 0
    for i,l in enumerate(nz):
        if l == str(i+1): ok += 1
        else: break
    print(f"{tag}: count first_good={ok} (need>=61) content_md5={md5s(c)} "
          f"toktr_md5={md5s(open(d/f'{tag}.toktr').read())} "
          f"shizhen_md5={md5s(content(tag,'shizhen'))}")

print("P8 determinism:", md5s(content("N0","count")) == md5s(content("N0b","count")))
print("P2 O1==N0 :", md5s(content("O1","count")) == md5s(content("N0","count")),
      md5s(content("O1","shizhen")) == md5s(content("N0","shizhen")),
      md5s(open(d/"O1.toktr").read()) == md5s(open(d/"N0.toktr").read()))
PY

for t in O1 N0 N0b; do
  echo "$t faults=$(grep -cE 'illegal|fault' $VERIFY/log/arm_$t.log) hang=$(grep -c ar5-hang $VERIFY/log/arm_$t.log)"
done
```

**判定表**

| 项 | 期望 | 不达标 |
|---|---|---|
| P8 确定性 | `N0.count md5 == N0b.count md5` | **整个 A/B 失效**，先查非确定性，别下结论 |
| **P2 逐字节** | `O1 == N0`（count / 出师表 / toktr **三路全同**） | ❌ 重编译漂移或新 Rust 未完全惰性 |
| P1 计数 | 三个 arm 前 61 行都对 | ❌ 引擎损坏（与 A1a 无关也报） |
| P5/P6 | `ar5-hang=0`、`faults=0` | ❌ 停 |
| 符号 | 旧 4 个计数 = 0；新 4 个都在 | 构建/装载错误，停 |

> 注意：`DSV41_TOKTRACE=1` 的 `[toktr]` 打印在捕获段**之外**（`chain_dev.rs:5620-5630`），
> 不改变执行路径，用它做 token 级字节证据是安全的。

---

## 2. 验证二：gate ON vs OFF 逐字节一致（"fold 不改数值"）

**要回答的问题**：staging 拷贝从独立 kernel 挪进 producer epilogue 后，AR 的**求和链与字节**是否
逐位不变（实施报告 §3 的三段论证：只搬不求和 / 同 kernel / 顺序不变）。

**设计要点**：**同一个新 pair、同一个二进制**，唯一变量是 `DSV41_AR_STORE_FUSE`。gate ON **同时开
attn+MoE**（§0.4-1），所以本验证覆盖"两个折一起"，MoE 单独的归因留给 §3.2。

```bash
run_arm N1 $VERIFY/new $VERIFY/new DSV41_AR_STORE_FUSE=1 DSV41_TOKTRACE=1
# （N0 已在 §1 跑过：新 pair gate OFF）
```

**判据**：与 §1.4 同一段 python + faults/hang 检查，把 `O1→N0` 换成 `N0→N1`。

| 项 | 期望 |
|---|---|
| **P2 逐字节** | `N0 == N1`：count / 出师表 / `[toktr]` 三路 md5 全同 |
| P1 计数 | 两个 arm 前 61 行都对 |
| P5/P6 | `ar5-hang=0`、`faults=0` |
| **门真的进了进程** | `arm_N1.env` 含 `DSV41_AR_STORE_FUSE=1`；`arm_N0.env` 含 `=0` |

> ⚠️ **"逐字节一致"本身不是 fold 生效的证据**——门没生效时也一致。生效证据在 §3。
> 两件事必须同时成立才叫通过：**数值一致（本验证）+ launch 计数按预期下降（§3）**。

---

## 3. 验证三：launch 计数 + 步时 + 捕获

### 3.1 nsys 计数 pass（fold 生效的**首要证据**）

**关键技巧（§0.4-4）**：fold 需要 v5，而 nsys 默认不展开整步图里的 kernel 节点；所以计数 pass 用
**`DSV41_GRAPH_STEP=0 DSV41_AR_V5=1`** ——`ar_v5()` 是 `graph || env`，env 腿照样把 v5 点亮，
于是 **v5 内核全在、图全无**，每个 kernel 都是普通 launch，nsys 的 `cuda_gpu_kern_sum` 干净可读。
同时 pin `DSV41_SPEC=0 DSV41_DSPARK=0`，让每步恰好 = 40 层 × 2 轮（attn + MoE），算术可对账。

```bash
NSYS=${NSYS:-/usr/local/cuda-13.2/bin/nsys}
count_pass() {  # count_pass <tag> <libdir> <bindir> <0|1>
  local tag=$1 libdir=$2 bindir=$3 gate=$4
  local port=$(( 8500 + RANDOM % 100 ))
  local rep=$VERIFY/nsys_$tag
  pkill -9 -x ferrite-serve 2>/dev/null; sleep 8
  ( cd /home/smith/src/ferrite && nohup env \
      CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 LD_LIBRARY_PATH=$libdir \
      DSV41_TIMING=1 DSV41_AR_V5=1 DSV41_GRAPH_STEP=0 \
      DSV41_SPEC=0 DSV41_DSPARK=0 \
      DSV41_AR_STORE_FUSE=$gate \
      $NSYS profile --trace=cuda --sample=none \
        --output=$rep --force-overwrite=true \
        $bindir/ferrite-serve --model dsv41 --serve --tp 8 \
          --model-dir /opt/dlami/nvme/models/DeepSeek-V4.1-Flash \
          --lib $libdir/libferrite_kernels.so --port $port \
      >$rep.log 2>&1 & )
  for i in $(seq 1 60); do sleep 5; grep -q "chain ready, serving" $rep.log && break; done
  # 小请求即可：nsys 下 v5 publish 的自旋被放大 ~300x（nsys_wave1.sh 头），只读计数、绝不读 ms
  curl -s --noproxy '*' -m 900 -X POST http://localhost:$port/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"dsv41","messages":[{"role":"user","content":"请从1数到20，每个数字单独一行。"}],"max_tokens":16,"temperature":0,"stream":false}' \
    > $VERIFY/out/nsys_$tag.json
  curl -s -m 10 -X POST http://localhost:$port/shutdown >/dev/null 2>&1
  sleep 5; pkill -INT -x nsys 2>/dev/null; sleep 12; pkill -9 -x nsys 2>/dev/null
  pkill -9 -x ferrite-serve 2>/dev/null; sleep 8
  $NSYS stats --report cuda_gpu_kern_sum --format csv $rep.nsys-rep > $VERIFY/log/$tag.kern.csv
}

count_pass C_OFF $VERIFY/new $VERIFY/new 0
count_pass C_ON  $VERIFY/new $VERIFY/new 1

# 计数解析（沿用 l49_ab.sh:700 的 csv 口径：instances 在第 3 列，名字在最后）
python3 - $VERIFY/log <<'PY'
import csv,sys,pathlib,collections
d=pathlib.Path(sys.argv[1])
def load(tag):
    c=collections.Counter()
    for r in csv.reader(open(d/f"{tag}.kern.csv",errors="ignore")):
        if len(r)<3: continue
        try: n=int(r[2])
        except: continue
        c[r[-1].strip()]+=n
    return c
off,on=load("C_OFF"),load("C_ON")
def fam(c,*pref):
    return sum(v for k,v in c.items() if any(p in k for p in pref))
print("store OFF/ON:", fam(off,"p2p_ar_store_v5_kernel"), fam(on,"p2p_ar_store_v5_kernel"))
print("pubred OFF/ON:", fam(off,"p2p_ar_pubred_v5"), fam(on,"p2p_ar_pubred_v5"))
print("top diff:", {k:(off[k],on.get(k,0)) for k in set(off)|set(on)
                    if off[k]!=on.get(k,0) and ("ar_" in k or "add_kernel" in k or "down_reduce" in k)})
PY
```

**期望（默认配置：TP8 / 复制共享专家 / ADD_EPI+HC_POST+FUSE_C 默认 ON / 无 MTP）**

每步每 rank：attn 40 轮 + MoE 40 轮 = 80 轮；每轮（未折）1 store + 1 pubred。

| 量 | gate OFF | gate ON | 说明 |
|---|---|---|---|
| `p2p_ar_store_v5_kernel` | 80 / 步 / rank | **40 / 步 / rank**（仅 rank 0） | attn 折在 8 rank 全生效（−40×8）；MoE 折只在 rank 1..7 生效（−40×7）；rank 0 因 ADD_EPI 保持两发 |
| 合计（8 rank） | **640 / 步** | **40 / 步**（≈ −600） | 预期 drop = 40×8 + 40×7 = 600 |
| `p2p_ar_pubred_v5*` | 640 / 步 | **640 / 步（必须一字不差）** | 折只搬 store，**轮数不变**——这是最干净的强判据 |
| `add_kernel` / `moe_down_reduce_kernel` | 计数不变 | 计数不变 | 折是"同一 kernel 多传参"，不是新 kernel |

**判定**：① pubred 计数**两侧完全相同**（差 0）；② store 计数**下降且降幅可对账**
（`OFF−ON == 40×8 + 40×(rank 数−1)`，rank 数由 `/proc/environ` 里的布局决定）；
③ 若 store 不降 ⇒ **门没生效**（查符号审计 §1.2 + `/proc/environ` 回读），A/B 作废。

> 与 §2 的关系：**§2 的"逐字节一致"必须与本节"计数下降"同时成立**，才算 fold 通过。
> 若两者矛盾（文本变了、计数没降），读数不可信，先修观测再判。

### 3.2 （可选）MoE 半的归因：旧 .so gate ON vs 新 .so gate ON

门是单一的，attn 折与 MoE 折同开。若 §2 失败、需要判定是 attn 还是 MoE：

```bash
count_pass C_OLD_ON $VERIFY/old $VERIFY/old 1    # 旧 .so 无 carrier 符号 ⇒ 只折 attn
# 对比 C_OLD_ON vs C_ON：
#   store(C_OLD_ON) = 640 − 320 = 320/步        （attn 折，8 rank）
#   store(C_ON)     = 40/步                      （attn + MoE，MoE 只 7 rank）
#   差 = 280/步 = 40×7  ⇒ 正是 A1a 的 MoE 贡献
```

这条把 **A1a 的 MoE 增量单独量出来**（旧树 = "attn 已在树内"的准确基线，见 §0.1）。
代价是每个 nsys pass ~10 分钟（自旋放大），按需跑。

### 3.3 步时对比（A/B 吞吐）

用 §2 的两个 serve（同一 binary+`.so`，唯一变量是门）的**每步墙钟**，口径按项目唯一可接受的方式：
`[dsv41] step pos=N: X.XXms` 的 **p50（稳态，跳过预热）**；**不要**从 nsys pass 读 ms。

```bash
step_p50() { grep -oP '\[dsv41\] step pos=\d+: \K[0-9.]+' $VERIFY/log/arm_$1.log \
             | tail -n +$(( $(grep -c 'step pos=' $VERIFY/log/arm_$1.log) / 5 )) \
             | sort -n | awk '{a[NR]=$1} END{print a[int(NR/2)+1]}'; }
echo "OFF p50=$(step_p50 N0)  ON p50=$(step_p50 N1)"
```

**期望**：`p50(ON) ≤ p50(OFF)`，设计口径 **−0.08~0.16 ms/步**（`ar-l4l5-optimization-design.md:201`
的 −1~2µs/轮 × 40 轮）。**判据 = 主判据是计数（§3.1），步时只要求"方向不反 + 不回归"**：
本项收益小、易被噪声吞掉（`b6-mrows-f32-design.md:423` 的 R6：**launch 计数是首要判据**）。
若要报数，必须同会话背靠背、串行、单请求一 serve，并附上 `[dsv41] step` 的 n/p10/p50/p90。

### 3.4 捕获兼容性（多轮连续请求）

用真捕获路径 **`DSV41_GRAPH_STEP=1`（默认）**；**不要用 `DSV41_GRAPH_MOE`**（§0.4-3：死路径）。
载体 launch 落在捕获段内、publish+reduce 在段外，回放时 `ar_carry` 是进程常量 ⇒ 表述不变
（`chain_dev.rs:14474-14479`）。历史上"图 ON + 多轮连续请求"有过 rank 漂移的非法访问
（`:5567-5579`），本项就是要把它压住。

```bash
run_multi() {  # run_multi <tag> <gate>  —— 单 serve 内 4 个连续请求，验证重捕获/回放
  local tag=$1 gate=$2 port=$(( 8600 + RANDOM % 100 ))
  pkill -9 -x ferrite-serve 2>/dev/null; sleep 8
  ( cd /home/smith/src/ferrite && nohup env \
      CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
      LD_LIBRARY_PATH=$VERIFY/new \
      DSV41_TIMING=1 DSV41_AR_V5=1 DSV41_GRAPH_STEP=1 \
      DSV41_AR_STORE_FUSE=$gate \
      $VERIFY/new/ferrite-serve --model dsv41 --serve --tp 8 \
        --model-dir /opt/dlami/nvme/models/DeepSeek-V4.1-Flash \
        --lib $VERIFY/new/libferrite_kernels.so --port $port \
      >$VERIFY/log/multi_$tag.log 2>&1 & )
  for i in $(seq 1 60); do sleep 5; grep -q "chain ready, serving" $VERIFY/log/multi_$tag.log && break; done
  for P in "请从1数到200，每个数字单独一行。" \
           "请完整背诵《出师表》全文，从先帝创业未半而中道崩殂开始。" \
           "1+1=" \
           "请从1数到20，每个数字单独一行。"; do
    curl -s --noproxy '*' -m 600 -X POST http://localhost:$port/v1/chat/completions \
      -H 'Content-Type: application/json' \
      -d "{\"model\":\"dsv41\",\"messages\":[{\"role\":\"user\",\"content\":\"$P\"}],\"max_tokens\":1000,\"temperature\":0,\"stream\":false}" \
      >> $VERIFY/out/multi_$tag.jsonl
    echo >> $VERIFY/out/multi_$tag.jsonl
  done
  curl -s -m 10 -X POST http://localhost:$port/shutdown >/dev/null 2>&1
  sleep 5; pkill -9 -x ferrite-serve 2>/dev/null; sleep 8
}
run_multi MG_ON 1
run_multi MG_OFF 0
```

**判据**

| 项 | 期望 |
|---|---|
| P5/P6 | `multi_MG_ON.log`：`illegal\|fault == 0`、`ar5-hang == 0` |
| 连续性 | 4 个请求全部返回（jsonl 4 行有效），无进程死亡、无 "a rank did not answer" |
| 捕获真的发生 | 图 ON 时无 `capture FAILED`；且**不是** `DSV41_GRAPH_MOE` 路径（那是死代码，见 §0.4-3） |
| **逐字节** | `MG_ON` 的四段响应 `content` 与 `MG_OFF` 的四段**两两相同**（同 config 只差门） |
| 交叉对照 | 与 §1 的无图 arm 对比：图 ON/OFF 的文本也一致（与门正交的已知属性） |

---

## 4. 三项验证的完整判据汇总（一张表）

| # | 验证 | 命令主入口 | 通过判据 | 失败含义 |
|---|---|---|---|---|
| **1** | 双产物重编 + gate OFF 逐字节 | §1.3 `run_arm O1/N0/N0b` + §1.4 | P8：N0==N0b；**P2：O1==N0（count+出师表+toktr）**；P1 前 61 行；faults=0；hang=0；旧符号 0 / 新符号 4 | 重编译漂移，或新 Rust 非惰性 |
| **2** | gate ON vs OFF 逐字节 | §2 `run_arm N1` + §1.4 同脚本 | **P2：N0==N1（三路）**；P1 前 61 行；faults=0；hang=0；`/proc/environ` 实读门进了 | fold 改数；或观测不可信 |
| **3a** | launch 计数 | §3.1 `count_pass C_OFF/C_ON` | pubred 计数**完全相同**；store 下降且降幅 = 40×8+40×(rank−1) | 门没生效（先查符号+env） |
| **3b** | （可选）MoE 归因 | §3.2 `C_OLD_ON` | store(C_OLD_ON)−store(C_ON) = 280/步 = 40×7 | 归因 A1a 的 MoE 增量 |
| **3c** | 步时 | §3.3 | p50(ON) ≤ p50(OFF)，−0.08~0.16ms/步；方向不反、无回归（计数为主判据） | 收益落空 |
| **3d** | 捕获 | §3.4 `run_multi MG_ON/MG_OFF` | 4 连request 全返回；faults=0；hang=0；无 capture FAILED；MG_ON==MG_OFF 四段逐字节 | 图与载体交互坏 |

**§2 与 §3a 必须成对成立**：逐字节一致但计数不降 = 什么都没测到；计数下降但文本变 = 快而不对。

---

## 5. 陷阱清单（执行前逐条读）

1. **旧 `.so` 必须配旧 binary**（§0.2，`cuda.rs:2126-2140` 的三重 build-id 校验）。混搭会
   "REFUSING TO START"，不是静默降级。
2. **构建顺序**：`build.sh` → `touch build.rs` → `cargo build`。反过来或漏 touch，cargo 增量会
   重新链接**旧 stamp**，导致永久不匹配（`dsv41_serve_ab.sh:34-40`）。
3. **一个 arm 一个 serve**：`DSV41_AR_STORE_FUSE` 是 `OnceLock`（进程常量），改门只能重启。
   同一 serve 里跑两个 prompt 会污染 process-level 累加器（`[dspark] steps=` 不按请求重置）。
4. **门是单一的**：`DSV41_AR_STORE_FUSE=1` 同时开 attn 折 + MoE 折。没有
   `DSV41_AR_ST_ATTN/_MOE`（代码里不存在）。MoE 单独归因走 §3.2 的旧/新 `.so` 对照。
5. **nsys 只读计数**：v5 publish 在 nsys 下自旋放大 ~300×（`nsys_wave1.sh:36-42`），
   该 pass 的 ms 一律作废；因此用 `max_tokens=16` 的小请求 + 长 watchdog。
6. **nsys 计数必须 `DSV41_GRAPH_STEP=0`**（否则整步图把 kernel 收进 graph node，
   `cuda_gpu_kern_sum` 读不到）；但**不能**把 `DSV41_AR_V5` 也关（那会连 v5 内核一起关掉，
   计数全 0）。正确组合：`GRAPH_STEP=0 AR_V5=1`。
7. **`DSV41_GRAPH_MOE` 是死路径**（无赋值点）。捕获用 `DSV41_GRAPH_STEP`。
8. **不要开额外观测**：`DSV41_V5_LEDGER` / `DSV41_STATS` 会引入 D2H 同步或强制退图
   （`dspark-correctness-chain.md` 的"观测干扰"）；`DSV41_TOKTRACE` 是安全的那一个（段外打印）。
9. **P1 计数只对前 61 行有效**；字节一致性则比全量——别用"line 62 变成 12"误判为回归。
10. **默认配置下 rank 0 不折 MoE**（ADD_EPI on ⇒ 无 producer 可挂）。预期计数里 rank 0 的
    store 仍在，是设计使然，不是 bug。

---

## 6. 本设计未覆盖

- **MoE 折的 additive 变体（A5 `moe_epi_add` / `DSV41_ADD_EPI=0` 下的 add_inplace 携带）**：
  实施报告 §7 明确留给后续。若要覆盖，需在 `DSV41_ADD_EPI=0` 下重跑 §2 + §3。
- **engram AR**（2 节点/步）与 **attn 侧折的独立复测**：属 Step 1 之前的遗留项（报告 §7）。
- **`DSV41_GRAPH_MOE` 死代码的清理**：发现于 §0.4-3，建议单独立项。

*工部 · 本文件为唯一产出；命令中所有路径（模型目录、nsys 路径、nvcc）以节点实际为准。*
