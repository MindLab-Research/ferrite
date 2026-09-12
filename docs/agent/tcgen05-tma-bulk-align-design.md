# tcgen05 TMA bulk 对齐 —— 根本修复设计

> 工部 · 2026-09-12 · **只读调查 + 本文件（唯一产出）**。未执行任何 GPU 命令、未改动任何源码。
> 上游输入：`docs/agent/tcgen05-residual-misalign-sanitizer-plan.md`（结论：byte-fallback 对 `cp.async.bulk` 原理无效）。
> 现场核对：`kernels/cuda/dsv41_experts_mxf4.cu`（HEAD 工作树）、`kernels/cuda/dsv41_kernels.cu`、
> `crates/ferrite-models/src/dsv41/{load.rs,weights.rs,device.rs,chain_dev.rs}`、
> `crates/ferrite-kernel/src/devrt.rs`。所有行号对当前工作树现场核对。

---

## 0. 结论摘要（先看这个）

1. **修复的形态不是「再加一个 fallback」，也不是「把 `al16` 门加硬」**，而是**把 16B 从
   launcher 的运行时门提升成布局的构造性不变量**，让门永远为真。
   理由有一条硬约束：`cp.async.bulk` 的 smem 目的侧**没有可拆的粒度**（B 侧直落 canonical unit，
   16B 是 MMA descriptor 语义的一部分），所以「源不对齐就降级」在这条路上不存在；
   **唯一能保证的是源与目的都在构造时就对齐**。

2. **门的现状是「四缺三」**（现场核对，见 §3）：
   - `tc5::e4`（`e4_launch_gateup` `:5218-5222`）**完整** —— `act` + 4 基址 + 4 stride 全查；
   - `tc5::mxf4`（`:4471-4475`）**漏 `act`**；
   - `e4x` grouped（`:6163-6167`）**漏 `bh_base` / `bhs_base` / `bh_stride` / `bhs_stride`**（任务里提到的漏项，已确认）；
   - `e4x` dense（`:5795-5796`）只查 3 个基址，**8 个 stride 一个没查**；
   - `dsv41_gemm_fp8_swapab`（`dsv41_kernels.cu:6361-6387`）**一个都没查**，
     而它的 `w` 走 `cp.async.bulk`（`:715`）、`a` 走 `uint4` 读写（`:656-658`），**两边都是 16B 硬要求**。

3. **「1 misaligned」最可能是被漏掉的那三个门之一，而不是 e4 的 bulk**：
   `tc5::e4` 的门把 `act`/4 基址/4 stride 全钉住了，且它 kernel 内部的行偏移
   （`row*kbytes`、`pk0`、`st*kKStep/2`、`e4_off`）在 `dim%128==0` 下**逐项都是 16B 的倍数**（§2.3 推导）。
   ⇒ 在 e4 这条路上，门若真被触发，Rust 侧会把它当**错误**报出来（`device.rs:5380` 的 `kerr`），
   不会变成 err 716 的 fault。**err 716 只可能来自门没覆盖的指针。**
   这是一条**可证伪的判据**：`CUDA_LAUNCH_BLOCKING=1` 下若报 `cuda error 1 (invalid argument)` 而不是 716，
   就证明「decline 已发生」；报 716 则证明「漏掉的指针」——**这一步必须在写任何代码之前先跑**（§6 V0）。

4. **门的失败模式本身是 bug 级的**：`expert_tcgen05_gate_up_mxf4` 的 Rust 包装
   （`device.rs:5293-5328`）在 `rc == 0` 时返回 `Ok(false)` 表示「门关着」，
   而非零 `rc` 走 `kerr` ⇒ **「门拒了」和「launch 失败」在 Rust 侧只有 `0` 与「非 0」一个比特的差别**。
   再叠上 `chain_dev.rs:16155-16177` 的 `ran_tc = ...` → `!ran_tc` 就回退 GEMV，
   于是一个**布局事故**看起来像「臂没生效」。这是本项目记载过的 #1 度量陷阱（A/B 失真）的翻版。

5. **对齐保证的方案 = 三层不变量 + 一个审计钩子**（§4）：
   - **层 1（根）布局**：per-expert `block` 与每个 plane 偏移都 `align_up(16)`（建议 128），
     行 pitch（`dim/2`、`nsf=dim/32`、`inter_local/2`）断言 16B 倍数；
   - **层 2（门）**：三个 launcher 共用一张对齐描述表，补齐所有漏项，失败时**响亮**（一次诊断 + 可选 `abort`）；
   - **层 3（证明）**：smem 侧用 `static_assert` 把「16B 由构造保证」钉死（现在只有 `alignas`，靠人眼）；
   - **审计**：`DSV41_ALIGN_AUDIT=1` 在**装载期**（不是 prefill 191ms 处）遍历每层每 expert 的
     6 个 plane + 每 rank 的 shard，打印并校验 `base&15 / stride%16 / pitch%16 / nsf%16`。

6. **代价**：`ALIGN=128` 下每个 expert 至多多花 889 B（6 个 plane 偏移 + block 各一次 127B），
   生产形状每 rank 每层 ≤ 42.7 KiB，40 层 ≈ 1.7 MiB per rank（相对 119.5 MiB/expert-pool 是 0.1%）。
   补白字节全为 0（pool 已被 `zero_at` 清零），**不改变任何权重值 ⇒ 数值必须逐字节不变**。

7. `-lineinfo`、`FERRITE_ALIGN_STRICT`、`DSV41_ALIGN_AUDIT` 的默认值、以及 **「门失败是 decline 还是 error」的策略变更**，
   都属**方案外**，请尚书省批准（§7）。

---

## 1. 现场核对：全部 `cp.async.bulk` 站点

`grep -rn "cp.async.bulk" kernels/cuda/*.cu`:  `dsv41_experts_mxf4.cu` 9 行、`dsv41_kernels.cu` 7 行，
其余文件只有注释/测试。实际**指令级**站点如下（去重定义与调用）：

| # | 位置 | 形态 | 源（global） | 目的（smem） | size |
|---|---|---|---|---|---|
| B1 | `mxf4.cu:3002`（`dsv41_w2_pf_bulk`） | `cp.async.bulk.prefetch.L2.global` | `:3038` `w2_base + e*w2_stride + j*kW2PfChunk`；`:3041` `w2s_base + e*w2s_stride + …` | — | `span` 截断到 chunk，`n &= ~15u` |
| B2 | `mxf4.cu:597→:715`（`swapab_bulk_g2s`，在 `dsv41_kernels.cu`） | `.shared::cluster.global` | `w + (m0+r)*k + k0` | `sw + slot*16*kSwapabRow + r*kSwapabRow` | `rb`（≤`kSwapabKStep`，16B 倍数） |
| B3 | `mxf4.cu:3376`（`tc5_bulk_g2s`） | 同上 | `:3567` `wrow + tid*(kPackK/2)`；`:3571` `act + k0` | `s.a_raw[sslot] + …`；`s.b_raw[sslot] + tid*kPackK` | `kPackK/2` / `kPackK` |
| B4 | `mxf4.cu:4086`（`m4_bulk_g2s`） | 同上 | `:4225` `wrow + kb0 + st*kAtomBytes + 16*kb`；`:4236` `act + kb0 + …` | `:4225` `s.a_op[sslot][st] + m4_off(tid,kb)`；`:4235` `s.b_op[sslot][st] + m4_off(0,kb)` | 16 |
| B5 | `mxf4.cu:4793`（`e4_bulk_g2s`） | 同上 | `:4977` `wrow + pk0 + st*(kKStep/2)`；`:4986` `act + g*kPackK + st*kKStep + 16*kb` | `:4977` `s.a_raw[sslot] + tid*(kPackK/2) + st*16`；`:4986` `s.b_op[sslot][st] + e4_off(0,kb)` | 16 |

> `e4m3_gemm_grouped_kernel` / `e4m3_gemm_kernel`（`e4x`）**不发 bulk**：它们的操作数读点是
> LDG（`ld_uint4_a16` 系已守）。它们的对齐风险是**另一类**（见 §3 的门漏项），
> 但 `bytes` 侧不涉及 bulk 的「静默错位」，只会 err 716。

---

## 2. 每个站点的对齐推导链（逐项分解）

`cp.async.bulk` 的 PTX 要求是**三条**：`dst` 16B、`src` 16B、`size % 16 == 0`。
`cp.async.bulk.prefetch.L2` 只要 `src` 16B + size。把三条拆成可验证的原子条件：

### 2.1 A 侧（权重，源 = `w1p/w3p` 的行）

```
src = pool + e*block + poff[k] + row*pitch + j*16
      └──┬──┘   └───┬───┘  └──┬───┘  └─┬─┘   └─┬─┘
      cudaMalloc  load.rs    load.rs  dim/2   kernel 常量
      256B 对齐    裸累加!    裸累加!  需 %16   恒 %16
```

| 因子 | 来源 | 现值（生产 dim=5120, inter=2304, world=8） | 是否被保证 |
|---|---|---|---|
| `pool` | `devrt.rs:1075` `cudaMalloc`（`debug_assert` 只查 `&0xF`） | ≥256B | ✅ |
| `e*block` | `load.rs:641-657` 的 `block` 是**六个 plane 的裸累加** | 2,611,200 B → `%16==0`、`%128==0` | ⚠️ **恰好成立，无断言** |
| `poff[k]` | 同上，裸累加 | 0 / 51200 / 819200 / … | ⚠️ **恰好成立，无断言** |
| `pitch = dim/2` | `kbytes = dim>>1`（`mxf4.cu:4923`） | 2560 → `%16==0` | ✅（`dim%128==0` 的 launcher 检查隐含 `dim%32==0`） |
| `j*16` | `pk0 = g*16`、`st*16`、`16*kb`、`kAtomBytes=64` | 16 的倍数 | ✅ 编译期 |
| `row` | `m0 + tid`，`m0 = blockIdx.x * kMTile`（128） | — | ✅ |
| `row*pitch` 的另一种：swapab 的 `(m0+r)*k` | `k` 由调用者给，launcher 查 `k % 32 == 0`（`:6367`） | — | ✅ |
| scale 行 `rr*nsf` | `nsf = dim>>5` | 160 → `%16==0` | ⚠️ **只在 `dim % 512 == 0` 时成立**（现在只查 `dim%128==0`） |

### 2.2 B 侧（激活，源 = `act`）

| 站点 | 表达式 | 需要 | 现状 |
|---|---|---|---|
| B5 | `act + g*kPackK + st*kKStep + 16*kb`（`kPackK=32`、`kKStep=32`） | 仅 `al16(act)` | ✅ **e4 门查了**（`:5220`） |
| B4 | `act + kb0 + st*kAtomBytes + 16*kb`（`kb0=g*32`、`kAtomBytes=64`） | 仅 `al16(act)` | ❌ **mxf4 门没查** |
| B3 | `act + k0`（`k0` 为 `act` 的 K 偏移） | 仅 `al16(act)` | 见对应 launcher |
| B2 | swapab 的激活不走 bulk，但 `:656-658` 用 `uint4` 直接读写 | `al16(a + k0p)` | ❌ **没查** |

### 2.3 smem 目的侧（编译期可证）

`Smem`（`mxf4.cu:4688-4702`，e4 臂）：
`a_raw` `alignas(1024)`、`a_op` `alignas(1024)`、`b_op` `alignas(128)`、`sf_stage` `alignas(1024)`；
`kARawBytes=2048`、`kAbBytes=4096`、`kBbBytes=256` 全 `%16==0`；`e4_off(r,kb) = 16*((r&7)+8*kb+16*(r>>3))`
**恒为 16 的倍数**；`tid*(kPackK/2)=tid*16`。
⇒ B5 的两侧都成立。**但没有任何 `static_assert` 把这个结论钉死**——
一个 `kPackK`/`kRing` 的调参或 `e4_off` 的重写就能静默破坏它。

> **正例（可以直接抄的写法）**：`dsv41_kernels.cu:503-513` 的
> `static_assert(kSwapabRow % 16 == 0, …)` + `kSwapabMbarBytes` 向上取整到 16
> ——这正是本设计要推广到全部 arm 的形状。

---

## 3. 门的覆盖矩阵（「哪些链接没有任何保证」）

| launcher | 行 | 已查 | **漏** |
|---|---|---|---|
| `tc5::e4` `e4_launch_gateup` | `mxf4.cu:5218-5222` | `act` + `w1/w1s/w3/w3s` 基址 + 4 stride | —（完整） |
| `tc5::mxf4` `m4_launch_gateup` | `mxf4.cu:4471-4475` | 4 基址 + 4 stride | **`act`** |
| `e4x` grouped `e4x_launch_grouped` | `mxf4.cu:6163-6167` | `a`、`a_scale`、`b_base`、`bs_base`、`b_stride`、`bs_stride` | **`bh_base`、`bhs_base`、`bh_stride`、`bhs_stride`**（`:6161` 只做 nullptr 检查） |
| `e4x` dense `e4x_launch_gemm` | `mxf4.cu:5795-5796` | `a`、`b`、`b_hi` | **8 个 stride 全部**（且该 arm 当前不发：`e4x_tile=false`） |
| `dsv41_gemm_fp8_swapab` | `dsv41_kernels.cu:6361-6387` | 5 个指针非空 + `n%16` + `k%32` | **`al16(a)`、`al16(w)`**（`a` 走 `uint4` 直读 `:656-658`，`w` 走 bulk `:715`） |

**旁证：`bhs_base` 的重要性有代码注释自证。** `mxf4.cu:5025-5032` 明确写着
「`w3sp`/`w1sp` 是 TP-sharded 的 W3/W1 **SCALE** 视图 —— grouped 臂的 `bhs_base` 的另一个名字。
shard 边界可以让某一个 rank 的视图偏几个字节而其他 rank 正常，一个奇数基址的 uint4 读就是 err 716」
——这段注释描述的现象与任务给的「`DevBuf::view` 的 base 不继承 16B」完全一致，
而**它只在 `tc5::e4` 的 SF prologue 里被 `ld_uint4_a16` 兜住了，在 grouped 臂的 launcher 里连门都没有**。

### 3.1 「D 位」：Rust 侧没有任何一条 16B 约束

- `DevBuf::view(ptr, bytes)`（`devrt.rs:340`、`device.rs:1580`）不做任何对齐检查；
- `as_u8_at(off)`（`devrt.rs:365`）不查 `off`；
- `chain_dev.rs` 里 **179 处** `wrapping_add`，其中直接喂给 bulk 源的真实例子：
  `:11458` `xq + g*k`、`:11460` `wo_a + g*olg*k`、`:16160` `xq4.as_u8()`（e4 的 `act`）。
  这些指针**没有任何一处**在 host 侧被断言过 16B——
  它们只是「碰巧」对齐，靠的是「`k % 32 == 0`」这类**由别处维护的隐式前提**。

**⇒ 「misaligned 的来源」的正确定位不是某一个具体 bug，而是：
16B 这个契约没有任何单一真源（single source of truth），
它被分散在 launcher 的门（4 缺 3）、loader 的隐式布局（裸累加恰好成立）、
kernel 的常量（靠人眼核对）三处，每一处都能单独破环。**

---

## 4. 修复设计：三层不变量 + 一个审计钩子

### 4.0 对齐保证的形式化命题

**命题**：若
1. `pool` 来自 `cudaMalloc`（≥256B）；
2. `block ≡ 0 (mod 16)` 且 `∀k: poff[k] ≡ 0 (mod 16)`；
3. 每个 plane 的行 pitch `≡ 0 (mod 16)`（即 `dim%32==0`、`(dim/32)%16==0`、`inter_local%32==0`）；
4. kernel 内部的行偏移都是 `row*pitch + j*16`（`j ∈ ℤ`，`row = m0 + tid`，`m0 = bx*128`）；

则对任意 `e, k, row, j`：`pool + e*block + poff[k] + row*pitch + j*16 ≡ 0 (mod 16)`。
——即 **A 侧 bulk 源无条件 16B 对齐**。

B 侧：`al16(act)` 且 `kPackK/kAtomBytes ≡ 0 (mod 16)` ⇒ 无条件对齐。
smem 侧：`alignas ≥ 16` + 每个 slot 步长 `≡ 0 (mod 16)` + `e4_off/m4_off ≡ 0 (mod 16)` ⇒ 无条件对齐。

**三层设计就是把 1/2/3/4 的每一条都变成「构造时成立 + 编译/装载期可验证」，并把门从「兜底」降级为「报警」。**

---

### 层 1 —— 布局不变量（修复的根）

#### 1a. per-expert `block` 与 plane 偏移对齐（`crates/ferrite-models/src/dsv41/load.rs`，`load_expert_pool`）

现状（`:641-666`，裸累加，**恰好**成立）：

```rust
        let (block, poff) = if ilv {
            let mut o = [0usize; 6];
            o[1] = 2 * w1b;
            o[3] = o[1] + plans[1].bytes;
            o[4] = o[3] + plans[3].bytes;
            o[5] = o[4] + plans[4].bytes;
            (o[5] + plans[5].bytes, o)
        } else {
            let mut o = [0usize; 6];
            let mut acc = 0usize;
            for (k, slot) in o.iter_mut().enumerate() {
                *slot = acc;
                acc += plans[k].bytes;
            }
            (acc, o)
        };
        debug_assert!(block * n_routed <= total.max(1));
        // one allocation; the K padding is already zero
        let pool = self.dev.alloc(total.max(1))?;
        self.dev.zero_at(pool.ptr, total.max(1))?;
```

改为：

```rust
        // ---- 16B pool geometry: the bulk-copy contract --------------------
        // Every pointer an expert kernel derives is
        //     pool + e*block + poff[k] + row*pitch [+ j*16]
        // and `cp.async.bulk{.prefetch}` demands 16 B on BOTH usable sides
        // (src, and dst for the G2S form). cudaMalloc supplies `pool`; `block`,
        // `poff` and `pitch` are OURS, so they are padded here instead of
        // being "accidentally right" for the shipped shape (the current
        // 2,611,200 B block is a multiple of 128 only by arithmetic luck, and
        // nothing asserts it).
        // See docs/agent/tcgen05-tma-bulk-align-design.md §4.
        const ALIGN: usize = 128;  // >= 16 (PTX); 128 also keeps a full L2 sector
        let up = |x: usize| (x + ALIGN - 1) & !(ALIGN - 1);
        let (block, poff) = if ilv {
            let mut o = [0usize; 6];
            o[1] = up(2 * w1b);
            o[3] = up(o[1] + plans[1].bytes);
            o[4] = up(o[3] + plans[3].bytes);
            o[5] = up(o[4] + plans[4].bytes);
            (up(o[5] + plans[5].bytes), o)
        } else {
            let mut o = [0usize; 6];
            let mut acc = 0usize;
            for (k, slot) in o.iter_mut().enumerate() {
                *slot = up(acc);
                acc = *slot + plans[k].bytes;
            }
            (up(acc), o)
        };
        debug_assert!(block % ALIGN == 0);
        debug_assert!(poff.iter().all(|o| o % ALIGN == 0));
        debug_assert!(block * n_routed >= total);
        // one allocation; BOTH the K padding and the alignment padding are zero
        let pool = self.dev.alloc(block * n_routed)?;
        self.dev.zero_at(pool.ptr, block * n_routed)?;
```

注意三点：
- `ilv` 分支的 `o[2]` 保持 0（w3 复用加倍区域，见 `:716-724` 的 view 逻辑），0 天然对齐；
- `dma_plan` 的落地地址与 view 的 `pb + poff[k]` **同时**读到 `poff`，改动一处即两端一致（`:701/:710` 与 `:723`）；
- `debug_assert!(block * n_routed <= total)` 的方向必须改成 `>=`。

#### 1b. 行 pitch 不变量（`crates/ferrite-models/src/dsv41/weights.rs` + 装载期校验）

```rust
/// 16 B: the alignment `cp.async.bulk*` requires on the global source (and on
/// the smem destination of the `.shared::cluster.global` form), and the granule
/// every weight-plane PITCH must be a multiple of. A violation is SILENT on the
/// bulk form (bytes land in the wrong place, no fault) -- see
/// docs/agent/tcgen05-tma-bulk-align-design.md.
pub const BULK_ALIGN: usize = 16;

/// The pool geometry the bulk paths depend on, checked ONCE per model at LOAD
/// time. A config that cannot be bulk-addressed must fail here (seconds) rather
/// than at the first prefill (191 ms in, with a detached err 716 and no kernel
/// name).
pub fn check_bulk_geometry(cfg: &Dsv41Config, world: usize) -> Result<()> {
    let dim = cfg.dim;
    let mut bad: Vec<String> = Vec::new();
    if dim % (2 * BULK_ALIGN) != 0 {
        bad.push(format!("dim={dim}: the fp4 row pitch dim/2 is not 16B (needs dim % 32 == 0)"));
    }
    if (dim / 32) % BULK_ALIGN != 0 {
        bad.push(format!(
            "nsf=dim/32={}: the e8m0 scale row pitch is not 16B (needs dim % 512 == 0); \
             the SF prologue's `rr * nsf` and the grouped arm's bhs rows both address it directly",
            dim / 32
        ));
    }
    for (what, v) in [("inter/world", cfg.moe_inter_dim / world), ("padded_inter", padded_inter(cfg.moe_inter_dim / world))] {
        if v % BULK_ALIGN != 0 {
            bad.push(format!("{what}={v} is not 16B (w2's ExpertCols row pitch inter_local/2)"));
        }
    }
    if bad.is_empty() { Ok(()) } else {
        Err(FerriteError::Config(format!("bulk-geometry: {}", bad.join("; "))))
    }
}
```

`Loader::load` 开头调用 `check_bulk_geometry(cfg, world)?`。
（生产形状：`dim=5120` ⇒ `2560%16==0`、`160%16==0`、`288%16==0`、`320%16==0` 全通过。）

#### 1c. Rust 侧的 view 算术带上约束（`crates/ferrite-kernel/src/devrt.rs`）

```rust
    /// A 16-byte-aligned view at `off` bytes into this buffer. `cp.async.bulk*`
    /// and `cp.async`16/`LDG.128` operands all require it, and this is the ONE
    /// place a view into a pooled allocation can be built: a bare
    /// `ptr.wrapping_add(off)` at a call site is how a 4-byte slip reaches a
    /// TMA source (dsv41_experts_mxf4.cu:5025-5032 documents the class).
    #[inline]
    pub fn u8_aligned_at(&self, off: usize) -> *const u8 {
        debug_assert_eq!(off & (BULK_ALIGN - 1), 0, "bulk/16B view must stay aligned");
        (self.ptr as *const u8).wrapping_add(off)
    }
```

`chain_dev.rs` 里所有「喂给 bulk 源的 view」改走它（至少 `:11458`、`:11460`、`:16160`；
其余 179 处按「是否进入 bulk 源」逐个分类，**不在本设计的一次提交里全改**，见 §6 的分批）。

> ⚠️ 这一条只加 `debug_assert`（release 无开销）。它的价值是**在 CI/parity 跑里立刻炸**，
> 而不是在生产里静默。

---

### 层 2 —— launcher 的门：补齐 + 响亮化

#### 2a. 一个共用的对齐描述表（`kernels/cuda/dsv41_experts_mxf4.cu`，放在第一个 launcher 之前，约 `:4425`）

```cpp
// ---- the bulk/16B alignment contract, in ONE place --------------------------
// `cp.async.bulk*` demands 16 B on the global source (and on the smem
// destination for the G2S form); the lengths must be multiples of 16. A
// violation is SILENT on the bulk form (the bytes land elsewhere, no fault) and
// err 716 on the LDG.128/`cp.async`16 form. Every launcher that can reach a
// bulk copy checks its own argument list through this helper, so a new arm
// cannot be born with half the check (the e4 arm had all of it, the mxf4 arm
// was missing `act`, the grouped arm the whole `bh*` set, the swapAB entry
// point nothing).
//
// FERRITE_ALIGN_STRICT=1 turns the refusal into a HARD STOP. That is the mode
// for CI and for the harness: a bare cudaErrorInvalidValue is swallowed as
// "the arm declined" on the Rust side (device.rs:5323-5327 -> Ok(false) ->
// chain_dev.rs:16209 falls back to the SIMT GEMV), i.e. a layout accident
// presents as "the arm never ran".
struct BulkAlign {
    const char* name;
    const void* p;
    long stride;   // 0 = no per-expert stride to check
};
inline bool bulk_align_ok(const char* who, const BulkAlign* v, int n) {
    static const int strict = [] {
        const char* e = getenv("FERRITE_ALIGN_STRICT");
        return (e != nullptr && e[0] == '1') ? 1 : 0;
    }();
    bool ok = true;
    for (int i = 0; i < n; ++i) {
        const unsigned a = (unsigned)((uintptr_t)v[i].p & 0xF);
        const long s = v[i].stride;
        if (a == 0 && ((s & 0xF) == 0)) continue;
        ok = false;
        // ONCE per process per launcher: the condition is a property of the
        // LAYOUT, not of the call, and this runs in the hot path.
        static int reported = 0;
        if (reported++ < 8)
            fprintf(stderr,
                    "[align] %s: %s base misaligned by %u B, stride misaligned by %ld B "
                    "-- the bulk/16B contract cannot be met (see "
                    "docs/agent/tcgen05-tma-bulk-align-design.md)\n",
                    who, v[i].name, a, s & 0xF);
    }
    if (!ok && strict) { fflush(stderr); abort(); }
    return ok;
}
```

#### 2b. 逐 launcher 的调用点

| launcher | 改法 |
|---|---|
| `tc5::e4` `:5218-5222` | 语义不变，改写成 `bulk_align_ok("e4", {{"act",act,0},{"w1",w1_base,w1_stride},…},9)`（**保留** `al16(act)`：它是 e4 独有的、也是唯一的 B 侧保证） |
| `tc5::mxf4` `:4471-4475` | **加 `{"act", act, 0}`** ← 这是 B4 的唯一漏洞 |
| `e4x` grouped `:6163-6167` | **加 `bh_base/bh_stride`、`bhs_base/bhs_stride`**，并把 `a/a_scale/b_base/bs_base` 一起收进同一张表 |
| `e4x` dense `:5795-5796` | 补 8 个 stride（`b_stride/bs_stride/bh_stride/bhs_stride`），基址表不变 |
| `dsv41_gemm_fp8_swapab`（`dsv41_kernels.cu:6361`） | **加 `{"a", a, 0}`、`{"w", w, 0}`**：`a` 走 `uint4` 直读（`:656-658`），`w` 走 bulk（`:715`）/`cp.async16`（`:740-741`），两个都是 16B 硬要求 |

#### 2c. **不给「硬 al16 ⇒ 该 rank decline」背书**（回应任务里的疑虑）

任务里写「硬 al16 会让该 rank decline——可能不是好方案」。**这个判断是对的**，但结论要分两层：

1. **decline 本身不是修复**：它只是把「静默错位」换成「静默换臂」。
   两者都不可接受——前者算错，后者让 A/B 度量失真。
2. **门的定位应当是报警器，真正干活的是层 1**：层 1 落地后，门在生产形状上**恒为真**，
   唯一会让它为真的是「上游出现了新的破环」（例如有人把 `inter_local` 改成非 32 倍数）。
   那种情况下 **decline 是错的、abort 也是错的**——正确的是**一次响亮的诊断 + 保持现状的语义**，
   即：默认打印 + decline（不改变「跑到哪儿了」的可预期性），
   `FERRITE_ALIGN_STRICT=1`（CI/harness）时 abort，让 CI 拦住它。

> 所以本设计的门**不收紧语义、只补齐覆盖面 + 让失败可见**。这与「不引入新 fallback」的约束一致：
> 门是既有的 fallback 触发点，我们只是让它不再沉默。

---

### 层 3 —— smem 侧的编译期证明

`kernels/cuda/dsv41_experts_mxf4.cu`，每个 arm 的 `Smem` 之后（e4 在 `:4708-4714` 一带）：

```cpp
// ---- the smem side of the same contract, PROVEN not eyeballed --------------
// The bulk destination needs 16 B and every per-slot stride must preserve it.
// `alignas` on the members plus 16B-multiple stage sizes is the whole argument;
// these asserts exist so a tuning change (kRing, kPackK, a kNTile tweak) cannot
// silently re-open the hole.
static_assert(kARawBytes % 16 == 0, "packed A staging: TMA dst stride");
static_assert(kAbBytes   % 16 == 0, "A operand: MMA descriptor 16B units");
static_assert(kBbBytes   % 16 == 0, "B operand: TMA dst + MMA descriptor");
static_assert(kLboBytes  % 16 == 0 && kSboBytes % 16 == 0, "descriptor strides");
static_assert(__builtin_offsetof(Smem, a_raw) % 16 == 0, "TMA dst base");
static_assert(__builtin_offsetof(Smem, a_op)  % 16 == 0, "MMA operand base");
static_assert(__builtin_offsetof(Smem, b_op)  % 16 == 0, "TMA dst base (B)");
static_assert(__builtin_offsetof(Smem, sf_stage) % 16 == 0, "SF staging base");
static_assert(kPackK % 16 == 0 && kKStep % 16 == 0, "per-copy offsets are k*16");
```

同样的三条加到 `tc5::mxf4`（`kAtomBytes=64`）、`tc5::gemm`（`kSwapabRow` 已有 `:504`）与 `e4x`。
再加一条把「行偏移是 16 的倍数」写成**可见形式**（不改语义，只改可读性/可断言性）：

```cpp
// e4_off returns a byte offset that IS a multiple of 16 by construction: the
// canonical unit is 16B. Written as `16 * unit` so the property is visible at
// the expression level (and so a future edit that adds a +8 is obvious).
__device__ __forceinline__ constexpr int e4_unit(int row, int kb) {
    return (row & 7) + 8 * kb + 16 * (row >> 3);
}
__device__ __forceinline__ constexpr int e4_off(int row, int kb) { return 16 * e4_unit(row, kb); }
```

---

### 层 4（可选项，**不推荐作为主修复**）—— A 侧的「对齐落位」

如果将来真的出现「源不可保证」且无法改布局的情形，**只有 A 侧**存在一个不改变数值的物理办法
（因为 e4 臂的 A 侧有 `a_raw` 这个 packed scratch + expansion pass）：

- 若 `src & 15 == off != 0`：从 `align_down(src,16)` 拷 `align_up(bytes+off,16)` 到
  `a_raw[slot] + off`（`a_raw` 两端各留 15B slack），expansion pass 按 `a_raw[slot] + off` 读。
  **语义不变**（expansion 只是字节→字节的重排），且**没有 fallback 分支**（是同一段代码的一个偏移）。

**B 侧不适用**：`s.b_op[slot][st] + e4_off(0,kb)` 是 **MMA descriptor 的 canonical unit**，
16B 是它的语义，不是可以平移的字节。
⇒ **这是「B 侧的 16B 必须靠布局不变量（层 1）」的硬理由**，也是本设计把重心放在层 1 的原因。

---

## 5. 具体代码改动清单

| # | 文件 | 位置 | 改动 | 风险 |
|---|---|---|---|---|
| C1 | `crates/ferrite-models/src/dsv41/load.rs` | `load_expert_pool` `:641-666` | `block`/`poff` 用 `align_up(128)`；alloc/zero 用 `block * n_routed`；debug_assert 方向修正 | 低（只写 0 字节进补白区） |
| C2 | `crates/ferrite-models/src/dsv41/weights.rs` | 新 pub 项 | `BULK_ALIGN`、`check_bulk_geometry()` | 低 |
| C3 | `crates/ferrite-models/src/dsv41/load.rs` | `Loader::load` 入口 | 调 `check_bulk_geometry(cfg, world)?` | 低（生产形状已通过） |
| C4 | `crates/ferrite-kernel/src/devrt.rs` | `DevBuf` impl `:363` 一带 | `u8_aligned_at(off)`（debug_assert） | 无（release 零开销） |
| C5 | `crates/ferrite-models/src/dsv41/chain_dev.rs` | `:11458`、`:11460`、`:16160` | 改走 `u8_aligned_at` | 极低 |
| C6 | `kernels/cuda/dsv41_experts_mxf4.cu` | `~:4425` | `BulkAlign` + `bulk_align_ok()` | 低 |
| C7 | 同上 | `:4471-4475`（mxf4） | **补 `act`** | 低 |
| C8 | 同上 | `:6163-6167`（grouped） | **补 `bh_base/bhs_base/bh_stride/bhs_stride`** | 低 |
| C9 | 同上 | `:5795-5796`（dense） | 补 8 个 stride | 低（arm 当前不发） |
| C10 | `kernels/cuda/dsv41_kernels.cu` | `:6361-6387`（swapab） | **补 `al16(a)`、`al16(w)`** | 低 |
| C11 | `kernels/cuda/dsv41_experts_mxf4.cu` | 各 `Smem` 之后 | 层 3 的 `static_assert` + `e4_unit/e4_off` 形式 | 无（编译期） |
| C12 | 同上 + `dsv41_kernels.cu` | 装载路径 | `DSV41_ALIGN_AUDIT=1` 的一次性审计（§6 V2） | 低（默认关） |

**不改**：任何 kernel 的数值路径、任何转义/解码逻辑、任何 epilogue。
C1 只改变**权重在显存里的地址**，不改变**读到的是什么字节** ⇒ 期望 replay 文本逐字节一致。

---

## 6. 验证方法

### V0（先跑，5 分钟，无任何代码改动）——把「门」和「漏项」分开

```bash
# 1) 这一轮真实的 env（不是 shell 的 env）
ssh ubuntu@<node> "tr '\0' '\n' < /proc/\$(pgrep -x ferrite-serve|head -1)/environ | grep -E '^DSV41_|^FERRITE_' | sort"

# 2) 用 CUDA_LAUNCH_BLOCKING=1 把 fault 归到真正的那次 launch
export CUDA_LAUNCH_BLOCKING=1
./target/release/ferrite-serve --model dsv41 --serve --tp 8 --model-dir <...> --port 8699
# 发一个 "你好" max_tokens=20
```

**判据（本设计最关键的二分）**：
- 报 `cuda error 1`（`invalid argument`，由 `kerr` 抛出，带 `dsv41_expert_tcgen05_gate_up_e4m3: cuda error 1`）
  ⇒ **门被触发**：一个**未补全的门**（C7/C8/C10）或 `act` 视图错位。
  对照 §3 的矩阵即可立刻定位到哪个参数——**这时 C7-C10 就是修复**。
- 报 `err 716 misaligned`，且上下文是某个 expert/tcgen05 kernel
  ⇒ **门没覆盖到**：按 §1/§2 表逐站点对照，用 `-lineinfo` + `--destroy-on-device-error kernel`
  拿到 `<kernel>:<行>:<指令>`；若是 `CPASYNC.BULK` ⇒ 走 C1-C3（层 1）。

### V1 编译期

- 层 3 的 `static_assert` 全绿即证明 smem 侧（`bash kernels/cuda/build.sh 103a`）。
- `cargo build --release` + `debug_assert` 生效的 `cargo test`（`check_bulk_geometry` 的单测：
  用 `dim=5120/inter=2304/world=8` 期望 `Ok`，用 `dim=6016`（`nsf=188`）期望 `Err`）。

### V2 装载期审计（新增，本设计的关键新增能力）

`DSV41_ALIGN_AUDIT=1` 时，`Loader::load_expert_pool` 在 pool 建成后遍历
「每层 × 每 expert × 6 plane」打印并校验：

```
[align] L0 e0 w1   base&15=0 stride%16=0 pitch%16=0
[align] L0 e0 w1s  base&15=0 stride%16=0 nsf%16=0
...
[align] L0 pool   block%128=0 poff=[0,51200,819200,...]
```

任一非 0 → `Err`（**在 prefill 之前、在 40GB 权重加载后、秒级**）。
这一条把「191ms 才炸、且报告不可归因」变成「装载期一行日志」——
**它是本设计里性价比最高的一个改动**，因为它对**所有** arm 生效，不依赖任何 arm 的具体检查。

### V3 独立 harness（复用现有的 `kernels/cuda/tests_tcgen05_misalign_repro.cu`）

现有 harness 已经用「每 plane 64B slack + 4/8/12 字节 slip」直通 kernel（绕过门）构造了：
- `control(slip=0)` / `slip=4/8/12`（各 plane）× `entry`（走门，作为对照）。
**本设计只需增补两类 case**：
- `--case layout` ：pool/plane 全部按 C1 的 `align_up` 规则构造 ⇒ **期望 0 fault**（正证据：层 1 足够）；
- `--case stride-slip` ：`block` 加 8 字节（模拟「裸累加」被改坏）⇒ **期望 门拒 + 诊断**（正证据：层 2 有效）。

配 `compute-sanitizer --tool memcheck --destroy-on-device-error kernel --show-backtrace device`
（plan §3.2/§4），确认指令类别是 `CPASYNC.BULK` 还是 `LDG.E.128`。

### V4 生产定版

`bash scripts/dsv41_tcgen05_mxf4_verify.sh` + `scripts/tcgen05_bench.sh`；
serve 端 replay 文本 parity（期望**逐字节相同**，见 §5 的「只改地址不改字节」论证）。

### V5 回归门槛（建议加进 CI）

`FERRITE_ALIGN_STRICT=1` + `DSV41_ALIGN_AUDIT=1` 跑一次冒烟 + 一次 replay；
任何一个新的破环都会**在 CI 里炸**而不是在 191ms 处。

---

## 7. 需要批准 / 上报的「方案外」项

1. **C7-C10 是会改变「跑到哪个 arm」的**：补齐后，原先静默 decline 的 rank 会**开始跑 tcgen05 臂**。
   若故障真是「漏项 + 错位」，它会让错位从「静默」变成「门拒」——
   **这是可观测性的改进，不是性能承诺**，请确认这符合本轮的验收口径。
2. **`FERRITE_ALIGN_STRICT` 与「门失败 = decline 还是 error」的策略**：本设计默认保持
   `cudaErrorInvalidValue → Rust 视为错误`（现状），但**建议**在上报链上加一句：
   `[align]` 诊断一旦出现，`chain_dev.rs` 的 decline 路径要打「arm declined by ALIGNMENT」而不是
   「shape declined」，否则两类 decline 混在一个 `Ok(false)` 里（本轮就是被这个坑住的）。
3. **`ALIGN = 128` vs `16`**：128 会改变 `w1_stride` 的具体数值（都不影响 kernel 的 `str16` 检查）。
   若希望**最小化**地址面变化，可只用 16（代价同样可忽略）。请选定一个。
4. **`-lineinfo`**（plan §8.1）仍然需要：本设计的 V0 二分依赖「kernel 名 + 行 + 指令」。
   若 `-lineinfo` 未获批，V0 只能给到「门 vs 716」，V3 的归因仍可做（harness 秒级）。
5. **不在本设计内、但应记一笔**：`e4x` dense 的 `e4x_tile=false` 硬编码
   （`chain_dev.rs:11523`）意味着 C9 现在**测不到**——建议 C9 与「打开 e4x_tile」一并排期，
   否则它是一条没有执行证据的防线。

---

## 8. 一页执行顺序

```
V0（不改代码，5 分钟）        CUDA_LAUNCH_BLOCKING=1 复现 → 二分「门触发 vs 716」
                                ├─ cuda error 1  ⇒ 未补全的门 → C7/C8/C10（+ act 视图检查）
                                └─ err 716       ⇒ 门未覆盖   → C1-C3（层 1）+ V2 审计定位

层 1（根）  C1 align_up(block/poff) + C2 check_bulk_geometry + C3 装载期调用
            C4/C5 给 Rust 的 view 算术加约束（debug_assert）
层 3（证明）C11 static_assert（先做，成本为零，把 smem 侧永久钉死）
层 2（门）  C6 共用表 + C7-C10 补齐四缺三 + 一次诊断 + FERRITE_ALIGN_STRICT
审计        C12 DSV41_ALIGN_AUDIT（装载期，全 arm 生效）

V1 编译期 + 单测 → V2 装载期审计 → V3 harness（layout / stride-slip）→ V4 生产定版 → V5 CI 门槛
```

**一句话**：这一轮的修复**不是**「再加一条 fallback」，而是**让 16B 成为布局的构造性事实**——
`block`/`poff`/`pitch` 三处在装载期对齐并断言（层 1）、smem 侧用 `static_assert` 钉死（层 3）、
门补齐并停止沉默（层 2）、装载期审计把「191ms 才发现」提前到「装载期一行日志」（V2）。
byte-fallback 在第 1/2 轮的失效不是实现问题，而是**它作用在了错误的层**：
bulk 的 16B 是**布局与语义**问题，必须在布局层解决。

---

*工部 · 只读调查 + 本文件（唯一产出）；未执行任何 GPU 命令、未改动任何源码。*
*全部 bulk 站点、launcher 门的覆盖矩阵、`Smem` 的 `alignas`/步长、Rust 侧 `block`/`poff` 的裸累加、
`kerr`/`Ok(false)` 的失败模式，均已对工作树 HEAD 现场核对；
C1-C12、`ALIGN` 取值、`FERRITE_ALIGN_STRICT` 策略明确标注为「待批准」。*
