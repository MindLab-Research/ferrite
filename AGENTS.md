# ferrite — Agent Working Guide

Rust-native inference engine for **GLM-5.3-Flash** (hybrid GatedDeltaNet linear attention + DSA sparse attention + MoE), single-node TP over CUDA graphs.
Read `README.md` for the design contract; this file is the operational guide: build/test loop, every runtime flag, demo configs, and the profiling workflow that actually works on this hardware.

## Repo layout (hot paths)

```
crates/ferrite-exec/src/tp.rs        TP cluster + mega-graph chain + MTP step (mtp_step, mega_chain_dev)
crates/ferrite-kernel/src/cuda.rs    CudaBackend: FFI, graphs, GDN/DSA/MoE device chains, MtpState
crates/ferrite-serve/src/main.rs     binary: load checkpoint → prefill → decode loop (one-shot)
kernels/cuda/ferrite_kernels.cu       ALL device kernels (sm_103a), build.sh → libferrite_kernels.so
```

## Build / deploy / test loop

Local (no GPU needed): `cargo check` (hard gate — every commit).
Remote: b300-4 `ssh ubuntu@43.202.208.136`, repo `~/ferrite`, model `/opt/dlami/nvme/models/GLM-5.3-Flash`, GPUs 4–7.

```bash
# local → remote (the remote has TWO remotes; origin's fetch refspec only tracks main):
git push origin main:perf-b1
# remote:
ssh ubuntu@43.202.208.136 'cd ~/ferrite && git fetch origin refs/heads/perf-b1 && git reset --hard FETCH_HEAD && \
  cd kernels/cuda && bash build.sh 103a && cd ~/ferrite && source ~/.cargo/env && cargo build --release'
```

`bash build.sh 103a` = sm_103a (B300). `103a` is required; the README's `100a` is stale.

## Environment flags

### Required for full-speed decode (the standard MTP run)

```bash
NCCL_NVLS_ENABLE=0 \
FERRITE_MEGA=1 FERRITE_NCCL=1 FERRITE_P2P=1 FERRITE_WORKER_POOL=1 \
FERRITE_LAYER_DEV=1 FERRITE_GDN_DEV=1 FERRITE_MOE_DEV=1 FERRITE_DSA_DEV=1 FERRITE_HEAD_DEV=1 \
FERRITE_MTP=1 \
CUDA_VISIBLE_DEVICES=4,5,6,7 \
LD_LIBRARY_PATH=$HOME/ferrite/kernels/cuda \
./target/release/ferrite-serve --backend cuda --tp 4 \
  --model-dir /opt/dlami/nvme/models/GLM-5.3-Flash \
  --lib kernels/cuda/libferrite_kernels.so \
  --max-tokens 500 --prompt "请背诵《出师表》"
```

- `NCCL_NVLS_ENABLE=0` is MANDATORY on this node: without it `ncclCommInitAll` fails → silent fallback to host all-reduce → ~2.4× slower step and MTP accept degrades (looks like a logic bug but is env).
- `FERRITE_P2P=1` (2026-09-08 VERIFIED — was previously rejected): P2P one-shot AR (10µs vs NCCL RING_LL 35µs per AR). The earlier accept crash (2.39→1.71) was an OLD-version bug (fixed by the mtp_step N-unification series); current: n=3 500-step 121.8 tok/s @ accept 2.43 (vs NCCL 110.2), text flawless, n=1 90.1 (no regression — P2P spin jitter offsets the small-n AR gain). The P2P sum order (staging r0+r1+r2+r3) matches NCCL ring's 4-rank single-chunk order — 1-ulp domain-consistent, draft+verify chains both P2P.
- Non-MTP regression: drop `FERRITE_MTP=1` (everything else identical). Baseline ~49 tok/s @200-step window, MTP ~65.
- Output line to read: `[serve] decode: N steps in Ts = X steps/s | real M tokens = Y tok/s`. With MTP, `real` counts accepted tokens (N×~2.4); always ALSO eyeball the generated text (乱码 must be caught by eye, never by token counts).

### Execution-path flags (all default OFF; 1 = enable)

| flag | effect |
|---|---|
| `FERRITE_MEGA` | one CUDA graph per seq for the whole decode step (mega-graph); the decode fast path |
| `FERRITE_NCCL` | NCCL TP all-reduce inside the captured graph |
| `FERRITE_WORKER_POOL` | persistent fan_out worker threads (no per-step thread spawn) |
| `FERRITE_LAYER_DEV` | per-layer device op chain (hc/norm/attn/ffn stay on GPU) |
| `FERRITE_GDN_DEV` / `FERRITE_MOE_DEV` / `FERRITE_DSA_DEV` / `FERRITE_HEAD_DEV` | device kernels for the respective layer types / lm_head+argmax |
| `FERRITE_MTP` | speculative decoding: draft=2 (layers.45 nextn), verify n=3 mega_v graph, greedy accept, single-kernel `ferrite_mtp_commit` |
| `FERRITE_DRAFT_GRAPH` | per-draft draft-chain graphs `mega_d{seq}_{i}` — **default ON** (0 = host chain fallback; 2 = device chain host-serial bisect mode). Each graph = cast_store + embed_one_dev + the full mtp_forward layer chain; replay = H2D 4B + dsa advance(1) + ONE launch per draft, with a stream sync between graphs (the pinned t0 slot is shared host memory). |
| `FERRITE_MTP_N` | MTP verify width N (drafts = N-1), 1..=8, default 3. N-UNIFIED: the draft chain runs N-1 mtp_forward steps, the verify graph runs N rows (`mega_v` captured at n=N), accept k ∈ 1..=N, commit snapshots are [N-1] contiguous. N=3 = the historical (d1,d2) chain, bit-identical. **N=1 = plain decode on the SAME mega1 path** (no MTP buffers, no ping-pong, no commit — FERRITE_MTP_N=1 with FERRITE_MTP=1 routes to the non-MTP decode_step_mega branch; zero overhead, output bit-identical to FERRITE_MTP unset). `FERRITE_ZERO_H2D=1` (experimental device-resident path) is N=3-only; other N values route to the generalized `mtp_step` with a log line. |
| `FERRITE_NCU` | cuProfilerStart/Stop window around the decode loop — pair with `ncu --profile-from-start off` so the 80s weight load is NOT profiled (ncu intercepts each H2D/kernel launch at ms cost; profiling the load stalls it 10+ min). See Profiling below. |
| `FERRITE_P2P` | NVLink P2P all-reduce (experimental path) |
| `FERRITE_GRAPH`, `FERRITE_GRAPH_LAYER`, `FERRITE_GRAPH_MOE`, `FERRITE_GRAPH_MID`, `FERRITE_GRAPH_DSA` | legacy per-segment graph capture (superseded by FERRITE_MEGA; kept for bisection) |

### Diagnostic flags

| flag | effect |
|---|---|
| `FERRITE_MTP_TIMING` | per-step `[mtp-tm] draft= verify= commitN=` wall times |
| `FERRITE_MTP_DEBUG` | per-step accept decision (d1/d2 vs argmax) |
| `FERRITE_TIMING` | mega replay per-step wall + per-layer DRY segment times |
| `FERRITE_MEGA_DRY` | skip graph capture, run the real chain per step (graph-bug bisection) |
| `FERRITE_MEGA_PROBE` | dump layer intermediates to `/tmp/orion` (host download — capture-illegal) |
| `FERRITE_GDN_PROBE` / `FERRITE_AR_PROBE` / `FERRITE_TRACE_NAN` / `FERRITE_TRACE_TOK` / `FERRITE_TRACE_MOE` / `FERRITE_PROBE` + `FERRITE_PROBE_DIR` | per-layer numerical probes |
| `FERRITE_PPROF` + `FERRITE_PPROF_OUT` | built-in 1 kHz CPU flamegraph (pprof crate) |
| `FERRITE_KERNEL_SO` | override dlopen path of `libferrite_kernels.so` |

## Performance state (perf-b1, 2026-09-08)

- Non-MTP baseline (n=1): **114.0 tok/s** (HTTP+SSE, 1200-token output, head/tail-trimmed 20/20; replay 8.89 ms/step, TP8). Was 91.2 (200-step one-shot) before the split-K + occupancy series.
- **B=16 concurrency (batched path, 2026-09-08 late): 314-319 tok/s end-to-end (8000 tok SSE), 441 tok/s steady replay 16 seqs = 36.3 ms/step.** Was 74 tok/s before the batch-size padding series (SGLang-style):
  - Graph keyed by PADDED size `megab_b{1,2,4,8,16,32}` (was per-composition → re-capture on every membership change, 1-2 s each). Per-seq pointer tables keyed (layer|family, SIZE) with STABLE device addresses, content H2D-refreshed only on membership change; padded rows (u64::MAX) use a shared dummy state (DSA dummy must span MAXT=8192 tokens — 1-token dummy faulted Xid 13).
  - **Inside capture, NO H2D on the tables** (a captured async memcpy node replays against freed stack → Xid 13; sync H2D inside capture → err 900). The dry-run fills, the capture reads.
  - Graphs are KEPT across retires/B=1 ticks (tables are content-refreshed, not embedded) — destroying cost a re-capture per membership change.
  - `--max-seqs` default was 4 → raise to 16 for the B=16 benchmark.
  - **P2P AR deadlocks at batched n>8** (dev0 at dry-run L0, peers at L35 — capture serializes ranks; epoch reset after dry-run did NOT fix). NCCL fallback works (drop FERRITE_P2P for batched runs). NCCL AR costs 3.7 ms/step at 16 seqs (90×41 µs) — the single biggest kernel.
  - `P2P_AR_MAX_N` was 16*4096=65536 < 16×5120=81920 → raised to 16*8192.
- **B=16 nsys per-step (steady, NCCL)**: AR 3.7, gemv_bf16_nt 3.0, moe_down 2.9, indexer_topk 1.7, matmul_tiled 1.7 (mostly prefill), hc_pre 1.6, moe_act 1.5, sparse 1.25, gdn_chunk 1.2, gemv_fp8 1.1 ms. **GPU is saturated at steady state (bucketed nsys: ~35.5 ms kernel per 36.3 ms step) — the host gap story was a measurement artifact (admission-diluted averages); optimize KERNELS, not host.**
- Next targets for 1600 tok/s @16 (need 10 ms/step): AR (P2P fix or fused AR), gemv 3.0 ms, moe_down 2.9 ms, indexer 1.7 ms.
- **SGLang production config (measured 2026-09-08, b300-2 decode 15.164.0.39:30002 / b300-3 prefill 3.37.20.114:30100, ssh -i ~/.ssh/b300-spot.pem)**: NCCL **2.28.3**, default env (no NCCL_NVLS_ENABLE → they do NOT use SHARP either; /dev/nvidia-nvswitch* is absent on these instances), **--enforce-disable-flashinfer-allreduce-fusion** (AR fusion OFF), decode: `--dcp-size 8` + EAGLE (num-steps 5 / draft-tokens 6) + `--cuda-graph-max-bs-decode 64` + kv fp8_e4m3 + HiCache; prefill: `--enable-prefill-cp --cp-strategy interleave --enable-dsa-prefill-cp-layersplit --disable-overlap-schedule`. → SGLang's throughput comes from DCP + EAGLE + big CUDA-graph batches, NOT from a faster AR.
- **AR tuning exhausted (all measured on 16-seq batched, 30.7-31.3 ms/step baseline)**: bf16 payload +2%, NCCL_ALGO=Tree +1%, NCCL_MAX_NCHANNELS=2 −1.5%, NCCL 2.28.3 −0.3%, NCCL_LAUNCH_MODE=GROUP deadlocks the capture. The 223µs/call is the NCCL 8-rank rendezvous floor (~100µs kernel + ~133µs start spread, last-arriving rank rotates). Only structural fixes remain: fewer ARs (EP instead of TP for the FFN) or AR/compute overlap.
- **Two regressions from the TP-occupancy series (2026-09-08, both FIXED by bisect; the text was gibberish/EOS for ~1h and token counts alone never showed it):**
  1. `6ab9bee` P2P AR parallelization: phase A mapped (token, peer) onto threads but reused `i` as the peer index INSIDE `if (i < n)`, with grid = (n+1023)/1024. At decode (n=1) only thread 0 ran → wrote peer 0's staging slot alone, yet stamped EVERY peer's ready flag → every other rank read the PREVIOUS epoch's staging value → wrong all-reduce → deterministic gibberish at every layer. FIX: phase A maps r0 → (token = r0/world, peer = r0%world) over n*world threads; launcher grid covers n*world. LESSON: when parallelizing a serial loop, the thread index must be decomposed, not reused.
  2. `HC_P345_NB` 16→64→256 CORRUPTS the output (NB=64 → all-EOS, NB=256 → "the the the"; bisected one-shot 400-tok). NB=16 is the last known-good value and the sweep showed NO perf gain (rest345 12.6µs at NB=16 vs 13.7 at 64 — compute-bound, not occupancy-bound). Root cause of the NB>16 corruption still TBD; **do not raise HC_P345_NB without re-validating the 出师表 text**.
- MEMSET elimination (2026-09-08): nsys `CUPTI_ACTIVITY_KIND_MEMSET` counted **128 4-byte `cudaMemsetAsync(ctr2)` per decode step** (hc_pre, 2/layer). Folded the zeroing into `hc_pre_mix_split_kernel` block 0 (rest345 is a later kernel on the same stream, so the write is visible before any atomicAdd) → replay 9.32→8.89 ms (**+6%, 107→114 tok/s**). General rule: **count CUPTI MEMSET/MEMCPY rows in nsys, not just kernels** — per-step GPU ops that are not kernels are invisible in cuda_gpu_kern_sum.

- MTP (draft=2, verify n=3): **123.1 tok/s @500-step** (113.4 @200-step), accept 2.46, text flawless.
- Per-step (200-step window, ~20.5 ms → 48.8 steps/s): verify ~16.6 ms + draft ~1.6 ms + commit ~0.0 ms (one kernel) + host ~0.5 ms + fan_out ~1.7 ms.
- MTP chain: A→B copy-in is recorded INSIDE the mega_v graph; accept commit is ONE kernel (`ferrite_mtp_commit`, k read zero-copy from pinned slot); draft chain = per-draft graphs `mega_d{seq}_{i}` (DEFAULT ON since 2026-09-08; `FERRITE_DRAFT_GRAPH=0` falls back to the host chain).
- **Draft-graph alias bug (2026-09-08, FIXED)**: `mtp_forward_raw_argmax` did `forget(h_out.as_ref())` — forgetting a `&DevBuf` is a NO-OP, so the owned `Option<DevBuf>` still dropped and returned MtpState's `h_d[i]` address to the pool; the NEXT call's enorm/hnorm allocs aliased it and overwrote the draft h relay → accept 2.39→1.93 (graph path) / 1.00 (host-serial device chain). FIX: `std::mem::forget(h_out)` (the owned value). Same-function `zero_h2d` path was affected too. Detector: `FERRITE_POOL_DEBUG=1` prints `[pool-dup] DOUBLE RELEASE` on pool double-release.
- Inter-graph sync in the draft replay loop is REQUIRED: the pinned t0 slot is shared host memory, so without a stream sync between graph launches the next `dsa_host_advance` lands before the just-launched graph's DSA kernels read it (measured accept 2.39 with sync vs 2.22 without).
- Debug knobs: `FERRITE_DRAFT_GRAPH=2` runs the device draft chain host-serial (bisect mode); `FERRITE_DRAFT_DRY_ONLY=1` skips the capture pass (dry-only bisect).

## Profiling (what works on this machine)

**LLM-serve kernel breakdown: nsys 2025.6.3** (`/usr/local/cuda-13.2/bin/nsys` — the CUDA 13.2 bundled one; the 2024.2.3 apt version still does NOT work: zero CUDA rows). Verified 2026-09-07 on a 3-kernel mini program (CUPTI injection healthy, cuda_gpu_kern_sum reports correct). ~5-10% overhead, full timeline, works with CUDA graphs:

```bash
sudo /usr/local/cuda-13.2/bin/nsys profile --trace=cuda --cuda-graph-trace=node \
  --sample=none -o /tmp/nsys_out --force-overwrite=true \
  env NCCL_NVLS_ENABLE=0 FERRITE_MEGA=1 FERRITE_NCCL=1 FERRITE_P2P=1 FERRITE_WORKER_POOL=1 \
  FERRITE_LAYER_DEV=1 FERRITE_GDN_DEV=1 FERRITE_MOE_DEV=1 FERRITE_DSA_DEV=1 FERRITE_HEAD_DEV=1 \
  CUDA_VISIBLE_DEVICES=4,5,6,7 LD_LIBRARY_PATH=$HOME/ferrite/kernels/cuda \
  ./target/release/ferrite-serve ... --max-tokens 20 ...
sudo /usr/local/cuda-13.2/bin/nsys stats --report cuda_gpu_kern_sum /tmp/nsys_out.nsys-rep
```

`--cuda-graph-trace=node` expands mega-graph replays into individual kernels in the report. Log FULL output to a file (never `grep|head` pipes — they swallow progress/errors and you can't see which stage hung). Validate the toolchain on the 3-kernel mini first when in doubt: `kernels/cuda/ncu_miniprof.cu` (cudaProfilerStart/Stop window) exists for exactly this.

**ncu: NEVER profile the whole serve with it — it WILL time out (burned ~40 min / 5+ failed attempts 2026-09-07).** ncu's injection overhead is 10-100× per CUDA call; the pre-decode phases alone (mmap load = 38287 cudaMemcpy API interceptions + mega-graph capture with 900+ kernel dry-runs) exceed any sane timeout — the profile window (FERRITE_NCU, decode step ≥1) never even starts (CSV contains only `==PROF== Connected` + `==ERROR== ... 124`). kernel-name filters and launch-count caps do NOT help: the overhead hits pre-window API calls too. ncu IS fine for single-kernel micro-benchmarks (mini program verified) and for known-hotspot deep dives launched outside the serve. In-serve per-kernel time = nsys; kernel deep-dive (SOL/occupancy) = ncu on an isolated repro of that kernel (see tests/bf16_widen_gpu.rs for the Rust-side harness pattern).

Other profiling rules:
- GPU-side timing events inside a captured CUDA graph do NOT work (`cudaEventElapsedTime` on graph-recorded events returns InvalidValue — sync-only; the dead `FERRITE_MEGA_EVTS` code was removed for this reason).
- `FERRITE_TIMING` per-layer numbers (at=/mid=/ffn=) are HOST-side Instant deltas around launch+sync — useful for RELATIVE layer comparison only; never quote them as kernel GPU time (they overstate ~10-20× and misled the "DSA at=7.5ms" hunt; the real per-step budget is replay ~15.6ms/45 layers = ~350µs/layer).
- CPU side: `FERRITE_PPROF=1 FERRITE_PPROF_OUT=serve.svg` (flamegraph over load+decode).

## GPU test discipline (hard rules)

1. **Exclusive GPU runs**: before starting a serve, `pgrep -af ferrite-serve` + `nvidia-smi --query-compute-apps` must be EMPTY on GPUs 4–7. A leftover panicked serve holds ~160 GB/GPU and breaks the next run's NCCL init.
2. **Never kill a GPU process while another test is running on the same GPUs** — NCCL bootstrap resources are shared; killing a zombie peer can hang the live run's collectives (observed: decode frozen mid-text, log mtime stale).
3. Kill by exact PID (`kill -9 <pid>`), never `pkill -f` (it matches your own ssh bash command line → suicide, exit 255).
4. A background run whose log mtime has not moved for minutes is hung — check `stat -c %y` + `pgrep`, don't wait blindly.
5. After `kill -9`, GPU memory release is asynchronous (defunct + CUDA context teardown can take ~30 s) — confirm `nvidia-smi` shows 0 MiB before the next run.
6. serve is one-shot: exits via `std::process::exit(0)` (exit-time drop of 1.17 TB weights SEGFAULTs → EXIT 139, which also loses profiler buffers).

## 2026-09-08 会话结论：通信不是瓶颈（A/B 实测），计算才是

**用户目标**：16 并发 ≥1600 tok/s（不开 MTP）；若用 MTP 则目标 3200。**当前：18.8 ms/步 = 851 tok/s**（会话内从 546 提升 **+56%**，每次改动都人眼验证文本）。

**决定性 A/B（同一 16-seq 负载，`FERRITE_AR_SKIP=1` 跳过全部 AR）**：跳 AR 22.05 ms vs 带 P2P AR 23.06 ms → **通信只占 1.78 ms = 7.5%**。32 并发同样（NO-AR 95.5 vs 102.8）。**"通信占 64%" 是陈旧读数**（`p2p_ar_publish` 的 max=55.6ms 是 dry-run/capture 期自旋超时；其稳态中位数只有 5.1µs）。

**本会话的有效 kernel 优化（全部文本验证，累计 29.33→18.8 ms）**：
1. **moe_fused_down 16 字节 lane**（每 lane uint4 覆盖 2 个连续 h 行，klen=256）：**7.5 → 1.9 ms**。根因：8 字节 lane 只有 1.16TB/s（request-rate 受限）。坑：scale 列索引必须 `(lane&15)>>3`；补丁内重复声明 `py[8]` 会遮蔽外层 → 输出全零。
2. **moe_fused_act padded per-warp smem staging**：A 片段 4B 跨 8 行 = 每指令只用到 32B sector 的一半。暂存 16 行 × 64 列 × 2 投影，**行距 padding 到 80B**（64B 会让每行落 bank 0 → 8 路冲突，反而慢）。act 74 → 62.8 µs。
3. **indexer 8 线程/pool**（原来是每线程串行 4096-MAC 点积，GPU 利用率 ~1%）：中位 100 → 32 µs。**shuffle 必须用 8-lane 组掩码**（0xffffffff 会让没有 pool 的组跳过 → 捕获死锁）。
4. **sparse_attn 输出 float4 + 4 槽位分组**（原来 4 字节读、跨槽 64KB 零合并）：121 → 51 µs（2.4x）。
5. **hc_pre_mix 4 行/block**（x 原来被 24 个 mix 行各读一遍 → L2 受限）：11.8 → 9.5 µs。
6. **gemv_fp8 8 行/组 + x 寄存器缓存**（x 原被每 (block,row) 重读）：51.7 → 41.7 µs。
7. sparse_attn 的 live_k 边界（原按固定 select_k_max=2048 循环）+ indexer 短上下文快路径。
8. dense-FFN / GDN host 路径的 AR 改 P2P 优先；run_matmul 走快路径。

**已验证无效/更差（勿重复）**：gemv row-major（**权重矩阵 1-3MB 全在 L2 内，token-major 的 16x 重读是 L2 命中，不是 DRAM 瓶颈**——实测 18.15 vs 17.99ms）、down TT=1/8/16、ROWS=16/64、gemv R=8（无 x 缓存时）、gemv 1024 线程/block、K 循环 unroll 4、token 循环 unroll 2、去掉 fp8 转换链（仅省 4.5%）、act 的共享 sa staging、act 的 128 列 staging（18.93ms，占用率下降）、HC_MIX_KS 8→2。

**教训（乱码=数值回归，且可能是"少算"）**：gemv row-major 曾测得 14.78ms/1082 tok/s 但输出 `!!!`——因为 launcher 的 grid 硬除 `rpb*8` 而 kernel 在 `nrows%8!=0` 时用 R=1，**只算了 1/8 的输出行**。**任何提速都必须同时人眼验证文本，否则"少算"会被误当成"优化"。**

**当前每步每卡分解（nsys，~300 步反推）**：p2p_ar_publish ~1.8（含 barrier）/ **moe_act 3.4** / matmul_tiled_bf16 3.3（多为 prefill）/ **gemv_fp8 2.4** / **moe_down 2.1** / **indexer 1.6** / hc_rest345 1.34 / hc_mix 1.04 / kpool 1.0 / gdn_chunk 1.0 / gdn_step 0.6 / sparse_attn 0.6 / NCCL AR 残余 0.64。

**下一步（按预期收益）**：① moe_act 仍 62% 峰值（3.4ms）——试 128 列 staging 或 2-token/block；② indexer 的 139µs 平均（中位 32µs，长尾待查）；③ kpool_compress 中位 114µs（11 次/步）；④ gdn_chunk/step 1.6ms；⑤ AR：publish 中位 5.1µs×90=0.46ms，可试 per-block flag 省掉 publish kernel。

## B=16 通信 vs 计算：实测推翻旧结论（2026-09-08 late, perf-b1）

**用户目标**：16 并发 ≥1600 tok/s（不开 MTP）；若用 MTP 则目标 3200。**当前：21.2 ms/步 = 755 tok/s**（会话内从 546 提升 +38%，每次改动都人眼验证文本）。

**决定性 A/B（同一 16-seq 负载，`FERRITE_AR_SKIP=1` 跳过全部 AR）**：跳 AR 22.05 ms vs 带 P2P AR 23.06 ms → **通信只占 1.78 ms = 7.5%**。32 并发同样（NO-AR 95.5 vs 102.8）。**"通信占 64%" 是陈旧读数**（`p2p_ar_publish` max=55.6ms 是 dry-run/capture 期自旋超时）。

**本会话落地的有效 kernel 优化（全部文本验证通过，累计 29.33→21.2 ms）**：
1. **moe_fused_down 的 16 字节 lane**（每 lane uint4 覆盖 2 个连续 h 行，klen=256）：**7.5 → 1.7 ms（4x）**。关键认知：该 kernel 原为 request-rate 受限（8B lane 只跑 1.16TB/s = 15% 峰值）。踩过的坑：scale 列索引必须用 `(lane&15)>>3`（lane 16-31 读的是第二行）；补丁里重复声明 `py[8]` 会遮蔽外层变量 → 输出全零。
2. **moe_fused_act 的 padded per-warp smem staging**：A 片段 4B 加载跨 8 行 = 每指令只用到 32B sector 的一半（2.3x 字节浪费）。暂存 16 行 × 64 列 × 2 投影，**行距必须 padding 到 80B**（64B 会让每行落在 bank 0 → 8 路冲突，实测反而慢 1.6ms）。→ act 74 → 62.7 µs/次。
3. **sparse_attn 的 live_k 边界**（原按固定 select_k_max=2048 循环，实际槽位 ~112）。
4. **indexer_topk 快路径**（select_k ≥ jmax 时跳过 O(k·n) 选择）。
5. **dense-FFN / GDN host 路径的 AR 改 P2P 优先**（原来直调 NCCL）。
6. 8 行数据预取到寄存器 + shuffle 归约外提 + TT=4 tokens/block + 去掉阻止展开的 `break`/守卫。

**已验证无效/更差（勿重复）**：gemv row-major、down TT=1/8/16、ROWS=16/64、gemv R=8 行/组、gemv 1024 线程/block、K 循环 unroll 4、token 循环 unroll 2、去掉 fp8 转换链（仅省 4.5% → 非转换瓶颈）、act 的共享 sa staging（无 padding 或 4KB/warp 版）。

**当前每步每卡分解（nsys，367 步反推）**：p2p_ar_publish 1.3-2.5（含 barrier 等待）/ moe_act 3.2 / matmul_tiled_bf16 2.7（多为 prefill）/ gemv_fp8 2.5 / indexer 1.9 / moe_down 1.7 / sparse_attn 1.7 / AR 残余 / hc 2.2 / kpool 0.86 / gdn_chunk 0.82 / gdn_step 0.48 / route 0.47。

**下一步**：① matmul_tiled_bf16 是 32×32 FMA（无 tensor core），确认是否为 decode 路径，是则改 cuBLAS/MMA；② indexer/sparse_attn 的 199/180µs 中位（短上下文下应更快）；③ hc 两个 kernel 2.2ms（每 block 仅 256 元素）；④ AR 的 90 次 barrier（49µs/次，理论 11µs）——可试 per-block flag 省掉 publish kernel。

## B=16 通信 vs 计算：实测推翻旧结论（2026-09-08 late, perf-b1）

**A/B 实测（同一 16-seq 负载，`FERRITE_AR_SKIP=1` 跳过全部 AR）：**

| 配置 | replay/step | aggregate |
|---|---|---|
| 跳过 AR（纯计算） | 24.88 ms | 643 tok/s |
| NCCL AR | 30.68 ms | 512 tok/s |
| P2P AR（3-kernel 版） | **26.16 ms** | **611 tok/s** |

→ **通信（AR）只占 4.45 ms = 15%**，计算占 24.88 ms = 85%。此前"AR 20.1 ms/64%"的结论来自一次陈旧/被 capture 期自旋超时污染的 nsys 读数（`p2p_ar_publish` 的 max=55.6ms 就是 500000 次自旋超时，只出现在 dry-run/capture 阶段）。**优化方向必须放在 kernel，而不是继续调 AR 参数。**

**nsys 每步每卡分解（367 步反推，GPU 约 92% 忙）：**

| kernel | ms/步/卡 | 备注 |
|---|---|---|
| moe_fused_down_sum_fp8 | 6.7 → ~4.5（已优化） | token-major、每 token 重读专家矩阵 |
| moe_fused_act_fp8_mma | 3.5 | fp8 MMA，8 warp K-split，效率约 45% |
| matmul_tiled_bf16 | 2.7 | M=16 时 32×32 tile 浪费一半 |
| gemv_fp8_v2 | 2.5 | 已改 row-major（无收益，L2 已吸收） |
| indexer_topk_batched | 2.0 | **top-k 是 O(k·n) 串行全块归约**（k≈live pools） |
| sparse_attn_v2_batched | 1.9 | |
| hc_pre_rest345 + mix + post | 2.4 | 90 次/步 |
| gdn_chunk + gdn_step | 1.3 | |
| NCCL AR 残余 | 1.5 | 仍有 ~4 次/步走 NCCL |
| p2p AR（store+publish+reduce） | 4.45 | 49µs/次 × 90 |
| moe_route | 0.5 | |

**模型规模（GLM-5.3-Flash config）**：hidden 4096、45 层（前 3 层 dense，inter 12288）、**288 路由专家 / topk 8 / moe_inter 2048**、`e_local=288`（**每个 rank 持有全部 288 专家**，专家矩阵按 **inter 切 256**，down 输出完整 4096）、kv_lora 512、q_lora 1536、64 heads × 256、index_topk 2048、MTP 1 层。

**权重读取的物理下限**：每步每卡全量专家权重 42 层 × 288 专家 × 25.2MB / 8 = **38 GB**；但 bs=16 只用 ~104/288 专家 → ~14 GB → 在 7.6 TB/s 下 **~1.8 ms/步**。当前 24.88 ms 计算 = 下限的 ~13 倍（SGLang ~5 倍），说明**冗余读取 + kernel 效率**两个方向都有空间。

## 2026-09-08 会话结论：通信不是瓶颈（A/B 实测），计算才是

**用户目标**：16 并发 ≥1600 tok/s（不开 MTP）；若用 MTP 则目标 3200。**当前：23.1 ms/步 = 693 tok/s**（会话内从 546 提升 +27%，文本每次都人眼验证）。

**决定性 A/B（同一 16-seq 负载，`FERRITE_AR_SKIP=1` 跳过全部 AR）**：

| 配置 | replay/步 | aggregate |
|---|---|---|
| 跳过 AR（纯计算） | 22.05 ms | 726 tok/s |
| P2P AR | **23.06 ms** | **693 tok/s** |

→ **通信只占 1.78 ms = 7.5%**。32 并发同样：NO-AR 95.5 ms vs 带 AR 102.8 ms（AR 仅 7.3 ms）。**"通信占 64%" 是陈旧读数**（`p2p_ar_publish` 的 max=55.6ms 是 dry-run/capture 期自旋超时）。优化必须放在 kernel。

**每步每卡分解（nsys 稳态，367 步反推）**：moe_down 7.5→~5（已优化）/ moe_act 3.5 / gemv_fp8 2.8 / sparse_attn 1.9 / indexer 1.5 / AR 1.8 / hc 2.3 / gdn 1.0 / cuBLAS 1.3 / kpool 0.6 / route 0.5。

**本会话已落地的 kernel 修复（全部文本验证通过）**：
1. **P2P batched 死锁根因**：多 block 时 last-block 检测（`atomicAdd(ctr)`）在图 replay 下失效 → 改 3-kernel（store 多 block / publish 1 block / reduce 多 block）+ 每 block 独立 seen 行 + float4 向量化（AR 从 236ms/步 → 1.78ms）。详见下节。
2. **moe_fused_down**：32 lane × 8B/行（原 16 lane 闲置一半）+ 去掉阻止展开的 `break`/`h<hidden` 守卫 + 8 行数据预取到寄存器 + shuffle 归约外提 + TT=4 tokens/block。6.6 → ~5 ms。
3. **moe_fused_act**：去掉 kk 循环里永不触发却阻止展开的 `break`。
4. **indexer_topk**：短上下文 `select_k >= jmax` 时跳过 O(k·n) 选择循环（集合语义，消费方过滤非法项）。
5. **gemv_fp8 row-major**：无收益且破坏输出 → 已回退（教训：改索引映射必须验证文本）。

**已验证无效/更差的尝试（勿重复）**：gemv row-major、down 的 TT=1/8/16、ROWS=16/64、gemv 的 R=8、K 循环 unroll 4、token 循环 unroll 2、去掉 fp8 转换链（只省 4.5% → 非转换瓶颈）。

**下一步（按预期收益）**：
1. **MoE down 仍是最慢单 kernel（~5ms/步，仅 16% 峰值带宽）**：act kernel（3.5ms）达 46% 峰值，差别是 act 每 block 读 128KB（单专家矩阵内），down 每 block 的 9 warp 各读不同专家的 2KB。**唯一验证过的有效手段是提高每 warp 连续读取量**，但 ROWS 试验反而变慢（原因未明，需 ncu）。建议：expert-major 分组（把同一专家的 token 聚到一个 block）或 cuBLAS grouped GEMM（`cublasGemmGroupedBatchedEx`，需先把权重反量化到 bf16 或用 fp8 分组 API）。
2. **act kernel 的 A-fragment 访存未合并**（8 行 × 16B/指令）→ smem staging 或 16×32 swizzle（TileRT 的 `_swizzle_qmma_16x32` 就是这个）可望 ~1.5x。
3. **indexer/sparse_attn**：长上下文时 top-k 仍是 O(k·n)（基数选择/bitonic 可解）。
4. **AR 1.78ms**：publish/poll 的中位 4.6µs 但均值被 capture 期污染；可试 per-block flag 省掉 publish kernel。

## P2P batched 死锁的完整根因链（2026-09-08，已修复，勿再重复试错）

单序列 P2P 能跑、batched 死锁，曾试过 5 种修法全部无效（epoch reset、单调 seen 协议、ctr 清零、flag 后加 fence、`atomicExch_system`）——**因为它们全部打在已废弃的 `p2p_ar_down_v2_kernel` 上**（`FileReplace` 匹配到第一个出现处）。真正的活跃 kernel 是 `p2p_ar_fused_v3_kernel`。

真实根因（三个叠加）：
1. **last-block 检测失效**：launcher 按 `blocks = ceil(n*world/1024)` 启动（16 seqs → 640 blocks），`gridDim.x != 1` 走 `atomicAdd(ctr)` 路径；注释自陈该路径 ctr 跨 replay 会卡住 → 没有任何 block 认为自己是最后一个 → 永不发布 stamp → 单调等待死锁。旧修复只覆盖了单序列的 `gridDim.x == 1` 分支。
2. **staging 写零合并**：`(token,peer)` 展平（`ii = r0/world, rr = r0%world`）让相邻线程写不同 GPU → 655K 次 4B NVLink 事务/AR，约 2.5ms/次（replay 236ms）。修：外层 peer、内层连续 token，每线程读一个 token 写所有 peer 的连续槽位。
3. **seen[] 共享行竞争**：多 block 同时 poll 同一 peer 时，block A 更新 seen[tr] 后 block B 读到 `prev == cur` → 永远等下一个 stamp（`prev=2 cur=2 myepoch=1`）。修：`seen[MAX_FBLOCKS][world]`，每 block 独立行。

最终结构（3 kernel/AR，全部 stream-ordered，无需任何 grid 级计数器）：
`p2p_ar_fused_v3`（多 block 合并写 staging，block 0 把 epoch 快照写入 ctr）→ `p2p_ar_publish_v3`（**1 block**：block 0 写全部 peer 的 stamp + 一次 `__threadfence_system`，8 个线程 poll）→ `p2p_ar_reduce_v3`（多 block，零轮询）。
**注意**：`__threadfence_system()` 不能放在 655K 个线程里（每个线程一次 → ~2ms/AR）；kernel 边界本身已保证 store 对下一个 kernel 可见。flag 用普通 volatile store + 一次 system fence（vLLM 的做法）即可。

**当前 P2P AR = 49µs/次**（store ~3.5µs + publish 中位 4.6µs + reduce + 3 次 graph node 启动）。仍比理论（~11µs）高 4 倍，是下一个通信优化目标。

## 已知可攻的 kernel 低效点（按收益排序）

1. **MoE act/down 的冗余读取**：token-major 循环让每个 token 单独遍历其 top-k 专家矩阵；bs=16 时 ~104 个专家被 128 次赋值覆盖 → 1.2-1.4x 冗余（不算大），但 kernel 效率只有 25-45% 峰值。已修 down 的 lane 利用率（32 lane × 8B，29.33→26.16ms）。
2. **indexer top-k 是 O(k·n)**：`for r in 0..select_k { 全块归约求 max }`，短上下文 t≈100 → 100 次串行归约 = 212µs/次 × 11 层。长上下文 t=2048 时更糟（DSA decay）。修法：基数选择/bitonic，或降低 select_k。
3. **matmul_tiled_bf16 在 M=16 时浪费一半 tile**（32×32 tile，M=16）；dense 层与注意力投影受影响。
4. **AR 49µs → 目标 <15µs**：减少 graph node 数（3→2）或合并 reduce。

## Known-good demo numbers (watch for regressions)

- MTP 出师表 200-step: `real 476 tokens` window, text must be flawless 《出师表》 through 将军向宠 section (乱码 = accept/commit bug, ALWAYS check by eye).
- 500-step: 58.9 tok/s (DSA decay visible), non-MTP 500-step 44.7.
- If accept rate collapses to exactly 1.0 with NCCL fallback → env missing NCCL_NVLS_ENABLE=0.
- **B=16 non-MTP (2026-09-08 late, the current best)**: `[megab] replay 16 seqs: 16.23-16.42 ms`
  = **975-985 tok/s** (300-token window, text = direct 《出师表》 recitation ✓); 1000-token window
  ~17.9 ms = **893-895 tok/s**. Progression this session: 546 → 985 tok/s.
  Kept changes: gemv_fp8 T=2 tokens/block, gated_rmsnorm 1 token/block x256, moe_act double-buffered
  cp.async.cg staging + ldmatrix, hc_pre_mix K-loop `#pragma unroll 4`, **moe_route 256-thread
  top-k**, **rmsnorm blockDim-sized reduce (was a latent 8-warp bug)**, **4/2-accumulator splits of
  the serial FMA chains (GDN +0.5%, gemv +0.1%)**.
  Run-to-run/thermal drift is ~±1%; always A/B in the same window.
  **Biggest remaining win: MLA absorption** (see `docs/agent/perf-roadmap.md`) — the DSA cache is
  non-absorbed (per-head up-projected k/v), costing ~1.5 ms/step at t=1000 (~8%).

## 2026-09-08 续：half2 FMA 迁移的收益与陷阱（gemv 有效 / moe_down 回退）

**背景**：gemv_fp8_v2 / moe_down 的内层都是 `fp8x2 -> half2 -> float2 -> 2 条标量 FMA`
（每 16 个元素约 40 条指令）。改成 `fp8x2 -> half2 -> __hfma2`（每 16 个元素 16 条）
在 gemv 上生效（17.61 -> 17.51 ms，+0.6%），在 moe_down 上**速度中性（17.51 vs 17.51）但
把模型推入 thinking 模式**（文本从直接背诵《出师表》变成 `<think 嗯，用户要求背诵…`）——
已回退。教训：

1. **`reinterpret_cast<const __half2*>(float_ptr)` 是错的**：这是把 float 的位模式当
   half 读（实测输出全 `!!!!!`，且"变快"3.4 ms —— 少算/垃圾值）。必须 `__floats2half2_rn`。
2. **速度中性但改变输出分布 = 必须回退**：fp16 只在 16 元素 chunk 内累加、随后并入 fp32，
   理论上误差 ~1e-3（远小于 fp8 权重 1e-2），但**实际翻转了模型行为**。数值改动即使"精度够"
   也可能越过 logit 的决策边界 —— 唯一可靠的判据是人眼文本，不是误差估计。
3. 迁移 half2 只对**指令受限且数值敏感度低**的内核安全；MoE 路径（act/down）不要动。

**其他中性/更差的尝试（勿重复）**：gemv WPR=2（17.87）、gemv 内层 unroll 4（17.67）、
moe_route warp-shuffle top-k（17.61，中性但修掉了 `bidx[threadIdx.x]` 越界写 [32] 数组）。

**act kernel 双 tile 预取（2026-09-08，已回退）**：把 `uint4 pf[4]`（1 个 tile）改成
`pf[8]`（2 个 tile，kb 消费 pf[cur] 时重填 pf[cur]=tile(kb+128)）→ **19.6ms（-12%）**，
文本仍正确。根因：pf[8] = 32 个额外寄存器压低占用，收益被抵消。**结论：act 的瓶颈不是
单个 warp 的 load 深度**（单 tile 版已经是 8 warp × 4 uint4 = 32 个在飞的 load），
继续加深度只会伤占用率。

## 2026-09-08 gemv_fp8 的 M 维复用（有效，+1.1%）

`gemv_fp8_v2_kernel` 改成 **每 block 处理 2 个 token**（grid.y = ceil(n/2)，t0 = blockIdx.y*2）：
权重行只 load 一次、fp8->half2 只转换一次，两个 token 共用（每个 token 保留自己的累加顺序
→ 逐位相同）。17.57-17.61 -> **17.40-17.42 ms（919 tok/s）**，文本正确。这是 n=16 下 GEMV
的正确打法（M 维复用），但收益只有预测的 1/3 —— 说明该 kernel 仍有很大比例是延迟而非指令。

**注意编译陷阱**：第一次提交漏了 `has1` 的作用域（声明在 row 循环的 `{}` 内、epilogue 在外），
`build.sh` 报 1 error 而 serve 用的是**旧 .so**，测出 17.62ms 的假结果。**每次必须看 build 输出的
error 数**（本会话第二次踩这个坑）。
