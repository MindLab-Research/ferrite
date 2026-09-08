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

## Known-good demo numbers (watch for regressions)

- MTP 出师表 200-step: `real 476 tokens` window, text must be flawless 《出师表》 through 将军向宠 section (乱码 = accept/commit bug, ALWAYS check by eye).
- 500-step: 58.9 tok/s (DSA decay visible), non-MTP 500-step 44.7.
- If accept rate collapses to exactly 1.0 with NCCL fallback → env missing NCCL_NVLS_ENABLE=0.
