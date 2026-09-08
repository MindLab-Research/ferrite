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
FERRITE_MEGA=1 FERRITE_NCCL=1 FERRITE_WORKER_POOL=1 \
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

## Performance state (perf-b1, 2026-09-06)

- Non-MTP baseline: 49.0 tok/s (200-step window, 出师表 prompt).
- MTP (draft=2, verify n=3): **64.8 tok/s @200-step** (58.9 @500-step — DSA cache O(len) decay, TODO 94), accept ≈2.4 tokens/step.
- Per-step (200-step window): verify ~30 ms + draft ~2.2 ms + commit ~0.0 ms (single kernel) + host ~4.5 ms.
- MTP chain: A→B copy-in is recorded INSIDE the mega_v graph; accept commit is ONE kernel (`ferrite_mtp_commit`, k read zero-copy from pinned slot); draft chain is host-launched `mtp_forward` ×2 (TODO: graph it).

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
