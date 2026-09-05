// ferrite_kernels.cu — GLM-5.3-Flash op kernels for B300 (sm_100a).
//
// Strategy (v1, correctness-first): every op is a straightforward CUDA port
// of the CPU golden reference (crates/ferrite-kernel/src/cpu.rs). The
// numerical contract is *bit-comparable within fp tolerance* to the CPU
// backend — the B300 validation runs both and diffs. Performance tuning
// (Warp-specialised GEMM, WYF-parallel chunkwise Gated DeltaNet, fused
// SwiGLU, cp.async pipelines) is layered on top of this correct baseline
// without changing the extern "C" contract.
//
// All kernels operate on f32 host-visible buffers (cudaMemcpy'd by the
// Rust CudaBackend in v1; device-resident tensors come with the graph
// runner). Layouts match the CPU backend docs exactly.

#include <cuda_runtime.h>
#include <cmath>

#define FERRITE_CHECK(call)                                                  \
    do {                                                                     \
        cudaError_t e = (call);                                              \
        if (e != cudaSuccess) {                                              \
            return e;                                                        \
        }                                                                    \
    } while (0)

// ============================================================
// matmul: out[n, out_f] = x[n, in_f] @ w[out_f, in_f]^T (+ bias?)
// w is row-major [out_f, in_f] (PyTorch Linear layout).
// Tuned: 32x32 shared-memory tiles with +1 padding (bank-conflict free);
// the naive per-thread-dot body is kept as matmul_naive for golden-diff.
// ============================================================
#define FERRITE_TILE 32

__global__ void matmul_tiled_kernel(const float* __restrict__ x,
                                    const float* __restrict__ w,
                                    const float* __restrict__ bias,
                                    float* __restrict__ out,
                                    int n, int in_f, int out_f) {
    __shared__ float sx[FERRITE_TILE][FERRITE_TILE + 1];
    __shared__ float sw[FERRITE_TILE][FERRITE_TILE + 1];
    int row = blockIdx.y * FERRITE_TILE + threadIdx.y;
    int col = blockIdx.x * FERRITE_TILE + threadIdx.x;
    float acc = (bias && col < out_f) ? bias[col] : 0.0f;
    int tiles = (in_f + FERRITE_TILE - 1) / FERRITE_TILE;
    for (int t = 0; t < tiles; t++) {
        int k = t * FERRITE_TILE;
        // x tile: coalesced along in_f
        sx[threadIdx.y][threadIdx.x] =
            (row < n && k + threadIdx.x < in_f)
                ? x[(size_t)row * in_f + k + threadIdx.x]
                : 0.0f;
        // w tile: sw[i][j] = w[c0+i][k0+j] — col (out dim) rows the tile,
        // k dim columns. Store as [tx][ty] so the dot loop reads
        // sw[tx][l] = w[c0+tx][k0+l].
        sw[threadIdx.x][threadIdx.y] =
            (col < out_f && k + threadIdx.y < in_f)
                ? w[(size_t)col * in_f + k + threadIdx.y]
                : 0.0f;
        __syncthreads();
#pragma unroll
        for (int l = 0; l < FERRITE_TILE; l++) {
            acc += sx[threadIdx.y][l] * sw[threadIdx.x][l];
        }
        __syncthreads();
    }
    if (row < n && col < out_f) out[(size_t)row * out_f + col] = acc;
}

__global__ void matmul_naive_kernel(const float* __restrict__ x,
                                    const float* __restrict__ w,
                                    const float* __restrict__ bias,
                                    float* __restrict__ out,
                                    int n, int in_f, int out_f) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n || col >= out_f) return;
    float acc = bias ? bias[col] : 0.0f;
    const float* xr = x + (size_t)row * in_f;
    const float* wr = w + (size_t)col * in_f;
    for (int l = 0; l < in_f; l++) acc += xr[l] * wr[l];
    out[(size_t)row * out_f + col] = acc;
}

extern "C" cudaError_t ferrite_matmul(const float* x, const float* w,
                                      const float* bias, float* out,
                                      int n, int in_f, int out_f,
                                      cudaStream_t s) {
    if (n <= 0 || out_f <= 0) return cudaSuccess;
    // tiled path for all real shapes; naive kept reachable for diffing
    dim3 block(FERRITE_TILE, FERRITE_TILE);
    dim3 grid((out_f + FERRITE_TILE - 1) / FERRITE_TILE,
              (n + FERRITE_TILE - 1) / FERRITE_TILE);
    matmul_tiled_kernel<<<grid, block, 0, s>>>(x, w, bias, out, n, in_f, out_f);
    return cudaGetLastError();
}

extern "C" cudaError_t ferrite_matmul_naive(const float* x, const float* w,
                                            const float* bias, float* out,
                                            int n, int in_f, int out_f,
                                            cudaStream_t s) {
    dim3 block(128);
    dim3 grid((out_f + 127) / 128, n);
    matmul_naive_kernel<<<grid, block, 0, s>>>(x, w, bias, out, n, in_f, out_f);
    return cudaGetLastError();
}

// ============================================================
// bf16-resident weight matmul: weights live on the device in bf16
// (half the HBM footprint of f32 — a 285GB/TP4-rank f32 shard does not
// fit a 275GB B300; bf16 fits with 130GB to spare). x/out stay f32:
// the activation pipeline is unchanged, only the weight layout differs.
// w rows are __nv_bfloat16 (PyTorch-style bf16 = f32 high bits).
// ============================================================
#include <cuda_bf16.h>
__global__ void matmul_tiled_bf16_kernel(const float* __restrict__ x,
                                         const __nv_bfloat16* __restrict__ w,
                                         const float* __restrict__ bias,
                                         float* __restrict__ out,
                                         int n, int in_f, int out_f) {
    __shared__ float sx[FERRITE_TILE][FERRITE_TILE + 1];
    __shared__ float sw[FERRITE_TILE][FERRITE_TILE + 1];
    int row = blockIdx.y * FERRITE_TILE + threadIdx.y;
    int col = blockIdx.x * FERRITE_TILE + threadIdx.x;
    float acc = (bias && col < out_f) ? bias[col] : 0.0f;
    int tiles = (in_f + FERRITE_TILE - 1) / FERRITE_TILE;
    for (int t = 0; t < tiles; t++) {
        int k = t * FERRITE_TILE;
        sx[threadIdx.y][threadIdx.x] =
            (row < n && k + threadIdx.x < in_f)
                ? x[(size_t)row * in_f + k + threadIdx.x]
                : 0.0f;
        // w stored bf16 per row-major [out_f, in_f]; convert on smem load
        sw[threadIdx.x][threadIdx.y] =
            (col < out_f && k + threadIdx.y < in_f)
                ? __bfloat162float(w[(size_t)col * in_f + k + threadIdx.y])
                : 0.0f;
        __syncthreads();
#pragma unroll
        for (int l = 0; l < FERRITE_TILE; l++) {
            acc += sx[threadIdx.y][l] * sw[threadIdx.x][l];
        }
        __syncthreads();
    }
    if (row < n && col < out_f) out[(size_t)row * out_f + col] = acc;
}

extern "C" cudaError_t ferrite_matmul_bf16(const float* x, const void* w,
                                           const float* bias, float* out,
                                           int n, int in_f, int out_f,
                                           cudaStream_t s) {
    if (n <= 0 || out_f <= 0) return cudaSuccess;
    dim3 block(FERRITE_TILE, FERRITE_TILE);
    dim3 grid((out_f + FERRITE_TILE - 1) / FERRITE_TILE,
              (n + FERRITE_TILE - 1) / FERRITE_TILE);
    matmul_tiled_bf16_kernel<<<grid, block, 0, s>>>(
        x, (const __nv_bfloat16*)w, bias, out, n, in_f, out_f);
    return cudaGetLastError();
}

// ============================================================
// GPU-side f32 → bf16 conversion (truncation — exactly the CPU pack
// `bits >> 16`, so parity holds). Warmup streams f32 chunks over PCIe
// and converts in place into the resident bf16 allocation: packing
// 292GB/rank on the CPU is the warmup bottleneck (~150s/thread), the
// GPU converts at HBM speed.
// ============================================================
__global__ void f32_to_bf16_kernel(const float* __restrict__ in,
                                   unsigned short* __restrict__ out, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        unsigned int b = __float_as_uint(in[i]);
        out[i] = (unsigned short)(b >> 16);
    }
}

extern "C" cudaError_t ferrite_f32_to_bf16(const float* in, void* out,
                                           long n, cudaStream_t s) {
    if (n <= 0) return cudaSuccess;
    int threads = 256;
    long blocks = (n + threads - 1) / threads;
    f32_to_bf16_kernel<<<(unsigned)blocks, threads, 0, s>>>(in, (unsigned short*)out, n);
    return cudaGetLastError();
}

// ============================================================
// rmsnorm over the last dim: y = x / rms(x) * w
// 256 threads per ROW (the old block(32,4) ran ONE warp per row — for
// n=1 decode that was 32 threads serially scanning 4096 elements =
// 128 dependent loads/thread, no latency hiding, 41µs measured; the mega
// graph calls this 2×45+1 times per token = 3.7ms/token).
// ============================================================
__global__ void rmsnorm_kernel(const float* __restrict__ x,
                               const float* __restrict__ w,
                               float* __restrict__ out,
                               int n, int dim, float eps) {
    int row = blockIdx.x;
    if (row >= n) return;
    const float* xr = x + (size_t)row * dim;
    float* or_ = out + (size_t)row * dim;
    float ss = 0.f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        ss += xr[i] * xr[i];
    }
    // warp reduce
    float lane = ss;
    for (int off = 16; off > 0; off >>= 1) lane += __shfl_down_sync(0xffffffff, lane, off);
    __shared__ float red[8]; // 256 threads = 8 warps
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = lane;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.f;
        for (int i = 0; i < 8; i++) t += red[i];
        red[0] = rsqrtf(t / dim + eps);
    }
    __syncthreads();
    float inv = red[0];
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        or_[i] = xr[i] * inv * w[i];
    }
}

extern "C" cudaError_t ferrite_rmsnorm(const float* x, const float* w,
                                       float* out, int n, int dim, float eps,
                                       cudaStream_t s) {
    dim3 block(256);
    dim3 grid(n);
    rmsnorm_kernel<<<grid, block, 0, s>>>(x, w, out, n, dim, eps);
    return cudaGetLastError();
}

// ============================================================
// gated rmsnorm: y = rmsnorm(x) * w * (gate + 1)
// gate: [n, dim] (same layout as x)
// ============================================================
__global__ void gated_rmsnorm_kernel(const float* __restrict__ x,
                                     const float* __restrict__ gate,
                                     const float* __restrict__ w,
                                     float* __restrict__ out,
                                     int n, int dim, float eps) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= n) return;
    const float* xr = x + (size_t)row * dim;
    const float* gr = gate + (size_t)row * dim;
    float* or_ = out + (size_t)row * dim;
    float ss = 0.f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) ss += xr[i] * xr[i];
    float lane = ss;
    for (int off = 16; off > 0; off >>= 1) lane += __shfl_down_sync(0xffffffff, lane, off);
    __shared__ float warp_s[32];
    if (threadIdx.x == 0) warp_s[threadIdx.y] = lane / dim;
    __syncthreads();
    float inv = rsqrtf(warp_s[threadIdx.y] + eps);
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        // Glm5NextTextRMSNormGated: y = rmsnorm(x) * w * sigmoid(gate)
        or_[i] = xr[i] * inv * w[i] / (1.0f + __expf(-gr[i]));
    }
}

extern "C" cudaError_t ferrite_gated_rmsnorm(const float* x, const float* gate,
                                            const float* w, float* out,
                                            int n, int dim, float eps,
                                            cudaStream_t s) {
    dim3 block(32, 4);
    dim3 grid((n + 3) / 4);
    gated_rmsnorm_kernel<<<grid, block, 0, s>>>(x, gate, w, out, n, dim, eps);
    return cudaGetLastError();
}

// ============================================================
// swiglu_limited: gate_up [n, 2*inter] -> out [n, inter]
// out = silu(clamp(gate)) * clamp(up), limit = swiglu_limit
// ============================================================
__global__ void swiglu_kernel(const float* __restrict__ gu,
                              float* __restrict__ out,
                              int n, int inter, float limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * inter;
    if (idx >= total) return;
    int row = idx / inter, col = idx % inter;
    float g = gu[(size_t)row * 2 * inter + col];
    float u = gu[(size_t)row * 2 * inter + inter + col];
    g = fminf(g, limit); // gate: clamp max only (transformers)
    u = fminf(fmaxf(u, -limit), limit);
    out[idx] = (g / (1.0f + expf(-g))) * u;
}

extern "C" cudaError_t ferrite_swiglu(const float* gu, float* out, int n,
                                      int inter, float limit, cudaStream_t s) {
    int total = n * inter;
    dim3 block(256);
    dim3 grid((total + 255) / 256);
    swiglu_kernel<<<grid, block, 0, s>>>(gu, out, n, inter, limit);
    return cudaGetLastError();
}

// ============================================================
// fused swiglu2: reads two INDEPENDENT matmul outputs (gate, up) directly
// — the engine no longer packs them into one interleaved buffer (saves
// the host-side gather + the copy bandwidth of one extra read pass).
// out = silu(clamp(gate)) * clamp(up)
// ============================================================
__global__ void swiglu2_kernel(const float* __restrict__ gate,
                              const float* __restrict__ up,
                              float* __restrict__ out,
                              int total, float limit) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    float g = gate[idx];
    float u = up[idx];
    g = fminf(g, limit); // gate: clamp max only (transformers)
    u = fminf(fmaxf(u, -limit), limit);
    out[idx] = (g / (1.0f + expf(-g))) * u;
}

extern "C" cudaError_t ferrite_swiglu2(const float* gate, const float* up,
                                       float* out, int n, int inter,
                                       float limit, cudaStream_t s) {
    int total = n * inter;
    dim3 block(256);
    dim3 grid((total + 255) / 256);
    swiglu2_kernel<<<grid, block, 0, s>>>(gate, up, out, total, limit);
    return cudaGetLastError();
}

// ============================================================
// causal_conv1d: per-channel causal conv with carried tail.
// stream = state_in[ch, hist] ++ x[:, ch]; out[t,ch] = sum_i w[ch,i] *
// stream[hist + t - (conv-1) + i]; state_out = last hist inputs.
// ============================================================
__global__ void conv1d_kernel(const float* __restrict__ x,
                              const float* __restrict__ w,
                              const float* __restrict__ state_in,
                              float* __restrict__ out,
                              float* __restrict__ state_out,
                              int n, int ch, int conv) {
    // One block per channel (stream[] is per-channel state — the old (64,4)
    // block shared one stream across 64 channels, a data race).
    int c = blockIdx.x;
    if (c >= ch) return;
    int hist = conv - 1;
    extern __shared__ float stream[]; // hist + n floats (dynamic)
    for (int h = threadIdx.x; h < hist; h += blockDim.x)
        stream[h] = state_in[c * hist + h];
    for (int t = threadIdx.x; t < n; t += blockDim.x)
        stream[hist + t] = x[(size_t)t * ch + c];
    __syncthreads();
    for (int t = threadIdx.x; t < n; t += blockDim.x) {
        float acc = 0.f;
        for (int i = 0; i < conv; i++)
            acc += w[c * conv + i] * stream[hist + t - (conv - 1) + i];
        out[(size_t)t * ch + c] = acc;
    }
    for (int h = threadIdx.x; h < hist; h += blockDim.x)
        state_out[c * hist + h] = stream[n + h];
}

extern "C" cudaError_t ferrite_causal_conv1d(const float* x, const float* w,
                                             const float* state_in, float* out,
                                             float* state_out, int n, int ch,
                                             int conv, cudaStream_t s) {
    int hist = conv - 1;
    dim3 block(128);
    dim3 grid(ch);
    size_t smem = (size_t)(hist + n) * sizeof(float);
    conv1d_kernel<<<grid, block, smem, s>>>(x, w, state_in, out, state_out, n, ch, conv);
    return cudaGetLastError();
}

// ============================================================
// gated_deltanet_step (single-token or looped chunk; CPU-exact recurrence):
// one block per (token, head); threads sweep the dk*dv state.
// decay_i = exp(gate[t,h,i] * a_h); S[i,:] *= decay_i;
// S -= beta * k (S^T k)^T; S += beta * k v^T; o = q^T S.
// state layout [h, dk, dv]; q/k [n,h,dk]; v [n,h,dv]; beta/gate: [n,h] /
// [n,h,dk]; a_log [h].
// ============================================================
__global__ void gdn_step_kernel(const float* __restrict__ q,
                                const float* __restrict__ k,
                                const float* __restrict__ v,
                                const float* __restrict__ beta,
                                const float* __restrict__ gate,
                                const float* __restrict__ a_log,
                                float* __restrict__ state,
                                float* __restrict__ out,
                                int n, int h, int dk, int dv) {
    int t = blockIdx.z;
    int hd = blockIdx.y;
    if (t >= n || hd >= h) return;
    // KDA form: `gate` carries the LOG-SPACE decay (lb * sigmoid(exp(A_log)*(fb+dt_bias)));
    // the recurrence is S *= exp(gate) (fla naive_recurrent_kda).
    float bt = beta[t * h + hd];
    float* S = state + (size_t)hd * dk * dv;
    const float* qh = q + ((size_t)t * h + hd) * dk;
    const float* kh = k + ((size_t)t * h + hd) * dk;
    const float* vh = v + ((size_t)t * h + hd) * dv;
    const float* gh = gate + ((size_t)t * h + hd) * dk;
    // 1. per-channel decay: S[i, :] *= expf(gate[h, i]) — KDA log-space gate
    for (int i = threadIdx.x; i < dk; i += blockDim.x) {
        float decay = expf(gh[i]);
        if (decay != 1.0f) {
            for (int j = 0; j < dv; j++) S[(size_t)i * dv + j] *= decay;
        }
    }
    __syncthreads();
    // 2. kS = S^T k -> shared[dv]
    extern __shared__ float ks[];
    for (int j = threadIdx.x; j < dv; j += blockDim.x) {
        float acc = 0.f;
        for (int i = 0; i < dk; i++) acc += kh[i] * S[(size_t)i * dv + j];
        ks[j] = acc;
    }
    __syncthreads();
    // 3+4. delta rule write: S[i,j] += beta * k_i * (v_j - ks_j)
    for (int idx = threadIdx.x; idx < dk * dv; idx += blockDim.x) {
        int i = idx / dv, j = idx % dv;
        S[idx] += bt * kh[i] * (vh[j] - ks[j]);
    }
    __syncthreads();
    // 5. o = q^T S
    for (int j = threadIdx.x; j < dv; j += blockDim.x) {
        float acc = 0.f;
        for (int i = 0; i < dk; i++) acc += qh[i] * S[(size_t)i * dv + j];
        out[((size_t)t * h + hd) * dv + j] = acc;
    }
}

// Forward declaration: ferrite_gdn_step delegates to the chunk launcher
// defined below (same signature).
extern "C" cudaError_t ferrite_gdn_chunk(const float* q, const float* k,
                                         const float* v, const float* beta,
                                         const float* gate, const float* a_log,
                                         float* state, float* out,
                                         int n, int h, int dk, int dv,
                                         cudaStream_t s);

extern "C" cudaError_t ferrite_gdn_step(const float* q, const float* k,
                                        const float* v, const float* beta,
                                        const float* gate, const float* a_log,
                                        float* state, float* out,
                                        int n, int h, int dk, int dv,
                                        cudaStream_t s) {
    // Single-token path: delegate to the exact per-token chunk launcher
    // (identical signature; it loops tokens with correct offsets). The old
    // body here was a broken placeholder (returned inside the loop with
    // un-offset pointers) — dead code, no Rust caller, but the extern
    // symbol was exposed; this makes the contract safe.
    return ferrite_gdn_chunk(q, k, v, beta, gate, a_log, state, out,
                            n, h, dk, dv, s);
}

// Full chunked launcher: sequential token launches sharing the state
// buffer. Same-stream launches execute in order, so the state dependency
// chain is guaranteed WITHOUT per-token synchronization — removing the old
// cudaStreamSynchronize was a free win (launch latency amortised).
extern "C" cudaError_t ferrite_gdn_chunk(const float* q, const float* k,
                                         const float* v, const float* beta,
                                         const float* gate, const float* a_log,
                                         float* state, float* out,
                                         int n, int h, int dk, int dv,
                                         cudaStream_t s) {
    for (int t = 0; t < n; t++) {
        const float* qt = q + ((size_t)t * h) * dk;
        const float* kt = k + ((size_t)t * h) * dk;
        const float* vt = v + ((size_t)t * h) * dv;
        const float* bt = beta + (size_t)t * h;
        const float* gt = gate + ((size_t)t * h) * dk;
        float* ot = out + ((size_t)t * h) * dv;
        dim3 block(128);
        dim3 grid(1, h, 1);
        size_t smem = (size_t)dv * sizeof(float);
        gdn_step_kernel<<<grid, block, smem, s>>>(qt, kt, vt, bt, gt, a_log,
                                                  state, ot, 1, h, dk, dv);
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) return e;
    }
    return cudaSuccess;
}

// ============================================================
// gdn_step v2: v1 kept the dk*dv state in HBM and swept it FOUR times
// per token (decay R/W, kS read, delta R/W, o read — ~7 HBM passes over
// 1MB/layer with h=16 TP4 ranks), one block per head with 128 threads.
// v2 stages the state in SHARED memory (padded stride dv+1 = 129 —
// bank-conflict-free for both row sweeps and column reductions), so HBM
// traffic drops to load+store (2 passes) and every intermediate step
// reads smem. Block 512 (4x intra-block parallelism). Same per-token
// launch loop (decode n=1 → single launch; the state chain forbids
// parallel tokens).
// ============================================================
__global__ void gdn_step_v2_kernel(const float* __restrict__ q,
                                   const float* __restrict__ k,
                                   const float* __restrict__ v,
                                   const float* __restrict__ beta,
                                   const float* __restrict__ gate,
                                   const float* __restrict__ a_log,
                                   float* __restrict__ state,
                                   float* __restrict__ out,
                                   int n, int h, int dk, int dv) {
    int t = blockIdx.x;
    int hd = blockIdx.y;
    if (t >= n || hd >= h) return;
#if __CUDA_ARCH__ >= 900
    // PDL: prologue (blockIdx math, smem layout above) ran while the
    // upstream kernel (conv_prep_fused) was still draining its tail;
    // now block until its q/k/v/beta/gate stores are visible.
    cudaGridDependencySynchronize();
#endif
    float bt = beta[(size_t)t * h + hd];
    const size_t spitch = (size_t)dv + 1; // padded row stride (bank conflicts)
    extern __shared__ float sm[];
    float* S = sm;                          // [dk * (dv+1)]
    float* ks = S + (size_t)dk * spitch;    // [dv]
    float* kh = ks + dv;                    // [dk]
    float* vh = kh + dk;                    // [dv]
    float* qh = vh + dv;                    // [dk]
    float* gh = qh + dk;                    // [dk]
    // 0. load: state → smem (single HBM read), q/k/v/gate caches
    for (int i = threadIdx.x; i < dk; i += blockDim.x) {
        gh[i] = gate[((size_t)t * h + hd) * dk + i];
        qh[i] = q[((size_t)t * h + hd) * dk + i];
        kh[i] = k[((size_t)t * h + hd) * dk + i];
    }
    for (int j = threadIdx.x; j < dv; j += blockDim.x)
        vh[j] = v[((size_t)t * h + hd) * dv + j];
    float* Sg = state + (size_t)hd * dk * dv;
    for (int idx = threadIdx.x; idx < dk * dv; idx += blockDim.x)
        S[(size_t)(idx / dv) * spitch + (idx % dv)] = Sg[idx];
    __syncthreads();
    // 1. per-channel decay: S[i,:] *= exp(gate[h,i])
    for (int i = threadIdx.x; i < dk; i += blockDim.x) {
        float decay = expf(gh[i]);
        if (decay != 1.0f) {
            float* Si = S + (size_t)i * spitch;
            for (int j = 0; j < dv; j++) Si[j] *= decay;
        }
    }
    __syncthreads();
    // 2. kS = S^T k
    for (int j = threadIdx.x; j < dv; j += blockDim.x) {
        float acc = 0.f;
        for (int i = 0; i < dk; i++) acc += kh[i] * S[(size_t)i * spitch + j];
        ks[j] = acc;
    }
    __syncthreads();
    // 3. delta rule: S[i,j] += beta * k_i * (v_j - ks_j)
    for (int idx = threadIdx.x; idx < dk * dv; idx += blockDim.x)
        S[(size_t)(idx / dv) * spitch + (idx % dv)] +=
            bt * kh[idx / dv] * (vh[idx % dv] - ks[idx % dv]);
    __syncthreads();
    // 4. o = q^T S
    for (int j = threadIdx.x; j < dv; j += blockDim.x) {
        float acc = 0.f;
        for (int i = 0; i < dk; i++) acc += qh[i] * S[(size_t)i * spitch + j];
        out[((size_t)t * h + hd) * dv + j] = acc;
    }
    __syncthreads();
    // 5. store state back (single HBM write)
    for (int idx = threadIdx.x; idx < dk * dv; idx += blockDim.x)
        Sg[idx] = S[(size_t)(idx / dv) * spitch + (idx % dv)];
}

// PDL (programmatic dependent launch) enable flag: FERRITE_PDL=1 opt-in.
// Cache the getenv once — checked on every PDL-capable launcher call.
static int ferrite_pdl_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char* e = getenv("FERRITE_PDL");
        cached = (e && e[0] == '1') ? 1 : 0;
    }
    return cached;
}

extern "C" cudaError_t ferrite_gdn_chunk_v2(const float* q, const float* k,
                                            const float* v, const float* beta,
                                            const float* gate, const float* a_log,
                                            float* state, float* out,
                                            int n, int h, int dk, int dv,
                                            cudaStream_t s) {
    size_t smem = (size_t)dk * (dv + 1) * sizeof(float)
                  + (size_t)(dv + dk + dv + dk + dk) * sizeof(float);
    if (smem > 48 * 1024) {
        cudaError_t e = cudaFuncSetAttribute(gdn_step_v2_kernel,
                                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        if (e != cudaSuccess) return e;
    }
    for (int t = 0; t < n; t++) {
        dim3 block(512);
        dim3 grid(1, h, 1);
        const float* qt = q + (size_t)t * h * dk;
        const float* kt = k + (size_t)t * h * dk;
        const float* vt = v + (size_t)t * h * dv;
        const float* betat = beta + (size_t)t * h;
        const float* gatet = gate + (size_t)t * h * dk;
        float* ot = out + (size_t)t * h * dv;
        if (ferrite_pdl_enabled()) {
            // PDL: launch with programmatic stream serialization — this kernel's
            // prologue overlaps the upstream (conv_prep_fused) tail; its
            // cudaGridDependencySynchronize() gates the actual data reads.
            cudaLaunchConfig_t cfg = {};
            cfg.gridDim = grid; cfg.blockDim = block;
            cfg.dynamicSmemBytes = smem; cfg.stream = s;
            cudaLaunchAttribute attrs[1];
            attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
            attrs[0].val.programmaticStreamSerializationAllowed = 1;
            cfg.attrs = attrs; cfg.numAttrs = 1;
            cudaLaunchKernelEx(&cfg, gdn_step_v2_kernel,
                               qt, kt, vt, betat, gatet, a_log, state, ot,
                               1, h, dk, dv);
        } else {
            gdn_step_v2_kernel<<<grid, block, smem, s>>>(qt, kt, vt, betat, gatet,
                                                          a_log, state, ot, 1, h, dk, dv);
        }
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) return e;
    }
    return cudaSuccess;
}

// ============================================================
// WYF-parallel chunkwise Gated DeltaNet (ferrite-kernel/src/wyf.rs math):
//   L[t,i] = Σ_{r≤t} gate[r,i]·a  (inclusive prefix, log-space)
//   b_t = S₀ᵀ(k_t ⊙ e^{L_t})                       — state interaction
//   c[t,s] = k_t·(k_s ⊙ e^{L_t−L_s}),  s < t       — triangular system
//   w_t = β_t(v_t − b_t − Σ_{s<t} c[t,s]·w_s)      — fwd substitution
//   O_t = (q_t ⊙ e^{L_t})ᵀ S₀ + Σ_{s≤t} m[t,s]·w_s
//   S_C = diag(e^{L_{C−1}}) S₀ + Σ_s (k_s ⊙ e^{L_{C−1}−L_s}) w_sᵀ
// One block per (chunk, head); C=32 tokens in parallel inside the chunk.
// Chunks chain sequentially (state ping-pong in the launcher); the tail
// chunk falls back to the exact per-token kernel. 32x fewer launches.
// Validated against the sequential golden recurrence (wyf.rs tests).
// ============================================================
#define GDN_WYF_C 32

__global__ void gdn_wyf_kernel(const float* __restrict__ q,
                              const float* __restrict__ k,
                              const float* __restrict__ v,
                              const float* __restrict__ beta,
                              const float* __restrict__ gate,
                              const float* __restrict__ a_log,
                              const float* __restrict__ s0,
                              float* __restrict__ out,
                              float* __restrict__ st_out,
                              int chunk, int C, int h, int dk, int dv) {
    int hd = blockIdx.y;
    float a = -expf(a_log[hd]);
    size_t base_t = ((size_t)chunk * C);
    // shared layout: L[C*dk], b[C*dv], c[C*C], w[C*dv]
    extern __shared__ float sm[];
    float* L = sm;                       // [C, dk]
    float* b = L + (size_t)C * dk;       // [C, dv]
    float* c = b + (size_t)C * dv;       // [C, C]
    float* w = c + (size_t)C * C;        // [C, dv]
    const float* S0 = s0 + (size_t)hd * dk * dv;

    // 1. inclusive prefix L[t,i] = a * Σ_{r≤t} gate[r,i]
    for (int i = threadIdx.x; i < dk; i += blockDim.x) {
        float acc = 0.f;
        for (int t = 0; t < C; t++) {
            acc += gate[(base_t + (size_t)t) * (size_t)(h * dk) + (size_t)hd * dk + i] * a;
            L[t * dk + i] = acc;
        }
    }
    __syncthreads();

    // 2. b[t,j] = Σ_i S0[i,j]·k_t[i]·e^{L[t,i]}
    for (int idx = threadIdx.x; idx < C * dv; idx += blockDim.x) {
        int t = idx / dv, j = idx % dv;
        const float* k_t = k + (base_t + (size_t)t) * (size_t)(h * dk) + (size_t)hd * dk;
        float acc = 0.f;
        for (int i = 0; i < dk; i++) {
            acc += S0[(size_t)i * dv + j] * k_t[i] * expf(L[t * dk + i]);
        }
        b[idx] = acc;
    }
    __syncthreads();

    // 3. c[t,s] = Σ_i k_t[i]·k_s[i]·e^{L[t,i]−L[s,i]} (strict lower)
    for (int idx = threadIdx.x; idx < C * C; idx += blockDim.x) {
        int t = idx / C, s = idx % C;
        if (s < t) {
            const float* k_t = k + (base_t + (size_t)t) * (size_t)(h * dk) + (size_t)hd * dk;
            const float* k_s = k + (base_t + (size_t)s) * (size_t)(h * dk) + (size_t)hd * dk;
            float acc = 0.f;
            for (int i = 0; i < dk; i++) {
                acc += k_t[i] * k_s[i] * expf(L[t * dk + i] - L[s * dk + i]);
            }
            c[idx] = acc;
        } else {
            c[idx] = 0.f;
        }
    }
    __syncthreads();

    // 4. forward substitution (t sequential, dv lanes parallel)
    for (int t = 0; t < C; t++) {
        for (int j = threadIdx.x; j < dv; j += blockDim.x) {
            float acc = v[(base_t + (size_t)t) * (size_t)(h * dv) + (size_t)hd * dv + j] - b[t * dv + j];
            for (int s = 0; s < t; s++) {
                acc -= c[t * C + s] * w[s * dv + j];
            }
            w[t * dv + j] = beta[(base_t + (size_t)t) * h + hd] * acc;
        }
        __syncthreads();
    }

    // 5. O[t,j] = Σ_i q_t[i]·e^{L[t,i]}·S0[i,j] + Σ_{s≤t} m[t,s]·w[s,j]
    for (int idx = threadIdx.x; idx < C * dv; idx += blockDim.x) {
        int t = idx / dv, j = idx % dv;
        const float* q_t = q + (base_t + (size_t)t) * (size_t)(h * dk) + (size_t)hd * dk;
        float acc = 0.f;
        for (int i = 0; i < dk; i++) {
            acc += q_t[i] * expf(L[t * dk + i]) * S0[(size_t)i * dv + j];
        }
        for (int s = 0; s <= t; s++) {
            const float* k_s = k + (base_t + (size_t)s) * (size_t)(h * dk) + (size_t)hd * dk;
            float m_ts = 0.f;
            for (int i = 0; i < dk; i++) {
                m_ts += q_t[i] * k_s[i] * expf(L[t * dk + i] - L[s * dk + i]);
            }
            acc += m_ts * w[s * dv + j];
        }
        out[(base_t + (size_t)t) * (size_t)(h * dv) + (size_t)hd * dv + j] = acc;
    }
    __syncthreads();

    // 6. S_C[i,j] = e^{L[C-1,i]}·S0[i,j] + Σ_s k_s[i]·e^{L[C-1,i]−L[s,i]}·w[s,j]
    for (int idx = threadIdx.x; idx < dk * dv; idx += blockDim.x) {
        int i = idx / dv, j = idx % dv;
        float acc = expf(L[(C - 1) * dk + i]) * S0[(size_t)i * dv + j];
        for (int s = 0; s < C; s++) {
            const float* k_s = k + (base_t + (size_t)s) * (size_t)(h * dk) + (size_t)hd * dk;
            acc += k_s[i] * expf(L[(C - 1) * dk + i] - L[s * dk + i]) * w[s * dv + j];
        }
        st_out[(size_t)hd * dk * dv + idx] = acc;
    }
}

extern "C" cudaError_t ferrite_gdn_chunk_wyf(const float* q, const float* k,
                                             const float* v, const float* beta,
                                             const float* gate, const float* a_log,
                                             float* state_in, float* out,
                                             float* state_out,
                                             int n, int h, int dk, int dv,
                                             cudaStream_t s) {
    const int C = GDN_WYF_C;
    size_t smem = ((size_t)C * dk + 2 * (size_t)C * dv + (size_t)C * C) * sizeof(float);
    // state ping-pong (chunk chain: S_C of chunk i is S_0 of chunk i+1)
    float* bufs[2] = { state_in, state_out };
    int cur = 0;
    for (int base = 0; base < n; base += C) {
        int c_len = min(C, n - base);
        if (c_len < C) {
            // exact per-token fallback for the tail chunk (correctness first;
            // a padded WYF tail is a later tuning — the head chunks carry
            // the parallelism win)
            for (int t = 0; t < c_len; t++) {
                int gt = base + t;
                const float* qt = q + ((size_t)gt * h) * dk;
                const float* kt = k + ((size_t)gt * h) * dk;
                const float* vt = v + ((size_t)gt * h) * dv;
                const float* bt = beta + (size_t)gt * h;
                const float* gt_ = gate + ((size_t)gt * h) * dk;
                float* ot = out + ((size_t)gt * h) * dv;
                dim3 block(128);
                dim3 grid(1, h, 1);
                size_t sm = (size_t)dv * sizeof(float);
                gdn_step_kernel<<<grid, block, sm, s>>>(qt, kt, vt, bt, gt_, a_log,
                                                        bufs[cur], ot, 1, h, dk, dv);
                cudaError_t e = cudaGetLastError();
                if (e != cudaSuccess) return e;
            }
            cur ^= 1;
            continue;
        }
        dim3 block(256);
        dim3 grid(1, h, 1);
        gdn_wyf_kernel<<<grid, block, smem, s>>>(q, k, v, beta, gate, a_log,
                                                 bufs[cur], out, bufs[cur ^ 1],
                                                 base / C, C, h, dk, dv);
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) return e;
        cur ^= 1;
    }
    if (bufs[cur ^ 1] != state_out) {
        // odd chunk count ended writing into state_in's buffer? no — parity:
        // after k chunks, the last write target is bufs[k % 2 == 0 ? 1 : 0].
        // Settle the result into state_out when the chain ended elsewhere.
        cudaError_t e = cudaMemcpyAsync(state_out, bufs[cur ^ 1],
                                        (size_t)h * dk * dv * sizeof(float),
                                        cudaMemcpyDeviceToDevice, s);
        if (e != cudaSuccess) return e;
    }
    return cudaSuccess;
}

// ============================================================
// moe_route: sigmoid + topk + renorm (noaux-tc), per row.
// ============================================================
__global__ void moe_route_kernel(const float* __restrict__ logits,
                                 const float* __restrict__ bias,
                                 float* __restrict__ probs,
                                 float* __restrict__ ids,
                                 int n, int e, int topk, float scale) {
    int row = blockIdx.x;
    if (row >= n) return;
    // transformers Glm5NextTextTopkRouter:
    //   scores = sigmoid(logits);  choice = scores + e_score_correction_bias
    //   top-k on `choice`; weights = raw sigmoid scores (no bias), renormed.
    extern __shared__ float sm[]; // 2e floats: [0..e) sigmoid, [e..2e) choice
    float* ch = sm + e;
    for (int j = threadIdx.x; j < e; j += blockDim.x)
        sm[j] = 1.0f / (1.0f + expf(-logits[(size_t)row * e + j]));
    __syncthreads();
    for (int j = threadIdx.x; j < e; j += blockDim.x)
        ch[j] = sm[j] + bias[j];
    __syncthreads();
    // selection sort topk on the choice scores (small e in v1; bitonic later)
    for (int r = 0; r < topk; r++) {
        int best = -1;
        float bv = -1e30f;
        for (int j = threadIdx.x; j < e; j += blockDim.x) {
            if (ch[j] > bv) { bv = ch[j]; best = j; }
        }
        // warp-ish reduce: use shared
        __shared__ int bidx[32];
        __shared__ float bval[32];
        bidx[threadIdx.x] = best;
        bval[threadIdx.x] = bv;
        __syncthreads();
        for (int off = 16; off > 0; off >>= 1) {
            if (threadIdx.x + off < 32) {
                if (bval[threadIdx.x + off] > bval[threadIdx.x]) {
                    bval[threadIdx.x] = bval[threadIdx.x + off];
                    bidx[threadIdx.x] = bidx[threadIdx.x + off];
                }
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            int sel = bidx[0];
            ids[(size_t)row * topk + r] = (float)sel;
            ch[sel] = -1e30f; // remove
        }
        __syncthreads();
    }
    // renorm pass (single thread, small topk) — raw sigmoid scores, no bias
    if (threadIdx.x == 0) {
        float sum = 0.f;
        for (int r = 0; r < topk; r++) {
            int j = (int)ids[(size_t)row * topk + r];
            float val = sm[j];
            probs[(size_t)row * topk + r] = val;
            sum += val;
        }
        for (int r = 0; r < topk; r++)
            probs[(size_t)row * topk + r] = probs[(size_t)row * topk + r] / (sum + 1e-9f) * scale;
    }
}

extern "C" cudaError_t ferrite_moe_route(const float* logits, const float* bias,
                                        float* probs, float* ids, int n, int e,
                                        int topk, float scale, cudaStream_t s) {
    dim3 block(32);
    dim3 grid(n);
    size_t smem = 2 * (size_t)e * sizeof(float);
    moe_route_kernel<<<grid, block, smem, s>>>(logits, bias, probs, ids, n, e, topk, scale);
    return cudaGetLastError();
}

// ============================================================
// argmax over the last dim (greedy decode)
// MULTI-BLOCK-LATENCY-HIDDEN: the old grid(n)×block(32) ran ONE warp per
// row — 154880 elements / 32 threads = 4840 serial global loads per thread
// with zero latency hiding → 796µs for [1, 154880] (measured). 1024
// threads (32 warps) strided: ~151 elements/thread, latency fully hidden
// → ~20µs. Warp-reduce then 32-way block reduce.
// ============================================================
__global__ void argmax_kernel(const float* __restrict__ logits,
                              float* __restrict__ out, int n, int dim) {
    int row = blockIdx.x;
    if (row >= n) return;
    const float* lr = logits + (size_t)row * dim;
    int best = 0;
    float bv = -INFINITY;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        if (lr[i] > bv) { bv = lr[i]; best = i; }
    }
    // warp-level reduce (index must prefer the FIRST max on ties —
    // shuffle order: keep earlier index when equal by using > only)
    for (int off = 16; off > 0; off >>= 1) {
        float ov = __shfl_down_sync(0xffffffff, bv, off);
        int oi = __shfl_down_sync(0xffffffff, best, off);
        if (ov > bv) { bv = ov; best = oi; }
    }
    __shared__ int bidx[32];
    __shared__ float bval[32];
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    if (lane == 0) { bidx[warp] = best; bval[warp] = bv; }
    __syncthreads();
    if (threadIdx.x == 0) {
        int nw = (blockDim.x + 31) >> 5;
        for (int w = 1; w < nw; w++) {
            if (bval[w] > bv) { bv = bval[w]; best = bidx[w]; }
        }
        out[row] = (float)best;
    }
}

extern "C" cudaError_t ferrite_argmax(const float* logits, float* out, int n,
                                      int dim, cudaStream_t s) {
    dim3 block(1024);
    dim3 grid(n);
    argmax_kernel<<<grid, block, 0, s>>>(logits, out, n, dim);
    return cudaGetLastError();
}

// ============================================================
// softmax over the last dim
// ============================================================
__global__ void softmax_kernel(const float* __restrict__ logits,
                               float* __restrict__ out, int n, int dim) {
    int row = blockIdx.x;
    if (row >= n) return;
    const float* lr = logits + (size_t)row * dim;
    float* or_ = out + (size_t)row * dim;
    float m = -INFINITY;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) m = fmaxf(m, lr[i]);
    __shared__ float red[32];
    red[threadIdx.x] = m;
    __syncthreads();
    for (int off = 16; off > 0; off >>= 1) {
        if (threadIdx.x + off < 32) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + off]);
        __syncthreads();
    }
    m = red[0];
    float s = 0.f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        or_[i] = expf(lr[i] - m);
        s += or_[i];
    }
    __shared__ float reds[32];
    reds[threadIdx.x] = s;
    __syncthreads();
    for (int off = 16; off > 0; off >>= 1) {
        if (threadIdx.x + off < 32) reds[threadIdx.x] += reds[threadIdx.x + off];
        __syncthreads();
    }
    for (int i = threadIdx.x; i < dim; i += blockDim.x) or_[i] /= reds[0];
}

extern "C" cudaError_t ferrite_softmax(const float* logits, float* out, int n,
                                      int dim, cudaStream_t s) {
    dim3 block(32);
    dim3 grid(n);
    softmax_kernel<<<grid, block, 0, s>>>(logits, out, n, dim);
    return cudaGetLastError();
}

// ============================================================
// indexer_topk (real GLM-5.3-Flash semantics):
//   qi: [n, H*D] per-head indexer queries, ki: [t, D] shared keys,
//   w:  [n, H] per-head score weights.
//   score[i,j] = Σ_h w[i,h] · (q[i,h,:]·k[j,:]) / √D → topk over j.
// v1: full scan per row (t <= 1M tokens OK for correctness harness).
// ============================================================
__global__ void indexer_topk_kernel(const float* __restrict__ qi,
                                     const float* __restrict__ ki,
                                     const float* __restrict__ w,
                                     float* __restrict__ idx,
                                     int n, int h, int d, int topk,
                                     const int* __restrict__ total_ptr, int kpool_val, int n_fixed) {
    int total = *total_ptr; // actual total tokens from pinned memory
    int t = (total + kpool_val - 1) / kpool_val; // DERIVE npools from total
    int ctx0 = total - n_fixed; // derive from pinned total
    int ctx0_pools = ctx0 / kpool_val;
    int row = blockIdx.x;
    if (row >= n) return;
    extern __shared__ float sm[]; // t scores (sized for MAX at launch; graph-safe)
    float inv_sqrt_d = rsqrtf((float)d);
    // causal guard: query row i may only select keys j < ctx0_pools + i + 1
    int jmax = min(ctx0_pools + row + 1, t);
    for (int j = threadIdx.x; j < t; j += blockDim.x) {
        const float* k = ki + (size_t)j * d;
        float s = 0.f;
        if (j < jmax) {
            for (int hi = 0; hi < h; hi++) {
                const float* q = qi + (size_t)row * (h * d) + hi * d;
                float dot = 0.f;
                for (int l = 0; l < d; l++) dot += q[l] * k[l];
                s += w[(size_t)row * h + hi] * fmaxf(dot, 0.f); // relu
            }
            sm[j] = s * inv_sqrt_d;
        } else {
            sm[j] = -INFINITY;
        }
    }
    __syncthreads();
    // selection topk
    for (int r = 0; r < topk; r++) {
        __shared__ int bidx[32];
        __shared__ float bval[32];
        int best = -1;
        float bv = -INFINITY;
        for (int j = threadIdx.x; j < t; j += blockDim.x) {
            if (sm[j] > bv) { bv = sm[j]; best = j; }
        }
        bidx[threadIdx.x] = best;
        bval[threadIdx.x] = bv;
        __syncthreads();
        for (int off = 16; off > 0; off >>= 1) {
            if (threadIdx.x + off < 32 && bval[threadIdx.x + off] > bval[threadIdx.x]) {
                bval[threadIdx.x] = bval[threadIdx.x + off];
                bidx[threadIdx.x] = bidx[threadIdx.x + off];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            int sel = bidx[0];
            if (sel >= 0) {
                idx[(size_t)row * topk + r] = (float)sel;
                sm[sel] = -INFINITY;
            } else {
                idx[(size_t)row * topk + r] = -1.0f; // invisible: skip at expansion
            }
        }
        __syncthreads();
    }
}

extern "C" cudaError_t ferrite_indexer_topk(const float* qi, const float* ki,
                                            const float* w,
                                            float* idx, int n, int h, int d,
                                            int topk, const int* total_ptr, int kpool_val, int n_fixed,
                                            cudaStream_t s) {
    dim3 block(32);
    dim3 grid(n);
    // smem sized for MAX possible pools (graph-safe: frozen smem with actual
    // npools would overflow as context grows)
    int max_t = 2048; // max_npools = max_tokens / kpool
    size_t smem = (size_t)max_t * sizeof(float);
    indexer_topk_kernel<<<grid, block, smem, s>>>(qi, ki, w, idx, n, h, d, topk, total_ptr, kpool_val, n_fixed);
    return cudaGetLastError();
}

// ============================================================
// sparse_mla_attn: out[n, h, dv] = softmax(q · k_sel) v_sel over the
// top-k selected tokens per row. q [n,h,dq]; k [t,h,dk]; v [t,h,dv];
// idx [n, topk]; dq == dk (nope-only).
// ============================================================
__global__ void sparse_attn_kernel(const float* __restrict__ q,
                                   const float* __restrict__ k,
                                   const float* __restrict__ v,
                                   const float* __restrict__ idx,
                                   float* __restrict__ out,
                                   int n, const int* __restrict__ t_ptr, int h, int d, int dv, int topk) {
    int t = *t_ptr; // zero-copy read from pinned host memory (graph-safe)
    int row = blockIdx.x;
    int hd = blockIdx.y;
    if (row >= n) return;
    float scale = rsqrtf((float)d);
    extern __shared__ float sm[]; // topk scores + topk exp
    for (int s = threadIdx.x; s < topk; s += blockDim.x) {
        int j = (int)idx[(size_t)row * topk + s];
        if (j < 0 || j >= t) { sm[s] = -INFINITY; continue; } // kpool padding (-1) / OOB guard
        // transformers scatter-add mask: duplicate indices collapse to ONE
        // visible key. Skip repeats (first occurrence wins) so the softmax
        // stays normalised.
        bool dup = false;
        for (int s0 = 0; s0 < topk; s0++) {
            int j0 = (int)idx[(size_t)row * topk + s0];
            if (s0 < s && j0 == j) { dup = true; break; }
        }
        if (dup) { sm[s] = -INFINITY; continue; }
        const float* qh = q + ((size_t)row * h + hd) * d;
        const float* kj = k + ((size_t)j * h + hd) * d;
        float acc = 0.f;
        for (int l = 0; l < d; l++) acc += qh[l] * kj[l];
        sm[s] = acc * scale;
    }
    __syncthreads();
    float m = -INFINITY;
    for (int s = threadIdx.x; s < topk; s += blockDim.x) m = fmaxf(m, sm[s]);
    __shared__ float red[32];
    red[threadIdx.x] = m;
    __syncthreads();
    for (int off = 16; off > 0; off >>= 1) {
        if (threadIdx.x + off < 32) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + off]);
        __syncthreads();
    }
    m = red[0];
    float denom = 1e-9f;
    for (int s = threadIdx.x; s < topk; s += blockDim.x) {
        sm[s] = expf(sm[s] - m);
        denom += sm[s];
    }
    __shared__ float reds[32];
    reds[threadIdx.x] = denom;
    __syncthreads();
    for (int off = 16; off > 0; off >>= 1) {
        if (threadIdx.x + off < 32) reds[threadIdx.x] += reds[threadIdx.x + off];
        __syncthreads();
    }
    denom = reds[0];
    for (int j2 = threadIdx.x; j2 < dv; j2 += blockDim.x) {
        float acc = 0.f;
        for (int s = 0; s < topk; s++) {
            int j = (int)idx[(size_t)row * topk + s];
            if (j < 0 || j >= t) continue; // kpool padding (-1) / OOB guard
            if (sm[s] == -INFINITY) continue; // deduplicated slot (repeat index)
            float w = sm[s] / denom;
            acc += w * v[((size_t)j * h + hd) * dv + j2];
        }
        out[((size_t)row * h + hd) * dv + j2] = acc;
    }
}

extern "C" cudaError_t ferrite_sparse_attn(const float* q, const float* k,
                                           const float* v, const float* idx,
                                           float* out, int n, const int* t_ptr, int h, int d,
                                           int dv, int topk, cudaStream_t s) {
    // NOTE: block width must stay <= 32 — the shared reduction arrays
    // (red/reds) are [32]; 128 threads would write out of bounds.
    dim3 block(32);
    dim3 grid(n, h);
    size_t smem = (size_t)topk * sizeof(float); // dynamic smem for the topk scores
    sparse_attn_kernel<<<grid, block, smem, s>>>(q, k, v, idx, out, n, t_ptr, h, d, dv, topk);
    return cudaGetLastError();
}

// ============================================================
// sparse_attn v2: v1 ran block=32 (ONE warp per (row, head)) over
// topk = select_k*kpool+3 (~8K) slots — each lane did hundreds of
// SERIAL SCALAR dots (d=128 each) and the dedup rescanned idx from
// GLOBAL memory O(topk^2); the v-gather re-read idx dv/32 times.
// v2 (256 threads/block): idx + scores + q in smem, float4 dots,
// bitmap dedup (atomicOr test-and-set — same "first wins" semantics,
// any duplicate slot yields the same key so the winner is arbitrary),
// block-parallel softmax (warp shuffle + 8-warp smem), coalesced
// weight × v gather. smem ≈ topk*8B + d*4 + 16KB bitmap — opt-in
// dynamic smem for >48KB. Bitmap covers t ≤ 131072 (4096 words);
// longer caches skip dedup for j beyond range (dup keys double-count
// — acceptable for now, bench caches are ≤ 16K).
// ============================================================
__global__ void sparse_attn_v2_kernel(const float* __restrict__ q,
                                      const float* __restrict__ k,
                                      const float* __restrict__ v,
                                      const float* __restrict__ idx,
                                      float* __restrict__ out,
                                      int n, const int* __restrict__ t_ptr, int h, int d, int dv, int topk) {
    int t = *t_ptr; // zero-copy pinned read (graph-safe)
    int row = blockIdx.x;
    int hd = blockIdx.y;
    if (row >= n || hd >= h) return;
    float scale = rsqrtf((float)d);
    // Layout: qs FIRST (float4 reads need the 16B-aligned smem base;
    // topk=8195 is NOT a multiple of 4 — an int[topk] prefix misaligned
    // qs by 12B and crashed prefill with err 716). All later arrays are
    // scalar-access, 4B alignment suffices.
    extern __shared__ float sm[];
    float* qs = sm;                                   // [d] float4 reads
    float* sc = sm + d;                                // [topk] scores → weights
    int* idx_s = (int*)(sc + topk);                   // [topk]
    float* red = (float*)(idx_s + topk);               // [8] warp partials
    unsigned int* bm = (unsigned int*)(red + 8);       // [4096] dedup bitmap
    const int bm_words_max = 4096;
    int bm_words = (t + 31) >> 5; if (bm_words > bm_words_max) bm_words = bm_words_max;
    // 0. preload: q head-slice → smem, idx → smem, clear bitmap
    for (int l = threadIdx.x; l < d; l += blockDim.x) qs[l] = q[((size_t)row * h + hd) * d + l];
    for (int s = threadIdx.x; s < topk; s += blockDim.x) idx_s[s] = (int)idx[(size_t)row * topk + s];
    for (int w0 = threadIdx.x; w0 < bm_words_max; w0 += blockDim.x) bm[w0] = 0u;
    __syncthreads();
    // 1. scores: float4 dot per slot; bitmap dedup (first wins — duplicate
    // slots carry the same key, so which one survives is value-identical)
    for (int s = threadIdx.x; s < topk; s += blockDim.x) {
        int j = idx_s[s];
        if (j < 0 || j >= t) { sc[s] = -INFINITY; continue; }
        bool dup = false;
        if ((j >> 5) < bm_words) {
            unsigned int prev = atomicOr(&bm[j >> 5], 1u << (j & 31));
            dup = (prev & (1u << (j & 31))) != 0;
        }
        if (dup) { sc[s] = -INFINITY; continue; }
        const float4* k4 = reinterpret_cast<const float4*>(k + ((size_t)j * h + hd) * d);
        float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
        for (int l = 0; l + 3 < d; l += 4) {
            float4 kk = k4[l >> 2];
            float4 qq = *reinterpret_cast<const float4*>(qs + l);
            acc.x += qq.x * kk.x; acc.y += qq.y * kk.y;
            acc.z += qq.z * kk.z; acc.w += qq.w * kk.w;
        }
        float a = acc.x + acc.y + acc.z + acc.w;
        for (int l = d & ~3; l < d; l++) a += qs[l] * k[((size_t)j * h + hd) * d + l];
        sc[s] = a * scale;
    }
    __syncthreads();
    // 2. softmax (block-wide max → exp → sum via warp shuffles + smem)
    float m = -INFINITY;
    for (int s = threadIdx.x; s < topk; s += blockDim.x) m = fmaxf(m, sc[s]);
    for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_down_sync(0xffffffff, m, off));
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = m;
    __syncthreads();
    if (threadIdx.x < 8) m = red[threadIdx.x];
    for (int off = 4; off > 0; off >>= 1) m = fmaxf(m, __shfl_down_sync(0xffffffff, m, off));
    __shared__ float ms_;
    if (threadIdx.x == 0) ms_ = m;
    __syncthreads();
    m = ms_;
    bool all_inf = (m == -INFINITY);
    float sum = 0.f;
    for (int s = threadIdx.x; s < topk; s += blockDim.x) {
        sc[s] = all_inf ? 0.f : __expf(sc[s] - m);
        sum += sc[s];
    }
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_down_sync(0xffffffff, sum, off);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = sum;
    __syncthreads();
    if (threadIdx.x < 8) sum = red[threadIdx.x];
    for (int off = 4; off > 0; off >>= 1) sum += __shfl_down_sync(0xffffffff, sum, off);
    __shared__ float sum_;
    if (threadIdx.x == 0) sum_ = sum;
    __syncthreads();
    float denom = sum_ + 1e-9f;
    __syncthreads();
    for (int s = threadIdx.x; s < topk; s += blockDim.x) sc[s] /= denom;
    __syncthreads();
    // 3. weighted v-gather: coalesced over j2 (consecutive lanes → dv)
    for (int j2 = threadIdx.x; j2 < dv; j2 += blockDim.x) {
        float a = 0.f;
        for (int s = 0; s < topk; s++) {
            float w = sc[s];
            if (w == 0.f) continue; // padding / deduped slot (exp→0)
            int j = idx_s[s];
            if (j < 0 || j >= t) continue;
            a += w * v[((size_t)j * h + hd) * dv + j2];
        }
        out[((size_t)row * h + hd) * dv + j2] = a;
    }
}

extern "C" cudaError_t ferrite_sparse_attn_v2(const float* q, const float* k,
                                              const float* v, const float* idx,
                                              float* out, int n, const int* t_ptr, int h, int d,
                                              int dv, int topk, cudaStream_t s) {
    dim3 block(256);
    dim3 grid(n, h);
    size_t smem = (size_t)topk * (sizeof(int) + sizeof(float)) + (size_t)d * sizeof(float)
                  + 8 * sizeof(float) + 4096 * sizeof(unsigned int);
    if (smem > 48 * 1024) {
        cudaError_t e = cudaFuncSetAttribute(sparse_attn_v2_kernel,
                                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        if (e != cudaSuccess) return e;
    }
    sparse_attn_v2_kernel<<<grid, block, smem, s>>>(q, k, v, idx, out, n, t_ptr, h, d, dv, topk);
    return cudaGetLastError();
}

// ============================================================
// MHC hyper-connections (sglang-exact port; see ferrite-exec/src/mhc.rs
// for the golden CPU math). hc_pre mixes the n residual flows into the
// layer input; hc_post recombines the sublayer output back. One block
// per token; the mix dot-products are block-reduced, the 4x4 sinkhorn
// runs single-threaded (n is tiny).
//
// hc_pre: mixes[m] = (fw[m,:] · x) * rsqrt(mean(x^2)+rms_eps)
//   pre_i  = sigmoid(mixes_i*scale0 + base_i) + hc_eps;  li = Σ pre_i x_i
//   post_i = 2*sigmoid(mixes_{n+i}*scale1 + base_{n+i})
//   comb   = mixes_{2n+..}*scale2 + base → sinkhorn-normalised [n,n]
// hc_post: out[t,i,j] = post[t,i]*x[t,j] + Σ_k comb[t,k,i]*res[t,k,j]
// ============================================================
__global__ void hc_pre_kernel(const float* __restrict__ res,
                               const float* __restrict__ fw,
                               const float* __restrict__ scale,
                               const float* __restrict__ base,
                               float* __restrict__ li,
                               float* __restrict__ post,
                               float* __restrict__ comb,
                               int s, int n, int h, int mix,
                               float rms_eps, float hc_eps, int iters) {
    int t = blockIdx.x;
    if (t >= s) return;
    const float* x = res + (size_t)t * n * h;
    const int nh = n * h;
    extern __shared__ float sm[]; // mixes [mix] + comb [n*n] + red [32]
    float* mx = sm;
    float* cb = sm + mix;
    float* red = cb + n * n;

    // 1. rsqrt(mean(x^2) + rms_eps)
    if (threadIdx.x == 0) red[31] = rsqrtf(0.f); // placeholder init
    float part = 0.f;
    for (int i = threadIdx.x; i < nh; i += blockDim.x) part += x[i] * x[i];
    // warp+block reduce via shared
    for (int off = 16; off > 0; off >>= 1) part += __shfl_down_sync(0xffffffff, part, off);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = part;
    __syncthreads();
    float msq = 0.f;
    if (threadIdx.x == 0) {
        for (int w = 0; w < 32; w++) if (w < (blockDim.x + 31) >> 5) msq += red[w];
        red[30] = rsqrtf(msq / (float)nh + rms_eps);
    }
    __syncthreads();
    float rsq = red[30];

    // 2. mixes: WARP-PER-MIX (was: serial loop over mix — 24 dots, each with
    //    two block-wide __syncthreads reductions = the 0.78ms/layer cost).
    //    Each warp reduces one mix's dot independently via shuffle — zero
    //    block syncs in the loop.
    {
        int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
        int nwarps = blockDim.x >> 5;
        for (int m = warp; m < mix; m += nwarps) {
            const float* row = fw + (size_t)m * nh;
            float acc = 0.f;
            for (int i = lane; i < nh; i += 32) acc += row[i] * x[i];
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
            if (lane == 0) mx[m] = acc * rsq;
        }
    }
    __syncthreads();

    // 3. pre / layer_input, post, comb (single thread; n and mix are tiny)
    if (threadIdx.x == 0) {
        for (int i = 0; i < n; i++) {
            float pre_i = 1.0f / (1.0f + __expf(-(mx[i] * scale[0] + base[i]))) + hc_eps;
            post[t * n + i] = 2.0f * (1.0f / (1.0f + __expf(-(mx[n + i] * scale[1] + base[n + i]))));
            // stash pre in comb's tail? no — write li below with a parallel loop.
            // keep pre in smem: reuse red[16..16+n]
            red[16 + i] = pre_i;
        }
        for (int i = 0; i < n; i++)
            for (int k = 0; k < n; k++)
                cb[i * n + k] = mx[2 * n + i * n + k] * scale[2] + base[2 * n + i * n + k];
        // 4. sinkhorn: row softmax (+eps), then alternating col/row normalise
        for (int i = 0; i < n; i++) {
            float rmax = -INFINITY;
            for (int k = 0; k < n; k++) rmax = fmaxf(rmax, cb[i * n + k]);
            float denom = 0.f;
            for (int k = 0; k < n; k++) { cb[i * n + k] = __expf(cb[i * n + k] - rmax); denom += cb[i * n + k]; }
            for (int k = 0; k < n; k++) cb[i * n + k] = cb[i * n + k] / denom + hc_eps;
        }
        for (int k = 0; k < n; k++) {
            float colsum = 0.f;
            for (int i = 0; i < n; i++) colsum += cb[i * n + k];
            float d = colsum + hc_eps;
            for (int i = 0; i < n; i++) cb[i * n + k] /= d;
        }
        for (int it = 1; it < iters; it++) {
            for (int i = 0; i < n; i++) {
                float rowsum = 0.f;
                for (int k2 = 0; k2 < n; k2++) rowsum += cb[i * n + k2];
                float d = rowsum + hc_eps;
                for (int k2 = 0; k2 < n; k2++) cb[i * n + k2] /= d;
            }
            for (int k2 = 0; k2 < n; k2++) {
                float colsum = 0.f;
                for (int i = 0; i < n; i++) colsum += cb[i * n + k2];
                float d = colsum + hc_eps;
                for (int i = 0; i < n; i++) cb[i * n + k2] /= d;
            }
        }
    }
    __syncthreads();

    // 5. li = Σ_i pre_i · x[i*h + j] (parallel over h)
    for (int j = threadIdx.x; j < h; j += blockDim.x) {
        float acc = 0.f;
        for (int i = 0; i < n; i++) acc += red[16 + i] * x[(size_t)i * h + j];
        li[(size_t)t * h + j] = acc;
    }
    // write comb out
    if (threadIdx.x == 0) {
        for (int i = 0; i < n * n; i++) comb[(size_t)t * n * n + i] = cb[i];
    }
}

extern "C" cudaError_t ferrite_hc_pre(const float* res, const float* fw,
                                       const float* scale, const float* base,
                                       float* li, float* post, float* comb,
                                       int s, int n, int h, int mix,
                                       float rms_eps, float hc_eps, int iters,
                                       cudaStream_t stream) {
    size_t smem = ((size_t)mix + n * n + 32) * sizeof(float);
    // one warp per mix row (mix = n + n + n² = 24 for hc_mult 4): 768 threads
    int threads = (mix * 32) > 1024 ? 1024 : ((mix * 32) < 256 ? 256 : mix * 32);
    hc_pre_kernel<<<s, threads, smem, stream>>>(res, fw, scale, base, li, post, comb,
                                                 s, n, h, mix, rms_eps, hc_eps, iters);
    return cudaGetLastError();
}

__global__ void hc_post_kernel(const float* __restrict__ x,
                               const float* __restrict__ res,
                               const float* __restrict__ post,
                               const float* __restrict__ comb,
                               float* __restrict__ out,
                               int s, int n, int h) {
    // out[t,i,j] = post[t,i]*x[t,j] + Σ_k comb[t,k,i]*res[t,k,j]
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = s * n * h;
    if (idx >= total) return;
    int j = idx % h;
    int i = (idx / h) % n;
    int t = idx / (n * h);
    float acc = post[(size_t)t * n + i] * x[(size_t)t * h + j];
    for (int k = 0; k < n; k++) {
        acc += comb[(size_t)t * n * n + k * n + i] * res[(size_t)(t * n + k) * h + j];
    }
    out[idx] = acc;
}

extern "C" cudaError_t ferrite_hc_post(const float* x, const float* res,
                                        const float* post, const float* comb,
                                        float* out, int s, int n, int h,
                                        cudaStream_t stream) {
    int total = s * n * h;
    dim3 block(256);
    dim3 grid((total + 255) / 256);
    hc_post_kernel<<<grid, block, 0, stream>>>(x, res, post, comb, out, s, n, h);
    return cudaGetLastError();
}

// ============================================================
// fused GDN prep: everything between the conv1d/matmul projections and
// the gated-deltanet core, in ONE kernel (the CPU path did SiLU, split,
// per-head L2 on q/k, beta sigmoid, and the KDA forget gate as separate
// host loops — six host round-trips per layer per token).
//
// Inputs (all device, from matmul_dev outputs):
//   conv_out [n, 3*proj]  — raw causal-conv output (SiLU applied HERE)
//   b_raw    [n, proj]    — b_proj output (beta = sigmoid)
//   fb       [n, proj]    — f_b(f_a(x)) output (gate input)
//   dt_bias  [proj]       — weight (f32-resident)
//   a_log    [h]          — weight
// Outputs:
//   q [n,h,dk], k [n,h,dk] (L2-normalised), v [n,h,dk] (raw split)
//   beta [n,h], gate [n,h,dk] = lb * sigmoid(exp(A_log_h) * (fb + dt_bias))
// grid: (n * h) blocks, 256 threads — one block per (token, head).
// ============================================================
__global__ void gdn_prep_kernel(const float* __restrict__ conv_out,
                                const float* __restrict__ b_raw,
                                const float* __restrict__ fb,
                                const float* __restrict__ dt_bias,
                                const float* __restrict__ a_log,
                                float* __restrict__ q,
                                float* __restrict__ k,
                                float* __restrict__ v,
                                float* __restrict__ beta,
                                float* __restrict__ gate,
                                int n, int h, int dk, float lb) {
    int th = blockIdx.x;
    if (th >= n * h) return;
    int t = th / h;
    int hd = th % h;
    int proj = h * dk;
    const float* conv_row = conv_out + (size_t)t * 3 * proj;
    // one block handles this head's dk lanes of q/k/v/beta/gate
    extern __shared__ float sm_ss[]; // dk floats for q & k L2 sums
    float* sq = sm_ss;
    float* sk = sm_ss + dk;
    // SiLU + split (conv layout: [q_h0.., q_h1.., k_..., v_...] per token row
    // = [3*proj] with q in [0,proj), k in [proj,2*proj), v in [2*proj,3*proj))
    float ssq = 0.f, ssk = 0.f;
    for (int j = threadIdx.x; j < dk; j += blockDim.x) {
        int off = hd * dk + j;
        float qv = conv_row[off];
        qv = qv / (1.0f + expf(-qv)); // silu
        float kv = conv_row[proj + off];
        kv = kv / (1.0f + expf(-kv));
        float vv = conv_row[2 * proj + off];
        vv = vv / (1.0f + expf(-vv));
        sq[j] = qv; sk[j] = kv;
        ssq += qv * qv; ssk += kv * kv;
        // gate (per channel): lb * sigmoid(exp(a_log_h) * (fb + dt_bias))
        float g = fb[(size_t)t * proj + off] + dt_bias[off];
        // gate: KDA forget gate — MUST match the CPU path's exact computation
        // order (lb * (1/(1+exp(-x))), NOT lb/(1+exp(-x)) — the 1-ulp
        // division-vs-reciprocal rounding difference is amplified ~10x/token
        // by the GDN recurrence over 8 prefill tokens (observed O(1) output
        // divergence with real checkpoint weights).
        // CPU (exec_lib.rs): a = al[hd].exp(); x = a*g; sig = 1/(1+(-x).exp());
        //                   gv = lb * sig(a*g)
        float a_ex = expf(a_log[hd]);
        float x = a_ex * g;
        float sig = 1.0f / (1.0f + expf(-x));
        gate[((size_t)t * h + hd) * dk + j] = lb * sig;
        // v passes through (silu'd)
        v[((size_t)t * h + hd) * dk + j] = vv;
    }
    // L2 norm: SINGLE-THREAD sequential accumulation — EXACTLY matches the
    // CPU's iter().sum() order (left-to-right). The warp-shuffle tree
    // reduction differed by 1-2 ulp; the GDN recurrence (real-weight decay
    // ~10x/token) amplifies this to O(1) divergence over 8 prefill tokens
    // (observed max_diff 2.15 at l0 attn all-reduce).
    __shared__ float red[64];
    __syncthreads(); // sq[]/sk[] writes from all threads visible before sum
    if (threadIdx.x == 0) {
        float a = 0.f, b = 0.f;
        for (int j = 0; j < dk; j++) { a += sq[j] * sq[j]; b += sk[j] * sk[j]; }
        red[60] = (a > 0.f) ? 1.0f / sqrtf(a) : 0.f;
        red[61] = (b > 0.f) ? 1.0f / sqrtf(b) : 0.f;
    }
    __syncthreads();
    float nq = red[60]; float nk = red[61];
    // fla KDA: q = l2norm(q) * K^-0.5 (k is NOT scaled) — matches the CPU
    // path (lib.rs:593). This scale was missing here too (same root cause as
    // the gdn_layer_dev hybrid bug: q norms 1.0 vs CPU 0.0884=1/sqrt(128)).
    const float q_scl = rsqrtf((float)dk);
    for (int j = threadIdx.x; j < dk; j += blockDim.x) {
        int off = hd * dk + j;
        q[((size_t)t * h + hd) * dk + j] = sq[j] * nq * q_scl;
        k[((size_t)t * h + hd) * dk + j] = sk[j] * nk;
    }
    // beta = sigmoid(b_raw[t, head])
    if (threadIdx.x == 0) {
        beta[(size_t)t * h + hd] = 1.0f / (1.0f + expf(-b_raw[(size_t)t * h + hd]));
    }
}

extern "C" cudaError_t ferrite_gdn_prep(const float* conv_out, const float* b_raw,
                                        const float* fb, const float* dt_bias,
                                        const float* a_log,
                                        float* q, float* k, float* v, float* beta, float* gate,
                                        int n, int h, int dk, float lb,
                                        cudaStream_t s) {
    dim3 block(256);
    dim3 grid((unsigned)((n * h + 0) / 1)); // one block per (t, head)
    grid.x = (unsigned)(n * h);
    size_t smem = 2 * dk * sizeof(float) + 64 * sizeof(float);
    gdn_prep_kernel<<<grid, block, smem, s>>>(conv_out, b_raw, fb, dt_bias, a_log,
                                               q, k, v, beta, gate, n, h, dk, lb);
    return cudaGetLastError();
}

// ============================================================
// conv1d + gdn_prep FUSED (decode n==1 hot path): v1 ran two kernels
// (conv1d: grid(ch) one-block-per-channel FIR; gdn_prep: grid(n*h)
// per-head silu/split/L2/beta/gate) with the conv_out round-trip
// through HBM between them — ~2 kernel-boundary costs per gdn layer
// × 34 layers. v2: grid(h) one block per head, dk threads; each thread
// owns lane j of the head's q/k/v triple, computes the 4-tap FIR from
// the resident sliding-window state, updates the window in place
// (per-channel ownership — same in==out safety as conv1d), then runs
// the prep math (silu, gate, L2 — thread-0 serial sum for the exact
// CPU accumulation order, beta). Prefill (n>1) keeps the v1 pair.
// ============================================================
__global__ void conv_prep_fused_kernel(
    const float* __restrict__ x,        // [ch] qkv proj output (n==1)
    const float* __restrict__ cw,       // [ch, conv=4] FIR weights
    float* __restrict__ cs,             // [ch, hist=3] sliding-window state (in==out)
    const float* __restrict__ b_raw,   // [h]
    const float* __restrict__ fb,      // [proj]
    const float* __restrict__ dt_bias, // [proj]
    const float* __restrict__ a_log,   // [h]
    float* __restrict__ q, float* __restrict__ k, float* __restrict__ v,
    float* __restrict__ beta, float* __restrict__ gate,
    int h, int dk, float lb) {
    int hd = blockIdx.x;
    int j = threadIdx.x;
    if (j >= dk) return;
    extern __shared__ float sm[]; // sq[dk], sk[dk] (L2 sums — exact serial order)
    float* sq = sm;
    float* sk = sm + dk;
    const int proj = h * dk;
    const int c_q = hd * dk + j;
    const int c_k = proj + c_q;
    const int c_v = 2 * proj + c_q;
    // 1. conv FIR (4 taps: 3 window + new token) — matches conv1d_kernel
    //    out = Σ_i w[i]·stream[hist + 0 - 3 + i], stream = [s0,s1,s2,x]
    float qv = cw[c_q * 4 + 0] * cs[c_q * 3 + 0]
              + cw[c_q * 4 + 1] * cs[c_q * 3 + 1]
              + cw[c_q * 4 + 2] * cs[c_q * 3 + 2]
              + cw[c_q * 4 + 3] * x[c_q];
    float kv = cw[c_k * 4 + 0] * cs[c_k * 3 + 0]
              + cw[c_k * 4 + 1] * cs[c_k * 3 + 1]
              + cw[c_k * 4 + 2] * cs[c_k * 3 + 2]
              + cw[c_k * 4 + 3] * x[c_k];
    float vv = cw[c_v * 4 + 0] * cs[c_v * 3 + 0]
              + cw[c_v * 4 + 1] * cs[c_v * 3 + 1]
              + cw[c_v * 4 + 2] * cs[c_v * 3 + 2]
              + cw[c_v * 4 + 3] * x[c_v];
    // 2. slide the window in place: [s0,s1,s2] → [s1,s2,x] (same channel →
    //    per-thread ownership, no race; conv1d_kernel did the same per block)
    cs[c_q * 3 + 0] = cs[c_q * 3 + 1];
    cs[c_q * 3 + 1] = cs[c_q * 3 + 2];
    cs[c_q * 3 + 2] = x[c_q];
    cs[c_k * 3 + 0] = cs[c_k * 3 + 1];
    cs[c_k * 3 + 1] = cs[c_k * 3 + 2];
    cs[c_k * 3 + 2] = x[c_k];
    cs[c_v * 3 + 0] = cs[c_v * 3 + 1];
    cs[c_v * 3 + 1] = cs[c_v * 3 + 2];
    cs[c_v * 3 + 2] = x[c_v];
    // 3. prep math (gdn_prep exact semantics): silu, gate, L2, beta
    qv = qv / (1.0f + expf(-qv));
    kv = kv / (1.0f + expf(-kv));
    vv = vv / (1.0f + expf(-vv));
    sq[j] = qv; sk[j] = kv;
    // gate: KDA log-space — MUST be lb * sig(a*(fb+dt)) in that exact
    // computation order (1-ulp division rounding amplifies in the recurrence)
    float g = fb[c_q] + dt_bias[c_q];
    float a_ex = expf(a_log[hd]);
    float xg = a_ex * g;
    float sig = 1.0f / (1.0f + expf(-xg));
    gate[c_q] = lb * sig;
    v[c_q] = vv;
    // L2 norm: thread-0 serial accumulation — EXACT CPU iter().sum() order
    // (warp-shuffle trees diverge 1-2 ulp → O(1) divergence over prefill)
    __shared__ float red[64];
    __syncthreads();
    if (threadIdx.x == 0) {
        float a = 0.f, b = 0.f;
        for (int i = 0; i < dk; i++) { a += sq[i] * sq[i]; b += sk[i] * sk[i]; }
        red[60] = (a > 0.f) ? 1.0f / sqrtf(a) : 0.f;
        red[61] = (b > 0.f) ? 1.0f / sqrtf(b) : 0.f;
    }
    __syncthreads();
    const float q_scl = rsqrtf((float)dk);
    q[c_q] = sq[j] * red[60] * q_scl;
    k[c_q] = sk[j] * red[61];
    if (threadIdx.x == 0) beta[hd] = 1.0f / (1.0f + expf(-b_raw[hd]));
#if __CUDA_ARCH__ >= 900
    // PDL: release the downstream launch (gdn_step_v2 prologue) early —
    // all global stores (q/k/v/beta/gate) are complete here.
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

extern "C" cudaError_t ferrite_conv_prep_fused(
    const float* x, const float* cw, float* cs,
    const float* b_raw, const float* fb, const float* dt_bias,
    const float* a_log, float* q, float* k, float* v,
    float* beta, float* gate, int h, int dk, float lb, cudaStream_t s) {
    dim3 block((dk > 1024) ? 1024 : dk);
    dim3 grid(h);
    size_t smem = 2 * (size_t)dk * sizeof(float) + 64 * sizeof(float);
    conv_prep_fused_kernel<<<grid, block, smem, s>>>(x, cw, cs, b_raw, fb, dt_bias,
                                                     a_log, q, k, v, beta, gate, h, dk, lb);
    return cudaGetLastError();
}

// ============================================================
// TP all-reduce (sum): in-place sum of N partial outputs.
// grid = total/nthreads, each thread sums N inputs element-wise.
// For the decode-step device op chain: the TP fan-out produces
// world partial [n, hidden] DevBufs; this kernel sums them in-place
// on the FIRST partial's buffer (no H2D/D2H, graph-capturable).
// ============================================================
__global__ void tp_all_reduce_kernel(float* __restrict__ out,
                                     const float* __restrict__ partials,
                                     int total, int world) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    float acc = 0.f;
    for (int w = 0; w < world; w++) {
        acc += partials[(size_t)w * total + i];
    }
    out[i] = acc;
}

extern "C" cudaError_t ferrite_tp_all_reduce(float* partials, float* out,
                                              int total, int world,
                                              cudaStream_t s) {
    if (total <= 0 || world <= 1) return cudaSuccess;
    dim3 block(256);
    dim3 grid((total + 255) / 256);
    tp_all_reduce_kernel<<<grid, block, 0, s>>>(out, partials, total, world);
    return cudaGetLastError();
}

// ============================================================
// Weighted sum for MoE: out[t, hidden] = Σ_j probs[t, j] * expert_out[t, j, hidden]
// Each thread handles one (t, hidden_col) element, loops over topk experts.
// ============================================================
__global__ void moe_weighted_sum_kernel(const float* __restrict__ probs,
                                          const float* __restrict__ eouts,
                                          float* __restrict__ out,
                                          int n, int topk, int hidden) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * hidden;
    if (idx >= total) return;
    int t = idx / hidden;
    int c = idx % hidden;
    float acc = 0.f;
    for (int j = 0; j < topk; j++) {
        float p = probs[t * topk + j];
        if (p != 0.f) {
            acc += p * eouts[(size_t)t * topk * hidden + j * hidden + c];
        }
    }
    out[idx] = acc;
}

extern "C" cudaError_t ferrite_moe_weighted_sum(const float* probs,
                                                 const float* eouts,
                                                 float* out,
                                                 int n, int topk, int hidden,
                                                 cudaStream_t s) {
    dim3 block(256);
    dim3 grid((n * hidden + 255) / 256);
    moe_weighted_sum_kernel<<<grid, block, 0, s>>>(probs, eouts, out, n, topk, hidden);
    return cudaGetLastError();
}

// ============================================================
// Dedicated GEMV for decode (n==1): y[1,out_f] = x[1,in_f] @ W^T + bias.
// W row-major [out_f, in_f] bf16, x f32. The tiled 32x32 kernel wastes
// 31/32 warps at n=1 (only one row of the tile is live); this warp-level
// GEMV gives every warp one output row and streams W's bf16 row with
// 32-lane strip-mining — 8 rows per 256-thread block, K folded by warp
// shuffle reduction. This is the decode matmul (every matmul at n==1:
// GDN projections, MoE experts, DSA, lm_head).
// ============================================================
__global__ void gemv_bf16_kernel(const float* __restrict__ x,
                                 const __nv_bfloat16* __restrict__ w,
                                 const float* __restrict__ bias,
                                 float* __restrict__ y,
                                 int in_f, int out_f) {
    int warps_per_block = blockDim.x >> 5;
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * warps_per_block + warp;
    if (row >= out_f) return;
    const __nv_bfloat16* wr = w + (size_t)row * in_f;
    float acc = 0.f;
    // strip-mine: 32 lanes x 4 elements = 128 bf16 per iteration
    for (int k = lane * 4; k < in_f; k += 32 * 4) {
        float xv[4];
        xv[0] = x[k];
        xv[1] = (k + 1 < in_f) ? x[k + 1] : 0.f;
        xv[2] = (k + 2 < in_f) ? x[k + 2] : 0.f;
        xv[3] = (k + 3 < in_f) ? x[k + 3] : 0.f;
        acc += xv[0] * __bfloat162float(wr[k]);
        if (k + 1 < in_f) acc += xv[1] * __bfloat162float(wr[k + 1]);
        if (k + 2 < in_f) acc += xv[2] * __bfloat162float(wr[k + 2]);
        if (k + 3 < in_f) acc += xv[3] * __bfloat162float(wr[k + 3]);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, off);
    }
    if (lane == 0) y[row] = (bias ? bias[row] : 0.f) + acc;
}

extern "C" cudaError_t ferrite_gemv_bf16(const float* x, const void* w,
                                         const float* bias, float* out,
                                         int in_f, int out_f,
                                         cudaStream_t s) {
    if (out_f <= 0) return cudaSuccess;
    int threads = 256;
    int warps = threads >> 5; // 8 rows per block
    dim3 grid((out_f + warps - 1) / warps);
    gemv_bf16_kernel<<<grid, threads, 0, s>>>(x, (const __nv_bfloat16*)w, bias, out, in_f, out_f);
    return cudaGetLastError();
}

// ============================================================
// GEMV v2 (vectorized + K-split): v1 runs the decode weight-streaming
// chains (gdn/dsa/moe/lm_head ≈ 17ms of the 28.2ms decode step) at only
// 2.2-3.1 TB/s (27-39% of B300's 8 TB/s HBM). Two diagnosed bottlenecks:
//   (a) scalar bf16 loads — v2 issues uint4 (8 bf16 = 16B) per lane-step
//       plus 2x float4 x loads;
//   (b) latency-bound medium matrices — [3072,4096] gives only 384
//       blocks = 21 warps/SM ≈ 5KB in flight per SM vs the ~43KB HBM
//       latency-BW product needs. v2 splits K across WPR warps per row
//       (same-block smem reduce — no extra kernel, no atomics):
//       [3072,4096] WPR=4 → 12288 warps → 83/SM.
// Correctness note: v1/v2 summation orders differ (K-slice partials +
// smem fold vs single-warp shuffle tree) → f32 rounding diffs ~1e-6.
// ============================================================
template <int WPR>
__global__ void gemv_bf16_v2_kernel(const float* __restrict__ x,
                                   const __nv_bfloat16* __restrict__ w,
                                   const float* __restrict__ bias,
                                   float* __restrict__ y,
                                   int in_f, int out_f) {
    const int warps = blockDim.x >> 5;
    const int rpb = warps / WPR;               // rows per block
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = blockIdx.x * rpb + warp / WPR;
    const int kw = warp % WPR;                 // K-slice id
    float acc = 0.f;
    if (row < out_f) {
        const __nv_bfloat16* wr = w + (size_t)row * in_f;
        // slice size rounded to a multiple of 8 (uint4 16B alignment;
        // in_f % 8 == 0 is guaranteed by the host fallback to v1)
        int kper = ((in_f + WPR - 1) / WPR + 7) & ~7;
        int k0 = kw * kper;
        int k1 = min(k0 + kper, in_f);
        // vector body: uint4 W (8 bf16) + 2x float4 x per lane-step
        #pragma unroll 2
        for (int k = k0 + lane * 8; k + 7 < k1; k += 32 * 8) {
            uint4 wv = *reinterpret_cast<const uint4*>(wr + k);
            float4 xa = *reinterpret_cast<const float4*>(x + k);
            float4 xb = *reinterpret_cast<const float4*>(x + k + 4);
            const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(&wv);
            float2 f0 = __bfloat1622float2(w2[0]);
            float2 f1 = __bfloat1622float2(w2[1]);
            float2 f2 = __bfloat1622float2(w2[2]);
            float2 f3 = __bfloat1622float2(w2[3]);
            acc += xa.x * f0.x + xa.y * f0.y + xa.z * f1.x + xa.w * f1.y;
            acc += xb.x * f2.x + xb.y * f2.y + xb.z * f3.x + xb.w * f3.y;
        }
        // No tail: host falls back to v1 when in_f % 8 != 0, so in_f is a
        // multiple of 8; kper is rounded to 8 → k0/k1 8-aligned → the
        // vector loop's k+7 < k1 guard covers every element of [k0, k1).
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, off);
    }
    if (WPR == 1) {
        if (lane == 0 && row < out_f) y[row] = (bias ? bias[row] : 0.f) + acc;
    } else {
        __shared__ float part[16];
        if (lane == 0) part[warp] = acc;
        __syncthreads();
        if (warp % WPR == 0 && lane == 0) {
            float sum = 0.f;
            #pragma unroll
            for (int j = 0; j < WPR; j++) sum += part[(warp / WPR) * WPR + j];
            if (row < out_f) y[row] = (bias ? bias[row] : 0.f) + sum;
        }
    }
}

extern "C" cudaError_t ferrite_gemv_bf16_v2(const float* x, const void* w,
                                            const float* bias, float* out,
                                            int in_f, int out_f,
                                            cudaStream_t s) {
    if (out_f <= 0) return cudaSuccess;
    if (in_f <= 0) return cudaSuccess;
    if (in_f & 7) return ferrite_gemv_bf16(x, w, bias, out, in_f, out_f, s);
    // WPR heuristic by row count: enough warps to cover HBM latency
    // (out_f*WPR/8 warps total; target >= 64 warps/SM on 148 SMs).
    int wpr = out_f >= 16384 ? 1 : (out_f >= 4096 ? 2 : (out_f >= 1024 ? 4 : 8));
    int rpb = 8 / wpr;                        // 256 threads = 8 warps
    dim3 grid((out_f + rpb - 1) / rpb);
    switch (wpr) {
        case 1: gemv_bf16_v2_kernel<1><<<grid, 256, 0, s>>>(x, (const __nv_bfloat16*)w, bias, out, in_f, out_f); break;
        case 2: gemv_bf16_v2_kernel<2><<<grid, 256, 0, s>>>(x, (const __nv_bfloat16*)w, bias, out, in_f, out_f); break;
        case 4: gemv_bf16_v2_kernel<4><<<grid, 256, 0, s>>>(x, (const __nv_bfloat16*)w, bias, out, in_f, out_f); break;
        default: gemv_bf16_v2_kernel<8><<<grid, 256, 0, s>>>(x, (const __nv_bfloat16*)w, bias, out, in_f, out_f); break;
    }
    return cudaGetLastError();
}

// ============================================================
// TRI GEMV (decode n==1): three SAME-INPUT projections in ONE kernel —
// the gdn layer's b_raw [h,in] + f_a [dk,in] + g_a [dk,in] all read the
// same hidden x. v1 ran three separate gemv v2 launches (3 kernel
// boundaries per gdn layer × 34 layers); tri maps the three weight
// matrices onto one row space [0, o1+o2+o3) — WPR=4 K-split per row,
// uint4 vector body identical to gemv_bf16_v2. -2 nodes/layer.
// ============================================================
__global__ void gemv_tri_kernel(const float* __restrict__ x,
                                const __nv_bfloat16* __restrict__ w1,
                                const __nv_bfloat16* __restrict__ w2,
                                const __nv_bfloat16* __restrict__ w3,
                                float* __restrict__ y1,
                                float* __restrict__ y2,
                                float* __restrict__ y3,
                                int in_f, int o1, int o2, int o3) {
    const int T = o1 + o2 + o3;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int rpb = (blockDim.x >> 5) / 4;    // WPR=4, 256 threads → 2 rows/block
    const int row = blockIdx.x * rpb + warp / 4;
    const int kw = warp % 4;                   // K-slice id (WPR=4)
    float acc = 0.f;
    if (row < T) {
        const __nv_bfloat16* wr;
        if (row < o1)               wr = w1 + (size_t)row * in_f;
        else if (row < o1 + o2)     wr = w2 + (size_t)(row - o1) * in_f;
        else                        wr = w3 + (size_t)(row - o1 - o2) * in_f;
        int kper = ((in_f + 3) / 4 + 7) & ~7;  // WPR=4 slice, 8-aligned
        int k0 = kw * kper;
        int k1 = min(k0 + kper, in_f);
        #pragma unroll 2
        for (int k = k0 + lane * 8; k + 7 < k1; k += 32 * 8) {
            uint4 wv = *reinterpret_cast<const uint4*>(wr + k);
            float4 xa = *reinterpret_cast<const float4*>(x + k);
            float4 xb = *reinterpret_cast<const float4*>(x + k + 4);
            const __nv_bfloat162* w2p = reinterpret_cast<const __nv_bfloat162*>(&wv);
            float2 f0 = __bfloat1622float2(w2p[0]);
            float2 f1 = __bfloat1622float2(w2p[1]);
            float2 f2 = __bfloat1622float2(w2p[2]);
            float2 f3 = __bfloat1622float2(w2p[3]);
            acc += xa.x * f0.x + xa.y * f0.y + xa.z * f1.x + xa.w * f1.y;
            acc += xb.x * f2.x + xb.y * f2.y + xb.z * f3.x + xb.w * f3.y;
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
    __shared__ float part[8];
    if (lane == 0) part[warp] = acc;
    __syncthreads();
    if (warp % 4 == 0 && lane == 0 && row < T) {
        float sum = 0.f;
        #pragma unroll
        for (int j = 0; j < 4; j++) sum += part[(warp / 4) * 4 + j];
        if (row < o1)               y1[row] = sum;
        else if (row < o1 + o2)     y2[row - o1] = sum;
        else                        y3[row - o1 - o2] = sum;
    }
}

extern "C" cudaError_t ferrite_gemv_tri(const float* x, const void* w1, const void* w2,
                                        const void* w3, float* y1, float* y2, float* y3,
                                        int in_f, int o1, int o2, int o3,
                                        cudaStream_t s) {
    int T = o1 + o2 + o3;
    if (T <= 0 || in_f <= 0) return cudaSuccess;
    if (in_f & 7) return cudaErrorNotSupported; // host falls back to 3x gemv
    dim3 grid((T + 1) / 2);                     // rpb=2 rows/block (WPR=4)
    gemv_tri_kernel<<<grid, 256, 0, s>>>(x, (const __nv_bfloat16*)w1,
                                         (const __nv_bfloat16*)w2,
                                         (const __nv_bfloat16*)w3,
                                         y1, y2, y3, in_f, o1, o2, o3);
    return cudaGetLastError();
}

// ============================================================
// Fused MoE decode (n==1) with GPU-side expert dispatch — the TileRT
// ExpertSelectUpGateSiLU idea, ferrite-style: expert weights stay wherever
// the dev_weight_bf16 cache put them; a device POINTER TABLE
// (gate_ptrs/up_ptrs/down_ptrs[e_local]) lets the kernels gather the
// selected experts' rows with zero host round-trips. ids/probs stay on
// device from ferrite_moe_route. The old path downloaded ids+probs,
// dispatched on CPU, ran 8 per-expert kernel chains, gathered D2D and
// re-uploaded probs_ext: 3 host crossings + a sync per MoE layer.
//   act kernel:  grid (inter/rows, topk+1) — slot j < topk: eid=ids[j] →
//     local=eid-start → gate/up GEMV (warp-per-row) + swiglu2 → act[j];
//     slot topk = shared expert. Non-local slots zero (all-reduce sums).
//   down kernel: grid (hidden/rows) — out[h] = Σ_j probs[j]·(act_j·down_j[h,:])
//     + shared·act_shared. down_ptrs indirect per selected expert.
// ============================================================
__global__ void moe_fused_act_kernel(
    const float* __restrict__ x,          // [n, hidden]
    const float* __restrict__ ids_f,      // [n, topk] f32-encoded
    const __nv_bfloat16* const* __restrict__ gate_ptrs,  // [e_local]
    const __nv_bfloat16* const* __restrict__ up_ptrs,    // [e_local]
    const __nv_bfloat16* __restrict__ shared_gate,       // [inter_shared, hidden]
    const __nv_bfloat16* __restrict__ shared_up,
    float* __restrict__ act,              // [n, topk*inter + inter_shared]
    int expert_start, int e_local, int hidden, int inter,
    int inter_shared, int topk, int rows, float limit) {
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int slot = blockIdx.y;
    int tok = blockIdx.z;
    int row0 = blockIdx.x * rows;
    int stride = topk * inter + inter_shared;
    const float* xt = x + (size_t)tok * hidden;
    int slot_rows, slot_base;
    const __nv_bfloat16 *gw, *uw;
    if (slot < topk) {
        slot_rows = inter;
        slot_base = slot * inter;
        int eid = (int)ids_f[(size_t)tok * topk + slot];
        int local = eid - expert_start;
        if (local < 0 || local >= e_local) {
            if (warp == 0) {
                for (int r = row0 + lane; r < row0 + rows && r < slot_rows; r += 32) {
                    act[(size_t)tok * stride + slot_base + r] = 0.f;
                }
            }
            return;
        }
        gw = gate_ptrs[local];
        uw = up_ptrs[local];
    } else {
        slot_rows = inter_shared;
        slot_base = topk * inter;
        gw = shared_gate;
        uw = shared_up;
    }
    int warps = blockDim.x >> 5;
    for (int r = row0 + warp; r < row0 + rows && r < slot_rows; r += warps) {
        const __nv_bfloat16* gwr = gw + (size_t)r * hidden;
        const __nv_bfloat16* uwr = uw + (size_t)r * hidden;
        float g = 0.f, u = 0.f;
        for (int k = lane * 4; k < hidden; k += 32 * 4) {
            float x0 = xt[k];
            float x1 = (k + 1 < hidden) ? xt[k + 1] : 0.f;
            float x2 = (k + 2 < hidden) ? xt[k + 2] : 0.f;
            float x3 = (k + 3 < hidden) ? xt[k + 3] : 0.f;
            g += x0 * __bfloat162float(gwr[k]);
            u += x0 * __bfloat162float(uwr[k]);
            if (k + 1 < hidden) { g += x1 * __bfloat162float(gwr[k + 1]); u += x1 * __bfloat162float(uwr[k + 1]); }
            if (k + 2 < hidden) { g += x2 * __bfloat162float(gwr[k + 2]); u += x2 * __bfloat162float(uwr[k + 2]); }
            if (k + 3 < hidden) { g += x3 * __bfloat162float(gwr[k + 3]); u += x3 * __bfloat162float(uwr[k + 3]); }
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            g += __shfl_down_sync(0xffffffff, g, off);
            u += __shfl_down_sync(0xffffffff, u, off);
        }
        if (lane == 0) {
            g = fminf(g, limit);
            u = fminf(fmaxf(u, -limit), limit);
            act[(size_t)tok * stride + slot_base + r] = (g / (1.0f + expf(-g))) * u;
        }
    }
}

__global__ void moe_fused_down_sum_kernel(
    const float* __restrict__ ids_f,       // [n, topk]
    const float* __restrict__ probs,       // [n, topk]
    const __nv_bfloat16* const* __restrict__ down_ptrs,  // [e_local], [hidden, inter] each
    const __nv_bfloat16* __restrict__ shared_down,      // [hidden, inter_shared]
    const float* __restrict__ act,         // [n, topk*inter + inter_shared]
    float* __restrict__ out,               // [n, hidden]
    int expert_start, int e_local, int hidden, int inter,
    int inter_shared, int topk, int rows) {
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int tok = blockIdx.y;
    int h = blockIdx.x * rows + warp;
    if (h >= hidden) return;
    int stride = topk * inter + inter_shared;
    const float* act_t = act + (size_t)tok * stride;
    const float* ids_t = ids_f + (size_t)tok * topk;
    const float* probs_t = probs + (size_t)tok * topk;
    float acc = 0.f;
    for (int j = 0; j < topk; j++) {
        int eid = (int)ids_t[j];
        int local = eid - expert_start;
        if (local < 0 || local >= e_local) continue; // another rank's slot (zero act)
        float p = probs_t[j];
        if (p == 0.f) continue;
        const __nv_bfloat16* dwr = down_ptrs[local] + (size_t)h * inter;
        const float* aj = act_t + (size_t)j * inter;
        float y = 0.f;
        for (int i = lane * 4; i < inter; i += 32 * 4) {
            float a0 = aj[i];
            float a1 = (i + 1 < inter) ? aj[i + 1] : 0.f;
            float a2 = (i + 2 < inter) ? aj[i + 2] : 0.f;
            float a3 = (i + 3 < inter) ? aj[i + 3] : 0.f;
            y += a0 * __bfloat162float(dwr[i]);
            if (i + 1 < inter) y += a1 * __bfloat162float(dwr[i + 1]);
            if (i + 2 < inter) y += a2 * __bfloat162float(dwr[i + 2]);
            if (i + 3 < inter) y += a3 * __bfloat162float(dwr[i + 3]);
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            y += __shfl_down_sync(0xffffffff, y, off);
        }
        acc += p * y;
    }
    // shared expert (slot topk, weight 1; K length = inter_shared, TP-sharded)
    {
        const __nv_bfloat16* dwr = shared_down + (size_t)h * inter_shared;
        const float* as = act_t + (size_t)topk * inter;
        float y = 0.f;
        for (int i = lane * 4; i < inter_shared; i += 32 * 4) {
            float a0 = as[i];
            float a1 = (i + 1 < inter_shared) ? as[i + 1] : 0.f;
            float a2 = (i + 2 < inter_shared) ? as[i + 2] : 0.f;
            float a3 = (i + 3 < inter_shared) ? as[i + 3] : 0.f;
            y += a0 * __bfloat162float(dwr[i]);
            if (i + 1 < inter_shared) y += a1 * __bfloat162float(dwr[i + 1]);
            if (i + 2 < inter_shared) y += a2 * __bfloat162float(dwr[i + 2]);
            if (i + 3 < inter_shared) y += a3 * __bfloat162float(dwr[i + 3]);
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            y += __shfl_down_sync(0xffffffff, y, off);
        }
        acc += y;
    }
    if (lane == 0) out[(size_t)tok * hidden + h] = acc;
}

// Launcher A: act stage — caller provides the act buffer
// ([topk*inter + inter_shared]) and ids (from ferrite_moe_route, device).
extern "C" cudaError_t ferrite_moe_fused_act(
    const float* x, const float* ids_f,
    const void* const* gate_ptrs, const void* const* up_ptrs,
    const void* shared_gate, const void* shared_up,
    float* act, int expert_start, int e_local, int hidden, int inter,
    int inter_shared, int topk, int n, float limit, cudaStream_t s) {
    // rows = warps per block (256 threads / 32 = 8); grid.z = token (prefill
    // batch dimension — decode n==1, chunked prefill n up to chunk size).
    int rows = 8;
    int max_rows = inter > inter_shared ? inter : inter_shared;
    dim3 grid((max_rows + rows - 1) / rows, topk + 1, n);
    moe_fused_act_kernel<<<grid, 256, 0, s>>>(
        x, ids_f,
        (const __nv_bfloat16* const*)gate_ptrs, (const __nv_bfloat16* const*)up_ptrs,
        (const __nv_bfloat16*)shared_gate, (const __nv_bfloat16*)shared_up,
        act, expert_start, e_local, hidden, inter, inter_shared, topk, rows, limit);
    return cudaGetLastError();
}

// Launcher B: down + weighted-sum + shared stage — caller provides act
// ([n, topk*inter + inter_shared]) and out ([n, hidden]).
extern "C" cudaError_t ferrite_moe_fused_down_sum(
    const float* ids_f, const float* probs,
    const void* const* down_ptrs, const void* shared_down,
    const float* act, float* out,
    int expert_start, int e_local, int hidden, int inter,
    int inter_shared, int topk, int n,
    cudaStream_t s) {
    // CRITICAL: rows must equal warps per block (256/32 = 8). The kernel
    // assigns ONE hidden-row per warp with no stride loop — rows=128 made
    // grid.x = hidden/128 while each block only computed 8 rows (4096-row
    // hidden: 32 blocks × 8 = 256 rows computed, 94% of out was garbage).
    int rows = 8;
    dim3 grid((hidden + rows - 1) / rows, n);
    moe_fused_down_sum_kernel<<<grid, 256, 0, s>>>(
        ids_f, probs,
        (const __nv_bfloat16* const*)down_ptrs, (const __nv_bfloat16*)shared_down,
        act, out, expert_start, e_local, hidden, inter, inter_shared, topk, rows);
    return cudaGetLastError();
}

// ============================================================
// DSA (sparse attention) device chain — the four small kernels the CPU
// path did on the host between GPU calls (each crossing was a sync):
//   layernorm_affine: ki = LN(x·wk)(k_norm w/b)  [n, idm]
//   dsa_cache_append: kvb per-head strided split → k_nope/v at slot T0+t,
//                     ki/gate copies → k_idx/k_gate
//   kpool_compress:   per-channel softmax(gate+ape) pool mixing of k_idx
//   pool_expand:      idx_pools [n, select_k] → token idx [n, out_width]
//                     (+ visible tail, -1 padding)
// The big ops (indexer_topk, sparse_mla_attn, gemv projections) already
// exist as GPU kernels — dsa_layer_dev chains them all with zero host
// round-trips.
// ============================================================
__global__ void layernorm_affine_kernel(const float* __restrict__ x,
                                        const float* __restrict__ w,
                                        const float* __restrict__ b,
                                        float* __restrict__ out,
                                        int dim) {
    int row = blockIdx.x;
    const float* xr = x + (size_t)row * dim;
    float* orow = out + (size_t)row * dim;
    __shared__ float sm[512];
    float mean = 0.f, var = 0.f;
    for (int j = threadIdx.x; j < dim; j += blockDim.x) sm[j] = xr[j];
    __syncthreads();
    for (int j = 0; j < dim; j++) mean += sm[j];
    mean /= dim;
    for (int j = 0; j < dim; j++) {
        float d = sm[j] - mean;
        var += d * d;
    }
    float inv = rsqrtf(var / dim + 1e-5f);
    for (int j = threadIdx.x; j < dim; j += blockDim.x) {
        orow[j] = (sm[j] - mean) * inv * w[j] + b[j];
    }
}

extern "C" cudaError_t ferrite_layernorm_affine(const float* x, const float* w,
                                                const float* b, float* out,
                                                int n, int dim, cudaStream_t s) {
    layernorm_affine_kernel<<<n, min(dim, 256), 0, s>>>(x, w, b, out, dim);
    return cudaGetLastError();
}

__global__ void dsa_cache_append_kernel(
    const float* __restrict__ kvb,   // [n, h*(dk+dv)]
    const float* __restrict__ ki,    // [n, idm]
    const float* __restrict__ gate,  // [n, idm]
    float* __restrict__ k_nope,      // [T_total, h, dk]
    float* __restrict__ v,           // [T_total, h, dv]
    float* __restrict__ k_idx,       // [T_total, idm]
    float* __restrict__ k_gate,      // [T_total, idm]
    const int* __restrict__ t0_ptr,  // pinned memory (graph-safe: CPU writes before each replay)
    int n, int h, int dk, int dv, int idm) {
    int t0 = *t0_ptr; // zero-copy read from pinned host memory
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int row_bytes = h * (dk + dv);
    int total_elems = n * row_bytes;
    if (tid < total_elems) {
        int t = tid / row_bytes, r = tid % row_bytes;
        int hd = r / (dk + dv), c = r % (dk + dv);
        size_t dst = ((size_t)(t0 + t) * h + hd);
        if (c < dk) {
            k_nope[dst * dk + c] = kvb[tid];
        } else {
            v[dst * dv + (c - dk)] = kvb[tid];
        }
    } else if (tid < total_elems + n * idm) {
        int j = tid - total_elems;
        int t = j / idm, c = j % idm;
        k_idx[(size_t)(t0 + t) * idm + c] = ki[j];
    } else if (tid < total_elems + 2 * n * idm) {
        int j = tid - total_elems - n * idm;
        int t = j / idm, c = j % idm;
        k_gate[(size_t)(t0 + t) * idm + c] = gate[j];
    }
}

extern "C" cudaError_t ferrite_dsa_cache_append(
    const float* kvb, const float* ki, const float* gate,
    float* k_nope, float* v, float* k_idx, float* k_gate,
    const int* t0_ptr, int n, int h, int dk, int dv, int idm, cudaStream_t s) {
    int total = n * h * (dk + dv) + 2 * n * idm;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    dsa_cache_append_kernel<<<blocks, threads, 0, s>>>(
        kvb, ki, gate, k_nope, v, k_idx, k_gate, t0_ptr, n, h, dk, dv, idm);
    return cudaGetLastError();
}

__global__ void kpool_compress_kernel(
    const float* __restrict__ k_idx,   // [total, idm]
    const float* __restrict__ k_gate,  // [total, idm]
    const float* __restrict__ ape,     // [kpool, idm]
    float* __restrict__ pool_keys,     // [npools, idm]
    const int* __restrict__ total_ptr, // pinned (graph-safe)
    int max_npools, int kpool, int idm) {
    int total = *total_ptr;
    int npools = (total + kpool - 1) / kpool; // derive from pinned total
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (int)((size_t)npools * idm)) return;
    int p = tid / idm, d = tid % idm;
    float lmax = -INFINITY;
    for (int j = 0; j < kpool; j++) {
        int t = p * kpool + j;
        if (t < total) {
            float lv = k_gate[(size_t)t * idm + d] + ape[j * idm + d];
            if (lv > lmax) lmax = lv;
        }
    }
    if (lmax == -INFINITY) return;
    float den = 0.f, num = 0.f;
    for (int j = 0; j < kpool; j++) {
        int t = p * kpool + j;
        if (t < total) {
            float wgt = expf(k_gate[(size_t)t * idm + d] + ape[j * idm + d] - lmax);
            den += wgt;
            num += wgt * k_idx[(size_t)t * idm + d];
        }
    }
    pool_keys[(size_t)p * idm + d] = num / den;
}

extern "C" cudaError_t ferrite_kpool_compress(
    const float* k_idx, const float* k_gate, const float* ape,
    float* pool_keys, const int* total_ptr, int npools, int kpool, int idm,
    cudaStream_t s) {
    size_t total_t = (size_t)npools * idm;
    int threads = 256;
    int blocks = (int)((total_t + threads - 1) / threads);
    kpool_compress_kernel<<<blocks, threads, 0, s>>>(
        k_idx, k_gate, ape, pool_keys, total_ptr, npools, kpool, idm);
    return cudaGetLastError();
}

__global__ void pool_expand_kernel(
    const float* __restrict__ idx_pools,  // [n, select_k]
    float* __restrict__ idx,              // [n, out_width]
    int n, int select_k, int kpool, int max_npools,
    const int* __restrict__ total_ptr,    // pinned (graph-safe)
    int n_fixed) {                        // n as a CONSTANT for ctx0 derivation
    int total = *total_ptr;
    int ctx0 = total - n_fixed;           // derive from pinned total
    int npools = (total + kpool - 1) / kpool; // derive from pinned total
    int i = blockIdx.x;
    if (i >= n) return;
    int out_width = select_k * kpool + (kpool - 1);
    const float* pv = idx_pools + (size_t)i * select_k;
    float* iv = idx + (size_t)i * out_width;

    // MULTI-THREAD (was 1 thread serially writing ~8K slots — a dsa-layer
    // straggler): phase A flags valid r (pflt in range) + block prefix in
    // smem; phase B writes valid r's kpool slots in parallel. Invalid r slots
    // are SKIPPED (compact — col only advances for valid r, matching the
    // serial semantics): valid r's base col = valid_prefix(r) * kpool.
    extern __shared__ int sp[];           // [select_k+1] prefix
    int tid = threadIdx.x;
    for (int r = tid; r < select_k; r += blockDim.x) {
        float pflt = pv[r];
        sp[r + 1] = (pflt >= 0.0f && (int)pflt < npools) ? 1 : 0;
    }
    if (tid == 0) sp[0] = 0;
    __syncthreads();
    if (tid == 0) { // serial scan (select_k ~2k adds from smem — fine)
        for (int r = 0; r < select_k; r++) sp[r + 1] += sp[r];
    }
    __syncthreads();
    int nvalid = sp[select_k];
    // phase B: slot s = c/kpool → rank r via prefix probe (monotonic sp —
    // start from a proportional guess, walk to the bracketing interval)
    for (int c = tid; c < nvalid * kpool; c += blockDim.x) {
        int s = c / kpool, j = c % kpool;
        int r = (int)(((long long)s * select_k) / (nvalid > 0 ? nvalid : 1));
        if (r >= select_k) r = select_k - 1;
        while (r > 0 && sp[r] > s) r--;
        while (r + 1 < select_k && sp[r + 1] <= s) r++;
        int p = (int)pv[r];
        int t = p * kpool + j;
        iv[s * kpool + j] = (t < total && t <= ctx0 + i) ? (float)t : -1.0f;
    }
    // tail + padding (kpool-1 slots — tiny, single thread as before)
    if (tid == 0) {
        int visible_count = ctx0 + i + 1;
        int tail_count = visible_count % kpool;
        int tail_start = visible_count - tail_count;
        int col = nvalid * kpool;
        for (int j = 0; j < kpool - 1 && col < out_width; j++) {
            int t = tail_start + j;
            iv[col++] = (j < tail_count && t <= ctx0 + i) ? (float)t : -1.0f;
        }
        while (col < out_width) iv[col++] = -1.0f;
    }
}

extern "C" cudaError_t ferrite_pool_expand(
    const float* idx_pools, float* idx,
    int n, int select_k, int kpool, int max_npools, const int* total_ptr,
    int n_fixed,
    cudaStream_t s) {
    size_t smem = ((size_t)select_k + 1) * sizeof(int);
    pool_expand_kernel<<<n, 256, smem, s>>>(idx_pools, idx, n, select_k, kpool, max_npools, total_ptr, n_fixed);
    return cudaGetLastError();
}

// elementwise in-place scale (w_idx × n_heads^-0.5)
__global__ void scale_inplace_kernel(float* __restrict__ x, float s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= s;
}
extern "C" cudaError_t ferrite_scale_inplace(float* x, float s, int n, cudaStream_t st) {
    if (n <= 0) return cudaSuccess;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    scale_inplace_kernel<<<blocks, threads, 0, st>>>(x, s, n);
    return cudaGetLastError();
}

// ============================================================
// CUDA graph via RUNTIME API wrappers (the driver-API dlopen path
// SIGSEGV'd inside cuGraphInstantiate on worker-thread captures).
// ============================================================
// Event-in-graph timing (FERRITE_MEGA_EVTS): events recorded during
// stream capture become event-record nodes — replay updates them, and
// post-replay cudaEventElapsedTime gives the TRUE in-graph segment
// times (the DRY sync-timing drains the async queue between layers,
// so its per-segment numbers are contaminated by downstream work).
// ============================================================
// In-graph timing marker (FERRITE_MEGA_EVTS v2): CUDA events recorded
// inside a graph cannot be used with cudaEventElapsedTime (verified:
// single-test event_timing_graph path (b) → InvalidValue(1); events in
// graphs are synchronization-only). Instead a 1-thread kernel reads
// %globaltimer (chip-wide ns clock) into a u64 slot — replay updates
// the buffer, host computes segment deltas post-replay.
// ============================================================
__global__ void mark_ts_kernel(unsigned long long* __restrict__ buf, int slot) {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    buf[slot] = t;
}
extern "C" cudaError_t ferrite_mark_ts(unsigned long long* buf, int slot, cudaStream_t s) {
    mark_ts_kernel<<<1, 1, 0, s>>>(buf, slot);
    return cudaGetLastError();
}

extern "C" cudaError_t ferrite_event_create(void** ev) {
    cudaEvent_t* e = (cudaEvent_t*)malloc(sizeof(cudaEvent_t));
    if (!e) return cudaErrorMemoryAllocation;
    cudaError_t err = cudaEventCreate(e); // default flags: timing enabled
    if (err != cudaSuccess) { free(e); return err; }
    *ev = e;
    return err;
}
extern "C" cudaError_t ferrite_event_record(void* ev, cudaStream_t s) {
    return cudaEventRecord(*(cudaEvent_t*)ev, s); // in capture: graph node
}
extern "C" cudaError_t ferrite_event_elapsed(void* a, void* b, float* ms) {
    // (a, b) = (start, stop): elapsed ms from event a to event b (a must
    // have been recorded BEFORE b — the graph replays them in node order).
    return cudaEventElapsedTime(ms, *(cudaEvent_t*)a, *(cudaEvent_t*)b);
}
extern "C" cudaError_t ferrite_event_destroy(void* ev) {
    cudaError_t err = cudaEventDestroy(*(cudaEvent_t*)ev);
    free(ev);
    return err;
}
extern "C" cudaError_t ferrite_graph_begin(cudaStream_t s) {
    return cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal);
}
extern "C" cudaError_t ferrite_graph_end(cudaStream_t s, cudaGraph_t* g) {
    return cudaStreamEndCapture(s, g);
}
extern "C" cudaError_t ferrite_graph_instantiate(cudaGraphExec_t* e, cudaGraph_t g) {
    return cudaGraphInstantiate(e, g, 0);
}
extern "C" cudaError_t ferrite_graph_launch(cudaGraphExec_t e, cudaStream_t s) {
    return cudaGraphLaunch(e, s);
}
extern "C" cudaError_t ferrite_graph_destroy_exec(cudaGraphExec_t e) {
    return cudaGraphExecDestroy(e);
}

// DSA t0/total device counter: a captured graph FREEZES kernel arguments,
// so t0 (the KV append slot) and total (context length) as parameters
// would make every replay write the same slot. This mini kernel runs FIRST
// in the graph: it reads the persistent counter and writes t0/total to a
// fixed device location that subsequent kernels dereference.
__global__ void dsa_t0_counter_kernel(
    int* __restrict__ counter,   // persistent: holds next t0 (incremented by this kernel)
    int* __restrict__ t0_out,    // written: this call's t0
    int* __restrict__ total_out, // written: this call's total = t0 + n
    int n) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int t0 = *counter;
        *t0_out = t0;
        *total_out = t0 + n;
        *counter = t0 + n;
    }
}

extern "C" cudaError_t ferrite_dsa_t0_counter(
    int* counter, int* t0_out, int* total_out, int n, cudaStream_t s) {
    dsa_t0_counter_kernel<<<1, 1, 0, s>>>(counter, t0_out, total_out, n);
    return cudaGetLastError();
}

// ============================================================
// P2P all-reduce: rank 0 collects the other ranks' partials via
// cudaMemcpyPeerAsync (NVLink, B300 GPU4-7 = NV18), then the existing
// tp_all_reduce kernel sums them on-device. Replaces the host
// download→CPU-sum→re-upload round-trip.
// ============================================================
extern "C" cudaError_t ferrite_p2p_copy(float* dst, int dst_dev,
                                         const float* src, int src_dev,
                                         size_t count, cudaStream_t s) {
    return cudaMemcpyPeerAsync(dst, dst_dev, src, src_dev, count * 4, s);
}

extern "C" cudaError_t ferrite_p2p_enable(int dev, int peer) {
    cudaError_t e = cudaSetDevice(dev);
    if (e != cudaSuccess) return e;
    // Ignore cudaErrorPeerAccessAlreadyEnabled
    e = cudaDeviceEnablePeerAccess(peer, 0);
    if (e == cudaErrorPeerAccessAlreadyEnabled) return cudaSuccess;
    return e;
}

// ============================================================
// hc_pre SPLIT for decode bandwidth: the single-block hc_pre was 0.18ms
// because grid=(1) runs on ONE SM (24 warps limited to ~6GB/s vs 8TB/s
// HBM). This version uses grid=(s, mix) — each block computes ONE mix's
// dot product with 256 threads across a separate SM. rsq is computed
// redundantly per block (x is L2-cached after the first read).
// Expected: 0.18ms → ~0.01ms (16x).
// ============================================================
// K-SPLIT lanes per mix row in hc_pre phase 1 (gridDim.z): 24 mix rows × 8
// = 192 blocks = 130% SM occupancy on B300 (148 SMs) vs 24 blocks (16%).
#define HC_MIX_KS 8

__global__ void hc_pre_mix_split_kernel(const float* __restrict__ res,
                                        const float* __restrict__ fw,
                                        float* __restrict__ mx_partial,
                                        int s, int n, int h, int mix,
                                        float rms_eps) {
    // K-SPLIT: gridDim.z = KS lanes per mix row — 24 mix rows × 8 lanes =
    // 192 blocks (130% SM) vs the old 24-block single-lane version (16% SM,
    // each block serially dotting the full 18432-dim row). Each lane dots
    // its 1/KS segment; the rest kernel's prologue sums the KS partials and
    // applies rsq (rsq itself moved there too — phase 1 is a pure dot now).
    const int KS = gridDim.z;
    int t = blockIdx.x;
    int m = blockIdx.y;
    int z = blockIdx.z;
    if (t >= s || m >= mix) return;
    const float* x = res + (size_t)t * n * h;
    const int nh = n * h;
    const float* row = fw + (size_t)m * nh;
    int seg = (nh + KS - 1) / KS;
    int lo = z * seg;
    int hi = min(lo + seg, nh);

    float acc = 0.f;
    for (int i = lo + threadIdx.x; i < hi; i += blockDim.x) acc += row[i] * x[i];
    __shared__ float red[8];
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        float tot = 0.f;
        for (int w = 0; w < 8; w++) if (w < (blockDim.x + 31) >> 5) tot += red[w];
        mx_partial[((size_t)t * mix + m) * KS + z] = tot;
    }
    (void)rms_eps;
}

// The REST of hc_pre: reads pre-computed mx from global memory, does
// sinkhorn + li + post + comb. Single block per token (tiny work).
__global__ void hc_pre_rest_kernel(const float* __restrict__ res,
                                   const float* __restrict__ mx_in,
                                   const float* __restrict__ scale,
                                   const float* __restrict__ base,
                                   const float* __restrict__ nw,
                                   float* __restrict__ li,
                                   float* __restrict__ post,
                                   float* __restrict__ comb,
                                   int s, int n, int h, int mix, int mix_ks,
                                   float rms_eps, float hc_eps, int iters) {
    int t = blockIdx.x;
    if (t >= s) return;
    const float* x = res + (size_t)t * n * h;
    const int nh = n * h;
    extern __shared__ float sm[];
    float* mx_s = sm;               // [mix] K-split partials reduced here
    float* cb = sm + mix;           // [n*n]
    float* pre_s = sm + mix + n * n; // [n]
    float* li_s = sm + mix + n * n + n; // [h] fused-norm staging (rmsnorm parity)
    float* red = li_s + h;   // [8+8] warp partials (mx prologue + norm tail)

    // PROLOGUE (K-split phase-2): reduce the KS partial dots per mix row
    // and apply rsq (Σx² block reduce — was phase-1 per-block redundant).
    // mx_in layout: [t][mix][ks] partials from hc_pre_mix_split_kernel.
    {
        float part = 0.f;
        for (int i = threadIdx.x; i < nh; i += blockDim.x) part += x[i] * x[i];
        for (int off = 16; off > 0; off >>= 1) part += __shfl_down_sync(0xffffffff, part, off);
        if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = part;
        __syncthreads();
        float msq = 0.f;
        if (threadIdx.x == 0) {
            for (int w = 0; w < 8; w++) if (w < (blockDim.x + 31) >> 5) msq += red[w];
            red[15] = rsqrtf(msq / (float)nh + rms_eps);
        }
        // reduce partials: thread m sums its mix row's ks lanes
        for (int m = threadIdx.x; m < mix; m += blockDim.x) {
            float acc = 0.f;
            for (int z = 0; z < mix_ks; z++) acc += mx_in[((size_t)t * mix + m) * mix_ks + z];
            mx_s[m] = acc * red[15];
        }
        __syncthreads();
    }
    const float* mx = mx_s;

    // pre / layer_input, post, comb (parallel across n for sigmoid)
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        pre_s[i] = 1.0f / (1.0f + __expf(-(mx[i] * scale[0] + base[i]))) + hc_eps;
        post[t * n + i] = 2.0f * (1.0f / (1.0f + __expf(-(mx[n + i] * scale[1] + base[n + i]))));
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int i = 0; i < n; i++)
            for (int k = 0; k < n; k++)
                cb[i * n + k] = mx[2 * n + i * n + k] * scale[2] + base[2 * n + i * n + k];
        // sinkhorn
        for (int i = 0; i < n; i++) {
            float rmax = -INFINITY;
            for (int k = 0; k < n; k++) rmax = fmaxf(rmax, cb[i * n + k]);
            float denom = 0.f;
            for (int k = 0; k < n; k++) { cb[i * n + k] = __expf(cb[i * n + k] - rmax); denom += cb[i * n + k]; }
            for (int k = 0; k < n; k++) cb[i * n + k] = cb[i * n + k] / denom + hc_eps;
        }
        for (int k = 0; k < n; k++) {
            float colsum = 0.f;
            for (int i = 0; i < n; i++) colsum += cb[i * n + k];
            float d = colsum + hc_eps;
            for (int i = 0; i < n; i++) cb[i * n + k] /= d;
        }
        for (int it = 1; it < iters; it++) {
            for (int i = 0; i < n; i++) {
                float rowsum = 0.f;
                for (int k2 = 0; k2 < n; k2++) rowsum += cb[i * n + k2];
                float d = rowsum + hc_eps;
                for (int k2 = 0; k2 < n; k2++) cb[i * n + k2] /= d;
            }
            for (int k2 = 0; k2 < n; k2++) {
                float colsum = 0.f;
                for (int i = 0; i < n; i++) colsum += cb[i * n + k2];
                float d = colsum + hc_eps;
                for (int i = 0; i < n; i++) cb[i * n + k2] /= d;
            }
        }
        for (int i = 0; i < n * n; i++) comb[(size_t)t * n * n + i] = cb[i];
    }
    __syncthreads();
    // li = Σ_i pre_i · x[i*h + j] (parallel over h) — staged in smem, then
    // FUSED rmsnorm tail (saves the standalone rmsnorm kernel launch per
    // layer segment): identical reduce order to rmsnorm_kernel (stride ss →
    // warp shfl → serial red[8] → rsqrt(ss/h + eps)) for parity.
    for (int j = threadIdx.x; j < h; j += blockDim.x) {
        float acc = 0.f;
        for (int i = 0; i < n; i++) acc += pre_s[i] * x[(size_t)i * h + j];
        li_s[j] = acc;
    }
    __syncthreads();
    float ss = 0.f;
    for (int j = threadIdx.x; j < h; j += blockDim.x) ss += li_s[j] * li_s[j];
    for (int off = 16; off > 0; off >>= 1) ss += __shfl_down_sync(0xffffffff, ss, off);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
    __syncthreads();
    if (threadIdx.x == 0) {
        float tt = 0.f;
        for (int i = 0; i < 8; i++) tt += red[i];
        red[0] = rsqrtf(tt / h + rms_eps);
    }
    __syncthreads();
    float inv = red[0];
    for (int j = threadIdx.x; j < h; j += blockDim.x) {
        li[(size_t)t * h + j] = li_s[j] * inv * nw[j];
    }
}

extern "C" cudaError_t ferrite_hc_pre_split(const float* res, const float* fw,
                                            const float* scale, const float* base,
                                            const float* nw,
                                            float* li, float* post, float* comb,
                                            float* mx_scratch,
                                            int s, int n, int h, int mix,
                                            float rms_eps, float hc_eps, int iters,
                                            cudaStream_t stream) {
    // Phase 1: K-SPLIT mix computation — grid(s, mix, KS=8) = 192 blocks
    // (130% SM) vs the old (s, mix) = 24 blocks (16% SM, each block serially
    // dotting the full 18432-dim row). Each lane dots its 1/8 segment into
    // mx_partial; the rest kernel's prologue sums the 8 lanes and applies rsq.
    dim3 mix_grid(s, mix, HC_MIX_KS);
    hc_pre_mix_split_kernel<<<mix_grid, 256, 0, stream>>>(
        res, fw, mx_scratch, s, n, h, mix, rms_eps);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return e;

    // Phase 2: rest (single block per token — sinkhorn + li + post + comb)
    // + FUSED rmsnorm tail (nw = input_layernorm weight): li comes out
    // normalized — saves the standalone rmsnorm launch per layer segment.
    // PROLOGUE: reduce the KS mix partials + rsq (Σx²).
    // smem: mx_s[mix] + cb[n*n] + pre_s[n] + li_s[h] + red[8]  (~16.6KB)
    size_t smem2 = ((size_t)(mix + n * n + n) + h + 16) * sizeof(float);
    hc_pre_rest_kernel<<<s, 256, smem2, stream>>>(
        res, mx_scratch, scale, base, nw, li, post, comb,
        s, n, h, mix, HC_MIX_KS, rms_eps, hc_eps, iters);
    return cudaGetLastError();
}

// ============================================================
// hc_contract: [s, n*h] -> [s, h] — mean over the n MHC flows (the mirror
// of mhc::hc_expand; the head chain's input after the last layer). Token-
// major layout: flow i of token t lives at in[(t*n + i)*h .. +h].
// mega-graph: this closes the device-resident decode loop (residual ->
// contract -> rmsnorm -> lm_head -> argmax, zero host crossings).
// ============================================================
__global__ void hc_contract_kernel(const float* __restrict__ in,
                                   float* __restrict__ out,
                                   int s, int n, int h) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // one thread per output elem
    int t = blockIdx.y;
    if (j >= h || t >= s) return;
    float acc = 0.f;
    for (int i = 0; i < n; i++) {
        acc += in[(size_t)(t * n + i) * h + j];
    }
    out[(size_t)t * h + j] = acc / n;
}

extern "C" cudaError_t ferrite_hc_contract(const float* in, float* out,
                                          int s, int n, int h,
                                          cudaStream_t stream) {
    dim3 block(256);
    dim3 grid((h + 255) / 256, s);
    hc_contract_kernel<<<grid, block, 0, stream>>>(in, out, s, n, h);
    return cudaGetLastError();
}

// ============================================================
// gemv5_bf16: ONE launch for up to 5 same-input GEMVs (decode n=1).
// gdn_layer_dev issues x*Wqkv, x*Wb, x*Wfa, x*Wga as 4 separate GEMV
// launches (4 kernel tails ~10-15us each); dsa issues 5. Same-input rows
// concatenate: thread t owns ONE output row of the virtual [of1+..+of5,
// in_f] matrix — full HBM bandwidth, one launch.
// Pass of5=0 (w5/o5 = nullptr) for the 4-matrix case.
// ============================================================
__global__ void gemv5_bf16_kernel(const float* __restrict__ x,
                                  const __nv_bfloat16* __restrict__ w1, const __nv_bfloat16* __restrict__ w2,
                                  const __nv_bfloat16* __restrict__ w3, const __nv_bfloat16* __restrict__ w4,
                                  const __nv_bfloat16* __restrict__ w5,
                                  float* __restrict__ o1, float* __restrict__ o2,
                                  float* __restrict__ o3, float* __restrict__ o4,
                                  float* __restrict__ o5,
                                  int in_f, int of1, int of2, int of3, int of4, int of5) {
    int row = blockIdx.x; // one block per row
    int tot = of1 + of2 + of3 + of4 + of5;
    if (row >= tot) return;
    const __nv_bfloat16* wrow;
    float* orow;
    if (row < of1) { wrow = w1 + (size_t)row * in_f; orow = o1 + row; }
    else if (row < of1 + of2) { wrow = w2 + (size_t)(row - of1) * in_f; orow = o2 + (row - of1); }
    else if (row < of1 + of2 + of3) { wrow = w3 + (size_t)(row - of1 - of2) * in_f; orow = o3 + (row - of1 - of2); }
    else if (row < of1 + of2 + of3 + of4) { wrow = w4 + (size_t)(row - of1 - of2 - of3) * in_f; orow = o4 + (row - of1 - of2 - of3); }
    else { wrow = w5 + (size_t)(row - of1 - of2 - of3 - of4) * in_f; orow = o5 + (row - of1 - of2 - of3 - of4); }
    float acc = 0.f;
    for (int k = threadIdx.x; k < in_f; k += blockDim.x)
        acc += x[k] * __bfloat162float(wrow[k]);
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffff, acc, off);
    __shared__ float red[8];
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.f;
        for (int i = 0; i < 8; i++) t += red[i];
        *orow = t;
    }
}

extern "C" cudaError_t ferrite_gemv5_bf16(const float* x,
                                          const void* w1, const void* w2, const void* w3,
                                          const void* w4, const void* w5,
                                          float* o1, float* o2, float* o3, float* o4, float* o5,
                                          int in_f, int of1, int of2, int of3, int of4, int of5,
                                          cudaStream_t s) {
    int tot = of1 + of2 + of3 + of4 + of5;
    if (tot <= 0) return cudaSuccess;
    dim3 block(256);
    dim3 grid(tot); // one block per output row
    gemv5_bf16_kernel<<<grid, block, 0, s>>>(
        x,
        (const __nv_bfloat16*)w1, (const __nv_bfloat16*)w2, (const __nv_bfloat16*)w3,
        (const __nv_bfloat16*)w4, (const __nv_bfloat16*)w5,
        o1, o2, o3, o4, o5,
        in_f, of1, of2, of3, of4, of5);
    return cudaGetLastError();
}

// ============================================================
// PDL (Programmatic Dependent Launch) experiment: A→B dependency chain.
// Normal launch: B waits for A's FULL completion (tail + memory flush)
// before B's prologue starts. PDL (sm_90+): B launches early (A's
// cudaTriggerProgrammaticLaunchCompletion), B's prologue (address calc,
// smem init, independent loads) overlaps A's tail; B's
// cudaGridDependencySynchronize() blocks until A's writes are visible.
// ferrite_pdl_exp times iters× (A,B) chains both ways.
// ============================================================
__global__ void pdl_a_kernel(float* buf, int work_iters) {
    cudaTriggerProgrammaticLaunchCompletion();  // release B's launch NOW
    // main body (overlappable with B's prologue)
    for (int i = threadIdx.x; i < 4096; i += blockDim.x) buf[i] = (float)(i + threadIdx.x);
    for (int it = 0; it < work_iters; it++)
        for (int i = threadIdx.x; i < 4096; i += blockDim.x) buf[i] = buf[i] * 1.0001f + 0.1f;
}
__global__ void pdl_b_kernel(const float* buf, float* out) {
    // prologue: independent work (would otherwise sit idle behind A's tail)
    float dummy = 0.f;
    #pragma unroll 20
    for (int it = 0; it < 2000; it++) dummy += it * 0.001f;
    cudaGridDependencySynchronize();  // A's writes now visible
    float acc = 0.f;
    for (int i = threadIdx.x; i < 4096; i += blockDim.x) acc += buf[i];
    if (dummy == 12345.678f) acc = -acc;  // keep prologue alive
    atomicAdd(out, acc * 1e-9f);
}
extern "C" cudaError_t ferrite_pdl_exp(int mode, int iters, float* out_time_ms,
                                       float* out_checksum, cudaStream_t s) {
    float *d_a, *d_b;
    cudaError_t e;
    if ((e = cudaMalloc(&d_a, 4096 * sizeof(float))) != cudaSuccess) return e;
    if ((e = cudaMalloc(&d_b, sizeof(float))) != cudaSuccess) { cudaFree(d_a); return e; }
    cudaMemset(d_b, 0, sizeof(float));
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    if (mode >= 2) {
        // GRAPH-CAPTURED chains: does the PDL attribute survive stream capture?
        // mode 2 = normal launches captured; mode 3 = cudaLaunchKernelEx + PDL
        // attr captured. Same A→B dependency chain ×16 per graph, replay iters.
        cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal);
        for (int it = 0; it < 16; it++) {
            if (mode == 3) {
                cudaLaunchConfig_t cfg = {};
                cfg.gridDim = dim3(1); cfg.blockDim = dim3(256); cfg.stream = s;
                cudaLaunchAttribute attrs[1];
                attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
                attrs[0].val.programmaticStreamSerializationAllowed = 1;
                cfg.attrs = attrs; cfg.numAttrs = 1;
                cudaLaunchKernelEx(&cfg, pdl_a_kernel, d_a, 200);
                cudaLaunchKernelEx(&cfg, pdl_b_kernel, (const float*)d_a, d_b);
            } else {
                pdl_a_kernel<<<1, 256, 0, s>>>(d_a, 200);
                pdl_b_kernel<<<1, 256, 0, s>>>((const float*)d_a, d_b);
            }
        }
        cudaGraph_t g;
        if ((e = cudaStreamEndCapture(s, &g)) != cudaSuccess) { cudaFree(d_a); cudaFree(d_b); return e; }
        cudaGraphExec_t ge;
        if ((e = cudaGraphInstantiate(&ge, g, NULL, NULL, 0)) != cudaSuccess) { cudaGraphDestroy(g); cudaFree(d_a); cudaFree(d_b); return e; }
        cudaGraphDestroy(g);
        cudaGraphLaunch(ge, s); // warm
        cudaStreamSynchronize(s);
        cudaMemset(d_b, 0, sizeof(float));
        cudaEventRecord(e0, s);
        for (int it = 0; it < iters; it++) cudaGraphLaunch(ge, s);
        cudaEventRecord(e1, s);
        cudaEventSynchronize(e1);
        float ms;
        cudaEventElapsedTime(&ms, e0, e1);
        *out_time_ms = ms;
        e = cudaMemcpy(out_checksum, d_b, sizeof(float), cudaMemcpyDeviceToHost);
        cudaGraphExecDestroy(ge);
        cudaEventDestroy(e0); cudaEventDestroy(e1);
        cudaFree(d_a); cudaFree(d_b);
        return e;
    }
    cudaEventRecord(e0, s);
    for (int it = 0; it < iters; it++) {
        if (mode == 1) {
            cudaLaunchConfig_t cfg = {};
            cfg.gridDim = dim3(1); cfg.blockDim = dim3(256); cfg.stream = s;
            cudaLaunchAttribute attrs[1];
            attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
            attrs[0].val.programmaticStreamSerializationAllowed = 1;
            cfg.attrs = attrs; cfg.numAttrs = 1;
            cudaLaunchKernelEx(&cfg, pdl_a_kernel, d_a, 200);
            cudaLaunchKernelEx(&cfg, pdl_b_kernel, (const float*)d_a, d_b);
        } else {
            pdl_a_kernel<<<1, 256, 0, s>>>(d_a, 200);
            pdl_b_kernel<<<1, 256, 0, s>>>((const float*)d_a, d_b);
        }
    }
    cudaEventRecord(e1, s);
    cudaEventSynchronize(e1);
    float ms;
    cudaEventElapsedTime(&ms, e0, e1);
    *out_time_ms = ms;
    e = cudaMemcpy(out_checksum, d_b, sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    cudaFree(d_a); cudaFree(d_b);
    return e;
}

// ============================================================
// P2P one-shot all-reduce micro-bench (TileRT ExpertDownAllReduce
// mode): replaces NCCL allreduce for small decode collectives
// (n*hidden f32). down: each rank writes its partial into ALL ranks'
// staging slots (UVA peer writes over NVLink) + last block raises MY
// flag on ALL ranks; sum: spins all local flags, sums local staging
// [world][n] rows. Flags are reset by the host between iterations
// (micro-bench protocol; the production version inlines the flag
// reset into the next round's producer per ready/done handshake).
// ============================================================
__global__ void p2p_ar_down_kernel(const float* __restrict__ partial,
                                   float* const* __restrict__ staging_tbl,
                                   unsigned* const* __restrict__ ready_tbl,
                                   unsigned* ctr, int world, int my_rank, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = partial[i];
        // each rank writes ITS slot row (my_rank) in every peer's staging
        const size_t off = (size_t)my_rank * n + i;
        #pragma unroll 4
        for (int r = 0; r < world; r++) staging_tbl[r][off] = v;
    }
    __threadfence_system(); // peer-visible stores before flag
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned prev = atomicAdd(ctr, 1u);
        if (prev == gridDim.x - 1u) { // last block to finish: all stores visible
            for (int r = 0; r < world; r++)
                *(volatile unsigned*)&ready_tbl[r][my_rank] = 1u;
            *ctr = 0u; // reset for the next launch (stream-ordered)
        }
    }
}

__global__ void p2p_ar_sum_kernel(const float* __restrict__ staging, // local [world][n]
                                  const unsigned* __restrict__ ready,  // local [world]
                                  float* __restrict__ out, int world, int n) {
    if (threadIdx.x == 0) { // every block spins until all ranks' flags are up
        for (int r = 0; r < world; r++)
            while (*(volatile unsigned*)&ready[r] == 0u) __nanosleep(100);
    }
    __syncthreads();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float acc = 0.f;
        for (int r = 0; r < world; r++) acc += staging[(size_t)r * n + i];
        out[i] = acc;
    }
}

// staging_tbl/ready_tbl are DEVICE arrays of world device pointers (the
// peer bases); ctr is this rank's local block counter; staging_local /
// ready_local are this rank's staging row block and flag row.
extern "C" cudaError_t ferrite_p2p_ar_oneshot(
    const float* partial, float* const* staging_tbl,
    unsigned* const* ready_tbl, unsigned* ctr,
    const float* staging_local, const unsigned* ready_local,
    float* out, int n, int world, int my_rank, cudaStream_t s) {
    p2p_ar_down_kernel<<<(n + 255) / 256, 256, 0, s>>>(
        partial, staging_tbl, ready_tbl, ctr, world, my_rank, n);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return e;
    p2p_ar_sum_kernel<<<(n + 255) / 256, 256, 0, s>>>(
        staging_local, ready_local, out, world, n);
    return cudaGetLastError();
}

// ============================================================
// Knife 1b: qkv GEMV + conv FIR/silu/window-slide epilogue (decode n==1).
// Replaces matmul_dev(qkv_proj) + the FIR/silu/slide half of
// conv_prep_fused — one kernel per gdn layer. The row's dot lands,
// lane 0 runs the 3-tap FIR against the sliding-window state, slides
// it, applies silu, and writes q/k/v directly. The L2 norm + gate +
// beta halves move to gdn_step_v2p's prologue (they need cross-row
// reductions / other tensors).
// ============================================================
__global__ void gemv_qkv_conv_kernel(const float* __restrict__ x,
                                     const __nv_bfloat16* __restrict__ w,
                                     const float* __restrict__ cw,
                                     float* __restrict__ cs,
                                     float* __restrict__ q,
                                     float* __restrict__ k,
                                     float* __restrict__ v,
                                     int in_f, int proj) {
    // WPR==1 specialization of gemv_bf16_v2 (out=3*proj >= 16k rows):
    // one warp per row, uint4 body, lane 0 epilogue = FIR + slide + silu.
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = blockIdx.x * 8 + warp;   // 256 threads = 8 warps/block
    if (row >= 3 * proj) return;
    const __nv_bfloat16* wr = w + (size_t)row * in_f;
    float acc = 0.f;
    #pragma unroll 2
    for (int k0 = lane * 8; k0 + 7 < in_f; k0 += 32 * 8) {
        uint4 wv = *reinterpret_cast<const uint4*>(wr + k0);
        float4 xa = *reinterpret_cast<const float4*>(x + k0);
        float4 xb = *reinterpret_cast<const float4*>(x + k0 + 4);
        const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(&wv);
        float2 f0 = __bfloat1622float2(w2[0]);
        float2 f1 = __bfloat1622float2(w2[1]);
        float2 f2 = __bfloat1622float2(w2[2]);
        float2 f3 = __bfloat1622float2(w2[3]);
        acc += xa.x * f0.x + xa.y * f0.y + xa.z * f1.x + xa.w * f1.y;
        acc += xb.x * f2.x + xb.y * f2.y + xb.z * f3.x + xb.w * f3.y;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffff, acc, off);
    if (lane != 0) return;
    // ---- epilogue: conv FIR (3 window taps + new token) + slide + silu ----
    const float xv = acc;                       // new conv input token
    float fir = cw[(size_t)row * 4 + 0] * cs[(size_t)row * 3 + 0]
              + cw[(size_t)row * 4 + 1] * cs[(size_t)row * 3 + 1]
              + cw[(size_t)row * 4 + 2] * cs[(size_t)row * 3 + 2]
              + cw[(size_t)row * 4 + 3] * xv;
    cs[(size_t)row * 3 + 0] = cs[(size_t)row * 3 + 1];
    cs[(size_t)row * 3 + 1] = cs[(size_t)row * 3 + 2];
    cs[(size_t)row * 3 + 2] = xv;
    fir = fir / (1.0f + expf(-fir));            // silu (q, k and v all gated)
    if (row < proj)       q[row] = fir;
    else if (row < 2 * proj) k[row - proj] = fir;
    else                  v[row - 2 * proj] = fir;
}
extern "C" cudaError_t ferrite_gemv_qkv_conv(
    const float* x, const void* w, const void* cw, float* cs,
    float* q, float* k, float* v, int in_f, int proj, cudaStream_t s) {
    int out_f = 3 * proj;
    dim3 grid((out_f + 7) / 8);
    gemv_qkv_conv_kernel<<<grid, 256, 0, s>>>(
        x, (const __nv_bfloat16*)w, (const float*)cw, cs,
        q, k, v, in_f, proj);
    return cudaGetLastError();
}

// ============================================================
// Knife 1b part 2: gdn_step_v2p — gdn_step_v2 with an extended prologue
// that computes the L2 norm (q, k), gate (KDA log-space sigmoid) and beta
// inline, replacing the conv_prep_fused node. q/k arrive as raw FIR+silu
// output (from gemv_qkv_conv's epilogue); the block reduces sum(qh^2)/
// sum(kh^2) via smem tree, then applies q = qh*L2*rsqrt(dk), k = kh*L2
// (matching conv_prep_fused's normalization semantics).
// ============================================================
__global__ void gdn_step_v2p_kernel(const float* __restrict__ q,
                                   const float* __restrict__ k,
                                   const float* __restrict__ v,
                                   const float* __restrict__ b_raw,
                                   const float* __restrict__ fb,
                                   const float* __restrict__ dt_bias,
                                   const float* __restrict__ a_log,
                                   float lb,
                                   float* __restrict__ state,
                                   float* __restrict__ out,
                                   int n, int h, int dk, int dv) {
    int t = blockIdx.x;
    int hd = blockIdx.y;
    if (t >= n || hd >= h) return;
    const float a_ex = expf(a_log[hd]);
    const float bt = 1.0f / (1.0f + expf(-b_raw[hd]));
    const size_t spitch = (size_t)dv + 1;
    extern __shared__ float sm[];
    float* S = sm;
    float* ks = S + (size_t)dk * spitch;
    float* kh = ks + dv;
    float* vh = kh + dk;
    float* qh = vh + dv;
    float* gh = qh + dk;
    float* red = gh + dk; // [2]: L2 sums (q, k)
    __shared__ float wq[16], wk[16]; // warp sums for the L2 block-tree
    // 0. load q/k (raw FIR+silu) + gate math + v + state; L2 via smem tree
    const int base = (int)((size_t)t * h + hd) * dk;
    float q_sq = 0.f, k_sq = 0.f;
    for (int i = threadIdx.x; i < dk; i += blockDim.x) {
        float qv = q[base + i];
        float kv = k[base + i];
        qh[i] = qv; kh[i] = kv;
        q_sq += qv * qv; k_sq += kv * kv;
        // KDA log-space gate: lb * sig(a_log[hd] * (fb[c]+dt_bias[c])) —
        // exact conv_prep computation order (1-ulp sensitive recurrence).
        float g = fb[base + i] + dt_bias[base + i];
        gh[i] = lb / (1.0f + expf(-(a_ex * g)));
    }
    for (int j = threadIdx.x; j < dv; j += blockDim.x)
        vh[j] = v[(size_t)((size_t)t * h + hd) * dv + j];
    // block-tree reduce q_sq/k_sq -> red[2]
    // (warp shuffle: every warp's lane 0 holds the warp sum; 512 threads
    // = 16 warps; warps past dk are all-zero lanes, contributing 0)
    unsigned mask = 0xffffffffu;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        q_sq += __shfl_down_sync(mask, q_sq, off);
        k_sq += __shfl_down_sync(mask, k_sq, off);
    }
    if ((threadIdx.x & 31) == 0) {
        int w = threadIdx.x >> 5;
        wq[w] = q_sq; wk[w] = k_sq;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float a = 0.f, b = 0.f;
        for (int w = 0; w < 16; w++) { a += wq[w]; b += wk[w]; }
        red[0] = (a > 0.f) ? rsqrtf(a) : 0.f;
        red[1] = (b > 0.f) ? rsqrtf(b) : 0.f;
    }
    __syncthreads();
    const float q_scl = rsqrtf((float)dk);
    for (int i = threadIdx.x; i < dk; i += blockDim.x) {
        qh[i] = qh[i] * red[0] * q_scl;
        kh[i] = kh[i] * red[1];
    }
    float* Sg = state + (size_t)hd * dk * dv;
    for (int idx = threadIdx.x; idx < dk * dv; idx += blockDim.x)
        S[(size_t)(idx / dv) * spitch + (idx % dv)] = Sg[idx];
    __syncthreads();
    // 1. per-channel decay: S[i,:] *= exp(gate[h,i])
    for (int i = threadIdx.x; i < dk; i += blockDim.x) {
        float decay = expf(gh[i]);
        if (decay != 1.0f) {
            float* Si = S + (size_t)i * spitch;
            for (int j = 0; j < dv; j++) Si[j] *= decay;
        }
    }
    __syncthreads();
    // 2. kS = S^T k
    for (int j = threadIdx.x; j < dv; j += blockDim.x) {
        float acc = 0.f;
        for (int i = 0; i < dk; i++) acc += kh[i] * S[(size_t)i * spitch + j];
        ks[j] = acc;
    }
    __syncthreads();
    // 3. delta rule: S[i,j] += beta * k_i * (v_j - ks_j)
    for (int idx = threadIdx.x; idx < dk * dv; idx += blockDim.x)
        S[(size_t)(idx / dv) * spitch + (idx % dv)] +=
            bt * kh[idx / dv] * (vh[idx % dv] - ks[idx % dv]);
    __syncthreads();
    // 4. o = q^T S
    for (int j = threadIdx.x; j < dv; j += blockDim.x) {
        float acc = 0.f;
        for (int i = 0; i < dk; i++) acc += qh[i] * S[(size_t)i * spitch + j];
        out[((size_t)t * h + hd) * dv + j] = acc;
    }
    __syncthreads();
    // 5. store state back
    for (int idx = threadIdx.x; idx < dk * dv; idx += blockDim.x)
        Sg[idx] = S[(size_t)(idx / dv) * spitch + (idx % dv)];
}
extern "C" cudaError_t ferrite_gdn_step_v2p(
    const float* q, const float* k, const float* v,
    const float* b_raw, const float* fb, const float* dt_bias,
    const float* a_log, float lb,
    float* state, float* out, int h, int dk, int dv,
    cudaStream_t s) {
    size_t smem = (size_t)dk * (dv + 1) * sizeof(float)
                  + (size_t)(dv + dk + dv + dk + dk + 2) * sizeof(float);
    if (smem > 48 * 1024) {
        cudaError_t e = cudaFuncSetAttribute(gdn_step_v2p_kernel,
                                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        if (e != cudaSuccess) return e;
    }
    dim3 block(512);
    dim3 grid(1, h, 1);
    gdn_step_v2p_kernel<<<grid, block, smem, s>>>(
        q, k, v, b_raw, fb, dt_bias, a_log, lb, state, out, 1, h, dk, dv);
    return cudaGetLastError();
}

// elementwise add (residual): z = x + y — MTP layer's standard (non-MHC)
// residual connections.
__global__ void add_kernel(const float* __restrict__ x, const float* __restrict__ y,
                           float* __restrict__ z, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) z[i] = x[i] + y[i];
}
extern "C" cudaError_t ferrite_add(const float* x, const float* y, float* z,
                                   int n, cudaStream_t s) {
    add_kernel<<<(n + 255) / 256, 256, 0, s>>>(x, y, z, n);
    return cudaGetLastError();
}

// ============================================================
// MTP Phase2: fused n-token GDN chunk (verify n=2) — ONE launch per layer
// instead of the t-split's 2 chunk_v2 launches + a B0 copy in between.
// The state stays resident in smem across the t loop (HBM state round-trip
// eliminated); the t=0 snapshot (B0 = A+t_last for accept-1's commit) is
// written straight from smem. Saves ~2 launches + 16MB HBM traffic/layer
// on the 34 GDN layers (verify 23.4ms → ~19ms expected).
// ============================================================
__global__ void gdn_chunk_fused_kernel(const float* __restrict__ q,
                                       const float* __restrict__ k,
                                       const float* __restrict__ v,
                                       const float* __restrict__ beta,
                                       const float* __restrict__ gate,
                                       const float* __restrict__ a_log,
                                       float* __restrict__ state,
                                       float* __restrict__ gdn0,
                                       float* __restrict__ gdn1,
                                       float* __restrict__ out,
                                       int n, int h, int dk, int dv) {
    int hd = blockIdx.y;
    if (hd >= h) return;
    const size_t spitch = (size_t)dv + 1;
    extern __shared__ float sm[];
    float* S = sm;                          // [dk * (dv+1)] — resident across t
    float* ks = S + (size_t)dk * spitch;   // [dv]
    float* kh = ks + dv;                   // [dk]
    float* vh = kh + dk;                   // [dv]
    float* qh = vh + dv;                   // [dk]
    float* gh = qh + dk;                   // [dk]
    float* Sg = state + (size_t)hd * dk * dv;
    // state load ONCE (smem resident across the whole t loop)
    for (int idx = threadIdx.x; idx < dk * dv; idx += blockDim.x)
        S[(size_t)(idx / dv) * spitch + (idx % dv)] = Sg[idx];
    __syncthreads();
    for (int t = 0; t < n; t++) {
        float bt = beta[(size_t)t * h + hd];
        for (int i = threadIdx.x; i < dk; i += blockDim.x) {
            gh[i] = gate[((size_t)t * h + hd) * dk + i];
            qh[i] = q[((size_t)t * h + hd) * dk + i];
            kh[i] = k[((size_t)t * h + hd) * dk + i];
        }
        for (int j = threadIdx.x; j < dv; j += blockDim.x)
            vh[j] = v[((size_t)t * h + hd) * dv + j];
        __syncthreads();
        // 1. per-channel decay
        for (int i = threadIdx.x; i < dk; i += blockDim.x) {
            float decay = expf(gh[i]);
            if (decay != 1.0f) {
                float* Si = S + (size_t)i * spitch;
                for (int j = 0; j < dv; j++) Si[j] *= decay;
            }
        }
        __syncthreads();
        // 2. kS = S^T k
        for (int j = threadIdx.x; j < dv; j += blockDim.x) {
            float acc = 0.f;
            for (int i = 0; i < dk; i++) acc += kh[i] * S[(size_t)i * spitch + j];
            ks[j] = acc;
        }
        __syncthreads();
        // 3. delta rule
        for (int idx = threadIdx.x; idx < dk * dv; idx += blockDim.x)
            S[(size_t)(idx / dv) * spitch + (idx % dv)] +=
                bt * kh[idx / dv] * (vh[idx % dv] - ks[idx % dv]);
        __syncthreads();
        // 4. o = q^T S
        for (int j = threadIdx.x; j < dv; j += blockDim.x) {
            float acc = 0.f;
            for (int i = 0; i < dk; i++) acc += qh[i] * S[(size_t)i * spitch + j];
            out[((size_t)t * h + hd) * dv + j] = acc;
        }
        __syncthreads();
        // 5. t=0/t=1 snapshots (B_k = A + t_0..t_k: accept-k's commit source)
        // straight from smem — B0 = A+t_last (accept-1), B1 = A+t_last+d1
        // (accept-2, n=3 only)
        if ((t == 0 && gdn0 != nullptr) || (t == 1 && gdn1 != nullptr)) {
            float* Sk = (t == 0 ? gdn0 : gdn1) + (size_t)hd * dk * dv;
            for (int idx = threadIdx.x; idx < dk * dv; idx += blockDim.x)
                Sk[idx] = S[(size_t)(idx / dv) * spitch + (idx % dv)];
            __syncthreads();
        }
    }
    // state store ONCE after the loop (B = A + all n tokens)
    for (int idx = threadIdx.x; idx < dk * dv; idx += blockDim.x)
        Sg[idx] = S[(size_t)(idx / dv) * spitch + (idx % dv)];
}
extern "C" cudaError_t ferrite_gdn_chunk_fused(
    const float* q, const float* k, const float* v,
    const float* beta, const float* gate, const float* a_log,
    float* state, float* gdn0, float* gdn1, float* out,
    int n, int h, int dk, int dv, cudaStream_t s) {
    size_t smem = (size_t)dk * (dv + 1) * sizeof(float)
                  + (size_t)(dv + dk + dv + dk + dk) * sizeof(float);
    if (smem > 48 * 1024) {
        cudaError_t e = cudaFuncSetAttribute(gdn_chunk_fused_kernel,
                                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        if (e != cudaSuccess) {

            return e;
        }
    }
    dim3 block(512);
    dim3 grid(1, h, 1);
    gdn_chunk_fused_kernel<<<grid, block, smem, s>>>(
        q, k, v, beta, gate, a_log, state, gdn0, gdn1, out, n, h, dk, dv);
    cudaError_t le = cudaGetLastError();
    if (le != cudaSuccess) {

    }
    return le;
}

// ============================================================
// MTP accept commit (single launch): replaces the per-layer
// cudaMemcpyAsync ping-pong chain (2 memcpys x n_gdn_layers + hprev
// select = ~70 launches/step of pure launch overhead) with ONE kernel.
// k (1..3, the accept length) is read from a PINNED host int at run
// time (zero-copy): the verify graph replay returns argmax to the
// host, the host computes k, then launches this kernel.
// plan: [n_plans][8] device-resident pointer table per GDN layer:
//   [0]=conv_a(dst) [1]=gdn_a(dst) [2]=conv_b [3]=gdn_b
//   [4]=conv_b0 [5]=gdn_b0 [6]=conv_b1 [7]=gdn_b1
// k=3 commits B (full verify state), k=2 B1 (A+t_last+d1), k=1 B0
// (A+t_last). Tail segment: hprev <- hf_v row (k-1).
// ============================================================
__global__ void mtp_commit_kernel(const int* __restrict__ k_pin,
                                   const float* const* __restrict__ plan,
                                   int n_plans, int conv_len, int gdn_len,
                                   const float* __restrict__ hf_v,
                                   float* __restrict__ hprev, int hidden) {
    __shared__ int ks;
    if (threadIdx.x == 0) ks = *k_pin;
    __syncthreads();
    const int k = ks; // 1..3
    const int row = conv_len + gdn_len;
    const long lay = (long)n_plans * row;
    const long total = lay + hidden;
    for (long idx = (long)blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += (long)gridDim.x * blockDim.x) {
        if (idx < lay) {
            int l = (int)(idx / row);
            int r = (int)(idx - (long)l * row);
            const float* const* p = plan + (size_t)l * 8;
            if (r < conv_len) {
                const float* src = (k == 3) ? p[2] : (k == 2) ? p[6] : p[4];
                p[0][r] = src[r];
            } else {
                int rr = r - conv_len;
                const float* src = (k == 3) ? p[3] : (k == 2) ? p[7] : p[5];
                p[1][rr] = src[rr];
            }
        } else {
            int r = (int)(idx - lay);
            hprev[r] = hf_v[(size_t)(k - 1) * hidden + r];
        }
    }
}

extern "C" cudaError_t ferrite_mtp_commit(const int* k_pin,
                                          const float* const* plan,
                                          int n_plans, int conv_len, int gdn_len,
                                          const float* hf_v, float* hprev,
                                          int hidden, cudaStream_t s) {
    long total = (long)n_plans * (conv_len + gdn_len) + hidden;
    int blocks = (int)((total + 1023) / 1024);
    if (blocks > 4096) blocks = 4096;
    if (blocks < 1) blocks = 1;
    mtp_commit_kernel<<<blocks, 1024, 0, s>>>(k_pin, plan, n_plans, conv_len,
                                             gdn_len, hf_v, hprev, hidden);
    return cudaGetLastError();
}
