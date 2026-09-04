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
// ============================================================
__global__ void rmsnorm_kernel(const float* __restrict__ x,
                               const float* __restrict__ w,
                               float* __restrict__ out,
                               int n, int dim, float eps) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= n) return;
    const float* xr = x + (size_t)row * dim;
    float* or_ = out + (size_t)row * dim;
    // one warp handles the row reduction (dim <= 16k covers 4096 hidden)
    float ss = 0.f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        ss += xr[i] * xr[i];
    }
    // warp reduce; each warp (threadIdx.y) writes its own shared slot
    __shared__ float warp_s[32];
    float lane = ss;
    for (int off = 16; off > 0; off >>= 1) lane += __shfl_down_sync(0xffffffff, lane, off);
    if (threadIdx.x == 0) warp_s[threadIdx.y] = lane / dim;
    __syncthreads();
    float inv = rsqrtf(warp_s[threadIdx.y] + eps);
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        or_[i] = xr[i] * inv * w[i];
    }
}

extern "C" cudaError_t ferrite_rmsnorm(const float* x, const float* w,
                                       float* out, int n, int dim, float eps,
                                       cudaStream_t s) {
    dim3 block(32, 4);
    dim3 grid((n + 3) / 4);
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
    __shared__ int bidx[32];
    __shared__ float bval[32];
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
    if (threadIdx.x == 0) out[row] = (float)bidx[0];
}

extern "C" cudaError_t ferrite_argmax(const float* logits, float* out, int n,
                                      int dim, cudaStream_t s) {
    dim3 block(32);
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
                                     int n, int t, int h, int d, int topk, int ctx0) {
    int row = blockIdx.x;
    if (row >= n) return;
    extern __shared__ float sm[]; // t scores (dynamic smem; t*4 bytes)
    float inv_sqrt_d = rsqrtf((float)d);
    // causal guard: query row i may only select keys j < ctx0 + i + 1
    int jmax = min(ctx0 + row + 1, t);
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
                                            float* idx, int n, int t, int h, int d,
                                            int topk, int ctx0, cudaStream_t s) {
    dim3 block(32);
    dim3 grid(n);
    size_t smem = (size_t)t * sizeof(float);
    indexer_topk_kernel<<<grid, block, smem, s>>>(qi, ki, w, idx, n, t, h, d, topk, ctx0);
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
                                   int n, int t, int h, int d, int dv, int topk) {
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
                                           float* out, int n, int t, int h, int d,
                                           int dv, int topk, cudaStream_t s) {
    // NOTE: block width must stay <= 32 — the shared reduction arrays
    // (red/reds) are [32]; 128 threads would write out of bounds.
    dim3 block(32);
    dim3 grid(n, h);
    size_t smem = (size_t)topk * sizeof(float); // dynamic smem for the topk scores
    sparse_attn_kernel<<<grid, block, smem, s>>>(q, k, v, idx, out, n, t, h, d, dv, topk);
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

    // 2. mixes: mix rows of fw against x (block-reduced dot each)
    for (int m = 0; m < mix; m++) {
        const float* row = fw + (size_t)m * nh;
        float acc = 0.f;
        for (int i = threadIdx.x; i < nh; i += blockDim.x) acc += row[i] * x[i];
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
        if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = acc;
        __syncthreads();
        if (threadIdx.x == 0) {
            float tot = 0.f;
            for (int w = 0; w < 32; w++) if (w < (blockDim.x + 31) >> 5) tot += red[w];
            mx[m] = tot * rsq;
        }
        __syncthreads();
    }

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
    hc_pre_kernel<<<s, 256, smem, stream>>>(res, fw, scale, base, li, post, comb,
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
    const float* __restrict__ x,
    const float* __restrict__ ids_f,       // [topk] f32-encoded (moe_route writes f32)
    const __nv_bfloat16* const* __restrict__ gate_ptrs,  // [e_local]
    const __nv_bfloat16* const* __restrict__ up_ptrs,    // [e_local]
    const __nv_bfloat16* __restrict__ shared_gate,       // [inter_shared, hidden]
    const __nv_bfloat16* __restrict__ shared_up,
    float* __restrict__ act,               // [topk*inter + inter_shared]
    int expert_start, int e_local, int hidden, int inter,
    int inter_shared, int topk, int rows, float limit) {
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int slot = blockIdx.y;
    int row0 = blockIdx.x * rows;
    // Layout: routed slots [0, topk) use inter rows each (full expert width);
    // the shared slot (== topk) uses inter_shared rows (TP-sharded width).
    int slot_rows, slot_base;
    const __nv_bfloat16 *gw, *uw;
    if (slot < topk) {
        slot_rows = inter;
        slot_base = slot * inter;
        int eid = (int)ids_f[slot];
        int local = eid - expert_start;
        if (local < 0 || local >= e_local) {
            // Another rank owns this expert → zero slot; the TP all-reduce
            // across ranks fills the total contribution.
            if (warp == 0) {
                for (int r = row0 + lane; r < row0 + rows && r < slot_rows; r += 32) {
                    act[(size_t)slot_base + r] = 0.f;
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
            float x0 = x[k];
            float x1 = (k + 1 < hidden) ? x[k + 1] : 0.f;
            float x2 = (k + 2 < hidden) ? x[k + 2] : 0.f;
            float x3 = (k + 3 < hidden) ? x[k + 3] : 0.f;
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
            // swiglu2 semantics (gate clamp, up clamp, silu(g)*u)
            g = fminf(g, limit);
            u = fminf(fmaxf(u, -limit), limit);
            act[(size_t)slot_base + r] = (g / (1.0f + expf(-g))) * u;
        }
    }
}

__global__ void moe_fused_down_sum_kernel(
    const float* __restrict__ ids_f,       // [topk]
    const float* __restrict__ probs,       // [topk]
    const __nv_bfloat16* const* __restrict__ down_ptrs,  // [e_local], [hidden, inter] each
    const __nv_bfloat16* __restrict__ shared_down,      // [hidden, inter_shared]
    const float* __restrict__ act,         // [topk*inter + inter_shared]
    float* __restrict__ out,               // [hidden]
    int expert_start, int e_local, int hidden, int inter,
    int inter_shared, int topk, int rows) {
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int h = blockIdx.x * rows + warp;
    if (h >= hidden) return;
    float acc = 0.f;
    for (int j = 0; j < topk; j++) {
        int eid = (int)ids_f[j];
        int local = eid - expert_start;
        if (local < 0 || local >= e_local) continue; // another rank's slot (zero act)
        float p = probs[j];
        if (p == 0.f) continue;
        const __nv_bfloat16* dwr = down_ptrs[local] + (size_t)h * inter;
        const float* aj = act + (size_t)j * inter;
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
        const float* as = act + (size_t)topk * inter;
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
    if (lane == 0) out[h] = acc;
}

// Launcher A: act stage — caller provides the act buffer
// ([topk*inter + inter_shared]) and ids (from ferrite_moe_route, device).
extern "C" cudaError_t ferrite_moe_fused_act(
    const float* x, const float* ids_f,
    const void* const* gate_ptrs, const void* const* up_ptrs,
    const void* shared_gate, const void* shared_up,
    float* act, int expert_start, int e_local, int hidden, int inter,
    int inter_shared, int topk, float limit, cudaStream_t s) {
    // rows = warps per block (256 threads / 32 = 8): each block covers 8
    // rows via the stride loop (r += warps).
    int rows = 8;
    int max_rows = inter > inter_shared ? inter : inter_shared;
    dim3 grid((max_rows + rows - 1) / rows, topk + 1);
    moe_fused_act_kernel<<<grid, 256, 0, s>>>(
        x, ids_f,
        (const __nv_bfloat16* const*)gate_ptrs, (const __nv_bfloat16* const*)up_ptrs,
        (const __nv_bfloat16*)shared_gate, (const __nv_bfloat16*)shared_up,
        act, expert_start, e_local, hidden, inter, inter_shared, topk, rows, limit);
    return cudaGetLastError();
}

// Launcher B: down + weighted-sum + shared stage — caller provides act
// ([topk*inter + inter_shared]) and out ([hidden]).
extern "C" cudaError_t ferrite_moe_fused_down_sum(
    const float* ids_f, const float* probs,
    const void* const* down_ptrs, const void* shared_down,
    const float* act, float* out,
    int expert_start, int e_local, int hidden, int inter,
    int inter_shared, int topk,
    cudaStream_t s) {
    // CRITICAL: rows must equal warps per block (256/32 = 8). The kernel
    // assigns ONE hidden-row per warp with no stride loop — rows=128 made
    // grid.x = hidden/128 while each block only computed 8 rows (4096-row
    // hidden: 32 blocks × 8 = 256 rows computed, 94% of out was garbage).
    int rows = 8;
    dim3 grid((hidden + rows - 1) / rows);
    moe_fused_down_sum_kernel<<<grid, 256, 0, s>>>(
        ids_f, probs,
        (const __nv_bfloat16* const*)down_ptrs, (const __nv_bfloat16*)shared_down,
        act, out, expert_start, e_local, hidden, inter, inter_shared, topk, rows);
    return cudaGetLastError();
}
