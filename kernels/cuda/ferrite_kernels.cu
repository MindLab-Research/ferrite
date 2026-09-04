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
