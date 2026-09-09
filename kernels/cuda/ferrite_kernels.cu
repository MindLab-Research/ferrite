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
#include <cstdio>
#include <cmath>
#include <cuda_fp8.h>

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
// Direct-load kernels (mmap disk→GPU path — weights never materialize on
// the CPU): the preload path H2Ds the checkpoint's raw bytes and converts
// IN DEVICE MEMORY.
//
// dequant_e4m3_block: fp8 e4m3 [rows, cols] × 128×128 block scales →
// bf16 resident (the CPU path's dequant_block + bf16-pack fused; the
// e4m3→float conversion uses the fp8 intrinsic (half-raw — exact: e4m3's
// 4-bit exponent/3-bit mantissa are representable in fp16), scale fetch
// per element by (row/128, col/128) block index, bf16 truncate `>> 16`
// — identical to the CPU pack's f32→bf16 rounding).
// ============================================================
__global__ void dequant_e4m3_block_kernel(const unsigned char* __restrict__ w,
                                          const float* __restrict__ scale,
                                          unsigned short* __restrict__ out,
                                          long rows, long cols,
                                          int srows, int scols) {
    long total = rows * cols;
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += (long)gridDim.x * blockDim.x) {
        long r = i / cols, c = i - r * cols;
        int sr = (int)(r >> 7); if (sr >= srows) sr = srows - 1;
        int sc = (int)(c >> 7); if (sc >= scols) sc = scols - 1;
        float v = __half2float(__nv_cvt_fp8_to_halfraw(w[i], __NV_E4M3))
                  * scale[(long)sr * scols + sc];
        out[i] = (unsigned short)(__float_as_uint(v) >> 16);
    }
}

extern "C" cudaError_t ferrite_dequant_e4m3_block(const void* w, const void* scale,
                                                  void* out, long rows, long cols,
                                                  int srows, int scols,
                                                  cudaStream_t s) {
    long total = rows * cols;
    if (total <= 0) return cudaSuccess;
    int threads = 256;
    long blocks = (total + threads - 1) / threads;
    if (blocks > (1L << 31) - 1) blocks = (1L << 31) - 1;
    dequant_e4m3_block_kernel<<<(unsigned)blocks, threads, 0, s>>>(
        (const unsigned char*)w, (const float*)scale, (unsigned short*)out,
        rows, cols, srows, scols);
    return cudaGetLastError();
}

// bf16 raw → f32 resident (the embed-table expand: checkpoint bf16 bytes
// H2D'd straight from the mmap, expanded on device — 2.5 GB never crosses
// a CPU conversion pass).
// NOTE: bit-convert explicitly (u16 << 16 into f32 bits) instead of
// __bfloat162float — the intrinsic path produced garbage on sm_103a
// (unit-test verified: got 0x473f0000(48896) for bf16 0xbf78(-0.97));
// the bit form is the mathematically identical expansion (bf16 IS the
// high 16 bits of f32) with zero intrinsic dependency.
__global__ void bf16_to_f32_kernel(const unsigned short* __restrict__ in,
                                    float* __restrict__ out, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __uint_as_float(((unsigned)in[i]) << 16);
}

extern "C" cudaError_t ferrite_bf16_to_f32(const void* in, void* out,
                                           long n, cudaStream_t s) {
    if (n <= 0) return cudaSuccess;
    int threads = 256;
    long blocks = (n + threads - 1) / threads;
    bf16_to_f32_kernel<<<(unsigned)blocks, threads, 0, s>>>(
        (const unsigned short*)in, (float*)out, n);
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
    // Cross-warp reduce sized by blockDim (the old version hardcoded 8 warps —
    // growing the block then silently dropped 3/4 of the sum and looked "8%
    // faster"). 32 slots covers up to 1024 threads.
    __shared__ float red[32];
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = lane;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.f;
        for (int i = 0; i < (int)(blockDim.x >> 5); i++) t += red[i];
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
    // 1024 threads/block: grid(n) is only 16 blocks at n=16, so per-block
    // latency dominates; 4 elems/thread instead of 16. The reduce above is
    // now blockDim-sized, so this is safe (it was NOT before).
    dim3 block(1024);
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
    // ONE token per block, 256 threads: the old block(32,4)/grid(n/4) gave only
    // 512 threads for the whole grid and each thread serially walked dim/32 =
    // 128 elements (pure latency; the kernel is called per GDN layer).
    const int row = blockIdx.x;
    if (row >= n) return;
    const float* xr = x + (size_t)row * dim;
    const float* gr = gate + (size_t)row * dim;
    float* or_ = out + (size_t)row * dim;
    float ss = 0.f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) ss += xr[i] * xr[i];
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) ss += __shfl_down_sync(0xffffffff, ss, off);
    __shared__ float warp_s[8];
    if ((threadIdx.x & 31) == 0) warp_s[threadIdx.x >> 5] = ss;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.f;
        for (int wq = 0; wq < (blockDim.x >> 5); wq++) t += warp_s[wq];
        warp_s[0] = t / dim;
    }
    __syncthreads();
    const float inv = rsqrtf(warp_s[0] + eps);
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        // Glm5NextTextRMSNormGated: y = rmsnorm(x) * w * sigmoid(gate)
        or_[i] = xr[i] * inv * w[i] / (1.0f + __expf(-gr[i]));
    }
}

extern "C" cudaError_t ferrite_gated_rmsnorm(const float* x, const float* gate,
                                            const float* w, float* out,
                                            int n, int dim, float eps,
                                            cudaStream_t s) {
    dim3 block(256);
    dim3 grid(n);
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
                              float* __restrict__ snaps,   // [n-1][ch, hist] t-snapshots (null = none) — MTP verify B_k
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
    // N-UNIFIED snapshots: state after token t = the last hist inputs
    // [x_{t-hist+1}..x_t] = stream[t+1 .. t+hist] (verified: hist=3,
    // state after t = [x_{t-2},x_{t-1},x_t] = stream[t+1..t+3]). The old
    // verify path did a PER-TOKEN conv1d launch + a D2D snapshot copy per t
    // (n=3: 3 launches + 2 cudaMemcpyAsync per GDN layer × 34 layers); this
    // writes the snapshots straight from smem in the SAME launch.
    if (snaps != nullptr) {
        for (int t = threadIdx.x; t < n - 1; t += blockDim.x) {
            float* sn = snaps + (size_t)t * ch * hist + (size_t)c * hist;
            #pragma unroll 4
            for (int hh = 0; hh < hist; hh++)
                sn[hh] = stream[t + 1 + hh];
        }
        __syncthreads();
    }
    for (int h = threadIdx.x; h < hist; h += blockDim.x)
        state_out[c * hist + h] = stream[n + h];
}

extern "C" cudaError_t ferrite_causal_conv1d(const float* x, const float* w,
                                             const float* state_in, float* out,
                                             float* state_out, float* snaps, int n, int ch,
                                             int conv, cudaStream_t s) {
    int hist = conv - 1;
    dim3 block(128);
    dim3 grid(ch);
    size_t smem = (size_t)(hist + n) * sizeof(float);
    conv1d_kernel<<<grid, block, smem, s>>>(x, w, state_in, out, state_out, snaps, n, ch, conv);
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
        // 4 accumulators: the fp32 dot was a serial FMA chain (the compiler
        // may not reassociate fp), so the 4-cycle FMA latency dominated.
        // Order changes -> ~1e-7 relative, far below the model's tolerance.
        float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
        int i = 0;
        for (; i + 3 < dk; i += 4) {
            a0 += kh[i]     * S[(size_t)(i)     * dv + j];
            a1 += kh[i + 1] * S[(size_t)(i + 1) * dv + j];
            a2 += kh[i + 2] * S[(size_t)(i + 2) * dv + j];
            a3 += kh[i + 3] * S[(size_t)(i + 3) * dv + j];
        }
        for (; i < dk; i++) a0 += kh[i] * S[(size_t)i * dv + j];
        float acc = (a0 + a1) + (a2 + a3);
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
        // 4 accumulators: the fp32 dot was a serial FMA chain (the compiler
        // may not reassociate fp), so the 4-cycle FMA latency dominated.
        // Order changes -> ~1e-7 relative, far below the model's tolerance.
        float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
        int i = 0;
        for (; i + 3 < dk; i += 4) {
            a0 += qh[i]     * S[(size_t)(i)     * dv + j];
            a1 += qh[i + 1] * S[(size_t)(i + 1) * dv + j];
            a2 += qh[i + 2] * S[(size_t)(i + 2) * dv + j];
            a3 += qh[i + 3] * S[(size_t)(i + 3) * dv + j];
        }
        for (; i < dk; i++) a0 += qh[i] * S[(size_t)i * dv + j];
        float acc = (a0 + a1) + (a2 + a3);
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
    // Was: one thread per ROW (only 128 of 512 threads active, 75% idle, and
    // each active thread ran a serial dv-long loop). Now: precompute the
    // per-row decay, then sweep all dk*dv elements with the full block.
    for (int i = threadIdx.x; i < dk; i += blockDim.x) gh[i] = expf(gh[i]);
    __syncthreads();
    for (int idx = threadIdx.x; idx < dk * dv; idx += blockDim.x) {
        const int i = idx / dv, j = idx - i * dv;
        S[(size_t)i * spitch + j] *= gh[i];
    }
    __syncthreads();
    // 2. kS = S^T k
    for (int j = threadIdx.x; j < dv; j += blockDim.x) {
        // 4 accumulators: the fp32 dot was a serial FMA chain (the compiler
        // may not reassociate fp), so the 4-cycle FMA latency dominated.
        // Order changes -> ~1e-7 relative, far below the model's tolerance.
        float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
        int i = 0;
        for (; i + 3 < dk; i += 4) {
            a0 += kh[i]     * S[(size_t)(i)     * spitch + j];
            a1 += kh[i + 1] * S[(size_t)(i + 1) * spitch + j];
            a2 += kh[i + 2] * S[(size_t)(i + 2) * spitch + j];
            a3 += kh[i + 3] * S[(size_t)(i + 3) * spitch + j];
        }
        for (; i < dk; i++) a0 += kh[i] * S[(size_t)i * spitch + j];
        float acc = (a0 + a1) + (a2 + a3);
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
        // 4 accumulators: the fp32 dot was a serial FMA chain (the compiler
        // may not reassociate fp), so the 4-cycle FMA latency dominated.
        // Order changes -> ~1e-7 relative, far below the model's tolerance.
        float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
        int i = 0;
        for (; i + 3 < dk; i += 4) {
            a0 += qh[i]     * S[(size_t)(i)     * spitch + j];
            a1 += qh[i + 1] * S[(size_t)(i + 1) * spitch + j];
            a2 += qh[i + 2] * S[(size_t)(i + 2) * spitch + j];
            a3 += qh[i + 3] * S[(size_t)(i + 3) * spitch + j];
        }
        for (; i < dk; i++) a0 += qh[i] * S[(size_t)i * spitch + j];
        float acc = (a0 + a1) + (a2 + a3);
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

// PDL v5 helper: launch with programmatic stream serialization (B-side PDL).
// The kernel's OPENING cudaGridDependencySynchronize() gates its data reads
// on the predecessor's completion; the launch overhead (grid init, prologue)
// overlaps the predecessor's tail — this is the mega-graph node-gap killer
// (~900 nodes × ~2µs launch gap per decode step). No-op semantics when
// FERRITE_PDL is unset (plain <<<>>>); when set, the graph captures the
// launch-with-attr as a PDL node (ferrite_pdl_exp mode 3 verified capture).
// NOTE: the KERNEL must start with cudaGridDependencySynchronize() before
// touching its predecessors' outputs — every pdl_or_plain'd kernel below
// carries the __CUDA_ARCH__ >= 900 guard block at entry.
// NOTE: the kernel<<<>>> launch syntax does NOT accept a template parameter
// (the 6-error "return value type does not match" — the launch triple-bracket
// is only valid on a literal kernel symbol). Both paths therefore go through
// cudaLaunchKernelEx (the C++ variadic template): WITHOUT the PDL attr it is
// a plain launch (identical node in the graph capture), WITH the attr it is
// the PDL node. graph capture records either faithfully (pdl_exp mode 3).
template <typename K, typename... Args>
static inline cudaError_t pdl_or_plain(K kern, dim3 grid, dim3 block,
                                       size_t smem, cudaStream_t stream, Args... args) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid; cfg.blockDim = block;
    cfg.dynamicSmemBytes = smem; cfg.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = 1;
    if (ferrite_pdl_enabled()) {
        cfg.attrs = attrs; cfg.numAttrs = 1;
    }
    return cudaLaunchKernelEx(&cfg, kern, args...);
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
// BATCHED per-seq GDN kernels (the batched decode, n=1 token per seq × B
// seqs): the per-seq loop (B × conv1d + B × gdn_chunk_v2 launches —
// measured +4.2ms of the n=4 step's +15ms over n=1: the small grids
// (conv grid(ch) with 1 live thread/block at n=1; gdn grid(1,h) = 64
// blocks = 43% SM) serialize per seq) collapses to ONE launch each with
// a per-seq STATE POINTER TABLE (the MoE expert-ptr-table pattern):
//   conv1d_batched: grid(B*ch/TPB) — each thread one (seq, channel):
//   the 3-tap FIR + slide vs state_ptrs[seq]'s channel slice. The FIR
//   accumulation order = conv1d_kernel's sequential i (bit-identical).
//   gdn_chunk_batched: grid(B, h) — the gdn_step_v2 body per (seq, head)
//   with Sg = state_ptrs[seq] + hd*dk*dv. The 5-phase accumulation is
//   the gdn_step_v2's exactly (decay → kS → delta → o → store). The
//   states live OUTSIDE the pooled DevBufs (dev_state — fixed addresses
//   for the graph's lifetime); the tables are per-layer DevBufs uploaded
//   once per composition (the graph records the frozen H2D + pointers).
// ============================================================
__global__ void conv1d_batched_kernel(
    const float* __restrict__ x,            // [B, ch] (the batched qkv rows)
    const float* __restrict__ w,           // [ch, conv]
    float* const* __restrict__ state_ptrs,  // [B] per-seq conv states [ch, hist]
    float* __restrict__ out,               // [B, ch]
    int B, int ch, int conv) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= B * ch) return;
    int seq = tid / ch, c = tid % ch;
    int hist = conv - 1;
    float* cs = state_ptrs[seq] + (size_t)c * hist;
    const float* wc = w + (size_t)c * conv;
    // FIR (the conv1d_kernel's sequential i order — bit-identical):
    // out = Σ_{i<conv} w[i]·stream[hist - (conv-1) + i], stream = [s0..s2, x]
    float fir = 0.f;
    for (int i = 0; i < hist; i++) fir += wc[i] * cs[i];
    fir += wc[hist] * x[tid];
    out[tid] = fir;
    // slide the window: [s0,s1,s2] → [s1,s2,x] (per-thread channel ownership)
    for (int i = 0; i + 1 < hist; i++) cs[i] = cs[i + 1];
    cs[hist - 1] = x[tid];
}

extern "C" cudaError_t ferrite_conv1d_batched(const float* x, const float* w,
                                              float* const* state_ptrs,
                                              float* out, int B, int ch, int conv,
                                              cudaStream_t s) {
    int total = B * ch;
    if (total <= 0) return cudaSuccess;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    conv1d_batched_kernel<<<blocks, threads, 0, s>>>(x, w, state_ptrs, out, B, ch, conv);
    return cudaGetLastError();
}

__global__ void gdn_chunk_batched_kernel(
    const float* __restrict__ q,            // [B, h, dk]
    const float* __restrict__ k,            // [B, h, dk]
    const float* __restrict__ v,            // [B, h, dv]
    const float* __restrict__ beta,         // [B, h]
    const float* __restrict__ gate,         // [B, h, dk]
    const float* __restrict__ a_log,        // [h]
    float* const* __restrict__ state_ptrs,  // [B] per-seq [h, dk, dv] states
    float* __restrict__ out,                // [B, h, dv]
    int h, int dk, int dv) {
    int seq = blockIdx.x;   // B
    int hd = blockIdx.y;     // h
    // gdn_step_v2's body (n=1) with the per-seq state indirection — the 5
    // phases' accumulation order is IDENTICAL (bit-equal per seq vs the
    // per-seq launches; the batched launch only changes the grid).
    float bt = beta[(size_t)seq * h + hd];
    const size_t spitch = (size_t)dv + 1; // padded row stride (bank conflicts)
    extern __shared__ float sm[];
    float* S = sm;                          // [dk * (dv+1)]
    float* ks = S + (size_t)dk * spitch;   // [dv]
    float* kh = ks + dv;                   // [dk]
    float* vh = kh + dk;                   // [dv]
    float* qh = vh + dv;                   // [dk]
    float* gh = qh + dk;                   // [dk]
    const int base = (int)((size_t)seq * h + hd);
    for (int i = threadIdx.x; i < dk; i += blockDim.x) {
        gh[i] = gate[(size_t)base * dk + i];
        qh[i] = q[(size_t)base * dk + i];
        kh[i] = k[(size_t)base * dk + i];
    }
    for (int j = threadIdx.x; j < dv; j += blockDim.x)
        vh[j] = v[(size_t)base * dv + j];
    float* Sg = state_ptrs[seq] + (size_t)hd * dk * dv;
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
        // 4 accumulators: the fp32 dot was a serial FMA chain (the compiler
        // may not reassociate fp), so the 4-cycle FMA latency dominated.
        // Order changes -> ~1e-7 relative, far below the model's tolerance.
        float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
        int i = 0;
        for (; i + 3 < dk; i += 4) {
            a0 += kh[i]     * S[(size_t)(i)     * spitch + j];
            a1 += kh[i + 1] * S[(size_t)(i + 1) * spitch + j];
            a2 += kh[i + 2] * S[(size_t)(i + 2) * spitch + j];
            a3 += kh[i + 3] * S[(size_t)(i + 3) * spitch + j];
        }
        for (; i < dk; i++) a0 += kh[i] * S[(size_t)i * spitch + j];
        float acc = (a0 + a1) + (a2 + a3);
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
        // 4 accumulators: the fp32 dot was a serial FMA chain (the compiler
        // may not reassociate fp), so the 4-cycle FMA latency dominated.
        // Order changes -> ~1e-7 relative, far below the model's tolerance.
        float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
        int i = 0;
        for (; i + 3 < dk; i += 4) {
            a0 += qh[i]     * S[(size_t)(i)     * spitch + j];
            a1 += qh[i + 1] * S[(size_t)(i + 1) * spitch + j];
            a2 += qh[i + 2] * S[(size_t)(i + 2) * spitch + j];
            a3 += qh[i + 3] * S[(size_t)(i + 3) * spitch + j];
        }
        for (; i < dk; i++) a0 += qh[i] * S[(size_t)i * spitch + j];
        float acc = (a0 + a1) + (a2 + a3);
        out[(size_t)base * dv + j] = acc;
    }
    __syncthreads();
    // 5. store state back
    for (int idx = threadIdx.x; idx < dk * dv; idx += blockDim.x)
        Sg[idx] = S[(size_t)(idx / dv) * spitch + (idx % dv)];
}

extern "C" cudaError_t ferrite_gdn_chunk_batched(const float* q, const float* k,
                                                 const float* v, const float* beta,
                                                 const float* gate, const float* a_log,
                                                 float* const* state_ptrs,
                                                 float* out, int B, int h, int dk, int dv,
                                                 cudaStream_t s) {
    if (B <= 0) return cudaSuccess;
    size_t smem = (size_t)dk * (dv + 1) * sizeof(float)
                  + (size_t)(dv + dk + dv + dk + dk) * sizeof(float);
    if (smem > 48 * 1024) {
        cudaError_t e = cudaFuncSetAttribute(gdn_chunk_batched_kernel,
                                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        if (e != cudaSuccess) return e;
    }
    dim3 block(512);
    dim3 grid(B, h, 1);
    gdn_chunk_batched_kernel<<<grid, block, smem, s>>>(
        q, k, v, beta, gate, a_log, state_ptrs, out, h, dk, dv);
    return cudaGetLastError();
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
    // 256 threads (8 warps): each round's scan over e=288 is 2 iterations
    // instead of 9 with a single warp. Warp reduce + one smem cross-warp step.
    __shared__ int wbest[8];
    __shared__ float wval[8];
    for (int r = 0; r < topk; r++) {
        int best = -1;
        float bv = -1e30f;
        for (int j = threadIdx.x; j < e; j += blockDim.x) {
            if (ch[j] > bv) { bv = ch[j]; best = j; }
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = __shfl_down_sync(0xffffffff, bv, off);
            const int oi = __shfl_down_sync(0xffffffff, best, off);
            if (ov > bv) { bv = ov; best = oi; }
        }
        const int wid = threadIdx.x >> 5;
        if ((threadIdx.x & 31) == 0) { wbest[wid] = best; wval[wid] = bv; }
        __syncthreads();
        if (threadIdx.x == 0) {
            int bi = -1;
            float bvv = -1e30f;
            for (int w = 0; w < (int)(blockDim.x >> 5); w++) {
                if (wval[w] > bvv) { bvv = wval[w]; bi = wbest[w]; }
            }
            if (bi >= 0) {
                ids[(size_t)row * topk + r] = (float)bi;
                ch[bi] = -1e30f; // remove
            } else {
                // All logits NaN (e.g. an upstream layer exploded): mark the
                // slot INVALID instead of indexing ch[-1] (an OOB smem write
                // that corrupted the sigmoid row) — consumers already skip
                // ids < expert_start.
                ids[(size_t)row * topk + r] = -1.0f;
            }
        }
        __syncthreads(); // ch[best] must be visible to all threads next round
    }
    // renorm pass (block-wide: 2 passes over topk + block reduce). Was a
    // single-thread loop over topk=2048 (~4.5µs x 42 calls/step = 0.19ms).
    float lsum = 0.f;
    for (int r = threadIdx.x; r < topk; r += blockDim.x) {
        int j = (int)ids[(size_t)row * topk + r];
        // j can be -1 (all-NaN logits → invalid slot): sm[-1] was an OOB smem read.
        float val = (j >= 0) ? sm[j] : 0.f;
        probs[(size_t)row * topk + r] = val;
        lsum += val;
    }
    for (int off = 16; off > 0; off >>= 1) lsum += __shfl_down_sync(0xffffffff, lsum, off);
    __shared__ float rsum[32];
    if ((threadIdx.x & 31) == 0) rsum[threadIdx.x >> 5] = lsum;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.f;
        int nw = (blockDim.x + 31) >> 5;
        for (int w = 0; w < nw; w++) s += rsum[w];
        rsum[0] = s + 1e-9f;
    }
    __syncthreads();
    const float rdenom = rsum[0];
    for (int r = threadIdx.x; r < topk; r += blockDim.x)
        probs[(size_t)row * topk + r] = probs[(size_t)row * topk + r] / rdenom * scale;
}

// ============================================================
// router_gemm_route_fused: the router GEMV + the route in ONE kernel
// (the "last block" pattern). The separate moe_route was 14.1µs × 44 layers
// = 0.62ms/step — 12µs launch overhead (the 1-block kernel's graph node
// dispatch) + 2µs execution. The fusion eliminates the launch overhead:
// the router GEMV's 160 blocks compute the logits, the last block (the
// atomic counter) does the route (sigmoid + topk + renorm). The route's
// FP is IDENTICAL to the moe_route_kernel (the same sigmoid, the same
// topk selection order, the same renorm).
// ============================================================
__global__ void router_gemm_route_fused_kernel(
    const float* __restrict__ x,           // [hidden] the input
    const __nv_bfloat16* __restrict__ w,   // [n_exp, hidden] the router weight
    const float* __restrict__ bias,        // [n_exp] the router bias
    float* __restrict__ probs,             // [n, topk] the output probs
    float* __restrict__ ids,               // [n, topk] the output ids
    float* __restrict__ logits,            // [n_exp] the scratch (the logits)
    unsigned* __restrict__ ctr,            // [1] the atomic counter
    int n_exp, int hidden, int topk, float scale) {

    int e = blockIdx.x; // 1 expert per block (grid = n_exp)
    if (e >= n_exp) return;
    int lane = threadIdx.x & 31;

    // ── GEMV: logit[e] = w[e,:] · x (uint4-vectorized, the same pattern as
    // gemv_bf16_v2<WPR=1> but per-block: 1 expert per block, 256 threads) ──
    const __nv_bfloat16* wr = w + (size_t)e * hidden;
    float acc = 0.f;
    // uint4 vectorized (8 bf16 per load — hidden % 8 == 0 for GLM-5.3)
    // 4 accumulators: the single `acc` is a 512-iteration serial FMA chain per
    // thread (hidden/8 iterations at blockDim=256); fp32 cannot be
    // reassociated by the compiler, so split it explicitly.
    float q0 = 0.f, q1 = 0.f, q2 = 0.f, q3 = 0.f;
    for (int k = threadIdx.x * 8; k + 7 < hidden; k += blockDim.x * 8) {
        float4 xa = *reinterpret_cast<const float4*>(x + k);
        float4 xb = *reinterpret_cast<const float4*>(x + k + 4);
        uint4 wv = *reinterpret_cast<const uint4*>(wr + k);
        const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(&wv);
        float2 f0 = __bfloat1622float2(w2[0]), f1 = __bfloat1622float2(w2[1]);
        float2 f2 = __bfloat1622float2(w2[2]), f3 = __bfloat1622float2(w2[3]);
        q0 += xa.x * f0.x + xa.y * f0.y;
        q1 += xa.z * f1.x + xa.w * f1.y;
        q2 += xb.x * f2.x + xb.y * f2.y;
        q3 += xb.z * f3.x + xb.w * f3.y;
    }
    acc = (q0 + q1) + (q2 + q3);
    for (int k = threadIdx.x + ((hidden >> 3) << 3); k < hidden; k += blockDim.x) {
        acc += x[k] * __bfloat162float(wr[k]);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, off);
    }
    __shared__ float ws[16];
    if (lane == 0) ws[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        float tot = 0.f;
        for (int i = 0; i < (blockDim.x + 31) >> 5; i++) tot += ws[i];
        logits[e] = tot;
    }
    __threadfence();
    __syncthreads();

    // ── The "last block" pattern: the atomic counter ──
    __shared__ int is_last;
    if (threadIdx.x == 0) {
        unsigned prev = atomicAdd(ctr, 1u);
        is_last = (prev == (unsigned)(n_exp - 1)) ? 1 : 0;
    }
    __syncthreads();
    if (!is_last) return;

    // ── ROUTE (the last block only): sigmoid + topk + renorm ──
    // The SAME code as moe_route_kernel (the identical sigmoid, the same
    // topk selection order, the same renorm — FP-safe).
    extern __shared__ float sm[]; // [n_exp] sigmoid + [n_exp] choice
    float* ch = sm + n_exp;
    for (int j = threadIdx.x; j < n_exp; j += blockDim.x)
        sm[j] = 1.0f / (1.0f + expf(-logits[j]));
    __syncthreads();
    for (int j = threadIdx.x; j < n_exp; j += blockDim.x)
        ch[j] = sm[j] + bias[j];
    __syncthreads();
    // selection topk (the same as moe_route_kernel)
    for (int r = 0; r < topk; r++) {
        __shared__ int bidx[32];
        __shared__ float bval[32];
        int best = -1;
        float bv = -1e30f;
        for (int j = threadIdx.x; j < n_exp; j += blockDim.x) {
            if (ch[j] > bv) { bv = ch[j]; best = j; }
        }
        for (int off = 16; off > 0; off >>= 1) {
            float ov = __shfl_down_sync(0xffffffff, bv, off);
            int oi = __shfl_down_sync(0xffffffff, best, off);
            if (ov > bv) { bv = ov; best = oi; }
        }
        int warp = threadIdx.x >> 5;
        if (lane == 0) { bidx[warp] = best; bval[warp] = bv; }
        __syncthreads();
        if (threadIdx.x == 0) {
            int sel = -1;
            float sv = -1e30f;
            for (int wv = 0; wv < (blockDim.x >> 5); wv++) {
                if (bval[wv] > sv) { sv = bval[wv]; sel = bidx[wv]; }
            }
            if (sel >= 0) {
                ids[r] = (float)sel;
                ch[sel] = -1e30f;
            } else {
                ids[r] = -1.0f;
            }
        }
        __syncthreads();
    }
    // renorm (block-wide: same math as moe_route_kernel's parallel renorm).
    float lsum2 = 0.f;
    for (int r = threadIdx.x; r < topk; r += blockDim.x) {
        int j = (int)ids[r];
        // j can be -1 (all-NaN logits → invalid slot): sm[-1] was an OOB smem read.
        float val = (j >= 0) ? sm[j] : 0.f;
        probs[r] = val;
        lsum2 += val;
    }
    for (int off = 16; off > 0; off >>= 1) lsum2 += __shfl_down_sync(0xffffffff, lsum2, off);
    __shared__ float rsum2[32];
    if ((threadIdx.x & 31) == 0) rsum2[threadIdx.x >> 5] = lsum2;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.f;
        int nw = (blockDim.x + 31) >> 5;
        for (int w = 0; w < nw; w++) s += rsum2[w];
        rsum2[0] = s + 1e-9f;
        *ctr = 0u; // reset for the next invocation (the mega graph replays)
    }
    __syncthreads();
    const float rdenom2 = rsum2[0];
    for (int r = threadIdx.x; r < topk; r += blockDim.x)
        probs[r] = probs[r] / rdenom2 * scale;
}

extern "C" cudaError_t ferrite_router_gemm_route_fused(
    const float* x, const void* w, const float* bias,
    float* probs, float* ids, float* logits, unsigned* ctr,
    int n_exp, int hidden, int topk, float scale, cudaStream_t s) {
    dim3 block(256);
    dim3 grid(n_exp);
    size_t smem = 2 * (size_t)n_exp * sizeof(float);
    router_gemm_route_fused_kernel<<<grid, block, smem, s>>>(
        x, (const __nv_bfloat16*)w, bias, probs, ids, logits, ctr,
        n_exp, hidden, topk, scale);
    return cudaGetLastError();
}

extern "C" cudaError_t ferrite_moe_route(const float* logits, const float* bias,
                                        float* probs, float* ids, int n, int e,
                                        int topk, float scale, cudaStream_t s) {
    // DIAGNOSTIC ONLY (FERRITE_MOE_SKIP=1): skip the launch to A/B the MoE's
    // share of the step. Output is garbage by construction; timing only.
    static const bool moe_skip_ = getenv("FERRITE_MOE_SKIP") != nullptr || getenv("FERRITE_SKIP_ROUTE") != nullptr;
    if (moe_skip_) return cudaSuccess;

    dim3 block(256);   // 8 warps: the top-k scan over e is 2 iters/round vs 9
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
                                     int n, int h, int d, int topk_max,
                                     const int* __restrict__ total_ptr, int kpool_val, int n_fixed) {
    int total = *total_ptr; // actual total tokens from pinned memory
    int t = (total + kpool_val - 1) / kpool_val; // DERIVE npools from total
    int select_k = min(topk_max, t); // LIVE select_k (graph-safe: grows with cache)
    int ctx0 = total - n_fixed; // derive from pinned total
    int ctx0_pools = ctx0 / kpool_val;
    int row = blockIdx.x;
    if (row >= n) return;
    extern __shared__ float sm[]; // t scores (sized for MAX at launch; graph-safe)
    float inv_sqrt_d = rsqrtf((float)d);
    // causal guard: query row i may only select keys j < ctx0_pools + i + 1
    int jmax = min(ctx0_pools + row + 1, t);
    // FAST PATH FIRST (v12): when all causally-valid pools are selected
    // (select_k >= jmax — ALWAYS true in decode: t pools ~250 << topk_max
    // 8195), idx is a direct enumeration and the H-split scoring below is
    // dead code (nothing reads sm[] on this path). 37.7us -> ~5us: skip the
    // 21us single-block scoring + its ~560KB qi/ki reads entirely.
    if (select_k >= jmax) {
        for (int r = threadIdx.x; r < select_k; r += blockDim.x)
            idx[(size_t)row * topk_max + r] = (r < jmax) ? (float)r : -1.0f;
        for (int r = select_k + threadIdx.x; r < topk_max; r += blockDim.x)
            idx[(size_t)row * topk_max + r] = -1.0f;
        return;
    }
    // ═══ H-SPLIT SCORING (v11): 4 threads per pool (4 heads each) — the
    // 1-block 85µs scoring (129 pools × 16 heads × 64 dims at 1 SM) becomes
    // 4× parallel (~21µs). The partials [4][t] in the smem's extended area
    // (sm[t .. t+4t)); block 0 reduces (quarter-ascending = head-ascending).
    // FP NOTE: the 4-partial sum (h0-3)+(h4-7)+(h8-11)+(h12-15) differs
    // from the serial h0+h1+...+h15 by ~1-2 ulp — the topk selection is
    // score-threshold based (robust to this if the scores aren't at ties).
    {
        int tid = threadIdx.x;
        int pool = tid >> 2;      // pool = tid / 4
        int quarter = tid & 3;    // h quarter (0-3)
        if (pool < t) {
            const float* k = ki + (size_t)pool * d;
            float s = 0.f;
            if (pool < jmax) {
                int h0 = quarter * (h >> 2);
                int h1 = h0 + (h >> 2);
                for (int hi = h0; hi < h1; hi++) {
                    const float* q = qi + (size_t)row * (h * d) + hi * d;
                    float dot = 0.f;
                    float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
                    for (int l = 0; l + 3 < d; l += 4) {
                        float4 qv = *reinterpret_cast<const float4*>(q + l);
                        float4 kv = *reinterpret_cast<const float4*>(k + l);
                        d0 += qv.x * kv.x; d1 += qv.y * kv.y; d2 += qv.z * kv.z; d3 += qv.w * kv.w;
                    }
                    dot = (d0 + d1) + (d2 + d3);
                    s += w[(size_t)row * h + hi] * fmaxf(dot, 0.f); // relu
                }
            }
            sm[t + pool * 4 + quarter] = s; // partial (extended smem area)
        }
    }
    __syncthreads();
    // Reduce the 4 partials (quarter-ascending = head-ascending order)
    for (int j = threadIdx.x; j < t; j += blockDim.x) {
        float total = sm[t + j * 4] + sm[t + j * 4 + 1] + sm[t + j * 4 + 2] + sm[t + j * 4 + 3];
        sm[j] = (j < jmax) ? total * inv_sqrt_d : -INFINITY;
    }
    __syncthreads();
    // (v12: the select_k >= jmax fast path moved BEFORE the scoring — the
    // H-split scoring is dead code on that branch, which is ALWAYS taken in
    // decode: t pools ~250 << topk_max 8195.)
    // selection topk (warp-shuffle reduce, blockDim-agnostic): scoring was
    // 32 threads (96 total on the verify chain — 96/4736 cores busy, 144us/
    // inst O(len)); 256 threads = 8x lanes. Strict > keeps the LOWEST lane /
    // warp index on ties — same selection as the old 32-thread tree.
    // GRAPH-SAFE: select_k derived LIVE from the pinned total (not frozen at
    // capture — the old frozen select_k made the verify graph's attention see
    // only capture-time npools while the draft's live select_k grew with the
    // cache → d1≠a0 → MTP accept collapse).
    for (int r = 0; r < select_k; r++) {
        __shared__ int bidx[8];
        __shared__ float bval[8];
        int best = -1;
        float bv = -INFINITY;
        for (int j = threadIdx.x; j < t; j += blockDim.x) {
            if (sm[j] > bv) { bv = sm[j]; best = j; }
        }
        for (int off = 16; off > 0; off >>= 1) {
            float ov = __shfl_down_sync(0xffffffff, bv, off);
            int oi = __shfl_down_sync(0xffffffff, best, off);
            if (ov > bv) { bv = ov; best = oi; }
        }
        int warp = threadIdx.x >> 5;
        if ((threadIdx.x & 31) == 0) { bidx[warp] = best; bval[warp] = bv; }
        __syncthreads();
        if (threadIdx.x == 0) {
            int sel = -1;
            float sv = -INFINITY;
            for (int w = 0; w < (blockDim.x >> 5); w++) {
                if (bval[w] > sv) { sv = bval[w]; sel = bidx[w]; }
            }
            if (sel >= 0) {
                idx[(size_t)row * topk_max + r] = (float)sel;
                sm[sel] = -INFINITY;
            } else {
                idx[(size_t)row * topk_max + r] = -1.0f; // invisible: skip at expansion
            }
        }
        __syncthreads();
    }
    // pad the (select_k..topk_max) tail: expand only reads r < select_k but
    // keep the buffer deterministic (graph-safe: fixed stride topk_max).
    for (int r = select_k + threadIdx.x; r < topk_max; r += blockDim.x) {
        idx[(size_t)row * topk_max + r] = -1.0f;
    }
}

extern "C" cudaError_t ferrite_indexer_topk(const float* qi, const float* ki,
                                            const float* w,
                                            float* idx, int n, int h, int d,
                                            int topk, const int* total_ptr, int kpool_val, int n_fixed,
                                            cudaStream_t s) {
    // H-SPLIT (v11): 4 threads per pool (4 heads each) — blockDim 544 = 17 warps
    // (129 pools × 4 quarters = 516 threads + 28 idle). The smem: t scores +
    // 4t partials = 5 × max_t floats.
    dim3 block(544);
    dim3 grid(n);
    // smem sized for MAX possible pools (graph-safe: frozen smem with actual
    // npools would overflow as context grows) + 4 partials per pool (h-split)
    int max_t = 2048; // max_npools = max_tokens / kpool
    size_t smem = (size_t)max_t * 5 * sizeof(float); // scores + 4 partials
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
                                      float* __restrict__ pm,   // [n,h,splits] partial max
                                      float* __restrict__ pl,   // [n,h,splits] partial sum
                                      float* __restrict__ po,   // [n,h,splits,dv] partial O
                                      int n, const int* __restrict__ t_ptr, int h, int d, int dv, int topk) {
#if __CUDA_ARCH__ >= 900
    cudaGridDependencySynchronize(); // PDL v5: launch overlaps predecessor tail
#endif
    int t = *t_ptr; // zero-copy pinned read (graph-safe)
    int row = blockIdx.x;
    int hd = blockIdx.y;
    if (row >= n || hd >= h) return;
    // SPLIT-K (SGLang MLA decode's num_splits): grid.z slices the topk slot
    // range so the block count is heads*splits instead of heads. At TP8 the
    // old grid (n,h) was 8 blocks on a 148-SM part (36KB smem → 2 SM) —
    // latency-bound 75µs. Partials are merged by sparse_attn_merge_kernel.
    const int sp = blockIdx.z;
    const int splits = gridDim.z;
    const int slots = (topk + splits - 1) / splits;
    const int s0 = sp * slots;
    const int s1 = min(s0 + slots, topk);
    float scale = rsqrtf((float)d);
    // Layout: qs FIRST (float4 reads need the 16B-aligned smem base;
    // topk=8195 is NOT a multiple of 4 — an int[topk] prefix misaligned
    // qs by 12B and crashed prefill with err 716). All later arrays are
    // scalar-access, 4B alignment suffices.
    extern __shared__ float sm[];
    float* qs = sm;                                   // [d] float4 reads
    float* sc = sm + d;                                // [topk] scores → weights
    int* idx_s = (int*)(sc + topk);                   // [topk]
    float* red = (float*)(idx_s + topk);               // [16] warp partials
    unsigned int* bm = (unsigned int*)(red + 16);       // [4096] dedup bitmap
    const int bm_words_max = 4096;
    int bm_words = (t + 31) >> 5; if (bm_words > bm_words_max) bm_words = bm_words_max;
    float* red2 = (float*)(bm + bm_words_max); // [SG*dv <= 512] stage-3 split partials
    // 0. preload: q head-slice → smem, idx → smem, clear bitmap
    for (int l = threadIdx.x; l < d; l += blockDim.x) qs[l] = q[((size_t)row * h + hd) * d + l];
    for (int s = threadIdx.x; s < topk; s += blockDim.x) idx_s[s] = (int)idx[(size_t)row * topk + s];
    for (int w0 = threadIdx.x; w0 < bm_words_max; w0 += blockDim.x) bm[w0] = 0u;
    __syncthreads();
    // 1. scores: float4 dot per slot; bitmap dedup (first wins — duplicate
    // slots carry the same key, so which one survives is value-identical)
    for (int s = s0 + threadIdx.x; s < s1; s += blockDim.x) {
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
    for (int s = s0 + threadIdx.x; s < s1; s += blockDim.x) m = fmaxf(m, sc[s]);
    for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_down_sync(0xffffffff, m, off));
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = m;
    __syncthreads();
    if (threadIdx.x < 16) m = red[threadIdx.x];
    for (int off = 8; off > 0; off >>= 1) m = fmaxf(m, __shfl_down_sync(0xffffffff, m, off));
    __shared__ float ms_;
    if (threadIdx.x == 0) ms_ = m;
    __syncthreads();
    m = ms_;
    bool all_inf = (m == -INFINITY);
    float sum = 0.f;
    for (int s = s0 + threadIdx.x; s < s1; s += blockDim.x) {
        sc[s] = all_inf ? 0.f : __expf(sc[s] - m);
        sum += sc[s];
    }
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_down_sync(0xffffffff, sum, off);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = sum;
    __syncthreads();
    if (threadIdx.x < 16) sum = red[threadIdx.x];
    for (int off = 8; off > 0; off >>= 1) sum += __shfl_down_sync(0xffffffff, sum, off);
    __shared__ float sum_;
    if (threadIdx.x == 0) sum_ = sum;
    __syncthreads();
    float denom = sum_ + 1e-9f;
    __syncthreads();
    for (int s = s0 + threadIdx.x; s < s1; s += blockDim.x) sc[s] /= denom;
    __syncthreads();
    // 3. weighted v-gather — split over slots: blockDim = SG x dv lanes, lane
    // (sg, j2) accumulates slots s = sg, sg+SG, ... (w==0 padding skips are
    // order-free), then per-column partials joined via red2. FP-safe class.
    {
        const int SG = (int)(blockDim.x / dv); // 512/128 = 4
        int sg = threadIdx.x / dv, j2 = threadIdx.x - sg * dv;
        if (sg < SG) {
            float a = 0.f;
            for (int s = s0 + sg; s < s1; s += SG) {
                float w = sc[s];
                if (w == 0.f) continue; // padding / deduped slot (exp→0)
                int j = idx_s[s];
                if (j < 0 || j >= t) continue;
                a += w * v[((size_t)j * h + hd) * dv + j2];
            }
            red2[(size_t)sg * dv + j2] = a;
        }
        __syncthreads();
        if (threadIdx.x < dv) {
            float a = 0.f;
            for (int g2 = 0; g2 < SG; g2++) a += red2[(size_t)g2 * dv + threadIdx.x];
            po[(((size_t)row * h + hd) * splits + sp) * dv + threadIdx.x] = a;
        }
        if (threadIdx.x == 0) {
            const size_t base = ((size_t)row * h + hd) * splits + sp;
            pm[base] = m;
            pl[base] = sum_;
        }
    }
}

// split-K merge (flash-decoding): combine the `splits` partial (max, sum, O)
// triples with a log-sum-exp rescale.
__global__ void sparse_attn_merge_kernel(const float* __restrict__ pm,
                                         const float* __restrict__ pl,
                                         const float* __restrict__ po,
                                         float* __restrict__ out,
                                         int n, int h, int dv, int splits) {
    int row = blockIdx.x;
    int hd = blockIdx.y;
    if (row >= n || hd >= h) return;
    const size_t base = ((size_t)row * h + hd) * splits;
    float m = -INFINITY;
    for (int i = 0; i < splits; i++) m = fmaxf(m, pm[base + i]);
    for (int j = threadIdx.x; j < dv; j += blockDim.x) {
        float l = 0.f, o = 0.f;
        for (int i = 0; i < splits; i++) {
            float w = (m == -INFINITY) ? 0.f : __expf(pm[base + i] - m);
            l += w * pl[base + i];
            o += w * po[(base + i) * dv + j];
        }
        out[((size_t)row * h + hd) * dv + j] = o / (l + 1e-9f);
    }
}

extern "C" cudaError_t ferrite_sparse_attn_v2(const float* q, const float* k,
                                              const float* v, const float* idx,
                                              float* out, float* scratch,
                                              int n, const int* t_ptr, int h, int d,
                                              int dv, int topk, int splits, cudaStream_t s) {
    if (splits < 1) splits = 1;
    // scratch layout: pm [n,h,splits] | pl [n,h,splits] | po [n,h,splits,dv]
    float* pm = scratch;
    float* pl = pm + (size_t)n * h * splits;
    float* po = pl + (size_t)n * h * splits;
    dim3 block(512);
    dim3 grid(n, h, splits);
    size_t smem = (size_t)topk * (sizeof(int) + sizeof(float)) + (size_t)d * sizeof(float)
                  + 16 * sizeof(float) + 4096 * sizeof(unsigned int)
                  + 512 * sizeof(float); // red2: SG(4) x dv(128) split partials
    if (smem > 48 * 1024) {
        cudaError_t e = cudaFuncSetAttribute(sparse_attn_v2_kernel,
                                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        if (e != cudaSuccess) return e;
    }
    cudaError_t e = pdl_or_plain(sparse_attn_v2_kernel, grid, block, smem, s,
                        q, k, v, idx, pm, pl, po, n, t_ptr, h, d, dv, topk);
    if (e != cudaSuccess) return e;
    // The merge is a no-op-ish copy at splits==1 (po's layout differs from
    // out's), so run it unconditionally.
    dim3 mgrid(n, h);
    dim3 mblock(128);
    sparse_attn_merge_kernel<<<mgrid, mblock, 0, s>>>(pm, pl, po, out, n, h, dv, splits);
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
        // 4 accumulators: a single fp32 accumulator is a serial FMA chain.
        float p0 = 0.f, p1 = 0.f, p2 = 0.f, p3 = 0.f;
        int i = 0;
        for (; i + 3 < n; i += 4) {
            p0 += red[16 + i]     * x[(size_t)(i)     * h + j];
            p1 += red[16 + i + 1] * x[(size_t)(i + 1) * h + j];
            p2 += red[16 + i + 2] * x[(size_t)(i + 2) * h + j];
            p3 += red[16 + i + 3] * x[(size_t)(i + 3) * h + j];
        }
        for (; i < n; i++) p0 += red[16 + i] * x[(size_t)i * h + j];
        li[(size_t)t * h + j] = (p0 + p1) + (p2 + p3);
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
#if __CUDA_ARCH__ >= 900
    cudaGridDependencySynchronize(); // PDL v5: launch overlaps predecessor tail
#endif
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
    return pdl_or_plain(hc_post_kernel, grid, block, 0, stream,
                        x, res, post, comb, out, s, n, h);
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
                                   int in_f, int out_f, int nrows) {
#if __CUDA_ARCH__ >= 900
    cudaGridDependencySynchronize(); // PDL v5: launch overlaps predecessor tail
#endif
    const int warps = blockDim.x >> 5;
    const int rpb = warps / WPR;               // rows per block
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int rowg = blockIdx.x * rpb + warp / WPR;   // global row in [0, nrows*out_f)
    const int token = rowg / out_f;
    const int row = rowg - token * out_f;
    const int kw = warp % WPR;                 // K-slice id
    float acc = 0.f;
    if (rowg < nrows * out_f) {
        const __nv_bfloat16* wr = w + (size_t)row * in_f;
        const float* xr = x + (size_t)token * in_f;
        // slice size rounded to a multiple of 8 (uint4 16B alignment;
        // in_f % 8 == 0 is guaranteed by the host fallback to v1)
        int kper = ((in_f + WPR - 1) / WPR + 7) & ~7;
        int k0 = kw * kper;
        int k1 = min(k0 + kper, in_f);
        // vector body: uint4 W (8 bf16) + 2x float4 x per lane-step
        #pragma unroll 2
        for (int k = k0 + lane * 8; k + 7 < k1; k += 32 * 8) {
            uint4 wv = *reinterpret_cast<const uint4*>(wr + k);
            float4 xa = *reinterpret_cast<const float4*>(xr + k);
            float4 xb = *reinterpret_cast<const float4*>(xr + k + 4);
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
        if (lane == 0 && rowg < nrows * out_f) y[rowg] = (bias ? bias[row] : 0.f) + acc;
    } else {
        __shared__ float part[16];
        if (lane == 0) part[warp] = acc;
        __syncthreads();
        if (warp % WPR == 0 && lane == 0) {
            float sum = 0.f;
            #pragma unroll
            for (int j = 0; j < WPR; j++) sum += part[(warp / WPR) * WPR + j];
            if (rowg < nrows * out_f) y[rowg] = (bias ? bias[row] : 0.f) + sum;
        }
    }
}

extern "C" cudaError_t ferrite_gemv_bf16_v2(const float* x, const void* w,
                                            const float* bias, float* out,
                                            int in_f, int out_f, int nrows,
                                            cudaStream_t s) {
    if (out_f <= 0 || nrows <= 0) return cudaSuccess;
    if (in_f <= 0) return cudaSuccess;
    if (in_f & 7) return ferrite_gemv_bf16(x, w, bias, out, in_f, out_f, s);
    // WPR heuristic by row count: enough warps to cover HBM latency
    // (out_f*WPR/8 warps total; target >= 64 warps/SM on 148 SMs).
    int wpr = out_f >= 16384 ? 1 : (out_f >= 4096 ? 2 : (out_f >= 1024 ? 4 : 8));
    int rpb = 8 / wpr;                        // 256 threads = 8 warps
    long total = (long)nrows * out_f;
    dim3 grid((total + rpb - 1) / rpb);
    dim3 block(256);
    const __nv_bfloat16* wb = (const __nv_bfloat16*)w;
    // PDL v5: the decode chain's dominant GEMV (fb/gb/o_proj on GDN, all
    // projections on DSA — ~200 nodes/step) launches with programmatic
    // stream serialization under FERRITE_PDL=1 (kernel-entry gridDepSync).
    switch (wpr) {
        case 1: return pdl_or_plain(gemv_bf16_v2_kernel<1>, grid, block, 0, s, x, wb, bias, out, in_f, out_f, nrows);
        case 2: return pdl_or_plain(gemv_bf16_v2_kernel<2>, grid, block, 0, s, x, wb, bias, out, in_f, out_f, nrows);
        case 4: return pdl_or_plain(gemv_bf16_v2_kernel<4>, grid, block, 0, s, x, wb, bias, out, in_f, out_f, nrows);
        default: return pdl_or_plain(gemv_bf16_v2_kernel<8>, grid, block, 0, s, x, wb, bias, out, in_f, out_f, nrows);
    }
}

// ============================================================
// N-token tall-skinny GEMV (v4): each WPR-warp group computes ONE output
// row for ALL n tokens — the weight row slice is read from HBM ONCE and
// dotted against the n activation rows (activations are n*in_f*4B ≈ 64KB
// at n=4/in=4096 — L2/L1-resident after the first blocks). This is the
// TRUE batched-GEMM weight streaming at small n:
//   - v2 batched (n rows in one launch): each (token,row) block reads the
//     SAME weight bytes n times — the L2 *partially* serves the n-th read
//     (measured n=4 batched decode: 33.5ms/step vs n=1's 16.1ms — ~2x
//     effective weight traffic);
//   - the tiled GEMM (ferrite_matmul_bf16): reads weights once but its
//     128-row tile computes the full tile regardless of n (measured n=4:
//     105ms/step — 6.5x the HBM floor);
//   - v4 (this kernel): weights once (the HBM floor), n accumulators per
//     warp, the per-token accumulation order IDENTICAL to v2 (the same
//     K-slice lane order, the same FMA chain, the same WPR shuffle/root
//     reduction) — bit-equal per-token results vs the n=1 GEMV, no greedy
//     flips. n=4 decode: ~16-18ms/step (the n=1 HBM floor + ε compute).
// Grid: (out_f + rpb - 1)/rpb blocks (NO ×n — each block computes one
// row's n outputs). Registers: NT accumulators/lane (NT<=16 — 16 floats).
// ============================================================
template <int NT, int WPR>
__global__ void gemv_bf16_nt_kernel(const float* __restrict__ x,
                                    const __nv_bfloat16* __restrict__ w,
                                    const float* __restrict__ bias,
                                    float* __restrict__ y,
                                    int in_f, int out_f) {
    const int warps = blockDim.x >> 5;
    const int rpb = warps / WPR;               // rows per block (as v2)
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = blockIdx.x * rpb + warp / WPR;   // ONE row per warp-group
    const int kw = warp % WPR;                 // K-slice id (as v2)
    float acc[NT];
    #pragma unroll
    for (int t = 0; t < NT; t++) acc[t] = 0.f;
    if (row < out_f) {
        const __nv_bfloat16* wr = w + (size_t)row * in_f;
        int kper = ((in_f + WPR - 1) / WPR + 7) & ~7;
        int k0 = kw * kper;
        int k1 = min(k0 + kper, in_f);
        // vector body: uint4 W (8 bf16) read ONCE per k-step; the n
        // activation rows' float4 pairs (x[t] is n*in_f ≤ 64KB — L1/L2 hits).
        // Per-token FMA chain = v2's exactly (same k order, same fma pairs).
        #pragma unroll 2
        for (int k = k0 + lane * 8; k + 7 < k1; k += 32 * 8) {
            uint4 wv = *reinterpret_cast<const uint4*>(wr + k);
            const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(&wv);
            float2 f0 = __bfloat1622float2(w2[0]);
            float2 f1 = __bfloat1622float2(w2[1]);
            float2 f2 = __bfloat1622float2(w2[2]);
            float2 f3 = __bfloat1622float2(w2[3]);
            #pragma unroll
            for (int t = 0; t < NT; t++) {
                const float* xr = x + (size_t)t * in_f;
                float4 xa = *reinterpret_cast<const float4*>(xr + k);
                float4 xb = *reinterpret_cast<const float4*>(xr + k + 4);
                acc[t] += xa.x * f0.x + xa.y * f0.y + xa.z * f1.x + xa.w * f1.y;
                acc[t] += xb.x * f2.x + xb.y * f2.y + xb.z * f3.x + xb.w * f3.y;
            }
        }
        // tail (in_f % (32*8*WPR) != 0): scalar per v2 (in_f%8==0 guaranteed
        // by the host; the kper rounding covers the K-slice tail identically)
    }
    // per-token warp shuffle (the same off order as v2 per token)
    #pragma unroll
    for (int t = 0; t < NT; t++) {
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            acc[t] += __shfl_down_sync(0xffffffff, acc[t], off);
        }
    }
    if (WPR == 1) {
        if (lane == 0 && row < out_f) {
            #pragma unroll
            for (int t = 0; t < NT; t++) {
                y[(size_t)t * out_f + row] = (bias ? bias[row] : 0.f) + acc[t];
            }
        }
    } else {
        __shared__ float part[NT][16];
        if (lane == 0) {
            #pragma unroll
            for (int t = 0; t < NT; t++) part[t][warp] = acc[t];
        }
        __syncthreads();
        if (warp % WPR == 0 && lane == 0) {
            #pragma unroll
            for (int t = 0; t < NT; t++) {
                float sum = 0.f;
                #pragma unroll
                for (int j = 0; j < WPR; j++) sum += part[t][(warp / WPR) * WPR + j];
                if (row < out_f) y[(size_t)t * out_f + row] = (bias ? bias[row] : 0.f) + sum;
            }
        }
    }
}

// ============================================================
// BF16 MMA batched decode GEMM: C[16, N] = A[16, K] * B[N, K]^T
// (m16n8k16 tensor core). The batched decode's rows ARE the batch —
// exactly the MMA m16 tile — so the weights stream ONCE and the tensor
// core hides the 16-token arithmetic. The FMA gemv_bf16_nt is
// compute-bound at n=16 (503 MFLOP / ~60 TFLOPS fp32 = 8.4us vs the
// 3.9us HBM floor) — measured 2.5x decay vs n=1; SGLang/cutlass avoid
// it with bf16 MMA and get BS16 = BS1 per-seq throughput.
// Grid: (ceil(N/32), 1); block: 128 threads (4 warps, each a 16x8 tile).
// ============================================================
__global__ void gemm_bf16_mma_kernel(const float* __restrict__ a,        // [16, K] fp32
                                     const __nv_bfloat16* __restrict__ b, // [N, K] bf16
                                     const float* __restrict__ bias,      // [N] or null
                                     float* __restrict__ c,               // [16, N] fp32
                                     int K, int N) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int n0 = blockIdx.x * 32 + warp * 8; // this warp's 8-col tile
    const int group = lane >> 2, tig = lane & 3;
    __shared__ __nv_bfloat16 sa[16][16];
    __shared__ __nv_bfloat16 sb[32][16];       // coalesced B tile (32 rows x k16)
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int k0 = 0; k0 < K; k0 += 16) {
        #pragma unroll
        for (int i = threadIdx.x; i < 256; i += 128) {
            int r = i >> 4, cc = i & 15;
            sa[r][cc] = __float2bfloat16(a[(size_t)r * K + k0 + cc]);
        }
        // B: coalesced 8-byte loads per thread into smem — the direct
        // per-lane fragment load was scattered (4B x 8 rows) and made the
        // MMA kernel SLOWER than the FMA gemv (measured 60ms vs 36ms).
        #pragma unroll
        for (int i = threadIdx.x; i < 512; i += 128) {
            int r = i >> 4, cc = i & 15;
            int nrow = blockIdx.x * 32 + r;
            sb[r][cc] = (nrow < N) ? b[(size_t)nrow * K + k0 + cc] : __float2bfloat16(0.f);
        }
        __syncthreads();
        unsigned a0 = *(const unsigned*)&sa[group][tig * 2];
        unsigned a1 = *(const unsigned*)&sa[group + 8][tig * 2];
        unsigned a2 = *(const unsigned*)&sa[group][tig * 2 + 8];
        unsigned a3 = *(const unsigned*)&sa[group + 8][tig * 2 + 8];
        // B fragment from smem: b0 = W[n0+group][k0+2tig .. +1]
        const int srow = warp * 8 + group;
        unsigned b0 = *(const unsigned*)&sb[srow][tig * 2];
        unsigned b1 = *(const unsigned*)&sb[srow][tig * 2 + 8];
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        __syncthreads();
    }
    const int n = n0 + tig * 2;
    // per-element guards: the old `if (n + 1 < N)` dropped the LAST element
    // of an odd out_f together with the out-of-bounds n+1 (all current shapes
    // are even, this makes odd N safe too)
    if (n < N) {
        const float bi0 = bias ? bias[n] : 0.f;
        c[(size_t)group * N + n] = acc[0] + bi0;
        c[(size_t)(group + 8) * N + n] = acc[2] + bi0;
    }
    if (n + 1 < N) {
        const float bi1 = bias ? bias[n + 1] : 0.f;
        c[(size_t)group * N + n + 1] = acc[1] + bi1;
        c[(size_t)(group + 8) * N + n + 1] = acc[3] + bi1;
    }
}

extern "C" cudaError_t ferrite_gemm_bf16_mma(const float* a, const void* w,
                                             const float* bias, float* out,
                                             int nrows, int in_f, int out_f,
                                             cudaStream_t s) {
    if (nrows != 16 || out_f <= 0 || in_f <= 0) return cudaErrorNotSupported;
    if (in_f & 15) return cudaErrorNotSupported; // k16 MMA step
    dim3 grid((out_f + 31) / 32);
    gemm_bf16_mma_kernel<<<grid, 128, 0, s>>>(a, (const __nv_bfloat16*)w, bias, out, in_f, out_f);
    return cudaGetLastError();
}

extern "C" cudaError_t ferrite_gemv_bf16_nt(const float* x, const void* w,
                                           const float* bias, float* out,
                                           int in_f, int out_f, int nrows,
                                           cudaStream_t s) {
    // DIAGNOSTIC ONLY (FERRITE_GEMV_SKIP=1): timing-only ablation.
    static const bool gemv_skip_ = getenv("FERRITE_GEMV_SKIP") != nullptr;
    if (gemv_skip_) return cudaSuccess;

    if (out_f <= 0 || nrows <= 0 || in_f <= 0) return cudaSuccess;
    if (in_f & 7) return cudaErrorNotSupported; // host falls back to v2 (→v1)
    int wpr = out_f >= 16384 ? 1 : (out_f >= 4096 ? 2 : (out_f >= 1024 ? 4 : 8));
    int rpb = 8 / wpr;
    dim3 grid((out_f + rpb - 1) / rpb);
    const __nv_bfloat16* wb = (const __nv_bfloat16*)w;
    // double dispatch (NT × WPR — the v2 heuristic's per-out_f K-split):
    // macro-instantiated switch (9 NT values × 4 WPR lanes = 36 kernels,
    // same code path per instantiation — no register bloat beyond NT).
#define NT_CASE(NTV, WPRV) gemv_bf16_nt_kernel<NTV, WPRV><<<grid, 256, 0, s>>>(x, wb, bias, out, in_f, out_f)
#define NT_SWITCH_W(NTV) \
    switch (wpr) { \
        case 1:  NT_CASE(NTV, 1);  break; \
        case 2:  NT_CASE(NTV, 2);  break; \
        case 4:  NT_CASE(NTV, 4);  break; \
        default: NT_CASE(NTV, 8);  break; \
    }
    switch (nrows) {
        case 2:  NT_SWITCH_W(2); break;
        case 3:  NT_SWITCH_W(3); break;
        case 4:  NT_SWITCH_W(4); break;
        case 5:  NT_SWITCH_W(5); break;
        case 6:  NT_SWITCH_W(6); break;
        case 7:  NT_SWITCH_W(7); break;
        case 8:  NT_SWITCH_W(8); break;
        case 12: NT_SWITCH_W(12); break;
        case 16: NT_SWITCH_W(16); break;
        default: return cudaErrorNotSupported; // host falls back to v2 batched
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
#if __CUDA_ARCH__ >= 900
    cudaGridDependencySynchronize(); // PDL v5: launch overlaps predecessor tail
#endif
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
    return pdl_or_plain(gemv_tri_kernel, grid, dim3(256), 0, s,
                        x, (const __nv_bfloat16*)w1, (const __nv_bfloat16*)w2,
                        (const __nv_bfloat16*)w3, y1, y2, y3, in_f, o1, o2, o3);
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
        // uint4-vectorized (8 bf16 weights + 2x float4 x per lane-step): the
        // per-lane k-summation order is ascending (same accumulation chain as
        // the scalar loop, 8-wide steps); lane boundary shift only changes the
        // cross-lane partial grouping, folded by the warp shuffle — validated
        // by 出师表 recitation (garbling = revert). hidden%8==0 (4096).
        for (int k = lane * 8; k + 7 < hidden; k += 32 * 8) {
            float4 xa = *reinterpret_cast<const float4*>(xt + k);
            float4 xb = *reinterpret_cast<const float4*>(xt + k + 4);
            uint4 gv = *reinterpret_cast<const uint4*>(gwr + k);
            uint4 uv = *reinterpret_cast<const uint4*>(uwr + k);
            const __nv_bfloat162* g2 = reinterpret_cast<const __nv_bfloat162*>(&gv);
            const __nv_bfloat162* u2 = reinterpret_cast<const __nv_bfloat162*>(&uv);
            float2 gf0 = __bfloat1622float2(g2[0]), gf1 = __bfloat1622float2(g2[1]);
            float2 gf2 = __bfloat1622float2(g2[2]), gf3 = __bfloat1622float2(g2[3]);
            float2 uf0 = __bfloat1622float2(u2[0]), uf1 = __bfloat1622float2(u2[1]);
            float2 uf2 = __bfloat1622float2(u2[2]), uf3 = __bfloat1622float2(u2[3]);
            g += xa.x * gf0.x + xa.y * gf0.y + xa.z * gf1.x + xa.w * gf1.y
               + xb.x * gf2.x + xb.y * gf2.y + xb.z * gf3.x + xb.w * gf3.y;
            u += xa.x * uf0.x + xa.y * uf0.y + xa.z * uf1.x + xa.w * uf1.y
               + xb.x * uf2.x + xb.y * uf2.y + xb.z * uf3.x + xb.w * uf3.y;
        }
        for (int k = lane * 8 + ((hidden >> 3) << 3); k < hidden; k += 32) {
            // tail (hidden % 8 != 0 — never on GLM but kept safe)
            g += xt[k] * __bfloat162float(gwr[k]);
            u += xt[k] * __bfloat162float(uwr[k]);
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
    // EXPERT-PARALLEL (v11): grid (hidden, n) — 1 hidden dim per block, 8 warps
    // = 8 experts (warp j → expert j). The old version (grid (hidden/rows, n),
    // 1 warp per hidden dim, 8-expert serial loop) was 18.9µs/layer — the
    // expert loop is the latency chain. Now: 8 warps compute the 8 experts'
    // dot products in PARALLEL, warp 0 sums the partials in j=0..7 order
    // (SAME summation order as the old serial acc += p*y — FP-SAFE).
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int tok = blockIdx.y;
    int h = blockIdx.x; // 1 hidden per block (grid (hidden, n))
    if (h >= hidden) return;
    int stride = topk * inter + inter_shared;
    const float* act_t = act + (size_t)tok * stride;
    const float* ids_t = ids_f + (size_t)tok * topk;
    const float* probs_t = probs + (size_t)tok * topk;
    __shared__ float part[32]; // per-warp partials (topk ≤ 8, 32 max warps)
    // Each warp j computes expert j's p*y dot product (parallel)
    {
        int j = warp; // warp ID = expert slot
        float py = 0.f;
        if (j < topk) {
            int eid = (int)ids_t[j];
            int local = eid - expert_start;
            if (local >= 0 && local < e_local) {
                float p = probs_t[j];
                if (p != 0.f) {
                    const __nv_bfloat16* dwr = down_ptrs[local] + (size_t)h * inter;
                    const float* aj = act_t + (size_t)j * inter;
                    float y = 0.f;
                    int i = lane * 8;
                    for (; i + 7 < inter; i += 32 * 8) {
                        float4 aa = *reinterpret_cast<const float4*>(aj + i);
                        float4 ab = *reinterpret_cast<const float4*>(aj + i + 4);
                        uint4 dv = *reinterpret_cast<const uint4*>(dwr + i);
                        const __nv_bfloat162* d2 = reinterpret_cast<const __nv_bfloat162*>(&dv);
                        float2 df0 = __bfloat1622float2(d2[0]), df1 = __bfloat1622float2(d2[1]);
                        float2 df2 = __bfloat1622float2(d2[2]), df3 = __bfloat1622float2(d2[3]);
                        y += aa.x * df0.x + aa.y * df0.y + aa.z * df1.x + aa.w * df1.y
                           + ab.x * df2.x + ab.y * df2.y + ab.z * df3.x + ab.w * df3.y;
                    }
                    for (; i < inter; i++) {
                        y += aj[i] * __bfloat162float(dwr[i]);
                    }
                    #pragma unroll
                    for (int off = 16; off > 0; off >>= 1) {
                        y += __shfl_down_sync(0xffffffff, y, off);
                    }
                    if (lane == 0) py = p * y;
                }
            }
        }
        if (lane == 0) part[warp] = py;
    }
    // Warp 0 ALSO computes the shared expert (slot topk) — 2 dot products
    // on the critical path (still 4.5× faster than the old 9-serial)
    float shared_y = 0.f;
    if (warp == 0) {
        const __nv_bfloat16* dwr = shared_down + (size_t)h * inter_shared;
        const float* as = act_t + (size_t)topk * inter;
        float y = 0.f;
        int i = lane * 8;
        for (; i + 7 < inter_shared; i += 32 * 8) {
            float4 aa = *reinterpret_cast<const float4*>(as + i);
            float4 ab = *reinterpret_cast<const float4*>(as + i + 4);
            uint4 dv = *reinterpret_cast<const uint4*>(dwr + i);
            const __nv_bfloat162* d2 = reinterpret_cast<const __nv_bfloat162*>(&dv);
            float2 df0 = __bfloat1622float2(d2[0]), df1 = __bfloat1622float2(d2[1]);
            float2 df2 = __bfloat1622float2(d2[2]), df3 = __bfloat1622float2(d2[3]);
            y += aa.x * df0.x + aa.y * df0.y + aa.z * df1.x + aa.w * df1.y
               + ab.x * df2.x + ab.y * df2.y + ab.z * df3.x + ab.w * df3.y;
        }
        for (; i < inter_shared; i++) {
            y += as[i] * __bfloat162float(dwr[i]);
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            y += __shfl_down_sync(0xffffffff, y, off);
        }
        if (lane == 0) shared_y = y;
    }
    __syncthreads();
    // Warp 0 lane 0: sum the 8 partials in j=0..7 order (SAME as the old
    // serial acc += p*y for j=0..7 — FP-SAFE) + the shared expert
    if (warp == 0 && lane == 0) {
        float acc = 0.f;
        for (int j = 0; j < topk; j++) {
            acc += part[j]; // j-ascending = the old serial order
        }
        acc += shared_y;
        out[(size_t)tok * hidden + h] = acc;
    }
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
    // v11 tried rows=2/blockDim=64 (2304 blocks): 62.0 vs 62.3 — NO gain (act
    // is memory-bandwidth-bound at ~3.2TB/s effective, more blocks ≠ more MLP).
    // TP-occupancy: grid.x = inter/rows. At TP8 inter/rank is 1/8, so rows=8
    // gave 216 blocks on 148 SM (18% occupancy, latency-bound 15.7µs). rows=4
    // doubles the block count; the kernel's inner loop is column-parallel.
    int rows = 4;
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
    // EXPERT-PARALLEL (v11): grid (hidden, n) — 1 hidden dim per block, 8 warps
    // = 8 experts (warp j → expert j). The old grid (hidden/8, n) had 1 warp
    // per hidden dim with the 8-expert serial loop (the latency chain: 9 dot
    // products × 512 dims serial per warp = 18.9µs/layer). Now: 8 warps
    // compute the 8 experts in PARALLEL, warp 0 also handles the shared
    // expert + sums partials in j=0..7 order (FP-SAFE — same summation
    // order as the old serial acc += p*y).
    dim3 grid(hidden, n);
    moe_fused_down_sum_kernel<<<grid, 256, 0, s>>>(
        ids_f, probs,
        (const __nv_bfloat16* const*)down_ptrs, (const __nv_bfloat16*)shared_down,
        act, out, expert_start, e_local, hidden, inter, inter_shared, topk, 8);
    return cudaGetLastError();
}

// ============================================================
// MoE fp8 variants: the experts' weights serve from the checkpoint-native
// F8_E4M3 bytes + 128x128 block scales (the fp8_map's Fp8Dev pairs) —
// HALF the HBM traffic of the bf16 pointer tables (moe was 4.5ms of the
// 21.7ms verify step, HBM-bound). Same grid/tile structure as the bf16
// kernels; the dot bodies swap bf16-uint4 loads for 16x-fp8-uint4 with the
// inline block-scale dequant (w_f32 = e4m3(b) * s[r>>7][k>>7], the
// checkpoint's own dequant_block semantics — NOT a re-quantization).
// scale layouts: act weights [inter, hidden] -> srow = s + (r>>7)*hscols
// (hscols = hidden/128); down weights [hidden, inter] -> srow = s + (h>>7)*dscols
// (dscols = inter/128).
// ============================================================
__global__ void moe_fused_act_fp8_kernel(
    const float* __restrict__ x,          // [n, hidden]
    const float* __restrict__ ids_f,      // [n, topk] f32-encoded
    const unsigned char* const* __restrict__ gate_w8_ptrs,   // [e_local] fp8 [inter, hidden]
    const float* const* __restrict__ gate_scale_ptrs,       // [e_local] [inter/128, hidden/128]
    const unsigned char* const* __restrict__ up_w8_ptrs,    // [e_local] fp8 [inter, hidden]
    const float* const* __restrict__ up_scale_ptrs,
    const unsigned char* __restrict__ shared_gate_w8,      // [inter_shared, hidden]
    const float* __restrict__ shared_gate_scale,
    const unsigned char* __restrict__ shared_up_w8,
    const float* __restrict__ shared_up_scale,
    float* __restrict__ act,              // [n, topk*inter + inter_shared]
    int expert_start, int e_local, int hidden, int inter,
    int inter_shared, int topk, int rows, float limit, int hscols) {
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int slot = blockIdx.y;
    int tok = blockIdx.z;
    int row0 = blockIdx.x * rows;
    int stride = topk * inter + inter_shared;
    const float* xt = x + (size_t)tok * hidden;
    int slot_rows, slot_base;
    const unsigned char *gw8, *uw8;
    const float *gs, *us; // block scales
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
        gw8 = gate_w8_ptrs[local]; gs = gate_scale_ptrs[local];
        uw8 = up_w8_ptrs[local];  us = up_scale_ptrs[local];
    } else {
        slot_rows = inter_shared;
        slot_base = topk * inter;
        gw8 = shared_gate_w8; gs = shared_gate_scale;
        uw8 = shared_up_w8;   us = shared_up_scale;
    }
    int warps = blockDim.x >> 5;
    for (int r = row0 + warp; r < row0 + rows && r < slot_rows; r += warps) {
        const unsigned char* gwr = gw8 + (size_t)r * hidden;
        const unsigned char* uwr = uw8 + (size_t)r * hidden;
        const float* gsr = gs + (size_t)(r >> 7) * hscols; // block-scale row
        const float* usr = us + (size_t)(r >> 7) * hscols;
        float g = 0.f, u = 0.f;
        // uint4 = 16 fp8 weights per lane-step; the 16-col group never
        // crosses a 128-col scale block (16 | 128), one s fetch per group.
        for (int k = lane * 16; k + 15 < hidden; k += 32 * 16) {
            const float4 xa = *reinterpret_cast<const float4*>(xt + k);
            const float4 xb = *reinterpret_cast<const float4*>(xt + k + 4);
            const float4 xc = *reinterpret_cast<const float4*>(xt + k + 8);
            const float4 xd = *reinterpret_cast<const float4*>(xt + k + 12);
            uint4 gv = *reinterpret_cast<const uint4*>(gwr + k);
            uint4 uv = *reinterpret_cast<const uint4*>(uwr + k);
            const unsigned char* g8 = reinterpret_cast<const unsigned char*>(&gv);
            const unsigned char* u8 = reinterpret_cast<const unsigned char*>(&uv);
            const float gs_c = gsr[k >> 7];
            const float us_c = usr[k >> 7];
            // fp8x2 batch convert (__nv_cvt_fp8x2_to_halfraw2): 2 e4m3 -> 1
            // half2 per op — HALVES the convert instruction count vs the
            // scalar __nv_cvt_fp8_to_halfraw (the single-warp gemv was
            // convert-bound, offsetting the fp8 HBM savings: 0.94x vs bf16).
            const float xv[16] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w,
                                  xc.x, xc.y, xc.z, xc.w, xd.x, xd.y, xd.z, xd.w};
            #pragma unroll
            for (int p = 0; p < 8; p++) {
                const __nv_fp8x2_storage_t gx2 = *reinterpret_cast<const __nv_fp8x2_storage_t*>(&g8[p * 2]);
                const __nv_fp8x2_storage_t ux2 = *reinterpret_cast<const __nv_fp8x2_storage_t*>(&u8[p * 2]);
                const float2 gf = __half22float2(*reinterpret_cast<const __half2*>(&__nv_cvt_fp8x2_to_halfraw2(gx2, __NV_E4M3)));
                const float2 uf = __half22float2(*reinterpret_cast<const __half2*>(&__nv_cvt_fp8x2_to_halfraw2(ux2, __NV_E4M3)));
                g += (gf.x * gs_c) * xv[p * 2] + (gf.y * gs_c) * xv[p * 2 + 1];
                u += (uf.x * us_c) * xv[p * 2] + (uf.y * us_c) * xv[p * 2 + 1];
            }
        }
        for (int k = ((hidden >> 4) << 4) + lane; k < hidden; k += 32) {
            // tail (hidden % 16 != 0 — never on GLM but kept safe)
            const float gs_c = gsr[k >> 7];
            const float us_c = usr[k >> 7];
            g += (__half2float(__nv_cvt_fp8_to_halfraw(gwr[k], __NV_E4M3)) * gs_c) * xt[k];
            u += (__half2float(__nv_cvt_fp8_to_halfraw(uwr[k], __NV_E4M3)) * us_c) * xt[k];
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

extern "C" cudaError_t ferrite_moe_fused_act_fp8(
    const float* x, const float* ids_f,
    const void* const* gate_w8_ptrs, const void* const* gate_scale_ptrs,
    const void* const* up_w8_ptrs, const void* const* up_scale_ptrs,
    const void* shared_gate_w8, const void* shared_gate_scale,
    const void* shared_up_w8, const void* shared_up_scale,
    float* act, int expert_start, int e_local, int hidden, int inter,
    int inter_shared, int topk, int n, float limit, int hscols,
    cudaStream_t s) {
    // TP-occupancy: grid.x = max_rows/rows. At TP8 inter/rank is 1/8, so
    // rows=8 left only 216 blocks on 148 SM (18% occupancy, latency-bound
    // 15.7µs x 40.6/step = 0.64ms). rows=4 doubles the block count.
    int rows = 4;
    int max_rows = inter > inter_shared ? inter : inter_shared;
    dim3 grid((max_rows + rows - 1) / rows, topk + 1, n);
    moe_fused_act_fp8_kernel<<<grid, 256, 0, s>>>(
        x, ids_f,
        (const unsigned char* const*)gate_w8_ptrs, (const float* const*)gate_scale_ptrs,
        (const unsigned char* const*)up_w8_ptrs, (const float* const*)up_scale_ptrs,
        (const unsigned char*)shared_gate_w8, (const float*)shared_gate_scale,
        (const unsigned char*)shared_up_w8, (const float*)shared_up_scale,
        act, expert_start, e_local, hidden, inter, inter_shared, topk, rows, limit, hscols);
    return cudaGetLastError();
}

// v0 (n=1 PATH): warp-serial expert loop — restored from 23acab6 as the n=1
// dispatch target. The v11/v12.1 expert-parallel + register-cache variants all
// regressed n=1 (90.9 → 87.2: 48 regs/lane occupancy drop + the 9-warp 288-
// thread block wastes 6 warps' slots at n=1's ~2 routed experts per token —
// only ~3 of 9 warps have work). v0 keeps 8 h-rows per block, warp = one h
// row, serial j (the topk experts of ONE token — j loop is short at n=1).
__global__ void moe_fused_down_sum_fp8_v0_kernel(
    const float* __restrict__ ids_f,       // [n, topk]
    const float* __restrict__ probs,       // [n, topk]
    const unsigned char* const* __restrict__ down_w8_ptrs,  // [e_local] fp8 [hidden, inter]
    const float* const* __restrict__ down_scale_ptrs,       // [e_local] [hidden/128, inter/128]
    const unsigned char* __restrict__ shared_down_w8,      // [hidden, inter_shared]
    const float* __restrict__ shared_down_scale,
    const float* __restrict__ act,         // [n, topk*inter + inter_shared]
    float* __restrict__ out,               // [n, hidden]
    int expert_start, int e_local, int hidden, int inter,
    int inter_shared, int topk, int dscols) {
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int tok = blockIdx.y;
    int h = blockIdx.x * 8 + warp;
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
        const unsigned char* dwr = down_w8_ptrs[local] + (size_t)h * inter;
        const float* dsr = down_scale_ptrs[local] + (size_t)(h >> 7) * dscols;
        const float* aj = act_t + (size_t)j * inter;
        float y = 0.f;
        int i = lane * 16;
        for (; i + 15 < inter; i += 32 * 16) {
            uint4 dv = *reinterpret_cast<const uint4*>(dwr + i);
            const float4 aa = *reinterpret_cast<const float4*>(aj + i);
            const float4 ab = *reinterpret_cast<const float4*>(aj + i + 4);
            const float4 ac = *reinterpret_cast<const float4*>(aj + i + 8);
            const float4 ad = *reinterpret_cast<const float4*>(aj + i + 12);
            const unsigned char* d8 = reinterpret_cast<const unsigned char*>(&dv);
            const float ds_c = dsr[i >> 7];
            const float xv[16] = {aa.x, aa.y, aa.z, aa.w, ab.x, ab.y, ab.z, ab.w,
                                  ac.x, ac.y, ac.z, ac.w, ad.x, ad.y, ad.z, ad.w};
#pragma unroll
            for (int p = 0; p < 8; p++) {
                const __nv_fp8x2_storage_t dx2 = *reinterpret_cast<const __nv_fp8x2_storage_t*>(&d8[p * 2]);
                const float2 df = __half22float2(*reinterpret_cast<const __half2*>(&__nv_cvt_fp8x2_to_halfraw2(dx2, __NV_E4M3)));
                y += (df.x * ds_c) * xv[p * 2] + (df.y * ds_c) * xv[p * 2 + 1];
            }
        }
        for (; i < inter; i++) {
            y += (__half2float(__nv_cvt_fp8_to_halfraw(dwr[i], __NV_E4M3)) * dsr[i >> 7]) * aj[i];
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            y += __shfl_down_sync(0xffffffff, y, off);
        }
        acc += p * y;
    }
    // shared expert (slot topk, weight 1; K length = inter_shared, TP-sharded)
    {
        const unsigned char* dwr = shared_down_w8 + (size_t)h * inter_shared;
        const float* dsr = shared_down_scale + (size_t)(h >> 7) * dscols;
        const float* as_ = act_t + (size_t)topk * inter;
        float y = 0.f;
        int i = lane * 16;
        for (; i + 15 < inter_shared; i += 32 * 16) {
            uint4 dv = *reinterpret_cast<const uint4*>(dwr + i);
            const float4 aa = *reinterpret_cast<const float4*>(as_ + i);
            const float4 ab = *reinterpret_cast<const float4*>(as_ + i + 4);
            const float4 ac = *reinterpret_cast<const float4*>(as_ + i + 8);
            const float4 ad = *reinterpret_cast<const float4*>(as_ + i + 12);
            const unsigned char* d8 = reinterpret_cast<const unsigned char*>(&dv);
            const float ds_c = dsr[i >> 7];
            const float xv[16] = {aa.x, aa.y, aa.z, aa.w, ab.x, ab.y, ab.z, ab.w,
                                  ac.x, ac.y, ac.z, ac.w, ad.x, ad.y, ad.z, ad.w};
#pragma unroll
            for (int p = 0; p < 8; p++) {
                const __nv_fp8x2_storage_t dx2 = *reinterpret_cast<const __nv_fp8x2_storage_t*>(&d8[p * 2]);
                const float2 df = __half22float2(*reinterpret_cast<const __half2*>(&__nv_cvt_fp8x2_to_halfraw2(dx2, __NV_E4M3)));
                y += (df.x * ds_c) * xv[p * 2] + (df.y * ds_c) * xv[p * 2 + 1];
            }
        }
        for (; i < inter_shared; i++) {
            y += (__half2float(__nv_cvt_fp8_to_halfraw(dwr[i], __NV_E4M3)) * dsr[i >> 7]) * as_[i];
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            y += __shfl_down_sync(0xffffffff, y, off);
        }
        acc += y;
    }
    if (lane == 0) out[(size_t)tok * hidden + h] = acc;
}

__global__ void moe_fused_down_sum_fp8_kernel(
    const float* __restrict__ ids_f,       // [n, topk]
    const float* __restrict__ probs,       // [n, topk]
    const unsigned char* const* __restrict__ down_w8_ptrs,  // [e_local] fp8 [hidden, inter]
    const float* const* __restrict__ down_scale_ptrs,       // [e_local] [hidden/128, inter/128]
    const unsigned char* __restrict__ shared_down_w8,      // [hidden, inter_shared]
    const float* __restrict__ shared_down_scale,
    const float* __restrict__ act,         // [n, topk*inter + inter_shared]
    float* __restrict__ out,               // [n, hidden]
    int expert_start, int e_local, int hidden, int inter,
    int inter_shared, int topk, int dscols, int nt) {
    // v12: grid (hidden/8, nt) — warp j owns expert slot j and loops EIGHT h
    // rows. The act row (tok, slot j — 6KB) is invariant across h: the first
    // read pulls it into L1 and the 7 re-reads hit L1 (the v11 grid
    // (hidden, n) re-read each act row from L2 once per h block = 662MB of
    // L2 traffic at the ~15TB/s L2 peak — THAT was the 44.6µs bottleneck,
    // not HBM (down weights are only ~21MB/rank). The down weight rows for
    // 8 consecutive h are one contiguous 12KB run per expert (better HBM
    // coalescing too). FP: per-h fold stays j-ascending (warp 0..7 each
    // reduce one h's part[j] column serially — the SAME summation order as
    // the old warp-serial acc += p*y chain, bit-identical partials from the
    // same lane dot + shuffle tree).
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int h0 = blockIdx.x * 8;
    if (h0 >= hidden) return;
    int stride = topk * inter + inter_shared;
    // ALL tokens per block (grid.y == 1). One block read only 18KB and spent
    // ~2us in fixed per-block latency -> 5.6GB/s/SM, while the act kernel
    // (128KB/block) reaches 22GB/s/SM. Processing MAXN tokens per block
    // amortizes that latency MAXN-fold; `part` holds the per-token partials.
    const int MAXN = 64;
    const int TT = 4; // tokens per block (middle ground: 512 blocks gave too
                      // little parallelism per SM, 8192 paid the fixed
                      // per-block latency 16x)
    __shared__ float part[MAXN][8][16]; // [tok][h row][slot]
    int j = warp;
    const int t0 = blockIdx.y * TT;
    const int t1 = (t0 + TT < nt) ? (t0 + TT) : nt;
    for (int base = t0; base < t1; base += MAXN) {
        const int cnt = ((t1 - base) < MAXN) ? (t1 - base) : MAXN;
        if (j <= topk) {
        float py[8];
        for (int tt = 0; tt < cnt; tt++) {
        const int tok = base + tt;
        const float* act_t = act + (size_t)tok * stride;
        const float* ids_t = ids_f + (size_t)tok * topk;
        const float* probs_t = probs + (size_t)tok * topk;
        #pragma unroll
        for (int hh = 0; hh < 8; hh++) py[hh] = 0.f;
        const float* aj = nullptr;
        const unsigned char* dbase = nullptr;
        const float* dsr_base = nullptr;
        int klen = 0;
        float p = 1.f;
        if (j < topk) {
            int eid = (int)ids_t[j];
            int local = eid - expert_start;
            if (local >= 0 && local < e_local) {
                p = probs_t[j];
                if (p != 0.f) {
                    dbase = down_w8_ptrs[local];
                    dsr_base = down_scale_ptrs[local];
                    aj = act_t + (size_t)j * inter;
                    klen = inter;
                }
            }
        } else {
            dbase = shared_down_w8;
            dsr_base = shared_down_scale;
            aj = act_t + (size_t)topk * inter;
            klen = inter_shared;
        }
        if (klen > 0) {
            // 16-BYTE LANES (uint4): one load covers 512B = TWO h-rows (rows
            // are contiguous, klen=256). The kernel was request-rate bound at
            // 1.16TB/s (~15% of peak) with 8-byte lanes; halving the request
            // count doubles the bytes per in-flight request.
            float4 ar[4];
            {
                const float4* a4 = reinterpret_cast<const float4*>(aj);
                const int base = (lane & 15) * 4; // act[16] per lane, same for both rows
                #pragma unroll
                for (int r = 0; r < 4; r++) ar[r] = a4[base + r];
            }
            if (klen == 256) {
                const int i0 = lane * 16;
                const int scol = (lane & 15) >> 3; // scale column: lanes 16-31 read the SECOND row's bytes [0..256)
                uint4 dv4[4];
                #pragma unroll
                for (int c = 0; c < 4; c++)
                    dv4[c] = *reinterpret_cast<const uint4*>(dbase + (size_t)(h0 + 2 * c) * klen + i0);
                #pragma unroll
                for (int c = 0; c < 4; c++) {
                    const unsigned char* d8 = reinterpret_cast<const unsigned char*>(&dv4[c]);
                    const float ds_c = dsr_base[(size_t)((h0 + 2 * c) >> 7) * dscols + scol];
                    const float* arf = reinterpret_cast<const float*>(ar);
                    // 4 accumulators (was 2): each chain was 4 deep and the
                    // fp32 FMA latency is 4 cycles, so the tail of every
                    // iteration stalled. Four independent chains hide it.
                    float ya0 = 0.f, ya1 = 0.f, yb0 = 0.f, yb1 = 0.f;
                    #pragma unroll
                    for (int q = 0; q < 8; q += 2) {
                        const __nv_fp8x2_storage_t dx2 = *reinterpret_cast<const __nv_fp8x2_storage_t*>(d8 + q * 2);
                        const float2 df = __half22float2(*reinterpret_cast<const __half2*>(&__nv_cvt_fp8x2_to_halfraw2(dx2, __NV_E4M3)));
                        ya0 += (df.x * ds_c) * arf[q * 2];
                        ya1 += (df.y * ds_c) * arf[q * 2 + 1];
                        const __nv_fp8x2_storage_t dx2b = *reinterpret_cast<const __nv_fp8x2_storage_t*>(d8 + (q + 1) * 2);
                        const float2 dfb = __half22float2(*reinterpret_cast<const __half2*>(&__nv_cvt_fp8x2_to_halfraw2(dx2b, __NV_E4M3)));
                        yb0 += (dfb.x * ds_c) * arf[(q + 1) * 2];
                        yb1 += (dfb.y * ds_c) * arf[(q + 1) * 2 + 1];
                    }
                    float y = (ya0 + ya1) + (yb0 + yb1);
                    // lanes 0..15 -> row h0+2c, lanes 16..31 -> row h0+2c+1
                    #pragma unroll
                    for (int off = 8; off > 0; off >>= 1) y += __shfl_down_sync(0xffffffff, y, off);
                    const float y1 = __shfl_sync(0xffffffff, y, 16);
                    if (lane == 0) { py[2 * c] = y; py[2 * c + 1] = y1; }
                }
            } else {
                #pragma unroll
                for (int hh = 0; hh < 8; hh++) {
                    int h = h0 + hh;
                    float y = 0.f;
                    for (int k = lane; k < klen; k += 32)
                        y += (__half2float(__nv_cvt_fp8_to_halfraw(dbase[(size_t)h * klen + k], __NV_E4M3))
                              * dsr_base[(size_t)(h >> 7) * dscols + (k >> 7)]) * aj[k];
                    #pragma unroll
                    for (int off = 16; off > 0; off >>= 1) y += __shfl_down_sync(0xffffffff, y, off);
                    py[hh] = y;
                }
            }
            }
        if (lane == 0) {
            #pragma unroll
            for (int hh = 0; hh < 8; hh++) part[tt][hh][j] = p * py[hh];
        }
        }
        }
        __syncthreads();
        // fold: cnt*8 (tok, h row) pairs, j-ascending (FP-safe)
        for (int idx = threadIdx.x; idx < cnt * 8; idx += blockDim.x) {
            int tt = idx >> 3, hh = idx & 7;
            int h = h0 + hh;
            if (h < hidden) {
                float acc = 0.f;
                for (int jj = 0; jj <= topk; jj++) acc += part[tt][hh][jj];
                out[(size_t)(base + tt) * hidden + h] = acc;
            }
        }
        __syncthreads();
    }
}

// ============================================================
// moe_fused_down_mma (W8A8, tensor core): the down projection of the fused
// MoE on the fp8 MMA. Block = (16 hidden rows, one token); the 8 warps split
// the K (inter/8, 32-aligned). A = the expert's down weight tile via
// ldmatrix.x4 (fp8, per-128 scale from down_scale_ptrs); B = the token's act
// row, quantized to e4m3 in the staging with a per-32-K-tile absmax (the act
// is fp32 in global). The 8 k-tiles accumulate into 8 separate fp32
// accumulators so each tile's (wscale * ascale) folds in at the end. The MMA
// computes a 16x8 tile; the 8 N columns are replicas of the same token (only
// column 0 is used), which still beats the SIMT fp8 FMA path by ~4x because
// the tensor-core MAC rate is ~32x a lane's.
// ============================================================
template <int UNUSED_MOE_DOWN>
__global__ void __launch_bounds__(256, 4) moe_down_mma_kernel(
    const float* __restrict__ ids_f,       // [n, topk]
    const float* __restrict__ probs,       // [n, topk]
    const unsigned char* const* __restrict__ down_w8_ptrs,  // [e_local] fp8 [hidden, inter]
    const float* const* __restrict__ down_scale_ptrs,       // [e_local] [hidden/128, inter/128]
    const unsigned char* __restrict__ shared_down_w8,      // [hidden, inter_shared]
    const float* __restrict__ shared_down_scale,
    const float* __restrict__ act,         // [n, topk*inter + inter_shared]
    float* __restrict__ out,               // [n, hidden]
    int expert_start, int e_local, int hidden, int inter,
    int inter_shared, int topk, int dscols, int n) {
    const int h0 = blockIdx.x * 16;
    const int tok = blockIdx.y;
    if (h0 >= hidden || tok >= n) return;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int c0 = (lane & 3) * 4;
    const int kper = ((inter + 7) / 8 + 31) & ~31;
    const int k0 = warp * kper;
    const int k1 = min(k0 + kper, inter);
    const int stride = topk * inter + inter_shared;
    __shared__ unsigned char sw[2][8][16][48];   // [stage][warp][h row][K=32+16]
    __shared__ unsigned char sa[2][8][48];       // [stage][warp][K] quantized act
    __shared__ float asc[8];                     // per-warp act scale (per k-tile)
    float acc[8][4];
    #pragma unroll
    for (int i = 0; i < 8; i++) { acc[i][0] = 0.f; acc[i][1] = 0.f; acc[i][2] = 0.f; acc[i][3] = 0.f; }
    float total = 0.f, total2 = 0.f;
    const int nslot = topk + (inter_shared > 0 ? 1 : 0);
    for (int slot = 0; slot < nslot; slot++) {
        int eid = (int)ids_f[(size_t)tok * topk + (slot < topk ? slot : 0)];
        const unsigned char* w8; const float* ws; const float* arow; float p;
        if (slot < topk) {
            int local = eid - expert_start;
            if (local < 0 || local >= e_local) continue;
            w8 = down_w8_ptrs[local]; ws = down_scale_ptrs[local];
            arow = act + (size_t)tok * stride + (size_t)slot * inter;
            p = probs[(size_t)tok * topk + slot];
        } else {
            if (inter_shared <= 0) break;
            w8 = shared_down_w8; ws = shared_down_scale;
            arow = act + (size_t)tok * stride + (size_t)topk * inter;
            p = 1.0f;
        }
        // quantize this warp's K slice of the act row once (32 values)
        {
            const int l0 = k0 + lane;
            float a0 = (l0 < inter) ? arow[l0] : 0.f;
            float am = fabsf(a0);
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) am = fmaxf(am, __shfl_down_sync(0xffffffff, am, off));
            am = __shfl_sync(0xffffffff, am, 0);
            const float ascale = am / 448.0f + 1e-12f;
            if (lane == 0) asc[warp] = ascale;
            // every lane needs the 32 quantized values in smem (B fragment is
            // shared across the N=8 replicas) -> 8 lanes write 4 each
            #pragma unroll
            for (int q = 0; q < 4; q++) {
                const int idx = q * 8 + lane;
                if (idx < 32) {
                    const int l = k0 + idx;
                    const float v = (l < inter) ? arow[l] : 0.f;
                    const float qv = fminf(fmaxf(v / ascale, -448.0f), 448.0f);
                    sa[0][warp][idx] = (unsigned char)__nv_cvt_float_to_fp8(qv, __NV_SATFINITE, __NV_E4M3);
                }
            }
        }
        __syncwarp();
        // load the weight tile (16 h rows x 32 K) into smem
        {
            const int t = lane;
            #pragma unroll
            for (int q = 0; q < 4; q++) {
                const int idx = t + q * 32;         // 0..127: row = idx/8, col8 = idx%8
                const int row = idx >> 3, c8 = (idx & 7) * 4;
                const int kk = k0 + c8;
                unsigned char* dst = sw[0][warp][row] + c8;
                if (kk + 3 < inter && h0 + row < hidden) {
                    const unsigned int* src = (const unsigned int*)(w8 + (size_t)(h0 + row) * inter + kk);
                    *(unsigned int*)dst = *src;
                } else {
                    unsigned char tmp[4] = {0, 0, 0, 0};
                    for (int e = 0; e < 4; e++) {
                        const int l = kk + e;
                        tmp[e] = (l < inter && h0 + row < hidden) ? w8[(size_t)(h0 + row) * inter + l] : 0;
                    }
                    *(unsigned int*)dst = *(unsigned int*)tmp;
                }
            }
        }
        __syncwarp();
        const float ascale = asc[warp];
        const int kb = k0 >> 7;
        const float wsc = ws[(size_t)(h0 >> 7) * dscols + kb];
        // one 16x8x32 MMA (K slice = 32 for this warp)
        {
            const unsigned saddr_a = (unsigned)__cvta_generic_to_shared(
                sw[0][warp][0] + (size_t)(lane & 15) * 48 + ((lane >> 4) * 16));
            unsigned ba[4];
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                         : "=r"(ba[0]), "=r"(ba[1]), "=r"(ba[2]), "=r"(ba[3]) : "r"(saddr_a));
            unsigned b[2];
            b[0] = *(const unsigned*)(sa[0][warp] + c0);
            b[1] = *(const unsigned*)(sa[0][warp] + c0 + 16);
            const int ki = (k0 >> 5) & 7;
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                : "=f"(acc[ki][0]), "=f"(acc[ki][1]), "=f"(acc[ki][2]), "=f"(acc[ki][3])
                : "r"(ba[0]), "r"(ba[1]), "r"(ba[2]), "r"(ba[3]), "r"(b[0]), "r"(b[1]),
                  "f"(0.f), "f"(0.f), "f"(0.f), "f"(0.f));
        }
        // fold this k-tile. C fragment (m16n8): lane l holds (row l/4, col
        // (l%4)*2), (row l/4, col+1), (row l/4+8, col), (row l/4+8, col+1).
        // The 8 N columns are replicas of the same token, so col 0 carries
        // the answer; the two M rows (l/4 and l/4+8) are distinct h rows.
        {
            const int ki = (k0 >> 5) & 7;
            const float f = wsc * ascale * p;
            if ((lane & 3) == 0) {
                total += acc[ki][0] * f;
                total2 += acc[ki][2] * f;
                acc[ki][0] = 0.f; acc[ki][1] = 0.f; acc[ki][2] = 0.f; acc[ki][3] = 0.f;
            }
        }
    }
    // reduce the 8 warps' K-slice sums (each warp covered a different K range)
    __shared__ float red[8][64];
    red[warp][lane * 2 + 0] = total;
    red[warp][lane * 2 + 1] = total2;
    __syncthreads();
    if (warp == 0) {
        float s0 = 0.f, s1 = 0.f;
        #pragma unroll
        for (int u = 0; u < 8; u++) {
            s0 += red[u][lane * 2 + 0];
            s1 += red[u][lane * 2 + 1];
        }
        if ((lane & 3) == 0) {
            const int hrow = h0 + (lane >> 2);
            if (hrow < hidden) out[(size_t)tok * hidden + hrow] = s0;
            if (hrow + 8 < hidden) out[(size_t)tok * hidden + hrow + 8] = s1;
        }
    }
}

// ============================================================
// moe_down_bf16_mma: the down projection on the tensor core with bf16
// operands. The fp8 (e4m3) variant was PROVEN mathematically correct
// (e4m3-exact act fill -> bad=0) but its act quantization (3-bit mantissa,
// up to 6.25% per element, ~2-3% on the dot) compounds over 42 layers and
// destroys the text. bf16 has an 8-bit mantissa (~0.4%), which survives.
//
// Weights stay fp8 in memory (half the bytes) and are converted to bf16 in
// the smem staging; the act is converted fp32 -> bf16 in the staging.
// Block = (16 hidden rows, one token); the 8 warps split the K
// (inter/8 = 32 each = two m16n8k16 tiles).
// ============================================================
__global__ void __launch_bounds__(256, 4) moe_down_bf16_mma_kernel(
    const float* __restrict__ ids_f,
    const float* __restrict__ probs,
    const unsigned char* const* __restrict__ down_w8_ptrs,
    const float* const* __restrict__ down_scale_ptrs,
    const unsigned char* __restrict__ shared_down_w8,
    const float* __restrict__ shared_down_scale,
    const float* __restrict__ act,
    float* __restrict__ out,
    int expert_start, int e_local, int hidden, int inter,
    int inter_shared, int topk, int dscols, int n) {
    const int h0 = blockIdx.x * 16;
    const int tok = blockIdx.y;
    if (h0 >= hidden || tok >= n) return;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int c0 = (lane & 3) * 4;
    const int kper = ((inter + 7) / 8 + 15) & ~15;   // 16-aligned (m16n8k16)
    const int k0 = warp * kper;
    const int k1 = min(k0 + kper, inter);
    const int stride = topk * inter + inter_shared;
    // A tile: 16 rows x 32 K bf16 (two m16n8k16 tiles), padded stride 40
    __shared__ __nv_bfloat16 sw[8][16][40];
    __shared__ __nv_bfloat16 sa[8][32];
    float total = 0.f, total2 = 0.f;
    const int nslot = topk + (inter_shared > 0 ? 1 : 0);
    for (int slot = 0; slot < nslot; slot++) {
        const unsigned char* w8; const float* ws; const float* arow; float p;
        if (slot < topk) {
            const int eid = (int)ids_f[(size_t)tok * topk + slot];
            const int local = eid - expert_start;
            if (local < 0 || local >= e_local) continue;
            w8 = down_w8_ptrs[local]; ws = down_scale_ptrs[local];
            arow = act + (size_t)tok * stride + (size_t)slot * inter;
            p = probs[(size_t)tok * topk + slot];
        } else {
            if (inter_shared <= 0) break;
            w8 = shared_down_w8; ws = shared_down_scale;
            arow = act + (size_t)tok * stride + (size_t)topk * inter;
            p = 1.0f;
        }
        // ---- B: this warp's 32 act values -> bf16 ----
        // BUGFIX: was `for (q < 2) base = q*16 + lane` with lane 0..31 ->
        // wrote base 16..47 into sa[8][32], corrupting the NEIGHBOURING
        // warps' smem (the bench read 1e34-magnitude garbage). 32 values =
        // one write per lane.
        {
            const int l = k0 + lane;
            sa[warp][lane] = __float2bfloat16((l < inter) ? arow[l] : 0.f);
        }
        // ---- A: weight rows h0..h0+15, K slice k0..k0+31 -> bf16 ----
        // 16 rows x 32 K = 512 elements / 32 lanes = 16 each
        #pragma unroll
        for (int q = 0; q < 16; q++) {
            const int idx = lane + q * 32;          // 0..511
            const int row = idx >> 5, kk = idx & 31;
            const int l = k0 + kk;
            unsigned char v8 = 0;
            if (l < inter && h0 + row < hidden)
                v8 = w8[(size_t)(h0 + row) * inter + l];
            sw[warp][row][kk] = __float2bfloat16(
                __half2float(__nv_cvt_fp8_to_halfraw(v8, __NV_E4M3)));
        }
        __syncwarp();
        const float wsc = ws[(size_t)(h0 >> 7) * dscols + (k0 >> 7)];
        // ---- two m16n8k16 MMAs (K = 32 for this warp) ----
        #pragma unroll
        for (int t = 0; t < 2; t++) {
            const int kb = t * 16;
            // A fragment for m16n8k16 via DIRECT smem loads (bisect: replaces
            // ldmatrix to rule out its addressing). m16n8k16 A layout per
            // lane l: (row l/4, k (l%4)*2) in a0, (row l/4+8) in a1,
            // (k +8) in a2, (row+8, k+8) in a3.
            unsigned a[4];
            {
                const int r0 = lane >> 2, cc = (lane & 3) * 2;
                a[0] = *(const unsigned*)&sw[warp][r0][kb + cc];
                a[1] = *(const unsigned*)&sw[warp][r0 + 8][kb + cc];
                a[2] = *(const unsigned*)&sw[warp][r0][kb + cc + 8];
                a[3] = *(const unsigned*)&sw[warp][r0 + 8][kb + cc + 8];
            }
            // B fragment: lane l holds B[(l%4)*2 ..][l/4] for the 16x8 tile
            unsigned b[2];
            {
                const __nv_bfloat16* bp = sa[warp] + kb + (lane & 3) * 2;
                b[0] = *(const unsigned*)bp;
                const __nv_bfloat16* bp2 = sa[warp] + kb + 8 + (lane & 3) * 2;
                b[1] = *(const unsigned*)bp2;
            }
            float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
            const float f = wsc * p;
            if ((lane & 3) == 0) { total += d0 * f; total2 += d2 * f; }
        }
    }
    __shared__ float red[8][64];
    red[warp][lane * 2 + 0] = total;
    red[warp][lane * 2 + 1] = total2;
    __syncthreads();
    if (warp == 0) {
        float s0 = 0.f, s1 = 0.f;
        #pragma unroll
        for (int u = 0; u < 8; u++) { s0 += red[u][lane * 2 + 0]; s1 += red[u][lane * 2 + 1]; }
        if ((lane & 3) == 0) {
            const int hrow = h0 + (lane >> 2);
            if (hrow < hidden) out[(size_t)tok * hidden + hrow] = s0;
            if (hrow + 8 < hidden) out[(size_t)tok * hidden + hrow + 8] = s1;
        }
    }
}

extern "C" cudaError_t ferrite_moe_down_bf16_mma(
    const float* ids_f, const float* probs,
    const unsigned char* const* down_w8_ptrs, const float* const* down_scale_ptrs,
    const unsigned char* shared_down_w8, const float* shared_down_scale,
    const float* act, float* out,
    int expert_start, int e_local, int hidden, int inter,
    int inter_shared, int topk, int dscols, int n, cudaStream_t s) {
    if (n <= 0 || n > 16 || (hidden & 15) != 0) return cudaErrorInvalidValue;
    dim3 grid((unsigned)(hidden / 16), (unsigned)n);
    moe_down_bf16_mma_kernel<<<grid, 256, 0, s>>>(
        ids_f, probs, down_w8_ptrs, down_scale_ptrs, shared_down_w8,
        shared_down_scale, act, out, expert_start, e_local, hidden, inter,
        inter_shared, topk, dscols, n);
    return cudaGetLastError();
}

extern "C" cudaError_t ferrite_moe_down_mma(
    const float* ids_f, const float* probs,
    const unsigned char* const* down_w8_ptrs, const float* const* down_scale_ptrs,
    const unsigned char* shared_down_w8, const float* shared_down_scale,
    const float* act, float* out,
    int expert_start, int e_local, int hidden, int inter,
    int inter_shared, int topk, int dscols, int n, cudaStream_t s) {
    if (n <= 0 || n > 16 || (hidden & 15) != 0) return cudaErrorInvalidValue;
    dim3 grid((unsigned)(hidden / 16), (unsigned)n);
    moe_down_mma_kernel<0><<<grid, 256, 0, s>>>(
        ids_f, probs, down_w8_ptrs, down_scale_ptrs, shared_down_w8,
        shared_down_scale, act, out, expert_start, e_local, hidden, inter,
        inter_shared, topk, dscols, n);
    return cudaGetLastError();
}

extern "C" cudaError_t ferrite_moe_fused_down_sum_fp8(
    const float* ids_f, const float* probs,
    const void* const* down_w8_ptrs, const void* const* down_scale_ptrs,
    const void* shared_down_w8, const void* shared_down_scale,
    const float* act, float* out,
    int expert_start, int e_local, int hidden, int inter,
    int inter_shared, int topk, int n, int dscols, cudaStream_t s) {
    // DIAGNOSTIC ONLY (FERRITE_MOE_SKIP=1): skip the launch to A/B the MoE's
    // share of the step. Output is garbage by construction; timing only.
    static const bool moe_skip_ = getenv("FERRITE_MOE_SKIP") != nullptr || getenv("FERRITE_SKIP_DOWN") != nullptr;
    if (moe_skip_) return cudaSuccess;

    if (n == 1) {
        // n=1 PATH (v0 warp-serial): the v12.1 expert-parallel + register-cache
        // variant's 48 regs/lane dropped occupancy — n=1 measured 90.9 (v0)
        // vs 87.2 (v12.1) tok/s. grid (hidden/8, 1): each warp owns ONE h row,
        // serial j loop over topk+shared. The v12.1 h-loop path stays for n>1.
        moe_fused_down_sum_fp8_v0_kernel<<<dim3((hidden + 7) / 8, 1, 1), 256, 0, s>>>(
            ids_f, probs,
            (const unsigned char* const*)down_w8_ptrs, (const float* const*)down_scale_ptrs,
            (const unsigned char*)shared_down_w8, (const float*)shared_down_scale,
            act, out, expert_start, e_local, hidden, inter, inter_shared, topk, dscols);
        return cudaGetLastError();
    }
    dim3 block(288); // 9 warps: topk routed (8) + shared
    // 4 tokens per block: 512 blocks (all tokens) starved the SMs; 8192
    // (one token) paid the fixed per-block latency 16x.
    dim3 grid((hidden + 7) / 8, (n + 3) / 4, 1);
    moe_fused_down_sum_fp8_kernel<<<grid, block, 0, s>>>(
        ids_f, probs,
        (const unsigned char* const*)down_w8_ptrs, (const float* const*)down_scale_ptrs,
        (const unsigned char*)shared_down_w8, (const float*)shared_down_scale,
        act, out, expert_start, e_local, hidden, inter, inter_shared, topk, dscols, n);
    return cudaGetLastError();
}


// ============================================================
// moe_fused_act_fp8_mma (W8A8): the act stage of the fused MoE on the
// tensor core — per (16-row block, slot, token): v1-mode per-block quant
// (x -> e4m3 smem, absmax/448), GATE mma + UP mma (both [16, K] against
// the SAME smem xq — the N=8 replica of B), then the swiglu epilogue on
// the two 16-row results. The W8A16 dequant loop (moe_fused_act_fp8)
// measured 0.94x bf16 — the mma path halves the expert weight HBM bytes
// (fp8) AND computes e4m3 x e4m3 directly (no per-element cvt).
// smem: xq[hidden] + reduce[256] + xs[1] + gate sacc[8][16] + up sacc[8][16].
// ============================================================
__global__ void __launch_bounds__(256, 3) moe_fused_act_fp8_mma_kernel(
    const float* __restrict__ x,          // [n, hidden]
    const float* __restrict__ ids_f,      // [n, topk]
    const unsigned char* const* __restrict__ gate_w8_ptrs,   // [e_local] [inter, hidden] e4m3
    const float* const* __restrict__ gate_scale_ptrs,       // [e_local] [inter/128, hidden/128]
    const unsigned char* const* __restrict__ up_w8_ptrs,    // [e_local] [inter, hidden]
    const float* const* __restrict__ up_scale_ptrs,
    const unsigned char* __restrict__ shared_gate_w8,      // [inter_shared, hidden]
    const float* __restrict__ shared_gate_scale,
    const unsigned char* __restrict__ shared_up_w8,
    const float* __restrict__ shared_up_scale,
    float* __restrict__ act,              // [n, topk*inter + inter_shared]
    int expert_start, int e_local, int hidden, int inter,
    int inter_shared, int topk, float limit,
    const unsigned char* __restrict__ xq, // [n, hidden] e4m3 — PRE-QUANTIZED (v2; null = per-block v1)
    const float* __restrict__ xs) {        // [n] per-token scales (v2)
    const int slot = blockIdx.y;
    const int tok = blockIdx.z;
    const int m0 = blockIdx.x * 16;        // 16 inter rows per block
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r0 = lane >> 2, c0 = (lane & 3) * 4;
    const int stride = topk * inter + inter_shared;
    const float* xt = x + (size_t)tok * hidden;
    int slot_rows, slot_base;
    const unsigned char *gw8, *uw8;
    const float *gs, *us;
    if (slot < topk) {
        slot_rows = inter;
        slot_base = slot * inter;
        int eid = (int)ids_f[(size_t)tok * topk + slot];
        int local = eid - expert_start;
        if (local < 0 || local >= e_local || m0 >= slot_rows) {
            // another rank's slot or tail rows: zero (act buffer pre-zeroed by
            // the caller for cross-rank slots; tail rows just skip writes)
            return;
        }
        gw8 = gate_w8_ptrs[local]; gs = gate_scale_ptrs[local];
        uw8 = up_w8_ptrs[local];  us = up_scale_ptrs[local];
    } else {
        slot_rows = inter_shared;
        slot_base = topk * inter;
        if (m0 >= slot_rows) return;
        gw8 = shared_gate_w8; gs = shared_gate_scale;
        uw8 = shared_up_w8;   us = shared_up_scale;
    }
    // ---- 1. quantize (v1 per-block | v2 PRE-QUANTIZED copy) ----
    // v2 (xq non-null): the per-token xq/xs were computed ONCE by
    // ferrite_quant_e4m3_tokens (1 block/token vs 864 re-quantizes here:
    // grid (max_rows/16, topk+1, n) re-quantized x[tok] 96x9x per layer,
    // each 16KB read + absmax reduce + 2 barriers = the 46µs kernel's
    // dominant cost, NOT the expert A-weights stream). Copy the 4KB row
    // into smem (coalesced, 1 read) + load xs[tok]. FP: xs is the SAME
    // absmax/448 (fmaxf commutative — the 8-warp serial fold order is
    // preserved in the quant kernel) — bit-identical act output.
    extern __shared__ unsigned char smem[];
    unsigned char* sx = smem;                       // [hidden] e4m3 xq
    float* sred = (float*)(smem + hidden);         // [256] absmax reduce
    float* sxs = (float*)(smem + hidden + 256 * 4); // [1] x_scale
    float* sgacc = sxs + 1;                        // [8][16] gate partials
    float* suacc = sgacc + 8 * 16;                 // [8][16] up partials
    if (xq != nullptr) {
        const unsigned char* xqt = xq + (size_t)tok * hidden;
        for (int k = threadIdx.x * 16; k + 15 < hidden; k += 256 * 16) {
            *reinterpret_cast<uint4*>(sx + k) = *reinterpret_cast<const uint4*>(xqt + k);
        }
        for (int k = threadIdx.x + ((hidden >> 4) << 4); k < hidden; k += 256)
            sx[k] = xqt[k]; // tail (hidden%16 != 0 — never on GLM)
        if (threadIdx.x == 0) sxs[0] = xs[tok];
        __syncthreads();
    } else {
        const float* xt2 = x + (size_t)tok * hidden;
        float amax = 1e-9f;
        for (int k = threadIdx.x; k < hidden; k += 256)
            amax = fmaxf(amax, fabsf(xt2[k]));
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            amax = fmaxf(amax, __shfl_down_sync(0xffffffff, amax, off));
        if ((threadIdx.x & 31) == 0) sred[threadIdx.x >> 5] = amax;
        __syncthreads();
        if (threadIdx.x == 0) {
            float m = sred[0];
            #pragma unroll
            for (int w = 1; w < 8; w++) m = fmaxf(m, sred[w]);
            sxs[0] = m / 448.0f;
        }
        __syncthreads();
        const float inv = 1.0f / sxs[0];
        for (int k = threadIdx.x; k < hidden; k += 256) {
            const float q = fminf(fmaxf(xt2[k] * inv, -448.0f), 448.0f);
            sx[k] = (unsigned char)__nv_cvt_float_to_fp8(q, __NV_SATFINITE, __NV_E4M3);
        }
        __syncthreads();
    }
    // ---- 2. gate mma + up mma (same smem xq; 8-warp K-split each) ----
    const int nblk = (hidden + 127) >> 7;
    const int bseg = (nblk + 7) / 8;
    const int kW = warp;
    const int k0 = kW * bseg * 128;
    const int k1 = min(k0 + bseg * 128, hidden);
    const int gs_ws = (m0 / 128) * ((hidden + 127) >> 7);   // scale row (16 rows share m/128)
    float g0 = 0.f, g1 = 0.f, u0 = 0.f, u1 = 0.f;
    // PER-WARP padded smem staging. The A-fragment loads were 4-byte across 8
    // rows = 8 sectors per instruction with only 16B used each (50% sector
    // efficiency -> the kernel streamed 2.3x the bytes it needs and was
    // bandwidth-bound at 3.2ms). Staging 16 rows x 64 cols x 2 projections
    // with 16-byte loads makes every sector fully used. The row stride is
    // PADDED to 80 bytes: stride 64 would start every row on bank 0 (8-way
    // conflict on the fragment reads) — 80 gives banks 0,20,8,28,16,4,24,12.
    const int SA_STRIDE = 80;
    // DOUBLE-BUFFERED cp.async staging: global -> smem directly (no register
    // round-trip) with two buffers, so the next tile's copy is in flight while
    // the MMAs run on the current one. The old single-buffer + register
    // prefetch still stalled ~536 cycles/tile (600-cycle DRAM latency vs the
    // ~64 cycles of MMA work) — that is why the 2-tile *register* prefetch was
    // slower (32 extra registers); cp.async costs smem instead (40KB/block).
    // 2-deep pipeline (40KB): ncu showed the 60KB 3-deep version capped
    // theoretical occupancy at 37.5% (shared-memory-limited, est. 39% speedup
    // available). One tile still stays in flight: the next tile is issued
    // BEFORE the wait_group 1, so the wait only has to drain the current one.
    __shared__ unsigned char sa[2][8][2 * 16 * 80];
    #define ACT_ISSUE(TILE, BUF) do { \
        for (int t = lane; t < 128; t += 32) { \
            const int proj = t >> 6, off = t & 63; \
            const int row = off >> 2, col = (off & 3) * 16; \
            const unsigned char* src_ = (proj ? uw8 : gw8) + (size_t)(m0 + row) * hidden + (TILE) + col; \
            unsigned char* dst_ = sa[BUF][warp] + proj * (16 * SA_STRIDE) + row * SA_STRIDE + col; \
            const unsigned int sd_ = (unsigned int)__cvta_generic_to_shared(dst_); \
            asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" :: "r"(sd_), "l"(src_)); \
        } \
        asm volatile("cp.async.commit_group;\n"); \
    } while (0)
    ACT_ISSUE(k0, 0);
    int sbuf = 0;
    for (int kb = k0; kb < k1; kb += 64, sbuf ^= 1) {
        // issue the NEXT tile into the other buffer first, then drain all but
        // the one in flight (wait_group 1 = the current tile is complete).
        if (kb + 64 < k1) ACT_ISSUE(kb + 64, sbuf ^ 1);
        if (kb + 64 < k1) asm volatile("cp.async.wait_group 1;\n");
        else asm volatile("cp.async.wait_group 0;\n");
        __syncwarp();
        float gd0 = 0.f, gd1 = 0.f, gd2 = 0.f, gd3 = 0.f;
        float ud0 = 0.f, ud1 = 0.f, ud2 = 0.f, ud3 = 0.f;
        #pragma unroll
        for (int kk = kb; kk < kb + 64; kk += 32) {
            const int kkl = kk - kb;
            // ldmatrix.x4 replaces 8 scalar 4-byte smem loads. The fp8
            // m16n8k32 A fragment is 16 rows x 32 K = four 8x8 b16 tiles, and
            // its per-lane layout (row l/4, k (l%4)*4) is exactly ldmatrix's
            // output. Lane l supplies its sub-tile's row address: (l&15) picks
            // the row, (l>>4) the 16-byte K half. SA_STRIDE=80 keeps every
            // address 16-byte aligned.
            const unsigned saddr_a = (unsigned)__cvta_generic_to_shared(
                sa[sbuf][warp] + (size_t)(lane & 15) * SA_STRIDE + kkl + ((lane >> 4) * 16));
            unsigned ba[4];
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                         : "=r"(ba[0]), "=r"(ba[1]), "=r"(ba[2]), "=r"(ba[3])
                         : "r"(saddr_a));
            const unsigned saddr_u = (unsigned)__cvta_generic_to_shared(
                sa[sbuf][warp] + 16 * SA_STRIDE + (size_t)(lane & 15) * SA_STRIDE + kkl + ((lane >> 4) * 16));
            unsigned b1_[4]; // up A fragments
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                         : "=r"(b1_[0]), "=r"(b1_[1]), "=r"(b1_[2]), "=r"(b1_[3])
                         : "r"(saddr_u));
            unsigned b[2];   // B: smem xq (n=8 replica)
            b[0] = *(const unsigned*)(sx + kk + c0);
            b[1] = *(const unsigned*)(sx + kk + c0 + 16);
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(gd0), "+f"(gd1), "+f"(gd2), "+f"(gd3)
                : "r"(ba[0]), "r"(ba[1]), "r"(ba[2]), "r"(ba[3]),
                  "r"(b[0]), "r"(b[1]));
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(ud0), "+f"(ud1), "+f"(ud2), "+f"(ud3)
                : "r"(b1_[0]), "r"(b1_[1]), "r"(b1_[2]), "r"(b1_[3]),
                  "r"(b[0]), "r"(b[1]));
        }
        const int kblk = kb >> 7;
        const float gw_sc = gs[gs_ws + kblk];
        const float uw_sc = us[gs_ws + kblk];
        if ((lane & 3) == 0) {
            g0 += gd0 * gw_sc; g1 += gd2 * gw_sc;   // (r0, col0), (r0+8, col0)
            u0 += ud0 * uw_sc; u1 += ud2 * uw_sc;
        }
        // Safe to fill the OTHER buffer now: this iteration's MMA operands are
        // already in registers.
        // (the next tile was already issued at the top of this iteration)
    }
    #undef ACT_ISSUE
    if ((lane & 3) == 0) {
        sgacc[warp * 16 + r0] = g0;
        sgacc[warp * 16 + r0 + 8] = g1;
        suacc[warp * 16 + r0] = u0;
        suacc[warp * 16 + r0 + 8] = u1;
    }
    __syncthreads();
    // ---- 3. swiglu epilogue: act[r] = silu(min(g, limit)) * clamp(u, ±limit) ----
    // g/u are e4m3(W)·w_scale·e4m3(x/x_s) dots — scale by x_s (the per-token
    // quant scale) before the nonlinearity (v1 gemv epilogue semantics).
    if (warp == 0 && lane < 16) {
        float g = 0.f, u = 0.f;
        for (int i = 0; i < 8; i++) {
            g += sgacc[i * 16 + lane];
            u += suacc[i * 16 + lane];
        }
        g *= sxs[0];
        u *= sxs[0];
        g = fminf(g, limit);
        u = fminf(fmaxf(u, -limit), limit);
        const int r = m0 + lane;
        if (r < slot_rows) {
            act[(size_t)tok * stride + slot_base + r] = (g / (1.0f + expf(-g))) * u;
        }
    }
    (void)e_local;
}

extern "C" cudaError_t ferrite_moe_fused_act_fp8_mma(
    const float* x, const float* ids_f,
    const void* const* gate_w8_ptrs, const void* const* gate_scale_ptrs,
    const void* const* up_w8_ptrs, const void* const* up_scale_ptrs,
    const void* shared_gate_w8, const void* shared_gate_scale,
    const void* shared_up_w8, const void* shared_up_scale,
    float* act, int expert_start, int e_local, int hidden, int inter,
    int inter_shared, int topk, int n, float limit, cudaStream_t s)
{
    int max_rows = inter > inter_shared ? inter : inter_shared;
    if (max_rows % 16 != 0 || hidden % 128 != 0) return cudaErrorNotSupported; // v1 alignment
    dim3 grid((unsigned)(max_rows / 16), topk + 1, n);
    const int smem = hidden + 256 * 4 + 4 + 2 * 8 * 16 * 4;
    cudaFuncSetAttribute(moe_fused_act_fp8_mma_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    moe_fused_act_fp8_mma_kernel<<<grid, 256, smem, s>>>(
        x, ids_f,
        (const unsigned char* const*)gate_w8_ptrs, (const float* const*)gate_scale_ptrs,
        (const unsigned char* const*)up_w8_ptrs, (const float* const*)up_scale_ptrs,
        (const unsigned char*)shared_gate_w8, (const float*)shared_gate_scale,
        (const unsigned char*)shared_up_w8, (const float*)shared_up_scale,
        act, expert_start, e_local, hidden, inter, inter_shared, topk, limit,
        nullptr, nullptr); // v1: per-block quantize fallback
    return cudaGetLastError();
}

// v2: pre-quantized variant — xq/xs come from ferrite_quant_e4m3_tokens
// (one quantize per token per layer vs 864 in-kernel re-quantizes).
extern "C" cudaError_t ferrite_moe_fused_act_fp8_mma_v2(
    const float* x, const float* ids_f,
    const void* const* gate_w8_ptrs, const void* const* gate_scale_ptrs,
    const void* const* up_w8_ptrs, const void* const* up_scale_ptrs,
    const void* shared_gate_w8, const void* shared_gate_scale,
    const void* shared_up_w8, const void* shared_up_scale,
    float* act, int expert_start, int e_local, int hidden, int inter,
    int inter_shared, int topk, int n, float limit,
    const void* xq, const void* xs, cudaStream_t s)
{
    // DIAGNOSTIC ONLY (FERRITE_MOE_SKIP=1): skip the launch to A/B the MoE's
    // share of the step. Output is garbage by construction; timing only.
    static const bool moe_skip_ = getenv("FERRITE_MOE_SKIP") != nullptr || getenv("FERRITE_SKIP_ACT") != nullptr;
    if (moe_skip_) return cudaSuccess;

    int max_rows = inter > inter_shared ? inter : inter_shared;
    if (max_rows % 16 != 0 || hidden % 128 != 0) return cudaErrorNotSupported;
    dim3 grid((unsigned)(max_rows / 16), topk + 1, n);
    const int smem = hidden + 256 * 4 + 4 + 2 * 8 * 16 * 4;
    cudaFuncSetAttribute(moe_fused_act_fp8_mma_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    moe_fused_act_fp8_mma_kernel<<<grid, 256, smem, s>>>(
        x, ids_f,
        (const unsigned char* const*)gate_w8_ptrs, (const float* const*)gate_scale_ptrs,
        (const unsigned char* const*)up_w8_ptrs, (const float* const*)up_scale_ptrs,
        (const unsigned char*)shared_gate_w8, (const float*)shared_gate_scale,
        (const unsigned char*)shared_up_w8, (const float*)shared_up_scale,
        act, expert_start, e_local, hidden, inter, inter_shared, topk, limit,
        (const unsigned char*)xq, (const float*)xs);
    return cudaGetLastError();
}

// ============================================================
// act quantize v2 (2026-09-08): per-token x quantize ONCE per layer —
// x [n, hidden] -> xq e4m3 [n, hidden] + xs [n]. The act mma kernel's
// per-block quantize (v1 no-barrier mode) re-quantized the SAME x[tok]
// (max_rows/16) × (topk+1) = 96 × 9 = 864 times per token per layer
// (n=3: 2592 blocks × 16KB read + absmax reduce + 2 barriers each —
// the 46µs act kernel is quantize-BOUND: A weights only stream
// ~5µs/layer at HBM). One 256-thr block per token: warp-shuffle max +
// serial 8-warp fold (the SAME reduction order as the act kernel's
// in-block quantize — fmaxf is order-commutative, xs bit-identical)
// + the same cvt clamp. The act kernel then copies xq into smem (4KB
// coalesced, ~0.5µs) instead of re-quantizing (~2µs × 864/SM-wave).
// ============================================================
__global__ void quant_e4m3_tokens_kernel(
    const float* __restrict__ x,          // [n, hidden]
    unsigned char* __restrict__ xq,      // [n, hidden] e4m3
    float* __restrict__ xs,              // [n] scale = absmax/448
    int n, int hidden) {
    // 1024 threads + float4 loads/stores (was 256 threads / scalar): the
    // per-step cost is ~200 calls (one per distinct (x, in_f) per layer), so
    // this kernel's latency is directly on the MMA gemv's critical path.
    int tok = blockIdx.x;
    if (tok >= n) return;
    const float4* x4 = (const float4*)(x + (size_t)tok * hidden);
    const int n4 = hidden >> 2;
    __shared__ float sred[32];
    float amax = 1e-9f;
    for (int k = threadIdx.x; k < n4; k += blockDim.x) {
        const float4 v = x4[k];
        amax = fmaxf(amax, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)),
                                 fmaxf(fabsf(v.z), fabsf(v.w))));
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        amax = fmaxf(amax, __shfl_down_sync(0xffffffff, amax, off));
    if ((threadIdx.x & 31) == 0) sred[threadIdx.x >> 5] = amax;
    __syncthreads();
    if (threadIdx.x == 0) {
        float m = sred[0];
        for (int w = 1; w < (blockDim.x >> 5); w++) m = fmaxf(m, sred[w]);
        xs[tok] = m / 448.0f;
    }
    __syncthreads();
    const float inv = 1.0f / xs[tok];
    unsigned char* qt = xq + (size_t)tok * hidden;
    for (int k = threadIdx.x; k < n4; k += blockDim.x) {
        const float4 v = x4[k];
        uchar4 o;
        o.x = (unsigned char)__nv_cvt_float_to_fp8(fminf(fmaxf(v.x * inv, -448.0f), 448.0f), __NV_SATFINITE, __NV_E4M3);
        o.y = (unsigned char)__nv_cvt_float_to_fp8(fminf(fmaxf(v.y * inv, -448.0f), 448.0f), __NV_SATFINITE, __NV_E4M3);
        o.z = (unsigned char)__nv_cvt_float_to_fp8(fminf(fmaxf(v.z * inv, -448.0f), 448.0f), __NV_SATFINITE, __NV_E4M3);
        o.w = (unsigned char)__nv_cvt_float_to_fp8(fminf(fmaxf(v.w * inv, -448.0f), 448.0f), __NV_SATFINITE, __NV_E4M3);
        *(uchar4*)(qt + k * 4) = o;
    }
}

extern "C" cudaError_t ferrite_quant_e4m3_tokens(
    const float* x, unsigned char* xq, float* xs,
    int n, int hidden, cudaStream_t s) {
    if (n <= 0) return cudaSuccess;
    quant_e4m3_tokens_kernel<<<n, 1024, 0, s>>>(x, xq, xs, n, hidden);
    return cudaGetLastError();
}

// Per-(token, slot) e4m3 quantization of the MoE act rows for the MMA down
// kernel (mirrors quant_e4m3_tokens but the rows are the STRIDED act slices:
// row (t, j) = act[t, j*inter .. +inter] for j<topk, and the shared row
// act[t, topk*inter .. +inter_shared]). One block per row.
__global__ void quant_act_rows_kernel(
    const float* __restrict__ act,        // [n, topk*inter + inter_shared]
    unsigned char* __restrict__ aq,       // same layout, e4m3
    float* __restrict__ as_,              // [n, topk+1] per-row scales
    int stride, int inter, int topk, int inter_shared) {
    const int t = blockIdx.x / (topk + 1);
    const int j = blockIdx.x % (topk + 1);
    const int klen = (j < topk) ? inter : inter_shared;
    const float* row = act + (size_t)t * stride + (size_t)j * inter;
    unsigned char* qrow = aq + (size_t)t * stride + (size_t)j * inter;
    __shared__ float sc[1];
    float am = 1e-9f;
    for (int k = threadIdx.x; k < klen; k += blockDim.x)
        am = fmaxf(am, fabsf(row[k]));
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) am = fmaxf(am, __shfl_down_sync(0xffffffff, am, off));
    __shared__ float red[8];
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = am;
    __syncthreads();
    if (threadIdx.x == 0) {
        float m = red[0];
        #pragma unroll
        for (int w = 1; w < 8; w++) m = fmaxf(m, red[w]);
        sc[0] = m / 448.0f;
        as_[(size_t)t * (topk + 1) + j] = sc[0];
    }
    __syncthreads();
    const float inv = 1.0f / sc[0];
    for (int k = threadIdx.x; k < klen; k += blockDim.x) {
        const float q = fminf(fmaxf(row[k] * inv, -448.0f), 448.0f);
        qrow[k] = (unsigned char)__nv_cvt_float_to_fp8(q, __NV_SATFINITE, __NV_E4M3);
    }
}

extern "C" cudaError_t ferrite_quant_act_rows(
    const float* act, unsigned char* aq, float* as_,
    int n, int stride, int inter, int topk, int inter_shared, cudaStream_t s) {
    if (n <= 0) return cudaSuccess;
    quant_act_rows_kernel<<<(unsigned)(n * (topk + 1)), 256, 0, s>>>(
        act, aq, as_, stride, inter, topk, inter_shared);
    return cudaGetLastError();
}

// ============================================================
// moe_down_e4m3_mma: the tensor-core MoE down projection.
// out[t, h] = Σ_j p_j · (act[t, j·inter..] · W_ej[h, :]^T) + shared.
//
// WHY: the SIMT fp8 down (moe_fused_down_sum_fp8) is INSTRUCTION-bound, not
// bandwidth-bound — the isolated bench (2026-09-09, /tmp/down_bench.cu)
// measured 108.8µs for random routing (109MB) vs 102.0µs when ALL tokens
// share 8 experts (8MB, fully L2-resident): the time is invariant to the
// weight traffic; the fp8→half2→float2 conversion chain per weight element
// dominates. This kernel runs the same GEMM on the e4m3 tensor cores
// (m16n8k32 — the weights are ALREADY e4m3; the act rows are pre-quantized
// by ferrite_quant_act_rows), mirroring the act kernel's proven staging:
// A = the weight rows [16 h, 32 k] via ldmatrix, B = the act row broadcast
// across the n8. The m16 spans 16 h-rows (1 real token column of C).
//
// Structure: grid (hidden/32, n) — 32 h-rows (2 m-tiles) × one token; the
// block loops the (topk+1) slots j-ASCENDING accumulating in registers —
// the per-token fold order is deterministic. 8 warps split K (each warp one
// k32 chunk of its slot's klen; klen=inter or inter_shared). The 272B smem
// row stride (17×16B) makes the ldmatrix bank-conflict-free; double-buffered
// cp.async pipelines the next slot's weight tile.
// ============================================================
__global__ void __launch_bounds__(256, 3) moe_down_e4m3_mma_kernel(
    const float* __restrict__ ids_f,       // [n, topk]
    const float* __restrict__ probs,       // [n, topk]
    const unsigned char* const* __restrict__ down_w8_ptrs,  // [e_local] [hidden, inter] e4m3
    const float* const* __restrict__ down_scale_ptrs,       // [e_local] [hidden/128, dscols]
    const unsigned char* __restrict__ shared_down_w8,       // [hidden, inter_shared]
    const float* __restrict__ shared_down_scale,
    const unsigned char* __restrict__ aq,   // [n, stride] e4m3 act rows (pre-quantized)
    const float* __restrict__ as_,          // [n, topk+1] per-row scales
    float* __restrict__ out,                // [n, hidden]
    int expert_start, int e_local, int hidden, int inter,
    int inter_shared, int topk, int dscols) {
    const int h0 = blockIdx.x * 32;
    const int t = blockIdx.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int stride = topk * inter + inter_shared;
    // smem: aq rows flat at 512B pitch (j<topk rows use 256B), staged weight
    // tile [32 rows][272B] ×2 buffers, warp partials, slot id/prob/scale.
    __shared__ unsigned char saq[(/*TOPK_MAX*/ 8 + 1) * 512];
    __shared__ unsigned char sW[2][32 * 272];
    __shared__ float part[8][32];
    __shared__ float ssp[9];   // as_[t,j] × p_j (folded slot scale); 0 = skip
    // ---- stage the token's act rows + slot metadata ----
    {
        const unsigned char* aq_t = aq + (size_t)t * stride;
        const int topk_inter = topk * inter;
        for (int off = threadIdx.x * 16; off < stride; off += 256 * 16) {
            // rows: [topk × inter] slices then ONE inter_shared slice
            int j, lo, klen;
            if (off < topk_inter) {
                j = off / inter; lo = off - j * inter; klen = inter;
            } else {
                j = topk; lo = off - topk_inter; klen = inter_shared;
            }
            if (lo + 16 <= klen)
                *reinterpret_cast<uint4*>(saq + j * 512 + lo) =
                    *reinterpret_cast<const uint4*>(aq_t + off);
        }
        // tail (stride%16 — never on GLM)
        if (threadIdx.x == 0) {
            for (int j = 0; j <= topk; j++) {
                const float p = (j < topk) ? probs[(size_t)t * topk + j] : 1.f;
                const float s = as_[(size_t)t * (topk + 1) + j];
                float v = s * p;
                if (j < topk) {
                    const int eid = (int)ids_f[(size_t)t * topk + j];
                    const int local = eid - expert_start;
                    if (local < 0 || local >= e_local || p == 0.f) v = 0.f;  // skip marker
                }
                ssp[j] = v;
            }
        }
        __syncthreads();
    }
    // ---- the slot loop with double-buffered weight staging ----
    #define DM_STAGE(J, BUF) do { \
        const float sp_ = ssp[J]; \
        if (sp_ != 0.f) { \
            const unsigned char* wbase = (J < topk) \
                ? down_w8_ptrs[(int)ids_f[(size_t)t * topk + J] - expert_start] \
                : shared_down_w8; \
            const int klen_ = (J < topk) ? inter : inter_shared; \
            /* 32 rows × klen bytes in 16B chunks; row stride 272 */ \
            for (int c = threadIdx.x; c < 32 * (klen_ >> 4); c += 256) { \
                const int r = c / (klen_ >> 4), cc = (c % (klen_ >> 4)) * 16; \
                const unsigned char* src_ = wbase + (size_t)(h0 + r) * klen_ + cc; \
                unsigned char* dst_ = sW[BUF] + r * 272 + cc; \
                const unsigned int sd_ = (unsigned int)__cvta_generic_to_shared(dst_); \
                asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" :: "r"(sd_), "l"(src_)); \
            } \
        } \
        asm volatile("cp.async.commit_group;\n"); \
    } while (0)
    DM_STAGE(0, 0);
    float accA0 = 0.f, accA1 = 0.f, accB0 = 0.f, accB1 = 0.f;  // m-tile 0/1 × (r0, r0+8)
    int buf = 0;
    for (int j = 0; j <= topk; j++, buf ^= 1) {
        if (j + 1 <= topk) DM_STAGE(j + 1, buf ^ 1);
        if (j + 1 <= topk) asm volatile("cp.async.wait_group 1;\n");
        else asm volatile("cp.async.wait_group 0;\n");
        __syncthreads();
        if (ssp[j] == 0.f) continue;
        const int klen = (j < topk) ? inter : inter_shared;
        const float* dsr = (j < topk)
            ? down_scale_ptrs[(int)ids_f[(size_t)t * topk + j] - expert_start]
            : shared_down_scale;
        const int srow = h0 >> 7;
        // per-warp k32 chunks: klen/32 chunks over 8 warps (klen=256 → 1 each, 512 → 2)
        for (int kc = warp; kc < (klen >> 5); kc += 8) {
            const float wsc = dsr[(size_t)srow * dscols + ((kc << 5) >> 7)] * ssp[j];
            const unsigned char* arow = saq + j * 512 + (kc << 5);
            #pragma unroll
            for (int m = 0; m < 2; m++) {
                // A fragment: 16 rows × 32 k from the staged tile (ldmatrix.x4)
                const unsigned saddr_a = (unsigned)__cvta_generic_to_shared(
                    sW[buf] + (size_t)(m * 16 + (lane & 15)) * 272 + (kc << 5) + ((lane >> 4) * 16));
                unsigned a[4];
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(saddr_a));
                unsigned b[2];
                const int c0 = (lane & 3) * 4;
                b[0] = *(const unsigned*)(arow + c0);
                b[1] = *(const unsigned*)(arow + c0 + 16);
                float gd0 = 0.f, gd1 = 0.f, gd2 = 0.f, gd3 = 0.f;
                asm volatile(
                    "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                    : "+f"(gd0), "+f"(gd1), "+f"(gd2), "+f"(gd3)
                    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
                if ((lane & 3) == 0) {
                    if (m == 0) { accA0 += gd0 * wsc; accA1 += gd2 * wsc; }
                    else        { accB0 += gd0 * wsc; accB1 += gd2 * wsc; }
                }
            }
        }
        __syncthreads();
    }
    #undef DM_STAGE
    // ---- cross-warp K reduction (warp-ascending, deterministic) ----
    if ((lane & 3) == 0) {
        const int r0 = lane >> 2;
        part[warp][r0] = accA0;
        part[warp][r0 + 8] = accA1;
        part[warp][16 + r0] = accB0;
        part[warp][16 + r0 + 8] = accB1;
    }
    __syncthreads();
    if (warp == 0) {
        for (int r = lane; r < 32; r += 32) {
            float s = 0.f;
            #pragma unroll
            for (int w = 0; w < 8; w++) s += part[w][r];
            out[(size_t)t * hidden + h0 + r] = s;
        }
    }
}

extern "C" cudaError_t ferrite_moe_down_e4m3_mma(
    const float* ids_f, const float* probs,
    const void* const* down_w8_ptrs, const void* const* down_scale_ptrs,
    const void* shared_down_w8, const void* shared_down_scale,
    const unsigned char* aq, const float* as_,
    float* out,
    int expert_start, int e_local, int hidden, int inter,
    int inter_shared, int topk, int n, int dscols, cudaStream_t s) {
    if (n <= 0) return cudaSuccess;
    if (hidden % 32 != 0 || (inter & 31) != 0 || (inter_shared & 31) != 0)
        return cudaErrorNotSupported;
    // the smem tile rows are 272B (256 data + 16 pad): klen must fit. GLM:
    // inter = inter_shared = moe_inter/tp = 256.
    if (inter > 256 || inter_shared > 256 || topk > 8) return cudaErrorNotSupported;
    dim3 grid((unsigned)(hidden / 32), (unsigned)n);
    moe_down_e4m3_mma_kernel<<<grid, 256, 0, s>>>(
        ids_f, probs,
        (const unsigned char* const*)down_w8_ptrs, (const float* const*)down_scale_ptrs,
        (const unsigned char*)shared_down_w8, (const float*)shared_down_scale,
        aq, as_, out, expert_start, e_local, hidden, inter, inter_shared, topk, dscols);
    return cudaGetLastError();
}

// ============================================================
// moe_down_e4m3_mma2: v2 — per-(assignment, h-tile) blocks.
// v1 (grid (hidden/32, n)) reads each block's 9 experts' 8KB slices — the
// same scattered-run pattern as the SIMT version (2.5TB/s vs the act
// kernel's 5TB/s). v2 gives every block ONE 32KB CONTIGUOUS run of ONE
// expert's weights (W_e[h0..h0+128, :]) — the act kernel's access shape.
// The per-(token, slot) contributions land in a partials buffer
// [n][topk+1][hidden]; a deterministic reduce (j-ascending) sums them.
// ============================================================
__global__ void __launch_bounds__(256, 3) moe_down_e4m3_mma2_kernel(
    const float* __restrict__ ids_f, const float* __restrict__ probs,
    const unsigned char* const* __restrict__ down_w8_ptrs,
    const float* const* __restrict__ down_scale_ptrs,
    const unsigned char* __restrict__ shared_down_w8,
    const float* __restrict__ shared_down_scale,
    const unsigned char* __restrict__ aq, const float* __restrict__ as_,
    float* __restrict__ partial,          // [n][topk+1][hidden]
    int expert_start, int e_local, int hidden, int inter,
    int inter_shared, int topk, int dscols) {
    const int h0 = blockIdx.x * 128;
    const int t = blockIdx.y / (topk + 1);
    const int j = blockIdx.y % (topk + 1);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const unsigned char* wbase; const float* dsr; int klen; float sp;
    if (j < topk) {
        const int eid = (int)ids_f[(size_t)t * topk + j];
        const int local = eid - expert_start;
        const float p = probs[(size_t)t * topk + j];
        if (local < 0 || local >= e_local || p == 0.f) return;  // the reduce skips this slot
        wbase = down_w8_ptrs[local]; dsr = down_scale_ptrs[local];
        klen = inter; sp = as_[(size_t)t * (topk + 1) + j] * p;
    } else {
        wbase = shared_down_w8; dsr = shared_down_scale;
        klen = inter_shared; sp = as_[(size_t)t * (topk + 1) + j];
    }
    __shared__ unsigned char sW[128 * 272];
    __shared__ unsigned char srow[512];
    __shared__ float part[8][129];
    // stage the aq row (≤256B)
    {
        const unsigned char* arow = aq + (size_t)t * (topk * inter + inter_shared) + (size_t)j * inter;
        for (int c = (int)threadIdx.x * 16; c < klen; c += 256 * 16)
            *reinterpret_cast<uint4*>(srow + c) = *reinterpret_cast<const uint4*>(arow + c);
    }
    // stage W_e[h0..h0+128, :klen] — 128 rows × klen bytes, contiguous 32KB
    for (int c = threadIdx.x; c < 128 * (klen >> 4); c += 256) {
        const int r = c / (klen >> 4), cc = (c % (klen >> 4)) * 16;
        const unsigned char* src_ = wbase + (size_t)(h0 + r) * klen + cc;
        unsigned char* dst_ = sW + r * 272 + cc;
        const unsigned int sd_ = (unsigned int)__cvta_generic_to_shared(dst_);
        asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" :: "r"(sd_), "l"(src_));
    }
    asm volatile("cp.async.commit_group;\n");
    asm volatile("cp.async.wait_group 0;\n");
    __syncthreads();
    // MMA: warp w owns k-chunk w (klen=256 → 8 warps exactly); 8 m-tiles
    const int kc = warp;
    if (kc < (klen >> 5)) {
        const float wsc = dsr[(size_t)(h0 >> 7) * dscols + ((kc << 5) >> 7)] * sp;
        const unsigned char* arow = srow + (kc << 5);
        float acc0[8] = {0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f};
        float acc1[8] = {0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f};
        #pragma unroll
        for (int m = 0; m < 8; m++) {
            const unsigned saddr_a = (unsigned)__cvta_generic_to_shared(
                sW + (size_t)(m * 16 + (lane & 15)) * 272 + (kc << 5) + ((lane >> 4) * 16));
            unsigned a[4];
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                         : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(saddr_a));
            unsigned b[2];
            const int c0 = (lane & 3) * 4;
            b[0] = *(const unsigned*)(arow + c0);
            b[1] = *(const unsigned*)(arow + c0 + 16);
            float gd0 = 0.f, gd1 = 0.f, gd2 = 0.f, gd3 = 0.f;
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(gd0), "+f"(gd1), "+f"(gd2), "+f"(gd3)
                : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
            if ((lane & 3) == 0) {
                acc0[m] += gd0 * wsc;   // row m*16 + r0
                acc1[m] += gd2 * wsc;   // row m*16 + r0 + 8
            }
        }
        if ((lane & 3) == 0) {
            const int r0 = lane >> 2;
            #pragma unroll
            for (int m = 0; m < 8; m++) {
                part[warp][m * 16 + r0] = acc0[m];
                part[warp][m * 16 + r0 + 8] = acc1[m];
            }
        }
    }
    __syncthreads();
    // cross-warp K-reduce (warp-ascending, deterministic) → the partial
    if (warp == 0) {
        const int KW = klen >> 5;
        for (int r = lane; r < 128; r += 32) {
            float s = 0.f;
            for (int w = 0; w < KW; w++) s += part[w][r];
            partial[(size_t)t * (topk + 1) * hidden + (size_t)j * hidden + h0 + r] = s;
        }
    }
}

__global__ void moe_down_e4m3_reduce_kernel(
    const float* __restrict__ partial,   // [n][topk+1][hidden]
    const float* __restrict__ ids_f, const float* __restrict__ probs,
    float* __restrict__ out,             // [n, hidden]
    int hidden, int topk, int expert_start, int e_local) {
    const int t = blockIdx.y;
    const int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= hidden) return;
    const size_t base = (size_t)t * (topk + 1) * hidden + h;
    float s = 0.f;
    for (int j = 0; j < topk; j++) {
        const int eid = (int)ids_f[(size_t)t * topk + j];
        const int local = eid - expert_start;
        const float p = probs[(size_t)t * topk + j];
        if (local >= 0 && local < e_local && p != 0.f)
            s += partial[base + (size_t)j * hidden];
    }
    s += partial[base + (size_t)topk * hidden];  // the shared expert (always local)
    out[(size_t)t * hidden + h] = s;
}

extern "C" cudaError_t ferrite_moe_down_e4m3_mma2(
    const float* ids_f, const float* probs,
    const void* const* down_w8_ptrs, const void* const* down_scale_ptrs,
    const void* shared_down_w8, const void* shared_down_scale,
    const unsigned char* aq, const float* as_,
    float* partial, float* out,
    int expert_start, int e_local, int hidden, int inter,
    int inter_shared, int topk, int n, int dscols, cudaStream_t s) {
    if (n <= 0) return cudaSuccess;
    if (hidden % 128 != 0 || (inter & 31) != 0 || (inter_shared & 31) != 0)
        return cudaErrorNotSupported;
    if (inter > 256 || inter_shared > 256 || topk > 8) return cudaErrorNotSupported;
    dim3 grid((unsigned)(hidden / 128), (unsigned)(n * (topk + 1)));
    moe_down_e4m3_mma2_kernel<<<grid, 256, 0, s>>>(
        ids_f, probs,
        (const unsigned char* const*)down_w8_ptrs, (const float* const*)down_scale_ptrs,
        (const unsigned char*)shared_down_w8, (const float*)shared_down_scale,
        aq, as_, partial, expert_start, e_local, hidden, inter, inter_shared, topk, dscols);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return e;
    dim3 rgrid((unsigned)((hidden + 255) / 256), (unsigned)n);
    moe_down_e4m3_reduce_kernel<<<rgrid, 256, 0, s>>>(
        partial, ids_f, probs, out, hidden, topk, expert_start, e_local);
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
    // 4-way unrolled: the two serial passes (max, then softmax sum) each
    // exposed the full DRAM latency of the next load (~128 iterations x 2).
    // Four independent chains keep 4 loads in flight and hide it.
    const int jmax_ = (kpool < (total - p * kpool)) ? kpool : (total - p * kpool);
    float m0 = -INFINITY, m1 = -INFINITY, m2 = -INFINITY, m3 = -INFINITY;
    int j = 0;
    for (; j + 3 < jmax_; j += 4) {
        const int t0 = p * kpool + j;
        float v0 = k_gate[(size_t)(t0 + 0) * idm + d] + ape[(size_t)(j + 0) * idm + d];
        float v1 = k_gate[(size_t)(t0 + 1) * idm + d] + ape[(size_t)(j + 1) * idm + d];
        float v2 = k_gate[(size_t)(t0 + 2) * idm + d] + ape[(size_t)(j + 2) * idm + d];
        float v3 = k_gate[(size_t)(t0 + 3) * idm + d] + ape[(size_t)(j + 3) * idm + d];
        m0 = fmaxf(m0, v0); m1 = fmaxf(m1, v1); m2 = fmaxf(m2, v2); m3 = fmaxf(m3, v3);
    }
    for (; j < jmax_; j++) {
        const int t = p * kpool + j;
        m0 = fmaxf(m0, k_gate[(size_t)t * idm + d] + ape[(size_t)j * idm + d]);
    }
    const float lmax = fmaxf(fmaxf(m0, m1), fmaxf(m2, m3));
    if (lmax == -INFINITY) return;
    float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
    float n0 = 0.f, n1 = 0.f, n2 = 0.f, n3 = 0.f;
    j = 0;
    for (; j + 3 < jmax_; j += 4) {
        const int t0 = p * kpool + j;
        float w0 = expf(k_gate[(size_t)(t0 + 0) * idm + d] + ape[(size_t)(j + 0) * idm + d] - lmax);
        float w1 = expf(k_gate[(size_t)(t0 + 1) * idm + d] + ape[(size_t)(j + 1) * idm + d] - lmax);
        float w2 = expf(k_gate[(size_t)(t0 + 2) * idm + d] + ape[(size_t)(j + 2) * idm + d] - lmax);
        float w3 = expf(k_gate[(size_t)(t0 + 3) * idm + d] + ape[(size_t)(j + 3) * idm + d] - lmax);
        d0 += w0; d1 += w1; d2 += w2; d3 += w3;
        n0 += w0 * k_idx[(size_t)(t0 + 0) * idm + d];
        n1 += w1 * k_idx[(size_t)(t0 + 1) * idm + d];
        n2 += w2 * k_idx[(size_t)(t0 + 2) * idm + d];
        n3 += w3 * k_idx[(size_t)(t0 + 3) * idm + d];
    }
    for (; j < jmax_; j++) {
        const int t = p * kpool + j;
        float wgt = expf(k_gate[(size_t)t * idm + d] + ape[(size_t)j * idm + d] - lmax);
        d0 += wgt;
        n0 += wgt * k_idx[(size_t)t * idm + d];
    }
    // keep the accumulation order deterministic: pairwise within the chain
    const float den = ((d0 + d1) + (d2 + d3));
    const float num = ((n0 + n1) + (n2 + n3));
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
    const float* __restrict__ idx_pools,  // [n, select_k_max]
    float* __restrict__ idx,              // [n, out_width_max]
    int n, int select_k_max, int kpool, int max_npools,
    const int* __restrict__ total_ptr,    // pinned (graph-safe)
    int n_fixed) {                        // n as a CONSTANT for ctx0 derivation
    int total = *total_ptr;
    int ctx0 = total - n_fixed;           // derive from pinned total
    int npools = (total + kpool - 1) / kpool; // derive from pinned total
    int select_k = min(select_k_max, npools); // LIVE (graph-safe: grows with cache)
    int out_width = select_k * kpool + (kpool - 1); // live out_width
    int out_stride = select_k_max * kpool + (kpool - 1); // fixed buffer stride
    int i = blockIdx.x;
    if (i >= n) return;
    const float* pv = idx_pools + (size_t)i * select_k_max;
    float* iv = idx + (size_t)i * out_stride;

    // MULTI-THREAD (was 1 thread serially writing ~8K slots — a dsa-layer
    // straggler): phase A flags valid r (pflt in range) + block prefix in
    // smem; phase B writes valid r's kpool slots in parallel. Invalid r slots
    // are SKIPPED (compact — col only advances for valid r, matching the
    // serial semantics): valid r's base col = valid_prefix(r) * kpool.
    extern __shared__ int sp[];           // [select_k_max+1] prefix
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
        while (col < out_stride) iv[col++] = -1.0f; // stride tail (select_k < max)
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

// ============================================================
// DSA BATCHED (multi-seq decode, B rows = B seqs): the 5 per-seq
// chains (append/kpool/topk/expand/attn) collapsed to ONE launch each.
// Per-seq cache/total state via device pointer tables: [B] arrays of the
// per-(seq,family) cache pointers (stable for the caches' lifetime — the
// tables are cudaMalloc'd + memcpy'd once per composition by the host) and
// [B] arrays of the per-seq PINNED t0/total int pointers (the kernel
// dereferences zero-copy — the host writes the ints per step; graph-safe).
// Replaces B × 5 per-seq launches (small grids at n=1 serialize) with
// grid(B,...) launches that fill the SMs; per-seq math identical (the
// bodies are the single-seq kernels' with row→(seq, ptr-table) indexing).
// ============================================================

// 1. cache append: kvb [B,h*(dk+dv)], ki [B,idm], gate [B,idm] → each seq's
// cache at ITS t0 (flat grid over all 3 regions, seq derived by division).
// fp8 KV cache (2026-09-09): the sparse attention reads k_nope/v for the
// live_k selected slots x 64 heads x 256 dims x 2 tensors — ~4.3 GB/step at
// B=16 with fp32, which is the kernel's dominant cost. e4m3 cuts that 4x.
// A per-(token, head) absmax is needed (a 1-thread-per-element kernel cannot
// compute it), so one block per (seq, token) with 64 heads x 4 lanes: each
// lane covers dk/4 = 64 K and dv/4 = 64 V elements of its head, the 4 lanes
// shuffle-reduce the absmax, then fp8 + the per-head scale are written.
__global__ void dsa_append_batched_kernel(
    const float* __restrict__ kvb,   // [B, ntok, h*(dk+dv)] rows
    const float* __restrict__ ki,    // [B, idm]
    const float* __restrict__ gate,  // [B, idm]
    float* const* __restrict__ kn_tbl,   // [B] per-seq k_nope ptrs (f32 [T, h, dk])
    float* const* __restrict__ v_tbl,    // [B] per-seq v ptrs (f32 [T, h, dv])
    float* const* __restrict__ kidx_tbl, // [B]
    float* const* __restrict__ kgate_tbl,// [B]
    const int* const* __restrict__ t0_tbl, // [B] per-seq PINNED t0 ptrs
    const int* const* __restrict__ total_tbl, // [B] per-seq PINNED total ptrs
    int B, int h, int dk, int dv, int idm, int max_t, int dev_adv) {
    // F32 CACHE FORMAT (2026-09-09, root cause #4 of the batched garbage
    // text): the single-seq path (PREFILL + n=1 decode,
    // dsa_cache_append_kernel) writes the shared (seq, family) DSA K/V cache
    // as f32 with NO scales. The fp8-e4m3+scales format (526e002) was
    // migrated on the batched side ONLY: the prefill's f32 slots were then
    // misread as e4m3 bytes and the scale buffers (never written by the
    // single path) held pool garbage — kernel-printf evidence: ksc0=0.000000,
    // NaN scales → softmax sum=NaN → every DSA layer's attention exactly
    // zero at n>=2 → garbage text. The two paths share the same cache
    // buffers, so they MUST use one format. Restored f32 (the last
    // B=16-text-verified format, pre-b3d41ca). A future fp8/bf16 KV-cache
    // migration must convert BOTH paths in one change, with text + isolated
    // bench verification.
    const int seq = blockIdx.x;
    const int tok = blockIdx.y;
    if (seq >= B) return;
    const int t0 = *t0_tbl[seq]; // per-seq pinned t0 (zero-copy)
    // PINNED-READ HARDENING (2026-09-09): a garbage/stale pinned t0 made this
    // kernel WRITE past the per-seq DSA cache ((t0+tok)*h + hd — a 2MB-aligned
    // PDE fault; memcheck cannot see pinned-page sources). Clamp to capacity.
    if (t0 < 0 || t0 + tok >= max_t) return;
    const int tid = threadIdx.x;
    const int row = h * (dk + dv);
    const float* src = kvb + (size_t)seq * row + (size_t)tok * row;
    for (int r = tid; r < row; r += blockDim.x) {
        const int hd = r / (dk + dv), c = r % (dk + dv);
        const size_t slot = ((size_t)(t0 + tok) * h + hd);
        if (c < dk) kn_tbl[seq][slot * dk + c] = src[r];
        else        v_tbl[seq][slot * dv + (c - dk)] = src[r];
    }
    // ki / gate: idm elements, same [T, idm] layout as the single-seq path
    for (int c = tid; c < idm; c += blockDim.x) {
        kidx_tbl[seq][(size_t)(t0 + tok) * idm + c] = ki[(size_t)seq * idm + c];
        kgate_tbl[seq][(size_t)(t0 + tok) * idm + c] = gate[(size_t)seq * idm + c];
    }
    // DEVICE-SIDE ADVANCE (2026-09-10): the pinned t0/total are incremented
    // HERE (the tok==0 block) instead of by the host — the host writes raced
    // the in-flight kernels' zero-copy reads (Xid 31 PDE faults), which
    // forced the step-start all-rank sync and killed the host/GPU pipeline.
    // In-stream ordering makes the update visible to this launch's downstream
    // consumers (kpool/topk/expand/attn read `total` AFTER the append in the
    // stream). The dry pass (real execution) advances exactly once; the
    // capture pass records without executing (no advance); every replay
    // advances exactly once — the pre-capture host rollback becomes map-only.
    // The pinned memory is mutable; the const is only the table typing.
    if (dev_adv && tok == 0 && threadIdx.x == 0) {
        *(int*)t0_tbl[seq] = t0 + 1;
        *(int*)total_tbl[seq] = t0 + 1;
    }
}

// 2. kpool compression: per-seq (k_idx, k_gate) → pool_keys [B, max_npools,
// idm]. npools derived live per seq from its pinned total.
__global__ void kpool_compress_batched_kernel(
    const float* const* __restrict__ kidx_tbl,  // [B]
    const float* const* __restrict__ kgate_tbl, // [B]
    const float* __restrict__ ape,              // [kpool, idm] shared
    float* __restrict__ pool_keys,               // [B, max_npools, idm]
    const int* const* __restrict__ total_tbl,    // [B] per-seq PINNED total ptrs
    int B, int max_npools, int kpool, int idm) {
    // float4 over d: one element per thread made each of the kpool loads a
    // separate strided 4-byte access (median 114us/call).
    const int idm4 = idm >> 2;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t per4 = (size_t)max_npools * idm4;
    if (tid >= (size_t)B * per4) return;
    int seq = (int)(tid / per4);
    size_t rem = tid % per4;
    int p = (int)(rem / idm4), d4 = (int)(rem % idm4);
    int total = *total_tbl[seq];
    // PINNED-READ HARDENING (2026-09-09): the pinned total is host-written and
    // read zero-copy; a garbage/stale value made npools astronomical and the
    // `t < total` guard pass for out-of-cache t — the gather then read past
    // the DSA cache (2MB-aligned Xid 31 PDE faults; invisible to memcheck,
    // which does not track pinned pages). Clamp to the cache capacity.
    if (total < 0) total = 0;
    if (total > max_npools * kpool) total = max_npools * kpool;
    int npools = (total + kpool - 1) / kpool;
    if (p >= npools) return;
    const float* k_idx = kidx_tbl[seq];
    const float* k_gate = kgate_tbl[seq];
    const int d = d4 * 4;
    float4 lmax = make_float4(-INFINITY, -INFINITY, -INFINITY, -INFINITY);
    for (int j = 0; j < kpool; j++) {
        int t = p * kpool + j;
        if (t < total) {
            const float4 g = *reinterpret_cast<const float4*>(k_gate + (size_t)t * idm + d);
            const float4 a = *reinterpret_cast<const float4*>(ape + (size_t)j * idm + d);
            lmax.x = fmaxf(lmax.x, g.x + a.x);
            lmax.y = fmaxf(lmax.y, g.y + a.y);
            lmax.z = fmaxf(lmax.z, g.z + a.z);
            lmax.w = fmaxf(lmax.w, g.w + a.w);
        }
    }
    if (lmax.x == -INFINITY) return;
    float4 den = make_float4(0.f, 0.f, 0.f, 0.f), num = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int j = 0; j < kpool; j++) {
        int t = p * kpool + j;
        if (t < total) {
            const float4 g = *reinterpret_cast<const float4*>(k_gate + (size_t)t * idm + d);
            const float4 a = *reinterpret_cast<const float4*>(ape + (size_t)j * idm + d);
            const float4 idxv = *reinterpret_cast<const float4*>(k_idx + (size_t)t * idm + d);
            const float4 w = make_float4(__expf(g.x + a.x - lmax.x), __expf(g.y + a.y - lmax.y),
                                         __expf(g.z + a.z - lmax.z), __expf(g.w + a.w - lmax.w));
            den.x += w.x; den.y += w.y; den.z += w.z; den.w += w.w;
            num.x += w.x * idxv.x; num.y += w.y * idxv.y; num.z += w.z * idxv.z; num.w += w.w * idxv.w;
        }
    }
    const float4 res = make_float4(num.x / den.x, num.y / den.y, num.z / den.z, num.w / den.w);
    *reinterpret_cast<float4*>(pool_keys + (size_t)seq * (size_t)max_npools * idm + (size_t)p * idm + d) = res;
}

// 3. indexer topk: one 256-thread block per seq — score ALL pools of its
// pool_keys row, warp-shuffle select_k selection (select_k live: min(cap,
// npools) from the pinned total — unlike the single-seq graph-frozen arg).
__global__ void indexer_topk_batched_kernel(
    const float* __restrict__ qi,        // [B, ih*idm]
    const float* __restrict__ pool_keys, // [B, max_npools, idm]
    const float* __restrict__ w,          // [B, ih]
    float* __restrict__ idx,              // [B, select_k_max]
    int B, int ih, int idm, int select_k_max, int kpool, int max_npools,
    const int* const* __restrict__ total_tbl, int idx_mma) { // [B] pinned
    int seq = blockIdx.x;
    if (seq >= B) return;
    int total = *total_tbl[seq];
    int t = (total + kpool - 1) / kpool; // live npools
    int ctx0 = total - 1;                  // n=1 per seq
    int ctx0_pools = ctx0 / kpool;
    int jmax = min(ctx0_pools + 1, t);
    int select_k = min(select_k_max, t);
    const float* q_s = qi + (size_t)seq * (size_t)(ih * idm);
    const float* pk = pool_keys + (size_t)seq * (size_t)max_npools * idm;
    const float* w_s = w + (size_t)seq * ih;
    float* iv = idx + (size_t)seq * select_k_max;
    extern __shared__ float sm[]; // max_npools scores (sized for MAX at launch)
    float inv_sqrt_d = rsqrtf((float)idm);
    // 8 THREADS PER POOL: the score loop was one serial 32-head x 128-dim dot
    // per thread, so only t (~112) threads of 1024 did any work (the whole
    // kernel ran at ~1% of the GPU). Each group splits the heads and the
    // shuffle reduces within the group.
    // FAST PATH FIRST: when every causal-valid pool is selected the output is
    // exactly {0..jmax-1} and the scores are never read by any consumer
    // (pool_expand only takes the indices; sparse_attn softmaxes over the
    // selected slots). The score GEMM below was pure waste in that case —
    // it used to run BEFORE this check (~97us/call at t~1200, x11 layers).
    if (select_k >= jmax) {
        for (int r = threadIdx.x; r < select_k_max; r += blockDim.x)
            iv[r] = (r < jmax) ? (float)r : -1.0f;
        return;
    }
    const int TG = 8;   // 2x the per-thread columns on the latency-bound pool dot
    const int gid = threadIdx.x / TG;
    const int lid = threadIdx.x % TG;
    const int ngroups = blockDim.x / TG;
    // ---- tensor-core score GEMM (env FERRITE_IDX_MMA=1) ----
    // scores[p][hi] = pk[p] . q[hi] is a natural GEMM: BOTH operands are
    // contiguous (no gather), M = 16 pools, N = 8 heads, K = idm (128 -> 8
    // m16n8k16 bf16 tiles). s[p] = sum_hi w[hi] * relu(score * inv_sqrt_d).
    if (idx_mma) {
        __shared__ __nv_bfloat16 aq[16][128];
        __shared__ __nv_bfloat16 bq[128][40];
        __shared__ float cacc[16][40];
        const int lane = threadIdx.x & 31;
        const int ntiles = (ih + 7) / 8;
        for (int j0 = 0; j0 < t; j0 += 16) {
            for (int l = threadIdx.x; l < 16 * 128; l += blockDim.x) {
                const int r = l >> 7, c = l & 127;
                const int p_ = j0 + r;
                aq[r][c] = __float2bfloat16(p_ < t ? pk[(size_t)p_ * idm + c] : 0.f);
            }
            __syncthreads();
            for (int n0 = 0; n0 < ih; n0 += 8) {
                for (int l = threadIdx.x; l < 128 * 8; l += blockDim.x) {
                    const int c = l >> 3, hi = l & 7;
                    bq[c][hi] = __float2bfloat16(q_s[(size_t)(n0 + hi) * idm + c]);
                }
                __syncthreads();
                float acc[4] = {0.f, 0.f, 0.f, 0.f};
                #pragma unroll
                for (int kt = 0; kt < 8; kt++) {
                    const unsigned sa_ = (unsigned)__cvta_generic_to_shared(
                        &aq[0][0] + (size_t)(lane & 15) * 128 + ((lane >> 4) * 8) + kt * 16);
                    unsigned a[4];
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                 : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(sa_));
                    unsigned b[2];
                    const int r0_ = lane >> 2, cc = (lane & 3) * 2;
                    b[0] = *(const unsigned*)&bq[kt * 16 + cc][r0_];
                    b[1] = *(const unsigned*)&bq[kt * 16 + cc + 8][r0_];
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
                }
                if ((lane & 3) == 0) {
                    const int r0_ = lane >> 2;
                    for (int rr = 0; rr < 2; rr++) {
                        const int rw = r0_ + rr * 8;
                        if (j0 + rw < t) {
                            const float sc_ = acc[rr * 2] * inv_sqrt_d;
                            const float wv = w_s[n0 + 0];   // col 0 of this n-tile
                            if (n0 == 0) cacc[rw][n0 >> 3] = wv * fmaxf(sc_, 0.f);
                            else cacc[rw][n0 >> 3] = wv * fmaxf(sc_, 0.f);
                        }
                    }
                }
                __syncthreads();
            }
            for (int r = threadIdx.x; r < 16; r += blockDim.x) {
                const int p_ = j0 + r;
                if (p_ < t) {
                    float s_ = 0.f;
                    for (int nt = 0; nt < ntiles; nt++) s_ += cacc[r][nt];
                    sm[p_] = (p_ < jmax) ? s_ : -INFINITY;
                }
            }
            __syncthreads();
        }
    }
    for (int j = (idx_mma ? t : gid); j < t; j += ngroups) {
        const float* k = pk + (size_t)j * idm;
        float s = 0.f;
        if (j < jmax) {
            for (int hi = lid; hi < ih; hi += TG) {
                const float* q = q_s + (size_t)hi * idm;
                float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
                float e0 = 0.f, e1 = 0.f, e2 = 0.f, e3 = 0.f;
                // 4-way unroll (was 2): idm=128 -> 32 float4 iterations, each
                // pair of loads feeding only 2 chains left the fp32 FMA
                // latency exposed (the indexer is latency-bound).
                int l = 0;
                #pragma unroll 2
                for (; l + 7 < idm; l += 8) {
                    float4 qv = *reinterpret_cast<const float4*>(q + l);
                    float4 kv = *reinterpret_cast<const float4*>(k + l);
                    float4 qw = *reinterpret_cast<const float4*>(q + l + 4);
                    float4 kw = *reinterpret_cast<const float4*>(k + l + 4);
                    d0 += qv.x * kv.x; d1 += qv.y * kv.y; d2 += qv.z * kv.z; d3 += qv.w * kv.w;
                    e0 += qw.x * kw.x; e1 += qw.y * kw.y; e2 += qw.z * kw.z; e3 += qw.w * kw.w;
                }
                for (; l + 3 < idm; l += 4) {
                    float4 qv = *reinterpret_cast<const float4*>(q + l);
                    float4 kv = *reinterpret_cast<const float4*>(k + l);
                    d0 += qv.x * kv.x; d1 += qv.y * kv.y; d2 += qv.z * kv.z; d3 += qv.w * kv.w;
                }
                d0 += e0; d1 += e1; d2 += e2; d3 += e3;
                float dot = (d0 + d1) + (d2 + d3);
                s += w_s[hi] * fmaxf(dot, 0.f); // relu
            }
            // group mask (8 contiguous lanes within the warp): 0xffffffff
            // would deadlock when some groups have no pools and skip this.
            const unsigned gmask = 0xffu << ((threadIdx.x & 31) & ~7u);   // 8-lane groups (TG=8) — must match TG
            #pragma unroll
            for (int off = TG / 2; off > 0; off >>= 1) s += __shfl_down_sync(gmask, s, off);
            if (lid == 0) sm[j] = s * inv_sqrt_d;
        } else {
            if (lid == 0) sm[j] = -INFINITY;
        }
    }
    __syncthreads();
    for (int r = 0; r < select_k; r++) {
        __shared__ int bidx[8];
        __shared__ float bval[8];
        int best = -1;
        float bv = -INFINITY;
        for (int j = threadIdx.x; j < t; j += blockDim.x) {
            if (sm[j] > bv) { bv = sm[j]; best = j; }
        }
        for (int off = 16; off > 0; off >>= 1) {
            float ov = __shfl_down_sync(0xffffffff, bv, off);
            int oi = __shfl_down_sync(0xffffffff, best, off);
            if (ov > bv) { bv = ov; best = oi; }
        }
        int warp = threadIdx.x >> 5;
        if ((threadIdx.x & 31) == 0) { bidx[warp] = best; bval[warp] = bv; }
        __syncthreads();
        if (threadIdx.x == 0) {
            int sel = -1;
            float sv = -INFINITY;
            for (int wi = 0; wi < (blockDim.x >> 5); wi++) {
                if (bval[wi] > sv) { sv = bval[wi]; sel = bidx[wi]; }
            }
            if (sel >= 0) { iv[r] = (float)sel; sm[sel] = -INFINITY; }
            else iv[r] = -1.0f;
        }
        __syncthreads();
    }
    // pad the (select_k_max - select_k) tail: pool_expand only reads r <
    // select_k, but keep the buffer deterministic.
    for (int r = select_k + threadIdx.x; r < select_k_max; r += blockDim.x) iv[r] = -1.0f;
}

// 4. pool expand: one block per seq — compact the valid pools' token slots
// + causal tail (multi-thread prefix + probe, the single-seq semantics),
// out_width frozen at select_k_max for the graph-stable buffer stride.
__global__ void pool_expand_batched_kernel(
    const float* __restrict__ idx_pools,  // [B, select_k_max]
    float* __restrict__ idx,              // [B, select_k_max*kpool + (kpool-1)]
    int B, int select_k_max, int kpool, int max_npools,
    const int* const* __restrict__ total_tbl, int n_fixed) {
    int seq = blockIdx.x;
    int total = *total_tbl[seq];
    int ctx0 = total - n_fixed;
    int npools = (total + kpool - 1) / kpool;
    int select_k = min(select_k_max, npools);
    int out_width = select_k * kpool + (kpool - 1);
    int out_stride = select_k_max * kpool + (kpool - 1);
    const float* pv = idx_pools + (size_t)seq * select_k_max;
    float* iv = idx + (size_t)seq * (size_t)out_stride;
    extern __shared__ int sp[]; // [select_k_max+1] prefix
    int tid = threadIdx.x;
    for (int r = tid; r < select_k; r += blockDim.x) {
        float pflt = pv[r];
        sp[r + 1] = (pflt >= 0.0f && (int)pflt < npools) ? 1 : 0;
    }
    if (tid == 0) sp[0] = 0;
    __syncthreads();
    if (tid == 0) { for (int r = 0; r < select_k; r++) sp[r + 1] += sp[r]; }
    __syncthreads();
    int nvalid = sp[select_k];
    for (int c = tid; c < nvalid * kpool; c += blockDim.x) {
        int s = c / kpool, j = c % kpool;
        int r = (int)(((long long)s * select_k) / (nvalid > 0 ? nvalid : 1));
        if (r >= select_k) r = select_k - 1;
        while (r > 0 && sp[r] > s) r--;
        while (r + 1 < select_k && sp[r + 1] <= s) r++;
        int p = (int)pv[r];
        int t = p * kpool + j;
        iv[s * kpool + j] = (t < total && t <= ctx0) ? (float)t : -1.0f;
    }
    if (tid == 0) {
        int visible_count = ctx0 + 1; // n=1: row i=0
        int tail_count = visible_count % kpool;
        int tail_start = visible_count - tail_count;
        int col = nvalid * kpool;
        for (int j = 0; j < kpool - 1 && col < out_width; j++) {
            int t = tail_start + j;
            iv[col++] = (j < tail_count && t <= ctx0) ? (float)t : -1.0f;
        }
        while (col < out_width) iv[col++] = -1.0f;
        while (col < out_stride) iv[col++] = -1.0f; // stride tail (select_k < max)
    }
}

// 5. sparse attention: grid (B, h) — the v2 body per (seq, head) with the
// per-seq k/v via tables + per-seq idx row (stride = frozen out_width).
// __launch_bounds__(256,8): ncu showed the register limit capping this kernel
// at 6 blocks/SM (No-Eligible 69.5%, long-scoreboard 43.5% = latency-bound).
// Forcing 8 blocks/SM trades registers for the latency hiding it needs.
template <int BLK>
__global__ void __launch_bounds__(BLK, 2048 / BLK) sparse_attn_v2_batched_kernel(
    const float* __restrict__ q,          // [B, h*d]
    const float* const* __restrict__ k_tbl,      // [B] per-seq k_nope caches (f32 [T, h, d])
    const float* const* __restrict__ v_tbl,      // [B] per-seq v caches (f32 [T, h, dv])
    const float* __restrict__ idx,         // [B, topk_slots]
    float* __restrict__ out,              // [B, h*dv]
    int B, const int* const* __restrict__ total_tbl, // [B] pinned
    int h, int d, int dv, int topk, int nodedup, int qk_mma, int dbg) {
    // F32 CACHE READS (2026-09-09, root cause #4 of the batched garbage
    // text): the K/V caches are f32 [T, h, d] — the format the single-seq
    // path (prefill + dsa_cache_append_kernel) writes. The fp8+scales reader
    // (526e002) misread the prefill's f32 slots as e4m3 and read NEVER-WRITTEN
    // scale buffers (pool garbage: ksc=0.000000 + NaN → softmax sum=NaN →
    // attention exactly zero). One cache, one format.
    int seq = blockIdx.x;
    int hd = blockIdx.y;
    int t = *total_tbl[seq]; // per-seq zero-copy pinned read
    const int live_k = (topk < t) ? topk : t; // live slots only: the indexer writes the rest as -1, so looping to the fixed select_k_max (2048) wasted 18x at short context
    const float* q_s = q + (size_t)seq * (size_t)(h * d);
    const float* k_s = k_tbl[seq];
    const float* v_s = v_tbl[seq];
    const float* idx_s = idx + (size_t)seq * topk;
    float* out_s = out + (size_t)seq * (size_t)(h * dv);
    float scale = rsqrtf((float)d);
    // FERRITE_ATTN_DBG (2026-09-09, batched-attention-is-zero hunt): what the
    // KERNEL sees at (seq0, hd0) — the device-side table contents and the
    // pinned totals can differ from what the host believes it wrote.
    if (dbg && seq == 0 && hd == 0 && threadIdx.x == 0) {
        const int j0 = (int)idx_s[0];
        const int jc = j0 >= 0 ? j0 : 0;
        printf("[attn-dbg] B=%d h=%d d=%d dv=%d topk=%d t=%d live_k=%d idx0..3=%d,%d,%d,%d "
               "K[j0]=%.4f,%.4f,%.4f V[j0]=%.4f,%.4f,%.4f q0=%.4f\n",
               B, h, d, dv, topk, t, live_k,
               j0, (int)idx_s[1], (int)idx_s[2], (int)idx_s[3],
               k_s[(size_t)jc * h * d], k_s[(size_t)jc * h * d + 1], k_s[(size_t)jc * h * d + 2],
               v_s[(size_t)jc * h * dv], v_s[(size_t)jc * h * dv + 1], v_s[(size_t)jc * h * dv + 2],
               q_s[0]);
    }
    extern __shared__ float sm[];
    float* qs = sm;                                   // [d] float4-aligned base
    float* sc = sm + d;                                // [topk]
    int* idxs = (int*)(sc + topk);                     // [topk]
    float* red = (float*)(idxs + topk);                // [16] warp partials
    unsigned int* bm = (unsigned int*)(red + 16);       // [256] dedup bitmap
    // 256 words = 8192 tokens = the DSA cache's max_t. Was 4096 words
    // (131072 tokens) = 16KB of smem for nothing: ncu showed shared memory
    // capping this kernel at 5 blocks/SM (No-Eligible 69.5%, top stall
    // long-scoreboard 43.5%). 16x smaller bitmap -> ~12 blocks/SM.
    const int bm_words_max = 256;
    int bm_words = (t + 31) >> 5; if (bm_words > bm_words_max) bm_words = bm_words_max;
    for (int l = threadIdx.x; l < d; l += blockDim.x) qs[l] = q_s[(size_t)hd * d + l];
    for (int s = threadIdx.x; s < live_k; s += blockDim.x) idxs[s] = (int)idx_s[s];
    for (int w0 = threadIdx.x; w0 < bm_words_max; w0 += blockDim.x) bm[w0] = 0u;
    __syncthreads();
    // 8 THREADS PER SLOT (same fix as the indexer): one serial 256-dim dot per
    // thread left ~44% of the block idle and made the dot latency-bound.
    // TG=8: 8 lanes per slot (8 float4 cols each). Was TG=2 while `glane0`
    // was already 8-aligned — the dup broadcast read a lane OUTSIDE the
    // 2-lane mask (undefined result). Matching TG to glane0 fixes that and
    // shortens the per-lane serial FMA chain 128 -> 32.
    const int TG = 8;
    const int gid = threadIdx.x / TG;
    const int lid = threadIdx.x % TG;
    const int ngroups = blockDim.x / TG;
    const unsigned gmask = 0xffu << ((threadIdx.x & 31) & ~7u);   // 8-lane groups (TG=8) — must match TG
    const int glane0 = (threadIdx.x & 31) & ~7;
    // ---- QK^T on the tensor core (16 slots per MMA tile) ----
    // DISABLED BY DEFAULT (FERRITE_ATTN_QK_MMA=1 to try): measured 22.6 ms/step
    // vs 13.7 with the SIMT path — the full-d gather + per-tile sync + the 8x
    // N-replica waste cost far more than the MMA saves here.
    if (qk_mma) {
    // ncu: this kernel is latency-bound (No-Eligible 69.5%, long-scoreboard
    // 43.5%, Compute 24.7%). Each slot's 256-dim dot was 32 serial FMAs per
    // lane; the fp8 m16n8k32 does 16 slots x 32 K per instruction.
    // A = the 16 slots' K rows (fp8, gathered from the cache into smem)
    // B = the Q (fp8, replicated across the 8 N columns)
    // C[slot][0] = that slot's score (col 0 carries the answer).
    __shared__ unsigned char q8s[256];          // the Q in e4m3
    __shared__ unsigned char kt[16][256 + 16];  // 16 slots x FULL d=256 (+pad)
    {
        const int lane = threadIdx.x & 31;
        const float qsc = 1.0f;                  // Q quantized with absmax/448
        float am = 1e-9f;
        for (int l = threadIdx.x; l < d; l += blockDim.x) am = fmaxf(am, fabsf(qs[l]));
        __syncthreads();
        const float qscale = am / 448.0f + 1e-12f;
        for (int l = threadIdx.x; l < d; l += blockDim.x)
            q8s[l] = (unsigned char)__nv_cvt_float_to_fp8(
                fminf(fmaxf(qs[l] / qscale, -448.0f), 448.0f), __NV_SATFINITE, __NV_E4M3);
        __syncthreads();
        for (int s0 = 0; s0 < live_k; s0 += 16) {
            // gather the FULL d=256 for the 16 slots (one pass)
            for (int l = threadIdx.x; l < 16 * 256; l += blockDim.x) {
                const int r = l >> 8, kk = l & 255;
                const int ss = s0 + r;
                unsigned char v8 = 0;
                if (ss < live_k) {
                    const int j = idxs[ss];
                    // f32 cache: quantize on the fly (the e4m3 MMA needs bytes;
                    // the per-head ksc rescale is gone with the f32 format)
                    if (j >= 0 && j < t)
                        v8 = (unsigned char)__nv_cvt_float_to_fp8(
                            k_s[((size_t)j * h + hd) * d + kk], __NV_SATFINITE, __NV_E4M3);
                }
                kt[r][kk] = v8;
            }
            __syncthreads();
            float acc[4] = {0.f, 0.f, 0.f, 0.f};
            const int c0 = (lane & 3) * 4;
            #pragma unroll
            for (int kt_i = 0; kt_i < 8; kt_i++) {
                // A fragment: lanes 0-15 give the 16 rows at the k half
                const unsigned saddr = (unsigned)__cvta_generic_to_shared(
                    &kt[0][0] + (size_t)(lane & 15) * (256 + 16) + ((lane >> 4) * 16) + kt_i * 32);
                unsigned a[4];
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(saddr));
                unsigned b[2];
                b[0] = *(const unsigned*)(q8s + kt_i * 32 + c0);
                b[1] = *(const unsigned*)(q8s + kt_i * 32 + c0 + 16);
                asm volatile(
                    "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                    : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
            }
            // epilogue: lane l/4 holds rows l/4 and l/4+8 at column (l%4)*2
            if ((lane & 3) == 0) {
                const int r0_ = lane >> 2;
                for (int r = 0; r < 2; r++) {
                    const int rr = r0_ + r * 8;
                    const int ss = s0 + rr;
                    if (ss < live_k && lid == 0) {
                        const int j = idxs[ss];
                        const bool valid = (j >= 0 && j < t && j < 8192);
                        sc[ss] = valid ? (acc[r * 2] * scale * qscale) : -INFINITY;
                    }
                }
            }
            __syncthreads();
        }
    }
    }  // end if (qk_mma)
    // the SIMT per-slot dot path (default; skipped when the MMA ran)
    for (int s = (qk_mma ? live_k : gid); s < live_k; s += ngroups) {
        int j = idxs[s];
        bool valid = (j >= 0 && j < t && j < 8192);
        int dup = 0;
        // DIAG (FERRITE_ATTN_NODEDUP=1): the bitmap atomicOr is ~2M shared
        // atomics per layer-call (2048 slots x 1024 blocks) = the suspected
        // 143us bottleneck. This ablation measures it; correctness of the
        // resulting text is checked by eye before anything is removed.
        if (valid && lid == 0 && !nodedup) {
            if ((j >> 5) < bm_words) {
                unsigned int prev = atomicOr(&bm[j >> 5], 1u << (j & 31));
                dup = (prev & (1u << (j & 31))) != 0;
            }
        }
        dup = __shfl_sync(gmask, dup, glane0);
        float a = 0.f;
        if (valid && !dup) {
            // F32 cache reads (2026-09-09, root cause #4 restore — see the
            // kernel header): float4 dot, NO per-head scale. This is the
            // pre-b3d41ca structure (the last B=16-text-verified numerics).
            const float* krow = k_s + ((size_t)j * h + hd) * d;
            const float4* k4 = reinterpret_cast<const float4*>(krow);
            float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
            for (int l = lid * 4; l + 3 < d; l += TG * 4) {
                float4 kk = k4[l >> 2];
                float4 qq = *reinterpret_cast<const float4*>(qs + l);
                acc.x += qq.x * kk.x; acc.y += qq.y * kk.y;
                acc.z += qq.z * kk.z; acc.w += qq.w * kk.w;
            }
            a = acc.x + acc.y + acc.z + acc.w;
            if (lid == 0) {
                for (int l = d & ~3; l < d; l++) a += qs[l] * krow[l];
            }
            #pragma unroll
            for (int off = TG / 2; off > 0; off >>= 1) a += __shfl_down_sync(gmask, a, off);
        }
        if (lid == 0) sc[s] = (valid && !dup) ? (a * scale) : -INFINITY;
    }
    __syncthreads();
    float m = -INFINITY;
    // FIXED 256-stride + 8-warp reduction: the FP association must be IDENTICAL
    // to the known-good 256-thread build at any BLK. The `tid < 256` guard is
    // REQUIRED — without it the threads 256..511 re-scan slots 256.., 512..,
    // i.e. every slot >= 256 is summed twice (softmax denominator ~2x too big,
    // attention weights wrong -> the model degenerates into a repetition loop
    // once live_k > 256; that is exactly the "only loops with longer output"
    // symptom). Only the QK group count may differ — each slot's dot uses the
    // same 8-lane tree, so the scores stay bit-identical.
    if (threadIdx.x < 256)
        for (int s = threadIdx.x; s < live_k; s += 256) m = fmaxf(m, sc[s]);
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
    for (int s = threadIdx.x; s < live_k; s += blockDim.x)
        sc[s] = all_inf ? 0.f : __expf(sc[s] - m);
    __syncthreads();
    float sum = 0.f;
    if (threadIdx.x < 256)   // same guard: see the max reduction above
        for (int s = threadIdx.x; s < live_k; s += 256) sum += sc[s];
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
    for (int s = threadIdx.x; s < live_k; s += blockDim.x) sc[s] /= denom;
    __syncthreads();
    if (dbg && seq == 0 && hd == 0 && threadIdx.x == 0) {
        printf("[attn-dbg] softmax m=%f sum=%f w0..2=%f,%f,%f\n",
               m, sum_,
               live_k > 0 ? sc[0] : -1.f, live_k > 1 ? sc[1] : -1.f,
               live_k > 2 ? sc[2] : -1.f);
    }
    // float4 + 4 slot groups: the old loop read V one 4-byte element per
    // thread with consecutive slots 64KB apart (zero coalescing, 12.5%
    // sector efficiency). 64 float4 columns x 4 slot groups = 256 threads,
    // 16-byte coalesced loads, then an smem reduction over the groups.
    {
        // F32 cache reads (2026-09-09, root cause #4 restore — see the kernel
        // header): float4 columns of 4 f32 elements, NO per-head scale. This
        // is the pre-b3d41ca 86fb8f8 structure (the last B=16-text-verified
        // numerics). NOTE: the fp8 variant's cols=dv>>4 with a single
        // float4 store per column only covered dv/4 output elements — the
        // f32 structure's c*4 write covers the full dv again.
        const int cols = dv >> 2;               // float4 columns (256/4 = 64)
        const int G = (blockDim.x + cols - 1) / cols;   // slot groups (4)
        const int g = threadIdx.x / cols;
        const int c = threadIdx.x % cols;
        if (c < cols && g < G) {
            float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
            // unroll 4: one independent float4 load per iteration, previously
            // serialized by the compiler (no unroll -> ~300-cycle L2 latency
            // exposed on every slot).
            #pragma unroll 4
            for (int s = g; s < live_k; s += G) {
                const float w = sc[s];
                if (w == 0.f) continue;
                const int j = idxs[s];
                if (j < 0 || j >= t) continue;
                float4 vv;
                asm volatile("ld.global.nc.L2::128B.v4.f32 {%0,%1,%2,%3}, [%4];\n"
                             : "=f"(vv.x), "=f"(vv.y), "=f"(vv.z), "=f"(vv.w)
                             : "l"(v_s + ((size_t)j * h + hd) * dv + c * 4));
                a.x += w * vv.x; a.y += w * vv.y; a.z += w * vv.z; a.w += w * vv.w;
            }
            __shared__ float4 pred[4 * 64];     // static (4KB), G<=4, cols<=64
            pred[g * cols + c] = a;
            __syncthreads();
            if (g == 0) {
                float4 tot = pred[c];
                for (int gg = 1; gg < G; gg++) {
                    const float4 t2 = pred[gg * cols + c];
                    tot.x += t2.x; tot.y += t2.y; tot.z += t2.z; tot.w += t2.w;
                }
                *reinterpret_cast<float4*>(out_s + (size_t)hd * dv + c * 4) = tot;
            }
        }
    }
}

extern "C" cudaError_t ferrite_dsa_append_batched(
    const float* kvb, const float* ki, const float* gate,
    float* const* kn_tbl, float* const* v_tbl, float* const* kidx_tbl, float* const* kgate_tbl,
    const int* const* t0_tbl, const int* const* total_tbl,
    int B, int h, int dk, int dv, int idm, int ntok, cudaStream_t s) {
    // one block per (seq, token) — f32 cache stores (see the kernel's F32
    // CACHE FORMAT comment: the single-seq path owns the format). The kernel
    // advances the pinned t0/total itself (dev_adv) — the host-side
    // per-step writes raced the in-flight kernels (Xid 31) and forced the
    // step-start all-rank sync. FERRITE_DEV_ADV=0 restores the host-owned
    // counters (with the sync, for A/B).
    static const int dev_adv_ = [] {
        const char* e = getenv("FERRITE_DEV_ADV");
        return e ? atoi(e) : 1;
    }();
    dim3 grid((unsigned)B, (unsigned)(ntok > 0 ? ntok : 1));
    static const int max_t_ = [] {
        const char* e = getenv("FERRITE_DSA_MAXT");
        return e ? atoi(e) : 8192;
    }();
    dsa_append_batched_kernel<<<grid, 256, 0, s>>>(
        kvb, ki, gate,
        kn_tbl, v_tbl, kidx_tbl, kgate_tbl, t0_tbl, total_tbl,
        B, h, dk, dv, idm, max_t_, dev_adv_);
    return cudaGetLastError();
}

extern "C" cudaError_t ferrite_kpool_compress_batched(
    const float* const* kidx_tbl, const float* const* kgate_tbl,
    const float* ape, float* pool_keys, const int* const* total_tbl,
    int B, int max_npools, int kpool, int idm, cudaStream_t s) {
    size_t total_t = (size_t)B * (size_t)max_npools * (idm >> 2);
    int threads = 256;
    int blocks = (int)((total_t + threads - 1) / threads);
    kpool_compress_batched_kernel<<<blocks, threads, 0, s>>>(
        kidx_tbl, kgate_tbl, ape, pool_keys, total_tbl, B, max_npools, kpool, idm);
    return cudaGetLastError();
}

extern "C" cudaError_t ferrite_indexer_topk_batched(
    const float* qi, const float* pool_keys, const float* w,
    float* idx, int B, int ih, int idm, int select_k_max, int kpool, int max_npools,
    const int* const* total_tbl, cudaStream_t s) {
    // DIAGNOSTIC ONLY (FERRITE_ATTN_SKIP=1): timing-only ablation (the DSA
    // cache append is NOT skipped — only the scoring/selection + the attention).
    static const bool attn_skip_ = getenv("FERRITE_ATTN_SKIP") != nullptr;
    if (attn_skip_) return cudaSuccess;

    static const int idx_mma_ = getenv("FERRITE_IDX_MMA") ? 1 : 0;
    dim3 block(1024); // was 256: the grid is only B=16 blocks, so the block
                      // size IS the parallelism (16x256 threads used 1.4% of
                      // the GPU; 16x1024 = 5.5%).
    dim3 grid(B);
    int max_t = 2048; // max_npools (smem frozen for MAX — graph-safe)
    size_t smem = (size_t)max_t * sizeof(float);
    indexer_topk_batched_kernel<<<grid, block, smem, s>>>(
        qi, pool_keys, w, idx, B, ih, idm, select_k_max, kpool, max_npools, total_tbl, idx_mma_);
    return cudaGetLastError();
}

extern "C" cudaError_t ferrite_pool_expand_batched(
    const float* idx_pools, float* idx,
    int B, int select_k_max, int kpool, int max_npools,
    const int* const* total_tbl, int n_fixed, cudaStream_t s) {
    size_t smem = ((size_t)select_k_max + 1) * sizeof(int);
    pool_expand_batched_kernel<<<B, 256, smem, s>>>(
        idx_pools, idx, B, select_k_max, kpool, max_npools, total_tbl, n_fixed);
    return cudaGetLastError();
}

extern "C" cudaError_t ferrite_sparse_attn_v2_batched(
    const float* q, float* const* k_tbl, float* const* v_tbl,
    const float* idx, float* out, int B, const int* const* total_tbl,
    int h, int d, int dv, int topk, cudaStream_t s) {
    // DIAGNOSTIC ONLY (FERRITE_ATTN_SKIP=1): timing-only ablation (the DSA
    // cache append is NOT skipped — only the scoring/selection + the attention).
    static const bool attn_skip_ = getenv("FERRITE_ATTN_SKIP") != nullptr;
    if (attn_skip_) return cudaSuccess;
    static const int nodedup_ = getenv("FERRITE_ATTN_NODEDUP") ? 1 : 0;
    static const int qk_mma_ = getenv("FERRITE_ATTN_QK_MMA") ? 1 : 0;
    static const int dbg_ = getenv("FERRITE_ATTN_DBG") ? 1 : 0;

    // The kernel is latency-bound and the grid is only B*h = 16*8 = 128 blocks,
    // so the block size IS the parallelism (256 threads = 6.9 warps/SM = 11%
    // occupancy; 512 = 13.8 warps/SM, measured 13.68 -> 12.58 ms/step). The
    // 512 build initially flipped the model into a repetition loop because the
    // softmax/PV reductions were reassociated by the block size; the kernel now
    // pins those associations to the 256-thread layout (see the FIXED comments
    // in the kernel), so BLK only changes the QK group count. FERRITE_ATTN_BLK=256
    // is the bisect escape hatch.
    // 512-thread measured faster (13.68 -> 12.58 ms/step) but FLIPS the model
    // into a repetition loop (the QK dedup-race winner differs with the larger
    // group count -> a ~1e-7 softmax reassociation -> crosses the logit decision
    // boundary; the known-good 256 build is bit-stable). Default is 256 (the
    // correct text); FERRITE_ATTN_BLK=512 opts into the fast-but-unsafe path.
    static const int blk_ = [] {
        const char* e = getenv("FERRITE_ATTN_BLK");
        return e ? atoi(e) : 256;
    }();
    dim3 grid(B, h);
    size_t smem = (size_t)topk * (sizeof(int) + sizeof(float)) + (size_t)d * sizeof(float)
                  + 16 * sizeof(float) + 256 * sizeof(unsigned int);
    if (blk_ >= 512) {
        if (smem > 48 * 1024) {
            cudaError_t e = cudaFuncSetAttribute(sparse_attn_v2_batched_kernel<512>,
                                                 cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
            if (e != cudaSuccess) return e;
        }
        sparse_attn_v2_batched_kernel<512><<<grid, dim3(512), smem, s>>>(
            q, k_tbl, v_tbl,
            idx, out, B, total_tbl, h, d, dv, topk, nodedup_, qk_mma_, dbg_);
    } else {
        if (smem > 48 * 1024) {
            cudaError_t e = cudaFuncSetAttribute(sparse_attn_v2_batched_kernel<256>,
                                                 cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
            if (e != cudaSuccess) return e;
        }
        sparse_attn_v2_batched_kernel<256><<<grid, dim3(256), smem, s>>>(
            q, k_tbl, v_tbl,
            idx, out, B, total_tbl, h, d, dv, topk, nodedup_, qk_mma_, dbg_);
    }
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
// 16 (was 8): the mix kernel's 48 blocks leave the GPU mostly idle and the
// launcher's own comment records a ~50us fixed per-launch cost that scales
// down with block count (1 blk 55us / 16 blk 11us / 192 blk 5.8us).
// Doubling the K-split doubles the blocks -> better amortization.
#define HC_MIX_KS 16
// Plan N v1: P345 column blocks per token (gridDim.y) — h=4096/16 = 256
// columns per block × 256 threads = 1 column/thread. 16 blocks spread the
// old single-block P3's 64KB x read over 16 SMs' L2 bandwidth.
// P345 grid.y: column blocks over h=nh/n. TP shrinks h per rank (hidden is
// replicated). NOTE 2026-09-08: NB>16 CORRUPTS the output (bisected:
// 6355741 NB=16 → correct 出师表; 27f6ba0 NB=64 → all-EOS; 117982e NB=256
// → "the the the"). Root cause TBD; 16 is the last known-good value and the
// NB sweep showed no measurable gain (rest345 is compute-bound, 12.6µs at
// NB=16 vs 13.7 at 64 — the extra blocks LOSE).
#define HC_P345_NB 16

__global__ void hc_pre_mix_split_kernel(const float* __restrict__ res,
                                        const float* __restrict__ fw,
                                        float* __restrict__ mx_partial,
                                        unsigned* __restrict__ ctr2,
                                        int s, int n, int h, int mix) {
#if __CUDA_ARCH__ >= 900
    // PDL (v5): this kernel's launch overlaps the PREDECESSOR's tail (the
    // attr is set by the launcher under FERRITE_PDL=1); gridDepSync gates
    // the res/fw reads until the predecessor's writes are visible. No-op on
    // a normal launch.
    cudaGridDependencySynchronize();
#endif
    // ctr2 zeroing folded in here (was a cudaMemsetAsync before every launch
    // — 2/step/layer = 128 extra 1.5µs GPU ops per decode step, nsys-counted).
    // Safe: rest345 (the only ctr2 reader) is a LATER kernel on the same
    // stream, so this write is visible before any atomicAdd.
    if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x < (unsigned)s) {
        ctr2[threadIdx.x] = 0u;
    }
    // K-SPLIT: gridDim.z = KS lanes per mix row — 24 mix rows × 8 lanes =
    // 192 blocks (130% SM) vs the old 24-block single-lane version (16% SM,
    // each block serially dotting the full 18432-dim row). Each lane dots
    // its 1/KS segment; the rest kernel's prologue sums the KS partials and
    // applies rsq (rsq itself moved there too — phase 1 is a pure dot now).
    const int KS = gridDim.z;
    int t = blockIdx.x;
    const int m0 = blockIdx.y * 4;   // 4 mix rows per block: x was re-read
                                     // once per mix row (24x -> L2-bound)
    int z = blockIdx.z;
    if (t >= s) return;
    const float* x = res + (size_t)t * n * h;
    const int nh = n * h;
    const float* row0 = fw + (size_t)m0 * nh;
    int seg = (nh + KS - 1) / KS;
    int lo = z * seg;
    int hi = min(lo + seg, nh);

    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    float sq = 0.f;
    // unroll 4: 5 independent loads per iteration (x + 4 weight rows) — without
    // the unroll the compiler serialized them and the loop stalled on the L2.
    #pragma unroll 4
    for (int i = lo + threadIdx.x; i < hi; i += blockDim.x) {
        const float xv = x[i];                 // ONE x read serves 4 rows
        sq += xv * xv;                         // Σx² rides free
        #pragma unroll
        for (int mm = 0; mm < 4; mm++) acc[mm] += row0[(size_t)mm * nh + i] * xv;
    }
    __shared__ float red[8];
    #pragma unroll
    for (int mm = 0; mm < 4; mm++) {
        float a = acc[mm];
        for (int off = 16; off > 0; off >>= 1) a += __shfl_down_sync(0xffffffff, a, off);
        if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = a;
        __syncthreads();
        if (threadIdx.x == 0 && m0 + mm < mix) {
            float tot = 0.f;
            for (int w = 0; w < 8; w++) if (w < (blockDim.x + 31) >> 5) tot += red[w];
            mx_partial[((size_t)t * mix + m0 + mm) * KS + z] = tot;
        }
        __syncthreads();
    }
    // Σx² partial (m0 == 0 lane only): the rest kernel's P1 prologue re-read
    // the full nh (~15µs x 90/step = 1.35ms) just for rsqrt(Σx²/nh + eps).
    if (m0 == 0) {
        for (int off = 16; off > 0; off >>= 1) sq += __shfl_down_sync(0xffffffff, sq, off);
        if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = sq;
        __syncthreads();
        if (threadIdx.x == 0) {
            float tot2 = 0.f;
            for (int w = 0; w < 8; w++) if (w < (blockDim.x + 31) >> 5) tot2 += red[w];
            mx_partial[(size_t)s * mix * KS + (size_t)t * KS + z] = tot2;
        }
    }
}

// ── Plan N v1: hc_pre_rest split into restA (P1/P2) + P345 (multi-block) ──
// The single-block rest kernel was 62µs/layer in serve (1-SM launch bound:
// P3's 64KB x-read cold at ~8GB/s effective + the serial P4/P5 tail). restA
// keeps only P1 (mx partial reduce + rsq) + P2 (sigmoid + sinkhorn + comb)
// — small serial work, ~15µs. P345 moves P3 (li_raw = Σ pre_s·x, column-
// independent) + P4 (Σli² partials) + P5 (·rsq·nw) onto a grid(s, NB)
// multi-block launch: 16 blocks × 256 threads split the h=4096 columns —
// the x read spreads over 16 SMs' L2 bandwidth instead of 1.
// FP notes: P3 keeps the per-column i-ascending FMA chain (bit-identical
// to the old in-smem P3); P4's reduction TREE changes (16 block partials
// serial-summed vs the old 32-warp tree — 1ulp risk, validated by 出师表);
// P5 is li_raw·inv·nw — same op order as before.

// ── Plan N v2: rest345 — ONE kernel grid(s, NB=16) replacing restA + p345 ──
// v1 data (nsys, 85 layers × 20 steps): restA 55.6µs (1 block!) + p345
// 11.2µs. The 55µs on a ~5µs workload is the 1-BLOCK kernel's fixed
// serve-real overhead (the old rest's 62µs was the same ~50µs + P3-P5 —
// the "P3 cold-read 40µs" hypothesis is DEAD: p345 reads x at 16 blocks
// in 11µs total). Kernel duration scales with block count on this
// workload class: 1 blk = 55µs, 16 blk = 11µs, 192 blk = 5.8µs.
// v2 kills the last 1-block kernel: P1 (mx partial reduce) + P2 (sigmoid
// pre_s) run REDUNDANTLY in every block (same inputs → bit-identical smem
// results — 768B of L2-hot reads × 16, free), the sinkhorn/comb/post
// (thread0-serial ~8µs at iters=20) runs on block 0 AFTER its P4 atomic —
// in parallel with the is_last block's P5 (no dependency: P5 needs only
// li_raw + partials, sinkhorn feeds hc_post). Wall clock ≈ max(block0
// ~13µs, is_last P5 ~7µs) ≈ 13µs vs v1's 67µs chain.
__global__ void hc_pre_rest345_kernel(const float* __restrict__ res,
                                     const float* __restrict__ mx_in,
                                     const float* __restrict__ scale,
                                     const float* __restrict__ base,
                                     const float* __restrict__ nw,
                                     float* __restrict__ pre_s_g,   // [s][n] (block0, compat/debug)
                                     float* __restrict__ post,      // [s][n] (block0)
                                     float* __restrict__ comb,      // [s][n*n] (block0)
                                     float* __restrict__ li,        // [s][h] in: li_raw out: li
                                     float* __restrict__ p4_part,   // [s][NB] Σli² partials
                                     unsigned* __restrict__ ctr,    // [s] is_last counter (pre-zeroed)
                                     int s, int n, int h, int mix, int mix_ks,
                                     float rms_eps, float hc_eps, int iters) {
#if __CUDA_ARCH__ >= 900
    // PDL (v5): launch overlaps mix_split's tail; gridDepSync gates the
    // mx_in reads (mix_split's partial writes) until it completes.
    cudaGridDependencySynchronize();
#endif
    const int NB = gridDim.y;
    int t = blockIdx.x;
    if (t >= s) return;
    const int b = blockIdx.y;
    const int nh = n * h;
    const int hpb = (h + NB - 1) / NB;
    const int col = b * hpb + (int)threadIdx.x;
    const float* x = res + (size_t)t * n * h;
    extern __shared__ float sm[];
    float* mx_s = sm;                    // [mix] (per-block redundant copy)
    float* cb = sm + mix;                // [n*n]
    float* ps = sm + mix + n * n;        // [n] pre_s (per-block copy)
    float* red = sm + mix + n * n + n;   // [48]
    float* xs = red + 48;                // [n * hpb] cp.async staging of the x rows
    __shared__ float red8[8];
    __shared__ int last;
    __shared__ float inv_s;

    // P1 (redundant per block, bit-identical): reduce the KS mix partials
    // + rsq. mx_in: [t][mix][ks] partials + Σx² tail [s][mix_ks] at
    // s*mix*mix_ks (written by mix_split's m==0 lanes).
    {
        if (threadIdx.x == 0) {
            float msq = 0.f;
            const float* xsq = mx_in + (size_t)s * mix * mix_ks;
            for (int z = 0; z < mix_ks; z++) msq += xsq[(size_t)t * mix_ks + z];
            red[39] = rsqrtf(msq / (float)nh + rms_eps);
        }
        __syncthreads();
        for (int m = threadIdx.x; m < mix; m += blockDim.x) {
            float acc = 0.f;
            for (int z = 0; z < mix_ks; z++) acc += mx_in[((size_t)t * mix + m) * mix_ks + z];
            mx_s[m] = acc * red[39];
        }
        __syncthreads();
    }
    // P2a: sigmoid pre_s (redundant per block — feeds this block's P3 from
    // SMEM). post (b==0 only, global).
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        ps[i] = 1.0f / (1.0f + __expf(-(mx_s[i] * scale[0] + base[i]))) + hc_eps;
        if (b == 0) post[t * n + i] = 2.0f * (1.0f / (1.0f + __expf(-(mx_s[n + i] * scale[1] + base[n + i]))));
    }
    __syncthreads();
    // P3: li_raw[col] = Σ_i ps[i]·x[i·h+col] — one column per thread,
    // i-ascending FMA chain (same numeric order as the old in-smem P3).
    float acc = 0.f;
    if (col < h) {
        // cp.async stage ALL n x-rows for this column block up front: the 16
        // strided global loads (~600ns each, 16KB apart -> ~9.6us of the 12us
        // per-block latency) now fly concurrently instead of 4 at a time.
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            if (i < n) {
                const unsigned dst = (unsigned)__cvta_generic_to_shared(&xs[i * hpb + threadIdx.x]);
                asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"
                             :: "r"(dst), "l"(x + (size_t)i * h + col));
            }
        }
        asm volatile("cp.async.commit_group;\n");
        asm volatile("cp.async.wait_group 0;\n");
        __syncthreads();
        float p0 = 0.f, p1 = 0.f, p2 = 0.f, p3 = 0.f;
        int i = 0;
        for (; i + 3 < n; i += 4) {
            p0 += ps[i]     * xs[(i)     * hpb + threadIdx.x];
            p1 += ps[i + 1] * xs[(i + 1) * hpb + threadIdx.x];
            p2 += ps[i + 2] * xs[(i + 2) * hpb + threadIdx.x];
            p3 += ps[i + 3] * xs[(i + 3) * hpb + threadIdx.x];
        }
        for (; i < n; i++) p0 += ps[i] * xs[i * hpb + threadIdx.x];
        acc = (p0 + p1) + (p2 + p3);
        li[(size_t)t * h + col] = acc;   // li_raw staged in place (P5 overwrites)
    }
    // P4: Σli_raw² block partial (warp tree + red8 serial sum)
    float ss = acc * acc;
    for (int off = 16; off > 0; off >>= 1) ss += __shfl_down_sync(0xffffffff, ss, off);
    if ((threadIdx.x & 31) == 0) red8[threadIdx.x >> 5] = ss;
    __syncthreads();
    if (threadIdx.x == 0) {
        float tot = 0.f;
        for (int w = 0; w < 8; w++) if (w < (blockDim.x + 31) >> 5) tot += red8[w];
        p4_part[(size_t)t * NB + b] = tot;
    }
    // is_last election (the last block to arrive runs P5 over ALL columns —
    // write visibility: peers' stores → threadfence → atomic).
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0)
        last = (atomicAdd(&ctr[t], 1u) == (unsigned)NB - 1u) ? 1 : 0;
    __syncthreads();
    // P2b (block0, warp1 lanes 0..15 — v4): the v3 4-thread+__syncthreads
    // sinkhorn was ~6µs (41 block-wide barriers); n*n=16 elements fit ONE
    // warp's registers — all row/col reductions are __shfl_xor butterflies
    // (row groups = lanes 4r..4r+3 → xor 1,2; col groups = lanes c,c+4,
    // c+8,c+12 → xor 4,8). NO barriers, ~1.3µs at iters=20. FP: the sum
    // order is a butterfly tree vs the serial k-ascending (1ulp class —
    // the P4-tree change already validated by 出师表; max/div are
    // order-independent). Runs on warp1 AFTER this block's P4 atomic —
    // parallel with the is_last block's P5 and never on the atomic-sync
    // critical path (warp0 handles is_last/P5; warp1's lanes exit the
    // sinkhorn loop independently).
    if (b == 0 && threadIdx.x >= 32 && threadIdx.x < 48) {
        const unsigned m16 = 0x0000ffffu; // warp1 lanes 0..15 (shfl group)
        const int ln = threadIdx.x - 32;  // 0..15 = r*4+c
        const float v0 = mx_s[2 * n + ln] * scale[2] + base[2 * n + ln];
        // initial row softmax: rmax (butterfly max — order-free), denom, /=, +eps
        float v = v0;
        float rmax = v;
        rmax = fmaxf(rmax, __shfl_xor_sync(m16, rmax, 1));
        rmax = fmaxf(rmax, __shfl_xor_sync(m16, rmax, 2));
        v = __expf(v - rmax);
        float denom = v;
        denom += __shfl_xor_sync(m16, denom, 1);
        denom += __shfl_xor_sync(m16, denom, 2);
        v = v / denom + hc_eps;
        // initial col normalise (xor 4, 8 butterflies)
        float cs = v;
        cs += __shfl_xor_sync(m16, cs, 4);
        cs += __shfl_xor_sync(m16, cs, 8);
        v /= cs + hc_eps;
        for (int it = 1; it < iters; it++) {
            // row normalise (xor 1, 2)
            float rs = v;
            rs += __shfl_xor_sync(m16, rs, 1);
            rs += __shfl_xor_sync(m16, rs, 2);
            v /= rs + hc_eps;
            // col normalise (xor 4, 8)
            float cs2 = v;
            cs2 += __shfl_xor_sync(m16, cs2, 4);
            cs2 += __shfl_xor_sync(m16, cs2, 8);
            v /= cs2 + hc_eps;
        }
        comb[(size_t)t * n * n + ln] = v;
        if (threadIdx.x - 32 < n) pre_s_g[(size_t)t * n + (threadIdx.x - 32)] = ps[threadIdx.x - 32];
    }
    // P4b + P5 (is_last block only): reduce partials → rsq, li = li_raw·inv·nw
    if (!last) return;
    if (threadIdx.x == 0) {
        float tt = 0.f;
        for (int q = 0; q < NB; q++) tt += p4_part[(size_t)t * NB + q];
        inv_s = rsqrtf(tt / (float)h + rms_eps);
        ctr[t] = 0u;   // reset for the next launch (stream/graph ordered)
    }
    __syncthreads();
    const float inv = inv_s;
    for (int c = threadIdx.x; c < h; c += blockDim.x)
        li[(size_t)t * h + c] = li[(size_t)t * h + c] * inv * nw[c];
}

extern "C" cudaError_t ferrite_hc_pre_split(const float* res, const float* fw,
                                            const float* scale, const float* base,
                                            const float* nw,
                                            float* li, float* post, float* comb,
                                            float* mx_scratch,
                                            int s, int n, int h, int mix,
                                            float rms_eps, float hc_eps, int iters,
                                            cudaStream_t stream) {
    // Plan N v2: two launches —
    //   1. mix_split  grid(s, mix, KS=8): mx partials + Σx² partials (192 blk)
    //   2. rest345    grid(s, NB=16): P1+P2a sigmoid REDUNDANTLY per block
    //                 (bit-identical smem), P3/P4 per column-block, sinkhorn
    //                 (block0, thread0) parallel with the is_last P5.
    // v1 data: restA(1 blk) 55.6µs — the 1-BLOCK kernel carries ~50µs fixed
    // serve-real overhead (block-count scaling: 1 blk 55µs / 16 blk 11µs /
    // 192 blk 5.8µs); the old rest's "62µs" was the same ~50µs + P3-P5 —
    // the P3 cold-read theory is dead (p345 read x at 16 blocks in 11µs).
    // Scratch layout: [mx: s*mix*KS][Σx²: s*KS][old ctr: s][pre_s: s*n]
    // [p4 partials: s*NB][rest345 ctr: s] — Rust allocates
    // s*(mix*8 + 8 + 1 + n + 17) floats.
    const int NB = HC_P345_NB;
    float* pre_s_g = mx_scratch + (size_t)s * mix * HC_MIX_KS + (size_t)s * HC_MIX_KS + s;
    float* p4 = pre_s_g + (size_t)s * n;
    unsigned* ctr2 = (unsigned*)(p4 + (size_t)s * NB);
    dim3 mix_grid(s, (mix + 3) / 4, HC_MIX_KS);
    if (ferrite_pdl_enabled()) {
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim = mix_grid; cfg.blockDim = dim3(256);
        cfg.dynamicSmemBytes = 0; cfg.stream = stream;
        cudaLaunchAttribute attrs[1];
        attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attrs[0].val.programmaticStreamSerializationAllowed = 1;
        cfg.attrs = attrs; cfg.numAttrs = 1;
        cudaLaunchKernelEx(&cfg, hc_pre_mix_split_kernel,
                           res, fw, mx_scratch, ctr2, s, n, h, mix);
    } else {
        hc_pre_mix_split_kernel<<<mix_grid, 256, 0, stream>>>(
            res, fw, mx_scratch, ctr2, s, n, h, mix);
    }
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return e;

    dim3 p345_grid(s, NB);
    const int hpb_l = (h + 15) / 16;   // must match the kernel's hpb (NB=16)
    size_t smem_r = (size_t)(mix + n * n + n + 48 + n * hpb_l) * sizeof(float);  // + xs staging
    if (ferrite_pdl_enabled()) {
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim = p345_grid; cfg.blockDim = dim3(256);
        cfg.dynamicSmemBytes = smem_r; cfg.stream = stream;
        cudaLaunchAttribute attrs[1];
        attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attrs[0].val.programmaticStreamSerializationAllowed = 1;
        cfg.attrs = attrs; cfg.numAttrs = 1;
        cudaLaunchKernelEx(&cfg, hc_pre_rest345_kernel,
                           res, mx_scratch, scale, base, nw, pre_s_g, post, comb, li,
                           p4, ctr2, s, n, h, mix, HC_MIX_KS, rms_eps, hc_eps, iters);
    } else {
        hc_pre_rest345_kernel<<<p345_grid, 256, smem_r, stream>>>(
            res, mx_scratch, scale, base, nw, pre_s_g, post, comb, li,
            p4, ctr2, s, n, h, mix, HC_MIX_KS, rms_eps, hc_eps, iters);
    }
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
// P2P one-shot AR v2 — PRODUCTION (in-graph, multi-call safe).
// v1 (above) is the micro-bench protocol: flags are never reset, so a
// SECOND call spins through stale 1s and reads half-written staging
// (race) — unusable for the decode chains (90 AR/step × thousands of
// steps). v2 fixes it with an EPOCH + PING-PONG protocol, no resets:
//   epoch  : per-rank device u32, advances once per call (the down's
//            last block, AFTER the flag writes). Monotonic; the sum
//            derives its round from it (stream order: down → sum).
//   staging: [2][world][n] — call e uses parity e&1 (ping-pong). The
//            rank j's down of call e+2 (same parity) is gated by the
//            transitive chain j.sum(e+1) ⟂ j.down(e+2) ≥ i.down(e+1) ≥
//            i.sum(e) — a 2-deep pipeline: rank i's sum(e) can never
//            race rank j's staging overwrite of call e+2.
//   flags  : per-rank [world] u32 EPOCH stamps (written via UVA by the
//            peers' down; the wrap-safe spin compares (int)(flag-e) >= 0).
//   ctr    : per-call block-arrivals (the down's last block does the
//            flag writes + the epoch advance + the ctr reset).
// Graph-capturable: the epoch/staging/flags/tables are persistent device
// buffers (fixed addresses across replays); the counters are RUNTIME
// device state the captured kernels read/write — each replay advances the
// epoch exactly like a dry-run call. Lockstep requirement: all ranks
// launch their graphs ~together (the fan_out worker pool guarantees).
// ============================================================
__global__ void p2p_ar_down_v2_kernel(
    const float* __restrict__ partial,          // this rank's partial [n]
    float* const* __restrict__ staging_tbl,     // [world] peers' staging bases ([2][world][stride])
    unsigned* const* __restrict__ ready_tbl,    // [world] peers' flag rows ([world] u32)
    unsigned* epoch, unsigned* ctr,             // this rank's device counters
    int world, int my_rank, int n, int stride) {
    unsigned e = *epoch; // this call's epoch (pre-advance)
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = partial[i];
        const size_t off = (size_t)(((e & 1u) * (unsigned)world + (unsigned)my_rank) * (unsigned)stride + (unsigned)i);
        #pragma unroll 4
        for (int r = 0; r < world; r++) staging_tbl[r][off] = v;
    }
    __threadfence_system(); // peer-visible stores before flag
    __syncthreads();
    if (threadIdx.x == 0) {
        // Single-block grids (every batched decode size here: n*world <=
        // 1024 threads) need NO arrival counter — this block IS the last.
        // The atomicAdd path left ctr stuck non-zero across replays (its
        // reset raced the graph capture), so no block ever saw "last" and
        // *epoch never advanced (diag: e==0 for EVERY AR) → peers' flags
        // were never re-stamped → the monotonic wait deadlocked.
        unsigned prev = (gridDim.x == 1u) ? 0u : atomicAdd(ctr, 1u);
        if (threadIdx.x == 0 && my_rank == 7 && e < 2u)
            printf("[p2p-prev] rank=%d e=%u prev=%u gridDim=%u ctr=%u\n",
                   my_rank, e, prev, (unsigned)gridDim.x, (unsigned)*ctr);
        if (prev == gridDim.x - 1u) { // last block: all stores fenced
            for (int r = 0; r < world; r++)
                // SYSTEM-scope atomic store. A plain volatile store is only
                // device-scope: it never becomes visible in the peer's
                // address space over NVLink/UVA, so both sides waited
                // forever for a flag that WAS written but invisible
                // (measured hang: prev=50 cur=50 myepoch=50, both ranks
                // stuck in phase B). A system fence cannot rescue a
                // device-scope store — the store itself must be sys-scope.
                atomicExch_system((unsigned int*)&ready_tbl[r][my_rank], e + 1u);
            __threadfence_system();
            *ctr = 0u;     // reset for the next call (stream-ordered)
            *epoch = e + 1u; // advance AFTER the flags (the next kernel sees it)
        }
    }
}

__global__ void p2p_ar_sum_v2_kernel(
    const float* __restrict__ staging_local,   // my [2][world][stride]
    const unsigned* __restrict__ ready_local,   // my [world] epoch stamps
    const unsigned* epoch,                      // (= e+1 after my down)
    float* __restrict__ out, int world, int n, int stride) {
    unsigned e2 = *epoch; // the round this sum completes (down's e+1)
    if (threadIdx.x == 0) { // every block spins until all ranks' flags reach e2
        for (int r = 0; r < world; r++)
            while ((int)((*(volatile unsigned*)&ready_local[r]) - e2) < 0) __nanosleep(100);
    }
    __syncthreads();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int par = (int)((e2 - 1u) & 1u);
        float acc = 0.f;
        for (int r = 0; r < world; r++)
            acc += staging_local[(size_t)(par * world + r) * stride + i];
        out[i] = acc;
    }
}

// v3: down+sum fused into ONE kernel — the v2 pair paid a full kernel
// boundary (~3-4us launch gap + re-read of epoch/ready) between the
// publish and the collect phases. The fused kernel: phase A publishes the
// partial (last block flags peers + advances epoch), then EVERY block spins
// on the peers' ready flags and reduces its own slice. Same parity double-
// buffer, same lockstep epoch protocol as v2 — semantics bit-identical.
__global__ void p2p_ar_fused_v3_kernel(
    const float* __restrict__ partial,          // this rank's partial [n]
    float* const* __restrict__ staging_tbl,     // [world] peers' staging bases
    unsigned* const* __restrict__ ready_tbl,    // [world] peers' flag rows
    unsigned* epoch, unsigned* ctr,             // this rank's device counters
    const float* __restrict__ staging_local,   // my [2][world][stride]
    const unsigned* __restrict__ ready_local,   // my [world] epoch stamps
    unsigned* __restrict__ seen,                // my [world] last-observed stamps
    float* __restrict__ out, int world, int my_rank, int n, int stride) {
    unsigned e = *epoch; // this call's epoch (pre-advance)
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // TP8 vs TP4: the old code serialized the world loop in ONE thread
    // (8 NVLink stores + 8 flag polls + 8 staging reads, each ~1.5µs at
    // world=8) — measured 15.3µs at TP8 vs 11.7µs at TP4, the whole TP8
    // gap. Map (token, peer) across threads instead.
    const int total = n * world;
    const int tr = (threadIdx.x < (unsigned)world) ? (int)threadIdx.x : -1;
    // phase A: down — publish my partial to all peers' staging slots.
    // (token, peer) mapped across threads: total = n * world. The previous
    // version reused `i` as the peer-mapped index INSIDE `if (i < n)`: at
    // decode (n=1) only thread 0 ran, writing peer 0 alone while still
    // stamping EVERY peer's flag → every other rank read a stale partial
    // (flag said "arrived", staging held the previous epoch's value) →
    // wrong all-reduce → gibberish text.
    {
        // COALESCED + 16B-VECTORIZED: consecutive threads write consecutive
        // tokens of the SAME peer (the old (token,peer) flattening sent
        // adjacent threads to different GPUs), and each thread moves 16B
        // instead of 4B. The 4-byte version needed 524K NVLink transactions
        // per AR (2.1MB in 49us = 43GB/s vs the 750GB/s memcpyPeer floor).
        const int step = gridDim.x * blockDim.x;
        const int n4 = n >> 2;
        const float4* p4 = reinterpret_cast<const float4*>(partial);
        for (int i4 = blockIdx.x * blockDim.x + threadIdx.x; i4 < n4; i4 += step) {
            const float4 v = p4[i4];
            const size_t base = (size_t)((e & 1u) * (unsigned)world + (unsigned)my_rank) * (unsigned)stride + (size_t)i4 * 4;
            for (int rr = 0; rr < world; rr++)
                *reinterpret_cast<float4*>(staging_tbl[rr] + base) = v;
        }
        for (int ii = n4 * 4 + blockIdx.x * blockDim.x + threadIdx.x; ii < n; ii += step) {
            float v = partial[ii];
            const size_t base = (size_t)((e & 1u) * (unsigned)world + (unsigned)my_rank) * (unsigned)stride + (unsigned)ii;
            for (int rr = 0; rr < world; rr++) staging_tbl[rr][base] = v;
        }
    }
    // NO __threadfence_system() here: the finish kernel runs only after this
    // kernel COMPLETES (stream order), and kernel completion makes every
    // store visible — a per-thread system fence across 655K threads was
    // costing ~2 ms per AR.
    if (blockIdx.x == 0 && threadIdx.x == 0) *ctr = e;
}

// Phase B: publish the epoch stamp + wait for the peers' stamps. ONE block
// only: with the reduce in a separate kernel (below) there is no reason for
// 80 blocks × 8 threads to hammer the same 8 flag words — that thundering
// herd delays the very store they wait for (measured ~200 µs/AR at 16 seqs
// vs ~15 µs at 1 seq with a single block polling).
__global__ void p2p_ar_publish_v3_kernel(
    unsigned* const* __restrict__ ready_tbl,
    unsigned* epoch, const unsigned* __restrict__ snap,
    const unsigned* __restrict__ ready_local,
    unsigned* __restrict__ seen, int world, int my_rank) {
    const unsigned e = *snap; // stable: *epoch is advanced only below
    if (threadIdx.x == 0) {
        for (int r = 0; r < world; r++)
            *(volatile unsigned*)&ready_tbl[r][my_rank] = e + 1u;
        // ONE system fence publishes the 8 stamps (vLLM's custom-AR pattern).
        __threadfence_system();
        *epoch = e + 1u;
    }
    __syncthreads(); // our own stamp (tr == my_rank) must be visible
    const int tr = (threadIdx.x < (unsigned)world) ? (int)threadIdx.x : -1;
    if (tr >= 0) { // one thread per peer polls its own flag (parallel)
        unsigned* my_seen = seen + (size_t)blockIdx.x * (unsigned)world;
        unsigned prev = my_seen[tr];
        unsigned cur = *(volatile unsigned*)&ready_local[tr];
        long spins = 0;
        while ((int)(cur - prev) <= 0) {
            __nanosleep(100);
            cur = *(volatile unsigned*)&ready_local[tr];
            if (++spins > 500000) { // ~50ms: diagnose + break instead of hanging
                if (tr == 0) {
                    printf("[p2p-hang] rank=%d peer=%d prev=%u cur=%u myepoch=%u\n",
                           my_rank, tr, prev, cur, e);
                }
                break;
            }
        }
        my_seen[tr] = cur;
    }
}

// Phase C: the reduce, with NO polling at all (the publish kernel above ran
// to completion first, so every peer's staging segment is already in place).
__global__ void p2p_ar_reduce_v3_kernel(
    const float* __restrict__ staging_local,
    float* __restrict__ out, const unsigned* __restrict__ snap,
    int world, int n, int stride) {
    const unsigned e = *snap;
    const int step = gridDim.x * blockDim.x;
    const int n4 = n >> 2;
    for (int i4 = blockIdx.x * blockDim.x + threadIdx.x; i4 < n4; i4 += step) {
        float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
        for (int r = 0; r < world; r++) {
            const float4 v = *reinterpret_cast<const float4*>(
                staging_local + (size_t)((e & 1u) * world + r) * stride + (size_t)i4 * 4);
            acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
        }
        *reinterpret_cast<float4*>(out + (size_t)i4 * 4) = acc;
    }
    for (int ii = n4 * 4 + blockIdx.x * blockDim.x + threadIdx.x; ii < n; ii += step) {
        float acc = 0.f;
        for (int r = 0; r < world; r++)
            acc += staging_local[(size_t)((e & 1u) * world + r) * stride + ii];
        out[ii] = acc;
    }
}

extern "C" cudaError_t ferrite_p2p_ar_fused_v3(
    const float* partial, float* const* staging_tbl,
    unsigned* const* ready_tbl, unsigned* epoch, unsigned* ctr,
    const float* staging_local, const unsigned* ready_local,
    unsigned* seen,
    float* out, int n, int world, int my_rank, int stride, cudaStream_t s) {
    // Multi-block store kernel (fast: 640 blocks × 1024 threads at 16 seqs,
    // vs one SM issuing 655K scalar stores) + a separate finish kernel that
    // publishes the stamp. Stream order is what makes the publish safe: the
    // finish kernel starts only after EVERY store of the store kernel is
    // visible, so no arrival counter / last-block detection is needed.
    int threads = 1024;
    int work = n * world;
    int blocks = (work + threads - 1) / threads;
    if (blocks < 1) blocks = 1;
    p2p_ar_fused_v3_kernel<<<blocks, threads, 0, s>>>(
        partial, staging_tbl, ready_tbl, epoch, ctr,
        staging_local, ready_local, seen, out, world, my_rank, n, stride);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    // Publish + wait on ONE block (8 pollers, no thundering herd).
    p2p_ar_publish_v3_kernel<<<1, 32, 0, s>>>(
        ready_tbl, epoch, ctr, ready_local, seen, world, my_rank);
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    // Reduce on many blocks — no polling at all (the publish kernel already
    // observed every peer's stamp).
    int rblocks = (n + threads - 1) / threads;
    if (rblocks < 1) rblocks = 1;
    p2p_ar_reduce_v3_kernel<<<rblocks, threads, 0, s>>>(
        staging_local, out, ctr, world, n, stride);
    return cudaGetLastError();
}

extern "C" cudaError_t ferrite_p2p_ar_oneshot_v2(
    const float* partial, float* const* staging_tbl,
    unsigned* const* ready_tbl, unsigned* epoch, unsigned* ctr,
    const float* staging_local, const unsigned* ready_local,
    float* out, int n, int world, int my_rank, int stride, cudaStream_t s) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    if (blocks < 1) blocks = 1;
    p2p_ar_down_v2_kernel<<<blocks, threads, 0, s>>>(
        partial, staging_tbl, ready_tbl, epoch, ctr, world, my_rank, n, stride);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return e;
    p2p_ar_sum_v2_kernel<<<blocks, threads, 0, s>>>(
        staging_local, ready_local, epoch, out, world, n, stride);
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
#if __CUDA_ARCH__ >= 900
    cudaGridDependencySynchronize(); // PDL v5: launch overlaps predecessor tail
#endif
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
    return pdl_or_plain(gemv_qkv_conv_kernel, grid, dim3(256), 0, s,
                       x, (const __nv_bfloat16*)w, (const float*)cw, cs,
                       q, k, v, in_f, proj);
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
#if __CUDA_ARCH__ >= 900
    cudaGridDependencySynchronize(); // PDL v5: launch overlaps predecessor tail
#endif
    int t = blockIdx.x;
    int hd = blockIdx.y;
    if (t >= n || hd >= h) return;
    // SPLIT-K over dv (SGLang num_splits pattern): grid.z slices the v
    // columns. Every stage is column-independent (ks[j], o[j], the delta
    // update and the state R/W touch only column j), so the split needs no
    // cross-block reduction — the q/k L2 norms are redundantly recomputed.
    // At TP8 the grid was (1,h)=8 blocks on 148 SM (66KB smem → 3 SM).
    const int dsp = blockIdx.z;
    const int dsplits = gridDim.z;
    const int dvl = (dv + dsplits - 1) / dsplits;
    const int dv0 = dsp * dvl;
    const int dv1 = min(dv0 + dvl, dv);
    if (dv0 >= dv) return;
    const float a_ex = expf(a_log[hd]);
    const float bt = 1.0f / (1.0f + expf(-b_raw[hd]));
    const size_t spitch = (size_t)(dv1 - dv0) + 1;
    extern __shared__ float sm[];
    const int dvl_ = dv1 - dv0;
    float* S = sm;
    float* ks = S + (size_t)dk * spitch;
    float* kh = ks + dvl_;
    float* vh = kh + dk;
    float* qh = vh + dvl_;
    float* gh = qh + dk;
    float* red = gh + dk; // [2]: L2 sums (q, k)
    float* dec = red + 2;          // [dk]: per-channel decay factor exp(gh[i])
    float* red2 = dec + dk;        // [splits*dv <= 512]: column-split partials
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
    for (int j = threadIdx.x; j < dvl_; j += blockDim.x)
        vh[j] = v[(size_t)((size_t)t * h + hd) * dv + dv0 + j];
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
        dec[i] = expf(gh[i]); // per-channel decay factor (absorbs old stage 1)
    }
    __syncthreads();
    // fused state load + per-channel decay (was a separate full-S read-modify-
    // write pass): S[i,j] = Sg[i,j] * dec[i] — bit-identical to the old
    // two-stage form (x*dec then use == load then *dec).
    float* Sg = state + (size_t)hd * dk * dv + dv0;
    for (int idx = threadIdx.x; idx < dk * dvl_; idx += blockDim.x)
        S[(size_t)(idx / dvl_) * spitch + (idx % dvl_)] =
            Sg[(size_t)(idx / dvl_) * dv + (idx % dvl_)] * dec[idx / dvl_];
    __syncthreads();
    // 1. kS = S^T k — column-parallel with intra-column split: blockDim threads
    // = splits x dv lanes, each lane reduces a dk/splits row block, per-column
    // partials joined via red2. (FP-safe class: 1-ulp order change in the ks
    // and o sums only, gdn approved.)
    const int splits = (int)(blockDim.x / dvl_);
    int rows = (dk + splits - 1) / splits;
    {
        int g = threadIdx.x / dvl_, j = threadIdx.x - g * dvl_;
        if (g < splits) {
            float acc = 0.f;
            int i0 = g * rows, i1 = min(i0 + rows, dk);
            for (int i = i0; i < i1; i++)
                acc += kh[i] * S[(size_t)i * spitch + j];
            red2[(size_t)g * dvl_ + j] = acc;
        }
        __syncthreads();
        if (threadIdx.x < dvl_) {
            float a = 0.f;
            for (int g2 = 0; g2 < splits; g2++)
                a += red2[(size_t)g2 * dvl_ + threadIdx.x];
            ks[threadIdx.x] = a;
        }
        __syncthreads();
    }
    // 2. delta rule: S[i,j] += beta * k_i * (v_j - ks_j)
    for (int idx = threadIdx.x; idx < dk * dvl_; idx += blockDim.x)
        S[(size_t)(idx / dvl_) * spitch + (idx % dvl_)] +=
            bt * kh[idx / dvl_] * (vh[idx % dvl_] - ks[idx % dvl_]);
    __syncthreads();
    // 3. o = q^T S — same column-split scheme as stage 1.
    {
        int g = threadIdx.x / dvl_, j = threadIdx.x - g * dvl_;
        if (g < splits) {
            float acc = 0.f;
            int i0 = g * rows, i1 = min(i0 + rows, dk);
            for (int i = i0; i < i1; i++)
                acc += qh[i] * S[(size_t)i * spitch + j];
            red2[(size_t)g * dvl_ + j] = acc;
        }
        __syncthreads();
        if (threadIdx.x < dvl_) {
            float a = 0.f;
            for (int g2 = 0; g2 < splits; g2++)
                a += red2[(size_t)g2 * dvl_ + threadIdx.x];
            out[((size_t)t * h + hd) * dv + dv0 + threadIdx.x] = a;
        }
        __syncthreads();
    }
    // 4. store state back
    for (int idx = threadIdx.x; idx < dk * dvl_; idx += blockDim.x)
        Sg[(size_t)(idx / dvl_) * dv + (idx % dvl_)] = S[(size_t)(idx / dvl_) * spitch + (idx % dvl_)];
}
extern "C" cudaError_t ferrite_gdn_step_v2p(
    const float* q, const float* k, const float* v,
    const float* b_raw, const float* fb, const float* dt_bias,
    const float* a_log, float lb,
    float* state, float* out, int h, int dk, int dv, int dsplits,
    cudaStream_t s) {
    if (dsplits < 1) dsplits = 1;
    const int dvl = (dv + dsplits - 1) / dsplits;
    size_t smem = (size_t)dk * (dvl + 1) * sizeof(float)
                  + (size_t)(dvl + dk + dvl + dk + dk + 2 + dk + 1024) * sizeof(float);
    if (smem > 48 * 1024) {
        cudaError_t e = cudaFuncSetAttribute(gdn_step_v2p_kernel,
                                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        if (e != cudaSuccess) return e;
    }
    dim3 block(1024);
    dim3 grid(1, h, dsplits);
    return pdl_or_plain(gdn_step_v2p_kernel, grid, block, smem, s,
                        q, k, v, b_raw, fb, dt_bias, a_log, lb, state, out, 1, h, dk, dv);
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
// MTP Phase2: fused n-token GDN chunk (verify n=2/3) — ONE launch per layer
// instead of the t-split's 2 chunk_v2 launches + a B0 copy in between.
// The state stays resident in smem across the t loop (HBM state round-trip
// eliminated); the t=0 snapshot (B0 = A+t_last for accept-1's commit) is
// written straight from smem. Saves ~2 launches + 16MB HBM traffic/layer
// on the 34 GDN layers.
//
// v2p UNIFICATION (n-arbitrary directive — the per-t body is gdn_step_v2p's
// core, so n=1 decode and n=3 verify share ONE logic):
//   1. col-split kS/o (v2p stage 1/3): blockDim = splits × dv lanes —
//      lane (g,j) reduces rows [g*dk/splits, (g+1)*dk/splits), per-column
//      partials joined via red2. The old one-thread-per-column serial
//      dk=128-step loop left 384/512 threads idle (dv=128 columns only).
//   2. uniform idx-strided decay (v2p stage 0 pattern): the old per-thread
//      full-row loop (thread i owns row i, serial j in 0..dv) idled
//      blockDim-dk threads; the idx loop spreads dk*dv evenly.
//   3. t=0's decay FUSED into the state load (S = Sg * exp(gate_0) —
//      bit-identical to load-then-multiply; v2p's proven pattern).
// FP class: kS/o's split-partials + serial g-join changes the summation
// order by 1 ulp (the same change v2p made on the n=1 path — gdn-approved,
// validated by 出师表). Snapshots gdn0/gdn1 keep the exact t-boundary
// semantics (post-delta, pre-next-decay).
// ============================================================
__global__ void gdn_chunk_fused_kernel(const float* __restrict__ q,
                                       const float* __restrict__ k,
                                       const float* __restrict__ v,
                                       const float* __restrict__ beta,
                                       const float* __restrict__ gate,
                                       const float* __restrict__ a_log,
                                       float* __restrict__ state,
                                       float* __restrict__ gdn_snaps, // [n-1][h*dk*dv] base (t-th snapshot = base + t*h*dk*dk); null = no snapshots
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
    float* dec = qh + dk;                  // [dk] per-channel decay exp(gate)
    float* red2 = dec + dk;                // [splits*dv <= 512] col-split partials
    float* Sg = state + (size_t)hd * dk * dv;
    const int splits = (int)(blockDim.x / dv); // 512/128 = 4
    int rows = (dk + splits - 1) / splits;
    // state load + t=0 decay FUSED (S = Sg * exp(gate_0[i]) — one pass,
    // bit-identical to the old load + multiply): the dec0 factor needs
    // gate[t=0] FIRST, so the t=0 gate read rides this pass (kh/qh/vh load
    // in the t-loop below as usual).
    {
        const size_t th0 = (size_t)0 * h + hd;
        for (int i = threadIdx.x; i < dk; i += blockDim.x)
            dec[i] = expf(gate[th0 * dk + i]);
        __syncthreads();
        for (int idx = threadIdx.x; idx < dk * dv; idx += blockDim.x)
            S[(size_t)(idx / dv) * spitch + (idx % dv)] = Sg[idx] * dec[idx / dv];
    }
    __syncthreads();
    for (int t = 0; t < n; t++) {
        float bt = beta[(size_t)t * h + hd];
        const size_t th = (size_t)t * h + hd;
        for (int i = threadIdx.x; i < dk; i += blockDim.x) {
            qh[i] = q[th * dk + i];
            kh[i] = k[th * dk + i];
        }
        for (int j = threadIdx.x; j < dv; j += blockDim.x)
            vh[j] = v[th * dv + j];
        __syncthreads();
        // 0. per-channel decay for t>0 (t=0 was fused into the load above).
        //    Uniform idx-stride (dk*dv spread over all threads — the old
        //    per-thread full-row loop idled blockDim-dk threads).
        if (t > 0) {
            for (int i = threadIdx.x; i < dk; i += blockDim.x)
                dec[i] = expf(gate[th * dk + i]);
            __syncthreads();
            for (int idx = threadIdx.x; idx < dk * dv; idx += blockDim.x)
                S[(size_t)(idx / dv) * spitch + (idx % dv)] *= dec[idx / dv];
            __syncthreads();
        }
        // 1. kS = S^T k — col-split (v2p stage 1): lane (g,j) reduces a
        // dk/splits row block, per-column partials joined via red2.
        {
            int g = threadIdx.x / dv, j = threadIdx.x - g * dv;
            if (g < splits) {
                float acc = 0.f;
                int i0 = g * rows, i1 = min(i0 + rows, dk);
                for (int i = i0; i < i1; i++)
                    acc += kh[i] * S[(size_t)i * spitch + j];
                red2[(size_t)g * dv + j] = acc;
            }
            __syncthreads();
            if (threadIdx.x < dv) {
                float a = 0.f;
                for (int g2 = 0; g2 < splits; g2++)
                    a += red2[(size_t)g2 * dv + threadIdx.x];
                ks[threadIdx.x] = a;
            }
            __syncthreads();
        }
        // 2. delta rule: S[i,j] += beta * k_i * (v_j - ks_j)
        for (int idx = threadIdx.x; idx < dk * dv; idx += blockDim.x)
            S[(size_t)(idx / dv) * spitch + (idx % dv)] +=
                bt * kh[idx / dv] * (vh[idx % dv] - ks[idx % dv]);
        __syncthreads();
        // 3. o = q^T S — same col-split scheme as stage 1.
        {
            int g = threadIdx.x / dv, j = threadIdx.x - g * dv;
            if (g < splits) {
                float acc = 0.f;
                int i0 = g * rows, i1 = min(i0 + rows, dk);
                for (int i = i0; i < i1; i++)
                    acc += qh[i] * S[(size_t)i * spitch + j];
                red2[(size_t)g * dv + j] = acc;
            }
            __syncthreads();
            if (threadIdx.x < dv) {
                float a = 0.f;
                for (int g2 = 0; g2 < splits; g2++)
                    a += red2[(size_t)g2 * dv + threadIdx.x];
                out[((size_t)t * h + hd) * dv + threadIdx.x] = a;
            }
            __syncthreads();
        }
        // 4. t-snapshots (B_j = A + t_0..t_j: accept-j+1's commit source),
        //    N-UNIFIED: snap t = A after applying tokens 0..t — ONE
        //    contiguous [n-1] scratch (Rust allocates [n-1][h*dk*dv]),
        //    snapshot t at gdn_snaps + t*(h*dk*dk). Straight from smem.
        //    (n=3 legacy: gdn0 = snap 0, gdn1 = snap 1 — identical layout.)
        if (gdn_snaps != nullptr && t < n - 1) {
            float* Sk = gdn_snaps + (size_t)t * h * dk * dv + (size_t)hd * dk * dv;
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
    // N-UNIFIED entry: gdn0 = the snapshots' contiguous base (the [n-1]
    // per-layer scratch), gdn1 unused (kept for the FFI's 8-slot ABI; the
    // kernel takes the base + stride h*dk*dv). Callers with the old
    // (gdn0, gdn1) pair pass base=gdn0 (the Rust side allocates one
    // contiguous [n-1] buffer).
    (void)gdn1;
    size_t smem = (size_t)dk * (dv + 1) * sizeof(float)
                  + (size_t)(dv + dk + dv + dk + dk + 512) * sizeof(float);
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
        q, k, v, beta, gate, a_log, state, gdn0, out, n, h, dk, dv);
    cudaError_t le = cudaGetLastError();
    if (le != cudaSuccess) {
    }
    return le;
}

// ============================================================
// MTP accept commit (single launch): replaces the per-layer
// cudaMemcpyAsync ping-pong chain (2 memcpys x n_gdn_layers + hprev
// select = ~70 launches/step of pure launch overhead) with ONE kernel.
// k (1..n, the accept length) is read from a PINNED host int at run
// time (zero-copy): the verify graph replay returns argmax to the
// host, the host computes k, then launches this kernel.
// N-UNIFIED (FERRITE_MTP_N — arbitrary verify width): plan row is
// SIX pointers per GDN layer (the n=3 hardcode's 8 collapsed — the
// per-t snapshots live in ONE contiguous [n-1][len] scratch each,
// so the snapshot bases + strides replace the per-snapshot pointers):
//   [0]=conv_a(dst) [1]=gdn_a(dst) [2]=conv_b [3]=gdn_b
//   [4]=conv_snaps_base [5]=gdn_snaps_base ([n-1] snapshots, stride
//       conv_len/gdn_len — snap i = A + verify tokens t_0..t_i)
// k=n commits B (full verify state); k=j<n commits snapshot j-1
// (A + t_0..t_{j-1}). Tail segment: hprev <- hf_v row (k-1).
// ============================================================
__global__ void mtp_commit_kernel(const int* __restrict__ k_pin,
                                   float* const* __restrict__ plan,
                                   int n_plans, int conv_len, int gdn_len,
                                   const float* __restrict__ hf_v,
                                   float* __restrict__ hprev, int hidden, int n) {
    __shared__ int ks;
    if (threadIdx.x == 0) ks = *k_pin;
    __syncthreads();
    const int k = ks; // 1..n
    const int row = 6; // pointers per plan row
    const long lay = (long)n_plans * (conv_len + gdn_len);
    const long total = lay + hidden;
    for (long idx = (long)blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += (long)gridDim.x * blockDim.x) {
        if (idx < lay) {
            int l = (int)(idx / (conv_len + gdn_len));
            int r = (int)(idx - (long)l * (conv_len + gdn_len));
            float* const* p = plan + (size_t)l * row;
            if (r < conv_len) {
                // k=n → B (full); k=j<n → snapshot j-1 (A + t_0..t_{j-1})
                const float* src = (k == n) ? p[2]
                    : (p[4] + (size_t)(k - 1) * conv_len);
                p[0][r] = src[r];
            } else {
                int rr = r - conv_len;
                const float* src = (k == n) ? p[3]
                    : (p[5] + (size_t)(k - 1) * gdn_len);
                p[1][rr] = src[rr];
            }
        } else {
            int r = (int)(idx - lay);
            hprev[r] = hf_v[(size_t)(k - 1) * hidden + r];
        }
    }
}

extern "C" cudaError_t ferrite_mtp_commit(const int* k_pin,
                                          float* const* plan,
                                          int n_plans, int conv_len, int gdn_len,
                                          const float* hf_v, float* hprev,
                                          int hidden, int n, cudaStream_t s) {
    long total = (long)n_plans * (conv_len + gdn_len) + hidden;
    int blocks = (int)((total + 1023) / 1024);
    if (blocks > 4096) blocks = 4096;
    if (blocks < 1) blocks = 1;
    mtp_commit_kernel<<<blocks, 1024, 0, s>>>(k_pin, plan, n_plans, conv_len,
                                             gdn_len, hf_v, hprev, hidden, n);
    return cudaGetLastError();
}

// ============================================================
// gemv_fp8_v2: uint4 16x fp8 weights + 128x128-block scale inline dequant
// (w_f32 = fp8_e4m3(raw) * s[row/128][col/128] — EXACTLY the checkpoint's
// dequant_block semantics, f32 x, f32 accumulate). Native-precision path:
// the weights stay in their checkpoint fp8 (the bf16 path re-quantized the
// dequantized f32 — this reads HALF the bytes; gemv/moe are HBM-bound).
// A 16-element uint4 lane never crosses a 128-col scale block (128%16==0).
// ============================================================
// WPR: warps per row (K-split). 4 warps cooperate on one output row — the
// single-warp/row v0 read HBM at ~1/3 the bf16_v2 rate (no latency hiding);
// the K-split matches bf16_v2's structure (tile per WPR-warps, smem partial
// reduce). kper is 16-aligned: a uint4 lane-step (16 fp8) never crosses a
// 128-col scale block boundary (128 % 16 == 0), and slice starts land on
// scale-block boundaries whenever in_f is 128-aligned (all real shapes).
template <int WPR>
// __launch_bounds__(256, 4): ncu showed 84 regs/thread -> Block Limit
// Registers = 2 -> 23% occupancy, 74.8% of cycles with NO eligible warp.
// Capping the registers to 64 lifted the occupancy to 45.6% (No-Eligible
// 48.9%, Duration 53.5->39.0us). 512-thread blocks measured WORSE (q_a
// 0.044 vs 0.031 ms) - keep 256.
__global__ void __launch_bounds__(256, 4) gemv_fp8_v2_kernel(const float* __restrict__ x,
                                   const unsigned char* __restrict__ w,
                                   const float* __restrict__ scale,
                                   const float* __restrict__ bias,
                                   float* __restrict__ y,
                                   int in_f, int out_f, int nrows,
                                   int srows, int scols) {
    (void)srows;
    const int warps = blockDim.x >> 5;
    const int rpb = warps / WPR;               // rows per block
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    // 8 rows per warp-group with the x slice CACHED in registers: the same
    // token's x row is re-read by every (block, row) otherwise (measured
    // 196MB of x vs 49MB of weights per call -> L2-bound).
    const int R = ((out_f & 7) == 0) ? 8 : 1;
    const int rowg0 = (blockIdx.x * rpb + warp / WPR) * R;   // row within token
    const int kw = warp % WPR;                 // K-slice id
    const int kper = ((in_f + WPR - 1) / WPR + 15) & ~15;  // uint4-aligned slice
    const int k0 = kw * kper;
    const int k1 = min(k0 + kper, in_f);
    __shared__ float part[32];
    // TWO tokens per block: the weight row is loaded and fp8->half2 converted
    // ONCE and reused for both tokens' dots — the per-token instruction count
    // drops ~32% (this GEMV is issue-bound at n=16, not bandwidth-bound).
    // Each token keeps its own accumulation order -> bit-identical output.
    const int t0 = blockIdx.y * 2;
    const bool has1 = (t0 + 1) < nrows;
    float4 xc[2][4];
    #pragma unroll
    for (int tt = 0; tt < 2; tt++) {
        const int k = k0 + lane * 16;
        if (t0 + tt < nrows && k + 15 < k1) {
            const float* xr0 = x + (size_t)(t0 + tt) * in_f + k;
            xc[tt][0] = *reinterpret_cast<const float4*>(xr0);
            xc[tt][1] = *reinterpret_cast<const float4*>(xr0 + 4);
            xc[tt][2] = *reinterpret_cast<const float4*>(xr0 + 8);
            xc[tt][3] = *reinterpret_cast<const float4*>(xr0 + 12);
        } else {
            xc[tt][0] = xc[tt][1] = xc[tt][2] = xc[tt][3] = make_float4(0.f, 0.f, 0.f, 0.f);
        }
    }
    // UNROLLED row loop (no `break`: it blocked unrolling, leaving the 8 rows'
    // loads serialized — the kernel is latency-bound, ~32us per block for 8KB).
    #pragma unroll
    for (int r = 0; r < R; r++) {
    const int row = rowg0 + r;
    if (row >= out_f) continue;
    float acc0 = 0.f, acc1 = 0.f;
    {
        const unsigned char* wr = w + (size_t)row * in_f;
        const float* xr0 = x + (size_t)t0 * in_f;
        const float* xr1 = x + (size_t)(t0 + 1) * in_f;
        const float* srow = scale + (size_t)(row >> 7) * scols;
        int k = k0 + lane * 16;
        if (k + 15 < k1) {
            uint4 wv;
            asm volatile("ld.global.nc.L2::128B.v4.u32 {%0,%1,%2,%3}, [%4];\n"
                         : "=r"(wv.x), "=r"(wv.y), "=r"(wv.z), "=r"(wv.w)
                         : "l"(wr + k));
            const unsigned char* w8 = reinterpret_cast<const unsigned char*>(&wv);
            const float sc = srow[k >> 7];
            const float xv0[16] = {xc[0][0].x, xc[0][0].y, xc[0][0].z, xc[0][0].w,
                                   xc[0][1].x, xc[0][1].y, xc[0][1].z, xc[0][1].w,
                                   xc[0][2].x, xc[0][2].y, xc[0][2].z, xc[0][2].w,
                                   xc[0][3].x, xc[0][3].y, xc[0][3].z, xc[0][3].w};
            const float xv1[16] = {xc[1][0].x, xc[1][0].y, xc[1][0].z, xc[1][0].w,
                                   xc[1][1].x, xc[1][1].y, xc[1][1].z, xc[1][1].w,
                                   xc[1][2].x, xc[1][2].y, xc[1][2].z, xc[1][2].w,
                                   xc[1][3].x, xc[1][3].y, xc[1][3].z, xc[1][3].w};
            #pragma unroll
            for (int p = 0; p < 8; p++) {
                const __nv_fp8x2_storage_t wx2 = *reinterpret_cast<const __nv_fp8x2_storage_t*>(&w8[p * 2]);
                const float2 wf = __half22float2(*reinterpret_cast<const __half2*>(&__nv_cvt_fp8x2_to_halfraw2(wx2, __NV_E4M3)));
                acc0 += (wf.x * sc) * xv0[p * 2] + (wf.y * sc) * xv0[p * 2 + 1];
                acc1 += (wf.x * sc) * xv1[p * 2] + (wf.y * sc) * xv1[p * 2 + 1];
            }
            k += 32 * 16;
        }
        #pragma unroll 2
        for (; k + 15 < k1; k += 32 * 16) {
            uint4 wv = *reinterpret_cast<const uint4*>(wr + k);
            const unsigned char* w8 = reinterpret_cast<const unsigned char*>(&wv);
            const float sc = srow[k >> 7];
            const float4 xa0 = *reinterpret_cast<const float4*>(xr0 + k);
            const float4 xb0 = *reinterpret_cast<const float4*>(xr0 + k + 4);
            const float4 xc0 = *reinterpret_cast<const float4*>(xr0 + k + 8);
            const float4 xd0 = *reinterpret_cast<const float4*>(xr0 + k + 12);
            const __half2 hx0[8] = {__floats2half2_rn(xa0.x, xa0.y), __floats2half2_rn(xa0.z, xa0.w),
                                    __floats2half2_rn(xb0.x, xb0.y), __floats2half2_rn(xb0.z, xb0.w),
                                    __floats2half2_rn(xc0.x, xc0.y), __floats2half2_rn(xc0.z, xc0.w),
                                    __floats2half2_rn(xd0.x, xd0.y), __floats2half2_rn(xd0.z, xd0.w)};
            const float4 xa1 = *reinterpret_cast<const float4*>(xr1 + k);
            const float4 xb1 = *reinterpret_cast<const float4*>(xr1 + k + 4);
            const float4 xc1 = *reinterpret_cast<const float4*>(xr1 + k + 8);
            const float4 xd1 = *reinterpret_cast<const float4*>(xr1 + k + 12);
            const __half2 hx1[8] = {__floats2half2_rn(xa1.x, xa1.y), __floats2half2_rn(xa1.z, xa1.w),
                                    __floats2half2_rn(xb1.x, xb1.y), __floats2half2_rn(xb1.z, xb1.w),
                                    __floats2half2_rn(xc1.x, xc1.y), __floats2half2_rn(xc1.z, xc1.w),
                                    __floats2half2_rn(xd1.x, xd1.y), __floats2half2_rn(xd1.z, xd1.w)};
            // half2 FMA path (see gemv header): 8 cvt + 8 __hfma2 per 16 values
            // per token, but the 8 cvt are SHARED by both tokens.
            // TWO half2 chains per token: a single 8-deep hfma2 chain is a
            // 32-cycle dependency; splitting it halves the latency. Same fp16
            // chunk accumulation, reassociated.
            __half2 a20 = __float2half2_rn(0.f), a21 = __float2half2_rn(0.f);
            __half2 b20 = __float2half2_rn(0.f), b21 = __float2half2_rn(0.f);
            #pragma unroll
            for (int p = 0; p < 8; p += 2) {
                const __nv_fp8x2_storage_t wx2 = *reinterpret_cast<const __nv_fp8x2_storage_t*>(&w8[p * 2]);
                const __half2_raw wraw = __nv_cvt_fp8x2_to_halfraw2(wx2, __NV_E4M3);
                const __half2 w2 = *reinterpret_cast<const __half2*>(&wraw);
                const __nv_fp8x2_storage_t wy2 = *reinterpret_cast<const __nv_fp8x2_storage_t*>(&w8[(p + 1) * 2]);
                const __half2_raw wraw2 = __nv_cvt_fp8x2_to_halfraw2(wy2, __NV_E4M3);
                const __half2 w3 = *reinterpret_cast<const __half2*>(&wraw2);
                a20 = __hfma2(w2, hx0[p], a20);
                a21 = __hfma2(w2, hx1[p], a21);
                b20 = __hfma2(w3, hx0[p + 1], b20);
                b21 = __hfma2(w3, hx1[p + 1], b21);
            }
            const __half2 s0 = __hadd2(a20, b20), s1 = __hadd2(a21, b21);
            acc0 += (__half2float(s0.x) + __half2float(s0.y)) * sc;
            acc1 += (__half2float(s1.x) + __half2float(s1.y)) * sc;
        }
        for (; k < k1; k++) {
            const float sc = srow[k >> 7];
            const float wf8 = __half2float(__nv_cvt_fp8_to_halfraw(wr[k], __NV_E4M3)) * sc;
            acc0 += wf8 * xr0[k];
            acc1 += wf8 * xr1[k];
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc0 += __shfl_down_sync(0xffffffff, acc0, off);
        acc1 += __shfl_down_sync(0xffffffff, acc1, off);
    }
    if (lane == 0) { part[warp] = acc0; part[16 + warp] = acc1; }
    __syncthreads();
    if (warp % WPR == 0 && lane == 0) {
        float s0 = 0.f, s1 = 0.f;
        #pragma unroll
        for (int j = 0; j < WPR; j++) {
            s0 += part[(warp / WPR) * WPR + j];
            s1 += part[16 + (warp / WPR) * WPR + j];
        }
        const float b = bias ? bias[row] : 0.f;
        y[(size_t)t0 * out_f + row] = b + s0;
        if (has1) y[(size_t)(t0 + 1) * out_f + row] = b + s1;
    }
    __syncthreads();
    }
}

// fp8 TRI fusion: three SAME-INPUT weight matrices in ONE launch (the GDN's
// b_proj/f_a/g_a and the DSA's wk/weights_proj/gate are 3 launches each at
// n>1 today; the bf16 gemv_tri_dev is n==1-only and its fp8 guard falls back
// to 3 launches). Rows [0,o1)->w1/y1, [o1,o1+o2)->w2/y2, rest->w3/y3, with a
// PER-ROW matrix selection so the R=8 row group may span a boundary.
// The three phases mirror gemv_fp8_v2 exactly (register-preloaded first
// k-step, half2 main loop, scalar tail, shuffle+smem reduce), so every output
// row is bit-identical to the corresponding separate gemv_fp8_v2 call.
template <int WPR>
__global__ void __launch_bounds__(256, 4) gemv_fp8_tri_kernel(
    const float* __restrict__ x,
    const unsigned char* __restrict__ w1, const float* __restrict__ s1,
    const unsigned char* __restrict__ w2, const float* __restrict__ s2,
    const unsigned char* __restrict__ w3, const float* __restrict__ s3,
    float* __restrict__ y1, float* __restrict__ y2, float* __restrict__ y3,
    int in_f, int o1, int o2, int o3, int nrows, int scols) {
    const int T = o1 + o2 + o3;
    const int warps = blockDim.x >> 5;
    const int rpb = warps / WPR;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int R = ((T & 7) == 0) ? 8 : 1;
    const int rowg0 = (blockIdx.x * rpb + warp / WPR) * R;
    const int kw = warp % WPR;
    const int kper = ((in_f + WPR - 1) / WPR + 15) & ~15;
    const int k0 = kw * kper;
    const int k1 = min(k0 + kper, in_f);
    __shared__ float part[32];
    const int t0 = blockIdx.y * 2;
    const bool has1 = (t0 + 1) < nrows;
    float4 xc[2][4];
    #pragma unroll
    for (int tt = 0; tt < 2; tt++) {
        const int kk = k0 + lane * 16;
        if (t0 + tt < nrows && kk + 15 < k1) {
            const float* xr0 = x + (size_t)(t0 + tt) * in_f + kk;
            xc[tt][0] = *reinterpret_cast<const float4*>(xr0);
            xc[tt][1] = *reinterpret_cast<const float4*>(xr0 + 4);
            xc[tt][2] = *reinterpret_cast<const float4*>(xr0 + 8);
            xc[tt][3] = *reinterpret_cast<const float4*>(xr0 + 12);
        } else {
            xc[tt][0] = xc[tt][1] = xc[tt][2] = xc[tt][3] = make_float4(0.f, 0.f, 0.f, 0.f);
        }
    }
    #pragma unroll
    for (int r = 0; r < R; r++) {
        const int row = rowg0 + r;
        if (row >= T) continue;
        const unsigned char* wsel; const float* ssel; float* ysel; int row_l; int on;
        if (row < o1) { wsel = w1; ssel = s1; ysel = y1; row_l = row; on = o1; }
        else if (row < o1 + o2) { wsel = w2; ssel = s2; ysel = y2; row_l = row - o1; on = o2; }
        else { wsel = w3; ssel = s3; ysel = y3; row_l = row - o1 - o2; on = o3; }
        float acc0 = 0.f, acc1 = 0.f;
        {
            const unsigned char* wr = wsel + (size_t)row_l * in_f;
            const float* xr0 = x + (size_t)t0 * in_f;
            const float* xr1 = x + (size_t)(t0 + 1) * in_f;
            const float* srow = ssel + (size_t)(row_l >> 7) * scols;
            int k = k0 + lane * 16;
            if (k + 15 < k1) {
                uint4 wv;
                asm volatile("ld.global.nc.L2::128B.v4.u32 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(wv.x), "=r"(wv.y), "=r"(wv.z), "=r"(wv.w)
                             : "l"(wr + k));
                const unsigned char* w8 = reinterpret_cast<const unsigned char*>(&wv);
                const float sc = srow[k >> 7];
                const float xv0[16] = {xc[0][0].x, xc[0][0].y, xc[0][0].z, xc[0][0].w,
                                       xc[0][1].x, xc[0][1].y, xc[0][1].z, xc[0][1].w,
                                       xc[0][2].x, xc[0][2].y, xc[0][2].z, xc[0][2].w,
                                       xc[0][3].x, xc[0][3].y, xc[0][3].z, xc[0][3].w};
                const float xv1[16] = {xc[1][0].x, xc[1][0].y, xc[1][0].z, xc[1][0].w,
                                       xc[1][1].x, xc[1][1].y, xc[1][1].z, xc[1][1].w,
                                       xc[1][2].x, xc[1][2].y, xc[1][2].z, xc[1][2].w,
                                       xc[1][3].x, xc[1][3].y, xc[1][3].z, xc[1][3].w};
                #pragma unroll
                for (int p = 0; p < 8; p++) {
                    const __nv_fp8x2_storage_t wx2 = *reinterpret_cast<const __nv_fp8x2_storage_t*>(&w8[p * 2]);
                    const float2 wf = __half22float2(*reinterpret_cast<const __half2*>(&__nv_cvt_fp8x2_to_halfraw2(wx2, __NV_E4M3)));
                    acc0 += (wf.x * sc) * xv0[p * 2] + (wf.y * sc) * xv0[p * 2 + 1];
                    acc1 += (wf.x * sc) * xv1[p * 2] + (wf.y * sc) * xv1[p * 2 + 1];
                }
                k += 32 * 16;
            }
            #pragma unroll 2
            for (; k + 15 < k1; k += 32 * 16) {
                uint4 wv = *reinterpret_cast<const uint4*>(wr + k);
                const unsigned char* w8 = reinterpret_cast<const unsigned char*>(&wv);
                const float sc = srow[k >> 7];
                const float4 xa0 = *reinterpret_cast<const float4*>(xr0 + k);
                const float4 xb0 = *reinterpret_cast<const float4*>(xr0 + k + 4);
                const float4 xc0 = *reinterpret_cast<const float4*>(xr0 + k + 8);
                const float4 xd0 = *reinterpret_cast<const float4*>(xr0 + k + 12);
                const __half2 hx0[8] = {__floats2half2_rn(xa0.x, xa0.y), __floats2half2_rn(xa0.z, xa0.w),
                                        __floats2half2_rn(xb0.x, xb0.y), __floats2half2_rn(xb0.z, xb0.w),
                                        __floats2half2_rn(xc0.x, xc0.y), __floats2half2_rn(xc0.z, xc0.w),
                                        __floats2half2_rn(xd0.x, xd0.y), __floats2half2_rn(xd0.z, xd0.w)};
                const float4 xa1 = *reinterpret_cast<const float4*>(xr1 + k);
                const float4 xb1 = *reinterpret_cast<const float4*>(xr1 + k + 4);
                const float4 xc1 = *reinterpret_cast<const float4*>(xr1 + k + 8);
                const float4 xd1 = *reinterpret_cast<const float4*>(xr1 + k + 12);
                const __half2 hx1[8] = {__floats2half2_rn(xa1.x, xa1.y), __floats2half2_rn(xa1.z, xa1.w),
                                        __floats2half2_rn(xb1.x, xb1.y), __floats2half2_rn(xb1.z, xb1.w),
                                        __floats2half2_rn(xc1.x, xc1.y), __floats2half2_rn(xc1.z, xc1.w),
                                        __floats2half2_rn(xd1.x, xd1.y), __floats2half2_rn(xd1.z, xd1.w)};
                __half2 a20 = __float2half2_rn(0.f), a21 = __float2half2_rn(0.f);
                __half2 b20 = __float2half2_rn(0.f), b21 = __float2half2_rn(0.f);
                #pragma unroll
                for (int p = 0; p < 8; p += 2) {
                    const __nv_fp8x2_storage_t wx2 = *reinterpret_cast<const __nv_fp8x2_storage_t*>(&w8[p * 2]);
                    const __half2_raw wraw = __nv_cvt_fp8x2_to_halfraw2(wx2, __NV_E4M3);
                    const __half2 w2 = *reinterpret_cast<const __half2*>(&wraw);
                    const __nv_fp8x2_storage_t wy2 = *reinterpret_cast<const __nv_fp8x2_storage_t*>(&w8[(p + 1) * 2]);
                    const __half2_raw wraw2 = __nv_cvt_fp8x2_to_halfraw2(wy2, __NV_E4M3);
                    const __half2 w3 = *reinterpret_cast<const __half2*>(&wraw2);
                    a20 = __hfma2(w2, hx0[p], a20);
                    a21 = __hfma2(w2, hx1[p], a21);
                    b20 = __hfma2(w3, hx0[p + 1], b20);
                    b21 = __hfma2(w3, hx1[p + 1], b21);
                }
                const __half2 s0 = __hadd2(a20, b20), s1 = __hadd2(a21, b21);
                acc0 += (__half2float(s0.x) + __half2float(s0.y)) * sc;
                acc1 += (__half2float(s1.x) + __half2float(s1.y)) * sc;
            }
            for (; k < k1; k++) {
                const float sc = srow[k >> 7];
                const float wf8 = __half2float(__nv_cvt_fp8_to_halfraw(wr[k], __NV_E4M3)) * sc;
                acc0 += wf8 * xr0[k];
                acc1 += wf8 * xr1[k];
            }
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            acc0 += __shfl_down_sync(0xffffffff, acc0, off);
            acc1 += __shfl_down_sync(0xffffffff, acc1, off);
        }
        if (lane == 0) { part[warp] = acc0; part[16 + warp] = acc1; }
        __syncthreads();
        if (warp % WPR == 0 && lane == 0) {
            float s0 = 0.f, s1 = 0.f;
            #pragma unroll
            for (int j = 0; j < WPR; j++) {
                s0 += part[(warp / WPR) * WPR + j];
                s1 += part[16 + (warp / WPR) * WPR + j];
            }
            ysel[(size_t)t0 * on + row_l] = s0;
            if (has1) ysel[(size_t)(t0 + 1) * on + row_l] = s1;
        }
        __syncthreads();
    }
}

extern "C" cudaError_t ferrite_gemv_fp8_tri(
    const float* x,
    const void* w1, const float* s1, const void* w2, const float* s2,
    const void* w3, const float* s3,
    float* y1, float* y2, float* y3,
    int in_f, int o1, int o2, int o3, int nrows, int scols, cudaStream_t s) {
    if (in_f <= 0 || nrows <= 0) return cudaSuccess;
    constexpr int WPR = 8;
    const int rpb = 256 / 32 / WPR;
    const int T = o1 + o2 + o3;
    const int Rl = ((T & 7) == 0) ? 8 : 1;
    dim3 grid((unsigned)((T + rpb * Rl - 1) / (rpb * Rl)),
              (unsigned)((nrows + 1) / 2), 1);
    gemv_fp8_tri_kernel<WPR><<<grid, 256, 0, s>>>(
        x, (const unsigned char*)w1, s1, (const unsigned char*)w2, s2,
        (const unsigned char*)w3, s3, y1, y2, y3, in_f, o1, o2, o3, nrows, scols);
    return cudaGetLastError();
}

// CUTLASS-style fp8 MMA gemv (tensor core), modeled on the proven
// moe_fused_act_fp8_mma structure: M = n<=16 tokens, N = 8 output rows per
// warp-group, K = in_f split across the block's 8 warps, 3-stage cp.async
// pipeline, ldmatrix.x4 for A, manual B fragment from per-warp smem tiles
// (no cross-warp races). grid = ceil(out_f/8) = 192 blocks at out_f=1536.
// Weights are read as native e4m3 (1 byte) — half the bytes of the bf16 SIMT
// gemv — and the MMA replaces ~2048 serial FMA per lane with tensor ops.
template <int UNUSED_WPR>
__global__ void __launch_bounds__(256, 3) gemv_fp8_mma_b16_kernel(
    const unsigned char* __restrict__ xq,   // [n<=16, in_f] e4m3
    const float* __restrict__ xs,           // [n] per-token scales
    const unsigned char* __restrict__ w,    // [out_f, in_f] e4m3
    const float* __restrict__ ws,           // [out_f/128, scols]
    float* __restrict__ out,                // [n, out_f]
    int n, int in_f, int out_f, int scols) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row0 = blockIdx.x * 8;
    if (row0 >= out_f) return;
    const int c0 = (lane & 3) * 4;
    const int kper = ((in_f + 7) / 8 + 63) & ~63;   // per-warp K slice, 64-aligned
    const int k0 = warp * kper;
    const int k1 = min(k0 + kper, in_f);
    // per-warp tiles, 3-deep (mirrors the act kernel's sa[3][8][...] layout)
    __shared__ unsigned char sx[3][8][16][80];      // [stage][warp][tok][K]
    __shared__ unsigned char sw[3][8][8][80];       // [stage][warp][row][K]
    __shared__ float red[8][32][4];
    #define GMMA_ISSUE(KB, ST) do { \
        { \
            const int t = lane >> 1, half_ = lane & 1; \
            const unsigned char* src_ = xq + (size_t)t * in_f + (KB) + half_ * 32; \
            unsigned char* dst_ = sx[ST][warp][t] + half_ * 32; \
            asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" \
                         :: "r"((unsigned)__cvta_generic_to_shared(dst_)), "l"(src_)); \
            asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" \
                         :: "r"((unsigned)__cvta_generic_to_shared(dst_ + 16)), "l"(src_ + 16)); \
        } \
        { \
            const int r = lane >> 2, q = lane & 3; \
            const unsigned char* src_ = w + (size_t)(row0 + r) * in_f + (KB) + q * 16; \
            unsigned char* dst_ = sw[ST][warp][r] + q * 16; \
            asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" \
                         :: "r"((unsigned)__cvta_generic_to_shared(dst_)), "l"(src_)); \
        } \
        asm volatile("cp.async.commit_group;\n"); \
    } while (0)
    GMMA_ISSUE(k0, 0);
    if (k0 + 64 < k1) GMMA_ISSUE(k0 + 64, 1);
    if (k0 + 128 < k1) GMMA_ISSUE(k0 + 128, 2);
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    int st = 0;
    for (int kb = k0; kb < k1; kb += 64, st = (st + 1) % 3) {
        if (kb + 192 < k1) asm volatile("cp.async.wait_group 2;\n");
        else if (kb + 128 < k1) asm volatile("cp.async.wait_group 1;\n");
        else asm volatile("cp.async.wait_group 0;\n");
        __syncwarp();
        #pragma unroll
        for (int kk = 0; kk < 64; kk += 32) {
            const unsigned saddr_a = (unsigned)__cvta_generic_to_shared(
                sx[st][warp][0] + (size_t)(lane & 15) * 80 + kk + ((lane >> 4) * 16));
            unsigned a[4];
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                         : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(saddr_a));
            unsigned b[2];
            b[0] = *(const unsigned*)(sw[st][warp][(lane >> 2)] + kk + c0);
            b[1] = *(const unsigned*)(sw[st][warp][(lane >> 2)] + kk + c0 + 16);
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
        }
        if (kb + 192 < k1) GMMA_ISSUE(kb + 192, st);
    }
    #undef GMMA_ISSUE
    // weight block scale: 8 rows of a warp share one 128-row block only if
    // row0%128 <= 120 (row0 is 8-aligned, so always true)
    const float wsc = ws[(size_t)(row0 >> 7) * scols + ((k0) >> 7)];
    red[warp][lane][0] = acc[0] * wsc; red[warp][lane][1] = acc[1] * wsc;
    red[warp][lane][2] = acc[2] * wsc; red[warp][lane][3] = acc[3] * wsc;
    __syncthreads();
    if (warp == 0) {
        float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
        #pragma unroll
        for (int u = 0; u < 8; u++) {
            s0 += red[u][lane][0]; s1 += red[u][lane][1];
            s2 += red[u][lane][2]; s3 += red[u][lane][3];
        }
        const int m0_ = lane >> 2, n0_ = (lane & 3) * 2;
        if (m0_ < n) {
            out[(size_t)m0_ * out_f + row0 + n0_] = s0 * xs[m0_];
            out[(size_t)m0_ * out_f + row0 + n0_ + 1] = s1 * xs[m0_];
        }
        if (m0_ + 8 < n) {
            out[(size_t)(m0_ + 8) * out_f + row0 + n0_] = s2 * xs[m0_ + 8];
            out[(size_t)(m0_ + 8) * out_f + row0 + n0_ + 1] = s3 * xs[m0_ + 8];
        }
    }
}

extern "C" cudaError_t ferrite_gemv_fp8_mma_b16(
    const unsigned char* xq, const float* xs,
    const unsigned char* w, const float* ws,
    float* out, int n, int in_f, int out_f, int scols, cudaStream_t s) {
    if (n <= 0 || n > 16 || (out_f & 7) != 0 || in_f <= 0) return cudaErrorInvalidValue;
    dim3 grid((unsigned)((out_f + 7) / 8));
    gemv_fp8_mma_b16_kernel<0><<<grid, 256, 0, s>>>(xq, xs, w, ws, out, n, in_f, out_f, scols);
    return cudaGetLastError();
}

extern "C" cudaError_t ferrite_gemv_fp8_v2(const float* x, const void* w,
                                          const float* scale, const float* bias,
                                          float* out, int in_f, int out_f,
                                          int nrows, int srows, int scols,
                                          cudaStream_t s) {
    // DIAGNOSTIC ONLY (FERRITE_GEMV_SKIP=1): timing-only ablation.
    static const bool gemv_skip_ = getenv("FERRITE_GEMV_SKIP") != nullptr;
    if (gemv_skip_) return cudaSuccess;

    if (out_f <= 0 || nrows <= 0 || in_f <= 0) return cudaSuccess;
    // WPR=8: the K-split doubles the block count and halves the per-warp K
    // slice. Micro-bench (kernels/cuda/gemv_bench.cu, n=16): q_a 46->42us,
    // lm_head 489->471us, n=1 12->8us. WPR=2 is worse (65us), WPR=16 invalid.
    constexpr int WPR = 8;                 // K-split warps per row
    const int rpb = 256 / 32 / WPR;        // rows per block (8 warps / 4)
    // grid.x = row tiles (R synced with the kernel: R=1 when out_f%8 != 0),
    // grid.y = TOKEN PAIRS (the kernel handles 2 tokens per block).
    const int Rl = ((out_f & 7) == 0) ? 8 : 1;
    dim3 grid((unsigned)((out_f + rpb * Rl - 1) / (rpb * Rl)),
              (unsigned)((nrows + 1) / 2), 1);
    dim3 block(256);
    gemv_fp8_v2_kernel<WPR><<<grid, block, 0, s>>>(x, (const unsigned char*)w, scale, bias, out,
                                                   in_f, out_f, nrows, srows, scols);
    return cudaGetLastError();
}

// ==== fp8 mma layout probe (W8A8 feasibility): m16n8k32 e4m3 (sm_90+) =====
// A[16,32] fp8 row-major, B[32,8] fp8 col-major(k,n), C[16,8] f32.
// Encodes A[i][k]=fp8((i*32+k)/448), B[k][n]=fp8((k*8+n+1)/448) so C[i][n]=
// Σ_k (i*32+k)/448*(k*8+n+1)/448 — reading C pins the PTX fragment layout.
__global__ void fp8_mma_probe_kernel(const unsigned char* __restrict__ A,
                                     const unsigned char* __restrict__ B,
                                     float* __restrict__ C) {
    const int t = threadIdx.x & 31;
    const int r0 = t >> 2, c0 = (t & 3) * 4;
    // m16n8k32 e4m3 fragments (PTX spec): A = 4 .b32/thread (16 fp8),
    // B = 2 .b32/thread (8 fp8), C = 4 f32/thread.
    // A row-major [m,k]: a0=(r0, k=c0..+3), a1=(r0+8, c0), a2=(r0, c0+16), a3=(r0+8, c0+16)
    // B col-major [k,n] (n-major stride 32, k contiguous): b0=(k=c0..+3, n=r0), b1=(k=c0+16, n=r0)
    // C m16n8 classic: c0/c1=(r0, cc*2/+1), c2/c3=(r0+8, cc*2/+1)
    unsigned a[4];
    a[0] = *(const unsigned*)(A + r0 * 32 + c0);
    a[1] = *(const unsigned*)(A + (r0 + 8) * 32 + c0);
    a[2] = *(const unsigned*)(A + r0 * 32 + c0 + 16);
    a[3] = *(const unsigned*)(A + (r0 + 8) * 32 + c0 + 16);
    unsigned b[2];
    b[0] = *(const unsigned*)(B + r0 * 32 + c0);
    b[1] = *(const unsigned*)(B + r0 * 32 + c0 + 16);
    float c0f = 0.f, c1f = 0.f, c2f = 0.f, c3f = 0.f;
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c0f), "+f"(c1f), "+f"(c2f), "+f"(c3f)
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]));
    const int cr = (t >> 2), cc = (t & 3) * 2;
    C[(cr) * 8 + cc] = c0f;
    C[(cr) * 8 + cc + 1] = c1f;
    C[(cr + 8) * 8 + cc] = c2f;
    C[(cr + 8) * 8 + cc + 1] = c3f;
}

extern "C" cudaError_t ferrite_fp8_mma_probe(const unsigned char* A, const unsigned char* B,
                                             float* C, cudaStream_t s) {
    fp8_mma_probe_kernel<<<1, 32, 0, s>>>(A, B, C);
    return cudaGetLastError();
}

// ============================================================
// gemv_fp8_mma (W8A8): tensor-core fp8 GEMV — activations quantized to
// e4m3 IN-KERNEL (per-token absmax/448, sglang per_token_group_quant
// semantics), weights already e4m3 + 128x128 block scales, and the dot runs
// on mma.sync.m16n8k32.e4m3 (sm_90+ tensor core). This is the true W8A8
// path (dequant-free compute: the mma multiplies e4m3 x e4m3 directly — no
// per-element cvt-to-float, which the W8A16 attempt showed offsets the fp8
// HBM savings: 0.96x vs bf16).
//
// Mapping (decode gemv n=1): M = 16 output rows per block, N = 8 (x
// replicated — B(k,n) = x_q[k] for all n), K = in_f stepped 32 per mma.
// Per k128 block (4 mmas) the f32 partials scale by w_scale[m/128][k/128]
// (block-quantized dot), accumulated across K-warps in smem. x_scale applied
// once at the end. Layout verified by fp8_mma_layout_probe (0-diff).
//
// Structure mirrors gemv_bf16_v2 (K-split warps; here 8 warps split K per
// 16-row block, block reduce via smem). The bf16 and W8A16 kernels stay as
// fallbacks (fp8-registered weights with misaligned shapes serve bf16).
// Requires: in_f % 128 == 0 and out_f % 16 == 0 (GLM weights all comply).
// ============================================================
// ============================================================
// gemv_fp8_mma v1 (W8A8, launcher DEFAULT): per-block independent quant —
// every block quantizes x into ITS OWN shared memory (fully parallel, no
// barrier), then mma reads the smem xq. Redundant work (grid x quant) but
// NO synchronization: beats the v3 cooperative barrier on EVERY shape
// (v3.1: small 0.28x / large 0.52x vs v1 0.60x / 1.23x — the spin-pass
// atomics of ~8.5k non-participant blocks serialize on one L2 address
// ~250us, and the two-round votes among 1.2k blocks cost ~70us more).
// v1 wins the HBM-bound large shapes 1.23x over bf16 (185us vs 227us on
// lm_head 154880x4096). smem: [xq e4m3 |in_f|][reduce 256][xs 1][sacc 8x16].
// ============================================================
__global__ void gemv_fp8_mma_kernel(
    const float* __restrict__ x,          // [in_f] f32 (n=1 decode row)
    const unsigned char* __restrict__ w,  // [out_f, in_f] e4m3 row-major
    const float* __restrict__ w_scale,    // [srows][scols] = [out/128][in/128]
    float* __restrict__ y,                // [out_f]
    int in_f, int out_f, int srows, int scols)
{
    (void)srows;
    const int m0 = blockIdx.x * 16;        // 16 output rows per block
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r0 = lane >> 2, c0 = (lane & 3) * 4; // probe-verified fragments
    extern __shared__ unsigned char smem[];
    unsigned char* sx = smem;                       // [in_f] quantized x (e4m3)
    float* sred = (float*)(smem + in_f);            // [256] absmax reduce
    float* sxs = (float*)(smem + in_f + 256 * 4);   // [1] x_scale
    float* sacc = sxs + 1;                          // [8][16] warp partials
    // ---- 1. per-block quantize (NO cross-block barrier — this is v1's win) ----
    {
        float amax = 1e-9f;
        for (int k = threadIdx.x; k < in_f; k += 256)
            amax = fmaxf(amax, fabsf(x[k]));
        // v2: warp-shuffle max + 2-level smem (was a 128→1 tree with one
        // __syncthreads PER level — 8 serialized barriers per block).
        // The 8-warp final max is SERIAL on thread 0: a __shfl_down_sync
        // inside an if(tid<8) branch is a full-mask shuffle with 8/32 lanes
        // present = warp deadlock on sm_103a.
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            amax = fmaxf(amax, __shfl_down_sync(0xffffffff, amax, off));
        if ((threadIdx.x & 31) == 0) sred[threadIdx.x >> 5] = amax;
        __syncthreads();
        if (threadIdx.x == 0) {
            float m = sred[0];
            #pragma unroll
            for (int w = 1; w < 8; w++) m = fmaxf(m, sred[w]);
            sxs[0] = m / 448.0f;
        }
        __syncthreads();
        const float inv = 1.0f / sxs[0];
        for (int k = threadIdx.x; k < in_f; k += 256) {
            const float q = fminf(fmaxf(x[k] * inv, -448.0f), 448.0f);
            sx[k] = (unsigned char)__nv_cvt_float_to_fp8(q, __NV_SATFINITE, __NV_E4M3);
        }
        __syncthreads();
    }
    // ---- 2. mma body (reads smem xq; fragments per fp8_mma_layout_probe) ----
    const int nblk = (in_f + 127) >> 7;
    const int bseg = (nblk + 7) / 8;
    const int kW = warp;
    const int k0 = kW * bseg * 128;
    const int k1 = min(k0 + bseg * 128, in_f);
    const int ws_row = (m0 / 128) * scols;
    float acc0 = 0.f, acc1 = 0.f;
    for (int kb = k0; kb < k1; kb += 128) {
        float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
        for (int kk = kb; kk < kb + 128; kk += 32) {
            if (kk + 32 > k1) break;
            unsigned a[4];
            a[0] = *(const unsigned*)(w + (size_t)(m0 + r0) * in_f + kk + c0);
            a[1] = *(const unsigned*)(w + (size_t)(m0 + r0 + 8) * in_f + kk + c0);
            a[2] = *(const unsigned*)(w + (size_t)(m0 + r0) * in_f + kk + c0 + 16);
            a[3] = *(const unsigned*)(w + (size_t)(m0 + r0 + 8) * in_f + kk + c0 + 16);
            unsigned b[2];
            b[0] = *(const unsigned*)(sx + kk + c0);
            b[1] = *(const unsigned*)(sx + kk + c0 + 16);
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
                  "r"(b[0]), "r"(b[1]));
        }
        const float wsc = w_scale[ws_row + (kb >> 7)];
        if ((lane & 3) == 0) { acc0 += d0 * wsc; acc1 += d2 * wsc; }
    }
    if ((lane & 3) == 0) {
        sacc[warp * 16 + r0] = acc0;
        sacc[warp * 16 + r0 + 8] = acc1;
    }
    __syncthreads();
    if (warp == 0 && lane < 16) {
        float t = 0.f;
        for (int i = 0; i < 8; i++) t += sacc[i * 16 + lane];
        y[m0 + lane] = t * sxs[0];
    }
}

// v3.1 (user-directed grid fix): cooperative quant on the first co_res
// blocks ONLY (spin barrier among co-resident participants — deadlock-free),
// the mma stage runs FULLY PARALLEL (grid = out_f/16, no row loop — the
// earlier row-loop cap left 86% of the GPU idle on lm_head, 0.54x). Blocks
// >= co_res start after earlier ones retire (SM occupancy order), spin-pass
// the ready flag once, and join the mma directly. Tail: block 0 waits on a
// full-grid completion vote (cnt3 — non-resident blocks only ADD, never
// spin, so no deadlock) then resets the barrier state for the next call
// (stream order serializes callers).
__global__ void gemv_fp8_mma_v3_kernel(
    const float* __restrict__ x,          // [in_f] f32 activations
    const unsigned char* __restrict__ w,   // [out_f, in_f] e4m3 row-major
    const float* __restrict__ w_scale,    // [srows][scols]
    float* __restrict__ y,                 // [out_f]
    unsigned int* __restrict__ scratch,    // [amax(int bits), cnt, cnt2, cnt3, xs, xq]
    int in_f, int out_f, int scols, int co_res)
{
    const int nb = gridDim.x;              // = out_f/16 (FULL parallel)
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int* g_amax = (int*)scratch;            // float-as-int atomicMax (abs >= 0)
    unsigned int* g_cnt = scratch + 1;
    unsigned int* g_cnt2 = scratch + 2;
    unsigned int* g_cnt3 = scratch + 3;
    float* g_xs = (float*)(scratch + 4);
    unsigned char* xq = (unsigned char*)(scratch + 6);

    // ---- phase A+B: first co_res blocks cooperatively quantize ----
    if (blockIdx.x < (unsigned)co_res) {
        float pmax = 1e-9f;
        for (int k = (blockIdx.x * 256) + threadIdx.x; k < in_f; k += co_res * 256)
            pmax = fmaxf(pmax, fabsf(x[k]));
        for (int off = 16; off > 0; off >>= 1)
            pmax = fmaxf(pmax, __shfl_down_sync(0xffffffff, pmax, off));
        if (lane == 0) atomicMax(g_amax, __float_as_int(pmax));
        if (threadIdx.x == 0) atomicAdd(g_cnt, 1u);
        if (threadIdx.x == 0) { while (atomicAdd(g_cnt, 0u) < (unsigned)co_res) __nanosleep(32); }
        __syncthreads();
        const float xs_ = __int_as_float(atomicAdd(g_amax, 0)) / 448.0f;
        if (threadIdx.x == 0 && blockIdx.x == 0) *g_xs = xs_;
        const float inv = 1.0f / xs_;
        for (int k = (blockIdx.x * 256) + threadIdx.x; k < in_f; k += co_res * 256) {
            const float q = fminf(fmaxf(x[k] * inv, -448.0f), 448.0f);
            xq[k] = (unsigned char)__nv_cvt_float_to_fp8(q, __NV_SATFINITE, __NV_E4M3);
        }
        __threadfence();
        if (threadIdx.x == 0) atomicAdd(g_cnt2, 1u);
        if (threadIdx.x == 0) { while (atomicAdd(g_cnt2, 0u) < (unsigned)co_res) __nanosleep(32); }
        __threadfence();
        __syncthreads();
    } else {
        // later waves: launched after earlier blocks retire; the quant is
        // long done — one volatile check passes immediately.
        if (threadIdx.x == 0) { while (atomicAdd(g_cnt2, 0u) < (unsigned)co_res) __nanosleep(32); }
        __syncthreads();
    }
    const float xs_ = *g_xs;

    // ---- mma stage: FULLY parallel, one 16-row block per block (no loop) ----
    const int m0 = blockIdx.x * 16;
    const int r0 = lane >> 2, c0 = (lane & 3) * 4;
    __shared__ float sacc[8 * 16];
    const int nblk = (in_f + 127) >> 7;
    const int bseg = (nblk + 7) / 8;
    const int kW = warp;
    const int k0 = kW * bseg * 128;
    const int k1 = min(k0 + bseg * 128, in_f);
    const int ws_row = (m0 / 128) * scols;
    float acc0 = 0.f, acc1 = 0.f;
    for (int kb = k0; kb < k1; kb += 128) {
        float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
        for (int kk = kb; kk < kb + 128; kk += 32) {
            if (kk + 32 > k1) break;
            unsigned a[4];
            a[0] = *(const unsigned*)(w + (size_t)(m0 + r0) * in_f + kk + c0);
            a[1] = *(const unsigned*)(w + (size_t)(m0 + r0 + 8) * in_f + kk + c0);
            a[2] = *(const unsigned*)(w + (size_t)(m0 + r0) * in_f + kk + c0 + 16);
            a[3] = *(const unsigned*)(w + (size_t)(m0 + r0 + 8) * in_f + kk + c0 + 16);
            unsigned b[2];
            b[0] = *(const unsigned*)(xq + kk + c0);
            b[1] = *(const unsigned*)(xq + kk + c0 + 16);
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
                  "r"(b[0]), "r"(b[1]));
        }
        const float wsc = w_scale[ws_row + (kb >> 7)];
        if ((lane & 3) == 0) { acc0 += d0 * wsc; acc1 += d2 * wsc; }
    }
    if ((lane & 3) == 0) {
        sacc[warp * 16 + r0] = acc0;
        sacc[warp * 16 + r0 + 8] = acc1;
    }
    __syncthreads();
    if (warp == 0 && lane < 16) {
        float t = 0.f;
        for (int i = 0; i < 8; i++) t += sacc[i * 16 + lane];
        y[m0 + lane] = t * xs_;
    }
    // ---- tail: block 0 resets the barrier state AFTER all blocks voted done
    // (non-resident blocks only ADD, never spin — no deadlock; stream order
    // serializes the next launch against this reset). ----
    __threadfence();
    if (threadIdx.x == 0) atomicAdd(g_cnt3, 1u);
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        while (atomicAdd(g_cnt3, 0u) < (unsigned)nb) __nanosleep(32);
        *g_amax = 0;
        *g_cnt = 0;
        *g_cnt2 = 0;
        *g_cnt3 = 0;
    }
}

extern "C" cudaError_t ferrite_gemv_fp8_mma(
    const float* x, const void* w, const float* w_scale,
    float* out, int in_f, int out_f, int srows, int scols,
    unsigned int* scratch, cudaStream_t s)
{
    (void)srows;
    (void)scratch;
    if (out_f % 16 != 0 || in_f % 128 != 0) return cudaErrorNotSupported;
    // v1 dispatch (default): fully-independent per-block quant — v3's
    // cooperative barrier lost on ALL shapes (the ~8.5k non-participant
    // blocks' spin-pass atomics serialize on one L2 address ~250us; the
    // two-round votes among 1184 blocks cost ~70us more). v1's redundant
    // quant is fully parallel and beats the barrier everywhere; it wins the
    // HBM-bound large shapes 1.23x over bf16 (185us vs 227us on lm_head).
    // v3.1 kernel kept above for the record (grid-fix data in the log).
    const int smem = in_f + 256 * 4 + 4 + 8 * 16 * 4;
    cudaFuncSetAttribute(gemv_fp8_mma_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    dim3 grid((unsigned)(out_f / 16));
    gemv_fp8_mma_kernel<<<grid, 256, smem, s>>>(
        x, (const unsigned char*)w, w_scale, out, in_f, out_f, srows, scols);
    return cudaGetLastError();
}

// ============================================================
// embed_expand (host-4ms root-out knife 1): host embed lookup (16KB row
// memcpy) + hc_expand (64-192KB Vec concat) + staging H2D — all folded into
// ONE kernel that reads token ids from a pinned slot (host writes 1-3 u32/f32
// ids — 12B vs 192KB) and writes the graph input res directly (the
// mega-graph's first node becomes this kernel instead of the staging memcpy).
// Table is the resident bf16 embed (dev_weight_bf16 cache).
// ============================================================
// NOTE: the table is F32 (dev_weight f32 cache — the embed output feeds the
// residual stream; a bf16 table changed the numeric domain and cost accept
// 2.38→2.18 (argmax ties flip — the same class as the W8A8 lesson). f32 =
// bit-identical to the host lookup. VRAM +1.2GB for 154880x4096.
__global__ void embed_expand_kernel(
    const float* __restrict__ table,         // [vocab, hidden] resident F32
    const int* __restrict__ ids,              // [n] token ids (pinned or device)
    float* __restrict__ out,                  // [n, mult, hidden] (graph res buf)
    int n, int hidden, int mult, int vocab)
{
    int t = blockIdx.x;                       // one block per token
    if (t >= n) return;
    int id = ids[t];
    if (id < 0 || id >= vocab) id = 0;
    const float* row = table + (size_t)id * hidden;
    float* dst = out + (size_t)t * mult * hidden;
    for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
        float v = row[j];
        for (int m = 0; m < mult; m++) {
            dst[(size_t)m * hidden + j] = v;
        }
    }
}

extern "C" cudaError_t ferrite_embed_expand(
    const void* table, const int* ids, float* out,
    int n, int hidden, int mult, int vocab, cudaStream_t s)
{
    if (n <= 0) return cudaSuccess;
    embed_expand_kernel<<<n, 256, 0, s>>>(
        (const float*)table, ids, out, n, hidden, mult, vocab);
    return cudaGetLastError();
}

// ============================================================
// Device-resident MTP accept kernel (N-UNIFIED, FERRITE_MTP_N): compares
// the n-1 drafts d[0..n-2] vs the verify argmax a[0..n-1] (ALL device
// buffers — the argmax outputs never cross to host), writes k (the
// longest matching prefix length, 1..n) and next_token = a[k-1] to
// device slots. The host reads 8 bytes D2H per step (k + next_token for
// API response + seq tracking) — the ENTIRE decode loop is zero-H2D
// after the initial prompt upload. n=3 (d1,d2,a0,a1,a2) is the default;
// any n: the prefix loop generalizes the old d1/d2 nesting exactly.
// ============================================================
__global__ void mtp_accept_kernel(
    const float* __restrict__ d,        // [n-1] draft argmax (device)
    const float* __restrict__ a,         // [n] verify argmax (device: a0..a_{n-1})
    int* __restrict__ k_out,             // [1] accept count 1..n
    int* __restrict__ next_token,        // [1] next "last" token for the loop
    int* __restrict__ n_accepted,       // [1] tokens accepted this step (host read)
    int n) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int k = 1;
        while (k < n && (int)d[k - 1] == (int)a[k - 1]) k++;
        *k_out = k;
        *next_token = (int)a[k - 1];
        *n_accepted = k;
    }
}

extern "C" cudaError_t ferrite_mtp_accept(
    const float* d, const float* a,
    int* k_out, int* next_token, int* n_accepted,
    int n, cudaStream_t s)
{
    mtp_accept_kernel<<<1, 1, 0, s>>>(d, a, k_out, next_token, n_accepted, n);
    return cudaGetLastError();
}

// ============================================================
// Device token-chain embed: reads token IDs from a DEVICE int buffer
// (the previous step's accept kernel output or the initial prompt H2D)
// and writes the expanded graph input [n, mult, hidden]. This is the
// ZERO-H2D replacement for the host embed lookup + hc_expand + staging
// upload: the token never leaves the device.
// ============================================================
__global__ void embed_expand_dev_kernel(
    const float* __restrict__ table,         // [vocab, hidden] resident F32
    const int* __restrict__ ids_dev,         // [n] token ids (DEVICE buf)
    float* __restrict__ out,                  // [n, mult, hidden] graph input
    int n, int hidden, int mult, int vocab)
{
    int t = blockIdx.x;
    if (t >= n) return;
    int id = ids_dev[t];
    if (id < 0 || id >= vocab) id = 0;
    const float* row = table + (size_t)id * hidden;
    float* dst = out + (size_t)t * mult * hidden;
    for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
        float v = row[j];
        for (int m = 0; m < mult; m++) {
            dst[(size_t)m * hidden + j] = v;
        }
    }
}

extern "C" cudaError_t ferrite_embed_expand_dev(
    const void* table, const int* ids_dev, float* out,
    int n, int hidden, int mult, int vocab, cudaStream_t s)
{
    if (n <= 0) return cudaSuccess;
    embed_expand_dev_kernel<<<n, 256, 0, s>>>(
        (const float*)table, ids_dev, out, n, hidden, mult, vocab);
    return cudaGetLastError();
}

// ============================================================
// Draft embed: reads ONE token from a device int slot (the previous argmax
// or accept output), embeds it (table lookup + MHC expand) into a device
// buffer — replaces the host embed lookup + hc_expand + DevBuf::upload
// (4KB H2D) with a single kernel reading 4B from device.
// ============================================================
__global__ void embed_one_kernel(
    const float* __restrict__ table,         // [vocab, hidden] resident F32
    const int* __restrict__ token_slot,      // [1] device int (the token)
    float* __restrict__ out,                  // [mult * hidden] (hc_expand'd)
    int hidden, int mult, int vocab)
{
    int id = token_slot[0];
    if (id < 0 || id >= vocab) id = 0;
    const float* row = table + (size_t)id * hidden;
    for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
        float v = row[j];
        for (int m = 0; m < mult; m++) {
            out[(size_t)m * hidden + j] = v;
        }
    }
}

extern "C" cudaError_t ferrite_embed_one(
    const void* table, const int* token_slot, float* out,
    int hidden, int mult, int vocab, cudaStream_t s)
{
    embed_one_kernel<<<1, 256, 0, s>>>(
        (const float*)table, token_slot, out, hidden, mult, vocab);
    return cudaGetLastError();
}

// f32 -> i32 cast + store (device-side argmax-to-token-chain link).
// Replaces the D2H(d1)→host cast→H2D(tokens_dev[1]) roundtrip whose
// cudaStreamSynchronize between draft1 and draft2 breaks NCCL AR
// channel state (float reduction order 1-ulp → argmax flips on
// near-ties — d1 diverged from 98347 to 702 with bit-identical x2).
__global__ void cast_store_i32_kernel(const float* src, int* dst) {
    dst[0] = (int)__ldg(src);
}

extern "C" cudaError_t ferrite_cast_store_i32(
    const void* src, void* dst, cudaStream_t s) {
    cast_store_i32_kernel<<<1, 1, 0, s>>>((const float*)src, (int*)dst);
    return cudaGetLastError();
}
