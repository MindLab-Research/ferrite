// DeepSeek-V4.1-Flash MoE routing from pre-computed gate scores.
//
// The released checkpoint stores `ffn.gate.weight` in **bf16** — there is no
// `ffn.gate.scale` — so the gate GEMM runs on the bf16 tensor-core path and the
// routing decision is a separate step. `dsv41_moe_route` cannot serve this: it
// fuses an fp8 gate GEMM the checkpoint does not provide.
//
// Routing maths, exactly as the reference's Gate does it (`noaux_tc`):
//   act    = sqrtsoftplus
//   sel    = act(score) + gate_bias      the correction bias steers SELECTION
//   idx    = top-k(sel, k)               ties -> lower index (deterministic)
//   w      = act(score)[idx]             weights come from the UNBIASED values
//   w      = w / sum(w)                  when norm_topk_prob
//   w      = w * route_scale
//
// One block per row, 256 threads. `hist` (optional) counts assignments per
// expert so a caller can bucket them without a second scan.
//
// STATUS: this kernel is correct for n_experts <= blockDim (verified exactly at
// n_experts=6/topk=3) but picks a DIFFERENT expert set from the CPU reference at
// the production shape (n_experts=384/topk=6: 12/12 assignments differ, and they
// still differ with deliberately well-separated scores, so it is not a
// near-tie/precision artefact). The per-thread second candidate (threads
// 0..n_experts-blockDim-1 handle two experts) is the prime suspect. The earlier
// crash was a separate bug — a `used[]` flag array sized [topk] and indexed by
// the expert id — which is fixed (consumed experts are now marked by writing
// -INFINITY into s_sel).

#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

namespace {

__device__ __forceinline__ float dsv41_act(float v, int score_func) {
    if (score_func == 0) return 1.f / (1.f + expf(-v));   // sigmoid
    if (score_func == 2) {                                // sqrtsoftplus
        const float sp = log1pf(expf(-fabsf(v))) + fmaxf(v, 0.f);  // stable
        return sqrtf(fmaxf(sp, 0.f));
    }
    return v;                                             // identity
}

__global__ void route_topk_kernel(const float* __restrict__ scores,
                                  const float* __restrict__ bias, float* __restrict__ weights,
                                  int32_t* __restrict__ indices, int rows, int n_experts, int topk,
                                  int norm_topk_prob, float route_scale, int score_func) {
    const int r = blockIdx.x;
    if (r >= rows) return;
    // layout: [n_experts] unbiased act | [n_experts] selection score | [topk] pick
    // A consumed expert is marked by writing -INFINITY into s_sel, so no
    // per-expert flag array is needed. (An earlier revision kept a `used` array
    // sized [topk] and indexed it by the expert id — an out-of-bounds smem
    // access that a small test shape happened not to trip.)
    extern __shared__ float sh[];
    float* s_act = sh;
    float* s_sel = s_act + n_experts;
    int* s_pick = (int*)(s_sel + n_experts);

    const float* sr = scores + (size_t)r * n_experts;
    for (int e = threadIdx.x; e < n_experts; e += blockDim.x) {
        const float a = dsv41_act(sr[e], score_func);
        s_act[e] = a;
        s_sel[e] = a + (bias ? bias[e] : 0.f);
    }
    __syncthreads();

    __shared__ float s_bv[32];
    __shared__ int s_bi[32];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
    for (int it = 0; it < topk; ++it) {
        float bv = -INFINITY;
        int bi = n_experts;
        for (int e = threadIdx.x; e < n_experts; e += blockDim.x) {
            const float v = s_sel[e];
            if (v > bv || (v == bv && e < bi)) {
                bv = v;
                bi = e;
            }
        }
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, off);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, off);
            if (ov > bv || (ov == bv && oi < bi)) {
                bv = ov;
                bi = oi;
            }
        }
        if (lane == 0) {
            s_bv[warp] = bv;
            s_bi[warp] = bi;
        }
        __syncthreads();
        if (warp == 0) {
            float v = (lane < nw) ? s_bv[lane] : -INFINITY;
            int i = (lane < nw) ? s_bi[lane] : n_experts;
            for (int off = 16; off > 0; off >>= 1) {
                const float ov = __shfl_xor_sync(0xffffffffu, v, off);
                const int oi = __shfl_xor_sync(0xffffffffu, i, off);
                if (ov > v || (ov == v && oi < i)) {
                    v = ov;
                    i = oi;
                }
            }
            if (lane == 0) s_pick[it] = i;
        }
        __syncthreads();
        if (threadIdx.x == 0 && s_pick[it] < n_experts) {
            s_sel[s_pick[it]] = -INFINITY;  // consume it
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float sum = 0.f;
        for (int t = 0; t < topk; ++t) {
            const int e = s_pick[t];
            sum += (e < n_experts) ? s_act[e] : 0.f;
        }
        for (int t = 0; t < topk; ++t) {
            const int e = s_pick[t];
            const float base = (e < n_experts) ? s_act[e] : 0.f;
            const float w = (norm_topk_prob && sum > 0.f) ? base / sum : base;
            indices[(size_t)r * topk + t] = e;
            weights[(size_t)r * topk + t] = w * route_scale;
        }
    }
}

__global__ void route_hist_kernel(const int32_t* __restrict__ indices, int32_t* __restrict__ hist,
                                  int n_asg, int n_experts) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_asg; i += gridDim.x * blockDim.x) {
        const int e = indices[i];
        if (e >= 0 && e < n_experts) atomicAdd(&hist[e], 1);
    }
}

}  // namespace

extern "C" int dsv41_route_topk(const float* scores, const float* bias, float* weights,
                                int32_t* indices, int32_t* hist, int rows, int n_experts, int topk,
                                int norm_topk_prob, float route_scale, int score_func,
                                cudaStream_t s) {
    if (rows <= 0 || n_experts <= 0 || topk <= 0 || topk > n_experts) {
        return (int)cudaErrorInvalidValue;
    }
    const size_t smem = (size_t)n_experts * 2 * sizeof(float) + (size_t)topk * sizeof(int);
    if (smem > 200 * 1024) return (int)cudaErrorInvalidValue;
    if (smem > 48 * 1024) {
        cudaError_t e = cudaFuncSetAttribute(route_topk_kernel,
                                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        if (e != cudaSuccess) return (int)e;
    }
    route_topk_kernel<<<rows, 256, smem, s>>>(scores, bias, weights, indices, rows, n_experts, topk,
                                              norm_topk_prob, route_scale, score_func);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return (int)e;
    if (hist != nullptr) {
        cudaMemsetAsync(hist, 0, (size_t)n_experts * sizeof(int32_t), s);
        const int n_asg = rows * topk;
        route_hist_kernel<<<(n_asg + 255) / 256, 256, 0, s>>>(indices, hist, n_asg, n_experts);
    }
    return (int)cudaGetLastError();
}

// =============================================================================
// GROUPED (PERMUTED) ROUTING — the expert-major layout the expert-centric
// routed GEMMs consume. Runtime gate: DSV41_EXPERT_GROUPED (default OFF).
// =============================================================================
//
// WHY THIS EXISTS. `dsv41_expert_gemm_e4m3_ext` (`tc5::e4x`,
// dsv41_experts_mxf4.cu) is expert-CENTRIC: ONE launch covers ONE expert over a
// DENSE `[rows, k]` activation tile (`grid = (n_total/kNTile, rows/kMTile)`,
// the expert chosen by `ids[slot]`), exactly like DeepGEMM's
// `m_grouped_gemm_nt_masked`. `moe_rows`' routing is per-(row, slot)
// (`route_idx_r[m][topk]`), so handing that table to a dense launch would apply
// slot s's expert to rows that route to a DIFFERENT one — a silent wrong
// answer, which is why the dense-tile arm declines today (see the
// `e4x_tile = false` note in chain_dev.rs::moe_rows). These three entry points
// build and use the missing form:
//
//   dsv41_route_group          route_idx_r[m][topk] -> counts / starts /
//                              expert_rows / expert_slots / perm_map / gather_src
//   dsv41_route_gather_rows    the activation rows, gathered into grouped order
//   dsv41_route_scatter_rows   the grouped output, scattered back to [m][topk][n]
//
// ⚠️ NAMING: `dsv41_gather_rows` / `dsv41_scatter_rows` (no `route_` infix) are
// ALREADY TAKEN by the attention KV gather/scatter in `dsv41_glue.cu`, with
// different signatures. These three entry points are deliberately named
// `dsv41_route_*` so the `.so` carries one symbol per behaviour — a duplicate
// `extern "C"` name would be a link error, and reusing one would hand a caller
// the wrong ABI.
//
// LAYOUT (the contract every consumer below relies on).
// Let `n_assign = m * topk` be the number of (row, slot) ASSIGNMENTS and let
// `i = r * topk + t` be the flat index of assignment (r, t).
//   * `g` in [0, n_assign) indexes the GROUPED order: experts ASCENDING, and
//     within one expert the assignments in ascending `i`. The order is a pure
//     function of `route_idx_r` (built by a serial scan, no atomics, no block
//     scheduling dependence), so a replayed CUDA graph rebuilds an IDENTICAL
//     permutation and a run is reproducible bit for bit.
//   * expert `e` owns the contiguous grouped block `[starts[e], starts[e] + counts[e])`.
//     `starts[n_experts]` is the total number of routable assignments
//     (`== n_assign` whenever every id is in range).
//   * `expert_rows[e * m_cap + j]` / `expert_slots[e * m_cap + j]` are the SOURCE
//     row `r` / slot `t` of that expert's j-th grouped row, at the fixed
//     `m_cap` stride so a kernel can index a whole expert block as `e * m_cap`.
//   * `perm_map[i]` = the grouped position of original assignment `i`
//     (original -> grouped), `-1` when `ids[i]` was out of range.
//   * `gather_src[g]` = the original assignment `i` at grouped position `g`
//     (grouped -> original), `-1` for a position no assignment reached.
//   * `active[0..*n_active)` lists the experts with `counts[e] > 0` in ascending
//     order, so the host never iterates 384 experts to find the ~topk*m that a
//     decode-sized block actually uses.
// Both maps are needed: the gather walks grouped -> original, the scatter walks
// original -> grouped, and inverting one on the fly would be an O(n_experts)
// scan per element inside the hot kernel.
//
// NUMERIC DOMAIN. All three kernels are PURE DATA MOVEMENT — every byte is
// copied verbatim, there is no arithmetic and no re-quantisation — so the
// grouped pipeline's numbers are the per-(row, slot) pipeline's numbers element
// for element. The GEMMs stay the SAME kernels with the same K order; grouping
// only decides which rows share a launch. This is what makes the layout change
// testable by an exact `memcmp` against the ungrouped buffers (see
// docs/agent/grouped-routing-design.md for the verification plan).

namespace {

// One block, `route_group_kernel_smem(n_experts)` bytes of static-equivalent
// dynamic smem. The assignment pass is SERIAL on thread 0: `n_assign` is
// `m * topk` (36 at the production verify shape: m = 6, topk = 6), so the loop
// is a few dozen iterations and the serial form is what makes the permutation
// deterministic. Nothing here is hot.
__global__ void route_group_kernel(const int32_t* __restrict__ ids, int32_t* __restrict__ counts,
                                   int32_t* __restrict__ starts,
                                   int32_t* __restrict__ expert_rows,
                                   int32_t* __restrict__ expert_slots,
                                   int32_t* __restrict__ perm_map,
                                   int32_t* __restrict__ gather_src, int32_t* __restrict__ active,
                                   int32_t* __restrict__ n_active, int m, int topk, int n_experts,
                                   int m_cap) {
    // Named dynamic-smem alias (route_group): the anonymous `extern __shared__
    // float sh[]` at the top of this TU (route_topk's) cannot be re-declared
    // with a different element type in the same TU — nvcc rejects the
    // incompatible redeclaration. Alias through a byte-typed extern instead.
    extern __shared__ unsigned char route_sm[];
    int* sh = reinterpret_cast<int*>(route_sm);
    int* s_counts = sh;              // [n_experts]
    int* s_cursor = sh + n_experts;  // [n_experts] running destination cursor
    const int n_assign = m * topk;

    for (int e = threadIdx.x; e < n_experts; e += blockDim.x) s_counts[e] = 0;
    // Poison the inverse map BEFORE anything can fill it: a position no
    // assignment reaches must be readable as "empty", not as whatever the
    // buffer held from the previous step (a stale `gather_src` would make the
    // gather copy a row of a DIFFERENT step's activation into a live grouped
    // slot). The gather turns -1 into a zero row and the scatter turns a -1
    // `perm_map` into a zeroed destination, so an out-of-range id can only
    // ever produce a zero contribution.
    for (int g = threadIdx.x; g < n_assign; g += blockDim.x) gather_src[g] = -1;
    __syncthreads();

    for (int i = threadIdx.x; i < n_assign; i += blockDim.x) {
        const int e = ids[i];
        if (e >= 0 && e < n_experts) atomicAdd(&s_counts[e], 1);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        // Exclusive prefix sum = the grouped block start of every expert. The
        // ids come from `dsv41_route_topk`, which consumes each picked expert
        // (writes -INFINITY into its selection score), so an expert can appear
        // at most once per row: counts[e] <= m <= m_cap for any well-formed
        // table. `m_cap = topk * VERIFY_ROWS` is the caller's worst case and
        // the `j < m_cap` guard below is belt-and-braces against a malformed
        // (duplicate-id) table — the grouped path must never write out of
        // bounds, and a truncated expert block would otherwise be a silent
        // wrong answer.
        int acc = 0;
        for (int e = 0; e < n_experts; ++e) {
            counts[e] = s_counts[e];
            starts[e] = acc;
            s_cursor[e] = acc;
            acc += s_counts[e];
        }
        starts[n_experts] = acc;
        int na = 0;
        for (int e = 0; e < n_experts; ++e) {
            if (s_counts[e] > 0) {
                if (active != nullptr) active[na] = e;
                ++na;
            }
        }
        if (n_active != nullptr) *n_active = na;

        for (int r = 0; r < m; ++r) {
            for (int t = 0; t < topk; ++t) {
                const int i = r * topk + t;
                const int e = ids[i];
                if (e < 0 || e >= n_experts) {
                    perm_map[i] = -1;
                    continue;
                }
                const int g = s_cursor[e]++;
                const int j = g - starts[e];
                if (j < m_cap) {
                    expert_rows[(size_t)e * m_cap + j] = r;
                    expert_slots[(size_t)e * m_cap + j] = t;
                }
                perm_map[i] = g;
                gather_src[g] = i;
            }
        }
    }
}

// The activation gather. Both operands are DENSE ROW-MAJOR with the same row
// pitch they already have in `xq4_r`/`xsc4_r`: the e4m3 activation is
// `row_bytes = dim` (one byte per value) and its per-32-block scales are
// `sc_row_bytes = (dim/32) * 4` bytes. Either pointer pair may be null (the
// caller may want the bytes only), and a grouped position whose `gather_src`
// is -1 is written as ZEROS rather than left stale.
//
// `gather_src[g]` is the flat assignment `i = r * topk + t`; the SOURCE ROW is
// `i / topk`, because the activation is quantised per ROW and every one of a
// row's topk slots reads the same bytes. That is also why the grouped buffer
// holds `m * topk` rows while the source holds `m`: the duplication is the
// price of making each expert's operand block contiguous, and it is what lets
// one expert's rows live in one dense launch.
__global__ void route_group_gather_kernel(const uint8_t* __restrict__ src_q,
                                   const float* __restrict__ src_sc,
                                   uint8_t* __restrict__ dst_q, float* __restrict__ dst_sc,
                                   const int32_t* __restrict__ gather_src, int n_assign, int topk,
                                   int row_bytes, int sc_row_bytes) {
    const int g = blockIdx.x;
    if (g >= n_assign) return;
    const int i = gather_src[g];
    const bool have_q = (src_q != nullptr && dst_q != nullptr && row_bytes > 0);
    const bool have_sc = (src_sc != nullptr && dst_sc != nullptr && sc_row_bytes > 0);
    // Row index of the source activation: -1 selects the zero fill below.
    const int row = (i >= 0) ? (i / topk) : -1;

    if (have_q) {
        uint8_t* dq = dst_q + (size_t)g * row_bytes;
        const uint8_t* sq = (row >= 0) ? src_q + (size_t)row * row_bytes : nullptr;
        // 16 bytes at a time when the row pitch allows it (`dim % 16 == 0` at
        // every production shape); a byte loop otherwise.
        if ((row_bytes & 15) == 0) {
            const int nvec = row_bytes >> 4;
            for (int c = threadIdx.x; c < nvec; c += blockDim.x) {
                uint4 v = make_uint4(0, 0, 0, 0);
                if (sq != nullptr) v = *reinterpret_cast<const uint4*>(sq + ((size_t)c << 4));
                *reinterpret_cast<uint4*>(dq + ((size_t)c << 4)) = v;
            }
        } else {
            for (int b = threadIdx.x; b < row_bytes; b += blockDim.x)
                dq[b] = (sq != nullptr) ? sq[b] : (uint8_t)0;
        }
    }
    if (have_sc) {
        float* dsc = dst_sc + (size_t)g * (sc_row_bytes >> 2);
        const float* ssc = (row >= 0) ? src_sc + (size_t)row * (sc_row_bytes >> 2) : nullptr;
        if ((sc_row_bytes & 15) == 0) {
            const int nvec = sc_row_bytes >> 4;
            for (int c = threadIdx.x; c < nvec; c += blockDim.x) {
                float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
                if (ssc != nullptr) v = *reinterpret_cast<const float4*>(ssc + (c << 2));
                *reinterpret_cast<float4*>(dsc + (c << 2)) = v;
            }
        } else {
            const int n = sc_row_bytes >> 2;
            for (int b = threadIdx.x; b < n; b += blockDim.x)
                dsc[b] = (ssc != nullptr) ? ssc[b] : 0.f;
        }
    }
}

// The output scatter: grouped `[n_assign, n]` back to the per-(row, slot)
// `[m * topk, n]` layout (`ex_act_r`'s row pitch is `topk * n`, so the flat
// assignment index IS the destination row). Driven by `perm_map` (original ->
// grouped), i.e. one block per DESTINATION row: every destination element is
// written once from a known source, and a `perm_map[i] < 0` writes zeros so a
// stale `ex_act_r` row from the previous step cannot survive into the epilogue.
__global__ void route_group_scatter_kernel(const float* __restrict__ src, float* __restrict__ dst,
                                    const int32_t* __restrict__ perm_map, int n_assign, int n) {
    const int i = blockIdx.x;
    if (i >= n_assign) return;
    const int g = perm_map[i];
    float* drow = dst + (size_t)i * n;
    if (g < 0) {
        for (int c = threadIdx.x; c < n; c += blockDim.x) drow[c] = 0.f;
        return;
    }
    const float* srow = src + (size_t)g * n;
    if ((n & 3) == 0) {
        const int nvec = n >> 2;
        for (int c = threadIdx.x; c < nvec; c += blockDim.x) {
            *reinterpret_cast<float4*>(drow + (c << 2)) =
                *reinterpret_cast<const float4*>(srow + (c << 2));
        }
    } else {
        for (int c = threadIdx.x; c < n; c += blockDim.x) drow[c] = srow[c];
    }
}

}  // namespace

// Dynamic shared memory the layout builder needs: one count and one cursor per
// expert. A plain function so the Rust side can size the launch without
// duplicating the formula.
extern "C" int dsv41_route_group_smem(int n_experts) { return 2 * n_experts * (int)sizeof(int32_t); }

// Build the grouped routing layout (see the layout contract above). Every
// pointer except `ids` may be null when the caller wants a subset; `counts`,
// `starts` and both maps are what the launchers actually need. `m_cap` is the
// per-expert row capacity of `expert_rows`/`expert_slots` (pass
// `topk * VERIFY_ROWS` for the worst case).
extern "C" int dsv41_route_group(const int32_t* ids, int32_t* counts, int32_t* starts,
                                 int32_t* expert_rows, int32_t* expert_slots, int32_t* perm_map,
                                 int32_t* gather_src, int32_t* active, int32_t* n_active, int m,
                                 int topk, int n_experts, int m_cap, cudaStream_t s) {
    if (ids == nullptr || m <= 0 || topk <= 0 || n_experts <= 0 || m_cap <= 0) {
        return (int)cudaErrorInvalidValue;
    }
    const int smem = 2 * n_experts * (int)sizeof(int32_t);
    if (smem > 48 * 1024) {
        cudaError_t e = cudaFuncSetAttribute(route_group_kernel,
                                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        if (e != cudaSuccess) return (int)e;
    }
    route_group_kernel<<<1, 256, smem, s>>>(ids, counts, starts, expert_rows, expert_slots,
                                            perm_map, gather_src, active, n_active, m, topk,
                                            n_experts, m_cap);
    return (int)cudaGetLastError();
}

// Gather the activation rows into grouped order (see the layout contract).
// Blocks: one per grouped position, 256 threads.
extern "C" int dsv41_route_gather_rows(const uint8_t* src_q, const float* src_sc, uint8_t* dst_q,
                                 float* dst_sc, const int32_t* gather_src, int n_assign, int topk,
                                 int row_bytes, int sc_row_bytes, cudaStream_t s) {
    if (gather_src == nullptr || n_assign <= 0 || topk <= 0) return (int)cudaErrorInvalidValue;
    if ((src_q != nullptr && row_bytes <= 0) || (src_sc != nullptr && sc_row_bytes <= 0)) {
        return (int)cudaErrorInvalidValue;
    }
    if ((sc_row_bytes & 3) != 0) return (int)cudaErrorInvalidValue;  // f32 scales
    route_group_gather_kernel<<<n_assign, 256, 0, s>>>(src_q, src_sc, dst_q, dst_sc, gather_src, n_assign,
                                                topk, row_bytes, sc_row_bytes);
    return (int)cudaGetLastError();
}

// Scatter the grouped expert output back to the per-(row, slot) layout.
// Blocks: one per destination row, 256 threads.
extern "C" int dsv41_route_scatter_rows(const float* src, float* dst, const int32_t* perm_map,
                                  int n_assign, int n, cudaStream_t s) {
    if (src == nullptr || dst == nullptr || perm_map == nullptr || n_assign <= 0 || n <= 0) {
        return (int)cudaErrorInvalidValue;
    }
    route_group_scatter_kernel<<<n_assign, 256, 0, s>>>(src, dst, perm_map, n_assign, n);
    return (int)cudaGetLastError();
}
