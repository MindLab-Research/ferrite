// Device-side `moe_align` — the TileLang MoE arm's segment-table builder.
//
// WHY THIS EXISTS. The TileLang MoE arm (`tilelang_gen/moe_bf16_shim.cu`,
// `DSV41_MOE_TILELANG`) consumes three tables: `order` (per segment, the flat
// assignment index of each row), `counts` (live rows per segment) and `eid`
// (the segment's expert id). The prototype built them in Python; ferrite's first
// wiring built them on the HOST (`chain_dev.rs::moe_align_host`), which means
// reading `route_idx_r` back with `download_u8` — a FULL device sync and an
// ILLEGAL CUDA-graph capture op. That is the arm's single real limitation: it
// could never enter the verify graph, which is where its ~7.1ms lives
// (docs/agent/moe-graph-design.md, verdict 1a').
//
// THE PROJECTION. `dsv41_route_group` (`dsv41_route.cu`) already computes, on the
// device, exactly the grouping `moe_align` computes. Its layout contract:
//   * `active[0..n_active)`  the experts with `counts > 0`, ASCENDING — i.e. the
//                            segments, with the empty experts' gaps removed;
//   * `counts_by_e[e]`       the assignments routed to expert e;
//   * `starts[e]`            expert e's contiguous block start in grouped order;
//   * `gather_src[g]`        the original flat assignment `i = row*topk + slot`
//                            at grouped position `g`.
// The tables the shim consumes are that same information re-indexed by SEGMENT:
//
//   seg                = 0 .. nseg-1, ordered as `active`
//   eid[seg]           = active[seg]
//   counts_seg[seg]    = min(counts_by_e[eid[seg]], bm)
//   order[seg*bm + r]  = gather_src[starts[eid[seg]] + r]        r < counts_seg[seg]
//   order[seg*bm + r]  = -1                                      r >= counts_seg[seg]
//   counts_seg[seg]    = 0, eid[seg] = 0                         seg >= nseg
//   *nseg_out          = min(*n_active_dev, seg_cap)
//
// ⚠️ THIS IS BIT-FOR-BIT `moe_align_host`. The host version sorts the assignments
// by `(expert asc, flat index asc)` with a STABLE sort; `route_group_kernel`'s
// serial scan walks `i` ASCENDING and appends into `s_cursor[e]++`, so it emits
// the same total order by construction. Hence `gather_src[starts[e] + r]` IS the
// host's `perm[a + r]`, and the two table sets are memcmp-equal at every
// well-formed routing table (docs/agent/moe-graph-design.md, the equivalence
// argument; enforced at runtime by the `DSV41_MOE_TL_HOST_ALIGN=1` diagnostic in
// `chain_dev.rs::moe_tilelang_tables_dev`).
//
// NUMERIC DOMAIN: pure data movement. No arithmetic, no atomics, no shared
// memory, no block-scheduling dependence — each output element is written by
// exactly one thread from exactly one input element, so a replayed CUDA graph
// rebuilds identical tables.
//
// ⚠️ NOT in `dsv41_route.cu` on purpose: that TU owns the TU-local anonymous
// `extern __shared__` dynamic-smem alias (route_topk's), and a second
// declaration there is typically a redeclaration error. This kernel needs no
// smem at all.
//
// rc contract, same as the tilelang shims: `0` = fired; `2` = DECLINED (the
// shape is outside this entry point's contract — the caller keeps the proven
// path); any other value = the real `cudaGetLastError()` code.

#include <cuda_runtime.h>
#include <cstdint>

namespace {

// One block, `max(seg_cap, 64)` threads, no shared memory. `blockDim.x >= seg_cap`
// is a launcher invariant, so every segment has exactly one owner and no thread
// ever writes another segment's rows.
__global__ void moe_align_from_group_kernel(const int32_t* __restrict__ active,
                                            const int32_t* __restrict__ n_active_dev,
                                            const int32_t* __restrict__ counts_by_e,
                                            const int32_t* __restrict__ starts,
                                            const int32_t* __restrict__ gather_src,
                                            int32_t* __restrict__ order,
                                            int32_t* __restrict__ counts_seg,
                                            int32_t* __restrict__ eid,
                                            int32_t* __restrict__ nseg_out, int seg_cap, int bm) {
    // (1) FULL zero fill FIRST. The generated GEMM launches with a fixed
    // `grid.y = SEG_CAP` and the two movers with `grid.y = SEG_CAP` too, so the
    // pad tail IS read: a stale `order` row would be gathered as garbage and a
    // stale `counts` would make the scatter write past the segment. The host
    // version zero-fills the same three tables up front (`vec![-1]` /
    // `vec![0]`), which is why this must happen BEFORE any live write.
    const int n_order = seg_cap * bm;
    for (int i = threadIdx.x; i < n_order; i += blockDim.x) order[i] = -1;
    for (int i = threadIdx.x; i < seg_cap; i += blockDim.x) {
        counts_seg[i] = 0;
        eid[i] = 0;
    }
    __syncthreads();  // the pad tail is visible before the live segment pass

    // (2) The live segments, one thread each. `n_active_dev` is the one value
    // here that lives on the device (it is `route_group_kernel`'s output), which
    // is the whole point: no host round trip. Every thread reads it (a
    // broadcast), one writes the result.
    const int na = (*n_active_dev < seg_cap) ? *n_active_dev : seg_cap;
    if (threadIdx.x == 0) *nseg_out = na;
    const int seg = threadIdx.x;
    if (seg >= na) return;  // the block's upper half only took part in the fill
    const int e = active[seg];
    const int c = counts_by_e[e];
    // `min(c, bm)` is belt-and-braces, not a live bound: the router consumes
    // every expert it picks, so `counts[e] <= m <= 6 <= BM = 16` for any table
    // it can emit. The host version DECLINES the step instead (`run > BM ->
    // None`); that branch is unreachable at every production shape, so declining
    // would only turn an impossible shape into a silent fallback. Clamping keeps
    // every write in bounds either way.
    const int live = (c < bm) ? c : bm;
    eid[seg] = e;
    counts_seg[seg] = live;
    const int base = starts[e];
    for (int r = 0; r < live; ++r) order[seg * bm + r] = gather_src[base + r];
}

}  // namespace

// Project the TileLang arm's three tables out of `dsv41_route_group`'s output.
// Every pointer is a DEVICE buffer; the caller must have run `dsv41_route_group`
// on the same stream first. `seg_cap`/`bm` are the generated kernels' frozen
// geometry (`TILELANG_SEG_CAP` / `TILELANG_BM`, 36 / 16 in production).
extern "C" int dsv41_moe_align_from_group(const int32_t* active, const int32_t* n_active_dev,
                                          const int32_t* counts_by_e, const int32_t* starts,
                                          const int32_t* gather_src, int32_t* order,
                                          int32_t* counts_seg, int32_t* eid, int32_t* nseg_out,
                                          int seg_cap, int bm, cudaStream_t s) {
    if (active == nullptr || n_active_dev == nullptr || counts_by_e == nullptr ||
        starts == nullptr || gather_src == nullptr || order == nullptr || counts_seg == nullptr ||
        eid == nullptr || nseg_out == nullptr) {
        return 2;
    }
    // One thread per segment (see the kernel's invariant), so `seg_cap` has to
    // fit a block. The production value is 36; a caller asking for more than a
    // block can hold is a contract break, not a shape to serve.
    if (seg_cap < 1 || seg_cap > 1024 || bm < 1) return 2;
    const int threads = (seg_cap > 64) ? seg_cap : 64;
    moe_align_from_group_kernel<<<1, threads, 0, s>>>(active, n_active_dev, counts_by_e, starts,
                                                      gather_src, order, counts_seg, eid, nseg_out,
                                                      seg_cap, bm);
    return (int)cudaGetLastError();
}
