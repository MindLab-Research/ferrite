//! CudaBackend — FFI bridge to libferrite_kernels.so (sm_100a).
//!
//! Enabled with `--features cuda`. The .so is produced by
//! `kernels/cuda/build.sh` (nvcc; compiling needs no GPU).
//!
//! v1 contract: host-tensor in → cudaMemcpy H2D → kernel → cudaMemcpy D2H
//! → host-tensor out. This is the *correctness* path for the B300 golden
//! harness (diff against CpuBackend); the performance path (device-
//! resident tensors + CUDA graph replay via `cuStreamBeginCapture`) keeps
//! the same extern contract — see the GraphCapable section.
//!
//! Numerical parity target: f32 tolerance 1e-5 vs the CPU backend.

#![cfg(feature = "cuda")]

use std::ffi::CString;
use std::sync::Arc;

use ferrite_types::{DType, FerriteError, Result, Shape, Tensor};

/// Opaque stream handle (cudaStream_t == void* at the ABI level).
pub type CuStream = *mut std::ffi::c_void;

extern "C" {
    // cudart (linked into libferrite_kernels.so's dependency closure)
    fn cudaSetDevice(dev: i32) -> i32;
    fn cudaGetDevice(dev: *mut i32) -> i32;
    fn cudaProfilerStart() -> i32;
    fn cudaProfilerStop() -> i32;
    fn cudaMalloc(ptr: *mut *mut std::ffi::c_void, size: usize) -> i32;
    fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> i32;
    fn cudaFree(ptr: *mut std::ffi::c_void) -> i32;
    fn cudaMemcpy(dst: *mut std::ffi::c_void, src: *const std::ffi::c_void, count: usize, kind: i32) -> i32;
    fn cudaMemcpy2D(
        dst: *mut std::ffi::c_void,
        dpitch: usize,
        src: *const std::ffi::c_void,
        spitch: usize,
        width: usize,
        height: usize,
        kind: i32,
    ) -> i32;
    fn cudaStreamCreate(stream: *mut CuStream) -> i32;
    fn cudaStreamSynchronize(stream: CuStream) -> i32;
    fn cudaDeviceSynchronize() -> i32;
    fn cudaGetErrorString(err: i32) -> *const std::os::raw::c_char;

    // cuBLAS (batched decode m=16 GEMM: bandwidth-bound, needs the split-K
    // /streaming cuBLAS already implements — see build.rs)
    fn cublasCreate_v2(handle: *mut *mut std::ffi::c_void) -> i32;
    fn cublasSetStream_v2(handle: *mut std::ffi::c_void, stream: CuStream) -> i32;
    fn cublasGemmEx(
        handle: *mut std::ffi::c_void,
        transa: i32,
        transb: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: *const f32,
        a: *const std::ffi::c_void,
        atype: i32,
        lda: i32,
        b: *const std::ffi::c_void,
        btype: i32,
        ldb: i32,
        beta: *const f32,
        c: *mut std::ffi::c_void,
        ctype: i32,
        ldc: i32,
        compute: i32,
        algo: i32,
    ) -> i32;

    // ferrite kernels (ferrite_kernels.cu bridge)
    fn ferrite_matmul(x: *const f32, w: *const f32, bias: *const f32, out: *mut f32,
                      n: i32, in_f: i32, out_f: i32, s: CuStream) -> i32;
    fn ferrite_matmul_bf16(x: *const f32, w: *const std::ffi::c_void,
                            bias: *const f32, out: *mut f32,
                            n: i32, in_f: i32, out_f: i32, s: CuStream) -> i32;
    fn ferrite_gemv_bf16(x: *const f32, w: *const std::ffi::c_void,
                          bias: *const f32, out: *mut f32,
                          in_f: i32, out_f: i32, s: CuStream) -> i32;
    fn ferrite_gemv_bf16_v2(x: *const f32, w: *const std::ffi::c_void,
                             bias: *const f32, out: *mut f32,
                             in_f: i32, out_f: i32, nrows: i32, s: CuStream) -> i32;
    fn ferrite_gemv_bf16_nt(x: *const f32, w: *const std::ffi::c_void,
                             bias: *const f32, out: *mut f32,
                             in_f: i32, out_f: i32, nrows: i32, s: CuStream) -> i32;
    fn ferrite_gemm_bf16_mma(a: *const f32, w: *const std::ffi::c_void,
                             bias: *const f32, out: *mut f32,
                             nrows: i32, in_f: i32, out_f: i32, s: CuStream) -> i32;
    fn ferrite_gemv_tri(x: *const f32, w1: *const std::ffi::c_void, w2: *const std::ffi::c_void,
                        w3: *const std::ffi::c_void, y1: *mut f32, y2: *mut f32, y3: *mut f32,
                        in_f: i32, o1: i32, o2: i32, o3: i32, s: CuStream) -> i32;
    fn ferrite_gemv_fp8_mma_b16(xq: *const u8, xs: *const f32, w: *const u8,
                                  ws: *const f32, out: *mut f32,
                                  n: i32, in_f: i32, out_f: i32, scols: i32,
                                  s: CuStream) -> i32;
    fn ferrite_gemv_fp8_v2(x: *const f32, w: *const std::ffi::c_void,
                           scale: *const f32, bias: *const f32, out: *mut f32,
                           in_f: i32, out_f: i32, nrows: i32,
                           srows: i32, scols: i32, s: CuStream) -> i32;
    fn ferrite_fp8_mma_probe(A: *const u8, B: *const u8, C: *mut f32, s: CuStream) -> i32;
    fn ferrite_fp8_quant(x: *const f32, xq: *mut u8, xs: *mut f32, in_f: i32, s: CuStream) -> i32;
    fn ferrite_moe_fused_act_fp8_mma(x: *const f32, ids_f: *const f32,
                                      gate_w8_ptrs: *const *const std::ffi::c_void,
                                      gate_scale_ptrs: *const *const std::ffi::c_void,
                                      up_w8_ptrs: *const *const std::ffi::c_void,
                                      up_scale_ptrs: *const *const std::ffi::c_void,
                                      shared_gate_w8: *const std::ffi::c_void,
                                      shared_gate_scale: *const std::ffi::c_void,
                                      shared_up_w8: *const std::ffi::c_void,
                                      shared_up_scale: *const std::ffi::c_void,
                                      act: *mut f32, expert_start: i32, e_local: i32,
                                      hidden: i32, inter: i32, inter_shared: i32,
                                      topk: i32, n: i32, limit: f32, s: CuStream) -> i32;
    // v2: pre-quantized xq/xs (ferrite_quant_e4m3_tokens ran ONCE per layer —
    // the act kernel re-quantized x[tok] 864x per layer at n=3 (grid
    // (max_rows/16, topk+1, n) × per-block quant; the 46µs act kernel is
    // quantize-BOUND, not A-weight-stream bound).
    fn ferrite_moe_fused_act_fp8_mma_v2(x: *const f32, ids_f: *const f32,
                                        gate_w8_ptrs: *const *const std::ffi::c_void,
                                        gate_scale_ptrs: *const *const std::ffi::c_void,
                                        up_w8_ptrs: *const *const std::ffi::c_void,
                                        up_scale_ptrs: *const *const std::ffi::c_void,
                                        shared_gate_w8: *const std::ffi::c_void,
                                        shared_gate_scale: *const std::ffi::c_void,
                                        shared_up_w8: *const std::ffi::c_void,
                                        shared_up_scale: *const std::ffi::c_void,
                                        act: *mut f32, expert_start: i32, e_local: i32,
                                        hidden: i32, inter: i32, inter_shared: i32,
                                        topk: i32, n: i32, limit: f32,
                                        xq: *const u8, xs: *const f32,
                                        s: CuStream) -> i32;
    fn ferrite_quant_e4m3_tokens(x: *const f32, xq: *mut u8, xs: *mut f32,
                                  n: i32, hidden: i32, s: CuStream) -> i32;
    fn ferrite_gemv_fp8_mma_v2(xq: *const u8, xs: *const f32, w: *const std::ffi::c_void,
                                w_scale: *const f32, out: *mut f32, in_f: i32, out_f: i32,
                                scols: i32, s: CuStream) -> i32;
    fn ferrite_gemv_fp8_mma(x: *const f32, w: *const std::ffi::c_void, w_scale: *const f32,
                            out: *mut f32, in_f: i32, out_f: i32,
                            srows: i32, scols: i32, scratch: *mut u32, s: CuStream) -> i32;
    fn ferrite_layernorm_affine(x: *const f32, w: *const f32, b: *const f32,
                                 out: *mut f32, n: i32, dim: i32, s: CuStream) -> i32;
    fn ferrite_embed_expand(table: *const std::ffi::c_void, ids: *const i32,
                            out: *mut f32, n: i32, hidden: i32, mult: i32,
                            vocab: i32, s: CuStream) -> i32;
    // N-UNIFIED (FERRITE_MTP_N): d = [n-1] draft argmax (d1..d_{n-1}), a = [n]
    // verify argmax (a0..a_{n-1}) — the kernel finds the longest matching
    // prefix k in 1..n. n=3 reduces to the old (d1, d2, a0, a1, a2) form.
    fn ferrite_mtp_accept(d: *const f32, a: *const f32,
                          k_out: *mut i32, next_token: *mut i32, n_accepted: *mut i32,
                          n: i32, s: CuStream) -> i32;
    fn ferrite_embed_expand_dev(table: *const std::ffi::c_void, ids_dev: *const i32,
                               out: *mut f32, n: i32, hidden: i32, mult: i32,
                               vocab: i32, s: CuStream) -> i32;
    fn ferrite_embed_one(table: *const std::ffi::c_void, token_slot: *const i32,
                        out: *mut f32, hidden: i32, mult: i32,
                        vocab: i32, s: CuStream) -> i32;
    fn ferrite_cast_store_i32(src: *const std::ffi::c_void, dst: *mut std::ffi::c_void, s: CuStream) -> i32;
    fn ferrite_dsa_cache_append(kvb: *const f32, ki: *const f32, gate: *const f32,
                                 k_nope: *mut f32, v: *mut f32, k_idx: *mut f32, k_gate: *mut f32,
                                 t0_ptr: *const i32, n: i32, h: i32, dk: i32, dv: i32, idm: i32,
                                 s: CuStream) -> i32;
    fn ferrite_kpool_compress(k_idx: *const f32, k_gate: *const f32, ape: *const f32,
                               pool_keys: *mut f32, total_ptr: *const i32, npools: i32, kpool: i32,
                               idm: i32, s: CuStream) -> i32;
    fn ferrite_pool_expand(idx_pools: *const f32, idx: *mut f32,
                            n: i32, select_k: i32, kpool: i32, max_npools: i32,
                            total_ptr: *const i32, n_fixed: i32,
                            s: CuStream) -> i32;
    fn ferrite_dsa_append_batched(kvb: *const f32, ki: *const f32, gate: *const f32,
                                   kn_tbl: *const *mut f32, v_tbl: *const *mut f32,
                                   kidx_tbl: *const *mut f32, kgate_tbl: *const *mut f32,
                                   t0_tbl: *const *const i32,
                                   b: i32, h: i32, dk: i32, dv: i32, idm: i32, ntok: i32,
                                   s: CuStream) -> i32;
    fn ferrite_kpool_compress_batched(kidx_tbl: *const *mut f32, kgate_tbl: *const *mut f32,
                                       ape: *const f32, pool_keys: *mut f32,
                                       total_tbl: *const *const i32,
                                       b: i32, max_npools: i32, kpool: i32, idm: i32,
                                       s: CuStream) -> i32;
    fn ferrite_indexer_topk_batched(qi: *const f32, pool_keys: *const f32, w: *const f32,
                                     idx: *mut f32, b: i32, ih: i32, idm: i32,
                                     select_k_max: i32, kpool: i32, max_npools: i32,
                                     total_tbl: *const *const i32,
                                     s: CuStream) -> i32;
    fn ferrite_pool_expand_batched(idx_pools: *const f32, idx: *mut f32,
                                   b: i32, select_k_max: i32, kpool: i32, max_npools: i32,
                                   total_tbl: *const *const i32, n_fixed: i32,
                                   s: CuStream) -> i32;
    fn ferrite_sparse_attn_v2_batched(q: *const f32, k_tbl: *const *mut f32, v_tbl: *const *mut f32,
                                       idx: *const f32, out: *mut f32, b: i32,
                                       total_tbl: *const *const i32,
                                       h: i32, d: i32, dv: i32, topk: i32,
                                       s: CuStream) -> i32;
    fn ferrite_scale_inplace(x: *mut f32, s: f32, n: i32, st: CuStream) -> i32;
    fn ferrite_pdl_exp(mode: i32, iters: i32, out_time_ms: *mut f32,
                       out_checksum: *mut f32, s: CuStream) -> i32;
    fn ferrite_p2p_ar_oneshot(partial: *const f32,
                               staging_tbl: *const *mut f32,
                               ready_tbl: *const *mut u32,
                               ctr: *mut u32,
                               staging_local: *const f32,
                               ready_local: *const u32,
                               out: *mut f32, n: i32, world: i32, my_rank: i32,
                               s: CuStream) -> i32;
    fn ferrite_p2p_ar_oneshot_v2(partial: *const f32,
                                  staging_tbl: *const *mut f32,
                                  ready_tbl: *const *mut u32,
                                  epoch: *mut u32,
                                  ctr: *mut u32,
                                  staging_local: *const f32,
                                  ready_local: *const u32,
                                  out: *mut f32, n: i32, world: i32, my_rank: i32,
                                  stride: i32,
                                  s: CuStream) -> i32;
    fn ferrite_p2p_ar_fused_v3(partial: *const f32,
                               staging_tbl: *const *mut f32,
                               ready_tbl: *const *mut u32,
                               epoch: *mut u32,
                               ctr: *mut u32,
                               staging_local: *const f32,
                               ready_local: *const u32,
                               seen: *mut u32,
                               out: *mut f32, n: i32, world: i32, my_rank: i32,
                               stride: i32,
                               s: CuStream) -> i32;
    fn ferrite_graph_begin(s: CuStream) -> i32;
    fn ferrite_graph_end(s: CuStream, g: *mut *mut std::ffi::c_void) -> i32;
    fn ferrite_graph_instantiate(e: *mut *mut std::ffi::c_void, g: *mut std::ffi::c_void) -> i32;
    fn ferrite_graph_launch(e: *mut std::ffi::c_void, s: CuStream) -> i32;
    fn ferrite_f32_to_bf16(in_: *const f32, out: *mut std::ffi::c_void,
                            n: i64, s: CuStream) -> i32;
    fn ferrite_dequant_e4m3_block(w: *const u8, scale: *const f32, out: *mut std::ffi::c_void,
                                  rows: i64, cols: i64, srows: i32, scols: i32,
                                  s: CuStream) -> i32;
    fn ferrite_bf16_to_f32(in_: *const std::ffi::c_void, out: *mut f32,
                           n: i64, s: CuStream) -> i32;
    fn ferrite_rmsnorm(x: *const f32, w: *const f32, out: *mut f32,
                       n: i32, dim: i32, eps: f32, s: CuStream) -> i32;
    fn ferrite_hc_contract(x: *const f32, out: *mut f32,
                           s: i32, n: i32, h: i32, stream: CuStream) -> i32;
    fn ferrite_gemv5_bf16(x: *const f32, w1: *const std::ffi::c_void, w2: *const std::ffi::c_void,
                          w3: *const std::ffi::c_void, w4: *const std::ffi::c_void, w5: *const std::ffi::c_void,
                          o1: *mut f32, o2: *mut f32, o3: *mut f32, o4: *mut f32, o5: *mut f32,
                          in_f: i32, of1: i32, of2: i32, of3: i32, of4: i32, of5: i32,
                          stream: CuStream) -> i32;
    fn ferrite_gated_rmsnorm(x: *const f32, gate: *const f32, w: *const f32, out: *mut f32,
                             n: i32, dim: i32, eps: f32, s: CuStream) -> i32;
    fn ferrite_swiglu(gu: *const f32, out: *mut f32, n: i32, inter: i32,
                      limit: f32, s: CuStream) -> i32;
    fn ferrite_swiglu2(gate: *const f32, up: *const f32, out: *mut f32,
                       n: i32, inter: i32, limit: f32, s: CuStream) -> i32;
    fn ferrite_causal_conv1d(x: *const f32, w: *const f32, state_in: *const f32,
                             out: *mut f32, state_out: *mut f32, snaps: *mut f32,
                             n: i32, ch: i32, conv: i32, s: CuStream) -> i32;
    fn ferrite_conv1d_batched(x: *const f32, w: *const f32,
                             state_ptrs: *const *mut f32,
                             out: *mut f32, b: i32, ch: i32, conv: i32,
                             s: CuStream) -> i32;
    fn ferrite_gdn_chunk_batched(q: *const f32, k: *const f32, v: *const f32,
                                  beta: *const f32, gate: *const f32, a_log: *const f32,
                                  state_ptrs: *const *mut f32,
                                  out: *mut f32, b: i32, h: i32, dk: i32, dv: i32,
                                  s: CuStream) -> i32;
    fn ferrite_gdn_chunk(q: *const f32, k: *const f32, v: *const f32,
                         beta: *const f32, gate: *const f32, a_log: *const f32,
                         state: *mut f32, out: *mut f32,
                         n: i32, h: i32, dk: i32, dv: i32, s: CuStream) -> i32;
    fn ferrite_gdn_chunk_v2(q: *const f32, k: *const f32, v: *const f32,
                            beta: *const f32, gate: *const f32, a_log: *const f32,
                            state: *mut f32, out: *mut f32,
                            n: i32, h: i32, dk: i32, dv: i32, s: CuStream) -> i32;
    fn ferrite_gdn_chunk_wyf(q: *const f32, k: *const f32, v: *const f32,
                             beta: *const f32, gate: *const f32, a_log: *const f32,
                             state_in: *mut f32, out: *mut f32, state_out: *mut f32,
                             n: i32, h: i32, dk: i32, dv: i32, s: CuStream) -> i32;
    fn ferrite_moe_route(logits: *const f32, bias: *const f32, probs: *mut f32, ids: *mut f32,
                         n: i32, e: i32, topk: i32,
                         scale: f32, s: CuStream) -> i32;
    fn ferrite_router_gemm_route_fused(
        x: *const f32, w: *const std::ffi::c_void, bias: *const f32,
        probs: *mut f32, ids: *mut f32, logits: *mut f32, ctr: *mut u32,
        n_exp: i32, hidden: i32, topk: i32, scale: f32, s: CuStream) -> i32;
    fn ferrite_indexer_topk(qi: *const f32, ki: *const f32, w: *const f32, idx: *mut f32,
                            n: i32, h: i32, d: i32, topk: i32,
                            total_ptr: *const i32, kpool_val: i32, n_fixed: i32, s: CuStream) -> i32;
    fn ferrite_sparse_attn(q: *const f32, k: *const f32, v: *const f32, idx: *const f32,
                           out: *mut f32, n: i32, t_ptr: *const i32, h: i32, d: i32, dv: i32,
                           topk: i32, s: CuStream) -> i32;
    fn ferrite_sparse_attn_v2(q: *const f32, k: *const f32, v: *const f32, idx: *const f32,
                              out: *mut f32, scratch: *mut f32, n: i32, t_ptr: *const i32, h: i32, d: i32, dv: i32,
                              topk: i32, splits: i32, s: CuStream) -> i32;
    fn ferrite_argmax(logits: *const f32, out: *mut f32, n: i32, dim: i32, s: CuStream) -> i32;
    // N-UNIFIED (FERRITE_MTP_N): plan row = 6 pointers/layer (conv_a, gdn_a,
    // conv_b, gdn_b, conv_snaps_base, gdn_snaps_base — the per-t snapshots
    // live in ONE contiguous [n-1][len] scratch each; snapshot j-1 at
    // base + (j-1)*len). k=n commits B; k=j<n commits snapshot j-1.
    fn ferrite_mtp_commit(k_pin: *const i32, plan: *const *mut f32,
                          n_plans: i32, conv_len: i32, gdn_len: i32,
                          hf_v: *const f32, hprev: *mut f32,
                          hidden: i32, n: i32, s: CuStream) -> i32;
    fn ferrite_softmax(logits: *const f32, out: *mut f32, n: i32, dim: i32, s: CuStream) -> i32;
    fn ferrite_hc_pre(res: *const f32, fw: *const f32, scale: *const f32, base: *const f32,
                      li: *mut f32, post: *mut f32, comb: *mut f32,
                      s: i32, n: i32, h: i32, mix: i32,
                      rms_eps: f32, hc_eps: f32, iters: i32, stream: CuStream) -> i32;
    fn ferrite_hc_pre_split(res: *const f32, fw: *const f32, scale: *const f32, base: *const f32,
                             nw: *const f32,
                             li: *mut f32, post: *mut f32, comb: *mut f32, mx_scratch: *mut f32,
                             s: i32, n: i32, h: i32, mix: i32,
                             rms_eps: f32, hc_eps: f32, iters: i32, stream: CuStream) -> i32;
    fn ferrite_hc_post(x: *const f32, res: *const f32, post: *const f32, comb: *const f32,
                       out: *mut f32, s: i32, n: i32, h: i32, stream: CuStream) -> i32;
    fn ferrite_gdn_prep(conv_out: *const f32, b_raw: *const f32, fb: *const f32,
                        dt_bias: *const f32, a_log: *const f32,
                        q: *mut f32, k: *mut f32, v: *mut f32, beta: *mut f32, gate: *mut f32,
                        n: i32, h: i32, dk: i32, lb: f32, stream: CuStream) -> i32;
    fn ferrite_conv_prep_fused(x: *const f32, cw: *const f32, cs: *const f32,
                               b_raw: *const f32, fb: *const f32, dt_bias: *const f32,
                               a_log: *const f32,
                               q: *mut f32, k: *mut f32, v: *mut f32,
                               beta: *mut f32, gate: *mut f32,
                               h: i32, dk: i32, lb: f32, stream: CuStream) -> i32;
    fn ferrite_gemv_qkv_conv(x: *const f32, w: *const std::ffi::c_void,
                             cw: *const f32, cs: *mut f32,
                             q: *mut f32, k: *mut f32, v: *mut f32,
                             in_f: i32, proj: i32, stream: CuStream) -> i32;
    fn ferrite_gdn_chunk_fused(q: *const f32, k: *const f32, v: *const f32,
                               beta: *const f32, gate: *const f32, a_log: *const f32,
                               state: *mut f32, gdn0: *mut f32, gdn1: *mut f32, out: *mut f32,
                               n: i32, h: i32, dk: i32, dv: i32, stream: CuStream) -> i32;
    fn ferrite_gdn_step_v2p(q: *const f32, k: *const f32, v: *const f32,
                            b_raw: *const f32, fb: *const f32, dt_bias: *const f32,
                            a_log: *const f32, lb: f32,
                            state: *mut f32, out: *mut f32,
                            h: i32, dk: i32, dv: i32, dsplits: i32, stream: CuStream) -> i32;
    fn ferrite_add(x: *const f32, y: *const f32, z: *mut f32,
                   n: i32, stream: CuStream) -> i32;
    fn cudaMemset(ptr: *mut std::ffi::c_void, val: i32, bytes: usize) -> i32;
}

const CUDA_MEMCPY_H2D: i32 = 1;
const CUDA_MEMCPY_D2H: i32 = 2;
const CUDA_MEMCPY_D2D: i32 = 3;
extern "C" {
    fn cudaMemcpyAsync(dst: *mut std::ffi::c_void, src: *const std::ffi::c_void,
                        count: usize, kind: i32, stream: CuStream) -> i32;
    fn cudaMemsetAsync(ptr: *mut std::ffi::c_void, val: i32, count: usize, stream: CuStream) -> i32;
}

fn ck(err: i32, what: &str) -> Result<()> {
    if err == 0 {
        Ok(())
    } else {
        let msg = unsafe {
            let p = cudaGetErrorString(err);
            if p.is_null() {
                "unknown".into()
            } else {
                std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
            }
        };
        Err(FerriteError::InvalidArg(format!("CUDA {what}: {msg} (err {err})")))
    }
}

// ============================================================
// Activation buffer pool — per-device, size-class bucketed.
// Per-op cudaMalloc/cudaFree are device-synchronising calls that dominate the
// op-latency budget (tens of thousands of ops per token across a TP cluster);
// pooled reuse removes them entirely after warmup.
//
// GLOBAL (Mutex<HashMap>) — NOT thread-local: fan_out spawns fresh threads
// per layer (90 spawns × 4 ranks per token); a thread-local pool was EMPTY
// in every worker, so every DevBuf::alloc paid cudaMalloc+cudaMallocHost
// (~70μs each, ~14k allocs/token ≈ 1s of pure allocation per token) and
// the buffers LEAKED when the thread exited (pool dropped, never freed).
// The global pool also makes CUDA graph capture possible (cudaMallocHost
// during capture is illegal — with a warm global pool, capture allocates
// nothing).
//
// CUDA-graph capture support: every DevBuf owns a PINNED host staging buffer
// (cudaMallocHost, allocated with the device buffer, pooled with it).
// upload/download go through it — cudaMemcpyAsync from pageable memory is
// ILLEGAL during stream capture (cudaErrorStreamCaptureUnsupported) and the
// tensor's Vec address changes every call, which would bake a stale pointer
// into the graph. The pinned stage is the fixed-address rendezvous: the CPU
// writes it (outside the graph), the recorded memcpy moves stage→device.
// ============================================================
static BUF_POOL: std::sync::OnceLock<
    std::sync::Mutex<
        std::collections::HashMap<(i32, u32), Vec<PoolPtrs>>,
    >,
> = std::sync::OnceLock::new();

/// Raw device/pinned-stage pointer pair — Send+Sync because the pool's
/// Mutex serialises all take/release, and CUDA device pointers are
/// process-global (not thread-bound).
#[derive(Clone, Copy)]
struct PoolPtrs(*mut std::ffi::c_void, *mut std::ffi::c_void);
unsafe impl Send for PoolPtrs {}
unsafe impl Sync for PoolPtrs {}

fn pool() -> &'static std::sync::Mutex<
    std::collections::HashMap<(i32, u32), Vec<PoolPtrs>>,
> {
    BUF_POOL.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

/// Dedicated pool for the BATCHED decode path (2026-09-09): the per-size
/// CUDA graphs (megab_bN) RECORD the per-layer DevBuf pointers, so those
/// addresses must stay stable across steps. The general LIFO pool gets
/// shuffled by interleaved PREFILL allocations during the admission ramp —
/// a prefill taking a graph-referenced buffer makes the replay read/write
/// the WRONG buffer (the intermittent 2MB-aligned Xid-31 faults at
/// live≈4-5, ~50% of B=16 runs). Prefills/one-shot keep the general pool.
static BATCH_BUF_POOL: std::sync::OnceLock<
    std::sync::Mutex<std::collections::HashMap<(i32, u32), Vec<PoolPtrs>>>,
> = std::sync::OnceLock::new();

fn batch_pool() -> &'static std::sync::Mutex<
    std::collections::HashMap<(i32, u32), Vec<PoolPtrs>>,
> {
    BATCH_BUF_POOL.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

/// True while the engine runs decode_step_batched (set around the fan_out;
/// GLOBAL because the allocations happen on the fan_out worker threads).
static IN_BATCH_DECODE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Mark the current execution as the batched decode step: DevBuf::alloc
/// routes to the dedicated batch pool (see BATCH_BUF_POOL above).
pub fn set_batch_decode(v: bool) {
    IN_BATCH_DECODE.store(v, std::sync::atomic::Ordering::Release);
}

/// True while THIS thread is inside a stream capture. THREAD-LOCAL: the
/// per-layer graphs capture inside fan_out workers — 4 ranks capture
/// concurrently and each ends independently; a GLOBAL flag would let the
/// first finisher re-enable sync for the others mid-capture (segfault).
/// Download skips its synchronisation while capturing — the graph's
/// end/replay syncs once at the tail.
thread_local! {
    static CAPTURING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

pub(crate) fn set_capturing(v: bool) {
    CAPTURING.with(|c| c.set(v));
}

fn is_capturing() -> bool {
    CAPTURING.with(|c| c.get())
}

fn buf_pool_release(dev: i32, class: u32, ptr: *mut std::ffi::c_void, stage: *mut std::ffi::c_void, batch: bool) {
    let mut p = if batch { batch_pool().lock().unwrap() } else { pool().lock().unwrap() };
    let v = p.entry((dev, class)).or_default();
    if std::env::var_os("FERRITE_POOL_DEBUG").is_some() && v.iter().any(|pp| pp.0 == ptr) {
        eprintln!("[pool-dup] DOUBLE RELEASE dev={dev} class={class} ptr={ptr:?} batch={batch}");
    }
    v.push(PoolPtrs(ptr, stage));
}

fn buf_pool_take(dev: i32, class: u32, batch: bool) -> Option<(*mut std::ffi::c_void, *mut std::ffi::c_void)> {
    let mut p = if batch { batch_pool().lock().unwrap() } else { pool().lock().unwrap() };
    p.get_mut(&(dev, class)).and_then(|v| v.pop()).map(|p| (p.0, p.1))
}

/// Drop all pooled activation buffers (weights are owned by the weight cache).
/// Called from CudaBackend::Drop; leaks of freed devices are reclaimed by CUDA
/// context teardown at exit.
pub fn clear_activation_pool() {
    let mut p = pool().lock().unwrap();
    for ((dev, _), ptrs) in p.drain() {
        unsafe { cudaSetDevice(dev) };
        for pp in ptrs {
            unsafe { cudaFree(pp.0) };
            if !pp.1.is_null() {
                unsafe { cudaFreeHost(pp.1) };
            }
        }
    }
}

extern "C" {
    fn cudaMallocHost(ptr: *mut *mut std::ffi::c_void, bytes: usize) -> i32;
    fn cudaFreeHost(ptr: *mut std::ffi::c_void) -> i32;
    fn cudaGraphExecDestroy(exec: *mut std::ffi::c_void) -> i32;
}

/// A device buffer (pooled) with its pinned host stage. `len` is the
/// requested length; `class` the size class (next power of two ≥ len).
///
/// PUBLIC: the device-resident op-chain phase (whole-layer DevBuf pipelines
/// feeding a single CUDA graph) composes ops at this level — matmul_dev and
/// friends take/return DevBuf so activations never cross the bus inside a
/// layer.
pub struct DevBuf {
    pub ptr: *mut std::ffi::c_void,
    pub len: usize,
    pub class: u32,
    pub dev: i32,
    pub stream: CuStream,
    /// Pinned host staging (cudaMallocHost) — the fixed-address rendezvous
    /// for graph-capturable H2D/D2H (see the module comment above).
    pub stage: *mut std::ffi::c_void,
    /// True when allocated from the dedicated batched-decode pool (its
    /// addresses are baked into the per-size CUDA graphs and must not be
    /// shuffled by interleaved prefill allocations).
    pub batch: bool,
    /// True = this buffer NEVER enters any pool and its Drop is a no-op
    /// (intentionally leaked for the CUDA graph's lifetime). Used for the
    /// graph's INPUT buffer: its (ptr, stage) are recorded in the graph, so
    /// it must not be recycled by the pool (aliasing → replay faults) nor
    /// allocated during the capture (a pool miss there = cudaMalloc inside
    /// capture = err 900).
    pub immortal: bool,
}

impl DevBuf {
    /// Pooled alloc: reuse a released (device, stage) pair of the same size
    /// class when available, else cudaMalloc + cudaMallocHost. The caller
    /// must have `enter()`ed the backend's device.
    pub fn alloc(dev: i32, stream: CuStream, len: usize) -> Result<Self> {
        // DEFENSIVE (2026-09-09): cudaSetDevice is THREAD-LOCAL. A worker or
        // engine thread whose binding differs from `dev` would allocate on the
        // WRONG device — the resulting cross-device pointer is still a "valid"
        // address for memcheck but is unmapped on `dev`, so kernels fault on it
        // (2MB-aligned Xid 31 PDE faults). Re-bind before allocating.
        let mut cur: i32 = -1;
        unsafe { cudaGetDevice(&mut cur) };
        if cur != dev {
            if std::env::var_os("FERRITE_DEV_MISMATCH").is_some() {
                eprintln!("[dev-mismatch] DevBuf::alloc(dev={dev}) from a thread bound to {cur} — re-binding");
            }
            unsafe { cudaSetDevice(dev) };
        }
        let class = (len.max(1) as u32).next_power_of_two();
        // Route to the dedicated batched-decode pool when the flag is set (see
        // BATCH_BUF_POOL above): the graph-recorded addresses stay stable.
        let batch = IN_BATCH_DECODE.load(std::sync::atomic::Ordering::Acquire);
        if let Some((ptr, stage)) = buf_pool_take(dev, class, batch) {
            return Ok(DevBuf { ptr, len, class, dev, stream, stage, batch, immortal: false });
        }
        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        // DIAGNOSTIC (FERRITE_POOL_MISS=1): a cudaMalloc inside a stream
        // capture is ILLEGAL (err 900) and would silently invalidate the graph
        // — the dry-run is supposed to warm every pool class first.
        if std::env::var_os("FERRITE_POOL_MISS").is_some() {
            eprintln!("[pool-miss] dev={dev} class={class} len={len} batch={batch} — cudaMalloc (capture-illegal if inside a capture)");
        }
        ck(unsafe { cudaMalloc(&mut ptr, class as usize * std::mem::size_of::<f32>()) }, "pooled malloc")?;
        let mut stage: *mut std::ffi::c_void = std::ptr::null_mut();
        ck(unsafe { cudaMallocHost(&mut stage, class as usize * std::mem::size_of::<f32>()) }, "pinned stage malloc")?;
        Ok(DevBuf { ptr, len, class, dev, stream, stage, batch, immortal: false })
    }

    /// An IMMORTAL DevBuf: direct cudaMalloc + cudaMallocHost, NEVER enters any
    /// pool, Drop is a no-op (intentionally leaked for the CUDA graph's
    /// lifetime). Use for graph-recorded INPUT buffers whose (ptr, stage) must
    /// stay stable: a pooled buffer would be recycled by the layer loop's
    /// reassignments (aliasing → replay faults) or starve the capture pass's
    /// pool (a miss inside the capture = cudaMalloc = err 900).
    pub fn alloc_immortal(dev: i32, stream: CuStream, len: usize) -> Result<Self> {
        let mut cur: i32 = -1;
        unsafe { cudaGetDevice(&mut cur) };
        if cur != dev {
            unsafe { cudaSetDevice(dev) };
        }
        let class = (len.max(1) as u32).next_power_of_two();
        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        ck(unsafe { cudaMalloc(&mut ptr, class as usize * std::mem::size_of::<f32>()) }, "immortal malloc")?;
        let mut stage: *mut std::ffi::c_void = std::ptr::null_mut();
        ck(unsafe { cudaMallocHost(&mut stage, class as usize * std::mem::size_of::<f32>()) }, "immortal pinned stage malloc")?;
        Ok(DevBuf { ptr, len, class, dev, stream, stage, batch: false, immortal: true })
    }
    /// H2D via the pinned stage — graph-capturable: the CPU copy into the
    /// stage happens outside any graph; the recorded memcpy moves
    /// stage→device at fixed addresses on both ends.
    pub fn upload(&self, host: &[f32]) -> Result<()> {
        assert!(host.len() <= self.len);
        unsafe {
            std::ptr::copy_nonoverlapping(host.as_ptr(), self.stage as *mut f32, host.len());
        }
        ck(unsafe {
            cudaMemcpyAsync(self.ptr, self.stage, host.len() * 4, CUDA_MEMCPY_H2D, self.stream)
        }, "memcpy H2D")
    }
    /// D2H via the pinned stage; synchronises the stream (the op tail) —
    /// EXCEPT during capture, when sync is illegal and the graph's
    /// end_verify does the single tail sync instead.
    pub fn download(&self, host: &mut [f32]) -> Result<()> {
        assert!(host.len() <= self.len);
        ck(unsafe {
            cudaMemcpyAsync(self.stage, self.ptr, host.len() * 4, CUDA_MEMCPY_D2H, self.stream)
        }, "memcpy D2H")?;
        if !is_capturing() {
            ck(unsafe { cudaStreamSynchronize(self.stream) }, "sync after D2H")?;
        }
        unsafe {
            std::ptr::copy_nonoverlapping(self.stage as *const f32, host.as_mut_ptr(), host.len());
        }
        Ok(())
    }
    pub fn as_f32(&self) -> *mut f32 {
        self.ptr as *mut f32
    }
    pub fn as_const_f32(&self) -> *const f32 {
        self.ptr as *const f32
    }
}

impl Drop for DevBuf {
    /// Return the (device, stage) pair to its pool (general or batched-decode)
    /// instead of freeing. IMMORTAL buffers (graph-recorded inputs) are
    /// intentionally leaked — their Drop is a no-op.
    fn drop(&mut self) {
        if self.immortal {
            return;
        }
        if !self.ptr.is_null() {
            buf_pool_release(self.dev, self.class, self.ptr, self.stage, self.batch);
        }
    }
}

/// Cached device-resident weight: keyed by the host Arc buffer pointer
/// Persistent fp8-dequant staging (grow-on-demand, NEVER freed per call).
/// The mmap TP preload fires ~21600 dequants; each call's malloc(2MB w8)
/// +malloc(sc)+malloc(4MB ptr)+free(w8)+free(sc) interleaves 2MB-page-sized
/// frees with 4MB retained outputs — the 2MB-page allocator fragments
/// (observed: 132GB free yet a 4MB malloc fails; EP mode's 5400 calls at
/// 8MB staging never fragments).
struct Fp8Stage {
    w8: *mut std::ffi::c_void,
    w8_cap: usize,
    sc: *mut std::ffi::c_void,
    sc_cap: usize,
}

/// (stable because the tensor is immutable), the host Arc is kept alive in
/// the cache so the pointer can never dangle.
struct CachedBuf {
    keep: Arc<Vec<f32>>,
    dev: *mut std::ffi::c_void,
    len: usize,
}

unsafe impl Send for CachedBuf {}
unsafe impl Sync for CachedBuf {}

/// A borrowed device pointer (no ownership — the cache frees it).
#[derive(Clone, Copy)]
struct DevRef {
    ptr: *mut std::ffi::c_void,
    len: usize,
}

impl DevRef {
    pub fn as_const_f32(&self) -> *const f32 {
        self.ptr as *const f32
    }
}

/// CUDA backend (v1: host↔device per op for activations; *weights* are
/// device-resident via the pointer-keyed cache — repeated uploads of the
/// same Arc'd weight tensor hit the cache, which is also the precondition
/// for CUDA-graph capture (stable device pointers across replays)).
/// MTP speculative-decoding buffers (fixed graph-capture addresses):
/// hf_dev = decode graph's h_final [hidden] (n=1), hf_v = verify graph's
/// h_final [2*hidden] (n=2: rows t_last/d1), hprev = draft's h input, and
/// per-GDN-layer (conv, gdn) ping-pong B scratch (verify writes B, accept
/// commits B→A, reject leaves A untouched).
pub struct MtpState {
    pub hf_dev: DevBuf,
    /// verify graph's h_final [N*hidden] (N = FERRITE_MTP_N rows; the
    /// commit kernel's hprev select reads row k-1).
    pub hf_v: DevBuf,
    pub hprev: DevBuf,
    /// per-GDN-layer (conv, gdn, conv_snaps, gdn_snaps) N-UNIFIED scratch:
    /// conv/gdn = the verify B states (the full t-loop chain),
    /// conv_snaps/gdn_snaps = the [N-1]-deep contiguous t-snapshot bases
    /// (snap i = A + tokens t_0..t_i, accept-(i+1)'s commit source — the
    /// kernel indexes base + i*len; the old fixed (conv0,gdn0,conv1,gdn1)
    /// pairs are snaps 0/1 of the same layout at N=3).
    pub scratch: Vec<(DevBuf, DevBuf, DevBuf, DevBuf)>,
    /// Single-kernel accept-commit plan: device-resident [n_gdn][6] pointer
    /// table (conv_a, gdn_a, conv_b, gdn_b, conv_snaps_base, gdn_snaps_base
    /// — the N-unified 6-pointer row; the kernel derives snap j-1 at
    /// base + (j-1)*len) + a pinned k slot. One ferrite_mtp_commit launch
    /// replaces 2*n_gdn cudaMemcpyAsync D2Ds + the hprev row select.
    pub commit: Option<MtpCommitPlan>,
    /// ZERO-H2D device token chain (user mandate: the entire decode loop
    /// must have no host-to-device transfers — D2H between steps is allowed
    /// for SSE). The token IDs never leave the device: argmax (device) →
    /// accept kernel (device) → embed kernel (device) → graph input
    /// (device) → next step. Host reads 8 bytes D2H per step (k +
    /// next_token for seq tracking + API response).
    ///
    /// [0] = last accepted token (written by accept kernel or initial prompt)
    /// [1..N-1] = the drafts d1..d_{N-1} (written by the draft chain argmax)
    pub tokens_dev: DevBuf,      // [N] i32 — the token chain (device)
    /// verify graph's argmax output [N] (a0..a_{N-1} — the graph writes
    /// here at replay; the accept kernel reads it to compare against the
    /// drafts)
    pub verify_argmax_dev: DevBuf, // [N] f32 — argmax of the verify graph
    /// accept kernel outputs (device, read D2H by host for seq + SSE)
    pub k_dev: DevBuf,            // [1] i32 — accept count 1..N
    pub next_token_dev: DevBuf,   // [1] i32 — next step's "last" token
    pub n_accepted_dev: DevBuf,   // [1] i32 — tokens accepted this step
    /// draft chain embeds (fixed bufs, embed_one kernel output; one per draft)
    pub emb_devs: Vec<DevBuf>,    // [N-1] × [hidden] f32 — draft i's embed
    /// draft chain h relay (fixed bufs): h_d[i] = draft i's MTP-residual h
    /// (mtp_forward's h_out), fed as draft i+1's h_prev. Fixed addresses are
    /// REQUIRED for the draft graphs (mega_d{seq}_{i}) — a pool-allocated h
    /// (the host path's fresh DevBuf) would move between capture and replay.
    /// The LAST draft exports no h (verify's hf_v commit replaces it), so
    /// only nd-1 = N-2 relays exist.
    pub h_d: Vec<DevBuf>,         // [N-2] × [hidden] f32 — draft i's h relay
    /// draft argmax outputs (device, from mtp_forward's argmax) — ONE
    /// contiguous [N-1] buffer (draft i's token at offset i; the accept
    /// kernel reads it as the d array).
    pub d_argmax_dev: DevBuf,     // [N-1] f32 — draft tokens (argmax outputs)
}

/// Device-resident commit pointer table for ferrite_mtp_commit. `plan`
/// packs 6 device pointers per GDN layer as f32 bit patterns (DevBuf is
/// f32-typed; 2 f32 per pointer). `k_pin` is a 4-byte cudaMallocHost slot —
/// the kernel reads it zero-copy at run time (k is only known AFTER the
/// verify graph replay returns argmax, so it cannot be baked into a graph).
/// `mtp_n` = the verify width (FERRITE_MTP_N) — the kernel's k range 1..=n.
pub struct MtpCommitPlan {
    pub plan: DevBuf,
    pub k_pin: *mut i32,
    pub n: usize,
    /// verify width N (FERRITE_MTP_N) — the commit kernel's k range 1..=N
    /// (k=N commits B; k=j<N commits snapshot j-1 at base + (j-1)*len).
    pub mtp_n: usize,
    pub conv_len: usize,
    pub gdn_len: usize,
    pub hidden: usize,
}

/// Device-resident fp8 weight (raw F8_E4M3 bytes + block scales). Send+Sync
/// raw pointers — owned by the fp8_map, freed on clear_weight_cache.
#[derive(Clone, Copy)]
pub struct Fp8Dev {
    pub w: *mut std::ffi::c_void,
    pub scale: *mut std::ffi::c_void,
    pub srows: i32,
    pub scols: i32,
}
unsafe impl Send for Fp8Dev {}
unsafe impl Sync for Fp8Dev {}

pub struct CudaBackend {
    stream: CuStream,
    /// Device index this backend is bound to. cudaSetDevice is THREAD-LOCAL:
    /// a TP cluster drives N backends from one thread, so every op must
    /// re-bind before allocating/launching (buffers must live on the same
    /// device as the stream).
    dev: i32,
    weights: std::sync::Mutex<std::collections::HashMap<(usize, usize), CachedBuf>>,
    /// CUDA graph capture state (driver-API handle for the instantiated
    /// graph exec; see the GraphCapable impl below).
    graph: std::sync::Mutex<GraphState>,
    /// Device-resident recurrent states, keyed (seq, layer) — GDN
    /// [h,dk,dk] state and conv tails. NOT pooled (must persist across
    /// tokens; pooled buffers would be reused by other ops).
    gdn_states: std::sync::Mutex<std::collections::HashMap<(u64, usize), DeviceState>>,
    conv_states: std::sync::Mutex<std::collections::HashMap<(u64, usize), DeviceState>>,
    /// DSA caches: device-resident k_nope/v/k_idx/k_gate per (seq, family),
    /// pre-allocated to max tokens. The CPU path grew host Vecs and cloned
    /// them per call (~MBs memcpy per DSA layer per token).
    dsa_caches: std::sync::Mutex<std::collections::HashMap<(u64, usize), DsaCacheState>>,
    /// Freed per-seq DSA caches, pooled by size (in floats). Reusing the SAME
    /// cudaMalloc VA avoids the driver's free+realloc remap path — measured
    /// 2026-09-09 as the trigger of the 2MB-aligned Xid 31 PDE faults on the
    /// batched step that follows a seq retirement.
    dsa_pool: std::sync::Mutex<std::collections::HashMap<usize, Vec<*mut std::ffi::c_void>>>,
    /// ptr -> its dsa_alloc size (floats), so dsa_release can pool it.
    dsa_sizes: std::sync::Mutex<std::collections::HashMap<usize, usize>>,
    /// MoE expert POINTER TABLES (fused GPU dispatch): per layer, three
    /// device buffers of e_local raw pointers (gate/up/down) into the
    /// dev_weight_bf16 cache — the fused kernels gather the selected
    /// experts' rows through them with zero host round-trips. Keyed by the
    /// first expert's gate tensor pointer (stable per layer).
    moe_ptrs: std::sync::Mutex<std::collections::HashMap<usize, MoePtrTable>>,
    /// FP8 bypass registry: f32 golden's (ptr, numel) → device (raw F8 bytes
    /// + 128x128 block scale). matmul_dev looks up EVERY weight by the
    /// caller-passed &Tensor — a hit serves the native-precision fp8 GEMV
    /// (half the bf16 HBM traffic), a miss keeps bf16. Keyed at set_fp8
    /// registration (same-name golden/fp8 shard pairing), zero call-site
    /// churn in the exec layer.
    fp8_map: std::sync::Mutex<std::collections::HashMap<(usize, usize), Fp8Dev>>,
    /// in_f -> (xq e4m3 buffer for n<=16, xs[n] scales, last x ptr, valid).
    /// The MMA gemv needs the x pre-quantized; ONE quant per layer serves all
    /// same-x gemvs. Buffers are never freed (a captured graph holds the
    /// addresses) and a capture never allocates.
    xq_cache: std::sync::Mutex<std::collections::HashMap<i32, (DevBuf, DevBuf, usize, bool)>>,
    /// fp8 expert pointer tables (per layer, keyed like moe_ptrs) — (w8,
    /// scale) device tables for the fused MoE kernels.
    moe_fp8_ptrs: std::sync::Mutex<std::collections::HashMap<usize, MoeFp8PtrTable>>,
    /// Batched per-seq GDN state tables: (layer, seq-set) → (conv_tbl,
    /// gdn_tbl) device pointer pairs — the batched kernels' device arrays
    /// of B state pointers. cudaMalloc'd + memcpy'd ONCE per composition
    /// (at the dry-run), cached (the capture re-uses the same pointers —
    /// no malloc during capture; the table content is FROZEN per
    /// composition: the (seq, layer) state addresses are stable). Purged
    /// when free_seq drops a member (the tables would dangle).
    gdn_tbl_cache: std::sync::Mutex<std::collections::HashMap<(usize, usize), (*mut std::ffi::c_void, *mut std::ffi::c_void)>>,
    /// Batched per-seq DSA tables: (family, seq-set) → 6 device pointer
    /// arrays ([B] k_nope / v / k_idx / k_gate cache ptrs + [B] PINNED
    /// t0 / total int ptrs — the kernels dereference the pinned ints
    /// zero-copy, host-written per step: graph-safe). cudaMalloc'd + memcpy'd
    /// once per composition; purged by free_seq when a member cache dies.
    dsa_tbl_cache: std::sync::Mutex<std::collections::HashMap<(usize, usize), DsaBatchTables>>,
    /// per-family dummy DSA cache for padded batch rows (see dsa_dummy)
    dsa_dummy_cache: std::sync::Mutex<std::collections::HashMap<usize, (*mut f32, *mut f32, *mut f32, *mut f32, *mut f32, *mut f32, *const i32, *const i32)>>,
    /// P2P one-shot AR state (v2 epoch+ping-pong; FERRITE_P2P): the per-rank
    /// persistent device buffers for the in-graph decode-chain all-reduce —
    /// see P2pArState. Set once at cluster setup (after the peers' UVA
    /// addresses are exchanged); the AR sites call p2p_ar_v2 (fallback to
    /// NCCL when unset). Mutex: phase 2 (tables) mutates after phase 1
    /// (alloc) under &self (the cluster setup); the decode path copies the
    /// small state out per call.
    p2p_ar: std::sync::Mutex<Option<P2pArState>>,
    /// W8A8 mega-quant scratch per in_f (v3 gemv): [amax_bits, cnt, cnt2, xs,
    /// xq[in_f]] — the cooperative in-kernel quant's barrier state + shared
    /// xq. Allocated once (lazy) per width; the kernel TAIL resets the
    /// barrier counters, so the buffer is reusable across calls (stream order).
    w8a8_scratch: std::sync::Mutex<std::collections::HashMap<usize, (*mut std::ffi::c_void, usize)>>,
    /// Persistent fp8-dequant staging (grow-on-demand, NEVER freed per call).
    /// Root cause this exists: the mmap TP preload fires ~21600 dequants
    /// (160 experts × 3 proj × 45 layers × rows/cols splits); each call's
    /// malloc(w8 2MB)+malloc(sc)+malloc(ptr 4MB)+free(w8)+free(sc) interleaves
    /// 2MB-page-sized frees with 4MB retained outputs — the 2MB-page
    /// allocator fragments (observed: 132GB free yet a 4MB malloc fails,
    /// EP mode's 5400 calls at 8MB staging never fragments). Reusing one
    /// staging buffer (sized to the largest request) removes the interleave.
    fp8_stage: std::sync::Mutex<Fp8Stage>,
    /// Preload output arena (bump allocator over ~1GB cudaMalloc blocks).
    /// B300 allocator quirk: 21600 small (4MB) cudaMalloc calls for the MoE-TP
    /// expert dequant outputs fail with cudaErrorMemoryAllocation at
    /// ~130GB used / 129GB free (the EP mode's 5400 16MB calls at 161GB
    /// never hit it) — the driver rejects small allocations once the count
    /// × interleave pattern crosses a threshold. Bump-slicing one 1GB block
    /// per ~256 outputs keeps the cudaMalloc count at ~90 total.
    bump: std::sync::Mutex<Vec<(*mut std::ffi::c_void, usize, usize)>>,
    /// Named CUDA graphs (per layer-segment): FERRITE_GRAPH_LAYER captures
    /// each segment's op sequence once and replays per token — the per-op
    /// launch gaps (~30μs × ~19 ops/layer) are the decode bottleneck after
    /// the device chains.
    graph_execs: std::sync::Mutex<std::collections::HashMap<String, usize>>,
    /// Fixed IO pointers of captured segment graphs (per name).
    graph_io: std::sync::Mutex<std::collections::HashMap<String, GraphIO>>,
    /// MTP speculative-decoding fixed buffers (see MtpState).
    pub mtp: std::sync::Mutex<Option<MtpState>>,
    /// Per-row GEMV for small-n matmul (n==2) — ONLY the MTP verify chain
    /// sets this: the n=2 tiled GEMM wastes a whole tile (verify 108ms vs
    /// 23ms). Prefill MUST keep the GEMM (its row-batched accumulation
    /// order sets the first greedy token; per-row flips it 背出师表→English).
    pub small_n_rows: std::sync::atomic::AtomicBool,
    /// cuBLAS handle for the batched m=16 decode GEMM (lazily created on the
    /// dry-run/replay path — never inside capture).
    cublas: std::sync::Mutex<*mut std::ffi::c_void>,
    /// pinned token-id slots (embed_expand's zero-copy kernel input; cached
    /// per n — decode graph n=1, verify graph n=3).
    pinned_ids_cache: std::sync::Mutex<std::collections::HashMap<usize, *mut i32>>,
}

// cudaStream_t is thread-safe (CUDA runtime serialises ops on a stream);
// the raw pointer is just an opaque handle.
unsafe impl Send for CudaBackend {}
unsafe impl Sync for CudaBackend {}

impl Drop for CudaBackend {
    fn drop(&mut self) {
        self.clear_weight_cache();
        for store in [self.gdn_states.get_mut().unwrap(), self.conv_states.get_mut().unwrap()] {
            for (_, st) in store.drain() {
                unsafe { cudaFree(st.ptr) };
            }
        }
    }
}

impl Default for CudaBackend {
    fn default() -> Self {
        Self::new()
    }
}

// fp8 dequant D2H verify counter (first 3 per process — mmap debug only)
static FP8_DBG_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
// hc_pre input dump counter (first call per process — mmap garbage hunt)
static HC_DBG_ONCE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

impl CudaBackend {
    /// Loads libferrite_kernels.so's dependency closure (cudart) and
    /// creates a stream. Call after the .so is in the loader path.
    pub fn new() -> Self {
        // The extern symbols resolve from libcudart which is linked by
        // libferrite_kernels.so; loading that .so first is the caller's
        // job (see `with_library`).
        let mut stream: CuStream = std::ptr::null_mut();
        let e = unsafe { cudaStreamCreate(&mut stream) };
        if e != 0 {
            panic!("cudaStreamCreate failed: {e} (is libferrite_kernels.so loaded? see CudaBackend::with_library)");
        }
        CudaBackend {
            stream,
            dev: 0,
            weights: std::sync::Mutex::new(std::collections::HashMap::new()),
            graph: std::sync::Mutex::new(GraphState::default()),
            gdn_states: std::sync::Mutex::new(std::collections::HashMap::new()),
            conv_states: std::sync::Mutex::new(std::collections::HashMap::new()),
            dsa_caches: std::sync::Mutex::new(std::collections::HashMap::new()),
            dsa_pool: std::sync::Mutex::new(std::collections::HashMap::new()),
            dsa_sizes: std::sync::Mutex::new(std::collections::HashMap::new()),
            moe_ptrs: std::sync::Mutex::new(std::collections::HashMap::new()),
            fp8_map: std::sync::Mutex::new(std::collections::HashMap::new()),
            xq_cache: std::sync::Mutex::new(std::collections::HashMap::new()),
            moe_fp8_ptrs: std::sync::Mutex::new(std::collections::HashMap::new()),
            gdn_tbl_cache: std::sync::Mutex::new(std::collections::HashMap::new()),
            dsa_tbl_cache: std::sync::Mutex::new(std::collections::HashMap::new()),
            dsa_dummy_cache: std::sync::Mutex::new(std::collections::HashMap::new()),
            p2p_ar: std::sync::Mutex::new(None),
            w8a8_scratch: std::sync::Mutex::new(std::collections::HashMap::new()),
            fp8_stage: std::sync::Mutex::new(Fp8Stage { w8: std::ptr::null_mut(), w8_cap: 0, sc: std::ptr::null_mut(), sc_cap: 0 }),
            bump: std::sync::Mutex::new(Vec::new()),
            graph_execs: std::sync::Mutex::new(std::collections::HashMap::new()),
            graph_io: std::sync::Mutex::new(std::collections::HashMap::new()),
            mtp: std::sync::Mutex::new(None),
            small_n_rows: std::sync::atomic::AtomicBool::new(false),
            cublas: std::sync::Mutex::new(std::ptr::null_mut()),
            pinned_ids_cache: std::sync::Mutex::new(std::collections::HashMap::new()),
        }
    }

    /// Bind this backend's device as the calling thread's current device.
    /// cudaSetDevice is thread-local state; TP ranks all call ops from the
    /// main thread, so each op entry re-binds before cudaMalloc/launch.
    /// PUBLIC: device-chain call sites (attn_shard) allocate DevBufs
    /// BEFORE the first op — cudaMalloc binds to the CALLING thread's
    /// current device, which in a fan_out thread is another rank's.
    #[inline]
    pub fn enter(&self) {
        unsafe {
            cudaSetDevice(self.dev);
        }
    }

    /// Upload a weight tensor to the device ONCE — subsequent calls with
    /// the same host Arc buffer hit the cache (pointer + length keyed; the
    /// Arc is kept alive inside the cache so the key can never dangle).
    /// This kills the per-op weight H2D of the naive path and is the
    /// precondition for CUDA-graph capture (stable device pointers).
    fn dev_weight(&self, t: &Tensor) -> Result<DevRef> {
        let key = (t.as_slice().as_ptr() as usize, t.numel());
        let mut cache = self.weights.lock().unwrap();
        if let Some(cb) = cache.get(&key) {
            if cb.len == t.numel() {
                return Ok(DevRef { ptr: cb.dev, len: cb.len });
            }
        }
        // MMAP PLACEHOLDER GUARD: a 4-elem stub that missed the f32 cache
        // would cudaMemcpy numel*4 bytes from a 16-byte Vec — a silent heap
        // OOB read → garbage f32 weights → garbage text with ZERO errors
        // logged. THE ROOT CAUSE of the mmap "!!!" bug: legacy-loaded weights
        // are REAL f32 tensors (dev_weight falls through and uploads them),
        // but mmap placeholders are 4-elem stubs — the fallthrough read heap
        // garbage (conv_w / indexer ape / every f32-consumer 2-D weight the
        // legacy path uploaded lazily on first dev_weight call).
        // RECOVERY: the mmap preloaded the same weight as bf16 residency
        // (key numel<<1|1 via preload_bf16_raw). Widen it on device
        // (bf16→f32, bit-identical to the legacy path's checkpoint
        // bf16→f32 widen) and register the f32 key.
        if t.as_slice().len() < t.numel() {
            let bkey = (t.as_slice().as_ptr() as usize, t.numel() << 1 | 1);
            if let Some(bb) = cache.get(&bkey) {
                if bb.len == t.numel() {
                    let numel = t.numel();
                    eprintln!(
                        "[widen] f32-consumer weight: numel={numel} shape={:?} — widening bf16 residency",
                        t.shape.0
                    );
                    let mut f32p: *mut std::ffi::c_void = std::ptr::null_mut();
                    ck(unsafe { cudaMalloc(&mut f32p, numel * 4) }, "dev_weight bf16→f32 widen malloc")?;
                    let conv = (|| -> Result<()> {
                        ck(
                            unsafe { ferrite_bf16_to_f32(bb.dev as *const _, f32p as *mut f32, numel as i64, self.stream) },
                            "dev_weight bf16→f32 widen",
                        )
                    })();
                    self.sync()?;
                    conv?;
                    cache.insert(key, CachedBuf { keep: t.data.clone(), dev: f32p, len: numel });
                    return Ok(DevRef { ptr: f32p, len: numel });
                }
            }
            return Err(FerriteError::InvalidArg(format!(
                "dev_weight: placeholder stub ({} bytes data vs {} numel) missed both f32 and bf16 caches — the weight was never preloaded",
                t.as_slice().len(),
                t.numel()
            )));
        }
        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        ck(unsafe { cudaMalloc(&mut ptr, t.numel() * 4) }, "weight malloc")?;
        ck(
            unsafe { cudaMemcpy(ptr, t.as_slice().as_ptr() as *const _, t.numel() * 4, CUDA_MEMCPY_H2D) },
            "weight H2D",
        )?;
        cache.insert(key, CachedBuf { keep: t.data.clone(), dev: ptr, len: t.numel() });
        Ok(DevRef { ptr, len: t.numel() })
    }

    /// Upload a weight tensor to the device ONCE in **bf16** — the resident
    /// layout for large matmul weights. A 285GB/TP4-rank f32 shard does not
    /// fit a 275GB B300; bf16 halves it to 142GB (TileRT's resident-weights
    /// model). The kernel converts bf16→f32 in registers; x/out stay f32.
    ///
    /// Large weights (≥8M elements = 32MB f32) convert ON THE GPU: the f32
    /// source is streamed to a scratch buffer in chunks and a kernel packs
    /// bf16 in place — CPU-side packing of 292GB/rank was the warmup
    /// bottleneck (~150s/thread). Small weights pack on the CPU (the
    /// bit-shift loop is vector-friendly). Both paths use identical
    /// truncation semantics (f32 high bits), so parity holds.
    fn dev_weight_bf16(&self, t: &Tensor) -> Result<DevRef> {
        let key = (t.as_slice().as_ptr() as usize, t.numel() << 1 | 1);
        let mut cache = self.weights.lock().unwrap();
        if let Some(cb) = cache.get(&key) {
            if cb.len == t.numel() {
                return Ok(DevRef { ptr: cb.dev, len: cb.len });
            }
        }
        // MMAP DIAGNOSTIC: placeholder (4-elem stub) falling through to the
        // upload path reads 4 zeros as the full weight → GARBAGE. This should
        // NEVER happen in the mmap path (direct_preload_shard preloads all
        // weights). If it fires, the preload missed this weight's cache key.
        if t.as_slice().len() < t.numel() {
            eprintln!(
                "[dev_weight_bf16] CACHE MISS placeholder: numel={} data_len={} key=({},{}) — uploading 4-elem stub as full weight → GARBAGE",
                t.numel(), t.as_slice().len(), t.as_slice().as_ptr() as usize, t.numel() << 1 | 1
            );
        }
        let n = t.numel();
        let src = t.as_slice();
        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        ck(unsafe { cudaMalloc(&mut ptr, n * 2) }, "weight bf16 malloc")?;
        if n >= 8 << 20 {
            // GPU-side conversion: stream f32 chunks to a scratch buffer and
            // pack bf16 in place into the resident allocation. **Each chunk's
            // H2D must wait for the previous kernel** — blocking cudaMemcpy
            // does NOT order against stream-queued kernels (the next chunk's
            // H2D overwrites the scratch while the previous kernel is still
            // reading it — corrupted weights → NaN downstream; this exact
            // race is why the equivalence tests (small weights, CPU pack
            // path) all passed while serve (25M-element qkv_proj, 4 chunks)
            // produced all-NaN attention outputs).
            const CHUNK: usize = 32 << 20; // 32M f32 = 128MB per step
            let mut scratch: *mut std::ffi::c_void = std::ptr::null_mut();
            ck(unsafe { cudaMalloc(&mut scratch, CHUNK * 4) }, "bf16 scratch malloc")?;
            let conv = (|| -> Result<()> {
                for (i, chunk) in src.chunks(CHUNK).enumerate() {
                    self.sync()?; // previous kernel finished reading scratch
                    ck(
                        unsafe { cudaMemcpy(scratch, chunk.as_ptr() as *const _, chunk.len() * 4, CUDA_MEMCPY_H2D) },
                        "bf16 chunk H2D",
                    )?;
                    ck(
                        unsafe {
                            ferrite_f32_to_bf16(
                                scratch as *const f32,
                                (ptr as *mut u8).add(i * CHUNK * 2) as *mut _,
                                chunk.len() as i64,
                                self.stream,
                            )
                        },
                        "bf16 GPU convert",
                    )?;
                }
                self.sync()
            })();
            unsafe { cudaFree(scratch) };
            conv?;
        } else {
            // pack f32 → bf16 on the CPU (truncate to high bits; PyTorch
            // bf16 semantics — matches ferrite_f32_to_bf16 exactly)
            let mut packed: Vec<u16> = vec![0u16; src.len()];
            for (dst, v) in packed.iter_mut().zip(src.iter()) {
                *dst = (v.to_bits() >> 16) as u16;
            }
            ck(
                unsafe { cudaMemcpy(ptr, packed.as_ptr() as *const _, packed.len() * 2, CUDA_MEMCPY_H2D) },
                "weight bf16 H2D",
            )?;
        }
        cache.insert(key, CachedBuf { keep: t.data.clone(), dev: ptr, len: n });
        Ok(DevRef { ptr, len: n })
    }

    /// Register an fp8 bypass for `golden` (the f32 weight Tensor callers
    /// keep passing around): uploads the raw F8 bytes + block scales, keyed
    /// by the golden's (ptr, numel). Every matmul_dev on that Tensor from
    /// then on serves from the fp8 kernel — zero call-site changes; a
    /// weight never registered (or a shard seam misaligned for fp8) just
    /// keeps the bf16 path. Registration is idempotent (re-register hits
    /// the cache and returns).
    pub fn register_fp8(
        &self,
        golden: &Tensor,
        rows: usize,
        cols: usize,
        data: &[u8],
        scale: &[f32],
    ) -> Result<()> {
        self.enter();
        let key = (golden.as_slice().as_ptr() as usize, golden.numel());
        {
            let m = self.fp8_map.lock().unwrap();
            if m.contains_key(&key) {
                return Ok(());
            }
        }
        if data.len() != rows * cols {
            return Err(FerriteError::InvalidArg(format!(
                "register_fp8: data {} != rows*cols {}*{}",
                data.len(), rows, cols
            )));
        }
        let srows = rows.div_ceil(128);
        let scols = cols.div_ceil(128);
        if scale.len() != srows * scols {
            return Err(FerriteError::InvalidArg(format!(
                "register_fp8: scale {} != {}*{} (block 128)",
                scale.len(), srows, scols
            )));
        }
        // BUMP-allocated (B300 quirk: the MoE-TP experts fire ~39k register_fp8
        // calls (288 experts × 3 × 45 layers) — direct cudaMalloc hits the
        // driver's small-allocation degradation (the same class as the 21600
        // preload OOM); bump-slicing 1GB blocks keeps the malloc count ~90.
        // The fp8 weights live for the process lifetime (bump blocks are
        // never freed — same contract as the bf16 weight cache).
        let w = self.bump_alloc(data.len().max(256))?;
        let sc = self.bump_alloc(scale.len() * 4)?;
        ck(unsafe { cudaMemcpy(w, data.as_ptr() as *const _, data.len(), CUDA_MEMCPY_H2D) }, "fp8 w H2D")?;
        ck(unsafe { cudaMemcpy(sc, scale.as_ptr() as *const _, scale.len() * 4, CUDA_MEMCPY_H2D) }, "fp8 scale H2D")?;
        self.fp8_map.lock().unwrap().insert(
            key,
            Fp8Dev { w, scale: sc, srows: srows as i32, scols: scols as i32 },
        );
        Ok(())
    }

    fn fp8_lookup(&self, t: &Tensor) -> Option<Fp8Dev> {
        let key = (t.as_slice().as_ptr() as usize, t.numel());
        self.fp8_map.lock().unwrap().get(&key).copied()
    }

    /// Number of registered fp8 bypass weights (diagnostics).
    pub fn fp8_registered(&self) -> usize {
        self.fp8_map.lock().unwrap().len()
    }

    /// Is this f32 golden served from the registered fp8 bypass? Preload
    /// skips the bf16 upload for hits (matmul_dev serves fp8; only
    /// non-fp8-able weights and the fused-MoE bf16 ptr tables stay bf16) —
    /// the fp8 + bf16 double residency OOM'd rank 3 at ~213GB/275GB.
    pub fn fp8_hit(&self, t: &Tensor) -> bool {
        self.fp8_lookup(t).is_some()
    }

    /// W8A8 v3.1 scratch for this width (lazy, one cudaMalloc + zero per
    /// in_f; layout [amax(int bits), cnt, cnt2, cnt3, pad, xs(f32), xq[in_f]]
    /// — 32B header + xq bytes; the kernel tail resets the barrier counters
    /// via the cnt3 full-grid vote (stream order serializes callers).
    fn w8a8_scratch_ptr(&self, in_f: usize) -> Result<*mut u32> {
        let key = in_f;
        if let Some(&(p, _)) = self.w8a8_scratch.lock().unwrap().get(&key) {
            return Ok(p as *mut u32);
        }
        let bytes = 32 + in_f;
        let mut p: *mut std::ffi::c_void = std::ptr::null_mut();
        ck(unsafe { cudaMalloc(&mut p, bytes) }, "w8a8 scratch malloc")?;
        ck(unsafe { cudaMemset(p, 0, bytes) }, "w8a8 scratch zero")?;
        self.w8a8_scratch.lock().unwrap().insert(key, (p, bytes));
        Ok(p as *mut u32)
    }

    /// Preload a weight into device-resident storage (bf16 for 2-D matmul
    /// weights — run_matmul reads bf16 exclusively; 1-D tensors stay f32
    /// for the elementwise kernels). Serve calls this over every shard
    /// weight at startup so inference never uploads weights again (the
    /// TileRT model: weights resident, only activations cross the bus).
    pub fn preload_weight(&self, t: &Tensor) -> Result<()> {
        if t.numel() == 0 {
            return Ok(()); // TP shard placeholder (empty expert slice)
        }
        if t.as_slice().len() < t.numel() {
            // fp8 single-store placeholder: no bf16 to upload — the fp8 map
            // serves it (register_fp8 uploaded the F8 bytes at set_fp8).
            return Ok(());
        }
        // cudaSetDevice is THREAD-LOCAL: a TP cluster drives N backends from
        // one thread, so every entry point must re-bind before cudaMalloc —
        // without this, ALL ranks' weights malloc onto whatever device the
        // thread last touched (observed: 4 ranks' 142GB each piling onto
        // GPU 7 at 247GB).
        self.enter();
        if t.shape.0.len() >= 2 {
            self.dev_weight_bf16(t).map(|_| ())
        } else {
            self.dev_weight(t).map(|_| ())
        }
    }

    /// bf16 raw → resident bf16 via a COLUMN window (TP col-split: down/
    /// o_proj/eh_proj) — cudaMemcpy2D strided H2D straight from the mmap
    /// slice (host pitch = full row, width = the shard's col window). No
    /// CPU gather pass: the page cache streams the strided window.
    pub fn preload_bf16_col_raw(
        &self,
        placeholder: &Tensor,
        src: &[u8],       // mmap slice of the FULL [rows, full_cols] bf16 weight
        rows: usize,
        full_cols: usize,
        c0: usize,
        c1: usize,
    ) -> Result<()> {
        let shard_cols = c1 - c0;
        let numel = placeholder.numel();
        if numel != rows * shard_cols {
            return Err(FerriteError::InvalidArg(format!(
                "preload_bf16_col_raw: placeholder numel {numel} != {rows}x{shard_cols}"
            )));
        }
        if src.len() != rows * full_cols * 2 {
            return Err(FerriteError::InvalidArg(format!(
                "preload_bf16_col_raw: src {} != {rows}x{full_cols} bf16",
                src.len()
            )));
        }
        self.enter();
        let key = (placeholder.as_slice().as_ptr() as usize, numel << 1 | 1);
        {
            let cache = self.weights.lock().unwrap();
            if let Some(cb) = cache.get(&key) {
                if cb.len == numel {
                    return Ok(());
                }
            }
        }
        let ptr = self.bump_alloc(numel * 2)?;
        ck(
            unsafe {
                cudaMemcpy2D(
                    ptr,
                    shard_cols * 2,                       // dst pitch: shard row bytes
                    src.as_ptr().add(c0 * 2) as *const _,  // first window column
                    full_cols * 2,                         // src pitch: full row bytes (mmap)
                    shard_cols * 2,                        // width: window bytes
                    rows,                                  // height
                    CUDA_MEMCPY_H2D,
                )
            },
            "bf16_col H2D (mmap strided window)",
        )?;
        self.weights.lock().unwrap().insert(
            key,
            CachedBuf { keep: placeholder.data.clone(), dev: ptr, len: numel },
        );
        Ok(())
    }

    /// fp8 e4m3 + block scales → resident bf16, dequantized on the GPU, for
    /// a TP COLUMN window (down_proj col-split): the fp8 window is a
    /// cudaMemcpy2D strided H2D from the mmap slice; the scale window is
    /// gathered on the host (KBs — [srows, c0/128..c1/128) of [srows,
    /// full_cols/128], the only small CPU pass on the direct path).
    /// Bump-allocate the output (see bump_alloc — B300 small-alloc quirk).
    pub fn preload_fp8_col_dequant(
        &self,
        placeholder: &Tensor,
        src: &[u8],        // mmap slice of the FULL [rows, full_cols] fp8 weight
        scale_full: &[f32], // full [srows, full_cols/128] scales (host-side, tiny)
        rows: usize,
        full_cols: usize,
        c0: usize,
        c1: usize,
    ) -> Result<()> {
        let shard_cols = c1 - c0;
        let numel = placeholder.numel();
        if numel != rows * shard_cols {
            return Err(FerriteError::InvalidArg(format!(
                "preload_fp8_col_dequant: numel {numel} != {rows}x{shard_cols}"
            )));
        }
        let srows = rows.div_ceil(128);
        let full_scols = full_cols.div_ceil(128);
        let sc0 = c0 / 128;
        let sc1 = c1.div_ceil(128);
        let scols = sc1 - sc0;
        if scale_full.len() != srows * full_scols {
            return Err(FerriteError::InvalidArg(format!(
                "preload_fp8_col_dequant: scale len {} != {srows}x{full_scols}",
                scale_full.len()
            )));
        }
        self.enter();
        let key = (placeholder.as_slice().as_ptr() as usize, numel << 1 | 1);
        {
            let cache = self.weights.lock().unwrap();
            if let Some(cb) = cache.get(&key) {
                if cb.len == numel {
                    return Ok(());
                }
            }
        }
        // fp8 window: strided H2D from the mmap slice (PERSISTENT fp8_stage —
        // same anti-fragmentation rationale as preload_fp8_dequant)
        let mut stage = self.fp8_stage.lock().unwrap();
        if stage.w8_cap < numel {
            if !stage.w8.is_null() {
                unsafe { cudaFree(stage.w8) };
            }
            ck(unsafe { cudaMalloc(&mut stage.w8, numel) }, "fp8_col w8 stage alloc")?;
            stage.w8_cap = numel;
        }
        // scale window: host gather (KBs), then H2D
        let mut scale: Vec<f32> = Vec::with_capacity(srows * scols);
        for sr in 0..srows {
            let base = sr * full_scols;
            scale.extend_from_slice(&scale_full[base + sc0..base + sc1]);
        }
        if stage.sc_cap < scale.len() * 4 {
            if !stage.sc.is_null() {
                unsafe { cudaFree(stage.sc) };
            }
            ck(unsafe { cudaMalloc(&mut stage.sc, scale.len() * 4) }, "fp8_col sc stage alloc")?;
            stage.sc_cap = scale.len() * 4;
        }
        let w8 = stage.w8;
        ck(
            unsafe {
                cudaMemcpy2D(
                    w8,
                    shard_cols,
                    src.as_ptr().add(c0) as *const _,
                    full_cols,
                    shard_cols,
                    rows,
                    CUDA_MEMCPY_H2D,
                )
            },
            "fp8_col H2D (mmap strided window)",
        )?;
        let sc = stage.sc;
        ck(unsafe { cudaMemcpy(sc, scale.as_ptr() as *const _, scale.len() * 4, CUDA_MEMCPY_H2D) }, "fp8_col scale H2D")?;
        // GPU dequant → resident bf16 (bump arena — see bump_alloc)
        let ptr = self.bump_alloc(numel * 2)?;
        let conv = (|| -> Result<()> {
            ck(
                unsafe {
                    ferrite_dequant_e4m3_block(
                        w8 as *const u8,
                        sc as *const f32,
                        ptr,
                        rows as i64,
                        shard_cols as i64,
                        srows as i32,
                        scols as i32,
                        self.stream,
                    )
                },
                "fp8_col GPU dequant",
            )
        })();
        self.sync()?;
        drop(stage);
        conv?;
        self.weights.lock().unwrap().insert(
            key,
            CachedBuf { keep: placeholder.data.clone(), dev: ptr, len: numel },
        );
        Ok(())
    }


    // ============================================================
    // DIRECT-LOAD preload API (mmap disk→GPU — weights never materialize
    // on the CPU): the three entry points pair with `ferrite_model::direct`
    // WeightViews. Cache keys are the PLACEHOLDER tensor's (ptr, numel)
    // — the same keys dev_weight/dev_weight_bf16 compute at runtime, so
    // the engine's zero-change lookups hit the direct-uploaded buffers.
    // ============================================================

    /// bf16 raw segments → resident bf16 (NO conversion — the checkpoint's
    /// bf16 IS the resident layout). Segments concatenate in order (fused
    /// qkv: q/k/v row-blocks); each cudaMemcpy's straight from the mmap
    /// slice (page cache → PCIe DMA).
    pub fn preload_bf16_raw(&self, placeholder: &Tensor, segments: &[&[u8]]) -> Result<()> {
        let numel = placeholder.numel();
        if numel == 0 {
            return Ok(()); // TP shard placeholder (empty expert slice)
        }
        let total_bytes: usize = segments.iter().map(|s| s.len()).sum();
        if total_bytes != numel * 2 {
            return Err(FerriteError::InvalidArg(format!(
                "preload_bf16_raw: {} bytes != numel {numel} * 2 (bf16)",
                total_bytes
            )));
        }
        self.enter();
        let key = (placeholder.as_slice().as_ptr() as usize, numel << 1 | 1);
        {
            let cache = self.weights.lock().unwrap();
            if let Some(cb) = cache.get(&key) {
                if cb.len == numel {
                    return Ok(()); // idempotent (re-preload of the same weight)
                }
            }
        }
        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        ck(unsafe { cudaMalloc(&mut ptr, numel * 2) }, "bf16_raw malloc")?;
        let mut off = 0usize;
        for seg in segments {
            if seg.is_empty() {
                continue;
            }
            ck(
                unsafe {
                    cudaMemcpy((ptr as *mut u8).add(off) as *mut _, seg.as_ptr() as *const _, seg.len(), CUDA_MEMCPY_H2D)
                },
                "bf16_raw segment H2D (mmap → device)",
            )?;
            off += seg.len();
        }
        // D2H readback diagnostic: verify the device data matches the source
        // (mmap → device memcpy is a faithful copy, but let's PROVE it)
        if std::env::var_os("FERRITE_MMAP_DEBUG").is_some() && numel >= 8 {
            let mut host_back: Vec<u8> = vec![0u8; 16];
            ck(
                unsafe { cudaMemcpy(host_back.as_mut_ptr() as *mut _, ptr, 16, CUDA_MEMCPY_D2H) },
                "bf16_raw D2H verify",
            )?;
            let src_first: Vec<u8> = segments.iter().flat_map(|s| s.iter().take(16).copied()).take(16).collect();
            let match_ = host_back == src_first[..16.min(src_first.len())];
            eprintln!(
                "[mmap-dbg] bf16_raw D2H verify: numel={} device={:02x?} src={:02x?} match={}",
                numel, &host_back[..8], &src_first[..8.min(src_first.len())], match_
            );
        }
        self.weights.lock().unwrap().insert(
            key,
            CachedBuf { keep: placeholder.data.clone(), dev: ptr, len: numel },
        );
        Ok(())
    }

    /// Bump-allocate from ~1GB arena blocks (B300 allocator quirk: 21600
    /// small 4MB cudaMalloc calls for MoE-TP expert dequant outputs fail
    /// with cudaErrorMemoryAllocation at ~130GB used / 129GB free — the EP
    /// mode's 5400 16MB calls at 161GB never hit it; the driver's small-
    /// allocation path degrades past ~14k live allocations). Preload outputs
    /// live for the process lifetime (weights cache) — bump blocks are
    /// never freed. Slicing one 1GB block per ~256 outputs keeps the total
    /// cudaMalloc count at ~90.
    fn bump_alloc(&self, bytes: usize) -> Result<*mut std::ffi::c_void> {
        let mut blocks = self.bump.lock().unwrap();
        // CHUNK is env-tunable (FERRITE_BUMP_CHUNK_GB): the driver's small/large
        // allocation paths differ wildly on this B300 driver (documented below);
        // fewer, bigger blocks = fewer page-table mappings to get wrong.
        static CHUNK: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        let chunk = *CHUNK.get_or_init(|| {
            std::env::var("FERRITE_BUMP_CHUNK_GB")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(1)
                << 30
        });
        let need = bytes.next_multiple_of(256);
        if let Some((base, cap, used)) = blocks.last_mut() {
            if *cap - *used >= need {
                let ptr = unsafe { (*base as *mut u8).add(*used) as *mut std::ffi::c_void };
                *used += need;
                return Ok(ptr);
            }
        }
        let sz = need.max(chunk);
        let mut base: *mut std::ffi::c_void = std::ptr::null_mut();
        ck(unsafe { cudaMalloc(&mut base, sz) }, "bump arena block")?;
        blocks.push((base, sz, need));
        Ok(base)
    }

    /// fp8 e4m3 + 128×128 block scales → resident bf16, dequantized ON
    /// THE GPU (the legacy path's `to_f32` + `dequant_block` + CPU pack —
    /// 3 passes over the weight + a 4× f32 materialization — replaced by
    /// one fp8 H2D + one kernel). H2D sources are the mmap slices.
    pub fn preload_fp8_dequant(
        &self,
        placeholder: &Tensor,
        data: &[u8],
        scale: &[f32],
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        let numel = placeholder.numel();
        if numel == 0 {
            return Ok(());
        }
        if data.len() != rows * cols {
            return Err(FerriteError::InvalidArg(format!(
                "preload_fp8_dequant: data {} != rows*cols {rows}*{cols}",
                data.len()
            )));
        }
        let srows = rows.div_ceil(128);
        let scols = cols.div_ceil(128);
        if scale.len() != srows * scols {
            return Err(FerriteError::InvalidArg(format!(
                "preload_fp8_dequant: scale {} != {srows}*{scols} (block 128)",
                scale.len()
            )));
        }
        self.enter();
        let key = (placeholder.as_slice().as_ptr() as usize, numel << 1 | 1);
        {
            let cache = self.weights.lock().unwrap();
            if let Some(cb) = cache.get(&key) {
                if cb.len == numel {
                    return Ok(());
                }
            }
        }        // staging: PERSISTENT fp8_stage (grow-on-demand, reused across calls —
        // the mmap TP preload fires ~21600 dequants; each call's malloc(2MB
        // w8)+free interleaved with 4MB retained outputs fragments the
        // 2MB-page allocator: observed 132GB free yet a 4MB malloc fails).
        let mut stage = self.fp8_stage.lock().unwrap();
        if stage.w8_cap < data.len() {
            if !stage.w8.is_null() {
                unsafe { cudaFree(stage.w8) };
            }
            ck(unsafe { cudaMalloc(&mut stage.w8, data.len()) }, "fp8_dequant w8 stage alloc")?;
            stage.w8_cap = data.len();
        }
        if stage.sc_cap < scale.len() * 4 {
            if !stage.sc.is_null() {
                unsafe { cudaFree(stage.sc) };
            }
            ck(unsafe { cudaMalloc(&mut stage.sc, scale.len() * 4) }, "fp8_dequant sc stage alloc")?;
            stage.sc_cap = scale.len() * 4;
        }
        let w8 = stage.w8;
        let sc = stage.sc;
        ck(unsafe { cudaMemcpy(w8, data.as_ptr() as *const _, data.len(), CUDA_MEMCPY_H2D) }, "fp8_dequant w8 H2D")?;
        ck(unsafe { cudaMemcpy(sc, scale.as_ptr() as *const _, scale.len() * 4, CUDA_MEMCPY_H2D) }, "fp8_dequant scale H2D")?;
        // resident bf16 output — BUMP arena (B300 quirk: 21600 small cudaMalloc
        // calls for MoE-TP expert dequant outputs fail with OOM at ~130GB
        // used / 129GB free; bump-slicing 1GB blocks keeps the cudaMalloc
        // count at ~90. EP mode's 5400 16MB calls at 161GB never hit it.)
        let ptr = self.bump_alloc(numel * 2)?;
        let conv = (|| -> Result<()> {
            ck(
                unsafe {
                    ferrite_dequant_e4m3_block(
                        w8 as *const u8,
                        sc as *const f32,
                        ptr,
                        rows as i64,
                        cols as i64,
                        srows as i32,
                        scols as i32,
                        self.stream,
                    )
                },
                "fp8 GPU dequant",
            )
        })();
        // staging NOT freed — persistent fp8_stage reuse (the mmap TP preload
        // fires ~21600 dequants; per-call free+malloc interleaved with 4MB
        // retained outputs fragments the 2MB-page allocator).
        self.sync()?;
        drop(stage);
        conv?;
        // D2H verify: read back first 8 bf16 elements of the dequant output and
        // compare with the CPU reference (e4m3 decode × scale → f32 → bf16).
        // Catches kernel/scale-indexing/staging bugs on the 95%-of-params
        // expert path (fp8 dequant is mmap-only — the legacy path dequants on
        // CPU via dequant_block; a divergence here = garbage experts →
        // constant-output bug). One run, first 3 experts per rank.
        if std::env::var_os("FERRITE_MMAP_DEBUG").is_some()
            && numel >= 8
            && rows >= 1
            && cols >= 8
            && FP8_DBG_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 3
        {
            let mut host_back: Vec<u8> = vec![0u8; 16];
            ck(
                unsafe { cudaMemcpy(host_back.as_mut_ptr() as *mut _, ptr, 16, CUDA_MEMCPY_D2H) },
                "fp8_dequant D2H verify",
            )?;
            let mut expect: Vec<u8> = Vec::with_capacity(16);
            for c in 0..8usize.min(cols).min(data.len()) {
                let b = data[c];
                let sign = if b & 0x80 != 0 { -1.0f32 } else { 1.0f32 };
                let e = ((b >> 3) & 0x0f) as i32;
                let m = (b & 0x07) as i32;
                let v: f32 = if e == 0 {
                    sign * (m as f32 / 8.0) * 2f32.powi(-6)
                } else {
                    sign * (1.0 + m as f32 / 8.0) * 2f32.powi(e - 7)
                };
                let sc_col = (c / 128).min(scols.saturating_sub(1));
                let f = v * scale[sc_col];
                let bits = f.to_bits() >> 16;
                expect.extend_from_slice(&(bits as u16).to_le_bytes());
            }
            let n8 = expect.len().min(8);
            let m = &host_back[..n8] == &expect[..n8];
            eprintln!(
                "[fp8-verify] numel={} {}x{} device={:02x?} expect={:02x?} match={}",
                numel, rows, cols, &host_back[..n8], &expect[..n8], m
            );
        }
        self.weights.lock().unwrap().insert(
            key,
            CachedBuf { keep: placeholder.data.clone(), dev: ptr, len: numel },
        );
        Ok(())
    }

    /// bf16 raw → resident f32 (GPU expand — the embed table's f32 cache:
    /// the bf16→f32 widening happens on device; 2.5 GB never crosses a CPU
    /// conversion pass). Cache key is the f32 dev_weight's (ptr, numel).
    pub fn preload_bf16_to_f32_raw(&self, placeholder: &Tensor, bf16_bytes: &[u8]) -> Result<()> {
        let numel = placeholder.numel();
        if numel == 0 {
            return Ok(());
        }
        if bf16_bytes.len() != numel * 2 {
            return Err(FerriteError::InvalidArg(format!(
                "preload_bf16_to_f32_raw: {} bytes != numel {numel} * 2",
                bf16_bytes.len()
            )));
        }
        self.enter();
        let key = (placeholder.as_slice().as_ptr() as usize, numel);
        {
            let cache = self.weights.lock().unwrap();
            if let Some(cb) = cache.get(&key) {
                if cb.len == numel {
                    return Ok(());
                }
            }
        }
        let mut stage = self.fp8_stage.lock().unwrap();
        if stage.w8_cap < numel * 2 {
            if !stage.w8.is_null() {
                unsafe { cudaFree(stage.w8) };
            }
            ck(unsafe { cudaMalloc(&mut stage.w8, numel * 2) }, "bf16f32 staging stage alloc")?;
            stage.w8_cap = numel * 2;
        }
        let staging = stage.w8;
        ck(
            unsafe { cudaMemcpy(staging, bf16_bytes.as_ptr() as *const _, numel * 2, CUDA_MEMCPY_H2D) },
            "bf16f32 H2D (mmap → device)",
        )?;
        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        ck(unsafe { cudaMalloc(&mut ptr, numel * 4) }, "bf16f32 out malloc")?;
        let conv = (|| -> Result<()> {
            ck(
                unsafe { ferrite_bf16_to_f32(staging, ptr as *mut f32, numel as i64, self.stream) },
                "bf16→f32 GPU expand",
            )
        })();
        self.sync()?;
        drop(stage);
        conv?;
        self.weights.lock().unwrap().insert(
            key,
            CachedBuf { keep: placeholder.data.clone(), dev: ptr, len: numel },
        );
        Ok(())
    }

    // ============================================================
    // GPU kernel unit-test probes (doc(hidden) — used by
    // tests/bf16_widen_gpu.rs to validate the mmap preload path's kernels
    // in isolation on real hardware; NOT part of the serving path).
    // ============================================================

    /// Direct ferrite_bf16_to_f32 kernel roundtrip: bf16 bytes → GPU
    /// staging → kernel → f32 readback. Tests the kernel itself (the mmap
    /// path's widen + preload_bf16_to_f32_raw both feed this kernel; the
    /// serve logs showed its output as garbage while memcpy paths verified
    /// correct — this probe isolates exactly that kernel).
    #[doc(hidden)]
    pub fn dbg_kernel_bf16_to_f32(&self, bf16: &[u8]) -> Result<Vec<f32>> {
        assert_eq!(bf16.len() % 2, 0);
        let numel = bf16.len() / 2;
        self.enter();
        let mut staging: *mut std::ffi::c_void = std::ptr::null_mut();
        ck(unsafe { cudaMalloc(&mut staging, bf16.len()) }, "dbg staging malloc")?;
        ck(
            unsafe { cudaMemcpy(staging, bf16.as_ptr() as *const _, bf16.len(), CUDA_MEMCPY_H2D) },
            "dbg staging H2D",
        )?;
        let mut out: *mut std::ffi::c_void = std::ptr::null_mut();
        ck(unsafe { cudaMalloc(&mut out, numel * 4) }, "dbg out malloc")?;
        ck(
            unsafe { ferrite_bf16_to_f32(staging, out as *mut f32, numel as i64, self.stream) },
            "dbg kernel bf16→f32",
        )?;
        self.sync()?;
        let mut v = vec![0f32; numel];
        ck(
            unsafe { cudaMemcpy(v.as_mut_ptr() as *mut _, out, numel * 4, 2 /* D2H */) },
            "dbg out D2H",
        )?;
        unsafe { cudaFree(staging) };
        unsafe { cudaFree(out) };
        Ok(v)
    }

    /// Read back the f32 residency a placeholder's dev_weight resolves to
    /// (covers both the direct preload key and the bf16-widen recovery path).
    #[doc(hidden)]
    pub fn dbg_dev_f32(&self, t: &Tensor) -> Result<Vec<f32>> {
        let dw = self.dev_weight(t)?;
        self.enter();
        let mut v = vec![0f32; dw.len];
        ck(
            unsafe { cudaMemcpy(v.as_mut_ptr() as *mut _, dw.ptr, dw.len * 4, 2 /* D2H */) },
            "dbg dev f32 D2H",
        )?;
        Ok(v)
    }

    /// Read back the bf16 residency a placeholder's dev_weight_bf16 resolves
    /// to, widened to f32 host-side (bf16 u16 << 16) for comparison.
    #[doc(hidden)]
    pub fn dbg_dev_bf16_as_f32(&self, t: &Tensor) -> Result<Vec<f32>> {
        let dw = self.dev_weight_bf16(t)?;
        self.enter();
        let mut raw = vec![0u16; dw.len];
        ck(
            unsafe { cudaMemcpy(raw.as_mut_ptr() as *mut _, dw.ptr, dw.len * 2, 2 /* D2H */) },
            "dbg dev bf16 D2H",
        )?;
        Ok(raw.into_iter().map(|b| f32::from_bits((b as u32) << 16)).collect())
    }

    /// f32 raw → resident f32 (VERBATIM — the mmap bytes ARE the device
    /// layout for f32 checkpoint weights: rare 1-D norms/biases/scales that
    /// the checkpoint stores as F32, not bf16. No conversion kernel — just
    /// cudaMemcpy. Cache key is the f32 dev_weight's (ptr, numel).
    pub fn preload_f32_raw(&self, placeholder: &Tensor, f32_bytes: &[u8]) -> Result<()> {
        let numel = placeholder.numel();
        if numel == 0 {
            return Ok(());
        }
        if f32_bytes.len() != numel * 4 {
            return Err(FerriteError::InvalidArg(format!(
                "preload_f32_raw: {} bytes != numel {numel} * 4",
                f32_bytes.len()
            )));
        }
        self.enter();
        let key = (placeholder.as_slice().as_ptr() as usize, numel);
        {
            let cache = self.weights.lock().unwrap();
            if let Some(cb) = cache.get(&key) {
                if cb.len == numel {
                    return Ok(());
                }
            }
        }
        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        ck(unsafe { cudaMalloc(&mut ptr, numel * 4) }, "f32raw malloc")?;
        ck(
            unsafe { cudaMemcpy(ptr, f32_bytes.as_ptr() as *const _, numel * 4, CUDA_MEMCPY_H2D) },
            "f32raw H2D (mmap → device)",
        )?;
        self.weights.lock().unwrap().insert(
            key,
            CachedBuf { keep: placeholder.data.clone(), dev: ptr, len: numel },
        );
        Ok(())
    }

    /// Free all cached device weights (explicit; the Drop impl does it too).
    pub fn clear_weight_cache(&self) {
        let mut cache = self.weights.lock().unwrap();
        for (_, cb) in cache.drain() {
            unsafe { cudaFree(cb.dev) };
        }
    }

    /// Load `libferrite_kernels.so` (and its cudart dependency) explicitly,
    /// binding the backend to CUDA device `device` (cudaSetDevice). Each rank
    /// of a TP deployment constructs one backend per GPU.
    pub fn with_device(so_path: &str, device: i32) -> Result<Self> {
        let c = CString::new(so_path).map_err(|_| FerriteError::InvalidArg("bad path".into()))?;
        let handle = unsafe { libc_dlopen(c.as_ptr(), 2) };
        if handle.is_null() {
            return Err(FerriteError::InvalidArg(format!(
                "dlopen({so_path}) failed — run kernels/cuda/build.sh first"
            )));
        }
        let err = unsafe { cudaSetDevice(device) };
        if err != 0 {
            return Err(FerriteError::InvalidArg(format!(
                "cudaSetDevice({device}) failed: {err}"
            )));
        }
        let mut b = Self::new();
        b.dev = device;
        Ok(b)
    }

    /// Load `libferrite_kernels.so` (and its cudart dependency) explicitly.
    pub fn with_library(so_path: &str) -> Result<Self> {
        let c = CString::new(so_path).map_err(|_| FerriteError::InvalidArg("bad path".into()))?;
        let handle = unsafe { libc_dlopen(c.as_ptr(), 2) };
        if handle.is_null() {
            return Err(FerriteError::InvalidArg(format!(
                "dlopen({so_path}) failed — run kernels/cuda/build.sh first"
            )));
        }
        Ok(Self::new())
    }

    /// Synchronise this backend's stream (public: tests and callers of the
    /// device-chain APIs need a barrier before wall-clock timings).
    pub fn sync(&self) -> Result<()> {
        ck(unsafe { cudaStreamSynchronize(self.stream) }, "sync")
    }

    /// Device ordinal this backend is bound to (for DevBuf::alloc at the
    /// device-chain call sites).
    pub fn dev(&self) -> i32 {
        self.dev
    }

    /// This backend's CUDA stream (for NCCL comm init / external enqueue).
    pub fn stream_handle(&self) -> CuStream {
        self.stream
    }

    /// PDL (programmatic dependent launch) experiment: times iters× (A→B)
    /// dependency chains, normal launch (mode 0) vs PDL launch
    /// (mode 1: cudaLaunchKernelEx + ProgrammaticStreamSerialization — B's
    /// prologue overlaps A's tail via cudaGridDependencySynchronize).
    /// Returns (time_ms, checksum).
    pub fn pdl_exp_dev(&self, mode: i32, iters: i32) -> Result<(f32, f32)> {
        self.enter();
        let mut t = 0f32;
        let mut c = 0f32;
        ck(unsafe { ferrite_pdl_exp(mode, iters, &mut t, &mut c, self.stream) }, "pdl_exp")?;
        Ok((t, c))
    }

    /// This backend's stream (device-chain ops submit here; the graph
    /// capture/replay uses it too).
    pub fn stream(&self) -> CuStream {
        self.stream
    }

    /// fp8 mma layout probe (W8A8 tensor-core feasibility): drives the raw
    /// mma.sync...e4m3 fragment layout experiment. `a`/`b` hold packed fp8
    /// bytes (as f32 words), `c` receives the f32 mma result.
    pub fn fp8_mma_probe_dev(&self, a: &DevBuf, b: &DevBuf, c: &mut DevBuf) -> Result<()> {
        self.enter();
        ck(unsafe {
            ferrite_fp8_mma_probe(a.as_f32() as *const u8, b.as_f32() as *const u8,
                                  c.as_f32(), self.stream)
        }, "fp8_mma_probe")
    }

    /// Device-resident matmul: x already on device, w uploaded here (the
    /// BufferCache will dedupe repeated weight uploads), result stays on
    /// device. Building block for fused op chains (expert FFN).
    /// Weights are resident in bf16 (dev_weight_bf16).
    /// f32 → bf16 cast (batched AR's half-payload path).
    pub fn cast_f32_to_bf16(&self, src: &DevBuf, dst: &DevBuf, n: usize) -> Result<()> {
        ck(
            unsafe { ferrite_f32_to_bf16(src.as_const_f32(), dst.as_f32() as *mut _, n as i64, self.stream) },
            "cast f32->bf16",
        )
    }

    /// bf16 → f32 cast.
    pub fn cast_bf16_to_f32(&self, src: &DevBuf, dst: &mut DevBuf, n: usize) -> Result<()> {
        ck(
            unsafe { ferrite_bf16_to_f32(src.as_const_f32() as *const _, dst.as_f32(), n as i64, self.stream) },
            "cast bf16->f32",
        )
    }

    /// cuBLAS bf16 batched GEMM for the decode m=16 case:
    /// C[16, N] = A[16, K] * W[N, K]^T. A (fp32 activations) is cast to
    /// bf16 first; the weights are already bf16. cuBLAS's split-K/streaming
    /// hides the tiny-m parallelism that sank the hand-rolled MMA kernel
    /// (grid N/32 = 96 blocks → 5% occupancy → 3x SLOWER than the FMA gemv).
    fn gemm_cublas(&self, x: &DevBuf, w: *const std::ffi::c_void,
                   n: i32, in_f: i32, out_f: i32) -> Result<DevBuf> {
        let do_ = DevBuf::alloc(self.dev, self.stream, (n * out_f) as usize)?;
        let xb = DevBuf::alloc(self.dev, self.stream, ((n * in_f) as usize + 1) / 2)?;
        ck(
            unsafe {
                ferrite_f32_to_bf16(x.as_const_f32(), xb.as_f32() as *mut std::ffi::c_void,
                                    (n * in_f) as i64, self.stream)
            },
            "cast bf16",
        )?;
        // Handle created on first use: the dry-run reaches here BEFORE the
        // capture pass, and cublasCreate is illegal inside a capture.
        let h = {
            let mut g = self.cublas.lock().unwrap();
            if g.is_null() {
                ck(unsafe { cublasCreate_v2(&mut *g) }, "cublasCreate")?;
                ck(unsafe { cublasSetStream_v2(*g, self.stream) }, "cublasSetStream")?;
            }
            *g
        };
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        // column-major view: C'[N,16] = W^T[N,K] * x^T[K,16]
        let st = unsafe {
            cublasGemmEx(
                h, 1, 0, out_f, n, in_f,
                &alpha, w, 14, in_f,
                xb.as_const_f32() as *const std::ffi::c_void, 14, in_f,
                &beta, do_.as_f32() as *mut std::ffi::c_void, 0, out_f,
                0, 99,
            )
        };
        if st != 0 {
            return Err(FerriteError::Config(format!("cublasGemmEx failed: {st}")));
        }
        Ok(do_)
    }

    /// Invalidate the per-x quant cache at a layer boundary (x is rewritten
    /// every layer). Buffers are kept (captured graphs hold the addresses).
    pub fn clear_xq_cache(&self) {
        let mut c = self.xq_cache.lock().unwrap();
        for v in c.values_mut() {
            v.3 = false;
        }
    }

    /// fp32 x -> e4m3 once per (x, in_f); reused by every same-x gemv of the
    /// layer. None => would need to allocate inside a graph capture (illegal).
    fn xq_cached(&self, x_dev: &DevBuf, n: i32, in_f: i32) -> Result<Option<(*const u8, *const f32)>> {
        let xp = x_dev.as_const_f32() as usize;
        let rp;
        {
            let mut c = self.xq_cache.lock().unwrap();
            match c.get_mut(&in_f) {
                Some((q, sc, px, valid)) if *valid && *px == xp => {
                    return Ok(Some((q.as_f32() as *const u8, sc.as_const_f32())));
                }
                Some((q, sc, px, valid)) => {
                    *px = xp;
                    *valid = true;
                    rp = (q.as_f32() as *const u8, sc.as_const_f32());
                }
                None => {
                    if is_capturing() {
                        return Ok(None);
                    }
                    let q = DevBuf::alloc(self.dev, self.stream, (16usize * in_f as usize) / 4 + 1)?;
                    let sc = DevBuf::alloc(self.dev, self.stream, 16usize)?;
                    rp = (q.as_f32() as *const u8, sc.as_const_f32());
                    c.insert(in_f, (q, sc, xp, true));
                }
            }
        }
        ck(unsafe {
            ferrite_quant_e4m3_tokens(x_dev.as_const_f32(), rp.0 as *mut u8,
                                      rp.1 as *mut f32, n, in_f, self.stream)
        }, "quant_e4m3_tokens")?;
        Ok(Some(rp))
    }

    pub fn matmul_dev(&self, x_dev: &DevBuf, w: &Tensor, n: i32, in_f: i32, out_f: i32) -> Result<DevBuf> {
        // Pre-allocate the n<=16 MMA quant buffer OUTSIDE graph capture (the
        // prefill touches every in_f); the capture then only records the quant
        // kernel, never a cudaMalloc (err 900 otherwise).
        if !is_capturing() && !self.xq_cache.lock().unwrap().contains_key(&in_f) {
            let q = DevBuf::alloc(self.dev, self.stream, (16usize * in_f as usize) / 4 + 1)?;
            let sc = DevBuf::alloc(self.dev, self.stream, 16usize)?;
            self.xq_cache.lock().unwrap().insert(in_f, (q, sc, 0, false));
        }
        // fp8 single-store guard: a placeholder Tensor (data.len() < numel)
        // with NO fp8 registration must fail loudly — its bf16 upload would
        // read 4 elements as the full weight (garbage), and the fp8 map is
        // the only real store (register failed or shard seam dropped it).
        // DIRECT-MMAP EXCEPTION: the direct preload populates the weights
        // cache with the placeholder's (ptr, numel) or (ptr, numel<<1|1) for
        // fp8-dequant weights — check BOTH key formats before erroring.
        if w.as_slice().len() < w.numel() && self.fp8_lookup(w).is_none() {
            let ptr = w.as_slice().as_ptr() as usize;
            let numel = w.numel();
            let cached = {
                let cache = self.weights.lock().unwrap();
                cache.contains_key(&(ptr, numel))
                    || cache.contains_key(&(ptr, numel << 1 | 1))
            };
            if !cached {
                return Err(FerriteError::InvalidArg(format!(
                    "matmul_dev: fp8 placeholder weight ({} elems data vs {} numel) has no fp8 registration",
                    w.as_slice().len(), w.numel()
                )));
            }
        }
        // fp8 bypass: registered (ptr,numel) → native-precision F8 GEMV
        // (half the bf16 HBM bytes). Serves ANY n (prefill n>3 included — the
        // gemv loop covers it; slower than a tiled fp8 GEMM but the numeric
        // domain stays uniform across prefill/decode, which matters more than
        // prefill speed here).
        if let Some(f8) = self.fp8_lookup(w) {
            // W8A8 tensor-core path (n=1 decode, GLM-aligned shapes): the
            // mma.sync m16n8k32 e4m3 gemv — activations quantized in-kernel
            // (per-token absmax/448), e4m3 x e4m3 multiplied directly on the
            // tensor core (NO per-element dequant — the W8A16 attempt's cvt
            // overhead offset the HBM savings: 0.96x vs bf16). Misaligned
            // shapes / n>1 fall back to the W8A16 gemv below (kept).
            // v2 split (standalone quant kernel + global-xq mma) measured
            // SLOWER on every shape (0.49x vs v1's 0.60x — the extra launch
            // costs more than the 32x de-duplicated per-block quant work);
            // the real fix is layer-level x quant sharing (quant once in the
            // upstream epilogue, all same-x gemvs consume), not more kernels.
            if n == 1 && in_f % 128 == 0 && out_f % 16 == 0 && f8.scols == (in_f as i32 + 127) / 128 {
                let do_ = DevBuf::alloc(self.dev, self.stream, out_f as usize)?;
                let scratch = self.w8a8_scratch_ptr(in_f as usize)?;
                let r = unsafe {
                    ferrite_gemv_fp8_mma(x_dev.as_const_f32(), f8.w, f8.scale as *const f32,
                                         do_.as_f32(), in_f, out_f, f8.srows, f8.scols, scratch, self.stream)
                };
                if r == 0 {
                    return Ok(do_);
                }
                // cudaErrorNotSupported (unaligned): fall through to W8A16
            }
            // CUTLASS-style fp8 tensor-core MMA (micro-bench: q_a 31->6us,
            // lm_head 353->41us, bit-identical to the fp64 reference).
            if std::env::var("FERRITE_GEMV_MMA_DEBUG").is_ok() {
                static CNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
                let c = CNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if n >= 2 && n <= 16 && c < 200 {
                    let hit = n >= 2 && n <= 16 && (out_f & 7) == 0 && (in_f & 63) == 0
                        && f8.scols == (in_f + 127) / 128;
                    eprintln!("[gemvdbg] n={} in={} out={} scols={}/{} {}",
                              n, in_f, out_f, f8.scols, (in_f + 127) / 128,
                              if hit { "MMA" } else { "SIMT" });
                }
            }
            if n >= 2 && n <= 16 && (out_f & 7) == 0 && (in_f & 63) == 0
                && f8.scols == (in_f + 127) / 128
                && std::env::var("FERRITE_GEMV_MMA").map(|v| v != "0").unwrap_or(true) {
                if let Ok(Some((xq, xs))) = self.xq_cached(x_dev, n, in_f) {
                    let do_ = DevBuf::alloc(self.dev, self.stream, n as usize * out_f as usize)?;
                    let r = unsafe {
                        ferrite_gemv_fp8_mma_b16(xq, xs, f8.w as *const u8, f8.scale as *const f32,
                                                 do_.as_f32(), n, in_f, out_f, f8.scols, self.stream)
                    };
                    if r != 0 {
                        eprintln!("[opcheck] gemv_fp8_mma_b16 returned err {r}");
                    }
                    if r == 0 {
                        return Ok(do_);
                    }
                }
            }
            let do_ = DevBuf::alloc(self.dev, self.stream, n as usize * out_f as usize)?;
            ck(unsafe {
                ferrite_gemv_fp8_v2(x_dev.as_const_f32(), f8.w as *const _, f8.scale as *const f32, std::ptr::null(),
                                    do_.as_f32(), in_f, out_f, n, f8.srows, f8.scols, self.stream)
            }, "gemv_fp8")?;
            return Ok(do_);
        }
        let dw = self.dev_weight_bf16(w)?;
        let do_ = DevBuf::alloc(self.dev, self.stream, n as usize * out_f as usize)?;
        let dbias: *const f32 = std::ptr::null();
        if n == 16 && dbias.is_null() {
            // cuBLAS bf16 batched GEMM (split-K/streaming): the FMA
            // gemv_bf16_nt is compute-bound at n=16 (measured 2.5x decay
            // vs n=1) — this is the SGLang/cutlass route.
            if let Ok(o) = self.gemm_cublas(x_dev, dw.ptr as *const _, n, in_f, out_f) {
                return Ok(o);
            }
        }
        if n == 1 || (n <= 16 && self.small_n_rows.load(std::sync::atomic::Ordering::Relaxed)) {
            // Decode GEMV v2: uint4 vectorized + K-split WPR — 2.09x over v1
            // (bench gemv_v2_bench: 3.11→6.80TB/s lm_head, 2.20→3.91 o_proj,
            // all shapes 1.45-2.18x). BATCHED: ONE launch covers n rows
            // (grid = n*out_f) — was n single-row launches, the MTP verify
            // chain's 19880 small-graph-node cause. Per-row accumulation order
            // (warp shuffle + WPR root) is unchanged, so the greedy argmax is
            // bit-identical; only the launch/graph-node count drops n×.
            // small_n_rows (the MTP verify chain n<=3, the BATCHED decode
            // n<=16): the tiled GEMM at tiny n wastes the tile (measured n=4
            // batched: 105ms/step vs n=1's 16ms — the 128-row tile computes
            // the full K per tile regardless of n). The GEMV batched rides
            // the L2 for the n rows' same-weight reads (verify n=3: +35% vs
            // n=1's 16ms). Prefill keeps the GEMM (its row-batched
            // accumulation order sets the first greedy token).
            // v4 (n>1): the TALL-SKINNY nt kernel FIRST — each warp-group
            // reads ONE weight row slice ONCE and dots it against ALL n
            // activation rows (the true batched-GEMM weight streaming:
            // weights 1×, activations n× from L2, per-token accumulation
            // IDENTICAL to v2 — no greedy flips). Measured v2-batched at
            // n=4: 33.5ms/step (weights ~2× effective, L2 partial reuse);
            // v4's target: the n=1 HBM floor (~16-18ms) at n=4 → ~2x.
            // Unsupported shapes (in_f%8, n not in the template set) return
            // NotSupported → the v2 batched below (per-row, L2 luck).
            if n == 16 {
                // bf16 MMA (m16n8k16): the tensor core hides the 16-token
                // arithmetic that makes the FMA nt kernel compute-bound at
                // n=16 (measured 2.5x decay vs n=1). Weights stream once.
                let r = unsafe {
                    ferrite_gemm_bf16_mma(x_dev.as_const_f32(), dw.ptr as *const _,
                                          dbias, do_.as_f32(), n, in_f, out_f, self.stream)
                };
                if r == 0 {
                    return Ok(do_);
                }
            }
            if n > 1 {
                let r = unsafe {
                    ferrite_gemv_bf16_nt(x_dev.as_const_f32(), dw.ptr as *const _,
                                         dbias, do_.as_f32(), in_f, out_f, n, self.stream)
                };
                if r == 0 {
                    return Ok(do_);
                }
            }
            ck(unsafe {
                ferrite_gemv_bf16_v2(x_dev.as_const_f32(), dw.ptr as *const _,
                                      dbias, do_.as_f32(), in_f, out_f, n, self.stream)
            }, "gemv_batch")?;
        } else {
            ck(unsafe {
                ferrite_matmul_bf16(x_dev.as_const_f32(), dw.ptr as *const _,
                                     dbias, do_.as_f32(), n, in_f, out_f, self.stream)
            }, "matmul_dev")?;
        }
        Ok(do_)
    }

    /// GEMV v2 (vectorized uint4 + K-split): the decode weight-streaming
    /// upgrade of matmul_dev's n==1 path — uint4 (8 bf16) loads + WPR
    /// warps/row K-split to cover HBM latency on medium matrices.
    /// Benchmarked 2.2-3.1 TB/s (v1) → target 6+ TB/s. A/B via gemv_v2_bench.
    pub fn gemv_v2_dev(&self, x_dev: &DevBuf, w: &Tensor, n: i32, in_f: i32, out_f: i32) -> Result<DevBuf> {
        let dw = self.dev_weight_bf16(w)?;
        let do_ = DevBuf::alloc(self.dev, self.stream, n as usize * out_f as usize)?;
        ck(unsafe {
            ferrite_gemv_bf16_v2(x_dev.as_const_f32(), dw.ptr as *const _,
                                 std::ptr::null(), do_.as_f32(), in_f, out_f, n, self.stream)
        }, "gemv_v2_dev")?;
        Ok(do_)
    }

    /// 3-in-1 GEMV (decode n==1): the gdn layer's b_raw [h,in] + f_a [dk,in]
    /// + g_a [dk,in] all read the SAME hidden x — one kernel maps the three
    /// weight matrices onto one row space (WPR=4 K-split, uint4 body) instead
    /// of three separate gemv v2 launches. -2 kernel boundaries per gdn
    /// layer × 34 layers.
    pub fn gemv_tri_dev(&self, x_dev: &DevBuf, w1: &Tensor, w2: &Tensor, w3: &Tensor,
                        in_f: i32, o1: i32, o2: i32, o3: i32) -> Result<(DevBuf, DevBuf, DevBuf)> {
        // fp8 domain-uniformity guard: if ANY of the three is registered fp8,
        // serve all three through matmul_dev (fp8 GEMV) — the bf16 tri-kernel
        // and the fp8 single GEMV produce different numerics, and the MTP
        // draft (n==1, tri) vs verify (n==3, matmul) chains MUST stay in one
        // domain or the accept decision flips (same failure class as the
        // NCCL graph-order d2 flips). Cost: 2 extra launches × 34 layers on
        // the draft chain (~0.3ms/step) until the fp8 tri kernel lands.
        if self.fp8_lookup(w1).is_some() {
            let y1 = self.matmul_dev(x_dev, w1, 1, in_f, o1)?;
            let y2 = self.matmul_dev(x_dev, w2, 1, in_f, o2)?;
            let y3 = self.matmul_dev(x_dev, w3, 1, in_f, o3)?;
            return Ok((y1, y2, y3));
        }
        let dw1 = self.dev_weight_bf16(w1)?;
        let dw2 = self.dev_weight_bf16(w2)?;
        let dw3 = self.dev_weight_bf16(w3)?;
        let y1 = DevBuf::alloc(self.dev, self.stream, o1 as usize)?;
        let y2 = DevBuf::alloc(self.dev, self.stream, o2 as usize)?;
        let y3 = DevBuf::alloc(self.dev, self.stream, o3 as usize)?;
        ck(unsafe {
            ferrite_gemv_tri(x_dev.as_const_f32(), dw1.ptr as *const _, dw2.ptr as *const _,
                             dw3.ptr as *const _, y1.as_f32(), y2.as_f32(), y3.as_f32(),
                             in_f, o1, o2, o3, self.stream)
        }, "gemv_tri")?;
        Ok((y1, y2, y3))
    }

    /// sparse_attn v2 (256-thread block, float4 dots, smem idx/bitmap dedup):
    /// the dsa attention core — v1 ran block=32 (ONE warp) over topk≈8K
    /// slots with serial scalar dots + O(topk²) global idx rereads. Parity
    /// tested against the CPU reference (dedup first-wins, padding, softmax)
    /// in gpu_smoke::sparse_attn_v2_parity.
    pub fn sparse_mla_attn_v2(&self, q: &Tensor, k_nope: &Tensor, v: &Tensor, idx: &Tensor, out: &mut Tensor) -> Result<()> {
        self.enter();
        let n = q.shape.0[0] as i32;
        let t = k_nope.shape.0[0] as i32;
        let h = q.shape.0[1] as i32;
        let d = *q.shape.0.last().unwrap() as i32;
        let dv = *v.shape.0.last().unwrap() as i32;
        let topk = *idx.shape.0.last().unwrap() as i32;
        let dq = DevBuf::alloc(self.dev, self.stream, q.numel())?; dq.upload(q.as_slice())?;
        let dk = DevBuf::alloc(self.dev, self.stream, k_nope.numel())?; dk.upload(k_nope.as_slice())?;
        let dv_ = DevBuf::alloc(self.dev, self.stream, v.numel())?; dv_.upload(v.as_slice())?;
        let di = DevBuf::alloc(self.dev, self.stream, idx.numel())?; di.upload(idx.as_slice())?;
        let do_ = DevBuf::alloc(self.dev, self.stream, out.numel())?;
        let t_ptr = &t as *const i32;
        // split-K (SGLang MLA decode's num_splits): grid = n*h*splits blocks.
        let splits = (256 / (n * h).max(1)).clamp(1, 32);
        let scratch = DevBuf::alloc(self.dev, self.stream,
            (n as usize) * (h as usize) * (splits as usize) * (2 + dv as usize))?;
        ck(unsafe { ferrite_sparse_attn_v2(dq.as_const_f32(), dk.as_const_f32(), dv_.as_const_f32(), di.as_const_f32(), do_.as_f32(), scratch.as_f32(), n, t_ptr, h, d, dv, topk, splits, self.stream) }, "sparse_attn_v2")?;
        let ov = Arc::get_mut(&mut out.data).expect("unique out");
        do_.download(ov)?;
        Ok(())
    }

    /// gdn_step v2 parity hook (kernel-level): runs the v2 recurrent core
    /// (state staged in smem, padded stride) and returns out + new state.
    /// Golden is the sequential CPU recurrence in the gpu_smoke test.
    pub fn gdn_step_v2_dev(&self, q: &Tensor, k: &Tensor, v: &Tensor, beta: &Tensor,
                           gate: &Tensor, a_log: &Tensor, state_in: &Tensor, n: usize,
                           h: usize, dk: usize, dv: usize,
                           out: &mut Tensor, state_out: &mut Tensor) -> Result<()> {
        self.enter();
        let nn = n as i32;
        let dq = DevBuf::alloc(self.dev, self.stream, q.numel())?; dq.upload(q.as_slice())?;
        let dk_ = DevBuf::alloc(self.dev, self.stream, k.numel())?; dk_.upload(k.as_slice())?;
        let dv_ = DevBuf::alloc(self.dev, self.stream, v.numel())?; dv_.upload(v.as_slice())?;
        let db = DevBuf::alloc(self.dev, self.stream, beta.numel())?; db.upload(beta.as_slice())?;
        let dg = DevBuf::alloc(self.dev, self.stream, gate.numel())?; dg.upload(gate.as_slice())?;
        let dal = DevBuf::alloc(self.dev, self.stream, a_log.numel())?; dal.upload(a_log.as_slice())?;
        let dst = DevBuf::alloc(self.dev, self.stream, state_in.numel())?; dst.upload(state_in.as_slice())?;
        let do_ = DevBuf::alloc(self.dev, self.stream, out.numel())?;
        ck(unsafe { ferrite_gdn_chunk_v2(dq.as_const_f32(), dk_.as_const_f32(), dv_.as_const_f32(),
                                         db.as_const_f32(), dg.as_const_f32(), dal.as_const_f32(),
                                         dst.as_f32(), do_.as_f32(), nn, h as i32, dk as i32, dv as i32,
                                         self.stream) }, "gdn_chunk_v2")?;
        let ov = Arc::get_mut(&mut out.data).expect("unique out");
        do_.download(ov)?;
        let sv = Arc::get_mut(&mut state_out.data).expect("unique state_out");
        dst.download(sv)?;
        Ok(())
    }

    /// conv1d+gdn_prep FUSED parity hook (decode n==1 hot path). `cs` is the
    /// sliding-window conv state — updated IN PLACE (downloaded back mutated).
    pub fn conv_prep_fused_dev(&self, x: &Tensor, cw: &Tensor, cs: &mut Tensor,
                               b_raw: &Tensor, fb: &Tensor, dt_bias: &Tensor, a_log: &Tensor,
                               h: usize, dk: usize, lb: f32,
                               q: &mut Tensor, k: &mut Tensor, v: &mut Tensor,
                               beta: &mut Tensor, gate: &mut Tensor) -> Result<()> {
        self.enter();
        let proj = h * dk;
        let dx = DevBuf::alloc(self.dev, self.stream, x.numel())?; dx.upload(x.as_slice())?;
        let dwc = DevBuf::alloc(self.dev, self.stream, cw.numel())?; dwc.upload(cw.as_slice())?;
        let dcs = DevBuf::alloc(self.dev, self.stream, cs.numel())?; dcs.upload(cs.as_slice())?;
        let db = DevBuf::alloc(self.dev, self.stream, b_raw.numel())?; db.upload(b_raw.as_slice())?;
        let dfb = DevBuf::alloc(self.dev, self.stream, fb.numel())?; dfb.upload(fb.as_slice())?;
        let ddt = DevBuf::alloc(self.dev, self.stream, dt_bias.numel())?; ddt.upload(dt_bias.as_slice())?;
        let dal = DevBuf::alloc(self.dev, self.stream, a_log.numel())?; dal.upload(a_log.as_slice())?;
        let dq = DevBuf::alloc(self.dev, self.stream, q.numel())?;
        let dk_ = DevBuf::alloc(self.dev, self.stream, k.numel())?;
        let dv_ = DevBuf::alloc(self.dev, self.stream, v.numel())?;
        let dbt = DevBuf::alloc(self.dev, self.stream, beta.numel())?;
        let dg = DevBuf::alloc(self.dev, self.stream, gate.numel())?;
        ck(unsafe { ferrite_conv_prep_fused(dx.as_const_f32(), dwc.as_const_f32(), dcs.as_f32(),
                                             db.as_const_f32(), dfb.as_const_f32(), ddt.as_const_f32(),
                                             dal.as_const_f32(), dq.as_f32(), dk_.as_f32(), dv_.as_f32(),
                                             dbt.as_f32(), dg.as_f32(),
                                             h as i32, dk as i32, lb, self.stream) }, "conv_prep_fused")?;
        for (o, d) in [(q, dq), (k, dk_), (v, dv_), (beta, dbt), (gate, dg), (cs, dcs)] {
            let ov = Arc::get_mut(&mut o.data).expect("unique out");
            d.download(ov)?;
        }
        Ok(())
    }

    /// Fused SwiGLU on device: reads two independent matmul outputs.
    pub fn swiglu2_dev(&self, gate: &DevBuf, up: &DevBuf, n: i32, inter: i32, limit: f32) -> Result<DevBuf> {
        let out = DevBuf::alloc(self.dev, self.stream, n as usize * inter as usize)?;
        ck(unsafe {
            ferrite_swiglu2(gate.as_const_f32(), up.as_const_f32(), out.as_f32(), n, inter, limit, self.stream)
        }, "swiglu2")?;
        Ok(out)
    }

    fn run_matmul(&self, x: &Tensor, w: &Tensor, bias: Option<&Tensor>, out: &mut Tensor) -> Result<()> {
        let n = x.shape.0[0] as i32;
        let in_f = x.shape.0[1] as i32;
        let out_f = w.shape.0[0] as i32;
        let dx = DevBuf::alloc(self.dev, self.stream, x.numel())?; dx.upload(x.as_slice())?;
        // FAST PATHS FIRST: matmul_dev routes n==16 to cuBLAS (nvjet), fp8
        // weights to the fp8 GEMV/MMA and has the bf16 MMA GEMM. This path
        // used to go straight to the 32x32 FMA tiled kernel (measured
        // 176us/call, ~19 calls/step in the batched decode). No regression
        // for large n: matmul_dev's non-16/small-n branch is the same tiled
        // kernel.
        if bias.is_none() {
            if let Ok(o) = self.matmul_dev(&dx, w, n, in_f, out_f) {
                let ov = Arc::get_mut(&mut out.data).expect("unique out");
                o.download(ov)?;
                return Ok(());
            }
        }
        // weights resident in bf16 (half the f32 footprint — the TP4 shard
        // does not fit a 275GB B300 in f32); kernel converts to f32 in registers
        let dw = self.dev_weight_bf16(w)?;
        let db = match bias {
            Some(b) => Some(self.dev_weight(b)?),
            None => None,
        };
        let do_ = DevBuf::alloc(self.dev, self.stream, out.numel())?;
        ck(unsafe {
            ferrite_matmul_bf16(dx.as_const_f32(), dw.ptr as *const _,
                                 db.as_ref().map_or(std::ptr::null(), |b| b.as_const_f32()),
                                 do_.as_f32(), n, in_f, out_f, self.stream)
        }, "matmul")?;
        let ov = Arc::get_mut(&mut out.data).expect("unique out");
        do_.download(ov)?;
        Ok(())
    }
}

extern "C" {
    #[link_name = "dlopen"]
    fn libc_dlopen(filename: *const std::os::raw::c_char, flags: i32) -> *mut std::ffi::c_void;
    #[link_name = "dlsym"]
    fn libc_dlsym(handle: *mut std::ffi::c_void, symbol: *const std::os::raw::c_char) -> *mut std::ffi::c_void;
}

// ============================================================
// CUDA graph capture — driver API via dlopen/dlsym (no link-time CUDA
// dependency; resolves libcuda.so.1 at first use).
// Contract mapping onto GraphCapable:
//   begin_capture → cuStreamBeginCapture(THREAD_LOCAL)
//   end_capture   → cuStreamEndCapture + cuGraphInstantiate (exec kept)
//   begin_verify  → cuGraphLaunch (replay into the SAME device buffers)
//   end_verify    → stream sync
// Precondition: replay is only correct when kernel argument pointers are
// stable — weights are (BufferCache device-resident); activations must be
// arena-allocated (engine-level, wired on the B300 validation harness).
// ============================================================
type FnStreamBeginCapture = unsafe extern "C" fn(*mut std::ffi::c_void, i32) -> i32;
type FnStreamEndCapture = unsafe extern "C" fn(*mut std::ffi::c_void, *mut *mut std::ffi::c_void) -> i32;
type FnGraphInstantiate = unsafe extern "C" fn(*mut *mut std::ffi::c_void, *mut std::ffi::c_void, u64) -> i32;
type FnGraphLaunch = unsafe extern "C" fn(*mut std::ffi::c_void, *mut std::ffi::c_void) -> i32;
type FnGraphDestroy = unsafe extern "C" fn(*mut std::ffi::c_void) -> i32;

#[allow(non_snake_case)]
struct DriverApi {
    cuStreamBeginCapture: FnStreamBeginCapture,
    cuStreamEndCapture: FnStreamEndCapture,
    cuGraphInstantiate: FnGraphInstantiate,
    cuGraphLaunch: FnGraphLaunch,
    cuGraphDestroy: FnGraphDestroy,
}

impl DriverApi {
    fn get() -> Option<&'static DriverApi> {
        static API: std::sync::OnceLock<Option<DriverApi>> = std::sync::OnceLock::new();
        API.get_or_init(|| {
            let name = c"libcuda.so.1";
            let h = unsafe { libc_dlopen(name.as_ptr(), 2) };
            if h.is_null() {
                // try without the .1
                let name2 = c"libcuda.so";
                let h2 = unsafe { libc_dlopen(name2.as_ptr(), 2) };
                if h2.is_null() {
                    return None;
                }
                return DriverApi::from_handle(h2);
            }
            DriverApi::from_handle(h)
        })
        .as_ref()
    }

    fn from_handle(h: *mut std::ffi::c_void) -> Option<DriverApi> {
        let sym = |name: &std::ffi::CStr| unsafe {
            let p = libc_dlsym(h, name.as_ptr());
            if p.is_null() { None } else { Some(p) }
        };
        let s_bc: &std::ffi::CStr = c"cuStreamBeginCapture";
        let s_ec: &std::ffi::CStr = c"cuStreamEndCapture";
        let s_gi: &std::ffi::CStr = c"cuGraphInstantiate";
        let s_gl: &std::ffi::CStr = c"cuGraphLaunch";
        let s_gd: &std::ffi::CStr = c"cuGraphDestroy";
        let bc = sym(s_bc)?;
        let ec = sym(s_ec)?;
        let gi = sym(s_gi)?;
        let gl = sym(s_gl)?;
        let gd = sym(s_gd)?;
        Some(DriverApi {
            cuStreamBeginCapture: unsafe { std::mem::transmute::<*mut std::ffi::c_void, FnStreamBeginCapture>(bc) },
            cuStreamEndCapture: unsafe { std::mem::transmute::<*mut std::ffi::c_void, FnStreamEndCapture>(ec) },
            cuGraphInstantiate: unsafe { std::mem::transmute::<*mut std::ffi::c_void, FnGraphInstantiate>(gi) },
            cuGraphLaunch: unsafe { std::mem::transmute::<*mut std::ffi::c_void, FnGraphLaunch>(gl) },
            cuGraphDestroy: unsafe { std::mem::transmute::<*mut std::ffi::c_void, FnGraphDestroy>(gd) },
        })
    }
}

/// Graph capture state for the CUDA backend.
#[derive(Default)]
struct GraphState {
    capturing: bool,
    graph_exec: Option<*mut std::ffi::c_void>,
}

impl CudaBackend {
    /// Named-graph replay: launch a previously captured graph on this
    /// backend's stream. Returns false if the name has no captured graph
    /// yet (caller should run+capture instead). One graph per (layer,
    /// segment, rank) — the per-op launch gaps (~30μs × ~19 ops/layer) are
    /// the decode bottleneck after the device chains.
    pub fn graph_replay(&self, name: &str) -> bool {
        let exec = self.graph_execs.lock().unwrap().get(name).copied();
        match exec {
            Some(exec) => {
                let r = unsafe { ferrite_graph_launch(exec as *mut std::ffi::c_void, self.stream) };
                r == 0
            }
            None => false,
        }
    }

    /// Begin stream capture (THREAD_LOCAL mode). Ops enqueued until
    /// graph_capture_end are RECORDED, not executed. The pool and weight
    /// caches must be warm (prefill does this) — cudaMalloc during capture
    /// is illegal.
    pub fn graph_capture_begin(&self) {
        // RUNTIME API wrappers (the driver-API dlopen path SIGSEGV'd inside
        // cuGraphInstantiate on worker-thread captures)
        let r = unsafe { ferrite_graph_begin(self.stream) };
        if r != 0 {
            panic!("cudaStreamBeginCapture failed: {r}");
        }
        set_capturing(true);
    }

    /// End capture, instantiate, store under `name`. The recorded ops did
    /// NOT execute — replay immediately if this pass's results are needed.
    pub fn graph_capture_end(&self, name: &str) {
        set_capturing(false);
        let mut graph: *mut std::ffi::c_void = std::ptr::null_mut();
        let r = unsafe { ferrite_graph_end(self.stream, &mut graph) };
        if r != 0 {
            panic!("cudaStreamEndCapture failed: {r}");
        }
        if graph.is_null() {
            panic!("cudaStreamEndCapture returned a NULL graph (capture invalidated?)");
        }
        let mut exec: *mut std::ffi::c_void = std::ptr::null_mut();
        let r = unsafe { ferrite_graph_instantiate(&mut exec, graph) };
        if r != 0 {
            panic!("cudaGraphInstantiate failed: {r}");
        }
        self.graph_execs.lock().unwrap().insert(name.to_string(), exec as usize);
    }
}

impl crate::graph::GraphCapable for CudaBackend {
    fn begin_capture(&self) {
        let api = DriverApi::get().expect("libcuda not loadable (no GPU present?)");
        let r = unsafe { (api.cuStreamBeginCapture)(self.stream, 1) }; // 1 = THREAD_LOCAL
        if r != 0 {
            panic!("cuStreamBeginCapture failed: {r}");
        }
        set_capturing(true);
        let mut g = self.graph.lock().unwrap();
        g.capturing = true;
    }

    fn end_capture(&self) -> crate::graph::OpTrace {
        set_capturing(false);
        let api = DriverApi::get().expect("libcuda not loadable");
        let mut graph: *mut std::ffi::c_void = std::ptr::null_mut();
        let r = unsafe { (api.cuStreamEndCapture)(self.stream, &mut graph) };
        if r != 0 {
            panic!("cuStreamEndCapture failed: {r}");
        }
        let mut exec: *mut std::ffi::c_void = std::ptr::null_mut();
        let r = unsafe { (api.cuGraphInstantiate)(&mut exec, graph, 0) };
        if r != 0 {
            unsafe { (api.cuGraphDestroy)(graph) };
            panic!("cuGraphInstantiate failed: {r}");
        }
        unsafe { (api.cuGraphDestroy)(graph) }; // exec is independent
        let mut g = self.graph.lock().unwrap();
        g.capturing = false;
        g.graph_exec = Some(exec);
        // The CUDA graph handle IS the trace (hardware-recorded op sequence);
        // the CPU-side OpTrace recorder is the CPU backend's equivalent.
        crate::graph::OpTrace::default()
    }

    fn begin_verify(&self, _trace: &crate::graph::OpTrace) {
        let api = DriverApi::get().expect("libcuda not loadable");
        let exec = self.graph.lock().unwrap().graph_exec;
        if let Some(exec) = exec {
            let r = unsafe { (api.cuGraphLaunch)(exec, self.stream) };
            if r != 0 {
                panic!("cuGraphLaunch failed: {r}");
            }
        }
    }

    fn end_verify(&self) -> bool {
        self.sync().is_ok()
    }
}

impl crate::KernelBackend for CudaBackend {
    #[cfg(feature = "cuda")]
    fn as_cuda(&self) -> Option<&CudaBackend> {
        Some(self)
    }

    fn matmul(&self, x: &Tensor, w: &Tensor, bias: Option<&Tensor>, out: &mut Tensor) -> Result<()> {
        self.enter();
        self.run_matmul(x, w, bias, out)
    }

    fn rmsnorm(&self, x: &Tensor, w: &Tensor, eps: f32, out: &mut Tensor) -> Result<()> {
        self.enter();
        let n = (x.numel() / w.numel()) as i32;
        let dim = w.numel() as i32;
        let dx = DevBuf::alloc(self.dev, self.stream, x.numel())?; dx.upload(x.as_slice())?;
        let dw = self.dev_weight(w)?;
        let do_ = DevBuf::alloc(self.dev, self.stream, out.numel())?;
        ck(unsafe { ferrite_rmsnorm(dx.as_const_f32(), dw.as_const_f32(), do_.as_f32(), n, dim, eps, self.stream) }, "rmsnorm")?;
        let ov = Arc::get_mut(&mut out.data).expect("unique out");
        do_.download(ov)?;
        Ok(())
    }

    fn gated_rmsnorm(&self, x: &Tensor, gate: &Tensor, w: &Tensor, eps: f32, out: &mut Tensor) -> Result<()> {
        self.enter();
        let n = (x.numel() / w.numel()) as i32;
        let dim = w.numel() as i32;
        let dx = DevBuf::alloc(self.dev, self.stream, x.numel())?; dx.upload(x.as_slice())?;
        let dg = DevBuf::alloc(self.dev, self.stream, gate.numel())?; dg.upload(gate.as_slice())?;
        let dw = self.dev_weight(w)?;
        let do_ = DevBuf::alloc(self.dev, self.stream, out.numel())?;
        ck(unsafe { ferrite_gated_rmsnorm(dx.as_const_f32(), dg.as_const_f32(), dw.as_const_f32(), do_.as_f32(), n, dim, eps, self.stream) }, "gated_rmsnorm")?;
        let ov = Arc::get_mut(&mut out.data).expect("unique out");
        do_.download(ov)?;
        Ok(())
    }

    fn swiglu_limited(&self, gate_up: &Tensor, limit: f32, out: &mut Tensor) -> Result<()> {
        self.enter();
        let n = out.shape.0[0] as i32;
        let inter = out.shape.0[1] as i32;
        let dgu = DevBuf::alloc(self.dev, self.stream, gate_up.numel())?; dgu.upload(gate_up.as_slice())?;
        let do_ = DevBuf::alloc(self.dev, self.stream, out.numel())?;
        ck(unsafe { ferrite_swiglu(dgu.as_const_f32(), do_.as_f32(), n, inter, limit, self.stream) }, "swiglu")?;
        let ov = Arc::get_mut(&mut out.data).expect("unique out");
        do_.download(ov)?;
        Ok(())
    }

    fn causal_conv1d(&self, x: &Tensor, w: &Tensor, state_in: &Tensor, out: &mut Tensor, state_out: &mut Tensor) -> Result<()> {
        self.enter();
        let n = x.shape.0[0] as i32;
        let ch = x.shape.0[1] as i32;
        let conv = w.shape.0[1] as i32;
        let dx = DevBuf::alloc(self.dev, self.stream, x.numel())?; dx.upload(x.as_slice())?;
        let dw = self.dev_weight(w)?;
        let dsi = DevBuf::alloc(self.dev, self.stream, state_in.numel())?; dsi.upload(state_in.as_slice())?;
        let do_ = DevBuf::alloc(self.dev, self.stream, out.numel())?;
        let dso = DevBuf::alloc(self.dev, self.stream, state_out.numel())?;
        ck(unsafe { ferrite_causal_conv1d(dx.as_const_f32(), dw.as_const_f32(), dsi.as_const_f32(), do_.as_f32(), dso.as_f32(), std::ptr::null_mut(), n, ch, conv, self.stream) }, "conv1d")?;
        let ov = Arc::get_mut(&mut out.data).expect("unique out");
        do_.download(ov)?;
        let sv = Arc::get_mut(&mut state_out.data).expect("unique state");
        dso.download(sv)?;
        Ok(())
    }

    fn gated_deltanet_step(&self, q: &Tensor, k: &Tensor, v: &Tensor, beta: &Tensor, gate: &Tensor, a_log: &Tensor, state_in: &Tensor, out: &mut Tensor, state_out: &mut Tensor) -> Result<()> {
        self.enter();
        self.gated_deltanet_chunk(q, k, v, beta, gate, a_log, state_in, out, state_out)
    }

    fn gated_deltanet_chunk(&self, q: &Tensor, k: &Tensor, v: &Tensor, beta: &Tensor, gate: &Tensor, a_log: &Tensor, state_in: &Tensor, out: &mut Tensor, state_out: &mut Tensor) -> Result<()> {
        self.enter();
        let n = q.shape.0[0] as i32;
        let h = a_log.numel() as i32;
        let dk = *q.shape.0.last().unwrap() as i32;
        let dv = *v.shape.0.last().unwrap() as i32;
        let dq = DevBuf::alloc(self.dev, self.stream, q.numel())?; dq.upload(q.as_slice())?;
        let dk_ = DevBuf::alloc(self.dev, self.stream, k.numel())?; dk_.upload(k.as_slice())?;
        let dv_ = DevBuf::alloc(self.dev, self.stream, v.numel())?; dv_.upload(v.as_slice())?;
        let db = DevBuf::alloc(self.dev, self.stream, beta.numel())?; db.upload(beta.as_slice())?;
        let dg = DevBuf::alloc(self.dev, self.stream, gate.numel())?; dg.upload(gate.as_slice())?;
        let dal = self.dev_weight(a_log)?;
        // WYF chunkwise: state ping-pong buffers (chunk chain), tail chunk
        // falls back to the exact per-token kernel inside the launcher.
        let dst_a = DevBuf::alloc(self.dev, self.stream, state_in.numel())?; dst_a.upload(state_in.as_slice())?;
        let do_ = DevBuf::alloc(self.dev, self.stream, out.numel())?;
        ck(unsafe { ferrite_gdn_chunk_v2(dq.as_const_f32(), dk_.as_const_f32(), dv_.as_const_f32(), db.as_const_f32(), dg.as_const_f32(), dal.as_const_f32(), dst_a.as_f32(), do_.as_f32(), n, h, dk, dv, self.stream) }, "gdn_chunk_v2")?;
        let ov = Arc::get_mut(&mut out.data).expect("unique out");
        do_.download(ov)?;
        let sv = Arc::get_mut(&mut state_out.data).expect("unique state");
        dst_a.download(sv)?;
        Ok(())
    }

    fn indexer_topk(
        &self,
        q_idx: &Tensor,
        k_idx: &Tensor,
        w: &Tensor,
        topk: usize,
        ctx0: usize,
        idx: &mut Tensor,
    ) -> Result<()> {
        self.enter();
        let n = q_idx.shape.0[0] as i32;
        let hd = q_idx.shape.0[1] as i32;
        let d = k_idx.shape.0[1] as i32;
        let t = k_idx.shape.0[0] as i32;
        let h = w.shape.0[1] as i32;
        if hd != h * d {
            return Err(FerriteError::InvalidArg(
                "indexer_topk: q_idx [n,H*D] vs w [n,H] head mismatch".into(),
            ));
        }
        let dq = DevBuf::alloc(self.dev, self.stream, q_idx.numel())?; dq.upload(q_idx.as_slice())?;
        let dk = DevBuf::alloc(self.dev, self.stream, k_idx.numel())?; dk.upload(k_idx.as_slice())?;
        let dw = DevBuf::alloc(self.dev, self.stream, w.numel())?; dw.upload(w.as_slice())?;
        let di = DevBuf::alloc(self.dev, self.stream, idx.numel())?;
        let total_i32 = (t * 4) as i32; // total = npools * kpool (approximate: use t*4 as total for the pinned path)
        let total_ptr = &total_i32 as *const i32;
        let kpool_const = 4i32;
        ck(unsafe { ferrite_indexer_topk(dq.as_const_f32(), dk.as_const_f32(), dw.as_const_f32(), di.as_f32(), n, h, d, topk as i32, total_ptr, kpool_const, n, self.stream) }, "indexer_topk")?;
        let ov = Arc::get_mut(&mut idx.data).expect("unique idx");
        di.download(ov)?;
        Ok(())
    }

    fn sparse_mla_attn(&self, q: &Tensor, k_nope: &Tensor, v: &Tensor, idx: &Tensor, out: &mut Tensor) -> Result<()> {
        self.enter();
        let n = q.shape.0[0] as i32;
        let t = k_nope.shape.0[0] as i32;
        let h = q.shape.0[1] as i32;
        let d = *q.shape.0.last().unwrap() as i32;
        let dv = *v.shape.0.last().unwrap() as i32;
        let topk = *idx.shape.0.last().unwrap() as i32;
        let dq = DevBuf::alloc(self.dev, self.stream, q.numel())?; dq.upload(q.as_slice())?;
        let dk = DevBuf::alloc(self.dev, self.stream, k_nope.numel())?; dk.upload(k_nope.as_slice())?;
        let dv_ = DevBuf::alloc(self.dev, self.stream, v.numel())?; dv_.upload(v.as_slice())?;
        let di = DevBuf::alloc(self.dev, self.stream, idx.numel())?; di.upload(idx.as_slice())?;
        let do_ = DevBuf::alloc(self.dev, self.stream, out.numel())?;
        let t_ptr = &t as *const i32;
        ck(unsafe { ferrite_sparse_attn(dq.as_const_f32(), dk.as_const_f32(), dv_.as_const_f32(), di.as_const_f32(), do_.as_f32(), n, t_ptr, h, d, dv, topk, self.stream) }, "sparse_attn")?;
        let ov = Arc::get_mut(&mut out.data).expect("unique out");
        do_.download(ov)?;
        Ok(())
    }

    fn moe_route(&self, logits: &Tensor, bias: &Tensor, topk: usize, routed_scaling: f32, probs: &mut Tensor, ids: &mut Tensor) -> Result<()> {
        self.enter();
        let n = logits.shape.0[0] as i32;
        let e = logits.shape.0[1] as i32;
        let dl = DevBuf::alloc(self.dev, self.stream, logits.numel())?; dl.upload(logits.as_slice())?;
        let db = self.dev_weight(bias)?;
        let dp = DevBuf::alloc(self.dev, self.stream, probs.numel())?;
        // ids on the CPU backend are f32-valued; the kernel writes i32.
        let di = DevBuf::alloc(self.dev, self.stream, n as usize * topk)?;
        ck(unsafe { ferrite_moe_route(dl.as_const_f32(), db.as_const_f32(), dp.as_f32(), di.as_f32(), n, e, topk as i32, routed_scaling, self.stream) }, "moe_route")?;
        let pv = Arc::get_mut(&mut probs.data).expect("unique probs");
        dp.download(pv)?;
        let iv = Arc::get_mut(&mut ids.data).expect("unique ids");
        di.download(iv)?;
        Ok(())
    }

    fn expert_ffn(&self, x: &Tensor, gate_w: &Tensor, up_w: &Tensor, down_w: &Tensor, swiglu_limit: f32, out: &mut Tensor) -> Result<()> {
        self.clear_xq_cache();
        self.enter();
        // Fused device-resident chain: upload x once, two matmuls + swiglu2
        // + down matmul all on device, single D2H at the end. (The old path
        // did two host round-trips plus a host-side gate/up gather.)
        let n = x.shape.0[0] as i32;
        let in_f = x.shape.0[1] as i32;
        let inter = gate_w.shape.0[0] as i32;
        let dx = DevBuf::alloc(self.dev, self.stream, x.numel())?;
        dx.upload(x.as_slice())?;
        let gate = self.matmul_dev(&dx, gate_w, n, in_f, inter)?;
        let up = self.matmul_dev(&dx, up_w, n, in_f, inter)?;
        let act = self.swiglu2_dev(&gate, &up, n, inter, swiglu_limit)?;
        let dout = self.matmul_dev(&act, down_w, n, inter, in_f)?;
        let ov = Arc::get_mut(&mut out.data).expect("unique out");
        dout.download(ov)?;
        Ok(())
    }

    fn argmax_lastdim(&self, logits: &Tensor, out: &mut Tensor) -> Result<()> {
        self.enter();
        let dim = *logits.shape.0.last().unwrap() as i32;
        let n = (logits.numel() / dim as usize) as i32;
        let dl = DevBuf::alloc(self.dev, self.stream, logits.numel())?; dl.upload(logits.as_slice())?;
        let do_ = DevBuf::alloc(self.dev, self.stream, out.numel())?;
        ck(unsafe { ferrite_argmax(dl.as_const_f32(), do_.as_f32(), n, dim, self.stream) }, "argmax")?;
        let ov = Arc::get_mut(&mut out.data).expect("unique out");
        do_.download(ov)?;
        Ok(())
    }

    fn softmax_lastdim(&self, logits: &Tensor, out: &mut Tensor) -> Result<()> {
        self.enter();
        let dim = *logits.shape.0.last().unwrap() as i32;
        let n = (logits.numel() / dim as usize) as i32;
        let dl = DevBuf::alloc(self.dev, self.stream, logits.numel())?; dl.upload(logits.as_slice())?;
        let do_ = DevBuf::alloc(self.dev, self.stream, out.numel())?;
        ck(unsafe { ferrite_softmax(dl.as_const_f32(), do_.as_f32(), n, dim, self.stream) }, "softmax")?;
        let ov = Arc::get_mut(&mut out.data).expect("unique out");
        do_.download(ov)?;
        Ok(())
    }

    // MHC hyper-connections on the GPU — replaces the per-token host loops
    // (24×16384 mixes dot + sinkhorn + weighted combine) that dominated the
    // layer boundary between the fan_out attention/FFN segments.
    fn hc_pre(
        &self,
        residual_flat: &Tensor,
        fn_w: &Tensor,
        scale: &Tensor,
        base: &Tensor,
        rms_eps: f32,
        hc_eps: f32,
        sinkhorn_iters: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        self.enter();
        let s = residual_flat.shape.0[0] as i32;
        let nh = residual_flat.shape.0[1] as i32;
        let mix = fn_w.shape.0[0] as i32;
        let n = ((-2.0 + (4.0 + 4.0 * mix as f64).sqrt()) / 2.0) as i32;
        let h = nh / n;
        let dr = DevBuf::alloc(self.dev, self.stream, residual_flat.numel())?;
        dr.upload(residual_flat.as_slice())?;
        let dfw = self.dev_weight(fn_w)?;
        let dsc = self.dev_weight(scale)?;
        let dba = self.dev_weight(base)?;
        let dli = DevBuf::alloc(self.dev, self.stream, (s * h) as usize)?;
        let dpost = DevBuf::alloc(self.dev, self.stream, (s * n) as usize)?;
        let dcomb = DevBuf::alloc(self.dev, self.stream, (s * n * n) as usize)?;
        ck(
            unsafe {
                ferrite_hc_pre(
                    dr.as_const_f32(), dfw.as_const_f32(), dsc.as_const_f32(), dba.as_const_f32(),
                    dli.as_f32(), dpost.as_f32(), dcomb.as_f32(),
                    s, n, h, mix, rms_eps, hc_eps, sinkhorn_iters as i32, self.stream,
                )
            },
            "hc_pre",
        )?;
        let mut li = Tensor::zeros(Shape::new([s as usize, h as usize]), DType::F32);
        {
            let v = Arc::get_mut(&mut li.data).expect("unique");
            dli.download(v)?;
        }
        let mut post = Tensor::zeros(Shape::new([s as usize, n as usize]), DType::F32);
        {
            let v = Arc::get_mut(&mut post.data).expect("unique");
            dpost.download(v)?;
        }
        let mut comb = Tensor::zeros(Shape::new([s as usize, n as usize, n as usize]), DType::F32);
        {
            let v = Arc::get_mut(&mut comb.data).expect("unique");
            dcomb.download(v)?;
        }
        Ok((li, post, comb))
    }

    fn hc_post(&self, x: &Tensor, residual: &Tensor, post: &Tensor, comb: &Tensor) -> Result<Tensor> {
        self.enter();
        let s = x.shape.0[0] as i32;
        let h = x.shape.0[1] as i32;
        let n = residual.shape.0[1] as i32;
        let dx = DevBuf::alloc(self.dev, self.stream, x.numel())?;
        dx.upload(x.as_slice())?;
        let drs = DevBuf::alloc(self.dev, self.stream, residual.numel())?;
        drs.upload(residual.as_slice())?;
        let dp = DevBuf::alloc(self.dev, self.stream, post.numel())?;
        dp.upload(post.as_slice())?;
        let dc = DevBuf::alloc(self.dev, self.stream, comb.numel())?;
        dc.upload(comb.as_slice())?;
        let do_ = DevBuf::alloc(self.dev, self.stream, (s * n * h) as usize)?;
        ck(
            unsafe { ferrite_hc_post(dx.as_const_f32(), drs.as_const_f32(), dp.as_const_f32(), dc.as_const_f32(), do_.as_f32(), s, n, h, self.stream) },
            "hc_post",
        )?;
        let mut out = Tensor::zeros(Shape::new([s as usize, n as usize, h as usize]), DType::F32);
        {
            let v = Arc::get_mut(&mut out.data).expect("unique");
            do_.download(v)?;
        }
        Ok(out)
    }
}

// ============================================================
// GDN layer device chain — the whole linear-attention forward as one
// DevBuf pipeline (zero host round-trips inside the layer): six
// projections → causal conv (resident state) → fused prep (silu+split+
// l2norm+beta+gate, one kernel) → gated-deltanet core (resident state)
// → gated rmsnorm → o_proj. The caller (TpCluster's device path / the
// future single CUDA graph) feeds [n, hidden] and gets the TP partial
// [n, hidden] back, both as DevBuf.
// ============================================================

/// Device-resident recurrent state (GDN [h,dk,dk] / conv tails) — NOT
/// pooled: it must persist across tokens, and pooled buffers get reused
/// by other ops between tokens.
pub struct DeviceState {
    pub ptr: *mut std::ffi::c_void,
    pub len: usize, // floats
}
unsafe impl Send for DeviceState {}
unsafe impl Sync for DeviceState {}

/// Per-layer MoE expert pointer table (device buffers of e_local raw
/// pointers into the bf16 weight cache) — the fused kernels' indirect
/// addressing for GPU-side expert dispatch.
pub struct MoePtrTable {
    pub gate_dev: *mut std::ffi::c_void,
    pub up_dev: *mut std::ffi::c_void,
    pub down_dev: *mut std::ffi::c_void,
    pub e_local: usize,
}

/// fp8 variant of the expert pointer table: per expert (w8_bytes, scale)
/// pairs for gate/up/down — the fused MoE kernels gather e4m3 rows + block
/// scales through these (HALF the bf16 tables' HBM traffic; the inline
/// dequant is the checkpoint's own semantics, not a re-quantization).
pub struct MoeFp8PtrTable {
    pub gate_w8: *mut std::ffi::c_void,
    pub gate_scale: *mut std::ffi::c_void,
    pub up_w8: *mut std::ffi::c_void,
    pub up_scale: *mut std::ffi::c_void,
    pub down_w8: *mut std::ffi::c_void,
    pub down_scale: *mut std::ffi::c_void,
    pub e_local: usize,
}
unsafe impl Send for MoeFp8PtrTable {}
unsafe impl Sync for MoeFp8PtrTable {}
unsafe impl Send for MoePtrTable {}
unsafe impl Sync for MoePtrTable {}

/// Weight set for one GDN layer's device chain (borrowed from the shard's
/// Engine weights — all hit the dev_weight caches after warmup preload).
pub struct GdnLayerWeights<'a> {
    pub qkv_proj: &'a Tensor,
    pub b_proj: &'a Tensor,
    pub f_a: &'a Tensor,
    pub f_b: &'a Tensor,
    pub g_a: &'a Tensor,
    pub g_b: &'a Tensor,
    pub conv_w: &'a Tensor,
    pub dt_bias: &'a Tensor,
    pub a_log: &'a Tensor,
    pub o_norm: &'a Tensor,
    pub o_proj: &'a Tensor,
}

/// Device-resident DSA cache: k_nope [max_t, h, dk], v [max_t, h, dv],
/// k_idx/k_gate [max_t, idm] — pre-allocated, appended in place by
/// ferrite_dsa_cache_append. The CPU path grew host Vecs and cloned them
/// per layer per token (MBs of memcpy per call).
pub struct DsaCacheState {
    pub k_nope: *mut std::ffi::c_void,
    pub v: *mut std::ffi::c_void,
    pub k_nope_scale: *mut std::ffi::c_void,
    pub v_scale: *mut std::ffi::c_void,
    pub k_idx: *mut std::ffi::c_void,
    pub k_gate: *mut std::ffi::c_void,
    pub max_tokens: usize,
    /// tokens appended so far (device-side counter; the CPU Vecs are gone)
    pub t_count: usize,
    /// PINNED t0/total (graph-safe): the CPU writes these before each
    /// graph replay; kernels read them zero-copy from host memory.
    /// [t0, total] — 2 ints, cudaMallocHost'd.
    pub pinned_t0: *mut i32,
    pub pinned_total: *mut i32,
}
unsafe impl Send for DsaCacheState {}
unsafe impl Sync for DsaCacheState {}

impl DsaCacheState {
    fn clone_raw(&self) -> (*mut std::ffi::c_void, *mut std::ffi::c_void, *mut std::ffi::c_void, *mut std::ffi::c_void, usize) {
        (self.k_nope, self.v, self.k_idx, self.k_gate, self.max_tokens)
    }
}

/// Device pointer tables for the BATCHED DSA kernels: [B] arrays of the
/// per-seq cache pointers (kn/v/kidx/kgate — stable for the caches'
/// lifetime) + [B] arrays of the per-seq PINNED t0/total int POINTERS
/// (kernels deref zero-copy; the ints are host-written per step —
/// graph-safe). One (family, seq-set) composition → one table set,
/// cudaMalloc'd + memcpy'd once, cached, purged by free_seq.
#[derive(Clone, Copy)]
pub struct DsaBatchTables {
    pub kn: *mut std::ffi::c_void,
    pub v: *mut std::ffi::c_void,
    pub kns: *mut std::ffi::c_void,
    pub vs: *mut std::ffi::c_void,
    pub kidx: *mut std::ffi::c_void,
    pub kgate: *mut std::ffi::c_void,
    pub t0p: *mut std::ffi::c_void,
    pub totp: *mut std::ffi::c_void,
}
unsafe impl Send for DsaBatchTables {}
unsafe impl Sync for DsaBatchTables {}

/// P2P one-shot all-reduce state (per rank, v2 epoch+ping-pong protocol —
/// the in-graph decode-chain AR): [2][world][max_n] ping-pong staging +
/// [world] epoch-stamp flags + epoch/ctr counters + the [world] tables of
/// the PEERS' staging/flag bases (UVA — same process, peer access enabled).
/// cudaMalloc'd ONCE at cluster setup (FERRITE_P2P), NEVER pool-allocated
/// (fixed addresses across graph captures); the counters are runtime
/// device state the captured kernels advance per replay.
#[derive(Clone, Copy)]
pub struct P2pArState {
    pub staging_local: *mut std::ffi::c_void, // [2][world][max_n] f32
    pub ready_local: *mut std::ffi::c_void,   // [world] u32 epoch stamps
    pub seen: *mut std::ffi::c_void,          // [world] u32 last-observed stamps
    pub epoch: *mut std::ffi::c_void,         // [1] u32 call counter
    pub ctr: *mut std::ffi::c_void,          // [1] u32 block arrivals
    pub staging_tbl: *mut std::ffi::c_void,   // [world] device ptrs (peers' staging bases)
    pub ready_tbl: *mut std::ffi::c_void,     // [world] device ptrs (peers' flag rows)
    pub world: usize,
    pub max_n: usize,
}
unsafe impl Send for P2pArState {}
unsafe impl Sync for P2pArState {}

/// Weight set for one DSA layer's device chain (borrowed from the shard
/// Engine's weights — all hit the dev_weight caches after preload).
pub struct DsaLayerWeights<'a> {
    pub q_a: &'a Tensor,
    pub q_a_ln: &'a Tensor,
    pub q_b: &'a Tensor,
    pub kv_a: &'a Tensor,
    pub kv_a_ln: &'a Tensor,
    pub kv_b: &'a Tensor,
    pub wq_b: &'a Tensor,
    pub wk: &'a Tensor,
    pub k_norm_w: &'a Tensor,
    pub k_norm_b: &'a Tensor,
    pub weights_proj: &'a Tensor,
    pub gate: &'a Tensor,
    pub ape: &'a Tensor,
    pub o_proj: &'a Tensor,
    // dims
    pub h: usize,
    pub dk: usize,
    pub dv: usize,
    pub ih: usize,
    pub idm: usize,
    pub kpool: usize,
    pub topk: usize,
    pub rms_eps: f32,
}

impl CudaBackend {
    /// Main GDN state ptr (the A side of the verify ping-pong).
    pub fn gdn_state_ptr(&self, seq: u64, layer: usize, len: usize) -> Result<*mut f32> {
        self.dev_state(&self.gdn_states, (seq, layer), len)
    }

    /// Main conv-tail state ptr (the A side of the verify ping-pong).
    pub fn conv_state_ptr(&self, seq: u64, layer: usize, len: usize) -> Result<*mut f32> {
        self.dev_state(&self.conv_states, (seq, layer), len)
    }

    pub fn dev_state(
        &self,
        store: &std::sync::Mutex<std::collections::HashMap<(u64, usize), DeviceState>>,
        key: (u64, usize),
        len: usize,
    ) -> Result<*mut f32> {
        let mut m = store.lock().unwrap();
        if let Some(st) = m.get(&key) {
            return Ok(st.ptr as *mut f32);
        }
        // Pooled: same free+realloc hazard as the DSA caches (2026-09-09) —
        // reusing the same VA avoids the driver's remap path.
        let ptr = self.dsa_alloc(len)?;
        m.insert(key, DeviceState { ptr, len });
        Ok(ptr as *mut f32)
    }

    /// Batched per-seq GDN state pointer tables (the batched decode): the
    /// (conv, gdn) device arrays of B per-seq state pointers for ONE layer —
    /// the batched kernels' (state_ptrs[seq]) indirection. Built ONCE per
    /// (layer, seq-set) at the dry-run (cudaMalloc + memcpy — no capture
    /// running), cached: the capture pass re-uses the SAME pointers (no
    /// malloc during capture, the graph's kernel args stable). The table
    /// content is frozen per composition (the (seq, layer) state addresses
    /// are stable for the states' lifetime). Purged by free_seq (a member's
    /// state freed → the table dangles).
    pub fn gdn_state_tables(
        &self,
        layer: usize,
        seqs: &[u64],
        conv_len: usize,
        gdn_len: usize,
    ) -> Result<(*const *mut f32, *const *mut f32)> {
        // KEYED BY (layer, SIZE) — NOT the composition. The device table's
        // address must be stable so ONE per-size CUDA graph serves any
        // seq-set of that size (SGLang's cuda-graph batch-size padding); a
        // composition-keyed table forced a graph re-capture on every
        // membership change (measured: 8 captures while 16 requests streamed
        // in → throughput collapsed to 74 tok/s). Content (the per-seq state
        // pointers) is refreshed on EVERY call; padded slots point at a
        // shared per-layer dummy state (their outputs are discarded).
        let size = seqs.len();
        let key = (layer, size);
        let (c_tbl, g_tbl) = {
            let mut m = self.gdn_tbl_cache.lock().unwrap();
            match m.get(&key) {
                Some(&(c, g)) => (c, g),
                None => {
                    let b = size * std::mem::size_of::<*mut f32>();
                    let mut p: *mut std::ffi::c_void = std::ptr::null_mut();
                    ck(unsafe { cudaMalloc(&mut p, b) }, "gdn tbl malloc")?;
                    let mut q: *mut std::ffi::c_void = std::ptr::null_mut();
                    ck(unsafe { cudaMalloc(&mut q, b) }, "gdn tbl malloc")?;
                    m.insert(key, (p, q));
                    (p, q)
                }
            }
        };
        let mut conv_ptrs = Vec::with_capacity(size);
        let mut gdn_ptrs = Vec::with_capacity(size);
        for &seq_r in seqs {
            if seq_r == u64::MAX {
                // padded row → shared dummy state (output discarded)
                conv_ptrs.push(self.dev_state(&self.conv_states, (u64::MAX, layer), conv_len)?);
                gdn_ptrs.push(self.dev_state(&self.gdn_states, (u64::MAX, layer), gdn_len)?);
            } else {
                conv_ptrs.push(self.dev_state(&self.conv_states, (seq_r, layer), conv_len)?);
                gdn_ptrs.push(self.dev_state(&self.gdn_states, (seq_r, layer), gdn_len)?);
            }
        }
        if conv_ptrs.len() < size {
            let dc = self.dev_state(&self.conv_states, (u64::MAX, layer), conv_len)?;
            let dg = self.dev_state(&self.gdn_states, (u64::MAX, layer), gdn_len)?;
            while conv_ptrs.len() < size {
                conv_ptrs.push(dc);
                gdn_ptrs.push(dg);
            }
        }
        let b = size * std::mem::size_of::<*mut f32>();
        // The table CONTENT is written by the dry-run (capturing() == false)
        // BEFORE the capture pass. Inside capture neither async nor sync
        // H2D is allowed (err 900 / a captured node replaying freed stack
        // memory → Xid 13), so skip it there — the recorded kernels read
        // the table the dry-run already filled.
        if !self.capturing() {
            ck(
                unsafe { cudaMemcpyAsync(c_tbl, conv_ptrs.as_ptr() as *const _, b, CUDA_MEMCPY_H2D, self.stream) },
                "gdn tbl update",
            )?;
            ck(
                unsafe { cudaMemcpyAsync(g_tbl, gdn_ptrs.as_ptr() as *const _, b, CUDA_MEMCPY_H2D, self.stream) },
                "gdn tbl update",
            )?;
        }
        Ok((c_tbl as *const *mut f32, g_tbl as *const *mut f32))
    }

    /// Batched DSA per-seq pointer tables: (family, seq-set) → 6 device
    /// [B] arrays — k_nope/v/k_idx/k_gate cache pointers (stable for the
    /// caches' lifetime) + the per-seq PINNED t0/total int POINTERS (the
    /// kernels dereference them zero-copy; the host writes the ints per
    /// step — the same graph-safe mechanism as the per-seq kernels' single
    /// pinned pointer). cudaMalloc'd + memcpy'd ONCE per composition, cached
    /// (the capture pass re-uses the same pointers); purged by free_seq.
    /// REQUIRES all (seq, family) caches to exist (the caller's bookkeeping
    /// loop get-or-creates them first).
    pub fn dsa_ptr_tables(
        &self,
        family: usize,
        seqs: &[u64],
        h: usize,
        dk: usize,
        dv: usize,
        idm: usize,
    ) -> Result<DsaBatchTables> {
        // KEYED BY (family, SIZE) — stable addresses so a per-size CUDA graph
        // can be reused for any seq-set of that size (see gdn_state_tables).
        // Padded slots point at a shared per-family dummy cache (1 token:
        // the padded rows' attention reads only slot 0; their outputs are
        // discarded).
        let size = seqs.len();
        let key = (family, size);
        let b = size * std::mem::size_of::<*mut f32>();
        let t = {
            let mut m = self.dsa_tbl_cache.lock().unwrap();
            match m.get(&key) {
                Some(&t) => t,
                None => {
                    let mk = || -> Result<*mut std::ffi::c_void> {
                        let mut p: *mut std::ffi::c_void = std::ptr::null_mut();
                        ck(unsafe { cudaMalloc(&mut p, b) }, "dsa tbl malloc")?;
                        Ok(p)
                    };
                    let t = DsaBatchTables {
                        kn: mk()?,
                        v: mk()?,
                        kns: mk()?,
                        vs: mk()?,
                        kidx: mk()?,
                        kgate: mk()?,
                        t0p: mk()?,
                        totp: mk()?,
                    };
                    m.insert(key, t);
                    t
                }
            }
        };
        let mut kn: Vec<*mut f32> = Vec::with_capacity(size);
        let mut vv: Vec<*mut f32> = Vec::with_capacity(size);
        let mut kns: Vec<*mut f32> = Vec::with_capacity(size);
        let mut vss: Vec<*mut f32> = Vec::with_capacity(size);
        let mut ki_: Vec<*mut f32> = Vec::with_capacity(size);
        let mut kg: Vec<*mut f32> = Vec::with_capacity(size);
        let mut t0p: Vec<*const i32> = Vec::with_capacity(size);
        let mut totp: Vec<*const i32> = Vec::with_capacity(size);
        let dummy = if seqs.contains(&u64::MAX) {
            Some(self.dsa_dummy(family, h, dk, dv, idm)?)
        } else {
            None
        };
        {
            let m = self.dsa_caches.lock().unwrap();
            for &s in seqs {
                if s == u64::MAX {
                    let (dk_, dv_, dks_, dvs_, di_, dg_, dt0, dtot) = dummy.expect("dummy built above");
                    kn.push(dk_);
                    vv.push(dv_);
                    kns.push(dks_);
                    vss.push(dvs_);
                    ki_.push(di_);
                    kg.push(dg_);
                    t0p.push(dt0);
                    totp.push(dtot);
                    continue;
                }
                let c = m
                    .get(&(s, family))
                    .ok_or_else(|| FerriteError::InvalidArg(format!("no dsa cache ({s},{family}) for batched tables")))?;
                kn.push(c.k_nope as *mut f32);
                vv.push(c.v as *mut f32);
                kns.push(c.k_nope_scale as *mut f32);
                vss.push(c.v_scale as *mut f32);
                ki_.push(c.k_idx as *mut f32);
                kg.push(c.k_gate as *mut f32);
                t0p.push(c.pinned_t0 as *const i32);
                totp.push(c.pinned_total as *const i32);
            }
        }
        if kn.len() < size {
            let (dk_, dv_, dks_, dvs_, di_, dg_, dt0, dtot) = self.dsa_dummy(family, h, dk, dv, idm)?;
            while kn.len() < size {
                kn.push(dk_);
                vv.push(dv_);
                kns.push(dks_);
                vss.push(dvs_);
                ki_.push(di_);
                kg.push(dg_);
                t0p.push(dt0);
                totp.push(dtot);
            }
        }
        // capture-aware (see gdn_state_tables): the dry-run already filled
        // the table; inside capture H2D is illegal (err 900 / Xid 13).
        let up = |dst: *mut std::ffi::c_void, src: *const std::ffi::c_void| -> Result<()> {
            if self.capturing() {
                Ok(())
            } else {
                ck(unsafe { cudaMemcpyAsync(dst, src, b, CUDA_MEMCPY_H2D, self.stream) }, "dsa tbl update")
            }
        };
        up(t.kn, kn.as_ptr() as *const _)?;
        up(t.v, vv.as_ptr() as *const _)?;
        up(t.kns, kns.as_ptr() as *const _)?;
        up(t.vs, vss.as_ptr() as *const _)?;
        up(t.kidx, ki_.as_ptr() as *const _)?;
        up(t.kgate, kg.as_ptr() as *const _)?;
        up(t.t0p, t0p.as_ptr() as *const _)?;
        up(t.totp, totp.as_ptr() as *const _)?;
        Ok(t)
    }

    /// Shared per-family dummy DSA cache for padded batch rows (1 token;
    /// zeros; pinned t0/total = 0/1 so the kernels read only slot 0).
    fn dsa_dummy(&self, family: usize, h: usize, dk: usize, dv: usize, idm: usize) -> Result<(*mut f32, *mut f32, *mut f32, *mut f32, *mut f32, *mut f32, *const i32, *const i32)> {
        let mut m = self.dsa_dummy_cache.lock().unwrap();
        if let Some(d) = m.get(&family) {
            return Ok(*d);
        }
        let mut alloc = |len: usize| -> Result<*mut f32> {
            let mut p: *mut std::ffi::c_void = std::ptr::null_mut();
            ck(unsafe { cudaMalloc(&mut p, len * 4) }, "dsa dummy malloc")?;
            ck(unsafe { cudaMemset(p, 0, len * 4) }, "dsa dummy zero")?;
            Ok(p as *mut f32)
        };
        // SAME token dimension as the real caches: the DSA kernels index the
        // cache by pool/topk offsets (not just t0), so a 1-token dummy
        // faults with Xid 13 Out-Of-Range (measured).
        const MAXT: usize = 8192;
        let kn = alloc(MAXT * h * dk)?;
        let vv = alloc(MAXT * h * dv)?;
        let dks_ = alloc(MAXT * h)?;
        let dvs_ = alloc(MAXT * h)?;
        let ki_ = alloc(MAXT * idm)?;
        let kg = alloc(MAXT * idm)?;
        let mut pt0: *mut i32 = std::ptr::null_mut();
        let mut ptot: *mut i32 = std::ptr::null_mut();
        ck(unsafe { cudaMallocHost(&mut pt0 as *mut *mut i32 as *mut *mut std::ffi::c_void, 4) }, "dsa dummy pinned")?;
        ck(unsafe { cudaMallocHost(&mut ptot as *mut *mut i32 as *mut *mut std::ffi::c_void, 4) }, "dsa dummy pinned")?;
        unsafe {
            *pt0 = 0;
            // total = MAXT (not 1): the indexer/pool kernels derive the
            // number of pools from it; total=1 gave npools=1 and the padded
            // rows' topk/pool loops spun (measured hang at size=16).
            *ptot = MAXT as i32;
        }
        let tup = (kn, vv, dks_, dvs_, ki_, kg, pt0 as *const i32, ptot as *const i32);
        m.insert(family, tup);
        Ok(tup)
    }

    /// Whole GDN (linear-attention) layer on device. `x` is the layer's
    /// normed input [n, hidden]; returns the o_proj partial [n, hidden]
    /// (TP all-reduce happens at the caller).
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_layer_dev(
        &self,
        x: &DevBuf,
        w: &GdnLayerWeights,
        seq: u64,
        layer: usize,
        n: usize,
        hidden: usize,
        h: usize,
        dk: usize,
        lb: f32,
        rms_eps: f32,
        conv_size: usize,
        state_override: Option<(*mut f32, *mut f32, *mut f32, *mut f32)>, // (conv, gdn, conv_snaps_base, gdn_snaps_base) N-UNIFIED verify scratch: B = full state, snaps = the [n-1] contiguous t-snapshots (snap i = A + t_0..t_i, accept-(i+1)'s commit source)
    ) -> Result<DevBuf> {
        self.clear_xq_cache();
        self.enter();
        let proj = h * dk;
        let ni = n as i32;
        // 1. six projections (bf16-resident weights). NOTE: the gemv5 fused
        // same-input kernel measured SLOWER than separate tiled GEMVs
        // (48.7us vs 40.5us — block-per-row co-op dot loses to the tiled
        // smem kernel's throughput; graph-replay has no launch tail to
        // save) — kept separate. gemv5_dev stays for a future tiled fused
        // version (weight concat at load time).
        // knife 1b (n==1): the qkv GEMV's epilogue does conv FIR + silu +
        // window slide inline (gemv_qkv_conv) — no qkv buffer materialized.
        let qkv = if n > 1 {
            Some(self.matmul_dev(x, w.qkv_proj, ni, hidden as i32, (3 * proj) as i32)?)
        } else {
            None
        };
        // PROBE: dump x (input) + qkv (first matmul output) — pinpoints
        // divergence to upload (x wrong) vs matmul/weights (qkv wrong)
        if std::env::var_os("FERRITE_GDN_PROBE").is_some() && layer == 0 && n > 1 {
            let dir = std::env::var("FERRITE_PROBE_DIR").unwrap_or_else(|_| "/tmp/orion".into());
            let d = |name: &str, v: &[f32]| {
                let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
                std::fs::write(format!("{dir}/gdn_dev_{name}_r{}.f32", crate::shard_idx()), b).ok();
            };
            let mut xh = vec![0f32; x.len];
            let _ = x.download(&mut xh);
            d("x", &xh);
            let mut qh = vec![0f32; n * 3 * proj];
            let _ = qkv.as_ref().unwrap().download(&mut qh);
            d("qkv", &qh);
            eprintln!("[gdn_probe] dev L0 x/qkv dumped: x {} qkv {} (n={} proj={})", xh.len(), qh.len(), n, proj);
        }
        let (b_raw, fa, ga) = if n == 1 {
            // 3-in-1 GEMV (b_raw || f_a || g_a — same input x): one kernel maps
            // the three weight matrices onto one row space (WPR=4 K-split) —
            // 3 kernel launches → 1 per gdn layer × 34 layers.
            self.gemv_tri_dev(x, w.b_proj, w.f_a, w.g_a, hidden as i32,
                              h as i32, dk as i32, dk as i32)?
        } else {
            (self.matmul_dev(x, w.b_proj, ni, hidden as i32, h as i32)?,
             self.matmul_dev(x, w.f_a, ni, hidden as i32, dk as i32)?,
             self.matmul_dev(x, w.g_a, ni, hidden as i32, dk as i32)?)
        };
        let fb = self.matmul_dev(&fa, w.f_b, ni, dk as i32, proj as i32)?;
        let gb = self.matmul_dev(&ga, w.g_b, ni, dk as i32, proj as i32)?;
        // 2. causal conv — resident tail state (RMW in place: the kernel
        // reads state_in into smem at block start, writes state_out at end,
        // one block per channel, so in==out is safe)
        let ch = 3 * proj;
        let hist = conv_size.saturating_sub(1).max(1);
        let dw_conv = self.dev_weight(w.conv_w)?;
        let conv_state = match state_override {
            Some((cs, _, _, _)) => cs,
            None => self.dev_state(&self.conv_states, (seq, layer), ch * hist)?,
        };
        let q = DevBuf::alloc(self.dev, self.stream, n * proj)?;
        let k = DevBuf::alloc(self.dev, self.stream, n * proj)?;
        let v = DevBuf::alloc(self.dev, self.stream, n * proj)?;
        let beta = DevBuf::alloc(self.dev, self.stream, n * h)?;
        let gate = DevBuf::alloc(self.dev, self.stream, n * proj)?;
        let dw_dt = self.dev_weight(w.dt_bias)?;
        let dw_al = self.dev_weight(w.a_log)?;
        if n == 1 {
            // knife 1b: qkv GEMV epilogue does conv FIR + silu + window slide
            // inline — q/k/v come out raw (un-L2'd); gdn_step_v2p's prologue
            // computes L2/gate/beta. Kills the standalone conv_prep node.
            let dw_qkv = self.dev_weight_bf16(w.qkv_proj)?;
            ck(
                unsafe {
                    ferrite_gemv_qkv_conv(
                        x.as_const_f32(), dw_qkv.ptr, dw_conv.as_const_f32(), conv_state,
                        q.as_f32(), k.as_f32(), v.as_f32(),
                        hidden as i32, proj as i32, self.stream,
                    )
                },
                "gemv_qkv_conv",
            )?;
        } else {
            let conv_out = DevBuf::alloc(self.dev, self.stream, n * ch)?;
            // N-UNIFIED (verify 算子 = n=1 算子): ONE conv1d launch for ALL n
            // tokens (the kernel loops t in smem) + the t-snapshots written
            // straight from the smem stream — the old verify path launched a
            // PER-TOKEN conv1d (n launches) + a D2D snapshot copy per t (the
            // n=3 GDN layer was 8 kernels vs n=1's 2). snaps = the verify
            // scratch base (cs_base + t*(ch*hist)); non-verify passes null.
            let snaps = match state_override {
                Some((_, _, cs_base, _)) => cs_base,
                None => std::ptr::null_mut(),
            };
            ck(
                unsafe {
                    ferrite_causal_conv1d(
                        qkv.as_ref().unwrap().as_const_f32(), dw_conv.as_const_f32(), conv_state,
                        conv_out.as_f32(), conv_state, snaps, ni, ch as i32, conv_size as i32, self.stream,
                    )
                },
                "conv1d_dev",
            )?;
            // 3. GPU pre-processing (ferrite_gdn_prep): silu + split + per-head L2
            // norm + KDA q-scale + beta + gate — ONE kernel, zero host round-trips.
            ck(
                unsafe {
                    ferrite_gdn_prep(
                        conv_out.as_const_f32(), b_raw.as_const_f32(), fb.as_const_f32(),
                        dw_dt.as_const_f32(), dw_al.as_const_f32(),
                        q.as_f32(), k.as_f32(), v.as_f32(), beta.as_f32(), gate.as_f32(),
                        ni, h as i32, dk as i32, lb, self.stream,
                    )
                },
                "gdn_prep",
            )?;
        }
        // PROBE (rank-isolated, prefill-only): download intermediates for
        // CPU-vs-dev divergence diff; normal path stays zero-crossing.
        if std::env::var_os("FERRITE_GDN_PROBE").is_some() && layer == 0 && n > 1 {
            let dir = std::env::var("FERRITE_PROBE_DIR").unwrap_or_else(|_| "/tmp/orion".into());
            let d = |name: &str, v: &[f32]| {
                let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
                std::fs::write(format!("{dir}/gdn_dev_{name}_r{}.f32", crate::shard_idx()), b).ok();
            };
            let mut bh0 = vec![0f32; n * h];
            let _ = b_raw.download(&mut bh0);
            d("braw", &bh0);
            let mut fh = vec![0f32; n * proj];
            let _ = fb.download(&mut fh);
            d("fb", &fh);
            let mut qh = vec![0f32; n * proj];
            let _ = q.download(&mut qh);
            d("q", &qh);
            let mut kh = vec![0f32; n * proj];
            let _ = k.download(&mut kh);
            d("k", &kh);
            let mut bth = vec![0f32; n * h];
            let _ = beta.download(&mut bth);
            d("beta", &bth);
            let mut gth = vec![0f32; n * proj];
            let _ = gate.download(&mut gth);
            d("gate", &gth);
            eprintln!(
                "[gdn_probe] dev L0 gpu-prep dumped r{} (q {} — conv_prep_fused path)",
                crate::shard_idx(), qh.len()
            );
        }
        // 4. gated-deltanet core — resident [h, dk, dk] state (per-head
        // blocks read-modify-write their own slice; single buffer is safe).
        // v2: state staged in smem (padded stride dk+1) — HBM 7 passes → 2.
        let gdn_state = match state_override {
            Some((_, gs, _, _)) => gs,
            None => self.dev_state(&self.gdn_states, (seq, layer), h * dk * dk)?,
        };
        let core = DevBuf::alloc(self.dev, self.stream, n * proj)?;
        if n == 1 {
            // knife 1b: v2p — prologue computes L2 norm, KDA gate and beta
            // inline (from fb/dt_bias/a_log/b_raw), replacing conv_prep's
            // second half. q/k arrive raw FIR+silu (un-L2'd).
            ck(
                unsafe {
                    ferrite_gdn_step_v2p(
                        q.as_const_f32(), k.as_const_f32(), v.as_const_f32(),
                        b_raw.as_const_f32(), fb.as_const_f32(),
                        dw_dt.as_const_f32(), dw_al.as_const_f32(), lb,
                        gdn_state, core.as_f32(), h as i32, dk as i32, dk as i32,
                        (256 / (h as i32).max(1)).clamp(1, 32), self.stream,
                    )
                },
                "gdn_step_v2p",
            )?;
        } else if let Some((_, _, _, gs_base)) = state_override {
            // MTP Phase2 (N-UNIFIED): fused single-launch n-token chunk — the
            // state stays resident in smem across the t loop (HBM round-trip
            // eliminated), the t-th snapshot (A + t_0..t_i, accept-(i+1)'s
            // commit source) is written straight from smem to the contiguous
            // [n-1] snapshot scratch (gs_base + i*h*dk*dk). Replaces the
            // 2-launch t-split + the in-between D2D copy. gdn0 = the snaps
            // base, gdn1 unused (kept for the FFI's ABI; the kernel takes the
            // base + derives snap i by stride).
            let dal = self.dev_weight(w.a_log)?;
            ck(
                unsafe {
                    ferrite_gdn_chunk_fused(
                        q.as_const_f32(), k.as_const_f32(), v.as_const_f32(),
                        beta.as_const_f32(), gate.as_const_f32(), dal.as_const_f32(),
                        gdn_state, gs_base, std::ptr::null_mut(), core.as_f32(),
                        ni, h as i32, dk as i32, dk as i32, self.stream,
                    )
                },
                "gdn_chunk_fused",
            )?;
        } else {
            ck(
                unsafe {
                    ferrite_gdn_chunk_v2(
                        q.as_const_f32(), k.as_const_f32(), v.as_const_f32(),
                        beta.as_const_f32(), gate.as_const_f32(), self.dev_weight(w.a_log)?.as_const_f32(),
                        gdn_state, core.as_f32(), ni, h as i32, dk as i32, dk as i32, self.stream,
                    )
                },
                "gdn_chunk_dev",
            )?;
        }
        // probe: core output NaN check
        if std::env::var_os("FERRITE_GDN_PROBE").is_some() && layer == 0 {
            let mut cb = vec![0f32; n * proj];
            core.download(&mut cb)?;
            let nan_c = cb.iter().filter(|x| x.is_nan()).count();
            let cmax = cb.iter().fold(0f32, |a, v| if v.is_finite() { a.max(v.abs()) } else { a });
            // also check gdn state
            let mut sb = vec![0f32; h * dk * dk];
            unsafe {
                ck(cudaMemcpy(sb.as_mut_ptr() as *mut std::ffi::c_void, gdn_state as *const std::ffi::c_void, h * dk * dk * 4, CUDA_MEMCPY_D2H), "state probe")?;
            }
            let nan_s = sb.iter().filter(|x| x.is_nan()).count();
            let smax = sb.iter().fold(0f32, |a, v| if v.is_finite() { a.max(v.abs()) } else { a });
            eprintln!(
                "[gdn_dev] L{layer} core NaN {nan_c}/{} core_max {cmax:.3e} state NaN {nan_s}/{} state_max {smax:.3e}",
                n * proj, h * dk * dk
            );
        }
        // 5. gated rmsnorm (core [n,h,dk] flat = [n*h, dk]; gb the gate)
        let o_norm_w = self.dev_weight(w.o_norm)?;
        let normed = DevBuf::alloc(self.dev, self.stream, n * proj)?;
        ck(
            unsafe {
                ferrite_gated_rmsnorm(
                    core.as_const_f32(), gb.as_const_f32(), o_norm_w.as_const_f32(),
                    normed.as_f32(), (n * h) as i32, dk as i32, rms_eps, self.stream,
                )
            },
            "gdn_norm_dev",
        )?;
        // 6. o_proj — TP partial out (probe: dump core + partial rank-isolated)
        let partial = self.matmul_dev(&normed, w.o_proj, ni, proj as i32, hidden as i32)?;
        if std::env::var_os("FERRITE_GDN_PROBE").is_some() && layer == 0 && n > 1 {
            let dir = std::env::var("FERRITE_PROBE_DIR").unwrap_or_else(|_| "/tmp/orion".into());
            let d = |name: &str, v: &[f32]| {
                let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
                std::fs::write(format!("{dir}/gdn_dev_{name}_r{}.f32", crate::shard_idx()), b).ok();
            };
            let mut ch = vec![0f32; n * proj];
            let _ = core.download(&mut ch);
            d("core", &ch);
            let mut ph = vec![0f32; n * hidden];
            let _ = partial.download(&mut ph);
            d("partial", &ph);
            eprintln!("[gdn_probe] dev L0 core/partial dumped r{} (n={} proj={} hidden={})", crate::shard_idx(), n, proj, hidden);
        }
        Ok(partial)
    }

    /// Whole DSA (sparse attention) layer on device, zero host round-trips:
    /// gemv projections → layernorm → cache append (device-resident KV +
    /// index caches) → kpool compress → indexer topk → pool expand →
    /// sparse attention → o_proj. The CPU path did 10 Tensor-level ops
    /// (each a sync) + host-side cache clones per layer per token
    /// (2.8ms/layer measured).
    #[allow(clippy::too_many_arguments)]
    pub fn dsa_layer_dev(
        &self,
        x: &DevBuf,
        w: &DsaLayerWeights,
        seq: u64,
        family: usize,
        n: usize,
        hidden: usize,
    ) -> Result<DevBuf> {
        self.clear_xq_cache();
        self.enter();
        let ni = n as i32;
        let (h, dk, dv, ih, idm, kpool) = (w.h, w.dk, w.dv, w.ih, w.idm, w.kpool);

        // 1. query path: qa → rmsnorm → qb [n, h*dk]. (gemv5 fused same-input
        // GEMV measured SLOWER than separate tiled — see gdn note.)
        let qa = self.matmul_dev(x, w.q_a, ni, hidden as i32, (w.q_a.shape.0[0]) as i32)?;
        let qa_ln = self.rmsnorm_dev(&qa, w.q_a_ln, w.rms_eps, n, w.q_a.shape.0[0])?;
        let qb = self.matmul_dev(&qa_ln, w.q_b, ni, w.q_a.shape.0[0] as i32, (h * dk) as i32)?;

        // 2. kv path: latent → rmsnorm → kvb [n, h*(dk+dv)]
        let latent = self.matmul_dev(x, w.kv_a, ni, hidden as i32, (w.kv_a.shape.0[0]) as i32)?;
        let kv_ln = self.rmsnorm_dev(&latent, w.kv_a_ln, w.rms_eps, n, w.kv_a.shape.0[0])?;
        let kvb = self.matmul_dev(&kv_ln, w.kv_b, ni, w.kv_a.shape.0[0] as i32, (h * (dk + dv)) as i32)?;

        // 3. indexer queries: qi = qa @ wq_b [n, ih*idm]
        let qi = self.matmul_dev(&qa_ln, w.wq_b, ni, w.q_a.shape.0[0] as i32, (ih * idm) as i32)?;

        // 4-6. index keys / per-head weights / kpool gate: three SAME-INPUT(x)
        // GEMVs fused into ONE tri kernel at decode (n==1) — ki = LN(x@wk),
        // w_idx = (x@weights_proj)·ih^-0.5, gate = x@compress_gate.
        let (ki_raw, w_idx, gate) = if n == 1 {
            let (a, b, c) = self.gemv_tri_dev(
                x, w.wk, w.weights_proj, w.gate,
                hidden as i32, idm as i32, ih as i32, idm as i32,
            )?;
            (a, b, c)
        } else {
            (
                self.matmul_dev(x, w.wk, ni, hidden as i32, idm as i32)?,
                self.matmul_dev(x, w.weights_proj, ni, hidden as i32, ih as i32)?,
                self.matmul_dev(x, w.gate, ni, hidden as i32, idm as i32)?,
            )
        };
        let ki = DevBuf::alloc(self.dev, self.stream, n * idm)?;
        let knw = self.dev_weight(w.k_norm_w)?;
        let knb = self.dev_weight(w.k_norm_b)?;
        ck(
            unsafe { ferrite_layernorm_affine(ki_raw.as_const_f32(), knw.as_const_f32(), knb.as_const_f32(), ki.as_f32(), ni, idm as i32, self.stream) },
            "dsa_layernorm",
        )?;
        ck(
            unsafe { ferrite_scale_inplace(w_idx.as_f32(), (ih as f32).sqrt().recip(), (n * ih) as i32, self.stream) },
            "dsa_widx_scale",
        )?;

        // 7. cache append (device-resident, in place at slot t0)
        let (k_nope_dev, v_dev, kn_scale_dev, v_scale_dev, k_idx_dev, k_gate_dev, t0) = {
            let mut m = self.dsa_caches.lock().unwrap();
            let (t0, ptrs) = match m.get(&(seq, family)) {
                Some(c) => (c.t_count, (c.k_nope, c.v, c.k_nope_scale, c.v_scale, c.k_idx, c.k_gate)),
                None => (0usize, (std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut())),
            };
            if ptrs.0.is_null() {
                // FERRITE_DSA_MAXT: diagnostic knob for the per-seq DSA cache
                // capacity (default 8192). Shrinking it tests whether the
                // batched-path PDE faults are tied to these huge (~0.5GB each)
                // allocations.
                let max_tokens = std::env::var("FERRITE_DSA_MAXT")
                    .ok()
                    .and_then(|v| v.parse::<usize>().ok())
                    .unwrap_or(8192usize);
                let kn = self.dsa_alloc(max_tokens * h * dk)?;
                let vv = self.dsa_alloc(max_tokens * h * dv)?;
                let kns = self.dsa_alloc(max_tokens * h)?;   // fp8 per-(token,head) scales
                let vss = self.dsa_alloc(max_tokens * h)?;
                let ki_ = self.dsa_alloc(max_tokens * idm)?;
                let kg = self.dsa_alloc(max_tokens * idm)?;
                // pinned t0/total (graph-safe zero-copy)
                let mut pt0: *mut i32 = std::ptr::null_mut();
                let mut ptot: *mut i32 = std::ptr::null_mut();
                ck(unsafe { cudaMallocHost(&mut pt0 as *mut *mut i32 as *mut *mut std::ffi::c_void, 4) }, "pinned t0")?;
                ck(unsafe { cudaMallocHost(&mut ptot as *mut *mut i32 as *mut *mut std::ffi::c_void, 4) }, "pinned total")?;
                unsafe { *pt0 = 0; *ptot = 0; }
                m.insert(
                    (seq, family),
                    DsaCacheState { k_nope: kn, v: vv, k_nope_scale: kns, v_scale: vss, k_idx: ki_, k_gate: kg, max_tokens, t_count: n, pinned_t0: pt0, pinned_total: ptot },
                );
                (kn, vv, kns, vss, ki_, kg, 0)
            } else {
                m.get_mut(&(seq, family)).unwrap().t_count += n;
                (ptrs.0, ptrs.1, ptrs.2, ptrs.3, ptrs.4, ptrs.5, t0)
            }
        };
        // Write t0/total to pinned memory (graph-safe: kernels read zero-copy,
        // CPU writes before each replay)
        let (pinned_t0, pinned_total) = {
            let m = self.dsa_caches.lock().unwrap();
            let c = m.get(&(seq, family)).unwrap();
            unsafe {
                *c.pinned_t0 = t0 as i32;
                *c.pinned_total = (t0 + n) as i32;
            }
            (c.pinned_t0 as *const i32, c.pinned_total as *const i32)
        };
        ck(
            unsafe {
                ferrite_dsa_cache_append(
                    kvb.as_const_f32(), ki.as_const_f32(), gate.as_const_f32(),
                    k_nope_dev as *mut f32, v_dev as *mut f32, k_idx_dev as *mut f32, k_gate_dev as *mut f32,
                    pinned_t0, ni, h as i32, dk as i32, dv as i32, idm as i32, self.stream,
                )
            },
            "dsa_cache_append",
        )?;
        let total = t0 + n;

        // 8. kpool compression: pool_keys [npools, idm]
        // max_npools for graph safety: the grid is sized for the MAX
        // possible pools; the kernel derives the ACTUAL npools from the
        // pinned total (a frozen grid with actual npools would miss pools
        // as the context grows).
        let max_npools = (8192 + kpool - 1) / kpool; // max_tokens / kpool
        let npools = (total + kpool - 1) / kpool;
        let pool_keys = DevBuf::alloc(self.dev, self.stream, max_npools * idm)?;
        let dape = self.dev_weight(w.ape)?;
        ck(
            unsafe {
                ferrite_kpool_compress(
                    k_idx_dev as *const f32, k_gate_dev as *const f32, dape.as_const_f32(), pool_keys.as_f32(),
                    pinned_total, max_npools as i32, kpool as i32, idm as i32, self.stream,
                )
            },
            "dsa_kpool",
        )?;

        // 9. indexer topk over pools — GRAPH-SAFE select_k: pass the CONSTANT
        // topk/kpool (select_k_max); the kernel derives the LIVE select_k =
        // min(select_k_max, npools) from the pinned total at replay time. The
        // OLD code froze min(topk/kpool, npools_at_capture) as a kernel arg —
        // the verify graph's attention then saw only capture-time pools while
        // the draft (live select_k) saw the growing cache → d1≠a0 → MTP
        // accept collapse (the root cause of both the 500-step decay and the
        // ZERO_H2D early collapse).
        let select_k_max = w.topk / kpool;
        let idx_pools = DevBuf::alloc(self.dev, self.stream, n * select_k_max)?;
        let ctx0 = total - n;
        // graph-safe: pass pinned total_ptr instead of frozen npools/ctx0 values
        ck(
            unsafe {
                ferrite_indexer_topk(
                    qi.as_const_f32(), pool_keys.as_const_f32(), w_idx.as_const_f32(),
                    idx_pools.as_f32(), ni, ih as i32, idm as i32,
                    select_k_max as i32, pinned_total, kpool as i32, ni, self.stream,
                )
            },
            "dsa_topk",
        )?;

        // 10. expand pools to token indices [n, out_width_max] — the kernel
        // derives the live select_k/out_width from the pinned total; the
        // buffer stride is frozen at the max (graph-safe), the -1 tail slots
        // beyond each row's live out_width are masked by the attn's j<0 check.
        let out_width = select_k_max * kpool + (kpool - 1);
        let idx = DevBuf::alloc(self.dev, self.stream, n * out_width)?;
        ck(
            unsafe {
                ferrite_pool_expand(
                    idx_pools.as_const_f32(), idx.as_f32(),
                    ni, select_k_max as i32, kpool as i32, max_npools as i32, pinned_total,
                    ni, self.stream,
                )
            },
            "dsa_pool_expand",
        )?;;

        // 11. sparse attention: q [n,h,dk] × k [T,h,dk] × v [T,h,dv] → out [n, h*dv]
        // v2: 256-thread block (v1 was 32 — one warp over topk≈8K slots with
        // serial scalar dots + O(topk²) global idx rereads for dedup).
        let attn_out = DevBuf::alloc(self.dev, self.stream, n * h * dv)?;
        // split-K (SGLang MLA decode num_splits): grid = n*h*splits. At TP8
        // the old grid (n,h) was 8 blocks on 148 SM (36KB smem → 2 SM).
        let splits = (256 / (n * h).max(1)).clamp(1, 32);
        let sk_scratch = DevBuf::alloc(self.dev, self.stream, n * h * splits * (2 + dv))?;
        ck(
            unsafe {
                ferrite_sparse_attn_v2(
                    qb.as_const_f32(), k_nope_dev as *const f32, v_dev as *const f32, idx.as_const_f32(),
                    attn_out.as_f32(), sk_scratch.as_f32(), ni, pinned_total, h as i32, dk as i32, dv as i32,
                    out_width as i32, splits as i32, self.stream,
                )
            },
            "dsa_sparse_attn",
        )?;

        // 12. o_proj — TP partial [n, hidden]
        let partial = self.matmul_dev(&attn_out, w.o_proj, ni, (h * dv) as i32, hidden as i32)?;
        Ok(partial)
    }

    /// Debug getter: the family's current t_count (host bookkeeping).
    pub fn dsa_t_count(&self, seq: u64, family: usize) -> Option<usize> {
        let m = self.dsa_caches.lock().unwrap();
        m.get(&(seq, family)).map(|c| c.t_count)
    }

    /// Debug getter: (pinned_t0, pinned_total, t_count) — the values the
    /// captured kernels read zero-copy (pinned) vs the host bookkeeping.
    pub fn dsa_pinned(&self, seq: u64, family: usize) -> Option<(i32, i32, usize)> {
        let m = self.dsa_caches.lock().unwrap();
        m.get(&(seq, family)).map(|c| unsafe {
            (*c.pinned_t0, *c.pinned_total, c.t_count)
        })
    }

    // ============================================================
    // BATCHED decode layers (n = B rows from B DIFFERENT seqs): the GEMM
    // projections run at n=B (weights read ONCE per step — the user's
    // batched-GEMM directive), while the per-seq recurrent state ops
    // (GDN conv/state, DSA cache) run as B × n=1 kernel launches with
    // each row's own (seq, layer/family) state pointers. Inside a CUDA
    // graph the B launches replay at ~µs node cost — the small per-seq
    // kernels are independent (different states) and the HBM-bound
    // projections (95% of step time) amortize across all B rows.
    // ============================================================

    /// Batched GDN layer: x [B, hidden] → partial [B, hidden]. The
    /// projections (qkv/b/fa/ga/fb/gb/o_proj) are n=B GEMMs; the conv FIR
    /// + gated-deltanet core run per-seq (row r against (seqs[r], layer)'s
    /// conv/gdn state — B × n=1 launches with row-slice pointers).
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_layer_dev_batched(
        &self,
        x: &DevBuf,
        w: &GdnLayerWeights,
        seqs: &[u64],
        layer: usize,
        n: usize,
        hidden: usize,
        h: usize,
        dk: usize,
        lb: f32,
        rms_eps: f32,
        conv_size: usize,
    ) -> Result<DevBuf> {
        self.enter();
        // ALIGN (2026-09-08): n==1 must take the SAME fused path as the
        // single-seq chain. gdn_layer_dev uses gemv_tri_dev (b/fa/ga 3-in-1)
        // and gemv_qkv_conv (the conv FIR + silu + window slide INLINED in
        // the qkv GEMV epilogue) at n=1; this batched path always ran the
        // unfused matmul×N + conv1d_batched — measured +8.5ms/step of fixed
        // cost across the 34 GDN layers (batched graph replayed 17.95ms vs
        // the mega graph's 9.46ms at B=1; 19.48ms at B=2 — B-independent).
        if n == 1 {
            return self.gdn_layer_dev(
                x, w, seqs[0], layer, 1, hidden, h, dk, lb, rms_eps, conv_size, None,
            );
        }
        let proj = h * dk;
        let ni = n as i32;
        let ch = 3 * proj;
        let hist = conv_size.saturating_sub(1).max(1);
        // 1. projections — n=B GEMMs (the batched-GEMM directive: weights
        // stream once for all B rows; per-row accumulation is the tiled
        // GEMM's, matching the prefill's numeric domain).
        let qkv = self.matmul_dev(x, w.qkv_proj, ni, hidden as i32, (3 * proj) as i32)?;
        let b_raw = self.matmul_dev(x, w.b_proj, ni, hidden as i32, h as i32)?;
        let fa = self.matmul_dev(x, w.f_a, ni, hidden as i32, dk as i32)?;
        let ga = self.matmul_dev(x, w.g_a, ni, hidden as i32, dk as i32)?;
        let fb = self.matmul_dev(&fa, w.f_b, ni, dk as i32, proj as i32)?;
        let gb = self.matmul_dev(&ga, w.g_b, ni, dk as i32, proj as i32)?;
        let dw_conv = self.dev_weight(w.conv_w)?;
        // 2. per-seq causal conv — BATCHED: ONE launch, B×ch threads (each
        // (seq, channel): the 3-tap FIR + slide vs state_ptrs[seq]'s slice —
        // the ptr table cached per (layer, seq-set)). The per-seq loop's B
        // small launches (grid(ch), 1 live thread/block at n=1) serialize
        // per seq; the batched grid fills the SMs. FIR accumulation order =
        // conv1d_kernel's sequential i (bit-equal per seq).
        let conv_out = DevBuf::alloc(self.dev, self.stream, n * ch)?;
        let (conv_tbl, gdn_tbl) = self.gdn_state_tables(layer, seqs, ch * hist, h * dk * dk)?;
        ck(
            unsafe {
                ferrite_conv1d_batched(
                    qkv.as_const_f32(), dw_conv.as_const_f32(),
                    conv_tbl, conv_out.as_f32(),
                    n as i32, ch as i32, conv_size as i32, self.stream,
                )
            },
            "conv1d_batched",
        )?;
        // 3. gdn_prep n=B (row-independent: silu + split + L2 + beta + gate)
        let q = DevBuf::alloc(self.dev, self.stream, n * proj)?;
        let k = DevBuf::alloc(self.dev, self.stream, n * proj)?;
        let v = DevBuf::alloc(self.dev, self.stream, n * proj)?;
        let beta = DevBuf::alloc(self.dev, self.stream, n * h)?;
        let gate = DevBuf::alloc(self.dev, self.stream, n * proj)?;
        let dw_dt = self.dev_weight(w.dt_bias)?;
        let dw_al = self.dev_weight(w.a_log)?;
        ck(
            unsafe {
                ferrite_gdn_prep(
                    conv_out.as_const_f32(), b_raw.as_const_f32(), fb.as_const_f32(),
                    dw_dt.as_const_f32(), dw_al.as_const_f32(),
                    q.as_f32(), k.as_f32(), v.as_f32(), beta.as_f32(), gate.as_f32(),
                    ni, h as i32, dk as i32, lb, self.stream,
                )
            },
            "gdn_prep_batched",
        )?;
        // 4. per-seq gated-deltanet core — BATCHED: ONE launch, grid (B, h)
        // (the gdn_step_v2 body per (seq, head) with state_ptrs[seq] — the B
        // per-seq grid(1,h) launches (43% SM each) serialize; the batched
        // grid(B,h) fills the SMs). The 5-phase accumulation is bit-equal
        // per seq (the same kernel body, the same per-(seq,head) indexing).
        let core = DevBuf::alloc(self.dev, self.stream, n * proj)?;
        ck(
            unsafe {
                ferrite_gdn_chunk_batched(
                    q.as_const_f32(), k.as_const_f32(), v.as_const_f32(),
                    beta.as_const_f32(), gate.as_const_f32(), dw_al.as_const_f32(),
                    gdn_tbl, core.as_f32(),
                    n as i32, h as i32, dk as i32, dk as i32, self.stream,
                )
            },
            "gdn_chunk_batched",
        )?;
        // 5. gated rmsnorm n=B (row-independent) + o_proj GEMM n=B
        let o_norm_w = self.dev_weight(w.o_norm)?;
        let normed = DevBuf::alloc(self.dev, self.stream, n * proj)?;
        ck(
            unsafe {
                ferrite_gated_rmsnorm(
                    core.as_const_f32(), gb.as_const_f32(), o_norm_w.as_const_f32(),
                    normed.as_f32(), (n * h) as i32, dk as i32, rms_eps, self.stream,
                )
            },
            "gdn_norm_batched",
        )?;
        let partial = self.matmul_dev(&normed, w.o_proj, ni, proj as i32, hidden as i32)?;
        Ok(partial)
    }

    /// Batched DSA layer: x [B, hidden] → partial [B, hidden]. Projections
    /// are n=B GEMMs; the cache append + kpool + indexer topk + pool expand
    /// + sparse attention run per-seq (row r against (seqs[r], family)'s
    /// DSA cache at its own t0 — B × n=1 launches). The per-(seq, family)
    /// host bookkeeping (t_count += 1, pinned t0/total write) runs here —
    /// same as dsa_layer_dev, per row.
    #[allow(clippy::too_many_arguments)]
    pub fn dsa_layer_dev_batched(
        &self,
        x: &DevBuf,
        w: &DsaLayerWeights,
        seqs: &[u64],
        family: usize,
        n: usize,
        hidden: usize,
    ) -> Result<DevBuf> {
        self.enter();
        // ALIGN (2026-09-08): n==1 must take the SAME path as the single-seq
        // chain. Measured [megab-timing] dsa11=46.7ms vs the mega chain's
        // dsa11=2.2-5.5ms — a 10-20x gap that IS the batched graph's 8.5ms
        // fixed cost at B=1 (the GDN n==1 alignment only recovered 0.33ms).
        if n == 1 {
            return self.dsa_layer_dev(x, w, seqs[0], family, 1, hidden);
        }
        let ni = n as i32;
        let (h, dk, dv, ih, idm, kpool) = (w.h, w.dk, w.dv, w.ih, w.idm, w.kpool);
        // 1-6. projections + layernorm + scale — n=B GEMMs (row-independent
        // norms). The kvb/ki/gate rows feed the per-seq cache appends below.
        let qa = self.matmul_dev(x, w.q_a, ni, hidden as i32, (w.q_a.shape.0[0]) as i32)?;
        let qa_ln = self.rmsnorm_dev(&qa, w.q_a_ln, w.rms_eps, n, w.q_a.shape.0[0])?;
        let qb = self.matmul_dev(&qa_ln, w.q_b, ni, w.q_a.shape.0[0] as i32, (h * dk) as i32)?;
        let latent = self.matmul_dev(x, w.kv_a, ni, hidden as i32, (w.kv_a.shape.0[0]) as i32)?;
        let kv_ln = self.rmsnorm_dev(&latent, w.kv_a_ln, w.rms_eps, n, w.kv_a.shape.0[0])?;
        let kvb = self.matmul_dev(&kv_ln, w.kv_b, ni, w.kv_a.shape.0[0] as i32, (h * (dk + dv)) as i32)?;
        let qi = self.matmul_dev(&qa_ln, w.wq_b, ni, w.q_a.shape.0[0] as i32, (ih * idm) as i32)?;
        let ki_raw = self.matmul_dev(x, w.wk, ni, hidden as i32, idm as i32)?;
        let w_idx = self.matmul_dev(x, w.weights_proj, ni, hidden as i32, ih as i32)?;
        let gate = self.matmul_dev(x, w.gate, ni, hidden as i32, idm as i32)?;
        let ki = DevBuf::alloc(self.dev, self.stream, n * idm)?;
        let knw = self.dev_weight(w.k_norm_w)?;
        let knb = self.dev_weight(w.k_norm_b)?;
        ck(
            unsafe { ferrite_layernorm_affine(ki_raw.as_const_f32(), knw.as_const_f32(), knb.as_const_f32(), ki.as_f32(), ni, idm as i32, self.stream) },
            "dsa_layernorm_batched",
        )?;
        ck(
            unsafe { ferrite_scale_inplace(w_idx.as_f32(), (ih as f32).sqrt().recip(), (n * ih) as i32, self.stream) },
            "dsa_widx_scale_batched",
        )?;

        // Per-seq HOST bookkeeping ONLY (get-or-create the (seq, family)
        // caches, advance t_count, write the pinned t0/total ints the batched
        // kernels read zero-copy via the [B] pointer tables). The returned
        // raw pointers are unused here — the tables reach them by (family,
        // seq-set).
        let max_npools = (8192 + kpool - 1) / kpool;
        let attn_out = DevBuf::alloc(self.dev, self.stream, n * h * dv)?;
        let dape = self.dev_weight(w.ape)?;
        for &seq_r in seqs {
            if seq_r == u64::MAX {
                continue; // padded row → the dummy table slot serves it
            }
            let _ = {
                let mut m = self.dsa_caches.lock().unwrap();
                let existing = m
                    .get(&(seq_r, family))
                    .filter(|c| !c.k_nope.is_null())
                    .map(|c| (c.k_nope, c.v, c.k_nope_scale, c.v_scale, c.k_idx, c.k_gate, c.pinned_t0, c.pinned_total, c.t_count));
                match existing {
                    Some((kn, vv, kns, vss, ki_, kg, pt0, ptot, t0)) => {
                        m.get_mut(&(seq_r, family)).unwrap().t_count += 1;
                        unsafe {
                            *pt0 = t0 as i32;
                            *ptot = (t0 + 1) as i32;
                        }
                        (kn, vv, kns, vss, ki_, kg, pt0 as *const i32, ptot as *const i32, t0 + 1)
                    }
                    None => {
                        let max_tokens = std::env::var("FERRITE_DSA_MAXT")
                    .ok()
                    .and_then(|v| v.parse::<usize>().ok())
                    .unwrap_or(8192usize);
                        let kn = self.dsa_alloc(max_tokens * h * dk)?;
                        let vv = self.dsa_alloc(max_tokens * h * dv)?;
                        let kns = self.dsa_alloc(max_tokens * h)?;
                        let vss = self.dsa_alloc(max_tokens * h)?;
                        let ki_ = self.dsa_alloc(max_tokens * idm)?;
                        let kg = self.dsa_alloc(max_tokens * idm)?;
                        let mut pt0: *mut i32 = std::ptr::null_mut();
                        let mut ptot: *mut i32 = std::ptr::null_mut();
                        ck(unsafe { cudaMallocHost(&mut pt0 as *mut *mut i32 as *mut *mut std::ffi::c_void, 4) }, "pinned t0")?;
                        ck(unsafe { cudaMallocHost(&mut ptot as *mut *mut i32 as *mut *mut std::ffi::c_void, 4) }, "pinned total")?;
                        unsafe { *pt0 = 0; *ptot = 1; }
                        m.insert(
                            (seq_r, family),
                            DsaCacheState { k_nope: kn, v: vv, k_nope_scale: kns, v_scale: vss, k_idx: ki_, k_gate: kg, max_tokens, t_count: 1, pinned_t0: pt0, pinned_total: ptot },
                        );
                        (kn, vv, kns, vss, ki_, kg, pt0 as *const i32, ptot as *const i32, 1)
                    }
                }
            };
        }
        // FERRITE_LAYER_SUM inside-DSA probe (2026-09-09: the batched DSA
        // attention is exactly-zero at every family — discriminate "dead
        // projections" vs "dead pinned totals"). Host side only, never inside
        // capture, rank 0 only.
        if !self.capturing()
            && crate::shard_idx() == 0
            && std::env::var_os("FERRITE_LAYER_SUM").is_some()
        {
            let mut desc = String::new();
            {
                let m = self.dsa_caches.lock().unwrap();
                for &seq_r in seqs {
                    if seq_r == u64::MAX {
                        continue;
                    }
                    if let Some(c) = m.get(&(seq_r, family)) {
                        unsafe {
                            desc.push_str(&format!(
                                " seq{:x}:t0={}/tot={}/tc={}",
                                seq_r as u32, *c.pinned_t0, *c.pinned_total, c.t_count
                            ));
                        }
                    }
                }
            }
            let mx = |b: &DevBuf| -> String {
                let mut h = vec![0f32; b.len];
                if b.download(&mut h).is_err() {
                    return "dl_err".into();
                }
                format!("{:.4}", h.iter().fold(0f32, |a, v| a.max(v.abs())))
            };
            eprintln!(
                "[dsap] fam{family} n={n}{desc} qb_mx={} kvb_mx={} qi_mx={}",
                mx(&qb),
                mx(&kvb),
                mx(&qi)
            );
        }
        // the (family, seq-set) batched pointer tables (cached per composition)
        let tbl = self.dsa_ptr_tables(family, seqs, h, dk, dv, idm)?;
        // 7. cache append — ONE launch: all B rows → each seq's cache at its
        // own t0 (the per-seq t0s via the pinned-ptr table, zero-copy).
        ck(
            unsafe {
                ferrite_dsa_append_batched(
                    kvb.as_const_f32(), ki.as_const_f32(), gate.as_const_f32(),
                    tbl.kn as *const *mut f32, tbl.v as *const *mut f32,
                    tbl.kidx as *const *mut f32, tbl.kgate as *const *mut f32,
                    tbl.t0p as *const *const i32,
                    // NTOK MUST BE 1 (ROOT CAUSE #3 of the B=16 crashes,
                    // 2026-09-09): the kernel grid is (B, ntok) and it reads
                    // `kvb + (seq + tok) * row` — the batched kvb is [B, row]
                    // with ONE decode token per seq, so any ntok > 1 reads
                    // rows seq+tok >= B (up to 3.75MB past the buffer at B=16
                    // → the probabilistic 2MB-aligned Xid-31 faults; at B<=8
                    // the overshoot stayed inside pool slack and went
                    // unnoticed) and scribbles garbage cache slots beyond
                    // total (never read, so outputs were bit-identical).
                    // The single-seq path passes ni legitimately there
                    // (kvb = [n tokens of ONE seq]); this batched layout must
                    // not copy that.
                    ni, h as i32, dk as i32, dv as i32, idm as i32, 1, self.stream,
                )
            },
            "dsa_append_batched",
        )?;
        // 8. kpool compression — ONE launch: per-seq pools from each seq's
        // k_idx/k_gate (npools derived live from each seq's pinned total).
        let pool_keys = DevBuf::alloc(self.dev, self.stream, n * max_npools * idm)?;
        ck(
            unsafe {
                ferrite_kpool_compress_batched(
                    tbl.kidx as *const *mut f32, tbl.kgate as *const *mut f32,
                    dape.as_const_f32(), pool_keys.as_f32(),
                    tbl.totp as *const *const i32,
                    ni, max_npools as i32, kpool as i32, idm as i32, self.stream,
                )
            },
            "dsa_kpool_batched",
        )?;
        // 9. indexer topk — grid(B), one 256-thread block per seq: its qi/w_idx
        // row vs its pool_keys row (select_k LIVE per seq from the pinned
        // total — not frozen at capture like the single-seq graph path).
        let select_k_max = w.topk / kpool;
        let idx_pools = DevBuf::alloc(self.dev, self.stream, n * select_k_max)?;
        ck(
            unsafe {
                ferrite_indexer_topk_batched(
                    qi.as_const_f32(), pool_keys.as_const_f32(), w_idx.as_const_f32(),
                    idx_pools.as_f32(), ni, ih as i32, idm as i32, select_k_max as i32,
                    kpool as i32, max_npools as i32, tbl.totp as *const *const i32, self.stream,
                )
            },
            "dsa_topk_batched",
        )?;
        // 10. expand pools → token indices (idx stride frozen at the cap —
        // the -1 tail slots beyond each seq's live out_width are masked
        // by the attention's j < 0 guard).
        let out_width = select_k_max * kpool + (kpool - 1);
        let idx = DevBuf::alloc(self.dev, self.stream, n * out_width)?;
        ck(
            unsafe {
                ferrite_pool_expand_batched(
                    idx_pools.as_const_f32(), idx.as_f32(),
                    ni, select_k_max as i32, kpool as i32, max_npools as i32,
                    tbl.totp as *const *const i32, 1, self.stream,
                )
            },
            "dsa_expand_batched",
        )?;
        // FERRITE_LAYER_SUM: kpool/topk/expand liveness — pool_keys real? idx
        // row0 valid pools or all -1?
        if !self.capturing()
            && crate::shard_idx() == 0
            && std::env::var_os("FERRITE_LAYER_SUM").is_some()
        {
            let mut pk = vec![0f32; pool_keys.len];
            let pk_mx = if pool_keys.download(&mut pk).is_ok() {
                format!("{:.4}", pk.iter().fold(0f32, |a, v| a.max(v.abs())))
            } else {
                "dl_err".into()
            };
            let mut ih = vec![0f32; idx.len];
            let idx_desc = if idx.download(&mut ih).is_ok() {
                let w = out_width.min(40).min(ih.len());
                format!("{:?}", ih[..w].iter().map(|v| *v as i32).collect::<Vec<_>>())
            } else {
                "dl_err".into()
            };
            eprintln!("[dsap] fam{family} poolk_mx={pk_mx} idx_r0={idx_desc}");
        }
        // 11. sparse attention — grid(B, h): each seq's qb row vs its own
        // k_nope/v cache over its idx row (per-seq total via pinned table).
        ck(
            unsafe {
                ferrite_sparse_attn_v2_batched(
                    qb.as_const_f32(), tbl.kn as *const *mut f32, tbl.v as *const *mut f32,
                    idx.as_const_f32(), attn_out.as_f32(), ni, tbl.totp as *const *const i32,
                    h as i32, dk as i32, dv as i32, out_width as i32, self.stream,
                )
            },
            "dsa_attn_batched",
        )?;
        // FERRITE_LAYER_SUM: the attention output liveness at the SOURCE
        // (before o_proj) — exact zeros here = the attention itself is dead.
        if !self.capturing()
            && crate::shard_idx() == 0
            && std::env::var_os("FERRITE_LAYER_SUM").is_some()
        {
            let mut h = vec![0f32; attn_out.len];
            if attn_out.download(&mut h).is_ok() {
                eprintln!(
                    "[dsap] fam{family} attn_out mx={:.4} nz={}",
                    h.iter().fold(0f32, |a, v| a.max(v.abs())),
                    h.iter().filter(|v| **v != 0f32).count()
                );
            }
        }
        // 12. o_proj GEMM n=B
        let partial = self.matmul_dev(&attn_out, w.o_proj, ni, (h * dv) as i32, hidden as i32)?;
        Ok(partial)
    }

    /// Debug getter: (k_nope device ptr, t_count) for a family's cache.
    pub fn mtp_family_cache(&self, seq: u64, family: usize) -> Result<(*mut std::ffi::c_void, usize)> {
        let m = self.dsa_caches.lock().unwrap();
        let c = m.get(&(seq, family))
            .ok_or_else(|| FerriteError::InvalidArg(format!("no dsa cache ({seq},{family})")))?;
        Ok((c.k_nope, c.t_count))
    }

    /// Mega-graph host-side DSA advance: write the pinned t0/total that the
    /// captured graph's kernels read zero-copy, and advance t_count — the
    /// same bookkeeping dsa_layer_dev's host logic does per call, minus the
    /// kernels (the graph executes those at replay). Call BEFORE every
    /// graph replay.
    pub fn dsa_host_advance(&self, seq: u64, family: usize, n: usize) {
        // HOST-WRITE ORDERING (2026-09-09 root cause): the pinned t0/total are
        // written by the HOST here and read zero-copy by the DSA kernels. Host
        // stores are NOT ordered w.r.t. the device's in-flight reads of the
        // same slot — if the PREVIOUS step's (or this step's earlier layer's)
        // kernels are still running they observe the new (larger) total and
        // index past the cache into an unmapped 2MB page (Xid 31 PDE fault).
        // Evidence: the batched chain runs clean whenever a per-layer sync is
        // present (FERRITE_MEGA_PROBE=1) and faults otherwise; a pre-step sync
        // alone is not enough because the writes happen per layer.
        // FERRITE_NO_ADV_SYNC=1 disables this for A/B.
        // ⚠️ CAPTURE-ILLEGAL when enabled: a stream sync inside the capture pass
        // returns err 900 and invalidates the capture (measured). Default OFF;
        // FERRITE_ADV_SYNC=1 opts in (only meaningful with FERRITE_MEGA_DRY=1).
        if std::env::var_os("FERRITE_ADV_SYNC").is_some() {
            let _ = self.sync();
        }
        let mut m = self.dsa_caches.lock().unwrap();
        if let Some(c) = m.get_mut(&(seq, family)) {
            let t0 = c.t_count;
            unsafe {
                *c.pinned_t0 = t0 as i32;
                *c.pinned_total = (t0 + n) as i32;
            }
            c.t_count += n;
        }
    }

    /// Mega-graph capture rollback: the capture pass ran dsa_layer_dev's
    /// host bookkeeping (t_count += n, pinned write) but its kernels were
    /// only RECORDED, not executed — undo the virtual advance so t_count
    /// equals the tokens actually in the cache, then advance before every
    /// replay keeps the invariant.
    pub fn dsa_host_rollback(&self, seq: u64, family: usize, n: usize) {
        let mut m = self.dsa_caches.lock().unwrap();
        if let Some(c) = m.get_mut(&(seq, family)) {
            c.t_count -= n;
        }
    }

    /// Free ONE sequence's per-seq GPU state (multi-seq serving lifecycle —
    /// finished/aborted requests release ~GBs of per-seq caches or the
    /// serve OOMs after a handful of requests). Runs on the engine thread
    /// (single writer — no replay can race the frees; the caller owns the
    /// schedule). Order matters: graph execs are destroyed BEFORE the
    /// buffers their recorded kernel params reference (DSA caches, pinned
    /// t0). The gdn{layer}/moe{layer} per-LAYER segment graphs are
    /// seq-independent shared assets and are NOT touched. MtpState is a
    /// per-rank singleton (MTP serving is single-seq); not touched here.
    ///
    /// GraphIO pins (x_stage/out_dev) are REMOVED from the map but NOT
    /// freed: x_stage is the input DevBuf's pinned staging — that DevBuf
    /// returns to the (per-thread) DevBuf pool at the step's end, and the
    /// pool re-dispenses the (ptr, stage) pair for the NEXT capture. A
    /// cudaFreeHost here corrupts the pool's next allocation (observed:
    /// the follow-up seq's mega capture failed with "memcpy H2D: invalid
    /// argument" — the H2D src was the freed stage). The pins leak
    /// ~64KB/seq — negligible vs the ~GBs of DSA caches freed here.
    pub fn free_seq(&self, seq: u64) -> Result<()> {
        // DIAGNOSTIC (FERRITE_NO_FREE=1): leak instead of freeing — isolates
        // whether the cudaFree itself is what poisons the next allocation.
        if std::env::var_os("FERRITE_NO_FREE").is_some() {
            return Ok(());
        }
        self.enter();
        // Pending stream work may still reference the seq's buffers (the
        // last decode's async kernels): sync before freeing. Best-effort —
        // a wedged stream (a failed capture can leave capture mode ON,
        // err 900 on every op) must not block the frees below.
        //
        // ROOT-CAUSE FIX (2026-09-09): a STREAM-only sync is not enough — the
        // engine's NCCL all-reduce runs on NCCL-internal streams and the
        // retire is issued from the HTTP driver thread while the engine
        // thread may still be enqueueing. Measured: `bench_scb.py 16 200`
        // (its warmup request retires first) faulted on the NEXT batched
        // step's first dry-run with a 2MB-aligned Xid 31 PDE fault (a freed
        // DSA cache base), while the same bench WITHOUT the warmup ran clean
        // (3039 tok, 0 faults). A device-wide sync closes that window.
        let dev_rc = unsafe { cudaDeviceSynchronize() };
        if dev_rc != 0 {
            eprintln!("[cluster] free_seq {seq}: pre-free device sync failed (err {dev_rc}) — retrying stream sync");
            if let Err(e) = self.sync() {
                eprintln!("[cluster] free_seq {seq}: pre-free stream sync failed ({e}) — freeing anyway");
            }
        }
        // 1. This seq's mega graphs (exact names — mega{seq}, mega_v{seq}):
        //    destroy the exec; drop the GraphIO MAP entries (the pins leak —
        //    see the doc comment).
        for name in [format!("mega{seq}"), format!("mega_v{seq}")] {
            if let Some(exec) = self.graph_execs.lock().unwrap().remove(&name) {
                unsafe { cudaGraphExecDestroy(exec as *mut std::ffi::c_void) };
            }
            self.graph_io.lock().unwrap().remove(&name);
        }
        // 2. DSA family caches: (seq, *) → 4 device buffers + 2 pinned ints.
        {
            let mut m = self.dsa_caches.lock().unwrap();
            let keys: Vec<usize> = m
                .keys()
                .filter(|(s, _)| *s == seq)
                .map(|(_, f)| *f)
                .collect();
            for f in keys {
                if let Some(c) = m.remove(&(seq, f)) {
                    // Pool the big device caches (do NOT cudaFree): reusing the
                    // same VA avoids the driver's free+realloc remap path that
                    // faulted the next batched step (2026-09-09).
                    self.dsa_release(c.k_nope);
                    self.dsa_release(c.v);
                    self.dsa_release(c.k_idx);
                    self.dsa_release(c.k_gate);
                    unsafe {
                        cudaFreeHost(c.pinned_t0 as *mut std::ffi::c_void);
                        cudaFreeHost(c.pinned_total as *mut std::ffi::c_void);
                    }
                }
            }
        }
        // 3. GDN/conv recurrent states: (seq, layer) → DeviceState.
        for store in [&self.gdn_states, &self.conv_states] {
            let mut m = store.lock().unwrap();
            let keys: Vec<usize> = m
                .keys()
                .filter(|(s, _)| *s == seq)
                .map(|(_, l)| *l)
                .collect();
            for l in keys {
                if let Some(st) = m.remove(&(seq, l)) {
                    // Pooled (see dsa_release) — no cudaFree/remap of the state.
                    self.dsa_release(st.ptr);
                }
            }
        }
        // 4/5. The batched GDN/DSA tables are now keyed by (layer|family,
        // SIZE) and shared across seq-sets — their device addresses must stay
        // stable for per-size graph reuse, and their content (the per-seq
        // state pointers) is refreshed on every call, so a freed seq's stale
        // pointer is overwritten before the next replay. Nothing to purge.
        Ok(())
    }

    fn dsa_alloc(&self, floats: usize) -> Result<*mut std::ffi::c_void> {
        // Reuse a pooled buffer of the same size (same VA as a previous seq's
        // cache) — avoids the driver's free+realloc remap path that faulted
        // the batched step after a retirement (2026-09-09).
        let reused = self.dsa_pool.lock().unwrap().get_mut(&floats).and_then(|v| v.pop());
        if let Some(p) = reused {
            ck(unsafe { cudaMemset(p, 0, floats * 4) }, "dsa cache zero (reused)")?;
            self.dsa_sizes.lock().unwrap().insert(p as usize, floats);
            return Ok(p);
        }
        let mut p: *mut std::ffi::c_void = std::ptr::null_mut();
        ck(unsafe { cudaMalloc(&mut p, floats * 4) }, "dsa cache malloc")?;
        ck(unsafe { cudaMemset(p, 0, floats * 4) }, "dsa cache zero")?;
        self.dsa_sizes.lock().unwrap().insert(p as usize, floats);
        Ok(p)
    }

    /// Return a dsa_alloc buffer to the size pool instead of cudaFree'ing it.
    fn dsa_release(&self, p: *mut std::ffi::c_void) {
        if p.is_null() {
            return;
        }
        if let Some(floats) = self.dsa_sizes.lock().unwrap().remove(&(p as usize)) {
            self.dsa_pool.lock().unwrap().entry(floats).or_default().push(p);
        }
    }
}

// ============================================================
// TP all-reduce on device: sum N partial outputs in-place (graph-
// capturable, no H2D/D2H). For the decode-step device op chain —
// the fan-out produces world partial [n, hidden] DevBufs; this sums
// them into the first partial's buffer.
// ============================================================
extern "C" {
    fn ferrite_tp_all_reduce(partials: *const f32, out: *mut f32,
                               total: i32, world: i32, s: CuStream) -> i32;
    fn ferrite_moe_weighted_sum(probs: *const f32, eouts: *const f32,
                                  out: *mut f32, n: i32, topk: i32, hidden: i32,
                                  s: CuStream) -> i32;
    fn ferrite_moe_fused_act(x: *const f32, ids_f: *const f32,
                              gate_ptrs: *const *const std::ffi::c_void,
                              up_ptrs: *const *const std::ffi::c_void,
                              shared_gate: *const std::ffi::c_void,
                              shared_up: *const std::ffi::c_void,
                              act: *mut f32, expert_start: i32, e_local: i32,
                              hidden: i32, inter: i32, inter_shared: i32,
                              topk: i32, n: i32, limit: f32,
                              s: CuStream) -> i32;
    fn ferrite_moe_fused_down_sum(ids_f: *const f32, probs: *const f32,
                                   down_ptrs: *const *const std::ffi::c_void,
                                   shared_down: *const std::ffi::c_void,
                                   act: *const f32, out: *mut f32,
                                   expert_start: i32, e_local: i32,
                                   hidden: i32, inter: i32, inter_shared: i32,
                                   topk: i32, n: i32,
                                   s: CuStream) -> i32;
    fn ferrite_moe_fused_act_fp8(x: *const f32, ids_f: *const f32,
                                  gate_w8_ptrs: *const *const std::ffi::c_void,
                                  gate_scale_ptrs: *const *const std::ffi::c_void,
                                  up_w8_ptrs: *const *const std::ffi::c_void,
                                  up_scale_ptrs: *const *const std::ffi::c_void,
                                  shared_gate_w8: *const std::ffi::c_void,
                                  shared_gate_scale: *const std::ffi::c_void,
                                  shared_up_w8: *const std::ffi::c_void,
                                  shared_up_scale: *const std::ffi::c_void,
                                  act: *mut f32, expert_start: i32, e_local: i32,
                                  hidden: i32, inter: i32, inter_shared: i32,
                                  topk: i32, n: i32, limit: f32, hscols: i32,
                                  s: CuStream) -> i32;
    fn ferrite_moe_down_bf16_mma(
        ids_f: *const f32, probs: *const f32,
        down_w8_ptrs: *const *const std::ffi::c_void,
        down_scale_ptrs: *const *const f32,
        shared_down_w8: *const std::ffi::c_void, shared_down_scale: *const f32,
        act: *const f32, out: *mut f32,
        expert_start: i32, e_local: i32, hidden: i32, inter: i32,
        inter_shared: i32, topk: i32, dscols: i32, n: i32, s: CuStream) -> i32;
    fn ferrite_moe_fused_down_sum_fp8(ids_f: *const f32, probs: *const f32,
                                      down_w8_ptrs: *const *const std::ffi::c_void,
                                      down_scale_ptrs: *const *const std::ffi::c_void,
                                      shared_down_w8: *const std::ffi::c_void,
                                      shared_down_scale: *const std::ffi::c_void,
                                      act: *const f32, out: *mut f32,
                                      expert_start: i32, e_local: i32,
                                      hidden: i32, inter: i32, inter_shared: i32,
                                      topk: i32, n: i32, dscols: i32,
                                      s: CuStream) -> i32;
}

impl CudaBackend {
    /// Sum `world` partial [total] buffers (contiguous) into `out` — the
    /// GPU all-reduce for the TP decode-step device op chain. Graph-capturable.
    pub fn tp_all_reduce_dev(&self, partials: &DevBuf, out: &mut DevBuf, total: usize, world: usize) -> Result<()> {
        self.enter();
        ck(unsafe {
            ferrite_tp_all_reduce(partials.as_const_f32(), out.as_f32(),
                                   total as i32, world as i32, self.stream)
        }, "tp_all_reduce")
    }

    /// MoE weighted sum: out[t, h] = Σ_j probs[t, j] * eouts[t, j, h].
    /// Graph-capturable (replaces the CPU expert accumulation loop).
    pub fn moe_weighted_sum_dev(&self, probs: &DevBuf, eouts: &DevBuf, out: &mut DevBuf, n: usize, topk: usize, hidden: usize) -> Result<()> {
        self.enter();
        ck(unsafe {
            ferrite_moe_weighted_sum(probs.as_const_f32(), eouts.as_const_f32(),
                                      out.as_f32(), n as i32, topk as i32, hidden as i32, self.stream)
        }, "moe_weighted_sum")
    }
}

// ============================================================
// MoE layer device chain: routing + expert FFNs + weighted sum,
// all on device (zero H2D/D2H inside the layer). The caller
// (CUDA graph capture) feeds [n, hidden] DevBuf and gets the
// TP partial [n, hidden] DevBuf back.
// ============================================================

/// Weight set for one expert's device chain.
pub struct ExpertWeights<'a> {
    pub gate: &'a Tensor,
    pub up: &'a Tensor,
    pub down: &'a Tensor,
}

impl CudaBackend {
    /// Full MoE layer on device: x_dev [n, hidden] → partial [n, hidden].
    /// routing (moe_route) → top-k expert FFNs (matmul_dev + swiglu2_dev)
    /// → weighted sum → + shared expert. All DevBuf, graph-capturable.
    /// Lazily build (and cache) this layer's expert POINTER TABLE: three
    /// device buffers of e_local raw pointers into the dev_weight_bf16
    /// cache. The fused MoE kernels gather the selected experts' rows
    /// through these with GPU-side dispatch — zero host round-trips, zero
    /// duplicated weight memory. Keyed by the first expert's gate tensor
    /// pointer (stable per layer).
    pub fn moe_expert_ptrs(
        &self,
        experts: &[ExpertWeights],
    ) -> Result<(usize, *mut std::ffi::c_void, *mut std::ffi::c_void, *mut std::ffi::c_void)> {
        self.enter();
        let key = experts.first().map(|e| e.gate.as_slice().as_ptr() as usize).unwrap_or(0);
        let e_local = experts.len();
        {
            let m = self.moe_ptrs.lock().unwrap();
            if let Some(t) = m.get(&key) {
                return Ok((t.e_local, t.gate_dev, t.up_dev, t.down_dev));
            }
        }
        let mut gates: Vec<*mut std::ffi::c_void> = Vec::with_capacity(e_local);
        let mut ups: Vec<*mut std::ffi::c_void> = Vec::with_capacity(e_local);
        let mut downs: Vec<*mut std::ffi::c_void> = Vec::with_capacity(e_local);
        for e in experts {
            gates.push(self.dev_weight_bf16(e.gate)?.ptr);
            ups.push(self.dev_weight_bf16(e.up)?.ptr);
            downs.push(self.dev_weight_bf16(e.down)?.ptr);
        }
        let mk = |v: &[*mut std::ffi::c_void]| -> Result<*mut std::ffi::c_void> {
            let mut p: *mut std::ffi::c_void = std::ptr::null_mut();
            ck(unsafe { cudaMalloc(&mut p, v.len() * std::mem::size_of::<*mut std::ffi::c_void>()) }, "moe ptr table malloc")?;
            ck(unsafe {
                cudaMemcpy(p, v.as_ptr() as *const _, v.len() * std::mem::size_of::<*mut std::ffi::c_void>(), CUDA_MEMCPY_H2D)
            }, "moe ptr table H2D")?;
            Ok(p)
        };
        let (g, u, d) = (mk(&gates)?, mk(&ups)?, mk(&downs)?);
        self.moe_ptrs.lock().unwrap().insert(key, MoePtrTable { gate_dev: g, up_dev: u, down_dev: d, e_local });
        Ok((e_local, g, u, d))
    }

    /// fp8 variant of the expert pointer table: (w8 bytes, block scales)
    /// device tables per expert for gate/up/down — the fused MoE kernels'
    /// e4m3 indirect addressing. Returns None when ANY expert weight misses
    /// the fp8 registration (caller falls back to the bf16 tables — domain
    /// uniformity is per-LAYER, never mixed).
    pub fn moe_expert_ptrs_fp8(
        &self,
        experts: &[ExpertWeights],
    ) -> Result<Option<MoeFp8PtrTable>> {
        self.enter();
        let key = experts.first().map(|e| e.gate.as_slice().as_ptr() as usize).unwrap_or(0);
        let e_local = experts.len();
        {
            let m = self.moe_fp8_ptrs.lock().unwrap();
            if let Some(t) = m.get(&key) {
                return Ok(Some(MoeFp8PtrTable {
                    gate_w8: t.gate_w8, gate_scale: t.gate_scale,
                    up_w8: t.up_w8, up_scale: t.up_scale,
                    down_w8: t.down_w8, down_scale: t.down_scale,
                    e_local: t.e_local,
                }));
            }
        }
        // every expert weight must be fp8-registered (all-or-nothing)
        let mut gate_w8 = Vec::with_capacity(e_local);
        let mut gate_sc = Vec::with_capacity(e_local);
        let mut up_w8 = Vec::with_capacity(e_local);
        let mut up_sc = Vec::with_capacity(e_local);
        let mut down_w8 = Vec::with_capacity(e_local);
        let mut down_sc = Vec::with_capacity(e_local);
        for e in experts {
            let (g, u, d) = match (self.fp8_lookup(e.gate), self.fp8_lookup(e.up), self.fp8_lookup(e.down)) {
                (Some(g), Some(u), Some(d)) => (g, u, d),
                _ => return Ok(None), // not fully fp8 → bf16 tables (this layer)
            };
            gate_w8.push(g.w); gate_sc.push(g.scale);
            up_w8.push(u.w); up_sc.push(u.scale);
            down_w8.push(d.w); down_sc.push(d.scale);
        }
        // table upload macro (six tables, same pattern — the bf16 builder's
        // closure, macro'd for the fp8 pair tables)
        macro_rules! upload_table {
            ($vals:expr) => {{
                let mut p: *mut std::ffi::c_void = std::ptr::null_mut();
                ck(unsafe { cudaMalloc(&mut p, $vals.len() * std::mem::size_of::<*mut std::ffi::c_void>()) }, "moe fp8 table malloc")?;
                ck(unsafe {
                    cudaMemcpy(p, $vals.as_ptr() as *const _, $vals.len() * std::mem::size_of::<*mut std::ffi::c_void>(), CUDA_MEMCPY_H2D)
                }, "moe fp8 table H2D")?;
                p
            }};
        }
        let t = MoeFp8PtrTable {
            gate_w8: upload_table!(gate_w8), gate_scale: upload_table!(gate_sc),
            up_w8: upload_table!(up_w8), up_scale: upload_table!(up_sc),
            down_w8: upload_table!(down_w8), down_scale: upload_table!(down_sc),
            e_local,
        };
        let out = MoeFp8PtrTable {
            gate_w8: t.gate_w8, gate_scale: t.gate_scale,
            up_w8: t.up_w8, up_scale: t.up_scale,
            down_w8: t.down_w8, down_scale: t.down_scale,
            e_local: t.e_local,
        };
        self.moe_fp8_ptrs.lock().unwrap().insert(key, t);
        // DIAG (FERRITE_MOE_PTRDBG=1): dump the 6 pointer tables' contents and the
        // bump-arena ranges — a table pointer outside every arena range means the
        // kernel will dereference a stale/unmapped allocation base (Xid 31 PDE fault).
        if std::env::var_os("FERRITE_MOE_PTRDBG").is_some() {
            let dump = |p: *mut std::ffi::c_void, name: &str| {
                let mut h = vec![0usize; e_local];
                unsafe {
                    cudaMemcpy(h.as_mut_ptr() as *mut _, p, e_local * 8, 2 /* D2H */);
                }
                let mn = h.iter().min().copied().unwrap_or(0);
                let mx = h.iter().max().copied().unwrap_or(0);
                let zero = h.iter().filter(|&&v| v == 0).count();
                eprintln!(
                    "[ptrdbg] {name}: n={} min={mn:#x} max={mx:#x} zero={zero} h[0]={:#x} h[1]={:#x}",
                    h.len(), h[0], h[1]
                );
            };
            let (base, cap, used) = { self.bump.lock().unwrap().last().copied().unwrap_or((std::ptr::null_mut(), 0, 0)) };
            eprintln!(
                "[ptrdbg] bump last block: base={:#x} cap={:#x} used={:#x}  blocks={}",
                base as usize, cap, used, self.bump.lock().unwrap().len()
            );
            dump(out.down_w8, "down_w8");
            dump(out.down_scale, "down_scale");
            dump(out.gate_w8, "gate_w8");
            dump(out.up_w8, "up_w8");
            eprintln!("[ptrdbg] layer key={key:#x} e_local={e_local}");
        }
        Ok(Some(out))
    }

    pub fn moe_layer_dev(
        &self,
        x_dev: &DevBuf,
        gate_w: &Tensor,           // router [e, hidden]
        bias_w: &Tensor,            // router bias [e] (f32)
        shared: &ExpertWeights,     // shared expert
        experts: &[ExpertWeights],  // routed experts (this rank's slice)
        expert_start: usize,        // first expert id on this rank
        probs_out: &mut DevBuf,     // [n, topk] routing probabilities
        n: usize,
        hidden: usize,
        topk: usize,
        e_total: usize,
        routed_scaling: f32,
        swiglu_limit: f32,
    ) -> Result<DevBuf> {
        self.enter();
        let ni = n as i32;
        let hi = hidden as i32;

        // 1. routing + 2. moe_route: FUSED for n==1 (the router GEMV + the
        // sigmoid+topk+renorm in ONE kernel — the "last block" pattern saves
        // the separate moe_route's 12µs launch overhead × 44 layers = 0.53ms)
        let dprobs = DevBuf::alloc(self.dev, self.stream, n * topk)?;
        let dids = DevBuf::alloc(self.dev, self.stream, n * topk)?;
        let dbias = self.dev_weight(bias_w)?;
        if ni == 1 {
            // FUSED: router GEMV + route (the "last block" pattern — the 160
            // expert blocks compute the logits, the last block does the route)
            let dw_gate = self.dev_weight_bf16(gate_w)?;
            let dlogits = DevBuf::alloc(self.dev, self.stream, e_total)?;
            let dctr = DevBuf::alloc(self.dev, self.stream, 4)?;
            ck(unsafe { cudaMemsetAsync(dctr.as_f32() as *mut _, 0, 4, self.stream) }, "ctr zero")?;
            ck(unsafe {
                ferrite_router_gemm_route_fused(
                    x_dev.as_const_f32(), dw_gate.ptr as *const _, dbias.as_const_f32(),
                    dprobs.as_f32(), dids.as_f32(), dlogits.as_f32(), dctr.as_f32() as *mut u32,
                    e_total as i32, hi, topk as i32, routed_scaling, self.stream)
            }, "router_gemm_route_fused")?;
        } else {
            // n > 1 (prefill): the old path (matmul + moe_route)
            let logits = self.matmul_dev(x_dev, gate_w, ni, hi, e_total as i32)?;
            ck(unsafe {
                ferrite_moe_route(logits.as_const_f32(), dbias.as_const_f32(),
                                  dprobs.as_f32(), dids.as_f32(),
                                  ni, e_total as i32, topk as i32, routed_scaling, self.stream)
            }, "moe_route_fused")?;
        }

        // ---- FUSED PATH (TileRT ExpertSelect idea): GPU-side expert dispatch
        // via the pointer table — ids/probs NEVER cross to the host; two
        // kernels (act + down_sum) replace the per-expert kernel chains, the
        // D2D gather and the probs_ext upload. Now batch-capable: grid carries
        // the token dim (n==1 decode, n>1 chunked prefill).
        // (routing is handled by the conditional above: n==1 → fused
        // router_gemm_route_fused; n>1 → matmul + moe_route)
        {
            // dbias already allocated above the routing conditional
            if probs_out.len >= n * topk {
                let (dst, src) = (probs_out.as_f32(), dprobs.as_const_f32());
                ck(unsafe {
                    cudaMemcpyAsync(dst as *mut _, src as *const _, n * topk * 4, CUDA_MEMCPY_D2D, self.stream)
                }, "probs D2D (fused)")?;
            }
            // Routed experts keep the FULL inter; the shared expert's inter is
            // TP-sharded (moe_intermediate_size / world). The act buffer is
            // [n, topk*inter + inter_shared] (see the kernels' slot layout).
            let inter = experts.first()
                .map(|e| e.gate.shape.0[0])
                .unwrap_or(shared.gate.shape.0[0]) as i32;
            let inter_shared = shared.gate.shape.0[0] as i32;
            let act = DevBuf::alloc(self.dev, self.stream, n * (topk * inter as usize + inter_shared as usize))?;
            let out = DevBuf::alloc(self.dev, self.stream, n * hidden)?;
            // fp8 fused path: experts + shared ALL fp8-registered → e4m3
            // tables (HALF the bf16 tables' HBM bytes — the moe segment was
            // 4.5ms of the 21.7ms verify step, HBM-bound). All-or-nothing per
            // layer (domain uniformity: never mix fp8/bf16 experts); a miss on
            // any weight falls back to the bf16 tables below.
            let shared_fp8 = match (self.fp8_lookup(shared.gate), self.fp8_lookup(shared.up), self.fp8_lookup(shared.down)) {
                (Some(g), Some(u), Some(d)) if self.fp8_lookup(gate_w).is_none() => Some((g, u, d)),
                _ => None,
            };
            if let (Some(tbl), Some((sg, su, sd))) = (self.moe_expert_ptrs_fp8(experts)?, shared_fp8) {
                // W8A8 tensor-core act (v1-mode per-block quant + gate/up dual
                // mma on the SAME smem xq + swiglu epilogue): the W8A16 dequant
                // loop measured 0.94x bf16 (cvt overhead offset the fp8 HBM
                // savings). Misaligned shapes (inter%16, hidden%128) fall back
                // to the dequant act below (kept).
                let inter_max = inter.max(inter_shared);
                let act_mma = inter_max % 16 == 0 && hidden % 128 == 0;
                // v2 (pre-quantized) only for n>1: at n=1 the 864-block grid
                // shares one x quantize (redundant-but-free ~9µs) while v2 adds
                // a quant kernel launch + 4KB smem copy per layer (measured
                // 90.4 -> 84.8 tok/s — the launch tail dominates). At n=3 the
                // 2592-block grid re-quantized 864x/token — v2 nets -0.5ms/step.
                let act_mma_v2 = act_mma && n > 1;
                if act_mma_v2 {
                    // v2 QUANTIZE-ONCE: x -> xq e4m3 + xs, ONE launch per layer
                    // (per token; the act kernel re-quantized x[tok] 864x =
                    // grid(max_rows/16, topk+1, n) × per-block quant — the act
                    // kernel was quantize-BOUND at 46µs, not A-weight bound).
                    let xq = DevBuf::alloc(self.dev, self.stream, n * hi as usize / 4 + 1)?; // e4m3 bytes as f32 slots
                    let xs = DevBuf::alloc(self.dev, self.stream, n)?;
                    ck(
                        unsafe {
                            ferrite_quant_e4m3_tokens(
                                x_dev.as_const_f32(), xq.as_f32() as *mut u8, xs.as_f32(),
                                ni, hi, self.stream,
                            )
                        },
                        "quant_e4m3_tokens",
                    )?;
                    let r = unsafe {
                        ferrite_moe_fused_act_fp8_mma_v2(
                            x_dev.as_const_f32(), dids.as_const_f32(),
                            tbl.gate_w8 as *const *const _, tbl.gate_scale as *const *const _,
                            tbl.up_w8 as *const *const _, tbl.up_scale as *const *const _,
                            sg.w, sg.scale, su.w, su.scale,
                            act.as_f32(),
                            expert_start as i32, tbl.e_local as i32, hi, inter, inter_shared,
                            topk as i32, ni, swiglu_limit,
                            xq.as_const_f32() as *const u8, xs.as_const_f32(),
                            self.stream,
                        )
                    };
                    if r != 0 {
                        eprintln!("[opcheck] moe_fused_act_fp8_mma_v2 returned err {r} (n={ni}) — the v2 act kernel is the first failing launch");
                    }
                    if r == 0 {
                        {
                            static ONCE: std::sync::Once = std::sync::Once::new();
                            ONCE.call_once(|| {
                                eprintln!(
                                    "[moe-dbg] e_local={} expert_start={} hi(hidden)={} inter={} inter_shared={} topk={} n={} experts.len()={}",
                                    tbl.e_local, expert_start, hi, inter, inter_shared, topk, ni, experts.len()
                                );
                            });
                        }
                        let dscols = self.fp8_lookup(shared.down).map(|f| f.scols).unwrap_or((inter as usize).div_ceil(128) as i32);
                        // bf16 tensor-core down: verified correct (bench bad=0
                        // for n=1..16 with unique per-block scales; serve text
                        // is coherent) and slightly faster than the SIMT path.
                        // Set FERRITE_MOE_DOWN_MMA=0 to fall back.
                        let use_down_mma = std::env::var("FERRITE_MOE_DOWN_MMA")
                            .map(|v| v != "0").unwrap_or(true);
                        let down_mma = if !use_down_mma { 1 } else { unsafe {
                            ferrite_moe_down_bf16_mma(
                                dids.as_const_f32(), dprobs.as_const_f32(),
                                tbl.down_w8 as *const *const _, tbl.down_scale as *const *const _,
                                sd.w, sd.scale as *const f32,
                                act.as_const_f32(), out.as_f32(),
                                expert_start as i32, tbl.e_local as i32, hi, inter, inter_shared,
                                topk as i32, dscols, ni, self.stream,
                            )
                        } };
                        if down_mma != 0 {
                            ck(unsafe {
                                ferrite_moe_fused_down_sum_fp8(
                                    dids.as_const_f32(), dprobs.as_const_f32(),
                                    tbl.down_w8 as *const *const _, tbl.down_scale as *const *const _,
                                    sd.w, sd.scale,
                                    act.as_const_f32(), out.as_f32(),
                                    expert_start as i32, tbl.e_local as i32, hi, inter, inter_shared,
                                    topk as i32, ni, dscols, self.stream,
                                )
                            }, "moe_fused_down_sum_fp8")?;
                        }
                        return Ok(out);
                    }
                    // v2 unsupported (unaligned) — fall through to v1
                }
                if act_mma {
                    let r = unsafe {
                        ferrite_moe_fused_act_fp8_mma(
                            x_dev.as_const_f32(), dids.as_const_f32(),
                            tbl.gate_w8 as *const *const _, tbl.gate_scale as *const *const _,
                            tbl.up_w8 as *const *const _, tbl.up_scale as *const *const _,
                            sg.w, sg.scale, su.w, su.scale,
                            act.as_f32(),
                            expert_start as i32, tbl.e_local as i32, hi, inter, inter_shared,
                            topk as i32, ni, swiglu_limit, self.stream,
                        )
                    };
                    if r != 0 {
                        eprintln!("[opcheck] moe_fused_act_fp8_mma (v1) returned err {r} — the act kernel is the first failing launch");
                    }
                    if r == 0 {
                        let dscols = self.fp8_lookup(shared.down).map(|f| f.scols).unwrap_or((inter as usize).div_ceil(128) as i32);
                        // WIP: the tensor-core down currently produces wrong
                        // values (serve text degraded to "!!!!"); opt in
                        // explicitly until the fragment/scale bug is fixed.
                        let use_down_mma = std::env::var("FERRITE_MOE_DOWN_MMA")
                            .map(|v| v == "1").unwrap_or(false);
                        let down_mma = if !use_down_mma { 1 } else { unsafe {
                            ferrite_moe_down_bf16_mma(
                                dids.as_const_f32(), dprobs.as_const_f32(),
                                tbl.down_w8 as *const *const _, tbl.down_scale as *const *const _,
                                sd.w, sd.scale as *const f32,
                                act.as_const_f32(), out.as_f32(),
                                expert_start as i32, tbl.e_local as i32, hi, inter, inter_shared,
                                topk as i32, dscols, ni, self.stream,
                            )
                        } };
                        if down_mma != 0 {
                            ck(unsafe {
                                ferrite_moe_fused_down_sum_fp8(
                                    dids.as_const_f32(), dprobs.as_const_f32(),
                                    tbl.down_w8 as *const *const _, tbl.down_scale as *const *const _,
                                    sd.w, sd.scale,
                                    act.as_const_f32(), out.as_f32(),
                                    expert_start as i32, tbl.e_local as i32, hi, inter, inter_shared,
                                    topk as i32, ni, dscols, self.stream,
                                )
                            }, "moe_fused_down_sum_fp8")?;
                        }
                        return Ok(out);
                    }
                    // act_mma not supported (unaligned) — fall through to dequant
                }
                let hscols = self.fp8_lookup(shared.gate).map(|f| f.scols).unwrap_or((hidden as usize).div_ceil(128) as i32);
                ck(unsafe {
                    ferrite_moe_fused_act_fp8(
                        x_dev.as_const_f32(), dids.as_const_f32(),
                        tbl.gate_w8 as *const *const _, tbl.gate_scale as *const *const _,
                        tbl.up_w8 as *const *const _, tbl.up_scale as *const *const _,
                        sg.w, sg.scale, su.w, su.scale,
                        act.as_f32(),
                        expert_start as i32, tbl.e_local as i32, hi, inter, inter_shared,
                        topk as i32, ni, swiglu_limit, hscols,
                        self.stream,
                    )
                }, "moe_fused_act_fp8")?;
                // down's scale cols = inter/128 (down weights are [hidden, inter])
                let dscols = self.fp8_lookup(shared.down).map(|f| f.scols).unwrap_or((inter as usize).div_ceil(128) as i32);
                ck(unsafe {
                    ferrite_moe_fused_down_sum_fp8(
                        dids.as_const_f32(), dprobs.as_const_f32(),
                        tbl.down_w8 as *const *const _, tbl.down_scale as *const *const _,
                        sd.w, sd.scale,
                        act.as_const_f32(), out.as_f32(),
                        expert_start as i32, tbl.e_local as i32, hi, inter, inter_shared,
                        topk as i32, ni, dscols, self.stream,
                    )
                }, "moe_fused_down_sum_fp8")?;
                return Ok(out);
            }
            // bf16 fused path (experts' ptr tables into the dev_weight_bf16 cache)
            let (e_local, g_ptrs, u_ptrs, d_ptrs) = self.moe_expert_ptrs(experts)?;
            let dsg = self.dev_weight_bf16(shared.gate)?;
            let dsu = self.dev_weight_bf16(shared.up)?;
            let dsd = self.dev_weight_bf16(shared.down)?;
            ck(unsafe {
                ferrite_moe_fused_act(
                    x_dev.as_const_f32(), dids.as_const_f32(),
                    g_ptrs as *const *const _, u_ptrs as *const *const _,
                    dsg.ptr, dsu.ptr, act.as_f32(),
                    expert_start as i32, e_local as i32, hi, inter, inter_shared,
                    topk as i32, ni, swiglu_limit, self.stream,
                )
            }, "moe_fused_act")?;
            ck(unsafe {
                ferrite_moe_fused_down_sum(
                    dids.as_const_f32(), dprobs.as_const_f32(),
                    d_ptrs as *const *const _, dsd.ptr,
                    act.as_const_f32(), out.as_f32(),
                    expert_start as i32, e_local as i32, hi, inter, inter_shared,
                    topk as i32, ni, self.stream,
                )
            }, "moe_fused_down_sum")?;
            return Ok(out);
        }

        ck(unsafe {
            let dbias = self.dev_weight(bias_w)?;
            // NOTE: n>1 (prefill) — the fused path returned earlier; this is
            // the fallback (non-fused MoE). The logits come from the matmul
            // in the n>1 branch of the routing conditional above.
            let logits = self.matmul_dev(x_dev, gate_w, ni, hi, e_total as i32)?;
            ferrite_moe_route(logits.as_const_f32(), dbias.as_const_f32(),
                              dprobs.as_f32(), dids.as_f32(),
                              ni, e_total as i32, topk as i32, routed_scaling, self.stream)
        }, "moe_route_dev")?;

        // 3. shared expert: x → gate/up/swiglu/down → shared_out [n, hidden]
        let shared_gate = self.matmul_dev(x_dev, shared.gate, ni, hi, shared.gate.shape.0[0] as i32)?;
        let shared_up = self.matmul_dev(x_dev, shared.up, ni, hi, shared.up.shape.0[0] as i32)?;
        let shared_inter = shared.gate.shape.0[0] as i32; // gate/up have same inter
        let shared_act = self.swiglu2_dev(&shared_gate, &shared_up, ni, shared_inter, swiglu_limit)?;
        let shared_out = self.matmul_dev(&shared_act, shared.down, ni, shared_inter, hi)?;

        // 4. routed experts — the ONLY CPU↔GPU boundary in the layer:
        //    download ids+probs (small: n×topk), CPU dispatches expert FFN
        //    chains (device-resident, no syncs inside), D2D-copies each
        //    output into the gather buffer. For CUDA graph capture this
        //    becomes a static all-experts run + GPU-side gather instead.
        let mut ids_host = vec![0f32; n * topk];
        dids.download(&mut ids_host)?;
        let mut probs_host = vec![0f32; n * topk];
        dprobs.download(&mut probs_host)?;
        if probs_out.len >= n * topk {
            let (dst, src) = (probs_out.as_f32(), dprobs.as_const_f32());
            ck(unsafe {
                cudaMemcpyAsync(dst as *mut _, src as *const _, n * topk * 4, CUDA_MEMCPY_D2D, self.stream)
            }, "probs D2D")?;
        }

        // 5. gather buffer [n, (topk+1) * hidden]: slots 0..topk-1 = routed
        //    expert outputs, slot topk = shared output. probs_ext = [probs, 1.0]
        //    so ONE weighted_sum call folds the shared expert in.
        //    Zero-fill upfront: experts NOT owned by this rank (TP shard) leave
        //    their slots zero — the all-reduce sums partials across ranks.
        let slots = topk + 1;
        let mut eouts = DevBuf::alloc(self.dev, self.stream, n * slots * hidden)?;
        ck(unsafe { cudaMemsetAsync(eouts.as_f32() as *mut _, 0, n * slots * hidden * 4, self.stream) }, "eouts zero")?;
        let mut probs_ext = vec![0f32; n * slots];
        probs_ext[..n * topk].copy_from_slice(&probs_host);
        for t in 0..n {
            probs_ext[t * slots + topk] = 1.0; // shared weight
        }
        let dprobs_ext = DevBuf::alloc(self.dev, self.stream, n * slots)?;
        dprobs_ext.upload(&probs_ext)?;

        // 6. per-token expert dispatch: for decode (n=1) exactly `topk`
        //    expert chains; each = 3 matmuls + swiglu2 (no H2D/D2H inside).
        //    Experts not owned by this rank (TP shard) leave their slots ZERO
        //    (upfront memset above) — the all-reduce sums partials across ranks.
        let e_count = experts.len();
        for t in 0..n {
            for j in 0..topk {
                let slot = t * slots + j;
                let eid = ids_host[t * topk + j] as usize;
                let local = eid.saturating_sub(expert_start);
                if local >= e_count {
                    continue; // another rank owns this expert → zero slot
                }
                let w = &experts[local];
                let inter = w.gate.shape.0[0] as i32;
                let g = self.matmul_dev(x_dev, w.gate, ni, hi, inter)?;
                let u = self.matmul_dev(x_dev, w.up, ni, hi, inter)?;
                let a = self.swiglu2_dev(&g, &u, ni, inter, swiglu_limit)?;
                let d = self.matmul_dev(&a, w.down, ni, inter, hi)?;
                // D2D gather into slot (graph-capturable, no host round-trip)
                ck(unsafe {
                    cudaMemcpyAsync(
                        (eouts.as_f32() as *mut std::ffi::c_void).add(slot * hidden * 4),
                        d.as_const_f32() as *const std::ffi::c_void,
                        hidden * 4, // one token's row (decode: n==1 contiguous)
                        CUDA_MEMCPY_D2D, self.stream,
                    )
                }, "expert out gather")?;
            }
        }
        // shared expert output → slot topk (same layout)
        for t in 0..n {
            let slot = t * slots + topk;
            ck(unsafe {
                cudaMemcpyAsync(
                    (eouts.as_f32() as *mut std::ffi::c_void).add(slot * hidden * 4),
                    (shared_out.as_const_f32() as *const std::ffi::c_void).add(t * hidden * 4),
                    hidden * 4, CUDA_MEMCPY_D2D, self.stream,
                )
            }, "shared out gather")?;
        }

        // 7. ONE weighted-sum kernel: out[t] = Σ_j probs_ext[t,j]·eouts[t,j] + shared
        let mut out = DevBuf::alloc(self.dev, self.stream, n * hidden)?;
        self.moe_weighted_sum_dev(&dprobs_ext, &eouts, &mut out, n, slots, hidden)?;
        let _ = dprobs;
        Ok(out)
    }
}

// ============================================================
// MHC hyper-connections + rmsnorm at the DevBuf level — the layer-chain
// components for the full decode-step device op chain (graph capture):
// hc_pre_dev → rmsnorm_dev → gdn_layer_dev / moe_layer_dev →
// tp_all_reduce_dev → hc_post_dev, ALL DevBuf (zero host round-trips).
// The kernels are the same ferrite_hc_pre/hc_post/rmsnorm the Tensor-level
// path uses — these wrappers just keep activations on device.
// ============================================================
impl CudaBackend {
    /// MHC pre-step on device: residual_flat [s, n*h] → (li [s,h],
    /// post [s,n], comb [s,n,n]). Weights (fn_w/scale/base) hit the
    /// dev_weight cache (f32, resident after preload). DevBuf in/out —
    /// the CPU sinkhorn/dot loops this replaces took ~0.9ms/layer.
    #[allow(clippy::too_many_arguments)]
    pub fn hc_pre_dev(
        &self,
        res: &DevBuf,
        fn_w: &Tensor,
        scale: &Tensor,
        base: &Tensor,
        norm_w: &Tensor,
        s: usize,
        nh: usize,
        rms_eps: f32,
        hc_eps: f32,
        sinkhorn_iters: usize,
    ) -> Result<(DevBuf, DevBuf, DevBuf)> {
        self.enter();
        let mix = fn_w.shape.0[0];
        let n = ((-2.0 + (4.0 + 4.0 * mix as f64).sqrt()) / 2.0) as usize;
        let h = nh / n;
        let dfw = self.dev_weight(fn_w)?;
        let dsc = self.dev_weight(scale)?;
        let dba = self.dev_weight(base)?;
        let dnw = self.dev_weight(norm_w)?;
        // hc-pre input dump (first call per process): the mmap path's hc_pre
        // output explodes (hn0 ±41056 vs expected ~O(1)) despite all weights
        // verifying equal — dump the ACTUAL device values of fw/scale/base/nw
        // to find which input is wrong. Expected (checkpoint): fw ~O(1),
        // scale [0.06-0.09], base ~-7.6, nw ~1.45, res ~0.001.
        if HC_DBG_ONCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 1 {
            let rd = |p: *const std::ffi::c_void, cnt: usize, tag: &str| unsafe {
                let mut buf = vec![0f32; cnt];
                cudaMemcpy(buf.as_mut_ptr() as *mut _, p, cnt * 4, 2 /* D2H */);
                eprintln!("[hc-dbg] {tag}: {:?} (ptr={:p})", &buf, p);
            };
            let _ = self.sync();
            rd(dfw.ptr as *const _, 4, "fw[24,16384] widen");
            rd(dsc.ptr as *const _, 3, "scale[3]");
            rd(dba.ptr as *const _, 4, "base[24]");
            rd(dnw.ptr as *const _, 4, "nw[4096]");
            rd(res.as_const_f32() as *const _, 4, "res");
        }
        let li = DevBuf::alloc(self.dev, self.stream, s * h)?;
        let post = DevBuf::alloc(self.dev, self.stream, s * n)?;
        let comb = DevBuf::alloc(self.dev, self.stream, s * n * n)?;
        // K-split partials: s * mix * HC_MIX_KS(8) lanes + the Σx² fusion tail
        // [s][KS] (the mix_split m==0 lanes ride free on their existing x
        // reads — the restA kernel's P1 reduces these instead of re-reading
        // the full nh, ~15µs × 90/step) + old ctr [s] + pre_s [s][n] (restA
        // writes, p345's 16 blocks read) + P345: p4 partials [s][16] + ctr [s].
        // Plan N v1: P3/P4/P5 moved off the single-block rest kernel onto a
        // grid(s,16) multi-block launch (the 1-SM 64KB x read was the
        // 40µs/layer serve-real bottleneck).
        // HC_P345_NB must match the kernel's #define (16 — see the .cu;
        // NB>16 corrupts the output, bisected 2026-09-08).
        let mx_scratch = DevBuf::alloc(self.dev, self.stream, s * (mix * 16 + 16 + 1 + n + 16 + 1))?;
        // SPLIT version (grid(s, mix) — one block per mix row): the
        // single-block kernel ran on ONE SM (~6GB/s of 8TB/s HBM); the mix
        // GEMV (16384×18432) was 57% of the decode step (A_hc+C_hc 24ms of
        // 42ms). The original err-900 capture concern (mx_scratch pool
        // class not warm in capture thread) is gone under the mega-graph:
        // the dry-run warms the pool on the SAME worker that captures.
        // FUSED rmsnorm tail in phase-2 rest kernel (nw): li comes out
        // normalized (input_layernorm) — callers no longer launch a
        // standalone rmsnorm after this.
        ck(
            unsafe {
                ferrite_hc_pre_split(
                    res.as_const_f32(), dfw.as_const_f32(), dsc.as_const_f32(), dba.as_const_f32(),
                    dnw.as_const_f32(),
                    li.as_f32(), post.as_f32(), comb.as_f32(),
                    mx_scratch.as_f32(),
                    s as i32, n as i32, h as i32, mix as i32,
                    rms_eps, hc_eps, sinkhorn_iters as i32, self.stream,
                )
            },
            "hc_pre_dev",
        )?;
        Ok((li, post, comb))
    }

    /// MHC post-step on device: x [s, h_out] + residual [s, n, h] →
    /// out [s, n, h] (per DevBuf — the all-reduce partial feeds straight
    /// in, the next hc_pre's residual feeds straight out).
    pub fn hc_post_dev(
        &self,
        x: &DevBuf,
        res: &DevBuf,
        post: &DevBuf,
        comb: &DevBuf,
        s: usize,
        n: usize,
        h: usize,
    ) -> Result<DevBuf> {
        self.enter();
        let out = DevBuf::alloc(self.dev, self.stream, s * n * h)?;
        ck(
            unsafe {
                ferrite_hc_post(
                    x.as_const_f32(), res.as_const_f32(), post.as_const_f32(), comb.as_const_f32(),
                    out.as_f32(), s as i32, n as i32, h as i32, self.stream,
                )
            },
            "hc_post_dev",
        )?;
        Ok(out)
    }

    /// RMSNorm on device: x [n, dim] → out (weight resident f32).
    /// DevBuf in/out — the layer chain's input_layernorm/post_attention
    /// layernorm without a host round-trip.
    pub fn rmsnorm_dev(&self, x: &DevBuf, w: &Tensor, eps: f32, n: usize, dim: usize) -> Result<DevBuf> {
        self.enter();
        let dw = self.dev_weight(w)?;
        let out = DevBuf::alloc(self.dev, self.stream, x.len)?;
        ck(
            unsafe { ferrite_rmsnorm(x.as_const_f32(), dw.as_const_f32(), out.as_f32(), n as i32, dim as i32, eps, self.stream) },
            "rmsnorm_dev",
        )?;
        Ok(out)
    }

    /// hc_contract on device: x [s, n*h] → out [s, h] — mean over the n MHC
    /// flows (mirror of mhc::hc_expand). The mega-graph's head-chain bridge:
    /// last layer's residual → contract → rmsnorm → lm_head, zero host
    /// crossings.
    pub fn hc_contract_dev(&self, x: &DevBuf, s: usize, n: usize, h: usize) -> Result<DevBuf> {
        self.enter();
        let out = DevBuf::alloc(self.dev, self.stream, s * h)?;
        ck(
            unsafe { ferrite_hc_contract(x.as_const_f32(), out.as_f32(), s as i32, n as i32, h as i32, self.stream) },
            "hc_contract_dev",
        )?;
        Ok(out)
    }

    /// Fused multi-GEMV (same input x, up to 5 weight matrices) — ONE launch
    /// for decode chains that project x through several matrices (gdn: qkv/b/
    /// fa/ga share x; dsa: qa/latent/ki/w_idx/gate share x). Kills N-1 kernel
    /// launches + their tail latencies (~10-15us each on B300). w5=None →
    /// of5=0 (4-matrix case). Returns the output DevBufs in order.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv5_dev(
        &self,
        x: &DevBuf,
        w1: &Tensor, w2: &Tensor, w3: &Tensor, w4: &Tensor,
        w5: Option<&Tensor>,
        in_f: i32,
        of1: i32, of2: i32, of3: i32, of4: i32,
    ) -> Result<(DevBuf, DevBuf, DevBuf, DevBuf, Option<DevBuf>)> {
        self.enter();
        let d1 = self.dev_weight_bf16(w1)?;
        let d2 = self.dev_weight_bf16(w2)?;
        let d3 = self.dev_weight_bf16(w3)?;
        let d4 = self.dev_weight_bf16(w4)?;
        let o1 = DevBuf::alloc(self.dev, self.stream, of1 as usize)?;
        let o2 = DevBuf::alloc(self.dev, self.stream, of2 as usize)?;
        let o3 = DevBuf::alloc(self.dev, self.stream, of3 as usize)?;
        let o4 = DevBuf::alloc(self.dev, self.stream, of4 as usize)?;
        let of5 = w5.map(|t| t.shape.0[0] as i32).unwrap_or(0);
        let d5: *const std::ffi::c_void = match w5 {
            Some(w5t) => self.dev_weight_bf16(w5t)?.ptr,
            None => std::ptr::null(),
        };
        let o5 = match w5 {
            Some(w5t) => DevBuf::alloc(self.dev, self.stream, w5t.numel())?,
            None => DevBuf::alloc(self.dev, self.stream, 1)?, // unused 4B slot
        };
        ck(
            unsafe {
                ferrite_gemv5_bf16(
                    x.as_const_f32(),
                    d1.ptr, d2.ptr, d3.ptr, d4.ptr, d5,
                    o1.as_f32(), o2.as_f32(), o3.as_f32(), o4.as_f32(),
                    if w5.is_some() { o5.as_f32() } else { std::ptr::null_mut() },
                    in_f, of1, of2, of3, of4, of5,
                    self.stream,
                )
            },
            "gemv5_dev",
        )?;
        Ok((o1, o2, o3, o4, if w5.is_some() { Some(o5) } else { None }))
    }
}

// ============================================================
// Per-layer-segment CUDA graphs (FERRITE_GRAPH_LAYER): each segment's
// op sequence (upload memcpy + kernels) is captured ONCE and replayed
// per token. The pool is per-device (fan_out ranks don't share) and each
// rank's op sequence is deterministic → buffer addresses are stable
// across tokens. The segment's INPUT staging and OUTPUT device buffers
// are registered as GraphIO and LEAKED (never returned to the pool —
// replay writes them; pool reuse would corrupt).
// ============================================================
/// Fixed IO pointers of a captured segment graph: the CPU writes the
/// input into `x_stage` (pinned, the recorded memcpy's source), launches
/// the graph, then downloads `out_dev`.
pub struct GraphIO {
    pub x_stage: *mut std::ffi::c_void,
    pub x_len: usize,
    pub out_dev: *mut std::ffi::c_void,
    pub out_len: usize,
    /// The graph's INPUT DEVICE buffer (res_dev — the stage→dev memcpy's
    /// destination). Present when captured with a DevBuf input (mega graphs);
    /// the device-embed replay path (graph_run_dev_input) writes it via
    /// embed_expand_dev instead of the host staging (12B of token ids vs
    /// n*mult*hidden f32 of host embed+hc_expand+staging — ~1ms host per
    /// MTP verify step).
    pub in_dev: *mut std::ffi::c_void,
    /// Input row count n (the embed kernel's grid) + hidden/mult for
    /// embed_expand_dev; 0 = no device input path (host staging only).
    pub in_n: usize,
    pub in_hidden: usize,
    pub in_mult: usize,
}
unsafe impl Send for GraphIO {}
unsafe impl Sync for GraphIO {}
impl Clone for GraphIO {
    fn clone(&self) -> Self {
        *self
    }
}
impl Copy for GraphIO {}

impl CudaBackend {
    pub fn graph_io_put(&self, name: &str, io: GraphIO) {
        self.graph_io.lock().unwrap().insert(name.to_string(), io);
    }
    pub fn graph_io_get(&self, name: &str) -> Option<GraphIO> {
        self.graph_io.lock().unwrap().get(name).cloned()
    }
    /// True while THIS thread is inside a stream capture. Debug/probe paths
    /// that D2H or sync MUST check this — a cudaMemcpy during capture is
    /// err 901 (invalidated), which then poisons the whole capture pass.
    pub fn capturing(&self) -> bool {
        is_capturing()
    }
    /// Whether a named graph exec exists (the draft graphs' replay-path
    /// probe — graph_replay would LAUNCH it, so existence needs its own check).
    pub fn graph_exists(&self, name: &str) -> bool {
        self.graph_execs.lock().unwrap().contains_key(name)
    }
    /// Destroy a named graph (exec + IO entry) — the batched decode's
    /// composition lifecycle: the "megab_{seqs}" graph's recorded kernel args
    /// embed per-seq state pointers; a seq's retirement frees those states,
    /// so the graph MUST be destroyed before the freed pointers replay.
    /// The next decode_step_batched recaptures for the new composition.
    pub fn graph_destroy(&self, name: &str) {
        if let Some(exec) = self.graph_execs.lock().unwrap().remove(name) {
            unsafe { cudaGraphExecDestroy(exec as *mut std::ffi::c_void) };
        }
        self.graph_io.lock().unwrap().remove(name);
    }
    /// Device-input graph replay: the capture recorded embed_expand_dev as
    /// the graph's FIRST node (the input is n×4B TOKEN IDS in a device
    /// buffer, not the n*mult*hidden f32 staging). Host writes the ids
    /// (12B H2D at n=3) → replay → out D2H. Replaces the host embed
    /// lookup + hc_expand + 576KB pinned staging write (~1ms of the MTP
    /// verify step's host budget).
    pub fn graph_run_ids(
        &self,
        name: &str,
        ids: &[u32],
        ids_dev: *mut i32,
        out: &mut [f32],
    ) -> Result<bool> {
        let Some(io) = self.graph_io_get(name) else { return Ok(false); };
        if io.in_n == 0 || ids.len() != io.in_n || out.len() != io.out_len {
            return Err(FerriteError::InvalidArg(format!(
                "graph_run_ids {name}: ids {} != in_n {} or out {} != {} (device-input path not captured)",
                ids.len(), io.in_n, out.len(), io.out_len
            )));
        }
        self.enter();
        let idata: Vec<i32> = ids.iter().map(|&t| t as i32).collect();
        ck(
            unsafe {
                cudaMemcpyAsync(
                    ids_dev as *mut std::ffi::c_void,
                    idata.as_ptr() as *const std::ffi::c_void,
                    idata.len() * 4,
                    CUDA_MEMCPY_H2D,
                    self.stream,
                )
            },
            "ids H2D",
        )?;
        if !self.graph_replay(name) {
            return Ok(false);
        }
        ck(
            unsafe {
                cudaMemcpyAsync(
                    out.as_mut_ptr() as *mut std::ffi::c_void,
                    io.out_dev,
                    out.len() * 4,
                    CUDA_MEMCPY_D2H,
                    self.stream,
                )
            },
            "graph_run_ids D2H",
        )?;
        self.sync()?;
        Ok(true)
    }

    /// Replay a segment graph with fresh input: write `input` to the
    /// captured staging, launch, download the output. (capture never
    /// executes — this is the steady-state path)
    pub fn graph_run(&self, name: &str, input: &[f32], out: &mut [f32]) -> Result<bool> {
        let Some(io) = self.graph_io_get(name) else { return Ok(false); };
        if input.len() != io.x_len || out.len() != io.out_len {
            return Err(FerriteError::InvalidArg(format!(
                "graph_run {name}: input {} != stage {} or out {} != {}",
                input.len(), io.x_len, out.len(), io.out_len
            )));
        }
        unsafe {
            std::ptr::copy_nonoverlapping(input.as_ptr(), io.x_stage as *mut f32, io.x_len);
        }
        if !self.graph_replay(name) {
            return Ok(false);
        }
        self.enter();
        ck(
            unsafe {
                cudaMemcpyAsync(
                    out.as_mut_ptr() as *mut std::ffi::c_void,
                    io.out_dev,
                    io.out_len * 4,
                    CUDA_MEMCPY_D2H,
                    self.stream,
                )
            },
            "graph_run D2H",
        )?;
        self.sync()?;
        Ok(true)
    }
}

/// H2D copy for small i32 arrays (the zero-H2D path's token slots — 4B
/// writes to the device-resident tokens_dev buffer; NOT a general-purpose
/// H2D, just the initial token + d1/d2 writes that the accept kernel
/// would otherwise write on device).
pub fn memcpy_htod_i32(dst: *mut i32, src: *const i32, n: usize, s: CuStream) -> i32 {
    unsafe {
        cudaMemcpyAsync(
            dst as *mut std::ffi::c_void,
            src as *const std::ffi::c_void,
            n * 4,
            CUDA_MEMCPY_H2D,
            s,
        )
    }
}

/// Blocking device→host copy (raw pointers — used by the graph capture
/// path where the DevBuf was forgotten but its address is registered).
pub fn memcpy_d2h_sync(src: *mut std::ffi::c_void, dst: *mut f32, floats: usize, s: CuStream) -> i32 {
    unsafe { cudaMemcpyAsync(dst as *mut _, src, floats * 4, CUDA_MEMCPY_D2H, s) };
    unsafe { cudaStreamSynchronize(s) }
}

/// Global serialization for graph capture (experiment): fan_out's 4 rank
/// workers capture CONCURRENTLY and cuGraphInstantiate crashed inside
/// libcuda (gdb: SIGSEGV in cuGraphInstantiate from worker #2+). Capture
/// is a one-time cost per segment — serializing the capture passes (not
/// the replays) costs nothing steady-state.
pub fn capture_lock() -> &'static std::sync::Mutex<()> {
    static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
    LOCK.get_or_init(|| std::sync::Mutex::new(()))
}

impl CudaBackend {
    /// Pin 4 bytes of host memory (zero-copy kernel read slot — the MTP
    /// commit kernel's k). cudaMallocHost'd; freed with cudaFreeHost when
    /// the backend drops (pool-free by design: one slot per seq lifetime).
    pub fn pinned_i32(&self) -> Result<*mut i32> {
        let mut p: *mut i32 = std::ptr::null_mut();
        ck(
            unsafe { cudaMallocHost(&mut p as *mut *mut i32 as *mut *mut std::ffi::c_void, 4) },
            "pinned_i32",
        )?;
        unsafe { *p = 0 };
        Ok(p)
    }

    /// Pin n i32 slots (zero-copy kernel read — the embed_expand token ids:
    /// the mega-graph's first node is the embed_expand kernel reading these;
    /// replay writes 1-3 i32 (12B) instead of the 48KB f32 staging upload).
    fn pinned_ids(&self, n: usize) -> Result<*mut i32> {
        if let Some(&p) = self.pinned_ids_cache.lock().unwrap().get(&n) {
            return Ok(p);
        }
        let mut p: *mut i32 = std::ptr::null_mut();
        ck(
            unsafe { cudaMallocHost(&mut p as *mut *mut i32 as *mut *mut std::ffi::c_void, n * 4) },
            "pinned_ids",
        )?;
        for i in 0..n {
            unsafe { *p.add(i) = 0 };
        }
        self.pinned_ids_cache.lock().unwrap().insert(n, p);
        Ok(p)
    }

    /// Write token ids into the graph's pinned id slot (host, 4B/token —
    /// the 12B replay input that replaces the 48KB f32 staging write).
    pub fn pinned_ids_write(&self, n: usize, ids: &[u32]) -> Result<()> {
        let p = self.pinned_ids(n)?;
        for (i, &t) in ids.iter().enumerate() {
            unsafe { *p.add(i) = t as i32 };
        }
        Ok(())
    }

    /// Pinned id slot pointer (for the embed_expand kernel's capture-time
    /// parameter — the graph records this address; replay writes through
    /// pinned_ids_write).
    pub fn pinned_ids_ptr(&self, n: usize) -> Result<*mut i32> {
        self.pinned_ids(n)
    }

    /// Argmax over the last dim of a device buffer [n, dim] → out [n].
    /// Device-to-device (the Tensor-level path downloaded the full logits
    /// row — 620KB for GLM's 154880 vocab).
    pub fn argmax_dev(&self, logits: &DevBuf, out: &mut DevBuf, n: usize, dim: usize) -> Result<()> {
        self.enter();
        ck(
            unsafe { ferrite_argmax(logits.as_const_f32(), out.as_f32(), n as i32, dim as i32, self.stream) },
            "argmax_dev",
        )?;
        Ok(())
    }
}

// ============================================================
// P2P all-reduce via NVLink (B300 GPU4-7 = NV18): rank 0 collects
// the other ranks' partials with cudaMemcpyPeerAsync, then the
// existing tp_all_reduce kernel sums on-device. Replaces the host
// download→CPU-sum→re-upload round-trip per attention/ffn segment.
// ============================================================
extern "C" {
    fn ferrite_p2p_copy(dst: *mut f32, dst_dev: i32, src: *const f32, src_dev: i32,
                         count: usize, s: CuStream) -> i32;
    fn ferrite_p2p_enable(dev: i32, peer: i32) -> i32;
}

impl CudaBackend {
    /// Enable P2P access between this device and `peer` (NVLink).
    pub fn p2p_enable(&self, peer: i32) -> Result<()> {
        self.enter();
        ck(unsafe { ferrite_p2p_enable(self.dev, peer) }, "p2p_enable")?;
        Ok(())
    }

    /// P2P all-reduce: collect `partials` (device pointers from each rank)
    /// into a contiguous buffer on THIS device, then sum with the
    /// tp_all_reduce kernel. All pointers must be [n] floats.
    pub fn p2p_all_reduce(
        &self,
        partial_ptrs: &[usize],  // device pointers, index = rank
        n: usize,
    ) -> Result<DevBuf> {
        self.enter();
        let world = partial_ptrs.len();
        if world <= 1 {
            let out = DevBuf::alloc(self.dev, self.stream, n)?;
            ck(unsafe {
                cudaMemcpyAsync(out.as_f32() as *mut _, partial_ptrs[0] as *const _,
                                 n * 4, CUDA_MEMCPY_D2D, self.stream)
            }, "p2p single copy")?;
            return Ok(out);
        }
        // staging: [world, n] contiguous on this device — LEAKED (not returned
        // to the pool): the pool would reuse this memory for the NEXT op's
        // allocation while the GPU is still executing the tp_all_reduce
        // kernel that reads from it (async race). Per-token leak is
        // world * n * 4 bytes = 4 × 4096 × 4 = 64KB → acceptable.
        let mut staging = DevBuf::alloc(self.dev, self.stream, world * n)?;
        for (rank, &ptr) in partial_ptrs.iter().enumerate() {
            if rank as i32 == self.dev {
                // same device: plain D2D copy
                ck(unsafe {
                    cudaMemcpyAsync(
                        (staging.as_f32() as *mut std::ffi::c_void).add(rank * n * 4),
                        ptr as *const std::ffi::c_void,
                        n * 4, CUDA_MEMCPY_D2D, self.stream)
                }, "p2p local copy")?;
            } else {
                ck(unsafe {
                    ferrite_p2p_copy(
                        staging.as_f32().add(rank * n),
                        self.dev,
                        ptr as *const f32,
                        rank as i32,
                        n, self.stream)
                }, "p2p peer copy")?;
            }
        }
        // sum on this device
        let out = DevBuf::alloc(self.dev, self.stream, n)?;
        let mut out_mut = out;
        self.tp_all_reduce_dev(&staging, &mut out_mut, n, world)?;
        // CRITICAL: sync before returning — staging goes back to the pool on
        // drop, and the NEXT allocation would reuse its memory while the
        // GPU is still reading it (tp_all_reduce kernel is async).
        self.sync()?;
        Ok(out_mut)
    }

    /// P2P one-shot all-reduce micro-bench path (TileRT
    /// ExpertDownAllReduce mode): down kernel writes this rank's partial
    /// into EVERY rank's staging row via UVA peer writes (NVLink) and
    /// raises its ready flag; sum kernel spins all world flags then sums
    /// the local staging rows. staging_tbl/ready_tbl are device arrays of
    /// world device pointers (peer bases), ctr is this rank's block
    /// counter, staging_local is [world][n] rows, ready_local is [world].
    pub fn p2p_ar_oneshot_dev(&self, partial: &DevBuf, staging_tbl: &DevBuf,
                              ready_tbl: &DevBuf, ctr: &DevBuf,
                              staging_local: &DevBuf, ready_local: &DevBuf,
                              out: &mut DevBuf, n: usize, world: usize,
                              my_rank: usize) -> Result<()> {
        self.enter();
        ck(unsafe {
            ferrite_p2p_ar_oneshot(
                partial.as_const_f32(),
                staging_tbl.as_f32() as *const *mut f32,
                ready_tbl.as_f32() as *const *mut u32,
                ctr.as_f32() as *mut u32,
                staging_local.as_const_f32(),
                ready_local.as_const_f32() as *const u32,
                out.as_f32(), n as i32, world as i32, my_rank as i32,
                self.stream)
        }, "p2p_ar_oneshot")?;
        Ok(())
    }

    /// Phase 1: allocate this rank's P2P AR v2 buffers (the [2][world][max_n]
    /// ping-pong staging + the [world] epoch flags + the epoch/ctr counters,
    /// all zeroed). Returns the (staging, ready) UVA addresses for the peers'
    /// pointer tables. The OnceLock holds the state — the cluster setup is
    /// the only caller (before any decode).
    pub fn p2p_ar_alloc(&self, world: usize, max_n: usize) -> Result<(usize, usize)> {
        self.enter();
        let st_len = 2 * world * max_n; // [2][world][max_n] f32 (ping-pong)
        let mut st: *mut std::ffi::c_void = std::ptr::null_mut();
        ck(unsafe { cudaMalloc(&mut st, st_len * 4) }, "p2p_ar staging malloc")?;
        ck(unsafe { cudaMemset(st, 0, st_len * 4) }, "p2p_ar staging zero")?;
        let mut rd: *mut std::ffi::c_void = std::ptr::null_mut();
        ck(unsafe { cudaMalloc(&mut rd, world * 4) }, "p2p_ar flags malloc")?;
        ck(unsafe { cudaMemset(rd, 0, world * 4) }, "p2p_ar flags zero")?;
        let mut sn: *mut std::ffi::c_void = std::ptr::null_mut();
        // [MAX_FBLOCKS][world]: every finish-kernel block polls with its OWN
        // seen row. With a single shared row, block A updated seen[tr] and
        // block B then read prev == cur == the new stamp → waited forever
        // (measured: prev=2 cur=2 myepoch=1, 18 stalls per step).
        const MAX_FBLOCKS: usize = 256;
        ck(unsafe { cudaMalloc(&mut sn, MAX_FBLOCKS * world * 4) }, "p2p_ar seen malloc")?;
        ck(unsafe { cudaMemset(sn, 0, MAX_FBLOCKS * world * 4) }, "p2p_ar seen zero")?;
        let mut ep: *mut std::ffi::c_void = std::ptr::null_mut();
        ck(unsafe { cudaMalloc(&mut ep, 4) }, "p2p_ar epoch malloc")?;
        ck(unsafe { cudaMemset(ep, 0, 4) }, "p2p_ar epoch zero")?;
        let mut ct: *mut std::ffi::c_void = std::ptr::null_mut();
        ck(unsafe { cudaMalloc(&mut ct, 4) }, "p2p_ar ctr malloc")?;
        ck(unsafe { cudaMemset(ct, 0, 4) }, "p2p_ar ctr zero")?;
        let _ = self.p2p_ar.lock().unwrap().replace(P2pArState {
            staging_local: st,
            ready_local: rd,
            seen: sn,
            epoch: ep,
            ctr: ct,
            staging_tbl: std::ptr::null_mut(),
            ready_tbl: std::ptr::null_mut(),
            world,
            max_n,
        });
        Ok((st as usize, rd as usize))
    }

    /// Phase 2: upload this rank's [world] pointer tables (the peers'
    /// staging/ready UVA bases — same process, peer access enabled).
    pub fn p2p_ar_tables(&self, staging_addrs: &[usize], ready_addrs: &[usize]) -> Result<()> {
        let mut m = self.p2p_ar.lock().unwrap();
        let Some(st) = m.as_mut() else {
            return Err(FerriteError::InvalidArg("p2p_ar_tables before alloc".into()));
        };
        self.enter();
        let mk = |addrs: &[usize]| -> Result<*mut std::ffi::c_void> {
            let mut p: *mut std::ffi::c_void = std::ptr::null_mut();
            ck(unsafe { cudaMalloc(&mut p, addrs.len() * 8) }, "p2p_ar tbl malloc")?;
            ck(unsafe { cudaMemcpy(p, addrs.as_ptr() as *const _, addrs.len() * 8, CUDA_MEMCPY_H2D) }, "p2p_ar tbl H2D")?;
            Ok(p)
        };
        st.staging_tbl = mk(staging_addrs)?;
        st.ready_tbl = mk(ready_addrs)?;
        Ok(())
    }

        /// Reset the P2P AR protocol state (epoch, ready flags, ctr). The dry-run
    /// advances the epoch but the capture pass does NOT (it only records), so
    /// ranks leave the capture with DIFFERENT epoch counters — the next
    /// replay's flag waits then mismatch and deadlock (dev0 at L0, peers at
    /// L35). Call once after the dry-run/before capture on EVERY rank so all
    /// replays start from epoch 0.
    pub fn p2p_ar_reset(&self) -> Result<()> {
        let st = {
            let m = self.p2p_ar.lock().unwrap();
            match m.as_ref() {
                Some(st) => *st,
                None => return Ok(()),
            }
        };
        if st.epoch.is_null() {
            return Ok(());
        }
        self.enter();
        ck(unsafe { cudaMemsetAsync(st.epoch, 0, 4, self.stream) }, "p2p epoch reset")?;
        ck(unsafe { cudaMemsetAsync(st.ctr, 0, 4, self.stream) }, "p2p ctr reset")?;
        // ready_local is the row PEERS write via NVLink; zero it so stale
        // stamps from before the reset don't satisfy the first wait. The
        // ping-pong staging needs no reset (fully overwritten each call).
        ck(unsafe { cudaMemsetAsync(st.ready_local, 0, st.world * 4, self.stream) }, "p2p ready reset")?;
        // seen[] holds each peer's last-observed stamp, one row per finish
        // block (see p2p_ar_alloc) — zero it so the first wait of the next
        // replay accepts the peers' first new stamp.
        ck(unsafe { cudaMemsetAsync(st.seen, 0, 256 * st.world * 4, self.stream) }, "p2p seen reset")?;
        Ok(())
    }

    /// One-shot AR v2 (epoch + ping-pong — in-graph, multi-call safe). IN-PLACE
    /// on `buf`: the down kernel stages it to every peer's ping-pong slot,
    /// the sum kernel spins the epoch flags and writes the reduced values
    /// back into `buf` (the sequential down→sum makes the in-place reuse
    /// safe). Returns false when unconfigured or n > max_n (the caller
    /// falls back to NCCL).
    pub fn p2p_ar_v2(&self, buf: &mut DevBuf, n: usize) -> Result<bool> {
        let st = {
            let m = self.p2p_ar.lock().unwrap();
            match m.as_ref() {
                Some(st) => *st,
                None => return Ok(false),
            }
        };
        if n > st.max_n || st.staging_tbl.is_null() {
            return Ok(false);
        }
        self.enter();
        // Zero the "all blocks arrived" counter BEFORE the launch. It is
        // normally reset by the LAST block of the previous call, but a
        // GRID-SIZE change (the per-size b2/b4/b8/b16 graphs each capture
        // their own n*world grid) can leave it non-zero → no block ever
        // sees prev == gridDim.x-1 → the peers' ready flags are never
        // stamped → every rank spins forever (measured diag:
        // myepoch=270 flag=270, off by exactly 1).
        // cudaMemsetAsync is legal inside capture (it becomes a graph node,
        // no host-memory dependency — unlike the table H2D copies).
        ck(unsafe { cudaMemsetAsync(st.ctr, 0, 4, self.stream) }, "p2p ctr zero")?;
        ck(unsafe {
            // v3: fused down+sum single kernel (saves the inter-kernel gap
            // + epoch re-read between publish and collect phases)
            ferrite_p2p_ar_fused_v3(
                buf.as_const_f32(),
                st.staging_tbl as *const *mut f32,
                st.ready_tbl as *const *mut u32,
                st.epoch as *mut u32,
                st.ctr as *mut u32,
                st.staging_local as *const f32,
                st.ready_local as *const u32,
                st.seen as *mut u32,
                buf.as_f32(),
                n as i32,
                st.world as i32,
                self.dev as i32,
                st.max_n as i32, // staging row stride ([2][world][max_n] layout)
                self.stream,
            )
        }, "p2p_ar_v2")?;
        Ok(true)
    }

    /// Elementwise add (residual): z = x + y [n]. MTP layer's standard
    /// (non-MHC) residual connections.
    pub fn add_dev(&self, x: &DevBuf, y: &DevBuf, n: usize) -> Result<DevBuf> {
        self.enter();
        let z = DevBuf::alloc(self.dev, self.stream, n)?;
        ck(unsafe {
            ferrite_add(x.as_const_f32(), y.as_const_f32(), z.as_f32(), n as i32, self.stream)
        }, "add_dev")?;
        Ok(z)
    }

    /// MTP eh_proj input segment for this rank: eh_proj is column-split
    /// (2h/world cols); rank r's slice of cat(enorm, hnorm) is
    /// [r*half, (r+1)*half) where half = 2h/world — ranks < world/2 read
    /// enorm, the rest read hnorm. One D2D copy.
    pub fn mtp_eh_seg_dev(&self, enorm: &DevBuf, hnorm: &DevBuf, rank: usize,
                          world: usize, h: usize) -> Result<DevBuf> {
        self.enter();
        let seg_len = 2 * h / world;
        let seg = DevBuf::alloc(self.dev, self.stream, seg_len)?;
        let (src, off) = if rank < world / 2 {
            (enorm, rank * seg_len)
        } else {
            (hnorm, (rank - world / 2) * seg_len)
        };
        ck(unsafe {
            cudaMemcpyAsync(
                seg.as_f32() as *mut std::ffi::c_void,
                (src.as_const_f32() as *const u8).add(off * 4) as *const std::ffi::c_void,
                seg_len * 4,
                CUDA_MEMCPY_D2D,
                self.stream,
            )
        }, "mtp_eh_seg")?;
        Ok(seg)
    }

    /// embed_expand (host-4ms knife 1): token ids → embed row lookup (bf16
    /// resident table) → MHC expand (row × mult) → writes `out` [n, mult,
    /// hidden] in ONE kernel. Replaces the host embed lookup (n×16KB row
    /// memcpy) + hc_expand (64-192KB Vec concat) + H2D upload with a 12B id
    /// write + one kernel. ids is a device i32 buf (n entries; host writes
    /// go through a small pinned staging or D2D from the argmax output).
    pub fn embed_expand_dev(
        &self,
        table: &Tensor,
        ids: *const i32,
        out: *mut f32,
        n: usize,
        hidden: usize,
        mult: usize,
    ) -> Result<()> {
        self.enter();
        // F32 table (NOT bf16): the embed row feeds the residual stream —
        // bf16 truncation changed the numeric domain (accept 2.38→2.18,
        // argmax ties flip; the W8A8 lesson). f32 = bit-identical to the
        // host lookup. +1.2GB VRAM for 154880x4096.
        let dw = self.dev_weight(table)?;
        let vocab = table.shape.0[0];
        ck(
            unsafe {
                ferrite_embed_expand(
                    dw.ptr,
                    ids,
                    out,
                    n as i32,
                    hidden as i32,
                    mult as i32,
                    vocab as i32,
                    self.stream,
                )
            },
            "embed_expand",
        )?;
        Ok(())
    }

    // ================================================================
    // Zero-H2D device-resident MTP chain (user mandate: the ENTIRE decode
    // loop must have no host-to-device transfers). The token never leaves
    // the device: argmax output (device) → embed kernel reads it → graph
    // input (device) → graph replay → argmax (device) → accept kernel
    // (device) → commit kernel reads k from device → next step.
    // Host reads 8 bytes D2H per step (k + next_token for API response).
    // ================================================================

    /// Embed ONE token from a device int slot (the argmax/accept output —
    /// never crosses to host) into the hc_expand'd graph input [mult, hidden].
    /// Replaces: host embed lookup + hc_expand 64KB Vec + DevBuf upload 4KB
    /// H2D → ONE kernel reading 4B from device.
    /// Device-side f32 argmax result -> i32 token slot (replaces the
    /// D2H->cast->H2D roundtrip between draft1/draft2 whose
    /// cudaStreamSynchronize broke NCCL AR channel continuity, causing
    /// 1-ulp float drift -> argmax flips on near-ties (d1 98347->702 with
    /// bit-identical x2). Keeps the entire draft chain on-device (true
    /// zero-H2D) AND preserves NCCL AR ordering (no sync between drafts).
    pub fn cast_store_i32(
        &self,
        src: *const std::ffi::c_void,
        dst: *mut std::ffi::c_void,
    ) -> Result<()> {
        self.enter();
        ck(unsafe { ferrite_cast_store_i32(src, dst, self.stream) }, "cast_store_i32")?;
        Ok(())
    }

    pub fn embed_one_dev(
        &self,
        table: &Tensor,
        token_slot: *const i32,
        out: *mut f32,
        hidden: usize,
        mult: usize,
    ) -> Result<()> {
        self.enter();
        let dw = self.dev_weight(table)?; // F32 (bit-identical numeric domain)
        let vocab = table.shape.0[0];
        ck(
            unsafe {
                ferrite_embed_one(dw.ptr, token_slot, out, hidden as i32, mult as i32, vocab as i32, self.stream)
            },
            "embed_one_dev",
        )?;
        Ok(())
    }

    /// Embed n tokens from a device int buf (the token chain: [last, d1, d2])
    /// into the graph input [n, mult, hidden] — the verify graph's input
    /// written entirely on device.
    pub fn embed_expand_dev_buf(
        &self,
        table: &Tensor,
        ids_dev: *const i32,
        out: *mut f32,
        n: usize,
        hidden: usize,
        mult: usize,
    ) -> Result<()> {
        self.enter();
        let dw = self.dev_weight(table)?;
        let vocab = table.shape.0[0];
        ck(
            unsafe {
                ferrite_embed_expand_dev(dw.ptr, ids_dev, out, n as i32, hidden as i32, mult as i32, vocab as i32, self.stream)
            },
            "embed_expand_dev",
        )?;
        Ok(())
    }

    /// MTP accept on device (N-UNIFIED): compares the n-1 drafts d[0..n-2]
    /// vs the verify argmax a[0..n-1] (ALL device bufs — the argmax outputs
    /// never cross to host), writes k (longest matching prefix, 1..n) +
    /// next_token + n_accepted to device int slots. The host reads these
    /// 8-12 bytes D2H per step for API response / seq tracking.
    pub fn mtp_accept_dev(
        &self,
        d: *const f32,
        a: *const f32,
        k_out: *mut i32,
        next_token: *mut i32,
        n_accepted: *mut i32,
        n: usize,
    ) -> Result<()> {
        self.enter();
        ck(
            unsafe { ferrite_mtp_accept(d, a, k_out, next_token, n_accepted, n as i32, self.stream) },
            "mtp_accept",
        )?;
        Ok(())
    }

    /// D2D copy with source offset (elements): dst[0..n] = src[src_off..src_off+n].
    /// Used by the MTP verify chain to export h_final's last row into the
    /// draft's fixed h_prev staging buffer (capture-safe graph node).
    pub fn copy_dev(&self, src: &DevBuf, src_off: usize, dst: *mut f32, n: usize) -> Result<()> {
        self.enter();
        ck(unsafe {
            cudaMemcpyAsync(
                dst as *mut std::ffi::c_void,
                (src.as_const_f32() as *const u8).add(src_off * 4) as *const std::ffi::c_void,
                n * 4,
                CUDA_MEMCPY_D2D,
                self.stream,
            )
        }, "copy_dev")?;
        Ok(())
    }

    /// Raw-ptr D2D copy (MTP ping-pong commit: verify scratch B -> main
    /// state A, or A -> B copy-in). Both sides are raw device pointers.
    pub fn copy_raw_dev(&self, src: *const f32, dst: *mut f32, n: usize) -> Result<()> {
        self.enter();
        ck(unsafe {
            cudaMemcpyAsync(
                dst as *mut std::ffi::c_void,
                src as *const std::ffi::c_void,
                n * 4,
                CUDA_MEMCPY_D2D,
                self.stream,
            )
        }, "copy_raw_dev")?;
        Ok(())
    }

    /// MTP accept commit with k from a DEVICE buffer (zero-H2D chain: the
    /// accept kernel wrote k to k_dev on device — no host round-trip).
    /// Same commit kernel as the pinned variant, but reads k from device.
    /// N-UNIFIED: cp.mtp_n = FERRITE_MTP_N — the kernel's k range 1..=n
    /// (k=n commits B; k=j<n commits snapshot j-1 at base + (j-1)*len).
    pub fn mtp_commit_dev(&self, k_dev: *const i32) -> Result<()> {
        let m = self.mtp.lock().unwrap();
        let m = m
            .as_ref()
            .ok_or_else(|| FerriteError::Config("mtp bufs missing".into()))?;
        let cp = m
            .commit
            .as_ref()
            .ok_or_else(|| FerriteError::Config("mtp commit plan missing".into()))?;
        self.enter();
        // the commit kernel reads k from a device pointer — the pinned k_pin
        // slot is still written for the legacy path, but the kernel parameter
        // points at the device slot for the zero-H2D chain.
        // For now we pass the device k pointer as the "pinned" pointer (the
        // kernel dereferences it the same way — zero-copy device read).
        ck(
            unsafe {
                ferrite_mtp_commit(
                    k_dev,
                    cp.plan.as_f32() as *const *mut f32,
                    cp.n as i32,
                    cp.conv_len as i32,
                    cp.gdn_len as i32,
                    m.hf_v.as_const_f32(),
                    m.hprev.as_f32(),
                    cp.hidden as i32,
                    cp.mtp_n as i32,
                    self.stream,
                )
            },
            "mtp_commit_dev",
        )?;
        Ok(())
    }

    /// MTP accept commit (single launch): writes k to the pinned slot
    /// (zero-copy kernel read) and fires ferrite_mtp_commit — the kernel
    /// selects B_k per GDN layer (k=n -> B, j<n -> snapshot j-1), copies
    /// it back to the main state A, and sets hprev <- hf_v row (k-1) for
    /// the next draft step. N-UNIFIED (k in 1..=FERRITE_MTP_N). Replaces
    /// 2*n_gdn cudaMemcpyAsync launches + the hprev select. Graph-unsafe
    /// (pinned k write) — always called OUTSIDE captures, right after the
    /// verify replay's D2H sync.
    pub fn mtp_commit(&self, k: i32) -> Result<()> {
        let m = self.mtp.lock().unwrap();
        let m = m
            .as_ref()
            .ok_or_else(|| FerriteError::Config("mtp bufs missing".into()))?;
        let cp = m
            .commit
            .as_ref()
            .ok_or_else(|| FerriteError::Config("mtp commit plan missing".into()))?;
        unsafe {
            *cp.k_pin = k;
        }
        self.enter();
        ck(
            unsafe {
                ferrite_mtp_commit(
                    cp.k_pin as *const i32,
                    cp.plan.as_f32() as *const *mut f32,
                    cp.n as i32,
                    cp.conv_len as i32,
                    cp.gdn_len as i32,
                    m.hf_v.as_const_f32(),
                    m.hprev.as_f32(),
                    cp.hidden as i32,
                    cp.mtp_n as i32,
                    self.stream,
                )
            },
            "mtp_commit",
        )?;
        Ok(())
    }
}

impl CudaBackend {
    /// Graph replay WITHOUT the D2H download: write the input to the
    /// captured staging, launch, return the output DEVICE pointer (for
    /// P2P all-reduce — the result stays on GPU).
    pub fn graph_run_dev(&self, name: &str, input: &[f32]) -> Result<Option<usize>> {
        let Some(io) = self.graph_io_get(name) else { return Ok(None); };
        if input.len() != io.x_len {
            return Err(FerriteError::InvalidArg(format!(
                "graph_run_dev {name}: input {} != stage {}", input.len(), io.x_len
            )));
        }
        unsafe {
            std::ptr::copy_nonoverlapping(input.as_ptr(), io.x_stage as *mut f32, io.x_len);
        }
        if !self.graph_replay(name) {
            return Ok(None);
        }
        Ok(Some(io.out_dev as usize))
    }
}

impl CudaBackend {
    /// Raw P2P copy (NVLink): copy `count` floats from `src` (on `src_dev`)
    /// to `dst` (on this device). Public for the hn broadcast in the P2P
    /// decode chain.
    pub fn p2p_copy_raw(&self, dst: *mut f32, src: *const f32, src_dev: i32, count: usize) -> Result<()> {
        self.enter();
        ck(unsafe { ferrite_p2p_copy(dst, self.dev, src, src_dev, count, self.stream) }, "p2p_copy_raw")?;
        Ok(())
    }
}

/// Set the current CUDA device (for NCCL init — needs device 0 context).
pub fn cuda_set_device(dev: i32) {
    unsafe { cudaSetDevice(dev) };
}

/// ncu capture-window control (`ncu --profile-from-start off`): FERRITE_NCU=1
/// opens the window right before the decode loop — the 80s weight load
/// (38k H2D uploads ncu intercepts at ms-level cost each, stalling the run
/// 10+ minutes) and prefill run at full speed, only the decode steps are
/// profiled. cuProfilerStart/Stop work for both ncu and nsys.
pub fn profiler_start() {
    unsafe { cudaProfilerStart() };
}

pub fn profiler_stop() {
    unsafe { cudaProfilerStop() };
}
