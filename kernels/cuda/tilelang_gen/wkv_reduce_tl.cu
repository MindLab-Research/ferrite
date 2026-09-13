// ============================================================================
// GENERATED — do not edit. Regenerate via kernels/tilelang/gen_wkv_aot.py
// on a B300 (tilelang 0.1.14, sm_103a); see tilelang_gen/PROVENANCE.md.
// Source: TileLang 0.1.14 JITKernel.get_kernel_source(), route A (fp8 mma +
// per-32 ue8m0 scale, K-split), pass_configs={TL_DISABLE_TMA_LOWER:1,
// TL_DISABLE_WARP_SPECIALIZED:1} (bare-pointer ABI), with a runtime-m predicate
// on the activation staging / reduction store (see the generator's header).
// ============================================================================
#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/copy_sm90.h>
#include <tl_templates/cuda/copy_sm100.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/scan.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void main_kernel(float* __restrict__ C, const float* __restrict__ P, int m);
extern "C" __global__ void __launch_bounds__(256, 1) main_kernel(float* __restrict__ C, const float* __restrict__ P, int m) {
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    if (((i * 8) + (((int)threadIdx.x) >> 5)) < m) {
      ulonglong4 __1;
        ulonglong4 __2;
          ulonglong4 __3;
            ulonglong4 __4;
              ulonglong4 __5;
                ulonglong4 __6;
                  ulonglong4 __7;
                    ulonglong4 v_ = tl::load_global_256(&(*(ulonglong4*)(P + ((((i * 4096) + ((((int)threadIdx.x) >> 5) * 512)) + (((int)blockIdx.x) * 256)) + ((((int)threadIdx.x) & 31) * 8)))));
                    ulonglong4 v__1 = tl::load_global_256(&(*(ulonglong4*)(P + (((((i * 4096) + ((((int)threadIdx.x) >> 5) * 512)) + (((int)blockIdx.x) * 256)) + ((((int)threadIdx.x) & 31) * 8)) + 8192))));
                    *(float2*)(&(__7.x)) = tl::add2(*(float2*)(&(v_.x)), *(float2*)(&(v__1.x)));
                    *(float2*)(&(__7.y)) = tl::add2(*(float2*)(&(v_.y)), *(float2*)(&(v__1.y)));
                    *(float2*)(&(__7.z)) = tl::add2(*(float2*)(&(v_.z)), *(float2*)(&(v__1.z)));
                    *(float2*)(&(__7.w)) = tl::add2(*(float2*)(&(v_.w)), *(float2*)(&(v__1.w)));
                  ulonglong4 v__2 = tl::load_global_256(&(*(ulonglong4*)(P + (((((i * 4096) + ((((int)threadIdx.x) >> 5) * 512)) + (((int)blockIdx.x) * 256)) + ((((int)threadIdx.x) & 31) * 8)) + 16384))));
                  *(float2*)(&(__6.x)) = tl::add2(*(float2*)(&(__7.x)), *(float2*)(&(v__2.x)));
                  *(float2*)(&(__6.y)) = tl::add2(*(float2*)(&(__7.y)), *(float2*)(&(v__2.y)));
                  *(float2*)(&(__6.z)) = tl::add2(*(float2*)(&(__7.z)), *(float2*)(&(v__2.z)));
                  *(float2*)(&(__6.w)) = tl::add2(*(float2*)(&(__7.w)), *(float2*)(&(v__2.w)));
                ulonglong4 v__3 = tl::load_global_256(&(*(ulonglong4*)(P + (((((i * 4096) + ((((int)threadIdx.x) >> 5) * 512)) + (((int)blockIdx.x) * 256)) + ((((int)threadIdx.x) & 31) * 8)) + 24576))));
                *(float2*)(&(__5.x)) = tl::add2(*(float2*)(&(__6.x)), *(float2*)(&(v__3.x)));
                *(float2*)(&(__5.y)) = tl::add2(*(float2*)(&(__6.y)), *(float2*)(&(v__3.y)));
                *(float2*)(&(__5.z)) = tl::add2(*(float2*)(&(__6.z)), *(float2*)(&(v__3.z)));
                *(float2*)(&(__5.w)) = tl::add2(*(float2*)(&(__6.w)), *(float2*)(&(v__3.w)));
              ulonglong4 v__4 = tl::load_global_256(&(*(ulonglong4*)(P + (((((i * 4096) + ((((int)threadIdx.x) >> 5) * 512)) + (((int)blockIdx.x) * 256)) + ((((int)threadIdx.x) & 31) * 8)) + 32768))));
              *(float2*)(&(__4.x)) = tl::add2(*(float2*)(&(__5.x)), *(float2*)(&(v__4.x)));
              *(float2*)(&(__4.y)) = tl::add2(*(float2*)(&(__5.y)), *(float2*)(&(v__4.y)));
              *(float2*)(&(__4.z)) = tl::add2(*(float2*)(&(__5.z)), *(float2*)(&(v__4.z)));
              *(float2*)(&(__4.w)) = tl::add2(*(float2*)(&(__5.w)), *(float2*)(&(v__4.w)));
            ulonglong4 v__5 = tl::load_global_256(&(*(ulonglong4*)(P + (((((i * 4096) + ((((int)threadIdx.x) >> 5) * 512)) + (((int)blockIdx.x) * 256)) + ((((int)threadIdx.x) & 31) * 8)) + 40960))));
            *(float2*)(&(__3.x)) = tl::add2(*(float2*)(&(__4.x)), *(float2*)(&(v__5.x)));
            *(float2*)(&(__3.y)) = tl::add2(*(float2*)(&(__4.y)), *(float2*)(&(v__5.y)));
            *(float2*)(&(__3.z)) = tl::add2(*(float2*)(&(__4.z)), *(float2*)(&(v__5.z)));
            *(float2*)(&(__3.w)) = tl::add2(*(float2*)(&(__4.w)), *(float2*)(&(v__5.w)));
          ulonglong4 v__6 = tl::load_global_256(&(*(ulonglong4*)(P + (((((i * 4096) + ((((int)threadIdx.x) >> 5) * 512)) + (((int)blockIdx.x) * 256)) + ((((int)threadIdx.x) & 31) * 8)) + 49152))));
          *(float2*)(&(__2.x)) = tl::add2(*(float2*)(&(__3.x)), *(float2*)(&(v__6.x)));
          *(float2*)(&(__2.y)) = tl::add2(*(float2*)(&(__3.y)), *(float2*)(&(v__6.y)));
          *(float2*)(&(__2.z)) = tl::add2(*(float2*)(&(__3.z)), *(float2*)(&(v__6.z)));
          *(float2*)(&(__2.w)) = tl::add2(*(float2*)(&(__3.w)), *(float2*)(&(v__6.w)));
        ulonglong4 v__7 = tl::load_global_256(&(*(ulonglong4*)(P + (((((i * 4096) + ((((int)threadIdx.x) >> 5) * 512)) + (((int)blockIdx.x) * 256)) + ((((int)threadIdx.x) & 31) * 8)) + 57344))));
        *(float2*)(&(__1.x)) = tl::add2(*(float2*)(&(__2.x)), *(float2*)(&(v__7.x)));
        *(float2*)(&(__1.y)) = tl::add2(*(float2*)(&(__2.y)), *(float2*)(&(v__7.y)));
        *(float2*)(&(__1.z)) = tl::add2(*(float2*)(&(__2.z)), *(float2*)(&(v__7.z)));
        *(float2*)(&(__1.w)) = tl::add2(*(float2*)(&(__2.w)), *(float2*)(&(v__7.w)));
      tl::store_global_256(&(*(ulonglong4*)(C + ((((i * 4096) + ((((int)threadIdx.x) >> 5) * 512)) + (((int)blockIdx.x) * 256)) + ((((int)threadIdx.x) & 31) * 8)))), __1);
    }
  }
}

