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
#include <tl_templates/cuda/instruction/mma.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/cuda_fp8.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/scan.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void main_kernel(const fp8_e4_t* __restrict__ A, const float* __restrict__ ASC, float* __restrict__ P, const fp8_e4_t* __restrict__ W, const fp8_e8_t* __restrict__ WSC, int m);
extern "C" __global__ void __launch_bounds__(128, 1) main_kernel(const fp8_e4_t* __restrict__ A, const float* __restrict__ ASC, float* __restrict__ P, const fp8_e4_t* __restrict__ W, const fp8_e8_t* __restrict__ WSC, int m) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* W_sh = ((void*)((char*)buf_dyn_shmem + 0));
  void* A_sh = ((void*)((char*)buf_dyn_shmem + 12288));
  float C_l[16];
  float C_p[16];
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(C_l + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 2; ++i_1) {
    tl::cp_async_gs<16>((&(((fp8_e4_t*)W_sh)[(((i_1 * 2048) + ((((int)threadIdx.x) >> 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16))])), (&(W[(((((((int)blockIdx.x) * 655360) + (i_1 * 327680)) + ((((int)threadIdx.x) >> 1) * 5120)) + (((int)blockIdx.y) * 640)) + ((((int)threadIdx.x) & 1) * 16))])));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_2 = 0; i_2 < 2; ++i_2) {
    tl::cp_async_gs<16>((&(((fp8_e4_t*)W_sh)[((((i_2 * 2048) + ((((int)threadIdx.x) >> 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 4096)])), (&(W[((((((((int)blockIdx.x) * 655360) + (i_2 * 327680)) + ((((int)threadIdx.x) >> 1) * 5120)) + (((int)blockIdx.y) * 640)) + ((((int)threadIdx.x) & 1) * 16)) + 32)])));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_3 = 0; i_3 < 2; ++i_3) {
    tl::cp_async_gs<16>((&(((fp8_e4_t*)W_sh)[((((i_3 * 2048) + ((((int)threadIdx.x) >> 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 8192)])), (&(W[((((((((int)blockIdx.x) * 655360) + (i_3 * 327680)) + ((((int)threadIdx.x) >> 1) * 5120)) + (((int)blockIdx.y) * 640)) + ((((int)threadIdx.x) & 1) * 16)) + 64)])));
  }
  tl::cp_async_commit();
  for (int ko = 0; ko < 17; ++ko) {
    __syncthreads();
    if ((((int)threadIdx.x) >> 3) < m) {
      *(fp8_e4_4_t*)(((fp8_e4_t*)A_sh) + ((((((int)threadIdx.x) >> 3) * 32) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + ((((int)threadIdx.x) & 3) * 4))) = *(fp8_e4_4_t*)(A + (((((((int)threadIdx.x) >> 3) * 5120) + (((int)blockIdx.y) * 640)) + (ko * 32)) + ((((int)threadIdx.x) & 7) * 4)));
    } else {
      fp8_e4_t broadcast_var_1 = fp8_e4_t(0x0p+0f/*0.000000e+00*/);
      *(fp8_e4_4_t*)(((fp8_e4_t*)A_sh) + ((((((int)threadIdx.x) >> 3) * 32) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + ((((int)threadIdx.x) & 3) * 4))) = make_fp8_e4_4_t(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1);
    }
    tl::cp_async_wait<2>();
    __syncthreads();
    {
      fp8_e4_t A_local[16];
      fp8_e4_t B_local[32];
      #pragma unroll
      for (int i_4 = 0; i_4 < 4; ++i_4) {
        float broadcast_var_2 = 0x0p+0f/*0.000000e+00*/;
        *(float4*)(C_p + (i_4 * 4)) = make_float4(broadcast_var_2, broadcast_var_2, broadcast_var_2, broadcast_var_2);
      }
      tl::ptx_ldmatrix_x4((&(((fp8_e4_t*)A_sh)[(((((int)threadIdx.x) & 15) * 32) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16))])), (&(A_local[0])));
      #pragma unroll
      for (int i_5 = 0; i_5 < 2; ++i_5) {
        tl::ptx_ldmatrix_x4((&(((fp8_e4_t*)W_sh)[(((((((ko % 3) * 4096) + ((((int)threadIdx.x) >> 5) * 1024)) + (i_5 * 512)) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 7) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16))])), (&(B_local[(i_5 * 16)])));
      }
      for (int j = 0; j < 2; ++j) {
        tl::mma_sync<tl::DataType::kFloat8_e4m3, tl::DataType::kFloat8_e4m3, tl::DataType::kFloat32, 16, 8, 32, false, true>(reinterpret_cast<float*>(C_p + (j * 8)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + (j * 16)));
        tl::mma_sync<tl::DataType::kFloat8_e4m3, tl::DataType::kFloat8_e4m3, tl::DataType::kFloat32, 16, 8, 32, false, true>(reinterpret_cast<float*>(C_p + ((j * 8) + 4)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + ((j * 16) + 8)));
      }
    }
    __syncthreads();
    #pragma unroll
    for (int i_6 = 0; i_6 < 2; ++i_6) {
      tl::cp_async_gs<16>((&(((fp8_e4_t*)W_sh)[(((((ko % 3) * 4096) + (i_6 * 2048)) + ((((int)threadIdx.x) >> 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16))])), (&(W[(((((((((int)blockIdx.x) * 655360) + (i_6 * 327680)) + ((((int)threadIdx.x) >> 1) * 5120)) + (((int)blockIdx.y) * 640)) + (ko * 32)) + ((((int)threadIdx.x) & 1) * 16)) + 96)])));
    }
    tl::cp_async_commit();
    #pragma unroll
    for (int i_7 = 0; i_7 < 8; ++i_7) {
      float ASC_local_cast[2];
      *(float2*)(ASC_local_cast + 0) = make_float2(ASC[(((((i_7 & 1) * 1280) + (((((int)threadIdx.x) & 31) >> 2) * 160)) + (((int)blockIdx.y) * 20)) + ko)], ASC[(((((i_7 & 1) * 1280) + (((((int)threadIdx.x) & 31) >> 2) * 160)) + (((int)blockIdx.y) * 20)) + ko)]);
      float2 __1;
        float2 __2;
          float2 v_ = *(float2*)(C_p + (i_7 * 2));
          float2 v__1 = *(float2*)(ASC_local_cast + 0);
          *(float2*)(&(__2.x)) = tl::mul2(*(float2*)(&(v_.x)), *(float2*)(&(v__1.x)));
        float2 v__2 = make_float2(((float)WSC[((((((int)blockIdx.x) * 640) + ((((int)threadIdx.x) >> 5) * 160)) + (((int)blockIdx.y) * 20)) + ko)]), ((float)WSC[((((((int)blockIdx.x) * 640) + ((((int)threadIdx.x) >> 5) * 160)) + (((int)blockIdx.y) * 20)) + ko)]));
        float2 v__3 = *(float2*)(C_l + (i_7 * 2));
        *(float2*)(&(__1.x)) = tl::fma2(*(float2*)(&(__2.x)), *(float2*)(&(v__2.x)), *(float2*)(&(v__3.x)));
      *(float2*)(C_l + (i_7 * 2)) = __1;
    }
  }
  __syncthreads();
  if ((((int)threadIdx.x) >> 3) < m) {
    *(fp8_e4_4_t*)(((fp8_e4_t*)A_sh) + ((((((int)threadIdx.x) >> 3) * 32) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + ((((int)threadIdx.x) & 3) * 4))) = *(fp8_e4_4_t*)(A + (((((((int)threadIdx.x) >> 3) * 5120) + (((int)blockIdx.y) * 640)) + ((((int)threadIdx.x) & 7) * 4)) + 544));
  } else {
    fp8_e4_t broadcast_var_3 = fp8_e4_t(0x0p+0f/*0.000000e+00*/);
    *(fp8_e4_4_t*)(((fp8_e4_t*)A_sh) + ((((((int)threadIdx.x) >> 3) * 32) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + ((((int)threadIdx.x) & 3) * 4))) = make_fp8_e4_4_t(broadcast_var_3, broadcast_var_3, broadcast_var_3, broadcast_var_3);
  }
  tl::cp_async_wait<2>();
  __syncthreads();
  {
    fp8_e4_t A_local_1[16];
    fp8_e4_t B_local_1[32];
    #pragma unroll
    for (int i_8 = 0; i_8 < 4; ++i_8) {
      float broadcast_var_4 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(C_p + (i_8 * 4)) = make_float4(broadcast_var_4, broadcast_var_4, broadcast_var_4, broadcast_var_4);
    }
    tl::ptx_ldmatrix_x4((&(((fp8_e4_t*)A_sh)[(((((int)threadIdx.x) & 15) * 32) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16))])), (&(A_local_1[0])));
    #pragma unroll
    for (int i_9 = 0; i_9 < 2; ++i_9) {
      tl::ptx_ldmatrix_x4((&(((fp8_e4_t*)W_sh)[(((((((((int)threadIdx.x) >> 5) * 1024) + (i_9 * 512)) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 7) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + 8192)])), (&(B_local_1[(i_9 * 16)])));
    }
    for (int j_1 = 0; j_1 < 2; ++j_1) {
      tl::mma_sync<tl::DataType::kFloat8_e4m3, tl::DataType::kFloat8_e4m3, tl::DataType::kFloat32, 16, 8, 32, false, true>(reinterpret_cast<float*>(C_p + (j_1 * 8)), reinterpret_cast<const unsigned*>(A_local_1 + 0), reinterpret_cast<const unsigned*>(B_local_1 + (j_1 * 16)));
      tl::mma_sync<tl::DataType::kFloat8_e4m3, tl::DataType::kFloat8_e4m3, tl::DataType::kFloat32, 16, 8, 32, false, true>(reinterpret_cast<float*>(C_p + ((j_1 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_1 + 0), reinterpret_cast<const unsigned*>(B_local_1 + ((j_1 * 16) + 8)));
    }
  }
  #pragma unroll
  for (int i_10 = 0; i_10 < 8; ++i_10) {
    float ASC_local_cast_1[2];
    *(float2*)(ASC_local_cast_1 + 0) = make_float2(ASC[(((((i_10 & 1) * 1280) + (((((int)threadIdx.x) & 31) >> 2) * 160)) + (((int)blockIdx.y) * 20)) + 17)], ASC[(((((i_10 & 1) * 1280) + (((((int)threadIdx.x) & 31) >> 2) * 160)) + (((int)blockIdx.y) * 20)) + 17)]);
    float2 __3;
      float2 __4;
        float2 v__4 = *(float2*)(C_p + (i_10 * 2));
        float2 v__5 = *(float2*)(ASC_local_cast_1 + 0);
        *(float2*)(&(__4.x)) = tl::mul2(*(float2*)(&(v__4.x)), *(float2*)(&(v__5.x)));
      float2 v__6 = make_float2(((float)WSC[((((((int)blockIdx.x) * 640) + ((((int)threadIdx.x) >> 5) * 160)) + (((int)blockIdx.y) * 20)) + 17)]), ((float)WSC[((((((int)blockIdx.x) * 640) + ((((int)threadIdx.x) >> 5) * 160)) + (((int)blockIdx.y) * 20)) + 17)]));
      float2 v__7 = *(float2*)(C_l + (i_10 * 2));
      *(float2*)(&(__3.x)) = tl::fma2(*(float2*)(&(__4.x)), *(float2*)(&(v__6.x)), *(float2*)(&(v__7.x)));
    *(float2*)(C_l + (i_10 * 2)) = __3;
  }
  __syncthreads();
  if ((((int)threadIdx.x) >> 3) < m) {
    *(fp8_e4_4_t*)(((fp8_e4_t*)A_sh) + ((((((int)threadIdx.x) >> 3) * 32) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + ((((int)threadIdx.x) & 3) * 4))) = *(fp8_e4_4_t*)(A + (((((((int)threadIdx.x) >> 3) * 5120) + (((int)blockIdx.y) * 640)) + ((((int)threadIdx.x) & 7) * 4)) + 576));
  } else {
    fp8_e4_t broadcast_var_5 = fp8_e4_t(0x0p+0f/*0.000000e+00*/);
    *(fp8_e4_4_t*)(((fp8_e4_t*)A_sh) + ((((((int)threadIdx.x) >> 3) * 32) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + ((((int)threadIdx.x) & 3) * 4))) = make_fp8_e4_4_t(broadcast_var_5, broadcast_var_5, broadcast_var_5, broadcast_var_5);
  }
  tl::cp_async_wait<1>();
  __syncthreads();
  {
    fp8_e4_t A_local_2[16];
    fp8_e4_t B_local_2[32];
    #pragma unroll
    for (int i_11 = 0; i_11 < 4; ++i_11) {
      float broadcast_var_6 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(C_p + (i_11 * 4)) = make_float4(broadcast_var_6, broadcast_var_6, broadcast_var_6, broadcast_var_6);
    }
    tl::ptx_ldmatrix_x4((&(((fp8_e4_t*)A_sh)[(((((int)threadIdx.x) & 15) * 32) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16))])), (&(A_local_2[0])));
    #pragma unroll
    for (int i_12 = 0; i_12 < 2; ++i_12) {
      tl::ptx_ldmatrix_x4((&(((fp8_e4_t*)W_sh)[((((((((int)threadIdx.x) >> 5) * 1024) + (i_12 * 512)) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 7) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16))])), (&(B_local_2[(i_12 * 16)])));
    }
    for (int j_2 = 0; j_2 < 2; ++j_2) {
      tl::mma_sync<tl::DataType::kFloat8_e4m3, tl::DataType::kFloat8_e4m3, tl::DataType::kFloat32, 16, 8, 32, false, true>(reinterpret_cast<float*>(C_p + (j_2 * 8)), reinterpret_cast<const unsigned*>(A_local_2 + 0), reinterpret_cast<const unsigned*>(B_local_2 + (j_2 * 16)));
      tl::mma_sync<tl::DataType::kFloat8_e4m3, tl::DataType::kFloat8_e4m3, tl::DataType::kFloat32, 16, 8, 32, false, true>(reinterpret_cast<float*>(C_p + ((j_2 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_2 + 0), reinterpret_cast<const unsigned*>(B_local_2 + ((j_2 * 16) + 8)));
    }
  }
  #pragma unroll
  for (int i_13 = 0; i_13 < 8; ++i_13) {
    float ASC_local_cast_2[2];
    *(float2*)(ASC_local_cast_2 + 0) = make_float2(ASC[(((((i_13 & 1) * 1280) + (((((int)threadIdx.x) & 31) >> 2) * 160)) + (((int)blockIdx.y) * 20)) + 18)], ASC[(((((i_13 & 1) * 1280) + (((((int)threadIdx.x) & 31) >> 2) * 160)) + (((int)blockIdx.y) * 20)) + 18)]);
    float2 __5;
      float2 __6;
        float2 v__8 = *(float2*)(C_p + (i_13 * 2));
        float2 v__9 = *(float2*)(ASC_local_cast_2 + 0);
        *(float2*)(&(__6.x)) = tl::mul2(*(float2*)(&(v__8.x)), *(float2*)(&(v__9.x)));
      float2 v__10 = make_float2(((float)WSC[((((((int)blockIdx.x) * 640) + ((((int)threadIdx.x) >> 5) * 160)) + (((int)blockIdx.y) * 20)) + 18)]), ((float)WSC[((((((int)blockIdx.x) * 640) + ((((int)threadIdx.x) >> 5) * 160)) + (((int)blockIdx.y) * 20)) + 18)]));
      float2 v__11 = *(float2*)(C_l + (i_13 * 2));
      *(float2*)(&(__5.x)) = tl::fma2(*(float2*)(&(__6.x)), *(float2*)(&(v__10.x)), *(float2*)(&(v__11.x)));
    *(float2*)(C_l + (i_13 * 2)) = __5;
  }
  __syncthreads();
  if ((((int)threadIdx.x) >> 3) < m) {
    *(fp8_e4_4_t*)(((fp8_e4_t*)A_sh) + ((((((int)threadIdx.x) >> 3) * 32) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + ((((int)threadIdx.x) & 3) * 4))) = *(fp8_e4_4_t*)(A + (((((((int)threadIdx.x) >> 3) * 5120) + (((int)blockIdx.y) * 640)) + ((((int)threadIdx.x) & 7) * 4)) + 608));
  } else {
    fp8_e4_t broadcast_var_7 = fp8_e4_t(0x0p+0f/*0.000000e+00*/);
    *(fp8_e4_4_t*)(((fp8_e4_t*)A_sh) + ((((((int)threadIdx.x) >> 3) * 32) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + ((((int)threadIdx.x) & 3) * 4))) = make_fp8_e4_4_t(broadcast_var_7, broadcast_var_7, broadcast_var_7, broadcast_var_7);
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  {
    fp8_e4_t A_local_3[16];
    fp8_e4_t B_local_3[32];
    #pragma unroll
    for (int i_14 = 0; i_14 < 4; ++i_14) {
      float broadcast_var_8 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(C_p + (i_14 * 4)) = make_float4(broadcast_var_8, broadcast_var_8, broadcast_var_8, broadcast_var_8);
    }
    tl::ptx_ldmatrix_x4((&(((fp8_e4_t*)A_sh)[(((((int)threadIdx.x) & 15) * 32) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16))])), (&(A_local_3[0])));
    #pragma unroll
    for (int i_15 = 0; i_15 < 2; ++i_15) {
      tl::ptx_ldmatrix_x4((&(((fp8_e4_t*)W_sh)[(((((((((int)threadIdx.x) >> 5) * 1024) + (i_15 * 512)) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 7) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + 4096)])), (&(B_local_3[(i_15 * 16)])));
    }
    for (int j_3 = 0; j_3 < 2; ++j_3) {
      tl::mma_sync<tl::DataType::kFloat8_e4m3, tl::DataType::kFloat8_e4m3, tl::DataType::kFloat32, 16, 8, 32, false, true>(reinterpret_cast<float*>(C_p + (j_3 * 8)), reinterpret_cast<const unsigned*>(A_local_3 + 0), reinterpret_cast<const unsigned*>(B_local_3 + (j_3 * 16)));
      tl::mma_sync<tl::DataType::kFloat8_e4m3, tl::DataType::kFloat8_e4m3, tl::DataType::kFloat32, 16, 8, 32, false, true>(reinterpret_cast<float*>(C_p + ((j_3 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_3 + 0), reinterpret_cast<const unsigned*>(B_local_3 + ((j_3 * 16) + 8)));
    }
  }
  #pragma unroll
  for (int i_16 = 0; i_16 < 8; ++i_16) {
    float ASC_local_cast_3[2];
    *(float2*)(ASC_local_cast_3 + 0) = make_float2(ASC[(((((i_16 & 1) * 1280) + (((((int)threadIdx.x) & 31) >> 2) * 160)) + (((int)blockIdx.y) * 20)) + 19)], ASC[(((((i_16 & 1) * 1280) + (((((int)threadIdx.x) & 31) >> 2) * 160)) + (((int)blockIdx.y) * 20)) + 19)]);
    float2 __7;
      float2 __8;
        float2 v__12 = *(float2*)(C_p + (i_16 * 2));
        float2 v__13 = *(float2*)(ASC_local_cast_3 + 0);
        *(float2*)(&(__8.x)) = tl::mul2(*(float2*)(&(v__12.x)), *(float2*)(&(v__13.x)));
      float2 v__14 = make_float2(((float)WSC[((((((int)blockIdx.x) * 640) + ((((int)threadIdx.x) >> 5) * 160)) + (((int)blockIdx.y) * 20)) + 19)]), ((float)WSC[((((((int)blockIdx.x) * 640) + ((((int)threadIdx.x) >> 5) * 160)) + (((int)blockIdx.y) * 20)) + 19)]));
      float2 v__15 = *(float2*)(C_l + (i_16 * 2));
      *(float2*)(&(__7.x)) = tl::fma2(*(float2*)(&(__8.x)), *(float2*)(&(v__14.x)), *(float2*)(&(v__15.x)));
    *(float2*)(C_l + (i_16 * 2)) = __7;
  }
  #pragma unroll
  for (int i_17 = 0; i_17 < 8; ++i_17) {
    *(float2*)(P + (((((((((int)blockIdx.y) * 8192) + ((i_17 & 1) * 4096)) + (((((int)threadIdx.x) & 31) >> 2) * 512)) + (((int)blockIdx.x) * 128)) + ((((int)threadIdx.x) >> 5) * 32)) + ((i_17 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(C_l + (i_17 * 2));
  }
}

