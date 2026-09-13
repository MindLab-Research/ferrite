#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/instruction/mma.h>
#include <tl_templates/cuda/intrin.h>
#include <tl_templates/cuda/barrier.h>
#include <tl_templates/cuda/copy_sm90.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/scan.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void main_kernel(__grid_constant__ const CUtensorMap A_desc, float* __restrict__ C, const int* __restrict__ Eid, const uint* __restrict__ Lut, __grid_constant__ const CUtensorMap Wq_desc, const uchar* __restrict__ Ws);
extern "C" __global__ void __launch_bounds__(512, 1) main_kernel(__grid_constant__ const CUtensorMap A_desc, float* __restrict__ C, const int* __restrict__ Eid, const uint* __restrict__ Lut, __grid_constant__ const CUtensorMap Wq_desc, const uchar* __restrict__ Ws) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* Bq = ((void*)((char*)buf_dyn_shmem + 0));
  void* A_sh = ((void*)((char*)buf_dyn_shmem + 16384));
  void* B_sh = ((void*)((char*)buf_dyn_shmem + 20480));
  void* Bs = ((void*)((char*)buf_dyn_shmem + 86016));
  __shared__ __align__(16) uint64_t mbarrier_mem[8];
  auto mbarrier = reinterpret_cast<Barrier*>(mbarrier_mem);
  float C_l[16];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(A_desc);
    tl::prefetch_tma_descriptor(Wq_desc);
  }
  if (tl::tl_shuffle_elect<0>()) {
    mbarrier[0].init(1);
    mbarrier[1].init(1);
    mbarrier[2].init(1);
    mbarrier[3].init(1);
    mbarrier[4].init(256);
    mbarrier[5].init(256);
    mbarrier[6].init(256);
    mbarrier[7].init(256);
  }
  tl::fence_barrier_init();
  __syncthreads();
  if (((int)threadIdx.x) < 256) {
    tl::warpgroup_reg_dealloc<24>();
    int e = Eid[((int)blockIdx.y)];
    for (int k = 0; k < 5; ++k) {
      mbarrier[((k & 1) + 4)].wait((((k & 3) >> 1) ^ 1));
      if (tl::tl_shuffle_elect<256>()) {
        mbarrier[(k & 1)].arrive_and_expect_tx(2048);
        tl::tma_load(A_desc, mbarrier[(k & 1)], (&(((bfloat16_t*)A_sh)[((k & 1) * 1024)])), (k * 64), (((int)blockIdx.y) * 16));
      }
      mbarrier[((k & 1) + 6)].wait((((k & 3) >> 1) ^ 1));
      if (tl::tl_shuffle_elect<256>()) {
        mbarrier[((k & 1) + 2)].arrive_and_expect_tx(8192);
        tl::tma_load(Wq_desc, mbarrier[((k & 1) + 2)], (&(((uchar*)Bq)[((k & 1) * 8192)])), (k * 32), (((int)blockIdx.x) * 256), e);
      }
    }
  } else {
    tl::warpgroup_reg_alloc<224>();
    int e_1 = Eid[((int)blockIdx.y)];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      float broadcast_var = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(C_l + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
    }
    for (int k_1 = 0; k_1 < 5; ++k_1) {
      tl::__sync_thread_partial(3, 256);
      uchar broadcast_var_1 = (uchar)0;
      uchar2 condval;
      if (((0 <= e_1) && (e_1 < 384))) {
        condval = *(uchar2*)(Ws + ((((((int64_t)e_1) * (int64_t)51200) + (((int64_t)((int)blockIdx.x)) * (int64_t)2560)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)255) * (int64_t)10)) + (((int64_t)k_1) * (int64_t)2)));
      } else {
        condval = make_uchar2(broadcast_var_1, broadcast_var_1);
      }
      *(uchar2*)(((uchar*)Bs) + (((k_1 & 1) * 512) + ((((int)threadIdx.x) & 255) * 2))) = condval;
      mbarrier[((k_1 & 1) + 2)].wait(((k_1 & 3) >> 1));
      tl::__sync_thread_partial(3, 256);
      #pragma unroll
      for (int i_1 = 0; i_1 < 32; ++i_1) {
        uint p = Lut[((int)((uchar*)Bq)[(((((k_1 & 1) * 8192) + (i_1 * 256)) + ((int)threadIdx.x)) - 256)])];
        ushort v_ = (ushort)(((int)((uchar*)Bs)[(((((k_1 & 1) * 512) + (i_1 * 16)) + (((int)threadIdx.x) >> 4)) - 16)]) << 7);
        bfloat16_t scb = (*(bfloat16_t *)(&(v_)));
        ushort v__1 = (ushort)(p & (uint)65535);
        ((bfloat16_t*)B_sh)[((((((((k_1 & 1) * 16384) + (i_1 * 512)) + (((((int)threadIdx.x) & 255) >> 5) * 64)) + (((((((int)threadIdx.x) & 255) >> 7) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))] = ((*(bfloat16_t *)(&(v__1))) * scb);
        ushort v__2 = (ushort)(p >> (uint)16);
        ((bfloat16_t*)B_sh)[(((((((((k_1 & 1) * 16384) + (i_1 * 512)) + (((((int)threadIdx.x) & 255) >> 5) * 64)) + (((((((int)threadIdx.x) & 255) >> 7) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 1)] = ((*(bfloat16_t *)(&(v__2))) * scb);
      }
      mbarrier[((k_1 & 1) + 6)].arrive();
      mbarrier[(k_1 & 1)].wait(((k_1 & 3) >> 1));
      {
        bfloat16_t A_local[8];
        bfloat16_t B_local[16];
        tl::__sync_thread_partial(3, 256);
        for (int ki = 0; ki < 4; ++ki) {
          tl::ptx_ldmatrix_x4((&(((bfloat16_t*)A_sh)[((((((k_1 & 1) * 1024) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local[0])));
          #pragma unroll
          for (int i_2 = 0; i_2 < 2; ++i_2) {
            tl::ptx_ldmatrix_x4((&(((bfloat16_t*)B_sh)[(((((((((k_1 & 1) * 16384) + (((((int)threadIdx.x) & 255) >> 5) * 2048)) + (i_2 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local[(i_2 * 8)])));
          }
          for (int j = 0; j < 2; ++j) {
            tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_l + (j * 8)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + (j * 8)));
            tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_l + ((j * 8) + 4)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + ((j * 8) + 4)));
          }
        }
      }
      mbarrier[((k_1 & 1) + 4)].arrive();
    }
    #pragma unroll
    for (int i_3 = 0; i_3 < 8; ++i_3) {
      *(float2*)(C + ((((((((((int)blockIdx.y) * 81920) + ((i_3 & 1) * 40960)) + (((((int)threadIdx.x) & 31) >> 2) * 5120)) + (((int)blockIdx.x) * 256)) + ((((int)threadIdx.x) >> 5) * 32)) + ((i_3 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 256)) = *(float2*)(C_l + (i_3 * 2));
    }
  }
}

