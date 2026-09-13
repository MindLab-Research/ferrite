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

extern "C" __global__ void main_kernel(__grid_constant__ const CUtensorMap A_desc, float* __restrict__ C, const int* __restrict__ Eid, __grid_constant__ const CUtensorMap W_desc);
extern "C" __global__ void __launch_bounds__(384, 1) main_kernel(__grid_constant__ const CUtensorMap A_desc, float* __restrict__ C, const int* __restrict__ Eid, __grid_constant__ const CUtensorMap W_desc) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* B_sh = ((void*)((char*)buf_dyn_shmem + 0));
  void* A_sh = ((void*)((char*)buf_dyn_shmem + 98304));
  __shared__ __align__(16) uint64_t mbarrier_mem[6];
  auto mbarrier = reinterpret_cast<Barrier*>(mbarrier_mem);
  float C_l[16];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(A_desc);
    tl::prefetch_tma_descriptor(W_desc);
  }
  if (tl::tl_shuffle_elect<0>()) {
    mbarrier[0].init(1);
    mbarrier[1].init(1);
    mbarrier[2].init(1);
    mbarrier[3].init(256);
    mbarrier[4].init(256);
    mbarrier[5].init(256);
  }
  tl::fence_barrier_init();
  __syncthreads();
  if (((int)threadIdx.x) < 128) {
    tl::warpgroup_reg_dealloc<24>();
    int e = Eid[((int)blockIdx.y)];
    for (int k = 0; k < 80; ++k) {
      mbarrier[((k % 3) + 3)].wait((((k % 6) / 3) ^ 1));
      if (tl::tl_shuffle_elect<128>()) {
        mbarrier[(k % 3)].expect_transaction(2048);
        tl::tma_load(A_desc, mbarrier[(k % 3)], (&(((bfloat16_t*)A_sh)[((k % 3) * 1024)])), (k * 64), (((int)blockIdx.y) * 16));
        mbarrier[(k % 3)].arrive_and_expect_tx(32768);
        tl::tma_load(W_desc, mbarrier[(k % 3)], (&(((bfloat16_t*)B_sh)[((k % 3) * 16384)])), (k * 64), (((int)blockIdx.x) * 256), e);
      }
    }
  } else {
    tl::warpgroup_reg_alloc<240>();
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      float broadcast_var = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(C_l + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
    }
    for (int k_1 = 0; k_1 < 80; ++k_1) {
      mbarrier[(k_1 % 3)].wait(((k_1 % 6) / 3));
      {
        bfloat16_t A_local[8];
        bfloat16_t B_local[16];
        for (int ki = 0; ki < 4; ++ki) {
          tl::ptx_ldmatrix_x4((&(((bfloat16_t*)A_sh)[((((((k_1 % 3) * 1024) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(A_local[0])));
          #pragma unroll
          for (int i_1 = 0; i_1 < 2; ++i_1) {
            tl::ptx_ldmatrix_x4((&(((bfloat16_t*)B_sh)[(((((((((k_1 % 3) * 16384) + ((((((int)threadIdx.x) >> 5) + 4) & 7) * 2048)) + (i_1 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])), (&(B_local[(i_1 * 8)])));
          }
          for (int j = 0; j < 2; ++j) {
            tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_l + (j * 8)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + (j * 8)));
            tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(C_l + ((j * 8) + 4)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + ((j * 8) + 4)));
          }
        }
      }
      mbarrier[((k_1 % 3) + 3)].arrive();
    }
    if (((((int)threadIdx.x) >> 8) + ((int)blockIdx.x)) < 3) {
      #pragma unroll
      for (int i_2 = 0; i_2 < 8; ++i_2) {
        *(float2*)(C + ((((((((((int)blockIdx.y) * 10240) + ((i_2 & 1) * 5120)) + (((((int)threadIdx.x) & 31) >> 2) * 640)) + (((int)blockIdx.x) * 256)) + ((((int)threadIdx.x) >> 5) * 32)) + ((i_2 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) - 128)) = *(float2*)(C_l + (i_2 * 2));
      }
    }
  }
}

