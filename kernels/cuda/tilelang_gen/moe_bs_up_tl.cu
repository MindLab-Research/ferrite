#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/instruction/tcgen05mma.h>
#include <tl_templates/cuda/tcgen_05.h>
#include <tl_templates/cuda/intrin.h>
#include <tl_templates/cuda/barrier.h>
#include <tl_templates/cuda/copy_sm90.h>
#include <tl_templates/cuda/copy_sm100.h>
#include <tl_templates/cuda/cuda_fp8.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/scan.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void main_kernel(__grid_constant__ const CUtensorMap A_desc, __grid_constant__ const CUtensorMap C_desc, const int* __restrict__ Eid, const uint* __restrict__ SFA, __grid_constant__ const CUtensorMap SFW1_desc, __grid_constant__ const CUtensorMap SFW3_desc, __grid_constant__ const CUtensorMap W1_desc, __grid_constant__ const CUtensorMap W3_desc);
extern "C" __global__ void __launch_bounds__(128, 1) main_kernel(__grid_constant__ const CUtensorMap A_desc, __grid_constant__ const CUtensorMap C_desc, const int* __restrict__ Eid, const uint* __restrict__ SFA, __grid_constant__ const CUtensorMap SFW1_desc, __grid_constant__ const CUtensorMap SFW3_desc, __grid_constant__ const CUtensorMap W1_desc, __grid_constant__ const CUtensorMap W3_desc) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* A_sh = ((void*)((char*)buf_dyn_shmem + 0));
  void* C_sh = ((void*)((char*)buf_dyn_shmem + 0));
  void* B_sh = ((void*)((char*)buf_dyn_shmem + 49152));
  void* SFA_sh = ((void*)((char*)buf_dyn_shmem + 98304));
  void* SFW_sh = ((void*)((char*)buf_dyn_shmem + 99840));
  __shared__ __align__(16) uint64_t loaded_mem[3];
  auto loaded = reinterpret_cast<Barrier*>(loaded_mem);
  __shared__ __align__(16) uint64_t sf_full_mem[3];
  auto sf_full = reinterpret_cast<Barrier*>(sf_full_mem);
  __shared__ __align__(16) uint64_t consumed_mem[3];
  auto consumed = reinterpret_cast<Barrier*>(consumed_mem);
  __shared__ __align__(16) uint64_t tmem_full_mem[1];
  auto tmem_full = reinterpret_cast<Barrier*>(tmem_full_mem);
  __shared__ __align__(16) uint C_tmem[1];
  __shared__ __align__(16) uint sfa_data[1];
  float C_l[128];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(A_desc);
    tl::prefetch_tma_descriptor(W1_desc);
    tl::prefetch_tma_descriptor(W3_desc);
    tl::prefetch_tma_descriptor(SFW1_desc);
    tl::prefetch_tma_descriptor(SFW3_desc);
    tl::prefetch_tma_descriptor(C_desc);
  }
  if (tl::tl_shuffle_elect<0>()) {
    loaded[0].init(32);
    loaded[1].init(32);
    loaded[2].init(32);
    sf_full[0].init(32);
    sf_full[1].init(32);
    sf_full[2].init(32);
    consumed[0].init(1);
    consumed[1].init(1);
    consumed[2].init(1);
    tmem_full[0].init(1);
  }
  tl::fence_barrier_init();
  tl::tcgen05_before_thread_sync();
  __syncthreads();
  tl::tcgen05_after_thread_sync();
  if ((((int)threadIdx.x) >> 5) == 0) {
    tl::tmem_allocate((&(C_tmem[0])), 128);
    tl::tmem_allocate((&(sfa_data[0])), 32);
    // ROOT-CAUSE FIX (2026-09-14, illegal-instr-5): PTX ISA requires the CTA to
    // relinquish its TMEM allocation permit BEFORE tcgen05.dealloc. TileLang
    // 0.1.14's codegen omits this call entirely (tl_templates has no such
    // function), so the generated kernel's dealloc at the tail violates the ISA
    // precondition and the tcgen05 unit traps with "illegal instruction".
    // Verified convention in this repo: tests_tcgen05_mxf8f6f4_1x.cu:874-875
    // and dsv41_experts_mxf4.cu:327-332 both call relinquish right after alloc.
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
  }
  tl::tcgen05_before_thread_sync();
  __syncthreads();
  tl::tcgen05_after_thread_sync();
  int e = Eid[((int)blockIdx.y)];
  if (((int)threadIdx.x) < 32) {
    for (int g = 0; g < 40; ++g) {
      consumed[(g % 3)].wait((((g / 3) & 1) ^ 1));
      if (tl::tl_shuffle_elect<32>()) {
        loaded[(g % 3)].expect_transaction(16384);
        tl::tma_load(A_desc, loaded[(g % 3)], (&(((fp8_e4_t*)A_sh)[((g % 3) * 16384)])), (g * 128), (((int)blockIdx.y) * 128));
        loaded[(g % 3)].expect_transaction(4096);
        tl::tma_load(W1_desc, loaded[(g % 3)], (&(((uint8_t*)B_sh)[((g % 3) * 16384)])), (g * 128), (((int)blockIdx.x) * 64), e);
        loaded[(g % 3)].expect_transaction(4096);
        tl::tma_load(W3_desc, loaded[(g % 3)], (&(((uint8_t*)B_sh)[(((g % 3) * 16384) + 8192)])), (g * 128), (((int)blockIdx.x) * 64), e);
        loaded[(g % 3)].expect_transaction(512);
        tl::tma_load((&(((uint*)SFA_sh)[((g % 3) * 128)])), (&(SFA[((g * 4608) + (((int)blockIdx.y) * 128))])), loaded[(g % 3)], 512);
        loaded[(g % 3)].expect_transaction(256);
        tl::tma_load(SFW1_desc, loaded[(g % 3)], (&(((uint*)SFW_sh)[((g % 3) * 128)])), ((g * 320) + (((int)blockIdx.x) * 64)), e);
        loaded[(g % 3)].expect_transaction(256);
        tl::tma_load(SFW3_desc, loaded[(g % 3)], (&(((uint*)SFW_sh)[(((g % 3) * 128) + 64)])), ((g * 320) + (((int)blockIdx.x) * 64)), e);
      }
      loaded[(g % 3)].arrive();
    }
  } else {
    if (((int)threadIdx.x) < 64) {
      for (int k = 0; k < 40; ++k) {
        loaded[(k % 3)].wait(((k / 3) & 1));
        sf_full[(k % 3)].wait(((k / 3) & 1));
        tl::tcgen05_after_thread_sync();
        void* chunk_ptr = (&(((uint*)SFA_sh)[((k % 3) * 128)]));
        tl::tcgen05_cp<false>(tl::make_sf_smem_desc(reinterpret_cast<void*>(chunk_ptr)), (*reinterpret_cast<uint32_t*>(sfa_data)) + 0);
        void* chunk_ptr_1 = (&(((uint*)SFW_sh)[((k % 3) * 128)]));
        tl::tcgen05_cp<false>(tl::make_sf_smem_desc(reinterpret_cast<void*>(chunk_ptr_1)), (*reinterpret_cast<uint32_t*>(sfa_data)) + 4);
        {
          tl::Tcgen05SMemDescriptor desc_a;
          tl::Tcgen05SMemDescriptor desc_b;
          tl::initialize_tcgen05_descriptor(desc_a, (&(((fp8_e4_t*)A_sh)[0])), 1, 64, 0, 0, 2);
          tl::increase_descriptor_offset<int>(desc_a, ((k % 3) * 16384));
          tl::initialize_tcgen05_descriptor(desc_b, (&(((uint8_t*)B_sh)[0])), 1, 64, 0, 0, 2);
          tl::increase_descriptor_offset<int>(desc_b, ((k % 3) * 16384));
          #pragma unroll
          for (int ki = 0; ki < 4; ++ki) {
            tl::tcgen05mma_blockscaled_ss<tl::DataType::kFloat8_e4m3, false>(uint64_t(desc_a + (ki * 32)), uint64_t(desc_b + (ki * 32)), (*reinterpret_cast<uint32_t*>(C_tmem)) + 0, ((0 < ki) ? 1 : ((k == 0) ? 0 : 1)), static_cast<uint32_t>(144708608), (*reinterpret_cast<uint32_t*>(sfa_data)) + 0, (*reinterpret_cast<uint32_t*>(sfa_data)) + 4);
          }
          tl::tcgen05_mma_arrive((&(consumed[(k % 3)])));
        }
      }
      tl::tcgen05_mma_arrive((&(tmem_full[0])));
    } else {
      if (((int)threadIdx.x) < 96) {
        for (int k_1 = 0; k_1 < 40; ++k_1) {
          loaded[(k_1 % 3)].wait(((k_1 / 3) & 1));
          tl::tcgen05_before_thread_sync();
          tl::__sync_thread_partial(3, 32);
          tl::tcgen05_after_thread_sync();
          void* chunk_ptr_2 = (&(((uint*)SFA_sh)[((k_1 % 3) * 128)]));
          tl::tcgen05_sf_warp_transpose(reinterpret_cast<uint32_t*>(chunk_ptr_2));
          void* chunk_ptr_3 = (&(((uint*)SFW_sh)[((k_1 % 3) * 128)]));
          tl::tcgen05_sf_warp_transpose(reinterpret_cast<uint32_t*>(chunk_ptr_3));
          tl::fence_proxy_async();
          sf_full[(k_1 % 3)].arrive();
        }
      }
    }
  }
  tmem_full[0].wait(0);
  tl::tcgen05_before_thread_sync();
  __syncthreads();
  tl::tcgen05_after_thread_sync();
  tl::tcgen05_ld_32dp32bNx<128, false>(C_tmem[0], 0, (&(C_l[0])));
  tl::tcgen05_before_thread_sync();
  __syncthreads();
  tl::tcgen05_after_thread_sync();
  #pragma unroll
  for (int i = 0; i < 32; ++i) {
    *(float4*)(((float*)C_sh) + ((((((i >> 3) * 4096) + (((int)threadIdx.x) * 32)) + (((((i & 7) >> 2) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + (((((i & 3) >> 1) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + ((((i & 1) + (((int)threadIdx.x) & 1)) & 1) * 4))) = *(float4*)(C_l + (i * 4));
  }
  tl::tcgen05_before_thread_sync();
  __syncthreads();
  tl::tcgen05_after_thread_sync();
  if (tl::tl_shuffle_elect<128>()) {
    tl::fence_proxy_async();
    #pragma unroll
    for (int i_1 = 0; i_1 < 4; ++i_1) {
      tl::tma_store(C_desc, (&(((float*)C_sh)[(i_1 * 4096)])), ((((int)blockIdx.x) * 128) + (i_1 * 32)), (((int)blockIdx.y) * 128));
    }
    tl::tma_store_arrive();
    tl::tma_store_wait<0, true>();
  }
  if ((((int)threadIdx.x) >> 5) == 0) {
    tl::tmem_deallocate((&(C_tmem[0])), 128);
    tl::tmem_deallocate((&(sfa_data[0])), 32);
  }
}

