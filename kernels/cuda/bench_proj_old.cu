// bench_proj_old.cu — 老 ferrite 投影族 micro-bench（五形状），TileLang 第二阶段的对照侧。
//
// 独立 TU：`#include "dsv41_kernels.cu"`，直接调 launcher（不是重写 kernel），CUDA event 计时。
//   �密四形状：dsv41_gemm_fp8_mx (m=1 eager gemv) / dsv41_gemm_fp8_mrows (m=1 / m=6 verify)
//   wo_a 分组：dsv41_wo_a_grouped_fp8 (rows=1 / 6)，含 G=1(verify@TP8) 与 G=8(draft@TP1)
//              —— 另外给「per-(group,row) 的 gemm_fp8_mx 回退」同口径对照
//
// 形状（以代码事实为准，见 kernels/cuda/dsv41_kernels.cu:8165 / chain_dev.rs:13270）：
//   wkv   n=512  k=5120 | wq_a n=1280 k=5120 | wq_b n=4096 k=1280 | wo_b n=5120 k=1024
//   wo_a  G=nlg  n=o_lora_rank=1024  k=hpg*head_dim=4096  a_stride=nlh*head_dim
//         verify@TP8: G=1, a_stride=4096 ; draft@TP1: G=8, a_stride=32768
//
// 编译： nvcc -O3 -std=c++17 -gencode arch=compute_103a,code=sm_103a bench_proj_old.cu -o bench_proj_old
#include "dsv41_kernels.cu"
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>

static float t_us(void (*fn)(void*), void* ctx, int iters) {
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaDeviceSynchronize();
    fn(ctx);
    cudaDeviceSynchronize();
    cudaEventRecord(a);
    for (int i = 0; i < iters; ++i) fn(ctx);
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms = 0; cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return ms * 1000.0f / iters;
}

static float t_us_empty(int iters) {
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaDeviceSynchronize();
    cudaEventRecord(a);
    for (int i = 0; i < iters; ++i) {}
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms = 0; cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return ms * 1000.0f / iters;
}

struct Ctx { const uint8_t* a; const float* asc; const uint8_t* w; const uint8_t* wsc;
             const float* bias; float* out; int m, n, k, os; };

static void run_mx(void* p) {
    Ctx* c = (Ctx*)p;
    dsv41_gemm_fp8_mx(c->a, c->asc, c->w, c->wsc, c->bias, c->out, c->m, c->n, c->k, nullptr);
}
static void run_mrows(void* p) {
    Ctx* c = (Ctx*)p;
    dsv41_gemm_fp8_mrows(c->a, c->asc, c->w, c->wsc, c->bias, c->out, c->m, c->n, c->k,
                         c->os, nullptr);
}

// ---- wo_a 分组 ----
struct GCtx { const uint8_t* a; const float* asc; const uint8_t* w; const uint8_t* wsc;
              float* out; int groups, rows, n, k, a_stride, out_stride; };
static void run_woa(void* p) {
    GCtx* c = (GCtx*)p;
    dsv41_wo_a_grouped_fp8(c->a, c->asc, c->w, c->wsc, nullptr, c->out, c->groups, c->rows,
                           c->n, c->k, c->a_stride, c->out_stride, nullptr);
}
// 回退口径：每 (group, row) 一次 m=1 的 gemm_fp8_mx
static void run_woa_loop(void* p) {
    GCtx* c = (GCtx*)p;
    for (int r = 0; r < c->rows; ++r) {
        for (int g = 0; g < c->groups; ++g) {
            const uint8_t* a = c->a + (size_t)r * c->a_stride + (size_t)g * c->k;
            const float*  as = c->asc + ((size_t)r * c->a_stride + (size_t)g * c->k) / 32;
            const uint8_t* w = c->w + (size_t)g * c->n * c->k;
            const uint8_t* ws = c->wsc + ((size_t)g * c->n / 32) * (c->k / 32);
            float* o = c->out + (size_t)r * c->out_stride + (size_t)g * c->n;
            dsv41_gemm_fp8_mx(a, as, w, ws, nullptr, o, 1, c->n, c->k, nullptr);
        }
    }
}

struct Shape { const char* name; int n, k; };

int main(int argc, char** argv) {
    int iters = 2000;
    if (argc > 1) iters = atoi(argv[1]);
    const int it6 = iters > 200 ? iters / 4 : iters;
    printf("== legacy ferrite projection micro-bench (iters=%d, m6 iters=%d) ==\n",
           iters, it6);
    printf("launch floor (host-only loop) = %.2f us\n", t_us_empty(iters));

    struct Shape shapes[] = {
        {"wkv",   512, 5120}, {"wq_a", 1280, 5120},
        {"wq_b", 4096, 1280}, {"wo_b", 5120, 1024},
    };
    printf("\n%-6s %6s %6s | %9s %9s %9s | %8s %8s\n",
           "shape","n","k","mx_m1_us","mrows_m1","mrows_m6","mr6/mr1","mr1/mx1");
    for (auto& s : shapes) {
        const int n = s.n, k = s.k;
        const int nbk = k / 32, nbn = n / 32;
        std::vector<uint8_t> a(8*(size_t)k, 0x38);
        std::vector<float>   asc(8*(size_t)nbk, 1.0f);
        std::vector<uint8_t> w((size_t)n*(size_t)k, 0x38);
        std::vector<uint8_t> wsc((size_t)nbn*(size_t)nbk, 0x7F); // 2^0
        std::vector<float>   out(8*(size_t)n, 0.0f);
        uint8_t *da,*dw,*dws; float *dasc,*dout;
        cudaMalloc(&da, a.size()); cudaMalloc(&dw, w.size()); cudaMalloc(&dws, wsc.size());
        cudaMalloc(&dasc, asc.size()*4); cudaMalloc(&dout, out.size()*4);
        cudaMemcpy(da,a.data(),a.size(),cudaMemcpyHostToDevice);
        cudaMemcpy(dw,w.data(),w.size(),cudaMemcpyHostToDevice);
        cudaMemcpy(dws,wsc.data(),wsc.size(),cudaMemcpyHostToDevice);
        cudaMemcpy(dasc,asc.data(),asc.size()*4,cudaMemcpyHostToDevice);
        Ctx c{da,dasc,dw,dws,nullptr,dout,1,n,k,n};
        float mx1 = t_us(run_mx, &c, iters);
        c.m = 1; float mr1 = t_us(run_mrows, &c, iters);
        c.m = 6; float mr6 = t_us(run_mrows, &c, it6);
        printf("%-6s %6d %6d | %9.2f %9.2f %9.2f | %8.2f %8.2f\n",
               s.name, n, k, mx1, mr1, mr6, mr6/mr1, mr1/mx1);
        cudaFree(da);cudaFree(dw);cudaFree(dws);cudaFree(dasc);cudaFree(dout);
    }

    // ---- wo_a 分组 ----
    // G=1 (verify @TP8, a_stride == k) 与 G=8 (draft @TP1, a_stride = 8k)
    struct WoaShape { const char* name; int groups, n, k, a_stride; };
    struct WoaShape woa[] = {
        {"wo_a_nlg1", 1, 1024, 4096, 4096},
        {"wo_a_nlg8", 8, 1024, 4096, 32768},
    };
    printf("\n%-10s %5s %6s %6s %9s | %9s %9s %9s %9s | %8s %8s\n",
           "shape","G","n","k","a_stride",
           "grp_r1_us","grp_r6_us","loop_r1_us","loop_r6_us","grp6/grp1","grp1/loop1");
    for (auto& s : woa) {
        const int G = s.groups, n = s.n, k = s.k, ASTR = s.a_stride;
        const int rows = 6;
        const int nbk = k / 32;
        std::vector<uint8_t> a((size_t)rows*(size_t)ASTR, 0x38);
        std::vector<float>   asc((size_t)rows*(size_t)(ASTR/32), 1.0f);
        std::vector<uint8_t> w((size_t)G*(size_t)n*(size_t)k, 0x38);
        std::vector<uint8_t> wsc((size_t)(G*n/32)*(size_t)nbk, 0x7F);
        std::vector<float>   out((size_t)rows*(size_t)(G*n), 0.0f);
        uint8_t *da,*dw,*dws; float *dasc,*dout;
        cudaMalloc(&da, a.size()); cudaMalloc(&dw, w.size()); cudaMalloc(&dws, wsc.size());
        cudaMalloc(&dasc, asc.size()*4); cudaMalloc(&dout, out.size()*4);
        cudaMemcpy(da,a.data(),a.size(),cudaMemcpyHostToDevice);
        cudaMemcpy(dw,w.data(),w.size(),cudaMemcpyHostToDevice);
        cudaMemcpy(dws,wsc.data(),wsc.size(),cudaMemcpyHostToDevice);
        cudaMemcpy(dasc,asc.data(),asc.size()*4,cudaMemcpyHostToDevice);
        GCtx g{da,dasc,dw,dws,dout,G,1,n,k,ASTR,G*n};
        int rc1 = dsv41_wo_a_grouped_fp8(da,dasc,dw,dws,nullptr,dout,G,1,n,k,ASTR,G*n,nullptr);
        int rc6 = dsv41_wo_a_grouped_fp8(da,dasc,dw,dws,nullptr,dout,G,6,n,k,ASTR,G*n,nullptr);
        if (rc1 == 0 && rc6 == 0) {
            g.rows = 1; float g1 = t_us(run_woa, &g, iters);
            g.rows = 6; float g6 = t_us(run_woa, &g, it6);
            g.rows = 1; float l1 = t_us(run_woa_loop, &g, iters);
            g.rows = 6; float l6 = t_us(run_woa_loop, &g, it6);
            printf("%-10s %5d %6d %6d %9d | %9.2f %9.2f %9.2f %9.2f | %8.2f %8.2f\n",
                   s.name, G, n, k, ASTR, g1, g6, l1, l6, g6/g1, g1/l1);
        } else {
            printf("%-10s %5d %6d %6d %9d | DECLINED rc=%d/%d (gate/abi) -> loop only\n",
                   s.name, G, n, k, ASTR, rc1, rc6);
            g.rows = 1; float l1 = t_us(run_woa_loop, &g, iters);
            g.rows = 6; float l6 = t_us(run_woa_loop, &g, it6);
            printf("           loop_r1=%.2f loop_r6=%.2f r6/r1=%.2f\n", l1, l6, l6/l1);
        }
        cudaFree(da);cudaFree(dw);cudaFree(dws);cudaFree(dasc);cudaFree(dout);
    }
    auto e = cudaGetLastError();
    printf("cudaGetLastError: %s\n", cudaGetErrorString(e));
    return 0;
}
