// dump_proj_old.cu — 用**确定性 hash 输入**跑老 ferrite 投影族 kernel，把输出 dump 成 f32 裸文件。
//
// 目的：给 TileLang phase-2 提供「route A vs 真实 ferrite kernel」的逐形状 maxrel/meanrel
//       —— 这是 EAGER 对照真正需要的口径（proto.md §4 只对到 f64 真值 / f32 参考式）。
//
// 输入生成必须与 python 侧 `phase2._hash_idx()` **逐字节一致**（无共享流，每个数组独立从 i=0 起）：
//   h(i) = (uint32)( ( (uint64)i * 0x9E3779B97F4A7C15 + 0x0123456789ABCDEF ) >> 32 )
//   a/w  : r = h(i);  b = (r % 127) | ((r>>20 & 1) << 7)   -> e4m3 字节（0..126 与 0x80..0xFE，全有限）
//   a_scale: 0.5f + (h(i) % 8) * 0.125f                    -> 二进制精确的 f32
//   w_scale: 0x7B + (h(i) % 6)                             -> ue8m0 = 2^-4 .. 2^1（避开 0xFF=NaN）
//
// 编译： nvcc -O3 -std=c++17 -gencode arch=compute_103a,code=sm_103a dump_proj_old.cu -o dump_proj_old
// 运行： ./dump_proj_old <outdir>
#include "dsv41_kernels.cu"
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <string>

static inline uint32_t h(uint64_t i) {
    uint64_t x = i * 0x9E3779B97F4A7C15ull + 0x0123456789ABCDEFull;
    return (uint32_t)(x >> 32);
}
static void fill_bytes(std::vector<uint8_t>& v) {
    for (size_t i = 0; i < v.size(); ++i) {
        const uint32_t r = h(i);
        v[i] = (uint8_t)((r % 127u) | (((r >> 20) & 1u) << 7));
    }
}
static void fill_scales(std::vector<float>& v) {
    for (size_t i = 0; i < v.size(); ++i) v[i] = 0.5f + (float)(h(i) % 8u) * 0.125f;
}
static void fill_wsc(std::vector<uint8_t>& v) {
    for (size_t i = 0; i < v.size(); ++i) v[i] = (uint8_t)(0x7B + (h(i) % 6u));
}

static void dump(const std::string& path, const float* dev, size_t n) {
    std::vector<float> hh(n);
    cudaMemcpy(hh.data(), dev, n * sizeof(float), cudaMemcpyDeviceToHost);
    FILE* f = fopen(path.c_str(), "wb");
    if (f) { fwrite(hh.data(), sizeof(float), n, f); fclose(f); }
    printf("  wrote %s (%zu f32)\n", path.c_str(), n);
}

int main(int argc, char** argv) {
    const std::string dir = (argc > 1) ? argv[1] : ".";
    const int m = 6;

    struct Shape { const char* name; int n, k; };
    struct Shape shapes[] = {
        {"wkv",   512, 5120}, {"wq_a", 1280, 5120},
        {"wq_b", 4096, 1280}, {"wo_b", 5120, 1024},
    };
    for (auto& s : shapes) {
        const int n = s.n, k = s.k;
        std::vector<uint8_t> a((size_t)m*(size_t)k), w((size_t)n*(size_t)k),
                             wsc((size_t)(n/32)*(size_t)(k/32));
        std::vector<float>   asc((size_t)m*(size_t)(k/32));
        fill_bytes(a); fill_scales(asc); fill_bytes(w); fill_wsc(wsc);
        uint8_t *da,*dw,*dws; float *dasc,*dout;
        cudaMalloc(&da, a.size()); cudaMalloc(&dw, w.size()); cudaMalloc(&dws, wsc.size());
        cudaMalloc(&dasc, asc.size()*4); cudaMalloc(&dout, (size_t)m*(size_t)n*4);
        cudaMemcpy(da,a.data(),a.size(),cudaMemcpyHostToDevice);
        cudaMemcpy(dw,w.data(),w.size(),cudaMemcpyHostToDevice);
        cudaMemcpy(dws,wsc.data(),wsc.size(),cudaMemcpyHostToDevice);
        cudaMemcpy(dasc,asc.data(),asc.size()*4,cudaMemcpyHostToDevice);
        int rc = dsv41_gemm_fp8_mrows(da,dasc,dw,dws,nullptr,dout,m,n,k,n,nullptr);
        printf("%-6s rc=%d\n", s.name, rc);
        dump(dir + "/old_" + s.name + ".f32", dout, (size_t)m*(size_t)n);
        cudaFree(da);cudaFree(dw);cudaFree(dws);cudaFree(dasc);cudaFree(dout);
    }

    struct WoaShape { const char* name; int groups, n, k, a_stride; };
    struct WoaShape woa[] = {
        {"wo_a_nlg1", 1, 1024, 4096, 4096},
        {"wo_a_nlg8", 8, 1024, 4096, 32768},
    };
    for (auto& s : woa) {
        const int G = s.groups, n = s.n, k = s.k, ASTR = s.a_stride;
        std::vector<uint8_t> a((size_t)m*(size_t)ASTR), w((size_t)G*(size_t)n*(size_t)k),
                             wsc((size_t)(G*n/32)*(size_t)(k/32));
        std::vector<float>   asc((size_t)m*(size_t)(ASTR/32));
        fill_bytes(a); fill_scales(asc); fill_bytes(w); fill_wsc(wsc);
        uint8_t *da,*dw,*dws; float *dasc,*dout;
        cudaMalloc(&da, a.size()); cudaMalloc(&dw, w.size()); cudaMalloc(&dws, wsc.size());
        cudaMalloc(&dasc, asc.size()*4); cudaMalloc(&dout, (size_t)m*(size_t)(G*n)*4);
        cudaMemcpy(da,a.data(),a.size(),cudaMemcpyHostToDevice);
        cudaMemcpy(dw,w.data(),w.size(),cudaMemcpyHostToDevice);
        cudaMemcpy(dws,wsc.data(),wsc.size(),cudaMemcpyHostToDevice);
        cudaMemcpy(dasc,asc.data(),asc.size()*4,cudaMemcpyHostToDevice);
        int rc = dsv41_wo_a_grouped_fp8(da,dasc,dw,dws,nullptr,dout,G,m,n,k,ASTR,G*n,nullptr);
        printf("%-10s rc=%d\n", s.name, rc);
        dump(dir + "/old_" + s.name + ".f32", dout, (size_t)m*(size_t)(G*n));
        cudaFree(da);cudaFree(dw);cudaFree(dws);cudaFree(dasc);cudaFree(dout);
    }
    auto e = cudaGetLastError();
    printf("cudaGetLastError: %s\n", cudaGetErrorString(e));
    return 0;
}
