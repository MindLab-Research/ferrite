// Run OUR dsv41_sparse_attn on the ENGINE'S OWN dumped inputs (kinds 31/32/34/50)
// and compare the result with the engine's own kind 51.
//
// Purpose: prove the harness context is faithful. If our kernel on the engine's
// real inputs reproduces the engine's kind 51 bit-for-bit, then the harness is a
// valid instrument and any remaining difference against the official is carried
// by the INPUTS (kv ring / idxs), not by the operator.
//
// Fixed record table (ours): 8B pos | 8B kind | n x 4B, n from the table.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>

extern "C" int dsv41_sparse_attn(const float*, const float*, const float*, const int32_t*, float*,
                                 int, int, int, int, const int*, int, int, float, cudaStream_t);

static std::vector<unsigned char> rd(const char* p) {
    FILE* f = fopen(p, "rb");
    if (!f) { printf("cannot open %s\n", p); exit(1); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<unsigned char> v(n);
    if ((long)fread(v.data(), 1, n, f) != n) { printf("short read\n"); exit(1); }
    fclose(f);
    return v;
}
static inline unsigned u32(float x) { unsigned u; __builtin_memcpy(&u, &x, 4); return u; }

int main(int argc, char** argv) {
    if (argc < 2) { printf("usage: t_mine <our_dump.bin> [step]\n"); return 1; }
    const int S = 15, H = 8, D = 512;
    auto buf = rd(argv[1]);
    int SZ[64]; memset(SZ, 0, sizeof(SZ));
    SZ[0]=5120;SZ[1]=5120;SZ[2]=5120;SZ[3]=6;SZ[4]=6;SZ[5]=512;SZ[6]=512;SZ[7]=160;SZ[8]=320;
    SZ[9]=32;SZ[10]=64;SZ[11]=64;SZ[12]=16;SZ[13]=16;SZ[20]=40;SZ[21]=4;SZ[22]=384;SZ[23]=5120;
    SZ[31]=16384;SZ[32]=640;SZ[34]=4;SZ[40]=5120;SZ[41]=5120;SZ[42]=16160;SZ[50]=4096;SZ[51]=4096;SZ[52]=1024;SZ[53]=5120;
    for (int i=14;i<20;i++) SZ[i]=32;

    // gather per-step q / kv-ring / idxs / kind51
    std::vector<std::vector<float> > q, kv, o51;
    std::vector<std::vector<int> > ix;
    std::vector<std::vector<float> > scs;   // per-step [win, index_topk, scale, clen]
    float sc[4] = {0,0,0,0};
    size_t off = 0;
    while (off + 16 <= buf.size()) {
        long long pos, kind;
        memcpy(&pos, buf.data()+off, 8); memcpy(&kind, buf.data()+off+8, 8); off += 16;
        if (kind < 0 || kind >= 64 || SZ[kind] == 0) { printf("stop: kind %lld at pos %lld\n", kind, pos); break; }
        size_t nb = (size_t)SZ[kind]*4;
        const float* fp = (const float*)(buf.data()+off);
        const int* ip = (const int*)(buf.data()+off);
        if (kind == 34) {
            memcpy(sc, buf.data()+off, 16);
            if ((size_t)pos >= scs.size()) scs.resize(pos+1);
            scs[pos].assign(sc, sc+4);
        }
        if ((size_t)pos >= q.size()) { q.resize(pos+1); kv.resize(pos+1); ix.resize(pos+1); o51.resize(pos+1); }
        if (kind == 50) q[pos].assign(fp, fp+SZ[50]);
        if (kind == 31) kv[pos].assign(fp, fp+SZ[31]);
        if (kind == 32) ix[pos].assign(ip, ip+SZ[32]);
        if (kind == 51) o51[pos].assign(fp, fp+SZ[51]);
        off += nb;
    }
    printf("steps: q=%zu kv=%zu ix=%zu o51=%zu  sc=[%.0f,%.0f,%.9f,%.0f]\n",
           q.size(), kv.size(), ix.size(), o51.size(), sc[0], sc[1], sc[2], sc[3]);
    const int win = (int)sc[0], itk = (int)sc[1];
    const int per = (win + itk <= 64) ? win + itk : 64;   // our idx dump holds 64 entries
    const int step = (argc > 2) ? atoi(argv[2]) : 1;
    if (step >= (int)q.size() || q[step].empty()) { printf("step %d unavailable\n", step); return 1; }

    // one call: 1 row (m=1), H heads, D dims, with our window/index_topk and our idxs
    std::vector<float> qq = q[step];
    // kv: the ring holds 32 rows; the official-style call needs the rows the idxs point at,
    // so pass the whole ring and use our own idx layout (matching the engine's call).
    std::vector<float> kvv = kv[step];
    std::vector<int> idd(1 * (win + itk), -1);
    for (int j = 0; j < per && j < (win + itk); j++) idd[j] = ix[step][j];
    std::vector<int> cl(1, (int)((scs.size() > (size_t)step && scs[step].size() == 4) ? scs[step][3] : sc[3]));

    float *dq,*dkv,*dsink,*dout; int32_t* didxs; int* dclen;
    cudaMalloc(&dq, qq.size()*4); cudaMalloc(&dkv, kvv.size()*4); cudaMalloc(&dsink, H*4);
    cudaMalloc(&dout, (size_t)S*H*D*4); cudaMalloc(&didxs, idd.size()*4); cudaMalloc(&dclen, cl.size()*4);
    cudaMemcpy(dq, qq.data(), qq.size()*4, cudaMemcpyHostToDevice);
    cudaMemcpy(dkv, kvv.data(), kvv.size()*4, cudaMemcpyHostToDevice);
    std::vector<float> sink = {-0.09726219f,-0.01387484f,0.07686966f,-0.04399407f,-0.19732477f,0.12044567f,-0.25218663f,-0.03711864f};
    cudaMemcpy(dsink, sink.data(), H*4, cudaMemcpyHostToDevice);
    cudaMemcpy(didxs, idd.data(), idd.size()*4, cudaMemcpyHostToDevice);
    cudaMemcpy(dclen, cl.data(), cl.size()*4, cudaMemcpyHostToDevice);
    cudaMemset(dout, 0, (size_t)H*D*4);
    int rc = dsv41_sparse_attn(dq, dkv, dsink, didxs, dout, 1, 1, H, D, dclen, win, itk, sc[2], 0);
    printf("rc=%d sync=%s\n", rc, cudaGetErrorString(cudaDeviceSynchronize()));
    std::vector<float> got((size_t)H*D);
    cudaMemcpy(got.data(), dout, (size_t)H*D*4, cudaMemcpyDeviceToHost);
    int bd = 0;
    for (int i = 0; i < H*D; i++) if (u32(got[i]) != u32(o51[step][i])) bd++;
    printf("harness vs engine kind51 (step %d): bitdiff=%d/%d %s\n", step, bd, H*D, bd==0 ? "HARNESS-FAITHFUL" : "");
    printf("got[:4]=%g %g %g %g   eng[:4]=%g %g %g %g\n", got[0],got[1],got[2],got[3], o51[step][0],o51[step][1],o51[step][2],o51[step][3]);
    return 0;
}
