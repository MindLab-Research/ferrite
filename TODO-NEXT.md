# ferrite GLM-5.3-Flash 部署 — 当前状态与下一步（2026-09-04）

## 当前状态
- **GPU TP8 推理链路全通**：`LD_LIBRARY_PATH=kernels/cuda ./target/release/ferrite-serve --model-dir /opt/dlami/nvme/models/GLM-5.3-Flash --backend cuda --tp 8`
- 加载 37330 tensors / 80s；生成 ~0.11 tok/s（f32 + CPU all-reduce + per-op H2D/D2H，未优化）
- **输出仍乱码**（"appa \ \ \..."）— DSA(kpool)/MoE 段数值未对齐
- cargo test 18 个套件全绿；GPU 13 op smoke 对齐 CPU

## 已数值验证对齐（vs transformers 公式，maxdiff 2.3e-7）
- layer 0 全链（GDN KDA + MHC + dense FFN）
- GDN：decay=lb*sigmoid(exp(A_log)*(fb+dt_bias))、conv SiLU、q/k L2norm+q×K^-0.5、o_norm sigmoid(gate)
- MHC：hc_post comb 转置、ffn 段 post_attention_layernorm
- MoE：sigmoid(logits)+bias 选、renorm(raw sigmoid)、swiglu gate 只 clamp max

## 剩余问题（乱码根因，按优先级）
1. **DSA 层 attn 输出偏差**（ref l2 88.9 vs fer 141；token0 应为 o_proj(v[0])=-0.83 而 fer=0.23）— kpool fallback 修复后未重验。诊断：FERRITE_PROBE=1 跑一次 → /tmp/l0_*.f32（layer 3 dump）+ /tmp/l3_dsa_idx.f32，对比 transformers 公式（per-channel softmax(gate+ape) 加权 pool → pool 级 top-k → 展开+tail → sparse attn）
2. **MoE 段未对比**（ffn 输出 ref vs fer）
3. kpool ctx0 语义：indexer_topk 的 ctx0 传 `ctx0/kpool`（floor）在 pool 边界可能差 1（正确为 pool p 可见 iff (p+1)*kpool-1 <= ctx0+i，即 jmax=ceil((ctx0+i+1)/kpool)）

## 诊断方法（已验证有效）
- FERRITE_PROBE=1 → dump layer0/layer3 中间量（l0_hn/attn/ffn/in/out, l3_dsa_*）
- python 直读 safetensors 复现 transformers 公式对比（先 curl 恢复 /tmp/glm5_next.py）
- CUDA_LAUNCH_BLOCKING=1 定位 err700 的 kernel

## 已修 bug 清单（详见 git diff / 归档记忆 73bee581）
checkpoint: data_base（safetensors 偏移是 8+hlen）；rayon 并行加载
CUDA: sm_103a、dlopen flags、cudaSetDevice per-op（enter()）、matmul sw 转置、rmsnorm warp 竞写、conv1d smem race、moe_route int/float、sparse_attn smem=0/block>32/t 参数/-1 skip、WYF 弃用
Engine: KDA 公式全套、hc_post 转置、TP ffn norm、MoE bias、dt_bias、indexer causal+relu+scale+kpool、DsaLayerCache.k_gate
