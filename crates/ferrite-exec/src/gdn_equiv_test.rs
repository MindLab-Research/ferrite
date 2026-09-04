//! GDN device-chain equivalence: gdn_layer_dev (CUDA) vs the CPU golden
//! path (Engine::linear_attn_forward) — one layer, same weights/input/state
//! origin (fresh), zero tolerance beyond fp accumulation. Pins the numeric
//! bug that produced garbage output ("!!!!") on the serve path.

#![cfg(all(test, feature = "cuda"))]

use std::sync::Arc;

use ferrite_kernel::cuda::{CudaBackend, DevBuf, GdnLayerWeights};
use ferrite_model::{random_weights, Glm53FlashConfig};
use ferrite_types::{DType, Shape, Tensor};

use crate::Engine;

fn close(a: &Tensor, b: &Tensor, tol: f32, what: &str) {
    let (av, bv) = (a.as_slice(), b.as_slice());
    assert_eq!(av.len(), bv.len(), "{what}: len mismatch");
    let mut mx = 0f32;
    for (x, y) in av.iter().zip(bv) {
        mx = mx.max((x - y).abs());
    }
    eprintln!("{what}: max_diff {mx:.3e}");
    assert!(mx < tol, "{what}: max_diff {mx:.3e} >= {tol:.3e}");
}

#[test]
fn gdn_device_chain_matches_cpu() {
    let so = std::env::var("FERRITE_KERNEL_SO").unwrap_or_else(|_| {
        let mut p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        p.push("../../kernels/cuda/libferrite_kernels.so");
        p.to_string_lossy().into_owned()
    });
    let dev = CudaBackend::with_device(&so, 0).expect("open cuda device 0 (run on the GPU box)");

    let cfg = Glm53FlashConfig::test_config();
    let w = random_weights(&cfg, 99);
    let la = &cfg.linear_attn;
    let (h, dk) = (la.num_heads, la.head_dim);
    let hidden = cfg.hidden_size;
    let n = 3;
    let pfx = "model.layers.0";
    let x = Tensor::from_f32(
        Shape::new([n, hidden]),
        (0..n * hidden).map(|i| ((i * 37 % 251) as f32) * 0.01 - 1.2).collect(),
    );

    // CPU golden: the engine's own linear_attn_forward (CpuBackend) —
    // same weights, same input, state starts at zero on both sides.
    let mut eng = crate::Engine::new(cfg.clone(), w.clone(), ferrite_kernel::CpuBackend::new());
    let out_cpu = eng
        .linear_attn_forward(0, 0, pfx, &x, n)
        .expect("cpu gdn forward");

    // Device chain: gdn_layer_dev (fresh backend → fresh state store).
    let q = |name: &str| eng.w(&format!("{pfx}.self_attn.{name}")).unwrap().clone();
    let gw = GdnLayerWeights {
        qkv_proj: &q("qkv_proj.weight"),
        b_proj: &q("b_proj.weight"),
        f_a: &q("f_a_proj.weight"),
        f_b: &q("f_b_proj.weight"),
        g_a: &q("g_a_proj.weight"),
        g_b: &q("g_b_proj.weight"),
        conv_w: &q("qkv_conv1d.weight"),
        dt_bias: &q("dt_bias"),
        a_log: &q("A_log"),
        o_norm: &q("o_norm.weight"),
        o_proj: &q("o_proj.weight"),
    };
    // THREE consecutive calls with different inputs — the device chain's
    // resident states (conv tails, gdn state) must accumulate across calls
    // exactly like the CPU path's HashMaps. (The single-call test passed
    // with diff 0; the serve garbage points at cross-call state carry.)
    for round in 0..3 {
        let x = Tensor::from_f32(
            Shape::new([n, hidden]),
            (0..n * hidden)
                .map(|i| ((i * 37 + round * 101 % 251) as f32) * 0.01 - 1.2)
                .collect(),
        );
        let out_cpu = eng
            .linear_attn_forward(0, 0, pfx, &x, n)
            .unwrap_or_else(|e| panic!("cpu round {round}: {e}"));
        let x_dev = DevBuf::alloc(dev.dev(), dev.stream(), x.numel()).expect("alloc");
        x_dev.upload(x.as_slice()).expect("upload");
        let partial = dev
            .gdn_layer_dev(
                &x_dev,
                &gw,
                0, // seq
                0, // layer
                n,
                hidden,
                h,
                dk,
                la.gate_lower_bound,
                cfg.rms_norm_eps,
                la.short_conv_kernel_size,
            )
            .expect("gdn_layer_dev");
        let mut out_gpu = Tensor::zeros(Shape::new([n, hidden]), DType::F32);
        {
            let v = Arc::get_mut(&mut out_gpu.data).expect("unique");
            partial.download(v).expect("download");
        }
        close(&out_cpu, &out_gpu, 2e-3, &format!("gdn_layer_dev round {round}"));
    }
}

/// TP-shard scenario: the serve path runs GDN layers on shard_weights_tp'd
/// weights with the shard's h (heads/world). This mirrors that exactly —
/// the serve garbage ("!!!!") only reproduced on TP4; the full-width
/// single-call and state-carry tests above pass with diff 0.
#[test]
fn gdn_device_chain_tp4_shard() {
    let so = std::env::var("FERRITE_KERNEL_SO").unwrap_or_else(|_| {
        let mut p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        p.push("../../kernels/cuda/libferrite_kernels.so");
        p.to_string_lossy().into_owned()
    });
    let dev = CudaBackend::with_device(&so, 0).expect("open cuda device 0");
    let cfg = Glm53FlashConfig::test_config();
    let w = random_weights(&cfg, 99);
    let world = 4;
    let w_shard = crate::tp::shard_weights_tp(&w, &cfg, 0, world);
    let mut shard_cfg = cfg.clone();
    shard_cfg.linear_attn.num_heads /= world;
    let mut eng = crate::Engine::new(shard_cfg.clone(), w_shard.clone(), ferrite_kernel::CpuBackend::new());
    let la = &cfg.linear_attn;
    let (h, dk) = (shard_cfg.linear_attn.num_heads, la.head_dim);
    let hidden = cfg.hidden_size;
    let n = 3;
    let pfx = "model.layers.0";
    let q = |name: &str| eng.w(&format!("{pfx}.self_attn.{name}")).unwrap().clone();
    let gw = GdnLayerWeights {
        qkv_proj: &q("qkv_proj.weight"),
        b_proj: &q("b_proj.weight"),
        f_a: &q("f_a_proj.weight"),
        f_b: &q("f_b_proj.weight"),
        g_a: &q("g_a_proj.weight"),
        g_b: &q("g_b_proj.weight"),
        conv_w: &q("qkv_conv1d.weight"),
        dt_bias: &q("dt_bias"),
        a_log: &q("A_log"),
        o_norm: &q("o_norm.weight"),
        o_proj: &q("o_proj.weight"),
    };
    for round in 0..3 {
        let x = Tensor::from_f32(
            Shape::new([n, hidden]),
            (0..n * hidden).map(|i| ((i * 41 + round * 89 % 251) as f32) * 0.008 - 0.9).collect(),
        );
        let out_cpu = eng
            .linear_attn_forward(0, 0, pfx, &x, n)
            .unwrap_or_else(|e| panic!("cpu round {round}: {e}"));
        let x_dev = DevBuf::alloc(dev.dev(), dev.stream(), x.numel()).expect("alloc");
        x_dev.upload(x.as_slice()).expect("upload");
        let partial = dev
            .gdn_layer_dev(
                &x_dev, &gw, 0, 0, n, hidden, h, dk,
                la.gate_lower_bound, cfg.rms_norm_eps, la.short_conv_kernel_size,
            )
            .expect("gdn_layer_dev");
        let mut out_gpu = Tensor::zeros(Shape::new([n, hidden]), DType::F32);
        {
            let v = Arc::get_mut(&mut out_gpu.data).expect("unique");
            partial.download(v).expect("download");
        }
        close(&out_cpu, &out_gpu, 2e-3, &format!("tp4-shard round {round}"));
    }
}
