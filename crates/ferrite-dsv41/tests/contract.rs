//! Enforces the DeepSeek-V4.1-Flash performance contract on the kernel sources.
//!
//! The contract (user requirement, restated in `kernels.rs` and the crate
//! README) is:
//!   1. no weight is ever dequantised into a bf16/f32 buffer;
//!   2. every large matmul is a tensor-core MMA over the native format.
//!
//! A comment cannot enforce that, so this test reads the CUDA sources and fails
//! on the patterns that would silently violate it. It is deliberately textual:
//! capturing the *shape* of a violation (a widened operand feed into an MMA, a
//! bf16 GEMM over expert weights) is what matters, and a false positive here is
//! a prompt to justify the code rather than a real blocker.

use std::fs;
use std::path::{Path, PathBuf};

fn kernel_sources() -> Vec<(PathBuf, String)> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../kernels/cuda");
    let mut out = Vec::new();
    if let Ok(rd) = fs::read_dir(&root) {
        for e in rd.flatten() {
            let p = e.path();
            let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
            if name.starts_with("dsv41") && name.ends_with(".cu") {
                out.push((p.clone(), fs::read_to_string(&p).unwrap_or_default()));
            }
        }
    }
    out
}

/// Strip line comments so documentation that *mentions* a forbidden pattern
/// (like this file does) does not trip the checks.
fn code_only(src: &str) -> String {
    src.lines()
        .map(|l| match l.find("//") {
            Some(i) => &l[..i],
            None => l,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn no_bf16_widening_of_quantised_weights() {
    let srcs = kernel_sources();
    if srcs.is_empty() {
        eprintln!("[contract] no dsv41 kernel sources present yet — skipping");
        return;
    }
    // Widening a quantised weight into a 16-bit buffer is the thing the
    // contract forbids. Converting *activations* between fp32/bf16 is fine, so
    // the check is anchored on the weight-side names the ABI uses.
    let weight_names = ["w1", "w3", "w2", "down_w", "gate_w", "wkv", "engram"];
    for (path, src) in &srcs {
        let code = code_only(src);
        for (i, line) in code.lines().enumerate() {
            let l = line.trim();
            if l.starts_with("//") || l.is_empty() {
                continue;
            }
            let mentions_weight = weight_names.iter().any(|w| l.contains(w));
            if !mentions_weight {
                continue;
            }
            // a conversion into bf16/f16 combined with a weight name
            let widens = (l.contains("__nv_bfloat16") || l.contains("cvt") || l.contains("bf16"))
                && (l.contains("=") || l.contains("store"));
            assert!(
                !widens,
                "{}:{}: quantised weights must stay packed (no bf16/f16 widening):\n    {}",
                path.display(),
                i + 1,
                l
            );
        }
    }
}

#[test]
fn experts_are_never_computed_in_fp8() {
    // User directive: the routed experts are fp4 in the checkpoint and must run
    // on fp4 tensor cores (NVFP4 / MXFP4). No fp8 expert path may exist.
    let abi = fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("src/kernels.rs"))
        .expect("kernels.rs");
    for forbidden in [
        "dsv41_expert_gate_up_fp8",
        "dsv41_expert_down_fp8",
        "convert_expert_fp4_to_e4m3",
    ] {
        assert!(
            !abi.contains(forbidden),
            "the fp8 expert path is forbidden (found {forbidden} in the ABI)"
        );
    }
    let w = fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("src/weights.rs"))
        .expect("weights.rs");
    assert!(
        !w.contains("convert_expert_fp4_to_e4m3"),
        "the fp4 -> e4m3 re-encode (the fp8 path's enabler) must not come back"
    );
    // and the expert kernels must not mention an e4m3 MMA
    for (path, raw) in kernel_sources() {
        let src = code_only(&raw);
        for line in src.lines() {
            let l = line.trim();
            if l.contains("expert") && l.contains("e4m3") {
                panic!(
                    "{}: the expert path must use fp4 tensor cores, not fp8:\n    {}",
                    path.display(),
                    l
                );
            }
        }
    }
}

#[test]
fn large_matmuls_use_tensor_cores() {
    let srcs = kernel_sources();
    if srcs.is_empty() {
        eprintln!("[contract] no dsv41 kernel sources present yet — skipping");
        return;
    }
    let mut mma_forms = 0usize;
    for (path, raw) in &srcs {
        // comments are stripped: the header documents the rejected forms on
        // purpose, and that documentation must not trip the check
        let src = code_only(raw);
        for pat in [
            "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32",
            "mma.sync.aligned.m16n8k16",
            "tcgen05.mma",
        ] {
            mma_forms += src.matches(pat).count();
        }
        // the rejected forms must not reappear in CODE
        for bad in ["kind::f8f6f4", "m16n8k32.row.col.kind::f8f6f4"] {
            assert!(
                !src.contains(bad),
                "{}: the fp4 mma.sync form is rejected by ptxas on sm_103a; \
                 use tcgen05 kind::mxf4 or the lossless e4m3 fallback (found {bad})",
                path.display()
            );
        }
    }
    assert!(
        mma_forms > 0,
        "the dsv41 kernels must contain tensor-core MMAs (fp8 m16n8k32 and/or tcgen05), found none"
    );
}

#[test]
fn native_formats_are_the_abi_types() {
    // The ABI must talk about packed bytes for quantised tensors, never f32/bf16
    // for the weights themselves (which is how a dequantising interface would
    // look).
    let abi = fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("src/kernels.rs"),
    )
    .expect("kernels.rs");
    for fn_name in [
        "dsv41_gemm_fp8_mx",
        "dsv41_expert_gate_up_fp4",
        "dsv41_expert_down_fp4",
        "dsv41_engram_gather",
    ] {
        let at = abi
            .find(fn_name)
            .unwrap_or_else(|| panic!("{fn_name} missing from the ABI"));
        let body = &abi[at..abi[at..].find(") -> i32").map(|e| at + e).unwrap_or(at + 400)];
        assert!(
            body.contains("*const u8"),
            "{fn_name}: weights enter as packed bytes (found: {body})"
        );
    }
}
