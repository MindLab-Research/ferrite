// Link the CUDA kernels .so + cudart for the `cuda` feature (GPU path).
// Non-CUDA builds (plain CPU) link nothing extra. The .so must be built by
// `kernels/cuda/build.sh` first.
fn main() {
    if std::env::var_os("CARGO_FEATURE_CUDA").is_some() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../kernels/cuda");
        let so = root.join("libferrite_kernels.so");
        if so.exists() {
            println!("cargo:rustc-link-search=native={}", root.display());
            println!("cargo:rustc-link-lib=dylib=ferrite_kernels");
        }
        let cuda_lib = match std::env::var("CUDA_HOME") {
            Ok(h) => format!("{h}/lib64"),
            Err(_) => {
                if std::path::Path::new("/usr/local/cuda/lib64").exists() {
                    "/usr/local/cuda/lib64".to_string()
                } else {
                    String::new()
                }
            }
        };
        if !cuda_lib.is_empty() {
            println!("cargo:rustc-link-search=native={cuda_lib}");
            println!("cargo:rustc-link-lib=dylib=cudart");
            // cuBLAS: the batched decode's m=16 GEMM is bandwidth-bound and
            // needs split-K/streaming that cuBLAS already implements (a
            // hand-rolled m16n8k16 kernel had only N/32 blocks → 5%
            // occupancy → 3x SLOWER than the FMA gemv it replaced).
            println!("cargo:rustc-link-lib=dylib=cublas");
        }
    }
    // KERNEL/BINARY SAME-SOURCE GATE (user rule 2026-09-10): the binary embeds
    // the build id that kernels/cuda/build.sh last stamped into .build_id
    // (git revision + .cu content hash). The .so carries the same string and
    // cuda.rs compares them at dlopen — a mismatched pair refuses to start,
    // so "rebuild only one artifact" can no longer be measured by accident.
    let kdir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../kernels/cuda");
    let stamp_path = kdir.join(".build_id");
    let build_id = std::fs::read_to_string(&stamp_path)
        .map(|s| s.trim().to_string())
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| {
            let git_id = std::process::Command::new("git")
                .args(["rev-parse", "HEAD"])
                .current_dir(env!("CARGO_MANIFEST_DIR"))
                .output()
                .ok()
                .filter(|o| o.status.success())
                .and_then(|o| String::from_utf8(o.stdout).ok())
                .map(|s| s.trim().to_string())
                .unwrap_or_else(|| "unknown".to_string());
            format!("{git_id}+cuNOSTAMP")
        });
    println!("cargo:rustc-env=FERRITE_BUILD_ID={build_id}");
    println!("cargo:rerun-if-changed={}", stamp_path.display());
    println!("cargo:rerun-if-changed=build.rs");
}
