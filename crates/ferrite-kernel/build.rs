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
        }
    }
    println!("cargo:rerun-if-changed=build.rs");
}
