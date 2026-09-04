fn main() {
    // ferrite-kernel's `--features cuda` backend declares extern "C" symbols
    // (cudaMalloc/Memcpy/Stream* from the CUDA runtime and the ferrite_*
    // kernels from libferrite_kernels.so). Link both so the bin resolves at
    // build time (the runtime still dlopens the .so, which must be built by
    // `kernels/cuda/build.sh` first).
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let kdir = std::path::Path::new(&manifest)
        .join("../../kernels/cuda")
        .canonicalize()
        .unwrap_or_else(|_| std::path::Path::new(&manifest).join("../../kernels/cuda"));
    println!("cargo:rustc-link-search=native={}", kdir.display());
    println!("cargo:rustc-link-lib=dylib=ferrite_kernels");
    if let Ok(cuda) = std::env::var("CUDA_HOME") {
        println!("cargo:rustc-link-search=native={cuda}/lib64");
    } else if std::path::Path::new("/usr/local/cuda/lib64").exists() {
        println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    }
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rerun-if-changed=build.rs");
}
