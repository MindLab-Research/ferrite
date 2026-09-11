// Build script for ferrite-kernel.
//
// Two jobs:
//   1. LINK (only under the `cuda` feature): point the linker at
//      kernels/cuda/libferrite_kernels.so + cudart/cuBLAS/cuBLASLt.
//   2. COMPILE-TIME HALF OF THE KERNEL/BINARY SAME-SOURCE GATE (always):
//      bake kernels/cuda/.build_id into this binary as `FERRITE_BUILD_ID`, so
//      cuda.rs / devrt.rs can dlsym the .so's `ferrite_kernel_build_id()` and
//      refuse to start on a mismatch.
//
// Why the gate has a COMPILE-TIME half at all (user rule 2026-09-11, "确保以后
// 不会再出现这个 so 的问题"): the runtime gate can only reject a mismatched
// PAIR — it cannot stop you from *producing* one. `.so` and `.build_id` are
// gitignored/untracked, so `git checkout` + `git clean` leave both behind, and
// incrementally relinking a fresh binary against a stale stamp is silent. This
// script makes the inconsistent states un-buildable so the failure surfaces at
// `cargo build`, not after a 2-minute serve.
//
// The invariant enforced here (fail-closed):
//   * .so + .build_id both present  -> OK, embed the stamp (+ staleness check).
//   * exactly one present           -> GATE FAIL (half-built tree — the
//                                      dangerous case that produced the bad A/Bs).
//   * neither present               -> allowed ONLY in a debug build
//                                      (`cargo check` on a box with no CUDA /
//                                      no kernels). A RELEASE build must not
//                                      produce a stamp-less binary, so it
//                                      GATE FAILS; the runtime gate would reject
//                                      it anyway, this just moves the error to
//                                      the build.
//
// Escapes (deliberate, for CI / CPU-only boxes — never for a measurement run):
//   FERRITE_ALLOW_NO_KERNELS=1     downgrade "neither present" to a warning
//                                  even in release.
//   FERRITE_REQUIRE_KERNELS=1      upgrade it to a hard error even in debug.
//   FERRITE_ALLOW_STALE_KERNELS=1  skip the mtime staleness check (see below).
use std::path::{Path, PathBuf};
use std::time::SystemTime;

/// Abort the build with a readable, actionable message. cargo shows the panic
/// payload, so this is the "compile error" of the gate.
fn gate_fail(msg: &str) -> ! {
    panic!(
        "\n\n\
         ======================================================================\n\
         ferrite kernel build gate: REFUSING TO BUILD THIS ARTIFACT\n\
         ======================================================================\n\
         {msg}\n\
         ======================================================================\n"
    );
}

/// Newest `.cu` source (path, mtime) in the kernel directory.
fn newest_cu(kdir: &Path) -> Option<(PathBuf, SystemTime)> {
    let mut newest: Option<(PathBuf, SystemTime)> = None;
    for entry in std::fs::read_dir(kdir).ok()?.flatten() {
        let p = entry.path();
        if p.extension().map(|e| e == "cu").unwrap_or(false) {
            if let Ok(t) = p.metadata().and_then(|m| m.modified()) {
                if newest.as_ref().map(|(_, nt)| t > *nt).unwrap_or(true) {
                    newest = Some((p, t));
                }
            }
        }
    }
    newest
}

/// `git rev-parse HEAD` of THIS tree, for the stamp-less sentinel only.
fn git_head() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

fn main() {
    let cuda = std::env::var_os("CARGO_FEATURE_CUDA").is_some();
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../kernels/cuda");
    let so = root.join("libferrite_kernels.so");
    let stamp_path = root.join(".build_id");

    // Re-run whenever a kernel source changes: this is what lets the staleness
    // check below actually fire on "edited a .cu, forgot to rebuild the .so".
    // (Previously only .build_id + build.rs were watched, so a changed .cu did
    // not even re-run this script.)
    if let Ok(rd) = std::fs::read_dir(&root) {
        for entry in rd.flatten() {
            let p = entry.path();
            if p.extension().map(|e| e == "cu").unwrap_or(false) {
                println!("cargo:rerun-if-changed={}", p.display());
            }
        }
    }
    println!("cargo:rerun-if-changed={}", stamp_path.display());
    println!("cargo:rerun-if-changed=build.rs");

    // ---------------------------------------------------------------- link
    if cuda && so.exists() {
        println!("cargo:rustc-link-search=native={}", root.display());
        println!("cargo:rustc-link-lib=dylib=ferrite_kernels");
    }
    if cuda {
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
            // cuBLASLt: explicit algorithm selection for m<=32 decode shapes,
            // where cublasGemmEx's heuristic picks splitK (nvjet_splitK +
            // splitKreduce ≈ 0.41ms/step of pure K-split overhead).
            println!("cargo:rustc-link-lib=dylib=cublasLt");
        }
    }

    // ----------------------------------------------------- same-source gate
    let so_exists = so.is_file();
    let stamp = std::fs::read_to_string(&stamp_path)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());
    let profile = std::env::var("PROFILE").unwrap_or_default();
    let release = profile == "release";
    let allow_no_kernels = std::env::var_os("FERRITE_ALLOW_NO_KERNELS").is_some();
    let require_kernels = std::env::var_os("FERRITE_REQUIRE_KERNELS").is_some();
    let allow_stale = std::env::var_os("FERRITE_ALLOW_STALE_KERNELS").is_some();

    let build_id = match (so_exists, stamp.as_deref()) {
        (true, Some(id)) => {
            // Anti-staleness: a .cu newer than the .so means the .so was built
            // from older sources, so `id` no longer describes the sources you
            // are compiling against. Fail-closed (a false positive only costs a
            // rebuild; a false negative costs an invalid measurement).
            if !allow_stale {
                if let Some((cu, cu_t)) = newest_cu(&root) {
                    if let Ok(so_t) = so.metadata().and_then(|m| m.modified()) {
                        if cu_t > so_t {
                            gate_fail(&format!(
                                "kernel sources are NEWER than the prebuilt .so — the .so is STALE.\n\
                                 \x20 newer source : {}\n\
                                 \x20 .so            : {}\n\
                                 \x20 .build_id      : {id}\n\
                                 The .so/.build_id are gitignored, so a checkout/reset does NOT refresh them.\n\
                                 Fix (the ONLY order that works):\n\
                                 \x20   cd kernels/cuda && bash build.sh 103a && cd ../.. && cargo build --release\n\
                                 Or rebuild via the driver: PHASES=0 scripts/dsv41_recovery_verify.sh\n\
                                 (opt out, not for measurements: FERRITE_ALLOW_STALE_KERNELS=1)",
                                cu.display(),
                                so.display(),
                            ));
                        }
                    }
                }
            }
            id.to_string()
        }
        (true, None) => gate_fail(
            "kernels/cuda/libferrite_kernels.so EXISTS but kernels/cuda/.build_id is MISSING.\n\
             Without a stamp the binary cannot prove it matches the .so, so the runtime gate\n\
             would refuse to start — and worse, a stale .so could slip through if it were\n\
             silently re-stamped. This half-built state is forbidden.\n\
             Fix: remove the orphan .so and rebuild both artifacts together:\n\
             \x20   cd kernels/cuda && rm -f libferrite_kernels.so .build_id\n\
             \x20   bash build.sh 103a && cd ../.. && cargo build --release",
        ),
        (false, Some(id)) => gate_fail(&format!(
            "kernels/cuda/.build_id = {id} EXISTS but kernels/cuda/libferrite_kernels.so is MISSING.\n\
             This is the signature of a one-sided rebuild (the .so was deleted / never produced\n\
             while a prior build left the stamp). Refusing to bake a stamp for a .so that does\n\
             not exist.\n\
             Fix: rebuild both artifacts in order:\n\
             \x20   cd kernels/cuda && bash build.sh 103a && cd ../.. && cargo build --release",
        )),
        (false, None) => {
            if require_kernels || (release && !allow_no_kernels) {
                gate_fail(&format!(
                    "no kernel artifacts in this tree: neither {} nor {} exists.\n\
                     A [{profile}] build would produce a binary that cannot load its kernels\n\
                     (FERRITE_BUILD_ID would be a *cuNOSTAMP* sentinel and the runtime gate\n\
                     refuses to start), so this is a hard error instead.\n\
                     Build the kernels first (CUDA toolkit required, no GPU needed):\n\
                     \x20   cd kernels/cuda && bash build.sh 103a && cd ../.. && cargo build --release\n\
                     If this really is a CPU-only / check-only build:\n\
                     \x20   cargo check                      (debug is allowed to skip kernels)\n\
                     \x20   FERRITE_ALLOW_NO_KERNELS=1 cargo build --release   (explicit opt-out)",
                    so.display(),
                    stamp_path.display(),
                ));
            }
            println!(
                "cargo:warning=ferrite-kernel: no libferrite_kernels.so / .build_id in this tree — \
                 embedding the *cuNOSTAMP* sentinel. This is only allowed for a debug (check) build; \
                 the CUDA runtime gate will refuse to start this binary."
            );
            format!("{}+cuNOSTAMP", git_head())
        }
    };
    println!("cargo:rustc-env=FERRITE_BUILD_ID={build_id}");
}
