//! Stage-1 real-weight validation: run the fp8 dense GEMM on a **real
//! checkpoint tensor** and compare against a CPU reference built from the same
//! bytes.
//!
//! This is the first end-to-end slice of the GPU path and it exercises, on real
//! 475 GiB-checkpoint data: the shard reader, the (identity) TP slice, the
//! upload, the device layer's dlopen/symbol table, the fp8 MMA kernel and the
//! scale semantics. It needs a GPU; gated on `DSV41_MODEL_DIR`.
//!
//!   DSV41_MODEL_DIR=/opt/dlami/nvme/models/DeepSeek-V4.1-Flash \
//!   DSV41_KERNELS=./kernels/cuda/libferrite_kernels.so CUDA_VISIBLE_DEVICES=0 \
//!     cargo test --release -p ferrite-dsv41 --test real_gemm -- --nocapture

use std::path::Path;

use ferrite_dsv41::config::Dsv41Config;
use ferrite_dsv41::device::Device;
use ferrite_dsv41::load::Loader;
use ferrite_dsv41::quant;
use ferrite_dsv41::weights::{SafetensorsIndex, Shard, TensorSpec};

fn env(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.is_empty())
}

/// Read a tensor's raw bytes straight from the shard (for the CPU reference).
fn read_host(dir: &Path, name: &str) -> Option<(String, Vec<usize>, Vec<u8>)> {
    let txt = std::fs::read_to_string(dir.join("model.safetensors.index.json")).ok()?;
    let v: serde_json::Value = serde_json::from_str(&txt).ok()?;
    let shard = v["weight_map"][name].as_str()?.to_string();
    let idx = SafetensorsIndex::read_header(&dir.join(&shard)).ok()?;
    let t = idx.tensors.get(name)?;
    let mut f = std::fs::File::open(dir.join(&shard)).ok()?;
    use std::io::{Read, Seek, SeekFrom};
    let mut len = [0u8; 8];
    f.read_exact(&mut len).ok()?;
    let hlen = u64::from_le_bytes(len);
    f.seek(SeekFrom::Start(8 + hlen + t.begin)).ok()?;
    let mut buf = vec![0u8; (t.end - t.begin) as usize];
    f.read_exact(&mut buf).ok()?;
    Some((t.dtype.clone(), t.shape.clone(), buf))
}

#[test]
fn fp8_gemm_on_real_checkpoint_matches_cpu_reference() {
    let (Some(dir), Some(so)) = (env("DSV41_MODEL_DIR"), env("DSV41_KERNELS")) else {
        eprintln!("[real_gemm] DSV41_MODEL_DIR / DSV41_KERNELS unset — skipped");
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    let cfg = Dsv41Config::production();
    let dev = Device::open(&so).expect("open device");
    let world = 1usize;
    let rank = 0usize;

    // ---- real weight + scale, straight off the checkpoint ----
    let name_w = "layers.6.attn.wq_a.weight";
    let name_s = "layers.6.attn.wq_a.scale";
    let (dt_w, shp_w, bytes_w) = read_host(&dir, name_w).expect("wq_a.weight");
    let (dt_s, shp_s, bytes_s) = read_host(&dir, name_s).expect("wq_a.scale");
    assert_eq!(dt_w, "F8_E4M3");
    assert_eq!(dt_s, "F8_E8M0");
    eprintln!(
        "[real_gemm] {name_w} {shp_w:?} {dt_w} ({} B) + scale {shp_s:?}",
        bytes_w.len()
    );

    // ---- load through the crate's own loader (exercises slicing + upload) ----
    let mut loader = Loader::new(&dir, dev).expect("loader");
    let spec_w = TensorSpec { name: name_w.into(), shape: shp_w.clone(), shard: Shard::Replicated };
    let s = tensor_specs_probe(&cfg, &spec_w, world, rank);
    assert_eq!(s, shp_w, "identity slice must be the whole tensor");
    let _ = &mut loader;

    let dev_w = dev.upload(&bytes_w).expect("upload w");
    let dev_s = dev.upload(&bytes_s).expect("upload scale");

    // ---- activation: synthetic, quantised the runtime way (block 32, pow2) ----
    let m = 16usize;
    let n = shp_w[0]; // 1280
    let k = shp_w[1]; // 5120
    let mut x = vec![0f32; m * k];
    let mut seed = 12345u64;
    for v in x.iter_mut() {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        *v = ((seed >> 33) as f32 / (1u64 << 31) as f32 - 0.5) * 4.0;
    }
    let (xq, xsc) = quant::act_quant_fp8(&x, 32, true); // same semantics as the kernel
    let dev_xq = dev.upload(&xq).expect("upload xq");
    let dev_xsc = dev.upload_f32(&xsc).expect("upload xsc");
    let dev_out = dev.alloc(m * n * 4).expect("out");

    dev.gemm_fp8_mx(
        dev_xq.as_u8(),
        dev_xsc.as_f32(),
        dev_w.as_u8(),
        dev_s.as_u8(),
        std::ptr::null(),
        dev_out.ptr as *mut f32,
        m as i32,
        n as i32,
        k as i32,
    )
    .expect("gemm_fp8_mx");
    dev.sync().expect("sync");
    let mut out = vec![0f32; m * n];
    dev.download_f32(&dev_out, &mut out).expect("download");

    // ---- CPU reference off the same bytes ----
    let wf = {
        let mut w = vec![0f32; n * k];
        quant::dequant_fp8_block(&bytes_w, &bytes_s, n, k, 32, &mut w);
        w
    };
    let nb_k = k / 32;
    let mut max_rel = 0f32;
    let mut checked = 0usize;
    for r in (0..m).step_by(4) {
        for j in (0..n).step_by(37) {
            let mut acc = 0f32;
            for kb in 0..nb_k {
                let mut part = 0f32;
                for kk in 0..32 {
                    let c = kb * 32 + kk;
                    part += quant::e4m3_decode(xq[r * k + c]) as f32 * wf[j * k + c];
                }
                acc += part * xsc[r * nb_k + kb];
            }
            let got = out[r * n + j];
            let rel = if acc.abs() > 1e-3 {
                (got - acc).abs() / acc.abs()
            } else {
                (got - acc).abs()
            };
            max_rel = max_rel.max(rel);
            checked += 1;
        }
    }
    eprintln!(
        "[real_gemm] checked {checked} elements, max RELATIVE diff = {max_rel:.3e} (fp32 accumulation-order only)"
    );
    assert!(
        max_rel < 2e-2,
        "fp8 GEMM on real weights disagrees with the reference: {max_rel}"
    );

    // ---- and the loader path must produce the same bytes on the device ----
    let t = loader
        .load_single(&spec_w, world, rank)
        .expect("loader.load_single");
    let mut back = vec![0u8; bytes_w.len()];
    let st = dev.download_u8(&t.buf, &mut back);
    st.expect("download loaded weight");
    assert_eq!(back, bytes_w, "the loader must reproduce the checkpoint bytes");
    eprintln!("[real_gemm] loader round-trip: {} bytes identical", back.len());
}

/// The identity slice for world=1.
fn tensor_specs_probe(_cfg: &Dsv41Config, spec: &TensorSpec, world: usize, rank: usize) -> Vec<usize> {
    ferrite_dsv41::weights::local_shape(_cfg, spec, world, rank)
}
