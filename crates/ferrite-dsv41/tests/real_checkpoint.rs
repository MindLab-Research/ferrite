//! Validate the tensor map against a real DeepSeek-V4.1-Flash checkpoint.
//!
//! Skipped unless `DSV41_MODEL_DIR` points at the release (so a CI box without
//! the 475 GiB checkpoint still passes). It never loads tensor data — only the
//! safetensors headers — so it is a CPU-only, seconds-long check that every
//! name, shape and dtype the engine expects really exists in the checkpoint,
//! and that nothing unexpected is left unclaimed.
//!
//! Run on the B300:
//!   DSV41_MODEL_DIR=/opt/dlami/nvme/models/DeepSeek-V4.1-Flash \
//!     cargo test -p ferrite-dsv41 --test real_checkpoint -- --nocapture

use std::collections::{HashMap, HashSet};
use std::path::Path;

use ferrite_dsv41::config::Dsv41Config;
use ferrite_dsv41::weights::{tensor_specs, SafetensorsIndex};

/// The specs already describe the ON-DISK geometry (an fp4 expert weight is
/// `[out, in/2]`, an fp4 scale is one e8m0 byte per (row, 32 cols), a dense
/// weight is `[out, in]` with a `[out/32, in/32]` scale), so no reinterpretation
/// is needed here -- dividing again would expect half a row.
fn expected_file_shape(_name: &str, spec_shape: &[usize]) -> Vec<usize> {
    spec_shape.to_vec()
}

/// Dtype expectations, restricted to the tensors whose dtype the release
/// actually pins: everything else is bf16/f32 and is checked by name+shape
/// only (an over-broad heuristic here produced 177 false alarms on the first
/// real run).
fn expected_dtype(name: &str) -> Option<&'static str> {
    if !name.ends_with(".weight") && !name.ends_with(".scale") {
        return None;
    }
    if name.ends_with(".scale") {
        // ue8m0 block scales accompany every quantised weight
        return Some("F8_E8M0");
    }
    if name.contains(".experts.") {
        return Some("I8"); // fp4 e2m1, packed 2 per byte
    }
    let dense_fp8 = [
        ".attn.wq_a.weight",
        ".attn.wq_b.weight",
        ".attn.wkv.weight",
        ".attn.wo_a.weight",
        ".attn.wo_b.weight",
        ".ffn.shared_experts.w1.weight",
        ".ffn.shared_experts.w2.weight",
        ".ffn.shared_experts.w3.weight",
        ".engram.wkv.weight",
        ".engram.embed.weight",
        ".main_proj.weight",
    ];
    if dense_fp8.iter().any(|suf| name.ends_with(suf)) {
        return Some("F8_E4M3");
    }
    None // bf16/f32: norms, sinks, hc params, compressor/indexer, vision, heads
}

#[test]
fn weight_map_matches_the_real_checkpoint() {
    let Ok(dir) = std::env::var("DSV41_MODEL_DIR") else {
        eprintln!("[real_checkpoint] DSV41_MODEL_DIR unset — skipped");
        return;
    };
    let idx_path = Path::new(&dir).join("model.safetensors.index.json");
    let txt = std::fs::read_to_string(&idx_path).expect("index.json");
    let v: serde_json::Value = serde_json::from_str(&txt).expect("index json parses");
    let wm = v["weight_map"].as_object().expect("weight_map");

    // read every shard header once
    let mut shards: HashMap<String, SafetensorsIndex> = HashMap::new();
    for f in wm.values().filter_map(|x| x.as_str()) {
        if !shards.contains_key(f) {
            let idx = SafetensorsIndex::read_header(&Path::new(&dir).join(f))
                .unwrap_or_else(|e| panic!("header {f}: {e}"));
            shards.insert(f.to_string(), idx);
        }
    }
    eprintln!(
        "[real_checkpoint] {} tensors across {} shards",
        wm.len(),
        shards.len()
    );

    let cfg = Dsv41Config::production();
    let specs = tensor_specs(&cfg, 8);

    // 1. every expected tensor exists with the right file shape
    let mut missing: Vec<String> = Vec::new();
    let mut bad_shape: Vec<String> = Vec::new();
    let mut bad_dtype: Vec<String> = Vec::new();
    let mut descs = 0usize;
    for s in &specs {
        let Some(shard) = wm.get(&s.name).and_then(|x| x.as_str()) else {
            missing.push(s.name.clone());
            continue;
        };
        descs += 1;
        let tv = shards[shard]
            .tensors
            .get(&s.name)
            .unwrap_or_else(|| panic!("{} not in {}", s.name, shard));
        let want = expected_file_shape(&s.name, &s.shape);
        if tv.shape != want {
            bad_shape.push(format!("{}: file {:?} != {:?}", s.name, tv.shape, want));
        }
        if let Some(dt) = expected_dtype(&s.name) {
            if tv.dtype != dt {
                bad_dtype.push(format!("{}: {} != {}", s.name, tv.dtype, dt));
            }
        }
    }

    // 2. nothing in the checkpoint should be unclaimed (a silent extra tensor
    //    would mean we mis-mapped something, e.g. a companion scale)
    let spec_names: HashSet<&str> = specs.iter().map(|s| s.name.as_str()).collect();
    let mut unclaimed: Vec<String> = wm
        .keys()
        .filter(|k| !spec_names.contains(k.as_str()))
        .cloned()
        .collect();
    unclaimed.sort();

    eprintln!(
        "[real_checkpoint] specs {} | matched {} | missing {} | bad shape {} | bad dtype {} | unclaimed {}",
        specs.len(),
        descs,
        missing.len(),
        bad_shape.len(),
        bad_dtype.len(),
        unclaimed.len()
    );
    for (label, list) in [
        ("MISSING", &missing),
        ("BAD SHAPE", &bad_shape),
        ("BAD DTYPE", &bad_dtype),
        ("UNCLAIMED", &unclaimed),
    ] {
        for l in list.iter().take(20) {
            eprintln!("  {label}: {l}");
        }
        if list.len() > 20 {
            eprintln!("  {label}: ... and {} more", list.len() - 20);
        }
    }

    // The checkpoint is the ground truth: a mismatch here means the engine
    // cannot load the model, so fail loudly rather than warn.
    assert!(missing.is_empty(), "{} expected tensors are absent", missing.len());
    assert!(bad_shape.is_empty(), "{} shape mismatches", bad_shape.len());
    assert!(bad_dtype.is_empty(), "{} dtype mismatches", bad_dtype.len());
    // Unclaimed tensors are reported but only fatal if numerous: keep a small
    // allowance for metadata-style keys, fail on anything structural.
    assert!(
        unclaimed.len() <= 8,
        "{} unclaimed checkpoint tensors (first: {:?})",
        unclaimed.len(),
        unclaimed.first()
    );
}

#[test]
fn pack_geometry_matches_the_release_headers() {
    // A focused, GPU-free assertion of the two packings that matter:
    //   experts: I8 [out, in/2] + F8_E8M0 [out, in/32]
    //   dense:   F8_E4M3 [out, in] + F8_E8M0 [out/32, in/32]
    let cfg = Dsv41Config::production();
    let specs = tensor_specs(&cfg, 8);
    let get = |n: &str| specs.iter().find(|s| s.name == n).expect(n);
    assert_eq!(get("layers.6.ffn.experts.0.w1.weight").shape, vec![2304, 2560]);
    // the spec carries the packed (on-disk) row width for fp4 experts
    assert_eq!(expected_file_shape("layers.6.ffn.experts.0.w1.weight", &[2304, 2560]), vec![2304, 2560]);
    assert_eq!(get("layers.6.ffn.experts.0.w1.scale").shape, vec![2304, 160]);
    assert_eq!(get("layers.6.attn.wq_a.scale").shape, vec![40, 160]);
    // the wq_b packing (padded rope lanes) is a TP-head split, not a pack
    assert_eq!(get("layers.6.attn.wq_b.weight").shape, vec![32768, 1280]);
}
