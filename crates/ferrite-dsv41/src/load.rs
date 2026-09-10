//! Real-checkpoint loader: safetensors -> device buffers, in the checkpoint's
//! own formats.
//!
//! Config-driven: the tensor list comes from [`crate::weights::tensor_specs`]
//! (itself derived from [`crate::config::Dsv41Config`]), so a different
//! geometry/TP degree needs no code change here.
//!
//! **No dequantisation.** Whatever the file holds is what lands in device
//! memory: fp8 e4m3 weights next to their ue8m0 32x32 scale tensors, fp4 e2m1
//! weights (I8-packed, 2 values per byte) next to their per-(row, 32) e8m0
//! scales, and bf16/f32 tensors untouched. Slicing for tensor parallelism is a
//! pure byte-range operation.

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use ferrite_types::{FerriteError, Result};

use crate::config::Dsv41Config;
use crate::device::{DevBuf, Device};
use crate::weights::{local_shape, tensor_specs, SafetensorsIndex, Shard, TensorSpec};

fn io<T>(r: std::io::Result<T>, what: &str) -> Result<T> {
    r.map_err(|e| FerriteError::Config(format!("{what}: {e}")))
}

/// On-disk element sizes for the dtypes the release uses.
fn dtype_size(dt: &str) -> usize {
    match dt {
        "F32" => 4,
        "BF16" => 2,
        "F8_E4M3" | "F8_E5M2" | "F8_E8M0" | "I8" | "U8" => 1,
        "I32" | "U32" => 4,
        "I64" => 8,
        _ => 1,
    }
}

/// A tensor living on the device in its checkpoint format.
pub struct DevTensor {
    pub buf: DevBuf,
    pub shape: Vec<usize>,
    pub dtype: String,
}

impl DevTensor {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
    pub fn ptr(&self) -> *mut std::ffi::c_void {
        self.buf.ptr
    }
    pub fn as_u8(&self) -> *const u8 {
        self.buf.ptr as *const u8
    }
    pub fn as_f32(&self) -> *const f32 {
        self.buf.ptr as *const f32
    }
    pub fn as_mut_f32(&self) -> *mut f32 {
        self.buf.ptr as *mut f32
    }
    pub fn as_i32(&self) -> *const i32 {
        self.buf.ptr as *const i32
    }
}

/// One expert's on-device weights (the rank's own experts only).
pub struct DevExpert {
    pub w1: DevTensor,
    pub w1_scale: DevTensor,
    pub w3: DevTensor,
    pub w3_scale: DevTensor,
    pub w2: DevTensor,
    pub w2_scale: DevTensor,
}

/// Per-layer device weights. Every optional group mirrors a config test
/// (`is_kv_source`, `is_index_source`, engram layers, ...).
#[derive(Default)]
pub struct LayerDev {
    pub hc_attn_fn: Option<DevTensor>,
    pub hc_attn_base: Option<DevTensor>,
    pub hc_attn_scale: Option<DevTensor>,
    pub hc_ffn_fn: Option<DevTensor>,
    pub hc_ffn_base: Option<DevTensor>,
    pub hc_ffn_scale: Option<DevTensor>,
    pub attn_norm: Option<DevTensor>,
    pub ffn_norm: Option<DevTensor>,
    pub wq_a: Option<DevTensor>,
    pub wq_a_scale: Option<DevTensor>,
    pub q_norm: Option<DevTensor>,
    pub wq_b: Option<DevTensor>,
    pub wq_b_scale: Option<DevTensor>,
    pub wkv: Option<DevTensor>,
    pub wkv_scale: Option<DevTensor>,
    pub kv_norm: Option<DevTensor>,
    pub wo_a: Option<DevTensor>,
    pub wo_a_scale: Option<DevTensor>,
    pub wo_b: Option<DevTensor>,
    pub wo_b_scale: Option<DevTensor>,
    pub attn_sink: Option<DevTensor>,
    /// compressor (kv sources only): wkv fp8 + (ratio>1) wgate fp8 + norm
    pub comp_wkv: Option<DevTensor>,
    pub comp_wkv_scale: Option<DevTensor>,
    pub comp_wgate: Option<DevTensor>,
    pub comp_wgate_scale: Option<DevTensor>,
    pub comp_norm: Option<DevTensor>,
    /// indexer (index sources only)
    pub idx_wq_b: Option<DevTensor>,
    pub idx_wq_b_scale: Option<DevTensor>,
    pub idx_weights: Option<DevTensor>,
    pub idx_wk: Option<DevTensor>,
    pub idx_k_norm: Option<DevTensor>,
    pub gate_w: Option<DevTensor>,
    pub gate_w_scale: Option<DevTensor>,
    pub gate_bias: Option<DevTensor>,
    pub gate_bias_vl: Option<DevTensor>,
    pub experts: Vec<DevExpert>,
    pub shared_w1: Option<DevTensor>,
    pub shared_w1_scale: Option<DevTensor>,
    pub shared_w3: Option<DevTensor>,
    pub shared_w3_scale: Option<DevTensor>,
    pub shared_w2: Option<DevTensor>,
    pub shared_w2_scale: Option<DevTensor>,
    /// engram (this layer carries a table)
    pub engram_embed: Option<DevTensor>,
    pub engram_embed_scale: Option<DevTensor>,
    pub engram_wkv: Option<DevTensor>,
    pub engram_wkv_scale: Option<DevTensor>,
    pub engram_q_weight: Option<DevTensor>,
    pub engram_k_weight: Option<DevTensor>,
}

/// All weights a rank needs.
#[derive(Default)]
pub struct Dsv41DevWeights {
    pub embed: Option<DevTensor>,
    pub norm: Option<DevTensor>,
    pub head: Option<DevTensor>,
    pub layers: Vec<LayerDev>,
    pub mtp: Vec<LayerDev>,
    /// `mtp.0.main_proj` / `main_norm`
    pub main_proj: Option<DevTensor>,
    pub main_proj_scale: Option<DevTensor>,
    pub main_norm: Option<DevTensor>,
    /// `mtp.last.norm`
    pub dspark_norm: Option<DevTensor>,
    pub markov_embed: Option<DevTensor>,
    pub markov_head: Option<DevTensor>,
    pub confidence_proj: Option<DevTensor>,
    /// vision (replicated)
    pub vision: Vec<(String, DevTensor)>,
}

/// Reads a checkpoint shard-by-shard on demand, uploading the rank's slice.
pub struct Loader<'a> {
    dir: PathBuf,
    /// tensor name -> shard file
    index: HashMap<String, String>,
    /// shard file -> header
    headers: HashMap<String, SafetensorsIndex>,
    dev: &'a Device,
    /// bytes uploaded so far (for the report)
    pub uploaded: u64,
    /// tensor-name prefixes to skip entirely (bring-up: the 189 GiB engram
    /// tables while the engram write-back is not yet wired)
    pub skip_prefixes: Vec<String>,
}

impl<'a> Loader<'a> {
    pub fn new(dir: &Path, dev: &'a Device) -> Result<Self> {
        let idx_path = dir.join("model.safetensors.index.json");
        let txt = std::fs::read_to_string(&idx_path).map_err(|e| {
            FerriteError::Config(format!("read {}: {e}", idx_path.display()))
        })?;
        let v: serde_json::Value = serde_json::from_str(&txt)
            .map_err(|e| FerriteError::Config(format!("index json: {e}")))?;
        let mut index = HashMap::new();
        for (k, shard) in v["weight_map"]
            .as_object()
            .ok_or_else(|| FerriteError::Config("no weight_map".into()))?
        {
            if let Some(s) = shard.as_str() {
                index.insert(k.clone(), s.to_string());
            }
        }
        Ok(Loader {
            dir: dir.to_path_buf(),
            index,
            headers: HashMap::new(),
            dev,
            uploaded: 0,
            skip_prefixes: Vec::new(),
        })
    }

    fn header(&mut self, name: &str) -> Result<&SafetensorsIndex> {
        let shard = self
            .index
            .get(name)
            .ok_or_else(|| FerriteError::Config(format!("{name} not in the checkpoint")))?
            .clone();
        if !self.headers.contains_key(&shard) {
            let h = SafetensorsIndex::read_header(&self.dir.join(&shard))?;
            self.headers.insert(shard.clone(), h);
        }
        Ok(self.headers.get(&shard).unwrap())
    }

    /// Raw bytes of one contiguous run inside a tensor's data.
    fn read_at(&mut self, name: &str, off: u64, len: usize) -> Result<Vec<u8>> {
        let shard = self.index.get(name).unwrap().clone();
        let h = self.header(name)?.tensors.get(name).cloned().ok_or_else(|| {
            FerriteError::Config(format!("{name} missing in {shard}"))
        })?;
        // the data section starts after the 8-byte length + the header json
        let path = self.dir.join(&shard);
        let hlen = {
            let mut f = io(File::open(&path), "open shard")?;
            let mut b = [0u8; 8];
            io(f.read_exact(&mut b), "read header len")?;
            u64::from_le_bytes(b)
        };
        let base = 8 + hlen + h.begin;
        let mut f = io(File::open(&path), "open shard")?;
        io(f.seek(SeekFrom::Start(base + off)), "seek")?;
        let mut buf = vec![0u8; len];
        io(f.read_exact(&mut buf), "read tensor bytes")?;
        self.uploaded += len as u64;
        Ok(buf)
    }

    /// Upload `spec`'s slice for `rank` of `world`.
    fn load_tensor(&mut self, spec: &TensorSpec, world: usize, rank: usize) -> Result<DevTensor> {
        if self.skip_prefixes.iter().any(|p| spec.name.starts_with(p.as_str())) {
            // A zero-length placeholder keeps the caller's Option logic intact
            // while loading nothing: the device bytes are never read because the
            // matching stage is not wired yet.
            return Ok(DevTensor {
                buf: self.dev.alloc(4)?,
                shape: vec![0],
                dtype: "SKIPPED".into(),
            });
        }
        let h = self.header(&spec.name)?.tensors.get(&spec.name).cloned().ok_or_else(|| {
            FerriteError::Config(format!("{} absent", spec.name))
        })?;
        let esz = dtype_size(&h.dtype);
        let local = local_shape(
            // local_shape only needs the shape-derived rules, not the config
            &Dsv41Config::production(),
            spec,
            world,
            rank,
        );
        // row-major strides of the GLOBAL tensor
        let global = &h.shape;
        let row_bytes: usize = global[1..].iter().product::<usize>() * esz;
        let (row0, rows) = match spec.shard {
            Shard::Replicated => (0usize, global[0]),
            Shard::Rows => {
                if spec.name.contains("engram.embed") {
                    let per = global[0].div_ceil(world);
                    (rank * per, per.min(global[0].saturating_sub(rank * per)))
                } else {
                    let per = global[0] / world;
                    (rank * per, per)
                }
            }
            Shard::Heads | Shard::Groups | Shard::Experts => {
                let per = global[0] / world;
                (rank * per, per)
            }
            Shard::Cols => (0, global[0]),
        };
        let bytes = if spec.shard == Shard::Cols {
            // column slice: gather the local columns of every row
            let inner: usize = global[1..].iter().product();
            let per = inner / world;
            let mut out = Vec::with_capacity(global[0] * per * esz);
            for r in 0..global[0] {
                let off = (r * inner + rank * per) as u64 * esz as u64;
                out.extend_from_slice(&self.read_at(&spec.name, off, per * esz)?);
            }
            out
        } else {
            let off = row0 as u64 * row_bytes as u64;
            let len = rows * row_bytes;
            if len == 0 {
                Vec::new()
            } else {
                self.read_at(&spec.name, off, len)?
            }
        };
        let buf = self.dev.upload(&bytes)?;
        let _ = local;
        Ok(DevTensor {
            buf,
            shape: local,
            dtype: h.dtype,
        })
    }

    /// Load ONE tensor (used by the staged bring-up and by the real-weight
    /// validation).
    pub fn load_single(&mut self, spec: &TensorSpec, world: usize, rank: usize) -> Result<DevTensor> {
        self.load_tensor(spec, world, rank)
    }

    /// Load everything the rank needs. `specs` come from `tensor_specs`.
    pub fn load(&mut self, cfg: &Dsv41Config, world: usize, rank: usize) -> Result<Dsv41DevWeights> {
        let specs = tensor_specs(cfg, world);
        // expert ownership: the rank keeps n_routed/world consecutive experts
        let expert_lo = |n: usize| (rank * (n / world), n / world);
        let mut w = Dsv41DevWeights {
            layers: (0..cfg.n_layers).map(|_| LayerDev::default()).collect(),
            mtp: (0..cfg.n_mtp_layers).map(|_| LayerDev::default()).collect(),
            ..Default::default()
        };
        let want = |n: &str| specs.iter().find(|s| s.name == n).cloned();
        // A macro rather than a closure: a closure capturing `self` mutably
        // would conflict with the very next `self.load_tensor` call, and it has
        // to span both the backbone and the draft loops.
        macro_rules! take {
            ($pfx:expr, $ld:expr, $key:expr, $field:ident) => {
                if let Some(sp) = want(&format!("{}.{}", $pfx, $key)) {
                    $ld.$field = Some(self.load_tensor(&sp, world, rank)?);
                }
            };
        }

        // ---- globals ----
        w.embed = Some(self.load_tensor(&want("embed.weight").unwrap(), world, rank)?);
        w.norm = Some(self.load_tensor(&want("norm.weight").unwrap(), world, rank)?);
        w.head = Some(self.load_tensor(&want("head.weight").unwrap(), world, rank)?);

        for l in 0..cfg.n_layers {
            let p = format!("layers.{l}");
            let p = p.as_str();
            let mut ld = LayerDev::default();
            let mut take = |ld: &mut LayerDev, key: &str, dst: fn(&mut LayerDev) -> &mut Option<DevTensor>| -> Result<()> {
                if let Some(s) = want(&format!("{p}.{key}")) {
                    *dst(ld) = Some(self.load_tensor(&s, world, rank)?);
                }
                Ok(())
            };
            take!(p, ld, "attn_norm.weight", attn_norm);
            take!(p, ld, "ffn_norm.weight", ffn_norm);
            take!(p, ld, "hc_attn_fn", hc_attn_fn);
            take!(p, ld, "hc_attn_base", hc_attn_base);
            take!(p, ld, "hc_attn_scale", hc_attn_scale);
            take!(p, ld, "hc_ffn_fn", hc_ffn_fn);
            take!(p, ld, "hc_ffn_base", hc_ffn_base);
            take!(p, ld, "hc_ffn_scale", hc_ffn_scale);
            take!(p, ld, "attn.wq_a.weight", wq_a);
            take!(p, ld, "attn.wq_a.scale", wq_a_scale);
            take!(p, ld, "attn.q_norm.weight", q_norm);
            take!(p, ld, "attn.wq_b.weight", wq_b);
            take!(p, ld, "attn.wq_b.scale", wq_b_scale);
            take!(p, ld, "attn.wkv.weight", wkv);
            take!(p, ld, "attn.wkv.scale", wkv_scale);
            take!(p, ld, "attn.kv_norm.weight", kv_norm);
            take!(p, ld, "attn.wo_a.weight", wo_a);
            take!(p, ld, "attn.wo_a.scale", wo_a_scale);
            take!(p, ld, "attn.wo_b.weight", wo_b);
            take!(p, ld, "attn.wo_b.scale", wo_b_scale);
            take!(p, ld, "attn.attn_sink", attn_sink);
            take!(p, ld, "attn.compressor.wkv.weight", comp_wkv);
            take!(p, ld, "attn.compressor.wkv.scale", comp_wkv_scale);
            take!(p, ld, "attn.compressor.wgate.weight", comp_wgate);
            take!(p, ld, "attn.compressor.wgate.scale", comp_wgate_scale);
            take!(p, ld, "attn.compressor.norm.weight", comp_norm);
            take!(p, ld, "attn.indexer.wq_b.weight", idx_wq_b);
            take!(p, ld, "attn.indexer.wq_b.scale", idx_wq_b_scale);
            take!(p, ld, "attn.indexer.weights_proj.weight", idx_weights);
            take!(p, ld, "attn.indexer.wk.weight", idx_wk);
            take!(p, ld, "attn.indexer.k_norm.weight", idx_k_norm);
            take!(p, ld, "ffn.gate.weight", gate_w);
            take!(p, ld, "ffn.gate.bias", gate_bias);
            take!(p, ld, "ffn.gate.bias_vl", gate_bias_vl);
            take!(p, ld, "ffn.shared_experts.w1.weight", shared_w1);
            take!(p, ld, "ffn.shared_experts.w1.scale", shared_w1_scale);
            take!(p, ld, "ffn.shared_experts.w3.weight", shared_w3);
            take!(p, ld, "ffn.shared_experts.w3.scale", shared_w3_scale);
            take!(p, ld, "ffn.shared_experts.w2.weight", shared_w2);
            take!(p, ld, "ffn.shared_experts.w2.scale", shared_w2_scale);
            take!(p, ld, "engram.embed.weight", engram_embed);
            take!(p, ld, "engram.embed.scale", engram_embed_scale);
            take!(p, ld, "engram.wkv.weight", engram_wkv);
            take!(p, ld, "engram.wkv.scale", engram_wkv_scale);
            take!(p, ld, "engram.q_weight", engram_q_weight);
            take!(p, ld, "engram.k_weight", engram_k_weight);

            // routed experts this rank owns
            let (n_routed, _) = cfg.moe_config(l);
            let (e0, ne) = expert_lo(n_routed);
            for e in e0..(e0 + ne).min(n_routed) {
                let g = |n: &str| want(&format!("{p}.ffn.experts.{e}.{n}")).unwrap();
                ld.experts.push(DevExpert {
                    w1: self.load_tensor(&g("w1.weight"), world, rank)?,
                    w1_scale: self.load_tensor(&g("w1.scale"), world, rank)?,
                    w3: self.load_tensor(&g("w3.weight"), world, rank)?,
                    w3_scale: self.load_tensor(&g("w3.scale"), world, rank)?,
                    w2: self.load_tensor(&g("w2.weight"), world, rank)?,
                    w2_scale: self.load_tensor(&g("w2.scale"), world, rank)?,
                });
            }
            w.layers[l] = ld;
        }

        // ---- DSpark draft layers ----
        for s in 0..cfg.n_mtp_layers {
            let p = format!("mtp.{s}");
            let p = p.as_str();
            let mut ld = LayerDev::default();
            take!(p, ld, "attn_norm.weight", attn_norm);
            take!(p, ld, "ffn_norm.weight", ffn_norm);
            take!(p, ld, "hc_attn_fn", hc_attn_fn);
            take!(p, ld, "hc_attn_base", hc_attn_base);
            take!(p, ld, "hc_attn_scale", hc_attn_scale);
            take!(p, ld, "hc_ffn_fn", hc_ffn_fn);
            take!(p, ld, "hc_ffn_base", hc_ffn_base);
            take!(p, ld, "hc_ffn_scale", hc_ffn_scale);
            take!(p, ld, "attn.wq_a.weight", wq_a);
            take!(p, ld, "attn.wq_a.scale", wq_a_scale);
            take!(p, ld, "attn.q_norm.weight", q_norm);
            take!(p, ld, "attn.wq_b.weight", wq_b);
            take!(p, ld, "attn.wq_b.scale", wq_b_scale);
            take!(p, ld, "attn.wkv.weight", wkv);
            take!(p, ld, "attn.wkv.scale", wkv_scale);
            take!(p, ld, "attn.kv_norm.weight", kv_norm);
            take!(p, ld, "attn.wo_a.weight", wo_a);
            take!(p, ld, "attn.wo_a.scale", wo_a_scale);
            take!(p, ld, "attn.wo_b.weight", wo_b);
            take!(p, ld, "attn.wo_b.scale", wo_b_scale);
            take!(p, ld, "attn.attn_sink", attn_sink);
            take!(p, ld, "ffn.gate.weight", gate_w);
            take!(p, ld, "ffn.gate.bias", gate_bias);
            take!(p, ld, "ffn.gate.bias_vl", gate_bias_vl);
            take!(p, ld, "ffn.shared_experts.w1.weight", shared_w1);
            take!(p, ld, "ffn.shared_experts.w1.scale", shared_w1_scale);
            take!(p, ld, "ffn.shared_experts.w3.weight", shared_w3);
            take!(p, ld, "ffn.shared_experts.w3.scale", shared_w3_scale);
            take!(p, ld, "ffn.shared_experts.w2.weight", shared_w2);
            take!(p, ld, "ffn.shared_experts.w2.scale", shared_w2_scale);
            let (n_routed, _) = cfg.moe_config(cfg.n_layers + s);
            let (e0, ne) = expert_lo(n_routed);
            for e in e0..(e0 + ne).min(n_routed) {
                let g = |n: &str| want(&format!("{p}.ffn.experts.{e}.{n}")).unwrap();
                ld.experts.push(DevExpert {
                    w1: self.load_tensor(&g("w1.weight"), world, rank)?,
                    w1_scale: self.load_tensor(&g("w1.scale"), world, rank)?,
                    w3: self.load_tensor(&g("w3.weight"), world, rank)?,
                    w3_scale: self.load_tensor(&g("w3.scale"), world, rank)?,
                    w2: self.load_tensor(&g("w2.weight"), world, rank)?,
                    w2_scale: self.load_tensor(&g("w2.scale"), world, rank)?,
                });
            }
            if s == 0 {
                take!(p, ld, "main_proj.weight", attn_norm); // placeholder
                if let Some(sp) = want(&format!("{p}.main_proj.weight")) {
                    w.main_proj = Some(self.load_tensor(&sp, world, rank)?);
                }
                if let Some(sp) = want(&format!("{p}.main_proj.scale")) {
                    w.main_proj_scale = Some(self.load_tensor(&sp, world, rank)?);
                }
                if let Some(sp) = want(&format!("{p}.main_norm.weight")) {
                    w.main_norm = Some(self.load_tensor(&sp, world, rank)?);
                }
            }
            if s + 1 == cfg.n_mtp_layers {
                if let Some(sp) = want(&format!("{p}.norm.weight")) {
                    w.dspark_norm = Some(self.load_tensor(&sp, world, rank)?);
                }
                if let Some(sp) = want(&format!("{p}.markov_head.embed.weight")) {
                    w.markov_embed = Some(self.load_tensor(&sp, world, rank)?);
                }
                if let Some(sp) = want(&format!("{p}.markov_head.head.weight")) {
                    w.markov_head = Some(self.load_tensor(&sp, world, rank)?);
                }
                if let Some(sp) = want(&format!("{p}.confidence_head.proj.weight")) {
                    w.confidence_proj = Some(self.load_tensor(&sp, world, rank)?);
                }
            }
            w.mtp[s] = ld;
        }

        // ---- vision (replicated; only loaded when enabled) ----
        if cfg.vision_enabled() {
            for s in &specs {
                if s.name.starts_with("vision.") || s.name.starts_with("aligner.") || s.name.starts_with("image_") {
                    let t = self.load_tensor(s, world, rank)?;
                    w.vision.push((s.name.clone(), t));
                }
            }
        }
        Ok(w)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_sizes_match_the_release() {
        assert_eq!(dtype_size("F8_E4M3"), 1);
        assert_eq!(dtype_size("F8_E8M0"), 1);
        assert_eq!(dtype_size("I8"), 1);
        assert_eq!(dtype_size("BF16"), 2);
        assert_eq!(dtype_size("F32"), 4);
    }

    #[test]
    fn expert_ownership_is_consecutive_and_complete() {
        // the loader keeps n_routed/world consecutive experts per rank; across
        // all ranks that must cover every expert exactly once
        let cfg = Dsv41Config::production();
        for world in [1usize, 2, 4, 8] {
            let n = cfg.n_routed_experts;
            let per = n / world;
            let mut seen = vec![0usize; n];
            for rank in 0..world {
                let lo = rank * per;
                for e in lo..(lo + per).min(n) {
                    seen[e] += 1;
                }
            }
            assert!(
                seen.iter().all(|&c| c == 1),
                "world {world}: expert ownership must partition (got {:?})",
                seen.iter().filter(|&&c| c != 1).count()
            );
        }
    }
}
