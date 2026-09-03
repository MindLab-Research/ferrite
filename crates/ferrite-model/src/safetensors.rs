//! Minimal safetensors loader — parses the HF safetensors format directly
//! (8-byte LE header length + JSON header + raw tensor blob) with zero
//! external dependencies, converting BF16 / F16 / FP8-E4M3 / F32 / U8
//! into f32 storage for the CPU reference path.
//!
//! The B300 GPU path will read the raw bytes instead (keep the same header
//! parse, skip the f32 conversion) — that is a `feature = "cuda"` concern.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use ferrite_types::{DType, FerriteError, Shape, Tensor, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RawDType {
    F32,
    F16,
    Bf16,
    Fp8E4m3,
    U8,
}

impl RawDType {
    fn from_str(s: &str) -> Option<Self> {
        match s {
            "F32" => Some(RawDType::F32),
            "F16" => Some(RawDType::F16),
            "BF16" => Some(RawDType::Bf16),
            "F8_E4M3" => Some(RawDType::Fp8E4m3),
            "U8" => Some(RawDType::U8),
            _ => None,
        }
    }

    fn element_size(self) -> usize {
        match self {
            RawDType::F32 => 4,
            RawDType::F16 | RawDType::Bf16 => 2,
            RawDType::Fp8E4m3 | RawDType::U8 => 1,
        }
    }

    fn ferrite_dtype(self) -> DType {
        match self {
            RawDType::F32 => DType::F32,
            RawDType::F16 => DType::F16,
            RawDType::Bf16 => DType::Bf16,
            RawDType::Fp8E4m3 => DType::Fp8E4m3,
            RawDType::U8 => DType::F32, // token ids etc.
        }
    }
}

#[derive(Debug)]
struct RawEntry {
    shape: Vec<usize>,
    dtype: RawDType,
    start: u64,
    end: u64,
}

fn parse_header(bytes: &[u8]) -> Result<(HashMap<String, RawEntry>, usize)> {
    if bytes.len() < 8 {
        return Err(FerriteError::Config("safetensors: file too short".into()));
    }
    let header_len = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
    if 8 + header_len > bytes.len() {
        return Err(FerriteError::Config("safetensors: header length out of range".into()));
    }
    let header: serde_json::Value = serde_json::from_slice(&bytes[8..8 + header_len]).map_err(
        |e| FerriteError::Config(format!("safetensors: header JSON parse: {e}")),
    )?;
    let obj = header
        .as_object()
        .ok_or_else(|| FerriteError::Config("safetensors: header is not an object".into()))?;
    let mut out = HashMap::new();
    for (name, v) in obj {
        if name == "__metadata__" {
            continue;
        }
        let vo = v
            .as_object()
            .ok_or_else(|| FerriteError::Config(format!("safetensors: bad entry {name}")))?;
        let dtype = vo
            .get("dtype")
            .and_then(|d| d.as_str())
            .and_then(RawDType::from_str)
            .ok_or_else(|| {
                FerriteError::Config(format!("safetensors: unknown dtype for {name}"))
            })?;
        let shape: Vec<usize> = vo
            .get("shape")
            .and_then(|s| s.as_array())
            .ok_or_else(|| FerriteError::Config(format!("safetensors: bad shape for {name}")))?
            .iter()
            .map(|d| d.as_u64().unwrap_or(0) as usize)
            .collect();
        let offs = vo
            .get("data_offsets")
            .and_then(|o| o.as_array())
            .and_then(|a| {
                if a.len() == 2 {
                    Some((a[0].as_u64()?, a[1].as_u64()?))
                } else {
                    None
                }
            })
            .ok_or_else(|| {
                FerriteError::Config(format!("safetensors: bad data_offsets for {name}"))
            })?;
        out.insert(name.clone(), RawEntry { shape, dtype, start: offs.0, end: offs.1 });
    }
    Ok((out, 8 + header_len))
}

/// BF16 bits → f32 (shift into the top 16 bits of an f32).
fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// IEEE f16 bits → f32.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as i32;
    let frac = (bits & 0x3ff) as u32;
    let f = if exp == 0 {
        if frac == 0 {
            0.0f32
        } else {
            // subnormal
            let mut e = -1i32;
            let mut m = frac;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3ff;
            let adjusted_exp = 127 - 15 + e;
            f32::from_bits((sign << 31) | ((adjusted_exp as u32) << 23) | (m << 13))
        }
    } else if exp == 0x1f {
        // inf / nan
        f32::from_bits((sign << 31) | (0xffu32 << 23) | (frac << 13))
    } else {
        let adjusted_exp = (exp - 15 + 127) as u32;
        f32::from_bits((sign << 31) | (adjusted_exp << 23) | (frac << 13))
    };
    if sign == 1 { -f } else { f }
}

/// FP8 E4M3 bits → f32 (1 sign, 4 exp bias 7, 3 mantissa; no inf, max 448).
fn e4m3_to_f32(bits: u8) -> f32 {
    let sign = ((bits >> 7) & 1) as u32;
    let exp = ((bits >> 3) & 0xf) as i32;
    let frac = (bits & 0x7) as u32;
    let v = if exp == 0 && frac == 0 {
        0.0f32
    } else if exp == 0 {
        // subnormal: value = frac * 2^-9
        (frac as f32) * (2.0f32).powi(-9)
    } else if exp == 0xf && frac == 0x7 {
        // NaN (E4M3 has no inf; 0x7F is NaN)
        f32::NAN
    } else {
        let e = exp - 7; // bias 7
        let mantissa = 1.0 + (frac as f32) / 8.0;
        mantissa * (2.0f32).powi(e)
    };
    if sign == 1 { -v } else { v }
}

/// Decode raw bytes of one tensor into f32 storage.
fn decode_entry(raw: &[u8], entry: &RawEntry) -> Vec<f32> {
    let n: usize = entry.shape.iter().product::<usize>().max(1);
    let out = match entry.dtype {
        RawDType::F32 => {
            let mut v = Vec::with_capacity(n);
            for i in 0..n {
                let b = raw[i * 4..i * 4 + 4].try_into().unwrap();
                v.push(f32::from_le_bytes(b));
            }
            v
        }
        RawDType::Bf16 => {
            let mut v = Vec::with_capacity(n);
            for i in 0..n {
                let b = raw[i * 2..i * 2 + 2].try_into().unwrap();
                v.push(bf16_to_f32(u16::from_le_bytes(b)));
            }
            v
        }
        RawDType::F16 => {
            let mut v = Vec::with_capacity(n);
            for i in 0..n {
                let b = raw[i * 2..i * 2 + 2].try_into().unwrap();
                v.push(f16_to_f32(u16::from_le_bytes(b)));
            }
            v
        }
        RawDType::Fp8E4m3 => (0..n).map(|i| e4m3_to_f32(raw[i])).collect(),
        RawDType::U8 => (0..n).map(|i| raw[i] as f32).collect(),
    };
    out
}

/// Load one .safetensors file into name → Tensor (f32 storage).
pub fn load_safetensors_file(path: impl AsRef<Path>) -> Result<HashMap<String, Tensor>> {
    let bytes = fs::read(path.as_ref()).map_err(|e| {
        FerriteError::Config(format!("safetensors: read {}: {e}", path.as_ref().display()))
    })?;
    let (entries, data_start) = parse_header(&bytes)?;
    let mut out = HashMap::with_capacity(entries.len());
    for (name, entry) in &entries {
        let raw = &bytes[data_start + entry.start as usize..data_start + entry.end as usize];
        let expected = (entry.end - entry.start) as usize;
        let expect_elems: usize = entry.shape.iter().product::<usize>().max(1);
        if expected != expect_elems * entry.dtype.element_size() {
            return Err(FerriteError::Config(format!(
                "safetensors: {name} byte length {expected} != {} * {}",
                expect_elems,
                entry.dtype.element_size()
            )));
        }
        let data = decode_entry(raw, entry);
        out.insert(
            name.clone(),
            Tensor::new(Shape::new(entry.shape.clone()), entry.dtype.ferrite_dtype(), data),
        );
    }
    Ok(out)
}

/// Load every *.safetensors in a directory (sharded checkpoints), merged.
pub fn load_safetensors_dir(dir: impl AsRef<Path>) -> Result<HashMap<String, Tensor>> {
    let mut out = HashMap::new();
    let mut files: Vec<_> = fs::read_dir(dir.as_ref())
        .map_err(|e| FerriteError::Config(format!("safetensors: read dir: {e}")))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map(|x| x == "safetensors").unwrap_or(false))
        .collect();
    files.sort();
    if files.is_empty() {
        return Err(FerriteError::Config(format!(
            "safetensors: no *.safetensors in {}",
            dir.as_ref().display()
        )));
    }
    for f in files {
        let part = load_safetensors_file(&f)?;
        out.extend(part);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_st_file(path: &std::path::Path, entries: &[(&str, Vec<u32>, Vec<u8>, &str)]) {
        // tiny safetensors writer for tests
        let mut header = String::from("{");
        let mut blob: Vec<u8> = Vec::new();
        let mut offs = Vec::new();
        for (name, shape, bytes, dtype) in entries {
            let start = blob.len() as u64;
            blob.extend_from_slice(bytes);
            let end = blob.len() as u64;
            offs.push((name, shape.clone(), *dtype, start, end));
        }
        let items: Vec<String> = offs
            .iter()
            .map(|(n, s, d, st, e)| {
                let shape: Vec<String> = s.iter().map(|x| x.to_string()).collect();
                format!("\"{n}\":{{\"dtype\":\"{d}\",\"shape\":[{}],\"data_offsets\":[{st},{e}]}}", shape.join(","))
            })
            .collect();
        header.push_str(&items.join(","));
        header.push('}');
        // safetensors spec: header (json + space padding) total length goes
        // into the 8-byte LE prefix; blob starts right after, 8-aligned.
        while (8 + header.len()) % 8 != 0 {
            header.push(' ');
        }
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(&blob);
        fs::write(path, out).unwrap();
    }

    #[test]
    fn roundtrip_f32_bf16_e4m3() {
        let dir = std::env::temp_dir().join("ferrite_st_test");
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.safetensors");
        // f32 [2,2], bf16 [3], e4m3 [2]
        let f32_data: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        let bf16_data: Vec<u8> = [0x3f80u16, 0xc000, 0x0000] // 1.0, -2.0, 0.0
            .iter().flat_map(|b| b.to_le_bytes()).collect();
        let e4m3_data: Vec<u8> = vec![0x38, 0xC0, 0x00]; // 1.0, -2.0, 0.0
        write_st_file(
            &path,
            &[
                ("w.f32", vec![2, 2], f32_data, "F32"),
                ("w.bf16", vec![3], bf16_data, "BF16"),
                ("w.fp8", vec![3], e4m3_data, "F8_E4M3"),
            ],
        );
        let t = load_safetensors_file(&path).unwrap();
        assert_eq!(t["w.f32"].as_slice(), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(t["w.bf16"].as_slice(), &[1.0, -2.0, 0.0]);
        assert_eq!(t["w.fp8"].as_slice()[0], 1.0);
        assert_eq!(t["w.fp8"].as_slice()[1], -2.0);
        assert_eq!(t["w.fp8"].as_slice()[2], 0.0);
        assert_eq!(t["w.f32"].dtype, DType::F32);
        assert_eq!(t["w.fp8"].dtype, DType::Fp8E4m3);
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn e4m3_conversions() {
        // 0x38 = exp 7 (0) mant 0 → 1.0; 0x40 = exp 8 → 2.0; 0x44 → 2.5?
        assert_eq!(e4m3_to_f32(0x38), 1.0);
        assert_eq!(e4m3_to_f32(0x40), 2.0);
        assert_eq!(e4m3_to_f32(0x44), 3.0); // (1+4/8)*2^1
        assert_eq!(e4m3_to_f32(0xB8), -1.0);
        assert_eq!(e4m3_to_f32(0x00), 0.0);
        assert_eq!(e4m3_to_f32(0x50), 8.0);
        assert_eq!(e4m3_to_f32(0x04), 4.0 * 2.0f32.powi(-9)); // subnormal frac=4
    }

    #[test]
    fn bf16_f16_conversions() {
        assert_eq!(bf16_to_f32(0x3F80), 1.0);
        assert_eq!(bf16_to_f32(0xC000), -2.0);
        assert_eq!(f16_to_f32(0x3C00), 1.0);
        assert_eq!(f16_to_f32(0x8000), -0.0);
        assert_eq!(f16_to_f32(0x4000), 2.0);
        assert_eq!(f16_to_f32(0x3800), 0.5);
    }
}
