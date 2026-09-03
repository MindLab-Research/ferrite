//! ferrite-types: core tensor primitives for the ferrite inference engine.
//!
//! First version is CPU-only with f32 storage for all dtypes. `DType` marks
//! the *intended* element type of the tensor (what a GPU backend would
//! eventually load); the CPU reference backend keeps everything in f32 so
//! numerical smoke tests are exact and deterministic.

use std::fmt;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    /// 32-bit float (reference backend native).
    F32,
    /// 16-bit brain float (intended on-device dtype).
    Bf16,
    /// 16-bit IEEE float.
    F16,
    /// 8-bit FP8 E4M3 (weights in the real GLM-5.3-Flash checkpoint).
    Fp8E4m3,
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DType::F32 => write!(f, "f32"),
            DType::Bf16 => write!(f, "bf16"),
            DType::F16 => write!(f, "f16"),
            DType::Fp8E4m3 => write!(f, "fp8_e4m3"),
        }
    }
}

/// Row-major shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape(pub Vec<usize>);

impl Shape {
    pub fn new(dims: impl IntoIterator<Item = usize>) -> Self {
        Shape(dims.into_iter().collect())
    }

    pub fn numel(&self) -> usize {
        // Empty product = 1 for rank-0 scalars; a zero dim makes it 0
        // (0-token shards are legal: empty DCP ranks have LSE=-inf).
        self.0.iter().product::<usize>()
    }

    pub fn rank(&self) -> usize {
        self.0.len()
    }

    /// Row-major strides (in elements).
    pub fn strides(&self) -> Vec<usize> {
        let mut s = vec![1usize; self.0.len()];
        for i in (0..self.0.len().saturating_sub(1)).rev() {
            s[i] = s[i + 1] * self.0[i + 1];
        }
        s
    }

    pub fn as_slice(&self) -> &[usize] {
        &self.0
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}]", self.0.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", "))
    }
}

/// Immutable CPU tensor. Data is always stored as f32 regardless of `dtype`;
/// `dtype` records what the tensor *means* (e.g. fp8 weights dequantized on
/// load by the CPU reference backend; a CUDA backend would keep the raw
/// bytes and interpret them natively).
#[derive(Debug, Clone)]
pub struct Tensor {
    pub shape: Shape,
    pub dtype: DType,
    pub data: Arc<Vec<f32>>,
}

impl Tensor {
    pub fn new(shape: Shape, dtype: DType, data: Vec<f32>) -> Self {
        assert_eq!(
            shape.numel(),
            data.len(),
            "tensor data length {} != shape {} numel",
            data.len(),
            shape.numel()
        );
        Tensor { shape, dtype, data: Arc::new(data) }
    }

    pub fn zeros(shape: Shape, dtype: DType) -> Self {
        let n = shape.numel();
        Tensor::new(shape, dtype, vec![0.0; n])
    }

    pub fn from_f32(shape: Shape, data: Vec<f32>) -> Self {
        Tensor::new(shape, DType::F32, data)
    }

    /// 2-D matrix [rows, cols].
    pub fn mat(rows: usize, cols: usize, data: Vec<f32>) -> Self {
        Tensor::from_f32(Shape::new([rows, cols]), data)
    }

    /// 1-D vector.
    pub fn vec(data: Vec<f32>) -> Self {
        Tensor::from_f32(Shape::new([data.len()]), data)
    }

    pub fn numel(&self) -> usize {
        self.shape.numel()
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    /// Row-major 2-D view helpers (last two dims).
    pub fn rows(&self) -> usize {
        assert!(self.shape.rank() >= 2, "need >=2 dims for rows(), got {}", self.shape);
        self.shape.0[self.shape.rank() - 2]
    }
    pub fn cols(&self) -> usize {
        assert!(self.shape.rank() >= 2, "need >=2 dims for cols(), got {}", self.shape);
        self.shape.0[self.shape.rank() - 1]
    }

    pub fn is_same_shape(&self, other: &Tensor) -> bool {
        self.shape == other.shape
    }
}

/// Errors shared across ferrite crates.
#[derive(Debug, thiserror::Error)]
pub enum FerriteError {
    #[error("shape mismatch: expected {expected}, got {got}")]
    ShapeMismatch { expected: Shape, got: Shape },
    #[error("invalid argument: {0}")]
    InvalidArg(String),
    #[error("index out of bounds: {index} (len {len})")]
    IndexOutOfBounds { index: usize, len: usize },
    #[error("config error: {0}")]
    Config(String),
    #[error("kv/state pool error: {0}")]
    Pool(String),
    #[error("scheduler error: {0}")]
    Scheduler(String),
}

pub type Result<T> = std::result::Result<T, FerriteError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_strides_row_major() {
        let s = Shape::new([2, 3, 4]);
        assert_eq!(s.numel(), 24);
        assert_eq!(s.strides(), vec![12, 4, 1]);
    }

    #[test]
    fn tensor_construct_and_access() {
        let t = Tensor::mat(2, 3, (0..6).map(|i| i as f32).collect());
        assert_eq!(t.as_slice()[4], 4.0);
        assert_eq!(t.rows(), 2);
        assert_eq!(t.cols(), 3);
        let z = Tensor::zeros(Shape::new([4]), DType::Bf16);
        assert_eq!(z.dtype, DType::Bf16);
        assert_eq!(z.numel(), 4);
    }
}
