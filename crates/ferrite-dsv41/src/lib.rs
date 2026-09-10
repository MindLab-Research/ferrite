//! DeepSeek-V4.1-Flash support for ferrite.
//!
//! Layout:
//!   * [`config`]  — the released config, with every derived layer role
//!   * [`quant`]   — fp4 (e2m1) / fp8-e4m3 / ue8m0 block-scale primitives,
//!                   both directions, bit-exact against the reference kernels
//!   * [`engram`]  — the n-gram hash tables: bucket layout, the compressed
//!                   token map, per-(layer, lookback) multipliers and the
//!                   per-token hash ids
//!   * [`ops`]     — CPU reference implementations of every new operator
//!                   (the numerical golden standard the CUDA kernels match)
//!   * [`chain`]   — the model chain: Transformer forward + the DSpark draft
//!   * [`weights`] — checkpoint tensor names -> the engine's TP-sharded layout

pub mod chain;
pub mod config;
pub mod dspark;
pub mod engram;
pub mod kernels;
pub mod ops;
pub mod quant;
pub mod weights;

pub use config::{Dsv41Config, KvMode};
