//! DeepSeek-V4.1-Flash support for ferrite — a re-export shim.
//!
//! The model definition moved to [`ferrite_models::dsv41`] (one engine, models
//! as data). This crate keeps the historical module paths so every existing
//! consumer — the `dsv41-run` binary in `src/bin` and the integration tests in
//! `tests/` — builds unchanged. It stays in the workspace while the serve path
//! converges onto the single `ferrite-serve --model dsv41` binary; the crate
//! itself is deleted in a later phase.
//!
//! The model-owned modules now live at:
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

pub use ferrite_models::dsv41::{
    chain, chain_dev, config, device, dspark, engram, frame, kernels, load, ops, quant, tp, vision,
    weights,
};
pub use ferrite_models::dsv41::frame::Dsv41Frame;
pub use ferrite_models::dsv41::{Dsv41Config, KvMode};
