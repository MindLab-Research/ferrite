//! ferrite-models: the per-model static definitions.
//!
//! One engine, models as data: the runtime (device layer, engine contract,
//! serve stack) is shared, and every model — its config, quantisation
//! primitives, weight layout and layer chain — is one module here.
//!
//! * [`dsv41`] — DeepSeek-V4.1-Flash (`Dsv41Config`, the fp4/fp8-MX
//!   primitives, the engram n-gram tables, the CPU golden ops, the layer
//!   chain, the vision tower and the checkpoint loader).

pub mod dsv41;
