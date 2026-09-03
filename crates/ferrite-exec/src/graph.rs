//! CUDA-graph feasibility runner for the engine's decode path.
//!
//! On the CPU reference this captures the op sequence of one decode step
//! and verifies that every subsequent decode step replays the *identical*
//! sequence — which is precisely the precondition for replaying a CUDA
//! graph (`cuGraphLaunch`) on B300: static op structure, dynamic data.
//!
//! Usage pattern (mirrors what the CUDA backend does with real graphs):
//! 1. First decode step runs cold with `begin_capture` — the trace is the
//!    "graph" (on GPU: `cuStreamEndCapture` + `cuGraphInstantiate`).
//! 2. Every later decode step runs under `begin_verify(trace)` and asserts
//!    the same op sequence (on GPU: `cuGraphLaunch` into static buffers).

use ferrite_kernel::GraphCapable;
use ferrite_types::Result;

use crate::Engine;

/// Graph-runner state attached to an engine.
#[derive(Debug, Default)]
pub struct GraphRunner {
    /// Captured decode-step op sequence (the "graph").
    trace: Option<ferrite_kernel::OpTrace>,
    /// Number of steps replayed against the captured trace.
    replays: usize,
    /// True iff every replay so far matched.
    stable: bool,
}

impl GraphRunner {
    pub fn new() -> Self {
        GraphRunner { trace: None, replays: 0, stable: true }
    }

    pub fn trace(&self) -> Option<&ferrite_kernel::OpTrace> {
        self.trace.as_ref()
    }

    pub fn replays(&self) -> usize {
        self.replays
    }

    pub fn stable(&self) -> bool {
        self.stable
    }
}

impl<B: GraphCapable> Engine<B> {
    /// Run one decode step under graph semantics: capture on first use,
    /// verify afterwards. Returns the generated token.
    pub fn graph_decode_step(&mut self, seq: u64, gr: &mut GraphRunner) -> Result<u32> {
        // decide mode before executing
        let mode = if gr.trace.is_none() {
            Mode::Capture
        } else if gr.stable {
            Mode::Verify
        } else {
            Mode::Cold
        };
        match mode {
            Mode::Capture => {
                self.backend.begin_capture();
                let tok = self.decode_step(seq)?;
                gr.trace = Some(self.backend.end_capture());
                Ok(tok)
            }
            Mode::Verify => {
                let trace = gr.trace.clone().unwrap();
                self.backend.begin_verify(&trace);
                let tok = self.decode_step(seq)?;
                let ok = self.backend.end_verify();
                gr.replays += 1;
                if !ok {
                    gr.stable = false;
                }
                Ok(tok)
            }
            Mode::Cold => self.decode_step(seq),
        }
    }
}

enum Mode {
    Capture,
    Verify,
    Cold,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrite_kernel::CpuBackend;
    use ferrite_model::{random_weights, Glm53FlashConfig};

    fn engine() -> Engine<CpuBackend> {
        let cfg = Glm53FlashConfig::test_config();
        let weights = random_weights(&cfg, 33);
        Engine::new(cfg, weights, CpuBackend::new())
    }

    #[test]
    fn decode_op_sequence_is_stable() {
        // CUDA-graph feasibility: the decode-step op sequence must be
        // identical across steps (and across sequences of the same config).
        let mut e = engine();
        let a = e.submit(vec![1, 2, 3, 4, 5], 6).unwrap();
        let mut gr = GraphRunner::new();
        // drive prefill to completion via normal steps
        let mut i = 0;
        while e.scheduler.seq(a).map(|s| s.prefilled < s.prompt.len()).unwrap_or(false) {
            e.step().unwrap();
            i += 1;
            assert!(i < 20, "prefill should finish");
        }
        e.step().unwrap(); // post_step promotes to decoding
        // graph decode steps
        let mut tokens = Vec::new();
        for _ in 0..4 {
            let t = e.graph_decode_step(a, &mut gr).unwrap();
            e.scheduler.record_token(a, t).unwrap();
            if let Some(s) = e.seqs.get_mut(&a) {
                s.tokens.push(t);
            }
            tokens.push(t);
        }
        assert!(gr.trace().is_some(), "captured a trace");
        assert!(gr.replays() >= 3, "replayed >=3 steps");
        assert!(gr.stable(), "op sequence stable — CUDA graph is feasible");
        assert!(tokens.iter().all(|&t| (t as usize) < 512));
    }

    #[test]
    fn sequence_stable_across_seqs() {
        // different sequences share the same decode op structure
        let mut e = engine();
        let a = e.submit(vec![7, 8, 9], 4).unwrap();
        let mut gr = GraphRunner::new();
        let mut i = 0;
        while e.scheduler.seq(a).map(|s| s.prefilled < s.prompt.len()).unwrap_or(false) {
            e.step().unwrap();
            i += 1;
            assert!(i < 20);
        }
        e.step().unwrap();
        for _ in 0..3 {
            let t = e.graph_decode_step(a, &mut gr).unwrap();
            e.scheduler.record_token(a, t).unwrap();
            if let Some(s) = e.seqs.get_mut(&a) {
                s.tokens.push(t);
            }
        }
        assert!(gr.stable(), "stable within seq a");
        // a new sequence (fresh states) must replay the same op sequence
        let b = e.submit(vec![20, 21], 3).unwrap();
        let mut j = 0;
        while e.scheduler.seq(b).map(|s| s.prefilled < s.prompt.len()).unwrap_or(false) {
            e.step().unwrap();
            j += 1;
            assert!(j < 20);
        }
        e.step().unwrap();
        for _ in 0..2 {
            let t = e.graph_decode_step(b, &mut gr).unwrap();
            e.scheduler.record_token(b, t).unwrap();
            if let Some(s) = e.seqs.get_mut(&b) {
                s.tokens.push(t);
            }
        }
        assert!(gr.stable(), "same op sequence across sequences — graph buckets by config, not data");
    }
}
