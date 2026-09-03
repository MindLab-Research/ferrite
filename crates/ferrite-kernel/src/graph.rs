//! CUDA-graph semantics, device-agnostic.
//!
//! A backend that is `GraphCapable` can *capture* the sequence of kernel
//! ops executed for one engine step and later *replay* that sequence with
//! fresh data in the same static buffers. On CUDA this maps 1:1 onto
//! stream capture (`cuStreamBeginCapture` → `cuGraphInstantiate` →
//! `cuGraphLaunch`), eliminating per-op launch overhead in the decode
//! loop — the must-have for SGLang-class decode latency.
//!
//! The CPU reference implements the same contract with an op *trace*:
//! capture records `(op, shapes)` tuples; verification replays a step and
//! asserts the structure is identical. Structural stability across steps
//! is exactly the precondition for CUDA graph reuse — if the op sequence
//! is stable per batch shape, the graph replays correctly.

use crate::KernelBackend;

/// One recorded kernel invocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpRecord {
    pub op: &'static str,
    /// Input/output shapes participating in the op.
    pub shapes: Vec<Vec<usize>>,
}

impl OpRecord {
    pub fn new(op: &'static str, shapes: Vec<Vec<usize>>) -> Self {
        OpRecord { op, shapes }
    }
}

/// A captured op sequence for one engine step.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct OpTrace {
    pub ops: Vec<OpRecord>,
}

impl OpTrace {
    pub fn len(&self) -> usize {
        self.ops.len()
    }
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }
}

/// Capture/replay contract (see module docs for the CUDA mapping).
pub trait GraphCapable: KernelBackend {
    /// Start recording every kernel op (CUDA: begin stream capture).
    fn begin_capture(&self);

    /// Stop recording and take the trace (CUDA: end capture + instantiate
    /// the graph; returns a handle on real GPUs).
    fn end_capture(&self) -> OpTrace;

    /// Start verifying the *next* executed sequence against `trace`
    /// (CUDA: this is a replay launch into the same static buffers; the
    /// CPU reference checks structural equality instead).
    fn begin_verify(&self, trace: &OpTrace);

    /// Stop verifying and report whether the executed sequence matched
    /// (CUDA: replay succeeded; CPU: op-by-op equality held).
    fn end_verify(&self) -> bool;
}

/// Trace state machine shared by the CPU recorder.
#[derive(Debug)]
pub enum TraceMode {
    Off,
    Capture,
    Verify {
        trace: OpTrace,
        pos: usize,
        ok: bool,
    },
}

impl Default for TraceMode {
    fn default() -> Self {
        TraceMode::Off
    }
}

#[derive(Debug, Default)]
pub struct Recorder {
    pub mode: TraceMode,
    pub ops: Vec<OpRecord>,
}

impl Recorder {
    pub fn new() -> Self {
        Recorder { mode: TraceMode::Off, ops: Vec::new() }
    }

    /// Called at the top of every kernel impl. Cheap when Off.
    pub fn record(&mut self, op: &'static str, shapes: Vec<Vec<usize>>) {
        match &mut self.mode {
            TraceMode::Off => {}
            TraceMode::Capture => {
                self.ops.push(OpRecord::new(op, shapes));
            }
            TraceMode::Verify { trace, pos, ok } => {
                if *ok {
                    let rec = OpRecord::new(op, shapes);
                    if *pos >= trace.ops.len() || trace.ops[*pos] != rec {
                        *ok = false;
                    } else {
                        *pos += 1;
                    }
                }
            }
        }
    }

    pub fn begin_capture(&mut self) {
        self.ops.clear();
        self.mode = TraceMode::Capture;
    }

    pub fn end_capture(&mut self) -> OpTrace {
        self.mode = TraceMode::Off;
        OpTrace { ops: std::mem::take(&mut self.ops) }
    }

    pub fn begin_verify(&mut self, trace: OpTrace) {
        self.mode = TraceMode::Verify { trace, pos: 0, ok: true };
    }

    pub fn end_verify(&mut self) -> bool {
        match std::mem::replace(&mut self.mode, TraceMode::Off) {
            TraceMode::Verify { trace, pos, ok } => ok && pos == trace.ops.len(),
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recorder_capture_and_verify() {
        let mut r = Recorder::new();
        r.begin_capture();
        r.record("matmul", vec![vec![4, 8], vec![16, 8]]);
        r.record("rmsnorm", vec![vec![4, 8]]);
        let trace = r.end_capture();
        assert_eq!(trace.len(), 2);
        // matching replay
        r.begin_verify(trace.clone());
        r.record("matmul", vec![vec![4, 8], vec![16, 8]]);
        r.record("rmsnorm", vec![vec![4, 8]]);
        assert!(r.end_verify(), "identical sequence verifies");
        // divergent replay (extra op)
        r.begin_verify(trace.clone());
        r.record("matmul", vec![vec![4, 8], vec![16, 8]]);
        r.record("rmsnorm", vec![vec![4, 8]]);
        r.record("matmul", vec![vec![4, 8], vec![16, 8]]);
        assert!(!r.end_verify(), "extra op fails");
        // shape change (batch bucket change)
        r.begin_verify(trace);
        r.record("matmul", vec![vec![8, 8], vec![16, 8]]);
        r.record("rmsnorm", vec![vec![8, 8]]);
        assert!(!r.end_verify(), "shape change fails — needs its own bucket");
    }
}
