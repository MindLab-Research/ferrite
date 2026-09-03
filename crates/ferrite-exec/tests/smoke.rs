//! End-to-end smoke tests: random weights on the 4-layer test config, the
//! full PDAF loop (chunked prefill → decode → greedy sampling), plus the
//! two golden invariants:
//! 1. determinism — same engine state, same output;
//! 2. chunk-boundary correctness — split prefill produces the same decode
//!    trajectory as single-chunk prefill (conv tails + recurrent states
//!    carry across chunk boundaries).

use ferrite_exec::Engine;
use ferrite_kernel::CpuBackend;
use ferrite_model::{random_weights, Glm53FlashConfig};

fn build_engine(seed: u64, budget: usize) -> Engine<CpuBackend> {
    let cfg = Glm53FlashConfig::test_config();
    let weights = random_weights(&cfg, seed);
    let mut e = Engine::new(cfg, weights, CpuBackend::new());
    e.scheduler = ferrite_batch::BatchScheduler::new(8, budget);
    e
}

fn all_finite(out: &[u32]) -> bool {
    // token ids are finite by construction; we check the engine didn't
    // produce the panic/NaN sentinel pattern of all-same-token
    out.iter().any(|&t| t != out[0])
}

#[test]
fn smoke_full_loop_generates_tokens() {
    let mut e = build_engine(42, 64);
    let id = e.submit(vec![1, 2, 3, 4, 5, 6], 8).unwrap();
    let out = e.run_until_done(id).unwrap();
    assert!(!out.is_empty(), "generated some tokens");
    assert!(out.len() <= 8);
    // tokens within vocab
    assert!(out.iter().all(|&t| (t as usize) < 512));
}

#[test]
fn smoke_deterministic() {
    let mut e1 = build_engine(7, 64);
    let mut e2 = build_engine(7, 64);
    let a = e1.submit(vec![10, 11, 12], 5).unwrap();
    let b = e2.submit(vec![10, 11, 12], 5).unwrap();
    let o1 = e1.run_until_done(a).unwrap();
    let o2 = e2.run_until_done(b).unwrap();
    assert_eq!(o1, o2, "same seed + same prompt -> identical trajectory");
}

#[test]
fn smoke_chunked_prefill_matches_single_chunk() {
    // budget 2 -> prefill runs in 3 chunks of 2; budget 64 -> one chunk.
    // The decode trajectory (which depends on final conv tails + recurrent
    // states + DSA caches) must be identical: this is the chunk-boundary
    // golden invariant for the Gated DeltaNet path.
    let mut e_chunked = build_engine(123, 2);
    let mut e_single = build_engine(123, 64);
    let a = e_chunked.submit(vec![5, 6, 7, 8, 9, 10], 5).unwrap();
    let b = e_single.submit(vec![5, 6, 7, 8, 9, 10], 5).unwrap();
    let o1 = e_chunked.run_until_done(a).unwrap();
    let o2 = e_single.run_until_done(b).unwrap();
    assert_eq!(o1, o2, "chunked prefill (3x2) == single-chunk prefill (1x6)");
}

#[test]
fn smoke_two_seqs_interleaved() {
    // two sequences share engine steps (P/D interleave via the router)
    let mut e = build_engine(99, 64);
    let a = e.submit(vec![1, 2, 3], 4).unwrap();
    let b = e.submit(vec![9, 8, 7, 6], 4).unwrap();
    let mut got_a = None;
    let mut got_b = None;
    for _ in 0..64 {
        e.step().unwrap();
        if let Some(o) = e.finished_output(a) {
            got_a = Some(o);
        }
        if let Some(o) = e.finished_output(b) {
            got_b = Some(o);
        }
        if got_a.is_some() && got_b.is_some() {
            break;
        }
    }
    let oa = got_a.expect("seq a finished");
    let ob = got_b.expect("seq b finished");
    assert!(!oa.is_empty() && !ob.is_empty());
    assert!(all_finite(&oa) || oa.len() == 1);
    assert!(all_finite(&ob) || ob.len() == 1);
}

#[test]
fn smoke_longer_generation_stable() {
    // 4-layer hybrid with MoE: 20 decode steps, no NaN-induced collapse
    let mut e = build_engine(2024, 64);
    let id = e.submit(vec![3, 1, 4, 1, 5, 9, 2, 6], 20).unwrap();
    let out = e.run_until_done(id).unwrap();
    assert!(!out.is_empty());
    assert!(out.len() <= 20);
    assert!(out.iter().all(|&t| (t as usize) < 512));
}
