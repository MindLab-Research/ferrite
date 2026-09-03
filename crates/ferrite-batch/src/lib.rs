//! ferrite-batch: sequence lifecycle + continuous batching.
//!
//! A `Sequence` walks `Waiting → Prefilling → Decoding → Finished`.
//! `BatchScheduler` assembles the next *scheduled batch* per engine step:
//! chunked-prefill work first (bounded by a token budget), then decode work
//! (bounded by batch size) — the two phases map onto the P/D executors of
//! the PDAF plan (ferrite-scheduler).

use std::collections::{HashMap, VecDeque};

use ferrite_types::{FerriteError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    /// Queued, nothing prefilled yet.
    Waiting,
    /// Some (maybe all) prompt tokens processed in chunks.
    Prefilling,
    /// Prompt done; generating tokens one step at a time.
    Decoding,
    /// EOS or max-len reached.
    Finished,
}

#[derive(Debug, Clone)]
pub struct Sequence {
    pub id: u64,
    pub prompt: Vec<u32>,
    pub output: Vec<u32>,
    pub phase: Phase,
    /// Prompt tokens already consumed by prefill (chunked).
    pub prefilled: usize,
    pub eos_token: u32,
    pub max_new_tokens: usize,
}

impl Sequence {
    /// Context length the engine must attend over (prompt + generated).
    pub fn context_len(&self) -> usize {
        self.prompt.len() + self.output.len()
    }

    /// Tokens still to prefill (for the chunked prefill budget).
    pub fn remaining_prefill(&self) -> usize {
        self.prompt.len() - self.prefilled
    }

    pub fn is_decode_ready(&self) -> bool {
        self.phase == Phase::Prefilling && self.prefilled == self.prompt.len()
    }
}

/// One engine step's worth of work, split by phase (P vs D in PDAF).
#[derive(Debug, Clone, Default)]
pub struct ScheduledBatch {
    /// (seq id, chunk token count) for prefill work this step.
    pub prefill: Vec<(u64, usize)>,
    /// seq ids decoding this step (one token each).
    pub decode: Vec<u64>,
}

impl ScheduledBatch {
    pub fn total_prefill_tokens(&self) -> usize {
        self.prefill.iter().map(|(_, c)| c).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.prefill.is_empty() && self.decode.is_empty()
    }
}

/// Continuous batching over sequences.
pub struct BatchScheduler {
    seqs: HashMap<u64, Sequence>,
    order: VecDeque<u64>,
    next_id: u64,
    /// Max decode entries per step (running requests cap).
    pub max_running: usize,
    /// Chunked prefill token budget per step.
    pub prefill_token_budget: usize,
}

impl BatchScheduler {
    pub fn new(max_running: usize, prefill_token_budget: usize) -> Self {
        BatchScheduler {
            seqs: HashMap::new(),
            order: VecDeque::new(),
            next_id: 0,
            max_running,
            prefill_token_budget,
        }
    }

    pub fn submit(&mut self, prompt: Vec<u32>, eos_token: u32, max_new_tokens: usize) -> Result<u64> {
        if prompt.is_empty() {
            return Err(FerriteError::InvalidArg("empty prompt".into()));
        }
        let id = self.next_id;
        self.next_id += 1;
        let seq = Sequence {
            id,
            prompt,
            output: Vec::new(),
            phase: Phase::Waiting,
            prefilled: 0,
            eos_token,
            max_new_tokens,
        };
        self.seqs.insert(id, seq);
        self.order.push_back(id);
        Ok(id)
    }

    pub fn seq(&self, id: u64) -> Result<&Sequence> {
        self.seqs
            .get(&id)
            .ok_or_else(|| FerriteError::InvalidArg(format!("unknown seq {id}")))
    }

    pub fn seq_mut(&mut self, id: u64) -> Result<&mut Sequence> {
        self.seqs
            .get_mut(&id)
            .ok_or_else(|| FerriteError::InvalidArg(format!("unknown seq {id}")))
    }

    pub fn running(&self) -> impl Iterator<Item = &Sequence> {
        self.seqs
            .values()
            .filter(|s| !matches!(s.phase, Phase::Waiting | Phase::Finished))
    }

    pub fn finished(&self) -> impl Iterator<Item = &Sequence> {
        self.seqs.values().filter(|s| s.phase == Phase::Finished)
    }

    /// Advance scheduling state after the engine executed a batch:
    /// promote finished-prefill seqs to decoding, retire finished seqs.
    pub fn post_step(&mut self, _batch: &ScheduledBatch) {
        for s in self.seqs.values_mut() {
            match s.phase {
                Phase::Prefilling if s.prefilled == s.prompt.len() => s.phase = Phase::Decoding,
                _ => {}
            }
        }
    }

    /// Record that seq `id` generated `token`.
    pub fn record_token(&mut self, id: u64, token: u32) -> Result<()> {
        let s = self.seq_mut(id)?;
        match s.phase {
            Phase::Decoding => {
                s.output.push(token);
                if token == s.eos_token || s.output.len() >= s.max_new_tokens {
                    s.phase = Phase::Finished;
                }
                Ok(())
            }
            _ => Err(FerriteError::InvalidArg(format!(
                "seq {id} not decoding (phase {:?})",
                s.phase
            ))),
        }
    }

    /// Assemble the next step: prefill chunks (budgeted) + decode entries,
    /// respecting the running cap. Waiting seqs only enter when a decode
    /// slot is free (continuous batching admission).
    pub fn next_batch(&mut self) -> ScheduledBatch {
        let mut batch = ScheduledBatch::default();
        // how many currently active (non-waiting, non-finished)
        let active = self
            .seqs
            .values()
            .filter(|s| !matches!(s.phase, Phase::Waiting | Phase::Finished))
            .count();
        let mut free = self.max_running.saturating_sub(active);

        // 1) prefill work for admitted seqs (Prefilling phase)
        let mut budget = self.prefill_token_budget;
        let ids: Vec<u64> = self.order.iter().copied().collect();
        for id in ids {
            if budget == 0 {
                break;
            }
            let (phase, rem) = {
                let s = &self.seqs[&id];
                (s.phase, s.remaining_prefill())
            };
            match phase {
                Phase::Prefilling => {
                    let chunk = rem.min(budget);
                    budget -= chunk;
                    batch.prefill.push((id, chunk));
                }
                Phase::Waiting if free > 0 => {
                    // admit
                    let chunk = rem.min(budget);
                    if chunk > 0 {
                        budget -= chunk;
                        free -= 1;
                        let s = self.seqs.get_mut(&id).unwrap();
                        s.phase = Phase::Prefilling;
                        batch.prefill.push((id, chunk));
                    }
                }
                _ => {}
            }
        }

        // 2) decode work
        for s in self.seqs.values() {
            if s.phase == Phase::Decoding && batch.decode.len() < self.max_running {
                batch.decode.push(s.id);
            }
        }
        batch
    }

    /// The engine calls this after consuming a prefill chunk of `n` tokens.
    pub fn advance_prefill(&mut self, id: u64, n: usize) -> Result<()> {
        let s = self.seq_mut(id)?;
        s.prefilled = (s.prefilled + n).min(s.prompt.len());
        Ok(())
    }

    pub fn finished_output(&self, id: u64) -> Result<Option<Vec<u32>>> {
        let s = self.seq(id)?;
        Ok(if s.phase == Phase::Finished {
            Some(s.output.clone())
        } else {
            None
        })
    }

    pub fn all_finished(&self) -> bool {
        !self.seqs.is_empty()
            && self.seqs.values().all(|s| s.phase == Phase::Finished)
    }

    /// Remove finished seqs (returns their ids for state-pool cleanup).
    pub fn reap_finished(&mut self) -> Vec<u64> {
        let done: Vec<u64> = self
            .seqs
            .iter()
            .filter(|(_, s)| s.phase == Phase::Finished)
            .map(|(id, _)| *id)
            .collect();
        for id in &done {
            self.seqs.remove(id);
        }
        self.order.retain(|id| !done.contains(id));
        done
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sched() -> BatchScheduler {
        BatchScheduler::new(4, 8)
    }

    #[test]
    fn lifecycle_wait_prefill_decode_finish() {
        let mut s = sched();
        let id = s.submit(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 99, 3).unwrap();
        // chunked prefill: budget 8 -> first batch prefills 8 tokens
        let b1 = s.next_batch();
        assert_eq!(b1.prefill, vec![(id, 8)]);
        assert!(b1.decode.is_empty());
        s.advance_prefill(id, 8).unwrap();
        s.post_step(&b1);
        assert_eq!(s.seq(id).unwrap().phase, Phase::Prefilling);
        // second batch: remaining 2 tokens
        let b2 = s.next_batch();
        assert_eq!(b2.prefill, vec![(id, 2)]);
        s.advance_prefill(id, 2).unwrap();
        s.post_step(&b2);
        assert_eq!(s.seq(id).unwrap().phase, Phase::Decoding);
        // decode steps
        let b3 = s.next_batch();
        assert_eq!(b3.decode, vec![id]);
        s.record_token(id, 42).unwrap();
        s.record_token(id, 43).unwrap();
        s.record_token(id, 99).unwrap(); // eos -> finished
        assert_eq!(s.seq(id).unwrap().phase, Phase::Finished);
        assert_eq!(s.finished_output(id).unwrap(), Some(vec![42, 43, 99]));
        assert!(s.all_finished());
        let reaped = s.reap_finished();
        assert_eq!(reaped, vec![id]);
    }

    #[test]
    fn continuous_admission() {
        let mut s = BatchScheduler::new(2, 100);
        let a = s.submit(vec![1, 2], 9, 1).unwrap();
        let b = s.submit(vec![3, 4], 9, 1).unwrap();
        let c = s.submit(vec![5, 6], 9, 1).unwrap();
        // max_running=2: a, b admitted; c waits
        let b1 = s.next_batch();
        let admitted: Vec<u64> = b1.prefill.iter().map(|(id, _)| *id).collect();
        assert!(admitted.contains(&a) && admitted.contains(&b));
        assert!(!admitted.contains(&c));
        assert_eq!(b1.total_prefill_tokens(), 4);
        s.advance_prefill(a, 2).unwrap();
        s.advance_prefill(b, 2).unwrap();
        s.post_step(&b1);
        // a finishes (max_new=1) -> slot frees, c admitted next step
        s.record_token(a, 9).unwrap(); // eos immediately
        let b2 = s.next_batch();
        let admitted2: Vec<u64> = b2.prefill.iter().map(|(id, _)| *id).collect();
        assert_eq!(admitted2, vec![c], "c admitted after a retired");
        assert_eq!(b2.decode, vec![b]);
    }

    #[test]
    fn mixed_prefill_decode_step() {
        // decoding seq + waiting seq in one scheduled step
        let mut s = BatchScheduler::new(4, 8);
        let a = s.submit(vec![1, 2, 3, 4], 9, 10).unwrap();
        let b = s.submit(vec![5, 6, 7, 8], 9, 10).unwrap();
        let b1 = s.next_batch();
        // budget 8 admits both in the first step
        assert_eq!(b1.total_prefill_tokens(), 8);
        s.advance_prefill(a, 4).unwrap();
        s.advance_prefill(b, 4).unwrap();
        s.post_step(&b1);
        // both decoding
        let b2 = s.next_batch();
        assert_eq!(b2.decode.len(), 2);
        assert!(b2.prefill.is_empty());
    }
}
