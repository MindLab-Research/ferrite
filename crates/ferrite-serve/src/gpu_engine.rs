//! GpuEngine: the CUDA TpCluster behind the ServeEngine seam.
//!
//! The HTTP layer (ferrite-http: axum routes + SSE + the driver thread)
//! speaks `ServeEngine`; this impl drives the real GLM-5.3-Flash TP4
//! cluster — per-seq prefill (`prefill_chunk`) + per-seq mega-graph
//! decode (`decode_step`) in a round-robin. Concurrency today is
//! INTERLEAVED decode (each tick = one decode step per live seq, each
//! ~20ms of graph replay on the GPU): N live requests each get ~1/N of
//! the single-stream token rate; true batched decode (the [B,...] state
//! row engine the scheduler's ExecBackend targets) is the next phase —
//! this seam is where it lands.
//!
//! Per-seq GPU state lifecycle: every request mints a fresh cluster seq
//! (u64 counter — never reused), and `deregister`/`cancel` release its
//! DSA caches + GDN states + mega graphs via `TpCluster::free_seq`
//! (~GBs per seq — without it the serve OOMs after a handful of
//! requests).
//!
//! MTP constraint (Wave 5): the legacy MTP step is the single-seq
//! round-robin — the verify graph is one seq's block (`mega_v{seq}`,
//! n = FERRITE_MTP_N rows) and the draft chain / accept / commit are per seq
//! — so with FERRITE_MTP=1 the engine forces max_seqs=1 by default. The
//! scratch is per seq (`ferrite_kernel::cuda::MtpStateB`) and the B-seq step
//! is wired behind `FERRITE_MTP_BATCHED=1` (`TpCluster::mtp_step_batched`,
//! plan from `ferrite_exec::mtp_batch`); the forcing-1 default lifts once
//! `mtp_batch::MTP_BATCH_READY` — the three kernel-side pieces (the
//! `ntok = n_v` mapped append call site, the single-launch batched commit, the
//! `(seqs, n_v)`-keyed B-row verify capture) — is true, see
//! `mtp_batch::MTP_BATCH_READY`'s doc for the checklist.

#![cfg(feature = "cuda")]

use std::collections::{HashMap, VecDeque};

use ferrite_dispatch::arena::{NodeId, SeqId, SeqTag, TypedArena};
use ferrite_dispatch::batch::{Admission, CacheStats, TickPlan};
use ferrite_dispatch::radix::RadixCache;
use ferrite_dispatch::state::StateId;
use ferrite_exec::mtp_batch;
use ferrite_exec::tp::TpCluster;
use ferrite_http::engine::ServeEngine;
use ferrite_kernel::CudaBackend;
use ferrite_types::{FerriteError, Result};

/// DSA cache allocation bound (ferrite-kernel's max_tokens per family) —
/// prompt + generation must stay under it.
const MAX_CTX: usize = 8100;

/// Wave-4 P0-B: the chunked-prefill budget (prompt tokens per
/// `prefill_chunk` call). `0` (or unset) keeps the historical whole-segment
/// call, so the two granularities can be A/B'd on the same binary. Read once
/// and cached — the house rule for every hot-path gate.
fn prefill_token_budget() -> usize {
    static F: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *F.get_or_init(|| {
        std::env::var("FERRITE_PREFILL_BUDGET")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(0)
    })
}

// ---------------------------------------------------------------------------
// Prefix cache (GLM_KV_CACHE) — parked sequences + a token-granular radix index
// ---------------------------------------------------------------------------
//
// ## What can be reused, and what cannot (the feasibility note this was built on)
//
// GLM's per-seq state is not shareable between sequences: `prefill_chunk`
// (ferrite-exec/src/tp.rs) seeds a seq's runtime from its OWN chunk
// (`Engine::ensure_seq` early-returns when the seq exists — there is no
// "seed from another seq" path), and everything a seq accumulates is keyed
// by the seq id — per-(seq,family) DSA KV (device path:
// `cuda.rs` `dsa_caches`), per-(seq,layer) GDN/conv recurrent states
// (`gdn_states`/`conv_states`). `TpCluster` exposes exactly ONE per-seq
// lifecycle operation, `free_seq`, and no copy/export/import. So "lend seq
// A's KV to seq B" is NOT available as a primitive: manufacturing it means a
// fresh device-to-device copy over 6 buffers per DSA family plus every
// GDN/conv state and their pinned position counters — a kernel-level
// feature, not engine wiring. (The host fallback path, with
// `FERRITE_*_DEV` unset, keeps the same state in host Vecs — cloneable in
// principle, but production runs the device chain, see AGENTS.md
// "Required for full-speed decode", and `layer_forward_tp` routes to
// `layer_forward_dev` with no n==1 guard, so prefill is device-side too.)
//
// What IS available — and what this cache is built on — is **parking**: a
// finished request's seq keeps its state if the engine simply skips
// `free_seq`, and that state is a valid *prefix state* for any later prompt
// that starts with the tokens the parked seq had processed. Adopting a
// parked seq costs nothing (the KV never moves) and is exact by
// construction: the parked state IS the state for that token prefix.
//
// ## The two adoption rules
//
// Let T = the parked seq's token history (the position its state sits at),
// P = the new prompt, `prompt_len` = the length of the prompt the parked run
// was actually given.
//
// 1. **Extension — radix longest-prefix.** `T` is a prefix of `P`
//    (`P[..T.len()] == T`): the parked state is exactly the state for
//    `P[..T.len()]`, so admission prefills only `P[T.len()..]`. Sound for
//    any decoding mode. This is the growing-prompt / multi-turn case, and
//    the case a radix tree answers by construction ("which cached history is
//    a prefix of my query?").
// 2. **Repeat — exact prompt match (DSV41 `KvCacheEntry` parity).** `P` is
//    exactly the prompt the parked run was given (`P.len() == prompt_len`
//    and the parked prefix equals `P` verbatim) and the parked continuation
//    is no longer than this request's `max_new`: the parked seq is resumed
//    and its already-generated tail REPLAYS as this request's output. Exact
//    because the engine is greedy (argmax; MTP accept is argmax+verify) —
//    the parked tail IS the continuation a fresh run of `P` produces, token
//    for token. Same assumption DSV41's P0 cache makes when it replays
//    `first_token` on a hit.
//
// Both rules CONSUME the entry (the parked seq becomes the adopting
// request's seq, and its state advances from there). A retired request
// re-parks its own, longer history, so a steady stream of one prompt keeps
// hitting: the second request adopts, the third adopts what the second
// parked, and so on.
//
// ## Index shape
//
// The tree is built at `page_size = 1`: GLM's per-seq KV is one flat buffer
// per (seq, family) with no page granularity, so the exact reuse boundary is
// a token, not a page multiple (the page-aligned block model belongs to the
// paged scheduler, not here). One node per token keeps `match_prefix`'s
// answer exact — the deepest matched node IS the longest parked history that
// is a prefix of the query.
//
// Two maps index the entries: `by_node` (radix node → entry, rule 1) and
// `by_prompt` (FNV-1a of the parked prompt → entries, rule 2 — the hash is
// only a bucket, the verbatim token comparison decides, so a collision is a
// MISS, never a wrong resume). Eviction is LRU over entries; radix nodes are
// shared along prefixes and the tree's `detach` requires a childless leaf,
// so a full cache rebuilds the (small: ≤ cap histories) trie instead.

/// Parked-seq cap default (`GLM_KV_CACHE_SLOTS`): each parked seq pins its
/// whole per-seq device state (~GBs), so the default is deliberately small.
const KV_CACHE_SLOTS_DEFAULT: usize = 2;
/// Do not park histories shorter than this (`GLM_KV_CACHE_MIN`, default 64):
/// a slot costs ~GBs of pinned state, so trivially short prompts are not
/// worth one.
const KV_CACHE_MIN_DEFAULT: usize = 64;

/// `GLM_KV_CACHE` — the prefix cache gate (default OFF, `0` = off; the
/// `DSV41_KV_CACHE` convention). Read once and cached.
fn kv_cache_enabled() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| {
        std::env::var("GLM_KV_CACHE")
            .map(|v| v != "0")
            .unwrap_or(false)
    })
}

fn kv_cache_slots() -> usize {
    static F: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *F.get_or_init(|| {
        std::env::var("GLM_KV_CACHE_SLOTS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(KV_CACHE_SLOTS_DEFAULT)
            .max(1)
    })
}

fn kv_cache_min_tokens() -> usize {
    static F: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *F.get_or_init(|| {
        std::env::var("GLM_KV_CACHE_MIN")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(KV_CACHE_MIN_DEFAULT)
    })
}

/// One parked sequence: the token history its state sits at, plus the prompt
/// length of the run that produced it (the exact-match key of rule 2).
struct PrefixEntry {
    /// Full history at park time (`rt.tokens` — prompt + generated).
    tokens: Vec<u32>,
    /// Length of the prompt that run received (rule 2's key).
    prompt_len: usize,
    /// The parked cluster seq (still allocated — parking skips `free_seq`).
    cluster_seq: u64,
    /// LRU clock.
    last_used: u64,
}

/// A successful adoption: which parked seq to run on, how many tokens its
/// state already covers, and whether the parked tail replays as output.
struct PrefixAdopt {
    cluster_seq: u64,
    matched: usize,
    /// true → the whole prompt is covered by the parked state; the parked
    /// tail is this request's (already-generated) output, so nothing is
    /// prefilled.
    replay: bool,
}

struct PrefixCache {
    radix: RadixCache,
    /// Entry slots (`None` = free). LRU is a scan over the live slots — the
    /// cap is single digits, a heap would be ceremony here.
    entries: Vec<Option<PrefixEntry>>,
    by_node: HashMap<NodeId, usize>,
    by_prompt: HashMap<u64, Vec<usize>>,
    cap: usize,
    clock: u64,
    hits: u64,
    misses: u64,
    /// Prompt tokens served from the cache (sum of `matched` over hits).
    hit_tokens: u64,
    evictions: u64,
}

impl PrefixCache {
    fn new(cap: usize) -> Result<Self> {
        let cap = cap.max(1);
        Ok(PrefixCache {
            // Token-granular trie (see the section note): capacity is an
            // arena hint, the arena grows as needed.
            radix: RadixCache::new(cap.saturating_mul(4096).max(4096), 1)?,
            entries: Vec::new(),
            by_node: HashMap::new(),
            by_prompt: HashMap::new(),
            cap,
            clock: 0,
            hits: 0,
            misses: 0,
            hit_tokens: 0,
            evictions: 0,
        })
    }

    /// FNV-1a over the token ids (little-endian) — `DSV41`'s `KvCache::hash`
    /// parity. Only ever compared against the stored tokens, so stability is
    /// what matters, not cryptography.
    fn hash(tokens: &[u32]) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for &t in tokens {
            for b in t.to_le_bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        h
    }

    /// Index a token history into the trie, one node per token; returns the
    /// node of its deepest token (the entry's key).
    fn index(radix: &mut RadixCache, tokens: &[u32]) -> Option<NodeId> {
        let mut node = radix.root();
        for &t in tokens {
            match radix.insert_branch(node, vec![t], StateId::opaque(0)) {
                Ok((n, _)) => node = n,
                Err(e) => {
                    eprintln!("[kv-cache] radix insert failed: {e}");
                    return None;
                }
            }
        }
        Some(node)
    }

    fn live_entries(&self) -> usize {
        self.entries.iter().filter(|e| e.is_some()).count()
    }

    fn cached_tokens(&self) -> usize {
        self.entries.iter().flatten().map(|e| e.tokens.len()).sum()
    }

    /// A free slot, else the LRU victim's (already emptied). Returns the
    /// slot index plus the cluster seqs the eviction released.
    fn slot_for_new(&mut self) -> (usize, Vec<u64>) {
        let mut release = Vec::new();
        if let Some(s) = self.entries.iter().position(|e| e.is_none()) {
            return (s, release);
        }
        if self.entries.len() < self.cap {
            self.entries.push(None);
            return (self.entries.len() - 1, release);
        }
        let mut victim = None;
        let mut stamp = u64::MAX;
        for (i, e) in self.entries.iter().enumerate() {
            if let Some(e) = e {
                if e.last_used < stamp {
                    stamp = e.last_used;
                    victim = Some(i);
                }
            }
        }
        let Some(victim) = victim else {
            // cap == 0 after clamping can't happen; defensive only.
            self.entries.push(None);
            return (self.entries.len() - 1, release);
        };
        if let Some(e) = self.entries[victim].take() {
            release.push(e.cluster_seq);
            self.evictions += 1;
        }
        (victim, release)
    }

    /// Rebuild the trie + both indexes from the live slots (used after an
    /// eviction and to compact away the path nodes adoption leaves behind).
    fn rebuild(&mut self) {
        let cap_nodes = self.radix.live_nodes().max(64);
        let Ok(mut fresh) = RadixCache::new(cap_nodes, 1) else {
            return;
        };
        let mut by_node = HashMap::new();
        let mut by_prompt: HashMap<u64, Vec<usize>> = HashMap::new();
        for slot in 0..self.entries.len() {
            let Some(e) = self.entries[slot].as_ref() else {
                continue;
            };
            let Some(node) = Self::index(&mut fresh, &e.tokens) else {
                continue;
            };
            by_node.insert(node, slot);
            let key = Self::hash(&e.tokens[..e.prompt_len.min(e.tokens.len())]);
            by_prompt.entry(key).or_default().push(slot);
        }
        self.radix = fresh;
        self.by_node = by_node;
        self.by_prompt = by_prompt;
    }

    /// Index a parked seq. Returns the cluster seqs to release (`free_seq`).
    fn park(&mut self, tokens: &[u32], prompt_len: usize, cluster_seq: u64) -> Vec<u64> {
        if tokens.is_empty() {
            return Vec::new();
        }
        let (slot, mut release) = self.slot_for_new();
        let Some(node) = Self::index(&mut self.radix, tokens) else {
            return release;
        };
        // Identical history already parked: the newer seq takes the node (the
        // older state is redundant) — release it.
        let dup = self.by_node.get(&node).copied().filter(|&old| old != slot);
        if let Some(old) = dup {
            if let Some(e) = self.entries[old].take() {
                release.push(e.cluster_seq);
            }
        }
        self.clock += 1;
        self.entries[slot] = Some(PrefixEntry {
            tokens: tokens.to_vec(),
            prompt_len,
            cluster_seq,
            last_used: self.clock,
        });
        self.by_node.insert(node, slot);
        if release.is_empty() {
            let key = Self::hash(&tokens[..prompt_len.min(tokens.len())]);
            self.by_prompt.entry(key).or_default().push(slot);
        } else {
            // An eviction (or a duplicate history) leaves stale index entries
            // and unreferenced trie nodes behind — rebuild both from the live
            // slots. The cache is at cap when this happens, so the rebuild
            // (O(cached tokens)) is paid once per park at cap.
            self.rebuild();
        }
        self.compact_if_bloated();
        release
    }

    /// Adoption leaves its path nodes behind (they are re-shared when the
    /// seq re-parks, but a long-lived serve must not accumulate them): rebuild
    /// once the trie is much larger than the entries it indexes.
    fn compact_if_bloated(&mut self) {
        if self.radix.live_nodes() > self.cached_tokens().saturating_mul(2) + 4096 {
            self.rebuild();
        }
    }

    /// Longest-prefix (rule 1, radix) or exact-prompt (rule 2, hash)
    /// adoption. A hit CONSUMES the entry (its seq now belongs to the
    /// adopting request).
    fn lookup(&mut self, prompt: &[u32], max_new: usize) -> Option<PrefixAdopt> {
        if prompt.is_empty() {
            return None;
        }
        let mut found: Option<(usize, PrefixAdopt)> = None;
        // Rule 1 — the deepest parked history that is a prefix of the prompt.
        if let Some(m) = self.radix.match_prefix(prompt) {
            if let Some(&slot) = self.by_node.get(&m.node) {
                if let Some(e) = self.entries[slot].as_ref() {
                    if e.tokens.len() <= prompt.len() && e.tokens[..] == prompt[..e.tokens.len()] {
                        found = Some((
                            slot,
                            PrefixAdopt {
                                cluster_seq: e.cluster_seq,
                                matched: e.tokens.len(),
                                replay: false,
                            },
                        ));
                    }
                }
            }
        }
        // Rule 2 — this prompt is exactly the one the parked run received.
        if found.is_none() {
            let key = Self::hash(prompt);
            let slots = self.by_prompt.get(&key).cloned().unwrap_or_default();
            for slot in slots {
                let Some(e) = self.entries[slot].as_ref() else {
                    continue;
                };
                if e.prompt_len == prompt.len()
                    && e.tokens.len() > prompt.len()
                    && e.tokens[..prompt.len()] == prompt[..]
                    && e.tokens.len() - prompt.len() <= max_new
                {
                    found = Some((
                        slot,
                        PrefixAdopt {
                            cluster_seq: e.cluster_seq,
                            matched: prompt.len(),
                            replay: true,
                        },
                    ));
                    break;
                }
            }
        }
        let Some((slot, adopt)) = found else {
            self.misses += 1;
            return None;
        };
        self.entries[slot] = None;
        self.by_node.retain(|_, s| *s != slot);
        self.by_prompt.retain(|_, v| {
            v.retain(|s| *s != slot);
            !v.is_empty()
        });
        self.hits += 1;
        self.hit_tokens += adopt.matched as u64;
        Some(adopt)
    }

    fn census(&self) -> (usize, usize, usize, usize) {
        (
            self.radix.live_nodes(),
            self.radix.total_blocks(),
            self.live_entries(),
            self.cached_tokens(),
        )
    }
}

#[cfg(test)]
mod prefix_cache_tests {
    //! Parked-seq cache semantics (pure host bookkeeping — no GPU needed).
    //! The adoption rules are the contract: rule 1 may only report the tokens
    //! the parked STATE actually covers, rule 2 only fires for the parked
    //! run's own prompt with a continuation that fits the new request.
    use super::*;

    fn prompt(n: usize) -> Vec<u32> {
        (100..100 + n as u32).collect()
    }

    /// A parked run of `n` prompt tokens plus `gen` generated ones.
    fn parked_history(n: usize, gen: usize) -> Vec<u32> {
        let mut h = prompt(n);
        h.extend(std::iter::repeat(7).take(gen));
        h
    }

    #[test]
    fn extension_adopts_only_the_covered_prefix() {
        let mut c = PrefixCache::new(2).unwrap();
        let hist = parked_history(8, 3);
        assert!(c.park(&hist, 8, 42).is_empty());

        // The new prompt starts with the parked state's whole history.
        let mut longer = hist.clone();
        longer.extend_from_slice(&[55, 56]);
        let a = c.lookup(&longer, 16).expect("extension hit");
        assert_eq!(a.cluster_seq, 42);
        assert_eq!(a.matched, hist.len());
        assert!(!a.replay, "an extension resumes at a real KV boundary");

        // Consumed: the seq now belongs to the adopting request.
        assert!(c.lookup(&longer, 16).is_none());
        assert_eq!((c.hits, c.misses), (1, 1));
    }

    #[test]
    fn repeat_adopts_the_same_prompt_and_replays() {
        let mut c = PrefixCache::new(2).unwrap();
        let p = prompt(8);
        assert!(c.park(&parked_history(8, 3), 8, 42).is_empty());

        let a = c.lookup(&p, 16).expect("repeat hit");
        assert_eq!(a.cluster_seq, 42);
        assert_eq!(a.matched, p.len(), "the whole prompt came from the cache");
        assert!(a.replay, "the parked tail is this request's output");
    }

    #[test]
    fn repeat_is_refused_when_the_tail_exceeds_max_new() {
        let mut c = PrefixCache::new(2).unwrap();
        let p = prompt(8);
        assert!(c.park(&parked_history(8, 9), 8, 42).is_empty());
        assert!(c.lookup(&p, 4).is_none(), "tail 9 > max_new 4");
        assert!(c.lookup(&p, 9).is_some(), "tail 9 == max_new 9");
    }

    #[test]
    fn divergent_and_shorter_prompts_miss() {
        let mut c = PrefixCache::new(2).unwrap();
        let hist = parked_history(8, 3);
        assert!(c.park(&hist, 8, 42).is_empty());

        // Diverges inside the parked history: no entry covers that boundary.
        let mut diverged = prompt(8);
        diverged[7] = 999;
        assert!(c.lookup(&diverged, 16).is_none());

        // A prompt that does not extend the parked history at all.
        assert!(c.lookup(&[77, 78], 16).is_none());
        assert_eq!((c.hits, c.misses), (0, 2));
    }

    #[test]
    fn cap_evicts_the_lru_and_reports_it_for_release() {
        let mut c = PrefixCache::new(1).unwrap();
        assert!(c.park(&parked_history(4, 1), 4, 1).is_empty());
        // A different history at cap: the LRU victim's seq is returned so the
        // caller can free it.
        let release = c.park(&parked_history(5, 1), 5, 2);
        assert_eq!(release, vec![1]);
        assert_eq!(c.evictions, 1);
        assert_eq!(c.live_entries(), 1);
        // The surviving entry is the newer one.
        let mut p = prompt(5);
        p.push(7);
        assert_eq!(c.lookup(&p, 8).map(|a| a.cluster_seq), Some(2));
    }

    #[test]
    fn identical_histories_release_the_older_seq() {
        let mut c = PrefixCache::new(4).unwrap();
        assert!(c.park(&parked_history(4, 2), 4, 1).is_empty());
        // Same history parked again (two runs, identical tokens): the newer
        // seq owns the node, the older one is released.
        assert_eq!(c.park(&parked_history(4, 2), 4, 2), vec![1]);
        // The surviving entry is the NEWER seq, at that exact history.
        let q = parked_history(4, 2);
        assert_eq!(c.lookup(&q, 8).map(|a| a.cluster_seq), Some(2));
    }

    #[test]
    fn a_prompt_that_merely_shortens_a_parked_history_misses() {
        // The parked state sits at T.len(); a query that is a PROPER prefix of
        // T (and is not the parked run's own prompt) cannot be resumed — there
        // is no state at that boundary. Rule 1 needs `T` to be a prefix of the
        // QUERY, rule 2 the parked prompt verbatim: neither holds here.
        let mut c = PrefixCache::new(2).unwrap();
        assert!(c.park(&parked_history(4, 2), 4, 1).is_empty());
        let mut p = prompt(4);
        p.push(7);
        assert!(c.lookup(&p, 8).is_none());
        assert_eq!(c.misses, 1);
    }

    #[test]
    fn fork_shares_the_trie_but_keys_distinct_boundaries() {
        let mut c = PrefixCache::new(4).unwrap();
        // Two parked runs whose prompts share a 4-token head but diverge.
        assert!(c.park(&parked_history(4, 2), 4, 1).is_empty());
        let mut other = prompt(4);
        other.push(31);
        assert!(c.park(&other, 5, 2).is_empty());
        // The shared head is a path node, not an entry: extending the parked
        // history must hit ITS entry, not the fork point.
        let mut p = prompt(4);
        p.extend_from_slice(&[7, 7, 8]);
        assert_eq!(c.lookup(&p, 8).map(|a| a.cluster_seq), Some(1));
        // The forked branch keys its own boundary.
        let mut q = prompt(4);
        q.push(31);
        q.push(9);
        assert_eq!(c.lookup(&q, 8).map(|a| a.cluster_seq), Some(2));
    }
}

/// One live/queued request on the engine side.
struct GpuSeq {
    /// The cluster-side sequence id (u64 counter, never reused).
    cluster_seq: u64,
    /// Prompt token ids (held until admission runs the prefill).
    prompt: Vec<u32>,
    prompt_len: usize,
    max_new: usize,
    /// rt.tokens watermark — the incremental read per tick.
    prev_len: usize,
    /// Retirement flag (status() → "retired"; the driver reads the final
    /// output once, then deregisters).
    retired: bool,
    /// GPU state released (idempotence guard for cancel→deregister).
    freed: bool,
    /// Final output snapshot (taken at retirement/cancel — the cluster
    /// runtime is freed, later output() reads serve from here).
    final_out: Option<Vec<u32>>,
}

pub struct GpuEngine {
    cluster: TpCluster<CudaBackend>,
    arena: TypedArena<SeqTag, GpuSeq>,
    /// Admitted (prefill done) — the round-robin decode set.
    live: Vec<SeqId>,
    /// Awaiting admission (prefill) — FIFO.
    queue: VecDeque<SeqId>,
    next_cluster: u64,
    /// Full stop-token set (the tokenizer's specials; retirement check on
    /// the incremental stream tail).
    stops: Vec<u32>,
    max_seqs: usize,
    ticks: u64,
    /// The current batched-decode graph's composition name
    /// ("megab_{s1}_{s2}..."). A membership change (admission / retirement /
    /// cancel) destroys the old graph — its captured kernel args embed the
    /// member seqs' per-seq state pointers, which free_seq releases. The
    /// next tick captures fresh for the new composition (~1-2s, amortized
    /// over 1000-token streams).
    batch_graph: Option<String>,
    /// The parked-seq prefix cache (GLM_KV_CACHE; `None` = gate off).
    prefix_cache: Option<PrefixCache>,
}

impl GpuEngine {
    pub fn new(cluster: TpCluster<CudaBackend>, stops: Vec<u32>, mut max_seqs: usize) -> Self {
        // Wave 5: the MTP step is the single-seq round-robin by default — the
        // verify graph is one seq's block (`mega_v{seq}`) and the draft chain /
        // accept / commit are per seq — so FERRITE_MTP forces max_seqs=1
        // unless the batched sub-branch is explicitly opted into. MtpState is
        // per seq (`MtpStateB`) and the B-seq step exists
        // (`TpCluster::mtp_step_batched`, FERRITE_MTP_BATCHED=1); while
        // `mtp_batch::MTP_BATCH_READY` is false that step runs the per-seq
        // fallback, so opting in with max_seqs > 1 today is a
        // WIRING-VALIDATION mode (correct — the legacy round-robin — but it
        // pays the per-seq graph captures), NOT a performance path.
        let batched_mtp = mtp_batch::batched_enabled();
        if std::env::var_os("FERRITE_MTP").is_some() && max_seqs > 1 && !batched_mtp {
            eprintln!(
                "[serve] FERRITE_MTP=1 with max_seqs={max_seqs}: forcing 1 (the MTP step is still the single-seq round-robin; FERRITE_MTP_BATCHED=1 opts into the batched B-seq step, ready={})",
                mtp_batch::MTP_BATCH_READY
            );
            max_seqs = 1;
        }
        if batched_mtp && std::env::var_os("FERRITE_MTP").is_some() {
            eprintln!(
                "[serve] FERRITE_MTP_BATCHED=1: batched MTP sub-branch ON (max_seqs={max_seqs}, kernels ready={})",
                mtp_batch::MTP_BATCH_READY
            );
        }
        eprintln!(
            "[serve] GpuEngine: max_seqs={max_seqs} stops={stops:?} (per-seq state ~GBs; free at retire)"
        );
        // Prefix cache (GLM_KV_CACHE; default OFF). Parking skips free_seq, so
        // each slot pins a seq's ~GBs of device state — the cap is the memory
        // knob, and the gate keeps the default path byte-identical.
        let prefix_cache = if kv_cache_enabled() {
            let slots = kv_cache_slots();
            match PrefixCache::new(slots) {
                Ok(c) => {
                    eprintln!(
                        "[kv-cache] GLM_KV_CACHE on: slots={slots} min_tokens={} (parked seqs hold ~GBs each; a retired request parks instead of freeing)",
                        kv_cache_min_tokens()
                    );
                    Some(c)
                }
                Err(e) => {
                    eprintln!("[kv-cache] disabled (init failed: {e})");
                    None
                }
            }
        } else {
            None
        };
        GpuEngine {
            cluster,
            arena: TypedArena::with_capacity(64),
            live: Vec::new(),
            queue: VecDeque::new(),
            next_cluster: 1,
            stops,
            max_seqs,
            ticks: 0,
            batch_graph: None,
            prefix_cache,
        }
    }

    /// The incremental output of a live seq (post-prompt tokens).
    fn incremental(&self, cluster_seq: u64, prompt_len: usize) -> Vec<u32> {
        self.cluster
            .shards
            .first()
            .and_then(|s| s.seq_runtime(cluster_seq))
            .map(|rt| rt.tokens[prompt_len.min(rt.tokens.len())..].to_vec())
            .unwrap_or_default()
    }

    /// Release the seq's GPU state + drop the arena entry (idempotent).
    ///
    /// With `GLM_KV_CACHE` on, a *finished* seq is PARKED instead of freed:
    /// skipping `free_seq` keeps its per-seq state alive as a prefix state
    /// for later prompts (see the prefix-cache section). A parked seq is
    /// freed when an LRU eviction reclaims its slot.
    fn free(&mut self, seq: SeqId) {
        // KEEP the per-size batched graphs across retires: the kernel args
        // reference the per-size POINTER TABLES (content-refreshed each
        // replay), not embedded seq pointers — a retired seq's slots are
        // simply overwritten on the next refresh. Destroying here forced a
        // 1-2s re-capture every membership change.
        // (free_seq still releases the seq's own GDN states / DSA caches.)
        // Read the params immutably, snapshot the output (needs &self),
        // then free the GPU state + update the arena (&mut self) —
        // sequenced to avoid the borrow conflict.
        let params = self
            .arena
            .get(seq)
            .filter(|g| !g.freed)
            .map(|g| (g.cluster_seq, g.prompt_len, g.retired));
        if let Some((cluster_seq, prompt_len, retired)) = params {
            let snapshot = self.incremental(cluster_seq, prompt_len);
            let parked = self.try_park(cluster_seq, prompt_len, retired);
            if let Some(g) = self.arena.get_mut(seq) {
                g.freed = true;
                if g.final_out.is_none() {
                    g.final_out = Some(snapshot);
                }
            }
            if !parked {
                self.cluster.free_seq(cluster_seq);
            }
            // DIAGNOSTIC (FERRITE_DESTROY_BG=1): the doc comment on
            // destroy_batch_graph says a retire MUST destroy the batched graphs
            // because their recorded kernel args reference the freed per-seq
            // state — but nothing calls it. Force-destroy them here to test
            // whether the first replay after a retire is what faults.
            if std::env::var_os("FERRITE_DESTROY_BG").is_some() {
                for sz in [1usize, 2, 4, 8, 16, 32, 64] {
                    self.cluster.destroy_batch_graph(&format!("megab_b{sz}"));
                }
            }
        }
        self.arena.remove(seq);
    }

    /// Park a finished seq in the prefix cache (no state copy — the seq keeps
    /// its GPU state; the engine only stops tracking it as a request).
    /// Returns true when the seq was parked (its GPUs state must then NOT be
    /// freed). LRU victims are released here.
    fn try_park(&mut self, cluster_seq: u64, prompt_len: usize, retired: bool) -> bool {
        if self.prefix_cache.is_none() || !retired {
            return false;
        }
        // The parked HISTORY is the mirror the state position is derived from
        // (host path: `t0 = c.k_nope.len()/(h*dk)`; device path: `t_count`).
        let tokens = match self
            .cluster
            .shards
            .first()
            .and_then(|s| s.seq_runtime(cluster_seq))
        {
            Some(rt) if rt.tokens.len() >= kv_cache_min_tokens() => rt.tokens.clone(),
            _ => return false,
        };
        let Some(cache) = self.prefix_cache.as_mut() else {
            return false;
        };
        let release = cache.park(&tokens, prompt_len, cluster_seq);
        eprintln!(
            "[kv-cache] parked cluster={cluster_seq} history={} prompt={prompt_len} slots={}/{} hits={} misses={}",
            tokens.len(),
            cache.live_entries(),
            cache.cap,
            cache.hits,
            cache.misses
        );
        for victim in release {
            if victim != cluster_seq {
                self.cluster.free_seq(victim);
            }
        }
        true
    }

    /// Adopt a parked seq whose state matches a prefix of this prompt.
    fn adopt_prefix(&mut self, prompt: &[u32], max_new: usize) -> Option<PrefixAdopt> {
        let cache = self.prefix_cache.as_mut()?;
        let adopt = cache.lookup(prompt, max_new);
        match &adopt {
            Some(a) => eprintln!(
                "[kv-cache] HIT ({}): cluster={} matched={} of {} prompt tokens (replay={}) hits={} misses={}",
                if a.replay { "repeat" } else { "prefix" },
                a.cluster_seq,
                a.matched,
                prompt.len(),
                a.replay,
                cache.hits,
                cache.misses
            ),
            None => eprintln!(
                "[kv-cache] MISS: no parked prefix of {} prompt tokens (hits={} misses={} entries={})",
                prompt.len(),
                cache.hits,
                cache.misses,
                cache.live_entries()
            ),
        }
        adopt
    }
}

impl ServeEngine for GpuEngine {
    fn submit(
        &mut self,
        prompt_ids: Vec<u32>,
        max_new_tokens: usize,
        _eos: u32,
    ) -> Result<SeqId> {
        if prompt_ids.len() + max_new_tokens > MAX_CTX {
            return Err(FerriteError::InvalidArg(format!(
                "context too long: prompt {} + max_new {} > {MAX_CTX} (DSA cache bound)",
                prompt_ids.len(),
                max_new_tokens
            )));
        }
        if prompt_ids.is_empty() {
            return Err(FerriteError::InvalidArg("empty prompt".into()));
        }
        let cluster_seq = self.next_cluster;
        self.next_cluster += 1;
        let prompt_len = prompt_ids.len();
        let seq = self.arena.insert(GpuSeq {
            cluster_seq,
            prompt: prompt_ids,
            prompt_len,
            max_new: max_new_tokens,
            prev_len: 0,
            retired: false,
            freed: false,
            final_out: None,
        });
        self.queue.push_back(seq);
        Ok(seq)
    }

    fn tick(&mut self, plan: &mut TickPlan) -> Result<()> {
        self.ticks += 1;
        // TICK TIMING (FERRITE_TIMING): the server-side total step time —
        // compare against the [megab] replay median to see whether the
        // ~0.9ms host gap is server-side (tick > replay → pipelining helps)
        // or client-side Python SSE parsing (tick ≈ replay → pipelining
        // is pointless).
        let tick_start = std::time::Instant::now();
        // FERRITE_NCU serve window (nsys --capture-range=cudaProfilerApi):
        // open the capture only once the batch SATURATES (live == max_seqs) —
        // skips the 80s weight load AND the admission ramp / per-size graph
        // captures; closed by run_serve's profiler_stop before exit (nsys
        // waits for cudaProfilerStop forever without it — the documented
        // "serve never exits" trap).
        if std::env::var_os("FERRITE_NCU").is_some() && self.live.len() >= self.max_seqs {
            static NCU_WIN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            if !NCU_WIN.swap(true, std::sync::atomic::Ordering::AcqRel) {
                eprintln!(
                    "[ncu-win] batch saturated (live={}): opening the profiler window",
                    self.live.len()
                );
                #[cfg(feature = "cuda")]
                ferrite_kernel::cuda::profiler_start();
            }
        }
        plan.admissions.clear();
        // 1. Admission — ONE prefill per tick (prefill is a blocking
        //    host-chain forward, ~0.4-2s for chat prompts; one-at-a-time
        //    bounds the stall the live decode set sees).
        if !self.queue.is_empty() && self.live.len() < self.max_seqs {
            let seq = self.queue.pop_front().expect("checked non-empty");
            // None → cancelled while queued (free already ran): drop silently.
            if let Some(g) = self.arena.get(seq) {
                let (mut cluster_seq, prompt, max_new) = (g.cluster_seq, g.prompt.clone(), g.max_new);
                let t0 = std::time::Instant::now();
                // Prefix cache (GLM_KV_CACHE): adopt a parked seq when its
                // state already covers a prefix of this prompt. The parked
                // state IS the state for those tokens, so only the remainder
                // still needs prefill — and in the exact-repeat case there is
                // no remainder at all (the parked tail replays as output).
                let mut prefix_hit = 0usize;
                let mut suffix: Vec<u32> = prompt.clone();
                if let Some(adopt) = self.adopt_prefix(&prompt, max_new) {
                    cluster_seq = adopt.cluster_seq;
                    prefix_hit = adopt.matched;
                    if adopt.replay {
                        suffix.clear();
                    } else {
                        suffix = prompt[adopt.matched..].to_vec();
                    }
                    if let Some(g) = self.arena.get_mut(seq) {
                        g.cluster_seq = cluster_seq;
                    }
                }
                // P0-B (Wave 4): feed the prompt in chunks instead of one whole
                // segment. The engine is n-variable already — `prefill_chunk`
                // accumulates the KV across successive calls on the same seq, so
                // the ONLY thing that changes is the caller's granularity. The
                // chunked path is the prerequisite for every 1M-context item
                // (bounded per-call workspace, indexer chunking, CP). `0` keeps
                // the historical whole-segment call so the two can be A/B'd.
                let budget = prefill_token_budget();
                if !suffix.is_empty() {
                    if budget == 0 || suffix.len() <= budget {
                        self.cluster.prefill_chunk(cluster_seq, &suffix)?;
                    } else {
                        for chunk in suffix.chunks(budget) {
                            self.cluster.prefill_chunk(cluster_seq, chunk)?;
                        }
                    }
                    // The host token mirror is the position record the
                    // incremental read indexes into, and `prefill_chunk` does
                    // not append to it — an adopted seq's mirror still holds
                    // the parked history, so re-sync it to the sequence its
                    // state now covers (the whole prompt).
                    if prefix_hit > 0 {
                        self.cluster.set_seq_tokens(cluster_seq, &prompt);
                    }
                }
                if let Some(g) = self.arena.get_mut(seq) {
                    g.prev_len = g.prompt_len; // rt.tokens = prompt after prefill
                }
                self.live.push(seq);
                eprintln!(
                    "[serve] admitted seq {seq:?} cluster={cluster_seq} prompt={} prefix_hit={prefix_hit} prefill={:.2}s budget={} (live={} queued={})",
                    prompt.len(),
                    t0.elapsed().as_secs_f32(),
                    if budget == 0 { 0 } else { budget },
                    self.live.len(),
                    self.queue.len()
                );
                plan.admissions.push(Admission { seq, row: 0, prefix_hit });
            }
        }
        // 2. Decode — the TRUE BATCHED path (non-MTP): ONE graph step for
        //    ALL live seqs. The projections run at n=B GEMM (weights stream
        //    once per step for all B rows — the batched-GEMM directive);
        //    the per-seq recurrent state ops (GDN conv/state, DSA caches) run
        //    as B × n=1 in-graph launches with each row's own state
        //    pointers. Composition change (admission/retirement) re-captures
        //    (~1-2s, amortized over 1000-token streams).
        //    MTP: the per-seq round-robin by default, and (Wave 5) the batched
        //    B-seq step behind FERRITE_MTP_BATCHED=1 — `mtp_step_batched`
        //    batches the drafts + the B×n_v verify block of ALL live seqs into
        //    one pass. FERRITE_MTP still forces max_seqs=1 unless that flag is
        //    set, so this branch degenerates to a single live seq by default.
        let mtp_mode = std::env::var_os("FERRITE_MTP").is_some();
        let batched_mtp = mtp_batch::batched_enabled();
        let mut retired: Vec<SeqId> = Vec::new();
        if !self.live.is_empty() {
            if mtp_mode {
                // the live cluster seqs in admission order — the batched MTP
                // composition (and the per-seq loop's iteration order).
                let live_seqs: Vec<u64> = self
                    .live
                    .iter()
                    .filter_map(|seq| self.arena.get(*seq).map(|g| g.cluster_seq))
                    .collect();
                if batched_mtp && live_seqs.len() > 1 {
                    // WAVE 5 STEP B: ONE B-seq MTP step — drafts, the B×n_v
                    // verify block, accept and commit for all live seqs in one
                    // pass, with each seq's state reached through the pointer
                    // tables (the same shape the batched decode uses). The plan
                    // (`ferrite_exec::mtp_batch`) is built inside; while
                    // MTP_BATCH_READY is false the per-seq fallback runs, so
                    // this is safe-but-unaccelerated until the kernel work
                    // lands. The seq_runtime token push is part of the step.
                    self.cluster.mtp_step_batched(&live_seqs)?;
                } else {
                    // per-seq round-robin (the legacy single-seq path —
                    // decode_step handles mega/MTP). Also the B=1 case: the
                    // batched B-row graph is 1.9x SLOWER at B=1 (measured
                    // [megab] 17.95ms vs [mega] 9.55ms), and the batched MTP
                    // step has nothing to batch — keep the per-seq path there.
                    for i in 0..self.live.len() {
                        let seq = self.live[i];
                        let cluster_seq = match self.arena.get(seq) {
                            Some(g) => g.cluster_seq,
                            None => continue,
                        };
                        self.cluster.decode_step(cluster_seq)?;
                    }
                }
                // Retirement checks — shared by both sub-branches (the decode
                // call above is the only difference).
                for i in 0..self.live.len() {
                    let seq = self.live[i];
                    let (cluster_seq, prompt_len, max_new, prev_len) = match self.arena.get(seq) {
                        Some(g) => (g.cluster_seq, g.prompt_len, g.max_new, g.prev_len),
                        None => continue,
                    };
                    let rt_len = self
                        .cluster
                        .shards
                        .first()
                        .and_then(|s| s.seq_runtime(cluster_seq))
                        .map(|rt| rt.tokens.len())
                        .unwrap_or(0);
                    let stopped = rt_len > prev_len
                        && self
                            .cluster
                            .shards
                            .first()
                            .and_then(|s| s.seq_runtime(cluster_seq))
                            .map(|rt| rt.tokens[prev_len..].iter().any(|t| self.stops.contains(t)))
                            .unwrap_or(false);
                    if let Some(g) = self.arena.get_mut(seq) {
                        g.prev_len = rt_len;
                    }
                    let generated = rt_len.saturating_sub(prompt_len);
                    if stopped || generated >= max_new {
                        let snapshot = self.incremental(cluster_seq, prompt_len);
                        if let Some(g) = self.arena.get_mut(seq) {
                            g.retired = true;
                            g.final_out = Some(snapshot);
                        }
                        retired.push(seq);
                    }
                }
            } else {
                // BATCHED: one decode_step_batched for the whole live set —
                // the graph composition is the ordered live cluster seqs.
                let live_seqs: Vec<u64> = self
                    .live
                    .iter()
                    .filter_map(|seq| self.arena.get(*seq).map(|g| g.cluster_seq))
                    .collect();
                if live_seqs.len() == 1 && std::env::var_os("FERRITE_FORCE_BATCHED_B1").is_none() {
                    // SINGLE seq: the per-seq mega (GEMV) path. The batched
                    // B-row GEMM graph was 1.9x SLOWER at B=1 (measured
                    // [megab] replay 17.95ms vs [mega] 9.55ms). The
                    // gdn_layer_dev_batched n==1 alignment (fused
                    // gemv_tri/gemv_qkv_conv) closes most of that gap —
                    // FERRITE_FORCE_BATCHED_B1=1 re-tests the batched path.
                    // KEEP the per-size batched graphs (the tables are
                    // content-refreshed, no embedded seq pointers) —
                    // destroying here forces a 1-2s re-capture when
                    // concurrency returns.
                    self.cluster.decode_step(live_seqs[0])?;
                } else {
                    // SGLang-style batch-size keying: tp.rs pads to
                    // 1/2/4/8/16/32 and captures ONE graph per padded size,
                    // so a membership change REUSES the graph (the per-size
                    // pointer tables' content is refreshed inside
                    // decode_step_batched) — no re-capture.
                    let size = [1usize, 2, 4, 8, 16, 32]
                        .iter()
                        .copied()
                        .find(|&s| s >= live_seqs.len())
                        .unwrap_or(live_seqs.len());
                    let batch_name = format!("megab_b{size}");
                    self.batch_graph = Some(batch_name);
                    self.cluster.decode_step_batched(&live_seqs)?;
                }
                // per-seq retirement checks (the incremental reads — same
                // logic as the per-seq loop, minus the decode_step call)
                for i in 0..self.live.len() {
                    let seq = self.live[i];
                    let (cluster_seq, prompt_len, max_new, prev_len) = match self.arena.get(seq) {
                        Some(g) => (g.cluster_seq, g.prompt_len, g.max_new, g.prev_len),
                        None => continue,
                    };
                    let rt_len = self
                        .cluster
                        .shards
                        .first()
                        .and_then(|s| s.seq_runtime(cluster_seq))
                        .map(|rt| rt.tokens.len())
                        .unwrap_or(0);
                    let stopped = rt_len > prev_len
                        && self
                            .cluster
                            .shards
                            .first()
                            .and_then(|s| s.seq_runtime(cluster_seq))
                            .map(|rt| rt.tokens[prev_len..].iter().any(|t| self.stops.contains(t)))
                            .unwrap_or(false);
                    if let Some(g) = self.arena.get_mut(seq) {
                        g.prev_len = rt_len;
                    }
                    let generated = rt_len.saturating_sub(prompt_len);
                    if stopped || generated >= max_new {
                        let snapshot = self.incremental(cluster_seq, prompt_len);
                        if let Some(g) = self.arena.get_mut(seq) {
                            g.retired = true;
                            g.final_out = Some(snapshot);
                        }
                        retired.push(seq);
                    }
                }
            }
        }
        for seq in retired {
            self.live.retain(|s| *s != seq);
        }
        if std::env::var_os("FERRITE_TIMING").is_some() {
            eprintln!("[tick] total: {:.2}ms (live={})", tick_start.elapsed().as_secs_f64() * 1e3, self.live.len());
        }
        Ok(())
    }

    fn output(&self, seq: SeqId) -> Result<Vec<u32>> {
        match self.arena.get(seq) {
            // Retired/cancelled: the frozen snapshot (the cluster runtime
            // may be freed already).
            Some(g) if g.final_out.is_some() => Ok(g.final_out.clone().expect("checked")),
            Some(g) => Ok(self.incremental(g.cluster_seq, g.prompt_len)),
            None => Err(FerriteError::InvalidArg("no such seq".into())),
        }
    }

    fn cancel(&mut self, seq: SeqId) -> Result<bool> {
        let was_live = self.live.contains(&seq) || self.queue.contains(&seq);
        self.live.retain(|s| *s != seq);
        self.queue.retain(|s| *s != seq);
        if self.arena.get(seq).is_some() {
            self.free(seq);
        }
        Ok(was_live)
    }

    fn deregister(&mut self, seq: SeqId) {
        self.live.retain(|s| *s != seq);
        self.queue.retain(|s| *s != seq);
        self.free(seq);
    }

    fn status(&self, seq: SeqId) -> Option<&'static str> {
        match self.arena.get(seq) {
            Some(g) if g.retired => Some("retired"),
            Some(_) => Some("live"),
            None => None,
        }
    }

    fn live_rows(&self) -> usize {
        self.live.len()
    }

    fn queued(&self) -> usize {
        self.queue.len()
    }

    fn cache_stats(&self) -> CacheStats {
        // GLM_KV_CACHE: the parked-seq prefix cache is the only cache this
        // engine has — the radix census maps onto it as: tree_nodes/blocks =
        // the token trie, tier_census.0 (device) = parked seqs (their state
        // lives in the seq's own device buffers), pages_in_use/free = cache
        // slots. Gate off → all zero (the historical report).
        match &self.prefix_cache {
            Some(c) => {
                let (nodes, blocks, parked, cached_tokens) = c.census();
                CacheStats {
                    tree_nodes: nodes,
                    tree_blocks: blocks,
                    evictable_tokens: cached_tokens,
                    protected_tokens: 0,
                    tier_census: (parked, 0, 0),
                    pages_in_use: parked,
                    pages_free: c.cap.saturating_sub(parked),
                    hits: c.hits,
                    misses: c.misses,
                }
            }
            None => CacheStats {
                tree_nodes: 0,
                tree_blocks: 0,
                evictable_tokens: 0,
                protected_tokens: 0,
                tier_census: (0, 0, 0),
                pages_in_use: 0,
                pages_free: 0,
                hits: 0,
                misses: 0,
            },
        }
    }

    fn stop_id(&self) -> u32 {
        self.stops.first().copied().unwrap_or(154_820)
    }

    /// The FULL stop set decides the finish reason: the model's turn-end
    /// token (augu id 154827 — not the primary <|end|> 154820) must label
    /// genuine stops as "stop", not "length".
    fn is_stop(&self, t: u32) -> bool {
        self.stops.contains(&t)
    }
}
