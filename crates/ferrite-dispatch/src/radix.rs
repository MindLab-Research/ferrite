//! Radix tree over token streams — prefix cache with hybrid-state sharing.
//!
//! ## Why a radix tree (and not a hash of prefixes)
//!
//! vLLM-style block hashing answers "is *this exact* prefix cached?" — a
//! binary question. A radix tree answers "what is the *longest* cached
//! prefix of my tokens, and where does it fork?" — which is what MTP
//! verification needs: the `k` accepted tokens fork the shared branch, and
//! rejected siblings are still *potential* branches worth keeping while
//! their pages are alive (SGLang's cache-aware beam insight). Sharing is
//! multi-parent: one system prompt fans out to N conversations without N
//! copies of its pages or GDN snapshots.
//!
//! ## Structure (SGLang `RadixCache` parity)
//!
//! - Nodes hold **page-aligned token blocks** (one block = `page_size`
//!  tokens = one DSA page; `match_prefix` truncates the query to a
//!  page multiple — partial trailing pages never share).
//! - The root is a sentinel whose block is empty. Children are keyed by
//!  first-token (`u32`) so path descent is a hash lookup per block, not
//!  a scan; inside a node, matching is a memcmp over the block.
//! - A node owns a [`StateId`] (registry snapshot: GDN state + page lease,
//!  possibly on the host tier — promotion is the registry's business).
//!  `match_prefix` returns the deepest node whose tokens the query extends.
//!
//! ## Lock/refcount discipline (SGLang `lock_ref` parity)
//!
//! Every live decode row *pins* the nodes along its committed path via
//! [`PinnedPath`]: `lock_ref > 0` nodes are **protected** (never
//! evictable); `lock_ref == 0` leaves are **evictable**. The tree tracks
//! both token sums (`evictable_tokens` / `protected_tokens`) as pins move
//! — the admission policy's pressure signal, no walks required.
//!
//! ## Eviction (SGLang `evict(num_tokens)` parity, tier-aware)
//!
//! [`RadixCache::evict_for_pages`] is **page-budget-driven**: it frees
//! LRU *unpinned leaves* until `pages_needed` are reclaimable, cascading
//! childless+unpinned parents (the cascade is amortized: an evicted leaf
//! pushes its parent onto the victim frontier, SGLang's heap discipline —
//! this tree uses a linear scan over the (small) frontier, the shape
//! per-tick eviction actually has). The caller (scheduler) receives the
//! victim list and drives the physical layer: **device pressure demotes**
//! the victim's snapshot to the host tier (prefix retained — the memory
//! hierarchy of SGLang's `HiMambaRadixCache`), **host pressure drops**
//! it (registry releases the snapshot + pages). The tree never touches
//! memory tiers itself — it only picks victims and updates token sums.
//!
//! ## Cache-aware scheduling (SGLang's admission insight)
//!
//! The longest-match call doubles as the scheduler's admission input:
//! a request whose prompt hits a cached prefix pays only its unique
//! suffix in prefill (GDN state restored from the node, DSA pages
//! leased zero-copy), and the prefix's pages gain a pin for the
//! request's lifetime — admission and cache pressure are one decision.
//!
//! ## Locking
//!
//! Single-threaded by design: the tree lives inside the scheduler's tick
//! critical section, where the whole engine state is already owned. The
//! two-phase tick (plan → replay → ingest) means tree mutation never
//! overlaps device work — no RCU/epoch machinery needed.

use std::collections::HashMap;

use ferrite_types::{FerriteError, Result};

use crate::arena::{NodeId, TypedArena};
use crate::state::StateId;

/// One page-aligned block in the tree.
#[derive(Debug, Clone)]
pub struct RadixNode {
    /// Tokens of this block (exactly `page_size`, except the live tail of
    /// a branch being decoded — tail nodes are never shared, only pinned).
    pub tokens: Vec<u32>,
    /// Registry snapshot backing this node (GDN state + page lease; may
    /// sit on the host tier — promotion on hit is registry business).
    /// `None` only transiently while the scheduler is mid-promotion.
    pub state: Option<StateId>,
    /// Logical token position of this block's first token in the stream
    /// (absolute; root children start at 0).
    pub start: usize,
    /// Parent handle (root's parent = itself — the sentinel convention).
    pub parent: NodeId,
    /// First-token → sibling bucket. Blocks are ATOMIC sharing units
    /// (page-aligned): two blocks with the same first token are *different*
    /// children (a shared chat-template header followed by diverging
    /// content is the common case — SGLang keys children by the whole
    /// block, and so do we: the bucket is a linear scan of full-block
    /// equality, not a single-token discriminator). A bucket holds the
    /// rare same-first-token forks; `match` and `insert` compare whole
    /// blocks, never the key alone.
    children: HashMap<u32, Vec<NodeId>>,
    /// Live decode rows whose committed path includes this node
    /// (SGLang `lock_ref`).
    lock_ref: u32,
    /// LRU clock (set on access; eviction picks the smallest leaf).
    last_used: u64,
    /// Total retained blocks in the subtree rooted here (eviction weight:
    /// freed pages ≈ subtree blocks; maintained on insert/evict).
    subtree_blocks: u32,
}

impl RadixNode {
    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }
    pub fn lock_ref(&self) -> u32 {
        self.lock_ref
    }
}

/// Monotonic LRU clock (ticks).
#[derive(Debug, Clone, Copy, Default)]
struct LruClock {
    tick: u64,
}

impl LruClock {
    fn next(&mut self) -> u64 {
        self.tick += 1;
        self.tick
    }
}

/// Match result of [`RadixCache::match_prefix`] (SGLang `MatchResult`).
#[derive(Debug, Clone)]
pub struct PrefixMatch {
    /// Deepest node whose block path equals the query prefix.
    pub node: NodeId,
    /// Total cached tokens covered (page-aligned by construction).
    pub matched_tokens: usize,
    /// Next token in the query after the match (descent continuation —
    /// `None` if the query is fully covered: prompt entirely cached).
    pub next_query_token: Option<u32>,
}

/// A pinned path — SGLang `inc_lock_ref`/`dec_lock_ref` made RAII-shaped:
/// the scheduler holds the guard while a row runs; dropping it unpins.
/// (No `Drop` impl: unpin order inside a tick matters — the scheduler
/// calls `unpin` explicitly at retirement, mirroring SGLang's explicit
/// `dec_lock_ref`.)
#[derive(Debug, Clone)]
pub struct PinnedPath {
    /// Nodes pinned, root → leaf order.
    pub nodes: Vec<NodeId>,
    /// Tokens each node contributes to the pin (protected sum bookkeeping).
    pub tokens: usize,
}

/// Eviction victim descriptor — what the scheduler must do at the
/// physical layer (demote to host, or drop — SGLang `HiMambaRadixCache`:
/// device tier demotes under device pressure, host LRU drops under host
/// pressure; the tree only picks).
#[derive(Debug, Clone, Copy)]
pub struct EvictVictim {
    pub node: NodeId,
    pub state: StateId,
    /// Pages the snapshot leases (what freeing it reclaims at most).
    pub pages: usize,
    /// Block tokens (LRU age and cost diagnostics).
    pub tokens: usize,
    /// Host-tier residents must drop (not demote); device ones may demote.
    pub on_host: bool,
}

/// The prefix tree + LRU bookkeeping + token sums. Owns nothing physical
/// — all state lives in `StateRegistry` and is referenced by [`StateId`].
pub struct RadixCache {
    nodes: TypedArena<crate::arena::NodeTag, RadixNode>,
    root: NodeId,
    clock: LruClock,
    page_size: usize,
    /// Retained blocks (eviction-pressure metric).
    total_blocks: usize,
    /// Token sums (SGLang `evictable_size_` / `protected_size_` parity).
    evictable_tokens: usize,
    protected_tokens: usize,
}

impl RadixCache {
    pub fn new(capacity_nodes: usize, page_size: usize) -> Result<Self> {
        if page_size == 0 {
            return Err(FerriteError::Scheduler("radix: page_size must be > 0".into()));
        }
        let mut nodes = TypedArena::with_capacity(capacity_nodes);
        let root = nodes.insert(RadixNode {
            tokens: Vec::new(),
            state: None,
            start: 0,
            parent: NodeId::from_raw(0),
            children: HashMap::new(),
            lock_ref: 0,
            last_used: 0,
            subtree_blocks: 0,
        });
        Ok(RadixCache {
            nodes,
            root,
            clock: LruClock::default(),
            page_size,
            total_blocks: 0,
            evictable_tokens: 0,
            protected_tokens: 0,
        })
    }

    pub fn root(&self) -> NodeId {
        self.root
    }

    pub fn page_size(&self) -> usize {
        self.page_size
    }

    pub fn node(&self, id: NodeId) -> Result<&RadixNode> {
        self.nodes
            .get(id)
            .ok_or_else(|| FerriteError::Scheduler("radix: stale node id".into()))
    }

    pub fn node_mut(&mut self, id: NodeId) -> Result<&mut RadixNode> {
        self.nodes
            .get_mut(id)
            .ok_or_else(|| FerriteError::Scheduler("radix: stale node id".into()))
    }

    pub fn live_nodes(&self) -> usize {
        self.nodes.len()
    }

    pub fn total_blocks(&self) -> usize {
        self.total_blocks
    }

    pub fn evictable_tokens(&self) -> usize {
        self.evictable_tokens
    }

    pub fn protected_tokens(&self) -> usize {
        self.protected_tokens
    }

    // -- match (SGLang `match_prefix`, page-aligned) ----------------------

    /// Longest-prefix match over page-aligned blocks.
    ///
    /// O(depth × bucket scan × page memcmp) — buckets are tiny (the
    /// same-first-token fork is rare; the common case is a single child
    /// per first token). The match is *committed blocks only*: a
    /// partially-filled live tail never shares (its pages are not frozen).
    pub fn match_prefix(&mut self, tokens: &[u32]) -> Option<PrefixMatch> {
        let clock = self.clock.next();
        let mut cur = self.root;
        let mut matched = 0usize;
        self.nodes.get_mut(cur).map(|n| n.last_used = clock);

        while matched + self.page_size <= tokens.len() {
            let first = tokens[matched];
            let child = match self.nodes.get(cur) {
                Some(n) => n.children.get(&first).and_then(|bucket| {
                    bucket
                        .iter()
                        .find(|&&c| {
                            let cn = self.nodes.get(c).expect("child id from live parent");
                            cn.tokens.len() == self.page_size
                                && cn.tokens[..] == tokens[matched..matched + self.page_size]
                                && cn.state.is_some()
                        })
                        .copied()
                }),
                None => break,
            };
            let Some(child) = child else { break };
            cur = child;
            matched += self.page_size;
            if let Some(n) = self.nodes.get_mut(cur) {
                n.last_used = clock;
            }
        }

        if matched == 0 {
            return None;
        }
        Some(PrefixMatch {
            node: cur,
            matched_tokens: matched,
            next_query_token: tokens.get(matched).copied(),
        })
    }

    // -- insert (block append; forks are sibling blocks) -------------------

    /// Attach a new child to `parent` holding a full block of `tokens`
    /// backed by `state` (the registry snapshot). The scheduler freezes
    /// one block at a time as a row's commit crosses page boundaries.
    ///
    /// **Identical block → reuse** (the radix-sharing payoff): if the
    /// parent already holds this exact block (two requests shared a
    /// prefix — e.g. the same chat-template header), the existing node
    /// is returned and the caller releases its fresh snapshot (one
    /// state per block is the invariant; block-equal prefixes imply
    /// equal GDN state). Divergent blocks with the same first token are
    /// *separate children* (the bucket) — no split is ever needed at
    /// page granularity (a mid-block fork just yields two sibling
    /// blocks sharing the parent).
    ///
    /// Returns `(node, inserted_new)`; `inserted_new == false` means the
    /// caller must release the state it minted for this block.
    pub fn insert_branch(
        &mut self,
        parent: NodeId,
        tokens: Vec<u32>,
        state: StateId,
    ) -> Result<(NodeId, bool)> {
        if tokens.is_empty() || tokens.len() > self.page_size {
            return Err(FerriteError::Scheduler(format!(
                "radix insert: block len {} not in 1..={}",
                tokens.len(),
                self.page_size
            )));
        }
        let key = tokens[0];
        // exact-block reuse (shared prefix: same block already cached)
        if let Some(bucket) = self
            .nodes
            .get(parent)
            .ok_or_else(|| FerriteError::Scheduler("radix insert: stale parent".into()))?
            .children
            .get(&key)
        {
            for &c in bucket {
                if let Some(cn) = self.nodes.get(c) {
                    if cn.tokens == tokens {
                        return Ok((c, false));
                    }
                }
            }
        }
        let start = {
            let p = self
                .nodes
                .get(parent)
                .ok_or_else(|| FerriteError::Scheduler("radix insert: stale parent".into()))?;
            p.start + p.tokens.len()
        };
        let child = self.nodes.insert(RadixNode {
            tokens,
            state: Some(state),
            start,
            parent,
            children: HashMap::new(),
            lock_ref: 0,
            last_used: self.clock.next(),
            subtree_blocks: 1,
        });
        self.nodes
            .get_mut(parent)
            .expect("parent re-borrowed")
            .children
            .entry(key)
            .or_default()
            .push(child);
        self.bubble_blocks(parent);
        self.total_blocks += 1;
        self.evictable_tokens += self.page_size;
        Ok((child, true))
    }

    fn bubble_blocks(&mut self, mut node: NodeId) {
        loop {
            let parent = match self.nodes.get(node) {
                Some(n) if n.parent != node => n.parent,
                _ => return,
            };
            if let Some(p) = self.nodes.get_mut(parent) {
                p.subtree_blocks = p.subtree_blocks.saturating_add(1);
            }
            node = parent;
        }
    }

    // -- pinning (SGLang inc/dec_lock_ref with token-sum accounting) -------

    /// Pin the path root→node (a decode row resumed from this match).
    /// Protected-token accounting follows SGLang: crossing 0→1 moves the
    /// node's block from the evictable sum to the protected sum.
    pub fn pin_path(&mut self, node: NodeId) -> PinnedPath {
        let mut path = vec![node];
        let mut tokens = 0usize;
        let mut cur = node;
        while let Some(n) = self.nodes.get(cur) {
            tokens += n.tokens.len();
            if n.parent == cur {
                break;
            }
            cur = n.parent;
            path.push(cur);
        }
        for &id in &path {
            if let Some(n) = self.nodes.get_mut(id) {
                if n.lock_ref == 0 {
                    // 0→1: block becomes protected
                    self.evictable_tokens = self.evictable_tokens.saturating_sub(n.tokens.len());
                    self.protected_tokens += n.tokens.len();
                }
                n.lock_ref += 1;
            }
        }
        PinnedPath { nodes: path, tokens }
    }

    /// Unpin a previously pinned path (SGLang dec_lock_ref semantics:
    /// 1→0 moves the block back to the evictable sum).
    pub fn unpin_path(&mut self, path: &PinnedPath) {
        for &id in &path.nodes {
            if let Some(n) = self.nodes.get_mut(id) {
                if n.lock_ref == 0 {
                    continue; // double-unpin guard (retire is idempotent)
                }
                n.lock_ref -= 1;
                if n.lock_ref == 0 {
                    self.protected_tokens = self.protected_tokens.saturating_sub(n.tokens.len());
                    self.evictable_tokens += n.tokens.len();
                }
            }
        }
    }

    // -- eviction (page-budget driven, cascade parents, tier-agnostic) -----

    /// Pick one unpinned LRU leaf (no mutation — the victim selector).
    ///
    /// The selector is deliberately split from [`Self::detach`]: the
    /// scheduler's page-budget reclaim first tries **demotion** (device →
    /// host → disk — the prefix survives, the tree keeps the node), and
    /// only drops (detaches) when every tier is full. SGLang conflates
    /// the two; that conflation is why their reclaim path cannot keep a
    /// cached prefix under pressure.
    pub fn evict_victim(&self) -> Option<(NodeId, StateId, usize)> {
        let mut victim: Option<NodeId> = None;
        let mut victim_stamp = u64::MAX;
        for (id, n) in self.nodes.iter() {
            if n.children.is_empty() && n.lock_ref == 0 && id != self.root {
                if n.last_used < victim_stamp {
                    victim_stamp = n.last_used;
                    victim = Some(id);
                }
            }
        }
        let victim = victim?;
        let n = self.nodes.get(victim).expect("victim just found");
        Some((victim, n.state?, n.tokens.len()))
    }

    /// Detach a victim leaf (the drop path — after the selector said no
    /// tier has room). Unlinks from its parent, drops subtree weights and
    /// token sums; the caller releases the registry snapshot.
    pub fn detach(&mut self, node: NodeId) -> Result<(StateId, usize)> {
        let (tokens, state, parent) = {
            let n = self
                .nodes
                .get(node)
                .ok_or_else(|| FerriteError::Scheduler("detach: stale node".into()))?;
            if !n.children.is_empty() {
                return Err(FerriteError::Scheduler("detach: node has children".into()));
            }
            (n.tokens.clone(), n.state, n.parent)
        };
        let state = state.ok_or_else(|| FerriteError::Scheduler("detach: node without state".into()))?;
        // unlink from the parent's sibling bucket (first-token key)
        if let Some(first) = tokens.first() {
            if let Some(p) = self.nodes.get_mut(parent) {
                if let Some(bucket) = p.children.get_mut(first) {
                    bucket.retain(|&c| c != node);
                    if bucket.is_empty() {
                        p.children.remove(first);
                    }
                }
            }
        }
        let mut cur = parent;
        loop {
            let n = match self.nodes.get_mut(cur) {
                Some(n) => n,
                None => break,
            };
            n.subtree_blocks = n.subtree_blocks.saturating_sub(1);
            let up = n.parent;
            if up == cur {
                break;
            }
            cur = up;
        }
        self.nodes.remove(node);
        self.total_blocks = self.total_blocks.saturating_sub(1);
        self.evictable_tokens = self.evictable_tokens.saturating_sub(tokens.len());
        Ok((state, tokens.len()))
    }

    /// Evict unpinned LRU leaves until at least `pages_needed` pages are
    /// reclaimable (page-budget driven — SGLang `evict(num_tokens)`).
    ///
    /// Victims are **detached from the tree** (unlinked + token sums
    /// updated) and returned for physical-layer disposition: the caller
    /// demotes device-tier snapshots to host (prefix retained) or drops
    /// host-tier ones (registry release), per its tier pressure. Parent
    /// cascade: an evicted leaf may leave its parent childless+unpinned —
    /// that parent becomes evictable next round (this call cascades
    /// greedily, as eviction under pressure is recursive by nature).
    ///
    /// Returns the victims (call until physical pressure clears, then
    /// stop — the returned list is *already detached*; releasing their
    /// registry state is the caller's physical-layer move).
    pub fn evict_for_pages(&mut self, pages_needed: usize) -> Vec<EvictVictim> {
        let mut victims = Vec::new();
        let mut reclaimed = 0usize;
        while reclaimed < pages_needed {
            // LRU scan over the (small) unpinned-leaf frontier — the
            // linear scan is deliberate: per-tick victim counts are tiny
            // (admission granularity), and a heap costs an allocation +
            // invariant maintenance per insert/touch. (SGLang uses a heap
            // at page granularity; our leaves are single blocks.)
            let mut victim: Option<NodeId> = None;
            let mut victim_stamp = u64::MAX;
            for (id, n) in self.nodes.iter() {
                if n.children.is_empty() && n.lock_ref == 0 && id != self.root {
                    if n.last_used < victim_stamp {
                        victim_stamp = n.last_used;
                        victim = Some(id);
                    }
                }
            }
            let Some(victim) = victim else { break };
            let (tokens, state, parent) = {
                let n = self.nodes.get(victim).expect("victim just found");
                (n.tokens.clone(), n.state, n.parent)
            };
            let Some(state) = state else { break };
            let pages = tokens.len().div_ceil(self.page_size).max(1);
            reclaimed += pages;
            // unlink from the parent's sibling bucket, drop subtree weight
            if let Some(first) = tokens.first() {
                if let Some(p) = self.nodes.get_mut(parent) {
                    if let Some(bucket) = p.children.get_mut(first) {
                        bucket.retain(|&c| c != victim);
                        if bucket.is_empty() {
                            p.children.remove(first);
                        }
                    }
                }
            }
            let mut cur = parent;
            loop {
                let n = match self.nodes.get_mut(cur) {
                    Some(n) => n,
                    None => break,
                };
                n.subtree_blocks = n.subtree_blocks.saturating_sub(1);
                let up = n.parent;
                if up == cur {
                    break;
                }
                cur = up;
            }
            self.nodes.remove(victim);
            self.total_blocks = self.total_blocks.saturating_sub(1);
            self.evictable_tokens = self.evictable_tokens.saturating_sub(tokens.len());
            victims.push(EvictVictim {
                node: victim,
                state,
                pages,
                tokens: tokens.len(),
                on_host: false,
            });
            // cascade: childless + unpinned parent is reclaimable this round
        }
        victims
    }

    // -- traversal helpers (scheduler / diagnostics) ----------------------

    /// Dump the node path's tokens (root → leaf), for prefill resumes and
    /// cache-behavior diagnostics.
    pub fn path_tokens(&self, node: NodeId, out: &mut Vec<u32>) -> Result<()> {
        let mut cur = node;
        let mut stack = Vec::new();
        while let Some(n) = self.nodes.get(cur) {
            if n.parent == cur {
                break;
            }
            stack.extend_from_slice(&n.tokens);
            cur = n.parent;
        }
        while let Some(t) = stack.pop() {
            out.push(t);
        }
        self.nodes
            .get(node)
            .ok_or_else(|| FerriteError::Scheduler("path_tokens: stale node".into()))?;
        Ok(())
    }
}

/// Reserved handle for self-referential sentinels (the root's parent
/// points at itself; the root is always the arena's first insertion).
impl NodeId {
    pub(crate) fn from_raw(idx: u32) -> Self {
        Self::new_unchecked(idx)
    }
}

impl StateId {
    /// Mint an **opaque** state handle for an engine that owns its physical
    /// state outside `StateRegistry`.
    ///
    /// The tree only *carries* the handle (it never dereferences it — `state`
    /// is what the scheduler's physical layer later promotes/demotes/releases).
    /// An engine whose per-seq state is not a registry snapshot (the CUDA
    /// `GpuEngine`'s parked sequences, whose KV lives in the seq's own device
    /// buffers) stamps one of these per block so the node is matchable, and
    /// keeps its own `NodeId → seq` map for the physical side. The registry is
    /// never consulted for such ids — mixing them into a real `StateRegistry`
    /// would be a lookup miss, not a bogus hit.
    pub fn opaque(raw: u32) -> Self {
        Self::new_unchecked(raw)
    }
}
