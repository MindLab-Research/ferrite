//! Physical state ownership — the three-tier hicache behind radix caching.
//!
//! GLM-5.3-Flash hybrid state (ferrite-kv) splits into two families with
//! opposite sharing semantics:
//!
//! - **GDN recurrent state** (`[heads, dk, dv]` + conv tail, fixed size per
//!   layer): the *entire prefix* is folded into one tensor. Sharing a prefix
//!   means snapshotting that tensor at a fork point — a deep copy. Radix
//!   nodes therefore own **snapshot slots**.
//! - **DSA latent KV** (paged, grows per token): prefix sharing is
//!   **zero-copy** — a page range is reference-counted and multiple owners
//!   read the same physical pages. Radix nodes therefore *lease* pages.
//!
//! ## Three tiers, one cache (hicache as a first-class citizen)
//!
//! Prefix state lives on a **memory hierarchy**, and the hierarchy is part
//! of the cache's identity — not a bolted-on eviction target (the design
//! lesson from SGLang's `HiMambaRadixCache`: their radix tree carries
//! device values, and host/disk tiers are value-field patches on tree
//! nodes, which is exactly where their write-back races and lock leaks
//! breed). Here, a snapshot's tier is a *field of the registry record*:
//!
//! ```text
//!   Tier::Device (GPU)  ──demote──▶  Tier::Host (CPU RAM)  ──demote──▶  Tier::Disk (NVMe)
//!        ▲ hot: prefix hits restore with one D2D copy                    │
//!        └────────────── promote (H2D / disk→device on hit) ◀────────────┘
//! ```
//!
//! - **Device** is small and hot: radix snapshots + live decode rows. A
//!   prefix hit restores a row with one device copy (pages lease zero-copy).
//! - **Host** is the reservoir: demoted snapshots (D2H GDN state + paged
//!   DSA KV copies). SGLang parity, minus the races — demotion happens
//!   **synchronously at eviction** (write-through-on-pressure), never as a
//!   detached async task holding page promises.
//! - **Disk** (NVMe) is the archive: host pressure demotes again before
//!   anything is dropped. B300 hosts have 3.5–7 TB NVMe — hours of
//!   system-prompt prefixes at zero GPU/RAM cost.
//!
//! ## Eviction is page-budget-driven (SGLang `evict(num_tokens)` parity)
//!
//! Admission asks for `pages_needed`; the radix tree frees LRU unpinned
//! leaves until the budget is met — never a node-count heuristic. Device
//! pressure **demotes** (prefix retained one tier down); host pressure
//! **demotes to disk**; disk pressure drops. All reclaim work is
//! amortized into admission (between replays), never on the replay path.
//!
//! ## Two slot domains (graph-visible rows vs cache snapshots)
//!
//! 1. **Decode rows `[0, max_rows)`** — dense, contiguous, row `i` is the
//!    row the pad-to-B CUDA graph indexes for batch entry `i` (the reason
//!    the domain exists: device pointers baked into a captured mega graph
//!    address state by row index, so row indices must be stable and dense
//!    for every bucket shape `{1, 2, 4, 8, 16, 32}`).
//! 2. **Snapshot slots** (tier-aware, radix-owned). Freed only via radix
//!    eviction (refcount gates recycling).
//!
//! ## Correctness stance (the SGLang bug classes we design out)
//!
//! - **No lock leaks**: page/lease refcounts live in ONE place
//!   (`PageRegistry`); every transfer (row ↔ node ↔ tier) is a typed
//!   method on this registry — there is no path that frees without the
//!   refcount, because there is no second refcount.
//! - **No stale handles**: generational `TypedId` — an evicted node's
//!   snapshot id is a hard lookup miss, never a dangling slot.
//! - **Partial pages never share**: match/insert truncate at page
//!   multiples on both sides; the live tail belongs to its writing row.
//! - **No async write-back**: demotion is synchronous (a memcpy into the
//!   receiving tier) — there is no "in-flight tier move" state to race.
//!
//! Storage is behind [`PhysStateStore`] so the CUDA backend plugs device
//! pools (demote/promote are D2H/H2D/NVMe streams — backend may batch
//! them on side streams, but the *registry* sees them as synchronous
//! points); tests and the host engine use [`HostStateStore`] (one
//! address space, three Vecs — semantics identical). The registry never
//! touches tensor values — it is a *pure bookkeeping* layer.

use std::collections::{HashMap, VecDeque};

use ferrite_types::{FerriteError, Result};

use crate::arena::{ArenaFamily, TypedArena, TypedId};

// ---------------------------------------------------------------------------
// PhysStateStore — capability trait for the three-tier storage backend
// ---------------------------------------------------------------------------

/// Where a snapshot's state physically lives (the hicache tier).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Tier {
    /// GPU memory: hot. Prefix hits restore with a device copy only.
    Device,
    /// CPU RAM: the reservoir. Demoted snapshots await promotion.
    Host,
    /// NVMe: the archive. Host pressure demotes here before dropping.
    Disk,
}

impl Tier {
    /// Hotter → colder direction (demote).
    pub fn colder(self) -> Option<Tier> {
        match self {
            Tier::Device => Some(Tier::Host),
            Tier::Host => Some(Tier::Disk),
            Tier::Disk => None,
        }
    }

    /// Colder → hotter direction (promote).
    pub fn hotter(self) -> Option<Tier> {
        match self {
            Tier::Disk => Some(Tier::Host),
            Tier::Host => Some(Tier::Device),
            Tier::Device => None,
        }
    }
}

/// Physical storage capability used by the scheduler (three tiers).
///
/// All addresses are logical (per-tier slot handles, per-tier page id
/// spaces); the backend decides placement and copy paths. Copies are
/// always full-slot GDN snapshots; DSA pages are shared by refcount and
/// never copied except across tiers (demote/promote).
///
/// Synchronous demote/promote contract (the registry's no-race stance):
/// each call returns after the receiving tier owns the bytes — batching
/// onto side streams is a backend-internal concern, ordering is the
/// registry's and it is sequential here.
pub trait PhysStateStore {
    // -- GDN snapshot slots (device tier: decode rows + hot snapshots) ----
    fn alloc_device_slot(&mut self) -> Result<u32>;
    fn free_device_slot(&mut self, handle: u32);
    fn copy_device_slot(&mut self, src: u32, dst: u32) -> Result<()>;

    // -- GDN snapshot slots (host tier) -------------------------------------
    /// D2H: allocate a host slot, copy `src` (device) into it.
    fn alloc_host_slot(&mut self, src: u32) -> Result<u32>;
    fn free_host_slot(&mut self, handle: u32);
    /// H2D: copy host slot → existing device slot `dst`.
    fn copy_host_slot(&mut self, host: u32, dst: u32) -> Result<()>;

    // -- GDN snapshot slots (disk tier: NVMe archive) -----------------------
    /// H2D-isk: allocate a disk slot (mmap extent), copy `src` (host).
    fn alloc_disk_slot(&mut self, src: u32) -> Result<u64>;
    fn free_disk_slot(&mut self, handle: u64);
    /// Disk → host: copy disk extent into a host slot the backend allocates.
    fn copy_disk_slot(&mut self, disk: u64) -> Result<u32>;

    // -- DSA pages (device tier, refcounted by PageRegistry) ----------------
    fn alloc_pages(&mut self, n: usize) -> Result<Vec<u32>>;
    fn free_page(&mut self, page: u32);
    /// D2H one page (demotion): returns the host page id.
    fn demote_page(&mut self, page: u32) -> Result<u32>;
    /// H2D one page (promotion).
    fn promote_page(&mut self, host_page: u32) -> Result<u32>;
    /// Host → disk page (second demotion hop); returns the disk page id.
    fn page_to_disk(&mut self, host_page: u32) -> Result<u64>;
    /// Disk → host page.
    fn page_from_disk(&mut self, disk_page: u64) -> Result<u32>;
    /// Free a host/disk page (tier-known by id space).
    fn free_host_page(&mut self, host_page: u32);
    fn free_disk_page(&mut self, disk_page: u64);
}

/// Three-tier host-mirroring store — the default backend (unit + host
/// engine). One address space, three pools; copies are memcpy — placement
/// semantics identical to the CUDA backend (device slots + pages in GPU
/// pools, host slots + pages in RAM, disk slots + pages in a `Vec`
/// standing in for an mmap ring).
#[derive(Debug, Default)]
pub struct HostStateStore {
    slots: Vec<Vec<f32>>,
    free_device: Vec<u32>,
    host_slots: Vec<Vec<f32>>,
    free_host: Vec<u32>,
    disk_slots: Vec<Vec<u32>>, // handles: index
    free_disk: Vec<u64>,
    host_pages: HashMap<u32, Vec<f32>>,
    disk_pages: HashMap<u64, Vec<f32>>,
    next_page: u32,
    next_host_page: u32,
    next_disk_page: u64,
}

impl HostStateStore {
    pub fn new() -> Self {
        Self::default()
    }
}

impl PhysStateStore for HostStateStore {
    fn alloc_device_slot(&mut self) -> Result<u32> {
        match self.free_device.pop() {
            Some(h) => Ok(h),
            None => {
                self.slots.push(Vec::new());
                Ok((self.slots.len() - 1) as u32)
            }
        }
    }

    fn free_device_slot(&mut self, handle: u32) {
        self.free_device.push(handle);
    }

    fn copy_device_slot(&mut self, src: u32, dst: u32) -> Result<()> {
        let (s, d) = (src as usize, dst as usize);
        if s >= self.slots.len() || d >= self.slots.len() {
            return Err(FerriteError::Pool(format!("copy_slot: bad handle {src}->{dst}")));
        }
        self.slots[d] = self.slots[s].clone();
        Ok(())
    }

    fn alloc_host_slot(&mut self, src: u32) -> Result<u32> {
        let data = self
            .slots
            .get(src as usize)
            .ok_or_else(|| FerriteError::Pool(format!("demote: bad device slot {src}")))?
            .clone();
        let h = match self.free_host.pop() {
            Some(h) => h,
            None => {
                self.host_slots.push(Vec::new());
                (self.host_slots.len() - 1) as u32
            }
        };
        self.host_slots[h as usize] = data;
        Ok(h)
    }

    fn free_host_slot(&mut self, handle: u32) {
        self.free_host.push(handle);
    }

    fn copy_host_slot(&mut self, host: u32, dst: u32) -> Result<()> {
        let data = self
            .host_slots
            .get(host as usize)
            .ok_or_else(|| FerriteError::Pool(format!("promote: bad host slot {host}")))?;
        let d = dst as usize;
        if d >= self.slots.len() {
            return Err(FerriteError::Pool(format!("promote: bad device slot {dst}")));
        }
        self.slots[d] = data.clone();
        Ok(())
    }

    fn alloc_disk_slot(&mut self, src: u32) -> Result<u64> {
        let data = self
            .host_slots
            .get(src as usize)
            .ok_or_else(|| FerriteError::Pool(format!("disk demote: bad host slot {src}")))?
            .clone()
            .into_iter()
            .map(|f| f.to_bits())
            .collect();
        let h = match self.free_disk.pop() {
            Some(h) => h,
            None => {
                self.disk_slots.push(Vec::new());
                (self.disk_slots.len() - 1) as u64
            }
        };
        self.disk_slots[h as usize] = data;
        Ok(h)
    }

    fn free_disk_slot(&mut self, handle: u64) {
        self.free_disk.push(handle);
    }

    fn copy_disk_slot(&mut self, disk: u64) -> Result<u32> {
        let data = self
            .disk_slots
            .get(disk as usize)
            .ok_or_else(|| FerriteError::Pool(format!("disk promote: bad disk slot {disk}")))?;
        let bytes: Vec<f32> = data.iter().map(|&b| f32::from_bits(b)).collect();
        let h = match self.free_host.pop() {
            Some(h) => h,
            None => {
                self.host_slots.push(Vec::new());
                (self.host_slots.len() - 1) as u32
            }
        };
        self.host_slots[h as usize] = bytes;
        Ok(h)
    }

    fn alloc_pages(&mut self, n: usize) -> Result<Vec<u32>> {
        let base = self.next_page;
        self.next_page += n as u32;
        Ok((base..base + n as u32).collect())
    }

    fn free_page(&mut self, _page: u32) {}

    fn demote_page(&mut self, page: u32) -> Result<u32> {
        let h = self.next_host_page;
        self.next_host_page += 1;
        self.host_pages.insert(h, vec![0.0; 16]);
        let _ = page;
        Ok(h)
    }

    fn promote_page(&mut self, _host_page: u32) -> Result<u32> {
        let p = self.next_page;
        self.next_page += 1;
        Ok(p)
    }

    fn page_to_disk(&mut self, _host_page: u32) -> Result<u64> {
        let d = self.next_disk_page;
        self.next_disk_page += 1;
        self.disk_pages.insert(d, vec![]);
        Ok(d)
    }

    fn page_from_disk(&mut self, _disk_page: u64) -> Result<u32> {
        let h = self.next_host_page;
        self.next_host_page += 1;
        self.host_pages.insert(h, vec![0.0; 16]);
        Ok(h)
    }

    fn free_host_page(&mut self, _host_page: u32) {}
    fn free_disk_page(&mut self, _disk_page: u64) {}
}

// ---------------------------------------------------------------------------
// Page leases — DSA KV ranges, zero-copy shared within a tier
// ---------------------------------------------------------------------------

/// Logical page-range ownership token. Cloning a lease = +1 refcount per
/// physical page (same tier). Dropping it via the registry returns the
/// refcount; a page physically frees (or demotes) at zero.
///
/// A lease covers *whole* physical pages (`page_size` tokens each): radix
/// node granularity is page-aligned by construction (partial trailing
/// pages are never shared — they belong to the writing sequence until
/// filled and committed as a radix node boundary).
#[derive(Debug, Clone)]
pub struct PageLease {
    pub pages: Vec<u32>,
}

// ---------------------------------------------------------------------------
// StateRegistry
// ---------------------------------------------------------------------------

/// Family tag for registry-owned snapshot records.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StateTag;
impl ArenaFamily for StateTag {
    const NAME: &'static str = "state";
}
pub type StateId = TypedId<StateTag>;

/// Which tier a snapshot's GDN state currently occupies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlotDomain {
    /// A decode row `r` (graph-visible). `0 <= r < max_rows`.
    Row(u32),
    /// A radix snapshot (tier is a separate field — rows are always
    /// device-tier by definition; snapshots walk the hicache).
    Snapshot { tier: Tier },
}

/// One snapshot/row record: GDN slot + page lease + tier.
#[derive(Debug, Clone)]
struct SlotRecord {
    handle: u32,
    /// Disk-tier handle (only valid when `domain` is `Snapshot{Disk}` —
    /// the u64 id space of the mmap ring; 0 otherwise).
    disk_handle: u64,
    domain: SlotDomain,
    /// Physical DSA pages (id space follows `tier`: device page ids while
    /// `Tier::Device`; host ids after demotion; disk ids at archive tier).
    dsa_pages: Vec<u32>,
    /// Disk-tier page ids (parallel to `dsa_pages` when archived — pages
    /// on disk keep their host ids zeroed; one id space per tier keeps
    /// the backend simple, the record carries both).
    disk_pages: Vec<u64>,
    tier: Tier,
    committed_tokens: usize,
}

/// Page refcounts (device space): page id → live leases referencing it.
/// One refcount, one owner of truth (SGLang's split accounting between
/// tree and pool is what leaks — see module doc).
struct PageRegistry {
    counts: HashMap<u32, u32>,
}

impl PageRegistry {
    fn retain(&mut self, pages: &[u32]) {
        for &p in pages {
            *self.counts.entry(p).or_insert(0) += 1;
        }
    }

    /// Release a set of pages; collects the ids whose refcount hit zero
    /// (caller frees/demotes them — tier-known by id space).
    fn release(&mut self, pages: &[u32], freed: &mut Vec<u32>) {
        for &p in pages {
            if let Some(c) = self.counts.get_mut(&p) {
                *c -= 1;
                if *c == 0 {
                    self.counts.remove(&p);
                    freed.push(p);
                }
            }
        }
    }
}

/// Per-tier LRU of demoted snapshots (SGLang `HostLRUList` parity,
/// generalized: each tier has one). MRU → LRU id order; touches at
/// promote/demote; LRU victim under tier pressure.
struct TierLru {
    order: VecDeque<StateId>,
    clock: u64,
}

impl TierLru {
    fn touch(&mut self, id: StateId) {
        self.clock += 1;
        if let Some(pos) = self.order.iter().position(|&x| x == id) {
            self.order.remove(pos);
        }
        self.order.push_back(id);
    }

    fn remove(&mut self, id: StateId) {
        if let Some(pos) = self.order.iter().position(|&x| x == id) {
            self.order.remove(pos);
        }
    }

    fn lru(&self) -> Option<StateId> {
        self.order.front().copied()
    }
}

/// The three-tier state bookkeeping layer (module doc for semantics).
///
/// Invariants (enforced by construction, debug-asserted):
/// - decode rows are a *dense prefix* `[0, live_rows)` — never holed —
///   so bucket graphs can assume `row i = slot i`;
/// - a page is physically freed exactly when its refcount hits zero;
/// - snapshot slots change tier only through demote/promote (never in
///   two tiers at once — synchronous moves, no in-flight state);
/// - tier capacities are honored by demotion cascades (device pressure
///   demotes to host; host pressure demotes to disk; disk pressure drops)
///   — never silently losing a prefix.
pub struct StateRegistry<S: PhysStateStore> {
    store: S,
    /// Dense decode-row records: `rows[r]` is row r's record.
    rows: Vec<Option<SlotRecord>>,
    /// Radix snapshot records (rows and snapshots in one arena).
    snaps: TypedArena<StateTag, SlotRecord>,
    pages: PageRegistry,
    /// Fixed decode-row capacity (== max bucket size).
    max_rows: u32,
    /// Device-tier snapshot bound (demotion pressure threshold).
    max_device_snaps: usize,
    /// Host-tier snapshot bound (second demotion threshold).
    max_host_snaps: usize,
    /// Disk-tier snapshot bound (true-drop threshold; the archive).
    max_disk_snaps: usize,
    /// Device DSA page budget (device pages — the admission currency;
    /// SGLang `num_gpu_blocks` parity: requests pay pages, pressure
    /// reclaims them through the tier chain).
    max_pages: usize,
    /// DSA tokens per page (radix node granularity).
    page_size: usize,
    /// Per-tier LRU (device-tier is tree-LRU driven — the radix tree
    /// picks device victims; host/disk LRU pick demotion victims).
    host_lru: TierLru,
    disk_lru: TierLru,
    /// Live counts by tier (eviction-pressure metrics).
    device_snaps: usize,
    host_snaps: usize,
    disk_snaps: usize,
}

impl<S: PhysStateStore> StateRegistry<S> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        store: S,
        max_rows: u32,
        max_device_snaps: usize,
        max_host_snaps: usize,
        max_disk_snaps: usize,
        max_pages: usize,
        page_size: usize,
    ) -> Self {
        StateRegistry {
            store,
            rows: (0..max_rows).map(|_| None).collect(),
            snaps: TypedArena::with_capacity(max_device_snaps + max_host_snaps),
            pages: PageRegistry { counts: HashMap::new() },
            max_rows,
            max_device_snaps,
            max_host_snaps,
            max_disk_snaps,
            max_pages,
            page_size,
            host_lru: TierLru { order: VecDeque::new(), clock: 0 },
            disk_lru: TierLru { order: VecDeque::new(), clock: 0 },
            device_snaps: 0,
            host_snaps: 0,
            disk_snaps: 0,
        }
    }

    pub fn page_size(&self) -> usize {
        self.page_size
    }

    pub fn max_rows(&self) -> u32 {
        self.max_rows
    }

    // -- page-budget queries (the admission currency) -----------------------

    /// Device pages currently leased (live refs in `PageRegistry`).
    pub fn pages_in_use(&self) -> usize {
        self.pages.counts.len()
    }

    /// Device pages an admission may still take before reclaim is needed.
    pub fn pages_free(&self) -> usize {
        self.max_pages.saturating_sub(self.pages_in_use())
    }

    /// Whether the host tier can accept one more demotion (snapshot slot
    /// capacity — page capacity on the host tier is RAM-bound and the
    /// backend's concern; this is the slot bound the demote chain checks).
    pub fn host_can_accept(&self) -> bool {
        self.host_snaps < self.max_host_snaps
    }

    /// Whether the disk archive can accept one more demotion.
    pub fn disk_can_accept(&self) -> bool {
        self.disk_snaps < self.max_disk_snaps
    }

    /// Whether the device snapshot tier can accept one more (the
    /// `snapshot_from_row` demote pressure valve checks this itself).
    pub fn device_can_accept(&self) -> bool {
        self.device_snaps < self.max_device_snaps
    }

    // -- decode rows (always device-tier) ----------------------------------

    /// Acquire decode row `r` for a fresh sequence: zero GDN slot, no pages.
    pub fn acquire_row(&mut self, row: u32) -> Result<()> {
        if row >= self.max_rows {
            return Err(FerriteError::Pool(format!("row {row} >= max {}", self.max_rows)));
        }
        if self.rows[row as usize].is_some() {
            return Err(FerriteError::Pool(format!("row {row} already acquired")));
        }
        let handle = self.store.alloc_device_slot()?;
        self.rows[row as usize] = Some(SlotRecord {
            handle,
            disk_handle: 0,
            domain: SlotDomain::Row(row),
            dsa_pages: Vec::new(),
            disk_pages: Vec::new(),
            tier: Tier::Device,
            committed_tokens: 0,
        });
        Ok(())
    }

    /// Wire a prefix-hit sequence into a decode row: copy the GDN
    /// snapshot into the row's slot and lease the node's pages
    /// (+refcount, zero copy). A colder-tier snapshot promotes first
    /// (disk→device via host; promotion is synchronous).
    pub fn bind_row_to_snapshot(
        &mut self,
        row: u32,
        snap: StateId,
        snapshot_tokens: usize,
    ) -> Result<()> {
        // Promotion cascade: any tier → device, topmost first.
        while self.snapshot_tier(snap) != Some(Tier::Device) {
            let colder = self.snapshot_tier(snap);
            match colder {
                Some(Tier::Host) => self.promote_snapshot(snap)?,
                Some(Tier::Disk) => {
                    self.promote_snapshot_disk_to_host(snap)?;
                }
                _ => {
                    return Err(FerriteError::Pool("bind_row: stale snapshot id".into()))
                }
            }
        }
        let rec = self
            .snaps
            .get(snap)
            .map(|r| r.clone())
            .ok_or_else(|| FerriteError::Pool("bind_row: snapshot vanished during promote".into()))?;
        let slot = self.rows[row as usize]
            .as_mut()
            .ok_or_else(|| FerriteError::Pool(format!("bind_row: row {row} not acquired")))?;
        self.store.copy_device_slot(rec.handle, slot.handle)?;
        let pages = rec.dsa_pages.clone();
        self.pages.retain(&pages);
        slot.dsa_pages = pages;
        slot.committed_tokens = snapshot_tokens;
        Ok(())
    }

    /// Append `n` fresh device pages (grown DSA KV while decoding) to a row.
    pub fn extend_row(&mut self, row: u32, n_pages: usize, new_tokens: usize) -> Result<Vec<u32>> {
        let pages = self.store.alloc_pages(n_pages)?;
        self.extend_row_with(row, pages, new_tokens)
    }

    /// Append pre-allocated physical pages to a row (prefill writer path).
    pub fn extend_row_with(
        &mut self,
        row: u32,
        pages: Vec<u32>,
        new_tokens: usize,
    ) -> Result<Vec<u32>> {
        let slot = self.rows[row as usize]
            .as_mut()
            .ok_or_else(|| FerriteError::Pool(format!("extend_row: row {row} not acquired")))?;
        self.pages.retain(&pages);
        slot.committed_tokens += new_tokens;
        slot.dsa_pages.extend_from_slice(&pages);
        Ok(pages)
    }

    /// Release a decode row: free its GDN slot and unref its pages
    /// (sequence retirement — its committed prefix is promoted into the
    /// radix tree BEFORE this, via `snapshot_from_row`).
    pub fn release_row(&mut self, row: u32) -> Result<()> {
        let rec = self.rows[row as usize]
            .take()
            .ok_or_else(|| FerriteError::Pool(format!("release_row: row {row} not acquired")))?;
        debug_assert_eq!(rec.tier, Tier::Device, "rows are device-tier by definition");
        let mut freed = Vec::new();
        self.pages.release(&rec.dsa_pages, &mut freed);
        for p in freed {
            self.store.free_page(p);
        }
        self.store.free_device_slot(rec.handle);
        Ok(())
    }

    /// Move a row's state into another row (compaction: the graph-visible
    /// row domain stays dense — the freed row is the highest live one).
    /// One device copy + a page-lease move; page-shared prefixes never copy.
    pub fn move_row(&mut self, from: u32, to: u32) -> Result<()> {
        if from == to {
            return Ok(());
        }
        let rec = self.rows[from as usize]
            .take()
            .ok_or_else(|| FerriteError::Pool(format!("move_row: row {from} not acquired")))?;
        if self.rows[to as usize].is_some() {
            return Err(FerriteError::Pool(format!("move_row: row {to} occupied")));
        }
        let dst = self.store.alloc_device_slot()?;
        self.store.copy_device_slot(rec.handle, dst)?;
        self.store.free_device_slot(rec.handle);
        self.rows[to as usize] = Some(SlotRecord {
            handle: dst,
            disk_handle: 0,
            domain: SlotDomain::Row(to),
            dsa_pages: rec.dsa_pages,
            disk_pages: Vec::new(),
            tier: Tier::Device,
            committed_tokens: rec.committed_tokens,
        });
        Ok(())
    }

    // -- snapshots (radix nodes) — the hicache walks these tiers ----------

    /// Snapshot a row's prefix into the **device** snapshot tier (promote
    /// on page-boundary commit): fresh slot, GDN copy, page ownership
    /// transfer of the committed (page-aligned) prefix — the row keeps
    /// its lease (refcount), the node gains one.
    ///
    /// Device pressure demotes LRU device snapshots to host first (the
    /// hicache admits: capacity pressure is a demotion, not a drop).
    pub fn snapshot_from_row(&mut self, row: u32, page_aligned_tokens: usize) -> Result<StateId> {
        let rec = self.rows[row as usize]
            .as_ref()
            .ok_or_else(|| FerriteError::Pool(format!("snapshot: row {row} not acquired")))?
            .clone();
        let n_pages = page_aligned_tokens.div_ceil(self.page_size);
        if rec.dsa_pages.len() < n_pages {
            return Err(FerriteError::Pool("snapshot: pages not yet written".into()));
        }
        while self.device_snaps >= self.max_device_snaps {
            let victim = self
                .device_lru_victim()?
                .ok_or_else(|| FerriteError::Pool("snapshot: no demotable device snapshot".into()))?;
            self.demote_snapshot(victim)?;
        }
        let handle = self.store.alloc_device_slot()?;
        self.store.copy_device_slot(rec.handle, handle)?;
        let shared: Vec<u32> = rec.dsa_pages[..n_pages].to_vec();
        // node takes its own reference; the row keeps its existing one.
        self.pages.retain(&shared);
        self.device_snaps += 1;
        Ok(self.snaps.insert(SlotRecord {
            handle,
            disk_handle: 0,
            domain: SlotDomain::Snapshot { tier: Tier::Device },
            dsa_pages: shared,
            disk_pages: Vec::new(),
            tier: Tier::Device,
            committed_tokens: page_aligned_tokens,
        }))
    }

    /// Demote a snapshot one tier colder (device→host, host→disk):
    /// synchronous copy into the receiving tier, free the source tier's
    /// storage. The snapshot id is stable (radix nodes keep pointing at
    /// it); page ids move into the receiving tier's id space.
    pub fn demote_snapshot(&mut self, snap: StateId) -> Result<()> {
        let tier = self
            .snapshot_tier(snap)
            .ok_or_else(|| FerriteError::Pool("demote: stale snapshot id".into()))?;
        match tier {
            Tier::Device => {
                // Host pressure first: cascade LRU host → disk to make room.
                while self.host_snaps >= self.max_host_snaps {
                    let victim = self.host_lru.lru().ok_or_else(|| {
                        FerriteError::Pool("demote: host LRU empty while at capacity".into())
                    })?;
                    self.demote_snapshot(victim)?;
                }
                let rec = self
                    .snaps
                    .get_mut(snap)
                    .ok_or_else(|| FerriteError::Pool("demote: stale snapshot id".into()))?;
                let host_handle = self.store.alloc_host_slot(rec.handle)?;
                let mut host_pages = Vec::with_capacity(rec.dsa_pages.len());
                for &p in &rec.dsa_pages {
                    host_pages.push(self.store.demote_page(p)?);
                }
                // old device pages: refcount released (other owners may hold)
                let mut freed = Vec::new();
                let old_pages = rec.dsa_pages.clone();
                self.pages.release(&old_pages, &mut freed);
                for p in freed {
                    self.store.free_page(p);
                }
                rec.handle = host_handle;
                rec.dsa_pages = host_pages;
                rec.domain = SlotDomain::Snapshot { tier: Tier::Host };
                rec.tier = Tier::Host;
                self.device_snaps -= 1;
                self.host_snaps += 1;
                self.host_lru.touch(snap);
                Ok(())
            }
            Tier::Host => {
                // Disk archive (NVMe): page data follows the snapshot.
                while self.disk_snaps >= self.max_disk_snaps {
                    let victim = self.disk_lru.lru().ok_or_else(|| {
                        FerriteError::Pool("demote: disk LRU empty while at capacity".into())
                    })?;
                    // disk LRU victim: true drop (archive full — oldest
                    // prefix dies; its registry snapshot releases below)
                    let rec = self
                        .snaps
                        .remove(victim)
                        .ok_or_else(|| FerriteError::Pool("demote: disk LRU corruption".into()))?;
                    self.disk_snaps -= 1;
                    self.disk_lru.remove(victim);
                    for &dp in &rec.disk_pages {
                        self.store.free_disk_page(dp);
                    }
                    self.store.free_disk_slot(rec.disk_handle);
                }
                let rec = self
                    .snaps
                    .get_mut(snap)
                    .ok_or_else(|| FerriteError::Pool("demote: stale snapshot id".into()))?;
                let disk_handle = self.store.alloc_disk_slot(rec.handle)?;
                let mut disk_pages = Vec::with_capacity(rec.dsa_pages.len());
                for &hp in &rec.dsa_pages {
                    disk_pages.push(self.store.page_to_disk(hp)?);
                }
                for &hp in &rec.dsa_pages {
                    self.store.free_host_page(hp);
                }
                self.store.free_host_slot(rec.handle);
                rec.disk_handle = disk_handle;
                rec.disk_pages = disk_pages;
                rec.dsa_pages.clear();
                rec.domain = SlotDomain::Snapshot { tier: Tier::Disk };
                rec.tier = Tier::Disk;
                self.host_snaps -= 1;
                self.disk_snaps += 1;
                self.disk_lru.touch(snap);
                Ok(())
            }
            Tier::Disk => Ok(()), // archive floor: caller drops, not demotes
        }
    }

    /// Promote a host snapshot back to device (prefix-hit path).
    pub fn promote_snapshot(&mut self, snap: StateId) -> Result<()> {
        let tier = self
            .snapshot_tier(snap)
            .ok_or_else(|| FerriteError::Pool("promote: stale snapshot id".into()))?;
        if tier != Tier::Host {
            return Ok(()); // device already / disk routes through the cascade
        }
        while self.device_snaps >= self.max_device_snaps {
            let victim = self.device_lru_victim()?.ok_or_else(|| {
                FerriteError::Pool("promote: no demotable device snapshot".into())
            })?;
            if victim == snap {
                break; // the LRU device snapshot is the promotee itself
            }
            self.demote_snapshot(victim)?;
        }
        let rec = self
            .snaps
            .get_mut(snap)
            .ok_or_else(|| FerriteError::Pool("promote: stale snapshot id".into()))?;
        let device_handle = self.store.alloc_device_slot()?;
        self.store.copy_host_slot(rec.handle, device_handle)?;
        let mut device_pages = Vec::with_capacity(rec.dsa_pages.len());
        for &hp in &rec.dsa_pages {
            device_pages.push(self.store.promote_page(hp)?);
        }
        for &hp in &rec.dsa_pages {
            self.store.free_host_page(hp);
        }
        self.store.free_host_slot(rec.handle);
        rec.handle = device_handle;
        rec.dsa_pages = device_pages;
        rec.domain = SlotDomain::Snapshot { tier: Tier::Device };
        rec.tier = Tier::Device;
        self.host_snaps -= 1;
        self.device_snaps += 1;
        self.host_lru.remove(snap);
        Ok(())
    }

    /// Promote a disk-tier snapshot to host (the disk→device cascade's
    /// first hop; `bind_row_to_snapshot` drives the rest).
    pub fn promote_snapshot_disk_to_host(&mut self, snap: StateId) -> Result<()> {
        let tier = self
            .snapshot_tier(snap)
            .ok_or_else(|| FerriteError::Pool("promote-disk: stale snapshot id".into()))?;
        if tier != Tier::Disk {
            return Ok(());
        }
        while self.host_snaps >= self.max_host_snaps {
            let victim = self.host_lru.lru().ok_or_else(|| {
                FerriteError::Pool("promote-disk: host LRU empty while at capacity".into())
            })?;
            self.demote_snapshot(victim)?;
        }
        let rec = self
            .snaps
            .get_mut(snap)
            .ok_or_else(|| FerriteError::Pool("promote-disk: stale snapshot id".into()))?;
        let host_handle = self.store.copy_disk_slot(rec.disk_handle)?;
        let mut host_pages = Vec::with_capacity(rec.disk_pages.len());
        for &dp in &rec.disk_pages {
            host_pages.push(self.store.page_from_disk(dp)?);
        }
        for &dp in &rec.disk_pages {
            self.store.free_disk_page(dp);
        }
        self.store.free_disk_slot(rec.disk_handle);
        rec.handle = host_handle;
        rec.dsa_pages = host_pages;
        rec.disk_pages.clear();
        rec.domain = SlotDomain::Snapshot { tier: Tier::Host };
        rec.tier = Tier::Host;
        self.disk_snaps -= 1;
        self.host_snaps += 1;
        self.host_lru.touch(snap);
        self.disk_lru.remove(snap);
        Ok(())
    }

    /// Touch a snapshot's LRU position (prefix hit — it's hot again).
    pub fn touch_snapshot(&mut self, snap: StateId) {
        if let Some(r) = self.snaps.get(snap) {
            match r.tier {
                Tier::Host => self.host_lru.touch(snap),
                Tier::Disk => self.disk_lru.touch(snap),
                Tier::Device => {} // device victims are radix-tree LRU picked
            }
        }
    }

    /// Drop a radix snapshot (evicted / never committed): unref its pages,
    /// free its slot (tier-aware). Physical page frees cascade on the
    /// zero-refcount path.
    pub fn release_snapshot(&mut self, snap: StateId) -> Result<()> {
        let rec = self
            .snaps
            .remove(snap)
            .ok_or_else(|| FerriteError::Pool("release_snapshot: stale id".into()))?;
        match rec.tier {
            Tier::Device => {
                self.device_snaps -= 1;
                self.store.free_device_slot(rec.handle);
                let mut freed = Vec::new();
                self.pages.release(&rec.dsa_pages, &mut freed);
                for p in freed {
                    self.store.free_page(p);
                }
            }
            Tier::Host => {
                self.host_snaps -= 1;
                self.host_lru.remove(snap);
                self.store.free_host_slot(rec.handle);
                for &hp in &rec.dsa_pages {
                    self.store.free_host_page(hp);
                }
            }
            Tier::Disk => {
                self.disk_snaps -= 1;
                self.disk_lru.remove(snap);
                self.store.free_disk_slot(rec.disk_handle);
                for &dp in &rec.disk_pages {
                    self.store.free_disk_page(dp);
                }
            }
        }
        Ok(())
    }

    /// Snapshot's tier (diagnostics + radix eviction policy input).
    pub fn snapshot_tier(&self, snap: StateId) -> Option<Tier> {
        self.snaps.get(snap).map(|r| r.tier)
    }

    /// Device-tier LRU victim for demotion: the radix tree picks *which*
    /// prefix (tree LRU = SGLang parity); among device snapshots this is
    /// the fallback for pressure-driven demotion (tree-blind, oldest
    /// insertion — pressure demotion of un-cached slots should not
    /// happen in practice; the tree always evicts first).
    fn device_lru_victim(&mut self) -> Result<Option<StateId>> {
        // Rows are never demotable (they are live compute); snapshots only.
        let victim = self
            .snaps
            .iter()
            .filter(|(_, r)| matches!(r.domain, SlotDomain::Snapshot { tier: Tier::Device }))
            .map(|(id, _)| id)
            .min_by_key(|id| id.gen()); // stable: oldest insertion gen
        Ok(victim)
    }

    // -- queries ------------------------------------------------------------

    pub fn live_snapshots(&self) -> usize {
        self.snaps.len()
    }

    /// Device-tier snapshot pressure (demote-pressure metric).
    pub fn device_snapshot_pressure(&self) -> bool {
        self.device_snaps >= self.max_device_snaps
    }

    /// Host-tier snapshot pressure (second demotion threshold).
    pub fn host_snapshot_pressure(&self) -> bool {
        self.host_snaps >= self.max_host_snaps
    }

    pub fn live_rows(&self) -> u32 {
        self.rows.iter().filter(|r| r.is_some()).count() as u32
    }

    pub fn free_rows(&self) -> impl Iterator<Item = u32> + '_ {
        self.rows
            .iter()
            .enumerate()
            .filter(|(_, r)| r.is_none())
            .map(|(i, _)| i as u32)
    }

    pub fn row_tokens(&self, row: u32) -> Result<usize> {
        self.rows
            .get(row as usize)
            .and_then(|r| r.as_ref())
            .map(|r| r.committed_tokens)
            .ok_or_else(|| FerriteError::Pool(format!("row_tokens: row {row} not acquired")))
    }

    /// Tier census (diagnostics: hicache occupancy).
    pub fn tier_census(&self) -> (usize, usize, usize) {
        (self.device_snaps, self.host_snaps, self.disk_snaps)
    }

    /// Borrow the backing store (exec backend device pointer plumbing).
    pub fn store(&mut self) -> &mut S {
        &mut self.store
    }
}
