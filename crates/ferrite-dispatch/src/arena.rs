//! Typed generational arena — the identity backbone of the dispatcher.
//!
//! Every long-lived object the scheduler touches (sequences, radix nodes,
//! state slots) lives in one arena per family and is addressed by a
//! [`TypedId`] handle carrying (index, generation). Freed indices are reused
//! with a bumped generation, so a stale handle that survives one logical
//! deletion becomes a hard lookup miss instead of a silent wrong-object
//! access (the classic ABA hazard of slot maps in schedulers).
//!
//! Design notes:
//! - Handles are `Copy`, 8 bytes, ordered — they double as sort keys and
//!   can be packed into device-visible row indices.
//! - `TypedId<T>` is zero-cost over `u32` (niche: no `Option<T>` overhead —
//!   `Option<TypedId<T>>` is 8 bytes).
//! - Each family is a distinct phantom-typed tag: a `NodeId` can never be
//!   used to index the sequence arena (compile error), which is the kind of
//!   mix-up schedulers die from at 3 a.m.

use std::marker::PhantomData;

/// Marker trait for arena families. Implemented by the tag types
/// ([`crate::SeqTag`], [`crate::NodeTag`], [`crate::SlotTag`]).
pub trait ArenaFamily: Copy + 'static {
    /// Human-readable family name for diagnostics.
    const NAME: &'static str;
}

/// Typed, generational handle into one [`TypedArena`] family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TypedId<F: ArenaFamily> {
    /// Slot index inside the arena (dense while entries live).
    idx: u32,
    /// Generation bump on free; lookups require a match.
    gen: u32,
    _fam: PhantomData<F>,
}

impl<F: ArenaFamily> TypedId<F> {
    #[inline]
    pub fn idx(self) -> usize {
        self.idx as usize
    }

    #[inline]
    pub fn gen(self) -> u32 {
        self.gen
    }

    #[inline]
    fn new(idx: u32, gen: u32) -> Self {
        Self { idx, gen, _fam: PhantomData }
    }

    /// Reserved handle for self-referential sentinels inserted as the
    /// arena's *first* element (e.g. the radix root, whose parent field
    /// points to itself). Generation 0 / index 0 is exactly the first
    /// insert in a fresh arena — the only sanctioned use.
    pub(crate) fn new_unchecked(idx: u32) -> Self {
        Self { idx, gen: 0, _fam: PhantomData }
    }
}

/// Dense-slot arena with generation counters and O(1) free-list reuse.
///
/// Lookups (`get`/`get_mut`/`remove`) verify the generation — an id from a
/// removed entry resolves to `None` rather than aliasing its successor.
pub struct TypedArena<F: ArenaFamily, T> {
    slots: Vec<Option<T>>,
    gens: Vec<u32>,
    free: Vec<u32>,
    live: usize,
    _fam: PhantomData<F>,
}

impl<F: ArenaFamily, T> TypedArena<F, T> {
    pub fn with_capacity(cap: usize) -> Self {
        let mut a = TypedArena {
            slots: Vec::with_capacity(cap),
            gens: Vec::with_capacity(cap),
            free: Vec::new(),
            live: 0,
            _fam: PhantomData,
        };
        a.slots.resize_with(cap, || None);
        a.gens.resize(cap, 0);
        a
    }

    pub fn insert(&mut self, value: T) -> TypedId<F> {
        self.live += 1;
        match self.free.pop() {
            Some(idx) => {
                // generation already bumped by `remove`
                let gen = self.gens[idx as usize];
                self.slots[idx as usize] = Some(value);
                TypedId::new(idx, gen)
            }
            None => {
                let idx = self.slots.len() as u32;
                self.slots.push(Some(value));
                self.gens.push(0);
                TypedId::new(idx, 0)
            }
        }
    }

    #[inline]
    pub fn get(&self, id: TypedId<F>) -> Option<&T> {
        self.slots
            .get(id.idx as usize)?
            .as_ref()
            .filter(|_| self.gens[id.idx as usize] == id.gen)
    }

    #[inline]
    pub fn get_mut(&mut self, id: TypedId<F>) -> Option<&mut T> {
        let gen_ok = self.gens.get(id.idx as usize) == Some(&id.gen);
        if !gen_ok {
            return None;
        }
        self.slots[id.idx as usize].as_mut()
    }

    /// Remove by live handle. Returns `None` (and does nothing) on stale ids.
    pub fn remove(&mut self, id: TypedId<F>) -> Option<T> {
        let slot = self.slots.get_mut(id.idx as usize)?;
        if self.gens[id.idx as usize] != id.gen || slot.is_none() {
            return None;
        }
        self.live -= 1;
        let value = slot.take();
        self.gens[id.idx as usize] = self.gens[id.idx as usize].wrapping_add(1);
        self.free.push(id.idx);
        value
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.live
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.live == 0
    }

    /// Iterate live (id, &T) pairs in index order — deterministic, cache
    /// friendly. The preferred iteration shape for scheduler ticks (stable
    /// order keeps pad-to-B bucket assignment reproducible run to run).
    pub fn iter(&self) -> impl Iterator<Item = (TypedId<F>, &T)> {
        self.slots
            .iter()
            .zip(self.gens.iter())
            .enumerate()
            .filter_map(|(i, (slot, gen))| slot.as_ref().map(|v| (TypedId::new(i as u32, *gen), v)))
    }

    /// Iterate live handles to mutable values, index order.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (TypedId<F>, &mut T)> {
        self.slots
            .iter_mut()
            .zip(self.gens.iter())
            .enumerate()
            .filter_map(|(i, (slot, gen))| {
                slot.as_mut()
                    .map(|v| (TypedId::new(i as u32, *gen), v))
            })
    }
}

// -- family tags -----------------------------------------------------------

/// Arena family: scheduler sequences (requests).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SeqTag;
impl ArenaFamily for SeqTag {
    const NAME: &'static str = "seq";
}

/// Arena family: radix-tree cache nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NodeTag;
impl ArenaFamily for NodeTag {
    const NAME: &'static str = "radix-node";
}

/// Arena family: physical state slots (decode rows / radix snapshots).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlotTag;
impl ArenaFamily for SlotTag {
    const NAME: &'static str = "state-slot";
}

pub type SeqId = TypedId<SeqTag>;
pub type NodeId = TypedId<NodeTag>;
pub type SlotId = TypedId<SlotTag>;

/// Shorthand constructors used across the crate.
pub fn seq_arena(cap: usize) -> TypedArena<SeqTag, crate::batch::Seq> {
    TypedArena::with_capacity(cap)
}
