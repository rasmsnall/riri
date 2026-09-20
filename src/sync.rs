//! Release and acquire ordering between blocks.
//!
//! Barriers order a block and warp collectives order a warp, but neither says
//! anything across blocks, so until now Riri treated every pair of accesses
//! from different blocks as concurrent. That is right for kernels that do not
//! synchronise, and wrong for the ones that do: the message-passing pattern
//! below is correct CUDA, and Riri used to report it as a race.
//!
//! ```text
//! producer                         consumer
//!   data[i] = value                  while flag == 0 {}     (relaxed loads)
//!   threadfence()                    threadfence()
//!   flag = 1        (relaxed)        read data[i]
//! ```
//!
//! What carries the ordering is a *vector clock* per thread: for each other
//! thread, the highest operation of it that this thread is known to have seen.
//! A release publishes the releasing thread's clock to a location, an acquire
//! joins that location's clock into the acquiring thread's own, and an access
//! by one thread happens-before an access by another exactly when the second
//! thread's clock has caught up with the first access.
//!
//! Clocks are sparse, holding entries only for threads actually synchronised
//! with, so a kernel that never fences pays for a few empty maps rather than
//! for a matrix over its threads.

use std::collections::BTreeMap;

/// How an atomic access or a fence orders the operations around it.
///
/// This is the subset of the C++ model that CUDA kernels reach for, spelled
/// the same way. `SeqCst` is accepted and treated as `AcqRel`: Riri models
/// synchronisation between threads, not a single total order over all atomics,
/// and nothing it checks can tell the two apart.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Ordering {
    /// No ordering, only atomicity.
    Relaxed,
    /// Later accesses by this thread cannot be seen to happen before this one.
    Acquire,
    /// Earlier accesses by this thread are published by this one.
    Release,
    /// Both.
    AcqRel,
    /// Treated as [`Ordering::AcqRel`].
    SeqCst,
}

impl Ordering {
    pub(crate) fn is_acquire(self) -> bool {
        matches!(
            self,
            Ordering::Acquire | Ordering::AcqRel | Ordering::SeqCst
        )
    }

    pub(crate) fn is_release(self) -> bool {
        matches!(
            self,
            Ordering::Release | Ordering::AcqRel | Ordering::SeqCst
        )
    }
}

/// Identifies a thread within a launch, as a packed block and thread index.
pub(crate) type Key = u64;

pub(crate) fn key(block: u32, thread: u32) -> Key {
    ((block as u64) << 32) | thread as u64
}

/// For each thread, the highest operation of it that the owner has seen.
///
/// Absent entries read as zero, and operation counters start at one, so a
/// clock that has never synchronised orders nothing. That is what keeps the
/// old behaviour for kernels that do not use fences.
#[derive(Clone, Default, Debug)]
pub(crate) struct Clock(BTreeMap<Key, u64>);

impl Clock {
    pub(crate) fn get(&self, k: Key) -> u64 {
        self.0.get(&k).copied().unwrap_or(0)
    }

    pub(crate) fn observe(&mut self, k: Key, seq: u64) {
        let slot = self.0.entry(k).or_insert(0);
        if seq > *slot {
            *slot = seq;
        }
    }

    /// Takes the later of each entry, which is how knowledge is transferred.
    pub(crate) fn join(&mut self, other: &Clock) {
        for (&k, &seq) in &other.0 {
            self.observe(k, seq);
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

/// Per-thread clocks for a launch.
pub(crate) struct SyncState {
    /// Operation counter per thread, bumped by every instrumented access.
    seq: Vec<u64>,
    /// What each thread has seen.
    clock: Vec<Clock>,
    /// Snapshot taken by a release fence, published by the next relaxed store.
    fence_release: Vec<Clock>,
    /// Gathered by relaxed loads, joined in by the next acquire fence.
    fence_acquire: Vec<Clock>,
    /// What each block collectively knows, accumulated at its barriers.
    block: Vec<Clock>,
}

impl SyncState {
    pub(crate) fn new(threads: usize, blocks: usize) -> Self {
        SyncState {
            seq: vec![0; threads],
            clock: vec![Clock::default(); threads],
            fence_release: vec![Clock::default(); threads],
            fence_acquire: vec![Clock::default(); threads],
            block: vec![Clock::default(); blocks],
        }
    }

    /// Contributes this thread's knowledge to its block, on the way into a
    /// barrier.
    ///
    /// Without this a thread that raises a flag after a barrier would publish
    /// only its own writes, and the block-mates' writes the barrier just
    /// ordered would look unpublished. That composition, a barrier then a
    /// release, is the whole point of a fence in a multi-thread block.
    pub(crate) fn barrier_arrive(&mut self, gid: usize, block: usize) {
        let mine = self.clock[gid].clone();
        self.block[block].join(&mine);
    }

    /// Takes on what the whole block knew, on the way out of a barrier.
    ///
    /// Every thread has contributed by the time the barrier releases, so the
    /// block clock is complete when this runs.
    pub(crate) fn barrier_depart(&mut self, gid: usize, block: usize) {
        let shared = self.block[block].clone();
        self.clock[gid].join(&shared);
    }

    /// Counts one operation for this thread and returns its number.
    pub(crate) fn tick(&mut self, gid: usize, k: Key) -> u64 {
        self.seq[gid] += 1;
        let seq = self.seq[gid];
        self.clock[gid].observe(k, seq);
        seq
    }

    pub(crate) fn clock(&self, gid: usize) -> &Clock {
        &self.clock[gid]
    }

    /// A release: publish what this thread has done into `target`.
    ///
    /// A release ordering publishes the thread's own clock. A relaxed store
    /// publishes whatever a preceding release fence left behind, which is how
    /// `threadfence()` followed by a plain atomic write becomes a release.
    pub(crate) fn release(&mut self, gid: usize, target: &mut Clock, ordering: Ordering) {
        if ordering.is_release() {
            let mine = self.clock[gid].clone();
            target.join(&mine);
        }
        if !self.fence_release[gid].is_empty() {
            let fenced = self.fence_release[gid].clone();
            target.join(&fenced);
        }
    }

    /// An acquire: take on what `source` published.
    ///
    /// A relaxed load does not order anything by itself, but a later acquire
    /// fence makes it count, so what it read is remembered until then.
    pub(crate) fn acquire(&mut self, gid: usize, source: &Clock, ordering: Ordering) {
        if source.is_empty() {
            return;
        }
        if ordering.is_acquire() {
            self.clock[gid].join(source);
        } else {
            self.fence_acquire[gid].join(source);
        }
    }

    /// `__threadfence()` and its one-sided forms.
    pub(crate) fn fence(&mut self, gid: usize, ordering: Ordering) {
        if ordering.is_acquire() {
            let pending = std::mem::take(&mut self.fence_acquire[gid]);
            self.clock[gid].join(&pending);
        }
        if ordering.is_release() {
            self.fence_release[gid] = self.clock[gid].clone();
        }
    }
}
