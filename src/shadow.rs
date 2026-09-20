//! Shadow memory: per-element access history used for race detection.
//!
//! Happens-before model (v0.2):
//! - Within a block, `sync_threads()` orders everything before it against
//!   everything after it. Each barrier bumps the block's *epoch*.
//! - Within a warp, a full-mask warp collective (`sync_warp`, a shuffle, a
//!   ballot) does the same for that warp alone, bumping its *warp epoch*.
//!   A partial mask bumps nothing, since it says nothing about the lanes it
//!   leaves out.
//! - Across blocks there is no ordering at all within one launch.
//! - Two accesses from the same thread are always ordered.
//! - Two atomics never race with each other.
//!
//! So two accesses are *concurrent* iff they come from different threads and
//! either from different blocks, or from the same block in the same epoch
//! and either in different warps or in the same warp epoch.

use crate::diag::{Access, AccessKind};
use crate::sync::{key, Clock};

/// Bound on remembered readers per element. Beyond this, some races between a
/// write and an old reader may be missed; this keeps memory bounded.
const MAX_READERS: usize = 8;

/// Whether two accesses are unordered, where `b` is the one happening now
/// and `now` is the clock of the thread making it.
///
/// `b` cannot happen-before `a`, since `a` was recorded first, so only one
/// direction needs testing.
pub(crate) fn concurrent(a: &Access, b: &Access, now: &Clock) -> bool {
    if a.block == b.block && a.thread == b.thread {
        return false;
    }
    if a.kind == AccessKind::Atomic && b.kind == AccessKind::Atomic {
        return false;
    }
    // Release and acquire ordering, at any scope. A thread that has never
    // synchronised has an empty clock, which orders nothing, so kernels that
    // do not fence behave exactly as they did before.
    if now.get(key(a.block, a.thread)) >= a.seq {
        return false;
    }
    if a.block != b.block {
        return true;
    }
    if a.epoch != b.epoch {
        return false;
    }
    if a.warp != b.warp {
        return true;
    }
    a.warp_epoch == b.warp_epoch
}

#[derive(Clone, Default)]
pub(crate) struct Cell {
    pub(crate) init: bool,
    last_write: Option<Access>,
    readers: Vec<Access>,
    /// What a release to this element has published, for an acquire to pick
    /// up. Only atomic accesses touch it.
    pub(crate) release: Clock,
}

impl Cell {
    pub(crate) fn initialised() -> Self {
        Cell {
            init: true,
            ..Default::default()
        }
    }

    /// Records a read; returns a concurrent earlier write, if any.
    pub(crate) fn on_read(&mut self, acc: Access, now: &Clock) -> Option<Access> {
        let conflict = self.last_write.filter(|w| concurrent(w, &acc, now));
        // Readers from this block in older epochs happen-before everything
        // now, so they can be forgotten. Cross-block readers must be kept.
        self.readers
            .retain(|r| r.block != acc.block || r.epoch == acc.epoch);
        let dup = self
            .readers
            .iter()
            .any(|r| r.block == acc.block && r.thread == acc.thread && r.epoch == acc.epoch);
        if !dup && self.readers.len() < MAX_READERS {
            self.readers.push(acc);
        }
        conflict
    }

    /// Records a write or atomic; returns a concurrent earlier access, if any.
    pub(crate) fn on_write(&mut self, acc: Access, now: &Clock) -> Option<Access> {
        let conflict = self
            .last_write
            .filter(|w| concurrent(w, &acc, now))
            .or_else(|| {
                self.readers
                    .iter()
                    .copied()
                    .find(|r| concurrent(r, &acc, now))
            });
        self.last_write = Some(acc);
        // Concurrent readers were just reported; ordered ones are dead.
        self.readers.clear();
        self.init = true;
        conflict
    }
}

#[cfg(test)]
impl Cell {
    /// `on_read` for a thread that has synchronised with nobody.
    fn on_read_t(&mut self, acc: Access) -> Option<Access> {
        self.on_read(acc, &Clock::default())
    }

    /// `on_write` for a thread that has synchronised with nobody.
    fn on_write_t(&mut self, acc: Access) -> Option<Access> {
        self.on_write(acc, &Clock::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::panic::Location;

    /// An access in warp 0 at warp epoch 0, for the block-level tests.
    fn acc(block: u32, thread: u32, epoch: u64, kind: AccessKind) -> Access {
        warp_acc(block, thread, epoch, 0, 0, kind)
    }

    fn warp_acc(
        block: u32,
        thread: u32,
        epoch: u64,
        warp: u32,
        warp_epoch: u64,
        kind: AccessKind,
    ) -> Access {
        Access {
            block,
            thread,
            epoch,
            warp,
            warp_epoch,
            seq: 1,
            kind,
            location: Location::caller(),
        }
    }

    #[test]
    fn same_epoch_different_threads_race() {
        let mut c = Cell::initialised();
        assert!(c.on_write_t(acc(0, 0, 0, AccessKind::Write)).is_none());
        assert!(c.on_read_t(acc(0, 1, 0, AccessKind::Read)).is_some());
    }

    #[test]
    fn barrier_orders_accesses() {
        let mut c = Cell::initialised();
        c.on_write_t(acc(0, 0, 0, AccessKind::Write));
        assert!(c.on_read_t(acc(0, 1, 1, AccessKind::Read)).is_none());
    }

    #[test]
    fn blocks_are_never_ordered() {
        let mut c = Cell::initialised();
        c.on_write_t(acc(0, 0, 0, AccessKind::Write));
        assert!(c.on_write_t(acc(1, 0, 5, AccessKind::Write)).is_some());
    }

    #[test]
    fn atomics_do_not_race_each_other() {
        let mut c = Cell::initialised();
        c.on_write_t(acc(0, 0, 0, AccessKind::Atomic));
        assert!(c.on_write_t(acc(1, 3, 0, AccessKind::Atomic)).is_none());
        assert!(c.on_read_t(acc(0, 7, 0, AccessKind::Read)).is_some());
    }

    #[test]
    fn write_after_concurrent_read_races() {
        let mut c = Cell::initialised();
        c.on_read_t(acc(0, 0, 0, AccessKind::Read));
        assert!(c.on_write_t(acc(0, 1, 0, AccessKind::Write)).is_some());
    }

    #[test]
    fn warp_collective_orders_accesses_in_that_warp() {
        let mut c = Cell::initialised();
        c.on_write_t(warp_acc(0, 0, 0, 0, 0, AccessKind::Write));
        assert!(c
            .on_read_t(warp_acc(0, 1, 0, 0, 1, AccessKind::Read))
            .is_none());
    }

    #[test]
    fn warp_collective_does_not_order_other_warps() {
        let mut c = Cell::initialised();
        c.on_write_t(warp_acc(0, 0, 0, 0, 1, AccessKind::Write));
        // Warp 1 never took part, so its accesses are still concurrent.
        assert!(c
            .on_read_t(warp_acc(0, 32, 0, 1, 0, AccessKind::Read))
            .is_some());
    }
}
