//! Shadow memory: per-element access history used for race detection.
//!
//! Happens-before model (v0.1):
//! - Within a block, `sync_threads()` orders everything before it against
//!   everything after it. Each barrier bumps the block's *epoch*.
//! - Across blocks there is no ordering at all within one launch.
//! - Two accesses from the same thread are always ordered.
//! - Two atomics never race with each other.
//!
//! So two accesses are *concurrent* iff they come from different threads and
//! either from different blocks or from the same block in the same epoch.

use crate::diag::{Access, AccessKind};

/// Bound on remembered readers per element. Beyond this, some races between a
/// write and an old reader may be missed; this keeps memory bounded.
const MAX_READERS: usize = 8;

pub(crate) fn concurrent(a: &Access, b: &Access) -> bool {
    let same_thread = a.block == b.block && a.thread == b.thread;
    let both_atomic = a.kind == AccessKind::Atomic && b.kind == AccessKind::Atomic;
    !same_thread && !both_atomic && (a.block != b.block || a.epoch == b.epoch)
}

#[derive(Clone, Default)]
pub(crate) struct Cell {
    pub(crate) init: bool,
    last_write: Option<Access>,
    readers: Vec<Access>,
}

impl Cell {
    pub(crate) fn initialised() -> Self {
        Cell { init: true, ..Default::default() }
    }

    /// Records a read; returns a concurrent earlier write, if any.
    pub(crate) fn on_read(&mut self, acc: Access) -> Option<Access> {
        let conflict = self.last_write.filter(|w| concurrent(w, &acc));
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
    pub(crate) fn on_write(&mut self, acc: Access) -> Option<Access> {
        let conflict = self
            .last_write
            .filter(|w| concurrent(w, &acc))
            .or_else(|| self.readers.iter().copied().find(|r| concurrent(r, &acc)));
        self.last_write = Some(acc);
        // Concurrent readers were just reported; ordered ones are dead.
        self.readers.clear();
        self.init = true;
        conflict
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::panic::Location;

    fn acc(block: u32, thread: u32, epoch: u64, kind: AccessKind) -> Access {
        Access { block, thread, epoch, kind, location: Location::caller() }
    }

    #[test]
    fn same_epoch_different_threads_race() {
        let mut c = Cell::initialised();
        assert!(c.on_write(acc(0, 0, 0, AccessKind::Write)).is_none());
        assert!(c.on_read(acc(0, 1, 0, AccessKind::Read)).is_some());
    }

    #[test]
    fn barrier_orders_accesses() {
        let mut c = Cell::initialised();
        c.on_write(acc(0, 0, 0, AccessKind::Write));
        assert!(c.on_read(acc(0, 1, 1, AccessKind::Read)).is_none());
    }

    #[test]
    fn blocks_are_never_ordered() {
        let mut c = Cell::initialised();
        c.on_write(acc(0, 0, 0, AccessKind::Write));
        assert!(c.on_write(acc(1, 0, 5, AccessKind::Write)).is_some());
    }

    #[test]
    fn atomics_do_not_race_each_other() {
        let mut c = Cell::initialised();
        c.on_write(acc(0, 0, 0, AccessKind::Atomic));
        assert!(c.on_write(acc(1, 3, 0, AccessKind::Atomic)).is_none());
        assert!(c.on_read(acc(0, 7, 0, AccessKind::Read)).is_some());
    }

    #[test]
    fn write_after_concurrent_read_races() {
        let mut c = Cell::initialised();
        c.on_read(acc(0, 0, 0, AccessKind::Read));
        assert!(c.on_write(acc(0, 1, 0, AccessKind::Write)).is_some());
    }
}
