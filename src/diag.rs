use std::collections::HashSet;
use std::fmt;
use std::panic::Location;
use std::sync::{Arc, Mutex};

/// What kind of memory operation an access was.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AccessKind {
    Read,
    Write,
    /// Atomic read-modify-write. Atomics never race with each other,
    /// but do race with concurrent plain reads and writes.
    Atomic,
}

/// One recorded memory access.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Access {
    /// Linear block index.
    pub block: u32,
    /// Linear thread index within the block.
    pub thread: u32,
    /// Barrier generation of the block when the access happened.
    pub epoch: u64,
    pub kind: AccessKind,
    /// Source location of the access in the kernel.
    pub location: &'static Location<'static>,
}

/// Which memory an access touched.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum MemSpace {
    Global { buffer: Arc<str> },
    Shared { block: u32, name: &'static str },
}

impl fmt::Display for MemSpace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MemSpace::Global { buffer } => write!(f, "global `{buffer}`"),
            MemSpace::Shared { block, name } => write!(f, "shared `{name}` (block {block})"),
        }
    }
}

/// A problem Riri found while executing a kernel.
#[derive(Clone, Debug, PartialEq)]
pub enum Diagnostic {
    DataRace {
        space: MemSpace,
        index: usize,
        first: Access,
        second: Access,
    },
    UninitRead {
        space: MemSpace,
        index: usize,
        access: Access,
    },
    OutOfBounds {
        space: MemSpace,
        index: usize,
        len: usize,
        access: Access,
    },
    BarrierDivergence {
        block: u32,
        /// Threads of the block waiting at the barrier.
        waiting: usize,
        /// Threads of the block that exited without reaching it.
        exited: usize,
        /// Where the waiting threads are blocked.
        barrier: &'static Location<'static>,
    },
    KernelPanic {
        block: u32,
        thread: u32,
        message: String,
    },
}

fn who(a: &Access) -> String {
    format!(
        "block {} thread {} ({:?} at {}:{})",
        a.block,
        a.thread,
        a.kind,
        a.location.file(),
        a.location.line()
    )
}

impl fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Diagnostic::DataRace { space, index, first, second } => write!(
                f,
                "data race on {space}[{index}]: {} conflicts with {} with no barrier between them",
                who(first),
                who(second)
            ),
            Diagnostic::UninitRead { space, index, access } => write!(
                f,
                "read of uninitialised {space}[{index}] by {}",
                who(access)
            ),
            Diagnostic::OutOfBounds { space, index, len, access } => write!(
                f,
                "out-of-bounds access {space}[{index}] (len {len}) by {}",
                who(access)
            ),
            Diagnostic::BarrierDivergence { block, waiting, exited, barrier } => write!(
                f,
                "barrier divergence in block {block}: {waiting} thread(s) wait at {}:{} but {exited} thread(s) exited without reaching it",
                barrier.file(),
                barrier.line()
            ),
            Diagnostic::KernelPanic { block, thread, message } => {
                write!(f, "kernel panic in block {block} thread {thread}: {message}")
            }
        }
    }
}

/// The result of one kernel launch under Riri.
#[derive(Clone, Debug)]
pub struct Report {
    /// Scheduler seed; re-run with the same seed to reproduce the schedule.
    pub seed: u64,
    pub diagnostics: Vec<Diagnostic>,
    /// True if the launch was cut short by a trap-like error.
    pub aborted: bool,
}

impl Report {
    pub fn is_clean(&self) -> bool {
        self.diagnostics.is_empty()
    }

    pub fn has_race(&self) -> bool {
        self.diagnostics.iter().any(|d| matches!(d, Diagnostic::DataRace { .. }))
    }

    /// Panics with a readable listing if any diagnostic was reported.
    #[track_caller]
    pub fn assert_clean(&self) {
        if !self.is_clean() {
            panic!("{self}");
        }
    }
}

impl fmt::Display for Report {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "riri: {} diagnostic(s) (seed {}{})",
            self.diagnostics.len(),
            self.seed,
            if self.aborted { ", launch aborted" } else { "" }
        )?;
        for d in &self.diagnostics {
            writeln!(f, "  - {d}")?;
        }
        Ok(())
    }
}

const MAX_DIAGNOSTICS: usize = 64;

type DedupKey = (String, &'static Location<'static>, Option<&'static Location<'static>>);

/// Collects diagnostics, de-duplicating by kind and source location so a
/// racy line in a 1024-thread kernel produces one report, not a thousand.
#[derive(Default)]
pub(crate) struct Reporter {
    inner: Mutex<(Vec<Diagnostic>, HashSet<DedupKey>)>,
}

impl Reporter {
    pub(crate) fn push(&self, d: Diagnostic) {
        let key: DedupKey = match &d {
            Diagnostic::DataRace { first, second, .. } => {
                let (a, b) = if first.location <= second.location {
                    (first.location, second.location)
                } else {
                    (second.location, first.location)
                };
                ("race".into(), a, Some(b))
            }
            Diagnostic::UninitRead { access, .. } => ("uninit".into(), access.location, None),
            Diagnostic::OutOfBounds { access, .. } => ("oob".into(), access.location, None),
            Diagnostic::BarrierDivergence { block, barrier, .. } => {
                (format!("barrier{block}"), barrier, None)
            }
            Diagnostic::KernelPanic { message, .. } => {
                (format!("panic:{message}"), Location::caller(), None)
            }
        };
        let mut g = self.inner.lock().unwrap();
        if g.0.len() < MAX_DIAGNOSTICS && g.1.insert(key) {
            g.0.push(d);
        }
    }

    pub(crate) fn take(&self) -> Vec<Diagnostic> {
        std::mem::take(&mut self.inner.lock().unwrap().0)
    }
}
