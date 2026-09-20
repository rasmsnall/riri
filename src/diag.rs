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
    /// Warp index within the block.
    pub warp: u32,
    /// Collective generation of the warp when the access happened.
    pub warp_epoch: u64,
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

/// Ways a lane can name the wrong threads in a warp collective.
///
/// These are undefined behaviour in CUDA rather than merely surprising, so
/// Riri treats the first two as fatal and ends the launch.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LaneProblem {
    /// The calling lane left itself out of its own member mask.
    CallerNotInMask,
    /// The mask names lanes that do not exist, because the block does not
    /// fill this warp. `valid` is the mask of lanes that do exist.
    MaskOutsideBlock { valid: u32 },
    /// A shuffle read from a lane that is not a member of the mask, so the
    /// value it would return was never contributed.
    SourceLaneNotInMask { src: u32 },
}

impl fmt::Display for LaneProblem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LaneProblem::CallerNotInMask => {
                write!(f, "the calling lane is not a member of its own mask")
            }
            LaneProblem::MaskOutsideBlock { valid } => write!(
                f,
                "the mask names lanes outside the block (lanes {} exist)",
                lane_list(*valid)
            ),
            LaneProblem::SourceLaneNotInMask { src } => {
                write!(f, "source lane {src} is not a member of the mask")
            }
        }
    }
}

/// Renders a lane bitmask as a compact list, e.g. `0-15,31`.
pub(crate) fn lane_list(mask: u32) -> String {
    if mask == 0 {
        return "none".to_string();
    }
    let mut parts: Vec<String> = Vec::new();
    let mut lane = 0u32;
    while lane < 32 {
        if mask & (1u32 << lane) != 0 {
            let start = lane;
            while lane + 1 < 32 && mask & (1u32 << (lane + 1)) != 0 {
                lane += 1;
            }
            if start == lane {
                parts.push(format!("{start}"));
            } else {
                parts.push(format!("{start}-{lane}"));
            }
        }
        lane += 1;
    }
    parts.join(",")
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
    /// A warp collective was reached by only some of the lanes its mask
    /// names. The missing lanes branched elsewhere, exited, or stopped at a
    /// different collective, so the warp can never reconverge here.
    WarpDivergence {
        block: u32,
        /// Warp index within the block.
        warp: u32,
        /// Name of the collective, e.g. `shfl_down_sync`.
        op: &'static str,
        /// Member mask the waiting lanes passed.
        mask: u32,
        /// Lanes that actually reached this collective.
        arrived: u32,
        /// Where the waiting lanes are blocked.
        at: &'static Location<'static>,
    },
    /// Two lanes reached the same collective with different member masks.
    /// Every participant must agree on who is taking part.
    WarpMaskMismatch {
        block: u32,
        warp: u32,
        op: &'static str,
        at: &'static Location<'static>,
        lane: u32,
        mask: u32,
        other_lane: u32,
        other_mask: u32,
    },
    /// A lane named a set of threads that cannot be honoured.
    WarpLaneError {
        block: u32,
        warp: u32,
        op: &'static str,
        at: &'static Location<'static>,
        lane: u32,
        mask: u32,
        problem: LaneProblem,
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
            Diagnostic::UninitRead { space, index, access } => {
                write!(f, "read of uninitialised {space}[{index}] by {}", who(access))
            }
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
            Diagnostic::WarpDivergence { block, warp, op, mask, arrived, at } => write!(
                f,
                "warp divergence in block {block} warp {warp}: `{op}` at {}:{} names lanes {} but only lanes {} arrived, so lanes {} never reach it",
                at.file(),
                at.line(),
                lane_list(*mask),
                lane_list(*arrived),
                lane_list(mask & !arrived)
            ),
            Diagnostic::WarpMaskMismatch {
                block,
                warp,
                op,
                at,
                lane,
                mask,
                other_lane,
                other_mask,
            } => write!(
                f,
                "warp mask mismatch in block {block} warp {warp}: `{op}` at {}:{} was called by lane {lane} with lanes {} but by lane {other_lane} with lanes {}",
                at.file(),
                at.line(),
                lane_list(*mask),
                lane_list(*other_mask)
            ),
            Diagnostic::WarpLaneError { block, warp, op, at, lane, mask, problem } => write!(
                f,
                "invalid warp collective in block {block} warp {warp}: lane {lane} called `{op}` at {}:{} with lanes {} but {problem}",
                at.file(),
                at.line(),
                lane_list(*mask)
            ),
            Diagnostic::KernelPanic { block, thread, message } => {
                write!(f, "kernel panic in block {block} thread {thread}: {message}")
            }
        }
    }
}

fn at(loc: &Location<'static>) -> String {
    format!("{}:{}", loc.file(), loc.line())
}

impl LaneProblem {
    /// The variant name, without the fields that vary by schedule.
    pub fn kind(&self) -> &'static str {
        match self {
            LaneProblem::CallerNotInMask => "caller-not-in-mask",
            LaneProblem::MaskOutsideBlock { .. } => "mask-outside-block",
            LaneProblem::SourceLaneNotInMask { .. } => "source-lane-not-in-mask",
        }
    }
}

impl Diagnostic {
    /// A stable identity for this finding: its kind and the source locations
    /// involved, but not the block, thread, or index that happened to hit it.
    ///
    /// Two runs that find the same bug by different interleavings agree here,
    /// which is what lets a schedule be shrunk while checking that it still
    /// reproduces the *same* problem rather than some other one.
    pub fn fingerprint(&self) -> String {
        match self {
            Diagnostic::DataRace { first, second, .. } => {
                let (a, b) = if first.location <= second.location {
                    (first.location, second.location)
                } else {
                    (second.location, first.location)
                };
                format!("race@{}~{}", at(a), at(b))
            }
            Diagnostic::UninitRead { access, .. } => format!("uninit@{}", at(access.location)),
            Diagnostic::OutOfBounds { access, .. } => format!("oob@{}", at(access.location)),
            Diagnostic::BarrierDivergence { barrier, .. } => format!("barrier@{}", at(barrier)),
            Diagnostic::WarpDivergence { op, at: loc, .. } => {
                format!("warp-divergence:{op}@{}", at(loc))
            }
            Diagnostic::WarpMaskMismatch { op, at: loc, .. } => {
                format!("warp-mask-mismatch:{op}@{}", at(loc))
            }
            Diagnostic::WarpLaneError { op, at: loc, problem, .. } => {
                format!("warp-lane-error:{op}:{}@{}", problem.kind(), at(loc))
            }
            Diagnostic::KernelPanic { message, .. } => format!("panic:{message}"),
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

    /// True if any lane reached a warp collective that its warp-mates did not.
    pub fn has_warp_divergence(&self) -> bool {
        self.diagnostics.iter().any(|d| matches!(d, Diagnostic::WarpDivergence { .. }))
    }

    /// Every finding's [`Diagnostic::fingerprint`], sorted and de-duplicated.
    pub fn fingerprints(&self) -> Vec<String> {
        let mut f: Vec<String> = self.diagnostics.iter().map(Diagnostic::fingerprint).collect();
        f.sort();
        f.dedup();
        f
    }

    /// True if this report contains a finding with the given fingerprint.
    pub fn contains(&self, fingerprint: &str) -> bool {
        self.diagnostics.iter().any(|d| d.fingerprint() == fingerprint)
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
            Diagnostic::WarpDivergence { block, warp, at, .. } => {
                (format!("warpdiv{block}:{warp}"), at, None)
            }
            Diagnostic::WarpMaskMismatch { block, warp, at, .. } => {
                (format!("warpmask{block}:{warp}"), at, None)
            }
            Diagnostic::WarpLaneError { problem, at, .. } => {
                (format!("warplane{problem:?}"), at, None)
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
