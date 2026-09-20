//! Searching across schedules, and shrinking a failing one.
//!
//! A single [`launch`] answers "does this seed find a bug". Exploration
//! answers the two questions that follow: *does any schedule find one*, and
//! *what is the shortest schedule that does*.
//!
//! The second matters more than it sounds. A failing seed already reproduces
//! exactly, but the interleaving behind it can be hundreds of decisions long,
//! which is a reproducer you cannot read. Shrinking turns it into a handful of
//! decisions followed by threads running in order, which is a reproducer you
//! can reason about and paste into a test.
//!
//! ```
//! use riri::{explore, GlobalBuf, LaunchConfig};
//!
//! let out = GlobalBuf::new("out", vec![0u32; 1]);
//! let found = explore(&LaunchConfig::new(1, 4), 16, |t| {
//!     out.write(t, 0, t.thread_linear() as u32);
//! });
//!
//! assert!(!found.is_clean());
//! ```
//!
//! [`launch`]: crate::launch

use std::fmt;

use crate::ctx::ThreadCtx;
use crate::diag::Report;
use crate::launch::LaunchConfig;
use crate::sched::{Choices, SplitMix64};

/// A recorded sequence of scheduling decisions, replayable with [`replay`].
///
/// Each entry is the global index of the thread that was given the turn. A
/// plan shorter than the run it drives is not a problem: once it is spent,
/// the scheduler keeps the current thread running while it can, so the tail
/// of the schedule is the least surprising one available.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Schedule {
    plan: Vec<u32>,
}

impl Schedule {
    pub fn new(plan: Vec<u32>) -> Self {
        Schedule { plan }
    }

    /// The decisions, as global thread indices.
    pub fn plan(&self) -> &[u32] {
        &self.plan
    }

    pub fn len(&self) -> usize {
        self.plan.len()
    }

    pub fn is_empty(&self) -> bool {
        self.plan.is_empty()
    }

    /// How many times the turn passes from one thread to a different one.
    ///
    /// This is the number that decides whether a schedule is readable. A
    /// failure needing two switches is a story; one needing ninety is not.
    pub fn switches(&self) -> usize {
        self.plan.windows(2).filter(|w| w[0] != w[1]).count()
    }
}

impl fmt::Display for Schedule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} decision(s), {} switch(es)",
            self.len(),
            self.switches()
        )
    }
}

/// Why a failing schedule is or is not available in shrunk form.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Shrink {
    /// The schedule was shrunk, and replaying it reproduces the finding.
    Minimised(Schedule),
    /// Shrinking was turned off with [`Explore::minimise`].
    Disabled,
    /// The launch made more decisions than Riri records, so the schedule is
    /// incomplete and cannot be replayed. Shrink the launch to shrink the
    /// schedule.
    TraceTruncated,
    /// Replaying the recorded schedule did not reproduce the finding, so the
    /// kernel does not behave the same way twice.
    ///
    /// The usual cause is a buffer the kernel captures and mutates, since
    /// every run then starts from the previous run's output. Use
    /// [`Explore::run_with`] to build fresh state for each run.
    NotReproducible,
}

impl Shrink {
    /// The shrunk schedule, if there is one.
    pub fn schedule(&self) -> Option<&Schedule> {
        match self {
            Shrink::Minimised(s) => Some(s),
            _ => None,
        }
    }
}

impl fmt::Display for Shrink {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Shrink::Minimised(s) => write!(f, "shrunk to {s}"),
            Shrink::Disabled => write!(f, "not shrunk, minimisation disabled"),
            Shrink::TraceTruncated => write!(f, "not shrunk, the schedule was too long to record"),
            Shrink::NotReproducible => {
                write!(
                    f,
                    "not shrunk, the kernel did not behave the same way twice"
                )
            }
        }
    }
}

/// The first failing schedule an exploration found.
#[derive(Clone, Debug)]
pub struct Failure {
    /// The seed that produced it. Re-run it with
    /// `LaunchConfig::seed` to get this exact launch back.
    pub seed: u64,
    pub report: Report,
    pub shrink: Shrink,
    /// How many scheduling decisions the original failing run made, before
    /// any shrinking. The ratio against the shrunk schedule is the whole
    /// point of shrinking.
    pub decisions: usize,
}

impl Failure {
    /// The fingerprint that shrinking preserved: the first finding reported.
    pub fn target(&self) -> Option<String> {
        self.report.diagnostics.first().map(|d| d.fingerprint())
    }
}

/// The result of searching across schedules.
#[derive(Clone, Debug)]
pub struct Exploration {
    pub seeds_tried: u64,
    /// The first seed that found something, if any.
    pub failure: Option<Failure>,
}

impl Exploration {
    pub fn is_clean(&self) -> bool {
        self.failure.is_none()
    }

    /// Panics with the failing seed, its findings, and its shrunk schedule if
    /// any schedule found a problem.
    #[track_caller]
    pub fn assert_clean(&self) {
        if !self.is_clean() {
            panic!("{self}");
        }
    }
}

impl fmt::Display for Exploration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.failure {
            None => writeln!(f, "riri: explored {} seed(s), all clean", self.seeds_tried),
            Some(fail) => {
                writeln!(
                    f,
                    "riri: explored {} seed(s), seed {} failed in {} decision(s), {}",
                    self.seeds_tried, fail.seed, fail.decisions, fail.shrink
                )?;
                for d in &fail.report.diagnostics {
                    writeln!(f, "  - {d}")?;
                }
                if let Some(s) = fail.shrink.schedule() {
                    writeln!(f, "  schedule: {:?}", s.plan())?;
                }
                Ok(())
            }
        }
    }
}

/// Searches across schedules for one that finds a problem.
///
/// Built with [`Explore::new`], run with [`Explore::run`] or
/// [`Explore::run_with`].
pub struct Explore {
    base: LaunchConfig,
    seeds: u64,
    minimise: bool,
    budget: usize,
}

impl Explore {
    /// Defaults: 64 seeds starting from the config's own, shrinking on, and a
    /// budget of 256 replays for shrinking.
    pub fn new(config: &LaunchConfig) -> Self {
        Explore {
            base: *config,
            seeds: 64,
            minimise: true,
            budget: 256,
        }
    }

    /// How many seeds to try. Exploration stops at the first failing one.
    pub fn seeds(mut self, seeds: u64) -> Self {
        self.seeds = seeds;
        self
    }

    /// Whether to shrink a failing schedule. On by default.
    pub fn minimise(mut self, minimise: bool) -> Self {
        self.minimise = minimise;
        self
    }

    /// How many replays shrinking may spend. Shrinking stops early when the
    /// budget runs out and returns the best schedule found so far.
    pub fn budget(mut self, replays: usize) -> Self {
        self.budget = replays;
        self
    }

    /// Runs one kernel across the configured seeds.
    ///
    /// The kernel runs many times, so any buffer it captures carries its
    /// contents from one run into the next. For a kernel whose behaviour
    /// depends on the values it reads, use [`Explore::run_with`] instead.
    pub fn run<F>(&self, kernel: F) -> Exploration
    where
        F: Fn(&ThreadCtx<'_>) + Sync,
    {
        self.search(|config, choices| crate::launch::run(config, choices, &kernel))
    }

    /// Runs across the configured seeds, calling `setup` to build fresh state
    /// and a kernel before every run.
    ///
    /// ```
    /// use riri::{Explore, GlobalBuf, LaunchConfig, ThreadCtx};
    ///
    /// let found = Explore::new(&LaunchConfig::new(1, 4)).seeds(8).run_with(|| {
    ///     let counter = GlobalBuf::new("counter", vec![0u32; 1]);
    ///     move |t: &ThreadCtx<'_>| {
    ///         let seen = counter.read(t, 0);
    ///         counter.write(t, 0, seen + 1);
    ///     }
    /// });
    ///
    /// assert!(!found.is_clean());
    /// ```
    pub fn run_with<S, K>(&self, setup: S) -> Exploration
    where
        S: Fn() -> K,
        K: Fn(&ThreadCtx<'_>) + Sync,
    {
        self.search(|config, choices| crate::launch::run(config, choices, setup()))
    }

    fn search<R>(&self, run_once: R) -> Exploration
    where
        R: Fn(&LaunchConfig, Choices) -> (Report, Vec<u32>, bool),
    {
        for i in 0..self.seeds {
            let seed = self.base.seed.wrapping_add(i);
            let mut config = self.base;
            config.seed = seed;

            let (report, trace, complete) = run_once(&config, Choices::Random(SplitMix64(seed)));
            if report.is_clean() {
                continue;
            }

            let decisions = trace.len();
            let shrink = self.shrink(&config, &report, trace, complete, &run_once);
            return Exploration {
                seeds_tried: i + 1,
                failure: Some(Failure {
                    seed,
                    report,
                    shrink,
                    decisions,
                }),
            };
        }
        Exploration {
            seeds_tried: self.seeds,
            failure: None,
        }
    }

    fn shrink<R>(
        &self,
        config: &LaunchConfig,
        report: &Report,
        trace: Vec<u32>,
        complete: bool,
        run_once: &R,
    ) -> Shrink
    where
        R: Fn(&LaunchConfig, Choices) -> (Report, Vec<u32>, bool),
    {
        if !self.minimise {
            return Shrink::Disabled;
        }
        if !complete {
            return Shrink::TraceTruncated;
        }
        let Some(target) = report.diagnostics.first().map(|d| d.fingerprint()) else {
            return Shrink::Disabled;
        };

        let mut budget = self.budget;
        let reproduces = |plan: &[u32], budget: &mut usize| {
            if *budget == 0 {
                return false;
            }
            *budget -= 1;
            let choices = Choices::Replay {
                plan: plan.to_vec(),
                cursor: 0,
            };
            run_once(config, choices).0.contains(&target)
        };

        // Replaying the full recording must reproduce the finding. When it
        // does not, the kernel is not a pure function of its schedule and
        // every shrinking step below would be reasoning about noise.
        if !reproduces(&trace, &mut budget) {
            return Shrink::NotReproducible;
        }

        // Shortest prefix that still reproduces. Doubling then bisecting
        // assumes that a longer prefix is at least as likely to reproduce as
        // a shorter one, which is a heuristic rather than a guarantee, so
        // this finds a short prefix and not necessarily the shortest.
        let n = trace.len();
        let mut hi = n;
        if reproduces(&[], &mut budget) {
            // The default policy alone finds it, so no decisions are needed.
            hi = 0;
        } else {
            let mut lo = 0;
            let mut step = 1;
            while step < n {
                if reproduces(&trace[..step], &mut budget) {
                    hi = step;
                    break;
                }
                lo = step;
                step *= 2;
            }
            while lo + 1 < hi {
                let mid = lo + (hi - lo) / 2;
                if reproduces(&trace[..mid], &mut budget) {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }
        }

        // Then drop switches: wherever the turn changes hands, try letting
        // the previous thread keep running instead.
        let mut plan = trace[..hi].to_vec();
        for i in 1..plan.len() {
            if budget == 0 {
                break;
            }
            if plan[i] == plan[i - 1] {
                continue;
            }
            let mut candidate = plan.clone();
            candidate[i] = candidate[i - 1];
            if reproduces(&candidate, &mut budget) {
                plan = candidate;
            }
        }

        Shrink::Minimised(Schedule::new(plan))
    }
}

/// Searches `seeds` schedules for one that finds a problem, shrinking the
/// first failure it finds.
///
/// Equivalent to `Explore::new(config).seeds(seeds).run(kernel)`.
pub fn explore<F>(config: &LaunchConfig, seeds: u64, kernel: F) -> Exploration
where
    F: Fn(&ThreadCtx<'_>) + Sync,
{
    Explore::new(config).seeds(seeds).run(kernel)
}

/// Runs a kernel against an exact schedule rather than a seed.
///
/// Decisions past the end of the schedule keep the current thread running
/// while it can, so a short schedule is a complete and reproducible
/// description of a run.
pub fn replay<F>(config: &LaunchConfig, schedule: &Schedule, kernel: F) -> Report
where
    F: Fn(&ThreadCtx<'_>) + Sync,
{
    let choices = Choices::Replay {
        plan: schedule.plan().to_vec(),
        cursor: 0,
    };
    crate::launch::run(config, choices, kernel).0
}
