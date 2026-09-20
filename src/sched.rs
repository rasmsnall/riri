//! Deterministic, seeded scheduler.
//!
//! Every simulated GPU thread is backed by an OS thread, but only the thread
//! holding the *turn* may run. At every instrumented operation the running
//! thread hands the turn to a runnable thread chosen by a seeded RNG, so a
//! given seed always reproduces the same interleaving.
//!
//! Threads can block in three ways: at a block-wide `sync_threads` barrier,
//! at a warp collective waiting for its mask-mates, or not at all. If no
//! thread is runnable while some thread is still blocked, the launch is
//! wedged, and [`Scheduler::detect_stuck`] turns that into a diagnostic
//! rather than a hang.

use std::panic::Location;
use std::sync::{Condvar, Mutex, MutexGuard};

use crate::diag::{Diagnostic, Reporter};

/// Panic payload used to unwind simulated threads when a launch aborts.
pub(crate) struct AbortSignal;

/// First half of a warp collective: everyone shows up and publishes a value.
pub(crate) const PHASE_ARRIVE: u8 = 0;
/// Second half: everyone has read what they needed, so the exchange slots
/// are free to be reused by the next collective.
pub(crate) const PHASE_DEPART: u8 = 1;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Status {
    Runnable,
    AtBarrier,
    AtWarp,
    Finished,
}

/// What a thread parked at a warp collective is waiting for.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct WarpWait {
    phase: u8,
    mask: u32,
    at: &'static Location<'static>,
    op: &'static str,
}

/// Upper bound on recorded scheduling decisions. A launch that runs past it
/// keeps executing but stops recording, and its schedule cannot be replayed.
pub(crate) const MAX_TRACE: usize = 1 << 20;

/// Where the scheduler's decisions come from.
///
/// Recording the *thread* chosen rather than its position among the runnable
/// threads is what makes a trace survive being edited: a plan that names a
/// thread which is no longer runnable falls back cleanly instead of silently
/// meaning something else.
pub(crate) enum Choices {
    /// Pick uniformly at random from the runnable threads.
    Random(SplitMix64),
    /// Follow a recorded plan. Once it runs out, keep the current thread
    /// running while it can, which is the schedule with the fewest switches
    /// and so the one that reads most like ordinary sequential code.
    Replay { plan: Vec<u32>, cursor: usize },
}

pub(crate) struct SplitMix64(pub(crate) u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

struct State {
    current: Option<usize>,
    status: Vec<Status>,
    barrier_gen: Vec<u64>,
    barrier_loc: Vec<Option<&'static Location<'static>>>,
    warp_wait: Vec<Option<WarpWait>>,
    warp_gen: Vec<u64>,
    choices: Choices,
    trace: Vec<u32>,
    trace_complete: bool,
    aborted: bool,
}

pub(crate) struct Scheduler {
    blocks: usize,
    threads_per_block: usize,
    warp_size: usize,
    warps_per_block: usize,
    state: Mutex<State>,
    /// One per thread rather than one for all of them.
    ///
    /// A handoff wakes exactly the thread that now holds the turn. Waking
    /// every blocked thread so that one of them can proceed costs a wakeup
    /// per thread per scheduling decision, which is quadratic in the launch
    /// and was what kept blocks small.
    cvs: Vec<Condvar>,
}

impl Scheduler {
    pub(crate) fn new(
        blocks: usize,
        threads_per_block: usize,
        warp_size: usize,
        choices: Choices,
    ) -> Self {
        let total = blocks * threads_per_block;
        let warps_per_block = threads_per_block.div_ceil(warp_size);
        let mut state = State {
            current: None,
            status: vec![Status::Runnable; total],
            barrier_gen: vec![0; blocks],
            barrier_loc: vec![None; blocks],
            warp_wait: vec![None; total],
            warp_gen: vec![0; blocks * warps_per_block],
            choices,
            trace: Vec::new(),
            trace_complete: true,
            aborted: false,
        };
        Self::pick_next(&mut state);
        Scheduler {
            blocks,
            threads_per_block,
            warp_size,
            warps_per_block,
            state: Mutex::new(state),
            cvs: (0..total).map(|_| Condvar::new()).collect(),
        }
    }

    fn pick_next(s: &mut State) {
        let runnable: Vec<usize> = (0..s.status.len())
            .filter(|&i| s.status[i] == Status::Runnable)
            .collect();
        if runnable.is_empty() {
            s.current = None;
            return;
        }

        let previous = s.current;
        let chosen = match &mut s.choices {
            Choices::Random(rng) => runnable[(rng.next() % runnable.len() as u64) as usize],
            Choices::Replay { plan, cursor } => {
                let planned = plan.get(*cursor).copied();
                *cursor += 1;
                match planned {
                    // A plan naming a thread that cannot run now is stale,
                    // which happens while a schedule is being shrunk. Fall
                    // through to the same default as a spent plan.
                    Some(gid) if runnable.contains(&(gid as usize)) => gid as usize,
                    _ => previous
                        .filter(|p| runnable.contains(p))
                        .unwrap_or(runnable[0]),
                }
            }
        };

        if s.trace.len() < MAX_TRACE {
            s.trace.push(chosen as u32);
        } else {
            s.trace_complete = false;
        }
        s.current = Some(chosen);
    }

    /// The decisions this launch made, and whether the record is complete.
    pub(crate) fn take_trace(&self) -> (Vec<u32>, bool) {
        let mut s = self.state.lock().unwrap();
        (std::mem::take(&mut s.trace), s.trace_complete)
    }

    fn block_range(&self, block: usize) -> std::ops::Range<usize> {
        block * self.threads_per_block..(block + 1) * self.threads_per_block
    }

    // ------------------------------------------------------------- warps ---

    /// Splits a global thread id into (block, warp within block, lane).
    fn locate(&self, gid: usize) -> (usize, usize, u32) {
        let block = gid / self.threads_per_block;
        let t = gid % self.threads_per_block;
        (block, t / self.warp_size, (t % self.warp_size) as u32)
    }

    /// Global thread id of lane 0 of a warp.
    fn warp_base(&self, block: usize, warp: usize) -> usize {
        block * self.threads_per_block + warp * self.warp_size
    }

    /// How many lanes of this warp actually exist. The last warp of a block
    /// is short when the block size is not a multiple of the warp size.
    fn lanes_in(&self, warp: usize) -> u32 {
        (self.threads_per_block - warp * self.warp_size).min(self.warp_size) as u32
    }

    /// Mask of the lanes of this warp that exist.
    pub(crate) fn valid_mask(&self, warp: usize) -> u32 {
        let n = self.lanes_in(warp);
        if n >= 32 {
            u32::MAX
        } else {
            (1u32 << n) - 1
        }
    }

    fn warp_gen_index(&self, block: usize, warp: usize) -> usize {
        block * self.warps_per_block + warp
    }

    /// Lanes of this warp currently parked at the same collective as `wait`.
    fn arrived_mask(&self, s: &State, block: usize, warp: usize, wait: &WarpWait) -> u32 {
        let base = self.warp_base(block, warp);
        (0..self.lanes_in(warp))
            .filter(|&l| {
                s.status[base + l as usize] == Status::AtWarp
                    && s.warp_wait[base + l as usize]
                        .is_some_and(|w| w.phase == wait.phase && w.at == wait.at)
            })
            .fold(0u32, |acc, l| acc | (1u32 << l))
    }

    // --------------------------------------------------------- scheduling ---

    /// Blocks until it is `me`'s turn. `Err` means the launch was aborted.
    pub(crate) fn wait_turn(&self, me: usize) -> Result<(), AbortSignal> {
        let mut s = self.state.lock().unwrap();
        loop {
            if s.aborted {
                return Err(AbortSignal);
            }
            if s.current == Some(me) {
                return Ok(());
            }
            s = self.cvs[me].wait(s).unwrap();
        }
    }

    fn hand_off_and_wait(
        &self,
        mut s: MutexGuard<'_, State>,
        me: usize,
        reporter: &Reporter,
    ) -> Result<(), AbortSignal> {
        Self::pick_next(&mut s);
        if s.current.is_none() {
            // Nobody can run but this thread is still live: the launch is
            // wedged. Work out who is waiting on whom and say so.
            self.detect_stuck(&mut s, reporter);
        }
        self.wake(&s);
        loop {
            if s.aborted {
                return Err(AbortSignal);
            }
            if s.current == Some(me) {
                return Ok(());
            }
            s = self.cvs[me].wait(s).unwrap();
        }
    }

    /// A scheduling point: possibly lets another thread run first.
    pub(crate) fn yield_now(&self, me: usize, reporter: &Reporter) -> Result<(), AbortSignal> {
        let s = self.state.lock().unwrap();
        self.hand_off_and_wait(s, me, reporter)
    }

    /// Block barrier generation and warp collective generation, read together
    /// so a single access sees a consistent pair.
    pub(crate) fn epochs(&self, block: usize, warp: usize) -> (u64, u64) {
        let s = self.state.lock().unwrap();
        (
            s.barrier_gen[block],
            s.warp_gen[self.warp_gen_index(block, warp)],
        )
    }

    // ------------------------------------------------------- deadlock ---

    /// Reports why the launch cannot make progress, then aborts it.
    ///
    /// Warp collectives are reported first: when a warp cannot reconverge,
    /// any block barrier behind it is a consequence, not the cause.
    fn detect_stuck(&self, s: &mut State, reporter: &Reporter) {
        let mut reported = false;
        for block in 0..self.blocks {
            for warp in 0..self.warps_per_block {
                let base = self.warp_base(block, warp);
                let stuck: Vec<u32> = (0..self.lanes_in(warp))
                    .filter(|&l| s.status[base + l as usize] == Status::AtWarp)
                    .collect();
                let Some(&first) = stuck.first() else {
                    continue;
                };
                let Some(wait) = s.warp_wait[base + first as usize] else {
                    continue;
                };
                let arrived = self.arrived_mask(s, block, warp, &wait);
                reporter.push(Diagnostic::WarpDivergence {
                    block: block as u32,
                    warp: warp as u32,
                    op: wait.op,
                    mask: wait.mask,
                    arrived,
                    at: wait.at,
                });
                reported = true;
            }
        }
        if !reported {
            for block in 0..self.blocks {
                let range = self.block_range(block);
                let waiting = range
                    .clone()
                    .filter(|&i| s.status[i] == Status::AtBarrier)
                    .count();
                if waiting == 0 {
                    continue;
                }
                let exited = range.filter(|&i| s.status[i] == Status::Finished).count();
                reporter.push(Diagnostic::BarrierDivergence {
                    block: block as u32,
                    waiting,
                    exited,
                    barrier: s.barrier_loc[block].unwrap_or_else(|| Location::caller()),
                });
            }
        }
        s.aborted = true;
    }

    fn fail(&self, s: &mut State, d: Diagnostic, reporter: &Reporter) {
        reporter.push(d);
        s.aborted = true;
        self.wake(s);
    }

    /// Wakes whoever needs to run next.
    ///
    /// An abort wakes everyone, because every thread has to notice it and
    /// unwind. Otherwise exactly one thread holds the turn, and only that one
    /// has anything to do.
    fn wake(&self, s: &State) {
        if s.aborted {
            for cv in &self.cvs {
                cv.notify_one();
            }
        } else if let Some(next) = s.current {
            self.cvs[next].notify_one();
        }
    }

    // -------------------------------------------------------- barriers ---

    fn barrier_divergence(&self, s: &mut State, block: usize, reporter: &Reporter) {
        let range = self.block_range(block);
        let waiting = range
            .clone()
            .filter(|&i| s.status[i] == Status::AtBarrier)
            .count();
        let exited = range.filter(|&i| s.status[i] == Status::Finished).count();
        let d = Diagnostic::BarrierDivergence {
            block: block as u32,
            waiting,
            exited,
            barrier: s.barrier_loc[block].unwrap_or_else(|| Location::caller()),
        };
        self.fail(s, d, reporter);
    }

    /// `__syncthreads()`: wait until every thread of the block arrives.
    pub(crate) fn barrier(
        &self,
        me: usize,
        loc: &'static Location<'static>,
        reporter: &Reporter,
    ) -> Result<(), AbortSignal> {
        let block = me / self.threads_per_block;
        let mut s = self.state.lock().unwrap();
        s.status[me] = Status::AtBarrier;
        s.barrier_loc[block].get_or_insert(loc);

        let range = self.block_range(block);
        if range.clone().any(|i| s.status[i] == Status::Finished) {
            self.barrier_divergence(&mut s, block, reporter);
            return Err(AbortSignal);
        }
        if range.clone().all(|i| s.status[i] == Status::AtBarrier) {
            for i in range {
                s.status[i] = Status::Runnable;
            }
            s.barrier_gen[block] += 1;
            s.barrier_loc[block] = None;
        }
        self.hand_off_and_wait(s, me, reporter)
    }

    // ------------------------------------------------ warp collectives ---

    /// One half of a warp collective: park until every lane named by `mask`
    /// is parked at the same place, then release them together.
    ///
    /// `mask` must already be known to name only lanes that exist and to
    /// include the caller; [`crate::warp`] checks that before calling.
    pub(crate) fn warp_rendezvous(
        &self,
        me: usize,
        phase: u8,
        mask: u32,
        at: &'static Location<'static>,
        op: &'static str,
        reporter: &Reporter,
    ) -> Result<(), AbortSignal> {
        let (block, warp, lane) = self.locate(me);
        let base = self.warp_base(block, warp);
        let wait = WarpWait {
            phase,
            mask,
            at,
            op,
        };

        let mut s = self.state.lock().unwrap();
        s.status[me] = Status::AtWarp;
        s.warp_wait[me] = Some(wait);

        let members: Vec<usize> = (0..self.lanes_in(warp))
            .filter(|&l| mask & (1u32 << l) != 0)
            .map(|l| base + l as usize)
            .collect();

        // A mask-mate that already left the kernel can never arrive.
        if members.iter().any(|&p| s.status[p] == Status::Finished) {
            let arrived = self.arrived_mask(&s, block, warp, &wait);
            let d = Diagnostic::WarpDivergence {
                block: block as u32,
                warp: warp as u32,
                op,
                mask,
                arrived,
                at,
            };
            self.fail(&mut s, d, reporter);
            return Err(AbortSignal);
        }

        // Participants must agree on who is taking part. This scans the whole
        // warp, not just this lane's members: a lane that claims us while we
        // do not claim it is exactly the disagreement worth catching, and
        // checking both directions makes detection independent of who parked
        // first. Two genuinely disjoint groups at the same source line claim
        // neither each other nor us, so they stay clean.
        for l in 0..self.lanes_in(warp) {
            let p = base + l as usize;
            if p == me || s.status[p] != Status::AtWarp {
                continue;
            }
            let Some(other) = s.warp_wait[p] else {
                continue;
            };
            if other.phase != phase || other.at != at || other.mask == mask {
                continue;
            }
            let claims_each_other = mask & (1u32 << l) != 0 || other.mask & (1u32 << lane) != 0;
            if !claims_each_other {
                continue;
            }
            let d = Diagnostic::WarpMaskMismatch {
                block: block as u32,
                warp: warp as u32,
                op,
                at,
                lane,
                mask,
                other_lane: l,
                other_mask: other.mask,
            };
            self.fail(&mut s, d, reporter);
            return Err(AbortSignal);
        }

        let complete = members
            .iter()
            .all(|&p| s.status[p] == Status::AtWarp && s.warp_wait[p] == Some(wait));
        if complete {
            for &p in &members {
                s.status[p] = Status::Runnable;
                s.warp_wait[p] = None;
            }
            // Leaving the collective orders everything before it against
            // everything after it, but only for a full-warp mask: a partial
            // mask says nothing about the lanes it leaves out.
            if phase == PHASE_DEPART && mask == self.valid_mask(warp) {
                s.warp_gen[self.warp_gen_index(block, warp)] += 1;
            }
        }
        self.hand_off_and_wait(s, me, reporter)
    }

    // ---------------------------------------------------------- exit ---

    /// The thread returned from the kernel.
    pub(crate) fn finish(&self, me: usize, reporter: &Reporter) {
        let (block, warp, lane) = self.locate(me);
        let mut s = self.state.lock().unwrap();
        s.status[me] = Status::Finished;

        // Warp-mates still waiting for this lane will never be released.
        let base = self.warp_base(block, warp);
        for l in 0..self.lanes_in(warp) {
            let p = base + l as usize;
            if s.status[p] != Status::AtWarp {
                continue;
            }
            let Some(wait) = s.warp_wait[p] else { continue };
            if wait.mask & (1u32 << lane) == 0 {
                continue;
            }
            let arrived = self.arrived_mask(&s, block, warp, &wait);
            let d = Diagnostic::WarpDivergence {
                block: block as u32,
                warp: warp as u32,
                op: wait.op,
                mask: wait.mask,
                arrived,
                at: wait.at,
            };
            self.fail(&mut s, d, reporter);
            return;
        }

        if self
            .block_range(block)
            .any(|i| s.status[i] == Status::AtBarrier)
        {
            self.barrier_divergence(&mut s, block, reporter);
            return;
        }

        Self::pick_next(&mut s);
        if s.current.is_none() && s.status.iter().any(|&st| st != Status::Finished) {
            // The last runnable thread exited while others are still parked.
            self.detect_stuck(&mut s, reporter);
        }
        self.wake(&s);
    }

    /// Stops the launch (trap-like errors). All waiting threads unwind.
    pub(crate) fn abort(&self) {
        let mut s = self.state.lock().unwrap();
        s.aborted = true;
        self.wake(&s);
    }

    pub(crate) fn was_aborted(&self) -> bool {
        self.state.lock().unwrap().aborted
    }
}
