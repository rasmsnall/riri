//! Deterministic, seeded scheduler.
//!
//! Every simulated GPU thread is backed by an OS thread, but only the thread
//! holding the *turn* may run. At every instrumented operation the running
//! thread hands the turn to a runnable thread chosen by a seeded RNG, so a
//! given seed always reproduces the same interleaving.

use std::panic::Location;
use std::sync::{Condvar, Mutex};

use crate::diag::{Diagnostic, Reporter};

/// Panic payload used to unwind simulated threads when a launch aborts.
pub(crate) struct AbortSignal;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Status {
    Runnable,
    AtBarrier,
    Finished,
}

struct SplitMix64(u64);

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
    rng: SplitMix64,
    aborted: bool,
}

pub(crate) struct Scheduler {
    threads_per_block: usize,
    state: Mutex<State>,
    cv: Condvar,
}

impl Scheduler {
    pub(crate) fn new(blocks: usize, threads_per_block: usize, seed: u64) -> Self {
        let total = blocks * threads_per_block;
        let mut state = State {
            current: None,
            status: vec![Status::Runnable; total],
            barrier_gen: vec![0; blocks],
            barrier_loc: vec![None; blocks],
            rng: SplitMix64(seed),
            aborted: false,
        };
        Self::pick_next(&mut state);
        Scheduler { threads_per_block, state: Mutex::new(state), cv: Condvar::new() }
    }

    fn pick_next(s: &mut State) {
        let runnable: Vec<usize> = (0..s.status.len())
            .filter(|&i| s.status[i] == Status::Runnable)
            .collect();
        s.current = if runnable.is_empty() {
            None
        } else {
            Some(runnable[(s.rng.next() % runnable.len() as u64) as usize])
        };
    }

    fn block_range(&self, block: usize) -> std::ops::Range<usize> {
        block * self.threads_per_block..(block + 1) * self.threads_per_block
    }

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
            s = self.cv.wait(s).unwrap();
        }
    }

    fn hand_off_and_wait(&self, mut s: std::sync::MutexGuard<'_, State>, me: usize) -> Result<(), AbortSignal> {
        Self::pick_next(&mut s);
        if s.current.is_none() {
            // Nobody can run but this thread is still live: a scheduler-level
            // deadlock. Should be unreachable; fail loudly rather than hang.
            s.aborted = true;
        }
        self.cv.notify_all();
        loop {
            if s.aborted {
                return Err(AbortSignal);
            }
            if s.current == Some(me) {
                return Ok(());
            }
            s = self.cv.wait(s).unwrap();
        }
    }

    /// A scheduling point: possibly lets another thread run first.
    pub(crate) fn yield_now(&self, me: usize) -> Result<(), AbortSignal> {
        let s = self.state.lock().unwrap();
        self.hand_off_and_wait(s, me)
    }

    pub(crate) fn epoch(&self, block: usize) -> u64 {
        self.state.lock().unwrap().barrier_gen[block]
    }

    fn divergence(&self, s: &mut State, block: usize, reporter: &Reporter) {
        let range = self.block_range(block);
        let waiting = range.clone().filter(|&i| s.status[i] == Status::AtBarrier).count();
        let exited = range.filter(|&i| s.status[i] == Status::Finished).count();
        reporter.push(Diagnostic::BarrierDivergence {
            block: block as u32,
            waiting,
            exited,
            barrier: s.barrier_loc[block].unwrap_or_else(|| Location::caller()),
        });
        s.aborted = true;
        self.cv.notify_all();
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
            self.divergence(&mut s, block, reporter);
            return Err(AbortSignal);
        }
        if range.clone().all(|i| s.status[i] == Status::AtBarrier) {
            for i in range {
                s.status[i] = Status::Runnable;
            }
            s.barrier_gen[block] += 1;
            s.barrier_loc[block] = None;
        }
        self.hand_off_and_wait(s, me)
    }

    /// The thread returned from the kernel.
    pub(crate) fn finish(&self, me: usize, reporter: &Reporter) {
        let block = me / self.threads_per_block;
        let mut s = self.state.lock().unwrap();
        s.status[me] = Status::Finished;
        if self.block_range(block).any(|i| s.status[i] == Status::AtBarrier) {
            self.divergence(&mut s, block, reporter);
            return;
        }
        Self::pick_next(&mut s);
        self.cv.notify_all();
    }

    /// Stops the launch (trap-like errors). All waiting threads unwind.
    pub(crate) fn abort(&self) {
        let mut s = self.state.lock().unwrap();
        s.aborted = true;
        self.cv.notify_all();
    }

    pub(crate) fn was_aborted(&self) -> bool {
        self.state.lock().unwrap().aborted
    }
}
