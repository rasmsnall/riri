use std::any::Any;
use std::collections::HashMap;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::ctx::ThreadCtx;
use crate::diag::{Diagnostic, Report, Reporter};
use crate::dim::Dim3;
use crate::sched::{AbortSignal, Scheduler};

/// Simulated threads are real OS threads, so keep launches test-sized.
pub const MAX_THREADS: u32 = 16_384;

static NEXT_LAUNCH_ID: AtomicU64 = AtomicU64::new(1);

/// Grid and block shape plus the scheduler seed.
#[derive(Clone, Copy, Debug)]
pub struct LaunchConfig {
    pub grid: Dim3,
    pub block: Dim3,
    pub seed: u64,
}

impl LaunchConfig {
    pub fn new(grid: impl Into<Dim3>, block: impl Into<Dim3>) -> Self {
        LaunchConfig { grid: grid.into(), block: block.into(), seed: 0 }
    }

    /// Sets the scheduler seed. Different seeds explore different
    /// interleavings; the same seed always reproduces the same one.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

pub(crate) struct BlockState {
    pub(crate) shared: Mutex<HashMap<&'static str, Arc<dyn Any + Send + Sync>>>,
}

pub(crate) struct LaunchState {
    pub(crate) id: u64,
    pub(crate) config: LaunchConfig,
    pub(crate) sched: Scheduler,
    pub(crate) reporter: Reporter,
    pub(crate) blocks: Vec<BlockState>,
}

/// Runs `kernel` once per simulated GPU thread and returns what Riri found.
///
/// The kernel receives a [`ThreadCtx`] for its thread. All memory it touches
/// must go through Riri's instrumented types ([`crate::GlobalBuf`],
/// [`crate::SharedArray`]) to be checked.
pub fn launch<F>(config: &LaunchConfig, kernel: F) -> Report
where
    F: Fn(&ThreadCtx<'_>) + Sync,
{
    let blocks = config.grid.count();
    let tpb = config.block.count();
    assert!(blocks > 0 && tpb > 0, "riri: grid and block must be non-empty");
    let total = blocks.checked_mul(tpb).expect("riri: launch too large");
    assert!(
        total <= MAX_THREADS,
        "riri: {total} threads exceeds the limit of {MAX_THREADS}; shrink the launch for testing"
    );

    let state = LaunchState {
        id: NEXT_LAUNCH_ID.fetch_add(1, Ordering::Relaxed),
        config: *config,
        sched: Scheduler::new(blocks as usize, tpb as usize, config.seed),
        reporter: Reporter::default(),
        blocks: (0..blocks).map(|_| BlockState { shared: Mutex::new(HashMap::new()) }).collect(),
    };

    std::thread::scope(|scope| {
        for gid in 0..total as usize {
            let state = &state;
            let kernel = &kernel;
            std::thread::Builder::new()
                .name(format!("riri-{gid}"))
                .stack_size(256 * 1024)
                .spawn_scoped(scope, move || run_thread(state, kernel, gid))
                .expect("riri: failed to spawn simulated thread");
        }
    });

    Report {
        seed: config.seed,
        diagnostics: state.reporter.take(),
        aborted: state.sched.was_aborted(),
    }
}

fn run_thread<F>(state: &LaunchState, kernel: &F, gid: usize)
where
    F: Fn(&ThreadCtx<'_>) + Sync,
{
    let tpb = state.config.block.count() as usize;
    let ctx = ThreadCtx { launch: state, block: (gid / tpb) as u32, thread: (gid % tpb) as u32, gid };

    if state.sched.wait_turn(gid).is_err() {
        return;
    }
    match catch_unwind(AssertUnwindSafe(|| kernel(&ctx))) {
        Ok(()) => state.sched.finish(gid, &state.reporter),
        Err(payload) if payload.is::<AbortSignal>() => {}
        Err(payload) => {
            let message = payload
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "<non-string panic payload>".into());
            state.reporter.push(Diagnostic::KernelPanic { block: ctx.block, thread: ctx.thread, message });
            state.sched.abort();
        }
    }
}
