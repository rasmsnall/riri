use std::any::Any;
use std::collections::HashMap;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::ctx::ThreadCtx;
use crate::diag::{Diagnostic, Report, Reporter};
use crate::dim::Dim3;
use crate::sched::{AbortSignal, Choices, Scheduler, SplitMix64};
use crate::sync::SyncState;
use crate::warp::WARP_SIZE;

/// Simulated threads are real OS threads, so keep launches test-sized.
pub const MAX_THREADS: u32 = 16_384;

static NEXT_LAUNCH_ID: AtomicU64 = AtomicU64::new(1);

/// Grid and block shape, warp width, and the scheduler seed.
#[derive(Clone, Copy, Debug)]
pub struct LaunchConfig {
    pub grid: Dim3,
    pub block: Dim3,
    pub seed: u64,
    /// Lanes per warp. Defaults to [`WARP_SIZE`].
    pub warp_size: u32,
}

impl LaunchConfig {
    pub fn new(grid: impl Into<Dim3>, block: impl Into<Dim3>) -> Self {
        LaunchConfig {
            grid: grid.into(),
            block: block.into(),
            seed: 0,
            warp_size: WARP_SIZE,
        }
    }

    /// Sets the scheduler seed. Different seeds explore different
    /// interleavings; the same seed always reproduces the same one.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Sets the number of lanes per warp.
    ///
    /// Must be a power of two no greater than 32, since member masks are
    /// `u32`. Useful for writing small, readable warp tests; 64-lane AMD
    /// wavefronts cannot be modelled with a `u32` mask and are not supported.
    pub fn warp_size(mut self, lanes: u32) -> Self {
        assert!(
            lanes.is_power_of_two() && lanes <= 32,
            "riri: warp size must be a power of two no greater than 32, got {lanes}"
        );
        self.warp_size = lanes;
        self
    }
}

pub(crate) struct BlockState {
    pub(crate) shared: Mutex<HashMap<&'static str, Arc<dyn Any + Send + Sync>>>,
}

/// Per-warp exchange slots, one per lane, used by warp collectives to hand
/// values between lanes without going through instrumented memory.
pub(crate) struct WarpState {
    pub(crate) slots: Mutex<Vec<Option<Box<dyn Any + Send>>>>,
}

pub(crate) struct LaunchState {
    pub(crate) id: u64,
    pub(crate) config: LaunchConfig,
    pub(crate) sched: Scheduler,
    pub(crate) reporter: Reporter,
    pub(crate) blocks: Vec<BlockState>,
    pub(crate) warps: Vec<WarpState>,
    pub(crate) warps_per_block: usize,
    /// Per-thread vector clocks. Only one thread runs at a time, so this is
    /// never contended.
    pub(crate) sync: Mutex<SyncState>,
}

/// Runs `kernel` once per simulated GPU thread and returns what Riri found.
///
/// The kernel receives a [`ThreadCtx`] for its thread. All memory it touches
/// must go through Riri's instrumented types ([`crate::GlobalBuf`],
/// [`crate::SharedArray`]) to be checked, and all lane-to-lane exchange
/// through [`crate::warp`].
pub fn launch<F>(config: &LaunchConfig, kernel: F) -> Report
where
    F: Fn(&ThreadCtx<'_>) + Sync,
{
    run(config, Choices::Random(SplitMix64(config.seed)), kernel).0
}

/// Runs a kernel against an explicit source of scheduling decisions, and
/// returns what was found alongside the decisions that were made.
///
/// The second element of the returned tuple is `false` when the launch made
/// more than `MAX_TRACE` decisions, in which case the trace is truncated and
/// cannot be replayed.
pub(crate) fn run<F>(config: &LaunchConfig, choices: Choices, kernel: F) -> (Report, Vec<u32>, bool)
where
    F: Fn(&ThreadCtx<'_>) + Sync,
{
    run_shared(config, choices, |_, ctx| kernel(ctx))
}

/// The launch body, handing each simulated thread both its context and a
/// share of the launch state.
///
/// The `Arc` is what lets a context-free kernel surface work: a thread can
/// park its share in a thread local and have free functions rebuild a
/// [`ThreadCtx`] from it, with no raw pointers and no lifetime erasure. See
/// [`crate::oxide`].
pub(crate) fn run_shared<F>(
    config: &LaunchConfig,
    choices: Choices,
    kernel: F,
) -> (Report, Vec<u32>, bool)
where
    F: Fn(&Arc<LaunchState>, &ThreadCtx<'_>) + Sync,
{
    let blocks = config.grid.count();
    let tpb = config.block.count();
    assert!(
        blocks > 0 && tpb > 0,
        "riri: grid and block must be non-empty"
    );
    assert!(
        config.warp_size.is_power_of_two() && config.warp_size <= 32,
        "riri: warp size must be a power of two no greater than 32, got {}",
        config.warp_size
    );
    let total = blocks.checked_mul(tpb).expect("riri: launch too large");
    assert!(
        total <= MAX_THREADS,
        "riri: {total} threads exceeds the limit of {MAX_THREADS}; shrink the launch for testing"
    );

    let ws = config.warp_size as usize;
    let warps_per_block = (tpb as usize).div_ceil(ws);

    let state = Arc::new(LaunchState {
        id: NEXT_LAUNCH_ID.fetch_add(1, Ordering::Relaxed),
        config: *config,
        sched: Scheduler::new(blocks as usize, tpb as usize, ws, choices),
        reporter: Reporter::default(),
        blocks: (0..blocks)
            .map(|_| BlockState {
                shared: Mutex::new(HashMap::new()),
            })
            .collect(),
        warps: (0..blocks as usize * warps_per_block)
            .map(|_| WarpState {
                slots: Mutex::new((0..ws).map(|_| None).collect()),
            })
            .collect(),
        warps_per_block,
        sync: Mutex::new(SyncState::new(total as usize, blocks as usize)),
    });

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

    let (trace, complete) = state.sched.take_trace();
    let report = Report {
        seed: config.seed,
        diagnostics: state.reporter.take(),
        aborted: state.sched.was_aborted(),
    };
    (report, trace, complete)
}

fn run_thread<F>(state: &Arc<LaunchState>, kernel: &F, gid: usize)
where
    F: Fn(&Arc<LaunchState>, &ThreadCtx<'_>) + Sync,
{
    let tpb = state.config.block.count() as usize;
    let ctx = ThreadCtx {
        launch: state,
        block: (gid / tpb) as u32,
        thread: (gid % tpb) as u32,
        gid,
    };

    if state.sched.wait_turn(gid).is_err() {
        return;
    }
    match catch_unwind(AssertUnwindSafe(|| kernel(state, &ctx))) {
        Ok(()) => state.sched.finish(gid, &state.reporter),
        Err(payload) if payload.is::<AbortSignal>() => {}
        Err(payload) => {
            let message = payload
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "<non-string panic payload>".into());
            state.reporter.push(Diagnostic::KernelPanic {
                block: ctx.block,
                thread: ctx.thread,
                message,
            });
            state.sched.abort();
        }
    }
}
