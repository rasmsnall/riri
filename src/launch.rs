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

/// Simulated threads are real OS threads, so a wave has to stay test-sized.
///
/// This bounds the threads resident at once, not the launch. A grid larger
/// than this runs in waves, once [`LaunchConfig::resident_blocks`] says how
/// many blocks fit at a time.
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
    /// How many blocks are resident at once. `None` means all of them.
    pub resident_blocks: Option<u32>,
}

impl LaunchConfig {
    pub fn new(grid: impl Into<Dim3>, block: impl Into<Dim3>) -> Self {
        LaunchConfig {
            grid: grid.into(),
            block: block.into(),
            seed: 0,
            warp_size: WARP_SIZE,
            resident_blocks: None,
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

    /// Limits how many blocks run at the same time, the way hardware does.
    ///
    /// A GPU schedules blocks onto its multiprocessors in waves, and CUDA is
    /// explicit that a kernel may not assume two blocks are resident together
    /// unless it was launched cooperatively. Riri runs every block at once by
    /// default, which is the permissive reading; setting this models the
    /// hardware instead.
    ///
    /// Two things follow. Peak cost becomes a wave rather than the whole
    /// launch, so [`MAX_THREADS`] stops bounding the grid and starts bounding
    /// the wave. And a kernel that waits on a block in a later wave deadlocks
    /// here, which is a real bug that all-resident scheduling hides.
    ///
    /// Race detection is unaffected. Accesses from different blocks are
    /// unordered whether or not they overlapped in time, so a race between
    /// waves is reported exactly as one within a wave is.
    ///
    /// Schedules are not recorded across waves, so a multi-wave launch cannot
    /// be replayed or shrunk.
    pub fn resident_blocks(mut self, blocks: u32) -> Self {
        assert!(blocks > 0, "riri: at least one block must be resident");
        self.resident_blocks = Some(blocks);
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

/// One wave's view of a launch.
///
/// The scheduler covers only the blocks resident right now. Everything that
/// has to outlive a wave, because a race between waves must still be seen, is
/// shared.
pub(crate) struct LaunchState {
    pub(crate) id: u64,
    pub(crate) config: LaunchConfig,
    pub(crate) sched: Scheduler,
    /// Index of this wave's first block within the grid.
    pub(crate) first_block: u32,
    pub(crate) reporter: Arc<Reporter>,
    pub(crate) blocks: Arc<Vec<BlockState>>,
    pub(crate) warps: Arc<Vec<WarpState>>,
    pub(crate) warps_per_block: usize,
    /// Per-thread vector clocks. Only one thread runs at a time, so this is
    /// never contended.
    pub(crate) sync: Arc<Mutex<SyncState>>,
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
    // The launch as a whole is unbounded: only what is resident at once has
    // to fit, and that is checked below once the wave size is known.
    let total = blocks.checked_mul(tpb).expect("riri: launch too large");

    let ws = config.warp_size as usize;
    let warps_per_block = (tpb as usize).div_ceil(ws);

    let resident = config.resident_blocks.unwrap_or(blocks).clamp(1, blocks);
    let wave_threads = resident.checked_mul(tpb).expect("riri: wave too large");
    assert!(
        wave_threads <= MAX_THREADS,
        "riri: {wave_threads} resident threads exceeds the limit of {MAX_THREADS}; lower resident_blocks or shrink the block"
    );
    let waves = blocks.div_ceil(resident);

    let id = NEXT_LAUNCH_ID.fetch_add(1, Ordering::Relaxed);
    let reporter = Arc::new(Reporter::default());
    let block_state: Arc<Vec<BlockState>> = Arc::new(
        (0..blocks)
            .map(|_| BlockState {
                shared: Mutex::new(HashMap::new()),
            })
            .collect(),
    );
    let warp_state: Arc<Vec<WarpState>> = Arc::new(
        (0..blocks as usize * warps_per_block)
            .map(|_| WarpState {
                slots: Mutex::new((0..ws).map(|_| None).collect()),
            })
            .collect(),
    );
    let sync = Arc::new(Mutex::new(SyncState::new(total as usize, blocks as usize)));

    let mut trace = Vec::new();
    // A plan cannot span waves, because each wave numbers its threads from
    // zero, so replaying one would drive the wrong threads.
    let mut complete = waves == 1;
    let mut aborted = false;
    let mut choices = Some(choices);

    let mut first_block = 0u32;
    for wave in 0..waves {
        let wave_blocks = resident.min(blocks - first_block);
        let wave_total = (wave_blocks * tpb) as usize;

        let picks = choices
            .take()
            .unwrap_or_else(|| Choices::Random(SplitMix64(config.seed.wrapping_add(wave as u64))));

        let state = Arc::new(LaunchState {
            id,
            config: *config,
            sched: Scheduler::new(wave_blocks as usize, tpb as usize, ws, picks),
            first_block,
            reporter: Arc::clone(&reporter),
            blocks: Arc::clone(&block_state),
            warps: Arc::clone(&warp_state),
            warps_per_block,
            sync: Arc::clone(&sync),
        });

        std::thread::scope(|scope| {
            for slot in 0..wave_total {
                let state = &state;
                let kernel = &kernel;
                std::thread::Builder::new()
                    .name(format!(
                        "riri-{}",
                        first_block as usize * tpb as usize + slot
                    ))
                    .stack_size(256 * 1024)
                    .spawn_scoped(scope, move || run_thread(state, kernel, slot))
                    .expect("riri: failed to spawn simulated thread");
            }
        });

        if waves == 1 {
            let (recorded, recorded_complete) = state.sched.take_trace();
            trace = recorded;
            complete = recorded_complete;
        }
        if state.sched.was_aborted() {
            // A trap ends the launch, as it would on the device, so the
            // blocks behind it never run.
            aborted = true;
            break;
        }
        first_block += wave_blocks;
    }

    let report = Report {
        seed: config.seed,
        diagnostics: reporter.take(),
        aborted,
    };
    (report, trace, complete)
}

/// Runs one simulated thread.
///
/// `slot` numbers the thread within its wave, which is all the scheduler
/// knows about. The block and thread the kernel sees, and the global id the
/// clocks use, belong to the launch.
fn run_thread<F>(state: &Arc<LaunchState>, kernel: &F, slot: usize)
where
    F: Fn(&Arc<LaunchState>, &ThreadCtx<'_>) + Sync,
{
    let tpb = state.config.block.count() as usize;
    let block = state.first_block + (slot / tpb) as u32;
    let thread = (slot % tpb) as u32;
    let ctx = ThreadCtx {
        launch: state,
        block,
        thread,
        gid: block as usize * tpb + thread as usize,
        slot,
    };

    if state.sched.wait_turn(slot).is_err() {
        return;
    }
    match catch_unwind(AssertUnwindSafe(|| kernel(state, &ctx))) {
        Ok(()) => state.sched.finish(slot, &state.reporter),
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
