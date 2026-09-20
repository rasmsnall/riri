//! Warp-level collectives, with convergence checking.
//!
//! On a GPU the threads of a block execute in warps, and the warp-level
//! primitives (`__shfl_sync`, `__ballot_sync`, `__syncwarp`, ...) exchange
//! values directly between lanes without going through memory. Each takes a
//! *member mask* naming the lanes that take part, and CUDA requires every
//! named lane to reach that same instruction. A lane that branched away,
//! exited, or stopped at a different collective makes the result undefined:
//! on real hardware this is the classic hang or silently wrong reduction.
//!
//! Riri checks that contract instead of trusting it. Every collective is a
//! rendezvous: lanes park until all their mask-mates arrive, and if that can
//! never happen Riri reports [`Diagnostic::WarpDivergence`] naming the lanes
//! that never showed up, rather than deadlocking.
//!
//! It also catches the mistakes that do not hang but corrupt results:
//! mask-mates that disagree about who is taking part
//! ([`Diagnostic::WarpMaskMismatch`]), and shuffles that read from a lane
//! outside the mask ([`Diagnostic::WarpLaneError`]).
//!
//! # Wrapping a collective
//!
//! Collectives are told apart by the call site, which `#[track_caller]` makes
//! the caller's line rather than a line in this module. A helper of your own
//! that wraps a collective should carry `#[track_caller]` for the same
//! reason:
//!
//! ```
//! use riri::{warp, ThreadCtx};
//!
//! #[track_caller]
//! fn broadcast(t: &ThreadCtx<'_>, mask: u32, v: u32) -> u32 {
//!     warp::shfl_sync(t, mask, v, 0)
//! }
//! ```
//!
//! Without it every call site reports the helper's own line, so two groups
//! that diverged before calling it look like one converged group. They
//! rendezvous, and the divergence goes unreported.
//!
//! ```
//! use riri::{launch, warp, GlobalBuf, LaunchConfig};
//!
//! let out = GlobalBuf::new("out", vec![0u32; 32]);
//! let report = launch(&LaunchConfig::new(1, 32).seed(1), |t| {
//!     // Every lane reads lane 0, so every lane must take part.
//!     let v = warp::shfl_sync(t, warp::FULL_MASK, t.lane_id(), 0);
//!     out.write(t, t.global_linear(), v);
//! });
//!
//! report.assert_clean();
//! assert!(out.to_vec().iter().all(|&x| x == 0));
//! ```
//!
//! [`Diagnostic::WarpDivergence`]: crate::Diagnostic::WarpDivergence
//! [`Diagnostic::WarpMaskMismatch`]: crate::Diagnostic::WarpMaskMismatch
//! [`Diagnostic::WarpLaneError`]: crate::Diagnostic::WarpLaneError

use std::any::Any;
use std::panic::Location;

use crate::ctx::ThreadCtx;
use crate::diag::{Diagnostic, LaneProblem};
use crate::sched::{PHASE_ARRIVE, PHASE_DEPART};

/// Lanes per warp unless [`LaunchConfig::warp_size`] says otherwise.
///
/// [`LaunchConfig::warp_size`]: crate::LaunchConfig::warp_size
pub const WARP_SIZE: u32 = 32;

/// Mask naming every lane of a full warp.
///
/// In a block whose size is not a multiple of the warp size the last warp is
/// short, and this mask names lanes that do not exist. Use
/// [`ThreadCtx::warp_valid_mask`] there instead.
pub const FULL_MASK: u32 = u32::MAX;

/// Values published by the lanes taking part in a collective.
struct Lanes<'a> {
    slots: &'a [Option<Box<dyn Any + Send>>],
    op: &'static str,
}

impl Lanes<'_> {
    fn get<T: Copy + 'static>(&self, lane: u32) -> T {
        let slot = self
            .slots
            .get(lane as usize)
            .and_then(|s| s.as_ref())
            .unwrap_or_else(|| panic!("riri: lane {lane} published no value to `{}`", self.op));
        *slot.downcast_ref::<T>().unwrap_or_else(|| {
            panic!("riri: `{}` called with different value types on different lanes", self.op)
        })
    }
}

fn lane_error(
    ctx: &ThreadCtx<'_>,
    op: &'static str,
    at: &'static Location<'static>,
    mask: u32,
    problem: LaneProblem,
) -> Diagnostic {
    Diagnostic::WarpLaneError {
        block: ctx.block_linear() as u32,
        warp: ctx.warp_id(),
        op,
        at,
        lane: ctx.lane_id(),
        mask,
        problem,
    }
}

/// Runs one warp collective: publish a value, wait for every lane in `mask`,
/// let `combine` read what the others published, then release together.
///
/// The second rendezvous matters: without it a lane could race ahead into
/// the next collective and overwrite its slot while a mask-mate is still
/// reading the current one.
fn collective<T, R>(
    ctx: &ThreadCtx<'_>,
    mask: u32,
    op: &'static str,
    at: &'static Location<'static>,
    value: T,
    combine: impl FnOnce(&Lanes<'_>) -> R,
) -> R
where
    T: Send + 'static,
{
    // A scheduling point, so collectives interleave with memory operations.
    ctx.schedule_point();

    let lane = ctx.lane_id();
    let valid = ctx.warp_valid_mask();
    if mask & !valid != 0 {
        ctx.trap(lane_error(ctx, op, at, mask, LaneProblem::MaskOutsideBlock { valid }));
    }
    if mask & (1u32 << lane) == 0 {
        ctx.trap(lane_error(ctx, op, at, mask, LaneProblem::CallerNotInMask));
    }

    ctx.warp_slots().lock().unwrap()[lane as usize] = Some(Box::new(value));
    ctx.warp_rendezvous(PHASE_ARRIVE, mask, at, op);

    let result = {
        let slots = ctx.warp_slots().lock().unwrap();
        combine(&Lanes { slots: &slots, op })
    };

    ctx.warp_rendezvous(PHASE_DEPART, mask, at, op);
    result
}

/// Reads `value` from another lane, reporting if that lane is not a member.
fn shuffle<T: Copy + Send + 'static>(
    ctx: &ThreadCtx<'_>,
    mask: u32,
    op: &'static str,
    at: &'static Location<'static>,
    value: T,
    src: Option<u32>,
) -> T {
    collective(ctx, mask, op, at, value, |lanes| {
        // No source lane means the shuffle reached past the end of the warp,
        // which CUDA defines as returning the lane's own value.
        let Some(src) = src else { return value };
        if mask & (1u32 << src) == 0 {
            ctx.report(lane_error(ctx, op, at, mask, LaneProblem::SourceLaneNotInMask { src }));
            return value;
        }
        lanes.get::<T>(src)
    })
}

/// `__shfl_sync`: every lane reads the value held by `src_lane`.
///
/// `src_lane` is taken modulo the warp size, as on the GPU.
#[track_caller]
pub fn shfl_sync<T: Copy + Send + 'static>(
    ctx: &ThreadCtx<'_>,
    mask: u32,
    value: T,
    src_lane: u32,
) -> T {
    shfl_sync_at(ctx, mask, value, src_lane, Location::caller())
}

pub(crate) fn shfl_sync_at<T: Copy + Send + 'static>(
    ctx: &ThreadCtx<'_>,
    mask: u32,
    value: T,
    src_lane: u32,
    at: &'static Location<'static>,
) -> T {
    let src = src_lane % ctx.warp_size();
    shuffle(ctx, mask, "shfl_sync", at, value, Some(src))
}

/// `__shfl_up_sync`: reads from the lane `delta` below this one.
///
/// Lanes with no such neighbour keep their own value.
#[track_caller]
pub fn shfl_up_sync<T: Copy + Send + 'static>(
    ctx: &ThreadCtx<'_>,
    mask: u32,
    value: T,
    delta: u32,
) -> T {
    shfl_up_sync_at(ctx, mask, value, delta, Location::caller())
}

pub(crate) fn shfl_up_sync_at<T: Copy + Send + 'static>(
    ctx: &ThreadCtx<'_>,
    mask: u32,
    value: T,
    delta: u32,
    at: &'static Location<'static>,
) -> T {
    let src = ctx.lane_id().checked_sub(delta);
    shuffle(ctx, mask, "shfl_up_sync", at, value, src)
}

/// `__shfl_down_sync`: reads from the lane `delta` above this one.
///
/// Lanes with no such neighbour keep their own value.
#[track_caller]
pub fn shfl_down_sync<T: Copy + Send + 'static>(
    ctx: &ThreadCtx<'_>,
    mask: u32,
    value: T,
    delta: u32,
) -> T {
    shfl_down_sync_at(ctx, mask, value, delta, Location::caller())
}

pub(crate) fn shfl_down_sync_at<T: Copy + Send + 'static>(
    ctx: &ThreadCtx<'_>,
    mask: u32,
    value: T,
    delta: u32,
    at: &'static Location<'static>,
) -> T {
    let src = ctx.lane_id().checked_add(delta).filter(|&s| s < ctx.warp_size());
    shuffle(ctx, mask, "shfl_down_sync", at, value, src)
}

/// `__shfl_xor_sync`: reads from the lane whose id is this one XOR
/// `lane_mask`, the butterfly exchange used by warp reductions.
#[track_caller]
pub fn shfl_xor_sync<T: Copy + Send + 'static>(
    ctx: &ThreadCtx<'_>,
    mask: u32,
    value: T,
    lane_mask: u32,
) -> T {
    shfl_xor_sync_at(ctx, mask, value, lane_mask, Location::caller())
}

pub(crate) fn shfl_xor_sync_at<T: Copy + Send + 'static>(
    ctx: &ThreadCtx<'_>,
    mask: u32,
    value: T,
    lane_mask: u32,
    at: &'static Location<'static>,
) -> T {
    let src = Some(ctx.lane_id() ^ lane_mask).filter(|&s| s < ctx.warp_size());
    shuffle(ctx, mask, "shfl_xor_sync", at, value, src)
}

pub(crate) fn ballot(
    ctx: &ThreadCtx<'_>,
    mask: u32,
    pred: bool,
    op: &'static str,
    at: &'static Location<'static>,
) -> u32 {
    collective(ctx, mask, op, at, pred, |lanes| {
        (0..32u32)
            .filter(|l| mask & (1u32 << l) != 0 && lanes.get::<bool>(*l))
            .fold(0u32, |acc, l| acc | (1u32 << l))
    })
}

/// `__ballot_sync`: one bit per member lane, set where `pred` is true.
#[track_caller]
pub fn ballot_sync(ctx: &ThreadCtx<'_>, mask: u32, pred: bool) -> u32 {
    ballot(ctx, mask, pred, "ballot_sync", Location::caller())
}

/// `__any_sync`: true if `pred` holds on any member lane.
#[track_caller]
pub fn any_sync(ctx: &ThreadCtx<'_>, mask: u32, pred: bool) -> bool {
    ballot(ctx, mask, pred, "any_sync", Location::caller()) != 0
}

/// `__all_sync`: true if `pred` holds on every member lane.
#[track_caller]
pub fn all_sync(ctx: &ThreadCtx<'_>, mask: u32, pred: bool) -> bool {
    ballot(ctx, mask, pred, "all_sync", Location::caller()) == mask
}

/// `__syncwarp`: reconverge the member lanes without exchanging a value.
///
/// Like the other collectives this orders memory accesses made by the warp,
/// but only when `mask` names the whole warp.
#[track_caller]
pub fn sync_warp(ctx: &ThreadCtx<'_>, mask: u32) {
    sync_warp_at(ctx, mask, Location::caller())
}

pub(crate) fn sync_warp_at(ctx: &ThreadCtx<'_>, mask: u32, at: &'static Location<'static>) {
    collective(ctx, mask, "sync_warp", at, (), |_| ())
}
