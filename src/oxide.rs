//! A `cuda_device`-shaped surface, so a kernel written for cuda-oxide can be
//! checked by Riri without being rewritten.
//!
//! cuda-oxide kernels take their identity from hardware registers rather than
//! from a context argument: `thread::index_1d()` and friends are free
//! functions. Riri gives each simulated GPU thread its own OS thread, so the
//! same shape works here by parking the thread's share of the launch in a
//! thread local and rebuilding a [`ThreadCtx`] from it on demand. No raw
//! pointers are involved, and Riri stays free of `unsafe`.
//!
//! The intended use is one kernel source compiled two ways:
//!
//! ```ignore
//! #[cfg_attr(not(riri), cuda_device::kernel)]
//! pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
//!     if let Some((c_elem, idx)) = c.get_mut_indexed() {
//!         let i = idx.get();
//!         *c_elem = a[i] + b[i];
//!     }
//! }
//! ```
//!
//! `cfg_attr` carries the `#[kernel]` attribute only on the GPU build, which
//! is why Riri needs no proc macro of its own and keeps its empty dependency
//! list. Under Riri the same function is an ordinary Rust function, called
//! once per simulated thread by [`launch`].
//!
//! # What this checks, and what it cannot
//!
//! In cuda-oxide's Tier 1 the inputs are `&[T]`, which nothing writes during
//! a launch, and the single write path is [`DisjointSlice`], whose checked
//! accessors hand each thread a distinct index. Races are therefore ruled out
//! by construction, and Riri has nothing to add.
//!
//! The value is in Tier 2, where the guarantee is a claim rather than a
//! proof. [`DisjointSlice::get_unchecked_mut`] asserts that the caller's
//! index is unique to it, and nothing verifies that on hardware. Under Riri
//! the access is instrumented like any other, so two threads claiming the
//! same element is an ordinary data race with both source lines named. Warp
//! collectives are checked the same way as the rest of [`crate::warp`].
//!
//! Not covered yet:
//!
//! - Shared memory, deliberately. cuda-oxide's `SharedArray` is a zero-sized
//!   marker: their compiler recognises the type and backs it with storage in
//!   address space 3, and every accessor on it is `unreachable!` off-device.
//!   Riri would have to supply storage of its own, and `Index` hands out
//!   references into storage that starts uninitialised, which needs
//!   `MaybeUninit` behind `unsafe`. Use [`ThreadCtx::shared`] instead, which
//!   is checked exactly as the rest of Riri is.
//! - 2D and tiled index spaces, the `_sync` shuffle forms, managed barriers,
//!   clusters, and TMA.
//!
//! # Fidelity
//!
//! `cuda-device` is not published to crates.io and its workspace pins a
//! specific nightly, so Riri cannot depend on it and this surface is written
//! from the published API reference rather than compiled against the real
//! crate. Signatures can drift. Treat a mismatch as a bug in Riri.

use std::cell::RefCell;
use std::marker::PhantomData;
use std::panic::Location;
use std::sync::Arc;

use crate::ctx::ThreadCtx;
use crate::diag::Report;
use crate::launch::{LaunchConfig, LaunchState};
use crate::mem::{ElemMut, GlobalBuf};
use crate::sched::{Choices, SplitMix64};

struct Bound {
    state: Arc<LaunchState>,
    block: u32,
    thread: u32,
    gid: usize,
}

thread_local! {
    static CURRENT: RefCell<Option<Bound>> = const { RefCell::new(None) };
}

/// Rebuilds this thread's context and hands it to `f`.
fn with_ctx<R>(f: impl FnOnce(&ThreadCtx<'_>) -> R) -> R {
    CURRENT.with(|cell| {
        let bound = cell.borrow();
        let bound = bound
            .as_ref()
            .expect("riri: a cuda-oxide device function was called outside riri::oxide::launch");
        let ctx = ThreadCtx {
            launch: &bound.state,
            block: bound.block,
            thread: bound.thread,
            gid: bound.gid,
        };
        f(&ctx)
    })
}

/// Clears the binding however the kernel leaves, including by unwinding.
struct Unbind;

impl Drop for Unbind {
    fn drop(&mut self) {
        CURRENT.with(|cell| *cell.borrow_mut() = None);
    }
}

/// Runs a cuda-oxide shaped kernel once per simulated GPU thread.
///
/// The closure takes no arguments: the kernel finds its own identity through
/// [`thread`], exactly as it does on the GPU. Call the kernel function inside
/// it, handing over whatever it takes by value.
///
/// ```
/// use riri::oxide::{self, DisjointSlice};
/// use riri::{GlobalBuf, LaunchConfig};
///
/// fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
///     if let Some((mut c_elem, idx)) = c.get_mut_indexed() {
///         let i = idx.get();
///         *c_elem = a[i] + b[i];
///     }
/// }
///
/// let a = vec![1.0f32; 64];
/// let b = vec![2.0f32; 64];
/// let out = GlobalBuf::new("c", vec![0.0f32; 64]);
/// let c = DisjointSlice::new(&out);
///
/// let report = oxide::launch(&LaunchConfig::new(2, 32), || {
///     vecadd(&a, &b, c.clone())
/// });
///
/// report.assert_clean();
/// assert!(out.to_vec().iter().all(|&x| x == 3.0));
/// ```
pub fn launch<F>(config: &LaunchConfig, kernel: F) -> Report
where
    F: Fn() + Sync,
{
    crate::launch::run_shared(
        config,
        Choices::Random(SplitMix64(config.seed)),
        |state, ctx| {
            CURRENT.with(|cell| {
                *cell.borrow_mut() = Some(Bound {
                    state: Arc::clone(state),
                    block: ctx.block,
                    thread: ctx.thread,
                    gid: ctx.gid,
                });
            });
            let _unbind = Unbind;
            kernel();
        },
    )
    .0
}

/// An opaque witness that a thread owns one index.
///
/// As in cuda-oxide it has no public constructor, cannot be copied, cloned,
/// sent, or shared, and so cannot be passed between threads to claim another
/// thread's element.
pub struct ThreadIndex<'k> {
    index: usize,
    _not_send: PhantomData<*const ()>,
    _scope: PhantomData<&'k ()>,
}

impl ThreadIndex<'_> {
    /// The index this witness stands for.
    pub fn get(&self) -> usize {
        self.index
    }
}

/// Thread identification and block-level synchronisation.
#[allow(non_snake_case)]
pub mod thread {
    use super::{with_ctx, ThreadIndex};
    use std::marker::PhantomData;
    use std::panic::Location;

    fn witness<'k>(index: usize) -> ThreadIndex<'k> {
        ThreadIndex {
            index,
            _not_send: PhantomData,
            _scope: PhantomData,
        }
    }

    /// `blockIdx.x * blockDim.x + threadIdx.x`, as an owned witness.
    pub fn index_1d<'k>() -> ThreadIndex<'k> {
        witness(with_ctx(|t| t.global_linear()))
    }

    pub fn threadIdx_x() -> u32 {
        with_ctx(|t| t.thread_idx().x)
    }

    pub fn threadIdx_y() -> u32 {
        with_ctx(|t| t.thread_idx().y)
    }

    pub fn threadIdx_z() -> u32 {
        with_ctx(|t| t.thread_idx().z)
    }

    pub fn blockIdx_x() -> u32 {
        with_ctx(|t| t.block_idx().x)
    }

    pub fn blockIdx_y() -> u32 {
        with_ctx(|t| t.block_idx().y)
    }

    pub fn blockIdx_z() -> u32 {
        with_ctx(|t| t.block_idx().z)
    }

    pub fn blockDim_x() -> u32 {
        with_ctx(|t| t.block_dim().x)
    }

    pub fn blockDim_y() -> u32 {
        with_ctx(|t| t.block_dim().y)
    }

    pub fn blockDim_z() -> u32 {
        with_ctx(|t| t.block_dim().z)
    }

    pub fn gridDim_x() -> u32 {
        with_ctx(|t| t.grid_dim().x)
    }

    /// Lanes per warp, as `warpSize`.
    pub fn warp_size() -> u32 {
        with_ctx(|t| t.warp_size())
    }

    /// `__syncthreads()`.
    #[track_caller]
    pub fn sync_threads() {
        let at = Location::caller();
        with_ctx(|t| t.sync_threads_at(at));
    }
}

/// The write path: a slice whose elements are claimed one per thread.
///
/// Backed by a [`GlobalBuf`], so every access through it is instrumented.
/// Cloning is cheap and shares the buffer, which is how each simulated thread
/// receives its own handle the way a GPU kernel receives its own argument.
pub struct DisjointSlice<T> {
    buf: GlobalBuf<T>,
}

impl<T> Clone for DisjointSlice<T> {
    fn clone(&self) -> Self {
        DisjointSlice {
            buf: self.buf.clone(),
        }
    }
}

impl<T: Copy + Send + 'static> DisjointSlice<T> {
    /// Wraps an instrumented buffer as the kernel's write path.
    pub fn new(buf: &GlobalBuf<T>) -> Self {
        DisjointSlice { buf: buf.clone() }
    }

    pub fn len(&self) -> usize {
        self.buf.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    /// Mints this thread's witness and resolves it in one call.
    ///
    /// `None` for a thread whose index is past the end, which is how a launch
    /// rounded up to whole blocks drops its tail.
    #[track_caller]
    pub fn get_mut_indexed(&mut self) -> Option<(ElemMut<'_, T>, ThreadIndex<'_>)> {
        let at = Location::caller();
        let index = with_ctx(|t| t.global_linear());
        let elem = with_ctx(|t| self.buf.elem_mut(t, index, true, at))?;
        Some((
            elem,
            ThreadIndex {
                index,
                _not_send: PhantomData,
                _scope: PhantomData,
            },
        ))
    }

    /// Resolves an explicit witness, bounds checked.
    #[track_caller]
    pub fn get_mut(&mut self, idx: ThreadIndex<'_>) -> Option<ElemMut<'_, T>> {
        let at = Location::caller();
        let index = idx.get();
        with_ctx(|t| self.buf.elem_mut(t, index, true, at))
    }

    /// Claims an element by raw index, with no check that the claim is unique.
    ///
    /// On hardware nothing verifies that two threads did not pass the same
    /// index. Under Riri the access is instrumented, so they do not get away
    /// with it: the second claim is reported as a data race naming both
    /// lines. Out of range is a trap rather than `None`, matching the
    /// unchecked contract.
    ///
    /// This mirrors an `unsafe fn` in cuda-oxide. It is safe here because
    /// Riri checks rather than trusts, which is the entire point of running
    /// the kernel under it.
    #[track_caller]
    pub fn get_unchecked_mut(&mut self, index: usize) -> ElemMut<'_, T> {
        let at = Location::caller();
        with_ctx(|t| self.buf.elem_mut(t, index, false, at))
            .expect("riri: unchecked access is never None")
    }
}

/// Warp-level primitives.
///
/// cuda-oxide's unsuffixed forms take no member mask, so Riri supplies the
/// whole warp, which is what the hardware instruction they lower to assumes.
/// A warp that is not converged at one of these is therefore reported, which
/// is the bug those forms invite.
pub mod warp {
    use super::with_ctx;
    use std::panic::Location;

    /// This thread's lane within its warp.
    pub fn lane_id() -> u32 {
        with_ctx(|t| t.lane_id())
    }

    /// This thread's warp within its block.
    pub fn warp_id() -> u32 {
        with_ctx(|t| t.warp_id())
    }

    #[track_caller]
    pub fn shuffle(val: u32, src_lane: u32) -> u32 {
        let at = Location::caller();
        with_ctx(|t| crate::warp::shfl_sync_at(t, t.warp_valid_mask(), val, src_lane, at))
    }

    #[track_caller]
    pub fn shuffle_f32(val: f32, src_lane: u32) -> f32 {
        let at = Location::caller();
        with_ctx(|t| crate::warp::shfl_sync_at(t, t.warp_valid_mask(), val, src_lane, at))
    }

    #[track_caller]
    pub fn shuffle_up(val: u32, delta: u32) -> u32 {
        let at = Location::caller();
        with_ctx(|t| crate::warp::shfl_up_sync_at(t, t.warp_valid_mask(), val, delta, at))
    }

    #[track_caller]
    pub fn shuffle_up_f32(val: f32, delta: u32) -> f32 {
        let at = Location::caller();
        with_ctx(|t| crate::warp::shfl_up_sync_at(t, t.warp_valid_mask(), val, delta, at))
    }

    #[track_caller]
    pub fn shuffle_down(val: u32, delta: u32) -> u32 {
        let at = Location::caller();
        with_ctx(|t| crate::warp::shfl_down_sync_at(t, t.warp_valid_mask(), val, delta, at))
    }

    #[track_caller]
    pub fn shuffle_down_f32(val: f32, delta: u32) -> f32 {
        let at = Location::caller();
        with_ctx(|t| crate::warp::shfl_down_sync_at(t, t.warp_valid_mask(), val, delta, at))
    }

    #[track_caller]
    pub fn shuffle_xor(val: u32, lane_mask: u32) -> u32 {
        let at = Location::caller();
        with_ctx(|t| crate::warp::shfl_xor_sync_at(t, t.warp_valid_mask(), val, lane_mask, at))
    }

    #[track_caller]
    pub fn shuffle_xor_f32(val: f32, lane_mask: u32) -> f32 {
        let at = Location::caller();
        with_ctx(|t| crate::warp::shfl_xor_sync_at(t, t.warp_valid_mask(), val, lane_mask, at))
    }

    #[track_caller]
    pub fn all(predicate: bool) -> bool {
        let at = Location::caller();
        with_ctx(|t| {
            crate::warp::ballot(t, t.warp_valid_mask(), predicate, "all", at) == t.warp_valid_mask()
        })
    }

    #[track_caller]
    pub fn any(predicate: bool) -> bool {
        let at = Location::caller();
        with_ctx(|t| crate::warp::ballot(t, t.warp_valid_mask(), predicate, "any", at) != 0)
    }

    #[track_caller]
    pub fn ballot(predicate: bool) -> u32 {
        let at = Location::caller();
        with_ctx(|t| crate::warp::ballot(t, t.warp_valid_mask(), predicate, "ballot", at))
    }

    #[track_caller]
    pub fn popc(predicate: bool) -> u32 {
        ballot(predicate).count_ones()
    }
}
