use std::any::Any;
use std::panic::Location;
use std::sync::{Arc, Mutex};

use crate::diag::{Access, AccessKind, Diagnostic};
use crate::dim::Dim3;
use crate::launch::LaunchState;
use crate::mem::SharedArray;
use crate::sched::AbortSignal;

/// Per-thread view of a running kernel: the equivalent of CUDA's built-in
/// `threadIdx`, `blockIdx`, `blockDim`, `gridDim`, plus `__syncthreads()`
/// and `__shared__` allocation.
pub struct ThreadCtx<'l> {
    pub(crate) launch: &'l LaunchState,
    pub(crate) block: u32,
    pub(crate) thread: u32,
    pub(crate) gid: usize,
}

impl<'l> ThreadCtx<'l> {
    pub fn thread_idx(&self) -> Dim3 {
        Dim3::from_linear(self.thread, self.launch.config.block)
    }

    pub fn block_idx(&self) -> Dim3 {
        Dim3::from_linear(self.block, self.launch.config.grid)
    }

    pub fn block_dim(&self) -> Dim3 {
        self.launch.config.block
    }

    pub fn grid_dim(&self) -> Dim3 {
        self.launch.config.grid
    }

    /// Linear thread index within the block.
    pub fn thread_linear(&self) -> usize {
        self.thread as usize
    }

    /// Linear block index within the grid.
    pub fn block_linear(&self) -> usize {
        self.block as usize
    }

    /// `blockIdx * blockDim + threadIdx`, linearised over the whole launch.
    pub fn global_linear(&self) -> usize {
        self.gid
    }

    // ------------------------------------------------------------ warps ---

    /// Lanes per warp for this launch. 32 unless the launch config says
    /// otherwise.
    pub fn warp_size(&self) -> u32 {
        self.launch.config.warp_size
    }

    /// Index of this thread's warp within its block.
    pub fn warp_id(&self) -> u32 {
        self.thread / self.warp_size()
    }

    /// This thread's lane within its warp, in `0..warp_size()`.
    pub fn lane_id(&self) -> u32 {
        self.thread % self.warp_size()
    }

    /// Mask of the lanes of this thread's warp that actually exist.
    ///
    /// This is the full mask except in a block whose size is not a multiple
    /// of the warp size, where the last warp is short.
    pub fn warp_valid_mask(&self) -> u32 {
        self.launch.sched.valid_mask(self.warp_id() as usize)
    }

    /// `__syncthreads()`. Every thread of the block must reach the same
    /// barrier; otherwise Riri reports barrier divergence.
    #[track_caller]
    pub fn sync_threads(&self) {
        self.sync_threads_at(Location::caller());
    }

    /// `sync_threads` with the barrier's source location passed in, for
    /// callers reached through a closure, where `#[track_caller]` does not
    /// survive.
    pub(crate) fn sync_threads_at(&self, loc: &'static Location<'static>) {
        if self.launch.sched.barrier(self.gid, loc, &self.launch.reporter).is_err() {
            std::panic::resume_unwind(Box::new(AbortSignal));
        }
    }

    /// A block-wide `__shared__` array. The first thread of the block to ask
    /// creates it; later calls with the same name return the same array.
    /// Elements start uninitialised: reading before writing is reported.
    ///
    /// Panics if the same name is reused with a different type or length.
    pub fn shared<T: Copy + Default + Send + 'static>(&self, name: &'static str, len: usize) -> SharedArray<T> {
        let mut map = self.launch.blocks[self.block as usize].shared.lock().unwrap();
        let entry = map
            .entry(name)
            .or_insert_with(|| Arc::new(SharedArray::<T>::new_inner(self.block, name, len)) as Arc<dyn Any + Send + Sync>)
            .clone();
        drop(map);
        let arr = SharedArray::<T>::from_any(entry)
            .unwrap_or_else(|| panic!("riri: shared array `{name}` reused with a different element type"));
        assert_eq!(arr.len(), len, "riri: shared array `{name}` reused with a different length");
        arr
    }

    // ----- crate-internal instrumentation hooks -----

    pub(crate) fn schedule_point(&self) {
        if self.launch.sched.yield_now(self.gid, &self.launch.reporter).is_err() {
            std::panic::resume_unwind(Box::new(AbortSignal));
        }
    }

    pub(crate) fn access(&self, kind: AccessKind, location: &'static Location<'static>) -> Access {
        let warp = self.warp_id();
        let (epoch, warp_epoch) = self.launch.sched.epochs(self.block as usize, warp as usize);
        Access {
            block: self.block,
            thread: self.thread,
            epoch,
            warp,
            warp_epoch,
            kind,
            location,
        }
    }

    /// The exchange slots this thread's warp uses to publish values to its
    /// mask-mates during a collective.
    pub(crate) fn warp_slots(&self) -> &Mutex<Vec<Option<Box<dyn Any + Send>>>> {
        let index = self.block as usize * self.launch.warps_per_block + self.warp_id() as usize;
        &self.launch.warps[index].slots
    }

    /// Parks at one half of a warp collective; unwinds if the launch aborts.
    pub(crate) fn warp_rendezvous(
        &self,
        phase: u8,
        mask: u32,
        at: &'static Location<'static>,
        op: &'static str,
    ) {
        let r = self
            .launch
            .sched
            .warp_rendezvous(self.gid, phase, mask, at, op, &self.launch.reporter);
        if r.is_err() {
            std::panic::resume_unwind(Box::new(AbortSignal));
        }
    }

    pub(crate) fn report(&self, d: Diagnostic) {
        self.launch.reporter.push(d);
    }

    /// Reports a fatal error and aborts the launch, like a GPU trap.
    pub(crate) fn trap(&self, d: Diagnostic) -> ! {
        self.launch.reporter.push(d);
        self.launch.sched.abort();
        std::panic::resume_unwind(Box::new(AbortSignal));
    }

    pub(crate) fn launch_id(&self) -> u64 {
        self.launch.id
    }
}
