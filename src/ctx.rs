use std::any::Any;
use std::panic::Location;
use std::sync::Arc;

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

    /// `__syncthreads()`. Every thread of the block must reach the same
    /// barrier; otherwise Riri reports barrier divergence.
    #[track_caller]
    pub fn sync_threads(&self) {
        let loc = Location::caller();
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
        if self.launch.sched.yield_now(self.gid).is_err() {
            std::panic::resume_unwind(Box::new(AbortSignal));
        }
    }

    pub(crate) fn access(&self, kind: AccessKind, location: &'static Location<'static>) -> Access {
        Access {
            block: self.block,
            thread: self.thread,
            epoch: self.launch.sched.epoch(self.block as usize),
            kind,
            location,
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
