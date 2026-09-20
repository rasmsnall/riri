use std::any::Any;
use std::ops::{Add, Deref, DerefMut, Sub};
use std::panic::Location;
use std::sync::{Arc, Mutex, MutexGuard};

use crate::ctx::ThreadCtx;
use crate::diag::{AccessKind, Diagnostic, MemSpace};
use crate::shadow::Cell;
use crate::sync::Ordering;

/// Shared implementation for any instrumented array.
struct Instrumented<T> {
    data: Vec<T>,
    shadow: Vec<Cell>,
    /// Launch the shadow state belongs to; reset when a new launch touches it.
    launch_id: u64,
}

/// What the access path needs to know about a buffer: where its data lives,
/// how to name it in a diagnostic, and how to clear its shadow state when a
/// new launch first touches it.
///
/// These three travel together at every call site, so they are one argument.
struct Buffer<'a, T, S, R> {
    mem: &'a Mutex<Instrumented<T>>,
    space: S,
    reset: R,
}

fn checked_access<T, S, R>(
    buf: Buffer<'_, T, S, R>,
    ctx: &ThreadCtx<'_>,
    index: usize,
    kind: AccessKind,
    location: &'static Location<'static>,
    ordering: Option<Ordering>,
    op: impl FnOnce(&mut T) -> T,
) -> T
where
    T: Copy,
    S: Fn() -> MemSpace,
    R: Fn(&mut Instrumented<T>),
{
    let Buffer { mem, space, reset } = buf;

    // Let another thread run first: this is where interleavings come from.
    ctx.schedule_point();

    let mut m = mem.lock().unwrap();
    if m.launch_id != ctx.launch_id() {
        reset(&mut m);
        m.launch_id = ctx.launch_id();
    }
    let len = m.data.len();
    if index >= len {
        drop(m);
        let (access, _) = ctx.access(kind, location);
        ctx.trap(Diagnostic::OutOfBounds {
            space: space(),
            index,
            len,
            access,
        });
    }

    // An acquire has to land before this access is weighed against earlier
    // ones, or it would not order the very access that performed it.
    if let Some(ordering) = ordering {
        let published = m.shadow[index].release.clone();
        ctx.acquire_from(&published, ordering);
    }

    let (acc, now) = ctx.access(kind, location);
    let cell = &mut m.shadow[index];
    let was_init = cell.init;
    let conflict = match kind {
        AccessKind::Read => cell.on_read(acc, &now),
        AccessKind::Write | AccessKind::Atomic => cell.on_write(acc, &now),
    };
    let value = op(&mut m.data[index]);
    if let Some(ordering) = ordering {
        ctx.release_into(&mut m.shadow[index].release, ordering);
    }
    drop(m);

    if kind != AccessKind::Write && !was_init {
        ctx.report(Diagnostic::UninitRead {
            space: space(),
            index,
            access: acc,
        });
    }
    if let Some(first) = conflict {
        ctx.report(Diagnostic::DataRace {
            space: space(),
            index,
            first,
            second: acc,
        });
    }
    value
}

// ---------------------------------------------------------------- global ---

/// A buffer in simulated global (device) memory.
///
/// Cheap to clone (reference-counted), so kernels can capture it by value or
/// by reference. Contents persist across launches; shadow state does not.
pub struct GlobalBuf<T> {
    name: Arc<str>,
    mem: Arc<Mutex<Instrumented<T>>>,
}

impl<T> Clone for GlobalBuf<T> {
    fn clone(&self) -> Self {
        GlobalBuf {
            name: self.name.clone(),
            mem: self.mem.clone(),
        }
    }
}

impl<T: Copy + Send + 'static> GlobalBuf<T> {
    /// Creates a buffer initialised from host data (like a host-to-device copy).
    pub fn new(name: &str, data: Vec<T>) -> Self {
        let shadow = vec![Cell::initialised(); data.len()];
        GlobalBuf {
            name: name.into(),
            mem: Arc::new(Mutex::new(Instrumented {
                data,
                shadow,
                launch_id: 0,
            })),
        }
    }

    /// Creates a buffer of device memory that nothing has written yet, the way
    /// an output allocation arrives before a kernel fills it.
    ///
    /// Reading an element before some thread has written it is reported as
    /// [`Diagnostic::UninitRead`], which is the check Compute Sanitizer's
    /// `initcheck` performs. The usual bug it finds is a kernel that fills
    /// only part of its output, leaving the rest to be read as whatever the
    /// allocator last left there.
    ///
    /// The elements hold `T::default()` so that something is there to read;
    /// those values carry no meaning, and a read that reaches them is the
    /// fault being reported. Once written, an element stays initialised for
    /// the rest of the buffer's life, including across later launches, since
    /// that is how device memory behaves.
    ///
    /// [`Diagnostic::UninitRead`]: crate::Diagnostic::UninitRead
    pub fn uninit(name: &str, len: usize) -> Self
    where
        T: Default,
    {
        GlobalBuf {
            name: name.into(),
            mem: Arc::new(Mutex::new(Instrumented {
                data: (0..len).map(|_| T::default()).collect(),
                shadow: vec![Cell::default(); len],
                launch_id: 0,
            })),
        }
    }

    pub fn len(&self) -> usize {
        self.mem.lock().unwrap().data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Copies the contents back to the host.
    pub fn to_vec(&self) -> Vec<T> {
        self.mem.lock().unwrap().data.clone()
    }

    fn space(&self) -> MemSpace {
        MemSpace::Global {
            buffer: self.name.clone(),
        }
    }

    fn reset(m: &mut Instrumented<T>) {
        // Global memory keeps its (initialised) values across launches;
        // only the access history is launch-local.
        for c in &mut m.shadow {
            let init = c.init;
            *c = Cell::default();
            c.init = init;
        }
    }

    #[track_caller]
    pub fn read(&self, ctx: &ThreadCtx<'_>, index: usize) -> T {
        checked_access(
            Buffer {
                mem: &self.mem,
                space: || self.space(),
                reset: Self::reset,
            },
            ctx,
            index,
            AccessKind::Read,
            Location::caller(),
            None,
            |v| *v,
        )
    }

    #[track_caller]
    pub fn write(&self, ctx: &ThreadCtx<'_>, index: usize, value: T) {
        checked_access(
            Buffer {
                mem: &self.mem,
                space: || self.space(),
                reset: Self::reset,
            },
            ctx,
            index,
            AccessKind::Write,
            Location::caller(),
            None,
            |v| {
                *v = value;
                value
            },
        );
    }
}

impl<T: Copy + Send + 'static> GlobalBuf<T> {
    /// An atomic read. With [`Ordering::Acquire`] it takes on whatever a
    /// matching release published.
    #[track_caller]
    pub fn atomic_load(&self, ctx: &ThreadCtx<'_>, index: usize, ordering: Ordering) -> T {
        self.atomic_load_at(ctx, index, ordering, Location::caller())
    }

    pub(crate) fn atomic_load_at(
        &self,
        ctx: &ThreadCtx<'_>,
        index: usize,
        ordering: Ordering,
        at: &'static Location<'static>,
    ) -> T {
        self.atomic_rmw(ctx, index, ordering, at, |v| *v)
    }

    /// An atomic write. With [`Ordering::Release`], or after a release fence,
    /// it publishes everything this thread did beforehand.
    #[track_caller]
    pub fn atomic_store(&self, ctx: &ThreadCtx<'_>, index: usize, value: T, ordering: Ordering) {
        self.atomic_store_at(ctx, index, value, ordering, Location::caller());
    }

    pub(crate) fn atomic_store_at(
        &self,
        ctx: &ThreadCtx<'_>,
        index: usize,
        value: T,
        ordering: Ordering,
        at: &'static Location<'static>,
    ) {
        self.atomic_rmw(ctx, index, ordering, at, |v| {
            *v = value;
            value
        });
    }
}

impl<T: Copy + Send + 'static> GlobalBuf<T> {
    /// The read-modify-write every atomic below is a shape of.
    ///
    /// The update runs while the element is held, so nothing can interleave
    /// between reading the old value and writing the new one. That is what
    /// makes it atomic here, and it is why a compare-and-exchange can decide
    /// and act in one step.
    /// `at` is passed in rather than captured, because the shim reaches these
    /// through a closure and `#[track_caller]` does not survive one.
    pub(crate) fn atomic_rmw(
        &self,
        ctx: &ThreadCtx<'_>,
        index: usize,
        ordering: Ordering,
        at: &'static Location<'static>,
        update: impl FnOnce(&mut T) -> T,
    ) -> T {
        checked_access(
            Buffer {
                mem: &self.mem,
                space: || self.space(),
                reset: Self::reset,
            },
            ctx,
            index,
            AccessKind::Atomic,
            at,
            Some(ordering),
            update,
        )
    }

    /// `atomicExch`: writes `value` and returns what was there.
    #[track_caller]
    pub fn atomic_swap(
        &self,
        ctx: &ThreadCtx<'_>,
        index: usize,
        value: T,
        ordering: Ordering,
    ) -> T {
        self.atomic_swap_at(ctx, index, value, ordering, Location::caller())
    }

    pub(crate) fn atomic_swap_at(
        &self,
        ctx: &ThreadCtx<'_>,
        index: usize,
        value: T,
        ordering: Ordering,
        at: &'static Location<'static>,
    ) -> T {
        self.atomic_rmw(ctx, index, ordering, at, |v| std::mem::replace(v, value))
    }
}

impl<T: Copy + Send + PartialEq + 'static> GlobalBuf<T> {
    /// `atomicCAS`: writes `new` only if the element still holds `current`.
    ///
    /// Returns the previous value either way, as `Ok` when the exchange
    /// happened and `Err` when it did not, matching `std`.
    #[track_caller]
    pub fn atomic_compare_exchange(
        &self,
        ctx: &ThreadCtx<'_>,
        index: usize,
        current: T,
        new: T,
        ordering: Ordering,
    ) -> Result<T, T> {
        self.atomic_compare_exchange_at(ctx, index, current, new, ordering, Location::caller())
    }

    pub(crate) fn atomic_compare_exchange_at(
        &self,
        ctx: &ThreadCtx<'_>,
        index: usize,
        current: T,
        new: T,
        ordering: Ordering,
        at: &'static Location<'static>,
    ) -> Result<T, T> {
        let mut exchanged = false;
        let previous = self.atomic_rmw(ctx, index, ordering, at, |v| {
            let previous = *v;
            if previous == current {
                *v = new;
                exchanged = true;
            }
            previous
        });
        if exchanged {
            Ok(previous)
        } else {
            Err(previous)
        }
    }
}

impl<T: Copy + Send + Ord + 'static> GlobalBuf<T> {
    /// `atomicMin`, returning the previous value.
    #[track_caller]
    pub fn atomic_min(&self, ctx: &ThreadCtx<'_>, index: usize, value: T, ordering: Ordering) -> T {
        self.atomic_min_at(ctx, index, value, ordering, Location::caller())
    }

    pub(crate) fn atomic_min_at(
        &self,
        ctx: &ThreadCtx<'_>,
        index: usize,
        value: T,
        ordering: Ordering,
        at: &'static Location<'static>,
    ) -> T {
        self.atomic_rmw(ctx, index, ordering, at, |v| {
            let previous = *v;
            if value < previous {
                *v = value;
            }
            previous
        })
    }

    /// `atomicMax`, returning the previous value.
    #[track_caller]
    pub fn atomic_max(&self, ctx: &ThreadCtx<'_>, index: usize, value: T, ordering: Ordering) -> T {
        self.atomic_max_at(ctx, index, value, ordering, Location::caller())
    }

    pub(crate) fn atomic_max_at(
        &self,
        ctx: &ThreadCtx<'_>,
        index: usize,
        value: T,
        ordering: Ordering,
        at: &'static Location<'static>,
    ) -> T {
        self.atomic_rmw(ctx, index, ordering, at, |v| {
            let previous = *v;
            if value > previous {
                *v = value;
            }
            previous
        })
    }
}

impl<T: Copy + Send + Sub<Output = T> + 'static> GlobalBuf<T> {
    /// `atomicSub`, returning the previous value.
    #[track_caller]
    pub fn atomic_sub(&self, ctx: &ThreadCtx<'_>, index: usize, value: T, ordering: Ordering) -> T {
        self.atomic_sub_at(ctx, index, value, ordering, Location::caller())
    }

    pub(crate) fn atomic_sub_at(
        &self,
        ctx: &ThreadCtx<'_>,
        index: usize,
        value: T,
        ordering: Ordering,
        at: &'static Location<'static>,
    ) -> T {
        self.atomic_rmw(ctx, index, ordering, at, |v| {
            let previous = *v;
            *v = previous - value;
            previous
        })
    }
}

impl<T: Copy + Send + Add<Output = T> + 'static> GlobalBuf<T> {
    /// `atomicAdd`: returns the previous value. Atomics never race with each
    /// other, only with concurrent plain accesses.
    #[track_caller]
    pub fn atomic_add(&self, ctx: &ThreadCtx<'_>, index: usize, value: T, ordering: Ordering) -> T {
        self.atomic_add_at(ctx, index, value, ordering, Location::caller())
    }

    pub(crate) fn atomic_add_at(
        &self,
        ctx: &ThreadCtx<'_>,
        index: usize,
        value: T,
        ordering: Ordering,
        at: &'static Location<'static>,
    ) -> T {
        self.atomic_rmw(ctx, index, ordering, at, |v| {
            let previous = *v;
            *v = previous + value;
            previous
        })
    }
}

/// A borrowed element of an instrumented buffer, held for as long as the
/// caller keeps it.
///
/// This exists so a surface that hands out `&mut T`, as cuda-oxide's
/// `DisjointSlice::get_mut` does, can be offered without Riri giving up
/// either its instrumentation or its freedom from `unsafe`. The lock is held
/// for the life of the borrow, which is sound here because only the thread
/// holding the turn runs. Holding one across a barrier would wedge the
/// launch, so do not.
pub struct ElemMut<'a, T> {
    guard: MutexGuard<'a, Instrumented<T>>,
    index: usize,
}

impl<T> Deref for ElemMut<'_, T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.guard.data[self.index]
    }
}

impl<T> DerefMut for ElemMut<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.guard.data[self.index]
    }
}

impl<T: Copy + Send + 'static> GlobalBuf<T> {
    /// Borrows one element for writing, recording the access first.
    ///
    /// `checked` decides what an out-of-range index means: `None` for a
    /// bounds-checked surface, or a trap for an unchecked one.
    ///
    /// `location` is passed in rather than captured, because the caller is
    /// usually reached through a closure and `#[track_caller]` does not
    /// survive one. Reporting a race inside Riri instead of in the kernel
    /// would defeat the point of reporting it at all.
    pub(crate) fn elem_mut(
        &self,
        ctx: &ThreadCtx<'_>,
        index: usize,
        checked: bool,
        location: &'static Location<'static>,
    ) -> Option<ElemMut<'_, T>> {
        ctx.schedule_point();

        let (acc, now) = ctx.access(AccessKind::Write, location);
        let mut m = self.mem.lock().unwrap();
        if m.launch_id != ctx.launch_id() {
            Self::reset(&mut m);
            m.launch_id = ctx.launch_id();
        }
        let len = m.data.len();
        if index >= len {
            drop(m);
            if checked {
                return None;
            }
            ctx.trap(Diagnostic::OutOfBounds {
                space: self.space(),
                index,
                len,
                access: acc,
            });
        }

        // Reporting takes a different lock, so this is safe to do while the
        // element stays borrowed.
        if let Some(first) = m.shadow[index].on_write(acc, &now) {
            ctx.report(Diagnostic::DataRace {
                space: self.space(),
                index,
                first,
                second: acc,
            });
        }
        Some(ElemMut { guard: m, index })
    }
}

// ---------------------------------------------------------------- shared ---

pub(crate) struct SharedInner<T> {
    block: u32,
    name: &'static str,
    mem: Mutex<Instrumented<T>>,
}

/// A block-wide `__shared__` array. Obtain one with [`ThreadCtx::shared`].
pub struct SharedArray<T> {
    inner: Arc<SharedInner<T>>,
}

impl<T> Clone for SharedArray<T> {
    fn clone(&self) -> Self {
        SharedArray {
            inner: self.inner.clone(),
        }
    }
}

impl<T: Copy + Default + Send + 'static> SharedArray<T> {
    pub(crate) fn new_inner(block: u32, name: &'static str, len: usize) -> SharedInner<T> {
        SharedInner {
            block,
            name,
            mem: Mutex::new(Instrumented {
                data: vec![T::default(); len],
                shadow: vec![Cell::default(); len],
                launch_id: 0,
            }),
        }
    }

    pub(crate) fn from_any(a: Arc<dyn Any + Send + Sync>) -> Option<Self> {
        a.downcast::<SharedInner<T>>()
            .ok()
            .map(|inner| SharedArray { inner })
    }

    pub fn len(&self) -> usize {
        self.inner.mem.lock().unwrap().data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn space(&self) -> MemSpace {
        MemSpace::Shared {
            block: self.inner.block,
            name: self.inner.name,
        }
    }

    fn reset(m: &mut Instrumented<T>) {
        // Shared memory lives for one launch: it starts uninitialised.
        for c in &mut m.shadow {
            *c = Cell::default();
        }
    }

    #[track_caller]
    pub fn read(&self, ctx: &ThreadCtx<'_>, index: usize) -> T {
        checked_access(
            Buffer {
                mem: &self.inner.mem,
                space: || self.space(),
                reset: Self::reset,
            },
            ctx,
            index,
            AccessKind::Read,
            Location::caller(),
            None,
            |v| *v,
        )
    }

    #[track_caller]
    pub fn write(&self, ctx: &ThreadCtx<'_>, index: usize, value: T) {
        checked_access(
            Buffer {
                mem: &self.inner.mem,
                space: || self.space(),
                reset: Self::reset,
            },
            ctx,
            index,
            AccessKind::Write,
            Location::caller(),
            None,
            |v| {
                *v = value;
                value
            },
        );
    }
}
