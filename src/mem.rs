use std::any::Any;
use std::ops::{Add, Deref, DerefMut};
use std::panic::Location;
use std::sync::{Arc, Mutex, MutexGuard};

use crate::ctx::ThreadCtx;
use crate::diag::{Access, AccessKind, Diagnostic, MemSpace};
use crate::shadow::Cell;

/// Shared implementation for any instrumented array.
struct Instrumented<T> {
    data: Vec<T>,
    shadow: Vec<Cell>,
    /// Launch the shadow state belongs to; reset when a new launch touches it.
    launch_id: u64,
}

fn checked_access<T: Copy>(
    mem: &Mutex<Instrumented<T>>,
    ctx: &ThreadCtx<'_>,
    space: impl Fn() -> MemSpace,
    index: usize,
    kind: AccessKind,
    location: &'static Location<'static>,
    reset: impl Fn(&mut Instrumented<T>),
    op: impl FnOnce(&mut T) -> T,
) -> T {
    // Let another thread run first: this is where interleavings come from.
    ctx.schedule_point();

    let acc: Access = ctx.access(kind, location);
    let mut m = mem.lock().unwrap();
    if m.launch_id != ctx.launch_id() {
        reset(&mut m);
        m.launch_id = ctx.launch_id();
    }
    let len = m.data.len();
    if index >= len {
        drop(m);
        ctx.trap(Diagnostic::OutOfBounds { space: space(), index, len, access: acc });
    }

    let cell = &mut m.shadow[index];
    let was_init = cell.init;
    let conflict = match kind {
        AccessKind::Read => cell.on_read(acc),
        AccessKind::Write | AccessKind::Atomic => cell.on_write(acc),
    };
    let value = op(&mut m.data[index]);
    drop(m);

    if kind != AccessKind::Write && !was_init {
        ctx.report(Diagnostic::UninitRead { space: space(), index, access: acc });
    }
    if let Some(first) = conflict {
        ctx.report(Diagnostic::DataRace { space: space(), index, first, second: acc });
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
        GlobalBuf { name: self.name.clone(), mem: self.mem.clone() }
    }
}

impl<T: Copy + Send + 'static> GlobalBuf<T> {
    /// Creates a buffer initialised from host data (like a host-to-device copy).
    pub fn new(name: &str, data: Vec<T>) -> Self {
        let shadow = vec![Cell::initialised(); data.len()];
        GlobalBuf {
            name: name.into(),
            mem: Arc::new(Mutex::new(Instrumented { data, shadow, launch_id: 0 })),
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
        MemSpace::Global { buffer: self.name.clone() }
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
        checked_access(&self.mem, ctx, || self.space(), index, AccessKind::Read, Location::caller(), Self::reset, |v| *v)
    }

    #[track_caller]
    pub fn write(&self, ctx: &ThreadCtx<'_>, index: usize, value: T) {
        checked_access(&self.mem, ctx, || self.space(), index, AccessKind::Write, Location::caller(), Self::reset, |v| {
            *v = value;
            value
        });
    }
}

impl<T: Copy + Send + Add<Output = T> + 'static> GlobalBuf<T> {
    /// `atomicAdd`: returns the previous value. Atomics never race with each
    /// other, only with concurrent plain accesses.
    #[track_caller]
    pub fn atomic_add(&self, ctx: &ThreadCtx<'_>, index: usize, value: T) -> T {
        checked_access(&self.mem, ctx, || self.space(), index, AccessKind::Atomic, Location::caller(), Self::reset, |v| {
            let old = *v;
            *v = old + value;
            old
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

        let acc = ctx.access(AccessKind::Write, location);
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
            ctx.trap(Diagnostic::OutOfBounds { space: self.space(), index, len, access: acc });
        }

        // Reporting takes a different lock, so this is safe to do while the
        // element stays borrowed.
        if let Some(first) = m.shadow[index].on_write(acc) {
            ctx.report(Diagnostic::DataRace { space: self.space(), index, first, second: acc });
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
        SharedArray { inner: self.inner.clone() }
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
        a.downcast::<SharedInner<T>>().ok().map(|inner| SharedArray { inner })
    }

    pub fn len(&self) -> usize {
        self.inner.mem.lock().unwrap().data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn space(&self) -> MemSpace {
        MemSpace::Shared { block: self.inner.block, name: self.inner.name }
    }

    fn reset(m: &mut Instrumented<T>) {
        // Shared memory lives for one launch: it starts uninitialised.
        for c in &mut m.shadow {
            *c = Cell::default();
        }
    }

    #[track_caller]
    pub fn read(&self, ctx: &ThreadCtx<'_>, index: usize) -> T {
        checked_access(&self.inner.mem, ctx, || self.space(), index, AccessKind::Read, Location::caller(), Self::reset, |v| *v)
    }

    #[track_caller]
    pub fn write(&self, ctx: &ThreadCtx<'_>, index: usize, value: T) {
        checked_access(&self.inner.mem, ctx, || self.space(), index, AccessKind::Write, Location::caller(), Self::reset, |v| {
            *v = value;
            value
        });
    }
}
