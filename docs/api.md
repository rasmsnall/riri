# riri: API Reference

**Document type** Interface specification
**Status** Complete. Describes the surface as built, at version 0.5.0.
**Audience** Anyone writing kernels to run under Riri.
**Companion documents** `architecture.md` for why the design is shaped this way.
**Version** 1.5
**Date** 2026-09-20

---

## Contents

- I. Introduction
  - 1. Purpose
  - 2. The shape of a kernel
  - 3. What is checked and what is not
- II. Launching
  - 1. `launch`
  - 2. `LaunchConfig`
  - 3. `Dim3`
  - 4. Limits
- III. The Thread Context
  - 1. Indices
  - 2. Warp geometry
  - 3. `sync_threads`
  - 4. `shared`
- IV. Memory
  - 1. `GlobalBuf`
  - 2. `SharedArray`
  - 3. Atomics and ordering
  - 4. Fences
  - 5. Lifetime of values and of shadow state
- V. Warp Collectives
  - 1. Member masks
  - 2. Shuffles
  - 3. Votes
  - 4. `sync_warp`
  - 5. Errors specific to collectives
- VI. Reports and Diagnostics
  - 1. `Report`
  - 2. `Diagnostic`
  - 3. Asserting in tests
- VII. Searching Across Schedules
  - 1. `explore` and `Explore`
  - 2. `run` and `run_with`
  - 3. `Exploration` and `Failure`
  - 4. `Schedule` and `replay`
  - 5. When shrinking declines
- VIII. The cuda-oxide Shim
  - 1. `oxide::launch`
  - 2. `thread`
  - 3. `DisjointSlice`
  - 4. `warp`
  - 5. What is not covered
- IX. Worked Examples
  - 1. A clean vector add
  - 2. A shared-memory reduction
  - 3. A warp reduction
  - 4. Pinning a failure with a schedule
- Appendix A. Type index

### List of Tables

- `<Table 2-1>` `LaunchConfig` fields
- `<Table 3-1>` `ThreadCtx` accessors
- `<Table 5-1>` Warp collective signatures
- `<Table 6-1>` `Report` methods
- `<Table 7-1>` `Explore` options
- `<Table 7-2>` `Shrink` outcomes
- `<Table 8-1>` Shim surface

---

## I. Introduction

### 1. Purpose

This document describes everything `riri` exports. The crate has one entry point,
`launch`, and every other type is reached from the `ThreadCtx` it hands to the kernel.

### 2. The shape of a kernel

A kernel is a closure taking a `&ThreadCtx`, called once per simulated GPU thread. It must
be `Sync`, because all simulated threads share it, and it returns nothing: results are
written into instrumented buffers.

```rust
use riri::{launch, GlobalBuf, LaunchConfig};

let out = GlobalBuf::new("out", vec![0u32; 128]);

let report = launch(&LaunchConfig::new(4, 32).seed(1), |t| {
    out.write(t, t.global_linear(), t.thread_linear() as u32);
});

report.assert_clean();
```

Buffers are created outside the launch and captured by the closure. They are
reference-counted, so capturing by reference or by value both work.

### 3. What is checked and what is not

Only accesses that go through `GlobalBuf`, `SharedArray`, and the `warp` collectives are
checked. A kernel that communicates through an `AtomicUsize` it captured itself, or through
a `Mutex`, is invisible to Riri and will be reported as clean regardless of what it does.

This is the central constraint of the current stage. See `architecture.md`, Chapter IX,
Section 2.

---

## II. Launching

### 1. `launch`

```rust
pub fn launch<F>(config: &LaunchConfig, kernel: F) -> Report
where
    F: Fn(&ThreadCtx<'_>) + Sync,
```

Runs `kernel` once per thread of the configured grid and returns what Riri found. The call
blocks until every simulated thread has finished or the launch has been aborted by a
trapped fault.

`launch` never panics because of a fault in the kernel. A kernel panic is captured and
returned as a `Diagnostic::KernelPanic`. It does panic on a malformed configuration, such
as an empty grid or an invalid warp size, because that is a bug in the test rather than in
the kernel.

### 2. `LaunchConfig`

```rust
LaunchConfig::new(grid, block)      // both impl Into<Dim3>
    .seed(u64)                      // default 0
    .warp_size(u32)                 // default 32
```

`<Table 2-1>` `LaunchConfig` fields

| Field | Type | Meaning |
|---|---|---|
| `grid` | `Dim3` | Number of blocks |
| `block` | `Dim3` | Threads per block |
| `seed` | `u64` | Fixes the schedule. The same seed replays the same interleaving |
| `warp_size` | `u32` | Lanes per warp. A power of two no greater than 32 |

`warp_size` exists mainly so that warp behaviour can be tested at a readable scale. A block
of 8 with a warp size of 8 is one full warp whose expected results can be written out by
hand. It is not a way to model AMD wavefronts: masks are `u32`, so 64 lanes cannot be
represented, and the builder rejects anything above 32.

### 3. `Dim3`

```rust
Dim3::new(x, y, z)
Dim3::count(&self) -> u32           // x * y * z
```

`From` is implemented for `u32`, `(u32, u32)`, and `(u32, u32, u32)`, so a launch is
usually written as `LaunchConfig::new(4, 32)` rather than with explicit `Dim3` values.
Linearisation is x-fastest, matching CUDA.

### 4. Limits

`MAX_THREADS` is 16,384. Each simulated thread is an OS thread, so the cap is a resource
limit. Exceeding it panics with a message naming the requested count.

---

## III. The Thread Context

### 1. Indices

`<Table 3-1>` `ThreadCtx` accessors

| Method | Returns | CUDA equivalent |
|---|---|---|
| `thread_idx()` | `Dim3` | `threadIdx` |
| `block_idx()` | `Dim3` | `blockIdx` |
| `block_dim()` | `Dim3` | `blockDim` |
| `grid_dim()` | `Dim3` | `gridDim` |
| `thread_linear()` | `usize` | Linear index within the block |
| `block_linear()` | `usize` | Linear index of the block |
| `global_linear()` | `usize` | `blockIdx * blockDim + threadIdx`, linearised |
| `warp_size()` | `u32` | `warpSize` |
| `warp_id()` | `u32` | Warp index within the block |
| `lane_id()` | `u32` | Lane within the warp |
| `warp_valid_mask()` | `u32` | Mask of the lanes of this warp that exist |

### 2. Warp geometry

Warps are cut from the linear thread index: thread `t` is in warp `t / warp_size` at lane
`t % warp_size`.

`warp_valid_mask()` matters whenever a block size is not a multiple of the warp size,
because the final warp is then short and `warp::FULL_MASK` names lanes that do not exist.
Passing it there is an error, not a silently ignored bit. Prefer `t.warp_valid_mask()` in
any kernel whose block size is not known to be a multiple of the warp size.

### 3. `sync_threads`

```rust
t.sync_threads();
```

The block-wide barrier. Every thread of the block must reach it. If a thread exits while
others wait, Riri reports `BarrierDivergence` and aborts.

It orders everything the block did before it against everything after it, which is how a
shared-memory handoff is made race-free.

### 4. `shared`

```rust
let tile = t.shared::<u32>("tile", 32);
```

Allocates or retrieves a block-wide shared array by name. The first thread of the block to
ask creates it; every later call with the same name in the same block returns the same
array. Elements start *uninitialised*, so reading one before any thread has written it is
reported as `UninitRead`.

Reusing a name with a different element type or a different length panics, since that is a
kernel bug rather than a detected fault.

---

## IV. Memory

### 1. `GlobalBuf`

```rust
let a = GlobalBuf::new("a", vec![1.0f32; 64]);

a.read(t, i) -> T
a.write(t, i, value)
a.len() -> usize
a.is_empty() -> bool
a.to_vec() -> Vec<T>
```

Simulated global memory, created on the host and readable back with `to_vec`. The name is
used only in diagnostics, and choosing a meaningful one is worth the keystrokes because it
is what appears in a race report.

Indexing out of range is a trapped fault, not a panic: Riri reports `OutOfBounds` and
aborts the launch.

### 2. `SharedArray`

Obtained from `ThreadCtx::shared`, with the same `read`, `write`, `len`, and `is_empty`.
There is no `to_vec`, because shared memory does not outlive the launch.

### 3. Atomics and ordering

```rust
a.atomic_load(t, i, ordering) -> T
a.atomic_store(t, i, value, ordering)
a.atomic_add(t, i, value, ordering) -> T      // returns the previous value
```

`atomic_add` is available where `T: Add<Output = T>`. Atomics never conflict with each other
at any scope, including across blocks, but they do conflict with concurrent plain reads and
writes, which is the bug this models.

`Ordering` is `Relaxed`, `Acquire`, `Release`, `AcqRel`, or `SeqCst`. `SeqCst` is accepted
and treated as `AcqRel`: Riri models synchronisation between threads, not a single total
order over all atomics, and nothing it checks can tell the two apart.

A release store paired with an acquire load orders everything the releasing thread did
beforehand against everything the acquiring thread does afterwards. That is what lets one
block hand data to another.

### 4. Fences

```rust
t.threadfence();              // __threadfence(), both halves
t.fence(Ordering::Release);   // one side only
```

The other spelling of the same handoff, and the one CUDA code usually uses. A release fence
makes the next relaxed store publish; an acquire fence makes the previous relaxed load
count.

```rust
// producer                           // consumer
data.write(t, 0, 42);                 while flag.atomic_load(t, 0, Relaxed) == 0 {}
t.threadfence();                      t.threadfence();
flag.atomic_store(t, 0, 1, Relaxed);  let v = data.read(t, 0);   // ordered
```

Remove either fence and the two `data` accesses are reported against each other, because
nothing then orders the blocks.

A barrier feeds this too: a thread that raises a flag after `sync_threads` publishes the
whole block's writes, not just its own. Warp collectives do not, so a lane releasing
straight after `sync_warp` publishes only its own work. See `architecture.md`, Chapter III,
Section 5.

### 5. Lifetime of values and of shadow state

Global buffers keep their values across launches, so a buffer can be written by one launch
and read by the next. Access history does not carry over: each launch starts with a clean
happens-before state, because ordering between launches is not something Riri models.

Shared arrays reset entirely, values and history both, and start uninitialised.

---

## V. Warp Collectives

All collectives live in the `riri::warp` module and take `&ThreadCtx` as their first
argument.

### 1. Member masks

Every collective takes a member mask naming the lanes that take part. The contract, which
Riri enforces rather than assumes, is:

1. The calling lane must be a member of the mask it passes. Otherwise fatal.
2. The mask must name only lanes the block has. Otherwise fatal.
3. Every lane named must reach the same collective with the same mask. Otherwise
   `WarpDivergence` or `WarpMaskMismatch`.

`warp::FULL_MASK` is every lane of a full warp. `warp::WARP_SIZE` is 32, the default.

Riri tells collectives apart by the call site, which `#[track_caller]` makes the caller's
line. If you wrap a collective in a helper of your own, mark that helper `#[track_caller]`
too:

```rust
#[track_caller]
fn warp_reduce(t: &ThreadCtx<'_>, mask: u32, mut v: u32) -> u32 {
    let mut delta = t.warp_size() / 2;
    while delta > 0 {
        v += warp::shfl_down_sync(t, mask, v, delta);
        delta /= 2;
    }
    v
}
```

Without it every call site collapses onto the helper's own line, and two diverged groups
calling it from different places look converged, so Riri reports nothing. See
`architecture.md`, Chapter V, Section 6.

### 2. Shuffles

`<Table 5-1>` Warp collective signatures

| Function | Signature after `ctx, mask` | Reads from |
|---|---|---|
| `shfl_sync` | `value: T, src_lane: u32` | `src_lane % warp_size` |
| `shfl_up_sync` | `value: T, delta: u32` | `lane - delta` |
| `shfl_down_sync` | `value: T, delta: u32` | `lane + delta` |
| `shfl_xor_sync` | `value: T, lane_mask: u32` | `lane ^ lane_mask` |
| `ballot_sync` | `pred: bool` | All member lanes, returns `u32` |
| `any_sync` | `pred: bool` | All member lanes, returns `bool` |
| `all_sync` | `pred: bool` | All member lanes, returns `bool` |
| `sync_warp` | nothing | Nothing, ordering only |

`T` must be `Copy + Send + 'static`, and every participating lane must pass the same type.

A shuffle whose source lane falls outside the warp, such as `shfl_down_sync` near the top,
returns the lane's own value, matching CUDA. A shuffle whose source lane is *inside* the
warp but *outside the mask* is different: that lane published nothing, so Riri reports
`SourceLaneNotInMask` and the lane keeps its own value. This is the check that catches a
warp reduction carrying a stale mask.

### 3. Votes

`ballot_sync` returns one bit per member lane, set where the predicate held. `any_sync` and
`all_sync` are the same rendezvous reduced to a `bool`. `all_sync` compares against the mask,
so it means "true on every member", not "true on every lane of the warp".

### 4. `sync_warp`

Reconverges the member lanes without exchanging a value, and orders their memory accesses,
but only when the mask names the whole warp. A partial mask orders nothing. See
`architecture.md`, Chapter III, Section 3 for why.

### 5. Errors specific to collectives

| Diagnostic | Cause | Disposition |
|---|---|---|
| `WarpDivergence` | A member never reaches the collective | Aborts |
| `WarpMaskMismatch` | Two lanes disagree about the mask | Aborts |
| `WarpLaneError { CallerNotInMask }` | A lane omitted itself | Aborts |
| `WarpLaneError { MaskOutsideBlock }` | Mask names absent lanes | Aborts |
| `WarpLaneError { SourceLaneNotInMask }` | Shuffle read a non-member | Reported, continues |

---

## VI. Reports and Diagnostics

### 1. `Report`

```rust
pub struct Report {
    pub seed: u64,
    pub diagnostics: Vec<Diagnostic>,
    pub aborted: bool,
}
```

`<Table 6-1>` `Report` methods

| Method | Meaning |
|---|---|
| `is_clean()` | No diagnostics at all |
| `has_race()` | At least one `DataRace` |
| `has_warp_divergence()` | At least one `WarpDivergence` |
| `assert_clean()` | Panics with the full listing if anything was found |

`Display` prints the seed, whether the launch aborted, and one line per diagnostic, which
is what the examples print.

### 2. `Diagnostic`

An enum, matched on directly when a test needs to assert something specific. Every variant
carries the block, the thread or lane, and the `&'static Location` of the offending line.
The full list is in `architecture.md`, Chapter VII.

At most 64 diagnostics are retained, de-duplicated by kind and source location, so a racy
line executed by a thousand threads produces one entry.

### 3. Asserting in tests

```rust
report.assert_clean();                       // the common case
assert!(report.has_race(), "{report}");      // include the report in the message
assert!(matches!(
    report.diagnostics.first(),
    Some(Diagnostic::WarpDivergence { mask: 0xFF, .. })
));
```

Passing `"{report}"` as the assertion message is worth doing habitually: when the assertion
fails, the listing explains why far better than the boolean does.

---

## VII. Searching Across Schedules

### 1. `explore` and `Explore`

```rust
pub fn explore<F>(config: &LaunchConfig, seeds: u64, kernel: F) -> Exploration

Explore::new(config)
    .seeds(u64)             // default 64
    .minimise(bool)         // default true
    .budget(usize)          // default 256 replays
```

`explore` runs the kernel across `seeds` schedules, starting from the config's own seed,
and stops at the first one that finds something. `Explore` is the same thing with the
knobs exposed.

`<Table 7-1>` `Explore` options

| Option | Default | Meaning |
|---|---|---|
| `seeds` | 64 | How many seeds to try before giving up |
| `minimise` | true | Whether to shrink the failing schedule |
| `budget` | 256 | How many replays shrinking may spend |

Worth knowing before reaching for this: Riri finds most faults structurally, so a sweep
rarely finds something a single seed did not. It earns its keep where control flow depends
on values the interleaving decides, and on the shrinking rather than the search. See
`architecture.md`, Chapter VIII, Section 1.

### 2. `run` and `run_with`

```rust
Explore::new(&config).run(|t| { ... })            // one kernel, run repeatedly
Explore::new(&config).run_with(|| { ... })        // fresh state built per run
```

`run` takes the same kernel shape as `launch`. Because exploration runs it many times, any
buffer it captures carries its contents from one run into the next.

`run_with` takes a closure that *returns* a kernel, and calls it before every run, so each
run gets its own buffers:

```rust
Explore::new(&config).run_with(|| {
    let out = GlobalBuf::new("out", vec![0u32; 32]);
    move |t: &ThreadCtx<'_>| out.write(t, t.thread_linear(), 1)
})
```

Use `run` when the kernel's behaviour does not depend on the values it reads, which is the
common case, and `run_with` when it does.

### 3. `Exploration` and `Failure`

```rust
pub struct Exploration {
    pub seeds_tried: u64,
    pub failure: Option<Failure>,
}

pub struct Failure {
    pub seed: u64,
    pub report: Report,
    pub shrink: Shrink,
    pub decisions: usize,   // length of the original failing schedule
}
```

`Exploration::is_clean` and `assert_clean` mirror `Report`. `Failure::target` gives the
fingerprint that shrinking preserved, which is the first finding in the report.

`Display` prints the seeds tried, the failing seed, the original and shrunk sizes, the
findings, and the schedule.

### 4. `Schedule` and `replay`

```rust
pub fn replay<F>(config: &LaunchConfig, schedule: &Schedule, kernel: F) -> Report
```

A `Schedule` is a list of global thread indices: who gets the turn at each decision. Past
the end of the list, the scheduler keeps the current thread running while it can, so a
short schedule is still a complete description of a run.

`Schedule::len` and `Schedule::switches` are the two numbers that say whether it is
readable. `Schedule::new` builds one by hand, which is useful for pinning a known case.

A schedule reproduces on its own. It does not use the seed, so it keeps working even if
the way seeds are drawn ever changes.

### 5. When shrinking declines

`<Table 7-2>` `Shrink` outcomes

| Variant | Meaning |
|---|---|
| `Minimised(Schedule)` | A schedule that replays and reproduces the finding |
| `Disabled` | `minimise(false)` was set |
| `TraceTruncated` | The run made more decisions than Riri records |
| `NotReproducible` | Replaying the recording lost the finding |

`NotReproducible` means the kernel does not behave the same way twice, almost always
because it captures a buffer that every run mutates. `run_with` is the fix. Riri reports
this rather than shrinking against noise.

An empty `Minimised` schedule is a real result, not a failure: it means the default
in-order schedule already reaches the bug.

---

## VIII. The cuda-oxide Shim

`riri::oxide` provides the surface a cuda-oxide kernel is written against, so the same
source runs under Riri. The attribute is the only difference, and `cfg_attr` carries it:

```rust
#[cfg_attr(not(riri), cuda_device::kernel)]
fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
    if let Some((mut c_elem, idx)) = c.get_mut_indexed() {
        let i = idx.get();
        *c_elem = a[i] + b[i];
    }
}
```

### 1. `oxide::launch`

```rust
pub fn launch<F>(config: &LaunchConfig, kernel: F) -> Report
where
    F: Fn() + Sync,
```

The closure takes no arguments, because the kernel finds its own identity through `thread`
exactly as it does on the GPU. Call the kernel function inside it, passing whatever it takes
by value:

```rust
let out = GlobalBuf::new("c", vec![0.0f32; 64]);
let c = DisjointSlice::new(&out);
let report = oxide::launch(&LaunchConfig::new(2, 32), || vecadd(&a, &b, c.clone()));
```

`DisjointSlice` is cheap to clone and shares its buffer, which is how each simulated thread
receives its own handle the way a GPU kernel receives its own argument.

Calling any device function outside a launch panics with a message saying so, rather than
reading another launch's state.

### 2. `thread`

`<Table 8-1>` Shim surface

| Item | Notes |
|---|---|
| `thread::index_1d()` | Returns an owned `ThreadIndex`, not copyable or sendable |
| `thread::threadIdx_{x,y,z}()` | `u32`, as on the GPU |
| `thread::blockIdx_{x,y,z}()`, `blockDim_{x,y,z}()`, `gridDim_x()` | `u32` |
| `thread::warp_size()` | `warpSize` |
| `thread::sync_threads()` | The block barrier, checked as in Chapter III |

### 3. `DisjointSlice`

```rust
DisjointSlice::new(&buf)                     // wrap an instrumented GlobalBuf
slice.get_mut_indexed() -> Option<(ElemMut<'_, T>, ThreadIndex<'_>)>
slice.get_mut(idx)      -> Option<ElemMut<'_, T>>
slice.get_unchecked_mut(i) -> ElemMut<'_, T>
slice.len()
```

`ElemMut` derefs to `T`, so `*elem = value` reads as it does against cuda-oxide's `&mut T`.
It holds a lock for the life of the borrow, which is fine because only one simulated thread
runs at a time, but do not hold one across `sync_threads`.

The checked accessors return `None` past the end, which is how a launch rounded up to whole
blocks drops its tail. `get_unchecked_mut` traps instead, matching the unchecked contract,
and is the one worth running under Riri: it asserts the index belongs to the calling thread
alone, and two threads claiming one element is reported as a data race naming both lines.

### 4. `warp`

`lane_id`, `warp_id`, `shuffle`, `shuffle_up`, `shuffle_down`, `shuffle_xor` (each with an
`_f32` form), `all`, `any`, `ballot`, `popc`.

cuda-oxide's unsuffixed forms take no member mask, so Riri supplies the whole warp, which is
what the instruction they lower to assumes. A warp that is not converged at one of these is
therefore reported, which is the bug those forms invite.

### 5. What is not covered

- **Shared memory.** cuda-oxide's `SharedArray` is a zero-sized marker whose storage its
  compiler provides and whose accessors all panic off-device, so Riri would have to supply
  storage of its own, and handing out references into it would need `unsafe`. Left out
  deliberately. Use `ThreadCtx::shared`. See `architecture.md`, Chapter IX, Section 5.
- The `_sync` shuffle forms, 2D and tiled index spaces, managed barriers, clusters, TMA.
- `cuda-device` is unpublished and pins a nightly toolchain, so this surface is written from
  the published API reference rather than compiled against the real crate. Treat a signature
  mismatch as a bug in Riri.

---

## IX. Worked Examples

### 1. A clean vector add

```rust
let a = GlobalBuf::new("a", (0..128).map(|i| i as f32).collect());
let b = GlobalBuf::new("b", vec![1.0f32; 128]);
let c = GlobalBuf::new("c", vec![0.0f32; 128]);

let report = launch(&LaunchConfig::new(4, 32).seed(1), |t| {
    let i = t.global_linear();
    c.write(t, i, a.read(t, i) + b.read(t, i));
});

report.assert_clean();
```

### 2. A shared-memory reduction

The barrier inside the loop is the whole point. Removing it is the bug
`cargo run --example reduction` demonstrates.

```rust
let report = launch(&LaunchConfig::new(4, 64).seed(2026), |t| {
    let tile = t.shared::<u32>("tile", 64);
    let i = t.thread_linear();
    tile.write(t, i, input.read(t, t.global_linear()));
    t.sync_threads();

    let mut stride = 32;
    while stride > 0 {
        if i < stride {
            let sum = tile.read(t, i) + tile.read(t, i + stride);
            tile.write(t, i, sum);
        }
        t.sync_threads();
        stride /= 2;
    }

    if i == 0 {
        partial.write(t, t.block_linear(), tile.read(t, 0));
    }
});
```

### 3. A warp reduction

Note that the shuffle is called unconditionally. Guarding it with `if lane < delta`, which
is a common habit, is exactly the divergence Riri reports.

```rust
let report = launch(&LaunchConfig::new(1, 32).seed(1), |t| {
    let mask = t.warp_valid_mask();
    let mut v = t.lane_id() + 1;
    let mut delta = 16;
    while delta > 0 {
        v += warp::shfl_down_sync(t, mask, v, delta);
        delta /= 2;
    }
    if t.lane_id() == 0 {
        out.write(t, 0, v);
    }
});
```

### 4. Pinning a failure with a schedule

Once exploration has found and shrunk a failure, the schedule is the regression test. It
needs no seed and does not depend on how seeds are drawn.

```rust
use riri::{replay, LaunchConfig, Schedule};

#[test]
fn the_flag_race_stays_fixed() {
    let schedule = Schedule::new(vec![1, 3, 3, 3, 1]);
    let report = replay(&LaunchConfig::new(1, 4), &schedule, kernel);
    report.assert_clean();
}
```

---

## Appendix A. Type index

| Item | Kind | Module |
|---|---|---|
| `launch` | Function | root |
| `LaunchConfig` | Struct | root |
| `MAX_THREADS` | Constant | root |
| `Dim3` | Struct | root |
| `ThreadCtx` | Struct | root |
| `GlobalBuf` | Struct | root |
| `SharedArray` | Struct | root |
| `Report` | Struct | root |
| `explore`, `replay` | Functions | root |
| `ElemMut` | Struct | root |
| `Ordering` | Enum | root |
| `launch`, `thread`, `warp`, `DisjointSlice`, `ThreadIndex` | cuda-oxide shim | `oxide` |
| `Explore`, `Exploration`, `Failure`, `Schedule`, `Shrink` | Exploration types | root |
| `Diagnostic` | Enum | root |
| `LaneProblem` | Enum | root |
| `Access`, `AccessKind`, `MemSpace` | Types within diagnostics | root |
| `WARP_SIZE`, `FULL_MASK` | Constants | `warp` |
| `shfl_sync`, `shfl_up_sync`, `shfl_down_sync`, `shfl_xor_sync` | Functions | `warp` |
| `ballot_sync`, `any_sync`, `all_sync`, `sync_warp` | Functions | `warp` |
