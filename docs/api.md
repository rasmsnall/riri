# riri: API Reference

**Document type** Interface specification
**Status** Complete. Describes the surface as built, at version 0.2.0.
**Audience** Anyone writing kernels to run under Riri.
**Companion documents** `architecture.md` for why the design is shaped this way.
**Version** 1.0
**Date** 2026-09-19

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
  - 3. Atomics
  - 4. Lifetime of values and of shadow state
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
- VII. Worked Examples
  - 1. A clean vector add
  - 2. A shared-memory reduction
  - 3. A warp reduction
  - 4. Sweeping seeds
- Appendix A. Type index

### List of Tables

- `<Table 2-1>` `LaunchConfig` fields
- `<Table 3-1>` `ThreadCtx` accessors
- `<Table 5-1>` Warp collective signatures
- `<Table 6-1>` `Report` methods

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

This is the central constraint of the current stage. See `architecture.md`, Chapter VIII,
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

### 3. Atomics

```rust
a.atomic_add(t, i, value) -> T      // returns the previous value
```

Available where `T: Add<Output = T>`. Atomics never conflict with each other at any scope,
including across blocks, but they do conflict with concurrent plain reads and writes, which
is the bug this models.

### 4. Lifetime of values and of shadow state

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

## VII. Worked Examples

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

### 4. Sweeping seeds

Detection does not depend on the seed, but the *reported pair* of conflicting accesses does.
Sweeping is useful when narrowing down which accesses are involved.

```rust
for seed in 0..32 {
    let report = launch(&LaunchConfig::new(1, 8).seed(seed), kernel);
    assert!(report.is_clean(), "seed {seed}: {report}");
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
| `Diagnostic` | Enum | root |
| `LaneProblem` | Enum | root |
| `Access`, `AccessKind`, `MemSpace` | Types within diagnostics | root |
| `WARP_SIZE`, `FULL_MASK` | Constants | `warp` |
| `shfl_sync`, `shfl_up_sync`, `shfl_down_sync`, `shfl_xor_sync` | Functions | `warp` |
| `ballot_sync`, `any_sync`, `all_sync`, `sync_warp` | Functions | `warp` |
