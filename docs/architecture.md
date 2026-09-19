# riri: Architecture

**Document type** Technical architecture specification
**Status** Complete and implemented as described. Detection runs end to end for block and warp scopes.
**Audience** Anyone integrating, operating, or modifying this library. No prior context assumed.
**Companion documents** `api.md` for the callable surface.
**Version** 1.0
**Date** 2026-09-19

---

## Contents

- I. Introduction
  - 1. Purpose
  - 2. Rationale
  - 3. Scope and non-goals
- II. Execution Model
  - 1. Simulated threads
  - 2. The turn
  - 3. Scheduling points
  - 4. Determinism
- III. Happens-Before Model
  - 1. Scopes and epochs
  - 2. The concurrency rule
  - 3. Why a partial mask orders nothing
  - 4. Atomics
- IV. Shadow Memory
  - 1. Per-element state
  - 2. Reads and writes
  - 3. The reader bound
- V. Warp Model
  - 1. Warp geometry
  - 2. Collectives as rendezvous
  - 3. Why there are two phases
  - 4. Convergence failures
  - 5. Mask errors
- VI. Failure Model
  - 1. Reported faults and trapped faults
  - 2. Deadlock as a diagnostic
  - 3. Why the warp is blamed before the barrier
  - 4. De-duplication
- VII. Diagnostics
- VIII. Assessment
  - 1. Advantages
  - 2. Disadvantages
  - 3. Conditions under which this design is inappropriate
- IX. Dependencies
- References
- Appendix A. Glossary

### List of Tables

- `<Table 2-1>` Module map
- `<Table 3-1>` Ordering by scope
- `<Table 5-1>` Warp collectives
- `<Table 7-1>` Diagnostics

### List of Figures

- `[Figure 1-1]` Loop superseded by this library
- `[Figure 1-2]` Loop implemented by this library
- `[Figure 2-1]` Handing off the turn
- `[Figure 5-1]` The two phases of a collective

---

## I. Introduction

### 1. Purpose

`riri` detects undefined behaviour in GPU kernels written in Rust, by running them on the
CPU. Every GPU thread becomes a simulated thread, a seeded scheduler interleaves them one
memory operation at a time, and every access is checked against a happens-before model.

It exists to remove the GPU from the debugging loop.

```
kernel -> run on hardware -> observe a wrong answer -> infer which access raced
```

[Figure 1-1] Loop superseded by this library

That loop is slow, needs hardware in the path, and is unreliable in a specific way: a race
whose interleaving did not occur on this run produces a correct answer and no signal at
all. Warp-synchronous code makes this worse, because it usually produces the right answer
until a compiler or architecture change removes the lockstep it was relying on.

```
kernel -> simulated threads -> shadow memory -> the line that broke the rule
```

[Figure 1-2] Loop implemented by this library

### 2. Rationale

The change is not merely a convenience. Detection here is *structural* rather than
observational.

A race is reported when two accesses are concurrent under the happens-before model and at
least one writes, which does not depend on the two accesses actually landing in a bad order
during this run. A warp convergence failure is reported when a lane named by a member mask
cannot reach the collective, which does not depend on the schedule either.

The practical consequence is that a failing seed is a reproducer, not a sighting. The same
seed replays the same interleaving exactly, so a failure found in CI reproduces on a laptop
with no hardware and no flakiness.

### 3. Scope and non-goals

In scope:

- SIMT kernels written against Riri's API and run as ordinary Rust closures.
- Data races on global and shared memory, within and across blocks.
- Barrier divergence at block scope.
- Warp collective convergence, member mask agreement, and shuffle source validity.
- Uninitialised shared-memory reads, out-of-bounds accesses, and kernel panics.

Not in scope:

- Executing on a GPU, or emitting PTX. Nothing here ever touches a driver.
- Running cuda-oxide or rust-cuda kernel source unchanged. That is Roadmap item 1, and the
  current stage is a library emulator, as recorded in Chapter VIII, Section 2.
- Rust aliasing rules, such as Tree Borrows, across lanes. That requires MIR interpretation.
- Performance modelling of any kind. Riri says nothing about occupancy, coalescing, or
  throughput.
- Warp sizes above 32. Member masks are `u32`.
- Weak memory and fences beyond atomics.

---

## II. Execution Model

### 1. Simulated threads

Each GPU thread is backed by one OS thread, spawned inside a `std::thread::scope` so that
the kernel closure can borrow from the caller. This is why launches are capped at 16,384
threads: the cap is a resource limit, not a modelling one.

`<Table 2-1>` Module map

| Module | Responsibility |
|---|---|
| `launch.rs` | `LaunchConfig`, spawning simulated threads, assembling the `Report` |
| `sched.rs` | The turn, barriers, warp rendezvous, deadlock detection |
| `ctx.rs` | `ThreadCtx`: indices, warp geometry, `sync_threads`, shared allocation |
| `mem.rs` | `GlobalBuf`, `SharedArray`, the instrumented access path |
| `shadow.rs` | Per-element access history and the concurrency rule |
| `warp.rs` | Warp collectives and mask validation |
| `diag.rs` | `Diagnostic`, `Report`, the de-duplicating reporter |
| `dim.rs` | `Dim3` |

`shadow.rs` is deliberately free of any scheduler dependency. It decides whether two
recorded accesses conflict, which is the intricate part, and can be tested without running
a kernel at all.

### 2. The turn

Only one simulated thread runs at a time: the one holding the *turn*. All others are
blocked on a condition variable. This is what makes an interleaving a decision rather than
an accident, and it is why no lock is needed to protect user data.

```
running thread -> reaches a scheduling point -> picks a runnable thread by seeded RNG
               -> hands over the turn -> blocks until chosen again
```

[Figure 2-1] Handing off the turn

A thread is *runnable* unless it is parked at a block barrier, parked at a warp collective,
or finished. The scheduler picks uniformly at random among runnable threads using
SplitMix64 seeded from `LaunchConfig::seed`.

### 3. Scheduling points

The turn changes hands at every instrumented operation: each `GlobalBuf` or `SharedArray`
access, and each warp collective. Between scheduling points a thread runs without
interruption, which means pure computation in a kernel is not interleaved.

This is sound for race detection because pure computation touches no shared state. It does
mean Riri models the *memory* interleaving, not the instruction interleaving.

### 4. Determinism

A seed fixes the entire schedule. Two runs with the same seed produce the same
interleaving, the same diagnostics, and the same output values, which is asserted by tests
in both `tests/detect.rs` and `tests/warp.rs`.

Different seeds explore different interleavings. Because detection is happens-before based
rather than observational, changing the seed changes *which* accesses are reported as the
conflicting pair, not *whether* a race is found.

---

## III. Happens-Before Model

### 1. Scopes and epochs

Ordering is tracked with epoch counters rather than full vector clocks. Each block has a
barrier epoch, incremented when a `sync_threads` releases. Each warp has a collective
epoch, incremented when a full-mask collective releases.

`<Table 3-1>` Ordering by scope

| Relationship between two accesses | Ordered? |
|---|---|
| Same thread | Always |
| Same block, different barrier epoch | Yes, the barrier orders them |
| Same warp, same barrier epoch, different warp epoch | Yes, the collective orders them |
| Different warps of one block, same barrier epoch | No |
| Different blocks | Never, within a single launch |
| Two atomics | Never conflict with each other |

### 2. The concurrency rule

Two accesses are concurrent, and therefore race if at least one of them writes, exactly
when none of the ordering relationships above applies. Stated as the implementation in
`shadow.rs` evaluates it, in order:

1. Same thread: not concurrent.
2. Both atomic: not concurrent.
3. Different blocks: concurrent.
4. Different barrier epochs: not concurrent.
5. Different warps: concurrent.
6. Otherwise: concurrent if and only if the warp epochs are equal.

### 3. Why a partial mask orders nothing

A full-mask collective bumps the warp epoch. A collective with a partial mask does not.

The reason is that a partial mask is a statement about the lanes it names and says nothing
whatsoever about the lanes it omits. A single per-warp counter cannot record "these four
lanes synchronised and those four did not" without becoming a per-lane clock, and bumping
it anyway would report ordering that does not exist, which is a false negative. Declining
to bump is the conservative direction: some accesses between partial-mask participants may
be reported as racing when a finer model would order them.

This trade is chosen deliberately. For a race detector a false positive is a nuisance and a
false negative is a defect.

### 4. Atomics

Atomics never conflict with each other, at any scope, including across blocks. They do
conflict with concurrent plain reads and writes, which is the bug worth catching: a kernel
that accumulates with `atomic_add` while another thread reads the same cell without
synchronisation.

---

## IV. Shadow Memory

### 1. Per-element state

Every element of every instrumented buffer carries a shadow cell: an initialised flag, the
last write, and a bounded set of recent readers. Shadow state is per launch. Global memory
keeps its values across launches but resets its access history; shared memory resets both,
since it is created fresh per launch and starts uninitialised.

### 2. Reads and writes

A read reports a conflict if the last write to that element is concurrent with it. It then
discards readers from its own block in older epochs, which are ordered against everything
now, and records itself.

A write reports a conflict against the last write or any concurrent reader, then replaces
the last write and clears the reader set: readers that were concurrent have just been
reported, and readers that were ordered are dead.

### 3. The reader bound

Each cell remembers at most 8 concurrent readers. Beyond that limit further readers are not
recorded, so a later write may miss a write-after-read race against a forgotten reader.

This is a bounded-memory trade, and it is a false negative rather than a false positive. It
only bites when more than 8 threads read one element concurrently and a ninth thread then
writes it.

---

## V. Warp Model

### 1. Warp geometry

Warps are formed from the linear thread index within a block. For warp size `w`, thread `t`
sits in warp `t / w` at lane `t % w`. A block whose size is not a multiple of the warp size
has a short final warp, and `ThreadCtx::warp_valid_mask` reports the lanes that exist.

Naming a lane that does not exist is an error rather than a silently ignored bit, because on
hardware it is undefined behaviour and it is usually a symptom of a full mask constant being
used in a block that cannot fill a warp.

### 2. Collectives as rendezvous

Every collective is a rendezvous. A lane publishes its value into a per-warp exchange slot,
parks, and is released only when every lane named by the mask has parked at the *same*
source location with the *same* mask.

`<Table 5-1>` Warp collectives

| Function | Exchange | Source lane |
|---|---|---|
| `shfl_sync` | Value from an arbitrary lane | `src_lane % warp_size` |
| `shfl_up_sync` | Value from `lane - delta` | None below lane 0, keeps own value |
| `shfl_down_sync` | Value from `lane + delta` | None past the warp, keeps own value |
| `shfl_xor_sync` | Value from `lane ^ lane_mask` | None past the warp, keeps own value |
| `ballot_sync` | One bit per member lane | Not applicable |
| `any_sync`, `all_sync` | Ballot, reduced | Not applicable |
| `sync_warp` | Nothing, ordering only | Not applicable |

### 3. Why there are two phases

A collective is two rendezvous, not one.

```
publish value -> [arrive] -> read mask-mates' values -> [depart] -> continue
```

[Figure 5-1] The two phases of a collective

Without the departure phase a lane released from the arrival phase could run ahead into its
*next* collective and overwrite its exchange slot while a mask-mate was still reading the
current one. The departure phase is a plain counting rendezvous among the same members, and
always completes, because the code path between the two phases is straight-line. It is also
where the warp epoch is bumped, so that ordering takes effect after the exchange rather than
during it.

### 4. Convergence failures

A convergence failure is any state in which the members of a mask can never all park at the
same collective. Three mechanisms detect it, and they overlap deliberately so that the
common cases produce precise messages and the remainder are still caught.

1. **A member has already exited.** Checked when a lane arrives. Reported immediately.
2. **A member exits while others wait.** Checked when a thread finishes, against the masks
   of any parked warp-mates that name it. Reported immediately.
3. **Anything else.** Caught by the deadlock detector described in Chapter VI, Section 2,
   including lanes stopped at different collectives and lanes stopped at a block barrier.

Because the first two fire as soon as the situation arises, the reported `arrived` mask
shows how many lanes had parked at that instant, which depends on the schedule. The `mask`
and the set of missing lanes do not depend on the schedule: they come from the code.

### 5. Mask errors

Three mask errors are distinguished from convergence failures, because they are local to
one lane and need no rendezvous to diagnose.

- **Caller not in mask.** A lane omitted itself from the mask it passed. Fatal.
- **Mask outside block.** The mask names lanes the block does not have. Fatal.
- **Source lane not in mask.** A shuffle read from a lane that published nothing. Reported,
  and the lane keeps its own value, because the collective itself completed correctly and
  the launch can continue.

The first two are checked before the lane parks, so they abort rather than producing a
convergence failure as a secondary effect.

---

## VI. Failure Model

### 1. Reported faults and trapped faults

A *reported* fault records a diagnostic and lets the launch continue, which allows one run
to surface several independent problems. Races, uninitialised reads, and out-of-range
shuffle sources are reported.

A *trapped* fault aborts the launch, unwinding every simulated thread, in the manner of a
GPU trap. Out-of-bounds accesses, the two fatal mask errors, convergence failures, and
barrier divergence are trapped. `Report::aborted` records that this happened.

### 2. Deadlock as a diagnostic

The scheduler's stuck state is a first-class diagnostic rather than an internal error. When
a thread hands off the turn and no thread is runnable, while at least one thread is still
live, the launch cannot progress. Rather than hanging, the scheduler inspects the parked
threads, attributes the wedge, and aborts with a diagnostic.

This path is also checked when the last runnable thread exits, so a launch whose final
thread finishes while others are parked terminates rather than waiting forever.

### 3. Why the warp is blamed before the barrier

When both a warp collective and a block barrier are stuck, the warp is reported and the
barrier is not.

A block barrier waits for every thread of the block. If some of those threads are parked in
a warp collective that can never complete, the barrier could never have been reached, so
reporting it would name a consequence and bury the cause. The rule is therefore: report
every stuck warp; only if there are none, report stuck barriers.

### 4. De-duplication

Diagnostics are de-duplicated by kind and source location, so a racy line executed by 1024
threads produces one report rather than a thousand. A race is keyed on the unordered pair of
the two locations, so the same pair is not reported twice with the roles swapped. At most 64
diagnostics are retained per launch.

---

## VII. Diagnostics

`<Table 7-1>` Diagnostics

| Variant | Meaning | Disposition |
|---|---|---|
| `DataRace` | Two concurrent accesses, at least one a write | Reported |
| `UninitRead` | Read of a shared element nobody has written | Reported |
| `OutOfBounds` | Index past the end of a buffer | Trapped |
| `BarrierDivergence` | Threads wait at a barrier others never reach | Trapped |
| `WarpDivergence` | A lane named by a mask never reaches the collective | Trapped |
| `WarpMaskMismatch` | Two lanes disagree about the member mask | Trapped |
| `WarpLaneError` | Caller not in mask, mask outside block, or bad shuffle source | Trapped, except a bad source |
| `KernelPanic` | The kernel closure panicked | Trapped |

Every variant carries the block, thread or lane, and the `&'static Location` of the
offending source line, which is captured with `#[track_caller]` on the public API rather
than by unwinding.

---

## VIII. Assessment

### 1. Advantages

- Detection does not depend on an unlucky schedule, so absence of a report is meaningful
  within the model's limits, and presence of one is reproducible.
- No GPU, no driver, no `unsafe`, and no dependencies. It is a normal test binary.
- Failures name a source line rather than a symptom.
- Warp divergence is reported instead of hanging, which is the failure mode that makes the
  same bug expensive to diagnose on hardware.

### 2. Disadvantages

- Kernels must be written against Riri's API. Code cannot be moved under Riri without
  being ported, which is the largest limitation and the reason Roadmap item 1 exists.
- One OS thread per simulated GPU thread caps launches at 16,384 threads and makes large
  launches slow.
- A collective is identified by its source location, so two lanes at the same line in
  different loop iterations are treated as converged. Divergence across iterations of one
  loop is not detected.
- Epoch counters are coarser than vector clocks. Partial-mask collectives are conservatively
  treated as ordering nothing, which can produce false positives among their participants.
- The 8-reader bound can miss a write-after-read race when more than 8 threads read an
  element concurrently.

### 3. Conditions under which this design is inappropriate

- Kernels whose correctness depends on weak memory ordering or fences beyond atomics.
- AMD wavefronts, or any hardware with a warp wider than 32 lanes.
- Performance work of any kind.
- Kernels that cannot be expressed as a Rust closure over Riri's memory types, such as those
  relying on inline PTX or on vendor intrinsics with no model here.

---

## IX. Dependencies

None. The library depends only on the Rust standard library, and has no dev-dependencies.
This is a deliberate constraint: Riri is intended to be cheap to add to a CI job, and a
detector with a large dependency tree is a harder thing to trust and to audit.

The minimum supported Rust version is 1.75, which CI verifies on every push alongside
stable.

---

## References

- NVIDIA, *CUDA C++ Programming Guide*, warp shuffle functions and the member mask contract.
- NVIDIA, *Compute Sanitizer User Manual*, `racecheck` and `synccheck`.
- NVlabs, *cuda-oxide: The Safety Model*, chapter "The hard problems", which records that
  `DisjointSlice` does not cover cooperative patterns and that warp convergence is not
  enforceable by the type system today.
  https://nvlabs.github.io/cuda-oxide/gpu-safety/the-safety-model.html
- Köpcke, Gorlatch, Steuwer, *Descend: A Safe GPU Systems Programming Language*, PLDI 2024.
- The Miri authors, *Miri: an interpreter for Rust's mid-level intermediate representation*.
- Villard et al., *Tree Borrows*, the aliasing model targeted by Roadmap item 3.

---

## Appendix A. Glossary

`<Table A-1>` Glossary of terms

| Term | Meaning |
|---|---|
| Barrier epoch | Per-block counter incremented when a `sync_threads` releases |
| Block | A group of threads that can synchronise with each other and share shared memory |
| Collective | A warp-level operation every member lane must reach together |
| Concurrent | Two accesses with no happens-before relationship in either direction |
| Lane | A thread's position within its warp |
| Member mask | The set of lanes a collective names as participants |
| Reported fault | A diagnostic that does not stop the launch |
| Shadow memory | Per-element access history used to decide whether accesses conflict |
| SIMT | Single instruction, multiple threads, the GPU execution model modelled here |
| Trapped fault | A diagnostic that aborts the launch, in the manner of a GPU trap |
| Turn | The right to run; exactly one simulated thread holds it at a time |
| Warp | A group of lanes that execute together and can exchange values directly |
| Warp epoch | Per-warp counter incremented when a full-mask collective releases |
