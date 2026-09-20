# riri: Architecture

**Document type** Technical architecture specification
**Status** Complete and implemented as described. Detection runs end to end for block and warp scopes, with schedule exploration and shrinking on top.
**Audience** Anyone integrating, operating, or modifying this library. No prior context assumed.
**Companion documents** `api.md` for the callable surface.
**Version** 1.6
**Date** 2026-09-20

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
  - 5. Release and acquire between blocks
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
  - 6. How a collective is identified
- VI. Failure Model
  - 1. Reported faults and trapped faults
  - 2. Deadlock as a diagnostic
  - 3. Why the warp is blamed before the barrier
  - 4. De-duplication
- VII. Diagnostics
- VIII. Schedule Exploration
  - 1. Why search at all
  - 2. Recording and replaying a schedule
  - 3. Shrinking
  - 4. When shrinking cannot be trusted
- IX. The cuda-oxide Shim
  - 1. What it is for
  - 2. Giving a context-free API a context
  - 3. Handing out a mutable element
  - 4. Carrying the caller's location
  - 5. Shared memory, and why it is absent
- X. Assessment
  - 1. Advantages
  - 2. Disadvantages
  - 3. Conditions under which this design is inappropriate
- XI. Dependencies
- References
- Appendix A. Glossary

### List of Tables

- `<Table 2-1>` Module map
- `<Table 3-1>` Ordering by scope
- `<Table 3-2>` What a release publishes
- `<Table 5-1>` Warp collectives
- `<Table 7-1>` Diagnostics
- `<Table 8-1>` Outcomes of shrinking
- `<Table 9-1>` Shim coverage

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
- Uninitialised reads of shared memory, and of global memory declared uninitialised,
  plus out-of-bounds accesses and kernel panics.

Not in scope:

- Executing on a GPU, or emitting PTX. Nothing here ever touches a driver.
- Running rust-cuda kernel source, or any surface other than Riri's own and the cuda-oxide
  shim of Chapter IX. The shim covers cuda-oxide's indexing, `DisjointSlice`, barriers and
  warp primitives, but not its shared memory, and it is written from the published API
  reference rather than compiled against `cuda-device`.
- Rust aliasing rules, such as Tree Borrows, across lanes. That requires MIR interpretation.
- Performance modelling of any kind. Riri says nothing about occupancy, coalescing, or
  throughput.
- Warp sizes above 32. Member masks are `u32`.
- Weak memory as *values*. Release and acquire are modelled, but an atomic load always
  returns a value that was genuinely stored, never a stale one a weakly ordered machine
  could hand back. Riri checks synchronisation, not visibility.

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
| `sync.rs` | Vector clocks, fences, release and acquire |
| `explore.rs` | Searching across seeds, and shrinking a failing schedule |
| `diag.rs` | `Diagnostic`, `Report`, the de-duplicating reporter |
| `oxide.rs` | The cuda-oxide shaped surface |
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

### 5. Release and acquire between blocks

Blocks cannot share a barrier, so the only way they communicate is through global memory
with an atomic flag and a fence. Until that was modelled, every pair of accesses from
different blocks counted as concurrent, which made a correct handoff unreportable as
correct. It was a false positive on exactly the code most worth checking.

What carries the ordering is a *vector clock* per thread: for each other thread, the
highest operation of it this thread is known to have seen. Every instrumented access bumps
the acting thread's own counter and records it.

`<Table 3-2>` What a release publishes

| Operation | Effect |
|---|---|
| Release store or RMW | The acting thread's clock joins into the element's clock |
| Relaxed store after a release fence | The fence's snapshot joins into the element's clock |
| Acquire load or RMW | The element's clock joins into the acting thread's |
| Relaxed load | The element's clock is held aside for a later acquire fence |
| `threadfence()` | Both halves: pending loads are taken on, and a snapshot is left for later stores |

An earlier access by one thread is then ordered against a later access by another exactly
when the second thread's clock has caught up with the first access's counter. A thread that
has never synchronised has an empty clock, which orders nothing, so a kernel that does not
fence behaves precisely as it did before this existed.

Barriers feed the same machinery. On the way in, a thread contributes its clock to its
block; on the way out it takes back what the block collectively knows. Without that, a
thread raising a flag after a barrier would publish only its own writes, and the
block-mates' writes the barrier had just ordered would look unpublished. That composition,
a barrier and then a release, is how a multi-thread block hands anything over, and getting
it wrong reported a race in correct code.

Warp collectives do not feed the clocks. They order a warp through its warp epoch, which is
within a block and so not what the clocks are for. A lane that releases immediately after a
`sync_warp` publishes only its own work.

---

## IV. Shadow Memory

### 1. Per-element state

Every element of every instrumented buffer carries a shadow cell: an initialised flag, the
last write, a bounded set of recent readers, and whatever a release has published to it.
Shadow state is per launch. Global memory keeps its values across launches but resets its
access history; shared memory resets both, since it is created fresh per launch and starts
uninitialised.

The initialised flag is what makes a read of memory nobody has written reportable. Shared
memory starts uninitialised by construction. A global buffer can too, through
`GlobalBuf::uninit`, which is how an output allocation arrives before a kernel fills it.
The flag survives the per-launch reset, because device memory written by one launch is not
uninitialised for the next.

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

### 6. How a collective is identified

A collective is identified by the source location of the call, captured with
`#[track_caller]` so that it is the caller's line rather than a line inside `warp.rs`.
Lanes rendezvous when they agree on that location and on the mask.

Two consequences follow, and they pull in opposite directions.

The reassuring one is that lanes cannot drift apart within a mask. The departure phase of
Chapter V, Section 3 releases mask-mates together, so a lane cannot reach iteration two of
a collective while a mask-mate is still at iteration one. Loops whose trip count varies by
lane are therefore caught in the ordinary way: the lane wanting another turn waits for
lanes that have moved on, and that is a convergence failure like any other.

The limiting one is that a location is not a program counter. A helper of the user's own
that wraps a collective reports *its* line for every call site unless it is itself marked
`#[track_caller]`. Two diverged groups calling that helper from different places then look
like one converged group, they rendezvous, and nothing is reported. This is a false
negative, and the only one in the warp model that is known and not yet closed.

Marking such a helper `#[track_caller]` restores the distinction, which makes it the
recommended practice for any wrapper around a collective. Closing the gap properly needs
call-path identity rather than a single frame, which is out of reach without either a
backtrace on every collective or MIR-level interpretation.

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

## VIII. Schedule Exploration

### 1. Why search at all

Detection is happens-before based, so for most faults the schedule is close to irrelevant:
a race between two accesses is reported because they are unordered, not because they landed
in a bad sequence. Searching across seeds therefore finds little that a single seed did not.

It earns its place on the cases where control flow depends on values that the interleaving
decides. A thread that publishes a flag late sends its peers down a different path, and only
some schedules take it. Those faults are invisible to one seed and ordinary to a sweep.

The larger benefit is the second half: turning a failure into something short.

### 2. Recording and replaying a schedule

Every scheduling decision is recorded as the *global index of the thread chosen*, not its
position among the runnable threads. Recording the thread is what lets a recording be edited
and still mean something: a plan naming a thread that is no longer runnable is recognisably
stale, and falls back rather than silently selecting a different thread.

Replay follows the plan while it lasts. Once it is spent, the scheduler keeps the current
thread running for as long as it can. That default matters as much as the plan: it is the
schedule with the fewest switches available, so the tail of any shrunk schedule is the least
surprising continuation rather than more noise.

A consequence worth stating plainly: a schedule is a complete reproducer by itself. It needs
no seed, and it survives any later change to how seeds are drawn.

### 3. Shrinking

Shrinking preserves a *fingerprint*: the kind of finding and the source locations involved,
with the block, thread, and index left out, since those vary between schedules. The question
at each step is "does this shorter schedule still find the same bug", not "does it produce an
identical report".

Two passes run in order, within a replay budget:

1. **Shortest prefix.** Try the empty plan, then double the prefix length until one
   reproduces, then bisect. This assumes a longer prefix is at least as likely to reproduce
   as a shorter one, which is a heuristic rather than a fact, so the result is a short prefix
   and not provably the shortest.
2. **Fewest switches.** Walk the surviving prefix, and wherever the turn changes hands, try
   letting the previous thread keep running instead. Keep the change when the finding
   survives.

The empty plan is tried first because it is the most useful possible answer. It means the
default in-order schedule already hits the bug, which tells the reader that no exotic
interleaving is involved.

### 4. When shrinking cannot be trusted

Shrinking replays the kernel many times, which assumes the kernel is a function of its
schedule. A kernel that captures a buffer and mutates it is not: each run starts from the
last run's output.

Rather than shrink against that, Riri replays the full recording first and checks that the
finding survives. If it does not, no shrinking is attempted and the reason is reported.

`<Table 8-1>` Outcomes of shrinking

| Outcome | Meaning |
|---|---|
| `Minimised` | A schedule that replays and reproduces the finding |
| `Disabled` | Shrinking was turned off by the caller |
| `TraceTruncated` | The run made more decisions than are recorded, so it cannot be replayed |
| `NotReproducible` | Replaying the full recording lost the finding, so the kernel carries state between runs |

The cure for `NotReproducible` is to build fresh state per run, which the API supports
directly. See `api.md`, Chapter VII.

---

## IX. The cuda-oxide Shim

### 1. What it is for

`crate::oxide` offers the surface a cuda-oxide kernel is written against, so such a kernel
runs under Riri unchanged. Only the `#[kernel]` attribute differs between the two builds,
and `cfg_attr` carries it on the GPU build alone, which is why Riri needs no proc macro and
keeps its empty dependency list.

The shim is not where the checking happens. It is an adapter onto the same instrumentation
every other kernel uses, so everything in Chapters III to VI applies unchanged.

Its value is narrower than it first appears. In Tier 1 the inputs are `&[T]`, which nothing
writes during a launch, and the only write path is `DisjointSlice`, whose checked accessors
give each thread a distinct index. Races there are impossible by construction and Riri adds
nothing. Tier 2 is the point: `get_unchecked_mut` asserts that an index belongs to the
calling thread alone, nothing on hardware checks it, and under Riri a second claim on the
same element is an ordinary data race naming both lines.

`<Table 9-1>` Shim coverage

| Surface | State |
|---|---|
| `thread::index_1d`, `threadIdx_*`, `blockIdx_*`, `blockDim_*` | Covered |
| `thread::sync_threads` | Covered |
| `DisjointSlice::get_mut_indexed`, `get_mut`, `get_unchecked_mut`, `len` | Covered |
| `warp` shuffles, votes, `lane_id`, `warp_id` | Covered, unsuffixed forms only |
| `SharedArray`, `DynamicSharedArray` | Absent by decision, see Section 5 |
| 2D and tiled index spaces, managed barriers, clusters, TMA | Absent |

### 2. Giving a context-free API a context

A cuda-oxide kernel takes its identity from hardware registers, so `thread::index_1d()` is
a free function with no context argument. Riri's own surface passes a `ThreadCtx`
explicitly, and the shim has to bridge the two.

It works because Riri gives each simulated GPU thread its own OS thread. A launch parks
each thread's share of the state, an `Arc<LaunchState>` plus its indices, in a thread local,
and every shim function rebuilds a `ThreadCtx` from it on demand.

The `Arc` is what keeps this free of `unsafe`. Parking a `&ThreadCtx` would mean erasing a
lifetime and storing a raw pointer; parking an owned handle does not. A guard clears the
binding when the kernel returns, including by unwinding, so a device function called outside
a launch gets a clear panic rather than another launch's state.

### 3. Handing out a mutable element

cuda-oxide's `get_mut` returns `&mut T`, and Riri keeps buffer contents behind a mutex, so
it cannot hand out a reference into them and still record the access.

`ElemMut` resolves this. It holds the mutex guard and implements `Deref` and `DerefMut`, so
`*elem = x` works exactly as it reads while the access is recorded before the borrow is
handed over. Holding the lock for the life of the borrow is sound here because only the
thread holding the turn runs, so there is nobody to contend with. Holding one across a
barrier would wedge the launch, which is documented rather than prevented.

### 4. Carrying the caller's location

Every shim function reaches Riri through a closure, and `#[track_caller]` does not survive
one: a location captured inside the closure is a line in `oxide.rs`, not in the kernel.

The first working version of the shim reported races inside Riri itself, which is useless
when naming the line is the product. The instrumentation entry points therefore take a
location rather than capturing one, and each shim function captures its own caller and
passes it down. The same applies to `sync_threads` and to every warp collective, which is
why `warp.rs` carries an `_at` variant of each public function.

### 5. Shared memory, and why it is absent

cuda-oxide's `SharedArray` is not a container. It is `#[repr(transparent)]` over
`PhantomData`, a zero-sized marker their compiler recognises and backs with storage in
address space 3, and every accessor on it is `unreachable!("called outside CUDA kernel
context")`. There is no off-device implementation to borrow.

Riri would therefore have to supply real storage. Per-block instancing, which looks like
the hard part, is not: blocks are never ordered against each other, so running them one at
a time would give each block its own instance and cost no detection at all. The obstacle is
narrower and firmer. `Index` and `IndexMut` hand out references, the storage starts
uninitialised, and a lock cannot return a reference, so the only route is `MaybeUninit`
behind `unsafe`. That would cost Riri its freedom from `unsafe`, which is a property worth
more than one surface.

The spelling is also unsettled. Under edition 2024 the `static mut` access in their own
examples runs into `static_mut_refs`, and they have added `as_raw_mut_ptr` with
`&raw mut SCRATCH` as the pattern for threads deriving disjoint pointers. Building against
`Index` now would be building against something already being superseded.

Kernels needing shared memory use `ThreadCtx::shared`, which is checked exactly as the rest
of Riri is.

---

## X. Assessment

### 1. Advantages

- Detection does not depend on an unlucky schedule, so absence of a report is meaningful
  within the model's limits, and presence of one is reproducible.
- No GPU, no driver, no `unsafe`, and no dependencies. It is a normal test binary.
- Failures name a source line rather than a symptom.
- Warp divergence is reported instead of hanging, which is the failure mode that makes the
  same bug expensive to diagnose on hardware.
- A failing schedule reduces to a short list of thread indices that reproduces on its own,
  which is a reproducer that fits in a test and does not depend on the seed scheme.

### 2. Disadvantages

- Kernels must be written against Riri's API or the cuda-oxide shim. Code using any other
  surface cannot be moved under Riri without being ported.
- The shim is written from cuda-oxide's published API reference rather than compiled against
  `cuda-device`, which is unpublished and pins a nightly toolchain. Signatures can drift and
  nothing here would notice.
- One OS thread per simulated GPU thread caps launches at 16,384 threads and makes large
  launches slow.
- A collective is identified by its source location, so a helper wrapping a collective
  hides divergence between its call sites unless it is marked `#[track_caller]`. See
  Chapter V, Section 6.
- Epoch counters are coarser than vector clocks. Partial-mask collectives are conservatively
  treated as ordering nothing, which can produce false positives among their participants.
- The 8-reader bound can miss a write-after-read race when more than 8 threads read an
  element concurrently.
- Prefix shrinking assumes monotonicity, so it returns a short schedule rather than a
  provably minimal one, and stops when its replay budget runs out.
- Searching across seeds adds little for faults that detection already finds structurally.
  It pays off only where control flow depends on values the interleaving decides.

### 3. Conditions under which this design is inappropriate

- Kernels whose correctness depends on weak memory ordering or fences beyond atomics.
- AMD wavefronts, or any hardware with a warp wider than 32 lanes.
- Performance work of any kind.
- Kernels that cannot be expressed as a Rust closure over Riri's memory types, such as those
  relying on inline PTX or on vendor intrinsics with no model here.

---

## XI. Dependencies

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
