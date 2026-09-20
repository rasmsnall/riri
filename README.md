# riri

A Miri-style undefined-behaviour detector for GPU kernels written in Rust. It runs SIMT
kernels on the CPU and reports which line broke which rule, reproducibly, with no GPU
involved.

It removes the GPU from the debugging loop. Instead of running a kernel on real hardware
and inferring a race from a wrong answer, every thread becomes a simulated thread, a
seeded scheduler interleaves them one memory operation at a time, and every access is
checked against a happens-before model.

```
kernel -> simulated threads -> seeded scheduler -> shadow memory -> diagnostics
```

## What it does

- **Finds races by reasoning, not by luck.** Detection is happens-before based, so a race
  is reported whichever interleaving the seed happened to produce. Warp convergence is
  checked structurally for the same reason: a diverged shuffle is caught on every seed,
  not on the unlucky one. Running the same seed twice reproduces the same schedule
  exactly, so a failure found in CI reproduces on a laptop.
- **Shrinks a failing schedule to something readable.** A failing seed already
  reproduces exactly, but the interleaving behind it can be hundreds of decisions long.
  Riri cuts it down to the few that matter and hands back a short list of thread indices
  that reproduces the finding on its own, with no seed involved. Often the answer is that
  no decisions are needed at all, which is itself worth knowing: the bug is not an exotic
  race, it is there whenever threads run in plain order.
- **Runs cuda-oxide shaped kernels.** A kernel written against `cuda_device` runs under
  Riri with its source unchanged, because the same surface is provided here:
  `thread::index_1d`, `DisjointSlice`, `warp::shuffle`, `threadfence`, and atomics that
  index as they do on the GPU, so `counters[i].fetch_add(1, AtomicOrdering::Relaxed)`
  needs no changing. The point is
  Tier 2: `get_unchecked_mut` asserts an index is the calling thread's alone, nothing on
  hardware verifies that, and under Riri two threads claiming one element is an ordinary
  data race with both lines named. See [`riri::oxide`](docs/api.md).
- **Understands warps, not just threads.** Warp collectives are rendezvous points. Every
  lane named by a member mask must reach the same collective with the same mask, and Riri
  checks that contract rather than trusting it. See [`docs/architecture.md`](docs/architecture.md),
  Chapter V.
- **Never hangs on the bug it is looking for.** A warp that cannot reconverge would
  deadlock on hardware. Riri detects that no thread can make progress, works out who was
  waiting on whom, and reports it. A block barrier stalled behind an unreconvergeable warp
  is named as the symptom it is, and the warp is blamed instead.
- **Follows release and acquire between blocks.** Blocks cannot use a barrier with each
  other, so they hand data over through global memory with a fence and a flag. Riri tracks
  that with a vector clock per thread, which means a correct handoff is reported clean and
  the same kernel without its fences is reported as the race it is. Kernels that never
  fence pay nothing: an unsynchronised clock orders nothing, which is the old behaviour
  exactly.
- **Models memory ordering per scope.** A block barrier orders the block. A full-mask warp
  collective orders that warp alone. A partial mask orders nothing, because it says nothing
  about the lanes it leaves out. Warp-synchronous code that is correct is reported clean,
  and code that leans on lockstep execution is not.
- **Runs in ordinary CI.** No GPU, no driver, no `unsafe`, and no dependencies outside the
  standard library. It is a normal `cargo test`.

It is deliberately stricter than real hardware in one direction: lanes are scheduled
independently rather than in lockstep, so warp-synchronous code that happens to work today
is still reported. That is the intent, since such code is undefined behaviour that survives
until a compiler or architecture change removes it.

## What it catches

| Bug | Detected |
|---|---|
| Data races on shared memory, such as a missing `sync_threads` | Yes |
| Data races on global memory, including across blocks | Yes |
| Plain accesses racing with atomics | Yes |
| Barrier divergence, such as `sync_threads` inside a divergent branch | Yes |
| Warp divergence: a lane named by a shuffle's mask never arrives | Yes |
| Lanes disagreeing about a collective's member mask | Yes |
| Shuffles reading a lane outside the mask | Yes |
| Cross-block handoffs missing a `threadfence` or release/acquire pair | Yes |
| Reads of uninitialised shared memory | Yes |
| Reads of global memory nothing has written | Yes, from `GlobalBuf::uninit` |
| Out-of-bounds accesses and kernel panics, reported as traps | Yes |
| cuda-oxide `get_unchecked_mut` claiming one element twice | Yes |
| Rust aliasing violations, Tree Borrows across lanes | Not yet, see Roadmap |
| Divergence between two call sites of one helper | Only if the helper is `#[track_caller]`, see Scope |

## Usage

A kernel is an ordinary Rust closure taking a [`ThreadCtx`]. Memory it touches goes through
Riri's instrumented types so it can be checked.

```rust
use riri::{launch, GlobalBuf, LaunchConfig};

let out = GlobalBuf::new("out", vec![0u32; 32]);

let report = launch(&LaunchConfig::new(1, 32).seed(3), |t| {
    let tile = t.shared::<u32>("tile", 32);
    let i = t.thread_linear();
    tile.write(t, i, i as u32);
    // BUG: missing t.sync_threads();
    out.write(t, i, tile.read(t, (i + 1) % 32));
});

println!("{report}");
```

```text
riri: 2 diagnostic(s) (seed 3)
  - data race on shared `tile` (block 0)[21]: block 0 thread 21 (Write at src/main.rs:8) conflicts with block 0 thread 20 (Read at src/main.rs:10) with no barrier between them
  - read of uninitialised shared `tile` (block 0)[19] by block 0 thread 18 (Read at src/main.rs:10)
```

One bug, seen two ways: the missing barrier both races and lets a thread read a tile
element that nobody has written yet.

Warp collectives are checked the same way. Every lane named by a member mask must reach the
collective, and if one branched away or exited, Riri says which:

```rust
use riri::{launch, warp, LaunchConfig};

let report = launch(&LaunchConfig::new(1, 32).seed(1), |t| {
    if t.lane_id() < 16 {
        // BUG: the mask names all 32 lanes, but only half of them are here.
        let _ = warp::shfl_sync(t, warp::FULL_MASK, t.lane_id(), 0);
    }
});

println!("{report}");
```

```text
riri: 1 diagnostic(s) (seed 1, launch aborted)
  - warp divergence in block 0 warp 0: `shfl_sync` at src/main.rs:6 names lanes 0-31 but only lanes 10 arrived, so lanes 0-9,11-31 never reach it
```

Riri reports the moment the first excluded lane exits, so how many lanes have parked by
then depends on the schedule. The mask and the missing lanes do not: they come from the
code, not the interleaving.

In a test, assert on the report rather than printing it:

```rust
report.assert_clean();              // panics with the full listing if anything was found
assert!(report.has_race());
assert!(report.has_warp_divergence());
```

Blocks cannot share a barrier, so they hand data over with a fence and a flag. Riri follows
that:

```rust
launch(&LaunchConfig::new(2, 1), |t| {
    if t.block_linear() == 0 {
        data.write(t, 0, 42);
        t.threadfence();
        flag.atomic_store(t, 0, 1, Ordering::Relaxed);
    } else {
        while flag.atomic_load(t, 0, Ordering::Relaxed) == 0 {}
        t.threadfence();
        let _ = data.read(t, 0);          // ordered, not a race
    }
});
```

Remove either fence and the read of `data` is reported against the write, because nothing
then orders the two blocks. `atomic_store(.., Ordering::Release)` paired with
`atomic_load(.., Ordering::Acquire)` does the same job without the fences.

Searching across schedules, rather than running one, is [`explore`]:

```rust
use riri::{Explore, GlobalBuf, LaunchConfig, ThreadCtx};

let found = Explore::new(&LaunchConfig::new(1, 4)).seeds(64).run_with(|| {
    let flag = GlobalBuf::new("flag", vec![0u32; 1]);
    let out = GlobalBuf::new("out", vec![0u32; 1]);
    move |t: &ThreadCtx<'_>| {
        let i = t.thread_linear();
        if i == 0 {
            flag.atomic_add(t, 0, 1, Ordering::Relaxed);
        } else if flag.atomic_add(t, 0, 0, Ordering::Relaxed) == 0 {
            // Racy, but only for threads that ran before thread 0 published.
            out.write(t, 0, i as u32);
        }
    }
});

println!("{found}");
```

```text
riri: explored 2 seed(s), seed 1 failed in 10 decision(s), shrunk to 5 decision(s), 2 switch(es)
  - data race on global `out`[0]: block 0 thread 1 (Write at src/main.rs:12) conflicts with block 0 thread 3 (Write at src/main.rs:12) with no barrier between them
  schedule: [1, 3, 3, 3, 1]
```

That schedule is the whole reproducer. Hand it to `replay` and the race comes back without
a seed, which also means it survives any later change to how seeds are drawn.

`run_with` builds fresh buffers for each run. Use plain `run` when the kernel's behaviour
does not depend on the values it reads; if a kernel carries state between runs, Riri says
so rather than shrinking against noise.

A kernel written for cuda-oxide runs with its source unchanged. Only the attribute differs,
and `cfg_attr` carries that:

```rust
use riri::oxide::{self, DisjointSlice};
use riri::{GlobalBuf, LaunchConfig};

#[cfg_attr(not(riri), cuda_device::kernel)]
fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
    if let Some((mut c_elem, idx)) = c.get_mut_indexed() {
        let i = idx.get();
        *c_elem = a[i] + b[i];
    }
}

let out = GlobalBuf::new("c", vec![0.0f32; 64]);
let c = DisjointSlice::new(&out);
let report = oxide::launch(&LaunchConfig::new(2, 32), || vecadd(&a, &b, c.clone()));
```

The kernel takes its identity from free functions rather than a context argument, exactly as
it does on the GPU, because each simulated thread parks its share of the launch in a thread
local. Tier 1 code like this is race-free by construction and Riri has nothing to add. Tier 2
is the point: `get_unchecked_mut` claims an index is the calling thread's alone, and under
Riri two threads claiming one element is a data race with both lines named.

## Examples

The first two run a correct kernel beside a buggy one and print both reports. The third
searches for a failing schedule and shrinks it:

```
cargo run --example reduction     # block reduction, barrier removed from the loop
cargo run --example warp_reduce   # warp reduction, shuffle hidden inside a branch
cargo run --example shrink        # searching schedules, then shrinking the failing one
cargo run --example oxide_kernel  # a cuda-oxide shaped kernel with a bad unchecked index
cargo run --example message_passing  # a block-to-block handoff, with and without fences
```

## Development

```
cargo fmt --all --check
cargo clippy --all-targets -- -D warnings
cargo build --all-targets
cargo test
cargo run --example reduction
cargo run --example warp_reduce
cargo run --example shrink
cargo run --example oxide_kernel
cargo run --example message_passing
```

These are exactly the gates CI runs. Build, test and the examples run on stable and on
1.75, the declared minimum supported version; formatting and lints run on stable only,
since both tools change their output between releases. There are no dependencies and no
feature flags.

`riri-mir` is not part of any of this. It needs a pinned nightly with `rustc-dev`, and it
has its own [README](riri-mir/README.md).

## Scope and limits

Riri interprets nothing. It executes kernels as ordinary Rust against an instrumented
memory surface, which is enough to prove out the detection model and to test kernel
*algorithms*, but is not yet the full Miri move.

- Only what goes through Riri is checked: `GlobalBuf`, `SharedArray`, `DisjointSlice`, and
  the warp collectives. A kernel that talks to itself through a captured `AtomicUsize` is
  invisible and will be reported clean.
- The cuda-oxide shim covers Tier 1 indexing, `DisjointSlice`, block barriers, and the warp
  primitives. Shared memory is left out on purpose: their `SharedArray` is a zero-sized
  marker whose storage the cuda-oxide compiler provides and whose accessors all panic
  off-device, so Riri would have to supply storage and could only hand out references into
  it with `unsafe`. Use `ThreadCtx::shared` for those kernels.
- `cuda-device` is unpublished and pins a nightly toolchain, so Riri cannot depend on it and
  the shim is written from the published API reference rather than compiled against the real
  crate. Signatures can drift.
- A collective is identified by its source location, which is the call site thanks to
  `#[track_caller]`. Wrapping a collective in a helper of your own collapses every call site
  onto the helper's line, and two diverged groups calling that helper look to Riri like one
  converged group, so the divergence is missed. Put `#[track_caller]` on any helper that
  wraps a collective and the call sites stay distinct. Lanes that go round a loop a
  different number of times *are* caught, because the departure rendezvous keeps mask-mates
  in lockstep.
- Member masks are `u32`, so warp sizes above 32 are not modelled. `LaunchConfig::warp_size`
  accepts any power of two up to 32, which is mainly useful for writing small readable warp
  tests. AMD 64-lane wavefronts are out of scope.
- Release and acquire are modelled; the relaxed-atomic *value* semantics are not. Riri
  executes atomics in the order its scheduler picks, so a load returns some value that was
  actually stored, never a stale one that a weakly ordered machine could hand back. It
  checks synchronisation, not visibility.
- Each element remembers at most 8 recent readers. Beyond that, some write-after-read races
  can be missed.
- Launches are capped at 16,384 threads, because each simulated thread is an OS thread.

## Roadmap

1. **MIR-level interpretation.** The real Miri move: interpret the kernel's MIR with SIMT
   threads, applying Tree Borrows across lanes, so arbitrary `unsafe` in a kernel is checked
   without rewriting it. It would also close the helper-location gap for free, since MIR
   gives real program counters. [`riri-mir/`](riri-mir/) reaches MIR through `rustc_public`
   and now interprets scalar bodies, matching Rust on arithmetic, signedness, casts and
   overflow traps. It has no memory model, no calls and no SIMT layer yet, so it checks
   nothing. It needs a pinned nightly with `rustc-dev` and is not part of this crate or
   its CI.

Shipped: the warp model with shuffle convergence checks in v0.2, schedule exploration and
shrinking in v0.3, the cuda-oxide shim in v0.4, release and acquire ordering in v0.5.

## Documentation

| Document | Contents |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | Why the design is shaped this way: execution model, happens-before, shadow memory, the warp model, the failure model |
| [`docs/api.md`](docs/api.md) | The callable surface: `launch`, `ThreadCtx`, memory types, warp collectives, reports |

## Why this exists

cuda-oxide makes the common case safe, where one thread writes one element under a checked
launch contract. Its documentation keeps an explicit list of what it does not enforce,
under the heading [The hard problems](https://nvlabs.github.io/cuda-oxide/gpu-safety/the-safety-model.html).
Two entries on that list are what Riri is for.

**Shared memory access patterns.** `DisjointSlice` solves the unique-index-write pattern
but not cooperative ones: reductions, scans, and producer/consumer pipelines, where threads
deliberately touch overlapping regions with synchronisation between phases. Those stay in
the tier that needs `unsafe`.

**Warp-level convergence.** Collectives such as `shfl_sync` and `ballot_sync` require every
lane in the participation mask to be converged at the call site, and cuda-oxide records
that the type system cannot enforce this today. Pass a full mask from a diverged warp and
the result is, in their words, "a silent hang", with no crash and no message.

Both are stated as solvable rather than permanent, so the type-level answer may well arrive.
Riri is the dynamic complement in the meantime, and dynamic checking keeps its value
afterwards for the same reason Miri still matters: types rule out what they can prove, and
something has to check the rest. A warp that cannot reconverge is reported here as a
diagnostic naming the missing lanes, rather than as a kernel that never finishes.

The only dynamic check for any of this today is NVIDIA Compute Sanitizer, which needs a real
GPU and knows nothing about Rust's semantics. Riri aims to be to GPU kernels what Miri is to
CPU `unsafe` code: a tool that runs in CI and names the line.

## Against Compute Sanitizer

The incumbent for GPU kernels is NVIDIA's Compute Sanitizer, which is four tools. Riri
covers most of the same ground and is ahead on races, but the two are not substitutes:
Compute Sanitizer points at a real binary on real hardware at real scale, and Riri checks a
kernel written against its own API or the cuda-oxide shim, capped at 16,384 threads.

| Check | Compute Sanitizer | Riri |
|---|---|---|
| Out-of-bounds access | `memcheck` | Yes, reported as a trap |
| Misaligned access, leaks | `memcheck` | No: Riri indexes elements and models no allocator |
| Data races | `racecheck`, shared memory | Shared, global, and cross-block with fences |
| Uninitialised reads | `initcheck`, global memory | Shared, and global from `GlobalBuf::uninit` |
| Synchronisation hazards | `synccheck` | Barrier and warp divergence, mask disagreement, shuffle sources |

Two differences are worth the space. Riri decides races from happens-before rather than from
the execution it watched, so it reports a race whose bad interleaving never occurred. And it
needs no GPU, so it runs on every pull request rather than on a machine with a device in it.

Compute Sanitizer works at the C++ and PTX level and, as cuda-oxide's documentation puts it,
knows nothing about Rust's semantics. Riri will never check a C++ kernel. They are tools for
different languages that happen to look for the same bugs.

## Prior art

- [Miri](https://github.com/rust-lang/miri), the inspiration. CPU only.
- NVIDIA Compute Sanitizer (`racecheck`, `synccheck`). Real GPU, C++ level.
- GPUVerify, static verification of OpenCL and CUDA kernels.
- Descend (Köpcke, Gorlatch, Steuwer, PLDI 2024) and
  [warp-types](https://crates.io/crates/warp-types), type-level safety. Riri is the dynamic
  complement for what types cannot prove yet.
- [loom](https://github.com/tokio-rs/loom), deterministic concurrency testing on the CPU.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.

[`ThreadCtx`]: docs/api.md
[`explore`]: docs/api.md
