# Riri

**A Miri-style undefined-behaviour detector for Rust GPU kernels.**

Riri runs SIMT kernels on the CPU. Every GPU thread becomes a simulated
thread, a seeded scheduler interleaves them one memory operation at a time,
and every access goes through shadow memory. No GPU needed, so it runs in
plain `cargo test` and in CI.

It catches the bugs that safe GPU Rust can't rule out yet:

| Bug | Detected |
| --- | --- |
| Data races on shared memory (missing `sync_threads`) | ✅ |
| Data races on global memory, including across blocks | ✅ |
| Plain accesses racing with atomics | ✅ |
| Barrier divergence (`sync_threads` inside a divergent branch) | ✅ |
| Warp shuffles on diverged warps (a lane in the mask never arrives) | ✅ |
| Lanes disagreeing about a collective's member mask | ✅ |
| Shuffles reading a lane outside the mask | ✅ |
| Reads of uninitialised shared memory | ✅ |
| Out-of-bounds accesses, kernel panics (reported as traps) | ✅ |
| Rust aliasing violations (Tree Borrows) across threads | 🔜 roadmap |

## Why

Projects like [cuda-oxide](https://github.com/NVlabs/cuda-oxide) make the
common case safe (one thread writes one element), but cooperative patterns
such as shared-memory reductions, scans, and producer/consumer pipelines still
need `unsafe`. Today the only dynamic check for those is NVIDIA Compute
Sanitizer, which needs a real GPU and knows nothing about Rust's semantics.
Riri aims to be to GPU kernels what Miri is to CPU `unsafe` code: a tool you
run in CI that tells you *which line* broke *which rule*, reproducibly.

## Example

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

One bug, two ways of seeing it: the missing barrier both races and lets a
thread read a tile element nobody has written yet.

Warp collectives are checked the same way. Every lane named by a member mask
must reach the collective; if one branched away or exited, Riri says which:

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

Riri reports the moment the first excluded lane exits, so how many lanes have
parked by then depends on the schedule. The mask and the missing lanes do
not: they come from the code, not the interleaving.

Two demos run a correct kernel beside a buggy one — the classic "barrier
removed from the loop" bug, and the warp-synchronous shuffle hidden inside a
branch:

```sh
cargo run --example reduction
cargo run --example warp_reduce
```

## How it works

- **Scheduler.** Each simulated thread is backed by an OS thread, but only one
  holds the *turn*. At every instrumented operation the turn passes to a
  runnable thread picked by a seeded RNG, so a seed reproduces a schedule
  exactly. Different seeds explore different interleavings.
- **Happens-before.** Within a block, `sync_threads()` orders everything
  before it against everything after it (each barrier bumps the block's
  *epoch*). Within a warp, a *full-mask* collective does the same for that
  warp alone (bumping its *warp epoch*); a partial mask bumps nothing, since
  it says nothing about the lanes it leaves out. Across blocks nothing is
  ordered within a launch. Atomics never race with each other. Two accesses
  race if they're concurrent under this model and at least one writes.
  Because detection is happens-before based, a race is found regardless of
  which interleaving the seed produced.
- **Barriers.** A block's barrier releases when all its threads arrive. If a
  thread exits while others wait, that's barrier divergence (the conservative
  CUDA C++ rule).
- **Warp collectives.** Each is a rendezvous: lanes park until every lane in
  the mask arrives at the *same* collective with the *same* mask, then they
  are released together. Riri never deadlocks on a warp that cannot
  reconverge — when no thread can run, it works out who was waiting on whom
  and reports that instead, naming the lanes that never arrived.
- **Shadow memory.** Per element: initialised flag, last write, recent
  readers. Diagnostics are de-duplicated by source location, so a racy line in
  a 1024-thread kernel produces one report.

## Current scope (v0.2)

Riri is at the *library emulator* stage: kernels are written against Riri's
API (`ThreadCtx`, `GlobalBuf`, `SharedArray`) and run as ordinary Rust
closures. That's enough to prove out the detection model and be useful for
testing kernel *algorithms*. Known limits:

- Kernels must use Riri's types; it doesn't yet run cuda-oxide or rust-cuda
  source unchanged.
- Lanes are scheduled independently rather than in lockstep, which is
  stricter than real hardware and closer to Volta+ independent thread
  scheduling.
- A collective is identified by its source location, so two lanes at the same
  line in *different loop iterations* are treated as converged. Divergence
  across iterations of the same loop is not caught yet.
- Member masks are `u32`, so warp sizes above 32 (AMD wavefronts) are not
  modelled. `LaunchConfig::warp_size` accepts any power of two up to 32,
  which is mostly useful for writing small, readable warp tests.
- No `__threadfence`/memory-order modelling beyond atomics.
- Each element remembers at most 8 recent readers; beyond that, some
  write-after-read races can be missed.
- Launches are capped at 16,384 threads (each is an OS thread).

## Roadmap

1. ~~**Warp model.** Lane masks, `shfl_sync`/`ballot_sync` with convergence
   checks, warp-synchronous bugs.~~ Done in v0.2.
2. **cuda-oxide API shim.** A `cuda_device`-compatible surface
   (`thread::index_1d`, `DisjointSlice`, `SharedArray`, `sync_threads`) so
   the *same kernel source* runs on the GPU and under Riri via a `cfg` switch.
   Also validates `DisjointSlice::get_unchecked_mut` uniqueness claims at
   runtime.
3. **Schedule exploration.** `check(seeds)` / systematic exploration, plus
   minimising a failing seed to a short interleaving.
4. **MIR-level interpretation.** The real Miri move: interpret the kernel's
   MIR with SIMT threads, applying Tree Borrows across lanes, so arbitrary
   `unsafe` in kernels is checked without rewriting it.
5. **Memory fences and weak memory** for global-memory communication.

## Prior art

- [Miri](https://github.com/rust-lang/miri): the inspiration; CPU only.
- NVIDIA Compute Sanitizer (`racecheck`, `synccheck`): real GPU, C++-level.
- GPUVerify: static
  verification of OpenCL/CUDA kernels.
- Descend (Köpcke, Gorlatch, Steuwer, PLDI 2024) and
  [warp-types](https://crates.io/crates/warp-types): type-level safety;
  Riri is the dynamic complement for what types can't prove yet.
- [loom](https://github.com/tokio-rs/loom): deterministic concurrency testing
  on the CPU.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.
