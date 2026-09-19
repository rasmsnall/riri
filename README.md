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
| Reads of uninitialised shared memory | ✅ |
| Out-of-bounds accesses, kernel panics (reported as traps) | ✅ |
| Warp shuffles on diverged warps | 🔜 roadmap |
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
riri: 1 diagnostic(s) (seed 3)
  - data race on shared `tile` (block 0)[5]: block 0 thread 5 (Write at src/main.rs:8) conflicts with block 0 thread 4 (Read at src/main.rs:10) with no barrier between them
```

Try the tree-reduction demo, which runs a correct kernel and one with the
classic "barrier removed from the loop" bug:

```sh
cargo run --example reduction
```

## How it works

- **Scheduler.** Each simulated thread is backed by an OS thread, but only one
  holds the *turn*. At every instrumented operation the turn passes to a
  runnable thread picked by a seeded RNG, so a seed reproduces a schedule
  exactly. Different seeds explore different interleavings.
- **Happens-before.** Within a block, `sync_threads()` orders everything
  before it against everything after it (each barrier bumps the block's
  *epoch*). Across blocks nothing is ordered within a launch. Atomics never
  race with each other. Two accesses race if they're concurrent under this
  model and at least one writes. Because detection is happens-before based,
  a race is found regardless of which interleaving the seed produced.
- **Barriers.** A block's barrier releases when all its threads arrive. If a
  thread exits while others wait, that's barrier divergence (the conservative
  CUDA C++ rule).
- **Shadow memory.** Per element: initialised flag, last write, recent
  readers. Diagnostics are de-duplicated by source location, so a racy line in
  a 1024-thread kernel produces one report.

## Current scope (v0.1)

Riri is at the *library emulator* stage: kernels are written against Riri's
API (`ThreadCtx`, `GlobalBuf`, `SharedArray`) and run as ordinary Rust
closures. That's enough to prove out the detection model and be useful for
testing kernel *algorithms*. Known limits:

- Kernels must use Riri's types; it doesn't yet run cuda-oxide or rust-cuda
  source unchanged.
- No warp model yet (every thread is scheduled independently, which is
  already stricter than Volta+ independent thread scheduling).
- No `__threadfence`/memory-order modelling beyond atomics.
- Each element remembers at most 8 recent readers; beyond that, some
  write-after-read races can be missed.
- Launches are capped at 16,384 threads (each is an OS thread).

## Roadmap

1. **Warp model.** Lane masks, `shfl_sync`/`ballot_sync` with convergence
   checks, warp-synchronous bugs.
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
