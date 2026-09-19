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
- **Understands warps, not just threads.** Warp collectives are rendezvous points. Every
  lane named by a member mask must reach the same collective with the same mask, and Riri
  checks that contract rather than trusting it. See [`docs/architecture.md`](docs/architecture.md),
  Chapter V.
- **Never hangs on the bug it is looking for.** A warp that cannot reconverge would
  deadlock on hardware. Riri detects that no thread can make progress, works out who was
  waiting on whom, and reports it. A block barrier stalled behind an unreconvergeable warp
  is named as the symptom it is, and the warp is blamed instead.
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
| Reads of uninitialised shared memory | Yes |
| Out-of-bounds accesses and kernel panics, reported as traps | Yes |
| Rust aliasing violations, Tree Borrows across lanes | Not yet, see Roadmap |
| Divergence between iterations of the same loop | Not yet, see Scope |

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

## Examples

Two demos run a correct kernel beside a buggy one, printing both reports:

```
cargo run --example reduction     # block reduction, barrier removed from the loop
cargo run --example warp_reduce   # warp reduction, shuffle hidden inside a branch
```

## Development

```
cargo build --all-targets
cargo test
cargo run --example reduction
cargo run --example warp_reduce
```

These are exactly the gates CI runs, on stable and on 1.75, the declared minimum supported
version. There are no dependencies and no feature flags.

## Scope and limits

Riri is at the *library emulator* stage. Kernels are written against Riri's API rather than
compiled from cuda-oxide or rust-cuda source, which is enough to prove out the detection
model and to test kernel *algorithms*, but is not yet the full Miri move.

- Kernels must use Riri's types. It does not yet run cuda-oxide or rust-cuda source
  unchanged. Closing that gap is Roadmap item 1.
- A collective is identified by its source location, so two lanes at the same line in
  *different loop iterations* are treated as converged. Divergence across iterations of one
  loop is not caught.
- Member masks are `u32`, so warp sizes above 32 are not modelled. `LaunchConfig::warp_size`
  accepts any power of two up to 32, which is mainly useful for writing small readable warp
  tests. AMD 64-lane wavefronts are out of scope.
- No `__threadfence` or memory-order modelling beyond atomics.
- Each element remembers at most 8 recent readers. Beyond that, some write-after-read races
  can be missed.
- Launches are capped at 16,384 threads, because each simulated thread is an OS thread.

## Roadmap

1. **cuda-oxide API shim.** A `cuda_device`-compatible surface so the *same kernel source*
   runs on the GPU and under Riri behind a `cfg` switch, and so
   `DisjointSlice::get_unchecked_mut` uniqueness claims are validated at runtime.
2. **Schedule exploration.** Systematic exploration across seeds, and minimising a failing
   seed to a short interleaving.
3. **MIR-level interpretation.** The real Miri move: interpret the kernel's MIR with SIMT
   threads, applying Tree Borrows across lanes, so arbitrary `unsafe` in a kernel is checked
   without rewriting it.
4. **Memory fences and weak memory** for global-memory communication.

Item 0, a warp model with shuffle convergence checks, shipped in v0.2.

## Documentation

| Document | Contents |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | Why the design is shaped this way: execution model, happens-before, shadow memory, the warp model, the failure model |
| [`docs/api.md`](docs/api.md) | The callable surface: `launch`, `ThreadCtx`, memory types, warp collectives, reports |

## Why this exists

Projects like [cuda-oxide](https://github.com/NVlabs/cuda-oxide) make the common case safe,
where one thread writes one element. Cooperative patterns such as shared-memory reductions,
scans, and producer/consumer pipelines still need `unsafe`, and cuda-oxide's own
documentation records that its `DisjointSlice` does not cover them.

The only dynamic check for those today is NVIDIA Compute Sanitizer, which needs a real GPU
and knows nothing about Rust's semantics. Riri aims to be to GPU kernels what Miri is to
CPU `unsafe` code: a tool that runs in CI and names the line.

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
