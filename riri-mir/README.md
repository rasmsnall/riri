# riri-mir

A rustc driver that extracts MIR, as the first step towards interpreting GPU
kernels rather than executing them against an instrumented API.

**Status: a skeleton.** It does no interpretation. It compiles a crate
normally, enters the `rustc_public` context once analysis is finished, and
prints the MIR of the functions it finds. What it establishes is that the
pipeline works, which is the part worth proving before an interpreter is
written against it.

## Why it is separate from `riri`

| | `riri` | `riri-mir` |
|---|---|---|
| Toolchain | stable, MSRV 1.75 | `nightly-2026-08-28`, pinned |
| Components | default | `rustc-dev`, `llvm-tools`, `rust-src` |
| Dependencies | none | the compiler itself |
| Shape | library, runs under `cargo test` | binary, a rustc driver |

`rustc_public` (formerly `stable_mir`) is a stable *interface* to an unstable
compiler: the API is meant to hold across nightlies, but it ships inside the
rustc repository and is reached through `rustc_private`, so this binary links
against rustc and cannot be built on stable. None of the library's properties
survive that, so the two share neither a toolchain nor a workspace. The root
`Cargo.toml` excludes this directory from both the workspace and the published
package.

The nightly is pinned rather than floating because the API holding does not
mean the ABI of the `rustc_private` crates holds. It is the nightly cuda-oxide
builds on, so `rustc_public` is known to work there.

## Setup

```sh
rustup toolchain install nightly-2026-08-28 \
  --component rustc-dev --component llvm-tools --component rust-src --profile minimal
```

Roughly a gigabyte, mostly `rustc-dev`.

## Running it

```sh
cargo build
```

The driver supplies its own `--sysroot`. On Windows the toolchain's `bin`
directory must be on `PATH` so `rustc_driver`'s DLL resolves:

```sh
export PATH="$(cygpath -u "$(rustc --print sysroot)")/bin:$PATH"
./target/debug/riri-mir --edition 2021 --crate-type lib fixtures/kernel.rs --out-dir fixtures/out
```

On Linux or macOS the equivalent is `LD_LIBRARY_PATH` or `DYLD_LIBRARY_PATH`
pointing at `$(rustc --print sysroot)/lib`.

Set `RIRI_MIR_ONLY` to a substring to print only the functions that match.

Output for the bundled fixture:

```text
fn kernel::scatter
  6 block(s), 15 local(s)
  bb0:
    Assign(_7, CheckedBinaryOp(Mul, Copy(_4), Copy(_2)))
    -> Assert { cond: Move((_7.1: bool)), expected: false, msg: Overflow(Mul, ...), target: 1, ... }
  ...
  bb3:
    Assign(_12, AddressOf(FakeForPtrMetadata, (*_1)))
    Assign(_13, UnaryOp(PtrMetadata, Move(_12)))
    Assign(_14, BinaryOp(Lt, Copy(_5), Copy(_13)))
    -> Assert { cond: Move(_14), expected: true, msg: BoundsCheck { len: Move(_13), index: Copy(_5) }, ... }
  bb4:
    Assign((*_1)[_5], Cast(IntToInt, Copy(_5), Ty { id: 2, kind: RigidTy(Uint(U32)) }))
    -> Goto { target: 5 }
```

That is the material an interpreter needs: assignments, checked arithmetic,
bounds checks as explicit `Assert` terminators, and indexed stores.

## Not in CI

Building this needs a pinned nightly and the `rustc-dev` component, which is a
large download for every run. The library's CI stays fast and stable-only.
This is built by hand for now.

## What comes next

In rough order, each of which is a project rather than a task:

1. A value model: scalars, aggregates, and pointers with provenance.
2. A memory model, so a `Place` resolves to an address and Riri's existing
   shadow memory can record the access.
3. Statement and terminator evaluation, plus shims for the intrinsics and
   library calls a kernel reaches.
4. The SIMT layer: one interpreter state per lane, driven by the scheduler the
   library already has.
5. Tree Borrows across lanes, which is the point of the exercise and the thing
   no amount of API instrumentation can reach.

Step 4 is where this rejoins `riri`: the scheduler, shadow memory, and
diagnostics are all reusable as they stand. Steps 1 to 3 are the cost of
admission, and they are what Miri spent years on.
