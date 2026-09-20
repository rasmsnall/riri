# riri-mir

A rustc driver that extracts MIR, as the first step towards interpreting GPU
kernels rather than executing them against an instrumented API.

**Status: it interprets scalars.** It compiles a crate normally, enters the
`rustc_public` context once analysis is finished, and either prints a
function's MIR or runs it.

What it evaluates is the shape a body actually has once rustc is done with it:
locals, assignments, checked arithmetic with the overflow `Assert` rustc
attaches, comparisons at the right signedness, `SwitchInt`, casts, and tuple
fields. What it does not have is memory, calls, or any of the SIMT layer, which
is to say it is not yet a GPU interpreter. See [What comes next](#what-comes-next).

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

Set `RIRI_MIR_ONLY` to a substring to print only the functions that match, or
`RIRI_MIR_RUN` to interpret them instead of printing them:

```sh
RIRI_MIR_RUN=sum_to_ten ./target/debug/riri-mir \
  --edition 2021 --crate-type lib fixtures/interp.rs --out-dir fixtures/out
```

```text
interp::sum_to_ten = 45_u32
```

Printing rather than running gives the MIR itself, which is what an
interpreter consumes:


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

## Checking it

`./check.sh` runs every body in `fixtures/interp.rs` and diffs the results
against `fixtures/expected.txt`. There is no CI for this crate, so that script
stands in for one.

The expected values are not hand-computed. `fixtures/check.rs` includes the
same file and runs the same functions natively, so the two can be compared
directly:

```text
interpreted                                          native
interp::sum_to_ten = 45_u32                          sum_to_ten = 45
interp::signed_division = -2_i32                     signed_division = -2
interp::truncating_cast = 44_u8                      truncating_cast = 44
interp::sign_extending_cast = -3_i64                 sign_extending_cast = -3
interp::comparisons = true                           comparisons = true
interp::shifts = 32_u32                              shifts = 32
interp::overflows stopped: attempt to add            overflows panicked = true
  with overflow
```

The last row is the one worth looking at: the interpreter honours the `Assert`
terminator rustc emits for checked arithmetic, and reports it in the words the
panic would have used. `comparisons` is the other one, since it returns `true`
only if `-1 < 1` was decided as signed.

## Not in CI

Building this needs a pinned nightly and the `rustc-dev` component, which is a
large download for every run. The library's CI stays fast and stable-only.
This is built by hand for now.

## What comes next

Done so far: a value model for scalars and aggregates, and evaluation of the
statements and terminators that do not touch memory.

What remains, in rough order, each a project rather than a task:

1. **Memory.** A `Place` has to resolve to an address rather than a slot, with
   provenance attached, before `Deref`, `Index`, `Ref` or `AddressOf` mean
   anything. This is the step that unlocks the rest, and the one where Riri's
   existing shadow memory starts being reusable.
2. **Calls.** A call stack, plus shims for the intrinsics and library functions
   a kernel reaches. Miri has hundreds of these; a kernel subset is smaller but
   not small.
3. **The SIMT layer.** One interpreter state per lane, driven by the scheduler
   the library already has. This is where the two halves of the project meet,
   and where the existing scheduler, shadow memory and diagnostics pay off.
4. **Tree Borrows across lanes**, which is the point of the exercise and the
   thing no amount of API instrumentation can reach.

Step 1 is the next one to do and the largest single jump. Everything up to now
has been a body that computes; nothing has touched a pointer.
