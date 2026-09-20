#!/usr/bin/env bash
# Runs every fixture body under the interpreter and diffs the results against
# what Rust itself produces.
#
# There is no CI for this crate, since it needs a pinned nightly with
# rustc-dev, so this stands in for one. Run it after touching the interpreter.
set -uo pipefail
cd "$(dirname "$0")"

FUNCTIONS=(
  sum_to_ten
  signed_division
  truncating_cast
  sign_extending_cast
  comparisons
  shifts
  overflows
)

SYSROOT="$(rustc --print sysroot)"
case "$(uname -s)" in
  MINGW* | MSYS* | CYGWIN*) export PATH="$(cygpath -u "$SYSROOT")/bin:$PATH" ;;
  Darwin) export DYLD_LIBRARY_PATH="$SYSROOT/lib:${DYLD_LIBRARY_PATH:-}" ;;
  *) export LD_LIBRARY_PATH="$SYSROOT/lib:${LD_LIBRARY_PATH:-}" ;;
esac

cargo build --quiet || exit 1
mkdir -p fixtures/out

DRIVER="./target/debug/riri-mir"
[ -x "$DRIVER.exe" ] && DRIVER="$DRIVER.exe"

actual=$(
  for f in "${FUNCTIONS[@]}"; do
    RIRI_MIR_RUN="$f" "$DRIVER" --edition 2021 --crate-type lib \
      fixtures/interp.rs --out-dir ./fixtures/out 2>&1 | grep "^interp::"
  done
)

if diff -u fixtures/expected.txt <(echo "$actual"); then
  echo "interpreter matches expected output (${#FUNCTIONS[@]} bodies)"
else
  echo "interpreter output changed" >&2
  exit 1
fi
