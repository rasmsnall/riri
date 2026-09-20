//! A cuda-oxide shaped kernel, run under Riri.
//!
//! cuda-oxide's Tier 1 makes the common case safe: one thread writes one
//! element, proved by a launch contract. Tier 2 is where a kernel drops to
//! `get_unchecked_mut` for speed, and there the uniqueness of the index is a
//! claim the compiler cannot check and the hardware will not complain about.
//!
//! This runs the same kernel twice, with the index arithmetic right and then
//! wrong. On a GPU the wrong one writes plausible-looking garbage. Here it is
//! an ordinary data race with both lines named.
//!
//! cargo run --example oxide_kernel

use riri::oxide::{self, thread, DisjointSlice};
use riri::{GlobalBuf, LaunchConfig};

const BLOCKS: u32 = 4;
const THREADS: u32 = 8;

/// Each thread writes the element it owns, addressed by hand.
///
/// `stride` should be the block size. Passing anything smaller makes two
/// blocks overlap, which is the bug.
fn scatter(mut out: DisjointSlice<u32>, stride: u32) {
    let i = thread::blockIdx_x() * stride + thread::threadIdx_x();
    let mut slot = out.get_unchecked_mut(i as usize);
    *slot = i;
}

fn main() {
    let n = (BLOCKS * THREADS) as usize;

    for (label, stride) in [("correct", THREADS), ("buggy", THREADS - 1)] {
        let out = GlobalBuf::new("out", vec![0u32; n]);
        let slice = DisjointSlice::new(&out);

        let config = LaunchConfig::new(BLOCKS, THREADS).seed(2026);
        let report = oxide::launch(&config, || scatter(slice.clone(), stride));

        let written = out.to_vec().iter().filter(|&&x| x != 0).count();
        println!("== {label} scatter: stride {stride}, {written} of {n} slots written");
        print!("{report}");
        println!();
    }
}
