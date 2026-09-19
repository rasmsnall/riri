//! A warp-level reduction, run twice: once correct, once written the way
//! warp-synchronous code often is — the shuffle tucked inside a branch so
//! that "only the lanes that still have work" call it, while the member mask
//! still names the whole warp.
//!
//! That version is undefined behaviour. On real hardware it usually appears
//! to work, because the lanes that skipped the shuffle were going to be
//! thrown away anyway, right up until a compiler or architecture change
//! turns it into a hang or a wrong answer.
//!
//! cargo run --example warp_reduce

use riri::{launch, warp, GlobalBuf, LaunchConfig, ThreadCtx};

const WARP: u32 = 8;
const BLOCKS: u32 = 4;

fn reduce(t: &ThreadCtx<'_>, input: &GlobalBuf<u32>, partial: &GlobalBuf<u32>, buggy: bool) {
    let mask = t.warp_valid_mask();
    let mut v = input.read(t, t.global_linear());

    let mut delta = WARP / 2;
    while delta > 0 {
        if buggy && t.lane_id() >= delta {
            // BUG: this lane skips a collective its warp-mates are waiting on.
        } else {
            v += warp::shfl_down_sync(t, mask, v, delta);
        }
        delta /= 2;
    }

    if t.lane_id() == 0 {
        partial.write(t, t.block_linear(), v);
    }
}

fn main() {
    let n = (WARP * BLOCKS) as usize;
    let input = GlobalBuf::new("input", (1..=n as u32).collect());
    let expected: u32 = (1..=n as u32).sum();

    for buggy in [false, true] {
        let partial = GlobalBuf::new("partial", vec![0u32; BLOCKS as usize]);
        let config = LaunchConfig::new(BLOCKS, WARP).warp_size(WARP).seed(2026);
        let report = launch(&config, |t| reduce(t, &input, &partial, buggy));

        let total: u32 = partial.to_vec().iter().sum();
        println!(
            "== {} warp reduction: total {total} (expected {expected})",
            if buggy { "buggy" } else { "correct" }
        );
        print!("{report}");
        println!();
    }
}
