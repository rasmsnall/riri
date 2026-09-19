//! A classic block-level tree reduction, run twice: once correct, once with
//! the barrier inside the loop "optimised away" — a bug that often passes on
//! real hardware because warps happen to run in lockstep.
//!
//! cargo run --example reduction

use riri::{launch, GlobalBuf, LaunchConfig, ThreadCtx};

const BLOCK: u32 = 64;
const BLOCKS: u32 = 4;

fn reduce(t: &ThreadCtx<'_>, input: &GlobalBuf<u32>, partial: &GlobalBuf<u32>, buggy: bool) {
    let tile = t.shared::<u32>("tile", BLOCK as usize);
    let i = t.thread_linear();
    tile.write(t, i, input.read(t, t.global_linear()));
    t.sync_threads();

    let mut stride = BLOCK as usize / 2;
    while stride > 0 {
        if i < stride {
            let sum = tile.read(t, i) + tile.read(t, i + stride);
            tile.write(t, i, sum);
        }
        if !buggy {
            t.sync_threads();
        }
        stride /= 2;
    }

    if i == 0 {
        partial.write(t, t.block_linear(), tile.read(t, 0));
    }
}

fn main() {
    let n = (BLOCK * BLOCKS) as usize;
    let input = GlobalBuf::new("input", (1..=n as u32).collect());
    let expected: u32 = (1..=n as u32).sum();

    for buggy in [false, true] {
        let partial = GlobalBuf::new("partial", vec![0u32; BLOCKS as usize]);
        let report = launch(&LaunchConfig::new(BLOCKS, BLOCK).seed(2026), |t| {
            reduce(t, &input, &partial, buggy)
        });
        let total: u32 = partial.to_vec().iter().sum();
        println!(
            "== {} reduction: total {total} (expected {expected})",
            if buggy { "buggy" } else { "correct" }
        );
        print!("{report}");
        println!();
    }
}
