//! Handing data from one block to another, run twice: once with the fences
//! and once without.
//!
//! Blocks cannot use a barrier with each other, so the only way to communicate
//! is through global memory plus atomics and fences. The producer writes its
//! data, fences, then raises a flag; the consumer waits on the flag, fences,
//! then reads the data.
//!
//! Drop the fences and the kernel usually still works on hardware, because a
//! write that reached memory before the flag did tends to be visible after it.
//! Nothing guarantees that, and the answer is wrong whenever it is not.
//!
//! cargo run --example message_passing

use riri::{launch, GlobalBuf, LaunchConfig, Ordering};

fn main() {
    for fenced in [true, false] {
        let data = GlobalBuf::new("data", vec![0u32; 4]);
        let flag = GlobalBuf::new("flag", vec![0u32; 1]);
        let seen = GlobalBuf::new("seen", vec![0u32; 4]);

        let report = launch(&LaunchConfig::new(2, 4).seed(2026), |t| {
            let lane = t.thread_linear();
            if t.block_linear() == 0 {
                data.write(t, lane, 100 + lane as u32);
                // One thread raises the flag once the block has written.
                t.sync_threads();
                if lane == 0 {
                    if fenced {
                        t.threadfence();
                    }
                    flag.atomic_store(t, 0, 1, Ordering::Relaxed);
                }
            } else {
                let mut arrived = false;
                for _ in 0..200 {
                    if flag.atomic_load(t, 0, Ordering::Relaxed) == 1 {
                        arrived = true;
                        break;
                    }
                }
                if arrived {
                    if fenced {
                        t.threadfence();
                    }
                    let value = data.read(t, lane);
                    seen.write(t, lane, value);
                }
            }
        });

        println!(
            "== {} handoff: consumer saw {:?}",
            if fenced { "fenced" } else { "unfenced" },
            seen.to_vec()
        );
        print!("{report}");
        println!();
    }
}
