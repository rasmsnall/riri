//! Searching across schedules, then shrinking the failing one.
//!
//! Two kernels, because they shrink to very different answers.
//!
//! The first is an ordinary missing-barrier race. Riri finds it structurally,
//! from the happens-before relation rather than from catching the accesses in
//! a bad order, so it does not need a peculiar interleaving at all. Shrinking
//! says so: the schedule collapses to nothing, meaning threads running in
//! plain order already hit it.
//!
//! The second only goes wrong when one thread is late. Shrinking has to keep
//! the decisions that hold it back, and what comes out is a short list of
//! thread indices that is the whole reproducer.
//!
//! cargo run --example shrink

use riri::{replay, Explore, GlobalBuf, LaunchConfig, Ordering, ThreadCtx};

fn main() {
    structural();
    println!();
    order_dependent();
}

/// A missing barrier. Any schedule finds it.
fn structural() {
    const THREADS: u32 = 64;
    let config = LaunchConfig::new(1, THREADS);

    let found = Explore::new(&config).seeds(64).run_with(|| {
        let out = GlobalBuf::new("out", vec![0u32; THREADS as usize]);
        move |t: &ThreadCtx<'_>| {
            let tile = t.shared::<u32>("tile", THREADS as usize);
            let i = t.thread_linear();
            tile.write(t, i, i as u32);
            // BUG: no t.sync_threads() before reading a neighbour's slot.
            out.write(t, i, tile.read(t, (i + 1) % THREADS as usize));
        }
    });

    println!("== missing barrier");
    print!("{found}");

    if let Some(f) = &found.failure {
        if let Some(s) = f.shrink.schedule() {
            println!(
                "   {} decisions shrunk to {}: {:?}",
                f.decisions,
                s.len(),
                s.plan()
            );
            if s.is_empty() {
                println!("   an empty schedule means threads in order already hit it");
            }
        }
    }
}

/// A race that only happens when thread 0 is slow to publish its flag. The
/// atomics are how the flag is read without racing on the flag itself, so the
/// only finding is the one worth looking at.
fn order_dependent() {
    const THREADS: u32 = 4;
    let config = LaunchConfig::new(1, THREADS);

    let build = || {
        let flag = GlobalBuf::new("flag", vec![0u32; 1]);
        let out = GlobalBuf::new("out", vec![0u32; 1]);
        move |t: &ThreadCtx<'_>| {
            let i = t.thread_linear();
            if i == 0 {
                flag.atomic_add(t, 0, 1, Ordering::Relaxed);
            } else if flag.atomic_add(t, 0, 0, Ordering::Relaxed) == 0 {
                // Racy, but only for the threads that ran before thread 0.
                out.write(t, 0, i as u32);
            }
        }
    };

    let found = Explore::new(&config).seeds(64).run_with(build);

    println!("== flag published too late");
    print!("{found}");

    let Some(failure) = &found.failure else {
        println!("   no schedule reached the racy branch");
        return;
    };
    let Some(schedule) = failure.shrink.schedule() else {
        return;
    };

    println!(
        "   {} decisions shrunk to {}: {:?}",
        failure.decisions,
        schedule.len(),
        schedule.plan()
    );

    // That schedule is the entire reproducer. No seed involved.
    let report = replay(&config, schedule, build());
    println!("   replaying it alone:");
    print!("   {report}");
}
