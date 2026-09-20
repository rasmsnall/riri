//! Running a grid in waves, the way hardware schedules blocks onto its
//! multiprocessors.
//!
//! Riri runs every block at once by default. That is the permissive reading
//! of CUDA, which does not promise two blocks are resident together unless
//! the launch was cooperative. `resident_blocks` models the hardware instead,
//! which both bounds the cost of a launch and exposes kernels that assume
//! co-residency.

use riri::{explore, launch, GlobalBuf, LaunchConfig, Ordering, Shrink};

#[test]
fn races_between_waves_are_still_found() {
    // The load-bearing claim. Detection is happens-before based, so accesses
    // from different blocks are unordered whether or not they overlapped in
    // time. Running one block at a time must not hide anything.
    let out = GlobalBuf::new("out", vec![0u32; 1]);

    let report = launch(
        &LaunchConfig::new(4u32, 1u32).resident_blocks(1).seed(1),
        |t| {
            out.write(t, 0, t.block_linear() as u32);
        },
    );

    assert!(report.has_race(), "{report}");
}

#[test]
fn every_block_still_runs_and_indices_stay_right() {
    let out = GlobalBuf::new("out", vec![0u32; 24]);

    let report = launch(
        &LaunchConfig::new(6u32, 4u32).resident_blocks(2).seed(1),
        |t| {
            // Pack the identity this thread believes it has.
            let packed = t.block_linear() as u32 * 100 + t.thread_linear() as u32;
            out.write(t, t.global_linear(), packed);
        },
    );

    report.assert_clean();
    let expected: Vec<u32> = (0..6)
        .flat_map(|b| (0..4).map(move |i| b * 100 + i))
        .collect();
    assert_eq!(out.to_vec(), expected);
}

#[test]
fn shared_memory_stays_per_block_across_waves() {
    let out = GlobalBuf::new("out", vec![0u32; 16]);

    let report = launch(
        &LaunchConfig::new(4u32, 4u32).resident_blocks(1).seed(2),
        |t| {
            let tile = t.shared::<u32>("tile", 4);
            let i = t.thread_linear();
            tile.write(t, i, t.block_linear() as u32);
            t.sync_threads();
            out.write(t, t.global_linear(), tile.read(t, (i + 1) % 4));
        },
    );

    report.assert_clean();
    // Every thread reads a neighbour's slot, which holds its own block index.
    let expected: Vec<u32> = (0..4).flat_map(|b| std::iter::repeat(b).take(4)).collect();
    assert_eq!(out.to_vec(), expected);
}

#[test]
fn a_block_cannot_see_one_in_a_later_wave() {
    // The producer is block 1 and the consumer block 0. With both resident
    // the handoff works. With one block at a time it cannot, because the
    // producer has not run yet, and that is exactly the assumption CUDA says
    // a kernel may not make.
    let run = |resident: u32| {
        let data = GlobalBuf::new("data", vec![0u32; 1]);
        let flag = GlobalBuf::new("flag", vec![0u32; 1]);
        let seen = GlobalBuf::new("seen", vec![0u32; 1]);

        let config = LaunchConfig::new(2u32, 1u32)
            .resident_blocks(resident)
            .seed(1);
        let report = launch(&config, |t| {
            if t.block_linear() == 1 {
                data.write(t, 0, 42);
                t.threadfence();
                flag.atomic_store(t, 0, 1, Ordering::Relaxed);
            } else {
                let mut arrived = false;
                for _ in 0..200 {
                    if flag.atomic_load(t, 0, Ordering::Relaxed) == 1 {
                        arrived = true;
                        break;
                    }
                }
                if arrived {
                    t.threadfence();
                    seen.write(t, 0, data.read(t, 0));
                }
            }
        });
        (report, seen.to_vec()[0])
    };

    let (both, seen_both) = run(2);
    both.assert_clean();
    assert_eq!(seen_both, 42, "co-resident blocks should hand over");

    let (waved, seen_waved) = run(1);
    waved.assert_clean();
    assert_eq!(seen_waved, 0, "a later wave cannot have published yet");
}

#[test]
fn a_multi_wave_launch_cannot_be_shrunk() {
    // Each wave numbers its threads from zero, so a recorded plan would
    // replay into the wrong threads. Riri declines rather than mislead.
    let out = GlobalBuf::new("out", vec![0u32; 1]);

    let found = explore(&LaunchConfig::new(4u32, 2u32).resident_blocks(2), 4, |t| {
        out.write(t, 0, t.global_linear() as u32);
    });

    let failure = found.failure.expect("expected a race");
    assert_eq!(failure.shrink, Shrink::TraceTruncated);
}

#[test]
fn a_single_wave_is_unchanged() {
    // Setting the residency to the whole grid must behave exactly as not
    // setting it, including staying shrinkable.
    let out = GlobalBuf::new("out", vec![0u32; 1]);

    let found = explore(&LaunchConfig::new(4u32, 2u32).resident_blocks(4), 4, |t| {
        out.write(t, 0, t.global_linear() as u32);
    });

    let failure = found.failure.expect("expected a race");
    assert!(
        matches!(failure.shrink, Shrink::Minimised(_)),
        "{}",
        failure.shrink
    );
}

#[test]
fn a_launch_larger_than_the_resident_cap_runs() {
    // 17,408 threads, past MAX_THREADS, which used to be a hard ceiling on
    // the whole launch and now bounds only what is resident at once.
    const BLOCKS: u32 = 17;
    const THREADS: u32 = 1024;
    let n = (BLOCKS * THREADS) as usize;
    let out = GlobalBuf::new("out", vec![0u32; n]);

    let report = launch(
        &LaunchConfig::new(BLOCKS, THREADS)
            .resident_blocks(4)
            .seed(1),
        |t| out.write(t, t.global_linear(), t.global_linear() as u32),
    );

    report.assert_clean();
    assert_eq!(out.to_vec(), (0..n as u32).collect::<Vec<_>>());
}
