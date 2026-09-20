//! Cross-block communication: the pattern Riri used to report as a race
//! whether or not it was one.
//!
//! Barriers order a block and collectives order a warp, so before fences
//! existed every pair of accesses from different blocks counted as
//! concurrent. That is right for kernels that do not synchronise and wrong
//! for the ones that do, which made correct message passing unreportable as
//! correct. These tests pin both directions.

use riri::{launch, GlobalBuf, LaunchConfig, Ordering, Report, ThreadCtx};

/// Producer in block 0, consumer in block 1.
///
/// `publish` and `consume` decide how each side orders itself, which is the
/// only difference between a correct kernel and a racy one here.
fn message_passing(
    seed: u64,
    publish: impl Fn(&ThreadCtx<'_>) + Sync,
    consume: impl Fn(&ThreadCtx<'_>) + Sync,
    store: Ordering,
    load: Ordering,
) -> (Report, u32) {
    let data = GlobalBuf::new("data", vec![0u32; 1]);
    let flag = GlobalBuf::new("flag", vec![0u32; 1]);
    let seen = GlobalBuf::new("seen", vec![0u32; 1]);

    let report = launch(&LaunchConfig::new(2, 1).seed(seed), |t| {
        if t.block_linear() == 0 {
            data.write(t, 0, 42);
            publish(t);
            flag.atomic_store(t, 0, 1, store);
        } else {
            // Spin, bounded so a schedule that never runs the producer ends
            // rather than hangs.
            let mut arrived = false;
            for _ in 0..200 {
                if flag.atomic_load(t, 0, load) == 1 {
                    arrived = true;
                    break;
                }
            }
            if arrived {
                consume(t);
                let value = data.read(t, 0);
                seen.write(t, 0, value);
            }
        }
    });

    let value = seen.to_vec()[0];
    (report, value)
}

#[test]
fn threadfence_orders_across_blocks() {
    let (report, seen) = message_passing(
        1,
        |t| t.threadfence(),
        |t| t.threadfence(),
        Ordering::Relaxed,
        Ordering::Relaxed,
    );

    assert_eq!(seen, 42, "the consumer never read the data: {report}");
    report.assert_clean();
}

#[test]
fn without_fences_the_same_kernel_races() {
    let (report, seen) = message_passing(1, |_| {}, |_| {}, Ordering::Relaxed, Ordering::Relaxed);

    assert_eq!(seen, 42, "the consumer never read the data: {report}");
    assert!(report.has_race(), "{report}");
}

#[test]
fn release_and_acquire_order_across_blocks() {
    // The same handoff without explicit fences, carried by the atomics.
    let (report, seen) = message_passing(1, |_| {}, |_| {}, Ordering::Release, Ordering::Acquire);

    assert_eq!(seen, 42, "the consumer never read the data: {report}");
    report.assert_clean();
}

#[test]
fn a_release_without_a_matching_acquire_is_not_enough() {
    // The producer publishes, but the consumer never takes it on.
    let (report, seen) = message_passing(1, |_| {}, |_| {}, Ordering::Release, Ordering::Relaxed);

    assert_eq!(seen, 42, "the consumer never read the data: {report}");
    assert!(report.has_race(), "{report}");
}

#[test]
fn an_acquire_without_a_matching_release_is_not_enough() {
    let (report, seen) = message_passing(1, |_| {}, |_| {}, Ordering::Relaxed, Ordering::Acquire);

    assert_eq!(seen, 42, "the consumer never read the data: {report}");
    assert!(report.has_race(), "{report}");
}

#[test]
fn ordering_holds_on_every_schedule() {
    // Synchronisation is structural, like everything else Riri decides, so
    // the answer must not depend on the interleaving.
    for seed in 0..16 {
        let (report, seen) = message_passing(
            seed,
            |t| t.threadfence(),
            |t| t.threadfence(),
            Ordering::Relaxed,
            Ordering::Relaxed,
        );
        assert_eq!(seen, 42, "seed {seed}: consumer never read: {report}");
        assert!(report.is_clean(), "seed {seed}: {report}");
    }
}

#[test]
fn fencing_does_not_order_an_unrelated_block() {
    // Blocks 0 and 1 hand off correctly. Block 2 writes the same element
    // without taking part, and must still be reported.
    let data = GlobalBuf::new("data", vec![0u32; 1]);
    let flag = GlobalBuf::new("flag", vec![0u32; 1]);

    let report = launch(&LaunchConfig::new(3, 1).seed(2), |t| {
        match t.block_linear() {
            0 => {
                data.write(t, 0, 42);
                t.threadfence();
                flag.atomic_store(t, 0, 1, Ordering::Relaxed);
            }
            1 => {
                for _ in 0..200 {
                    if flag.atomic_load(t, 0, Ordering::Relaxed) == 1 {
                        break;
                    }
                }
                t.threadfence();
                let _ = data.read(t, 0);
            }
            _ => {
                data.write(t, 0, 99);
            }
        }
    });

    assert!(report.has_race(), "{report}");
}

#[test]
fn a_thread_that_never_fences_is_unaffected() {
    // The plain racing kernel, to show the clock machinery stays dormant.
    let out = GlobalBuf::new("out", vec![0u32; 1]);
    let report = launch(&LaunchConfig::new(2, 4).seed(1), |t| {
        out.write(t, 0, t.global_linear() as u32);
    });

    assert!(report.has_race(), "{report}");
}
