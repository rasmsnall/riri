//! Searching across schedules, and shrinking a failing one down to something
//! a person can read.

use riri::{explore, replay, Explore, GlobalBuf, LaunchConfig, Schedule, Shrink, ThreadCtx};

/// Every thread writes the same cell, so any schedule races.
fn racy(t: &ThreadCtx<'_>, out: &GlobalBuf<u32>) {
    out.write(t, 0, t.thread_linear() as u32);
}

#[test]
fn a_clean_kernel_stays_clean_across_seeds() {
    let out = GlobalBuf::new("out", vec![0u32; 32]);
    let found = explore(&LaunchConfig::new(1, 32), 64, |t| {
        out.write(t, t.thread_linear(), 1);
    });

    found.assert_clean();
    assert_eq!(found.seeds_tried, 64);
}

#[test]
fn a_racy_kernel_is_found_and_shrunk() {
    let out = GlobalBuf::new("out", vec![0u32; 1]);
    let found = explore(&LaunchConfig::new(1, 8), 64, |t| racy(t, &out));

    let failure = found.failure.expect("expected a failure");
    assert!(failure.report.has_race(), "{}", failure.report);

    let schedule = match &failure.shrink {
        Shrink::Minimised(s) => s,
        other => panic!("expected a shrunk schedule, got {other}"),
    };

    // Two threads writing the same cell is enough, so the shrunk schedule
    // should be tiny rather than the whole run.
    assert!(schedule.len() <= 8, "schedule was {schedule}");
    assert!(schedule.switches() <= 2, "schedule was {schedule}");
}

#[test]
fn the_shrunk_schedule_replays_the_same_finding() {
    let out = GlobalBuf::new("out", vec![0u32; 1]);
    let config = LaunchConfig::new(1, 8);
    let found = explore(&config, 64, |t| racy(t, &out));

    let failure = found.failure.expect("expected a failure");
    let target = failure.target().expect("expected a target fingerprint");
    let schedule = failure.shrink.schedule().expect("expected a shrunk schedule");

    let mut replayed = LaunchConfig::new(1, 8);
    replayed.seed = failure.seed;
    let report = replay(&replayed, schedule, |t| racy(t, &out));

    assert!(report.contains(&target), "replay lost the finding: {report}");
}

#[test]
fn replaying_a_schedule_is_deterministic() {
    let out = GlobalBuf::new("out", vec![0u32; 1]);
    let config = LaunchConfig::new(1, 4);
    let schedule = Schedule::new(vec![3, 2, 1, 0]);

    let first = format!("{}", replay(&config, &schedule, |t| racy(t, &out)));
    let second = format!("{}", replay(&config, &schedule, |t| racy(t, &out)));

    assert_eq!(first, second);
}

#[test]
fn a_spent_schedule_keeps_the_current_thread_running() {
    // An empty plan means every decision falls to the default, which is to
    // let the running thread continue. That is still a valid schedule, and
    // it still has to find a race on a cell every thread writes.
    let out = GlobalBuf::new("out", vec![0u32; 1]);
    let report = replay(&LaunchConfig::new(1, 4), &Schedule::new(Vec::new()), |t| {
        racy(t, &out)
    });

    assert!(report.has_race(), "{report}");
}

#[test]
fn shrinking_can_be_turned_off() {
    let out = GlobalBuf::new("out", vec![0u32; 1]);
    let found = Explore::new(&LaunchConfig::new(1, 8))
        .seeds(4)
        .minimise(false)
        .run(|t| racy(t, &out));

    let failure = found.failure.expect("expected a failure");
    assert_eq!(failure.shrink, Shrink::Disabled);
}

#[test]
fn a_kernel_that_carries_state_is_reported_as_unreproducible() {
    // This kernel behaves differently every run, because the counter it
    // reads is the one the previous run left behind. Shrinking cannot reason
    // about such a kernel, and says so rather than producing nonsense.
    let runs = GlobalBuf::new("runs", vec![0u32; 1]);
    let out = GlobalBuf::new("out", vec![0u32; 1]);

    let found = Explore::new(&LaunchConfig::new(1, 4)).seeds(4).run(|t| {
        let seen = runs.atomic_add(t, 0, 1);
        if seen < 4 {
            out.write(t, 0, 1);
        }
    });

    let failure = found.failure.expect("expected a failure");
    assert_eq!(failure.shrink, Shrink::NotReproducible);
}

#[test]
fn run_with_gives_each_run_its_own_state() {
    // The same kernel as above, but with fresh buffers per run, so every run
    // behaves identically and shrinking works.
    let found = Explore::new(&LaunchConfig::new(1, 4)).seeds(4).run_with(|| {
        let runs = GlobalBuf::new("runs", vec![0u32; 1]);
        let out = GlobalBuf::new("out", vec![0u32; 1]);
        move |t: &ThreadCtx<'_>| {
            let seen = runs.atomic_add(t, 0, 1);
            if seen < 4 {
                out.write(t, 0, 1);
            }
        }
    });

    let failure = found.failure.expect("expected a failure");
    assert!(
        matches!(failure.shrink, Shrink::Minimised(_)),
        "expected a shrunk schedule, got {}",
        failure.shrink
    );
}

#[test]
fn shrinking_a_long_run_cuts_it_down() {
    // A shared-memory handoff with no barrier. The full run makes hundreds of
    // decisions; the bug needs a handful.
    let out = GlobalBuf::new("out", vec![0u32; 64]);
    let config = LaunchConfig::new(1, 64);
    let found = Explore::new(&config).seeds(8).run(|t| {
        let tile = t.shared::<u32>("tile", 64);
        let i = t.thread_linear();
        tile.write(t, i, i as u32);
        out.write(t, i, tile.read(t, (i + 1) % 64));
    });

    let failure = found.failure.expect("expected a failure");
    let schedule = failure.shrink.schedule().expect("expected a shrunk schedule");

    // Three buffer operations per thread across 64 threads is on the order of
    // 200 decisions. The shrunk schedule should be a small fraction of that.
    assert!(schedule.len() < 64, "schedule was {schedule}");
}

#[test]
fn exploration_is_deterministic() {
    let run = || {
        let out = GlobalBuf::new("out", vec![0u32; 1]);
        let found = explore(&LaunchConfig::new(1, 8), 16, |t| racy(t, &out));
        let failure = found.failure.expect("expected a failure");
        (failure.seed, failure.shrink.schedule().cloned())
    };

    assert_eq!(run(), run());
}
