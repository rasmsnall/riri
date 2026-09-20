use riri::{launch, Diagnostic, GlobalBuf, LaunchConfig, MemSpace, Ordering};

fn count<F: Fn(&Diagnostic) -> bool>(r: &riri::Report, f: F) -> usize {
    r.diagnostics.iter().filter(|d| f(d)).count()
}

#[test]
fn clean_vecadd() {
    let n = 128;
    let a = GlobalBuf::new("a", (0..n).map(|i| i as f32).collect());
    let b = GlobalBuf::new("b", vec![1.0f32; n]);
    let c = GlobalBuf::new("c", vec![0.0f32; n]);

    let report = launch(&LaunchConfig::new(4, 32).seed(1), |t| {
        let i = t.global_linear();
        c.write(t, i, a.read(t, i) + b.read(t, i));
    });

    report.assert_clean();
    assert_eq!(
        c.to_vec(),
        (0..n).map(|i| i as f32 + 1.0).collect::<Vec<_>>()
    );
}

#[test]
fn every_thread_writes_same_slot() {
    let out = GlobalBuf::new("out", vec![0u32; 1]);
    let report = launch(&LaunchConfig::new(1, 32), |t| {
        out.write(t, 0, t.thread_linear() as u32)
    });
    assert!(report.has_race(), "{report}");
}

#[test]
fn cross_block_race_even_after_barriers() {
    // Barriers order threads within a block, never across blocks.
    let out = GlobalBuf::new("out", vec![0u32; 4]);
    let report = launch(&LaunchConfig::new(2, 4), |t| {
        t.sync_threads();
        out.write(t, t.thread_linear(), 1);
    });
    assert!(report.has_race(), "{report}");
}

#[test]
fn shared_memory_missing_barrier() {
    let out = GlobalBuf::new("out", vec![0u32; 32]);
    let report = launch(&LaunchConfig::new(1, 32).seed(3), |t| {
        let tile = t.shared::<u32>("tile", 32);
        let i = t.thread_linear();
        tile.write(t, i, i as u32);
        // BUG: no t.sync_threads() here.
        out.write(t, i, tile.read(t, (i + 1) % 32));
    });
    assert!(report.has_race(), "{report}");
    assert!(report.diagnostics.iter().any(|d| matches!(
        d,
        Diagnostic::DataRace {
            space: MemSpace::Shared { name: "tile", .. },
            ..
        }
    )));
}

#[test]
fn shared_memory_with_barrier_is_clean() {
    let out = GlobalBuf::new("out", vec![0u32; 32]);
    for seed in 0..8 {
        let report = launch(&LaunchConfig::new(1, 32).seed(seed), |t| {
            let tile = t.shared::<u32>("tile", 32);
            let i = t.thread_linear();
            tile.write(t, i, i as u32);
            t.sync_threads();
            out.write(t, i, tile.read(t, (i + 1) % 32));
        });
        report.assert_clean();
    }
    assert_eq!(out.to_vec()[31], 0);
    assert_eq!(out.to_vec()[0], 1);
}

#[test]
fn uninitialised_shared_read() {
    let report = launch(&LaunchConfig::new(1, 8), |t| {
        let tile = t.shared::<u32>("tile", 16);
        // Only the first 8 slots are ever written.
        tile.write(t, t.thread_linear(), 7);
        t.sync_threads();
        let _ = tile.read(t, 8 + t.thread_linear());
    });
    assert!(
        count(&report, |d| matches!(d, Diagnostic::UninitRead { .. })) > 0,
        "{report}"
    );
}

#[test]
fn barrier_in_divergent_branch() {
    let report = launch(&LaunchConfig::new(1, 32), |t| {
        if t.thread_linear() < 16 {
            t.sync_threads();
        }
    });
    assert!(report.aborted);
    assert!(
        report
            .diagnostics
            .iter()
            .any(|d| matches!(d, Diagnostic::BarrierDivergence { .. })),
        "{report}"
    );
}

#[test]
fn atomics_do_not_race() {
    let sum = GlobalBuf::new("sum", vec![0u32; 1]);
    let report = launch(&LaunchConfig::new(4, 32), |t| {
        sum.atomic_add(t, 0, 1, Ordering::Relaxed);
    });
    report.assert_clean();
    assert_eq!(sum.to_vec()[0], 128);
}

#[test]
fn plain_read_racing_atomics() {
    let sum = GlobalBuf::new("sum", vec![0u32; 1]);
    let seen = GlobalBuf::new("seen", vec![0u32; 32]);
    let report = launch(&LaunchConfig::new(1, 32), |t| {
        sum.atomic_add(t, 0, 1, Ordering::Relaxed);
        seen.write(t, t.thread_linear(), sum.read(t, 0));
    });
    assert!(report.has_race(), "{report}");
}

#[test]
fn out_of_bounds_traps() {
    let buf = GlobalBuf::new("buf", vec![0u8; 30]);
    let report = launch(&LaunchConfig::new(1, 32), |t| {
        buf.write(t, t.thread_linear(), 1)
    });
    assert!(report.aborted);
    assert!(
        report
            .diagnostics
            .iter()
            .any(|d| matches!(d, Diagnostic::OutOfBounds { len: 30, .. })),
        "{report}"
    );
}

#[test]
fn kernel_panic_is_reported() {
    let report = launch(&LaunchConfig::new(1, 4), |t| {
        assert!(t.thread_linear() != 2, "thread two is cursed");
    });
    assert!(report.aborted);
    assert!(report.diagnostics.iter().any(
        |d| matches!(d, Diagnostic::KernelPanic { message, .. } if message.contains("cursed"))
    ));
}

#[test]
fn same_seed_same_schedule() {
    // A deliberately racy kernel whose output depends on the interleaving.
    let run = |seed| {
        let out = GlobalBuf::new("out", vec![0u32; 1]);
        let report = launch(&LaunchConfig::new(2, 16).seed(seed), |t| {
            let v = out.read(t, 0);
            out.write(t, 0, v * 31 + t.global_linear() as u32);
        });
        (out.to_vec()[0], format!("{report}"))
    };
    assert_eq!(run(42), run(42));
    // And different seeds should explore different interleavings.
    assert!(
        (0..16)
            .map(run)
            .map(|r| r.0)
            .collect::<std::collections::HashSet<_>>()
            .len()
            > 1
    );
}

#[test]
fn diagnostics_are_deduplicated() {
    let out = GlobalBuf::new("out", vec![0u32; 1]);
    let report = launch(&LaunchConfig::new(4, 64), |t| out.write(t, 0, 1));
    assert_eq!(report.diagnostics.len(), 1, "{report}");
}
