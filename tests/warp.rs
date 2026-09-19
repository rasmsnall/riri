//! Warp-level collectives: the values they produce, and the convergence
//! mistakes Riri is meant to catch.
//!
//! Most tests use a small warp size so a whole warp fits in a few lanes and
//! the expected results can be written out by hand.

use riri::{launch, warp, Diagnostic, GlobalBuf, LaneProblem, LaunchConfig};

fn has<F: Fn(&Diagnostic) -> bool>(r: &riri::Report, f: F) -> bool {
    r.diagnostics.iter().any(f)
}

// ------------------------------------------------------------- results ---

#[test]
fn shuffle_down_reduction_is_clean() {
    let out = GlobalBuf::new("out", vec![0u32; 1]);
    let report = launch(&LaunchConfig::new(1, 8).warp_size(8).seed(1), |t| {
        let mask = t.warp_valid_mask();
        let mut v = t.lane_id() + 1;
        let mut delta = 4;
        while delta > 0 {
            v += warp::shfl_down_sync(t, mask, v, delta);
            delta /= 2;
        }
        if t.lane_id() == 0 {
            out.write(t, 0, v);
        }
    });

    report.assert_clean();
    assert_eq!(out.to_vec()[0], (1..=8).sum::<u32>());
}

#[test]
fn shuffle_past_the_end_keeps_own_value() {
    let out = GlobalBuf::new("out", vec![0u32; 8]);
    let report = launch(&LaunchConfig::new(1, 8).warp_size(8).seed(1), |t| {
        let v = warp::shfl_down_sync(t, t.warp_valid_mask(), t.lane_id(), 6);
        out.write(t, t.thread_linear(), v);
    });

    report.assert_clean();
    assert_eq!(out.to_vec(), vec![6, 7, 2, 3, 4, 5, 6, 7]);
}

#[test]
fn ballot_any_and_all() {
    let bits = GlobalBuf::new("bits", vec![0u32; 1]);
    let flags = GlobalBuf::new("flags", vec![0u32; 2]);
    let report = launch(&LaunchConfig::new(1, 8).warp_size(8).seed(2), |t| {
        let mask = t.warp_valid_mask();
        let lane = t.lane_id();
        let b = warp::ballot_sync(t, mask, lane % 2 == 0);
        let any = warp::any_sync(t, mask, lane == 7);
        let all = warp::all_sync(t, mask, lane < 8);
        if lane == 0 {
            bits.write(t, 0, b);
            flags.write(t, 0, any as u32);
            flags.write(t, 1, all as u32);
        }
    });

    report.assert_clean();
    assert_eq!(bits.to_vec()[0], 0b0101_0101);
    assert_eq!(flags.to_vec(), vec![1, 1]);
}

#[test]
fn partial_mask_is_fine_when_every_member_arrives() {
    let out = GlobalBuf::new("out", vec![9u32; 8]);
    let report = launch(&LaunchConfig::new(1, 8).warp_size(8).seed(4), |t| {
        let lane = t.lane_id();
        // Only the even lanes take part, and they all agree on that.
        if lane % 2 == 0 {
            let v = warp::shfl_sync(t, 0x55, lane, 2);
            out.write(t, t.thread_linear(), v);
        }
    });

    report.assert_clean();
    assert_eq!(out.to_vec(), vec![2, 9, 2, 9, 2, 9, 2, 9]);
}

#[test]
fn same_seed_same_schedule() {
    let run = || {
        let out = GlobalBuf::new("out", vec![0u32; 8]);
        let r = launch(&LaunchConfig::new(1, 8).warp_size(8).seed(99), |t| {
            let v = warp::shfl_xor_sync(t, t.warp_valid_mask(), t.lane_id(), 1);
            out.write(t, t.thread_linear(), v);
        });
        (format!("{r}"), out.to_vec())
    };
    let (first, values) = run();
    assert_eq!(run(), (first, values.clone()));
    assert_eq!(values, vec![1, 0, 3, 2, 5, 4, 7, 6]);
}

// --------------------------------------------------------- convergence ---

#[test]
fn shuffle_on_a_diverged_warp() {
    let out = GlobalBuf::new("out", vec![0u32; 8]);
    let report = launch(&LaunchConfig::new(1, 8).warp_size(8).seed(1), |t| {
        let lane = t.lane_id();
        if lane < 4 {
            // BUG: the mask names the whole warp, but only half of it is here.
            let v = warp::shfl_sync(t, t.warp_valid_mask(), lane, 0);
            out.write(t, lane as usize, v);
        }
    });

    assert!(report.has_warp_divergence(), "{report}");
    // Riri reports as soon as the first high lane exits, so how many low
    // lanes have parked by then depends on the schedule. What must hold is
    // that the mask named the whole warp and no high lane ever arrived.
    assert!(
        has(&report, |d| matches!(
            d,
            Diagnostic::WarpDivergence { mask: 0xFF, arrived, op: "shfl_sync", .. }
                if arrived & 0xF0 == 0 && arrived & 0x0F != 0
        )),
        "{report}"
    );
}

#[test]
fn lanes_disagreeing_about_the_mask() {
    let report = launch(&LaunchConfig::new(1, 8).warp_size(8).seed(1), |t| {
        let lane = t.lane_id();
        // BUG: the low lanes think the whole warp takes part, the high lanes
        // think only they do.
        let mask = if lane < 4 { 0xFF } else { 0xF0 };
        let _ = warp::shfl_sync(t, mask, lane, 4);
    });

    assert!(
        has(&report, |d| matches!(d, Diagnostic::WarpMaskMismatch { op: "shfl_sync", .. })),
        "{report}"
    );
}

#[test]
fn shuffle_reading_a_lane_outside_the_mask() {
    let report = launch(&LaunchConfig::new(1, 8).warp_size(8).seed(1), |t| {
        // Only the high lanes take part, but they read lane 0, which did not
        // contribute a value.
        if t.lane_id() >= 4 {
            let _ = warp::shfl_sync(t, 0xF0, t.lane_id(), 0);
        }
    });

    assert!(
        has(&report, |d| matches!(
            d,
            Diagnostic::WarpLaneError {
                problem: LaneProblem::SourceLaneNotInMask { src: 0 },
                ..
            }
        )),
        "{report}"
    );
}

#[test]
fn caller_left_itself_out_of_the_mask() {
    let report = launch(&LaunchConfig::new(1, 8).warp_size(8).seed(1), |t| {
        // Lanes 0-3 are not members of the mask they pass.
        let _ = warp::shfl_sync(t, 0xF0, t.lane_id(), 4);
    });

    assert!(report.aborted, "{report}");
    assert!(
        has(&report, |d| matches!(
            d,
            Diagnostic::WarpLaneError { problem: LaneProblem::CallerNotInMask, .. }
        )),
        "{report}"
    );
}

#[test]
fn mask_naming_lanes_the_block_does_not_have() {
    // A block of 6 with 4-lane warps: warp 1 holds only lanes 0 and 1, so a
    // full mask names threads that do not exist.
    let report = launch(&LaunchConfig::new(1, 6).warp_size(4).seed(1), |t| {
        let _ = warp::shfl_sync(t, warp::FULL_MASK, t.lane_id(), 0);
    });

    assert!(report.aborted, "{report}");
    assert!(
        has(&report, |d| matches!(
            d,
            Diagnostic::WarpLaneError { problem: LaneProblem::MaskOutsideBlock { .. }, .. }
        )),
        "{report}"
    );
}

#[test]
fn lane_exits_while_its_warp_mates_wait() {
    let report = launch(&LaunchConfig::new(1, 8).warp_size(8).seed(6), |t| {
        // Lane 7 returns early, so the collective can never complete.
        if t.lane_id() == 7 {
            return;
        }
        let _ = warp::shfl_sync(t, t.warp_valid_mask(), t.lane_id(), 0);
    });

    assert!(report.has_warp_divergence(), "{report}");
}

#[test]
fn some_lanes_at_a_barrier_others_at_a_shuffle() {
    // The block barrier waits for lanes that are stuck in a collective and
    // vice versa. Riri blames the warp, not the barrier.
    let report = launch(&LaunchConfig::new(1, 8).warp_size(8).seed(1), |t| {
        if t.lane_id() < 4 {
            let _ = warp::shfl_sync(t, t.warp_valid_mask(), t.lane_id(), 0);
        } else {
            t.sync_threads();
        }
    });

    assert!(report.has_warp_divergence(), "{report}");
}

#[test]
fn divergence_is_caught_on_every_schedule() {
    // Convergence is checked structurally, not by hoping for an unlucky
    // interleaving, so every seed must find it — and none may hang.
    for seed in 0..32 {
        let report = launch(&LaunchConfig::new(1, 8).warp_size(8).seed(seed), |t| {
            if t.lane_id() % 2 == 0 {
                let _ = warp::shfl_sync(t, t.warp_valid_mask(), t.lane_id(), 0);
            }
        });
        assert!(report.has_warp_divergence(), "seed {seed}: {report}");
    }
}

// ------------------------------------------------------ happens-before ---

#[test]
fn full_mask_collective_orders_shared_memory_in_that_warp() {
    let out = GlobalBuf::new("out", vec![0u32; 8]);
    let report = launch(&LaunchConfig::new(1, 8).warp_size(8).seed(5), |t| {
        let tile = t.shared::<u32>("tile", 8);
        let i = t.thread_linear();
        tile.write(t, i, i as u32);
        warp::sync_warp(t, t.warp_valid_mask());
        out.write(t, i, tile.read(t, (i + 1) % 8));
    });

    report.assert_clean();
    assert_eq!(out.to_vec(), vec![1, 2, 3, 4, 5, 6, 7, 0]);
}

#[test]
fn syncing_one_warp_does_not_order_another() {
    // Two warps of four share a tile. `sync_warp` says nothing about the
    // other warp, so reading across the boundary is still a race.
    let out = GlobalBuf::new("out", vec![0u32; 8]);
    let report = launch(&LaunchConfig::new(1, 8).warp_size(4).seed(5), |t| {
        let tile = t.shared::<u32>("tile", 8);
        let i = t.thread_linear();
        tile.write(t, i, i as u32);
        warp::sync_warp(t, t.warp_valid_mask());
        out.write(t, i, tile.read(t, (i + 4) % 8));
    });

    assert!(report.has_race(), "{report}");
}

#[test]
fn partial_mask_collective_does_not_order_memory() {
    // Only half the warp syncs, so it cannot stand in for a barrier even
    // among the lanes that took part.
    let out = GlobalBuf::new("out", vec![0u32; 8]);
    let report = launch(&LaunchConfig::new(1, 8).warp_size(8).seed(7), |t| {
        let tile = t.shared::<u32>("tile", 8);
        let i = t.thread_linear();
        tile.write(t, i, i as u32);
        if t.lane_id() < 4 {
            warp::sync_warp(t, 0x0F);
            out.write(t, i, tile.read(t, (i + 1) % 4));
        }
    });

    assert!(report.has_race(), "{report}");
}
