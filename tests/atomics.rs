//! The atomic operations beyond `atomic_add`, and the cuda-oxide shaped
//! surface over them.

use riri::oxide::{self, AtomicOrdering, DeviceAtomicSlice};
use riri::{launch, Diagnostic, GlobalBuf, LaunchConfig, MemSpace, Ordering};

#[test]
fn swap_hands_back_the_previous_value() {
    let cell = GlobalBuf::new("cell", vec![0u32; 1]);
    let seen = GlobalBuf::new("seen", vec![0u32; 4]);

    let report = launch(&LaunchConfig::new(1, 4).seed(1), |t| {
        let me = t.thread_linear() as u32 + 1;
        let previous = cell.atomic_swap(t, 0, me, Ordering::Relaxed);
        seen.write(t, t.thread_linear(), previous);
    });

    report.assert_clean();
    // Every thread saw a different predecessor, and the last one is left.
    let mut all: Vec<u32> = seen.to_vec();
    all.push(cell.to_vec()[0]);
    all.sort_unstable();
    assert_eq!(all, vec![0, 1, 2, 3, 4]);
}

#[test]
fn compare_exchange_lets_exactly_one_thread_win() {
    let lock = GlobalBuf::new("lock", vec![0u32; 1]);
    let winners = GlobalBuf::new("winners", vec![0u32; 1]);

    let report = launch(&LaunchConfig::new(2, 8).seed(1), |t| {
        if lock
            .atomic_compare_exchange(t, 0, 0, 1, Ordering::AcqRel)
            .is_ok()
        {
            winners.atomic_add(t, 0, 1, Ordering::Relaxed);
        }
    });

    report.assert_clean();
    assert_eq!(winners.to_vec()[0], 1, "more than one thread took the lock");
}

#[test]
fn min_and_max_reduce() {
    let low = GlobalBuf::new("low", vec![u32::MAX; 1]);
    let high = GlobalBuf::new("high", vec![0u32; 1]);

    let report = launch(&LaunchConfig::new(2, 8).seed(1), |t| {
        let v = t.global_linear() as u32;
        low.atomic_min(t, 0, v, Ordering::Relaxed);
        high.atomic_max(t, 0, v, Ordering::Relaxed);
    });

    report.assert_clean();
    assert_eq!(low.to_vec()[0], 0);
    assert_eq!(high.to_vec()[0], 15);
}

#[test]
fn sub_counts_down() {
    let remaining = GlobalBuf::new("remaining", vec![16u32; 1]);

    let report = launch(&LaunchConfig::new(2, 8).seed(1), |t| {
        remaining.atomic_sub(t, 0, 1, Ordering::Relaxed);
    });

    report.assert_clean();
    assert_eq!(remaining.to_vec()[0], 0);
}

#[test]
fn a_plain_access_still_races_with_an_atomic() {
    // Widening the atomic surface must not have widened what counts as safe.
    let cell = GlobalBuf::new("cell", vec![0u32; 1]);

    let report = launch(&LaunchConfig::new(2, 1).seed(1), |t| {
        if t.block_linear() == 0 {
            cell.atomic_max(t, 0, 5, Ordering::Relaxed);
        } else {
            cell.write(t, 0, 9);
        }
    });

    assert!(report.has_race(), "{report}");
}

// ------------------------------------------------ the cuda-oxide shape ---

#[test]
fn oxide_atomics_read_as_they_do_on_the_gpu() {
    // `counters[i].fetch_add(1, AtomicOrdering::Relaxed)` is the line a
    // cuda-oxide kernel writes, and it needs no changing to run here.
    let buf = GlobalBuf::new("counters", vec![0u32; 4]);
    let counters = DeviceAtomicSlice::new(&buf);

    let report = oxide::launch(&LaunchConfig::new(2, 8).seed(1), || {
        let slot = oxide::thread::threadIdx_x() as usize % 4;
        counters[slot].fetch_add(1, AtomicOrdering::Relaxed);
    });

    report.assert_clean();
    assert_eq!(buf.to_vec(), vec![4u32; 4]);
}

#[test]
fn oxide_fences_order_across_blocks() {
    let data = GlobalBuf::new("data", vec![0u32; 1]);
    let flagbuf = GlobalBuf::new("flag", vec![0u32; 1]);
    let flag = DeviceAtomicSlice::new(&flagbuf);
    let seen = GlobalBuf::new("seen", vec![0u32; 1]);

    let report = oxide::launch(&LaunchConfig::new(2, 1).seed(1), || {
        if oxide::thread::blockIdx_x() == 0 {
            let mut d = oxide::DisjointSlice::new(&data);
            *d.get_unchecked_mut(0) = 42;
            oxide::threadfence();
            flag[0].store(1, AtomicOrdering::Relaxed);
        } else {
            let mut arrived = false;
            for _ in 0..200 {
                if flag[0].load(AtomicOrdering::Relaxed) == 1 {
                    arrived = true;
                    break;
                }
            }
            if arrived {
                oxide::threadfence();
                let mut s = oxide::DisjointSlice::new(&seen);
                *s.get_unchecked_mut(0) = 1;
            }
        }
    });

    report.assert_clean();
    assert_eq!(seen.to_vec()[0], 1, "the consumer never ran: {report}");
}

#[test]
fn an_oxide_atomic_on_uninitialised_memory_names_the_kernel_line() {
    // The location must be the kernel's, not a line inside the shim.
    let buf = GlobalBuf::uninit("total", 1);
    let totals = DeviceAtomicSlice::new(&buf);

    let report = oxide::launch(&LaunchConfig::new(1, 4).seed(1), || {
        totals[0].fetch_add(1, AtomicOrdering::Relaxed);
    });

    let found = report.diagnostics.iter().find_map(|d| match d {
        Diagnostic::UninitRead {
            space: MemSpace::Global { .. },
            access,
            ..
        } => Some(access.location.file()),
        _ => None,
    });

    let file = found.unwrap_or_else(|| panic!("expected an uninit read: {report}"));
    assert!(
        file.contains("atomics.rs"),
        "reported inside Riri rather than the kernel: {file}"
    );
}
