//! Reads of device memory nothing has written.
//!
//! This is the check Compute Sanitizer calls `initcheck`. Riri already
//! reported it for shared memory, which starts uninitialised by construction;
//! a global buffer needed a way to start that way too, since the usual bug is
//! a kernel that fills only part of its output.

use riri::{launch, Diagnostic, GlobalBuf, LaunchConfig, MemSpace, Ordering};

#[test]
fn reading_global_memory_nobody_wrote_is_reported() {
    let out = GlobalBuf::uninit("out", 32);
    let copy = GlobalBuf::new("copy", vec![0u32; 32]);

    let report = launch(&LaunchConfig::new(1, 32).seed(1), |t| {
        let i = t.thread_linear();
        copy.write(t, i, out.read(t, i));
    });

    assert!(
        report.diagnostics.iter().any(|d| matches!(
            d,
            Diagnostic::UninitRead {
                space: MemSpace::Global { .. },
                ..
            }
        )),
        "{report}"
    );
}

#[test]
fn writing_before_reading_is_clean() {
    let out = GlobalBuf::uninit("out", 32);

    let report = launch(&LaunchConfig::new(1, 32).seed(1), |t| {
        let i = t.thread_linear();
        out.write(t, i, i as u32);
        let seen = out.read(t, i);
        out.write(t, i, seen + 1);
    });

    report.assert_clean();
    assert_eq!(out.to_vec()[5], 6);
}

#[test]
fn a_kernel_that_fills_only_half_its_output_is_caught() {
    // The bug initcheck exists for: the tail is never written, and something
    // downstream reads it.
    let out = GlobalBuf::uninit("out", 32);
    let sink = GlobalBuf::new("sink", vec![0u32; 32]);

    let report = launch(&LaunchConfig::new(1, 32).seed(1), |t| {
        let i = t.thread_linear();
        if i < 16 {
            out.write(t, i, i as u32);
        }
        t.sync_threads();
        sink.write(t, i, out.read(t, i));
    });

    assert!(
        report.diagnostics.iter().any(|d| matches!(
            d,
            Diagnostic::UninitRead {
                space: MemSpace::Global { .. },
                ..
            }
        )),
        "{report}"
    );
}

#[test]
fn an_atomic_on_uninitialised_memory_is_caught() {
    // An accumulator nobody zeroed. The read half of the read-modify-write
    // is still a read of memory that holds nothing.
    let total = GlobalBuf::uninit("total", 1);

    let report = launch(&LaunchConfig::new(1, 8).seed(1), |t| {
        total.atomic_add(t, 0, 1, Ordering::Relaxed);
    });

    assert!(
        report.diagnostics.iter().any(|d| matches!(
            d,
            Diagnostic::UninitRead {
                space: MemSpace::Global { .. },
                ..
            }
        )),
        "{report}"
    );
}

#[test]
fn initialisation_carries_across_launches() {
    // Device memory keeps its contents between launches, so a buffer one
    // kernel filled is not uninitialised for the next.
    let out = GlobalBuf::uninit("out", 8);
    let config = LaunchConfig::new(1, 8).seed(1);

    let first = launch(&config, |t| {
        out.write(t, t.thread_linear(), 7);
    });
    first.assert_clean();

    let second = launch(&config, |t| {
        let i = t.thread_linear();
        let seen = out.read(t, i);
        out.write(t, i, seen);
    });

    second.assert_clean();
    assert_eq!(out.to_vec(), vec![7u32; 8]);
}

#[test]
fn an_initialised_buffer_is_unaffected() {
    let out = GlobalBuf::new("out", vec![0u32; 8]);
    let report = launch(&LaunchConfig::new(1, 8).seed(1), |t| {
        let i = t.thread_linear();
        let seen = out.read(t, i);
        out.write(t, i, seen + 1);
    });

    report.assert_clean();
}
