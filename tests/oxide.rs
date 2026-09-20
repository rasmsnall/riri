//! Kernels written in cuda-oxide's shape, run under Riri.
//!
//! The point of these is not that Riri can execute them, but that the claims
//! cuda-oxide has to take on trust are checked once they run here.

use riri::oxide::{self, thread, warp, DisjointSlice};
use riri::{GlobalBuf, LaunchConfig};

/// The Tier 1 vector add from cuda-oxide's own documentation, unchanged
/// except for the `#[kernel]` attribute a GPU build would carry.
fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
    if let Some((mut c_elem, idx)) = c.get_mut_indexed() {
        let i = idx.get();
        *c_elem = a[i] + b[i];
    }
}

#[test]
fn tier_one_vecadd_is_clean() {
    let a: Vec<f32> = (0..128).map(|i| i as f32).collect();
    let b = vec![1.0f32; 128];
    let out = GlobalBuf::new("c", vec![0.0f32; 128]);
    let c = DisjointSlice::new(&out);

    let report = oxide::launch(&LaunchConfig::new(4, 32).seed(1), || {
        vecadd(&a, &b, c.clone())
    });

    report.assert_clean();
    assert_eq!(
        out.to_vec(),
        (0..128).map(|i| i as f32 + 1.0).collect::<Vec<_>>()
    );
}

#[test]
fn indices_past_the_end_get_nothing() {
    // 64 threads over a 48 element buffer: the tail must fall away rather
    // than trap, which is what `get_mut_indexed` returning None is for.
    let out = GlobalBuf::new("c", vec![0u32; 48]);
    let c = DisjointSlice::new(&out);

    let report = oxide::launch(&LaunchConfig::new(2, 32).seed(1), || {
        let mut c = c.clone();
        if let Some((mut elem, idx)) = c.get_mut_indexed() {
            *elem = idx.get() as u32;
        };
    });

    report.assert_clean();
    assert_eq!(out.to_vec()[47], 47);
}

#[test]
fn unchecked_access_claiming_one_element_twice_is_caught() {
    // This is the claim cuda-oxide cannot check: `get_unchecked_mut` asserts
    // the index is this thread's alone. Here every thread claims element 0.
    let out = GlobalBuf::new("c", vec![0u32; 32]);
    let c = DisjointSlice::new(&out);

    let report = oxide::launch(&LaunchConfig::new(1, 32).seed(1), || {
        let mut c = c.clone();
        let mut elem = c.get_unchecked_mut(0);
        *elem = thread::threadIdx_x();
    });

    assert!(report.has_race(), "{report}");
}

#[test]
fn unchecked_access_with_distinct_indices_is_clean() {
    let out = GlobalBuf::new("c", vec![0u32; 32]);
    let c = DisjointSlice::new(&out);

    let report = oxide::launch(&LaunchConfig::new(1, 32).seed(1), || {
        let mut c = c.clone();
        let i = thread::index_1d().get();
        let mut elem = c.get_unchecked_mut(i);
        *elem = i as u32;
    });

    report.assert_clean();
    assert_eq!(out.to_vec(), (0..32).collect::<Vec<u32>>());
}

#[test]
fn unchecked_access_out_of_range_traps() {
    let out = GlobalBuf::new("c", vec![0u32; 4]);
    let c = DisjointSlice::new(&out);

    let report = oxide::launch(&LaunchConfig::new(1, 8).seed(1), || {
        let mut c = c.clone();
        let mut elem = c.get_unchecked_mut(thread::index_1d().get());
        *elem = 1;
    });

    assert!(report.aborted, "{report}");
}

#[test]
fn block_indices_match_the_launch_shape() {
    let out = GlobalBuf::new("c", vec![0u32; 8]);
    let c = DisjointSlice::new(&out);

    let report = oxide::launch(&LaunchConfig::new(2, 4).seed(1), || {
        let mut c = c.clone();
        let packed = thread::blockIdx_x() * 100 + thread::threadIdx_x() * 10 + thread::blockDim_x();
        if let Some((mut elem, _)) = c.get_mut_indexed() {
            *elem = packed;
        };
    });

    report.assert_clean();
    assert_eq!(out.to_vec(), vec![4, 14, 24, 34, 104, 114, 124, 134]);
}

#[test]
fn sync_threads_orders_the_block() {
    let out = GlobalBuf::new("c", vec![0u32; 8]);
    let c = DisjointSlice::new(&out);

    let report = oxide::launch(&LaunchConfig::new(1, 8).seed(3), || {
        let mut c = c.clone();
        if let Some((mut elem, idx)) = c.get_mut_indexed() {
            *elem = idx.get() as u32;
        }
        thread::sync_threads();
    });

    report.assert_clean();
}

#[test]
fn a_shuffle_on_a_diverged_warp_is_caught() {
    // cuda-oxide's unsuffixed shuffle carries no member mask, so it assumes
    // the whole warp. Half a warp calling it is exactly the silent hang their
    // own documentation warns about.
    let report = oxide::launch(&LaunchConfig::new(1, 8).warp_size(8).seed(1), || {
        if warp::lane_id() < 4 {
            let _ = warp::shuffle(warp::lane_id(), 0);
        }
    });

    assert!(report.has_warp_divergence(), "{report}");
}

#[test]
fn a_converged_warp_reduction_is_clean() {
    let out = GlobalBuf::new("c", vec![0u32; 1]);
    let c = DisjointSlice::new(&out);

    let report = oxide::launch(&LaunchConfig::new(1, 8).warp_size(8).seed(1), || {
        let mut v = warp::lane_id() + 1;
        let mut delta = 4;
        while delta > 0 {
            v += warp::shuffle_down(v, delta);
            delta /= 2;
        }
        if warp::lane_id() == 0 {
            let mut c = c.clone();
            let mut elem = c.get_unchecked_mut(0);
            *elem = v;
        }
    });

    report.assert_clean();
    assert_eq!(out.to_vec()[0], (1..=8).sum::<u32>());
}

#[test]
fn votes_see_the_whole_warp() {
    let out = GlobalBuf::new("c", vec![0u32; 3]);
    let c = DisjointSlice::new(&out);

    let report = oxide::launch(&LaunchConfig::new(1, 8).warp_size(8).seed(1), || {
        let lane = warp::lane_id();
        let bits = warp::ballot(lane % 2 == 0);
        let any = warp::any(lane == 7);
        let all = warp::all(lane < 8);
        if lane == 0 {
            let mut c = c.clone();
            *c.get_unchecked_mut(0) = bits;
            *c.get_unchecked_mut(1) = any as u32;
            *c.get_unchecked_mut(2) = all as u32;
        }
    });

    report.assert_clean();
    assert_eq!(out.to_vec(), vec![0b0101_0101, 1, 1]);
}

#[test]
#[should_panic(expected = "outside riri::oxide::launch")]
fn a_device_function_outside_a_launch_says_so() {
    let _ = thread::threadIdx_x();
}
