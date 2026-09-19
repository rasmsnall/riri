//! # Riri
//!
//! A Miri-style undefined-behaviour detector for Rust GPU kernels.
//!
//! Riri runs a SIMT kernel on the CPU: every GPU thread becomes a simulated
//! thread, and a deterministic, seeded scheduler interleaves them one
//! instrumented operation at a time. Every memory access goes through shadow
//! memory, so Riri can report:
//!
//! - **data races** on global and shared memory (happens-before via
//!   `sync_threads` barriers within a block; no ordering across blocks),
//! - **barrier divergence** (some threads of a block wait at a barrier that
//!   other threads of the same block never reach),
//! - **uninitialised shared-memory reads**,
//! - **out-of-bounds accesses** and **kernel panics** (reported as traps).
//!
//! No GPU is required, so Riri runs in ordinary `cargo test` and CI.
//!
//! ```
//! use riri::{launch, GlobalBuf, LaunchConfig};
//!
//! let a = GlobalBuf::new("a", vec![1.0f32; 64]);
//! let b = GlobalBuf::new("b", vec![2.0f32; 64]);
//! let c = GlobalBuf::new("c", vec![0.0f32; 64]);
//!
//! let report = launch(&LaunchConfig::new(2, 32).seed(7), |t| {
//!     let i = t.global_linear();
//!     c.write(t, i, a.read(t, i) + b.read(t, i));
//! });
//!
//! report.assert_clean();
//! assert!(c.to_vec().iter().all(|&x| x == 3.0));
//! ```

mod ctx;
mod diag;
mod dim;
mod launch;
mod mem;
mod sched;
mod shadow;

pub use ctx::ThreadCtx;
pub use diag::{Access, AccessKind, Diagnostic, MemSpace, Report};
pub use dim::Dim3;
pub use launch::{launch, LaunchConfig};
pub use mem::{GlobalBuf, SharedArray};
