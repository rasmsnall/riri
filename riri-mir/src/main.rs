//! A rustc driver that extracts MIR for Riri.
//!
//! This is the first step towards *interpreting* kernels rather than
//! executing them against an instrumented API, which is the roadmap item that
//! would let arbitrary `unsafe` in a kernel be checked without rewriting it.
//!
//! It does no interpretation yet. It runs an ordinary compilation, enters the
//! `rustc_public` context once analysis is done, and prints the MIR of the
//! functions it finds. What it proves is the pipeline: that Riri can reach
//! monomorphized, borrow-checked MIR through a stable interface, which is the
//! part worth de-risking before any interpreter work starts.
//!
//! # Why this is a separate crate
//!
//! `rustc_public` is a stable interface to an unstable compiler. The API is
//! meant to hold across nightlies, but it ships inside the rustc repository
//! and is reached through `rustc_private`, so this binary links against the
//! compiler itself and needs the `rustc-dev` component. The library crate is
//! stable, dependency-free, and MSRV 1.75, and none of that can survive here,
//! so the two share neither a toolchain nor a workspace.
//!
//! # Running it
//!
//! ```text
//! cargo build
//! riri-mir --edition 2021 --crate-type lib path/to/kernel.rs --out-dir out
//! ```
//!
//! The sysroot is supplied automatically. On Windows the toolchain's `bin`
//! directory has to be on `PATH` so that `rustc_driver`'s DLL resolves; the
//! README has the one-liner.
//!
//! Set `RIRI_MIR_ONLY` to a substring to print only the functions that match.

#![feature(rustc_private)]

extern crate rustc_driver;
extern crate rustc_interface;
extern crate rustc_middle;
extern crate rustc_public;
extern crate rustc_public_bridge;
extern crate rustc_session;

mod interp;
mod value;

use rustc_driver::{Callbacks, Compilation};
use rustc_middle::ty::TyCtxt;
use rustc_public::CrateDef;

struct Riri;

impl Callbacks for Riri {
    fn after_analysis(
        &mut self,
        _compiler: &rustc_interface::interface::Compiler,
        tcx: TyCtxt<'_>,
    ) -> Compilation {
        // `run` opens the scope in which stable MIR queries are available.
        // The context borrows from the compiler's arena and cannot outlive
        // it, which is why this is a closure rather than a handle to keep.
        let _ = rustc_public::rustc_internal::run(tcx, dump);
        // Carry on with the real compilation, so this stays a drop-in
        // replacement for rustc rather than a fork of the build.
        Compilation::Continue
    }
}

fn dump() {
    let run = std::env::var("RIRI_MIR_RUN").ok();
    let only = std::env::var("RIRI_MIR_ONLY").ok();
    for item in rustc_public::all_local_items() {
        let name = item.name();
        if let Some(filter) = &only {
            if !name.contains(filter.as_str()) {
                continue;
            }
        }
        // An item with no body is a declaration, not a definition.
        let Some(body) = item.body() else { continue };

        // With RIRI_MIR_RUN set, interpret the matching function instead of
        // printing it.
        if let Some(target) = &run {
            if name.contains(target.as_str()) {
                match interp::Interp::run(&body) {
                    Ok(value) => println!("{name} = {value}"),
                    Err(why) => println!("{name} stopped: {why}"),
                }
            }
            continue;
        }

        println!("fn {name}");
        println!("  {} block(s), {} local(s)", body.blocks.len(), body.locals().len());
        for (index, block) in body.blocks.iter().enumerate() {
            println!("  bb{index}:");
            for statement in &block.statements {
                println!("    {:?}", statement.kind);
            }
            println!("    -> {:?}", block.terminator.kind);
        }
    }
}

/// The sysroot this driver was built against.
///
/// A driver is not rustc, so nothing supplies this for it. Asking `rustc` is
/// reliable here because the pinned `rust-toolchain.toml` makes the rustup
/// shim resolve to the same nightly this links against.
fn sysroot() -> Option<String> {
    let out = std::process::Command::new("rustc").arg("--print").arg("sysroot").output().ok()?;
    let path = String::from_utf8(out.stdout).ok()?;
    Some(path.trim().to_string())
}

fn main() {
    let mut args: Vec<String> = std::env::args().collect();
    if !args.iter().any(|a| a.starts_with("--sysroot")) {
        if let Some(path) = sysroot() {
            args.push("--sysroot".into());
            args.push(path);
        }
    }
    rustc_driver::run_compiler(&args, &mut Riri);
}
