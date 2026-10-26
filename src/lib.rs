//! Command-line utilities for saved symjit applications (`.sjb`), built on the `symjit` crate
//! (`symjit::backend`): `sjb2wasm` compiles one into a WebAssembly module, `sjb2obj` into a
//! relocatable object file and C header, `rust2sjb` saves a `#[kernel]` function of a Rust
//! file (`source`, `kernel`), `sjbinfo` prints what one holds (`info`), `sjb2ptx`
//! compiles one into a CUDA PTX module (`ptx`), and `pacman` reports on and extracts the
//! kernels of pyAmpliCol's evaluator containers (`pacbin`, `pacman`), its recurrence plans
//! (`plan`) and the plane tables of its compiled-mode kernels (`bindings`), and replays RustiCol
//! arena recordings with them (`recording`, `replay`) or builds them into a runner (`runner`).

// the generators copied from symjit name their register map `ϕ`
#![allow(uncommon_codepoints)]

pub mod bindings;
pub mod info;
pub mod kernel;
pub mod linker;
mod obj;
pub mod pacbin;
pub mod pacman;
pub mod plan;
pub mod ptx;
pub mod recording;
pub mod replay;
pub mod runner;
pub mod sha256;
pub mod sjb;
pub mod source;
pub mod wasm;

pub use sjb::SavedModel;
