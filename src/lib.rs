//! Command-line utilities for saved symjit applications (`.sjb`), built on the `symjit` crate
//! (`symjit::backend`): `sjb2wasm` compiles one into a WebAssembly module, `sjb2obj` into a
//! relocatable object file and C header, `rust2sjb` saves a `#[kernel]` function of a Rust
//! file (`source`, `kernel`), `sjbinfo` prints what one holds (`info`), and `sjb2ptx`
//! compiles one into a CUDA PTX module (`ptx`).

// the generators copied from symjit name their register map `ϕ`
#![allow(uncommon_codepoints)]

pub mod info;
pub mod kernel;
pub mod linker;
mod obj;
pub mod ptx;
pub mod sjb;
pub mod source;
pub mod wasm;

pub use sjb::SavedModel;
