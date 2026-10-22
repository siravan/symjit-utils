# symjit-utils

[SymJit](https://github.com/siravan/symjit) is a lightweight just-in-time (JIT) compiler that directly translates symbolic expressions into machine code without using a separate library such as LLVM. Its main utility is to generate fast numerical functions to feed into various numerical solvers. In Python, it acts as a bridge between Sympy and Numpy/Scipy solvers. SymJit is also the JIT backend for the [Symbolica](./docs/SYMBOLICA.md) computer algebra system.

**Symjit-utils** are a collection of binary utilities that expand the functionality of Symjit. Central to Symjit-utils is the Symjit-Binary (`.sjb`) format. This is the binary serialization format of Symjit. It contains a linearized Intermediate-Representation (IR). Of note, it does not include compiled machine codes; therefore, `.sjb` files are mostly portable. 

The binary utilities in Symjt-utils are:

* `sjb2wasm`: translates an `.sjb` file to WebAssembly (wasm).
* `sjb2obj`: compiles an `.sjb` file and saves the compiled kernel as an object file (`.o`), which can be linked into C/C++ programs.
* `sjb2ptx`: translates an `.sjb` file into CUDA ptx format.
* `sjbinfo`: prints the details of an `.sjb` file, useful for debugging and optimization. 
* `rust2sjb`: compiles a limited subset of Rust into `.sjb`. 

See `examples/viete` for an example of how to use different programs.
