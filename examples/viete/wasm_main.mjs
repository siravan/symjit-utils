// Calls the WebAssembly module of viete.sjb (sjb2wasm) with x = 0.5 and prints the result.
//
//     node wasm_main.mjs viete.wasm
//
// The model is a kernel: its input is a parameter and its output goes to `mem`, so the call is
// run(outs, 0, 0, inputs) (docs/WASM.md). It calls no math functions (sqrt is an instruction),
// so the module has no imports.

import { readFileSync } from "node:fs";

const path = process.argv[2] ?? "viete.wasm";
const module = new WebAssembly.Module(readFileSync(path));
const { run, memory, __heap_base } = new WebAssembly.Instance(module, {}).exports;

const f64 = new Float64Array(memory.buffer);
const outs = __heap_base.value; // one output
const inputs = outs + 8; // one input

f64[inputs / 8] = 0.5;
if (run(outs, 0, 0, inputs) !== 0) {
    throw new Error("run: the stack is exhausted");
}
const p = f64[outs / 8];
console.log(`wasm: viete(0.5) = ${p.toPrecision(17)}  (pi - viete(0.5) = ${(Math.PI - p).toExponential(3)})`);
