# sjb2asm: symjit's machine code as assembly text

`sjb2asm` compiles a saved symjit application with symjit, for any of its three architectures on
any host, and lists one of its kernels: the address, the bytes and the instruction of each line,
with the data area as hex values. With `--asm`, it writes the kernel as assembly text that GNU `as`
assembles back to the kernel's bytes. It uses no
external tool. The decoders know exactly the instructions symjit emits on x86-64, AArch64 and
RISC-V. The names of branch targets and constants come from the labels symjit's assembler used
(`MachineCode::labels`, symjit 2.29.2). The code is never run.

```
sjb2asm [OPTIONS] <INPUT.sjb | INPUT.json>
sjb2asm --raw --arch ARCH [OPTIONS] <KERNEL.bin>

  -a, --arch ARCH     x86_64, aarch64, riscv64 (or symjit's ty: amd-avx, amd-sse, native);
                      default: the file's own target
  -k, --kernel KIND   scalar (default), simd or fast
  -s, --asm           assembler input (symbolic labels, no addresses or bytes)
  -x, --hex           the listing with addresses and bytes (the default)
  -o, --output FILE   write to FILE instead of standard output
      --raw           the input is a kernel's bytes (needs --arch)
      --data OFFSET   with --raw: where the data area starts (decimal or 0x..)
```

## Inputs

- **A `.sjb` file**: `f.save(path)` in Python, `Application::save` in Rust.
- **A "symjit-application" JSON file**:
  - `f.dump(path, what='json')` in Python, or `Application::to_json` in Rust;
  - pacrunner's `programs/<k>/kernel.json` files are of this kind.

  A complexified MIR (`'json-complex'`) cannot be loaded, just as symjit cannot load one.
- **Raw bytes, with `--raw`**: the file `f.dump(path, what='scalar')` writes (also `'simd'` and
  `'fast'`).
  - Raw bytes have no labels. Branch targets are named `.L_at_<offset>`.
  - The data area starts at `--data`. Without `--data`, it starts at the lowest address that an
    instruction refers to forward, jumps and calls excepted. This equals `@data` on every kernel
    checked so far (the tests and 60 pacrunner kernels).
  - A kernel that refers to no data at all is decoded to its end.

The target is the file's own (`ty` in its configuration; `native` is the host) unless `--arch`
names another. symjit compiles a loaded application for the `ty` its file names, so sjb2asm writes
`--arch` into the JSON before loading it.

Kernels:
- **simd** exists on x86-64 with AVX and on AArch64. RISC-V has none.
- **fast** needs one output, no parameters and few states.

x86-64 kernels use the host CPU's features: AVX and FMA, and AVX-512 if the application enables it
and the CPU has it. On a host without AVX, they are the SSE kernels. `--arch amd-sse` gives the SSE
kernels anywhere. So an x86-64 listing depends on the machine that makes it.

## Output

By default (or `--hex`), each line has the address (the offset in the kernel), the bytes and
the instruction. The labels head their lines (`@self:`, `@data:`, `_const_0_:`). Targets are
numbers with their label names (`0x1a0 <_func_exp_>`). The data area is `.quad`s with their hex
values and, in comments, the doubles they hold:

```
$ sjb2asm --arch aarch64 f.sjb          # f = compile_func([x, y], [x+y, x*y+2.5])
// symjit scalar kernel, aarch64, 208 bytes: code 0x0..0xc8, data 0xc8..0xd0
@self:
       0:  d10103ff          sub sp, sp, #64
       4:  a90077fe          stp x30, x29, [sp, #0]
...
@data:
_const_0_:
      c8:  4004000000000000  .quad 0x4004000000000000    // 2.5
```

With `--asm`, the output is assembler input:

```
$ sjb2asm --arch x86_64 --asm f.sjb
# symjit scalar kernel, x86_64, 352 bytes: code 0x0..0x100, data 0x100..0x160
	.text
	.intel_syntax noprefix
	.p2align 5
	.globl symjit_scalar
	.type symjit_scalar, %function
symjit_scalar:
.L_self:
	push rbp
	.insn 0x81/5, rsp, 48{:s32}    # sub rsp, 48
	{disp8} mov qword ptr [rsp], r13
	...
	vaddsd xmm4, xmm4, qword ptr [rip+.L_const_0_]    # .L_const_0_ = 2.5
	...
.L_data:
.L_const_0_:
	.quad 0x4004000000000000    # 2.5
```

The text assembles back to the kernel's bytes:

```
as k.s -o k.o
ld -Ttext=0 -e symjit_scalar k.o
objcopy -O binary -j .text
```

For a foreign architecture, use the cross binutils (`aarch64-linux-gnu-as`, ...).

- **Labels**: they become local symbols. `@self` is `.L_self`, `_const_0_` is `.L_const_0_` and
  `@data` is `.L_data`. The entry is the global function `symjit_<kernel>`.
- **Data area**: the constants follow the code as `.quad`s, with their values in comments.
- **Function table**: a kernel that calls math functions keeps their addresses in its function
  table (`_func_<name>_`, `_simd_<name>_`). The addresses are valid only in the process that
  compiled it, so they change from run to run.
- **Spelling**: each instruction is spelled so that `as` encodes it as symjit did.
  - RISC-V: `add rd, rs, zero` rather than `mv`. `.option norvc` and `.option norelax` keep every
    instruction 4 bytes long.
  - AArch64: `adrp` and `:lo12:` name the constants.
  - x86-64 (Intel syntax): the pseudo-prefixes `{load}`/`{store}`, `{disp8}`/`{disp32}`, `{vex3}`,
    `{evex}` and `rex.w` select symjit's encodings. The few that no prefix gives become `.insn`
    lines, with the instruction in a comment:
    - scalar VEX instructions with VEX.L=1;
    - a 32-bit immediate that fits in a byte.

    `.insn` needs binutils 2.40 or newer.
- **Syntax**: GNU's, for ELF (Linux).

## Code

- `src/disasm/` is the decoders, moved from symjit's `asm` branch (96fbdb3, never merged into
  symjit):
  - `mod.rs`: `listing`, `data_start`;
  - `riscv.rs`, `arm.rs`, `x86.rs`.
- Their tests are generated from symjit's encoder tests, which pin each word against an
  independent assembler:

  ```
  python3 src/disasm/gen_tests.py riscv|arm|x86 SYMJIT > src/disasm/<arch>_tests.rs
  ```

  `SYMJIT` is a symjit checkout (default `../stable`). Regenerate the tests when symjit's encoders
  change. A new instruction then fails the decoder test until the decoder learns it.
- `src/asm.rs` loads the input, compiles it for the target and picks the kernel.
- `tests/sjb2asm.rs` compiles kernels on all three architectures, real and complex, and checks
  that:
  - every instruction before the data area decodes;
  - labels and label references are named;
  - with the binutils installed, the listing (and the raw listing) assembles back to the same
    bytes;
  - the data area found without labels is `@data`.

  `SJB2ASM_SAMPLES=DIR` also checks every `*.sjb` and `kernel.json` under `DIR`.
