// The C header written next to a PTX module: the kernel's name, its counts, its layout and
// how to launch it.

use super::{Layout, PtxOptions, Shape, BLOCK_SIZE};

/// `complex`: the model is complex (its counts are in doubles, a complex value takes two);
/// `ptx_file`, `header_file`: the names for the header's first lines.
pub fn header(
    options: &PtxOptions,
    shape: &Shape,
    complex: bool,
    ptx_file: &str,
    header_file: &str,
) -> String {
    let n = &options.name;
    let up = n.to_uppercase();
    let guard = format!("{}_PTX_H", up);
    let values = if complex {
        "Counts are in doubles: complex values are (re, im) pairs of consecutive slots."
    } else {
        "All values are doubles."
    };
    let inputs = if shape.row_first {
        "the model's parameters (a Symbolica model: different for each point)"
    } else {
        "the states"
    };
    let address = match options.layout {
        Layout::Soa => {
            "slot k of point i is at in[k * ld_in + i] (ld_in >= n), out[k * ld_out + i]"
        }
        Layout::Aos => {
            "slot k of point i is at in[i * ld_in + k] (ld_in >= COUNT_IN), out[i * ld_out + k]"
        }
    };

    format!(
        "/* {header_file}: the PTX kernel `{n}` in {ptx_file}, written by sjb2ptx {version}.

   One GPU thread per point. Load the module with the CUDA driver (cuModuleLoadData compiles
   it for the GPU) and launch `{n}` with a one-dimensional grid of at least n threads:

       .entry {n}(u64 in, u64 out, u64 params, u64 n, u64 ld_in, u64 ld_out)

       void *args[] = {{&in, &out, &params, &n, &ld_in, &ld_out}};   // device pointers, sizes
       cuLaunchKernel(f, (n + {up}_BLOCK_SIZE - 1) / {up}_BLOCK_SIZE, 1, 1,
                      {up}_BLOCK_SIZE, 1, 1, 0, stream, args, NULL);

   `in` holds {up}_COUNT_IN inputs per point, {inputs};
   `out` receives {up}_COUNT_OUT outputs per point (the observables, then any derivatives);
   `params` holds {up}_COUNT_PARAMS shared parameters, read by every thread (0 if none).
   Layout {layout}: {address}.
   `in` and `out` must not overlap. {values}
*/

#ifndef {guard}
#define {guard}

#define {up}_KERNEL \"{n}\"
#define {up}_COUNT_IN {count_in}
#define {up}_COUNT_OUT {count_out}
#define {up}_COUNT_PARAMS {count_params}
#define {up}_LAYOUT_SOA {soa}
#define {up}_COMPLEX {complex}
#define {up}_BLOCK_SIZE {block}

#endif /* {guard} */
",
        version = env!("CARGO_PKG_VERSION"),
        layout = options.layout.name().to_uppercase(),
        count_in = shape.count_in,
        count_out = shape.count_out,
        count_params = shape.count_params,
        soa = (options.layout == Layout::Soa) as u8,
        complex = complex as u8,
        block = BLOCK_SIZE,
    )
}
