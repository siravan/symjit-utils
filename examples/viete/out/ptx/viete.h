/* viete.h: the PTX kernel `viete` in viete.ptx, written by sjb2ptx 0.1.0.

   One GPU thread per point. Load the module with the CUDA driver (cuModuleLoadData compiles
   it for the GPU) and launch `viete` with a one-dimensional grid of at least n threads:

       .entry viete(u64 in, u64 out, u64 params, u64 n, u64 ld_in, u64 ld_out)

       void *args[] = {&in, &out, &params, &n, &ld_in, &ld_out};   // device pointers, sizes
       cuLaunchKernel(f, (n + VIETE_BLOCK_SIZE - 1) / VIETE_BLOCK_SIZE, 1, 1,
                      VIETE_BLOCK_SIZE, 1, 1, 0, stream, args, NULL);

   `in` holds VIETE_COUNT_IN inputs per point, the model's parameters (a Symbolica model: different for each point);
   `out` receives VIETE_COUNT_OUT outputs per point (the observables, then any derivatives);
   `params` holds VIETE_COUNT_PARAMS shared parameters, read by every thread (0 if none).
   Layout SOA: slot k of point i is at in[k * ld_in + i] (ld_in >= n), out[k * ld_out + i].
   `in` and `out` must not overlap. All values are doubles.
*/

#ifndef VIETE_PTX_H
#define VIETE_PTX_H

#define VIETE_KERNEL "viete"
#define VIETE_COUNT_IN 1
#define VIETE_COUNT_OUT 1
#define VIETE_COUNT_PARAMS 0
#define VIETE_LAYOUT_SOA 1
#define VIETE_COMPLEX 0
#define VIETE_BLOCK_SIZE 128

#endif /* VIETE_PTX_H */
