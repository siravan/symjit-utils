/* viete_cuda.h: runs the PTX kernel `viete` on an NVIDIA GPU; written by sjb2ptx 0.1.0.

   Build with `cc your_program.c viete_cuda.c -ldl`: the driver loads libcuda.so.1 at run time
   and needs neither the CUDA toolkit nor -lcuda.

       viete_cuda *ctx;
       if (viete_cuda_init(&ctx, 0) != 0) { puts(viete_cuda_error(ctx)); ... }
       viete_cuda_evaluate(ctx, n, in, ld_in, out, ld_out, params);   // host arrays
       viete_cuda_free(ctx);

   Each point has 1 inputs and 1 outputs (doubles); `params` holds 0
   shared parameters (NULL if none). Layout SoA: slot k of point i is at in[k * ld_in + i] (ld_in >= n), and likewise for `out`.
   Functions return 0 on success and -1 on errors (viete_cuda_error describes the last one).
   A context is used by one thread at a time.
*/

#ifndef VIETE_CUDA_H
#define VIETE_CUDA_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define VIETE_CUDA_COUNT_IN 1
#define VIETE_CUDA_COUNT_OUT 1
#define VIETE_CUDA_COUNT_PARAMS 0
#define VIETE_CUDA_LAYOUT_SOA 1

typedef struct viete_cuda viete_cuda;

/* Loads libcuda, initializes `device` (its primary context) and compiles the kernel. `*ctx`
   is set even on failure, for viete_cuda_error; free it with viete_cuda_free. */
int viete_cuda_init(viete_cuda **ctx, int device);

/* Evaluates n points of host arrays: copies them to the GPU in batches, runs the kernel and
   copies the outputs back. */
int viete_cuda_evaluate(viete_cuda *ctx, size_t n, const double *in, size_t ld_in, double *out,
                      size_t ld_out, const double *params);

/* Launches the kernel on device arrays (CUdeviceptr values) on `stream` (a CUstream, or NULL)
   without waiting for it. */
int viete_cuda_evaluate_device(viete_cuda *ctx, size_t n, unsigned long long in, size_t ld_in,
                             unsigned long long out, size_t ld_out, unsigned long long params,
                             void *stream);

/* The most points viete_cuda_evaluate sends to the GPU at a time (default 1048576; fewer if
   the GPU's memory is short). */
void viete_cuda_set_batch(viete_cuda *ctx, size_t max_points);

/* The last error (with the CUDA error or the PTX compiler's log), or "". */
const char *viete_cuda_error(const viete_cuda *ctx);

void viete_cuda_free(viete_cuda *ctx);

#ifdef __cplusplus
}
#endif

#endif /* VIETE_CUDA_H */
