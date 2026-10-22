/* Runs the PTX kernel of viete.sjb (sjb2ptx --driver: viete_cuda.c and viete_cuda.h) on the GPU
   over a range of inputs that includes 0.5, and prints the result at 0.5.

       cc -I<dir of viete_cuda.h> ptx_main.c viete_cuda.c -ldl -lm -o ptx_main

   One GPU thread per point; the input of a point is the model's parameter (docs/PTX.md). Exits
   with status 77 when there is no usable GPU. */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "viete_cuda.h"

#define N 4096 /* points 0.25, ..., 0.75 (step 1/8192): x[2048] = 0.5 exactly */

int main(void) {
    static double in[N], out[N];
    for (size_t i = 0; i < N; i++)
        in[i] = 0.25 + (double)i / (2 * N);

    viete_cuda *ctx;
    if (viete_cuda_init(&ctx, 0) != 0) {
        fprintf(stderr, "ptx: no GPU: %s\n", viete_cuda_error(ctx));
        viete_cuda_free(ctx);
        return 77;
    }
    /* SoA with one input and one output: ld_in = ld_out = N */
    if (viete_cuda_evaluate(ctx, N, in, N, out, N, NULL) != 0) {
        fprintf(stderr, "ptx: %s\n", viete_cuda_error(ctx));
        viete_cuda_free(ctx);
        return 1;
    }
    viete_cuda_free(ctx);

    size_t k = N / 2;
    if (in[k] != 0.5) {
        fprintf(stderr, "ptx: in[%zu] = %g, not 0.5\n", k, in[k]);
        return 1;
    }
    printf("ptx:  viete(0.5) = %.17g  (pi - viete(0.5) = %.3e; %d points from %g to %g)\n", out[k],
           M_PI - out[k], N, in[0], in[N - 1]);
    return 0;
}
