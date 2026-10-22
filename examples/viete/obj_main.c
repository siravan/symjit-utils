/* Calls the object file of viete.sjb (sjb2obj: viete.o and viete.h) with x = 0.5 and prints
   the result.

       cc -I<dir of viete.h> obj_main.c viete.o -lm -o obj_main

   The model is a kernel: its input is a parameter and its output goes to `mem`, so the call is
   viete(outs, NULL, 0, inputs) (docs/OBJ.md). */

#include <math.h>
#include <stdio.h>

#include "viete.h"

int main(void) {
    double outs[VIETE_MEM_SIZE] = {0};
    double inputs[VIETE_COUNT_PARAMS] = {0.5};

    viete(outs, NULL, 0, inputs);
    printf("obj:  viete(0.5) = %.17g  (pi - viete(0.5) = %.3e)\n", outs[0], M_PI - outs[0]);
    return 0;
}
