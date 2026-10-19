/* sjb2ptx math library: declarations shared by symjit_math.c, symjit_complex.c and cephes.c.
 *
 * The entry points (`__sj_<op>`, `__sj_cplx_<op>`, <op> being symjit's name of the function)
 * are the only symbols gen.sh keeps public; everything else is internal to the library. */
#ifndef SJ_H
#define SJ_H

#include <math.h>

typedef struct {
    double re, im;
} cplx;

double __lgamma_r(double, int *);

void cephes_sici(double x, double *si, double *ci);
void cephes_shichi(double x, double *shi, double *chi);

#endif
