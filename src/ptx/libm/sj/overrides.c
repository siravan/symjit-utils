/* sjb2ptx math library: the musl functions that are single instructions.
 *
 * Compiled with -fno-math-errno, the builtins become sqrt.rn.f64, abs.f64 and cvt.rmi.f64.f64
 * on the GPU (sqrtsd and andpd on x86-64), which are exact or correctly rounded as musl's C
 * versions are. x86-64 has no floor instruction before SSE4.1, so the host build compiles
 * musl's floor.c instead. */

double sqrt(double x) { return __builtin_sqrt(x); }

double fabs(double x) { return __builtin_fabs(x); }

#ifdef __NVPTX__
double floor(double x) { return __builtin_floor(x); }
#endif
