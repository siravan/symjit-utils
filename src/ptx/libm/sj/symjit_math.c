/* sjb2ptx math library: symjit's real functions (v227 rust/code.rs, `VirtualTable`) on musl.
 *
 * Names are symjit's: `ln` is the natural logarithm and `log` the decimal one (v227), `power`
 * is pow, `gamma` tgamma, `loggamma` lgamma, `arg` of a real is 0. The reciprocal functions and
 * sinc are composed as code.rs composes them. erf, erfc, tgamma and lgamma are musl's (v227 uses
 * Cephes); Si, Ci, Shi and Chi are Cephes' (cephes.c). */

#include "sj.h"

double __sj_sin(double x) { return sin(x); }
double __sj_cos(double x) { return cos(x); }
double __sj_tan(double x) { return tan(x); }
double __sj_csc(double x) { return 1.0 / sin(x); }
double __sj_sec(double x) { return 1.0 / cos(x); }
double __sj_cot(double x) { return 1.0 / tan(x); }

double __sj_sinc(double x)
{
    if (x == 0.0)
        return 1.0;
    return sin(x) / x;
}

double __sj_sinh(double x) { return sinh(x); }
double __sj_cosh(double x) { return cosh(x); }
double __sj_tanh(double x) { return tanh(x); }
double __sj_csch(double x) { return 1.0 / sinh(x); }
double __sj_sech(double x) { return 1.0 / cosh(x); }
double __sj_coth(double x) { return 1.0 / tanh(x); }

double __sj_arcsin(double x) { return asin(x); }
double __sj_arccos(double x) { return acos(x); }
double __sj_arctan(double x) { return atan(x); }
double __sj_arcsinh(double x) { return asinh(x); }
/* musl 1.2.5's acosh tests |x| and is finite for x <= -2; it is NaN below 1 */
double __sj_arccosh(double x)
{
    if (x < 1.0)
        return (x - x) / (x - x);
    return acosh(x);
}
double __sj_arctanh(double x) { return atanh(x); }

double __sj_cbrt(double x) { return cbrt(x); }
double __sj_exp(double x) { return exp(x); }
double __sj_ln(double x) { return log(x); }
double __sj_log(double x) { return log10(x); }
double __sj_expm1(double x) { return expm1(x); }
double __sj_log1p(double x) { return log1p(x); }
double __sj_exp2(double x) { return exp2(x); }
double __sj_log2(double x) { return log2(x); }

double __sj_erf(double x) { return erf(x); }
double __sj_erfc(double x) { return erfc(x); }
double __sj_gamma(double x) { return tgamma(x); }

double __sj_loggamma(double x)
{
    int sign;
    return __lgamma_r(x, &sign);
}

double __sj_Si(double x)
{
    double si, ci;
    cephes_sici(x, &si, &ci);
    return si;
}

double __sj_Ci(double x)
{
    double si, ci;
    cephes_sici(x, &si, &ci);
    return ci;
}

double __sj_Shi(double x)
{
    double shi, chi;
    cephes_shichi(x, &shi, &chi);
    return shi;
}

double __sj_Chi(double x)
{
    double shi, chi;
    cephes_shichi(x, &shi, &chi);
    return chi;
}

double __sj_power(double x, double y) { return pow(x, y); }
double __sj_atan2(double x, double y) { return atan2(x, y); }
double __sj_arg(double x) { return 0.0; }
