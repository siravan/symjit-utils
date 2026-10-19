/* sjb2ptx math library: symjit's complex functions (v227 rust/code.rs without its `libm`
 * feature), on musl's real functions.
 *
 * The formulas are num_complex 0.4.6's (src/lib.rs; MIT OR Apache-2.0), ported operation by
 * operation: constants such as i, 1 and 2 are complex numbers multiplied, added and divided as
 * num_complex does it, because the signs of zeros and the NaNs of infinite arguments depend on
 * it. recip, tanh_c and tan_c are code.rs's own. As in code.rs, `log` is the natural logarithm
 * (not log10, unlike the real `log`) and `arg` gives (arg z, 0). */

#include "sj.h"

static cplx C(double re, double im)
{
    cplx z = { re, im };
    return z;
}

static cplx add(cplx a, cplx b) { return C(a.re + b.re, a.im + b.im); }
static cplx sub(cplx a, cplx b) { return C(a.re - b.re, a.im - b.im); }
static cplx neg(cplx a) { return C(-a.re, -a.im); }

static cplx mul(cplx a, cplx b)
{
    return C(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
}

static cplx div(cplx a, cplx b)
{
    double norm_sqr = b.re * b.re + b.im * b.im;
    double re = a.re * b.re + a.im * b.im;
    double im = a.im * b.re - a.re * b.im;
    return C(re / norm_sqr, im / norm_sqr);
}

static int eq(cplx a, cplx b) { return a.re == b.re && a.im == b.im; }

static const cplx ONE = { 1.0, 0.0 };
static const cplx TWO = { 2.0, 0.0 };
static const cplx I = { 0.0, 1.0 };

static double norm(cplx z) { return hypot(z.re, z.im); }
static double arg(cplx z) { return atan2(z.im, z.re); }

static cplx from_polar(double r, double theta) { return C(r * cos(theta), r * sin(theta)); }

static cplx c_exp(cplx z)
{
    double re = z.re, im = z.im;
    if (isinf(re)) {
        if (re < 0.0) {
            if (!isfinite(im))
                return C(0.0, 0.0);
        } else if (im == 0.0 || !isfinite(im)) {
            if (isinf(im))
                im = NAN;
            return C(re, im);
        }
    } else if (isnan(re) && im == 0.0) {
        return z;
    }
    return from_polar(exp(re), im);
}

static cplx c_ln(cplx z) { return C(log(norm(z)), arg(z)); }

static cplx c_sqrt(cplx z)
{
    if (z.im == 0.0) {
        if (!signbit(z.re))
            return C(sqrt(z.re), z.im);
        double re = 0.0;
        double im = sqrt(-z.re);
        return !signbit(z.im) ? C(re, im) : C(re, -im);
    } else if (z.re == 0.0) {
        double x = sqrt(fabs(z.im) / 2.0);
        return !signbit(z.im) ? C(x, x) : C(x, -x);
    } else {
        return from_polar(sqrt(norm(z)), arg(z) / 2.0);
    }
}

static cplx c_cbrt(cplx z)
{
    if (z.im == 0.0) {
        if (!signbit(z.re))
            return C(cbrt(z.re), z.im);
        double re = cbrt(-z.re) / 2.0;
        double im = sqrt(3.0) * re;
        return !signbit(z.im) ? C(re, im) : C(re, -im);
    } else if (z.re == 0.0) {
        double im = cbrt(fabs(z.im)) / 2.0;
        double re = sqrt(3.0) * im;
        return !signbit(z.im) ? C(re, im) : C(re, -im);
    } else {
        return from_polar(cbrt(norm(z)), arg(z) / 3.0);
    }
}

static cplx c_powc(cplx z, cplx e)
{
    if (e.re == 0.0 && e.im == 0.0)
        return ONE;
    return c_exp(mul(e, c_ln(z)));
}

static cplx c_sin(cplx z) { return C(sin(z.re) * cosh(z.im), cos(z.re) * sinh(z.im)); }
static cplx c_cos(cplx z) { return C(cos(z.re) * cosh(z.im), -sin(z.re) * sinh(z.im)); }
static cplx c_sinh(cplx z) { return C(sinh(z.re) * cos(z.im), cosh(z.re) * sin(z.im)); }
static cplx c_cosh(cplx z) { return C(cosh(z.re) * cos(z.im), sinh(z.re) * sin(z.im)); }

/* arcsin(z) = -i ln(sqrt(1-z^2) + iz) */
static cplx c_asin(cplx z)
{
    return mul(neg(I), c_ln(add(c_sqrt(sub(ONE, mul(z, z))), mul(I, z))));
}

/* arccos(z) = -i ln(i sqrt(1-z^2) + z) */
static cplx c_acos(cplx z)
{
    return mul(neg(I), c_ln(add(mul(I, c_sqrt(sub(ONE, mul(z, z)))), z)));
}

/* arctan(z) = (ln(1+iz) - ln(1-iz))/(2i) */
static cplx c_atan(cplx z)
{
    if (eq(z, I))
        return C(0.0, INFINITY);
    if (eq(z, neg(I)))
        return C(0.0, -INFINITY);
    return div(sub(c_ln(add(ONE, mul(I, z))), c_ln(sub(ONE, mul(I, z)))), mul(TWO, I));
}

/* arcsinh(z) = ln(z + sqrt(1+z^2)) */
static cplx c_asinh(cplx z) { return c_ln(add(z, c_sqrt(add(ONE, mul(z, z))))); }

/* arccosh(z) = 2 ln(sqrt((z+1)/2) + sqrt((z-1)/2)) */
static cplx c_acosh(cplx z)
{
    return mul(TWO, c_ln(add(c_sqrt(div(add(z, ONE), TWO)), c_sqrt(div(sub(z, ONE), TWO)))));
}

/* arctanh(z) = (ln(1+z) - ln(1-z))/2 */
static cplx c_atanh(cplx z)
{
    if (eq(z, ONE))
        return C(INFINITY, 0.0);
    if (eq(z, neg(ONE)))
        return C(-INFINITY, 0.0);
    return div(sub(c_ln(add(ONE, z)), c_ln(sub(ONE, z))), TWO);
}

/* code.rs: 1/w by Smith's algorithm; 1/inf = 0 with the signs of conj(1/w) */
static cplx recip(cplx w)
{
    if (isinf(w.re) || isinf(w.im))
        return C(__builtin_copysign(0.0, w.re), __builtin_copysign(-0.0, w.im));
    if (fabs(w.re) >= fabs(w.im)) {
        double r = w.im / w.re;
        double d = w.re + w.im * r;
        return C(1.0 / d, -r / d);
    } else {
        double r = w.re / w.im;
        double d = w.re * r + w.im;
        return C(r / d, -1.0 / d);
    }
}

/* code.rs: tanh z = (sinh x cosh x + i sin y cos y) / (sinh^2 x + cos^2 y) */
static cplx tanh_c(cplx z)
{
    double x = z.re, y = z.im;
    double sy = sin(y), cy = cos(y);
    if (fabs(x) > 22.0) {
        double e = exp(-2.0 * fabs(x));
        return C(__builtin_copysign(1.0, x), 4.0 * sy * cy * e);
    } else {
        double s = sinh(x), c = cosh(x);
        double d = s * s + cy * cy;
        return C(s * c / d, sy * cy / d);
    }
}

/* code.rs: tan(z) = -i tanh(iz) */
static cplx tan_c(cplx z)
{
    cplx t = tanh_c(C(-z.im, z.re));
    return C(t.im, -t.re);
}

cplx __sj_cplx_sin(double re, double im) { return c_sin(C(re, im)); }
cplx __sj_cplx_cos(double re, double im) { return c_cos(C(re, im)); }
cplx __sj_cplx_tan(double re, double im) { return tan_c(C(re, im)); }
cplx __sj_cplx_csc(double re, double im) { return recip(c_sin(C(re, im))); }
cplx __sj_cplx_sec(double re, double im) { return recip(c_cos(C(re, im))); }
cplx __sj_cplx_cot(double re, double im) { return recip(tan_c(C(re, im))); }

cplx __sj_cplx_sinc(double re, double im)
{
    cplx x = C(re, im);
    if (re == 0.0 && im == 0.0)
        return ONE;
    return div(c_sin(x), x);
}

cplx __sj_cplx_sinh(double re, double im) { return c_sinh(C(re, im)); }
cplx __sj_cplx_cosh(double re, double im) { return c_cosh(C(re, im)); }
cplx __sj_cplx_tanh(double re, double im) { return tanh_c(C(re, im)); }
cplx __sj_cplx_csch(double re, double im) { return recip(c_sinh(C(re, im))); }
cplx __sj_cplx_sech(double re, double im) { return recip(c_cosh(C(re, im))); }
cplx __sj_cplx_coth(double re, double im) { return recip(tanh_c(C(re, im))); }

cplx __sj_cplx_arcsin(double re, double im) { return c_asin(C(re, im)); }
cplx __sj_cplx_arccos(double re, double im) { return c_acos(C(re, im)); }
cplx __sj_cplx_arctan(double re, double im) { return c_atan(C(re, im)); }
cplx __sj_cplx_arcsinh(double re, double im) { return c_asinh(C(re, im)); }
cplx __sj_cplx_arccosh(double re, double im) { return c_acosh(C(re, im)); }
cplx __sj_cplx_arctanh(double re, double im) { return c_atanh(C(re, im)); }

cplx __sj_cplx_cbrt(double re, double im) { return c_cbrt(C(re, im)); }
cplx __sj_cplx_exp(double re, double im) { return c_exp(C(re, im)); }
cplx __sj_cplx_ln(double re, double im) { return c_ln(C(re, im)); }
cplx __sj_cplx_log(double re, double im) { return c_ln(C(re, im)); }

/* x^z: code.rs's `Complex::new(xr, xi).powc(*z)` */
cplx __sj_cplx_power(double re, double im, double zre, double zim)
{
    return c_powc(C(re, im), C(zre, zim));
}

cplx __sj_cplx_arg(double re, double im) { return C(arg(C(re, im)), 0.0); }
