// SPDX-License-Identifier: 0BSD
//
// pacman recurrence runner: one CUDA kernel runs a whole RustiCol Direct-Arena recurrence plan
// (contributions, finalizations, closures), given as an array of 64-bit row words.
//
// Each thread is one phase-space point and walks every word in order. Rows depend only on rows
// of the same point, so the kernel needs no synchronisation, and all threads of a warp take the
// same branch of the dispatch. The plan is data: one build serves every process.
//
// A word holds, from the least significant bit: the op (4 bits), then the op's fields, whose
// widths the caller chooses per plan (`width`, see `pacman_recurrence.h` below):
//   P  current plane (component), A  amplitude plane, F  momentum form, X  factor index,
//   S  executor slot (its scale), K  small component count.
//
//   op 1  clear currents          P base, P count
//   op 2–6  contribution          1 initialize, S slot, P parent0, P parent1, P destination,
//                                 F form0, F form1, X exact factor
//           2 Weyl–vector→Weyl a, 3 b, 4 antisymmetric tensor–vector, 5 vector wedge vector,
//           6 colour-ordered three-vector
//   op 7–9  propagator            P base, F form, X exact factor
//           7 Weyl a, 8 Weyl b, 9 Feynman vector
//   op 10 identity finalization   P base, K count, X exact factor
//   op 11 closure-reduce          P parent0, P parent1, A amplitude, X exact factor,
//                                 X first component factor, K count
//
// The arithmetic is RustiCol's (`recurrence_intrinsic_direct.rs`, `recurrence_closure_direct.rs`)
// operation for operation; built with -fmad=false, the `f64` kernel equals RustiCol's bit for bit.
// The default kernel, `df64`, emulates f64 with pairs of f32 (for GPUs whose FP64 is slow, e.g.
// RTX); see `Options` below for PACMAN_RECURRENCE_KERNEL.
//
// Build: nvcc -O3 -fmad=false -Xcompiler -fPIC -shared -cudart static -o librecurrence.so
//        recurrence_runner.cu

#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <algorithm>
#include <unordered_map>
#include <vector>

// ---- the interface (pacman_recurrence.h) ---------------------------------------------------

#define PACMAN_RECURRENCE_ABI 1

enum { W_P, W_A, W_F, W_X, W_S, W_K, W_COUNT };

// Planes are plane-major with `stride` doubles per plane, as RustiCol's arena. A run list holds
// pairs (first plane, plane count). The current arena stays on the device between calls: only
// the runs named are copied.
typedef struct {
    uint32_t abi_version;
    uint32_t struct_size;
    const uint64_t *words;
    uint64_t words_key;            // nonzero: `words` never changes under this key (cached)
    uint32_t word_count;
    uint8_t width[8];              // W_P..W_K
    double *current_re, *current_im;
    double *amplitude_re, *amplitude_im;
    const double *momenta;         // momentum_planes planes (form-major, 4 per form)
    const double *scale_re, *scale_im;
    const double *factor_re, *factor_im;
    const uint32_t *upload_current, *upload_amplitude, *download_amplitude, *download_current;
    uint32_t current_planes, amplitude_planes, momentum_planes, scale_count, factor_count;
    uint32_t upload_current_runs, upload_amplitude_runs, download_amplitude_runs,
        download_current_runs;
    uint32_t stride, points;
    // since the reduction: a caller may pass a struct that ends here (the fields below are 0)
    const uint32_t *zero_amplitude;  // amplitude runs set to zero on the device, not uploaded
    uint32_t zero_amplitude_runs;
    uint32_t flags;                  // PACMAN_RECURRENCE_KEEP_AMPLITUDES
} pacman_recurrence_call;

// the download_amplitude runs stay on the device (not copied back) for pacman_recurrence_v1_reduce
#define PACMAN_RECURRENCE_KEEP_AMPLITUDES 1u

// After a run with PACMAN_RECURRENCE_KEEP_AMPLITUDES: for each point i < points,
//   output[i] += weights[k] * fma(re, re, im * im)   for k = 0 .. count-1 in order,
// with (re, im) the run's amplitude plane planes[k] at point i: RustiCol's LC diagonal reduction
// (`accumulate_lc_diagonal_amplitude`), operation for operation. `output` is the caller's tile.
typedef struct {
    uint32_t abi_version;
    uint32_t struct_size;
    const uint32_t *planes;
    const double *weights;
    double *output;
    uint32_t count, points;
} pacman_recurrence_reduce;

enum {
    PACMAN_RECURRENCE_OK = 0,
    PACMAN_RECURRENCE_INVALID = 2,
    PACMAN_RECURRENCE_DEVICE_ERROR = 3,
};

extern "C" int pacman_recurrence_v1_run(const pacman_recurrence_call *call);
extern "C" int pacman_recurrence_v1_reduce(const pacman_recurrence_reduce *reduce);

// ---- device code ---------------------------------------------------------------------------

// df64: a double emulated by a pair of floats (high, low), for GPUs with slow FP64. Accurate
// double-word algorithms (Joldes, Muller, Popescu, ACM TOMS 44(2), 2017): + AccurateDWPlusDW
// (Alg. 6), x DWTimesDW3 (Alg. 12), / DWDivDW2 (Alg. 17) with DWTimesFP3 (Alg. 9). They need
// correctly rounded f32 operations that are not contracted (-fmad=false) and an explicit fma.
struct D {
    float hi, lo;
    __device__ __forceinline__ D() {}
    __device__ __forceinline__ D(float h, float l) : hi(h), lo(l) {}
    // for the constants of the formulas (0, +-1, 2): exact in f32
    __device__ __forceinline__ explicit D(double x) : hi((float)x), lo(0.0f) {}
};

__device__ __forceinline__ void two_sum(float a, float b, float &s, float &e) {
    s = __fadd_rn(a, b);
    float bb = __fsub_rn(s, a);
    e = __fadd_rn(__fsub_rn(a, __fsub_rn(s, bb)), __fsub_rn(b, bb));
}
__device__ __forceinline__ void fast_two_sum(float a, float b, float &s, float &e) {
    s = __fadd_rn(a, b);
    e = __fsub_rn(b, __fsub_rn(s, a));
}
__device__ __forceinline__ D operator+(D x, D y) {
    float sh, sl, th, tl, vh, vl;
    two_sum(x.hi, y.hi, sh, sl);
    two_sum(x.lo, y.lo, th, tl);
    fast_two_sum(sh, __fadd_rn(sl, th), vh, vl);
    D z;
    fast_two_sum(vh, __fadd_rn(tl, vl), z.hi, z.lo);
    return z;
}
__device__ __forceinline__ D operator-(D x) { return D(-x.hi, -x.lo); }
__device__ __forceinline__ D operator-(D x, D y) { return x + (-y); }
__device__ __forceinline__ D operator*(D x, D y) {
    float ch = __fmul_rn(x.hi, y.hi);
    float cl1 = __fmaf_rn(x.hi, y.hi, -ch);
    float tl0 = __fmul_rn(x.lo, y.lo);
    float tl1 = __fmaf_rn(x.hi, y.lo, tl0);
    float cl2 = __fmaf_rn(x.lo, y.hi, tl1);
    D z;
    fast_two_sum(ch, __fadd_rn(cl1, cl2), z.hi, z.lo);
    return z;
}
__device__ __forceinline__ D operator/(D x, D y) {
    float th = __fdiv_rn(x.hi, y.hi);
    float rh = __fmul_rn(y.hi, th);
    float rl1 = __fmaf_rn(y.hi, th, -rh);
    float rl3 = __fmaf_rn(y.lo, th, rl1);
    float r_hi, r_lo;
    fast_two_sum(rh, rl3, r_hi, r_lo);
    float pi_h = __fsub_rn(x.hi, r_hi);
    float delta = __fadd_rn(pi_h, __fsub_rn(x.lo, r_lo));
    D z;
    fast_two_sum(th, __fdiv_rn(delta, y.hi), z.hi, z.lo);
    return z;
}

// complex numbers over double (`C`, RustiCol's arithmetic) or df64
template <class R> struct Cx {
    R re, im;
};
typedef Cx<double> C;

template <class R> __device__ __forceinline__ Cx<R> cx(R re, R im) { Cx<R> c; c.re = re; c.im = im; return c; }
template <class R> __device__ __forceinline__ Cx<R> add(Cx<R> a, Cx<R> b) { return cx(a.re + b.re, a.im + b.im); }
template <class R> __device__ __forceinline__ Cx<R> sub(Cx<R> a, Cx<R> b) { return cx(a.re - b.re, a.im - b.im); }
template <class R> __device__ __forceinline__ Cx<R> neg(Cx<R> a) { return cx(-a.re, -a.im); }
template <class R> __device__ __forceinline__ Cx<R> mul(Cx<R> a, Cx<R> b) {
    return cx(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
}
template <class R> __device__ __forceinline__ Cx<R> mul_i(Cx<R> a) { return cx(-a.im, a.re); }
template <class R> __device__ __forceinline__ Cx<R> mul_real(Cx<R> a, R s) { return cx(a.re * s, a.im * s); }

// a value of the arena (planes of doubles; a df64 arena holds float2 in the same 8 bytes)
template <class R> __device__ __forceinline__ R ldr(const double *base, size_t at);
template <> __device__ __forceinline__ double ldr<double>(const double *base, size_t at) { return base[at]; }
template <> __device__ __forceinline__ D ldr<D>(const double *base, size_t at) {
    float2 v = reinterpret_cast<const float2 *>(base)[at];
    return D(v.x, v.y);
}
__device__ __forceinline__ void str(double *base, size_t at, double v) { base[at] = v; }
__device__ __forceinline__ void str(double *base, size_t at, D v) {
    reinterpret_cast<float2 *>(base)[at] = make_float2(v.hi, v.lo);
}

struct Arena {
    double *cre, *cim, *are, *aim;
    const double *mom;
    const double *sre, *sim, *fre, *fim;
    size_t ld;
};

template <class R>
__device__ __forceinline__ R reciprocal_minkowski_norm(R p0, R p1, R p2, R p3) {
    return R(1.0) / (p0 * p0 - p1 * p1 - p2 * p2 - p3 * p3);
}

template <class R> __device__ __forceinline__ Cx<R> minkowski_dot(const Cx<R> *l, const Cx<R> *r) {
    return sub(sub(sub(mul(l[0], r[0]), mul(l[1], r[1])), mul(l[2], r[2])), mul(l[3], r[3]));
}

template <class R> __device__ __forceinline__ Cx<R> minkowski_real_dot(const Cx<R> *l, const R *q) {
    return sub(sub(sub(mul_real(l[0], q[0]), mul_real(l[1], q[1])), mul_real(l[2], q[2])),
               mul_real(l[3], q[3]));
}

// ---- the executors' arithmetic, in f64 and df64: operands in, values out ----

// components read from parent0 and parent1 and written, per contribution op (2..6)
template <int OP> struct Shape {
    static constexpr int c0 = OP == 4 ? 6 : OP == 5 || OP == 6 ? 4 : 2;
    static constexpr int c1 = 4;
    static constexpr int cd = OP == 4 || OP == 6 ? 4 : OP == 5 ? 6 : 2;
};

__host__ __device__ __forceinline__ void contribution_shape(int op, int *c0, int *c1, int *cd) {
    *c0 = op == 4 ? 6 : op == 5 || op == 6 ? 4 : 2;
    *c1 = 4;
    *cd = op == 4 || op == 6 ? 4 : op == 5 ? 6 : 2;
}

// contribution values before the scale (`o`, cd of them); q0, q1: the momentum forms (op 6)
template <class R>
__device__ __forceinline__ void contribution_values(int op, const Cx<R> *l, const Cx<R> *r, const R *q0,
                                                    const R *q1, Cx<R> *o) {
    typedef Cx<R> C;
    if (op == 2) {
        o[0] = add(add(sub(neg(mul_i(mul(l[0], r[3]))), mul_i(mul(l[1], r[1]))), mul(l[1], r[2])),
                   mul_i(mul(l[0], r[0])));
        o[1] = add(add(sub(neg(mul(l[0], r[2])), mul_i(mul(l[0], r[1]))), mul_i(mul(l[1], r[0]))),
                   mul_i(mul(l[1], r[3])));
    } else if (op == 3) {
        o[0] = add(add(add(neg(mul(l[1], r[2])), mul_i(mul(l[0], r[0]))), mul_i(mul(l[0], r[3]))),
                   mul_i(mul(l[1], r[1])));
        o[1] = add(add(add(neg(mul_i(mul(l[1], r[3]))), mul(l[0], r[2])), mul_i(mul(l[0], r[1]))),
                   mul_i(mul(l[1], r[0])));
    } else if (op == 4) {
        o[0] = add(add(mul(l[0], r[1]), mul(l[1], r[2])), mul(l[2], r[3]));
        o[1] = add(add(mul(l[0], r[0]), mul(l[3], r[2])), mul(l[4], r[3]));
        o[2] = add(sub(mul(l[1], r[0]), mul(l[3], r[1])), mul(l[5], r[3]));
        o[3] = sub(sub(mul(l[2], r[0]), mul(l[4], r[1])), mul(l[5], r[2]));
    } else if (op == 5) {
        o[0] = sub(mul(l[0], r[1]), mul(l[1], r[0]));
        o[1] = sub(mul(l[0], r[2]), mul(l[2], r[0]));
        o[2] = sub(mul(l[0], r[3]), mul(l[3], r[0]));
        o[3] = sub(mul(l[1], r[2]), mul(l[2], r[1]));
        o[4] = sub(mul(l[1], r[3]), mul(l[3], r[1]));
        o[5] = sub(mul(l[2], r[3]), mul(l[3], r[2]));
    } else {  // op == 6
        C vector_dot = minkowski_dot(l, r);
        C left_dot_parent1 = minkowski_real_dot(l, q1);
        C right_dot_parent0 = minkowski_real_dot(r, q0);
        for (int k = 0; k < 4; k++) {
            C momentum_term = mul_real(vector_dot, q0[k] - q1[k]);
            C current_term =
                mul_real(sub(mul(left_dot_parent1, r[k]), mul(right_dot_parent0, l[k])), R(2.0));
            o[k] = add(momentum_term, current_term);
        }
    }
}

// propagator values (ops 7..9): the 2 (Weyl) or 4 (vector) components `c` become `o`, scaled
template <class R>
__device__ __forceinline__ int propagator_values(int op, const Cx<R> *c, const R *p, Cx<R> row_factor,
                                                 Cx<R> *o) {
    typedef Cx<R> C;
    if (op == 9) {
        C scale = mul_real(mul(row_factor, cx(R(0.0), R(-1.0))),
                           reciprocal_minkowski_norm(p[0], p[1], p[2], p[3]));
        for (int k = 0; k < 4; k++) o[k] = mul(c[k], scale);
        return 4;
    }
    C scale = mul_real(row_factor, reciprocal_minkowski_norm(p[0], p[1], p[2], p[3]));
    C o0, o1;
    if (op == 7) {
        o0 = add(add(sub(neg(mul_i(mul_real(c[0], p[3]))), mul_i(mul_real(c[1], p[1]))),
                     mul_real(c[1], p[2])),
                 mul_i(mul_real(c[0], p[0])));
        o1 = add(add(sub(neg(mul_real(c[0], p[2])), mul_i(mul_real(c[0], p[1]))),
                     mul_i(mul_real(c[1], p[0]))),
                 mul_i(mul_real(c[1], p[3])));
    } else {
        o0 = add(add(add(neg(mul_real(c[1], p[2])), mul_i(mul_real(c[0], p[0]))),
                     mul_i(mul_real(c[0], p[3]))),
                 mul_i(mul_real(c[1], p[1])));
        o1 = add(add(add(neg(mul_i(mul_real(c[1], p[3]))), mul_real(c[0], p[2])),
                     mul_i(mul_real(c[0], p[1]))),
                 mul_i(mul_real(c[1], p[0])));
    }
    o[0] = mul(o0, scale);
    o[1] = mul(o1, scale);
    return 2;
}

// `execute_identity_finalization_rows`: factor × value
template <class R> __device__ __forceinline__ Cx<R> identity_value(R fr, R fi, Cx<R> v) {
    return cx(fr * v.re - fi * v.im, fr * v.im + fi * v.re);
}

// `execute_closure_reduce_rows`, one component: the amplitude increment
template <class R>
__device__ __forceinline__ Cx<R> closure_increment(Cx<R> l, Cx<R> r, R cr, R ci, R rfr, R rfi) {
    R scale_re = cr * rfr - ci * rfi;
    R scale_im = cr * rfi + ci * rfr;
    R product_re = l.re * r.re - l.im * r.im;
    R product_im = l.re * r.im + l.im * r.re;
    return cx(product_re * scale_re - product_im * scale_im,
              product_re * scale_im + product_im * scale_re);
}

// ---- programs: the words rewritten by the host (`prepare` below), 32 bytes per instruction ----
//
// w[0] = kind | op << 8 | flags << 16; the fields follow in w[1..7]. The program is the plan's
// rows with the contribution runs stably sorted by destination (each destination keeps its order
// of rows, so its sum is unchanged) and the destination of a run of rows on one destination kept
// in registers (F_FIRST loads or starts it, F_LAST stores it).

enum { K_CLEAR = 1, K_CONTRIB, K_PROP, K_IDENTITY, K_CLOSURE };
enum { F_INIT = 1, F_FIRST = 2, F_LAST = 4 };

struct IOp {
    uint32_t w[8];
};

__device__ __forceinline__ IOp load_op(const IOp *ops, uint32_t n) {
    const uint4 *q = reinterpret_cast<const uint4 *>(ops + n);
    uint4 x = __ldg(q), y = __ldg(q + 1);
    IOp o;
    o.w[0] = x.x; o.w[1] = x.y; o.w[2] = x.z; o.w[3] = x.w;
    o.w[4] = y.x; o.w[5] = y.y; o.w[6] = y.z; o.w[7] = y.w;
    return o;
}

// contributions w[1] scale slot, w[2] p0, w[3] p1, w[4] dst, w[5] f0, w[6] f1, w[7] factor;
// clear w[1] base, w[2] count; propagator w[1] base, w[2] form, w[3] factor; identity w[1] base,
// w[2] count, w[3] factor; closure w[1] p0, w[2] p1, w[3] amplitude, w[4] factor, w[5] first,
// w[6] count.
template <class R>
__device__ __forceinline__ Cx<R> ld_cr(const Arena &a, uint32_t plane, size_t i) {
    return cx(ldr<R>(a.cre, plane * a.ld + i), ldr<R>(a.cim, plane * a.ld + i));
}

template <class R, int OP>
__device__ __forceinline__ void contribution(const Arena &a, const IOp &o, int flags, Cx<R> *acc,
                                                 size_t i) {
    typedef Cx<R> C;
    constexpr int op = OP, c0 = Shape<OP>::c0, c1 = Shape<OP>::c1, cd = Shape<OP>::cd;
    uint32_t p0 = o.w[2], p1 = o.w[3], dst = o.w[4];
    C scale = mul(cx(ldr<R>(a.sre, o.w[1]), ldr<R>(a.sim, o.w[1])), cx(ldr<R>(a.fre, o.w[7]), ldr<R>(a.fim, o.w[7])));
    C l[6], r[4], v[6];
    R q0[4], q1[4];
#pragma unroll
    for (int k = 0; k < c0; k++) l[k] = ld_cr<R>(a, p0 + k, i);
#pragma unroll
    for (int k = 0; k < c1; k++) r[k] = ld_cr<R>(a, p1 + k, i);
    if (op == 6)
#pragma unroll
        for (int k = 0; k < 4; k++) {
            q0[k] = ldr<R>(a.mom, (o.w[5] * 4 + k) * a.ld + i);
            q1[k] = ldr<R>(a.mom, (o.w[6] * 4 + k) * a.ld + i);
        }
    contribution_values(op, l, r, q0, q1, v);
#pragma unroll
    for (int k = 0; k < cd; k++) {
        C c = mul(v[k], scale);
        if (!(flags & F_FIRST)) acc[k] = add(acc[k], c);
        else if (flags & F_INIT) acc[k] = c;
        else acc[k] = add(ld_cr<R>(a, dst + k, i), c);
    }
    if (flags & F_LAST)
#pragma unroll
        for (int k = 0; k < cd; k++) {
            str(a.cre, (dst + k) * a.ld + i, acc[k].re);
            str(a.cim, (dst + k) * a.ld + i, acc[k].im);
        }
}

template <class R, int OP>
__device__ __forceinline__ void propagator(const Arena &a, const IOp &o, size_t i) {
    typedef Cx<R> C;
    constexpr int m = OP == 9 ? 4 : 2;
    uint32_t base = o.w[1];
    R p[4];
#pragma unroll
    for (int k = 0; k < 4; k++) p[k] = ldr<R>(a.mom, (o.w[2] * 4 + k) * a.ld + i);
    C c[4], v[4];
#pragma unroll
    for (int k = 0; k < m; k++) c[k] = ld_cr<R>(a, base + k, i);
    propagator_values(OP, c, p, cx(ldr<R>(a.fre, o.w[3]), ldr<R>(a.fim, o.w[3])), v);
#pragma unroll
    for (int k = 0; k < m; k++) {
        str(a.cre, (base + k) * a.ld + i, v[k].re);
        str(a.cim, (base + k) * a.ld + i, v[k].im);
    }
}

// R = double: RustiCol's arithmetic; R = D: df64 on a df64 arena (all planes, momenta, scales and
// factors hold float2 (high, low) in place of their doubles; see `to_df64`, `to_f64_planes`).
template <class R>
__global__ void run_program(const IOp *__restrict__ ops, uint32_t count, Arena a, uint32_t points) {
    typedef Cx<R> C;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= points) return;
    C acc[6];
    for (uint32_t n = 0; n < count; n++) {
        IOp o = load_op(ops, n);
        int kind = o.w[0] & 0xff, op = (o.w[0] >> 8) & 0xff, flags = o.w[0] >> 16;
        if (kind == K_CONTRIB) {
            switch (op) {
            case 2: contribution<R, 2>(a, o, flags, acc, i); break;
            case 3: contribution<R, 3>(a, o, flags, acc, i); break;
            case 4: contribution<R, 4>(a, o, flags, acc, i); break;
            case 5: contribution<R, 5>(a, o, flags, acc, i); break;
            default: contribution<R, 6>(a, o, flags, acc, i); break;
            }
        } else if (kind == K_CLEAR) {
            for (uint32_t k = 0; k < o.w[2]; k++) {
                str(a.cre, (o.w[1] + k) * a.ld + i, R(0.0));
                str(a.cim, (o.w[1] + k) * a.ld + i, R(0.0));
            }
        } else if (kind == K_PROP) {
            if (op == 7) propagator<R, 7>(a, o, i);
            else if (op == 8) propagator<R, 8>(a, o, i);
            else propagator<R, 9>(a, o, i);
        } else if (kind == K_IDENTITY) {
            R fr = ldr<R>(a.fre, o.w[3]), fi = ldr<R>(a.fim, o.w[3]);
            for (uint32_t k = 0; k < o.w[2]; k++) {
                size_t at = (o.w[1] + k) * a.ld + i;
                C v = identity_value(fr, fi, cx(ldr<R>(a.cre, at), ldr<R>(a.cim, at)));
                str(a.cre, at, v.re);
                str(a.cim, at, v.im);
            }
        } else if (kind == K_CLOSURE) {
            R rfr = ldr<R>(a.fre, o.w[4]), rfi = ldr<R>(a.fim, o.w[4]);
            size_t at = o.w[3] * a.ld + i;
            R dr = ldr<R>(a.are, at), di = ldr<R>(a.aim, at);
            for (uint32_t k = 0; k < o.w[6]; k++) {
                C v = closure_increment(ld_cr<R>(a, o.w[1] + k, i), ld_cr<R>(a, o.w[2] + k, i),
                                        ldr<R>(a.fre, o.w[5] + k), ldr<R>(a.fim, o.w[5] + k), rfr, rfi);
                dr = dr + v.re;
                di = di + v.im;
            }
            str(a.are, at, dr);
            str(a.aim, at, di);
        }
    }
}

// in-place conversion of `count` doubles to df64 (float2 in the same 8 bytes)
__global__ void to_df64(double *p, size_t count) {
    for (size_t k = blockIdx.x * (size_t)blockDim.x + threadIdx.x; k < count; k += (size_t)gridDim.x * blockDim.x) {
        double x = p[k];
        float hi = __double2float_rn(x);
        float lo = __double2float_rn(x - (double)hi);
        reinterpret_cast<float2 *>(p)[k] = make_float2(hi, lo);
    }
}

__global__ void to_df64_planes(double *p, size_t planes, size_t stride, size_t points) {
    for (size_t k = blockIdx.x * (size_t)blockDim.x + threadIdx.x; k < planes * points; k += (size_t)gridDim.x * blockDim.x) {
        size_t at = k / points * stride + k % points;
        double x = p[at];
        float hi = __double2float_rn(x);
        float lo = __double2float_rn(x - (double)hi);
        reinterpret_cast<float2 *>(p)[at] = make_float2(hi, lo);
    }
}

__global__ void to_f64_planes(double *p, size_t planes, size_t stride, size_t points) {
    for (size_t k = blockIdx.x * (size_t)blockDim.x + threadIdx.x; k < planes * points; k += (size_t)gridDim.x * blockDim.x) {
        size_t at = k / points * stride + k % points;
        float2 v = reinterpret_cast<float2 *>(p)[at];
        p[at] = (double)v.x + (double)v.y;
    }
}

// ---- host side -----------------------------------------------------------------------------

namespace {

struct Device {
    double *cre = nullptr, *cim = nullptr, *are = nullptr, *aim = nullptr, *mom = nullptr;
    double *small = nullptr;  // scales, then factors
    size_t cre_capacity = 0, cim_capacity = 0, are_capacity = 0, aim_capacity = 0,
           momentum_capacity = 0, small_capacity = 0;
    // the last run with PACMAN_RECURRENCE_KEEP_AMPLITUDES (for a reduction)
    bool kept = false, kept_df64 = false;
    size_t kept_stride = 0, kept_points = 0, kept_planes = 0;
    // reduction buffers
    uint32_t *rplanes = nullptr;
    double *rweights = nullptr, *routput = nullptr;
    size_t rcount_capacity = 0, routput_capacity = 0;
    // statistics (PACMAN_RUNNER_STATS=1)
    uint64_t calls = 0, points = 0, copies = 0, bytes = 0, reductions = 0;
    double seconds = 0;
    bool stats = false;
};

Device device;

// ---- programs (host) ---------------------------------------------------------------------------

struct Row {
    int op;           // the word's op (1..11)
    uint32_t f[8];    // its fields in word order
};

std::vector<Row> decode(const uint64_t *words, uint32_t count, const uint8_t *width) {
    std::vector<Row> rows;
    rows.reserve(count);
    for (uint32_t n = 0; n < count; n++) {
        uint64_t w = words[n];
        auto take = [&](int bits) {
            uint32_t v = (uint32_t)(w & ((1ull << bits) - 1));
            w >>= bits;
            return v;
        };
        Row r{};
        r.op = (int)take(4);
        const int P = width[W_P], A = width[W_A], F = width[W_F], X = width[W_X], S = width[W_S],
                  K = width[W_K];
        int fields[8] = {0}, nf = 0;
        if (r.op == 1) { fields[0] = P; fields[1] = P; nf = 2; }
        else if (r.op >= 2 && r.op <= 6) {
            int f[8] = {1, S, P, P, P, F, F, X};
            for (int k = 0; k < 8; k++) fields[k] = f[k];
            nf = 8;
        } else if (r.op >= 7 && r.op <= 9) { fields[0] = P; fields[1] = F; fields[2] = X; nf = 3; }
        else if (r.op == 10) { fields[0] = P; fields[1] = K; fields[2] = X; nf = 3; }
        else if (r.op == 11) {
            int f[6] = {P, P, A, X, X, K};
            for (int k = 0; k < 6; k++) fields[k] = f[k];
            nf = 6;
        }
        for (int k = 0; k < nf; k++) r.f[k] = take(fields[k]);
        rows.push_back(r);
    }
    return rows;
}

// The contribution runs stably sorted by destination, where that is safe: no row of the run reads
// a component the run writes, and the destinations of the run do not overlap. Each destination
// keeps its own order of rows, so its value is unchanged bit for bit.
void sort_contribution_runs(std::vector<Row> &rows) {
    size_t n = 0;
    while (n < rows.size()) {
        if (rows[n].op < 2 || rows[n].op > 6) { n++; continue; }
        size_t end = n;
        while (end < rows.size() && rows[end].op >= 2 && rows[end].op <= 6) end++;
        std::unordered_map<uint32_t, uint32_t> dst_base;  // component -> destination base
        bool safe = true;
        for (size_t k = n; k < end && safe; k++) {
            int c0, c1, cd;
            contribution_shape(rows[k].op, &c0, &c1, &cd);
            for (int q = 0; q < cd; q++) {
                auto it = dst_base.emplace(rows[k].f[4] + q, rows[k].f[4]);
                if (it.first->second != rows[k].f[4]) safe = false;  // overlapping destinations
            }
        }
        for (size_t k = n; k < end && safe; k++) {
            int c0, c1, cd;
            contribution_shape(rows[k].op, &c0, &c1, &cd);
            for (int q = 0; q < c0 && safe; q++) safe = !dst_base.count(rows[k].f[2] + q);
            for (int q = 0; q < c1 && safe; q++) safe = !dst_base.count(rows[k].f[3] + q);
        }
        if (safe)
            std::stable_sort(rows.begin() + n, rows.begin() + end,
                             [](const Row &x, const Row &y) { return x.f[4] < y.f[4]; });
        n = end;
    }
}

IOp make_op(int kind, int op, int flags) {
    IOp o{};
    o.w[0] = (uint32_t)kind | (uint32_t)op << 8 | (uint32_t)flags << 16;
    return o;
}

std::vector<IOp> program(const std::vector<Row> &rows) {
    std::vector<IOp> ops;
    for (size_t n = 0; n < rows.size(); n++) {
        const Row &r = rows[n];
        if (r.op >= 2 && r.op <= 6) {
            bool first = n == 0 || rows[n - 1].op < 2 || rows[n - 1].op > 6 || rows[n - 1].f[4] != r.f[4];
            bool last = n + 1 == rows.size() || rows[n + 1].op < 2 || rows[n + 1].op > 6 ||
                        rows[n + 1].f[4] != r.f[4];
            IOp o = make_op(K_CONTRIB, r.op, (r.f[0] ? F_INIT : 0) | (first ? F_FIRST : 0) | (last ? F_LAST : 0));
            for (int k = 1; k < 8; k++) o.w[k] = r.f[k];
            ops.push_back(o);
        } else {
            int kind = r.op == 1 ? K_CLEAR : r.op <= 9 ? K_PROP : r.op == 10 ? K_IDENTITY : K_CLOSURE;
            IOp o = make_op(kind, r.op, 0);
            for (int k = 0; k < 7; k++) o.w[k + 1] = r.f[k];
            ops.push_back(o);
        }
    }
    return ops;
}

// One prepared program: its instructions on the device.
struct Prepared {
    uint32_t count;
    IOp *ops;
};

bool failed(cudaError_t e, const char *step) {
    if (e == cudaSuccess) return false;
    fprintf(stderr, "pacman recurrence runner: %s: %s\n", step, cudaGetErrorString(e));
    return true;
}

bool grow(double **p, size_t *capacity, size_t doubles, const char *what) {
    if (doubles <= *capacity) return true;
    if (*p) cudaFree(*p);
    *p = nullptr;
    *capacity = 0;
    if (failed(cudaMalloc((void **)p, doubles * sizeof(double)), what)) return false;
    *capacity = doubles;
    return true;
}

// copy the plane runs of re and im between host and device
// `n` planes of `stride` doubles from `src` to `dst`: only their first `points` values (the tile
// can be much longer than the call's points)
bool copy_planes(void *dst, const void *src, size_t n, size_t stride, size_t points, cudaMemcpyKind kind) {
    size_t pitch = stride * sizeof(double);
    if (failed(cudaMemcpy2D(dst, pitch, src, pitch, points * sizeof(double), n, kind), "copy")) return false;
    device.copies++;
    device.bytes += n * points * sizeof(double);
    return true;
}

// copy the plane runs of re and im between host and device
bool copy_runs(double *dre, double *dim, double *hre, double *him, const uint32_t *runs,
               uint32_t count, uint32_t planes, size_t stride, size_t points, bool to_device) {
    for (uint32_t r = 0; r < count; r++) {
        uint32_t first = runs[2 * r], n = runs[2 * r + 1];
        if (first > planes || n > planes - first) return false;
        size_t offset = (size_t)first * stride;
        cudaMemcpyKind kind = to_device ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost;
        if (to_device ? !copy_planes(dre + offset, hre + offset, n, stride, points, kind) ||
                            !copy_planes(dim + offset, him + offset, n, stride, points, kind)
                      : !copy_planes(hre + offset, dre + offset, n, stride, points, kind) ||
                            !copy_planes(him + offset, dim + offset, n, stride, points, kind))
            return false;
    }
    return true;
}

// the kernel: PACMAN_RECURRENCE_KERNEL=df64 (the default: f32-pair arithmetic on a df64 copy of
// the arena, for GPUs whose FP64 is slow) | f64 (RustiCol's arithmetic bit for bit),
// PACMAN_RECURRENCE_BLOCK (threads per block, 128), PACMAN_RECURRENCE_SORT=0 (do not sort the
// contribution rows by destination)
struct Options {
    bool df64 = true;
    int block = 128;
};

bool sort_option() {
    const char *v = getenv("PACMAN_RECURRENCE_SORT");
    return !(v && !strcmp(v, "0"));
}

Options options() {
    static Options o;
    static bool read = false;
    if (read) return o;
    read = true;
    const char *k = getenv("PACMAN_RECURRENCE_KERNEL");
    if (k && !strcmp(k, "f64")) o.df64 = false;
    else if (k && *k && strcmp(k, "df64"))
        fprintf(stderr, "pacman recurrence runner: unknown PACMAN_RECURRENCE_KERNEL=%s, using df64\n", k);
    if (const char *v = getenv("PACMAN_RECURRENCE_BLOCK")) o.block = std::max(32, std::min(1024, atoi(v)));
    return o;
}

// the program of `call`'s words, built once per words key
std::unordered_map<uint64_t, Prepared> prepared_programs;

const Prepared *prepare(const pacman_recurrence_call *call) {
    static Prepared scratch{0, nullptr};
    if (call->words_key) {
        auto it = prepared_programs.find(call->words_key);
        if (it != prepared_programs.end()) return &it->second;
    }
    std::vector<Row> rows = decode(call->words, call->word_count, call->width);
    if (sort_option()) sort_contribution_runs(rows);
    std::vector<IOp> ops = program(rows);
    Prepared p{(uint32_t)ops.size(), nullptr};
    if (failed(cudaMalloc((void **)&p.ops, std::max<size_t>(1, ops.size()) * sizeof(IOp)), "program") ||
        failed(cudaMemcpy(p.ops, ops.data(), ops.size() * sizeof(IOp), cudaMemcpyHostToDevice), "program"))
        return nullptr;
    if (call->words_key) return &(prepared_programs[call->words_key] = p);
    if (scratch.ops) cudaFree(scratch.ops);
    scratch = p;
    return &scratch;
}

bool launch(const Prepared &p, const Arena &a, uint32_t points) {
    Options o = options();
    unsigned block = (unsigned)o.block, grid = (points + block - 1) / block;
    if (o.df64) run_program<D><<<grid, block>>>(p.ops, p.count, a, points);
    else run_program<double><<<grid, block>>>(p.ops, p.count, a, points);
    return !failed(cudaGetLastError(), "launch");
}

// in-place f64 <-> df64 conversions of `doubles` values, and of plane runs of re and im
bool convert(void (*f)(double *, size_t), double *p, size_t doubles) {
    if (doubles == 0) return true;
    unsigned grid = (unsigned)std::min<size_t>(4096, (doubles + 255) / 256);
    f<<<grid, 256>>>(p, doubles);
    return !failed(cudaGetLastError(), "conversion");
}

// the first `points` values of `planes` planes of `stride` doubles
bool convert_planes(bool to_df, double *p, size_t planes, size_t stride, size_t points) {
    size_t count = planes * points;
    if (count == 0) return true;
    unsigned grid = (unsigned)std::min<size_t>(4096, (count + 255) / 256);
    if (to_df) to_df64_planes<<<grid, 256>>>(p, planes, stride, points);
    else to_f64_planes<<<grid, 256>>>(p, planes, stride, points);
    return !failed(cudaGetLastError(), "conversion");
}

bool convert_runs(bool to_df, double *re, double *im, const uint32_t *runs, uint32_t count,
                  size_t stride, size_t points) {
    for (uint32_t r = 0; r < count; r++) {
        size_t offset = (size_t)runs[2 * r] * stride, n = runs[2 * r + 1];
        if (!convert_planes(to_df, re + offset, n, stride, points) ||
            !convert_planes(to_df, im + offset, n, stride, points))
            return false;
    }
    return true;
}

void print_stats() {
    if (!device.stats || device.calls == 0) return;
    fprintf(stderr,
            "pacman recurrence runner: %llu calls, %llu points, %llu reductions, %llu copies (%.1f MB), "
            "%.3f s (%.2f us/point)\n",
            (unsigned long long)device.calls, (unsigned long long)device.points,
            (unsigned long long)device.reductions, (unsigned long long)device.copies, device.bytes / 1e6,
            device.seconds,
            device.points ? device.seconds * 1e6 / device.points : 0.0);
}

double now() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

}  // namespace

static void initialize() {
    static bool initialized = false;
    if (!initialized) {
        initialized = true;
        const char *s = getenv("PACMAN_RUNNER_STATS");
        device.stats = s && *s && strcmp(s, "0") != 0;
        atexit(print_stats);
    }
}

extern "C" int pacman_recurrence_v1_run(const pacman_recurrence_call *in) {
    initialize();
    const size_t v1_size = offsetof(pacman_recurrence_call, zero_amplitude);
    if (!in || in->abi_version != PACMAN_RECURRENCE_ABI ||
        (in->struct_size != sizeof(pacman_recurrence_call) && in->struct_size != v1_size))
        return PACMAN_RECURRENCE_INVALID;
    pacman_recurrence_call full;
    memset(&full, 0, sizeof full);
    memcpy(&full, in, in->struct_size);
    const pacman_recurrence_call *call = &full;
    if (call->points == 0 || call->points > call->stride || !call->words ||
        (call->zero_amplitude_runs && !call->zero_amplitude))
        return PACMAN_RECURRENCE_INVALID;
    device.kept = false;
    bool keep = call->flags & PACMAN_RECURRENCE_KEEP_AMPLITUDES;
    double start = device.stats ? now() : 0;
    size_t stride = call->stride;
    size_t current_doubles = (size_t)call->current_planes * stride;
    size_t amplitude_doubles = (size_t)call->amplitude_planes * stride;
    // a grown arena loses its contents: the caller uploads what the words read
    if (!grow(&device.cre, &device.cre_capacity, current_doubles, "current arena") ||
        !grow(&device.cim, &device.cim_capacity, current_doubles, "current arena") ||
        !grow(&device.are, &device.are_capacity, amplitude_doubles, "amplitudes") ||
        !grow(&device.aim, &device.aim_capacity, amplitude_doubles, "amplitudes"))
        return PACMAN_RECURRENCE_DEVICE_ERROR;
    size_t momentum_doubles = (size_t)call->momentum_planes * stride;
    size_t small = 2 * ((size_t)call->scale_count + call->factor_count);
    if (!grow(&device.mom, &device.momentum_capacity, momentum_doubles ? momentum_doubles : 1,
              "momenta") ||
        !grow(&device.small, &device.small_capacity, small ? small : 1, "factors"))
        return PACMAN_RECURRENCE_DEVICE_ERROR;

    // inputs: momenta, scales and factors, the current and amplitude runs named
    double *dscale_re = device.small, *dscale_im = dscale_re + call->scale_count;
    double *dfactor_re = dscale_im + call->scale_count, *dfactor_im = dfactor_re + call->factor_count;
    if ((momentum_doubles &&
         !copy_planes(device.mom, call->momenta, call->momentum_planes, stride, call->points,
                      cudaMemcpyHostToDevice)) ||
        (call->scale_count &&
         (failed(cudaMemcpy(dscale_re, call->scale_re, call->scale_count * sizeof(double),
                            cudaMemcpyHostToDevice), "scales") ||
          failed(cudaMemcpy(dscale_im, call->scale_im, call->scale_count * sizeof(double),
                            cudaMemcpyHostToDevice), "scales"))) ||
        (call->factor_count &&
         (failed(cudaMemcpy(dfactor_re, call->factor_re, call->factor_count * sizeof(double),
                            cudaMemcpyHostToDevice), "factors") ||
          failed(cudaMemcpy(dfactor_im, call->factor_im, call->factor_count * sizeof(double),
                            cudaMemcpyHostToDevice), "factors"))))
        return PACMAN_RECURRENCE_DEVICE_ERROR;
    if (!copy_runs(device.cre, device.cim, call->current_re, call->current_im, call->upload_current,
                   call->upload_current_runs, call->current_planes, stride, call->points, true) ||
        !copy_runs(device.are, device.aim, call->amplitude_re, call->amplitude_im,
                   call->upload_amplitude, call->upload_amplitude_runs, call->amplitude_planes,
                   stride, call->points, true))
        return PACMAN_RECURRENCE_DEVICE_ERROR;
    // amplitudes the caller has cleared: zero on the device (zero is zero in df64 too)
    for (uint32_t r = 0; r < call->zero_amplitude_runs; r++) {
        uint32_t first = call->zero_amplitude[2 * r], n = call->zero_amplitude[2 * r + 1];
        if (first > call->amplitude_planes || n > call->amplitude_planes - first)
            return PACMAN_RECURRENCE_INVALID;
        size_t pitch = stride * sizeof(double), width = (size_t)call->points * sizeof(double);
        if (failed(cudaMemset2D(device.are + (size_t)first * stride, pitch, 0, width, n), "zero") ||
            failed(cudaMemset2D(device.aim + (size_t)first * stride, pitch, 0, width, n), "zero"))
            return PACMAN_RECURRENCE_DEVICE_ERROR;
    }

    Arena a;
    a.cre = device.cre;
    a.cim = device.cim;
    a.are = device.are;
    a.aim = device.aim;
    a.mom = device.mom;
    a.sre = dscale_re;
    a.sim = dscale_im;
    a.fre = dfactor_re;
    a.fim = dfactor_im;
    a.ld = stride;
    // df64: the uploaded planes, the momenta and the constants become df64 in place; the device
    // arena stays df64 between calls (planes copied back are converted to f64 and back again)
    bool df64 = options().df64;
    if (df64 && (!convert_runs(true, device.cre, device.cim, call->upload_current,
                               call->upload_current_runs, stride, call->points) ||
                 !convert_runs(true, device.are, device.aim, call->upload_amplitude,
                               call->upload_amplitude_runs, stride, call->points) ||
                 !convert_planes(true, device.mom, call->momentum_planes, stride, call->points) || !convert(to_df64, device.small, small)))
        return PACMAN_RECURRENCE_DEVICE_ERROR;
    const Prepared *prepared = prepare(call);
    if (!prepared) return PACMAN_RECURRENCE_DEVICE_ERROR;
    if (!launch(*prepared, a, call->points))
        return PACMAN_RECURRENCE_DEVICE_ERROR;
    uint32_t amplitude_runs = keep ? 0 : call->download_amplitude_runs;
    if (df64 && (!convert_runs(false, device.are, device.aim, call->download_amplitude,
                               amplitude_runs, stride, call->points) ||
                 !convert_runs(false, device.cre, device.cim, call->download_current,
                               call->download_current_runs, stride, call->points)))
        return PACMAN_RECURRENCE_DEVICE_ERROR;
    if (failed(cudaDeviceSynchronize(), "kernel")) return PACMAN_RECURRENCE_DEVICE_ERROR;

    if (!copy_runs(device.are, device.aim, call->amplitude_re, call->amplitude_im,
                   call->download_amplitude, amplitude_runs,
                   call->amplitude_planes, stride, call->points, false) ||
        !copy_runs(device.cre, device.cim, call->current_re, call->current_im,
                   call->download_current, call->download_current_runs, call->current_planes,
                   stride, call->points, false))
        return PACMAN_RECURRENCE_DEVICE_ERROR;
    if (df64 && (!convert_runs(true, device.are, device.aim, call->download_amplitude,
                               amplitude_runs, stride, call->points) ||
                 !convert_runs(true, device.cre, device.cim, call->download_current,
                               call->download_current_runs, stride, call->points)))
        return PACMAN_RECURRENCE_DEVICE_ERROR;
    if (keep) {
        device.kept = true;
        device.kept_df64 = df64;
        device.kept_stride = stride;
        device.kept_points = call->points;
        device.kept_planes = call->amplitude_planes;
    }
    device.calls++;
    device.points += call->points;
    if (device.stats) device.seconds += now() - start;
    return PACMAN_RECURRENCE_OK;
}

// ---- the LC diagonal reduction on the device --------------------------------------------------

// a df64 value as the double it stands for: as `to_f64_planes` converts it, so the reduction of a df64 run
// equals the host's reduction of its copied-back amplitudes bit for bit
__device__ __forceinline__ double as_double(double x) { return x; }
__device__ __forceinline__ double as_double(D x) { return __dadd_rn((double)x.hi, (double)x.lo); }

template <class R>
__global__ void reduce_kernel(const double *are, const double *aim, size_t ld, const uint32_t *planes,
                              const double *weights, uint32_t count, double *output, uint32_t points) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= points) return;
    double acc = output[i];
    for (uint32_t k = 0; k < count; k++) {
        size_t at = (size_t)__ldg(planes + k) * ld + i;
        double re = as_double(ldr<R>(are, at)), im = as_double(ldr<R>(aim, at));
        // weight * real.mul_add(real, imaginary * imaginary), then +=
        acc = __dadd_rn(acc, __dmul_rn(__ldg(weights + k), __fma_rn(re, re, __dmul_rn(im, im))));
    }
    output[i] = acc;
}

extern "C" int pacman_recurrence_v1_reduce(const pacman_recurrence_reduce *r) {
    initialize();
    if (!r || r->abi_version != PACMAN_RECURRENCE_ABI || r->struct_size != sizeof(pacman_recurrence_reduce) ||
        (r->count && (!r->planes || !r->weights)) || !r->output)
        return PACMAN_RECURRENCE_INVALID;
    if (!device.kept || r->points != device.kept_points) return PACMAN_RECURRENCE_INVALID;
    for (uint32_t k = 0; k < r->count; k++)
        if (r->planes[k] >= device.kept_planes) return PACMAN_RECURRENCE_INVALID;
    double start = device.stats ? now() : 0;
    if (r->count > device.rcount_capacity) {
        if (device.rplanes) cudaFree(device.rplanes);
        if (device.rweights) cudaFree(device.rweights);
        device.rplanes = nullptr;
        device.rweights = nullptr;
        device.rcount_capacity = 0;
        if (failed(cudaMalloc((void **)&device.rplanes, r->count * sizeof(uint32_t)), "reduction") ||
            failed(cudaMalloc((void **)&device.rweights, r->count * sizeof(double)), "reduction"))
            return PACMAN_RECURRENCE_DEVICE_ERROR;
        device.rcount_capacity = r->count;
    }
    if (r->points > device.routput_capacity) {
        if (device.routput) cudaFree(device.routput);
        device.routput = nullptr;
        device.routput_capacity = 0;
        if (failed(cudaMalloc((void **)&device.routput, r->points * sizeof(double)), "reduction"))
            return PACMAN_RECURRENCE_DEVICE_ERROR;
        device.routput_capacity = r->points;
    }
    size_t bytes = r->points * sizeof(double);
    if ((r->count && (failed(cudaMemcpy(device.rplanes, r->planes, r->count * sizeof(uint32_t),
                                        cudaMemcpyHostToDevice), "reduction") ||
                      failed(cudaMemcpy(device.rweights, r->weights, r->count * sizeof(double),
                                        cudaMemcpyHostToDevice), "reduction"))) ||
        failed(cudaMemcpy(device.routput, r->output, bytes, cudaMemcpyHostToDevice), "reduction"))
        return PACMAN_RECURRENCE_DEVICE_ERROR;
    unsigned block = 128, grid = (r->points + block - 1) / block;
    if (device.kept_df64)
        reduce_kernel<D><<<grid, block>>>(device.are, device.aim, device.kept_stride, device.rplanes,
                                          device.rweights, r->count, device.routput, r->points);
    else
        reduce_kernel<double><<<grid, block>>>(device.are, device.aim, device.kept_stride, device.rplanes,
                                               device.rweights, r->count, device.routput, r->points);
    if (failed(cudaGetLastError(), "reduction") ||
        failed(cudaMemcpy(r->output, device.routput, bytes, cudaMemcpyDeviceToHost), "reduction"))
        return PACMAN_RECURRENCE_DEVICE_ERROR;
    device.copies += 2;
    device.bytes += 2 * bytes;
    device.reductions++;
    if (device.stats) device.seconds += now() - start;
    return PACMAN_RECURRENCE_OK;
}
