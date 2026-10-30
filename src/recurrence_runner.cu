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
// operation for operation; built with -fmad=false, the results equal RustiCol's bit for bit.
//
// Build: nvcc -O3 -fmad=false -Xcompiler -fPIC -shared -cudart static -o librecurrence.so
//        recurrence_runner.cu

#include <cuda_runtime.h>
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
} pacman_recurrence_call;

enum {
    PACMAN_RECURRENCE_OK = 0,
    PACMAN_RECURRENCE_INVALID = 2,
    PACMAN_RECURRENCE_DEVICE_ERROR = 3,
};

extern "C" int pacman_recurrence_v1_run(const pacman_recurrence_call *call);

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

struct Widths {
    int w[W_COUNT];
};

__device__ __forceinline__ uint32_t take(uint64_t &word, int bits) {
    uint32_t v = (uint32_t)(word & ((1ull << bits) - 1));
    word >>= bits;
    return v;
}

__device__ __forceinline__ C ld_c(const Arena &a, uint32_t plane, size_t i) {
    return cx(a.cre[plane * a.ld + i], a.cim[plane * a.ld + i]);
}

__device__ __forceinline__ double ld_p(const Arena &a, uint32_t form, int mu, size_t i) {
    return a.mom[(form * 4 + mu) * a.ld + i];
}

// `add_scaled_current` / `set_scaled_current`: value × scale, stored or added
__device__ __forceinline__ void st_scaled(const Arena &a, uint32_t plane, size_t i, C value,
                                          C scale, bool accumulate) {
    C c = mul(value, scale);
    double *pr = a.cre + plane * a.ld + i, *pi = a.cim + plane * a.ld + i;
    if (accumulate) {
        *pr += c.re;
        *pi += c.im;
    } else {
        *pr = c.re;
        *pi = c.im;
    }
}

__device__ __forceinline__ C factor(const Arena &a, uint32_t id) { return cx(a.fre[id], a.fim[id]); }

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

// ---- the executors' arithmetic, shared by every kernel variant: operands in, values out ----

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

// ---- v1: the words as RustiCol encodes them, every operand in global memory ----

template <int OP>
__device__ __forceinline__ void contribution(const Arena &a, const Widths &w, uint64_t word, size_t i) {
    constexpr int op = OP, c0 = Shape<OP>::c0, c1 = Shape<OP>::c1, cd = Shape<OP>::cd;
    bool accumulate = !take(word, 1);
    uint32_t slot = take(word, w.w[W_S]);
    uint32_t p0 = take(word, w.w[W_P]), p1 = take(word, w.w[W_P]), dst = take(word, w.w[W_P]);
    uint32_t f0 = take(word, w.w[W_F]), f1 = take(word, w.w[W_F]);
    uint32_t x = take(word, w.w[W_X]);
    // `execute_rows`: context scale × row factor
    C scale = mul(cx(a.sre[slot], a.sim[slot]), factor(a, x));
    C l[6], r[4], o[6];
    double q0[4], q1[4];
#pragma unroll
    for (int k = 0; k < c0; k++) l[k] = ld_c(a, p0 + k, i);
#pragma unroll
    for (int k = 0; k < c1; k++) r[k] = ld_c(a, p1 + k, i);
    if (op == 6)
#pragma unroll
        for (int k = 0; k < 4; k++) {
            q0[k] = ld_p(a, f0, k, i);
            q1[k] = ld_p(a, f1, k, i);
        }
    contribution_values(op, l, r, q0, q1, o);
#pragma unroll
    for (int k = 0; k < cd; k++) st_scaled(a, dst + k, i, o[k], scale, accumulate);
}

template <int OP>
__device__ __forceinline__ void propagator(const Arena &a, const Widths &w, uint64_t word, size_t i) {
    constexpr int op = OP, n = OP == 9 ? 4 : 2;
    uint32_t base = take(word, w.w[W_P]), form = take(word, w.w[W_F]), x = take(word, w.w[W_X]);
    double p[4];
#pragma unroll
    for (int k = 0; k < 4; k++) p[k] = ld_p(a, form, k, i);
    C c[4], o[4];
#pragma unroll
    for (int k = 0; k < n; k++) c[k] = ld_c(a, base + k, i);
    propagator_values(op, c, p, factor(a, x), o);
#pragma unroll
    for (int k = 0; k < n; k++) {
        a.cre[(base + k) * a.ld + i] = o[k].re;
        a.cim[(base + k) * a.ld + i] = o[k].im;
    }
}

__global__ void run_plan(const uint64_t *__restrict__ words, uint32_t word_count, Widths w,
                         Arena a, uint32_t points) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= points) return;
    for (uint32_t n = 0; n < word_count; n++) {
        uint64_t word = __ldg(words + n);
        int op = (int)take(word, 4);
        switch (op) {
        case 1: {
            uint32_t base = take(word, w.w[W_P]), count = take(word, w.w[W_P]);
            for (uint32_t k = 0; k < count; k++) {
                a.cre[(base + k) * a.ld + i] = 0.0;
                a.cim[(base + k) * a.ld + i] = 0.0;
            }
            break;
        }
        case 2: contribution<2>(a, w, word, i); break;
        case 3: contribution<3>(a, w, word, i); break;
        case 4: contribution<4>(a, w, word, i); break;
        case 5: contribution<5>(a, w, word, i); break;
        case 6: contribution<6>(a, w, word, i); break;
        case 7: propagator<7>(a, w, word, i); break;
        case 8: propagator<8>(a, w, word, i); break;
        case 9: propagator<9>(a, w, word, i); break;
        case 10: {
            // `execute_identity_finalization_rows`: factor × value
            uint32_t base = take(word, w.w[W_P]), count = take(word, w.w[W_K]),
                     x = take(word, w.w[W_X]);
            double fr = a.fre[x], fi = a.fim[x];
            for (uint32_t k = 0; k < count; k++) {
                double *pr = a.cre + (base + k) * a.ld + i, *pi = a.cim + (base + k) * a.ld + i;
                C v = identity_value(fr, fi, cx(*pr, *pi));
                *pr = v.re;
                *pi = v.im;
            }
            break;
        }
        case 11: {
            // `execute_closure_reduce_rows`
            uint32_t p0 = take(word, w.w[W_P]), p1 = take(word, w.w[W_P]),
                     amp = take(word, w.w[W_A]), x = take(word, w.w[W_X]),
                     first = take(word, w.w[W_X]), count = take(word, w.w[W_K]);
            double rfr = a.fre[x], rfi = a.fim[x];
            double *dr = a.are + amp * a.ld + i, *di = a.aim + amp * a.ld + i;
            for (uint32_t k = 0; k < count; k++) {
                C v = closure_increment(ld_c(a, p0 + k, i), ld_c(a, p1 + k, i), a.fre[first + k],
                                        a.fim[first + k], rfr, rfi);
                *dr += v.re;
                *di += v.im;
            }
            break;
        }
        default:
            return;  // the host validates the words
        }
    }
}

// ---- programs: the words rewritten by the host (`prepare` below), 32 bytes per instruction ----
//
// w[0] = kind | op << 8 | flags << 16; the fields follow in w[1..7].
//   acc kind: the plan's rows with the contribution runs stably sorted by destination (each
//     destination keeps its order of rows, so its sum is unchanged) and the destination of a run
//     of rows on one destination kept in registers (F_FIRST loads or starts it, F_LAST stores it).
//   cache kind: every current component the program touches lives in a shared-memory slot of the
//     thread while it is used; the host schedules the slots (Belady: evict the value used farthest
//     ahead; spill an evicted value only if it is read again), with explicit fill (global → slot),
//     spill (slot → global) and zero instructions. Operands are slot numbers (bytes).

enum { K_CLEAR = 1, K_CONTRIB, K_PROP, K_IDENTITY, K_CLOSURE, K_FILL, K_SPILL, K_ZERO };
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

// acc: contributions w[1] scale slot, w[2] p0, w[3] p1, w[4] dst, w[5] f0, w[6] f1, w[7] factor;
// clear w[1] base, w[2] count; propagator w[1] base, w[2] form, w[3] factor; identity w[1] base,
// w[2] count, w[3] factor; closure w[1] p0, w[2] p1, w[3] amplitude, w[4] factor, w[5] first,
// w[6] count.
template <class R>
__device__ __forceinline__ Cx<R> ld_cr(const Arena &a, uint32_t plane, size_t i) {
    return cx(ldr<R>(a.cre, plane * a.ld + i), ldr<R>(a.cim, plane * a.ld + i));
}

template <class R, int OP>
__device__ __forceinline__ void acc_contribution(const Arena &a, const IOp &o, int flags, Cx<R> *acc,
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
__device__ __forceinline__ void acc_propagator(const Arena &a, const IOp &o, size_t i) {
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

template <class R>
__global__ void run_acc(const IOp *__restrict__ ops, uint32_t count, Arena a, uint32_t points) {
    typedef Cx<R> C;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= points) return;
    C acc[6];
    for (uint32_t n = 0; n < count; n++) {
        IOp o = load_op(ops, n);
        int kind = o.w[0] & 0xff, op = (o.w[0] >> 8) & 0xff, flags = o.w[0] >> 16;
        if (kind == K_CONTRIB) {
            switch (op) {
            case 2: acc_contribution<R, 2>(a, o, flags, acc, i); break;
            case 3: acc_contribution<R, 3>(a, o, flags, acc, i); break;
            case 4: acc_contribution<R, 4>(a, o, flags, acc, i); break;
            case 5: acc_contribution<R, 5>(a, o, flags, acc, i); break;
            default: acc_contribution<R, 6>(a, o, flags, acc, i); break;
            }
        } else if (kind == K_CLEAR) {
            for (uint32_t k = 0; k < o.w[2]; k++) {
                str(a.cre, (o.w[1] + k) * a.ld + i, R(0.0));
                str(a.cim, (o.w[1] + k) * a.ld + i, R(0.0));
            }
        } else if (kind == K_PROP) {
            if (op == 7) acc_propagator<R, 7>(a, o, i);
            else if (op == 8) acc_propagator<R, 8>(a, o, i);
            else acc_propagator<R, 9>(a, o, i);
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

// cache: slots s of the thread at shared[(2 s + part) * blockDim.x + threadIdx.x] (re, im).
// Operand slots are bytes in w[2..5] (16 of them, in operand order); fills and spills hold up to 7
// pairs `slot | plane << 8` in w[1..7], zero up to 16 slot bytes in w[2..5], the count in flags
// >> 3. Contributions: w[1] scale slot, w[6] f0 | f1 << 16, w[7] factor; propagators: w[1] form,
// w[7] factor; identity: w[1] count, w[7] factor; closure: w[1] amplitude, w[6] first | count
// << 16, w[7] factor.
__device__ __forceinline__ uint32_t slot_of(const IOp &o, int k) {
    return (o.w[2 + (k >> 2)] >> (8 * (k & 3))) & 0xff;
}

template <class R> struct Slots {
    double *mine;  // a slot holds re and im: two doubles, or two float2 (df64) in their place
    uint32_t B;
    __device__ __forceinline__ Cx<R> get(uint32_t s) const {
        return cx(ldr<R>(mine, (2 * s) * B), ldr<R>(mine, (2 * s + 1) * B));
    }
    __device__ __forceinline__ void put(uint32_t s, Cx<R> v) const {
        str(mine, (2 * s) * B, v.re);
        str(mine, (2 * s + 1) * B, v.im);
    }
};

template <class R, int OP>
__device__ __forceinline__ void cache_contribution(const Arena &a, const Slots<R> &m, const IOp &o,
                                                   int flags, size_t i) {
    typedef Cx<R> C;
    constexpr int op = OP, c0 = Shape<OP>::c0, c1 = Shape<OP>::c1, cd = Shape<OP>::cd;
    C scale = mul(cx(ldr<R>(a.sre, o.w[1]), ldr<R>(a.sim, o.w[1])), cx(ldr<R>(a.fre, o.w[7]), ldr<R>(a.fim, o.w[7])));
    C l[6], r[4], v[6];
    R q0[4], q1[4];
#pragma unroll
    for (int k = 0; k < c0; k++) l[k] = m.get(slot_of(o, k));
#pragma unroll
    for (int k = 0; k < c1; k++) r[k] = m.get(slot_of(o, c0 + k));
    if (op == 6)
#pragma unroll
        for (int k = 0; k < 4; k++) {
            q0[k] = ldr<R>(a.mom, ((o.w[6] & 0xffff) * 4 + k) * a.ld + i);
            q1[k] = ldr<R>(a.mom, ((o.w[6] >> 16) * 4 + k) * a.ld + i);
        }
    contribution_values(op, l, r, q0, q1, v);
#pragma unroll
    for (int k = 0; k < cd; k++) {
        uint32_t s = slot_of(o, c0 + c1 + k);
        C c = mul(v[k], scale);
        m.put(s, (flags & F_INIT) ? c : add(m.get(s), c));
    }
}

template <class R, int OP>
__device__ __forceinline__ void cache_propagator(const Arena &a, const Slots<R> &m, const IOp &o, size_t i) {
    typedef Cx<R> C;
    constexpr int nc = OP == 9 ? 4 : 2;
    R p[4];
#pragma unroll
    for (int k = 0; k < 4; k++) p[k] = ldr<R>(a.mom, (o.w[1] * 4 + k) * a.ld + i);
    C c[4], v[4];
#pragma unroll
    for (int k = 0; k < nc; k++) c[k] = m.get(slot_of(o, k));
    propagator_values(OP, c, p, cx(ldr<R>(a.fre, o.w[7]), ldr<R>(a.fim, o.w[7])), v);
#pragma unroll
    for (int k = 0; k < nc; k++) m.put(slot_of(o, k), v[k]);
}

// R = double: RustiCol's arithmetic; R = D: df64 on a df64 arena (all planes, momenta, scales and
// factors hold float2 (high, low) in place of their doubles; see `to_df64`, `to_f64`).
template <class R>
__global__ void run_cache(const IOp *__restrict__ ops, uint32_t count, Arena a, uint32_t points) {
    typedef Cx<R> C;
    extern __shared__ double shared[];
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= points) return;
    Slots<R> sm{shared + threadIdx.x, blockDim.x};
    for (uint32_t n = 0; n < count; n++) {
        IOp o = load_op(ops, n);
        int kind = o.w[0] & 0xff, op = (o.w[0] >> 8) & 0xff, flags = o.w[0] >> 16;
        int m = flags >> 3;
        if (kind == K_FILL) {
            for (int k = 0; k < m; k++) {
                uint32_t pair = o.w[1 + k];
                sm.put(pair & 0xff, ld_cr<R>(a, pair >> 8, i));
            }
        } else if (kind == K_SPILL) {
            for (int k = 0; k < m; k++) {
                uint32_t pair = o.w[1 + k];
                C v = sm.get(pair & 0xff);
                str(a.cre, (size_t)(pair >> 8) * a.ld + i, v.re);
                str(a.cim, (size_t)(pair >> 8) * a.ld + i, v.im);
            }
        } else if (kind == K_ZERO) {
            for (int k = 0; k < m; k++) sm.put(slot_of(o, k), cx(R(0.0), R(0.0)));
        } else if (kind == K_CONTRIB) {
            switch (op) {
            case 2: cache_contribution<R, 2>(a, sm, o, flags, i); break;
            case 3: cache_contribution<R, 3>(a, sm, o, flags, i); break;
            case 4: cache_contribution<R, 4>(a, sm, o, flags, i); break;
            case 5: cache_contribution<R, 5>(a, sm, o, flags, i); break;
            default: cache_contribution<R, 6>(a, sm, o, flags, i); break;
            }
        } else if (kind == K_PROP) {
            if (op == 7) cache_propagator<R, 7>(a, sm, o, i);
            else if (op == 8) cache_propagator<R, 8>(a, sm, o, i);
            else cache_propagator<R, 9>(a, sm, o, i);
        } else if (kind == K_IDENTITY) {
            R fr = ldr<R>(a.fre, o.w[7]), fi = ldr<R>(a.fim, o.w[7]);
            for (uint32_t k = 0; k < o.w[1]; k++)
                sm.put(slot_of(o, k), identity_value(fr, fi, sm.get(slot_of(o, k))));
        } else if (kind == K_CLOSURE) {
            uint32_t first = o.w[6] & 0xffff, nk = o.w[6] >> 16;
            R rfr = ldr<R>(a.fre, o.w[7]), rfi = ldr<R>(a.fim, o.w[7]);
            size_t at = o.w[1] * a.ld + i;
            R dr = ldr<R>(a.are, at), di = ldr<R>(a.aim, at);
            for (uint32_t k = 0; k < nk; k++) {
                C v = closure_increment(sm.get(slot_of(o, k)), sm.get(slot_of(o, nk + k)),
                                        ldr<R>(a.fre, first + k), ldr<R>(a.fim, first + k), rfr, rfi);
                dr = dr + v.re;
                di = di + v.im;
            }
            str(a.are, at, dr);
            str(a.aim, at, di);
        }
    }
}

// in-place conversions of `count` doubles between f64 and df64 (float2 in the same 8 bytes)
__global__ void to_df64(double *p, size_t count) {
    for (size_t k = blockIdx.x * (size_t)blockDim.x + threadIdx.x; k < count; k += (size_t)gridDim.x * blockDim.x) {
        double x = p[k];
        float hi = __double2float_rn(x);
        float lo = __double2float_rn(x - (double)hi);
        reinterpret_cast<float2 *>(p)[k] = make_float2(hi, lo);
    }
}

__global__ void to_f64(double *p, size_t count) {
    for (size_t k = blockIdx.x * (size_t)blockDim.x + threadIdx.x; k < count; k += (size_t)gridDim.x * blockDim.x) {
        float2 v = reinterpret_cast<float2 *>(p)[k];
        p[k] = (double)v.x + (double)v.y;
    }
}

// ---- host side -----------------------------------------------------------------------------

namespace {

struct Device {
    double *cre = nullptr, *cim = nullptr, *are = nullptr, *aim = nullptr, *mom = nullptr;
    double *small = nullptr;  // scales, then factors
    size_t cre_capacity = 0, cim_capacity = 0, are_capacity = 0, aim_capacity = 0,
           momentum_capacity = 0, small_capacity = 0;
    std::unordered_map<uint64_t, std::pair<uint64_t *, uint32_t>> words;
    uint64_t *scratch_words = nullptr;
    size_t scratch_capacity = 0;
    // statistics (PACMAN_RUNNER_STATS=1)
    uint64_t calls = 0, points = 0, copies = 0, bytes = 0;
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

std::vector<IOp> acc_program(const std::vector<Row> &rows) {
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

// The components a row touches, in operand order, with their access: 1 read, 2 write, 3 both.
void accesses(const Row &r, std::vector<std::pair<uint32_t, int>> &out) {
    out.clear();
    if (r.op == 1) {
        for (uint32_t k = 0; k < r.f[1]; k++) out.push_back({r.f[0] + k, 2});
    } else if (r.op >= 2 && r.op <= 6) {
        int c0, c1, cd;
        contribution_shape(r.op, &c0, &c1, &cd);
        for (int k = 0; k < c0; k++) out.push_back({r.f[2] + k, 1});
        for (int k = 0; k < c1; k++) out.push_back({r.f[3] + k, 1});
        for (int k = 0; k < cd; k++) out.push_back({r.f[4] + k, r.f[0] ? 2 : 3});
    } else if (r.op >= 7 && r.op <= 9) {
        for (int k = 0; k < (r.op == 9 ? 4 : 2); k++) out.push_back({r.f[0] + k, 3});
    } else if (r.op == 10) {
        for (uint32_t k = 0; k < r.f[1]; k++) out.push_back({r.f[0] + k, 3});
    } else if (r.op == 11) {
        for (uint32_t k = 0; k < r.f[5]; k++) out.push_back({r.f[0] + k, 1});
        for (uint32_t k = 0; k < r.f[5]; k++) out.push_back({r.f[1] + k, 1});
    }
}

struct CacheStats {
    uint64_t fills = 0, spills = 0, zeros = 0, accesses = 0;
};

// The cache program with `slots` slots per thread (16..255). With `live_out`, every value still
// dirty at the end is spilled (the caller copies currents back).
std::vector<IOp> cache_program(const std::vector<Row> &rows, int slots, bool live_out,
                               CacheStats *stats) {
    const uint32_t NONE = 0xffffffffu;
    // per component: the rows that touch it, and how
    std::unordered_map<uint32_t, std::vector<std::pair<uint32_t, int>>> uses;
    std::vector<std::pair<uint32_t, int>> acc;
    for (uint32_t n = 0; n < rows.size(); n++) {
        accesses(rows[n], acc);
        for (auto &x : acc) {
            auto &u = uses[x.first];
            if (!u.empty() && u.back().first == n) u.back().second |= x.second;
            else u.push_back({n, x.second});
        }
    }
    std::unordered_map<uint32_t, size_t> cursor;  // component -> index of its current use
    auto next_use = [&](uint32_t c, uint32_t after, int *mode) -> uint32_t {
        auto &u = uses[c];
        size_t &k = cursor[c];
        while (k < u.size() && u[k].first <= after) k++;
        if (k == u.size()) { *mode = 0; return NONE; }
        *mode = u[k].second;
        return u[k].first;
    };
    std::vector<uint32_t> holder(slots, NONE);  // slot -> component
    std::vector<uint32_t> next(slots, NONE);    // slot -> its value's next use (row)
    std::vector<int> next_mode(slots, 0);
    std::vector<char> dirty(slots, 0);
    std::unordered_map<uint32_t, int> where;     // component -> slot
    std::vector<IOp> ops;
    std::vector<uint32_t> fill_pairs, spill_pairs, zero_slots;
    auto flush_pairs = [&](int kind, std::vector<uint32_t> &pairs) {
        for (size_t k = 0; k < pairs.size(); k += 7) {
            int m = (int)std::min<size_t>(7, pairs.size() - k);
            IOp o = make_op(kind, 0, m << 3);
            for (int q = 0; q < m; q++) o.w[1 + q] = pairs[k + q];
            ops.push_back(o);
        }
        pairs.clear();
    };
    auto set_slot = [](IOp &o, int k, uint32_t s) { o.w[2 + (k >> 2)] |= s << (8 * (k & 3)); };
    std::vector<int> row_slots;
    for (uint32_t n = 0; n < rows.size(); n++) {
        const Row &r = rows[n];
        accesses(r, acc);
        // a clear whose value is overwritten (or never used) before it is read is dropped
        if (r.op == 1) {
            bool needed = false;
            for (auto &x : acc) {
                int mode;
                next_use(x.first, n, &mode);
                if (mode & 1) needed = true;
            }
            if (!needed) continue;
        }
        row_slots.assign(acc.size(), -1);
        // pin the row's resident components, then place the missing ones
        std::vector<char> pinned(slots, 0);
        for (size_t k = 0; k < acc.size(); k++) {
            auto it = where.find(acc[k].first);
            if (it != where.end()) { row_slots[k] = it->second; pinned[it->second] = 1; }
        }
        for (size_t k = 0; k < acc.size(); k++) {
            if (row_slots[k] >= 0) continue;
            auto it = where.find(acc[k].first);  // a repeated operand placed just now
            if (it != where.end()) { row_slots[k] = it->second; continue; }
            int s = -1;
            for (int q = 0; q < slots && s < 0; q++) if (holder[q] == NONE) s = q;
            if (s < 0) {  // evict the value used farthest ahead
                uint32_t far = 0;
                for (int q = 0; q < slots; q++) {
                    if (pinned[q]) continue;
                    uint32_t t = next[q] == NONE ? NONE : next[q];
                    if (s < 0 || t > far) { s = q; far = t; }
                }
                uint32_t c = holder[s];
                bool live = (next[s] != NONE && (next_mode[s] & 1)) || (next[s] == NONE && live_out);
                if (dirty[s] && live) {
                    spill_pairs.push_back((uint32_t)s | c << 8);
                    if (stats) stats->spills++;
                }
                where.erase(c);
            }
            holder[s] = acc[k].first;
            where[acc[k].first] = s;
            dirty[s] = 0;
            pinned[s] = 1;
            row_slots[k] = s;
            if (acc[k].second & 1) {
                fill_pairs.push_back((uint32_t)s | acc[k].first << 8);
                if (stats) stats->fills++;
            }
        }
        flush_pairs(K_SPILL, spill_pairs);
        flush_pairs(K_FILL, fill_pairs);
        if (stats) stats->accesses += acc.size();
        // the row itself
        if (r.op == 1) {
            for (size_t k = 0; k < acc.size(); k += 16) {
                int m = (int)std::min<size_t>(16, acc.size() - k);
                IOp o = make_op(K_ZERO, 0, m << 3);
                for (int q = 0; q < m; q++) set_slot(o, q, row_slots[k + q]);
                ops.push_back(o);
                if (stats) stats->zeros += m;
            }
        } else {
            IOp o;
            if (r.op >= 2 && r.op <= 6) {
                o = make_op(K_CONTRIB, r.op, r.f[0] ? F_INIT : 0);
                o.w[1] = r.f[1];
                o.w[6] = r.f[5] | r.f[6] << 16;
                o.w[7] = r.f[7];
            } else if (r.op <= 9) {
                o = make_op(K_PROP, r.op, 0);
                o.w[1] = r.f[1];
                o.w[7] = r.f[2];
            } else if (r.op == 10) {
                o = make_op(K_IDENTITY, r.op, 0);
                o.w[1] = r.f[1];
                o.w[7] = r.f[2];
            } else {
                o = make_op(K_CLOSURE, r.op, 0);
                o.w[1] = r.f[2];
                o.w[6] = r.f[4] | r.f[5] << 16;
                o.w[7] = r.f[3];
            }
            for (size_t k = 0; k < acc.size(); k++) set_slot(o, (int)k, (uint32_t)row_slots[k]);
            ops.push_back(o);
        }
        for (size_t k = 0; k < acc.size(); k++) {
            int s = row_slots[k];
            if (acc[k].second & 2) dirty[s] = 1;
            int mode;
            next[s] = next_use(acc[k].first, n, &mode);
            next_mode[s] = mode;
        }
    }
    if (live_out) {
        for (int q = 0; q < slots; q++)
            if (holder[q] != NONE && dirty[q]) {
                spill_pairs.push_back((uint32_t)q | holder[q] << 8);
                if (stats) stats->spills++;
            }
        flush_pairs(K_SPILL, spill_pairs);
    }
    return ops;
}

// One prepared program: the kernel variant and its instructions on the device.
enum Variant { V1, ACC, CACHE };

struct Prepared {
    Variant variant;
    int slots;
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
bool copy_runs(double *dre, double *dim, double *hre, double *him, const uint32_t *runs,
               uint32_t count, uint32_t planes, size_t stride, bool to_device) {
    for (uint32_t r = 0; r < count; r++) {
        uint32_t first = runs[2 * r], n = runs[2 * r + 1];
        if (first > planes || n > planes - first) return false;
        size_t offset = (size_t)first * stride, bytes = (size_t)n * stride * sizeof(double);
        cudaMemcpyKind kind = to_device ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost;
        if (failed(cudaMemcpy(to_device ? (void *)(dre + offset) : (void *)(hre + offset),
                              to_device ? (const void *)(hre + offset) : (const void *)(dre + offset),
                              bytes, kind), "copy") ||
            failed(cudaMemcpy(to_device ? (void *)(dim + offset) : (void *)(him + offset),
                              to_device ? (const void *)(him + offset) : (const void *)(dim + offset),
                              bytes, kind), "copy"))
            return false;
        device.copies += 2;
        device.bytes += 2 * bytes;
    }
    return true;
}

// the kernel variant: PACMAN_RECURRENCE_KERNEL=v1|acc|cache|df64|df64cache (v1 and acc and
// cache are RustiCol's arithmetic bit for bit; df64 and df64cache run acc and cache in df64 on a
// df64 copy of the arena), PACMAN_RECURRENCE_SLOTS (cache, 32), PACMAN_RECURRENCE_BLOCK (threads
// per block; 128, cache 64), PACMAN_RECURRENCE_SORT=0 (do not sort the contribution rows by
// destination)
struct Options {
    Variant variant = V1;
    bool df64 = false;
    int slots = 32, block = 0;
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
    if (k && !strcmp(k, "acc")) o.variant = ACC;
    else if (k && !strcmp(k, "cache")) o.variant = CACHE;
    else if (k && !strcmp(k, "df64")) { o.variant = ACC; o.df64 = true; }
    else if (k && !strcmp(k, "df64cache")) { o.variant = CACHE; o.df64 = true; }
    else if (k && *k && strcmp(k, "v1"))
        fprintf(stderr, "pacman recurrence runner: unknown PACMAN_RECURRENCE_KERNEL=%s, using v1\\n", k);
    if (const char *v = getenv("PACMAN_RECURRENCE_SLOTS")) o.slots = std::max(16, std::min(255, atoi(v)));
    if (const char *v = getenv("PACMAN_RECURRENCE_BLOCK")) o.block = std::max(32, std::min(1024, atoi(v)));
    if (o.block == 0) o.block = o.variant == CACHE ? (o.df64 ? 128 : 64) : 128;
    if (o.df64 && !getenv("PACMAN_RECURRENCE_SLOTS")) o.slots = 16;
    return o;
}

// the program of `call`'s words for the chosen variant, built once per words key (and live_out)
std::unordered_map<uint64_t, Prepared> prepared_programs;

Prepared build_program(const uint64_t *words, uint32_t count, const uint8_t *width, Variant variant,
                       int slots, bool live_out, CacheStats *stats) {
    Prepared p{variant, slots, 0, nullptr};
    if (variant == V1) return p;
    std::vector<Row> rows = decode(words, count, width);
    if (sort_option()) sort_contribution_runs(rows);
    std::vector<IOp> ops = variant == ACC ? acc_program(rows) : cache_program(rows, slots, live_out, stats);
    p.count = (uint32_t)ops.size();
    if (failed(cudaMalloc((void **)&p.ops, std::max<size_t>(1, ops.size()) * sizeof(IOp)), "program") ||
        failed(cudaMemcpy(p.ops, ops.data(), ops.size() * sizeof(IOp), cudaMemcpyHostToDevice), "program"))
        p.ops = nullptr;
    return p;
}

const Prepared *prepare(const pacman_recurrence_call *call, bool live_out) {
    static Prepared scratch{V1, 0, 0, nullptr};
    Options o = options();
    if (o.variant == V1) {
        scratch.variant = V1;
        return &scratch;
    }
    uint64_t key = call->words_key ? (call->words_key * 4 + (live_out ? 2 : 0)) : 0;
    if (key) {
        auto it = prepared_programs.find(key);
        if (it != prepared_programs.end()) return &it->second;
    }
    Prepared p = build_program(call->words, call->word_count, call->width, o.variant, o.slots,
                               live_out, nullptr);
    if (!p.ops) return nullptr;
    if (key) return &(prepared_programs[key] = p);
    if (scratch.ops) cudaFree(scratch.ops);
    scratch = p;
    return &scratch;
}

bool launch(const Prepared &p, const uint64_t *dwords, uint32_t word_count, Widths w, const Arena &a,
            uint32_t points) {
    Options o = options();
    unsigned block = (unsigned)o.block, grid = (points + block - 1) / block;
    if (p.variant == V1) {
        run_plan<<<grid, block>>>(dwords, word_count, w, a, points);
    } else if (p.variant == ACC) {
        if (o.df64) run_acc<D><<<grid, block>>>(p.ops, p.count, a, points);
        else run_acc<double><<<grid, block>>>(p.ops, p.count, a, points);
    } else {
        size_t bytes = (size_t)p.slots * 2 * block * sizeof(double);
        static size_t allowed = 0;
        if (bytes > allowed) {
            if (failed(cudaFuncSetAttribute(run_cache<double>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)bytes),
                       "shared memory") ||
                failed(cudaFuncSetAttribute(run_cache<D>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)bytes),
                       "shared memory"))
                return false;
            allowed = bytes;
        }
        if (o.df64) run_cache<D><<<grid, block, bytes>>>(p.ops, p.count, a, points);
        else run_cache<double><<<grid, block, bytes>>>(p.ops, p.count, a, points);
    }
    return !failed(cudaGetLastError(), "launch");
}

// in-place f64 <-> df64 conversions of `doubles` values, and of plane runs of re and im
bool convert(void (*f)(double *, size_t), double *p, size_t doubles) {
    if (doubles == 0) return true;
    unsigned grid = (unsigned)std::min<size_t>(4096, (doubles + 255) / 256);
    f<<<grid, 256>>>(p, doubles);
    return !failed(cudaGetLastError(), "conversion");
}

bool convert_runs(void (*f)(double *, size_t), double *re, double *im, const uint32_t *runs,
                  uint32_t count, size_t stride) {
    for (uint32_t r = 0; r < count; r++) {
        size_t offset = (size_t)runs[2 * r] * stride, doubles = (size_t)runs[2 * r + 1] * stride;
        if (!convert(f, re + offset, doubles) || !convert(f, im + offset, doubles)) return false;
    }
    return true;
}

void print_stats() {
    if (!device.stats || device.calls == 0) return;
    fprintf(stderr,
            "pacman recurrence runner: %llu calls, %llu points, %llu copies (%.1f MB), %.3f s "
            "(%.2f us/point)\n",
            (unsigned long long)device.calls, (unsigned long long)device.points,
            (unsigned long long)device.copies, device.bytes / 1e6, device.seconds,
            device.points ? device.seconds * 1e6 / device.points : 0.0);
}

double now() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

}  // namespace

extern "C" int pacman_recurrence_v1_run(const pacman_recurrence_call *call) {
    static bool initialized = false;
    if (!initialized) {
        initialized = true;
        const char *s = getenv("PACMAN_RUNNER_STATS");
        device.stats = s && *s && strcmp(s, "0") != 0;
        atexit(print_stats);
    }
    if (!call || call->abi_version != PACMAN_RECURRENCE_ABI ||
        call->struct_size != sizeof(pacman_recurrence_call) || call->points == 0 ||
        call->points > call->stride || !call->words)
        return PACMAN_RECURRENCE_INVALID;
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

    // the words: cached under their key, else copied every call
    const uint64_t *dwords;
    auto cached = call->words_key ? device.words.find(call->words_key) : device.words.end();
    if (cached != device.words.end() && cached->second.second == call->word_count) {
        dwords = cached->second.first;
    } else {
        size_t bytes = (size_t)call->word_count * sizeof(uint64_t);
        uint64_t *target;
        if (call->words_key) {
            if (failed(cudaMalloc((void **)&target, bytes ? bytes : 8), "words"))
                return PACMAN_RECURRENCE_DEVICE_ERROR;
            device.words[call->words_key] = {target, call->word_count};
        } else {
            if (call->word_count > device.scratch_capacity) {
                if (device.scratch_words) cudaFree(device.scratch_words);
                if (failed(cudaMalloc((void **)&device.scratch_words, bytes), "words"))
                    return PACMAN_RECURRENCE_DEVICE_ERROR;
                device.scratch_capacity = call->word_count;
            }
            target = device.scratch_words;
        }
        if (failed(cudaMemcpy(target, call->words, bytes, cudaMemcpyHostToDevice), "words"))
            return PACMAN_RECURRENCE_DEVICE_ERROR;
        dwords = target;
    }

    // inputs: momenta, scales and factors, the current and amplitude runs named
    double *dscale_re = device.small, *dscale_im = dscale_re + call->scale_count;
    double *dfactor_re = dscale_im + call->scale_count, *dfactor_im = dfactor_re + call->factor_count;
    if ((momentum_doubles &&
         failed(cudaMemcpy(device.mom, call->momenta, momentum_doubles * sizeof(double),
                           cudaMemcpyHostToDevice), "momenta")) ||
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
                   call->upload_current_runs, call->current_planes, stride, true) ||
        !copy_runs(device.are, device.aim, call->amplitude_re, call->amplitude_im,
                   call->upload_amplitude, call->upload_amplitude_runs, call->amplitude_planes,
                   stride, true))
        return PACMAN_RECURRENCE_DEVICE_ERROR;

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
    Widths w;
    for (int k = 0; k < W_COUNT; k++) w.w[k] = call->width[k];
    // df64: the uploaded planes, the momenta and the constants become df64 in place; the device
    // arena stays df64 between calls (planes copied back are converted to f64 and back again)
    bool df64 = options().df64;
    if (df64 && (!convert_runs(to_df64, device.cre, device.cim, call->upload_current,
                               call->upload_current_runs, stride) ||
                 !convert_runs(to_df64, device.are, device.aim, call->upload_amplitude,
                               call->upload_amplitude_runs, stride) ||
                 !convert(to_df64, device.mom, momentum_doubles) || !convert(to_df64, device.small, small)))
        return PACMAN_RECURRENCE_DEVICE_ERROR;
    bool live_out = call->download_current_runs > 0;
    const Prepared *prepared = prepare(call, live_out);
    if (!prepared) return PACMAN_RECURRENCE_DEVICE_ERROR;
    if (!launch(*prepared, dwords, call->word_count, w, a, call->points))
        return PACMAN_RECURRENCE_DEVICE_ERROR;
    if (df64 && (!convert_runs(to_f64, device.are, device.aim, call->download_amplitude,
                               call->download_amplitude_runs, stride) ||
                 !convert_runs(to_f64, device.cre, device.cim, call->download_current,
                               call->download_current_runs, stride)))
        return PACMAN_RECURRENCE_DEVICE_ERROR;
    if (failed(cudaDeviceSynchronize(), "kernel")) return PACMAN_RECURRENCE_DEVICE_ERROR;

    if (!copy_runs(device.are, device.aim, call->amplitude_re, call->amplitude_im,
                   call->download_amplitude, call->download_amplitude_runs,
                   call->amplitude_planes, stride, false) ||
        !copy_runs(device.cre, device.cim, call->current_re, call->current_im,
                   call->download_current, call->download_current_runs, call->current_planes,
                   stride, false))
        return PACMAN_RECURRENCE_DEVICE_ERROR;
    if (df64 && (!convert_runs(to_df64, device.are, device.aim, call->download_amplitude,
                               call->download_amplitude_runs, stride) ||
                 !convert_runs(to_df64, device.cre, device.cim, call->download_current,
                               call->download_current_runs, stride)))
        return PACMAN_RECURRENCE_DEVICE_ERROR;
    device.calls++;
    device.points += call->points;
    if (device.stats) device.seconds += now() - start;
    return PACMAN_RECURRENCE_OK;
}
