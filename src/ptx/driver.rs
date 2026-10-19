// The C driver written next to a PTX module (`sjb2ptx --driver`): `<name>_cuda.c`, which embeds
// the module and runs the kernel through the CUDA driver API, and its header `<name>_cuda.h`.
//
// The driver loads `libcuda.so.1` with dlopen and declares the few driver API types and
// functions it uses itself (as cuda.h of CUDA 13 defines them), so a program using it needs
// neither the CUDA toolkit nor `-lcuda` to build: `cc user.c name_cuda.c -ldl`.

use super::{Layout, PtxOptions, Shape, BLOCK_SIZE};

/// `<name>_cuda.h`: the driver's API.
pub fn driver_header(options: &PtxOptions, shape: &Shape, complex: bool) -> String {
    let n = &options.name;
    let up = n.to_uppercase();
    let layout = match options.layout {
        Layout::Soa => "SoA: slot k of point i is at in[k * ld_in + i] (ld_in >= n)",
        Layout::Aos => "AoS: slot k of point i is at in[i * ld_in + k] (ld_in >= COUNT_IN)",
    };
    let values = if complex {
        " Complex values take two consecutive slots (re, im)."
    } else {
        ""
    };
    format!(
        "/* {n}_cuda.h: runs the PTX kernel `{n}` on an NVIDIA GPU; written by sjb2ptx {version}.

   Build with `cc your_program.c {n}_cuda.c -ldl`: the driver loads libcuda.so.1 at run time
   and needs neither the CUDA toolkit nor -lcuda.

       {n}_cuda *ctx;
       if ({n}_cuda_init(&ctx, 0) != 0) {{ puts({n}_cuda_error(ctx)); ... }}
       {n}_cuda_evaluate(ctx, n, in, ld_in, out, ld_out, params);   // host arrays
       {n}_cuda_free(ctx);

   Each point has {count_in} inputs and {count_out} outputs (doubles); `params` holds {count_params}
   shared parameters (NULL if none). Layout {layout}, and likewise for `out`.{values}
   Functions return 0 on success and -1 on errors ({n}_cuda_error describes the last one).
   A context is used by one thread at a time.
*/

#ifndef {up}_CUDA_H
#define {up}_CUDA_H

#include <stddef.h>

#ifdef __cplusplus
extern \"C\" {{
#endif

#define {up}_CUDA_COUNT_IN {count_in}
#define {up}_CUDA_COUNT_OUT {count_out}
#define {up}_CUDA_COUNT_PARAMS {count_params}
#define {up}_CUDA_LAYOUT_SOA {soa}

typedef struct {n}_cuda {n}_cuda;

/* Loads libcuda, initializes `device` (its primary context) and compiles the kernel. `*ctx`
   is set even on failure, for {n}_cuda_error; free it with {n}_cuda_free. */
int {n}_cuda_init({n}_cuda **ctx, int device);

/* Evaluates n points of host arrays: copies them to the GPU in batches, runs the kernel and
   copies the outputs back. */
int {n}_cuda_evaluate({n}_cuda *ctx, size_t n, const double *in, size_t ld_in, double *out,
                      size_t ld_out, const double *params);

/* Launches the kernel on device arrays (CUdeviceptr values) on `stream` (a CUstream, or NULL)
   without waiting for it. */
int {n}_cuda_evaluate_device({n}_cuda *ctx, size_t n, unsigned long long in, size_t ld_in,
                             unsigned long long out, size_t ld_out, unsigned long long params,
                             void *stream);

/* The most points {n}_cuda_evaluate sends to the GPU at a time (default 1048576; fewer if
   the GPU's memory is short). */
void {n}_cuda_set_batch({n}_cuda *ctx, size_t max_points);

/* The last error (with the CUDA error or the PTX compiler's log), or \"\". */
const char *{n}_cuda_error(const {n}_cuda *ctx);

void {n}_cuda_free({n}_cuda *ctx);

#ifdef __cplusplus
}}
#endif

#endif /* {up}_CUDA_H */
",
        version = env!("CARGO_PKG_VERSION"),
        count_in = shape.count_in,
        count_out = shape.count_out,
        count_params = shape.count_params,
        soa = (options.layout == Layout::Soa) as u8,
    )
}

/// The module as a C string literal, one line per literal.
fn c_string(text: &str) -> String {
    let mut s = String::new();
    for line in text.lines() {
        s.push_str("    \"");
        for c in line.chars() {
            match c {
                '\\' => s.push_str("\\\\"),
                '"' => s.push_str("\\\""),
                '\t' => s.push_str("\\t"),
                c => s.push(c),
            }
        }
        s.push_str("\\n\"\n");
    }
    s
}

/// `<name>_cuda.c`: the driver, with the module `ptx` embedded.
pub fn driver_source(options: &PtxOptions, shape: &Shape, ptx: &str) -> String {
    let n = &options.name;
    let soa = options.layout == Layout::Soa;
    format!(
        r#"/* {n}_cuda.c: runs the PTX kernel `{n}` (embedded below) through the CUDA driver API;
   written by sjb2ptx {version}. See {n}_cuda.h. */

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "{n}_cuda.h"

/* the CUDA driver API types and functions used here, as cuda.h declares them */
typedef int CUresult;
typedef int CUdevice;
typedef void *CUcontext;
typedef void *CUmodule;
typedef void *CUfunction;
typedef void *CUstream;
typedef unsigned long long CUdeviceptr;

typedef struct {{
    size_t srcXInBytes, srcY;
    int srcMemoryType;
    const void *srcHost;
    CUdeviceptr srcDevice;
    void *srcArray;
    size_t srcPitch;
    size_t dstXInBytes, dstY;
    int dstMemoryType;
    void *dstHost;
    CUdeviceptr dstDevice;
    void *dstArray;
    size_t dstPitch;
    size_t WidthInBytes, Height;
}} CUDA_MEMCPY2D;

enum {{ CU_MEMORYTYPE_HOST = 1, CU_MEMORYTYPE_DEVICE = 2 }};
enum {{ CU_JIT_ERROR_LOG_BUFFER = 5, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6 }};

typedef struct {{
    CUresult (*Init)(unsigned);
    CUresult (*DeviceGet)(CUdevice *, int);
    CUresult (*DevicePrimaryCtxRetain)(CUcontext *, CUdevice);
    CUresult (*DevicePrimaryCtxRelease)(CUdevice);
    CUresult (*CtxPushCurrent)(CUcontext);
    CUresult (*CtxPopCurrent)(CUcontext *);
    CUresult (*CtxSynchronize)(void);
    CUresult (*ModuleLoadDataEx)(CUmodule *, const void *, unsigned, int *, void **);
    CUresult (*ModuleGetFunction)(CUfunction *, CUmodule, const char *);
    CUresult (*ModuleUnload)(CUmodule);
    CUresult (*MemGetInfo)(size_t *, size_t *);
    CUresult (*MemAlloc)(CUdeviceptr *, size_t);
    CUresult (*MemFree)(CUdeviceptr);
    CUresult (*MemcpyHtoD)(CUdeviceptr, const void *, size_t);
    CUresult (*Memcpy2DUnaligned)(const CUDA_MEMCPY2D *);
    CUresult (*LaunchKernel)(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned,
                             unsigned, unsigned, CUstream, void **, void **);
    CUresult (*GetErrorString)(CUresult, const char **);
}} cuda_api;

#define COUNT_IN {count_in}
#define COUNT_OUT {count_out}
#define COUNT_PARAMS {count_params}
#define LAYOUT_SOA {soa}
#define BLOCK_SIZE {block}
#define DEFAULT_BATCH ((size_t)1 << 20)

struct {n}_cuda {{
    void *lib;
    cuda_api cu;
    CUdevice device;
    CUcontext context;
    CUmodule module;
    CUfunction kernel;
    int has_context;
    size_t max_batch;
    /* device buffers of {n}_cuda_evaluate, for `capacity` points */
    CUdeviceptr d_in, d_out, d_params;
    size_t capacity;
    char error[4096];
    char log[16384];
}};

static const char PTX[] =
{ptx};

static int fail({n}_cuda *ctx, const char *what, CUresult r) {{
    const char *s = NULL;
    if (ctx->cu.GetErrorString) ctx->cu.GetErrorString(r, &s);
    snprintf(ctx->error, sizeof ctx->error, "%s failed: CUDA error %d (%s)", what, r,
             s ? s : "unknown");
    return -1;
}}

#define CHECK(call, what) do {{ CUresult r_ = (call); if (r_ != 0) {{ fail(ctx, what, r_); goto done; }} }} while (0)

static int load_api({n}_cuda *ctx) {{
    ctx->lib = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!ctx->lib) {{
        snprintf(ctx->error, sizeof ctx->error, "cannot load libcuda.so.1 (the NVIDIA driver): %s",
                 dlerror());
        return -1;
    }}
    struct {{ void **slot; const char *name; }} fns[] = {{
        {{(void **)&ctx->cu.Init, "cuInit"}},
        {{(void **)&ctx->cu.DeviceGet, "cuDeviceGet"}},
        {{(void **)&ctx->cu.DevicePrimaryCtxRetain, "cuDevicePrimaryCtxRetain"}},
        {{(void **)&ctx->cu.DevicePrimaryCtxRelease, "cuDevicePrimaryCtxRelease_v2"}},
        {{(void **)&ctx->cu.CtxPushCurrent, "cuCtxPushCurrent_v2"}},
        {{(void **)&ctx->cu.CtxPopCurrent, "cuCtxPopCurrent_v2"}},
        {{(void **)&ctx->cu.CtxSynchronize, "cuCtxSynchronize"}},
        {{(void **)&ctx->cu.ModuleLoadDataEx, "cuModuleLoadDataEx"}},
        {{(void **)&ctx->cu.ModuleGetFunction, "cuModuleGetFunction"}},
        {{(void **)&ctx->cu.ModuleUnload, "cuModuleUnload"}},
        {{(void **)&ctx->cu.MemGetInfo, "cuMemGetInfo_v2"}},
        {{(void **)&ctx->cu.MemAlloc, "cuMemAlloc_v2"}},
        {{(void **)&ctx->cu.MemFree, "cuMemFree_v2"}},
        {{(void **)&ctx->cu.MemcpyHtoD, "cuMemcpyHtoD_v2"}},
        {{(void **)&ctx->cu.Memcpy2DUnaligned, "cuMemcpy2DUnaligned_v2"}},
        {{(void **)&ctx->cu.LaunchKernel, "cuLaunchKernel"}},
        {{(void **)&ctx->cu.GetErrorString, "cuGetErrorString"}},
    }};
    for (size_t k = 0; k < sizeof fns / sizeof fns[0]; k++) {{
        *fns[k].slot = dlsym(ctx->lib, fns[k].name);
        if (!*fns[k].slot) {{
            snprintf(ctx->error, sizeof ctx->error, "libcuda.so.1 has no %s", fns[k].name);
            return -1;
        }}
    }}
    return 0;
}}

int {n}_cuda_init({n}_cuda **pctx, int device) {{
    {n}_cuda *ctx = calloc(1, sizeof *ctx);
    *pctx = ctx;
    if (!ctx) return -1;
    ctx->max_batch = DEFAULT_BATCH;
    if (load_api(ctx) != 0) return -1;

    int ret = -1;
    int pushed = 0;
    CHECK(ctx->cu.Init(0), "cuInit");
    CHECK(ctx->cu.DeviceGet(&ctx->device, device), "cuDeviceGet");
    CHECK(ctx->cu.DevicePrimaryCtxRetain(&ctx->context, ctx->device), "cuDevicePrimaryCtxRetain");
    ctx->has_context = 1;
    CHECK(ctx->cu.CtxPushCurrent(ctx->context), "cuCtxPushCurrent");
    pushed = 1;

    int options[2] = {{CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES}};
    void *values[2] = {{ctx->log, (void *)(size_t)sizeof ctx->log}};
    CUresult r = ctx->cu.ModuleLoadDataEx(&ctx->module, PTX, 2, options, values);
    if (r != 0) {{
        fail(ctx, "compiling the PTX module (cuModuleLoadDataEx)", r);
        size_t len = strlen(ctx->error);
        snprintf(ctx->error + len, sizeof ctx->error - len, ": %s", ctx->log);
        ctx->module = NULL;
        goto done;
    }}
    CHECK(ctx->cu.ModuleGetFunction(&ctx->kernel, ctx->module, "{n}"), "cuModuleGetFunction");
    ret = 0;
done:
    if (pushed) {{
        CUcontext old;
        ctx->cu.CtxPopCurrent(&old);
    }}
    return ret;
}}

void {n}_cuda_set_batch({n}_cuda *ctx, size_t max_points) {{
    if (ctx) ctx->max_batch = max_points > 0 ? max_points : 1;
}}

const char *{n}_cuda_error(const {n}_cuda *ctx) {{
    return ctx ? ctx->error : "out of memory";
}}

static void free_buffers({n}_cuda *ctx) {{
    if (ctx->d_in) ctx->cu.MemFree(ctx->d_in);
    if (ctx->d_out) ctx->cu.MemFree(ctx->d_out);
    if (ctx->d_params) ctx->cu.MemFree(ctx->d_params);
    ctx->d_in = ctx->d_out = ctx->d_params = 0;
    ctx->capacity = 0;
}}

void {n}_cuda_free({n}_cuda *ctx) {{
    if (!ctx) return;
    if (ctx->has_context) {{
        if (ctx->cu.CtxPushCurrent(ctx->context) == 0) {{
            CUcontext old;
            free_buffers(ctx);
            if (ctx->module) ctx->cu.ModuleUnload(ctx->module);
            ctx->cu.CtxPopCurrent(&old);
        }}
        ctx->cu.DevicePrimaryCtxRelease(ctx->device);
    }}
    if (ctx->lib) dlclose(ctx->lib);
    free(ctx);
}}

/* launches the kernel for n points (in the current context) */
static CUresult launch({n}_cuda *ctx, size_t n, CUdeviceptr in, size_t ld_in, CUdeviceptr out,
                       size_t ld_out, CUdeviceptr params, CUstream stream) {{
    unsigned long long n_ = n, ld_in_ = ld_in, ld_out_ = ld_out;
    void *args[] = {{&in, &out, &params, &n_, &ld_in_, &ld_out_}};
    unsigned blocks = (unsigned)((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    return ctx->cu.LaunchKernel(ctx->kernel, blocks, 1, 1, BLOCK_SIZE, 1, 1, 0, stream, args,
                                NULL);
}}

/* a copy of `rows` rows of `width` bytes between host and device memory */
static CUresult copy2d({n}_cuda *ctx, int to_device, const void *host, size_t host_pitch,
                       CUdeviceptr dev, size_t dev_pitch, size_t width, size_t rows) {{
    if (width == 0 || rows == 0) return 0;
    CUDA_MEMCPY2D c;
    memset(&c, 0, sizeof c);
    if (to_device) {{
        c.srcMemoryType = CU_MEMORYTYPE_HOST;
        c.srcHost = host;
        c.srcPitch = host_pitch;
        c.dstMemoryType = CU_MEMORYTYPE_DEVICE;
        c.dstDevice = dev;
        c.dstPitch = dev_pitch;
    }} else {{
        c.srcMemoryType = CU_MEMORYTYPE_DEVICE;
        c.srcDevice = dev;
        c.srcPitch = dev_pitch;
        c.dstMemoryType = CU_MEMORYTYPE_HOST;
        c.dstHost = (void *)host;
        c.dstPitch = host_pitch;
    }}
    c.WidthInBytes = width;
    c.Height = rows;
    return ctx->cu.Memcpy2DUnaligned(&c);
}}

/* device buffers for `points` points (fewer if memory is short); the points they hold */
static size_t reserve({n}_cuda *ctx, size_t points) {{
    if (points <= ctx->capacity) return ctx->capacity;
    free_buffers(ctx);
    size_t per_point = 8 * (size_t)(COUNT_IN + COUNT_OUT);
    size_t free_bytes = 0, total = 0;
    if (ctx->cu.MemGetInfo(&free_bytes, &total) == 0 && per_point > 0) {{
        size_t fit = free_bytes / 2 / per_point;
        if (fit < points) points = fit > 0 ? fit : 1;
    }}
    while (points > 0) {{
        CUdeviceptr a = 0, b = 0;
        if (ctx->cu.MemAlloc(&a, 8 * (COUNT_IN > 0 ? (size_t)COUNT_IN : 1) * points) == 0) {{
            if (ctx->cu.MemAlloc(&b, 8 * (COUNT_OUT > 0 ? (size_t)COUNT_OUT : 1) * points) == 0) {{
                ctx->d_in = a;
                ctx->d_out = b;
                ctx->capacity = points;
                return points;
            }}
            ctx->cu.MemFree(a);
        }}
        points /= 2;
    }}
    return 0;
}}

int {n}_cuda_evaluate({n}_cuda *ctx, size_t n, const double *in, size_t ld_in, double *out,
                      size_t ld_out, const double *params) {{
    if (!ctx || !ctx->kernel) return -1;
    ctx->error[0] = 0;
    if (n == 0) return 0;
    if (LAYOUT_SOA ? (ld_in < n || ld_out < n) : (ld_in < COUNT_IN || ld_out < COUNT_OUT)) {{
        snprintf(ctx->error, sizeof ctx->error,
                 "ld_in = %zu or ld_out = %zu is too small for n = %zu points", ld_in, ld_out, n);
        return -1;
    }}
    if (COUNT_PARAMS > 0 && !params) {{
        snprintf(ctx->error, sizeof ctx->error, "the model needs %d parameters", COUNT_PARAMS);
        return -1;
    }}

    int ret = -1;
    CUcontext old;
    CUresult r = ctx->cu.CtxPushCurrent(ctx->context);
    if (r != 0) return fail(ctx, "cuCtxPushCurrent", r);

    size_t batch = reserve(ctx, n < ctx->max_batch ? n : ctx->max_batch);
    if (batch > ctx->max_batch) batch = ctx->max_batch;
    if (batch == 0) {{
        snprintf(ctx->error, sizeof ctx->error, "cannot allocate GPU memory");
        goto done;
    }}
    if (COUNT_PARAMS > 0) {{
        if (!ctx->d_params) CHECK(ctx->cu.MemAlloc(&ctx->d_params, 8 * COUNT_PARAMS), "cuMemAlloc");
        CHECK(ctx->cu.MemcpyHtoD(ctx->d_params, params, 8 * COUNT_PARAMS), "cuMemcpyHtoD");
    }}

    for (size_t first = 0; first < n; first += batch) {{
        size_t m = n - first < batch ? n - first : batch;
#if LAYOUT_SOA
        /* COUNT_IN rows of m points; on the device the rows are m apart */
        CHECK(copy2d(ctx, 1, in + first, 8 * ld_in, ctx->d_in, 8 * m, 8 * m, COUNT_IN), "copying the inputs");
        CHECK(launch(ctx, m, ctx->d_in, m, ctx->d_out, m, ctx->d_params, NULL), "cuLaunchKernel");
        CHECK(copy2d(ctx, 0, out + first, 8 * ld_out, ctx->d_out, 8 * m, 8 * m, COUNT_OUT), "copying the outputs");
#else
        /* m rows of COUNT_IN slots; on the device the rows are dense */
        CHECK(copy2d(ctx, 1, in + first * ld_in, 8 * ld_in, ctx->d_in, 8 * COUNT_IN, 8 * COUNT_IN, m), "copying the inputs");
        CHECK(launch(ctx, m, ctx->d_in, COUNT_IN, ctx->d_out, COUNT_OUT, ctx->d_params, NULL), "cuLaunchKernel");
        CHECK(copy2d(ctx, 0, out + first * ld_out, 8 * ld_out, ctx->d_out, 8 * COUNT_OUT, 8 * COUNT_OUT, m), "copying the outputs");
#endif
    }}
    CHECK(ctx->cu.CtxSynchronize(), "running the kernel");
    ret = 0;
done:
    ctx->cu.CtxPopCurrent(&old);
    return ret;
}}

int {n}_cuda_evaluate_device({n}_cuda *ctx, size_t n, unsigned long long in, size_t ld_in,
                             unsigned long long out, size_t ld_out, unsigned long long params,
                             void *stream) {{
    if (!ctx || !ctx->kernel) return -1;
    ctx->error[0] = 0;
    if (n == 0) return 0;
    int ret = -1;
    CUcontext old;
    CUresult r = ctx->cu.CtxPushCurrent(ctx->context);
    if (r != 0) return fail(ctx, "cuCtxPushCurrent", r);
    CHECK(launch(ctx, n, in, ld_in, out, ld_out, params, stream), "cuLaunchKernel");
    ret = 0;
done:
    ctx->cu.CtxPopCurrent(&old);
    return ret;
}}
"#,
        version = env!("CARGO_PKG_VERSION"),
        count_in = shape.count_in,
        count_out = shape.count_out,
        count_params = shape.count_params,
        soa = soa as u8,
        block = BLOCK_SIZE,
        ptx = c_string(ptx).trim_end(),
    )
}
