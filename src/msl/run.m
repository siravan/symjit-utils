/* run.m: runs Metal kernels of sjb2msl (src/msl/mod.rs). The source is compiled at run time
 * (newLibraryWithSource, MTLMathModeSafe: IEEE arithmetic, no fast math), so only the Metal
 * framework is needed, not Xcode.
 *
 *   cc -O2 -fobjc-arc run.m -framework Metal -framework Foundation -o run
 *
 *   run arena SOURCE.metal KERNELS.txt ARENA.f32 LD N REPEAT OUT.f32 [LO.f32]
 *       kernels of arena planes (MslOptions::mapping), one name per line of KERNELS.txt, run in
 *       order on points 0..N of a plane-major f32 arena of LD points per plane; the arena is
 *       restored before each of REPEAT runs, the last one written to OUT.f32
 *   run plain SOURCE.metal NAME IN.f32 COUNT_IN COUNT_OUT PARAMS.f32 N OUT.f32 [LO.f32]
 *       one kernel on N points: inputs SoA (in[k*N + i]), outputs SoA, PARAMS.f32 the shared
 *       parameters (may be empty)
 * LO.f32 (mixed precision): the low parts of the df64 inputs, laid out as the arena or IN.f32.
 *
 * Prints `compile <s> s, <k> pipelines <s> s` and, for `arena`, `best <s> s of <REPEAT> runs`
 * (GPU time of one command buffer holding every dispatch). RUN_TG=T sets the threadgroup size
 * (default: the pipeline's maximum, at most 256); RUN_EACH=1 also times each kernel alone.
 * Exit status 77: no Metal device. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void *slurp(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "run: cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *p = malloc(n + 1);
    if (n > 0 && fread(p, 1, n, f) != (size_t)n) { fprintf(stderr, "run: cannot read %s\n", path); exit(1); }
    p[n] = 0;
    fclose(f);
    *len = n;
    return p;
}

static void spill(const char *path, const void *p, size_t len) {
    FILE *f = fopen(path, "wb");
    if (!f || fwrite(p, 1, len, f) != len) { fprintf(stderr, "run: cannot write %s\n", path); exit(1); }
    fclose(f);
}

static double now(void) { return [[NSDate date] timeIntervalSince1970]; }

static id<MTLLibrary> compile(id<MTLDevice> d, const char *path) {
    size_t len;
    char *src = slurp(path, &len);
    MTLCompileOptions *o = [MTLCompileOptions new];
    if (@available(macOS 15.0, *)) {
        o.mathMode = MTLMathModeSafe;
        o.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
    } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        o.fastMathEnabled = NO;
#pragma clang diagnostic pop
    }
    NSError *e = nil;
    double t0 = now();
    id<MTLLibrary> lib = [d newLibraryWithSource:[NSString stringWithUTF8String:src] options:o error:&e];
    if (!lib) { fprintf(stderr, "run: %s: %s\n", path, e.localizedDescription.UTF8String); exit(1); }
    printf("compile %.2f s\n", now() - t0);
    free(src);
    return lib;
}

static id<MTLComputePipelineState> pipeline(id<MTLDevice> d, id<MTLLibrary> lib, const char *name) {
    id<MTLFunction> f = [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
    if (!f) { fprintf(stderr, "run: no kernel %s\n", name); exit(1); }
    NSError *e = nil;
    id<MTLComputePipelineState> p = [d newComputePipelineStateWithFunction:f error:&e];
    if (!p) { fprintf(stderr, "run: %s: %s\n", name, e.localizedDescription.UTF8String); exit(1); }
    return p;
}

static void dispatch(id<MTLComputeCommandEncoder> en, id<MTLComputePipelineState> p, uint32_t n) {
    NSUInteger tg = p.maxTotalThreadsPerThreadgroup;
    const char *env = getenv("RUN_TG");
    if (env && atoi(env) > 0) tg = (NSUInteger)atoi(env);
    if (tg > 256) tg = 256;
    if (tg > p.maxTotalThreadsPerThreadgroup) tg = p.maxTotalThreadsPerThreadgroup;
    [en setComputePipelineState:p];
    [en dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
}

static double gpu_time(id<MTLCommandBuffer> cb) {
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted) {
        fprintf(stderr, "run: the command buffer failed: %s\n", cb.error.localizedDescription.UTF8String);
        exit(3);
    }
    return cb.GPUEndTime - cb.GPUStartTime;
}

static int arena(id<MTLDevice> d, int argc, char **argv) {
    if (argc != 9 && argc != 10) { fprintf(stderr, "usage: run arena SOURCE KERNELS.txt ARENA.f32 LD N REPEAT OUT.f32 [LO.f32]\n"); return 2; }
    id<MTLLibrary> lib = compile(d, argv[2]);
    size_t len;
    char *list = slurp(argv[3], &len);
    NSMutableArray *ps = [NSMutableArray array];
    NSMutableArray *names = [NSMutableArray array];
    double t0 = now();
    for (char *line = strtok(list, "\n"); line; line = strtok(NULL, "\n")) {
        if (!*line) continue;
        [ps addObject:pipeline(d, lib, line)];
        [names addObject:[NSString stringWithUTF8String:line]];
    }
    printf("%lu pipelines %.2f s\n", (unsigned long)ps.count, now() - t0);
    uint32_t ld = (uint32_t)strtoul(argv[5], NULL, 10), n = (uint32_t)strtoul(argv[6], NULL, 10);
    int repeat = atoi(argv[7]);
    size_t bytes;
    float *init = slurp(argv[4], &bytes);
    if (bytes % ((size_t)ld * 4) || n > ld) { fprintf(stderr, "run: an arena of %zu bytes for LD %u, N %u\n", bytes, ld, n); return 1; }
    id<MTLBuffer> buf = [d newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> lo = buf;
    if (argc == 10) {
        size_t lo_bytes;
        float *l = slurp(argv[9], &lo_bytes);
        if (lo_bytes != bytes) { fprintf(stderr, "run: %s holds %zu bytes, the arena %zu\n", argv[9], lo_bytes, bytes); return 1; }
        lo = [d newBufferWithBytes:l length:lo_bytes options:MTLResourceStorageModeShared];
        free(l);
    }
    id<MTLCommandQueue> q = [d newCommandQueue];
    double best = 1e30;
    for (int r = 0; r < repeat; r++) {
        memcpy(buf.contents, init, bytes);
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
        [en setBuffer:buf offset:0 atIndex:0];
        [en setBytes:&n length:4 atIndex:1];
        [en setBytes:&ld length:4 atIndex:2];
        [en setBuffer:lo offset:0 atIndex:3];
        for (id<MTLComputePipelineState> p in ps) dispatch(en, p, n);
        [en endEncoding];
        double t = gpu_time(cb);
        if (t < best) best = t;
    }
    printf("best %.6f s of %d runs, %.4f us/point\n", best, repeat, best / n * 1e6);
    spill(argv[8], buf.contents, bytes);
    if (getenv("RUN_EACH")) {
        /* each kernel in its own command buffer, on the final arena (the inputs of every
         * kernel hold their values; only the time matters) */
        for (NSUInteger k = 0; k < ps.count; k++) {
            double t = 1e30;
            for (int r = 0; r < (repeat < 3 ? repeat : 3); r++) {
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
                [en setBuffer:buf offset:0 atIndex:0];
                [en setBytes:&n length:4 atIndex:1];
                [en setBytes:&ld length:4 atIndex:2];
        [en setBuffer:lo offset:0 atIndex:3];
                dispatch(en, ps[k], n);
                [en endEncoding];
                double g = gpu_time(cb);
                if (g < t) t = g;
            }
            id<MTLComputePipelineState> p = ps[k];
            printf("kernel %s %.6f s (threadgroup max %lu, width %lu)\n", [names[k] UTF8String], t,
                   (unsigned long)p.maxTotalThreadsPerThreadgroup, (unsigned long)p.threadExecutionWidth);
        }
    }
    return 0;
}

static int plain(id<MTLDevice> d, int argc, char **argv) {
    if (argc != 10 && argc != 11) { fprintf(stderr, "usage: run plain SOURCE NAME IN.f32 COUNT_IN COUNT_OUT PARAMS.f32 N OUT.f32 [LO.f32]\n"); return 2; }
    id<MTLLibrary> lib = compile(d, argv[2]);
    id<MTLComputePipelineState> p = pipeline(d, lib, argv[3]);
    uint32_t count_in = (uint32_t)strtoul(argv[5], NULL, 10), count_out = (uint32_t)strtoul(argv[6], NULL, 10);
    uint32_t n = (uint32_t)strtoul(argv[8], NULL, 10);
    size_t in_len, par_len;
    float *in = slurp(argv[4], &in_len);
    float *par = slurp(argv[7], &par_len);
    if (in_len != (size_t)count_in * n * 4) { fprintf(stderr, "run: %zu input bytes, expected %u x %u floats\n", in_len, count_in, n); return 1; }
    /* Metal buffers cannot be empty */
    id<MTLBuffer> bin = [d newBufferWithBytes:in length:in_len ? in_len : 4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bpar = [d newBufferWithBytes:par length:par_len ? par_len : 4 options:MTLResourceStorageModeShared];
    size_t out_len = (size_t)count_out * n * 4;
    id<MTLBuffer> bout = [d newBufferWithLength:out_len ? out_len : 4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> blo = bin;
    if (argc == 11) {
        size_t lo_len;
        float *l = slurp(argv[10], &lo_len);
        if (lo_len != in_len) { fprintf(stderr, "run: %s holds %zu bytes, the inputs %zu\n", argv[10], lo_len, in_len); return 1; }
        blo = [d newBufferWithBytes:l length:lo_len ? lo_len : 4 options:MTLResourceStorageModeShared];
    }
    id<MTLCommandQueue> q = [d newCommandQueue];
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
    [en setBuffer:blo offset:0 atIndex:6];
    [en setBuffer:bin offset:0 atIndex:0];
    [en setBuffer:bout offset:0 atIndex:1];
    [en setBuffer:bpar offset:0 atIndex:2];
    [en setBytes:&n length:4 atIndex:3];
    [en setBytes:&n length:4 atIndex:4];
    [en setBytes:&n length:4 atIndex:5];
    dispatch(en, p, n);
    [en endEncoding];
    gpu_time(cb);
    spill(argv[9], bout.contents, out_len);
    return 0;
}

int main(int argc, char **argv) {
    @autoreleasepool {
        id<MTLDevice> d = MTLCreateSystemDefaultDevice();
        if (!d) { fprintf(stderr, "run: no Metal device\n"); return 77; }
        if (argc > 1 && !strcmp(argv[1], "arena")) return arena(d, argc, argv);
        if (argc > 1 && !strcmp(argv[1], "plain")) return plain(d, argc, argv);
        fprintf(stderr, "usage: run arena|plain ... (see the top of run.m)\n");
        return 2;
    }
}
