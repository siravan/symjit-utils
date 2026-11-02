#define READ_FIRST 1u
#define WRITTEN 2u
#define COPY_BACK 4u /* written and copied back */

/* the GPU, opened once: the device, its queue, one pipeline per leaf, and the arena, in blocks of
   BLOCK_POINTS points: block b holds NUM_PLANES f32 planes of BLOCK_POINTS points (point j of
   plane p at at(p, j)), and LO the low parts of the planes FIRST_F64 .. NUM_PLANES (mixed) in
   blocks of LO_PLANES planes. A kernel runs on one block, its planes BLOCK_POINTS apart: short
   strides keep a block's planes in the caches and the TLB (as a call of BLOCK_POINTS points). */
static int STATE; /* 0: not opened yet, 1: ready, -1: failed */
static id<MTLDevice> DEVICE;
static id<MTLCommandQueue> QUEUE;
static NSArray<id<MTLComputePipelineState>> *PIPELINES;
static id<MTLBuffer> ARENA, LO;
static size_t BLOCKS; /* the blocks the arena has room for */
static uint32_t BLOCK_POINTS;
#define LO_PLANES (MIXED && NUM_PLANES > FIRST_F64 ? NUM_PLANES - FIRST_F64 : 1u)

/* the arena's float of plane p, point j (of the call) */
static inline size_t at(uint32_t p, uint32_t j) {
    return ((size_t)(j / BLOCK_POINTS) * NUM_PLANES + p) * BLOCK_POINTS + j % BLOCK_POINTS;
}
static int COPY_ALL; /* PACMAN_RUNNER_COPY_BACK=all or PACMAN_SAVE_ARENA: every written plane */
/* one call at a time: the arena is shared */
static pthread_mutex_t LOCK = PTHREAD_MUTEX_INITIALIZER;
static unsigned char FLAGS[NUM_PLANES];

/* PACMAN_RUNNER_STATS=1: calls, points and time, printed when the library is unloaded */
static struct {
    int on;
    unsigned long long calls, points, blocks;
    double seconds, gpu, convert, open;
} STATS;

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

__attribute__((destructor)) static void print_stats(void) {
    if (STATS.on && STATS.calls > 0) {
        fprintf(stderr,
                "pacman runner (%s of %s, Metal%s): %llu calls, %llu points in %llu blocks of %u, "
                "%.3f s (%.2f us/point; GPU %.3f s, conversions %.3f s), opened in %.2f s\n",
                METADATA.execution, METADATA.process, MIXED ? ", mixed f32/df64" : ", f32",
                STATS.calls, STATS.points, STATS.blocks, BLOCK_POINTS, STATS.seconds,
                STATS.points ? 1e6 * STATS.seconds / STATS.points : 0.0, STATS.gpu,
                STATS.convert, STATS.open);
    }
}

static void fail(const char *step, NSError *e) {
    fprintf(stderr, "pacman runner (%s of %s): %s failed%s%s\n", METADATA.execution,
            METADATA.process, step, e != nil ? ": " : "",
            e != nil ? e.localizedDescription.UTF8String : "");
}

/* the device, the leaves' source compiled (Metal caches it: the first time takes long) and
   their pipelines, once */
static int metal_open(void) {
    if (STATE != 0) {
        return STATE;
    }
    STATE = -1;
    double started = now();
    const char *stats = getenv("PACMAN_RUNNER_STATS");
    STATS.on = stats != NULL && stats[0] != '\0' && strcmp(stats, "0") != 0;
    const char *copy_back = getenv("PACMAN_RUNNER_COPY_BACK");
    const char *capture = getenv("PACMAN_SAVE_ARENA");
    COPY_ALL = (copy_back != NULL && strcmp(copy_back, "all") == 0) ||
               (capture != NULL && capture[0] != '\0');
    const char *block = getenv("PACMAN_RUNNER_BLOCK");
    BLOCK_POINTS = block != NULL ? (uint32_t)strtoul(block, NULL, 10) : 4096u;
    if (BLOCK_POINTS < 1) BLOCK_POINTS = 4096u;
    @autoreleasepool {
        DEVICE = MTLCreateSystemDefaultDevice();
        if (DEVICE == nil) {
            fprintf(stderr, "pacman runner: no Metal device\n");
            return -1;
        }
        NSString *source = [[NSString alloc]
            initWithBytesNoCopy:(void *)pacman_metal_source
                         length:(NSUInteger)(pacman_metal_source_end - pacman_metal_source)
                       encoding:NSUTF8StringEncoding
                   freeWhenDone:NO];
        MTLCompileOptions *options = [MTLCompileOptions new];
        if (@available(macOS 15.0, *)) {
            options.mathMode = MTLMathModeSafe;
            options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
        } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
            options.fastMathEnabled = NO;
#pragma clang diagnostic pop
        }
        NSError *e = nil;
        id<MTLLibrary> library = [DEVICE newLibraryWithSource:source options:options error:&e];
        if (library == nil) {
            fail("compiling the kernels", e);
            return -1;
        }
        NSMutableArray<id<MTLComputePipelineState>> *pipelines =
            [NSMutableArray arrayWithCapacity:NUM_LEAVES];
        for (uint32_t k = 0; k < NUM_LEAVES; k++) {
            NSString *name = [NSString stringWithFormat:@"pacman_leaf%u", k];
            id<MTLFunction> f = [library newFunctionWithName:name];
            if (f == nil) {
                fail("finding a leaf's kernel", nil);
                return -1;
            }
            id<MTLComputePipelineState> p = [DEVICE newComputePipelineStateWithFunction:f
                                                                                  error:&e];
            if (p == nil) {
                fail("building a leaf's pipeline", e);
                return -1;
            }
            [pipelines addObject:p];
        }
        PIPELINES = pipelines;
        QUEUE = [DEVICE newCommandQueue];
    }
    STATS.open = now() - started;
    STATE = 1;
    return 1;
}

const pacman_runner_metadata *pacman_runner_v1_metadata(void) {
    return &METADATA;
}

/* the reduction of the f32 arena's amplitudes (as reduce_host on the widened values) */
static void reduce_arena(const pacman_runner_reduction *r, uint32_t count, double *out) {
    const float *arena = ARENA.contents;
    uint32_t re0 = METADATA.regions[PACMAN_AMPLITUDE_RE][0];
    uint32_t im0 = METADATA.regions[PACMAN_AMPLITUDE_IM][0];
    for (uint32_t i = 0; i < count; i++) out[i] = 0.0;
    for (uint32_t g = 0; g < r->num_groups; g++) {
        double weight = r->weights[g];
        for (uint32_t i = 0; i < count; i++) {
            double re = 0.0, im = 0.0;
            for (uint32_t k = r->offsets[g]; k < r->offsets[g + 1]; k++) {
                re += (double)arena[at(re0 + r->indices[k], i)];
                im += (double)arena[at(im0 + r->indices[k], i)];
            }
            out[i] += weight * (re * re + im * im);
        }
    }
}

/* the arena for `count` points */
static int reserve(uint32_t count) {
    size_t blocks = ((size_t)count + BLOCK_POINTS - 1) / BLOCK_POINTS;
    if (blocks <= BLOCKS) return 0;
    ARENA = nil;
    LO = nil;
    BLOCKS = 0;
    size_t bytes = blocks * NUM_PLANES * BLOCK_POINTS * sizeof(float);
    ARENA = [DEVICE newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (ARENA == nil) {
        fprintf(stderr, "pacman runner: cannot allocate an arena of %.1f MB on the GPU (at most %.1f MB)\n",
                bytes * 1e-6, DEVICE.maxBufferLength * 1e-6);
        return -1;
    }
    LO = [DEVICE newBufferWithLength:blocks * LO_PLANES * BLOCK_POINTS * sizeof(float)
                             options:MTLResourceStorageModeShared];
    if (LO == nil) {
        ARENA = nil;
        fprintf(stderr, "pacman runner: cannot allocate the low parts on the GPU\n");
        return -1;
    }
    BLOCKS = blocks;
    return 0;
}

static void dispatch(id<MTLComputeCommandEncoder> en, id<MTLComputePipelineState> p, uint32_t n) {
    NSUInteger tg = p.maxTotalThreadsPerThreadgroup;
    if (tg > 256) tg = 256;
    [en setComputePipelineState:p];
    [en dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
}

/* pacman_runner_v1_run, and with `reduction` pacman_runner_v1_run_reduce; under LOCK */
static int run(double *const *planes, const uint32_t *leaves, uint32_t count,
               uint32_t point_start, uint32_t point_count,
               const pacman_runner_reduction *reduction, double *out) {
    if (metal_open() != 1) {
        return PACMAN_RUNNER_DEVICE_ERROR;
    }
    /* the planes to convert in (read before the leaves write them: RustiCol writes them before
       the call) and back (written; by default only the amplitudes, which RustiCol reduces; the
       currents are scratch) */
    memset(FLAGS, 0, sizeof FLAGS);
    for (uint32_t k = 0; k < count; k++) {
        uint32_t leaf = leaves == NULL ? k : leaves[k];
        const uint32_t *map = LEAF_PLANES[leaf];
        for (uint32_t i = 0; i < LEAF_INPUTS[leaf]; i++) {
            if (!(FLAGS[map[i]] & WRITTEN)) FLAGS[map[i]] |= READ_FIRST;
        }
        for (uint32_t i = LEAF_INPUTS[leaf]; i < LEAF_INPUTS[leaf] + LEAF_OUTPUTS[leaf]; i++) {
            FLAGS[map[i]] |= WRITTEN;
        }
    }
    uint32_t amplitudes = METADATA.regions[PACMAN_AMPLITUDE_RE][0];
    uint32_t amplitudes_end =
        METADATA.regions[PACMAN_AMPLITUDE_IM][0] + METADATA.regions[PACMAN_AMPLITUDE_IM][1];
    for (uint32_t p = 0; p < NUM_PLANES; p++) {
        if ((FLAGS[p] & WRITTEN) && (COPY_ALL || (p >= amplitudes && p < amplitudes_end))) {
            FLAGS[p] |= COPY_BACK;
        }
    }

    double started = now();
    if (reserve(point_count) != 0) {
        return PACMAN_RUNNER_DEVICE_ERROR;
    }
    /* in: rounded to f32; a df64 input x is (f32(x), f32(x - f32(x))) */
    float *arena = ARENA.contents, *lo = LO.contents;
    for (uint32_t b = 0; b < point_count; b += BLOCK_POINTS) {
        uint32_t n = point_count - b < BLOCK_POINTS ? point_count - b : BLOCK_POINTS;
        for (uint32_t p = 0; p < NUM_PLANES; p++) {
            if (!(FLAGS[p] & READ_FIRST)) continue;
            const double *src = planes[p] + point_start + b;
            float *hi = arena + at(p, b);
            if (MIXED && p >= FIRST_F64) {
                float *low = lo + ((size_t)(b / BLOCK_POINTS) * LO_PLANES + (p - FIRST_F64)) *
                                      BLOCK_POINTS;
                for (uint32_t i = 0; i < n; i++) {
                    float h = (float)src[i];
                    hi[i] = h;
                    low[i] = (float)(src[i] - (double)h);
                }
            } else {
                for (uint32_t i = 0; i < n; i++) hi[i] = (float)src[i];
            }
        }
    }
    double converted = now();
    STATS.convert += converted - started;

    /* the leaves, block by block, in one command buffer (a serial encoder: each dispatch sees
       the writes of the ones before) */
    int status = PACMAN_RUNNER_OK;
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [QUEUE commandBuffer];
        id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
        uint32_t ld = BLOCK_POINTS;
        [en setBytes:&ld length:sizeof ld atIndex:2];
        for (uint32_t b = 0; b < point_count; b += BLOCK_POINTS) {
            uint32_t n = point_count - b < BLOCK_POINTS ? point_count - b : BLOCK_POINTS;
            size_t block = b / BLOCK_POINTS;
            [en setBuffer:ARENA offset:block * NUM_PLANES * BLOCK_POINTS * sizeof(float) atIndex:0];
            [en setBytes:&n length:sizeof n atIndex:1];
            [en setBuffer:LO
                   offset:MIXED ? block * LO_PLANES * BLOCK_POINTS * sizeof(float) : 0
                  atIndex:3];
            for (uint32_t k = 0; k < count; k++) {
                dispatch(en, PIPELINES[leaves == NULL ? k : leaves[k]], n);
            }
            STATS.blocks++;
        }
        [en endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted) {
            fail("running the leaves", cb.error);
            status = PACMAN_RUNNER_DEVICE_ERROR;
        } else {
            STATS.gpu += cb.GPUEndTime - cb.GPUStartTime;
        }
    }
    if (status != PACMAN_RUNNER_OK) return status;

    /* out: widened (exact) */
    double finished = now();
    if (reduction != NULL) {
        reduce_arena(reduction, point_count, out);
    }
    if (reduction == NULL || COPY_ALL) {
        for (uint32_t b = 0; b < point_count; b += BLOCK_POINTS) {
            uint32_t n = point_count - b < BLOCK_POINTS ? point_count - b : BLOCK_POINTS;
            for (uint32_t p = 0; p < NUM_PLANES; p++) {
                if (!(FLAGS[p] & COPY_BACK)) continue;
                double *dst = planes[p] + point_start + b;
                const float *hi = arena + at(p, b);
                for (uint32_t i = 0; i < n; i++) dst[i] = (double)hi[i];
            }
        }
    }
    double done = now();
    STATS.convert += done - finished;
    STATS.calls++;
    STATS.points += point_count;
    STATS.seconds += done - started;
    return PACMAN_RUNNER_OK;
}

static int checked_run(double *const *planes, uint32_t num_planes, const uint32_t *leaves,
                       uint32_t num_leaves, uint32_t point_start, uint32_t point_count,
                       const pacman_runner_reduction *reduction, double *out) {
    if (planes == NULL || num_planes != NUM_PLANES || point_count == 0 ||
        point_start > UINT32_MAX - point_count) {
        return PACMAN_RUNNER_BAD_ARGUMENTS;
    }
    for (uint32_t p = 0; p < NUM_PLANES; p++) {
        if (planes[p] == NULL) {
            return PACMAN_RUNNER_BAD_ARGUMENTS;
        }
    }
    uint32_t count = leaves == NULL ? NUM_LEAVES : num_leaves;
    for (uint32_t k = 0; k < count; k++) {
        if (leaves != NULL && leaves[k] >= NUM_LEAVES) {
            return PACMAN_RUNNER_BAD_ARGUMENTS;
        }
    }
    pthread_mutex_lock(&LOCK);
    int status = run(planes, leaves, count, point_start, point_count, reduction, out);
    pthread_mutex_unlock(&LOCK);
    return status;
}

int pacman_runner_v1_run(double *const *planes, uint32_t num_planes, const uint32_t *leaves,
                         uint32_t num_leaves, uint32_t point_start, uint32_t point_count) {
    return checked_run(planes, num_planes, leaves, num_leaves, point_start, point_count, NULL,
                       NULL);
}

int pacman_runner_v1_run_reduce(double *const *planes, uint32_t num_planes,
                                const uint32_t *leaves, uint32_t num_leaves, uint32_t point_start,
                                uint32_t point_count, const pacman_runner_reduction *reduction,
                                double *out) {
    if (!reduction_ok(reduction, out)) {
        return PACMAN_RUNNER_BAD_ARGUMENTS;
    }
    return checked_run(planes, num_planes, leaves, num_leaves, point_start, point_count,
                       reduction, out);
}
