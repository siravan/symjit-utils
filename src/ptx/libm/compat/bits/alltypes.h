/* sjb2ptx: the types musl's math.h and endian.h need (both targets are LP64, FLT_EVAL_METHOD 0) */
#include <stdint.h>
#if defined(__NEED_double_t) && !defined(__DEFINED_double_t)
typedef float float_t;
typedef double double_t;
#define __DEFINED_double_t
#endif
