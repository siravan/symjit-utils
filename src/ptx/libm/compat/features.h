/* sjb2ptx: musl's src/include/features.h and include/features.h, for a freestanding build
   without symbol aliases (PTX has none) */
#ifndef FEATURES_H
#define FEATURES_H
#define _FEATURES_H
#define __restrict restrict
#define __inline inline
#define __REDIR(x,y) __typeof__(x) x __asm__(#y)
#define weak __attribute__((__weak__))
#define hidden
#define weak_alias(old, new)
#endif
