#ifndef UTIL_H
#define UTIL_H

#if _MSC_VER
#include <intrin.h>
#endif

// Count trailing zeros
inline unsigned countTrailingZeros(unsigned x)
{
    #if _MSC_VER
    unsigned long bitIdx;
    _BitScanForward(&bitIdx, x);
    return bitIdx;
    #else
    return __builtin_ctz(x);
    #endif
}

#endif // UTIL_H
