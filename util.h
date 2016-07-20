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

inline unsigned bitCount(unsigned x)
{
    #if _MSC_VER
    return __popcnt(x);
    #else
    return __builtin_popcount(x);
    #endif
}

inline unsigned bitCount(unsigned long long x)
{
    #if _MSC_VER
    return __popcnt64(x);
    #else
    return __builtin_popcount(x);
    #endif
}

#endif // UTIL_H
