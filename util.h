#ifndef UTIL_H
#define UTIL_H

#if _MSC_VER
#include <intrin.h>
#endif

namespace omp {

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
    #if _MSC_VER && _M_X64
    return (unsigned)__popcnt64(x);
    #elif _MSC_VER
    return __popcnt((unsigned)x) + __popcnt((unsigned)(x >> 32));
    #else
    return __builtin_popcount(x);
    #endif
}

}

#endif // UTIL_H
