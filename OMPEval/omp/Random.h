#ifndef OMP_RANDOM_H
#define OMP_RANDOM_H

#include "../libdivide/libdivide.h"
#include <cstdint>
#include <climits>

namespace omp {

// Fast 64-bit PRNG with a period of 2^128-1.
class XoroShiro128Plus
{
public:
    typedef uint64_t result_type;

    XoroShiro128Plus(uint64_t seed)
    {
        mState[0] = ~(mState[1] = seed);
    }

    uint64_t operator()()
    {
        uint64_t s0 = mState[0];
        uint64_t s1 = mState[1];
        uint64_t result = s0 + s1;
        s1 ^= s0;
        mState[0] = rotl(s0, 55) ^ s1 ^ (s1 << 14);
        mState[1] = rotl(s1, 36);
        return result;
    }

    static uint64_t min()
    {
        return 0;
    }

    static uint64_t max()
    {
        return ~(uint64_t)0;
    }

private:
    static uint64_t rotl(uint64_t x, unsigned k)
    {
        // MSVC and most g++ versions will compile this to rotl on x64.
        return (x << k) | (x >> (64 - k));
    }

    uint64_t mState[2];
};

// Generates non-repeating pseudo random numbers in specified range using a linear congruential generator.
class UniqueRng64
{
public:
    UniqueRng64(uint64_t range)
        : mRange(range)
    {
        // Find next power of two minus 1
        mMask = range - 1;
        mMask |= mMask >> 1;
        mMask |= mMask >> 2;
        mMask |= mMask >> 4;
        mMask |= mMask >> 8;
        mMask |= mMask >> 16;
        mMask |= mMask >> 32;
    }

    uint64_t operator()(uint64_t idx) const
    {
        do {
            idx = (a * idx + c) & mMask;
        } while (idx >= mRange);
        return idx;
    }

private:
    // Probably not ideal constants in most cases but guarantee a full period for any LCG.
    static const uint64_t a = 4 * 0xbce1fb1361e7685 + 1;
    static const uint64_t c = 0x170a96c613336ed9;
    uint64_t mMask, mRange;
};

// Simple and fast uniform int distribution for small ranges. Has a bias similar to the classic modulo
// method, but it's good enough for most poker simulations.
template<typename T = unsigned, unsigned tBits = 21> // 64 / 3
class FastUniformIntDistribution
{
public:
    typedef T TUnsigned;

    FastUniformIntDistribution()
    {
        init(0, 1);
    }

    FastUniformIntDistribution(T min, T max)
    {
        init(min, max);
    }

    void init(T min, T max)
    {
        mMin = min;
        mDiff = max - min + 1;
        mBuffer = 0;
        mBufferUsesLeft = 0;
    }

    template<class TRng>
    T operator()(TRng& rng)
    {
        static_assert(sizeof(typename TRng::result_type) == sizeof(uint64_t), "64-bit RNG required.");
        if (mBufferUsesLeft == 0) {
            mBuffer = rng();
            mBufferUsesLeft = sizeof(mBuffer) * CHAR_BIT / tBits;
        }
        unsigned res = ((uint64_t)((unsigned)mBuffer & MASK) * mDiff) >> tBits;
        mBuffer >>= tBits;
        --mBufferUsesLeft;
        return mMin + res;
    }

private:
    static const unsigned MASK = (1u << tBits) - 1;

    uint64_t mBuffer;
    unsigned mBufferUsesLeft;
    unsigned mDiff, mMin;
};

// A bit slower distribution without bias (still faster than std::uniform_int_distribution!)
template<typename T = unsigned>
class FastUniformIntDistribution2
{
public:
    typedef T TUnsigned;

    FastUniformIntDistribution2()
    {
        init(0, 1);
    }

    FastUniformIntDistribution2(T min, T max)
    {
        init(min, max);
    }

    void init(T min, T max)
    {
        mMin = min;
        mDiff = max -  min + 1;
        mBufferUsesLeft = 0;
        mFastDivider = libdivide::libdivide_u64_gen(max -  min + 1);
        initializeConstants();
    }

    template<class TRng>
    T operator()(TRng& rng)
    {
        static_assert(sizeof(typename TRng::result_type) == sizeof(uint64_t), "64-bit RNG required.");
        if (mBufferUsesLeft == 0)
            getBits(rng);
        uint64_t quotient = libdivide_u64_do(mBuffer, &mFastDivider);
        TUnsigned remainder = (TUnsigned)(mBuffer - quotient * mDiff);
        mBuffer = quotient;
        --mBufferUsesLeft;
        return mMin + remainder;
    }

private:

    template<class TRng>
    void getBits(TRng& rng)
    {
        do {
            mBuffer = rng() & mMask;
        } while (mBuffer >= mMaxBufferVal);

        mBufferUsesLeft = mMaxBufferUses;
    }

    void initializeConstants()
    {
        // Handle range of 1 as a special case.
        if (mDiff <= 1) {
            mMask = mMaxBufferVal = -1;
            mMaxBufferUses = -1;
            return;
        }

        // Find biggest power of mDiff without overflow.
        mMaxBufferUses = 1;
        mMask = -1;
        uint64_t diffPow = mDiff;
        while (mMask / diffPow >= mDiff) {
            ++mMaxBufferUses;
            diffPow *= mDiff;
        }
        // Check if multiplying one more time gives exactly 2^64
        if ((diffPow & 0xffffffffull) == 0 && ((diffPow >> 32) * mDiff == 0x100000000ull)) {
            ++mMaxBufferUses;
            diffPow *= mDiff;
        }
        mMaxBufferVal = diffPow - 1;

        while (((mMask >> 1) & mMaxBufferVal) == mMaxBufferVal)
            mMask >>= 1;
    }

    libdivide::libdivide_u64_t mFastDivider;
    T mMin;
    TUnsigned mDiff;
    uint64_t mBuffer, mMaxBufferVal, mMask;
    unsigned mBufferUsesLeft, mMaxBufferUses;
};

}

#endif // OMP_RANDOM_H
