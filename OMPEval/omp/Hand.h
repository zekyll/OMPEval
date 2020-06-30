#ifndef OMP_HAND_H
#define OMP_HAND_H

#include "Util.h"
#include "Constants.h"
#include <array>
#include <memory>
#include <cstdint>
#include <cassert>
#if OMP_SSE2
    #include <xmmintrin.h> // SSE1
    #include <emmintrin.h> // SSE2
    #if OMP_SSE4
        #include <smmintrin.h> // SSE4.1
    #endif
#endif

namespace omp {

// Structure that combines the data from multiple cards so that hand strength can be evaluated efficiently.
// It is essential when combining hands that exactly one of them was initialized using Hand::empty().
struct Hand
{
    // Default constructor. Leaves the struct uninitialized for performance reasons.
    Hand()
    {
        #if OMP_SSE2
        omp_assert((uintptr_t)&mData % sizeof(__m128i) == 0);
        #endif
    }

    // Copy constructor.
    Hand(const Hand& other)
    {
        #if OMP_SSE2
        omp_assert((uintptr_t)&mData % sizeof(__m128i) == 0);
        #endif
        *this = other;
    }

    // Create a Hand from a card. CardIdx is an integer between 0 and 51, so that CARD = 4 * RANK + SUIT, where
    // rank ranges from 0 (deuce) to 12 (ace) and suit is from 0 (spade) to 3 (diamond).
    Hand(unsigned cardIdx)
    {
        #if OMP_SSE2
        omp_assert((uintptr_t)&mData % sizeof(__m128i) == 0);
        #endif
        omp_assert(cardIdx < CARD_COUNT);
        *this = CARDS[cardIdx];
    }

    // Initialize hand from two cards.
    Hand(std::array<uint8_t,2> holeCards)
    {
        #if OMP_SSE2
        omp_assert((uintptr_t)&mData % sizeof(__m128i) == 0);
        #endif
        omp_assert(holeCards[0] < CARD_COUNT && holeCards[1] < CARD_COUNT);
        *this = CARDS[holeCards[0]] + CARDS[holeCards[1]];
    }

    // Combine with another hand.
    Hand operator+(const Hand& hand2) const
    {
        Hand ret = *this;
        ret += hand2;
        return ret;
    }

    // Combine with another hand.
    Hand& operator+=(const Hand& hand2)
    {
        omp_assert(!(mask() & hand2.mask()));
        #if OMP_SSE2
        mData = _mm_add_epi32(mData, hand2.mData); // sse2
        #else
        mKey += hand2.mKey;
        mMask |= hand2.mMask;
        #endif
        return *this;
    }

    // Remove cards from this hand.
    Hand operator-(const Hand& hand2) const
    {
        Hand ret = *this;
        ret -= hand2;
        return ret;
    }

    // Remove cards from this hand.
    Hand& operator-=(const Hand& hand2)
    {
        omp_assert((mask() & hand2.mask()) == hand2.mask());
        #if OMP_SSE2
        mData = _mm_sub_epi32(mData, hand2.mData); // sse2
        #else
        mKey -= hand2.mKey;
        mMask -= hand2.mMask;
        #endif
        return *this;
    }

    // Equality comparison.
    bool operator==(const Hand& hand2) const
    {
        return mask() == hand2.mask() && key() == hand2.key();
    }

    // Initialize a new empty hand.
    static Hand empty()
    {
        // Initializes suit counters to 3 so that the flush check bits gets set by the 5th suited card.
        return EMPTY;
    }

    // Number of cards for specific suit.
    unsigned suitCount(unsigned suit) const
    {
        return (counters() >> (4 * suit + (SUITS_SHIFT - 32)) & 0xf) - 3;
    }

    // Total number of cards.
    unsigned count() const
    {
        return (counters() >> (CARD_COUNT_SHIFT - 32)) & 0xf;
    }

    // Returns true if hand has 5 or more cards of the same suit.
    bool hasFlush() const
    {
        // Hand has a 4-bit counter for each suit. They start at 3 so the 4th bit gets set when
        // there is 5 or more cards of that suit. We can check for flush by simply masking
        // out the flush check bits.

        #if OMP_SSE4 && !OMP_X64
        // On 32-bit systems it's faster to use PTEST.
        const __m128i fcmask = _mm_set_epi32(0, 0, FLUSH_CHECK_MASK32, 0); // sse2
        return !_mm_testz_si128(mData, fcmask); // sse4.1
        #else
        return !!(key() & FLUSH_CHECK_MASK64);
        #endif
    }

    // Returns a 32-bit key that is unique for each card rank combination.
    uint32_t rankKey() const
    {
        #if OMP_SSE4 && !OMP_X64
        return _mm_extract_epi32(mData, 0); // sse4.1
        #else
        return (uint32_t)key();
        #endif
    }

    // Returns a card mask for the suit that has 5 or more cards.
    uint16_t flushKey() const
    {
        // Get the index of the flush check bit and use it to get the card mask for that suit.
        unsigned flushCheckBits = counters() & FLUSH_CHECK_MASK32;
        unsigned shift = countLeadingZeros(flushCheckBits) << 2;
        #if OMP_SSE2
        return _mm_extract_epi16(_mm_srli_epi64(mData, shift), 4); // sse2
        #else
        return mMask >> shift;
        #endif
    }

private:
    static Hand CARDS[CARD_COUNT];
    static const Hand EMPTY;
    static const unsigned CARD_COUNT_SHIFT = 32;
    static const unsigned SUITS_SHIFT = 48;
    static const uint64_t FLUSH_CHECK_MASK64 = 0x8888ull << SUITS_SHIFT;
    static const uint32_t FLUSH_CHECK_MASK32 = 0x8888ull << (SUITS_SHIFT - 32);

    // Returns the counters.
    uint32_t counters() const
    {
        #if OMP_SSE4
        return _mm_extract_epi32(mData, 1); // sse4.1
        #else
        return key() >> 32;
        #endif
    }

    // Low 64-bits. (Key & counters.)
    uint64_t key() const
    {
        #if OMP_SSE2 && OMP_X64
        return _mm_cvtsi128_si64(mData); // sse2, x64 only
        #elif OMP_SSE2
        uint64_t ret;
        _mm_storel_pi((__m64*)&ret, *(__m128*)&mData); // sse, x86 only
        return ret;
        #else
        return mKey;
        #endif
    }

    // High 64-bits.
    uint64_t mask() const
    {
        #if OMP_SSE4 && OMP_X64
        return _mm_extract_epi64(mData, 1); // sse4.1, x64 only
        #elif OMP_SSE2
        uint64_t ret;
        _mm_storeh_pi((__m64*)&ret, *(__m128*)&mData); // sse, x86 only
        return ret;
        #else
        return mMask;
        #endif
    }

    Hand(uint64_t key, uint64_t mask)
    {
        #if OMP_SSE2
        omp_assert((uintptr_t)this % sizeof(Hand) == 0);
        union {
            uint64_t a[2];
            __m128i b;
        };
        a[0] = key;
        a[1] = mask;
        mData = b;
        #else
        mKey = key;
        mMask = mask;
        #endif
    }

    // Bits 0-31: Key to non-flush lookup table (linear combination of the rank constants).
    // Bits 32-35: Card counter.
    // Bits 48-63: Suit counters.
    // Bits 64-128: Bit mask for all cards (suits are in 16-bit groups).
    #if OMP_SSE2
    __m128i mData;
    #else
    uint64_t mKey;
    uint64_t mMask;
    #endif

    friend class HandEvaluator;
};

}

// Some 32-bit compilers don't align __m128i correctly inside containers, which causes segfault. We need a
// custom allocator to fix the alignment.
OMP_ALIGNED_STD_ALLOCATOR(omp::Hand)

#endif // OMP_HAND_H
