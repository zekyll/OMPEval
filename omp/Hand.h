#ifndef HAND
#define HAND

#include "Util.h"
#include "Constants.h"
#include <array>
#include <cstdint>
#include <cassert>
#if OMP_SSE4
    #include <xmmintrin.h>
    #include <smmintrin.h>
#endif

namespace omp {

// Structure that combines the data from multiple cards so that hand strength can be evaluated efficiently.
// It is essential when combining hands that exactly one of them was initialized using Hand::empty().
struct Hand
{
    // Default constructor. Leaves the struct uninitialized for performance reasons.
    Hand()
    {
    }

    // Create a Hand from a card. CardIdx is an integer between 0 and 51, so that CARD = 4 * RANK + SUIT, where
    // rank ranges from 0 (deuce) to 12 (ace) and suit is from 0 (spade) to 3 (diamond).
    Hand(unsigned cardIdx)
    {
        omp_assert(cardIdx < CARD_COUNT);
        *this = CARDS[cardIdx];
    }

    // Initialize hand from two cards.
    Hand(std::array<uint8_t,2> holeCards)
    {
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
        #if OMP_SSE4
        mData = _mm_add_epi64(mData, hand2.mData);
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
        #if OMP_SSE4
        mData = _mm_sub_epi64(mData, hand2.mData);
        #else
        mKey -= hand2.mKey;
        mMask -= hand2.mMask;
        #endif
        return *this;
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
        return (suits() >> (4 * suit) & 0xf) - 3;
    }

    // Total number of cards.
    unsigned count() const
    {
        return key() >> CARD_COUNTER_SHIFT;
    }

private:
    static Hand CARDS[CARD_COUNT];
    static const Hand EMPTY;
    static const unsigned CARD_COUNTER_SHIFT = 48;
    static const unsigned SUITS_SHIFT = 32;
    static const uint64_t FLUSH_CHECK_MASK = 0x8888ull << SUITS_SHIFT;

    // Returns the suit counters.
    unsigned suits() const
    {
        return key() >> SUITS_SHIFT;
    }

    uint64_t key() const
    {
        #if OMP_SSE4
        return _mm_extract_epi64(mData, 1);
        #else
        return mKey;
        #endif
    }

    uint64_t mask() const
    {
        #if OMP_SSE4
        return _mm_extract_epi64(mData, 0);
        #else
        return mMask;
        #endif
    }

    Hand(uint64_t key, uint64_t mask)
    {
        #if OMP_SSE4
        uint64_t x[2] = {mask, key};
        mData = _mm_load_si128 ((__m128i*)x);
        #else
        mKey = key;
        mMask = mask;
        #endif
    }

    // Bits 0-31: key to non-flush lookup table (linear combination of the rank constants)
    // Bits 32-47: suit counts
    // Bits 48-51: card count
    // Bits 64-128: bit mask for all cards (suits are in 16-bit groups).
    #if OMP_SSE4
    __m128i mData;
    #else
    uint64_t mKey;
    uint64_t mMask;
    #endif

    friend class HandEvaluator;
};

}

#endif // HAND
