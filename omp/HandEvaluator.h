#ifndef OMPEVAL_H
#define OMPEVAL_H

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

    // Initialize a new empty hand.
    static Hand empty()
    {
        // Initializes suit counters to 3 so that the flush check bits gets set by the 5th suited card.
        return EMPTY;
    }

    // Returns the suit counters.
    unsigned suits() const
    {
        return key() >> SUITS_SHIFT;
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
    static Hand EMPTY;
    static const unsigned CARD_COUNTER_SHIFT = 48;
    static const unsigned SUITS_SHIFT = 32;
    static const uint64_t FLUSH_CHECK_MASK = 0x8888ull << SUITS_SHIFT;

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

// Evaluates hands with any number of cards up to 7.
class HandEvaluator
{
public:
    HandEvaluator();

    // Returns the rank of a hand as a 16-bit integer. Higher value is better. Can also rank hands with less than 5
    // cards. A missing card is considered the worst kicker, e.g. K < KQJT8 < A < AK < KKAQJ < AA < AA2 < AA4 < AA432.
    // Hand category can be extracted by dividing the value by 4096. 1=highcard, 2=pair, etc.
    uint16_t evaluate(const Hand& hand, bool flushPossible = true)
    {
        omp_assert(hand.count() <= 7 && hand.count() == bitCount(hand.mask()));
        // Hand has a 4-bit counter for each suit. It starts at 3 so the 4th bit gets set when there is 5 or more cards
        // of that suit.
        uint64_t flushCheck = hand.key() & Hand::FLUSH_CHECK_MASK;
        if (!flushPossible || !flushCheck) {
            // Get lookup key from the low 32 bits.
            unsigned key = (uint32_t)hand.key();
            return LOOKUP[perfHash(key)];
        } else {
            // Get the index of the flush check bit and use it to get the card mask for that suit.
            unsigned shift = countTrailingZeros((unsigned)(flushCheck >> 35)) << 2;
            unsigned flushKey = (uint16_t)(hand.mask() >> shift); // Get card mask of the correct suit.
            omp_assert(flushKey < FLUSH_LOOKUP_SIZE);
            return FLUSH_LOOKUP[flushKey];
        }
    }

private:
    static unsigned perfHash(unsigned key)
    {
        omp_assert(key <= MAX_KEY);
        return (key & PERF_HASH_COLUMN_MASK) + PERF_HASH_ROW_OFFSETS[key >> PERF_HASH_ROW_SHIFT];
    }

    static void staticInit();
    static void calculatePerfectHash();
    static unsigned populateLookup(uint64_t rankCounts, unsigned ncards, unsigned handValue, unsigned endRank,
                                   unsigned maxPair, unsigned maxTrips, unsigned maxStraight, bool flush = false);
    static unsigned getKey(uint64_t rankCounts, bool flush);
    static unsigned getBiggestStraight(uint64_t rankCounts);

    // Rank multipliers for non-flush and flush hands.
    static const unsigned RANKS[RANK_COUNT];
    static const unsigned FLUSH_RANKS[RANK_COUNT];

    // Turn on to recalculate and output the offset array.
    static const bool RECALCULATE_PERF_HASH = false;

    // Determines in how many rows the original lookup table is divided (2^shift). More rows means slightly smaller
    // lookup table but much bigger offset table.
    static const unsigned PERF_HASH_ROW_SHIFT = 11;
    static const unsigned PERF_HASH_COLUMN_MASK = (1 << PERF_HASH_ROW_SHIFT) - 1;

    // Minimum number of cards required for evaluating a hand. Can be set to higher value to decrease lookup
    // table size (requires hash recalculation).
    static const unsigned MIN_CARDS = 0;

    // Lookup tables
    static const unsigned MAX_KEY;
    static const size_t FLUSH_LOOKUP_SIZE = 8192;
    static uint16_t* ORIG_LOOKUP;
    static uint16_t LOOKUP[190641 + RECALCULATE_PERF_HASH * 100000000];
    static uint16_t FLUSH_LOOKUP[FLUSH_LOOKUP_SIZE];
    static uint32_t PERF_HASH_ROW_OFFSETS[8982 + RECALCULATE_PERF_HASH * 100000];
};

}

#endif // OMPEVAL_H
