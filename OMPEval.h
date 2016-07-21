#ifndef OMPEVAL_H
#define OMPEVAL_H

#include "util.h"
#include <array>
#include <cstdint>
#include <xmmintrin.h>

#ifndef OMP_SSE
    #define OMP_SSE 0
#endif

// Some constants.
struct Pok
{
    static const unsigned CARD_COUNT = 52;
    static const unsigned RANK_COUNT = 13;
    static const unsigned SUIT_COUNT = 4;
    static const unsigned SUIT_MASK = 0x3;
    static const unsigned RANK_MASK = 0x3c;
    static const unsigned RANK_SHIFT = 2;

    static const unsigned HAND_CATEGORY_OFFSET = 0x1000; // 4096

    static const unsigned HIGH_CARD = 1 * HAND_CATEGORY_OFFSET;
    static const unsigned PAIR = 2 * HAND_CATEGORY_OFFSET;
    static const unsigned TWO_PAIR = 3 * HAND_CATEGORY_OFFSET;
    static const unsigned THREE_OF_A_KIND = 4 * HAND_CATEGORY_OFFSET;
    static const unsigned STRAIGHT = 5 * HAND_CATEGORY_OFFSET;
    static const unsigned FLUSH = 6 * HAND_CATEGORY_OFFSET;
    static const unsigned FULL_HOUSE = 7 * HAND_CATEGORY_OFFSET;
    static const unsigned FOUR_OF_A_KIND = 8 * HAND_CATEGORY_OFFSET;
    static const unsigned STRAIGHT_FLUSH = 9 * HAND_CATEGORY_OFFSET;
};

// Structure that combines the data from multiple cards so that hand strength can be evaluated efficiently.
// It is essential when combining hands that exactly one of them was initialized using Hand::empty().
struct Hand : public Pok
{
    // Default constructor Leaves the struct uninitialized for performance reasons.
    Hand()
    {
    }

    // Initialize hand from one card.
    Hand(unsigned cardIdx)
    {
        *this = CARDS[cardIdx];
    }

    // Initialize hand from two cards.
    Hand(std::array<char,2> holeCards)
    {
        *this = CARDS[holeCards[0]];
        combine(holeCards[1]);
    }

    // Add a card to hand. Card is between 0 and 51, so that CARD = 4 * RANK + SUIT, where rank ranges
    // from 0 (deuce) to 12 (ace) and suit is from 0 (spade) to 3 (diamond).
    void combine(unsigned cardIdx)
    {
        combine(CARDS[cardIdx]);
    }

    // Combine with another hand.
    void combine(const Hand& hand2)
    {
        #if _MSC_VER && OMP_SSE
        mData = _mm_add_epi64(mData, hand2.mData);
        #else
        mKey += hand2.mKey;
        mMask |= hand2.mMask;
        #endif
    }

    void combineNoFlush(unsigned cardIdx)
    {
        combineNoFlush(CARDS[cardIdx]);
    }

    void combineNoFlush(const Hand& hand2)
    {
        #if _MSC_VER && OMP_SSE
        mData.m128i_u64[0] += hand2.mData.m128i_u64[0];
        #else
        mKey += hand2.mKey;
        #endif
    }

    // Initialize an empty hand.
    static Hand empty()
    {
        // Initialize suit counters to 3 so that the flush check bits gets set by the 5th suited card.
        return {0x333300000000, 0};
    }

    // Returns the suit counters.
    unsigned suits()
    {
        return key() >> 32;
    }

private:
    static Hand CARDS[CARD_COUNT];
    static const uint64_t FLUSH_CHECK_MASK = 0x888800000000;


    uint64_t key() const
    {
        #if _MSC_VER && OMP_SSE
        return mData.m128i_u64[0];
        #else
        return mKey;
        #endif
    }

    uint64_t mask() const
    {
        #if _MSC_VER && OMP_SSE
        return mData.m128i_u64[1];
        #else
        return mMask;
        #endif
    }

    Hand(uint64_t key, uint64_t mask)
    {
        #if _MSC_VER && OMP_SSE
        mData.m128i_u64[0] = key;
        mData.m128i_u64[1] = mask;
        #else
        mKey = key;
        mMask = mask;
        #endif
    }

    // Bits 0-31: key to non-flush lookup table (linear combination of the rank constants)
    // Bits 32-48: suit counters
    // Bits 64-128: bit mask for all cards (suits are in 16-bit groups).
    #if _MSC_VER && OMP_SSE
    __m128i mData;
    #else
    uint64_t mKey;
    uint64_t mMask;
    #endif

    friend class HandEvaluator;
};

// Evaluates hands with any number of cards up to 7.
class HandEvaluator : public Pok
{
public:
    HandEvaluator();

    // Returns the rank of a hand as a 16-bit integer. Higher value is better. Can also rank hands with less than 5
    // cards. A missing card is considered the worst kicker, e.g. K < KQJT8 < A < AK < KKAQJ < AA < AA2 < AA4 < AA432.
    // Hand category can be extracted by dividing the value by 4096. 1=highcard, 2=pair, etc.
    uint16_t evaluate(const Hand& hand, bool flushPossible = true)
    {
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
            return FLUSH_LOOKUP[flushKey];
        }
    }

private:
    static unsigned perfHash(unsigned key)
    {
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

#endif // OMPEVAL_H
