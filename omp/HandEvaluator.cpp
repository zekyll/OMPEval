#include "HandEvaluator.h"

#include "OffsetTable.hxx"
#include "Util.h"
#include <vector>
#include <iostream>
#include <algorithm>
#include <utility>
#include <cstring>

namespace omp {

// Rank multipliers that guarantee a unique key for every rank combination in a 0-7 card hand.
const unsigned HandEvaluator::RANKS[]{0x2000, 0x8001, 0x11000, 0x3a000, 0x91000, 0x176005, 0x366000,
        0x41a013, 0x47802e, 0x479068, 0x48c0e4, 0x48f211, 0x494493};

// Rank multipliers for flush hands where only 1 of each rank is allowed. We could choose smaller numbers here but
// we want to get the key from a bitmask, so powers of 2 are used.
const unsigned HandEvaluator::FLUSH_RANKS[]{1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096};

Hand Hand::CARDS[]{};
const Hand Hand::EMPTY(0x3333ull << SUITS_SHIFT, 0);
uint16_t HandEvaluator::LOOKUP[]{};
uint16_t* HandEvaluator::ORIG_LOOKUP = nullptr;
uint16_t HandEvaluator::FLUSH_LOOKUP[]{};
const unsigned HandEvaluator::MAX_KEY = 4 * RANKS[12] + 3 * RANKS[11];
bool HandEvaluator::cardInit = (initCardConstants(), true);

// Does a thread-safe (guaranteed by C++11) one time initialization of static data.
HandEvaluator::HandEvaluator()
{
    static bool initVar = (staticInit(), true);
    (void)initVar;
}

// Initialize card constants.
void HandEvaluator::initCardConstants()
{
    for (unsigned c = 0; c < CARD_COUNT; ++c) {
        unsigned rank = c / 4, suit = c % 4;
        Hand::CARDS[c] = Hand((1ull << (4 * suit + Hand::SUITS_SHIFT)) + (1ull << Hand::CARD_COUNT_SHIFT)
                              + RANKS[rank], 1ull << ((3 - suit) * 16 + rank));
    }
}

// Initialize static class data.
void HandEvaluator::staticInit()
{
    // Temporary table for hash recalculation.
    if (RECALCULATE_PERF_HASH_OFFSETS)
        ORIG_LOOKUP = new uint16_t[MAX_KEY + 1];

    static const unsigned RC = RANK_COUNT;

    // 1. High card
    unsigned handValue = HIGH_CARD;
    handValue = populateLookup(0, 0, handValue, RC, 0, 0, 0);

    // 2. Pair
    handValue = PAIR;
    for (unsigned r = 0; r < RC; ++r)
        handValue = populateLookup(2ull << 4 * r, 2, handValue, RC, 0, 0, 0);

    // 3. Two pairs
    handValue = TWO_PAIR;
    for (unsigned r1 = 0; r1 < RC; ++r1)
        for (unsigned r2 = 0; r2 < r1; ++r2)
            handValue = populateLookup((2ull << 4 * r1) + (2ull << 4 * r2), 4, handValue, RC, r2, 0, 0);

    // 4. Three of a kind
    handValue = THREE_OF_A_KIND;
    for (unsigned r = 0; r < RC; ++r)
        handValue = populateLookup(3ull << 4 * r, 3, handValue, RC, 0, r, 0);

    // 4. Straight
    handValue = STRAIGHT;
    handValue = populateLookup(0x1000000001111ull, 5, handValue, RC, RC, RC, 3); // wheel
    for (unsigned r = 4; r < RC; ++r)
        handValue = populateLookup(0x11111ull << 4 * (r - 4), 5, handValue, RC, RC, RC, r);

    // 6. FLUSH
    handValue = FLUSH;
    handValue = populateLookup(0, 0, handValue, RC, 0, 0, 0, true);

    // 7. Full house
    handValue = FULL_HOUSE;
    for (unsigned r1 = 0; r1 < RC; ++r1)
        for (unsigned r2 = 0; r2 < RC; ++r2)
            if (r2 != r1)
                handValue = populateLookup((3ull << 4 * r1) + (2ull << 4 * r2), 5, handValue, RC, r2, r1, RC);

    // 8. Quads
    handValue = FOUR_OF_A_KIND;
    for (unsigned r = 0; r < RC; ++r)
        handValue = populateLookup(4ull << 4 * r, 4, handValue, RC, RC, RC, RC);

    // 9. Straight flush
    handValue = STRAIGHT_FLUSH;
    handValue = populateLookup(0x1000000001111ull, 5, handValue, RC, 0, 0, 3, true); // low straight flush
    for (unsigned r = 4; r < RC; ++r)
        handValue = populateLookup(0x11111ull << 4 * (r - 4), 5, handValue, RC, 0, 0, r, true);

    if (RECALCULATE_PERF_HASH_OFFSETS) {
        calculatePerfectHashOffsets();
        delete[] ORIG_LOOKUP;
    }
}

// Iterates recursively over the the remaining cards ranks in a hand and writes the hand values for each combination
// to lookup table. Parameters maxPair, maxTrips, maxStraight are used for checking that the hand
// doesn't improve (except kickers).
unsigned HandEvaluator::populateLookup(uint64_t ranks, unsigned ncards, unsigned handValue, unsigned endRank,
                                            unsigned maxPair, unsigned maxTrips, unsigned maxStraight, bool flush)
{
    // Only increment hand value counter for every valid 5 card combination. (Or smaller hands if enabled.)
    if (ncards <= 5 && ncards >= (MIN_CARDS < 5 ? MIN_CARDS : 5))
        ++handValue;

    // Write hand value to lookup when we have required number of cards.
    if (ncards >= MIN_CARDS || (flush && ncards >= 5)) {
        unsigned key = getKey(ranks, flush);

        // Write flush and non-flush hands in different tables
        if (flush) {
            FLUSH_LOOKUP[key] = handValue;
        } else if (RECALCULATE_PERF_HASH_OFFSETS) {
            ORIG_LOOKUP[key] = handValue;
        } else {
            omp_assert(LOOKUP[perfHash(key)] == 0 || LOOKUP[perfHash(key)] == handValue);
            LOOKUP[perfHash(key)] = handValue;
        }

        if (ncards == 7)
            return handValue;
    }

    // Iterate next card rank.
    for (unsigned r = 0; r < endRank; ++r) {
        uint64_t newRanks = ranks + (1ull << (4 * r));

        // Check that hand doesn't improve.
        unsigned rankCount = ((newRanks >> (r * 4)) & 0xf);
        if (rankCount == 2 && r >= maxPair)
            continue;
        if (rankCount == 3 && r >= maxTrips)
            continue;
        if (rankCount >= 4) // Don't allow new quads or more than 4 of same rank.
            continue;
        if (getBiggestStraight(newRanks) > maxStraight)
            continue;

        handValue = populateLookup(newRanks, ncards + 1, handValue, r + 1, maxPair, maxTrips, maxStraight, flush);
    }

    return handValue;
}

// Calculate lookup table key from rank counts.
unsigned HandEvaluator::getKey(uint64_t ranks, bool flush)
{
    unsigned key = 0;
    for (unsigned r = 0; r < RANK_COUNT; ++r)
        key += ((ranks >> r * 4) & 0xf) * (flush ? FLUSH_RANKS[r] : RANKS[r]);
    return key;
}

// Returns index of the highest straight card or 0 when no straight.
unsigned HandEvaluator::getBiggestStraight(uint64_t ranks)
{
    uint64_t rankMask = (0x1111111111111 & ranks) | (0x2222222222222 & ranks) >> 1 | (0x4444444444444 & ranks) >> 2;
    for (unsigned i = 9; i-- > 0; )
        if (((rankMask >> 4 * i) & 0x11111ull) == 0x11111ull)
            return i + 4;
    if ((rankMask & 0x1000000001111) == 0x1000000001111)
        return 3;
    return 0;
}

// Perfect hashing based on the algorithm described in
// http://www.drdobbs.com/architecture-and-design/generating-perfect-hash-functions/184404506
void HandEvaluator::calculatePerfectHashOffsets()
{
    // Store locations of all non-zero elements in original lookup table, divided into rows.
    std::vector<std::pair<size_t,std::vector<size_t>>> rows;
    for (size_t i = 0; i < MAX_KEY + 1; ++i) {
        if (ORIG_LOOKUP[i]) {
            size_t rowIdx = i >> PERF_HASH_ROW_SHIFT;
            if (rowIdx >= rows.size())
                rows.resize(rowIdx + 1);
            rows[rowIdx].second.push_back(i);
        }
    }

    // Need to store the original row indexes because we need them after sorting.
    for (size_t i = 0; i < rows.size(); ++i)
        rows[i].first = i;

    // Try to fit the densest rows first. Results in slightly smaller table.
    std::sort(rows.begin(), rows.end(), [](const std::pair<size_t,std::vector<size_t>>& lhs,
              const std::pair<size_t,std::vector<size_t>> & rhs){
        return lhs.second.size() > rhs.second.size();
    });

    // Goes through every row and for each of them try to find the first offset that doesn't cause any collisions with
    // previous rows. Does a very naive brute force search.
    size_t maxIdx = 0;
    for (size_t i = 0; i < rows.size(); ++i) {
        size_t offset = 0; //-(rows[i].second[0] & PERF_HASH_COLUMN_MASK); makes no difference so let's avoid negative
        for (;;++offset) {
            bool ok = true;
            for (auto x : rows[i].second) {
                unsigned val = LOOKUP[(x & PERF_HASH_COLUMN_MASK) + offset];
                if (val && val != ORIG_LOOKUP[x]) { // Allow collisions if value is the same
                    ok = false;
                    break;
                }
            }
            if (ok)
                break;
        }
        //std::cout << "row=" << i << " size=" << rows[i].second.size() << " offset=" << offset << std::endl;
        PERF_HASH_ROW_OFFSETS[rows[i].first] = (uint32_t)(offset - (rows[i].first << PERF_HASH_ROW_SHIFT));
        for (size_t key : rows[i].second) {
            size_t newIdx = (key & PERF_HASH_COLUMN_MASK) + offset;
            maxIdx = std::max<size_t>(maxIdx, newIdx);
            LOOKUP[newIdx] = ORIG_LOOKUP[key];
        }
    }

    // Output offset array.
    std::cout << "offsets: " << std::endl;
    for (size_t i = 0; i < rows.size(); ++i) {
        if (i % 8 == 0)
            std::cout << std::endl;
        std::cout << std::hex << "0x" << PERF_HASH_ROW_OFFSETS[i] << std::dec << ", ";
    }

    // Output stats.
    std::cout << std::endl;
    outputTableStats("FLUSH_LOOKUP", FLUSH_LOOKUP, 2, FLUSH_LOOKUP_SIZE);
    outputTableStats("ORIG_LOOKUP", ORIG_LOOKUP, 2, MAX_KEY + 1);
    outputTableStats("LOOKUP", LOOKUP, 2, maxIdx + 1);
    outputTableStats("OFFSETS", PERF_HASH_ROW_OFFSETS, 4, rows.size());
    std::cout << "lookup table size: " << maxIdx + 1 << std::endl;
    std::cout << "offset table size: " << rows.size() << std::endl;
}

// Output stats about memory usage of a lookup table.
void HandEvaluator::outputTableStats(const char* name, const void* p, size_t elementSize, size_t count)
{
    char dummy[64]{};
    size_t totalCacheLines = 0, usedCacheLines = 0, usedElements = 0;
    for (size_t i = 0; i < elementSize * count; i += 64) {
        ++totalCacheLines;
        bool used = false;
        for (size_t j = 0; j < 64 && i + j < elementSize * count; j += elementSize) {
            if (std::memcmp((const char*)p + i + j, dummy, elementSize)) {
                ++usedElements;
                used = true;
            }
        }
        usedCacheLines += used;
    }
    std::cout << name << ": cachelines: " << usedCacheLines << "/" << totalCacheLines
         << "  kbytes: " << usedCacheLines / 16  << "/" << totalCacheLines / 16
         << "  elements: " << usedElements << "/" << count
         << std::endl;
}

}
