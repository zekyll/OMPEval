#ifndef OMP_COMBINED_RANGE_H
#define OMP_COMBINED_RANGE_H

#include "HandEvaluator.h"
#include "Util.h"
#include <vector>
#include <array>
#include <cstdint>

namespace omp {

// Combines hole card ranges of multiple players into one big range that includes all valid holecard combinations
// from the original ranges (aka outer join). Purpose is to improve the efficiency of the rejection sampling method
// used in monte carlo simulation by eliminating conflicting combos already before the simulation.
// This is necessary with highly overlapping ranges like AK vs AK vs AK vs AK.
class CombinedRange
{
public:
    struct Combo
    {
        uint64_t cardMask;
        std::array<std::array<uint8_t,2>,MAX_PLAYERS> holeCards;
        Hand evalHands[MAX_PLAYERS];
    };

    // Default constructor (0 players).
    CombinedRange();

    // Create a range for one player.
    CombinedRange(unsigned playerIdx, const std::vector<std::array<uint8_t,2>>& holeCards);

    // Combine with another range and return the result.
    CombinedRange join(const CombinedRange& range2) const;

    // Calculate the size of the joined range without actually doing it.
    uint64_t estimateJoinSize(const CombinedRange& range2) const;

    // Takes multiple ranges and combines as many of them as possible, while keeping range sizes below the limit.
    static std::vector<CombinedRange> joinRanges(const std::vector<std::vector<std::array<uint8_t,2>>>& holeCardRanges,
                                              size_t maxSize);

    // Randomize order of combos (good for random walk simulation).
    void shuffle();

    unsigned playerCount() const
    {
        return mPlayerCount;
    }

    const std::array<unsigned, MAX_PLAYERS>& players() const
    {
        return mPlayers;
    }

    const std::vector<Combo,AlignedAllocator<Combo>>& combos() const
    {
        return mCombos;
    }

    size_t size() const
    {
        return mSize;
    }

private:

    std::vector<Combo,AlignedAllocator<Combo>> mCombos;
    std::array<unsigned, MAX_PLAYERS> mPlayers;
    unsigned mPlayerCount;
    size_t mSize;
};

}

#endif // OMP_COMBINED_RANGE_H
