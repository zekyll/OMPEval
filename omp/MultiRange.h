#ifndef MULTIRANGE_H
#define MULTIRANGE_H

#include "HandEvaluator.h"
#include <vector>
#include <array>
#include <cstdint>

namespace omp {

// Combines hole card ranges of multiple players into one big range that includes all valid holecard combinations
// from the original ranges (aka outer join). Purpose is to improve the efficiency of the rejection sampling method
// used in monte carlo simulation by eliminating conflicting combos already before the simulation.
// This is necessary with highly overlapping ranges like AK vs AK vs AK vs AK.
class MultiRange
{
public:
    static const unsigned MAX_PLAYERS = 6;

    struct Combo
    {
        uint64_t cardMask;
        std::array<char,2> holeCards[MAX_PLAYERS];
        Hand evalState[MAX_PLAYERS];
    };

    // Default constructor (0 players).
    MultiRange();

    // Create a range for one player.
    MultiRange(unsigned playerIdx, const std::vector<std::array<char,2>>& holeCards);

    // Combine with another range and return the result.
    MultiRange join(const MultiRange& range2) const;

    // Calculate the size of the joined range without actually doing it.
    uint64_t estimateJoinSize(const MultiRange& range2) const;

    // Takes multiple ranges and combines as many of them as possible, while keeping range sizes below the limit.
    static std::vector<MultiRange> joinRanges(const std::vector<std::vector<std::array<char,2>>>& holeCardRanges,
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

    const std::vector<Combo>& combos() const
    {
        return mCombos;
    }

private:

    std::vector<Combo> mCombos;
    std::array<unsigned, MAX_PLAYERS> mPlayers;
    unsigned mPlayerCount;
};

}

#endif // MULTIRANGE_H
