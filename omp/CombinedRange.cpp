#include "CombinedRange.h"

#include "Random.h"
#include <algorithm>
#include <iterator>
#include <random>
#include <cassert>

namespace omp {

CombinedRange::CombinedRange()
    : mPlayerCount(0), mSize(0)
{
}

CombinedRange::CombinedRange(unsigned playerIdx, const std::vector<std::array<uint8_t,2>>& holeCards)
{
    mPlayerCount = 1;
    mPlayers[0] = playerIdx;
    for (auto& h: holeCards) {
        Combo c{1ull << h[0] | 1ull << h[1], {h}, {Hand(h)}};
        mCombos.emplace_back(c);
    }
    mSize = mCombos.size();
}

CombinedRange CombinedRange::join(const CombinedRange& range2) const
{
    omp_assert(mPlayerCount + range2.mPlayerCount <= MAX_PLAYERS);

    CombinedRange newRange;
    newRange.mPlayerCount = mPlayerCount + range2.mPlayerCount;
    std::copy(mPlayers.begin(), mPlayers.begin() + mPlayerCount, newRange.mPlayers.begin());
    std::copy(range2.mPlayers.begin(), range2.mPlayers.begin() + range2.mPlayerCount,
              newRange.mPlayers.begin() + mPlayerCount);

    for (const Combo& c1 : mCombos) {
        for (const Combo& c2 : range2.mCombos) {
            if (c1.cardMask & c2.cardMask)
                continue;
            Combo c;
            c.cardMask = c1.cardMask | c2.cardMask;
            std::copy(std::begin(c1.holeCards), std::begin(c1.holeCards) + mPlayerCount, std::begin(c.holeCards));
            std::copy(std::begin(c2.holeCards), std::begin(c2.holeCards) + range2.mPlayerCount, std::begin(c.holeCards) + mPlayerCount);
            for (unsigned i = 0; i < newRange.mPlayerCount; ++i)
                c.evalHands[i] = Hand(c.holeCards[i]);
            newRange.mCombos.push_back(c);
        }
    }
    newRange.mSize = newRange.mCombos.size();

    return newRange;
}

uint64_t CombinedRange::estimateJoinSize(const CombinedRange& range2) const
{
    omp_assert(mPlayerCount + range2.mPlayerCount <= MAX_PLAYERS);
    uint64_t size = 0;
    for (const Combo& c1 : mCombos) {
        for (const Combo& c2 : range2.mCombos) {
            if (c1.cardMask & c2.cardMask)
                continue;
            ++size;
        }
    }
    return size;
}

std::vector<CombinedRange> CombinedRange::joinRanges(
        const std::vector<std::vector<std::array<uint8_t,2>>>& holeCardRanges, size_t maxSize)
{
    std::vector<CombinedRange> combinedRanges;
    for (unsigned i = 0; i < holeCardRanges.size(); ++i)
        combinedRanges.emplace_back(CombinedRange{i, holeCardRanges[i]});

    for (;;) {
        uint64_t bestSize = ~0ull;
        unsigned besti = 0, bestj = 0;
        for (unsigned i = 0; i < combinedRanges.size(); ++i) {
            for (unsigned j = 0; j < i; ++j) {
                uint64_t newSize = combinedRanges[i].estimateJoinSize(combinedRanges[j]);
                if (newSize < bestSize)
                    besti = i, bestj = j, bestSize = newSize;
            }
        }

        if (bestSize <= maxSize) {
            combinedRanges[besti] = combinedRanges[besti].join(combinedRanges[bestj]);
            combinedRanges.erase(combinedRanges.begin() + bestj);
        } else {
            break;
        }
    }

    return combinedRanges;
}

void CombinedRange::shuffle()
{
    XoroShiro128Plus rng(std::random_device{}());
    std::shuffle(mCombos.begin(), mCombos.end(), rng);
}

}
