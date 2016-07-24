#include "MultiRange.h"

#include <cassert>

namespace omp {

MultiRange::MultiRange()
    : mPlayerCount(0)
{
}

MultiRange::MultiRange(unsigned playerIdx, const std::vector<std::array<char,2>>& holeCards)
{
    mPlayerCount = 1;
    mPlayers[0] = playerIdx;
    for (auto& h: holeCards) {
        Combo c{1ull << h[0] | 1ull << h[1], {h}, {Hand(h)}};
        mCombos.emplace_back(c);
    }
}

MultiRange MultiRange::join(const MultiRange& range2) const
{
    assert(mPlayerCount + range2.mPlayerCount <= MAX_PLAYERS);

    MultiRange newRange;
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
            std::copy(c1.holeCards, c1.holeCards + mPlayerCount, c.holeCards);
            std::copy(c2.holeCards, c2.holeCards + range2.mPlayerCount, c.holeCards + mPlayerCount);
            for (unsigned i = 0; i < newRange.mPlayerCount; ++i)
                c.evalState[i] = Hand(c.holeCards[i]);
            newRange.mCombos.push_back(c);
        }
    }

    return newRange;
}

uint64_t MultiRange::estimateJoinSize(const MultiRange& range2) const
{
    assert(mPlayerCount + range2.mPlayerCount <= MAX_PLAYERS);
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

std::vector<MultiRange> MultiRange::joinRanges(
        const std::vector<std::vector<std::array<char,2>>>& holeCardRanges, size_t maxSize)
{
    std::vector<MultiRange> multiRanges;
    for (unsigned i = 0; i < holeCardRanges.size(); ++i)
        multiRanges.emplace_back(MultiRange{i, holeCardRanges[i]});

    for (;;) {
        uint64_t bestSize = ~0ull;
        unsigned besti, bestj;
        for (unsigned i = 0; i < multiRanges.size(); ++i) {
            for (unsigned j = 0; j < i; ++j) {
                uint64_t newSize = multiRanges[i].estimateJoinSize(multiRanges[j]);
                if (newSize < bestSize)
                    besti = i, bestj = j, bestSize = newSize;
            }
        }

        if (bestSize <= maxSize) {
            multiRanges[besti] = multiRanges[besti].join(multiRanges[bestj]);
            multiRanges.erase(multiRanges.begin() + bestj);
        } else {
            break;
        }
    }

    return multiRanges;
}

}
