#ifndef OMP_EQUITYCALCULATOR_H
#define OMP_EQUITYCALCULATOR_H

#include "CombinedRange.h"
#include "Random.h"
#include "CardRange.h"
#include "HandEvaluator.h"
#include "Constants.h"
#include "Util.h"
#include <chrono>
#include <thread>
#include <mutex>
#include <atomic>
#include <unordered_map>
#include <array>
#include <cstdint>

namespace omp {

// Calculates all-in equities in Texas Holdem for given player hand ranges, board cards and dead cards. Supports both
// exact enumeration and monte carlo simulation.
class EquityCalculator
{
public:

    struct Results
    {
        // Number of players.
        unsigned players = 0;
        // Equity by player (between 0 and 1).
        double equity[MAX_PLAYERS] = {};
        // Wins by player.
        uint64_t wins[MAX_PLAYERS] = {};
        // Ties by player, adjusted for equity: 2-way splits = 1/2, 3-way = 1/3 etc..
        double ties[MAX_PLAYERS] = {};
        // Wins for each combination of winning players. Index ranges from 0 to 2^(n-1), where
        // bit 0 is player 1, bit 1 player 2 etc).
        uint64_t winsByPlayerMask[1 << MAX_PLAYERS] = {};
        // Total hand count / hand count for last update period.
        uint64_t hands = 0, intervalHands = 0;
        // Total speed in hands/s / speed for last update period.
        double speed = 0, intervalSpeed = 0;
        // Total duration / duration of the last update period.
        double time = 0, intervalTime = 0;
        // Standard deviation for the total equity of first player.
        double stdev = 0;
        // Single-hand standard deviation.
        double stdevPerHand = 0;
        // Progress from 0 to 1. Based on hand count for enumeration, and stdev target for monte carlo.
        double progress = 0;
        // Number of different combinations of starting hands for all players.
        uint64_t preflopCombos = 0;
        // Number of preflop combos that were skipped due the having same cards. (Enumeration only.)
        uint64_t skippedPreflopCombos = 0;
        // How many of the preflop combos were actually enumerated.
        uint64_t evaluatedPreflopCombos = 0;
        // How many showdowns were actually evaluated (instead of using lookups or isomorphism).
        uint64_t evaluations = 0;
        // Whether enumeration or monte carlo was used.
        bool enumerateAll = false;
        // Is calculation finished. (Includes stopping.)
        bool finished = false;
    };

    // Start a new calculation. Returns false if calculation is impossible for given hand ranges and board/dead cards.
    // After calling start() succesfully, wait() must be called in order wait for threads to finish.
    // handRanges: hand ranges for each player
    // boardCards/deadCards: bitmasks for board and dead cards
    // enumerateAll: true for exact enumeration, false for monte carlo
    // stdevTarget: stops monte carlo when standard deviation is smaller than this, use 0 for infinite simulation
    // callback: function that is called periodically with incomplete results
    // updateInterval: how often callback is called
    // threadCount: number of threads to spawn, 0 for maximum parallelism supported by hardware
    bool start(const std::vector<CardRange>& handRanges, uint64_t boardCards = 0, uint64_t deadCards = 0,
               bool enumerateAll = false, double stdevTarget = 5e-5,
               std::function<void(const Results&)> callback = nullptr,
               double updateInterval = 0.2, unsigned threadCount = 0);

    // Force current calculation to stop before it's ready. Still must call wait()!
    void stop()
    {
        mStopped = true;
    }

    // Wait for calculation to finish. Must always be called once for every successful start() call!
    void wait()
    {
        for (auto& t : mThreads)
            t.join();
    }

    // Set a time limit for the calculation in seconds. Use 0 to disable. Disabled by default.
    void setTimeLimit(double seconds)
    {
        std::lock_guard<std::mutex> lock(mMutex);
        mTimeLimit = seconds <= 0 ? INFINITE : seconds;
    }

    // Set a hand limit for the calculation or 0 to disable. Disabled by default.
    void setHandLimit(uint64_t handLimit)
    {
        std::lock_guard<std::mutex> lock(mMutex);
        mHandLimit = handLimit == 0 ? INFINITE : handLimit;
    }

    // Get results from previous update.
    Results getResults()
    {
        std::lock_guard<std::mutex> lock(mMutex);
        return mUpdateResults;
    }

    // Hand ranges used in current calculation.
    const std::vector<CardRange>& handRanges() const
    {
        return mOriginalHandRanges;
    }

private:
    typedef XoroShiro128Plus Rng;

    static const size_t MAX_LOOKUP_SIZE = 1000000;
    static const size_t MAX_COMBINED_RANGE_SIZE = 10000;
    static const uint64_t INFINITE = ~0ull;

    // Temporary storage for results.
    struct BatchResults
    {
        BatchResults(unsigned nplayers)
        {
            for (unsigned i = 0; i < nplayers; ++i)
                playerIds[i] = i;
        }

        uint64_t skippedPreflopCombos = 0;
        uint64_t uniquePreflopCombos = 0;
        uint64_t evalCount = 0;
        uint8_t playerIds[MAX_PLAYERS];
        unsigned winsByPlayerMask[1 << MAX_PLAYERS] = {};
    };

    // Ad-hoc struct used when sorting hands.
    struct HandWithPlayerIdx
    {
        std::array<uint8_t,2> cards;
        unsigned playerIdx;
    };

    void simulateRegularMonteCarlo();
    void simulateRandomWalkMonteCarlo();
    bool randomizeHoleCards(uint64_t &usedCardsMask, unsigned* comboIndexes, Hand* playerHands,
                            Rng& rng, FastUniformIntDistribution<unsigned,21>*comboDists);
    OMP_FORCE_INLINE void randomizeBoard(Hand& board, unsigned remainingCards, uint64_t usedCardsMask,
                        Rng& rng, FastUniformIntDistribution<unsigned,16>& cardDist);
    template<bool tFlushPossible = true>
    OMP_FORCE_INLINE void evaluateHands(const Hand* playerHands, unsigned nplayers, const Hand& board,
            BatchResults* stats, unsigned weight);
    void enumerate();
    void enumerateBoard(const HandWithPlayerIdx* playerHands, unsigned nplayers,
                   const Hand& board, uint64_t usedCardsMask, BatchResults* stats);
    void enumerateBoardRec(const Hand* playerHands, unsigned nplayers, BatchResults* stats,
                           const Hand& board, unsigned* deck, unsigned ndeck,  unsigned* suitCounts,
                           unsigned k, unsigned start, unsigned weight);
    bool lookupResults(uint64_t hash, BatchResults& results);
    bool lookupPrecalculatedResults(uint64_t hash, BatchResults& results) const;
    void storeResults(uint64_t hash, const BatchResults& results);
    static unsigned transformSuits(HandWithPlayerIdx* playerHands, unsigned nplayers,
                                   uint64_t* boardCards, uint64_t* usedCards);
    static uint64_t calculateUniquePreflopId(const HandWithPlayerIdx* playerHands, unsigned nplayers);
    static Hand getBoardFromBitmask(uint64_t board);
    static std::vector<std::vector<std::array<uint8_t,2>>> removeInvalidCombos(const std::vector<CardRange>& handRanges,
                                                               uint64_t reservedCards);
    std::pair<uint64_t,uint64_t> reserveBatch(uint64_t batchCount);
    uint64_t getPreflopCombinationCount();
    uint64_t getPostflopCombinationCount();

    void updateResults(const BatchResults& stats, bool finished);
    double combineResults(const BatchResults& batch);
    void outputLookupTable() const;

    std::vector<std::thread> mThreads;

    // Shared between threads, protected by mMutex.
    std::mutex mMutex;
    std::atomic<bool> mStopped;
    unsigned mUnfinishedThreads;
    std::chrono::high_resolution_clock::time_point mLastUpdate;
    Results mResults, mUpdateResults;
    double mBatchSum, mBatchSumSqr, mBatchCount;
    uint64_t mEnumPosition;
    std::unordered_map<uint64_t, BatchResults> mLookup;

    // Constant shared data
    std::vector<CardRange> mOriginalHandRanges; // Original ranges without before card removal.
    std::vector<std::vector<std::array<uint8_t,2>>> mHandRanges; // Ranges after card removal.
    CombinedRange mCombinedRanges[MAX_PLAYERS];
    unsigned mCombinedRangeCount;
    uint64_t mDeadCards, mBoardCards;
    HandEvaluator mEval;
    double mStdevTarget = 5e-5, mTimeLimit = (double)INFINITE, mUpdateInterval = 0.1;
    uint64_t mHandLimit = INFINITE;
    std::function<void(const Results& results)> mCallback;

    // Precalculated results for 2 player preflop situations. Uses a sorted array for lowest memory use.
    static const std::vector<uint64_t> PRECALCULATED_2PLAYER_RESULTS;
};

}

#endif // OMP_EQUITYCALCULATOR_H
