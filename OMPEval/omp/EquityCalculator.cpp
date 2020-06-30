#include "EquityCalculator.h"

#include "Util.h"
#include "../libdivide/libdivide.h"
#include <random>
#include <iostream>
#include <algorithm>
#include <cmath>

namespace omp {

// Start new calculation and spawn threads.
bool EquityCalculator::start(const std::vector<CardRange>& handRanges, uint64_t boardCards, uint64_t deadCards,
                             bool enumerateAll, double stdevTarget, std::function<void(const Results&)> callback,
                             double updateInterval, unsigned threadCount)
{
    if (handRanges.size() == 0 || handRanges.size() > MAX_PLAYERS)
        return false;
    if (bitCount(boardCards) > BOARD_CARDS)
        return false;
    if (2 * handRanges.size() + bitCount(deadCards) + BOARD_CARDS > CARD_COUNT)
        return false;

    // Set up card ranges.
    mDeadCards = deadCards;
    mBoardCards = boardCards;
    mOriginalHandRanges = handRanges;
    mHandRanges = removeInvalidCombos(handRanges, mDeadCards | mBoardCards);
    std::vector<CombinedRange> combinedRanges = CombinedRange::joinRanges(mHandRanges, MAX_COMBINED_RANGE_SIZE);
    for (unsigned i = 0; i < combinedRanges.size(); ++i) {
        if (combinedRanges[i].combos().size() == 0)
            return false;
        if (!enumerateAll)
            combinedRanges[i].shuffle();
        mCombinedRanges[i] = combinedRanges[i];
    }
    mCombinedRangeCount = (unsigned)combinedRanges.size();

    // Set up simulation settings.
    mEnumPosition = 0;
    mBatchSum = mBatchSumSqr = mBatchCount = 0;
    mResults = Results();
    mResults.players = (unsigned)handRanges.size();
    mResults.enumerateAll = enumerateAll;
    mUpdateResults = mResults;
    mStdevTarget = stdevTarget;
    mCallback = callback;
    mUpdateInterval = updateInterval;
    mStopped = false;
    mLastUpdate = std::chrono::high_resolution_clock::now();
    if (threadCount == 0)
        threadCount = std::thread::hardware_concurrency();
    mUnfinishedThreads = threadCount;

    // Start threads.
    mThreads.clear();
    for (unsigned i = 0; i < threadCount; ++i) {
        mThreads.emplace_back([this,enumerateAll]{
            if (enumerateAll)
                enumerate();
            else
                simulateRandomWalkMonteCarlo();
        });
    }

    // Started successfully.
    return true;
}

// Regular monte carlo simulation.
void EquityCalculator::simulateRegularMonteCarlo()
{
    unsigned nplayers = (unsigned)mHandRanges.size();
    Hand fixedBoard = getBoardFromBitmask(mBoardCards);
    unsigned remainingCards = BOARD_CARDS - fixedBoard.count();
    BatchResults stats(nplayers);

    Rng rng{std::random_device{}()};
    FastUniformIntDistribution<unsigned,16> cardDist(0, CARD_COUNT - 1);
    FastUniformIntDistribution<unsigned,21> comboDists[MAX_PLAYERS];
    unsigned combinedRangeCount = mCombinedRangeCount;
    for (unsigned i = 0; i < mCombinedRangeCount; ++i)
        comboDists[i] = FastUniformIntDistribution<unsigned,21>(0, (unsigned)mCombinedRanges[i].combos().size() - 1);

    for (;;) {
        // Randomize hands and check for duplicate holecards.
        uint64_t usedCardsMask = 0;
        Hand playerHands[MAX_PLAYERS];
        bool ok = true;
        for (unsigned i = 0; i < combinedRangeCount; ++i) {
            unsigned comboIdx = comboDists[i](rng);
            const CombinedRange::Combo& combo = mCombinedRanges[i].combos()[comboIdx];
            if (usedCardsMask & combo.cardMask) {
                ok = false;
                break;
            }
            for (unsigned j = 0; j < mCombinedRanges[i].playerCount(); ++j) {
                unsigned playerIdx = mCombinedRanges[i].players()[j];
                playerHands[playerIdx] = combo.evalHands[j];
            }
            usedCardsMask |= combo.cardMask;
        }

        // Conflicting holecards, try again.
        if (!ok) {
            if (++stats.skippedPreflopCombos > 1000 && stats.evalCount == 0) {
                break;
            } continue;
        }

        Hand board = fixedBoard;
        randomizeBoard(board, remainingCards, usedCardsMask | mDeadCards | mBoardCards, rng, cardDist);
        evaluateHands(playerHands, nplayers, board, &stats, 1);

        // Update periodically.
        if ((stats.evalCount & 0xfff) == 0) {
            updateResults(stats, false);
            stats = BatchResults(nplayers);
            if (mStopped)
                break;
        }
    }

    updateResults(stats, true);
}

// Monte carlo simulation using a random walk. On each iteration a random player is chosen and the next feasible
// combo is picked for that player. To prove that each preflop really has equal probability of being
// visited the preflop combinations can be thought of as a directed k-regular graph. The transition probability
// matrix P then has k non-zero values on each row and column, and all non-zero elements have value of 1/k.
// It is easy to see that (1,1,...,1) * P = (1,1,...,1), i.e. (1,1,...,1) is a stable distribution.
void EquityCalculator::simulateRandomWalkMonteCarlo()
{
    unsigned nplayers = (unsigned)mHandRanges.size();
    Hand fixedBoard = getBoardFromBitmask(mBoardCards);
    unsigned remainingCards = 5 - fixedBoard.count();
    BatchResults stats(nplayers);

    Rng rng{std::random_device{}()};
    FastUniformIntDistribution<unsigned,16> cardDist(0, CARD_COUNT - 1);
    FastUniformIntDistribution<unsigned,21> comboDists[MAX_PLAYERS];
    FastUniformIntDistribution<unsigned,16> combinedRangeDist(0, mCombinedRangeCount - 1);
    for (unsigned i = 0; i < mCombinedRangeCount; ++i)
        comboDists[i] = FastUniformIntDistribution<unsigned,21>(0, (unsigned)mCombinedRanges[i].combos().size() - 1);

    uint64_t usedCardsMask;
    Hand playerHands[MAX_PLAYERS];
    unsigned comboIndexes[MAX_PLAYERS];

    // Set initial state.
    if (randomizeHoleCards(usedCardsMask, comboIndexes, playerHands, rng, comboDists)) {
        // Loop until stopped.
        for (;;) {
            // Randomize board and evaluate for current holecards.
            Hand board = fixedBoard;
            randomizeBoard(board, remainingCards, usedCardsMask, rng, cardDist);
            evaluateHands(playerHands, nplayers, board, &stats, 1);

            // Update results periodically.
            if ((stats.evalCount & 0xfff) == 0) {
                updateResults(stats, false);
                if (mStopped)
                    break;
                stats = BatchResults(nplayers);
                // Occasionally do a full randomization, because in some rare cases the random walk might
                // not be able to visit all preflop combinations by changing just one hand at a time.
                // This shouldn't happen if MAX_COMBINED_RANGE_SIZE is big enough, but extra randomization never hurts.
                if (!randomizeHoleCards(usedCardsMask, comboIndexes, playerHands, rng, comboDists))
                    break;
            }

            // Choose random player and iterate to next valid combo. If current combo is the only one that is valid
            // then will loop back to itself.
            unsigned combinedRangeIdx = combinedRangeDist(rng);
            const CombinedRange& combinedRange = mCombinedRanges[combinedRangeIdx];
            unsigned comboIdx = comboIndexes[combinedRangeIdx]; // Caching array accessess for 3% speedup!
            usedCardsMask -= combinedRange.combos()[comboIdx].cardMask;
            uint64_t mask = 0;
            do {
                if (comboIdx == 0)
                    comboIdx = (unsigned)combinedRange.size();
                --comboIdx;
                mask = combinedRange.combos()[comboIdx].cardMask;
            } while (mask & usedCardsMask);
            usedCardsMask |= mask;
            for (unsigned i = 0; i < combinedRange.playerCount(); ++i) {
                unsigned playerIdx = combinedRange.players()[i];
                playerHands[playerIdx] = combinedRange.combos()[comboIdx].evalHands[i];
            }
            comboIndexes[combinedRangeIdx] = comboIdx;
        }
    }

    updateResults(stats, true);
}

// Randomize holecards using rejection sampling. Returns false if maximum number of attempts was reached.
bool EquityCalculator::randomizeHoleCards(uint64_t &usedCardsMask, unsigned* comboIndexes, Hand* playerHands,
                                          Rng& rng, FastUniformIntDistribution<unsigned,21>* comboDists)
{
    unsigned n = 0;
    for(bool ok = false; !ok && n < 1000; ++n) {
        ok = true;
        usedCardsMask = mDeadCards | mBoardCards;
        for (unsigned i = 0; i < mCombinedRangeCount; ++i) {
            unsigned comboIdx = comboDists[i](rng);
            comboIndexes[i] = comboIdx;
            const CombinedRange::Combo& combo = mCombinedRanges[i].combos()[comboIdx];
            if (usedCardsMask & combo.cardMask) {
                ok = false;
                break;
            }
            for (unsigned j = 0; j < mCombinedRanges[i].playerCount(); ++j) {
                unsigned playerIdx = mCombinedRanges[i].players()[j];
                playerHands[playerIdx] = combo.evalHands[j];
            }
            usedCardsMask |= combo.cardMask;
        }
    }
    return n < 1000;
}

// Naive method of randomizing the board by using rejection sampling.
void EquityCalculator::randomizeBoard(Hand& board, unsigned remainingCards, uint64_t usedCardsMask,
                                      Rng& rng, FastUniformIntDistribution<unsigned,16>& cardDist)
{
    omp_assert(remainingCards + bitCount(usedCardsMask) <= CARD_COUNT && remainingCards <= BOARD_CARDS);
    for(unsigned i = 0; i < remainingCards; ++i) {
        unsigned card;
        uint64_t cardMask;
        do {
            card = cardDist(rng);
            cardMask = 1ull << card;
        } while (usedCardsMask & cardMask);
        usedCardsMask |= cardMask;
        board += Hand(card);
    }
}

// Evaluates a single showdown with one or more players and stores the result.
template<bool tFlushPossible>
void EquityCalculator::evaluateHands(const Hand* playerHands, unsigned nplayers, const Hand& board, BatchResults* stats,
                                     unsigned weight)
{
    omp_assert(board.count() == BOARD_CARDS);
    ++stats->evalCount;
    unsigned bestRank = 0;
    unsigned winnersMask = 0;
    for (unsigned i = 0, m = 1; i < nplayers; ++i, m <<= 1) {
        Hand hand = board + playerHands[i];
        unsigned rank = mEval.evaluate<tFlushPossible>(hand);
        if (rank > bestRank) {
            bestRank = rank;
            winnersMask = m;
        } else if (rank == bestRank) {
            winnersMask |= m;
        }
    }

    stats->winsByPlayerMask[winnersMask] += weight;
}

// Calculates exact equities by enumerating through all possible combinations.
void EquityCalculator::enumerate()
{
    uint64_t enumPosition = 0, enumEnd = 0;
    uint64_t preflopCombos = getPreflopCombinationCount();
    unsigned nplayers = (unsigned)mHandRanges.size();
    BatchResults stats(nplayers);
    UniqueRng64 urng(preflopCombos);
    Hand fixedBoard = getBoardFromBitmask(mBoardCards);
    libdivide::libdivide_u64_t fastDividers[MAX_PLAYERS];
    unsigned combinedRangeCount = mCombinedRangeCount;
    for (unsigned i = 0; i < combinedRangeCount; ++i)
        fastDividers[i] = libdivide::libdivide_u64_gen(mCombinedRanges[i].combos().size());

    // Lookup overhead becomes too much if postflop tree is very small.
    uint64_t postflopCombos = getPostflopCombinationCount();
    bool useLookup = postflopCombos > 500;

    // Disable random preflop enumeration order if postflop is too small (bad for caching). It's also makes no sense
    // if all the combos don't fit in the lookup table.
    bool randomizeOrder = postflopCombos > 10000 && preflopCombos <= 2 * MAX_LOOKUP_SIZE;

    for (;;++enumPosition) {
        // Ask for more work if we don't have any.
        if (enumPosition >= enumEnd) {
            uint64_t batchSize = std::max<uint64_t>(2000000 / postflopCombos, 1);
            std::tie(enumPosition, enumEnd) = reserveBatch(batchSize);
            if (enumPosition >= enumEnd)
                break;
        }

        // Use a quasi-RNG to randomize the preflop enumeration order, while still making sure
        // every combo is evaluated once.
        uint64_t randomizedEnumPos = randomizeOrder ? urng(enumPosition) : enumPosition;

        // Map enumeration index to actual hands and check duplicate card.
        bool ok = true;
        uint64_t usedCardsMask = mBoardCards | mDeadCards;
        HandWithPlayerIdx playerHands[MAX_PLAYERS];
        for (unsigned i = 0; i < combinedRangeCount; ++i) {
            uint64_t quotient = libdivide_u64_do(randomizedEnumPos, &fastDividers[i]);
            uint64_t remainder = randomizedEnumPos - quotient * mCombinedRanges[i].combos().size();
            randomizedEnumPos = quotient;

            const CombinedRange::Combo& combo = mCombinedRanges[i].combos()[(size_t)remainder];
            if (usedCardsMask & combo.cardMask) {
                ok = false;
                break;
            }
            usedCardsMask |= combo.cardMask;
            for (unsigned j = 0; j < mCombinedRanges[i].playerCount(); ++j) {
                unsigned playerIdx = mCombinedRanges[i].players()[j];
                playerHands[playerIdx].cards = combo.holeCards[j];
                playerHands[playerIdx].playerIdx = playerIdx;
            }
        }

        if(!ok) {
            ++stats.skippedPreflopCombos; //TODO fix skipcount
        } else {
            // Transform preflop into canonical form so that suit and player isomoprhism can be detected.
            uint64_t boardCards = mBoardCards;
            uint64_t deadCards = mDeadCards;
            if (useLookup) {
                // Sort players based on their hand.
                std::sort(playerHands, playerHands + nplayers, [](const HandWithPlayerIdx& lhs,
                          const HandWithPlayerIdx& rhs){
                    if (lhs.cards[0] >> 2 != rhs.cards[0] >> 2)
                        return lhs.cards[0] >> 2 < rhs.cards[0] >> 2;
                    if (lhs.cards[1] >> 2 != rhs.cards[1] >> 2)
                        return lhs.cards[1] >> 2 < rhs.cards[1] >> 2;
                    if ((lhs.cards[0] & 3) != (rhs.cards[0] & 3))
                        return (lhs.cards[0] & 3) < (rhs.cards[0] & 3);
                    return (lhs.cards[1] & 3) < (rhs.cards[1] & 3);
                });

                // Save original player indexes cause we eventually want the results for the original order.
                for (unsigned i = 0; i < nplayers; ++i)
                    stats.playerIds[i] = playerHands[i].playerIdx;

                // Suit isomorphism.
                transformSuits(playerHands, nplayers, &boardCards, &deadCards);
                usedCardsMask = boardCards | deadCards;
                for (unsigned j = 0; j < nplayers; ++j)
                    usedCardsMask |= (1ull << playerHands[j].cards[0]) | (1ull << playerHands[j].cards[1]);

                // Get cached results if this combo has already been calculated.
                uint64_t preflopId = calculateUniquePreflopId(playerHands, nplayers);
                if (lookupResults(preflopId, stats)) {
                    for (unsigned i = 0; i < nplayers; ++i)
                        stats.playerIds[i] = playerHands[i].playerIdx;
                    stats.evalCount = 0;
                    stats.uniquePreflopCombos = 0;
                } else {
                    // Do full postflop enumeration.
                    ++stats.uniquePreflopCombos;
                    Hand board = getBoardFromBitmask(boardCards);
                    enumerateBoard(playerHands, nplayers, board, usedCardsMask, &stats);
                    storeResults(preflopId, stats);
                }
            } else {
                ++stats.uniquePreflopCombos;
                enumerateBoard(playerHands, nplayers, fixedBoard, usedCardsMask, &stats);
            }
        }

        //TODO combine lookup results here so we don't need update so often
        if (stats.evalCount >= 10000 || stats.skippedPreflopCombos >= 10000 || useLookup) {
            updateResults(stats, false);
            stats = BatchResults(nplayers);
            if (mStopped)
                break;
        }
    }

    updateResults(stats, true);
}

// Starts the postflop enumeration.
void EquityCalculator::enumerateBoard(const HandWithPlayerIdx* playerHands, unsigned nplayers,
                                 const Hand& board, uint64_t usedCardsMask, BatchResults* stats)
{
    Hand hands[MAX_PLAYERS];
    for (unsigned i = 0; i < nplayers; ++i)
        hands[i] = Hand(playerHands[i].cards);

    // Take a shortcut when no board cards left to iterate.
    unsigned remainingCards = BOARD_CARDS - board.count();
    if (remainingCards == 0) {
        evaluateHands(hands, nplayers, board, stats, 1);
        return;
    }

    // Initialize deck. This also determines the enumeration order. Iterating ranks in descending order is ~5%
    // faster for some reason. Could be better branch prediction, because lower cards affect hand value less. It's
    // unlikely to be due to caching, because reversing the evaluator's rank multipliers has no effect.
    unsigned deck[CARD_COUNT];
    unsigned ndeck = 0;
    for (unsigned c = CARD_COUNT; c-- > 0;) {
        if(!(usedCardsMask & (1ull << c)))
            deck[ndeck++] = c;
    }

    // Calculate the maximum card count for each suit that any player can have after holecards and fixed board cards.
    unsigned suitCounts[SUIT_COUNT] = {};
    for (unsigned i = 0; i < nplayers; ++i) {
        if ((playerHands[i].cards[0] & 3) == (playerHands[i].cards[1] & 3)) {
            suitCounts[playerHands[i].cards[0] & 3] = std::max(2u, suitCounts[playerHands[i].cards[0] & 3]);
        } else {
            suitCounts[playerHands[i].cards[0] & 3] = std::max(1u, suitCounts[playerHands[i].cards[0] & 3]);
            suitCounts[playerHands[i].cards[1] & 3] = std::max(1u, suitCounts[playerHands[i].cards[1] & 3]);
        }
    }
    for (unsigned i = 0; i < SUIT_COUNT; ++i)
        suitCounts[i] += board.suitCount(i);

    enumerateBoardRec(hands, nplayers, stats, board, deck, ndeck, suitCounts, remainingCards, 0, 1);
}

// Enumerates board cards recursively. Detects some isomorphic subtrees by looking at the number of cards for
// each suit. Suits that cannot create a flush anymore (called here "irrelevant suits") are handled at the same time,
// which gives roughly a speedup of 3x.
void EquityCalculator::enumerateBoardRec(const Hand* playerHands, unsigned nplayers, BatchResults* stats,
                                const Hand& board, unsigned* deck, unsigned ndeck, unsigned* suitCounts,
                                unsigned cardsLeft, unsigned start, unsigned weight)
{
    // More efficient version for the innermost loop.
    if (cardsLeft == 1)
    {
        // Even simpler version for non-flush rivers.
        if (suitCounts[0] < 4 && suitCounts[1] < 4 && suitCounts[2] < 4 && suitCounts[3] < 4) {
            for (unsigned i = start; i < ndeck; ) {
                unsigned multiplier = 1;

                Hand newBoard = board + deck[i];

                // Count how many cards there are with same rank.
                unsigned rank = deck[i] >> 2;
                for (++i; i < ndeck && deck[i] >> 2 == rank; ++i)
                    ++multiplier;

                evaluateHands<false>(playerHands, nplayers, newBoard, stats, multiplier * weight);
            }
        } else {
            unsigned lastRank = ~0;
            for (unsigned i = start; i < ndeck; ++i) {
                unsigned multiplier = 1;

                if (suitCounts[deck[i] & 3] < 4) {
                    unsigned rank = deck[i] >> 2;
                    if (rank == lastRank)
                        continue;
                    // Since this is last card there's no need to do reorder deck cards; we just count the
                    // irrelevant suits in current rank.
                    for (unsigned j = i + 1; j < ndeck && deck[j] >> 2 == rank; ++j) {
                        if (suitCounts[deck[j] & 3] < 4)
                            ++multiplier;
                    }
                    lastRank = rank;
                }

                Hand newBoard = board + deck[i];
                evaluateHands(playerHands, nplayers, newBoard, stats, multiplier * weight);
            }
        }
        return;
    }

    // General version.
    for (unsigned i = start; i < ndeck; ++i) {
        Hand newBoard = board;

        unsigned suit = deck[i] & 3;

        if (suitCounts[suit] + cardsLeft < 5) {
            unsigned irrelevantCount = 1;
            unsigned rank = deck[i] >> 2;

            // Go through all the cards with same rank (they're always consecutive) and count the irrelevat suits.
            for (unsigned j = i + 1; j < ndeck && deck[j] >> 2 == rank; ++j) {
                unsigned suit2 = deck[j] & 3;
                if (suitCounts[suit2] + cardsLeft < 5) {
                    // Move all the irrelevant suits before other suits so they don't get used again.
                    if (j != i + irrelevantCount)
                        std::swap(deck[j], deck[i + irrelevantCount]);
                    ++irrelevantCount;
                }
            }

            // When there are multiple cards with irrelevant suits we have to choose how many of them to use,
            // and the number of isomorphic subtrees depends on it.
            for (unsigned repeats = 1; repeats <= std::min(irrelevantCount, cardsLeft); ++repeats) {
                static const unsigned BINOM_COEFF[5][5] = {{0}, {0, 1}, {1, 2, 1}, {1, 3, 3, 1}, {1, 4, 6, 4, 1}};
                unsigned newWeight = BINOM_COEFF[irrelevantCount][repeats] * weight;
                newBoard += deck[i + repeats - 1];
                if (repeats == cardsLeft)
                    evaluateHands(playerHands, nplayers, newBoard, stats, newWeight);
                else
                    enumerateBoardRec(playerHands, nplayers, stats, newBoard, deck, ndeck, suitCounts,
                                  cardsLeft - repeats, i + irrelevantCount, newWeight);
            }

            i += irrelevantCount - 1;
        } else {
            newBoard += deck[i];
            ++suitCounts[suit];
            enumerateBoardRec(playerHands, nplayers, stats, newBoard, deck, ndeck, suitCounts,
                              cardsLeft - 1, i + 1, weight);
            --suitCounts[suit];
        }
    }
}

// Lookup cached results for particular preflop.
bool EquityCalculator::lookupResults(uint64_t preflopId, BatchResults& results)
{
    if (!mDeadCards && !mBoardCards && lookupPrecalculatedResults(preflopId, results))
        return true;

    std::lock_guard<std::mutex> lock(mMutex);
    auto it = mLookup.find(preflopId);
    if (it != mLookup.end())
        results = it->second;
    return it != mLookup.end();
}

// Lookup precalculated results.
bool EquityCalculator::lookupPrecalculatedResults(uint64_t preflopId, BatchResults& results) const
{
    struct Cmp
    {
        // Need two overloads because of the asymmetry.
        bool operator()(uint32_t preflopId, uint64_t a) { return preflopId < (a & 0x3fffff); }
        bool operator()(uint64_t a, uint32_t preflopId) { return (a & 0x1fffff) < preflopId; }
    };

    // Binary search.
    auto it = std::lower_bound(PRECALCULATED_2PLAYER_RESULTS.begin(), PRECALCULATED_2PLAYER_RESULTS.end(),
                               (uint32_t)preflopId, Cmp());
    if (it == PRECALCULATED_2PLAYER_RESULTS.end() || (*it & 0x3fffff) != preflopId)
        return false;

    // Unpack results.
    results.winsByPlayerMask[1] = (*it >> 22) & 0x1fffff;
    results.winsByPlayerMask[3] = (*it >> 43) & 0x1fffff;
    results.winsByPlayerMask[2] = 1712304 - results.winsByPlayerMask[1] - results.winsByPlayerMask[3];

    return true;
}

// Store results for one preflop in the lookup table.
void EquityCalculator::storeResults(uint64_t preflopId, const BatchResults& results)
{
    std::lock_guard<std::mutex> lock(mMutex); //TODO read-write lock
    mLookup.emplace(preflopId, results);
    // Make sure the hash map doesn't eat all memory. Not a great way of doing it but the lookup
    // table is quite useless with that many preflop combos anyway.
    if (mLookup.size() >= MAX_LOOKUP_SIZE)
        mLookup.clear();
}

// Transforms suits in such way that suit isomorphism can be easily detected. Goes through all the holecards, board
// cards and dead cards. First encountered suit is mapped to "virtual" suit 0, second suit maps to 1 and so on.
unsigned EquityCalculator::transformSuits(HandWithPlayerIdx* playerHands, unsigned nplayers,
                                          uint64_t* boardCards, uint64_t* deadCards)
{
    unsigned transform[SUIT_COUNT] = {~0u, ~0u, ~0u, ~0u};
    unsigned suitCount = 0;

    //TODO transform fixed cards before main enumeration loop.

    uint64_t newBoardCards = 0;
    for (unsigned i = 0; i < CARD_COUNT; ++i) {
        if ((*boardCards >> i) & 1) {
            unsigned suit = i & SUIT_MASK;
            if (transform[suit] == ~0u)
                transform[suit] = suitCount++;
            unsigned newCard = (i & RANK_MASK) | transform[suit];
            newBoardCards |= 1ull << newCard;
        }
    }
    *boardCards = newBoardCards;

    uint64_t newDeadCards = 0;
    for (unsigned i = 0; i < CARD_COUNT; ++i) {
        if ((*deadCards >> i) & 1) {
            unsigned suit = i & SUIT_MASK;
            if (transform[suit] == ~0u)
                transform[suit] = suitCount++;
            unsigned newCard = (i & RANK_MASK) | transform[suit];
            newDeadCards |= 1ull << newCard;
        }
    }
    *deadCards = newDeadCards;

    // Holecards need to be handled after any fixed cards, because the lookup is only based on them.
    for (unsigned i = 0; i < nplayers; ++i) {
        for (uint8_t& c : playerHands[i].cards) {
            unsigned suit = c & SUIT_MASK;
            if (transform[suit] == ~0u)
                transform[suit] = suitCount++;
            c = (c & RANK_MASK) | transform[suit];
        }
    }

    return suitCount;
}

// Calculates a unique 64-bit id for each combination of starting hands.
uint64_t EquityCalculator::calculateUniquePreflopId(const HandWithPlayerIdx* playerHands, unsigned nplayers)
{
    uint64_t preflopId = 0;
    // Basically we just map the preflop to a number in base 1327, where each digit represents a hand.
    for (unsigned i = 0; i < nplayers; ++i) {
        preflopId *= (CARD_COUNT * (CARD_COUNT - 1) >> 1) + 1; //1327
        auto h = playerHands[i].cards;
        if (h[0] < h[1])
            std::swap(h[0], h[1]);
        preflopId += (h[0] * (h[0] - 1) >> 1) + h[1] + 1; // map a hand to range [0, 1326]
    }
    return preflopId;
}

Hand EquityCalculator::getBoardFromBitmask(uint64_t cards)
{
    Hand board = Hand::empty();
    for (unsigned c = 0; c < CARD_COUNT; ++c) {
        if (cards & (1ull << c))
            board += c;
    }
    return board;
}

// Removes combos that conflict with board and dead cards.
std::vector<std::vector<std::array<uint8_t,2>>> EquityCalculator::removeInvalidCombos(
        const std::vector<CardRange>& handRanges, uint64_t reservedCards)
{
    std::vector<std::vector<std::array<uint8_t,2>>> result;
    for (auto& hr : handRanges) {
        result.push_back(std::vector<std::array<uint8_t,2>>{});
        for (auto& h : hr.combinations()) {
            uint64_t handMask = (1ull << h[0]) | (1ull << h[1]);
            if (!(reservedCards & handMask))
                result.back().push_back(h);
        }
    }
    return result;
}

// Work allocation for enumeration threads.
std::pair<uint64_t,uint64_t> EquityCalculator::reserveBatch(uint64_t batchCount)
{
    std::lock_guard<std::mutex> lock(mMutex);

    uint64_t totalBatchCount = getPreflopCombinationCount();
    uint64_t start = mEnumPosition;
    uint64_t end = std::min<uint64_t>(totalBatchCount, mEnumPosition + batchCount);
    mEnumPosition = end;

    return {start, end};
}

// Number of different preflops with given hand ranges, assuming no conflicts between players' hands.
uint64_t EquityCalculator::getPreflopCombinationCount()
{
    uint64_t combos = 1;
    for (unsigned i = 0; i < mCombinedRangeCount; ++i)
        combos *= mCombinedRanges[i].combos().size();
    return combos;
}

// Calculates size of the postflop tree, i.e. n choose k, where n is remaining deck size and k is number
// of undealt board cards.
uint64_t EquityCalculator::getPostflopCombinationCount()
{
    omp_assert(bitCount(mBoardCards) <= BOARD_CARDS);
    unsigned cardsInDeck = CARD_COUNT;
    cardsInDeck -= bitCount(mDeadCards | mBoardCards);
    cardsInDeck -= 2 * (unsigned)mHandRanges.size();
    unsigned boardCardsRemaining = BOARD_CARDS - bitCount(mBoardCards);
    uint64_t postflopCombos = 1;
    for (unsigned i = 0; i < boardCardsRemaining; ++i)
        postflopCombos *= cardsInDeck - i;
    for (unsigned i = 0; i < boardCardsRemaining; ++i)
        postflopCombos /= i + 1;
    return postflopCombos;
}

// Results aggregation for both enumeration and monte carlo.
void EquityCalculator::updateResults(const BatchResults& stats, bool threadFinished)
{
    auto t = std::chrono::high_resolution_clock::now();

    std::lock_guard<std::mutex> lock(mMutex);

    double batchEquity = combineResults(stats);

    // Store values for stdev calculation
    if (!threadFinished) {
        mBatchSum += batchEquity;
        mBatchSumSqr += batchEquity * batchEquity;
        mBatchCount += 1;
    }

    mResults.finished = threadFinished && --mUnfinishedThreads == 0;

    double dt = 1e-9 * std::chrono::duration_cast<std::chrono::nanoseconds>(t - mLastUpdate).count();
    //std::cout << mResults.hands << " " << mHandLimit << std::endl;
    if (mResults.time + dt >= mTimeLimit || mResults.hands + mResults.intervalHands >= mHandLimit)
        mStopped = true;

    // Periodic update through callback.
    if (dt >= mUpdateInterval || mResults.finished) {
        mResults.intervalTime = dt;
        mResults.time += mResults.intervalTime;
        mResults.hands += mResults.intervalHands;
        mResults.intervalSpeed = mResults.intervalHands / (mResults.intervalTime + 1e-9);
        mResults.speed = mResults.hands / (mResults.time + 1e-9);
        mResults.intervalHands = 0;
        mResults.stdev = std::sqrt(1e-9 + mBatchSumSqr - mBatchSum * mBatchSum / mBatchCount) / mBatchCount;
        mResults.stdevPerHand = mResults.stdev * std::sqrt(mResults.hands);
        if (mResults.enumerateAll) {
            mResults.progress = (double)mEnumPosition / getPreflopCombinationCount();
        } else {
            double estimatedHands = std::pow(mResults.stdev / mStdevTarget, 2) * mResults.hands;
            mResults.progress = mResults.hands / estimatedHands;
        }
        mResults.preflopCombos = getPreflopCombinationCount();

        if (!mResults.enumerateAll && mResults.stdev < mStdevTarget) //TODO use max stdev of any player
            mStopped = true;

        for (unsigned i = 0; i < mResults.players; ++i)
            mResults.equity[i] = (mResults.wins[i] + mResults.ties[i]) / (mResults.hands + 1e-9);

        mUpdateResults = mResults;

        if (mCallback)
            mCallback(mResults);

        mLastUpdate = t;
    }

    //if (finished)
    //    outputLookupTable();
}

// Sum batch results in the main results structure.
double EquityCalculator::combineResults(const BatchResults& batch)
{
    uint64_t batchHands = 0;
    double batchEquity = 0;

    for (unsigned i = 0; i < (1u << mResults.players); ++i) {
        mResults.intervalHands += batch.winsByPlayerMask[i];
        batchHands += batch.winsByPlayerMask[i];
        unsigned winnerCount = bitCount(i);
        unsigned actualPlayerMask = 0;
        for (unsigned j = 0; j < mResults.players; ++j) {
            if (i & (1 << j)) {
                if (winnerCount == 1) {
                    mResults.wins[batch.playerIds[j]] += batch.winsByPlayerMask[i];
                    if (batch.playerIds[j] == 0)
                        batchEquity += batch.winsByPlayerMask[i];
                } else {
                    mResults.ties[batch.playerIds[j]] += batch.winsByPlayerMask[i] / (double)winnerCount;
                    if (batch.playerIds[j] == 0)
                        batchEquity += batch.winsByPlayerMask[i] / (double)winnerCount;
                }
                actualPlayerMask |= 1 << batch.playerIds[j];
            }
        }
        mResults.winsByPlayerMask[actualPlayerMask] += batch.winsByPlayerMask[i];
    }

    mResults.evaluations += batch.evalCount;
    mResults.skippedPreflopCombos += batch.skippedPreflopCombos;
    mResults.evaluatedPreflopCombos += batch.uniquePreflopCombos;

    return batchEquity / (batchHands + 1e-9);
}

// Helper function for printing out precalculated lookup tables.
void EquityCalculator::outputLookupTable() const
{
    std::vector<std::array<unsigned,3>> a;
    for (auto& e: mLookup) {
        a.push_back({(unsigned)e.first, (unsigned)e.second.winsByPlayerMask[1],
                     (unsigned)e.second.winsByPlayerMask[3]});
    }
    std::sort(a.begin(), a.end(), [](const std::array<unsigned,3>& lhs, const std::array<unsigned,3>& rhs){
        return lhs[0] < rhs[0];
    });

    std::cout << std::hex;
    for (size_t i = 0; i < a.size(); ++i) {
        if (i % 6 == 0)
            std::cout << std::endl;
        std::cout << " 0x" << ((uint64_t)a[i][0] | (uint64_t)a[i][1] << 22 | (uint64_t)a[i][2] << 43) << ",";
    }
    std::cout << std::endl;
    std::cout.flush();
}

//Not used atm.
const std::vector<uint64_t> EquityCalculator::PRECALCULATED_2PLAYER_RESULTS {
};

}
