
#include "omp/HandEvaluator.h"
#include "omp/EquityCalculator.h"
#include "omp/Random.h"
#include "ttest/ttest.h"
#include <iostream>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <list>
#include <numeric>
#include <cmath>

using namespace std;
using namespace omp;

class UtilTest : public ttest::TestBase
{
    TTEST_CASE("countLeadingZeros")
    {
        TTEST_EQUAL(countLeadingZeros(1u), 31u);
        TTEST_EQUAL(countLeadingZeros(0xf0u), 24u);
        TTEST_EQUAL(countLeadingZeros(~0u), 0u);
    }

    TTEST_CASE("countTrailingZeros")
    {
        TTEST_EQUAL(countTrailingZeros((uint32_t)1), 0u);
        TTEST_EQUAL(countTrailingZeros((uint32_t)0x0f000000), 24u);
        TTEST_EQUAL(countTrailingZeros((uint32_t)~0u), 0u);
    }

    TTEST_CASE("bitCount")
    {
        TTEST_EQUAL(bitCount(~0u), sizeof(unsigned) * CHAR_BIT);
        TTEST_EQUAL(bitCount(0xf0u), 4u);
        TTEST_EQUAL(bitCount(~0ul), sizeof(unsigned long) * CHAR_BIT);
        TTEST_EQUAL(bitCount(0xf0ul), 4u);
        TTEST_EQUAL(bitCount(~0ull), sizeof(unsigned long long) * CHAR_BIT);
        TTEST_EQUAL(bitCount(0x0ff00000000000f0ull), 12u);
    }

    TTEST_CASE("alignedNew")
    {
        char* p = (char*)alignedNew(1, 512);
        TTEST_EQUAL((uintptr_t)p % 512, 0u);
        alignedDelete(p);
    }
};

class HandTest : public ttest::TestBase
{
    #if OMP_SSE2
    TTEST_CASE("has 16-byte alignment and size")
    {
        TTEST_EQUAL(OMP_ALIGNOF(Hand), 16u);
        TTEST_EQUAL(sizeof(Hand), 16u);
    }

    TTEST_CASE("allocated objects are aligned")
    {
        struct HandHash
        {
            size_t operator()(const Hand& h) const
            {
                return h.rankKey();
            }
        };

        Hand h;
        TTEST_EQUAL((uintptr_t)&h % sizeof(Hand), 0u);

        vector<Hand> v;
        unordered_set<Hand, HandHash> s;
        unordered_map<Hand,Hand,HandHash> m;
        list<Hand> l;
        for (unsigned i = 0; i < 52; ++i) {
            TTEST_EQUAL((uintptr_t)&*v.insert(v.end(), i) % sizeof(Hand), 0u);
            TTEST_EQUAL((uintptr_t)&*s.insert(i).first % sizeof(Hand), 0u);
            TTEST_EQUAL((uintptr_t)&(m[i] = Hand(i)) % sizeof(Hand), 0u);
            TTEST_EQUAL((uintptr_t)&*l.insert(l.end(), i) % sizeof(Hand), 0u);
        }
    }
    #endif

    TTEST_CASE("empty()")
    {
        Hand h = Hand::empty();
        TTEST_EQUAL(h.count(), 0u);
        TTEST_EQUAL(h.suitCount(0), 0u);
    }

    TTEST_CASE("adding & removing cards")
    {
        Hand h = Hand::empty() + Hand(5);
        TTEST_EQUAL(h.count(), 1u);
        TTEST_EQUAL(h.suitCount(1), 1u);
        h += Hand(51);
        TTEST_EQUAL(h.count(), 2u);
        TTEST_EQUAL(h.suitCount(0), 0u);
        TTEST_EQUAL(h.suitCount(1), 1u);
        TTEST_EQUAL(h.suitCount(3), 1u);
        h += Hand(3);
        TTEST_EQUAL(h.count(), 3u);
        TTEST_EQUAL(h.suitCount(0), 0u);
        TTEST_EQUAL(h.suitCount(1), 1u);
        TTEST_EQUAL(h.suitCount(3), 2u);
        h -= Hand(51);
        TTEST_EQUAL(h.count(), 2u);
        TTEST_EQUAL(h.suitCount(0), 0u);
        TTEST_EQUAL(h.suitCount(1), 1u);
        TTEST_EQUAL(h.suitCount(3), 1u);
        h = h - (Hand(3) + Hand(5));
        TTEST_EQUAL(h.count(), 0u);
    }

    TTEST_CASE("flush check")
    {
        Hand h = Hand::empty();
        TTEST_EQUAL(h.hasFlush(), false);
        h += Hand(4) + 8 + 12 + 16 + 17;
        TTEST_EQUAL(h.hasFlush(), false);
        h += 0;
        TTEST_EQUAL(h.hasFlush(), true);
    }

    TTEST_CASE("rankKey()")
    {
        TTEST_EQUAL((Hand(4) + Hand(8)).rankKey(), (Hand(9) + Hand(5)).rankKey());
        TTEST_EQUAL((Hand(4) + Hand(8)).rankKey() == (Hand(12) + Hand(5)).rankKey(), false);
    }

    void enumRankCombos(unsigned n, const Hand& h, unordered_set<uint32_t>& keys,
            unsigned s = 0, unsigned k = 0)
    {
        keys.insert(h.rankKey());
        if (n == 0)
            return;
        for (unsigned r = k; r < RANK_COUNT; ++r, s = 0) {
            if (s == 4)
                continue;
            enumRankCombos(n - 1, h + Hand(r * 4  + s), keys, s + 1, r);
        }
    }

    TTEST_CASE("rankKey() has no collisions")
    {
        unordered_set<uint32_t> keys;
        enumRankCombos(7, Hand::empty(), keys);
        TTEST_EQUAL(keys.size(), 76155u);
    }

    TTEST_CASE("flushKey()")
    {
        Hand h = Hand::empty() + 6 + 10 + 14 + 18 + 22 + 26 + 3;
        TTEST_EQUAL(h.flushKey(), 0x7eu);
    }
};

class HandEvaluatorTest : public ttest::TestBase
{
    HandEvaluator e;
    uint64_t counts[10]{};

    void enumerate(unsigned cardsLeft, const Hand& h = Hand::empty(), unsigned start = 0)
    {
        for (unsigned c = start; c < 52; ++c) {
            if (cardsLeft == 1)
                ++counts[e.evaluate(h + c) >> HAND_CATEGORY_SHIFT];
            else
                enumerate(cardsLeft - 1, h + c, c + 1);
        }
    }

    TTEST_BEFORE()
    {
        fill(begin(counts), end(counts), 0);
    }

    TTEST_CASE("0 cards")
    {
        TTEST_EQUAL(e.evaluate(Hand::empty()), HAND_CATEGORY_OFFSET + 1);
    }

    TTEST_CASE("enumerate 1 card hands")
    {
        uint64_t expected[10]{0, 52};
        enumerate(1);
        for (unsigned i = 0; i < 10; ++i)
            TTEST_EQUAL(counts[i], expected[i]);
    }

    TTEST_CASE("enumerate 2 cards hands")
    {
        uint64_t expected[10]{0, 1248, 78};
        enumerate(2);
        for (unsigned i = 0; i < 10; ++i)
            TTEST_EQUAL(counts[i], expected[i]);
    }

    TTEST_CASE("enumerate 3 cards hands")
    {
        uint64_t expected[10]{0, 18304, 3744, 0, 52};
        enumerate(3);
        for (unsigned i = 0; i < 10; ++i)
            TTEST_EQUAL(counts[i], expected[i]);
    }

    TTEST_CASE("enumerate 4 cards hands")
    {
        uint64_t expected[10]{0, 183040, 82368, 2808, 2496, 0, 0, 0, 13};
        enumerate(4);
        for (unsigned i = 0; i < 10; ++i)
            TTEST_EQUAL(counts[i], expected[i]);
    }

    TTEST_CASE("enumerate 5 cards hands")
    {
        uint64_t expected[10]{0, 1302540, 1098240, 123552, 54912, 10200, 5108, 3744, 624, 40};
        enumerate(5);
        for (unsigned i = 0; i < 10; ++i)
            TTEST_EQUAL(counts[i], expected[i]);
    }

    TTEST_CASE("enumerate 6 cards hands")
    {
        uint64_t expected[10]{0, 6612900, 9730740, 2532816, 732160, 361620, 205792, 165984, 14664, 1844};
        enumerate(6);
        for (unsigned i = 0; i < 10; ++i)
            TTEST_EQUAL(counts[i], expected[i]);
    }

    TTEST_CASE("enumerate 7 cards hands")
    {
        uint64_t expected[10]{0, 23294460, 58627800, 31433400, 6461620, 6180020, 4047644,
                3473184, 224848, 41584};
        enumerate(7);
        for (unsigned i = 0; i < 10; ++i)
            TTEST_EQUAL(counts[i], expected[i]);
    }
};

class EquityCalculatorTest : public ttest::TestBase
{
    EquityCalculator eq;

    struct TestCase
    {
        vector<string> ranges;
        string board, dead;
        vector<uint64_t> expectedResults;
    };

    vector<TestCase> TESTDATA = [&]{
        vector<TestCase> td;
        td.emplace_back(TestCase{{"AA", "KK"}, "", "",
                {0, 50371344, 10986372, 285228}});
        td.emplace_back(TestCase{ { "AK", "random"}, "2c3c", "",
                {0, 159167583, 108567320, 6233737}});
        td.emplace_back(TestCase{ {"random", "AA", "33"}, "2c3c8h", "6h",
                {0, 808395, 1681125, 20076, 12151512, 0, 0, 0}});
        td.emplace_back(TestCase{ {"random", "random", "AK"}, "4hAd3c4c7c", "6h",
                {0, 1461364, 1461364, 6386, 6760010, 42420, 42420, 108}});
        td.emplace_back(TestCase{ {"3d7d", "2h9h", "2c9c"}, "5d5h5c", "3s3c",
                {0, 183, 28, 0, 28, 0, 380, 201}});
        td.emplace_back(TestCase{ {"AA,KK", "KK,QQ", "QQ,AA" }, "", "",
                {0, 348272820, 119882736, 37653912, 303253020, 74015280, 1266624, 3904200}});
        return td;
    }(); // Workaround for MSVC2013's incomplete initializer list support.

    void enumTest(const TestCase& tc)
    {
        std::vector<CardRange> ranges2(tc.ranges.begin(), tc.ranges.end());
        if (!eq.start(ranges2, CardRange::getCardMask(tc.board), CardRange::getCardMask(tc.dead), true))
                throw ttest::TestException("Invalid hand ranges!");
        eq.wait();
        auto results = eq.getResults();
        for (unsigned i = 0; i < (1u << tc.ranges.size()); ++i)
            TTEST_EQUAL(results.winsByPlayerMask[i], tc.expectedResults[i]);
    }

    void monteCarloTest(const TestCase& tc)
    {
        double hands = accumulate(tc.expectedResults.begin(), tc.expectedResults.end(), 0.0);
        std::vector<CardRange> ranges2(tc.ranges.begin(), tc.ranges.end());
        bool timeout = false;
        auto callback = [&](const EquityCalculator::Results& r){
            double maxErr = 0;
            for (unsigned i = 0; i < (1u << tc.ranges.size()); ++i)
                maxErr = max(std::abs(tc.expectedResults[i] / hands - (double) r.winsByPlayerMask[i] / r.hands), maxErr);
            if (maxErr < 2e-4)
                eq.stop();
            if (r.time > 10) {
                timeout = true;
                eq.stop();
            }
        };
        if (!eq.start(ranges2, CardRange::getCardMask(tc.board), CardRange::getCardMask(tc.dead),
                false, 0, callback, 0.1))
            throw ttest::TestException("Invalid hand ranges!");
        eq.wait();
        if (timeout)
            throw ttest::TestException("Didn't converge to correct results in time!");
    }

    TTEST_BEFORE()
    {
        eq.setTimeLimit(0);
        eq.setHandLimit(0);
    }

    TTEST_CASE("start() returns false when too many board cards")
    {
        TTEST_EQUAL(eq.start({"random"}, CardRange::getCardMask("2c3c4c5c6c7c")), false);
    }

    TTEST_CASE("start() returns false when too many players")
    {
        TTEST_EQUAL(eq.start({"AA", "KK", "QQ", "JJ", "TT", "99", "88"}), false);
    }

    TTEST_CASE("start() returns false when too few cards left in the deck")
    {
        // 2*2 + (4 + 1) + 43 = 52
        TTEST_EQUAL(eq.start({"33", "33"}, 0xf, 0xffffffffffe00), true);
        eq.stop();
        eq.wait();
        // 2*2 + (4 + 1) + 44 = 53
        TTEST_EQUAL(eq.start({"33", "33"}, 0xf, 0xfffffffffff00), false);
    }

    TTEST_CASE("start() returns false when hand range is empty after card removal")
    {
        TTEST_EQUAL(eq.start({"random", ""}), false);
        TTEST_EQUAL(eq.start({"AA", "22"}, CardRange::getCardMask("AsAh"), CardRange::getCardMask("Ac")), false);
    }

    TTEST_CASE("start() returns false when no feasible combination of holecards")
    {
        TTEST_EQUAL(eq.start({"AA", "AK"}, CardRange::getCardMask("As"), CardRange::getCardMask("Ah")), false);
    }

    TTEST_CASE("time limit")
    {
        eq.setTimeLimit(0.5);
        auto callback = [&](const EquityCalculator::Results& r){
            if (r.time >= 2)
                eq.stop();
        };
        eq.start({"random", "random"}, 0, 0, false, 0, callback, 1.0);
        eq.wait();
        auto r = eq.getResults();
        TTEST_EQUAL(r.time >= 0.45 && r.time <= 0.55, true);
    }

    TTEST_CASE("hand limit")
    {
        eq.setHandLimit(3000000);
        auto callback = [&](const EquityCalculator::Results& r){
            if (r.time >= 2)
                eq.stop();
        };
        eq.start({"random", "random"}, 0, 0, false, 0, callback, 1.0);
        eq.wait();
        auto r = eq.getResults();
        TTEST_EQUAL(r.hands >= 3000000 && r.hands <= 3000000 + 16 * 0x1000, true);
    }

    TTEST_CASE("test 1 - enumeration") { enumTest(TESTDATA[0]); }
    TTEST_CASE("test 1 - monte carlo") { monteCarloTest(TESTDATA[0]); }
    TTEST_CASE("test 2 - enumeration") { enumTest(TESTDATA[1]); }
    TTEST_CASE("test 2 - monte carlo") { monteCarloTest(TESTDATA[1]); }
    TTEST_CASE("test 3 - enumeration") { enumTest(TESTDATA[2]); }
    TTEST_CASE("test 3 - monte carlo") { monteCarloTest(TESTDATA[2]); }
    TTEST_CASE("test 4 - enumeration") { enumTest(TESTDATA[3]); }
    TTEST_CASE("test 4 - monte carlo") { monteCarloTest(TESTDATA[3]); }
    TTEST_CASE("test 5 - enumeration") { enumTest(TESTDATA[4]); }
    TTEST_CASE("test 5 - monte carlo") { monteCarloTest(TESTDATA[4]); }
    TTEST_CASE("test 6 - enumeration") { enumTest(TESTDATA[5]); }
    TTEST_CASE("test 6 - monte carlo") { monteCarloTest(TESTDATA[5]); }
};

void printBuildInfo()
{
    cout << "=== Build information ===" << endl;
    cout << "" << (sizeof(void*) * 8) << "-bit" << endl;
    #if OMP_x64
    cout << "x64" << endl;
    #endif
    #if OMP_SSE2
    cout << "SSE2" << endl;
    #else
    cout << "No SSE" << endl;
    #endif
    #if OMP_SSE4
    cout << "SSE4" << endl;
    #endif
}

int main()
{
    printBuildInfo();

    cout << endl << "=== Tests ===" << endl;
    cout << "Util:" << endl;
    UtilTest().run();
    cout << "Hand:" << endl;
    HandTest().run();
    cout << "HandEvaluator:" << endl;
    HandEvaluatorTest().run();
    cout << "EquityCalculator:" << endl;
    EquityCalculatorTest().run();

    cout << endl << endl << "=== Benchmarks ===" << endl;
    void benchmark();
    benchmark();
    cout << endl << "Done." << endl;
}
