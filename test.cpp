
#include "omp/HandEvaluator.h"
#include "omp/EquityCalculator.h"
#include "omp/Random.h"
#include "ttest/ttest.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <numeric>
#include <cmath>

using namespace std;
using namespace omp;

class HandTest : public ttest::TestBase
{
    TTEST_CASE("empty()")
	{
        Hand h = Hand::empty();
        TTEST_EQUAL(h.count(), 0);
        TTEST_EQUAL(h.suitCount(0), 0);
	}

    TTEST_CASE("adding & removing cards")
	{
        Hand h = Hand::empty() + Hand(5);
        TTEST_EQUAL(h.count(), 1);
        TTEST_EQUAL(h.suitCount(1), 1);
        h += Hand(51);
        TTEST_EQUAL(h.count(), 2);
        TTEST_EQUAL(h.suitCount(0), 0);
        TTEST_EQUAL(h.suitCount(1), 1);
        TTEST_EQUAL(h.suitCount(3), 1);
        h += Hand(3);
        TTEST_EQUAL(h.count(), 3);
        TTEST_EQUAL(h.suitCount(0), 0);
        TTEST_EQUAL(h.suitCount(1), 1);
        TTEST_EQUAL(h.suitCount(3), 2);
        h -= Hand(51);
        TTEST_EQUAL(h.count(), 2);
        TTEST_EQUAL(h.suitCount(0), 0);
        TTEST_EQUAL(h.suitCount(1), 1);
        TTEST_EQUAL(h.suitCount(3), 1);
        h = h - (Hand(3) + Hand(51));
        TTEST_EQUAL(h.count(), 0);
	}
};

class HandEvaluatorTest : public ttest::TestBase
{
    HandEvaluator e;
    uint64_t counts[10]{};

    void enumerate(unsigned cardsLeft, Hand h = Hand::empty(), unsigned start = 0)
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
        double hands = accumulate(tc.expectedResults.begin(), tc.expectedResults.end(), 0);
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

// Evaluating hands in sequential order.
void sequentialEvaluationBenchmark()
{
    cout << endl << "Sequential evaluation:" << endl;

    HandEvaluator eval;
    unsigned sum = 0;
    uint64_t count = 0;

    auto t1 = chrono::high_resolution_clock::now();

    for (unsigned i = 0; i < 5; ++i) {
        for (unsigned c1 = 0; c1 < 52; c1++) {
            for (unsigned c2 = c1 + 1; c2 < 52; c2++) {
                for (unsigned c3 = c2 + 1; c3 < 52; c3++) {
                    for (unsigned c4 = c3 + 1; c4 < 52; c4++) {
                        for (unsigned c5 = c4 + 1; c5 < 52; c5++) {
                            Hand h5 = Hand::empty() + c1 + c2 + c3 + c4 + c5;
                            for (unsigned c6 = c5 + 1; c6 < 52; c6++) {
                                Hand h6 = h5 + c6;
                                for (unsigned c7 = c6 + 1; c7 < 52; c7++) {
                                    ++count;
                                    Hand h7 = h6 + c7;
                                    sum += eval.evaluate(h7);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    auto t2 = chrono::high_resolution_clock::now();
    double t = 1e-9 * chrono::duration_cast<chrono::nanoseconds>(t2 - t1).count();
    cout << count << " evals  " << (1e-6 * count / t) << "M/s  " << t << "s  " << sum << endl;
}

// Evaluating random hands.
void randomEvaluationBenchmark()
{
    cout << endl << "Random order evaluation (card arrays):" << endl;
    XoroShiro128Plus rng(0);
    FastUniformIntDistribution<unsigned> rnd(0, 51);
    HandEvaluator eval;
    uint64_t count = 0;
    unsigned sum = 0;

    vector<array<uint8_t,7>> table;
    for (unsigned i = 0; i < 10000000; ++i) {
        uint64_t usedCardsMask = 0;
        table.push_back({});
        for(auto& card : table.back()) {
            uint64_t cardMask;
            do {
                card = rnd(rng);
                cardMask = 1ull << card;
            } while (usedCardsMask & cardMask);
            usedCardsMask |= cardMask;
        }
    }

    auto t1 = chrono::high_resolution_clock::now();

    for (int i = 0; i < 50; ++i) {
        for (auto& h: table) {
            Hand hand = Hand::empty() + h[0] + h[1] + h[2] + h[3] + h[4] + h[5] + h[6];
            sum += eval.evaluate(hand);
            ++count;
        }
    }

    auto t2 = chrono::high_resolution_clock::now();
    double t = 1e-9 * chrono::duration_cast<chrono::nanoseconds>(t2 - t1).count();
    cout << count << " evals  " << (1e-6 * count / t) << "M/s  " << t << "s  " << sum << endl;
}

// Evaluating random hands.
void randomEvaluationBenchmark2()
{
    cout << endl << "Random order evaluation (precalculated Hand objects):" << endl;
    XoroShiro128Plus rng(0);
    FastUniformIntDistribution<unsigned> rnd(0, 51);
    HandEvaluator eval;
    uint64_t count = 0;
    unsigned sum = 0;

    vector<Hand> table;
    for (unsigned i = 0; i < 10000000; ++i) {
        uint64_t usedCardsMask = 0;
        table.push_back(Hand::empty());
        for(unsigned j = 0; j < 7; ++j) {
            unsigned card;
            uint64_t cardMask;
            do {
                card = rnd(rng);
                cardMask = 1ull << card;
            } while (usedCardsMask & cardMask);
            usedCardsMask |= cardMask;
            table.back() += card;
        }
    }

    auto t1 = chrono::high_resolution_clock::now();

    for (int i = 0; i < 50; ++i) {
        for (auto& hand: table) {
            sum += eval.evaluate(hand);
            ++count;
        }
    }

    auto t2 = chrono::high_resolution_clock::now();
    double t = 1e-9 * chrono::duration_cast<chrono::nanoseconds>(t2 - t1).count();
    cout << count << " evals  " << (1e-6 * count / t) << "M/s  " << t << "s  " << sum << endl;
}

int main()
{
    cout << "=== Tests ===" << endl;
    cout << "Hand:" << endl;
    HandTest().run();
    cout << "HandEvaluator:" << endl;
    HandEvaluatorTest().run();
    cout << "EquityCalculator:" << endl;
    EquityCalculatorTest().run();

    cout << endl << endl << "=== Benchmarks ===" << endl;
    sequentialEvaluationBenchmark();
    randomEvaluationBenchmark();
    randomEvaluationBenchmark2();
    cout << endl << "Done." << endl;
}
