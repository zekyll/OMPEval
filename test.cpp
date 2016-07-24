
#include "OMPEval.h"
#include <iostream>
#include <chrono>
#include <random>

using namespace std;
using namespace omp;

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
                            Hand h5 = Hand::empty();
                            h5.combine(c1);
                            h5.combine(c2);
                            h5.combine(c3);
                            h5.combine(c4);
                            h5.combine(c5);
                            for (unsigned c6 = c5 + 1; c6 < 52; c6++) {
                                Hand h6 = h5;
                                h6.combine(c6);
                                for (unsigned c7 = c6 + 1; c7 < 52; c7++) {
                                    ++count;
                                    Hand h7 = h6;
                                    h7.combine(c7);
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
    cout << endl << "Random order evaluation (precalculated Hand objects):" << endl;
    mt19937_64 rng(0);
    uniform_int_distribution<unsigned> rnd(0, 51);
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
            table.back().combine(card);
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

// Evaluating random hands.
void randomEvaluationBenchmark2()
{
    cout << endl << "Random order evaluation (card arrays):" << endl;
    mt19937_64 rng(0);
    uniform_int_distribution<unsigned> rnd(0, 51);
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
        for (auto& hand: table) {
            Hand h = Hand::empty();
            h.combine(hand[0]);
            h.combine(hand[1]);
            h.combine(hand[2]);
            h.combine(hand[3]);
            h.combine(hand[4]);
            h.combine(hand[5]);
            h.combine(hand[6]);
            sum += eval.evaluate(h);
            ++count;
        }
    }

    auto t2 = chrono::high_resolution_clock::now();
    double t = 1e-9 * chrono::duration_cast<chrono::nanoseconds>(t2 - t1).count();
    cout << count << " evals  " << (1e-6 * count / t) << "M/s  " << t << "s  " << sum << endl;
}

int main()
{
    sequentialEvaluationBenchmark();
    randomEvaluationBenchmark();
    randomEvaluationBenchmark2();
    cout << endl << "Done." << endl;
}
