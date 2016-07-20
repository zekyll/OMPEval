
#include "OMPEval.h"
#include <iostream>
#include <chrono>

using namespace std;

// Measure evaluation speed when enumerating card combinations in order.
void enumerationSpeedTest()
{
    cout << "Enumeration speed test:" << endl;

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

int main()
{
    enumerationSpeedTest();
    cout << "Done." << endl;
}
