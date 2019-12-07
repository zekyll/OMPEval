#include "omp/EquityCalculator.h"
#include "omp/HandEvaluator.h"
#include <iostream>
using namespace omp;
using namespace std;

int main()
{
    HandEvaluator eval;
    Hand h = Hand::empty(); // Final hand must include empty() exactly once!
    h += Hand(51) + Hand(48) + Hand(0) + Hand(1) + Hand(2); // AdAs2s2h2c
    std::cout << eval.evaluate(h) << std::endl; // 28684 = 7 * 4096 + 12

    EquityCalculator eq;
    eq.start({"AK", "QQ"});
    eq.wait();
    auto r1 = eq.getResults();
    std::cout << r1.equity[0] << " " << r1.equity[1] << std::endl;

    vector<CardRange> ranges{"QQ+,AKs,AcQc", "A2s+", "random"};
    uint64_t board = CardRange::getCardMask("2c4c5h");
    uint64_t dead = CardRange::getCardMask("Jc");
    double stdErrMargin = 2e-5; // stop when standard error below 0.002%
    auto callback = [&eq](const EquityCalculator::Results& results) {
        cout << results.equity[0] << " " << 100 * results.progress
                << " " << 1e-6 * results.intervalSpeed << endl;
        if (results.time > 5) // Stop after 5s
            eq.stop();
    };
    double updateInterval = 0.25; // Callback called every 0.25s.
    unsigned threads = 0; // max hardware parallelism (default)
    eq.start(ranges, board, dead, false, stdErrMargin, callback, updateInterval, threads);
    eq.wait();
    auto r2 = eq.getResults();
    cout << endl << r2.equity[0] << " " << r2.equity[1] << " " << r2.equity[2] << endl;
    cout << r2.wins[0] << " " << r2.wins[1] << " " << r2.wins[2] << endl;
    cout << r2.hands << " " << r2.time << " " << 1e-6 * r2.speed << " " << r2.stdev << endl;
}