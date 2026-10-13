// GPU-free test: g++ -std=c++17 -Wall -Wextra -Werror
//   async/test_residual_schedule.cpp -o /tmp/test_residual_schedule
#include "residual_schedule.h"

#include <cassert>
#include <climits>
#include <iostream>
#include <vector>

static std::vector<int> run(int D, long long limit, bool fixed,
                            const std::vector<long long>& expected_checks) {
    long long it = 0;
    std::vector<int> rounds;
    std::vector<long long> checks;
    while (it < limit) {
        const auto round = next_residual_round(it, limit, D, 10, fixed);
        assert(round.sweeps > 0 && round.sweeps <= D);
        assert(round.sweeps <= limit - it);
        if (!fixed) assert(round.sweeps <= 10 - it % 10);
        it += round.sweeps;
        rounds.push_back(round.sweeps);
        if (round.check) checks.push_back(it);
    }
    assert(it == limit);
    assert(checks == expected_checks);
    return rounds;
}

int main() {
    for (int D : {1, 2, 4, 8}) {
        run(D, 30, false, {10, 20, 30});
        run(D, 23, false, {10, 20, 23});
        run(D, 3, false, {3}); // cap before the first checkpoint
        run(D, 23, true, {23});
    }
    assert((run(4, 20, false, {10, 20}) ==
            std::vector<int>{4, 4, 2, 4, 4, 2}));
    assert((run(8, 20, false, {10, 20}) ==
            std::vector<int>{8, 2, 8, 2}));
    assert((run(4, 23, false, {10, 20, 23}) ==
            std::vector<int>{4, 4, 2, 4, 4, 2, 3}));
    assert((run(8, 23, false, {10, 20, 23}) ==
            std::vector<int>{8, 2, 8, 2, 3}));
    // Fixed mode does not split at M or stop early.
    assert((run(4, 23, true, {23}) == std::vector<int>{4, 4, 4, 4, 4, 3}));
    assert((run(8, 23, true, {23}) == std::vector<int>{8, 8, 7}));
    const auto large = next_residual_round(LLONG_MAX - 3, LLONG_MAX, 8, 10, false);
    assert(large.sweeps == 3 && large.check); // no checkpoint arithmetic overflow
    const auto every = next_residual_round(0, 5, 8, 1, false);
    assert(every.sweeps == 1 && every.check);
    run(8, 0, false, {});
    std::cout << "Residual scheduling tests passed\n";
}
