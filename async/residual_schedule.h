#pragma once

#include <algorithm>

struct ResidualRound {
    int sweeps;
    bool check;
};

// Called only with it < limit and positive D and M. Fixed mode keeps full
// D-sweep rounds and samples only at the end; convergence mode aligns to M.
inline ResidualRound next_residual_round(long long it, long long limit,
                                         int D, int M, bool fixed_mode) {
    long long sweeps = std::min<long long>(D, limit - it);
    if (!fixed_mode)
        sweeps = std::min<long long>(sweeps, M - it % M);
    const long long end = it + sweeps;
    return {static_cast<int>(sweeps),
            end == limit || (!fixed_mode && end % M == 0)};
}
