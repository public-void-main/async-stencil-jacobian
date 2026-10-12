// cpu_reference.cpp -- C++ CPU Jacobi cross-check (NOT a replacement for
// Vaibhav's NumPy reference.py; use it to sanity-check iteration counts and to
// produce .npy grids for the GPU comparison).
//
// Build: g++ -O2 -std=c++17 cpu_reference.cpp -o cpu_ref
// Usage: ./cpu_ref --N 512 --source gaussian --r 10 [--iters K] [--out grid.npy]
//   --iters K : run exactly K sweeps (no convergence stop) -- for grid validation
//   default   : run until residual < tol (checked every M iterations)

#include "common.h"
#include <chrono>
#include <cstdlib>

Result cpu_solve(const Config& cfg, const Problem& p, long long fixed_iters) {
    const int N = cfg.N;
    std::vector<float> a = p.u0, b = p.u0;       // ping-pong, both carry boundary
    Result res;
    long long it = 0;
    double last_upd = 0.0;
    auto t0 = std::chrono::steady_clock::now();
    const long long limit = fixed_iters > 0 ? fixed_iters : cfg.max_iters;
    while (it < limit) {
        float mx = 0.0f;
        for (int y = 1; y < N - 1; ++y)
            for (int x = 1; x < N - 1; ++x) {
                size_t i = (size_t)y * N + x;
                float sum = ((a[i - N] + a[i + N]) + a[i - 1]) + a[i + 1];  // N+S+W+E
                float v = ((a[i] + p.r * sum) + p.src[i]) / p.denom;
                mx = std::max(mx, std::fabs(v - a[i]));
                b[i] = v;
            }
        std::swap(a, b);
        ++it;
        last_upd = mx;
        if (fixed_iters == 0 && it % cfg.M == 0) {
            res.residual_history.push_back({it, last_upd});
            if (last_upd < cfg.tol) { res.converged = true; break; }
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    res.iterations = it;
    res.final_residual = last_upd;
    res.time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    res.final_grid = a;
    return res;
}

#ifndef NO_MAIN
int main(int argc, char** argv) {
    Config cfg; long long iters = 0; const char* out = nullptr;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto nxt = [&]() -> const char* { return (i + 1 < argc) ? argv[++i] : ""; };
        if      (k == "--N")         cfg.N = atoi(nxt());
        else if (k == "--source")  { if (!parse_source(nxt(), cfg.source)) { fprintf(stderr, "bad --source\n"); return 2; } }
        else if (k == "--r")         cfg.r = atof(nxt());
        else if (k == "--iters")     iters = atoll(nxt());
        else if (k == "--out")       out = nxt();
        else if (k == "--max-iters") cfg.max_iters = atoll(nxt());
        else if (k == "--tolerance") cfg.tol = atof(nxt());
        else { fprintf(stderr, "unknown arg %s\n", k.c_str()); return 2; }
    }
    Problem p = init_problem(cfg);
    Result r = cpu_solve(cfg, p, iters);
    if (out) write_npy(out, r.final_grid, cfg.N);
    print_result_json(r);
    return 0;
}
#endif
