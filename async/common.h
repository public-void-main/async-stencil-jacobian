// common.h -- shared by the CPU cross-check, the sync solver and the async solver.
// Plain C++ (no CUDA calls): compiles with g++ and nvcc.
//
// Source of truth: experiment_spec.json (harness_ml/). Summary:
//   u_new = (u_old + r*(north+south+west+east) + dt_times_source) / (1 + 4r)
//   where u_old is the CURRENT ITERATE of the cell (reference.py: `old`), NOT a frozen
//   previous time step. Evaluation order (float32, matches reference.py bit-for-bit
//   when FMA is off):  ((u_old + r*(((N+S)+W)+E)) + src) / denom
//   float32 state. Dirichlet: top edge = 100, bottom/left/right = 0,
//   corners belong to the top edge. Top edge = row y=0 (confirmed in problem.py).
//   Source peak dt*f = 1 at (N/2, N/2) (integer floor), zero on the boundary ring.
//     point   : single cell
//     gaussian: exp(-(dx^2+dy^2)/(2 sigma^2)), sigma = max(N/16,1) cells
//               (confirmed in problem.py)
//   Initial interior u = 0. Fixed point = steady state, so convergence is SLOW
//   (~N^2 iterations: reference.py gives 10700 iters at N=64, r=1).
//   residual  = max |update| of one sweep, sampled every M=10 iterations.
//   converged when residual < 1e-4 (strictly). (Was 1e-5, which sits at the float32 noise floor.)
//   NOTE: 1 ulp of float32 at 100 is 7.6e-6, so the old tol of 1e-5 sat at the float noise
//   floor (residual flickered 7.6e-6/1.1e-5/1.5e-5). 1e-4 is safely above it. Compile every solver with FMA off (nvcc -fmad=false,
//   g++ -ffp-contract=off) so results match reference.py bit-for-bit.
//
// Layout: idx = y*N + x (row-major). Boundary ring is never updated.
// 1 iteration = 1 Jacobi sweep over all interior cells.

#pragma once
#include <vector>
#include <string>
#include <utility>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <algorithm>

enum class SourceType { POINT = 0, GAUSSIAN = 1 };

struct Config {
    int       N         = 512;
    SourceType source   = SourceType::GAUSSIAN;
    double    r         = 1.0;         // alpha*dt/h^2
    double    tol       = 1e-4;
    int       M         = 10;          // residual interval (iterations)
    long long max_iters = 2000000;
    int       D         = 1;           // async: local sweeps per halo refresh
};

struct Problem {
    int   N;
    float r, denom;                    // denom = 1 + 4r
    std::vector<float> u0;             // initial grid incl. Dirichlet boundary
    std::vector<float> src;            // dt*f  (constant for the solve)
};

struct Result {
    bool      converged      = false;
    long long iterations     = 0;
    double    time_ms        = 0.0;
    double    final_residual = 0.0;
    std::vector<std::pair<long long, double>> residual_history;   // {iteration, residual}
    std::vector<float> final_grid;
};

inline bool is_boundary(int x, int y, int N) {
    return x == 0 || y == 0 || x == N - 1 || y == N - 1;
}

inline bool parse_source(const std::string& s, SourceType& out) {
    if (s == "point")    { out = SourceType::POINT;    return true; }
    if (s == "gaussian") { out = SourceType::GAUSSIAN; return true; }
    return false;
}

inline Problem init_problem(const Config& cfg) {
    const int N = cfg.N;
    Problem p;
    p.N = N;
    p.r = (float)cfg.r;
    p.denom = 1.0f + 4.0f * p.r;
    p.u0.assign((size_t)N * N, 0.0f);
    p.src.assign((size_t)N * N, 0.0f);

    // Dirichlet: top row = 100 (corners included), other edges stay 0.
    for (int x = 0; x < N; ++x) p.u0[x] = 100.0f;

    const int    c     = N / 2;                  // floor(N/2)
    const double sigma = N / 16.0;
    for (int y = 1; y < N - 1; ++y)
        for (int x = 1; x < N - 1; ++x) {
            double dts = 0.0;
            if (cfg.source == SourceType::POINT) {
                dts = (x == c && y == c) ? 1.0 : 0.0;
            } else {
                double dx = x - c, dy = y - c;
                dts = std::exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma));
            }
            size_t i = (size_t)y * N + x;
            p.src[i] = (float)dts;                         // dt*f
        }
    return p;
}

// NumPy .npy (v1.0, float32, C order, shape (N,N)) -- loadable with np.load.
inline bool write_npy(const char* path, const std::vector<float>& g, int N) {
    FILE* f = fopen(path, "wb");
    if (!f) return false;
    char dict[128];
    int dl = snprintf(dict, sizeof dict,
        "{'descr': '<f4', 'fortran_order': False, 'shape': (%d, %d), }", N, N);
    int total = 10 + dl + 1;                       // prefix + dict + '\n'
    int pad = (64 - total % 64) % 64;
    uint16_t hlen = (uint16_t)(dl + pad + 1);
    fwrite("\x93NUMPY\x01\x00", 1, 8, f);
    fwrite(&hlen, 2, 1, f);                        // little-endian host assumed
    fwrite(dict, 1, dl, f);
    for (int i = 0; i < pad; ++i) fputc(' ', f);
    fputc('\n', f);
    fwrite(g.data(), sizeof(float), g.size(), f);
    fclose(f);
    return true;
}

inline double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
        m = std::max(m, std::fabs((double)a[i] - (double)b[i]));
    return m;
}

// Print the JSON result line required by INTEGRATION_CONTRACT.md (last stdout line).
inline void print_result_json(const Result& r) {
    printf("{\"iterations\": %lld, \"time_ms\": %.6f, \"final_residual\": %.9e, "
           "\"converged\": %s, \"residual_history\": [",
           r.iterations, r.time_ms, r.final_residual, r.converged ? "true" : "false");
    for (size_t i = 0; i < r.residual_history.size(); ++i)
        printf("%s[%lld, %.9e]", i ? ", " : "",
               r.residual_history[i].first, r.residual_history[i].second);
    printf("]}\n");
    fflush(stdout);
}
