// async_solver.cu -- block-asynchronous Jacobi solver (Role B). One process = one run.
//
// Build (RTX 50-series, CUDA 12.8+):
//   nvcc -O3 -arch=sm_120 -fmad=false async_solver.cu -o build/async_jacobi
//   -fmad=false matters: the stop test sits at the float32 noise floor (1 ulp at 100 =
//   7.6e-6 < tol 1e-5), so FMA contraction changes iteration counts (N=128, r=100: 31620 vs
//   31950 on CPU). Sync and async MUST be built with the same flag, and the same update
//   expression, or R_I is polluted by rounding noise. -fmad=false also matches NumPy.
//   optional tile shape: -DTX=16 -DTY=16   (default 32x8, same as the sync solver)
//
// Run (examples):
//   ./build/async_jacobi --N 512 --source gaussian --r 10 --D 4
//   ./build/async_jacobi --N 512 --source point --r 1 --D 1 --barrier            # debug: == sync
//   ./build/async_jacobi --N 32 --source point --r 1 --D 1 --barrier --iters 100 --dump-npy g.npy
//
// Output: the LAST stdout line is one JSON object (INTEGRATION_CONTRACT.md):
//   {"iterations":..,"time_ms":..,"final_residual":..,"converged":..,"residual_history":[[it,res],..]}
// Everything else goes to stderr.
//
// Counting convention: 1 iteration = 1 Jacobi sweep. Residual (max |update| of one
// sweep) is sampled every M iterations; each free-running launch is exactly M sweeps
// (with D = 4, 8 the last round of a launch is shorter), so iteration counts are
// directly comparable with the sync solver.

#include "common.h"
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstring>
#include "async_kernel.cuh"

#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t e__ = (call);                                                \
        if (e__ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", cudaGetErrorName(e__),\
                    __FILE__, __LINE__, cudaGetErrorString(e__));                \
            exit(3);                                                             \
        }                                                                        \
    } while (0)

// Host-side check of the true residual of a final grid (outside the timed region).
static double true_max_update(const std::vector<float>& u, const Problem& p) {
    const int N = p.N; double m = 0.0;
    for (int y = 1; y < N - 1; ++y)
        for (int x = 1; x < N - 1; ++x) {
            size_t i = (size_t)y * N + x;
            float sum = ((u[i - N] + u[i + N]) + u[i - 1]) + u[i + 1];
            float v = ((u[i] + p.r * sum) + p.src[i]) / p.denom;
            m = std::max(m, (double)std::fabs(v - u[i]));
        }
    return m;
}

int main(int argc, char** argv) {
    Config cfg;
    bool barrier = false;
    long long fixed_iters = 0;                 // >0: run exactly this many sweeps
    const char* dump = nullptr;
    std::string strategy;
    bool D_given = false;

    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto nxt = [&]() -> const char* { return (i + 1 < argc) ? argv[++i] : ""; };
        if      (k == "--N")                 cfg.N = atoi(nxt());
        else if (k == "--source")          { if (!parse_source(nxt(), cfg.source)) { fprintf(stderr, "bad --source\n"); return 2; } }
        else if (k == "--r")                 cfg.r = atof(nxt());
        else if (k == "--D")               { cfg.D = atoi(nxt()); D_given = true; }
        else if (k == "--strategy")          strategy = nxt();          // e.g. D4
        else if (k == "--tolerance")         cfg.tol = atof(nxt());
        else if (k == "--residual-interval") cfg.M = atoi(nxt());
        else if (k == "--max-iters")         cfg.max_iters = atoll(nxt());
        else if (k == "--barrier")           barrier = true;
        else if (k == "--json")              { /* JSON is always printed; accepted for solvers.json */ }
        else if (k == "--iters")             fixed_iters = atoll(nxt());
        else if (k == "--dump-npy" || k == "--out") dump = nxt();
        else if (k == "--run-id" || k == "--seed")  (void)nxt();        // accepted, unused
        else { fprintf(stderr, "unknown arg %s\n", k.c_str()); return 2; }
    }
    if (!D_given && strategy.size() > 1 && strategy[0] == 'D') cfg.D = atoi(strategy.c_str() + 1);
    if (barrier) cfg.D = 1;
    if (cfg.N % TX || cfg.N % TY) { fprintf(stderr, "N must be divisible by %d and %d\n", TX, TY); return 2; }
    if (cfg.D < 1 || cfg.M < 1)   { fprintf(stderr, "bad D or M\n"); return 2; }

    const int N = cfg.N;
    const size_t bytes = (size_t)N * N * sizeof(float);
    Problem prob = init_problem(cfg);                       // not timed

    CUDA_CHECK(cudaFree(0));                                // create context before timing
    float *d_a = nullptr, *d_b = nullptr, *d_src = nullptr;
    unsigned int* d_res = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_src, bytes));
    CUDA_CHECK(cudaMalloc(&d_res, sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpy(d_a, prob.u0.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, prob.u0.data(), bytes, cudaMemcpyHostToDevice)); // boundary in both
    CUDA_CHECK(cudaMemcpy(d_src, prob.src.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_res, 0, sizeof(unsigned int)));

    dim3 block(TX, TY), grid(N / TX, N / TY);

    // Occupancy info (stderr only; for profiling tables)
    {
        int nb = 0, dev = 0; cudaDeviceProp pr;
        CUDA_CHECK(cudaGetDevice(&dev));
        CUDA_CHECK(cudaGetDeviceProperties(&pr, dev));
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, async_jacobi_kernel, TX * TY, 0));
        double occ = 100.0 * nb * (TX * TY / 32) / (pr.maxThreadsPerMultiProcessor / 32);
        fprintf(stderr, "[info] %s | tile %dx%d | blocks/SM %d | theoretical occupancy %.1f%% | SMs %d | blocks in grid %d\n",
                pr.name, TX, TY, nb, occ, pr.multiProcessorCount, grid.x * grid.y);
    }

    // Warm-up: a launch with 0 sweeps loads the module and touches nothing.
    async_jacobi_kernel<<<grid, block>>>(d_a, d_a, d_src, N, prob.r, prob.denom, 1, 0, 0, d_res);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    const bool fixed_mode = fixed_iters > 0;
    const long long limit = fixed_mode ? fixed_iters : cfg.max_iters;
    Result res;
    float* cur = d_a;                      // holds the current state
    float* nxt = d_b;                      // barrier mode only
    long long it = 0;
    double resid = 0.0;
    bool conv = false;

    auto read_residual = [&]() {
        unsigned int bits = 0;
        CUDA_CHECK(cudaMemcpy(&bits, d_res, sizeof bits, cudaMemcpyDeviceToHost));
        float f; memcpy(&f, &bits, sizeof f);
        return (double)f;
    };

    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventRecord(e0));

    if (barrier) {
        // Debug / sync-equivalent: 1 sweep per launch, ping-pong, kernel boundary = global barrier.
        while (it < limit) {
            const bool check = ((it + 1) % cfg.M == 0) || (fixed_mode && it + 1 == limit);
            if (check) CUDA_CHECK(cudaMemsetAsync(d_res, 0, sizeof(unsigned int)));
            async_jacobi_kernel<<<grid, block>>>(cur, nxt, d_src, N, prob.r, prob.denom, 1, 1, check ? 1 : 0, d_res);
            std::swap(cur, nxt);
            ++it;
            if (check) {
                resid = read_residual();
                if (!fixed_mode) {
                    res.residual_history.push_back({it, resid});
                    if (resid < cfg.tol) { conv = true; break; }
                }
            }
        }
    } else {
        // Free-running: in-place, one launch = M sweeps in rounds of D, no grid barrier inside.
        while (it < limit) {
            const int sw = (int)std::min<long long>(cfg.M, limit - it);
            const bool check = !fixed_mode || (it + sw == limit);
            if (check) CUDA_CHECK(cudaMemsetAsync(d_res, 0, sizeof(unsigned int)));
            async_jacobi_kernel<<<grid, block>>>(cur, cur, d_src, N, prob.r, prob.denom, cfg.D, sw, check ? 1 : 0, d_res);
            it += sw;
            if (check) {
                resid = read_residual();
                if (!fixed_mode) {
                    res.residual_history.push_back({it, resid});
                    if (resid < cfg.tol) { conv = true; break; }
                }
            }
        }
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));

    res.converged = conv;
    res.iterations = it;
    res.time_ms = ms;
    res.final_residual = resid;
    res.final_grid.resize((size_t)N * N);
    CUDA_CHECK(cudaMemcpy(res.final_grid.data(), cur, bytes, cudaMemcpyDeviceToHost));

    // Sanity (outside timing): the true max update of the final grid.
    fprintf(stderr, "[info] mode=%s D=%d iters=%lld time=%.3f ms reported_res=%.3e true_res=%.3e\n",
            barrier ? "barrier" : "async", cfg.D, it, ms, resid, true_max_update(res.final_grid, prob));
    if (dump && !write_npy(dump, res.final_grid, N)) fprintf(stderr, "could not write %s\n", dump);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_src); cudaFree(d_res);
    print_result_json(res);                                  // LAST stdout line
    return 0;
}
