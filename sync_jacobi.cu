// sync_jacobi.cu -- synchronous Jacobi baseline (Role A), contract-compliant version.
// Keeps the frozen solver's structure: one thread per cell, ping-pong buffers, global
// barrier = kernel boundary every iteration, max-|update| residual every M iterations.
// Changes vs jacobi_sync_frozen.cu: the agreed update equation (r + source term), 2D
// 32x8 blocks (the size chosen in sync-block-comparison.txt), CLI + JSON output for the
// harness, one run per process, shared problem definition (common.h).
//
// Build (from repo root; common.h lives in async/):
//   nvcc -O3 -arch=sm_120 -fmad=false -I async baseline/sync_jacobi.cu -o build/sync_jacobi
//   -fmad=false: use the SAME flag as the async solver (stop test sits at the float noise
//   floor, FMA contraction shifts iteration counts) and it matches NumPy bit-for-bit.
//
// Run: ./build/sync_jacobi --N 512 --source gaussian --r 10 --json
//      ./build/sync_jacobi --N 32 --source point --r 1 --iters 100 --dump-npy g.npy
// Last stdout line = JSON (INTEGRATION_CONTRACT.md); diagnostics go to stderr.

#include "common.h"
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstring>

#ifndef TX
#define TX 32
#endif
#ifndef TY
#define TY 8
#endif
#define NWARPS ((TX * TY) / 32)

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t e__ = (call);                                                 \
        if (e__ != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", cudaGetErrorName(e__),\
                    __FILE__, __LINE__, cudaGetErrorString(e__));                 \
            exit(3);                                                              \
        }                                                                         \
    } while (0)

// One thread per cell. u_new = ((old + r*(((N+S)+W)+E)) + src) / (1+4r).
// Residual (only when check != 0): max |update| over all interior cells, reduced
// per warp/block, then one atomicMax per block (same reduction as the async kernel,
// so the two solvers pay the same residual cost).
__global__ void jacobi_step(const float* __restrict__ old_grid,
                            float* __restrict__ new_grid,
                            const float* __restrict__ src,
                            int N, float r, float denom,
                            int check, unsigned int* max_bits)
{
    __shared__ float wmax[NWARPS];
    const int gx = blockIdx.x * TX + threadIdx.x;
    const int gy = blockIdx.y * TY + threadIdx.y;
    float diff = 0.0f;

    if (gx < N && gy < N && gx > 0 && gy > 0 && gx < N - 1 && gy < N - 1) {
        const size_t i = (size_t)gy * N + gx;
        const float old = old_grid[i];
        const float sum = ((old_grid[i - N] + old_grid[i + N]) + old_grid[i - 1]) + old_grid[i + 1];
        const float v = ((old + r * sum) + src[i]) / denom;
        new_grid[i] = v;                       // Dirichlet boundary is never written
        diff = fabsf(v - old);
    }

    if (check) {                               // uniform across the launch
        const int tid = threadIdx.y * TX + threadIdx.x;
        const int lane = tid & 31, wid = tid >> 5;
        float m = diff;
        for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
        if (lane == 0) wmax[wid] = m;
        __syncthreads();
        if (wid == 0) {
            m = (lane < NWARPS) ? wmax[lane] : 0.0f;
            for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
            if (lane == 0) atomicMax(max_bits, __float_as_uint(m));
        }
    }
}

int main(int argc, char** argv) {
    Config cfg;
    long long fixed_iters = 0;
    const char* dump = nullptr;

    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto nxt = [&]() -> const char* { return (i + 1 < argc) ? argv[++i] : ""; };
        if      (k == "--N")                 cfg.N = atoi(nxt());
        else if (k == "--source")          { if (!parse_source(nxt(), cfg.source)) { fprintf(stderr, "bad --source\n"); return 2; } }
        else if (k == "--r")                 cfg.r = atof(nxt());
        else if (k == "--tolerance")         cfg.tol = atof(nxt());
        else if (k == "--residual-interval") cfg.M = atoi(nxt());
        else if (k == "--max-iters")         cfg.max_iters = atoll(nxt());
        else if (k == "--iters")             fixed_iters = atoll(nxt());
        else if (k == "--dump-npy" || k == "--out") dump = nxt();
        else if (k == "--strategy" || k == "--run-id" || k == "--seed") (void)nxt();
        else if (k == "--json")              { /* always printed */ }
        else { fprintf(stderr, "unknown arg %s\n", k.c_str()); return 2; }
    }
    if (cfg.N % TX || cfg.N % TY || cfg.M < 1) { fprintf(stderr, "N must be divisible by %d and %d; M>=1\n", TX, TY); return 2; }

    const int N = cfg.N;
    const size_t bytes = (size_t)N * N * sizeof(float);
    Problem prob = init_problem(cfg);                         // not timed

    CUDA_CHECK(cudaFree(0));
    float *d_a, *d_b, *d_s, *d_src; unsigned int* d_res;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_s, bytes));                      // scratch for the warm-up launch
    CUDA_CHECK(cudaMalloc(&d_src, bytes));
    CUDA_CHECK(cudaMalloc(&d_res, sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpy(d_a, prob.u0.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, prob.u0.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_src, prob.src.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_res, 0, sizeof(unsigned int)));

    dim3 block(TX, TY), grid(N / TX, N / TY);
    {   // occupancy (stderr only)
        int nb = 0, dev = 0; cudaDeviceProp pr;
        CUDA_CHECK(cudaGetDevice(&dev));
        CUDA_CHECK(cudaGetDeviceProperties(&pr, dev));
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, jacobi_step, TX * TY, 0));
        fprintf(stderr, "[info] %s | block %dx%d | blocks/SM %d | theoretical occupancy %.1f%%\n",
                pr.name, TX, TY, nb, 100.0 * nb * (TX * TY / 32) / (pr.maxThreadsPerMultiProcessor / 32));
    }

    // Warm-up on scratch (writes only d_s) so module load is not in the timed region.
    jacobi_step<<<grid, block>>>(d_a, d_s, d_src, N, prob.r, prob.denom, 0, d_res);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    const bool fixed_mode = fixed_iters > 0;
    const long long limit = fixed_mode ? fixed_iters : cfg.max_iters;
    float *cur = d_a, *nxt = d_b;
    long long it = 0; double resid = 0.0; bool conv = false;
    Result res;

    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventRecord(e0));
    while (it < limit) {
        const bool check = ((it + 1) % cfg.M == 0) || (fixed_mode && it + 1 == limit);
        if (check) CUDA_CHECK(cudaMemsetAsync(d_res, 0, sizeof(unsigned int)));
        jacobi_step<<<grid, block>>>(cur, nxt, d_src, N, prob.r, prob.denom, check ? 1 : 0, d_res);
        std::swap(cur, nxt);
        ++it;
        if (check) {
            unsigned int bits = 0;
            CUDA_CHECK(cudaMemcpy(&bits, d_res, sizeof bits, cudaMemcpyDeviceToHost));
            float f; memcpy(&f, &bits, sizeof f); resid = f;
            if (!fixed_mode) {
                res.residual_history.push_back({it, resid});
                if (resid < cfg.tol) { conv = true; break; }
            }
        }
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));

    res.converged = conv; res.iterations = it; res.time_ms = ms; res.final_residual = resid;
    res.final_grid.resize((size_t)N * N);
    CUDA_CHECK(cudaMemcpy(res.final_grid.data(), cur, bytes, cudaMemcpyDeviceToHost));
    fprintf(stderr, "[info] sync iters=%lld time=%.3f ms (%.2f us/iter) residual=%.3e\n",
            it, ms, 1000.0 * ms / (it ? it : 1), resid);
    if (dump && !write_npy(dump, res.final_grid, N)) fprintf(stderr, "could not write %s\n", dump);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_s); cudaFree(d_src); cudaFree(d_res);
    print_result_json(res);
    return 0;
}
