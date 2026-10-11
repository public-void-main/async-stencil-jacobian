
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define VALIDATION_LIMIT 10000

#define CHECK(call) do {                                      \
    cudaError_t err = (call);                                 \
    if (err != cudaSuccess) {                                 \
        fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,    \
                cudaGetErrorString(err));                     \
        exit(EXIT_FAILURE);                                   \
    }                                                         \
} while (0)

/* One CUDA thread per cell; five-point Jacobi stencil. */
__global__ void jacobi_step(
    const float *old_grid,
    float *new_grid,
    unsigned int *max_residual_bits,
    int N,
    int check_residual)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * N;

    if (idx >= total)
        return;

    int i = idx / N;
    int j = idx % N;

    // Fixed boundaries remain unchanged.
    if (i == 0 || i == N - 1 ||
        j == 0 || j == N - 1)
        return;

    float value = 0.25f * (
        old_grid[(i - 1) * N + j] +
        old_grid[(i + 1) * N + j] +
        old_grid[i * N + j - 1] +
        old_grid[i * N + j + 1]
    );

    new_grid[idx] = value;

    if (check_residual) {
        float diff = fabsf(value - old_grid[idx]);
        unsigned int bits = __float_as_uint(diff);

        // Maximum absolute update across all interior cells.
        atomicMax(max_residual_bits, bits);
    }
}

/* Top boundary = 100; all other cells = 0. */
void initialize_grid(float *grid, int N)
{
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            grid[i * N + j] = (i == 0) ? 100.0f : 0.0f;
        }
    }
}

/* CPU reference: same initial condition and Jacobi update. */
void cpu_jacobi(float *grid, int N, int iterations)
{
    size_t bytes = (size_t)N * N * sizeof(float);

    float *a = (float *)malloc(bytes);
    float *b = (float *)malloc(bytes);

    if (a == NULL || b == NULL) {
        fprintf(stderr, "CPU memory allocation failed\n");
        free(a);
        free(b);
        exit(EXIT_FAILURE);
    }

    initialize_grid(a, N);
    initialize_grid(b, N);

    float *old_grid = a;
    float *new_grid = b;

    for (int iter = 0; iter < iterations; iter++) {
        for (int i = 1; i < N - 1; i++) {
            for (int j = 1; j < N - 1; j++) {
                new_grid[i * N + j] = 0.25f * (
                    old_grid[(i - 1) * N + j] +
                    old_grid[(i + 1) * N + j] +
                    old_grid[i * N + j - 1] +
                    old_grid[i * N + j + 1]
                );
            }
        }

        float *temp = old_grid;
        old_grid = new_grid;
        new_grid = temp;
    }

    for (int i = 0; i < N * N; i++) {
        grid[i] = old_grid[i];
    }

    free(a);
    free(b);
}

void run_solver(int N, int max_iters, float tolerance)
{
    size_t bytes = (size_t)N * N * sizeof(float);

    float *h_grid = (float *)malloc(bytes);

    if (h_grid == NULL) {
        fprintf(stderr, "Host memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    initialize_grid(h_grid, N);

    float *d_a = NULL;
    float *d_b = NULL;
    unsigned int *d_max_residual = NULL;

    CHECK(cudaMalloc((void **)&d_a, bytes));
    CHECK(cudaMalloc((void **)&d_b, bytes));
    CHECK(cudaMalloc((void **)&d_max_residual,
                     sizeof(unsigned int)));

    CHECK(cudaMemcpy(d_a, h_grid, bytes,
                     cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, h_grid, bytes,
                     cudaMemcpyHostToDevice));

    int total = N * N;
    int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;

    float *old_grid = d_a;
    float *new_grid = d_b;

    float residual = INFINITY;
    int iterations = 0;
    int converged = 0;

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    // Time GPU iterations, including residual checks.
    CHECK(cudaEventRecord(start));

    for (int iter = 1; iter <= max_iters; iter++) {
        int check_residual = (iter % 10 == 0);

        if (check_residual) {
            CHECK(cudaMemset(d_max_residual, 0,
                             sizeof(unsigned int)));
        }

        jacobi_step<<<blocks, BLOCK_SIZE>>>(
            old_grid,
            new_grid,
            d_max_residual,
            N,
            check_residual
        );

        CHECK(cudaGetLastError());

        // Swap after launching each iteration.
        float *temp = old_grid;
        old_grid = new_grid;
        new_grid = temp;

        iterations = iter;

        if (check_residual) {
            unsigned int bits;

            CHECK(cudaMemcpy(
                &bits,
                d_max_residual,
                sizeof(bits),
                cudaMemcpyDeviceToHost
            ));

            union {
                unsigned int u;
                float f;
            } conversion;

            conversion.u = bits;
            residual = conversion.f;

            if (residual < tolerance) {
                converged = 1;
                break;
            }
        }
    }

    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0.0f;
    CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    CHECK(cudaMemcpy(h_grid, old_grid, bytes,
                     cudaMemcpyDeviceToHost));

    printf("\n====================================\n");
    printf("Grid size N           : %d\n", N);
    printf("Iterations            : %d\n", iterations);
    printf("Final residual        : %.8e\n", residual);
    printf("Tolerance             : %.8e\n", tolerance);
    printf("Convergence status    : %s\n",
           converged ? "TOLERANCE REACHED" : "MAX ITERATIONS REACHED");
    printf("GPU time              : %.3f ms\n", milliseconds);
    printf("Center value (GPU)    : %.6f\n",
           h_grid[(N / 2) * N + (N / 2)]);

    // Validate against CPU at manageable iteration counts.
    if (iterations <= VALIDATION_LIMIT) {
        float *cpu_grid = (float *)malloc(bytes);

        if (cpu_grid == NULL) {
            fprintf(stderr, "CPU reference allocation failed\n");
            exit(EXIT_FAILURE);
        }

        cpu_jacobi(cpu_grid, N, iterations);

        float max_error = 0.0f;
        double sum_squared_error = 0.0;

        for (int i = 0; i < N * N; i++) {
            float error = fabsf(h_grid[i] - cpu_grid[i]);

            if (error > max_error) {
                max_error = error;
            }

            sum_squared_error += (double)error * error;
        }

        double rmse = sqrt(sum_squared_error / ((double)N * N));

        printf("Maximum CPU/GPU error : %.8e\n", max_error);
        printf("CPU/GPU RMSE          : %.8e\n", rmse);

        free(cpu_grid);
    } else {
        printf("CPU/GPU validation    : skipped for long run\n");
        printf("Use the earlier 10000-iteration comparison "
               "as the baseline check.\n");
    }

    printf("====================================\n");

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));

    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_b));
    CHECK(cudaFree(d_max_residual));

    free(h_grid);
}

int main(void)
{
    int device = 0;
    cudaDeviceProp prop;

    CHECK(cudaGetDeviceProperties(&prop, device));
    CHECK(cudaSetDevice(device));

    printf("GPU: %s\n", prop.name);
    printf("Compute capability: %d.%d\n",
           prop.major, prop.minor);

    const int max_iters = 2000000;
    const float tolerance = 1.0e-5f;

    run_solver(512, max_iters, tolerance);
    run_solver(1024, max_iters, tolerance);

    CHECK(cudaDeviceReset());

    return 0;
}
