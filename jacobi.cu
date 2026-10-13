
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#ifndef BLOCK_X
#define BLOCK_X 16
#endif

#ifndef BLOCK_Y
#define BLOCK_Y 16
#endif

#define VALIDATION_LIMIT 10000

#define CHECK(call) do {                                      \
    cudaError_t err = (call);                                 \
    if (err != cudaSuccess) {                                 \
        fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,    \
                cudaGetErrorString(err));                     \
        exit(EXIT_FAILURE);                                   \
    }                                                         \
} while (0)

/*
 * Two-dimensional CUDA Jacobi kernel.
 * Each thread handles one grid cell.
 */
__global__ void jacobi_step(
    const float *old_grid,
    float *new_grid,
    unsigned int *max_residual_bits,
    int N,
    int check_residual)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= N || j >= N)
        return;

    int idx = i * N + j;

    // Fixed boundaries: never update boundary cells.
    if (i == 0 || i == N - 1 ||
        j == 0 || j == N - 1)
        return;

    // Five-point Jacobi stencil.
    float value = 0.25f * (
        old_grid[(i - 1) * N + j] +
        old_grid[(i + 1) * N + j] +
        old_grid[i * N + j - 1] +
        old_grid[i * N + j + 1]
    );

    new_grid[idx] = value;

    if (check_residual) {
        float diff = fabsf(value - old_grid[idx]);

        // Atomic maximum of nonnegative float bit patterns.
        unsigned int bits = __float_as_uint(diff);
        atomicMax(max_residual_bits, bits);
    }
}

/* Top boundary = 100; all other cells = 0. */
void initialize_grid(float *grid, int N)
{
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            grid[i * N + j] =
                (i == 0) ? 100.0f : 0.0f;
        }
    }
}

/* CPU reference using the same Jacobi update. */
void cpu_jacobi(float *grid, int N, int iterations)
{
    size_t bytes = (size_t)N * N * sizeof(float);

    float *a = (float *)malloc(bytes);
    float *b = (float *)malloc(bytes);

    if (a == NULL || b == NULL) {
        fprintf(stderr, "CPU allocation failed\n");
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

    for (int i = 0; i < N * N; i++)
        grid[i] = old_grid[i];

    free(a);
    free(b);
}

void run_solver(int N, int max_iters, float tolerance)
{
    size_t bytes = (size_t)N * N * sizeof(float);

    float *h_grid = (float *)malloc(bytes);

    if (h_grid == NULL) {
        fprintf(stderr, "Host allocation failed\n");
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

    dim3 block(BLOCK_X, BLOCK_Y);

    dim3 grid(
        (N + BLOCK_X - 1) / BLOCK_X,
        (N + BLOCK_Y - 1) / BLOCK_Y
    );

    float *old_grid = d_a;
    float *new_grid = d_b;

    float residual = INFINITY;
    int iterations = 0;
    int converged = 0;

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    CHECK(cudaEventRecord(start));

    for (int iter = 1; iter <= max_iters; iter++) {
        int check = (iter % 10 == 0);

        if (check) {
            CHECK(cudaMemset(d_max_residual, 0,
                             sizeof(unsigned int)));
        }

        jacobi_step<<<grid, block>>>(
            old_grid,
            new_grid,
            d_max_residual,
            N,
            check
        );

        CHECK(cudaGetLastError());

        // Swap the ping-pong buffers.
        float *temp = old_grid;
        old_grid = new_grid;
        new_grid = temp;

        iterations = iter;

        if (check) {
            unsigned int bits;

            CHECK(cudaMemcpy(&bits, d_max_residual,
                             sizeof(bits),
                             cudaMemcpyDeviceToHost));

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
    printf("GPU                   : ");
    {
        cudaDeviceProp prop;
        int device;
        CHECK(cudaGetDevice(&device));
        CHECK(cudaGetDeviceProperties(&prop, device));
        printf("%s\n", prop.name);
    }

    printf("Grid size             : %d x %d\n", N, N);
    printf("Block shape           : %d x %d\n",
           BLOCK_X, BLOCK_Y);
    printf("Threads per block     : %d\n",
           BLOCK_X * BLOCK_Y);
    printf("Iterations            : %d\n", iterations);
    printf("Final residual        : %.8e\n", residual);
    printf("Tolerance             : %.8e\n", tolerance);
    printf("Convergence status    : %s\n",
           converged ? "TOLERANCE REACHED"
                     : "MAX ITERATIONS REACHED");
    printf("GPU time              : %.3f ms\n", milliseconds);
    printf("Center value          : %.6f\n",
           h_grid[(N / 2) * N + (N / 2)]);

    // CPU validation for runs of 10,000 iterations or fewer.
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

            if (error > max_error)
                max_error = error;

            sum_squared_error += (double)error * error;
        }

        printf("Maximum CPU/GPU error : %.8e\n", max_error);
        printf("CPU/GPU RMSE          : %.8e\n",
               sqrt(sum_squared_error / ((double)N * N)));

        free(cpu_grid);
    } else {
        printf("CPU/GPU validation    : skipped (long run)\n");
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
