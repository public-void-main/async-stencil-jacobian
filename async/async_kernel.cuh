// async_kernel.cuh -- block-asynchronous Jacobi kernel (Role B).
//
// Each CUDA block owns a TX x TY tile of the N x N grid and keeps it in shared
// memory. One "round" =
//     1. refresh the 1-cell halo from GLOBAL memory (other blocks' latest values,
//        possibly stale, possibly mid-update by another block -- the intended race)
//     2. do D local Jacobi sweeps in shared memory with the halo frozen
//     3. write the tile back to global memory
// A launch performs `sweeps` sweeps in total = ceil(sweeps/D) rounds; the last
// round may be shorter. There is NO grid-wide barrier inside the kernel.
//
// Two modes, selected by the host through the arguments:
//   free-running : in == out (same buffer), sweeps = M, D = 1,2,4,8
//   barrier/debug: in != out (ping-pong),  sweeps = 1, D = 1
//                  -> kernel boundary is a global barrier; equals sync Jacobi.
//
// Rules enforced here:
//   * Dirichlet boundary cells (global ring) are never changed and never written
//     back; their value is read once and kept in shared memory.
//   * src (= dt*f) is constant and is kept in a register.
//   * Same update as reference.py: u = ((old + r*(((N+S)+W)+E)) + src) / (1+4r),
//     where old is the cell's own current value (NOT a frozen time step).
//   * Residual (optional): max |new-old| over the tile in the LAST sweep of the
//     launch, reduced per block, then atomicMax over blocks (non-negative floats
//     compare correctly as unsigned ints).
//
// Requirements: N % TX == 0 and N % TY == 0 (checked on the host).
// Cross-block reads use __ldcg (bypass L1, read L2) and writes use __stcg so a
// block never sees a stale copy of another block's data from its own L1.

#pragma once

#ifndef TX
#define TX 32
#endif
#ifndef TY
#define TY 8
#endif
#define NTHREADS (TX * TY)
#define NWARPS   (NTHREADS / 32)

__global__ void async_jacobi_kernel(const float* in, float* out,
                                    const float* __restrict__ src,
                                    int N, float r, float denom,
                                    int D, int sweeps, int want_res,
                                    unsigned int* max_upd_bits)
{
    __shared__ float s[2][TY + 2][TX + 2];
    __shared__ float wmax[NWARPS];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int gx = blockIdx.x * TX + tx;
    const int gy = blockIdx.y * TY + ty;
    const size_t gi = (size_t)gy * N + gx;

    const bool fixed_cell = (gx == 0) || (gy == 0) || (gx == N - 1) || (gy == N - 1);
    const float src_c = fixed_cell ? 0.0f : src[gi];

    // Load this block's own tile once; later rounds only refresh the halo,
    // because only this block ever writes these cells.
    s[0][ty + 1][tx + 1] = __ldcg(in + gi);

    int   cur   = 0;
    float myres = 0.0f;

    for (int done = 0; done < sweeps; done += D) {
        const int nsw = min(D, sweeps - done);

        // ---- 1. halo refresh from global memory (both smem buffers) ----
        if (tx == 0) {
            float v = (gx > 0) ? __ldcg(in + gi - 1) : 0.0f;
            s[0][ty + 1][0] = v;  s[1][ty + 1][0] = v;
        }
        if (tx == TX - 1) {
            float v = (gx < N - 1) ? __ldcg(in + gi + 1) : 0.0f;
            s[0][ty + 1][TX + 1] = v;  s[1][ty + 1][TX + 1] = v;
        }
        if (ty == 0) {
            float v = (gy > 0) ? __ldcg(in + gi - N) : 0.0f;
            s[0][0][tx + 1] = v;  s[1][0][tx + 1] = v;
        }
        if (ty == TY - 1) {
            float v = (gy < N - 1) ? __ldcg(in + gi + N) : 0.0f;
            s[0][TY + 1][tx + 1] = v;  s[1][TY + 1][tx + 1] = v;
        }
        __syncthreads();

        // ---- 2. nsw local sweeps in shared memory, halo frozen ----
        for (int k = 0; k < nsw; ++k) {
            const float old = s[cur][ty + 1][tx + 1];
            float v = old;                              // fixed cells keep their value
            if (!fixed_cell) {
                const float sum = ((s[cur][ty][tx + 1] + s[cur][ty + 2][tx + 1])
                                   + s[cur][ty + 1][tx]) + s[cur][ty + 1][tx + 2];
                v = ((old + r * sum) + src_c) / denom;
                if (done + k == sweeps - 1) myres = fabsf(v - old);
            }
            s[cur ^ 1][ty + 1][tx + 1] = v;
            __syncthreads();                            // one barrier per sweep (smem ping-pong)
            cur ^= 1;
        }

        // ---- 3. write the tile back (interior cells only, never Dirichlet) ----
        __stcg(out + gi, s[cur][ty + 1][tx + 1]);
    }

    // ---- optional residual: block max, then one atomicMax per block ----
    if (want_res) {                                     // uniform across the block
        const int tid = ty * TX + tx;
        const int lane = tid & 31, wid = tid >> 5;
        float v = myres;
        for (int o = 16; o > 0; o >>= 1)
            v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
        if (lane == 0) wmax[wid] = v;
        __syncthreads();
        if (wid == 0) {
            v = (lane < NWARPS) ? wmax[lane] : 0.0f;
            for (int o = 16; o > 0; o >>= 1)
                v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
            if (lane == 0) atomicMax(max_upd_bits, __float_as_uint(v));
        }
    }
}
