# Async Stencil Solvers on GPU with ML-Guided Synchronization Selection

Manipal Institute of Technology, Manipal Academy of Higher Education
Parallel Computing mini project (PCAP)

| Name | Roll No. | Reg. No. | Role |
|---|---|---|---|
| Bhavana Peri | 6 | 240905034 | A: Sync baseline and GPU profiling |
| Medhavi Mehta | 21 | 240905122 | B: Block-async solver and staleness study |
| Vaibhav Bhatia | 52 | 240905504 | C: Reference, harness, dataset and ML |

**Deadline: 12 Oct 2026** | **GPU: NVIDIA GeForce RTX 5060 (Blackwell, sm_120, 8 GB)**

---

## 1. Project in brief

We solve 2D diffusion on a structured grid with a five-point **Jacobi** stencil on the GPU (one CUDA thread per grid cell).

- **Synchronous Jacobi:** all blocks finish an iteration before the next begins (kernel-boundary sync). Safe, but pays launch and synchronization overhead.
- **Block-asynchronous Jacobi:** sync is kept inside a thread block, but blocks do not wait for each other every iteration. Neighbour-block values may be stale.
- **Trade-off:** less synchronization overhead vs. extra iterations caused by stale data.
- **ML extension:** an offline decision tree predicts the best strategy {Sync, D=1, 2, 4, 8} from pre-runtime features.

**Research question:** can relaxing global synchronization reduce total time-to-convergence despite needing more iterations, and can a lightweight ML model predict the best strategy beforehand?

**Contribution:** empirical and architecture-aware study. We do not claim a new solver or new convergence theory.

## 2. Agreed decisions (fixed on Day 1)

1. **Equation (implicit transient step, so convergence takes tens to hundreds of iterations):**
   `u_new = (u_old + r * (sum of 4 neighbours) + dt * f) / (1 + 4r)`, where `r = alpha*dt/h^2`
2. **D** = number of local sweeps a block performs in shared memory between halo refreshes from global memory.
3. **Tolerance:** residual < 1e-4 (raised from 1e-5: 1 ulp of float32 at 100 is 7.6e-6, so 1e-5 sits at the rounding-noise floor and the stop point was luck-dependent). **Residual interval:** M = 10.
4. **Results CSV:** see `results/SCHEMA.md`.
5. **Compile for RTX 5060:** CUDA 12.8+, `-arch=sm_120`.

## 3. Scope (2.5-day version)

**Must have**
1. Validated synchronous CUDA solver
2. Block-asynchronous solver, D in {1, 2, 4, 8}
3. Comparison of iterations, time, convergence curves, speedup S and iteration ratio R_I
4. Basic profiling (occupancy, bandwidth)

**Should have**
5. Small decision-tree strategy selector vs. always-sync

**Dropped unless time remains:** online adaptive controller, N = 2048/4096, Neumann boundaries, full block-size sweep.

**Experiment grid:** N in {512, 1024}; Dirichlet only; sources point and Gaussian; r in {1, 10, 100}; strategies Sync, D1, D2, D4, D8; 3 repeats each.

## 4. Metrics

- Speedup: `S = T_sync / T_async`
- Iteration overhead: `R_I = I_async / I_sync`
- Time model: `T_total = I * T_iter + T_overhead`
- Also: residual decay, effective bandwidth (bytes moved / time), occupancy

## 5. Repository layout

```
baseline/     synchronous CUDA solver (Bhavana)
async/        block-asynchronous CUDA solver (Medhavi)
harness_ml/   CPU reference, generator, driver, plots, ML (Vaibhav)
results/      CSVs and figures (SCHEMA.md defines the columns)
report/       final report and figures
```

Common solver interface (all solvers follow it):
`solve(config) -> {iterations, time_ms, residual_history, final_grid}`

## 6. Setup

```bash
# check toolchain (needs CUDA 12.8+ for the RTX 5060)
nvcc --version
nvidia-smi

# compile a CUDA file for the RTX 5060
nvcc -O3 -arch=sm_120 baseline/sync_jacobi.cu -o build/sync_jacobi

# python side
python -m venv .venv && source .venv/bin/activate
pip install numpy pandas matplotlib scikit-learn
```

If Nsight Compute fails on the 5060, use `cudaOccupancyMaxActiveBlocksPerMultiprocessor` for occupancy and compute bandwidth as bytes moved / time, and note this in the report.

---

## 7. Task plan

### Day 1: Saturday 10 Oct (Build)

**Everyone (first hour)**
- [ ] Agree on the equation, D definition, tolerance, M and CSV columns (Section 2)
- [ ] Create the shared repo and folders
- [ ] Decide which machine with the RTX 5060 runs all experiments

**Bhavana (A): Sync solver**
- [ ] Check CUDA 12.8+ and compile a trivial kernel with `-arch=sm_120`
- [ ] Write the sync kernel (one thread per cell, ping-pong buffers, fixed boundaries)
- [ ] Add the residual check every 10 iterations (GPU reduction)
- [ ] Time one run at N = 512 and 1024 with CUDA events
- [ ] Validate against Vaibhav's CPU reference once ready
- [ ] Freeze the sync solver and push by tonight

**Medhavi (B): Async kernel**
- [ ] Skim Frommer and Szyld [3] and the CUDA memory-consistency section (max 1 hour)
- [ ] Write a half-page design note (buffer layout, halo refresh, how blocks read neighbours)
- [ ] Write the async kernel: load tile + halo into shared memory, D local sweeps, write back, no grid barrier
- [ ] Add the debug mode (D = 1 with a barrier) for comparison with sync
- [ ] Get it compiling and running at N = 512 by tonight
- [ ] Push to the repo

**Vaibhav (C): Reference and harness**
- [ ] NumPy CPU reference solver (finish first, A depends on it)
- [ ] Problem generator (point and Gaussian sources, r in {1, 10, 100}, fixed seeds)
- [ ] Experiment driver: config x strategy x 3 repeats, append to CSV
- [ ] Use the common solver interface (stubs until A and B push)
- [ ] First plotting script (time vs. strategy, residual curve)
- [ ] Push to the repo

### Day 2: Sunday 11 Oct (Experiments and draft)

**Bhavana (A): Profiling and write-up**
- [ ] Fix sync issues found by C's tests and confirm the frozen version
- [ ] Pick the block size (try 16x16 and 32x8, keep the faster)
- [ ] Compute occupancy and effective bandwidth for sync
- [ ] Try Nsight Compute on the sync kernel; use the fallback if it fails
- [ ] Afternoon: repeat occupancy and bandwidth for the async solver
- [ ] Put the numbers in a table
- [ ] Write sections: GPU architecture, baseline implementation, profiling

**Medhavi (B): Async verification and results**
- [ ] Check that D = 1 with barrier matches sync; fix bugs first
- [ ] Check that async converges to the same solution (compare final grids)
- [ ] If not converging by midday, reduce to D in {1, 2, 4}
- [ ] Hand the working async solver to C for the full run
- [ ] Compute S and R_I from the data
- [ ] Write down where async wins and loses, and why
- [ ] Write sections: async design and results

**Vaibhav (C): Full run, plots, ML**
- [ ] Plug A's and B's solvers into the driver; run one config end to end
- [ ] Run the full grid (about 1-2 hours; do not use the GPU for anything else)
- [ ] Check the CSV for missing or failed runs and rerun them
- [ ] Plots: residual curves, time vs. D, S and R_I table
- [ ] Label each config with the fastest strategy (median time); flag near-ties
- [ ] Train a decision tree with leave-one-out CV
- [ ] Compare with always-sync and best fixed D (total time and regret)
- [ ] Write sections: setup and ML (state the dataset is small)

**Everyone (evening)**
- [ ] Send finished sections to one person for merging
- [ ] Fill in the reference gaps ([5], [8]-[10])
- [ ] Note open problems for tomorrow

### Day 3: Monday 12 Oct (Finish and submit)

**Bhavana (A)**
- [ ] Check that every profiling number matches the saved outputs
- [ ] State GPU (RTX 5060), CUDA version, driver and compile flags
- [ ] Finalize this README's run instructions
- [ ] Package the code folder
- [ ] Proofread Vaibhav's sections

**Medhavi (B)**
- [ ] Check that plots have labelled axes, legends and captions
- [ ] Add a limitations paragraph (races, cases where async lost or failed, single GPU)
- [ ] Confirm D, M, S and R_I are used identically across the report
- [ ] Proofread Bhavana's sections

**Vaibhav (C)**
- [ ] Check that the final CSV matches every table and figure
- [ ] State random seeds and the one-command script to reproduce results
- [ ] Add a caution that the ML result is a proof of concept (about 12 configs)
- [ ] Merge everything into one document and do a full consistency read
- [ ] Proofread Medhavi's sections

**Everyone (final hours)**
- [ ] Every figure and table captioned and referenced in the text
- [ ] Title page: names, roll numbers, registration numbers
- [ ] Export the PDF and check that it renders
- [ ] **Submit early, not in the last hour**

### Handoffs

| When | From -> To | What |
|---|---|---|
| End of Day 1 | C -> A, B | CPU reference, generator, driver |
| End of Day 1 | A -> B, C | Frozen sync solver |
| Day 2 morning | B -> C | Working async solver |
| Day 2 midday | C -> all | Full dataset and plots |

### If we fall behind (cut in this order)
1. ML selector shrinks to one decision tree, no cross-validation
2. Profiling shrinks to occupancy plus computed bandwidth
3. Drop D = 8 or one source type
4. **Never drop** the sync vs. async comparison

---

## 8. Reading order

1. **Before coding:** [8] CUDA Programming Guide, [9] Best Practices, [15] Shan and Araya-Polo (GPU stencils), [10] Nsight Compute docs
2. **Before the async solver:** [3] Frommer and Szyld, [4] Saad (basic iterative methods chapter), [1] Chazan and Miranker, [2] Baudet
3. **Before the ML stage:** [11] Dufrechou et al., [14] Stylianou and Weiland, [12] Dinkelbach et al., [13] ICTC 2022
4. **Optional:** [5] Demmel et al. (communication avoidance), [6], [7] SuiteSparse

Note: the ML references [11]-[14] concern sparse-matrix formats and triangular solvers. Borrow the methodology, not the claims.

## 9. References

1. D. Chazan and W. Miranker, "Chaotic relaxation," *Linear Algebra Appl.*, vol. 2, no. 2, pp. 199-222, 1969.
2. G. M. Baudet, "Asynchronous iterative methods for multiprocessors," *J. ACM*, vol. 25, no. 2, pp. 226-244, 1978.
3. A. Frommer and D. B. Szyld, "On asynchronous iterations," *J. Comput. Appl. Math.*, vol. 123, no. 1-2, pp. 201-216, 2000.
4. Y. Saad, *Iterative Methods for Sparse Linear Systems*, 2nd ed., SIAM, 2003.
5. J. Demmel et al., "Communication-avoiding algorithms for linear algebra and beyond," *Acta Numerica*, vol. 26, pp. 1-108. (add year and details)
6. T. A. Davis and Y. Hu, "The University of Florida Sparse Matrix Collection," *ACM TOMS*, vol. 38, no. 1, pp. 1-25, 2011.
7. T. A. Davis et al., "The SuiteSparse Matrix Collection Website Interface," *JOSS*, vol. 4, no. 35, p. 1244, 2019.
8. NVIDIA, *CUDA C++ Programming Guide*. (add version and URL)
9. NVIDIA, *CUDA C++ Best Practices Guide*. (add version and URL)
10. NVIDIA, *Nsight Compute Documentation*. (add version and URL)
11. E. Dufrechou, P. Ezzatti, M. Freire, E. S. Quintana-Orti, "Machine learning for optimal selection of sparse triangular system solvers on GPUs," *J. Parallel Distrib. Comput.*, vol. 158, pp. 47-55, 2021, doi: 10.1016/j.jpdc.2021.07.013.
12. M. Dinkelbach et al., "Auto-Selection of an Optimal Sparse Matrix Format in the Neuro-Simulator ANNarchy," *Front. Neuroinform.*, vol. 16, 2022, doi: 10.3389/fninf.2022.877945.
13. "Analysis of Several Sparse Formats for Matrices used in Sparse-Matrix Dense-Matrix Multiplication for Machine Learning on GPUs," *Proc. ICTC*, 2022, doi: 10.1109/ICTC55196.2022.9952814.
14. C. Stylianou and M. Weiland, "Optimizing Sparse Linear Algebra Through Automatic Format Selection and Machine Learning," arXiv:2303.05098, 2023.
15. B. Shan and M. Araya-Polo, "Evaluation of Programming Models and Performance for Stencil Computation on Current GPU Architectures," arXiv:2404.04441, 2024.

## 10. Role-C implementation handoff

Vaibhav's assigned CPU reference, problem generator, experiment driver, result
auditing, plots and offline decision-tree selector are implemented in
`harness_ml/`. The shared numerical and experiment definitions are fixed in
`harness_ml/experiment_spec.json`; CUDA owners should use
`harness_ml/INTEGRATION_CONTRACT.md` for the minimal executable handoff.

No synthetic timings are included as project results. `results/runs.csv`, final
figures and ML conclusions must be generated only from the integrated sync and
async CUDA executables.
