"""Host-only guards for the existing total solve time boundaries.

These source checks do not replace GPU timing or numerical validation.
"""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]


class TimingContractTests(unittest.TestCase):
    def test_async_candidate_verification_remains_timed(self):
        source = (ROOT / "async/async_solver.cu").read_text()
        start = source.index("CUDA_CHECK(cudaEventRecord(e0));")
        stop = source.index("CUDA_CHECK(cudaEventRecord(e1));")
        for branch in ("if (barrier)", "// Race-free delayed-halo mode"):
            position = source.index(branch, start)
            copy = source.index("candidate.data(), cur,", position)
            verify = source.index("true_max_update(candidate, prob)", copy)
            accept = source.index("if (checked_resid < cfg.tol)", verify)
            self.assertTrue(start < position < copy < verify < accept < stop)
        final_copy = source.index(
            "cudaMemcpy(res.final_grid.data(), cur, bytes, cudaMemcpyDeviceToHost)"
        )
        final_check = source.index("true_max_update(res.final_grid, prob)")
        self.assertTrue(stop < final_copy < final_check)

    def test_sync_loop_and_residual_copy_remain_timed(self):
        source = (ROOT / "sync_jacobi.cu").read_text()
        start = source.index("CUDA_CHECK(cudaEventRecord(e0));")
        loop = source.index("while (it < limit)", start)
        residual_copy = source.index(
            "cudaMemcpy(&bits, d_res, sizeof bits, cudaMemcpyDeviceToHost)", loop
        )
        accept = source.index("if (resid < cfg.tol)", residual_copy)
        stop = source.index("CUDA_CHECK(cudaEventRecord(e1));", accept)
        final_copy = source.index(
            "cudaMemcpy(res.final_grid.data(), cur, bytes, cudaMemcpyDeviceToHost)",
            stop,
        )
        self.assertTrue(start < loop < residual_copy < accept < stop < final_copy)
        self.assertNotIn("true_max_update", source[loop:stop])


if __name__ == "__main__":
    unittest.main()
