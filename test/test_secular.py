import numpy as np
import sys
import os
import unittest

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from numpy import float64
from numpy.typing import NDArray
from src.secular import solve_secular_equation, SecularSolution
from numpy.linalg import norm

class TestSecularSolver(unittest.TestCase):
    def setUp(self):
        self.rho = 1.0
        self.eps = np.finfo(float).eps
        self.tol_ortho = 1e-13
        self.tol_recon = 1e-10
    
    def check_orthogonality(self, matrix: NDArray[float64], name: str = "U"):
        if matrix is None:
            return
        n = matrix.shape[1]
        gram = matrix.T @ matrix
        identity = np.eye(n)
        diff = norm(gram - identity, ord='fro')
        print(f"    > Checking orthogonality of {name}: ||{name}^T * {name} - I||_F = {diff:e}")
        self.assertTrue(diff < self.tol_ortho, f"{name} is not orthogonal within tolerance! Error: {diff}")
    
    def test_standard_case(self):
        print("\n=== Test 1: Standard Case (No Deflation) ===")
        d = np.array([0.0, 2.0, 3.0])
        z = np.array([0.5, 0.5, 0.5])
        
        sol = solve_secular_equation(d, z, self.rho)
        
        print(f"Roots: {sol.all_roots}")

        self.assertEqual(len(sol.all_roots), 3) # Expecting 3 roots
        self.assertFalse(np.any(sol.deflated_mask), "No deflation should occur")

        # Check roots are interlaced with d
        self.assertTrue(d[0] < sol.all_roots[0] < d[1])
        self.assertTrue(d[1] < sol.all_roots[1] < d[2])
        self.assertTrue(d[2] < sol.all_roots[2])

        # Check orthogonality of U and V
        self.check_orthogonality(sol.U_local, "U")
        self.check_orthogonality(sol.V_local, "V")

        # Check reconstruction accuracy
        Sigma = np.diag(sol.all_roots)
        e = np.array([1.0, 0.0, 0.0])  # Placeholder for the first standard basis vector
        Target = np.diag(d) + self.rho * np.outer(z, e)
        Reconstructed = sol.U_local @ Sigma @ sol.V_local.T

        err = norm(Target - Reconstructed, ord='fro') / norm(Target, ord='fro')
        print(f"  > Reconstruction Error (MM^T): {err:.2e}")
        self.assertTrue(err < self.tol_recon, f"Reconstruction error too high: {err}")

    def test_deflation_small(self):
        print("\n=== Test 2: Deflation Case (Small z Component) ===")
        d = np.array([0.0, 2.0, 3.0])
        z = np.array([1.0, 1e-15, 1.0])

        sol = solve_secular_equation(d, z, self.rho)
        
        print(f"Mask: {sol.deflated_mask}")
        print(f"Roots: {sol.all_roots}")

        # Check deflation mask.
        self.assertTrue(sol.deflated_mask[1], "Index 1 (d=2.0) 应该被收缩")
        self.assertFalse(sol.deflated_mask[0])
        self.assertFalse(sol.deflated_mask[2])

        # Check roots
        # The root corresponding to d=2.0 should be exactly 2.0.
        self.assertAlmostEqual(sol.all_roots[1], 2.0, places=10)

        # Check orthogonality of U and V
        self.check_orthogonality(sol.U_local, "U")
        self.check_orthogonality(sol.V_local, "V")

        # Check reconstruction accuracy
        Sigma = np.diag(sol.all_roots)
        e = np.array([1.0, 0.0, 0.0])  # Placeholder for the first standard basis vector
        Target = np.diag(d) + self.rho * np.outer(z, e)
        Reconstructed = sol.U_local @ Sigma @ sol.V_local.T

        err = norm(Target - Reconstructed, ord='fro') / norm(Target, ord='fro')
        print(f"  > Reconstruction Error (MM^T): {err:.2e}")
        self.assertTrue(err < self.tol_recon, f"Reconstruction error too high: {err}")

    def test_deflation_close(self):
        print("\n=== Test 3: Deflation by Close D ===")
        d = np.array([0.0, 2.0, 2.0 + 1e-14, 4.0])
        z = np.array([1.0, 1.0, 1.0, 1.0])
        sol = solve_secular_equation(d, z, self.rho)

        print(f"Mask: {sol.deflated_mask}")
        print(f"Roots: {sol.all_roots}")
        print(f"Givens Rotations: {len(sol.givens_rotations)}")

        # Check Givens rotations
        self.assertTrue(len(sol.givens_rotations) >= 1, "Expected at least one Givens rotation for close d components")

        # Check deflation mask
        is_deflated_cluster = sol.deflated_mask[1] or sol.deflated_mask[2]
        self.assertTrue(is_deflated_cluster, "At least one of the close d components should be deflated")

        # Check orthogonality of U and V
        self.check_orthogonality(sol.U_local, "U")
        self.check_orthogonality(sol.V_local, "V")

        # Check reconstruction accuracy
        Sigma = np.diag(sol.all_roots)
        e = np.array([1.0, 0.0, 0.0, 0.0])  # Placeholder for the first standard basis vector
        Target = np.diag(d) + self.rho * np.outer(z, e)
        Reconstructed = sol.U_local @ Sigma @ sol.V_local.T

        err = norm(Target - Reconstructed, ord='fro') / norm(Target, ord='fro')
        print(f"  > Reconstruction Error (MM^T): {err:.2e}")
        self.assertTrue(err < self.tol_recon, f"Reconstruction error too high: {err}")

    def test_permutation_consistency(self):
        print("\n=== Test 4: Permutation Consistency ===")
        d = np.array([3.0, 1.0, 2.0])
        z = np.array([0.3, 0.1, 0.2])

        sol = solve_secular_equation(d, z, self.rho)

        # Check that roots are in the right order after permutation
        d_sorted = d[sol.perm_indices]
        self.assertTrue(np.all(d_sorted[:-1] <= d_sorted[1:]), "d should be sorted according to perm_indices")

        print(sol.U_local)
        
        # Check if z is permuted correctly
        z_sorted = z[sol.perm_indices]
        self.assertEqual(z_sorted[0], 0.1) # expected 1.0
        self.assertEqual(z_sorted[1], 0.2) # expected 2.0
        self.assertEqual(z_sorted[2], 0.3) # expected 3.0

    def test_accuracy_check(self):
        print("\n=== Test 5: Numerical Accuracy ===")
        n = 20
        d = np.sort(np.random.rand(n))[::-1] * 10.0
        d[0] = 0.0  # Ensure first singular value is zero
        z = np.random.rand(n) * 2.0
        rho = 1.0

        sol = solve_secular_equation(d, z, rho)
        roots = sol.all_roots

        # Verify secular equation at each root
        for i, is_def in enumerate(sol.deflated_mask):
            if not is_def:
                root = roots[i]
                value = 1.0
                for j in range(len(d)):
                    value += (rho * (sol.z_secular[j]**2)) / ((sol.d_secular[j]**2) - (root**2))
                print(f"Root: {root}, Residual: {value}")
                self.assertAlmostEqual(value, 0.0, places=10, msg=f"Secular equation not satisfied at root {root}")
                

if __name__ == '__main__':
    unittest.main()