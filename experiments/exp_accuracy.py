# experiments/exp_accuracy.py
"""
Experiment to evaluate the accuracy of the DCSVDSolver implementation.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.dnc_svd import DCSVDSolver
from numpy.typing import NDArray
from numpy import float64
from typing import Tuple

def generate_test_matrix(n: int = 20) -> Tuple[NDArray[float64], NDArray[float64]]:
    """
    Generates a random bidiagonal matrix for testing.
    
    :param n: Size of the matrix.
    :type n: int
    :return: Diagonal and lower diagonal of the bidiagonal matrix.
    :rtype: Tuple[NDArray[float64], NDArray[float64]]
    """
    np.random.seed(42)
    d = np.sort(np.random.rand(n))[::-1] * 10.0
    d[0] = 0.0  # Ensure first singular value is zero
    e = np.random.rand(n) * 2.0
    return d.astype(float64), e.astype(float64)

def visualize_accuracy():
    n = 30
    d, e = generate_test_matrix(n)
    # e = np.zeros(n, dtype=float64)  # Make it diagonal for simplicity
    B = (d, e)

    print("Running DCSVDSolver...")
    solver = DCSVDSolver(tol=1e-15)
    try:
        U, S, Vt = solver.solve(B)
    except Exception as ex:
        print(f"Solver call failed: {ex}")
        return
    
    # Compute ground truth using numpy SVD
    bidiag_matrix = np.zeros((n + 1, n), dtype=float64)
    np.fill_diagonal(bidiag_matrix, d)
    np.fill_diagonal(bidiag_matrix[1:], e)
    U_gt, S_gt, Vt_gt = np.linalg.svd(bidiag_matrix)

    # Create canvas for plotting
    fig = plt.figure(figsize=(15, 10))
    plt.suptitle(f"D&C SVD Solver Accuracy Report (N={n})", fontsize=16, y=0.95)

    # Plot 1: Reconstruction Error
    ax1 = plt.subplot(2, 2, 1)
    # Reconstruct: B_recon = U * diag(S) * Vt
    diag = np.zeros((n + 1, n), dtype=float64)
    np.fill_diagonal(diag, S)
    B_recon = U @ diag @ Vt
    B = np.zeros((n + 1, n), dtype=float64)
    np.fill_diagonal(B, d)
    np.fill_diagonal(B[1:], e)
    Error_Matrix = np.abs(B - B_recon)

    from matplotlib.colors import LogNorm
    sns.heatmap(Error_Matrix, ax=ax1, cmap="magma", norm=LogNorm(vmin=1e-16, vmax=1e-12))
    ax1.set_title("Reconstruction Error $|B - U \Sigma V^T|$")

    # Plot 2: Orthogonality check.
    ax2 = plt.subplot(2, 2, 2)
    # Check U^T * U - I
    Ortho_Err = np.abs(U.T @ U - np.eye(n + 1))
    sns.heatmap(Ortho_Err, ax=ax2, cmap="viridis", norm=LogNorm(vmin=1e-16, vmax=1e-12))
    ax2.set_title("Orthogonality Check $|U^T U - I|$")

    # Plot 3: Singular Values Comparison
    S_sorted = np.sort(S)[::-1]
    S_gt_sorted = np.sort(S_gt)[::-1]
    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(S_gt_sorted, 'k-', linewidth=3, alpha=0.5, label='NumPy (LAPACK)')
    ax3.plot(S_sorted, 'r.', markersize=8, label='My Solver')
    ax3.set_yscale('log')
    ax3.set_title("Singular Values Distribution (Log Scale)")
    ax3.set_ylabel("Singular Value $\sigma_i$")
    ax3.set_xlabel("Index")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # S_sorted = np.sort(S)[::-1]
    
    rel_error = np.abs(S_sorted - S_gt_sorted) / (S_gt_sorted + 1e-15)
    print(rel_error)
    max_rel_err = np.max(rel_error)

    ax3.text(0.5, 0.5, f"Max Rel Error: {max_rel_err:.2e}", 
             transform=ax3.transAxes, fontsize=12, 
             bbox=dict(facecolor='white', alpha=0.8))
    
    # Plot 4: Diagonalization check
    ax4 = plt.subplot(2, 2, 4)
    Diag_Check = np.abs(U.T @ B @ Vt.T)
    sns.heatmap(Diag_Check, ax=ax4, cmap="icefire", norm=LogNorm(vmin=1e-16, vmax=np.max(S)))
    ax4.set_title("Diagonalization Check $|U^T B V|$")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('svd_accuracy.pdf', dpi=300)
    print("Report saved to 'svd_accuracy.pdf'")
    plt.show()

if __name__ == "__main__":
    visualize_accuracy()