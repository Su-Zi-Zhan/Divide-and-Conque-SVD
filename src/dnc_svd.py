# .src/dnc_svd.py
"""
Module for performing Singular Value Decomposition (SVD) on matrices.
"""
import numpy as np

from numpy import float64
from numpy.typing import NDArray
from numpy.linalg import norm
from typing import Tuple
from .utils import givens
from .secular import solve_secular_equation

class DCSVDSolver:
    """
    A class to perform Singular Value Decomposition (SVD) on a given matrix.
    """
    def __init__(self, tol: float = 1e-12):
        """
        Initializes the DCSVDSolver with a specified tolerance and statistics tracking.

        Parameters:
        tol (float): Tolerance for small singular values.
        """
        self.tol = tol
        self.stats = {
            'deflation_count': 0,
            'max_depth': 0,
            'recursion_calls': 0
        }

    def solve(self, B: NDArray) -> Tuple[NDArray, NDArray, NDArray]:
        """
        Performs SVD on the input matrix B. This is user facing method.
        
        :param B: Bidiagonal matrix to decompose.
        :type B: NDArray
        :return: U, S, Vt matrices from the SVD.
        :rtype: Tuple[NDArray, NDArray, NDArray]
        """
        if isinstance(B, Tuple):
            d, e = B
            d = d.astype(float64)
            e = e.astype(float64)
            return self._recursive_step(d, e, depth=0)
        else:
            n = d.shape[0]
            d = np.diag(B).copy().astype(float64)
            e = np.diag(B, k=-1).copy().astype(float64)
            return self._recursive_step(d, e, depth=0)
        
    
    def _recursive_step(self, d: NDArray[float64], e: NDArray[float64], depth: int)\
        -> Tuple[NDArray, NDArray, NDArray]:
        """
        A private method to perform a recursive step in the SVD algorithm.
        
        :param d: Diagonal elements of the bidiagonal matrix.
        :type d: NDArray
        :param e: Superdiagonal elements of the bidiagonal matrix.
        :type e: NDArray
        :param depth: Current recursion depth.
        :type depth: int
        :return: U, S, Vt matrices from the SVD.
        :rtype: Tuple[NDArray, NDArray, NDArray]
        """
        self.stats['recursion_calls'] += 1
        self.stats['max_depth'] = max(self.stats['max_depth'], depth)
        
        n = d.shape[0]
        if n == 0:
            return np.eye(N=1, M=1, dtype=float64), np.zeros((0), dtype=float64), np.zeros((0, 0), dtype=float64)
        if n <= 2:
            B_small = np.zeros((n + 1, n), dtype=float64)
            np.fill_diagonal(B_small, d)
            np.fill_diagonal(B_small[1:], e)
            U, S, Vt = np.linalg.svd(B_small, full_matrices=True)
            Vt = Vt.T
            perm = np.argsort(S)
            S_sorted = S[perm]
            u_perm = np.concatenate((perm, [n]))
            U_sorted = U[:, u_perm]
            Vt_sorted = Vt[:, perm]
            return U_sorted, S_sorted, Vt_sorted
        # Divide the problem

        # Check for decoupling

        # Standard case: divide and conquer
        k, alpha, beta, subprob1, subprob2 = self._divide(d, e)
        U1, S1, Vt1 = self._recursive_step(subprob1[0], subprob1[1], depth + 1)
        U2, S2, Vt2 = self._recursive_step(subprob2[0], subprob2[1], depth + 1)
        U, S, Vt = self._merge(k, alpha, beta, U1, S1, Vt1, U2, S2, Vt2)
        
        return U, S, Vt

    def _divide(self, d: NDArray[float64], e: NDArray[float64]) -> Tuple:
        """
        A private method to divide the problem into smaller subproblems.
        
        :param self: The instance of the class.
        :param d: Diagonal elements of the bidiagonal matrix.
        :type d: NDArray[float64]
        :param e: Lowerdiagonal elements of the bidiagonal matrix.
        :type e: NDArray[float64]
        :return: A tuple containing the divided subproblems.
        :rtype: Tuple
        """
        n = d.shape[0]
        k = n // 2

        beta = e[k - 1]
        alpha = d[k - 1]

        # Create subproblem 1
        d1 = d[:k - 1].copy()
        e1 = e[:k - 1].copy()

        # Create subproblem 2
        d2 = d[k:].copy()
        e2 = e[k:].copy()

        return k, alpha, beta, (d1, e1), (d2, e2)
    
    def _merge(self, k: int, alpha: float64, beta: float64,
               U1: NDArray, S1: NDArray, Vt1: NDArray,
               U2: NDArray, S2: NDArray, Vt2: NDArray) -> Tuple[NDArray, NDArray, NDArray]:
        """
        A private method to merge the results of two subproblems.
        
        :param self: The instance of the class.
        :param k: Breakpoint index.
        :type k: int
        :param alpha: alpha value at the breakpoint.
        :type alpha: float64
        :param beta: beta value at the breakpoint.
        :type beta: float64
        :param U1, S1, Vt1: SVD results from the first subproblem.
        :type U1, S1, Vt1: NDArray
        :param U2, S2, Vt2: SVD results from the second subproblem.
        :type U2, S2, Vt2: NDArray
        :return: Merged U, S, Vt matrices.
        :rtype: Tuple[NDArray, NDArray, NDArray]
        """
        n1 = S1.shape[0]
        n2 = S2.shape[0]
        n = n1 + n2

        # Construct z vector
        z_hat = np.zeros(n + 2, dtype=float64)
        if n1 > 0:
            # z = alpha q1
            #     alpha Q1
            #     beta Q2
            #     beta q2
            z_hat[0] = U1[n1, -1] * alpha
            z_hat[1:n1 + 1] = U1[n1, :n1].copy() * alpha
            z_hat[n1 + 1:] = U2[0, :].copy() * beta

        # Normalize z and adjust rho
        rho = 1.0
        # norm_z = norm(z_hat)
        #if norm_z > 0:
            #z_hat /= norm_z
            #rho *= norm_z * norm_z
        
        # Givens rotation to zero out last element of z
        c0, s0, r0 = givens(z_hat[0], z_hat[-1])
        z_hat[0] = r0
        z_hat[-1] = 0.0
        # Construct U and Vt
        U = np.zeros((n + 2, n + 2), dtype=float64)
        Vt = np.zeros((n + 1, n + 1), dtype=float64)
        # Fill in U and Vt with appropriate values
        # U = q1 Q1 0  0
        #     0  0  Q2 q2
        U[:n1 + 1, 1:n1 + 1] = U1[:, :n1]
        U[:n1 + 1, 0] = U1[:, n1]
        U[n1 + 1:, n1 + 1:] = U2
        # Vt = 0  Vt1  0
        #      1  0    0
        #      0  0  Vt2
        Vt[:n1, 1:n1 + 1] = Vt1
        Vt[n1, 0] = 1
        Vt[n1 + 1:, n1 + 1:] = Vt2
        # Apply Givens rotation to U
        q1 = U[:, 0].copy()
        q2 = U[:, -1].copy()
        U[:, 0] = c0 * q1 + s0 * q2
        U[:, -1] = c0 * q2 - s0 * q1
        
        # Construct M for secular equation. M = diag(d) + rho * z * e1^T
        d = np.zeros(n + 1, dtype=float64)
        d[1:n1 + 1] = S1
        d[n1 + 1:] = S2
        z = z_hat[:n+1]

        # sorting
        perm = np.argsort(d)
        d_sorted = d[perm]
        z_sorted = z[perm]

        U_sorted_basis = U[:, :n+1][:, perm]
        Vt_sorted_basis = Vt[:, :n+1][:, perm]

        # Solve secular equation
        sol = solve_secular_equation(d_sorted, z_sorted, rho, eps=self.tol, compute_eigvec=True)
        
        S = sol.all_roots
        U[:, :n + 1] = U_sorted_basis @ sol.U_local
        Vt = Vt_sorted_basis @ sol.V_local

        return U, np.array(S, dtype=float64), Vt