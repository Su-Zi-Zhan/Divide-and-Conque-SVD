# .src/secular.py
"""
Compute the solutions to the secular equations.
"""
import numpy as np

from numpy.typing import NDArray
from numpy.linalg import norm
from numpy import float64, abs
from typing import Tuple, List, NamedTuple, Optional
from .utils import givens

class SecularSolution(NamedTuple):
    all_roots: NDArray[float64]     # Singular values (roots of secular equation)
    perm_indices: NDArray[np.intp]  # Permutation indices to map back to original order
    deflated_mask: NDArray[np.bool] # Mask indicating which roots were deflated
    givens_rotations: List[Tuple]   # List of Givens rotations applied during deflation

    d_secular: NDArray[float64]     # Irreducible diagonal elements for secular equation
    z_secular: NDArray[float64]     # Irreducible first column elements for secular equation
    roots_secular: List[float]      # Computed roots of the secular equation
    roots_deflated: List[float]     # Deflated roots

    k_secular: int                  # Number of secular roots

    U_local: Optional[NDArray[float64]] = None # Left singular vectors (optional)
    V_local: Optional[NDArray[float64]] = None # Right singular vectors (optional)

def secular_function_left(offset: float, i: int, d: NDArray[float64], z: NDArray[float64], rho: float) -> Tuple[float, float, float]:
    """
    Computes the left-hand side of the modified secular equation.
    
    :param offset: Offset value mu = omega - d[i]
    :type offset: float
    :param i: Index i in the secular equation
    :type i: int
    :param d: Array of diagonal elements d
    :type d: NDArray[float64]
    :param z: Array of first column elements z
    :type z: NDArray[float64]
    :param rho: Scalar value rho
    :type rho: float
    :return: result of the secular function evaluation, absolute value of the summation of positive and negative terms, and derivative of the secular function
    :rtype: Tuple[float, float, float]
    """
    n = d.shape[0]
    result = 1.0
    abs_sum = 1.0
    der_sum = 0.0

    eps = np.finfo(float).eps * 100 # To avoid division by zero
    if offset < eps:
        offset = eps

    for j in range(n):
        delta = d[j] - d[i]

        factor1, factor2 = (delta - offset), (d[j] + d[i] + offset)
        
        denominator = factor1 * factor2
        if abs(denominator) < eps:
            denominator = eps if denominator >= 0 else -eps

        term = rho * (z[j] ** 2) / denominator
        der_sum += abs(term * (factor2 - factor1) / denominator)
        result += term

        abs_sum += abs(term)

    return result, abs_sum, der_sum

def secular_function_right(offset: float, i: int, d: NDArray[float64], z: NDArray[float64], rho: float) -> Tuple[float, float, float]:
    """
    Computes the right-hand side of the modified secular equation.
    
    :param offset: Offset value mu = d[i+1] - omega
    :type offset: float
    :param i: Index i in the secular equation
    :type i: int
    :param d: Array of diagonal elements d
    :type d: NDArray[float64]
    :param z: Array of first column elements z
    :type z: NDArray[float64]
    :param rho: Scalar value rho
    :type rho: float
    :return: result of the secular function evaluation, absolute value of the summation of positive and negative terms and derivative of the secular function
    :rtype: Tuple[float, float, float]
    """
    n = d.shape[0]

    result = 1.0
    abs_sum = 1.0
    der_sum = 0.0
    ub = d[i + 1] if i + 1 < n else d[i] + rho * np.linalg.norm(z)

    eps = np.finfo(float).eps * 100 # To avoid division by zero
    if offset < eps:
        offset = eps

    for j in range(n):
        delta = d[j] - ub
        factor1, factor2 = (delta + offset), (d[j] + ub - offset)

        denominator = factor1 * factor2
        if abs(denominator) < eps:
            denominator = eps if denominator >= 0 else -eps
        term = rho * (z[j] ** 2) / denominator
        result += term
        der_sum += abs(term * (factor1 - factor2) / denominator)

        abs_sum += abs(term)
    
    return result, abs_sum, der_sum

def stopping_criterion(n: int, residual: float, abs_sum: float, tol: float = np.finfo(float).eps) -> bool:
    """
    Determines if the stopping criterion for the secular equation solver is met.
    
    :param n: Size of the problem
    :type n: int
    :param residual: Current residual value
    :type residual: float
    :param abs_sum: Absolute sum of terms in the secular function
    :type abs_sum: float
    :param tol: Tolerance level for convergence
    :type tol: float
    :return: True if the stopping criterion is met, False otherwise
    :rtype: bool
    """
    return abs(residual) <= n * tol * abs_sum

def rational_interpolation(offset: float, f: float, df: float) -> float:
    """
    Performs rational interpolation for the secular equation.
    
    :param offset: Offset value.
    :type offset: float
    :param f: Secular function value at the current offset.
    :type f: float
    :param df: Derivative of the secular function at the current offset.
    :type df: float
    :return: Next step.
    :rtype: float
    """
    denominator = df
    if df < np.finfo(float).eps:
        denominator = np.finfo(float).eps * 100
    if offset < np.finfo(float).eps: # Avoid division by zero
        return - f / denominator
    denominator += abs(f / offset)
    return - f / denominator

def secular_single_root(i: int, d: NDArray[float64], z: NDArray[float64], rho: float, max_iter: int = 1000, tol: float = np.finfo(float).eps) -> float:
    """
    Computes a single root of the secular equation using rational interpolation and bisection.
    
    :param i: Index i in the secular equation
    :type i: int
    :param d: Array of diagonal elements d
    :type d: NDArray[float64]
    :param z: Array of first column elements z
    :type z: NDArray[float64]
    :param rho: Scalar value rho
    :type rho: float
    :param max_iter: Maximum number of iterations
    :type max_iter: int
    :param tol: Tolerance level for convergence
    :type tol: float
    :return: Computed i-th root of the secular equation.
    :rtype: float
    """
    n = d.shape[0]
    pole_lb = d[i]
    if i + 1 < n:
        pole_ub = d[i + 1]
    else:
        pole_ub = d[i] + rho * np.linalg.norm(z)

    search_lb = pole_lb
    search_ub = pole_ub
    
    omega = (search_lb + search_ub) / 2.0

    for iteration in range(max_iter):
        # Judge which side to evaluate
        left_offset = omega - pole_lb
        right_offset = pole_ub - omega
        left_search = left_offset <= right_offset or i == n - 1

        if left_search:
            f, abs_sum, df = secular_function_left(left_offset, i, d, z, rho)
            offset = left_offset
        else:
            f, abs_sum, df = secular_function_right(right_offset, i, d, z, rho)
            offset = right_offset
        
        # Check stopping criterion
        if stopping_criterion(n, f, abs_sum, tol):
            return omega
        
        if f > 0:
            search_ub = omega
        else:
            search_lb = omega
        
        # Update offsets.
        step = rational_interpolation(offset, f, df)
        next_omega = omega + step

        if next_omega == omega: # Too small step
            return omega
        if (search_ub - search_lb) <= 2 * np.finfo(float).eps * max(1.0, abs(omega)): # Interval too small
            return omega

        # Bisection safeguard
        if next_omega <= search_lb or next_omega >= search_ub:
            step = (search_lb + search_ub) / 2.0 - omega

        omega += step

    if iteration == max_iter - 1:
        print(f"Failed at i={i}, f={f}, step={step}, interval=({search_lb}, {search_ub})")
        raise RuntimeError("Maximum iterations reached without convergence in secular_single_root.")
    
    return omega

def compute_secular_eigenvectors(d: NDArray[float64], z: NDArray[float64], roots: NDArray[float64], rho: float) -> Tuple[NDArray[float64], NDArray[float64]]:
    """
    Computes the eigenvectors corresponding to the roots of the secular equation.
    
    :param d: Array of diagonal elements d
    :type d: NDArray[float64]
    :param z: Array of first column elements z
    :type z: NDArray[float64]
    :param roots: Array of computed roots of the secular equation
    :type roots: NDArray[float64]
    :param rho: Scalar value rho
    :type rho: float
    :return: U: Left singular vectors, V: Right singular vectors
    :rtype: Tuple[NDArray[float64], NDArray[float64]]
    """
    n = d.shape[0]
    m = roots.shape[0]

    U = np.zeros((n, m), dtype=float64)
    V = np.zeros((n, m), dtype=float64)
    z_hat = np.zeros(n, dtype=float64)
    d_sq = d * d
    roots_sq = roots * roots

    # Reconstruct z vector
    for k in range(n):
        diff_sigma = np.abs(roots_sq - d_sq[k])
        diff_sigma[diff_sigma < np.finfo(float).eps] = np.finfo(float).eps
        log_numerator = np.sum(np.log(diff_sigma))

        diff_d = np.abs(d_sq - d_sq[k])
        diff_d_masked = np.concatenate((diff_d[:k], diff_d[k+1:]))
        diff_d_masked[diff_d_masked < np.finfo(float).eps] = np.finfo(float).eps
        log_denominator = np.sum(np.log(diff_d_masked))

        log_z_hat_sq = log_numerator - log_denominator - np.log(abs(rho))

        z_hat[k] = np.exp(log_z_hat_sq / 2.0)

        z_hat[k] = np.copysign(z_hat[k], z[k])  # Restore sign
    
    # Compute eigenvectors
    Denom = d_sq[:, np.newaxis] - roots_sq[np.newaxis, :]

    # Extreme value safeguarding
    sign_denom = np.sign(Denom)
    sign_denom[sign_denom == 0] = 1
    denom_abs = np.abs(Denom)
    denom_abs[denom_abs < np.finfo(float).eps] = np.finfo(float).eps
    Denom_safe = sign_denom * denom_abs
    U = z_hat[:, np.newaxis] / Denom_safe
    V = d[:, np.newaxis] * U
    V[0, :] = -1

    for j in range(m):
        # Normalize V
        norm_v = norm(V[:, j])
        if norm_v > np.finfo(float).eps:
            V[:, j] /= norm_v
        
        # Normalize U
        norm_u = norm(U[:, j])
        if norm_u > np.finfo(float).eps:
            U[:, j] /= norm_u
    return U, V

def small_element(value: float, tol: float = np.finfo(float).eps * 100) -> bool:
    """
    Checks if a given value is considered small based on a tolerance.
    We consider the relative error.
    
    :param value: Value to be checked
    :type value: float
    :param tol: Tolerance level
    :type tol: float
    :return: True if the value is small, False otherwise
    :rtype: bool
    """
    return abs(value) <= tol

def local_deflation(d: NDArray[float64], z: NDArray[float64], rho: float, tol: float)\
        -> Tuple[NDArray[np.float64], NDArray[np.float64], List[float], List[int], NDArray[np.bool], List[Tuple]]:
    """
    Performs local deflation on the secular equation problem.
    1. Sorts for d.
    2. Deflates for small z components.
    3. Deflates for close d's.
    
    :param d: Array of diagonal elements d
    :type d: NDArray[float64]
    :param z: Array of first column elements z
    :type z: NDArray[float64]
    :param rho: Scalar value rho
    :type rho: float
    :param tol: Tolerance level for deflation
    :type tol: float
    :return: d_secular: irreducible diagonal elements for secular equation,
             z_secular: irreducible first column elements for secular equation,
             deflated_roots: list of deflated roots,
             perm_indices: permutation indices to map back to original order,
             deflated_indices: mask of deflated roots,
             givens_rotations: list of Givens rotations applied during deflation
    :rtype: Tuple[NDArray[float64], NDArray[float64], List[float], List[int], NDArray[np.bool], List[Tuple]]
    """
    n = d.shape[0]

    # Step 1: Sort d and permute z accordingly
    perm_indices = np.argsort(d)
    d_sorted = d[perm_indices].copy()
    z_sorted = z[perm_indices].copy()

    deflated_mask = np.zeros(n, dtype=bool)                 # True if deflated
    givens_rotations = []                                   # Store Givens rotations

    # Step 2: Sweep and deflation.
    for i in range(n):
        if deflated_mask[i]:
            continue

        # Case 1: Small z component
        if small_element(z_sorted[i], tol):
            deflated_mask[i] = True
            continue

        # Case 2: Close d components
        if i < n - 1 and not deflated_mask[i + 1]:
            diff = abs(d_sorted[i + 1] - d_sorted[i])
            if diff <= tol:
                # Apply Givens rotation to (z[i], z[i+1])
                c, s, r = givens(z_sorted[i], z_sorted[i + 1])
                givens_rotations.append((i, i + 1, c, s))
                
                # Update z components
                z_sorted[i] = r
                z_sorted[i + 1] = 0.0

                # Deflate the (i+1)-th component
                deflated_mask[i + 1] = True
    
    # Collect non-deflated components
    d_secular = d_sorted[~deflated_mask]
    z_secular = z_sorted[~deflated_mask]

    # Collect deflated roots and their original indices
    deflated_roots = d_sorted[deflated_mask].tolist()

    return d_secular, z_secular, deflated_roots, perm_indices.tolist(), deflated_mask, givens_rotations

# Main solver function for secular equation
def solve_secular_equation(d: NDArray[float64], z: NDArray[float64], rho: float, max_iter: int = 1000, eps: float = np.finfo(float).eps, compute_eigvec: bool = True)\
        -> SecularSolution:
    """
    Solves the secular equation after performing local deflation.
    
    :param d: Array of diagonal elements d
    :type d: NDArray[float64]
    :param z: Array of first column elements z
    :type z: NDArray[float64]
    :param rho: Scalar value rho
    :type rho: float
    :param max_iter: Maximum number of iterations for root finding
    :type max_iter: int
    :param eps: Tolerance level for convergence
    :type eps: float
    :param compute_eigvec: Flag to indicate if eigenvectors should be computed
    :type compute_eigvec: bool
    :return: SecularSolution named tuple containing all relevant results
    :rtype: SecularSolution
    """
    # Perform local deflation
    n = d.shape[0]
    tol = 2 * n * n * eps * (np.max(abs(d)) + abs(rho) * z.dot(z))

    # Local deflation
    d_sec, z_sec, def_roots, perm, def_mask, givens_rots = local_deflation(d, z, rho, tol)

    roots_sec = []
    m = len(d_sec)
    if m > 0:
        for i in range(m):
            root = secular_single_root(i=i, d=d_sec, z=z_sec, rho=rho, max_iter=max_iter)
            roots_sec.append(root)
    # Compute eigenvectors if required
    U, V = None, None
    if compute_eigvec:
        U, V = np.eye(n, dtype=float64), np.eye(n, dtype=float64)

        if m > 0:
            U_sec, V_sec = compute_secular_eigenvectors(d_sec, z_sec, np.array(roots_sec, dtype=float64), rho)

            active_indices = np.where(~def_mask)[0]
            U[np.ix_(active_indices, active_indices)] = U_sec
            V[np.ix_(active_indices, active_indices)] = V_sec
        
        # Apply Givens rotations to U and V
        for (i, j, c, s) in reversed(givens_rots):
            ui = U[i, :].copy()
            uj = U[j, :].copy()
            U[i, :] = c * ui - s * uj
            U[j, :] = s * ui + c * uj

            vi = V[i, :].copy()
            vj = V[j, :].copy()
            V[i, :] = c * vi - s * vj
            V[j, :] = s * vi + c * vj
        
        # Apply permutation to U and V
        U = U[np.argsort(perm), :]
        V = V[np.argsort(perm), :]

    
    # Combine deflated roots and computed roots
    all_roots = np.zeros(n, dtype=float64)
    all_roots[~def_mask] = roots_sec
    all_roots[def_mask] = def_roots

    return SecularSolution(
        all_roots=all_roots,
        perm_indices=np.array(perm, dtype=np.intp),
        deflated_mask=def_mask,
        givens_rotations=givens_rots,
        d_secular=d_sec,
        z_secular=z_sec,
        roots_secular=roots_sec,
        roots_deflated=def_roots,
        k_secular=m,
        U_local=U,
        V_local=V
    )