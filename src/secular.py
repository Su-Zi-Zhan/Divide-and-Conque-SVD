# .src/secular.py
"""
Compute the solutions to the secular equations.
"""
import numpy as np

from numpy.typing import NDArray
from numpy.linalg import norm
from numpy import float64, abs
from typing import Tuple, List
from .utils import givens

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

def local_deflation(d: NDArray[float64], z: NDArray[float64], rho: float, eps: float = np.finfo(float).eps)\
        -> Tuple[NDArray[np.float64], NDArray[np.float64], List[float], List[int], NDArray, List[Tuple]]:
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
    :rtype: Tuple[NDArray[float64], NDArray[float64], List[float], List[int], NDArray, List[Tuple]]
    """
    n = d.shape[0]

    # Step 1: Sort d and permute z accordingly
    perm_indices = np.argsort(d)
    d_sorted = d[perm_indices].copy()
    z_sorted = z[perm_indices].copy()

    deflated_mask = np.zeros(n, dtype=bool)                 # True if deflated
    givens_rotations = []                                   # Store Givens rotations
    m_norm_estimated = np.max(abs(d)) + abs(rho) * z.dot(z) # Estimate of ||M||_2
    tol = 2 * n * n * eps * m_norm_estimated                # Tolerance for deflation


    # Step 2: Sweep and deflation.
    for i in range(n):
        if deflated_mask[i]:
            continue

        # Case 1: Small z component
        if small_element(z_sorted[i], tol):
            deflated_mask[i] = True
            continue

        # Case 2: Small d components
        if i == 0 and small_element(d_sorted[i], tol):
            d_sorted[i] = tol
            continue
        elif small_element(d_sorted[i], tol):
            d_sorted[i] = 0
            continue

        # Case 3: Close d components
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
def solve_secular_equation(d: NDArray[float64], z: NDArray[float64], rho: float, max_iter: int = 1000, eps: float = np.finfo(float).eps)\
        -> NDArray[float64]:
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
    :return: Array of computed roots of the secular equation.
    :rtype: NDArray[float64]
    """
    # Perform local deflation
    pass
