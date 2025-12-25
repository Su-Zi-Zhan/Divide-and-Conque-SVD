# .src/secular.py
"""
Compute the solutions to the secular equations.
"""
import numpy as np

from numpy.typing import NDArray
from numpy import float64
from typing import Tuple

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
