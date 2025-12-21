# ./src/utils.py
"""
Utility functions for the project.
"""
import numpy as np

from numpy.typing import NDArray
from typing import Tuple
from math import sqrt

def givens(a: float, b: float) -> Tuple[float, float]:
    """
    Compute Givens rotation angles.
    :param a: vector element to rotate.
    :param b: vector element to rotate.
    :return: (c, s), where c represents cosine and s represents sine.
    """
    if b == 0:
        c = 1
        s = 0
    elif abs(b) > abs(a):
        t = a / b
        s = 1 / sqrt(1 + t * t)
        c = s * t
    else:
        t = b / a
        c = 1 / sqrt(1 + t * t)
        s = c * t
    return c, s

