import numpy as np
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from numpy.typing import NDArray
from numpy import float64
from src.secular import secular_single_root

def test_secular_single_root():
    d = np.array([0.0, 1.0, 2.0], dtype=float64)
    z = np.array([1.0, 2.0, 1.0], dtype=float64)
    z /= np.linalg.norm(z)
    rho = 0.5

    root1 = secular_single_root(0, d, z, rho)
    root2 = secular_single_root(1, d, z, rho)
    root3 = secular_single_root(2, d, z, rho)

    print(f"Computed roots: {root1}\n{root2}\n{root3}")

if __name__ == "__main__":
    test_secular_single_root()