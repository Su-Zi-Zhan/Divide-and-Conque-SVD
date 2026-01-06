# .experiments/exp_accuracy.py
"""
Experiment to evaluate the accuracy of the secular solver.
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.secular import solve_secular_equation
# from src.dnc_svd import DCSVDSolver

def plot_secular_function():
    d = np.array([0.0, 2.0, 2.5, 5.0])
    z = np.array([0.5, 0.4, 0.5, 0.5])
    rho = 1.0
    
    
    print("Solving secular equation...")
    sol = solve_secular_equation(d, z, rho, compute_eigvec=True)
    
    d_sec = sol.d_secular
    z_sec = sol.z_secular
    roots = sol.all_roots 
    

    def secular_val(x, d_arr, z_arr, rho_val):
        res = 1.0
        for i in range(len(d_arr)):
            # f(x) = 1 + rho * sum( z_i^2 / (d_i^2 - x^2) )
            res += rho_val * (z_arr[i]**2) / (d_arr[i]**2 - x**2)
        return res

    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_min = 0
    x_max = max(d) + 2.0
    
    intervals = []
    sorted_d = np.sort(d_sec)
    
    # First interval [0, d0)
    intervals.append(np.linspace(0, sorted_d[0] - 0.01, 200))
    
    # Middle intervals (di, di+1)
    for i in range(len(sorted_d) - 1):
        intervals.append(np.linspace(sorted_d[i] + 0.01, sorted_d[i+1] - 0.01, 200))
        
    # Last interval (dn, max)
    intervals.append(np.linspace(sorted_d[-1] + 0.01, x_max, 200))
    
    # Plot the curves
    for x_vals in intervals:
        y_vals = [secular_val(x, d_sec, z_sec, rho) for x in x_vals]
        ax.plot(x_vals, y_vals, 'b-', linewidth=1.5)

    # 4. Mark key points
    # Draw the 0 axis
    ax.axhline(0, color='k', linestyle='-', linewidth=0.5)
    
    # Draw poles - vertical dashed lines
    for p in d_sec:
        ax.axvline(p, color='r', linestyle='--', alpha=0.5, label='Poles ($d_i$)' if p == d_sec[0] else "")
        
    valid_roots = [r for r in sol.roots_secular]

    ax.plot(valid_roots, np.zeros_like(valid_roots), 'go', markersize=8, label='Computed Roots ($\sigma_i$)', zorder=5)
    
    # 5. Beautify the plot
    ax.set_ylim(-5, 5) # Limit y-axis range because it tends to infinity near poles
    ax.set_xlim(0, x_max)
    ax.set_xlabel('$\sigma$ (Singular Values)', fontsize=12)
    ax.set_ylabel('$f(\sigma)$', fontsize=12)
    ax.set_title(f'Secular Equation Function\n$f(\sigma) = 1 + \\rho \sum \\frac{{z_k^2}}{{d_k^2 - \sigma^2}}$', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('secular_function.pdf', dpi=300)
    print("Plot saved to secular_function.pdf")
    
    # 如果环境支持，plt.show()

if __name__ == "__main__":
    plot_secular_function()