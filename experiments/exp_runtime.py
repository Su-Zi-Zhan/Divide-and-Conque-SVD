# .experiments/exp_runtime.py
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
from src.dnc_svd import DCSVDSolver

def benchmark_svd(min_size=50, max_size=500, step=50, repeat=3):
    """
    测试 D&C SVD 和 NumPy SVD 的运行时间
    """
    sizes = range(min_size, max_size + 1, step)
    dnc_times = []
    numpy_times = []
    
    print(f"开始测试，矩阵规模从 {min_size} 到 {max_size}...")
    
    # 初始化求解器
    solver = DCSVDSolver(tol=1e-12)
    
    for n in tqdm(sizes):
        dnc_avg = 0.0
        numpy_avg = 0.0
        
        for _ in range(repeat):
            # 1. 构造随机二对角矩阵 (Lower Bidiagonal)
            # 你的代码逻辑是提取 k=-1 (下对角线)
            d = np.random.rand(n).astype(np.float64)
            e = np.random.rand(n - 1).astype(np.float64)
            
            # 构造完整矩阵用于 NumPy 测试
            B = np.diag(d) + np.diag(e, k=-1)
            
            # --- 测试 D&C SVD ---
            start_time = time.time()
            # 直接传入 (d, e) 元组，避免内部再次提取的开销
            solver.solve((d, e)) 
            dnc_avg += time.time() - start_time
            
            # --- 测试 NumPy SVD ---
            start_time = time.time()
            np.linalg.svd(B, full_matrices=True) # 保持 full_matrices=True 以公平对比
            numpy_avg += time.time() - start_time
            
        dnc_times.append(dnc_avg / repeat)
        numpy_times.append(numpy_avg / repeat)
        
    return list(sizes), dnc_times, numpy_times

def plot_results(sizes, dnc_times, numpy_times):
    """
    绘制可视化图表
    """
    sizes = np.array(sizes)
    dnc_times = np.array(dnc_times)
    
    # --- 计算参考线 ---
    # 我们以 D&C SVD 的最后一个点为基准进行对齐
    # Ref = k * n^3
    # k = time[-1] / size[-1]^3
    scale_factor = dnc_times[-1] / (sizes[-1]**3)
    ref_n2 = scale_factor * (sizes**2)
    
    plt.figure(figsize=(14, 6))
    
    # 图 1: 线性刻度 (Linear Scale)
    plt.subplot(1, 2, 1)
    plt.plot(sizes, dnc_times, 'o-', label='Your D&C SVD', linewidth=2, markersize=6)
    plt.plot(sizes, numpy_times, 's--', label='NumPy SVD (LAPACK)', linewidth=2, markersize=6)
    # 添加参考线
    plt.plot(sizes, ref_n2, 'k:', label=r'$O(n^2)$ Reference', linewidth=1.5, alpha=0.7)
    
    plt.xlabel('Matrix Size (N)', fontsize=12)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.title('Runtime: D&C vs NumPy (Linear Scale)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=11)
    
    # 图 2: 对数刻度 (Log-Log Scale)
    # 这是验证复杂度的关键图
    plt.subplot(1, 2, 2)
    plt.loglog(sizes, dnc_times, 'o-', label='Your D&C SVD', linewidth=2)
    plt.loglog(sizes, numpy_times, 's--', label='NumPy SVD', linewidth=2)
    # 添加参考线
    plt.loglog(sizes, ref_n2, 'k:', label=r'$O(n^3)$ Reference', linewidth=2)
    
    plt.xlabel('Matrix Size (N)', fontsize=12)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.title('Runtime Complexity (Log-Log Scale)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7, which="both")
    plt.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig('svd_runtime_comparison.pdf', dpi=300)
    plt.show()

if __name__ == "__main__":
    # 这里的 max_size 可以根据你的电脑性能调整
    # 分治法是 Python 实现的，对于大矩阵可能会比 C 实现的 NumPy 慢，这是正常的
    sizes, t_dnc, t_np = benchmark_svd(min_size=10, max_size=1000, step=20, repeat=5)
    plot_results(sizes, t_dnc, t_np)