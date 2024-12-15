import numpy as np
from scipy.linalg import solve_continuous_are

def lyapunov_stability(A):
    """
    使用李亚普诺夫方程求解P矩阵并判断系统稳定性
    :param A: 系统矩阵 A (n x n)
    :return: 系统稳定性结论
    """
    # 选择正定矩阵 Q, 这里我们选择单位矩阵 I
    Q = np.eye(A.shape[0])
    
    # 使用 SciPy 的 solve_continuous_are 函数求解李亚普诺夫方程
    P = solve_continuous_are(A, np.zeros(A.shape), Q, np.eye(A.shape[0]))

    # 判断 P 是否正定，即其特征值是否全为正
    eigenvalues_P = np.linalg.eigvals(P)
    
    if np.all(eigenvalues_P > 0):
        return "系统是渐近稳定的"
    else:
        return "系统是不稳定的"

# 示例：定义系统矩阵 A
A = np.array([[0, 1], [-1, -1]])

# 使用李亚普诺夫方程判断系统稳定性
stability = lyapunov_stability(A)
print(stability)
