import numpy as np
from scipy.linalg import solve_continuous_lyapunov, eigvals

# 定义系统矩阵 A
A = np.array([[0, 1],
              [-1, -1]])

# 定义正定矩阵 Q
Q = np.eye(2)  # 选择单位矩阵 Q

# 求解李雅普诺夫方程 A^T P + P A = -Q
P = solve_continuous_lyapunov(A.T, -Q)
print("P 矩阵为：\n", P)

eig_P = np.linalg.eigvals(P)
print("P 的特征值为：", eig_P)

if np.all(eig_P > 0):
    print("方法一：P 矩阵是正定的 (所有特征值 > 0)")
else:
    print("方法一：P 矩阵不是正定的 (存在特征值 <= 0)")

det1 = P[0, 0]
det2 = np.linalg.det(P)

print("P 的第一个主子矩阵的行列式 (det1):", det1)
print("P 的整个矩阵的行列式 (det2):", det2)

if det1 > 0 and det2 > 0:
    print("方法二：P 矩阵是正定的 (希尔维斯特判据成立)")
else:
    print("方法二：P 矩阵不是正定的 (希尔维斯特判据不成立)")
