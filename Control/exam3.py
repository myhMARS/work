import numpy as np


def state_stability(A):
    """
    使用李亚普诺夫第一法分析状态稳定性
    :param A: 系统矩阵 A (n x n)
    :return: 系统的状态稳定性结论
    """
    # 选择对称正定矩阵 P
    P = np.eye(A.shape[0])  # 简单选择单位矩阵作为正定矩阵

    # 计算 V(x) 的导数，即 dV/dt = x^T (A^T P + P A) x
    A_T_P_P_A = np.dot(A.T, P) + np.dot(P, A)

    # 判断 A^T P + P A 是否为负定矩阵
    eigenvalues = np.linalg.eigvals(A_T_P_P_A)  # 计算矩阵的特征值

    if np.all(eigenvalues < 0):
        return "系统的状态是渐近稳定的"
    elif np.all(eigenvalues <= 0):
        return "系统的状态是稳定的"
    else:
        return "系统的状态是不可稳定的"


def output_stability(A, C):
    """
    使用李亚普诺夫第一法分析输出稳定性
    :param A: 系统矩阵 A (n x n)
    :param C: 输出矩阵 C (m x n)
    :return: 系统的输出稳定性结论
    """
    # 选择对称正定矩阵 P
    P = np.eye(A.shape[0])  # 简单选择单位矩阵作为正定矩阵

    # 计算 V(x) 的导数，即 dV/dt = x^T (A^T P + P A) x
    A_T_P_P_A = np.dot(A.T, P) + np.dot(P, A)

    # 计算输出矩阵 C 的导数部分
    output_part = np.dot(C.T, C)  # 计算 C^T C

    # 判断 A^T P + P A 和 C^T C 是否满足稳定性条件
    eigenvalues_A = np.linalg.eigvals(A_T_P_P_A)
    eigenvalues_output = np.linalg.eigvals(output_part)

    # 如果系统是渐近稳定的，并且输出矩阵也收敛
    if np.all(eigenvalues_A < 0) and np.all(eigenvalues_output > 0):
        return "系统的输出是渐近稳定的"
    elif np.all(eigenvalues_A <= 0):
        return "系统的输出是稳定的"
    else:
        return "系统的输出是不可稳定的"


# 示例：定义系统矩阵 A 和输出矩阵 C
A = np.array([[-1, 0], [0, -1]])
B = np.array([[1], [1]])
C = np.array([[0, 1]])

# 使用李亚普诺夫第一法分析状态稳定性
state_stability_result = state_stability(A)
print(state_stability_result)

# 使用李亚普诺夫第一法分析输出稳定性
output_stability_result = output_stability(A, C)
print(output_stability_result)
