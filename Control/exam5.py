import numpy as np
import control as ctrl


def design_state_feedback(A, B, desired_poles):
    """
    设计状态反馈矩阵 K，使闭环系统的极点与期望极点一致
    :param A: 系统状态矩阵 A (n x n)
    :param B: 系统输入矩阵 B (n x m)
    :param desired_poles: 期望的极点位置
    :return: 状态反馈矩阵 K
    """
    # 检查系统是否能控
    controllability_matrix = np.hstack([np.linalg.matrix_power(A, i) @ B for i in range(A.shape[0])])

    if np.linalg.matrix_rank(controllability_matrix) != A.shape[0]:
        raise ValueError("系统不可控，无法设计状态反馈矩阵")

    # 使用 control 库的 place 函数来计算状态反馈矩阵 K
    K = ctrl.place(A, B, desired_poles)

    return K


# 示例：定义系统矩阵 A 和 B
A = np.array([[0, 1, 0], [0, -1, 1], [0, 0, -2]])
B = np.array([[0], [0], [1]])

# 设定期望的闭环极点
desired_poles = [-2, -1 + 1j, -1 - 1j]

# 设计状态反馈矩阵 K
K = design_state_feedback(A, B, desired_poles)
print("状态反馈矩阵 K:")
print(K)

# 验证设计结果：计算闭环系统的极点
A_cl = A - np.dot(B, K)
eigvals_cl = np.linalg.eigvals(A_cl)
print("闭环系统的极点：")
print(eigvals_cl)
