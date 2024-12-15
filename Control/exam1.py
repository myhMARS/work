import numpy as np

def check_controllability(A, B):
    """
    检查系统能控性
    """
    n = A.shape[0]  # 系统的阶数
    controllability_matrix = B
    for i in range(1, n):
        controllability_matrix = np.hstack((controllability_matrix, np.linalg.matrix_power(A, i) @ B))
    
    rank = np.linalg.matrix_rank(controllability_matrix)
    is_controllable = rank == n
    return is_controllable, controllability_matrix, rank

def check_observability(A, C):
    """
    检查系统能观性
    """
    n = A.shape[0]  # 系统的阶数
    observability_matrix = C
    for i in range(1, n):
        observability_matrix = np.vstack((observability_matrix, C @ np.linalg.matrix_power(A, i)))
    
    rank = np.linalg.matrix_rank(observability_matrix)
    is_observable = rank == n
    return is_observable, observability_matrix, rank

# 示例：定义系统
A = np.array([[-3, 1], [1, -3]])
B = np.array([[1,1], [1,1]])
C = np.array([[1, 1],[1, -1 ]])

# 检查能控性
controllable, C_matrix, C_rank = check_controllability(A, B)
print("能控性:", "是" if controllable else "否")
print("能控性矩阵:\n", C_matrix)
print("能控性矩阵秩:", C_rank)

# 检查能观性
observable, O_matrix, O_rank = check_observability(A, C)
print("能观性:", "是" if observable else "否")
print("能观性矩阵:\n", O_matrix)
print("能观性矩阵秩:", O_rank)
