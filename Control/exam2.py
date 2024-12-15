import numpy as np

def controllable_canonical_form(A, B):
    """
    转化为能控标准二型
    """
    n = A.shape[0]
    
    # 构建能控性矩阵
    controllability_matrix = B
    for i in range(1, n):
        controllability_matrix = np.hstack((controllability_matrix, np.linalg.matrix_power(A, i) @ B))
    
    # 检查是否满秩
    if np.linalg.matrix_rank(controllability_matrix) < n:
        raise ValueError("系统不可控，无法转化为能控标准二型")
    
    # 相似变换矩阵
    T = controllability_matrix
    T_inv = np.linalg.inv(T)
    
    # 转化后的矩阵
    A_c = T_inv @ A @ T
    B_c = T_inv @ B
    return A_c, B_c

def observable_canonical_form(A, C):
    """
    转化为能观标准一型
    """
    n = A.shape[0]
    
    # 构建能观性矩阵
    observability_matrix = C
    for i in range(1, n):
        observability_matrix = np.vstack((observability_matrix, C @ np.linalg.matrix_power(A, i)))
    
    # 检查是否满秩
    if np.linalg.matrix_rank(observability_matrix) < n:
        raise ValueError("系统不可观，无法转化为能观标准一型")
    
    # 相似变换矩阵
    T = observability_matrix.T
    T_inv = np.linalg.inv(T)
    
    # 转化后的矩阵
    A_o = T_inv @ A @ T
    C_o = C @ T
    return A_o, C_o

# 示例：定义系统
A = np.array([[1, 2], [4, 3]])
B = np.array([[1], [1]])
C = np.array([[0, 1]])

# 转化为能控标准二型
A_c, B_c = controllable_canonical_form(A, B)
print("能控标准二型 A_c:\n", A_c)
print("能控标准二型 B_c:\n", B_c)

# 转化为能观标准一型
A_o, C_o = observable_canonical_form(A, C)
print("能观标准一型 A_o:\n", A_o)
print("能观标准一型 C_o:\n", C_o)
