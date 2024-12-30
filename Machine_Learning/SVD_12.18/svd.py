import numpy as np

A = np.array([[1, 2, 3, 4],[5, 6, 7, 8],[9, 10, 11, 12]])
U, s, Vh = np.linalg.svd(A)
S = np.zeros_like(A)
for i in range(len(s)):
    S[i][i]=s[i]

print(U)
print(s)
print(S)
print(Vh.T)
tmp = np.dot(U[:,:1],np.diag(s[:1]))
res = np.dot(tmp,Vh[:1,:])
print(U@S@Vh)
