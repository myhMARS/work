import numpy as np

r = 0.618


def f(x):
    return np.exp(-x) + x ** 2


def solve(a, b, l):
    k = 1
    la = a + (1 - r) * (b - a)
    mu = a + r * (b - a)
    while b - a >= l:

        print(f"第{k}次迭代:la={la},mu={mu}")
        if f(la) < f(mu):
            a = a
            b = mu
            mu = la
            la = a + (1 - r) * (b - a)
        else:
            a = la
            b = b
            la = mu
            mu = a + r * (b - a)

        k += 1

    print(f"第{k}次迭代:la={la},mu={mu}")
    print(f"最小值点:{(a + b) / 2:.3f}")
    print(f"最小值:{f((a + b) / 2):.3f}")


solve(0, 1, 0.2)
