import numpy as np

E = np.exp(1)
F_0 = 1
F_1 = 1
fib_seq = [F_0, F_1]

def f(x):
    return float(np.exp(-x)) + x**2

def solve(a, b, ep):
    while fib_seq[-1] < (b-a) / ep:
        fib_seq.append(fib_seq[-1] + fib_seq[-2])

    print(fib_seq,len(fib_seq))

    la = a + (fib_seq[-3] * (b-a) / fib_seq[-1])
    mu = a + (fib_seq[-2] * (b-a)/ fib_seq[-1])

    print(la,mu)
    k = 1

    la_res = f(la)
    mu_res = f(mu)
        
    while k < len(fib_seq) - 2:
        if la_res > mu_res:
            a = la
            la = mu
            mu = a + (fib_seq[-2-k] * (b-a) / fib_seq[-1-k])
            la_res = mu_res
            mu_res = f(mu)
        else:
            b = mu
            mu = la
            la = a + (fib_seq[-3-k] * (b-a) / fib_seq[-1-k])
            mu_res = la_res
            la_res = f(la)
        print(f"第{k}次迭代:a={a},b={b}")
        k += 1


    if k == len(fib_seq) - 2:
        la = mu = (a+b) / 2
        print(f"第{k}次迭代:a={a},b={b}")

    k += 1

    if k == len(fib_seq) - 1:
        la_n = la
        mu_n = la + ep
        
        if f(la_n) > f(mu_n):
            a = la_n
        else:
            b = la_n
    print(f"第{k+1}次迭代:a={a},b={b}")
    print(f"最小值点{(a+b)/2}")
    print(f"最小值{f((a+b)/2)}")
solve(0,1,0.15)
