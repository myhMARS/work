'''
FR共轭方向算法(二元二次函数)
2023.10.20
'''
import numpy as np
from sympy import symbols, diff, solve
 
x1 = symbols("x1")
x2 = symbols("x2")
λ = symbols("λ")
f = x1 ** 2 + 2 * x2 ** 2 - 2 * x1 * x2 + 2 * x2 + 2 
 
 
def fletcher_reeves(x1_init, x2_init, ε):
    # 一阶求导
    grad_1 = diff(f, x1)
    grad_2 = diff(f, x2)
 
    x1_curr = x1_init
    x2_curr = x2_init
 
    first_grad_1_value = grad_1.subs({x1: x1_curr, x2: x2_curr})
    first_grad_2_value = grad_2.subs({x1: x1_curr, x2: x2_curr})
 
    g1 = np.array([first_grad_1_value, first_grad_2_value], dtype=float)
    norm_result_1 = np.linalg.norm(g1, ord=2, axis=0)
    if norm_result_1 <= ε:
        print("算法停止，该精度下的最优解是[%f,%f]" % (float(x1_curr), float(x2_curr)))
    else:
        x1_new = x1_curr - λ * first_grad_1_value
        x2_new = x2_curr - λ * first_grad_2_value
 
        f_new = f.subs({x1: x1_new, x2: x2_new})
        grad_3 = diff(f_new, λ)
        λ_value = solve(grad_3, λ)[0]
 
        x1_curr = x1_new.subs(λ, λ_value)
        x2_curr = x2_new.subs(λ, λ_value)
 
        print("第{}次迭代".format(1), "当前最优解为[%f,%f]" % (float(x1_curr), float(x2_curr)))
 
        second_grad_1_value = grad_1.subs({x1: x1_curr, x2: x2_curr})
        second_grad_2_value = grad_2.subs({x1: x1_curr, x2: x2_curr})
 
        g2 = np.array([second_grad_1_value, second_grad_2_value], dtype=float)
        norm_result_2 = np.linalg.norm(g2, ord=2, axis=0)
        if norm_result_2 <= ε:
            print("算法停止，该精度下的最优解是[%f,%f]" % (float(x1_curr), float(x2_curr)))
        else:
            α1 = norm_result_2 ** 2 / norm_result_1 ** 2
            p2_1 = -1 * second_grad_1_value - α1 * first_grad_1_value
            p2_2 = -1 * second_grad_2_value - α1 * first_grad_2_value
 
            x1_new = x1_curr + λ * p2_1
            x2_new = x2_curr + λ * p2_2
 
            f_new = f.subs({x1: x1_new, x2: x2_new})
            grad_4 = diff(f_new, λ)
            λ_value = solve(grad_4, λ)[0]
 
            x1_curr = x1_new.subs(λ, λ_value)
            x2_curr = x2_new.subs(λ, λ_value)
 
            print("第{}次迭代".format(2), "当前最优解为[%f,%f]" % (float(x1_curr), float(x2_curr)))
 
            grad_1_value = grad_1.subs({x1: x1_curr, x2: x2_curr})
            grad_2_value = grad_2.subs({x1: x1_curr, x2: x2_curr})
 
            g3 = np.array([grad_1_value, grad_2_value], dtype=float)
            norm_result_3 = np.linalg.norm(g3, ord=2, axis=0)
            if norm_result_3 <= ε:
                print("算法停止，该精度下的最优解是[%f,%f]" % (float(x1_curr), float(x2_curr)))
 
    return x1_curr, x2_curr
 
 
result = fletcher_reeves(0, 0, 0.1)
