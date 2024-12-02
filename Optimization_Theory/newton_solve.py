from sympy import symbols, diff, hessian
import numpy as np
import math


def grad_len(grad):
    return math.sqrt(sum(g**2 for g in grad))


def newton_dou(x_init, fun, epsilon):
    while True:
        grad = np.array([diff(fun, x1).subs({x1: x_init[0], x2: x_init[1]}),
                         diff(fun, x2).subs({x1: x_init[0], x2: x_init[1]})], dtype=float)
        """
        hessian = np.array([[diff(fun, x1, 2).subs({x1: x_init[0], x2: x_init[1]}),
                             diff(diff(fun, x1), x2).subs({x1: x_init[0], x2: x_init[1]})],
                            [diff(diff(fun, x2), x1).subs({x1: x_init[0], x2: x_init[1]}),
                             diff(fun, x2, 2).subs({x1: x_init[0], x2: x_init[1]})]], dtype=float)
        """
        hessian_ = hessian(fun, (x1, x2))
        hessian_ = np.array(hessian_.subs({x1: x_init[0], x2: x_init[1]}), dtype=float)

        inverse_hessian = np.linalg.inv(hessian_)
        x_new = x_init - np.matmul(inverse_hessian, grad)

        print('x:',x_new)
        print('fun:',fun.subs({x1: x_new[0], x2: x_new[1]}))

        if grad_len(grad) < epsilon:
            return x_new
        x_init = x_new


if __name__ == "__main__":
    x1, x2 = symbols('x1 x2')
    fun = (x2 - x1 ** 2) ** 2 +(1 - x1) ** 2
    data =[0.0, 0.0]
    epsilon = 0.1
    x_min = newton_dou(data, fun, epsilon)
    print(f"所求的最小值点: x1为{x_min[0]:.5f}, x2为{x_min[1]:.5f}")
    print(f"所求的最小值为: {fun.subs({x1: x_min[0], x2: x_min[1]}):.5f}")
