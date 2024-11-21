import math
import numpy as np
from sympy import symbols, diff, solve


x1, x2 = symbols('x1 x2')

f =2 * (x1 ** 2) + x2 ** 2

grad1 = diff(f, x1)
grad2 = diff(f, x2)
x1_value = 1
x2_value = 1

epsilon = 0.1
max_iter = 100
iter_count = 0

while True:
    grad1_value = grad1.subs({
        x1: x1_value,
        x2: x2_value
    }).evalf()
    grad2_value = grad2.subs({
        x1: x1_value,
        x2: x2_value
    }).evalf()
    
    if math.sqrt(grad1_value ** 2 + grad2_value ** 2) < epsilon:
        break

    t = symbols('t')
    x1_updated = x1_value - grad1_value * t
    x2_updated = x2_value - grad2_value * t


    f_updated = f.subs({
        x1: x1_updated,
        x2: x2_updated
    })
    grad_t = diff(f_updated, t)
    t_value = solve(grad_t, t)[0]

    x1_value -= grad1_value * t_value
    x2_value -= grad2_value * t_value
    iter_count += 1

    print(f'iter:{iter_count}, min_porint:({x1_value}, {x2_value}), func_num:{f.subs({x1: x1_value,x2: x2_value})}')
