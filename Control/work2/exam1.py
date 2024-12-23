import matplotlib.pyplot as plt
import numpy as np
import control as ctrl

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


ki_value = 1
kd_value = 0
kp_value = 1

G = ctrl.TransferFunction([1], [1, 1])

t = np.linspace(0, 10, 1000)

plt.figure(figsize=(10, 6))

# for ki in ki_value:
C = ctrl.TransferFunction([kd_value, kp_value, ki_value], [1, 0])
system = ctrl.feedback(C * G, 1)
t, y = ctrl.step_response(system, T=t)
plt.plot(t, y)

plt.title("影响比例增益 Kp 对系统响应的影响")
plt.xlabel("时间 (s)")
plt.ylabel("系统输出")
plt.grid()
plt.legend()
plt.show()

