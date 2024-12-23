import numpy as np
import matplotlib.pyplot as plt
import control as ctrl

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 定义系统参数
a = 20
alpha = 1.3  # 劳动强度系数
h = 175  # 身高 (cm)
S = 5  # 性别参数
x0 = 90  # 初始体重 (kg)
r_target = 65  # 目标体重 (kg)

G = ctrl.TransferFunction([1], [7000, 10 * alpha])
D = ctrl.TransferFunction([-alpha * (6.25 * h - 5 * a + S)], [1, 0])
R = ctrl.TransferFunction([r_target], [1, 0])


def pid(kp, ki, kd):
    return ctrl.TransferFunction([kd, kp, ki], [1, 0])


t = np.linspace(0, 3000, 1000)
plt.figure(figsize=(10, 6))

# 1. 仅添加比例控制器
kp = 200
ki = 0
kd = 0
C1 = pid(kp, ki, kd)
G_cl1 = ((C1 + D / R + 7000 * x0 / R) * G) / (1 + C1 * G)
t1, x1 = ctrl.step_response(G_cl1, T=t)
plt.plot(t1, x1 * 65, label="仅添加比例控制器 (kp=200, ki=0, kd=0)")

# 2. 添加比例和积分控制器
kp = 0
ki = 1
kd = 0
C2 = pid(kp, ki, kd)
G_cl2 = ((C2 + D / R + 7000 * x0 / R) * G) / (1 + C2 * G)
t2, x2 = ctrl.step_response(G_cl2, T=t)
plt.plot(t2, x2 * 65, label="添加积分控制器 (kp=0, ki=1, kd=0)")

# 3. 添加比例积分微分控制器
kp = 200
ki = 1
kd = 0
C3 = pid(kp, ki, kd)
G_cl3 = ((C3 + D / R + 7000 * x0 / R) * G) / (1 + C3 * G)
t3, x3 = ctrl.step_response(G_cl3, T=t)
plt.plot(t3, x3 * 65, label="添加比例积分控制器 (kp=200, ki=1, kd=0)")

# 绘制图像
plt.title("不同控制器对体重随时间变化的影响")
plt.xlabel("时间 (s)")
plt.ylabel("体重 (kg)")
plt.legend()
plt.grid()
plt.show()
