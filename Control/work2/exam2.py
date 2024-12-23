import numpy as np
import matplotlib.pyplot as plt
import control as ctrl

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 定义参数
S = 5
cases = [
    {"x0": 70, "h": 175, "alpha": 1.3, "Ei": 2500, "Ea": 0},  # 案例1
    {"x0": 70, "h": 175, "alpha": 1.3, "Ei": 2100, "Ea": 0},  # 案例2
    {"x0": 70, "h": 175, "alpha": 1.3, "Ei": 2500, "Ea": 500},  # 案例3
]

# 时间范围
t = np.linspace(0, 3000, 1000)

# 绘图
plt.figure(figsize=(10, 6))

for i, case in enumerate(cases, start=1):
    x0 = case["x0"]
    alpha = case["alpha"]
    h = case["h"]
    Ei = case["Ei"]
    Ea = case["Ea"]

    # 定义传递函数 G(s) = 1 / (7000s + 10 * alpha)
    G = ctrl.TransferFunction([1], [7000, 10 * alpha])

    # 定义扰动 d(t) = -alpha(6.25h - 5a + S)
    d = -alpha * (6.25 * h - 5 * 20 + S)  # 固定参数 S 和年龄 a = 20

    # 定义输入 u(t) = Ei - Ea
    u = Ei - Ea

    # 计算系统输出 x(t)
    x_t = u * ctrl.step_response(G, T=t)[1] + d * ctrl.step_response(G, T=t)[1] + 7000 * x0 * \
          ctrl.impulse_response(G, T=t)[1]

    # 绘制曲线
    plt.plot(t, x_t, label=f"案例{i}")

# 添加标题、坐标轴和图例
plt.title("各案例体重随时间变化曲线")
plt.xlabel("时间 (s)")
plt.ylabel("体重 x(t)")
plt.grid()
plt.legend()
plt.show()
