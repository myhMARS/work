import yfinance as yf
import numpy as np

# Step 1: 获取股票历史数据
# 使用 yfinance 获取指定股票（如苹果公司 AAPL）的历史数据。
# 下载的字段包括 Open, High, Low, Close, Adj Close（调整收盘价）和 Volume（成交量）。
stock_symbol = 'AAPL'  # 可以更换为其他股票代码
data = yf.download(stock_symbol, start='2024-01-01', end='2024-12-05')

# Step 2: 计算每日涨跌幅
# 使用 `pct_change` 函数计算调整收盘价的日收益率（涨跌幅）。
# Returns 表示股票每日涨跌幅，值为 (今日收盘价 - 昨日收盘价) / 昨日收盘价。
data['Returns'] = data['Adj Close'].pct_change()


# Step 3: 状态离散化
# 定义一个函数，将涨跌幅（连续值）离散化为 3 个状态：
# -1: 下跌 (Returns < 0), 0: 横盘 (Returns = 0), 1: 上涨 (Returns > 0)。
def discretize_returns(x):
    if x < 0:
        return -1  # 下跌
    elif x > 0:
        return 1  # 上涨
    else:
        return 0  # 横盘


# 将每个交易日的涨跌幅映射到对应的状态
data['State'] = data['Returns'].apply(discretize_returns)

# Step 4: 构建状态转移矩阵
# 定义 3 个可能的状态：-1, 0, 1
states = [-1, 0, 1]
# 初始化一个 3x3 的转移矩阵，存储状态间的转移次数
transition_matrix = np.zeros((3, 3))

# 遍历涨跌状态序列，统计相邻状态之间的转移次数
for i in range(1, len(data)):
    current_state = data['State'].iloc[i - 1]  # 当前状态
    next_state = data['State'].iloc[i]  # 下一状态
    # 将状态映射到矩阵索引，例如 -1 -> 0, 0 -> 1, 1 -> 2
    transition_matrix[current_state + 1, next_state + 1] += 1

# 将转移次数矩阵转换为概率矩阵
# 每一行表示当前状态，从该状态转移到其他状态的概率分布
transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)

# 输出转移概率矩阵
print("Transition Matrix:")
print(transition_matrix)

# Step 5: 基于当前状态预测未来状态
# 获取当前状态
current_state = data['State'].iloc[-1]  # 当前状态
current_state_index = current_state + 1  # 将状态映射到矩阵索引

# 假设预测未来 2 天的状态
num_days = 2
predicted_states = [current_state]  # 初始化预测状态序列

# 使用转移概率矩阵进行预测
for _ in range(num_days):
    # 根据当前状态的概率分布随机选择下一状态
    next_state_prob = transition_matrix[current_state_index]
    next_state = np.random.choice(states, p=next_state_prob)  # 按概率选择
    predicted_states.append(next_state)
    # 更新当前状态索引
    current_state_index = next_state + 1

# 输出预测结果
print(f"\nPredicted states for the next {num_days} days: {predicted_states[1:]}")
