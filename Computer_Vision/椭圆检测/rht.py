import cv2
import numpy as np


# 数据存储
class Data:
    def __init__(self):
        self.greyScale = None
        self.medianFilter = None
        self.sobel = None
        self.edge = None
        self.histogram = None


data = Data()

# 读取图像
img = cv2.imread("source/e1.PNG")  # 替换为实际路径

# ******************************************
# * GREYSCALE ***
# ******************************************
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
data.greyScale = img_gray

# ******************************************
# * MEDIAN FILTERING ***
# ******************************************
img_median = cv2.medianBlur(img_gray, 3)  # 窗口大小为 3
data.medianFilter = img_median

# ******************************************
# * EDGE DETECT SOBEL ***
# ******************************************
sobel_x = cv2.Sobel(img_median, cv2.CV_64F, 1, 0, ksize=3)  # Sobel 水平方向
sobel_y = cv2.Sobel(img_median, cv2.CV_64F, 0, 1, ksize=3)  # Sobel 垂直方向
sobel = cv2.magnitude(sobel_x, sobel_y)  # 计算梯度幅值
data.sobel = np.uint8(np.clip(sobel, 0, 255))  # 转为 8 位图像

# ******************************************
# * DEL BORDER ***
# ******************************************
mask = np.zeros_like(img_gray, dtype=np.uint8)  # 假设 mask 是一个二值掩码
cv2.rectangle(mask, (10, 10), (mask.shape[1] - 10, mask.shape[0] - 10), 255, -1)  # 示例：创建一个中心区域
img_edge = cv2.bitwise_and(data.sobel, mask)
img_edge = cv2.Canny(img_edge, 30, 100)
data.edge = img_edge

# ******************************************
# * HISTOGRAM ***
# ******************************************
histogram = cv2.calcHist([data.edge], [0], None, [256], [0, 256])  # 灰度图直方图
data.histogram = histogram

# 保存结果（可选）
# cv2.imwrite("output_greyscale.jpg", data.greyScale)
# cv2.imwrite("output_median.jpg", data.medianFilter)
# cv2.imwrite("output_sobel.jpg", data.sobel)
# cv2.imwrite("output_edge.jpg", data.edge)
cv2.imshow("edge", data.edge)
cv2.waitKey(0)
# 可视化直方图（可选）
import matplotlib.pyplot as plt

plt.figure()
plt.title("Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
plt.plot(histogram)
plt.xlim([0, 256])
plt.show()
