import cv2
import matplotlib.pyplot as plt

# 读取输入图像
image_path = "img.jpg"
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 1. Sobel 算子边缘检测
sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)  # x方向梯度
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)  # y方向梯度
sobel_combined = cv2.magnitude(sobel_x, sobel_y)     # 组合梯度

# 2. Canny 算子边缘检测
canny_edges = cv2.Canny(img, threshold1=50, threshold2=150)


plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(sobel_combined, cmap='gray')
plt.title('Sobel Edge Detection')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(canny_edges, cmap='gray')
plt.title('Canny Edge Detection')
plt.axis('off')

plt.tight_layout()
plt.show()
