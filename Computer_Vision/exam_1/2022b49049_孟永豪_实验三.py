import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: 选择图像数据
image = cv2.imread('Lena.bmp', cv2.IMREAD_GRAYSCALE)  # 使用灰度图像处理
if image is None:
    print("无法加载图像，请确保路径正确。")
    exit()


# 添加噪声的函数
def add_noise(img, noise_type="gaussian"):
    row, col = img.shape
    if noise_type == "gaussian":
        mean = 0
        sigma = 1
        gauss = np.random.normal(mean, sigma, (row, col)).astype('uint8')
        noisy = cv2.add(img, gauss)
    elif noise_type == "salt_pepper":
        noisy = img.copy()
        prob = 0.02
        num_salt = np.ceil(prob * img.size * 0.5).astype(int)
        num_pepper = np.ceil(prob * img.size * 0.5).astype(int)
        coords_salt = [np.random.randint(0, i, num_salt) for i in img.shape]
        coords_pepper = [np.random.randint(0, i, num_pepper) for i in img.shape]
        noisy[coords_salt] = 255
        noisy[coords_pepper] = 0
    else:
        raise ValueError("未知噪声类型")
    return noisy


# 添加噪声
noisy_image = add_noise(image, noise_type="gaussian")

# Step 2: 均值滤波
mean_filtered = cv2.blur(noisy_image, (5, 5))

# Step 3: 高斯滤波
gaussian_filtered = cv2.GaussianBlur(noisy_image, (5, 5), 0)

# Step 4: 中值滤波
median_filtered = cv2.medianBlur(noisy_image, 3)

# 显示结果
titles = ['Original Image', 'Noisy Image', 'Mean Filter', 'Gaussian Filter', 'Median Filter']
images = [image, noisy_image, mean_filtered, gaussian_filtered, median_filtered]

plt.figure(figsize=(12, 8))
for i in range(len(images)):
    plt.subplot(2, 3, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')
plt.tight_layout()
plt.show()
