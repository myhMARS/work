import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage.io import imread, imsave
import os

image_path = 'img.jpg'  # 替换为你的图像路径
image = imread(image_path)
# imsave('r_image.jpg', image)
image = image / 255.0

# original_shape = image.shape
pixels = image.reshape(-1, 3)
origin = []
zips = []
restructed = []
x = []
for n in range(5, 100, 5):
    x.append(n)
    n_colors = n
    kmeans = KMeans(n_clusters=n_colors, random_state=42)
    kmeans.fit(pixels)
    centers = kmeans.cluster_centers_  # 获取颜色中心
    labels = kmeans.labels_

    # gray_image = np.ones(image.shape[:2], dtype=np.uint8) * 255
    # zip_image = gray_image / (labels + 1).reshape(*gray_image.shape)
    zip_image = labels.reshape(image.shape[:2]).astype(np.uint8)
    print(zip_image.shape)

    reconstructed_image = centers[labels].reshape(image.shape)

    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 3, 1)
    # plt.title("Original Image")
    # plt.imshow(image)
    # plt.axis('off')
    #
    # plt.subplot(1, 3, 2)
    # plt.title(f"Zip Image ({n_colors} colors)")
    # plt.imshow(zip_image, cmap='gray')
    # plt.axis('off')
    #
    # plt.subplot(1, 3, 3)
    # plt.title(f"Reconstructed Image ({n_colors} colors)")
    # plt.imshow(reconstructed_image)
    # plt.axis('off')
    #
    # plt.tight_layout()
    # plt.show()

    zip_image_path = 'zip_image.jpg'
    imsave(zip_image_path, zip_image, as_gray=True)
    compressed_image_path = 'reconstructed_image.jpg'
    imsave(compressed_image_path, (reconstructed_image * 255).astype(np.uint8))
    zips.append(os.stat('zip_image.jpg').st_size)
    restructed.append(os.stat('reconstructed_image.jpg').st_size)
plt.plot(x, [os.stat('img.jpg').st_size] * len(x), label='Original Image Size',linestyle='-')
plt.plot(x, zips, label='Zip Image Size',linestyle='-.')
plt.plot(x, restructed, label='Reconstructed Image Size',linestyle='--')
plt.xlabel('K of kmeans')
plt.ylabel('Size of Image')
plt.legend()
plt.show()
