import cv2
import numpy as np
import matplotlib.pyplot as plt


def display_histogram(image, bins):
    plt.figure(figure=(10, 5))
    for i, n in enumerate(bins):
        quantized_image = np.floor(image / (256 / n)) * (256 / n)
        quantized_image = quantized_image.astype(np.uint8)

        plt.subplot(1, len(bins), i + 1)
        plt.hist(quantized_image.ravel(), bins=n, range=(0, 256), color='gray')
        plt.title(f'{n} gray level')
    plt.show()


def adjust_grayscale(image, alpha, beta):
    adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted_image


def equalize_histogram(image):
    equalized_image = cv2.equalizeHist(image)
    return equalized_image


if __name__ == '__main__':
    image_path = 'img.jpg'
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    bins = [64, 128, 256]
    display_histogram(image,bins)

    alpha, beta = 1.2, 30
    adjusted_image = adjust_grayscale(image, alpha, beta)
    cv2.imshow('adjust_grayscale',adjusted_image)
    cv2.waitKey(0)

    equalized_image = equalize_histogram(image)
    cv2.imshow('equalize_histogram', equalized_image)
    cv2.waitKey(0)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.hist(image.ravel(), bins=256, range=(0, 256), color='gray')
    plt.title('Origin')
    plt.subplot(1, 2, 2)
    plt.hist(equalized_image.ravel(), bins=256, range=(0, 256), color='gray')
    plt.title('Equalized')
    plt.show()

    cv2.destroyAllWindows()
