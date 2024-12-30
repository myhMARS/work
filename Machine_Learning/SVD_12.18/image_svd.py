import numpy as np
import  matplotlib.pyplot as plt
from PIL import Image


def load_image(image_path):
    image = Image.open(image_path).convert('L')
    return np.array(image)

def svd_decomposition(matrix):
    U, s, Vt = np.linalg.svd(matrix)
    return U, s, Vt

def reconstruct_image(U, s, Vt, num_singular_values):
    s_diag = np.diag(s[:num_singular_values])
    U_approx = U[:, :num_singular_values]
    Vt_approx = Vt[:num_singular_values, :]
    reconstruct_matrix = U_approx @ s_diag @ Vt_approx
    return reconstruct_matrix

def display_images(original_image, reconstructed_image):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original_image, cmap='gray')
    axes[0].set_title('original_image')
    axes[0].axis('off')
    
    axes[1].imshow(reconstructed_image, cmap='gray')
    axes[1].set_title('reconstruct_image')
    axes[1].axis('off')

    plt.show()


if __name__ == '__main__':
    image_path = 'Lena.bmp'
    original_image = load_image(image_path)
    U, s, Vt = svd_decomposition(original_image)
    num_singular_values = 50
    reconstructed_image = reconstruct_image(U,s,Vt,num_singular_values)
    display_images(
        original_image,
        reconstructed_image
    )
