import cv2
import numpy as np


def convolution(image, kernel):

    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    print(image_height, image_width, kernel_height, kernel_width, pad_height, pad_width)

    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')

    # Setta a 0 tutti i valori dell'array dell'immagine convoluta
    convolved_image = np.zeros_like(image)

    for i in range(image_height):
        for j in range(image_width):

            # Le regioni estratte sono della stessa grandezza del kernel da applicare
            image_region = padded_image[i:i + kernel_height, j:j + kernel_width]

            # Applica il kernel alla regione dell'immagine
            sum = 0
            for k in range(kernel_height):
                for l in range(kernel_width):
                    sum += kernel[k][l] * image_region[k][l]


            # Normalizzo l'immagine
            convolved_image[i][j] = sum * (1 / (kernel_width * kernel_height))

    return convolved_image

image = cv2.imread("dog.jpg", cv2.IMREAD_GRAYSCALE)
height, width = image.shape

print(f"Size: {height}x{width}")

blur_kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])

convolved_image, handmade_conv = convolution(image, blur_kernel)

cv2.imshow("Immagine in input: ", image)
cv2.imshow("Immagine convoluta: ", convolved_image)
cv2.imshow("Immagine convoluta a mano: ", handmade_conv)
cv2.waitKey(0)
cv2.destroyAllWindows()
