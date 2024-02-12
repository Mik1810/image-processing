import sys

import cv2
import numpy as np


def convolution(image, kernel):
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')

    # Setta a 0 tutti i valori dell'array dell'immagine convoluta
    convolved_image = np.zeros_like(image)

    norm_value = np.sum(kernel)

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
            convolved_image[i][j] = sum * (1 / norm_value)

    return convolved_image


def blur(image):
    blur_kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    convolved_image = convolution(image, blur_kernel)
    return convolved_image


def blur2(image):
    blur_kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
    convolved_image = convolution(image, blur_kernel)
    return convolved_image


def sharp(image, k):
    if k < 1:
        print("Coefficiente di sharpening troppo piccolo!")
        sys.exit()

    for _ in range(k):
        blurred_image = blur(image)
        diff_image = np.zeros_like(image)
        sharpened_image = np.zeros_like(image)

        diff_image = image - blurred_image
        image = image + diff_image

    return image


if __name__ == "__main__":
    path = input("Inserisci percorso file: ")

    while True:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        image = np.array(image)
        if image is not None:
            break

    height, width = image.shape
    print(f"Size: {height}x{width}")
    value = int(input("1. Applica sfocatura.\n2. Applica sfocatura gaussiana.\n3. Sharpen image.\n4. Altro\n\n"))
    if value == 1:
        blurred_image = blur(image)
        cv2.imshow("Immagine in input: ", image)
        cv2.imshow("Immagine sfocata con filtro box: ", blurred_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        sys.exit()
    if value == 2:
        blurred_image = blur2(image)
        cv2.imshow("Immagine in input: ", image)
        cv2.imshow("Immagine sfocata con filtro gaussiano: ", blurred_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        sys.exit()
    if value == 3:
        k = int(input("Inserisci coefficiente di sharpening: "))
        print("Caricamento...")
        sharpened_image = sharp(image, k)
        cv2.imshow("Immagine in input: ", image)
        cv2.imshow(f"Immagine sharpened {k} volte: ", sharpened_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        sys.exit()

    rows = int(input("Inserisci il numero di righe: "))
    kernel = []
    for i in range(rows):
        row = input(f"Inserisci i valori della riga {i}: ").split(" ")
        row = list(map(int, row))
        kernel.append(row)

    kernel = np.array(kernel)
    print(kernel)
