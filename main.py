import sys
import time
import random
import cv2
import numpy as np
from matplotlib import pyplot as plt


def convolution(image, kernel):
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

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

            convolved_image[i][j] = sum

    return convolved_image


def blur(image, blur_kernel=None):
    if blur_kernel is None:
        blur_kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        norm_value = np.sum(blur_kernel)
        blur_kernel = blur_kernel * (1 / norm_value)
    convolved_image = convolution(image, blur_kernel)
    return convolved_image


def blur2(image):
    blur_kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
    norm_value = np.sum(blur_kernel)
    blur_kernel = blur_kernel * (1 / norm_value)
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


def sobel(image):
    kernel_vertical = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    kernel_orizontal = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    horizontal = convolution(image, kernel_orizontal)
    vertical = convolution(image, kernel_vertical)
    total = horizontal + vertical
    return horizontal, vertical, total


def roberts(image):
    kernel_1 = np.array([[1, 0], [0, -1]])
    kernel_2 = np.array([[0, 1], [-1, 0]])

    gx = convolution(image, kernel_1)
    gy = convolution(image, kernel_2)
    return gx, gy


def fft(image):
    fft_image = np.fft.fft2(image)

    # Sposto l'origine al centro dell'immagine
    fft_shifted = np.fft.fftshift(fft_image)
    magnitude_spectrum = np.log(np.abs(fft_shifted) + 1)

    # Visualizza l'immagine originale e il suo spettro di frequenz
    return magnitude_spectrum


def median(image, n, m):
    image_height, image_width = image.shape

    pad_height = n // 2
    pad_width = m // 2

    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')

    # Setta a 0 tutti i valori dell'array dell'immagine
    median_image = np.zeros_like(image)

    for i in range(image_height):
        for j in range(image_width):
            # Le regioni estratte sono della stessa grandezza del kernel da applicare
            image_region = padded_image[i:i + n, j:j + m]

            # Applica il kernel alla regione dell'immagine
            flattened = np.ravel(image_region)

            # Calcolare il valore mediano
            median_image[i][j] = np.median(flattened)
    return median_image


def laplace(image):
    kernel1 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    kernel2 = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
    return convolution(image, kernel1), convolution(image, kernel2)


def histogram(image):
    original_histogram = cv2.calcHist([image], [0], None, [256], [0, 256])

    # Immagine equalizzata
    equalized_image = cv2.equalizeHist(image)

    # Istogramma equalizzato
    equalized_histogram = cv2.calcHist([equalized_image], [0], None, [256], [0, 256])

    # Visualizza l'immagine e l'istogramma
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Immagine originale')

    plt.subplot(2, 2, 2)
    plt.bar(np.arange(256), original_histogram[:, 0], color='black', width=1.0)
    plt.title('Istogramma originale')
    plt.xlabel('Livello di grigio')
    plt.ylabel('Frequenza')

    plt.subplot(2, 2, 3)
    plt.imshow(equalized_image, cmap='gray')
    plt.title('Immagine equalizzata')

    plt.subplot(2, 2, 4)
    plt.bar(np.arange(256), equalized_histogram[:, 0], color='black', width=1.0)
    plt.title('Istogramma equalizzato')
    plt.xlabel('Livello di grigio')
    plt.ylabel('Frequenza')

    plt.tight_layout()
    plt.show()


def salt_pepper_noise(image):
    noisy_image = np.zeros_like(image)

    image_height, image_width = image.shape
    for i in range(image_height):
        for j in range(image_width):
            if random.randint(0, 20) == 0:
                noisy_image[i][j] = 0
            elif random.randint(0, 20) == 0:
                noisy_image[i][j] = 255
            else:
                noisy_image[i][j] = image[i][j]
    return noisy_image


def bit_slice(image):
    masks = (1, 2, 4, 8, 16, 32, 64, 128)
    height, width = image.shape
    images = []

    for k in range(len(masks)):
        sliced_image = np.zeros_like(image)
        for i in range(height):
            for j in range(width):
                sliced_image[i][j] = image[i][j] & masks[k]
        images.append(sliced_image)

    # Utilizza la somma delle ultime 4 componenti (8, 7, 6, 5)
    # Comprime l'immagine usando solo 4 bit, la metà rispetto all'originale
    image_compressed = sum([i for i in images[-4:]])
    images.append(image_compressed)
    fig, axs = plt.subplots(3, 3, figsize=(12, 12))

    for i in range(3):
        for j in range(3):
            axs[i, j].imshow(images[i * 3 + j], cmap='gray')
            axs[i, j].axis('off')
            if i == 2 and j == 2:
                axs[i, j].set_title(f'Somma dai 4 livelli più significativi')
            else:
                axs[i, j].set_title(f'Bit {i * 3 + j + 1}')
    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.title('Immagine originale')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    path = input("Inserisci percorso file: ")

    while True:
        image = cv2.imread("./images/" + path, cv2.IMREAD_GRAYSCALE)
        image = np.array(image)
        if image is not None:
            break

    height, width = image.shape
    print(f"Size: {height}x{width}")
    value = int(input("1. Applica sfocatura con filtro box.\n"
                      "2. Applica sfocatura gaussiana.\n"
                      "3. Sharpen image\n"
                      "4. Estrazione dei contorni con gradiente di Sobel\n"
                      "5. Estrazione dei contorni con gradiente di Roberts\n"
                      "6. Fast Fourier Transform\n"
                      "7. Denoising di un'immagine con filtro mediano\n"
                      "8. Estrazione dei contorni con filtro laplaciano\n"
                      "9. Istrogramma\n"
                      "10. Rumore sale e pepe\n"
                      "11. Bit Slicing Image\n"
                      "12. Altro\n\n"))
    match value:
        case 1:
            start_time = time.time()
            blurred_image = blur(image)
            end_time = time.time()
            print("Time: ", end_time - start_time, "s")
            cv2.imshow("Immagine in input: ", image)
            cv2.imshow("Immagine sfocata con filtro box: ", blurred_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            sys.exit()
        case 2:
            start_time = time.time()
            blurred_image = blur2(image)
            end_time = time.time()
            print("Time: ", end_time - start_time, "s")
            cv2.imshow("Immagine in input: ", image)
            cv2.imshow("Immagine sfocata con filtro gaussiano: ", blurred_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            sys.exit()
        case 3:
            k = int(input("Inserisci coefficiente di sharpening: "))
            print("Caricamento...")
            start_time = time.time()
            sharpened_image = sharp(image, k)
            end_time = time.time()
            print("Time: ", end_time - start_time, "s")
            cv2.imshow("Immagine in input: ", image)
            cv2.imshow(f"Immagine sharpened {k} volte: ", sharpened_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            sys.exit()
        case 4:
            start_time = time.time()
            horizontal, vertical, total = sobel(image)
            end_time = time.time()
            print("Time: ", end_time - start_time, "s")
            cv2.imshow("Bordi orizzontali: ", horizontal)
            cv2.imshow("Bordi verticali: ", vertical)
            cv2.imshow("Bordi totali: ", total)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            sys.exit()
        case 5:
            start_time = time.time()
            gx, gy = roberts(image)
            end_time = time.time()
            print("Time: ", end_time - start_time, "s")
            cv2.imshow("G_x: ", gx)
            cv2.imshow("G_y: ", gy)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            sys.exit()
        case 6:
            start_time = time.time()
            spectrum = fft(image)
            end_time = time.time()
            print("Time: ", end_time - start_time, "s")
            plt.subplot(121), plt.imshow(image, cmap='gray')
            plt.title('Immagine originale'), plt.xticks([]), plt.yticks([])
            plt.subplot(122), plt.imshow(spectrum, cmap='gray')
            plt.title('Spettro di frequenza'), plt.xticks([]), plt.yticks([])
            plt.show()
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            sys.exit()
        case 7:
            start_time = time.time()
            median_image = median(image, 15, 15)
            end_time = time.time()
            print("Time: ", end_time - start_time, "s")
            cv2.imshow("Immagine in input: ", image)
            cv2.imshow("Immagine con filtro mediano: ", median_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            sys.exit()
        case 8:
            start_time = time.time()
            laplace_image1, laplace_image2 = laplace(image)
            end_time = time.time()
            print("Time: ", end_time - start_time, "s")
            cv2.imshow("Immagine in input: ", image)
            cv2.imshow("Immagine con filtro laplaciano 1: ", laplace_image1)
            cv2.imshow("Immagine con filtro laplaciano 2: ", laplace_image2)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            sys.exit()
        case 9:
            histogram(image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            sys.exit()
        case 10:
            start_time = time.time()
            noisy_image = salt_pepper_noise(image)
            denoised_image = median(image, 3, 3)
            end_time = time.time()
            print(f"Time: {end_time - start_time}s")
            cv2.imshow("Immagine in input: ", image)
            cv2.imshow("Immagine con filtro sale e pepe: ", noisy_image)
            cv2.imshow("Immagine con rumore rimosso: ", denoised_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            sys.exit()
        case 11:
            start_time = time.time()
            bit_slice(image)
            end_time = time.time()
            print(f"Time: {end_time - start_time}s")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            sys.exit()
        case _:
            pass

    rows = int(input("Inserisci il numero di righe: "))
    kernel = []
    for i in range(rows):
        row = input(f"Inserisci i valori della riga {i}: ").split(" ")
        row = list(map(int, row))
        kernel.append(row)

    kernel = np.array(kernel)
    kernel = kernel * (1 / np.sum(kernel))
    cv2.imshow("Immagine risultante: ", blur(image, kernel))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
