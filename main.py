# Python program to read
# image using PIL module

# importing PIL
from PIL import Image
import matplotlib.pyplot as plt

masks = (1, 2, 4, 8, 16, 32, 64, 128)

img = Image.open("res/dog.jpeg")
gray_img = img.convert("L")
gray_img.save('res/grey.png', 'PNG')

imgdata = list(gray_img.getdata())
print(imgdata)

img_data_len = len(imgdata)

for j in range(len(masks)):
    pixels = imgdata[:]
    for i in range(img_data_len):
        print(f"Bit:{pixels[i]}, Mask:{masks[j]}")
        pixels[i] &= masks[j]
        print(f"Bit dopo:{pixels[i]}")

    gray_img.putdata(pixels)
    gray_img.save('res/out'+str(j+1)+'.png', 'PNG')

    print(img.format)
    print(img.mode)
