import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

import ipdb #dubugger

from utils import (
    generate_histogram_img, 
    resize, 
    convert_to_gray, 
    convert_to_rgb, 
    show_img, 
    convert_to_hsv, 
    histogram,
    equalize,
    filter_median,
    generate_sobel_and_prewitt,
    hist_gray
)

#path = "img/lena.png"

#img = cv.imread(path)

#ipdb.set_trace()

# Redimensionar imagem
#img = cv.imread(path)
#img = resize(img, 800, 600)
#show_img(img, "Imagem Redimensionada (800x600)")




# Conversão para HSV
#img = cv.imread(path)
#img = convert_to_hsv(img)
#show_img(img, "Convertida para HSV")


#Histograma Colorido 
#path = "img/lena.png"
#img = cv.imread(path)
#histogram(img)
#show_img(img, "Convertida para HSV")


#histograma cinza

path = "img/lena.png"
img = cv.imread(path)
hist_gray(img)


"""
#Imagem Equalizada 
path = "img/lena.png"
img = cv.imread(path)
img = convert_to_gray(img)
img_equalized = equalize(img)
show_img(img, "Imagem Equalizada") 
"""

"""
#Filtro Mediano 
path = "img/lena.png"
img = cv.imread(path)
img_filtered = filter_median(img, 5)
show_img(img_filtered, "Imagem Com Filtro Mediano")
show_img(img, "Imagem Original")
"""


""" Filtro Sobel e Prewitt """
path = "img/coins.png"
img = cv.imread(path)
generate_sobel_and_prewitt(img, 3)

#generate_histogram_img(path, "Lena")