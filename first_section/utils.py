
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def generate_histogram_img(path, text):
    
    img = cv.imread(path)
    
    show_img(img, text)
    
    plt.hist(img.ravel(),256,[0,256])
    
    plt.show()
    cv.waitKey(0)

def histogram(img):
    blue_color = cv.calcHist([img], [0], None, [256], [0,256])
    red_color = cv.calcHist([img], [1], None, [256], [0,256])
    green_color = cv.calcHist([img], [2], None, [256], [0,256])
    
    
    plt.subplot(3, 1, 1)
    plt.title("Histograma do Azul")
    plt.hist(blue_color, color="blue")
    
    plt.subplot(3, 1, 2)
    plt.title("histograma do Verde")
    plt.hist(green_color, color="green")
    
    plt.subplot(3, 1, 3)
    plt.title("histograma do Vermelho")
    plt.hist(red_color, color="red")
    
    plt.tight_layout()
    plt.show()
    
    # histograma Colorido
    plt.title("Histogram Colorido")
    plt.hist(blue_color, color="blue")
    plt.hist(green_color, color="green")
    plt.hist(red_color, color="red")
    
    plt.show()
    
def generate_sobel_and_prewitt(img, kernel_size):
    img = convert_to_gray(img)
    
    # Sobel
    sobel_x = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=kernel_size)
    sobel_y = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=kernel_size)
    
    plt.subplot(2, 2, 3)
    plt.imshow(np.abs(sobel_x) + np.abs(sobel_y), cmap="gray")
    plt.title("Filtro Sobel")
    
    # Prewitt
    
    prewitt_x = cv.filter2D(img, cv.CV_64F, np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]))
    prewitt_y = cv.filter2D(img, cv.CV_64F, np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]))
    
    plt.subplot(2, 2, 4)
    plt.imshow(np.abs(prewitt_x) + np.abs(prewitt_y), cmap="gray")
    plt.title("Filtro Prewitt")
    
    plt.show()

def convert_to_gray(img):
    return cv.cvtColor(img, cv.COLOR_RGB2GRAY)

def convert_to_rgb(img):
    return cv.cvtColor(img, cv.COLOR_BGR2RGB)

def convert_to_hsv(img):
    return cv.cvtColor(img, cv.COLOR_BGR2HSV)

def resize(img, width, height):
    return cv.resize(img, (width, height))

def equalize(img):
    return cv.equalizeHist(img)


def hist_gray(img):
    img_gray = convert_to_gray(img)
    histogram = cv.calcHist([img_gray], [0], None, [256], [0,256])
    
    plt.plot(histogram, color="gray")


def filter_median(img, kernel_size):
    return cv.medianBlur(img, kernel_size)


def show_img(img, text):
    cv.imshow(text, img)
    cv.waitKey(0)
    cv.destroyAllWindows()