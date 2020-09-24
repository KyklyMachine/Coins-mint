import cv2
import numpy as np
import cv2 as cv
import sys
from PIL import Image

def viewImage(image, name_of_window):                   #Т
    cv2.namedWindow(name_of_window, cv2.WINDOW_NORMAL)  #Е
    cv2.imshow(name_of_window, image)                   #С
    cv2.waitKey(0)                                      #Т
    cv2.destroyAllWindows()                             #Ы

image = cv2.imread("IMG.jpg")


def Image_To_Binary(image):
    ######## Обрезка входного изображения ########
    image = cv.GaussianBlur(image, (3, 3), 0)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    kernel_size = 3
    ddepth = cv.CV_16S
    dst = cv.Laplacian(gray, ddepth, ksize=kernel_size)

    th, threshed = cv2.threshold(gray, 90, 160, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    morphed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel)
    cnts = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    cnt = sorted(cnts, key=cv2.contourArea)[-1]
    x,y,w,h = cv2.boundingRect(cnt)
    image = image[y:y+h, x:x+w]

    ######## Перевод изображения в бинарный вид ########
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, threshold_image = cv2.threshold(image, 35, 200, 4) # Подходит и значение "1" -- НАДО ВЫБРАТЬ!
    hsv_min = np.array((0, 0, 0), np.uint8)
    hsv_max = np.array((80, 200, 90), np.uint8)
    hsv = cv.cvtColor(threshold_image, cv.COLOR_BGR2HSV )
    thresh = cv.inRange(hsv, hsv_min, hsv_max )
    viewImage(thresh, "2")


Image_To_Binary(image)
