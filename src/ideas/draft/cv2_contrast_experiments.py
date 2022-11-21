import cv2
import numpy as np
from PIL import Image


def getStructuringElement():
    npz = np.load('../../../examples/input/20210331STM0104727.npz')['raw_image_low']
    npz = (npz / npz.max()) * 255
    gray = np.array(Image.fromarray(npz).convert('RGB'))

    # img = cv2.imread('Lenna.png')
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    k_size = (11, 11)  # (5, 5)
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, k_size)
    close = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel1)
    div = np.float32(gray) / close
    res = np.uint8(cv2.normalize(div, div, 0, 255, cv2.NORM_MINMAX))
    Image.fromarray(res).show()



def convertScaleAbs():
    img = cv2.imread('b.jpg',0) # loads in grayscale

    alpha = 1
    beta = 0
    res = cv2.multiply(img, alpha)
    res = cv2.add(res, beta)

    res = cv2.convertScaleAbs(img, alpha = alpha, beta = beta)



def get_AdaptiveThreshold():
    im = cv2.LoadImage("9jU1Um.jpg", cv2.CV_LOAD_IMAGE_GRAYSCALE)
    cv2.AdaptiveThreshold(
        im, im, 255, cv2.CV_ADAPTIVE_THRESH_MEAN_C, cv2.CV_THRESH_BINARY, blockSize=31, param1=15
    )


def main():
    getStructuringElement()


if __name__ == '__main__':
    main()
