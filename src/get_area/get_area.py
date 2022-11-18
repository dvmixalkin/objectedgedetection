import cv2
import numpy as np
from .eliminate_holes_and_tiny_objects import eliminate_holes_and_tiny_objects


def remove_platform(thresh, pad):
    h, w = thresh.shape[0], thresh.shape[1]
    wo_pad = np.zeros_like(thresh)
    wo_pad[pad:h-pad, pad:w-pad] = thresh[pad:h-pad, pad:w-pad]
    return wo_pad


def get_area(img_crop, area_type='mask', blur_mode='gaussian_blur', ksize=(7, 7), quantile=0.5, pad=50):

    assert blur_mode in ['simple_blur', 'gaussian_blur', 'no_blur']
    height, width = img_crop.shape[0], img_crop.shape[1]
    if blur_mode == 'simple_blur':
        image = cv2.blur(img_crop, ksize)
    elif blur_mode == 'gaussian_blur':
        image = cv2.GaussianBlur(img_crop, ksize, 0)
    else:
        image = img_crop

    # @TODO Automated thresholding
    # ret1, thresh1 = auto_thresholding(image)
    ret, thresh = cv2.threshold(image, image.mean(), 255, cv2.THRESH_BINARY_INV)

    # check for platform and remove it if exists
    thresh = remove_platform(thresh, pad=pad)

    # remove holes from mask
    cleared_mask = eliminate_holes_and_tiny_objects(
        thresh, width, height, eps=None, return_type=area_type, store_single=False
    )
    # Image.fromarray(cleared_mask).show()
    return cleared_mask
