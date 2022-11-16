import cv2
import numpy as np
from .filter_with_thresholds import auto_thresholding
from .eliminate_holes_and_tiny_objects import eliminate_holes_and_tiny_objects


def get_mask(img_crop, return_type='mask', blur_mode='gaussian_blur', ksize=(7, 7), quantile=0.5):

    assert blur_mode in ['simple_blur', 'gaussian_blur', 'no_blur']
    height, width = img_crop.shape[0], img_crop.shape[1]
    if blur_mode == 'simple_blur':
        image = cv2.blur(img_crop, ksize)
    elif blur_mode == 'gaussian_blur':
        image = cv2.GaussianBlur(img_crop, ksize, 0)
    else:
        image = img_crop

    # @TODO Automated thresholding
    ret, thresh = auto_thresholding(image)
    cleared_mask = np.array(
        eliminate_holes_and_tiny_objects(thresh, width, height, eps=None, return_type=return_type))

    return cleared_mask
