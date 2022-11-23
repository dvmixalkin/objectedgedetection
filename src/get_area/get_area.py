import cv2
import numpy as np
from PIL import Image
from .eliminate_holes_and_tiny_objects import eliminate_holes_and_tiny_objects

import rasterio
from rasterio import features
import shapely
from shapely.geometry import Polygon


def auto_thresholding(image_orig, start_position=None, step=17, pad=40):
    target_square = image_orig.shape[0] * image_orig.shape[1] * 0.9
    mask_candidates = []
    im_mean = image_orig.mean()
    image = image_orig  # - im_mean
    # image = np.invert(image_mean > 0)
    if start_position is None:
        for threshold in range(0, 256, step):
            ret, thresh = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)
            max_area = 0
            target_poly = None
            for shape, value in features.shapes(thresh.astype(np.uint8), mask=(thresh > 0),
                                                transform=rasterio.Affine(1.0, 0, 0, 0, 1.0, 0)):
                poly = shapely.geometry.shape(shape)
                if poly.area > max_area:
                    target_poly = poly
                    max_area = poly.area
            if target_poly is not None:
                mask_candidates.append([target_poly.area, ret, thresh])
        tmp = [abs(triplet[0] - target_square) for triplet in mask_candidates]
        nearest_square_idx = np.argmin(tmp)
        nearest_ret = int(mask_candidates[nearest_square_idx][1])
        eps = 8
    else:
        nearest_ret = int(start_position)
        eps = pad
    mask_candidates = []
    for threshold in range(max(nearest_ret - eps, 0), min(nearest_ret + eps, 255), 2):
        ret, thresh = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)
        max_area = 0
        target_poly = None
        for shape, value in features.shapes(thresh.astype(np.uint8), mask=(thresh > 0),
                                            transform=rasterio.Affine(1.0, 0, 0, 0, 1.0, 0)):
            poly = shapely.geometry.shape(shape)
            if poly.area > max_area:
                target_poly = poly
                max_area = poly.area
        if target_poly is not None:
            mask_candidates.append([target_poly.area, ret, thresh])
    tmp = [abs(triplet[0] - target_square) for triplet in mask_candidates]
    nearest_square_idx = np.argmin(tmp)
    ret = mask_candidates[nearest_square_idx][1]
    thresh = mask_candidates[nearest_square_idx][2]
    return ret, thresh


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
        image = cv2.GaussianBlur(img_crop.astype(np.uint8), ksize, 0)
    else:
        image = img_crop

    # @TODO Automated thresholding
    # ret, thresh = auto_thresholding(image, start_position=image.mean())

    # new block
    thresh = np.ones_like(image) * 255
    mask = image < (image.mean() * 2.)
    thresh[mask] = image[mask]
    thresh = np.invert(thresh)
    # Image.fromarray(thresh).show()

    # ret, thresh = cv2.threshold(image.astype(float), image.mean()*0.8, 255, cv2.THRESH_BINARY_INV)
    # check for platform and remove it if exists

    thresh = remove_platform(thresh, pad=pad)
    # to smooth lone pixels
    thresh = cv2.GaussianBlur(thresh.astype(np.uint8), (15,15), 0) > 0
    # remove holes from mask
    cleared_mask = eliminate_holes_and_tiny_objects(
        thresh, width, height, eps=None, return_type=area_type, store_single=False
    )
    # Image.fromarray(cleared_mask).show()
    return cleared_mask
