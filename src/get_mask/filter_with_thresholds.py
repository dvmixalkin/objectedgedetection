import cv2
import numpy as np
import rasterio
import shapely
from rasterio import features
from shapely.geometry import Polygon


def auto_thresholding(image_orig, step=17):
    target_square = image_orig.shape[0] * image_orig.shape[1] * 0.9
    mask_candidates = []
    im_mean = image_orig.mean()
    image = image_orig - im_mean
    # image = np.invert(image_mean > 0)

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

    mask_candidates = []
    for threshold in range(nearest_ret - 8, nearest_ret + 8, 2):
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
