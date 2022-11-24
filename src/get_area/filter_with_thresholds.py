import cv2
import numpy as np
import rasterio
import shapely
from rasterio import features
from shapely.geometry import Polygon


def search_threshold(image: np.ndarray, step: int, start_position: int = 0, finish_position: int = 256) -> list:
    """
    Atomic structure

    Algo is looking for best threshold value to make segmentation on image and get the closest mask;

    :param image: image to process through OpenCV filters
    :param step: iterator step to increase threshold value
    :param start_position: if is not None - used to reduce range of values of thresholds
    :param finish_position: if is not None - used to reduce range of values of thresholds
    :return: list of triplets with: area, threshold, segmentation mask
    """
    # initial empty list
    mask_candidates = []

    # iter through all values in range(start_position, finish_position)
    for threshold in range(start_position, finish_position, step):

        # thresholding image with given threshold
        ret, thresh = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)

        # set initial area as 0 (basic_area)
        max_area = 0

        # set initial poly as None
        target_poly = None

        # iter through objects, which were taken from current mask
        for shape, value in features.shapes(thresh.astype(np.uint8), mask=(thresh > 0),
                                            transform=rasterio.Affine(1.0, 0, 0, 0, 1.0, 0)):

            # convert to Polygon object
            poly = shapely.geometry.shape(shape)

            # get current Polygon object area and compare it with 'basic_area'
            if poly.area > max_area:

                # if current polygon area is greater than basic_area -> replace it with current polygon
                target_poly = poly
                max_area = poly.area

        # in case if target_poly is not None --> add this polygon to summary polygon list
        if target_poly is not None:
            mask_candidates.append([target_poly.area, ret, thresh])

    return mask_candidates


def auto_thresholding_v1(image, step=17):
    """
    Atomic structure

    Algo searches the best combination of threshold and ist mask for further;

    :param image: image to process through OpenCV filters
    :param step: iterator step to increase threshold value
    :return: threshold value and segmentation mask
    """
    # init target area to find the closest threshold and get best mask
    target_square = image.shape[0] * image.shape[1] * 0.9

    # get triplets list with candidates for further searching iterations
    mask_candidates = search_threshold(image, step)

    # compute differences in area between target and candidates
    tmp = [abs(triplet[0] - target_square) for triplet in mask_candidates]

    # get the best one's index
    nearest_square_idx = np.argmin(tmp)

    # get the best candidate threshold value
    nearest_ret = int(mask_candidates[nearest_square_idx][1])

    # get refined triplets list with candidates for final searching iterations
    mask_candidates = search_threshold(image, step=2, start_position=nearest_ret - 8, finish_position=nearest_ret + 8)

    # compute differences in area between target and candidates
    tmp = [abs(triplet[0] - target_square) for triplet in mask_candidates]

    # get the best one's index
    nearest_square_idx = np.argmin(tmp)

    # get the best candidate threshold value
    ret = mask_candidates[nearest_square_idx][1]

    # get best candidate segmentation mask
    thresh = mask_candidates[nearest_square_idx][2]
    return ret, thresh


def auto_thresholding_v2(image_orig, start_position=None, step=17, pad=50):

    # init target area to find the closest threshold and get best mask
    target_square = image_orig.shape[0] * image_orig.shape[1] * 0.9

    # primary normalization ops(especially, mean subtraction)
    image = image_orig  # - im_mean

    # if start_position is not defined manually - compute it
    if start_position is None:

        # get triplets list with candidates for further searching iterations
        mask_candidates = search_threshold(image, step)

        # compute differences in area between target and candidates
        tmp = [abs(triplet[0] - target_square) for triplet in mask_candidates]

        # get the best one's index
        nearest_square_idx = np.argmin(tmp)

        # get the best candidate threshold value
        nearest_ret = int(mask_candidates[nearest_square_idx][1])

        # set epsilon value(steps from centroid)
        eps = 8

    # if start_position is defined manually - try to find refined triplets list with candidates
    # for final searching iterations
    else:

        # set manual candidate threshold value
        nearest_ret = int(start_position)

        # set epsilon value(steps from centroid)
        eps = pad

    # get refined triplets list with candidates for final searching iterations
    mask_candidates = search_threshold(
        image, step=2,
        start_position=max(nearest_ret - eps, 0),
        finish_position=min(nearest_ret + eps, 255)
    )

    # compute differences in area between target and refined candidates
    tmp = [abs(triplet[0] - target_square) for triplet in mask_candidates]

    # get the best one's index
    nearest_square_idx = np.argmin(tmp)

    # get the best candidate threshold value
    ret = mask_candidates[nearest_square_idx][1]

    # get best candidate segmentation mask
    thresh = mask_candidates[nearest_square_idx][2]

    return ret, thresh


def auto_thresholding_v3(image_orig: np.ndarray):

    from matplotlib import pyplot
    x, bins = np.histogram(image_orig, bins=256)
    pyplot.stairs(x, bins)
    pass
