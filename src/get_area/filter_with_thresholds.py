import cv2
import matplotlib.pyplot as plt
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


def dual_plot(image, mask):
    cmap_val = 'gray'
    from matplotlib import pyplot as plt
    x, bins = np.histogram(image, bins=256)

    fig = plt.figure(figsize=(15, 15))
    # nrows, ncols, index
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.axis("off")
    ax1.title.set_text('Image')

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.axis("off")
    ax2.title.set_text("Mask")

    ax1.imshow(image, cmap=cmap_val)
    ax2.imshow(mask, cmap=cmap_val)


def triple_plot(image, mask1, mask2):
    cmap_val = 'gray'
    from matplotlib import pyplot as plt
    x, bins = np.histogram(image, bins=256)

    fig = plt.figure(figsize=(15, 15))
    # nrows, ncols, index
    ax1 = fig.add_subplot(3, 1, 1)
    ax1.axis("off")
    ax1.title.set_text('Image')

    ax2 = fig.add_subplot(3, 1, 2)
    ax2.axis("off")
    ax2.title.set_text("Hard_threshold")

    ax3 = fig.add_subplot(3, 1, 3)
    ax3.axis("off")
    ax3.title.set_text("Soft_threshold")

    ax1.imshow(image, cmap=cmap_val)
    ax2.imshow(mask1, cmap=cmap_val)
    ax3.imshow(mask2, cmap=cmap_val)


def auto_thresholding_v3(image_orig: np.ndarray):

    # get pixel value distribution
    x, bins = np.histogram(image_orig, bins=256)
    from PIL import Image
    from matplotlib import pyplot as plt
    plt.stairs(x, bins)

    def find_threshold(image_orig):
        x, bins = np.histogram(image_orig, bins=255)
        #  Алгоритм нахождени я индексов бинов на краях с мксимальными значениями

        divider = 5
        interval_stats = []
        for element in range((len(x) // divider)):
            interval_stats.append([element * divider, (element + 1) * divider, x[element * divider:(element + 1) * divider].sum()])

        np_interval_stats = np.array(interval_stats)
        peaks = np_interval_stats[np_interval_stats[:, 2] > np.quantile(np_interval_stats[:, 2], 0.8)]

        rupture_values = []
        for idx in range(peaks.shape[0]-1):
            rupture_values.append(peaks[idx+1, 0] - peaks[idx, 1])
        divisors_point = rupture_values.index(max(rupture_values)) + 1
        up_side, bot_side = peaks[:divisors_point], peaks[divisors_point:]

        up_filtered = np_interval_stats[:, 0] >= up_side[-1, 1]
        bot_filtered = np_interval_stats[:, 1] <= bot_side[0, 0]
        intersection = up_filtered * bot_filtered
        to_search_minimum = np_interval_stats[intersection]
        min_value_index = np.argmin(to_search_minimum[:, 2])
        min_value = to_search_minimum[min_value_index]

        hard_threshold = up_side[-1, 1]
        soft_threshold = min_value[1]
        return [hard_threshold, soft_threshold]

    hard_threshold, soft_threshold = find_threshold(image_orig)

    ret_h, thresh_h = cv2.threshold(image_orig, hard_threshold, 255, cv2.THRESH_BINARY_INV)
    ret_s, thresh_s = cv2.threshold(image_orig, soft_threshold, 255, cv2.THRESH_BINARY_INV)

    dual_plot(image_orig, image_orig<5)
    from src.get_data import equalize_this_v2
    dual_plot(image_orig,
              equalize_this_v2(image_file=equalize_this_v2(image_file=image_orig, gray_scale=True), gray_scale=True))
    pass
