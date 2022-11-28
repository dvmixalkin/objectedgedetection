import numpy as np
import cv2
from ..eliminate_holes_and_tiny_objects import (eliminate_holes,
                                                eliminate_tiny_objects,
                                                eliminate_holes_and_tiny_objects, poly2mask)


# @TODO remove imports when complete the work
import random
from PIL import Image


def find_threshold(image_orig, divider=5, quantile=0.8):

    # collect statistic for each pixel value (0-255)
    x, bins = np.histogram(image_orig, bins=255)

    # clamp to dense intervals collected data
    interval_stats = []
    for element in range((len(x) // divider)):
        interval_stats.append(
            [element * divider, (element + 1) * divider, x[element * divider:(element + 1) * divider].sum()])

    # convert to numpy array
    np_interval_stats = np.array(interval_stats)

    # get peaks through filtering with
    peaks = np_interval_stats[np_interval_stats[:, 2] > np.quantile(np_interval_stats[:, 2], quantile)]

    # find ruptures in peak value array
    rupture_values = []
    for idx in range(peaks.shape[0] - 1):
        rupture_values.append(peaks[idx + 1, 0] - peaks[idx, 1])

    # separate this array in rupture point to up_side and bot_side
    divisors_point = rupture_values.index(max(rupture_values)) + 1
    up_side, bot_side = peaks[:divisors_point], peaks[divisors_point:]

    # get array intervals between 2 peaks
    # lower bound of upper side
    up_filtered = np_interval_stats[:, 0] >= up_side[-1, 1]

    # upper bound of lower side
    bot_filtered = np_interval_stats[:, 1] <= bot_side[0, 0]
    # intervals between bounds
    intersection = up_filtered * bot_filtered
    to_search_minimum = np_interval_stats[intersection]

    # find interval index with minimal pixel count
    min_value_index = np.argmin(to_search_minimum[:, 2])

    # get interval with minimal pixel count
    min_value = to_search_minimum[min_value_index]

    # get minimal threshold value
    hard_threshold = up_side[-1, 1]

    # get maximal threshold value
    soft_threshold = min_value[1]
    return [hard_threshold, soft_threshold]


def process_interval(mask, interval):

    # get mask of given interval
    interval_mask = mask[:, interval[0]:interval[1]].astype(bool)

    # get count of active pixels in every row
    stats = interval_mask.sum(axis=1)

    # get average count
    mean = stats.mean()

    # get filtered mask
    mask = stats < mean

    # make zero rows with count lower than mean
    interval_mask[mask] = False
    return interval_mask * 255


def get_divisors(hard_thresh_mask, interval_range=[150, 200]):
    refined_mask = hard_thresh_mask.copy()
    height, width = refined_mask.shape

    # define interval size
    min_interval_width, max_interval_width = interval_range

    # if image width is greater than max_interval_width -> loop root removing
    if width > max_interval_width:

        # set current processed columns
        processed_width = 0

        # loop operation, while processed_width < image width
        while True:

            # get random interval size(in range 150-200)
            interval_width = random.randint(min_interval_width, max_interval_width)

            # set interval range
            current_interval = [processed_width, processed_width + interval_width]

            # increase cumulative variable(processed columns)
            processed_width += interval_width

            # if processed_width is greater than image width -> stop cycle
            if processed_width > width:
                break

            # apply to target image processed interval
            refined_mask[:, current_interval[0]:current_interval[1]] = process_interval(hard_thresh_mask, current_interval)

        # process remaining interval
        last_interval = [processed_width, width]
        refined_mask[:, last_interval[0]:last_interval[1]] = process_interval(hard_thresh_mask, last_interval)
    else:
        NotImplemented

    # remove holes
    wo_holes_thresh_h = eliminate_holes_and_tiny_objects(refined_mask, return_type='mask', store_single=False, eps=200)
    # Image.fromarray(wo_holes_thresh_h).show()

    # get vertical bounds of each cargo place

    return wo_holes_thresh_h


def auto_thresholding_v3(image_orig: np.ndarray, pad: list):

    # # get pixel value distribution
    # from matplotlib import pyplot as plt
    # x, bins = np.histogram(image_orig, bins=255)
    # plt.stairs(x, bins)

    # contour_coordinates = get_contours(image_orig)
    from .utils import n_plot

    try:
        hard_threshold, soft_threshold = find_threshold(image_orig, quantile=0.8)
        ret_h, thresh_h = cv2.threshold(image_orig, hard_threshold, 255, cv2.THRESH_BINARY_INV)
        ret_s, thresh_s = cv2.threshold(image_orig, soft_threshold, 255, cv2.THRESH_BINARY_INV)
        ret = [ret_h, ret_s]
        n_plot(
            images_dict={
                'image_orig': image_orig,
                'thresh_h': thresh_h,
                'thresh_s': thresh_s
            },
            axis=0
        )
    except:
        pass
    try:
        hard_threshold_v1, soft_threshold_v1 = find_threshold(image_orig, quantile=0.7)
        ret_h_v1, thresh_h_v1 = cv2.threshold(image_orig, hard_threshold_v1, 255, cv2.THRESH_BINARY_INV)
        ret_s_v1, thresh_s_v1 = cv2.threshold(image_orig, soft_threshold_v1, 255, cv2.THRESH_BINARY_INV)
        ret_v1 = [ret_h_v1, ret_s_v1]
        n_plot(
            images_dict={
                'image_orig': image_orig,
                'thresh_h_v1': thresh_h_v1,
                'thresh_s_v1': thresh_s_v1
            },
            axis=0
        )
    except:
        pass
    try:
        hard_threshold_v2, soft_threshold_v2 = find_threshold(image_orig, quantile=0.9)
        ret_h_v2, thresh_h_v2 = cv2.threshold(image_orig, hard_threshold_v2, 255, cv2.THRESH_BINARY_INV)
        ret_s_v2, thresh_s_v2 = cv2.threshold(image_orig, soft_threshold_v2, 255, cv2.THRESH_BINARY_INV)
        ret_v2 = [ret_h_v2, ret_s_v2]
        n_plot(
            images_dict={
                'image_orig': image_orig,
                'thresh_h_v2': thresh_h_v2,
                'thresh_s_v2': thresh_s_v2
            },
            axis=0
        )
    except:
        pass

    wo_holes = eliminate_holes(thresh_h_v1, thresh_h_v2.shape, return_type='mask')
    wo_tiny_objects = eliminate_tiny_objects(wo_holes, wo_holes.shape, return_type='polygon')
    thresh = wo_tiny_objects  # get_divisors(thresh_h_v1, interval_range=[50, 100])

    # 2) fill main objects holes
    # thresh = None

    return ret, thresh
