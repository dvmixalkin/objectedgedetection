import numpy as np
from .utils import search_threshold


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
