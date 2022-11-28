import numpy as np
from .utils import search_threshold


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
