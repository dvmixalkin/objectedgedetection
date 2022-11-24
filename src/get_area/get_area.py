import cv2
import numpy as np

from .eliminate_holes_and_tiny_objects import eliminate_holes_and_tiny_objects, mask2poly, polygon2mask
from .filter_with_thresholds import auto_thresholding_v1, auto_thresholding_v2, auto_thresholding_v3

# unused temporal modules
from PIL import Image
from src.ideas.visualize import vis_image, vis_contours, vis_polygon


def manual_statistical_removing(thresh):
    # ============================== remove small polygons =========================================================== #
    # convert to binary mask
    thresh_copy = thresh.copy() > 0

    # convert binary mask to list of polygons
    polygons = mask2poly(thresh_copy)

    # get areas of all polygons
    areas = [polygon.area for polygon in polygons]

    # calculate threshold for polygon areas
    threshold = np.quantile(np.array(areas), 0.9)

    # get polygon indexes with area above threshold
    idxs = [idx for idx, area in enumerate(areas) if area > threshold]

    # get filtered polygons
    filtered_polygons = [polygons[i] for i in idxs]

    # convert back polygons to mask
    mask = polygon2mask(filtered_polygons, thresh.shape)

    # ============================== remove horizontal artifacts ===================================================== #
    mask_copy = mask.copy()

    # count active pixel num in every row of mask
    full_lines_mask = (mask_copy > 0).sum(axis=1)

    # compute threshold for pixel num in row
    indexes = full_lines_mask > (np.quantile(full_lines_mask, 0.975) + np.median(full_lines_mask))/2

    # set zeros to rows with num pixels above threshold
    mask_copy[indexes] = 0

    # convert cleared mask to polygon
    polygons = mask2poly(mask_copy)

    # get areas of all polygons
    areas = [polygon.area for polygon in polygons]

    # calculate threshold for polygon areas
    threshold = np.median(np.array(areas))

    # get polygon indexes with area above threshold
    idxs = [idx for idx, area in enumerate(areas) if area > threshold]

    # get filtered polygons
    filtered_polygons = [polygons[i] for i in idxs]

    # convert back polygons to mask
    cleaned_mask = polygon2mask(filtered_polygons, thresh.shape)

    # ================-> tries to combine cleared mask width <-=========================================================
    object_mask = cleaned_mask > 0
    binary_filtered = thresh > thresh[object_mask].mean()
    polygons = mask2poly(binary_filtered)
    # get areas of all polygons
    areas = [polygon.area for polygon in polygons]

    # calculate threshold for polygon areas
    threshold = np.quantile(np.array(areas), 0.85)

    # get polygon indexes with area above threshold
    idxs = [idx for idx, area in enumerate(areas) if area > threshold]

    # get filtered polygons
    filtered_polygons = [polygons[i] for i in idxs]

    # convert back polygons to mask
    res_mask = polygon2mask(filtered_polygons, thresh.shape)

    return res_mask


def remove_platform(thresh, pad):
    h, w = thresh.shape[0], thresh.shape[1]

    # thresh = manual_statistical_removing(thresh)

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
    # ret, thresh = auto_thresholding_v2(image, start_position=image.mean() * 2)  #
    ret, thresh = auto_thresholding_v3(image)

    # # new block
    # thresh = np.ones_like(image) * 255
    # mask = image < (image.mean() * 2.)
    # thresh[mask] = image[mask]
    # thresh = np.invert(thresh)
    # # Image.fromarray(thresh).show()

    ret, thresh = cv2.threshold(image.astype(float), image.mean()*2., 255, cv2.THRESH_BINARY_INV)
    # check for platform and remove it if exists

    thresh = remove_platform(thresh, pad=pad)

    # to smooth lone pixels
    thresh = cv2.GaussianBlur(thresh.astype(np.uint8), (15, 15), 0) > 0
    # remove holes from mask
    cleared_mask = eliminate_holes_and_tiny_objects(
        thresh, width, height, eps=None, return_type=area_type, store_single=False
    )
    # Image.fromarray(cleared_mask).show()
    return cleared_mask
