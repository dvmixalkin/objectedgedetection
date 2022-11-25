import cv2
import numpy as np

from .eliminate_holes_and_tiny_objects import eliminate_holes_and_tiny_objects, mask2poly, polygon2mask
from .filter_with_thresholds import auto_thresholding_v1, auto_thresholding_v2, auto_thresholding_v3
from ..split_area.search_anomaly.utils.get_np_points_from_polygon import get_np_points_from_polygon

# unused temporal modules
from PIL import Image
from src.ideas.visualize import vis_image, vis_contours, vis_polygon


def dirty_polygon_cleaner(dirty_mask, axis):
    object_mask = dirty_mask > 0

    # binary_filtered = thresh > thresh[object_mask].mean()
    polygons = mask2poly(object_mask)

    # get areas of all polygons
    areas = [polygon.area for polygon in polygons]

    # calculate threshold for polygon areas
    threshold = np.quantile(np.array(areas), 0.85) if axis==1 else np.array(areas).mean()/2

    # get polygon indexes with area above threshold
    idxs = [idx for idx, area in enumerate(areas) if area >= threshold]

    # get filtered polygons
    filtered_polygons = [polygons[i] for i in idxs]

    # convert back polygons to mask
    res_mask = polygon2mask(filtered_polygons, dirty_mask.shape)
    return res_mask


def clean_mask_along_axis(mask, axis):
    mask_copy = mask.copy()

    # count active pixel num in every row of mask
    full_lines_mask = (mask_copy > 0).sum(axis=axis)

    # set zeros to rows with num pixels above threshold
    if axis == 0:
        # compute threshold for pixel num in column
        indexes = full_lines_mask < full_lines_mask.mean() / 2
        mask_copy[:, indexes] = 0
    else:
        # compute threshold for pixel num in row
        indexes = full_lines_mask > (np.quantile(full_lines_mask, 0.975) + np.median(full_lines_mask)) / 2
        mask_copy[indexes, :] = 0
    # convert cleared mask to polygon
    polygons = mask2poly(mask_copy)

    # get areas of all polygons
    areas = [polygon.area for polygon in polygons]

    # calculate threshold for polygon areas
    threshold = np.mean(np.array(areas))

    # get polygon indexes with area above threshold
    idxs = [idx for idx, area in enumerate(areas) if area > threshold]

    # get filtered polygons
    filtered_polygons = [polygons[i] for i in idxs]

    # convert back polygons to mask
    dirty_mask = polygon2mask(filtered_polygons, mask.shape)
    cleaned_mask = dirty_polygon_cleaner(dirty_mask, axis)
    return cleaned_mask


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
    mask = clean_mask_along_axis(mask, axis=1)

    # ============================== remove vertical artifacts ===================================================== #
    mask = clean_mask_along_axis(mask, axis=0)

    coordinates = np.hstack([get_np_points_from_polygon(poly) for poly in mask2poly(mask)])
    x_min = coordinates[..., 0].min()
    y_min = coordinates[..., 1].min()
    x_max = coordinates[..., 0].max()
    y_max = coordinates[..., 1].max()
    coordinates = [x_min ,y_min, x_max, y_max]
    return mask, coordinates


def remove_platform(thresh, pad, original=True):
    h, w = thresh.shape[0], thresh.shape[1]

    if not original:
        thresh, coordinates = manual_statistical_removing(thresh)

    x_min = pad if original else min(pad, coordinates[0])
    y_min = pad if original else min(pad, coordinates[1])
    x_max = w-pad if original else max(w-pad, coordinates[2])
    y_max = h-pad if original else max(h-pad, coordinates[3])

    wo_pad = np.zeros_like(thresh)
    wo_pad[y_min:y_max, x_min:x_max] = thresh[y_min:y_max, x_min:x_max]
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

    # # to smooth lone pixels
    # thresh = cv2.GaussianBlur(thresh.astype(np.uint8), (15, 15), 0) > 0
    # remove holes from mask
    cleared_mask = eliminate_holes_and_tiny_objects(
        thresh, width, height, eps=None, return_type=area_type, store_single=False
    )
    # Image.fromarray(cleared_mask).show()
    return cleared_mask
