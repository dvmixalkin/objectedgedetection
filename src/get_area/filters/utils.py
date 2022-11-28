import cv2
import numpy as np
import rasterio
import shapely
from rasterio import features
from shapely.geometry import Polygon
from matplotlib import pyplot as plt


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


def init_plt_image(figure, name, location):
    nrows, ncols, index = location
    axn = figure.add_subplot(nrows, ncols, index)
    axn.axis("off")
    axn.title.set_text(name)
    return axn


def n_plot(images_dict, axis=0, cmap_val='gray'):
    fig = plt.figure(figsize=(15, 15))

    # nrows, ncols, index
    num_items = len(images_dict.items())
    for idx, (key, val) in enumerate(images_dict.items()):
        location = [num_items, 1, idx+1] if axis == 0 else [1, num_items, idx+1]
        init_plt_image(fig, key, location).imshow(val, cmap=cmap_val)


def dual_plot(image, mask):
    image_dict = {
        'Image': image,
        "Mask": mask
    }
    n_plot(image_dict, axis=0, cmap_val='gray')


def triple_plot(image, mask1, mask2):
    image_dict = {
        'Image': image,
        "Mask1": mask1,
        "Mask2": mask2,
    }
    n_plot(image_dict, axis=0, cmap_val='gray')
