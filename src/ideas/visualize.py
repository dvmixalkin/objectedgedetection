import cv2
import numpy as np
import os

import shapely.geometry
from PIL import Image
import random
from src.split_area.search_anomaly.utils.get_np_points_from_polygon import get_np_points_from_polygon


def vis_image(npz_array):
    image = (npz_array / npz_array.max()) * 255
    Image.fromarray(image.astype(np.uint8)).show()


def vis_polygon(image, polygon, is_closed=True,
                x_bounds_coordinates=None,
                y_bounds_coordinates=None,
                hole_point_coordinate=None,
                color=(255, 255, 153),
                thickness=2):
    height, width = image.shape
    if len(image.shape) == 2:
        img = (np.stack([image, image, image]).transpose(1, 2, 0)).astype(np.uint8)
    elif len(image.shape) == 3:
        img = image
    else:
        raise NotImplemented

    im = img.copy()

    im = cv2.polylines(
            img=im.astype(np.uint8),
            pts=[np.asarray(polygon)],
            isClosed=is_closed,
            color=color,
            thickness=thickness)
    color_start = (0, 255, 0)
    color_end = (255, 0, 0)
    if hole_point_coordinate is not None:
        im = cv2.circle(im, hole_point_coordinate, radius=15, color=(255, 0, 0), thickness=-1)

    if x_bounds_coordinates is not None:
        if x_bounds_coordinates[0] is not None:
            start_point = (x_bounds_coordinates[0], 0)
            end_point = (x_bounds_coordinates[0], height)
            im = cv2.line(im, start_point, end_point, color_start, thickness)
        if x_bounds_coordinates[1] is not None:
            start_point = (x_bounds_coordinates[1], 0)
            end_point = (x_bounds_coordinates[1], height)
            im = cv2.line(im, start_point, end_point, color_end, thickness)
    if y_bounds_coordinates is not None:
        if y_bounds_coordinates[0] is not None:
            start_point = (0, y_bounds_coordinates[0])
            end_point = (width, y_bounds_coordinates[0])
            im = cv2.line(im, start_point, end_point, color_start, thickness)
        if y_bounds_coordinates[1] is not None:
            start_point = (0, y_bounds_coordinates[1])
            end_point = (width, y_bounds_coordinates[1])
            im = cv2.line(im, start_point, end_point, color_end, thickness)
    Image.fromarray(im).show()


def vis_contours(image, contours, show_contours=True, save_path=None):
    if len(image.shape) == 2:
        img = (np.stack([image, image, image]).transpose(1, 2, 0)).astype(np.uint8)
    elif len(image.shape) == 3:
        img = image
    else:
        raise NotImplemented

    im = img.copy()
    for idx, r in enumerate(contours):
        if isinstance(r, shapely.geometry.Polygon):
            r = get_np_points_from_polygon(r)
        if not isinstance(r, np.ndarray):
            r = np.asarray(r)

        im = cv2.polylines(
            img=im,
            pts=[r],
            isClosed=True,
            color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
            thickness=5)
    if show_contours:
        Image.fromarray(im).show()
    if save_path is not None:
        if isinstance(save_path, tuple):
            dst_folder = save_path[0]
            image_path = save_path[1]

            if not os.path.exists(dst_folder):
                os.makedirs(dst_folder, exist_ok=True)

            save_path = os.path.join(dst_folder, image_path)
        Image.fromarray(im).save(save_path)
