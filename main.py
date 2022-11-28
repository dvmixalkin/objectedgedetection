# custom imports
from src.get_data import get_image, get_cropped_image, get_annotation, get_coordinates_to_crop
from src.get_area import get_area
from src.split_area import split_area

# debug imports
from src.ideas.visualize import vis_image, vis_contours, vis_polygon

import numpy as np
import glob
from pathlib import Path
import json
import io


class EdgeDetector:
    def __init__(self, get_crop_coordinate_, get_image_crop_, get_area_, split_area_):
        self.get_crop_coordinate = get_crop_coordinate_
        self.get_image_crop = get_image_crop_
        self.get_area = get_area_
        self.split_area = split_area_

    @staticmethod
    def check_image_for_bytes(image_orig):
        try:
            npz_archive = np.load(io.BytesIO(image_orig))
            image_orig = get_image(npz_archive)
        except:
            pass
        return image_orig

    @staticmethod
    def check_annotation_for_bytes(annotations):
        if isinstance(annotations, bytes):
            annotations = json.loads(annotations)
        return annotations

    @staticmethod
    def upscale_to_world_coordinates(polygons, cropp_coordinates, pad):
        world_coordinates_polygons = []
        for polygon in polygons:
            np_polygon = np.array(polygon)
            np_polygon[:, 0] += cropp_coordinates[0] - pad[0]
            np_polygon[:, 1] += cropp_coordinates[1] - pad[1]
            world_coordinates_polygons.append(np_polygon.tolist())
        return world_coordinates_polygons

    def body(self, image_object, anno_object, anno_format='yolo_output', pad=None):
        # 0) get data and check for correctness
        if pad is None:
            pad = [50, 50, 50, 0]
        image_orig = self.check_image_for_bytes(image_object).astype(np.uint8)
        annotation = self.check_annotation_for_bytes(anno_object)

        # 1) get crop coordinates:
        # 1 - original 2 - per_object
        cropp_coordinates = self.get_crop_coordinate(image_orig, annotation, frmt=anno_format, version=1)

        all_poly_coordinates = []
        for cropp_coordinate in cropp_coordinates:
            # 2) crop image by given coordinates
            cropped_image = self.get_image_crop(image_orig, cropp_coordinate, pad=pad)

            # 3) get mask -> polygon of given image
            polygons = self.get_area(
                cropped_image,
                area_type='polygon',
                blur_mode='gaussian_blur',  # gaussian
                ksize=(7, 7),
                quantile=0.5,
                pad=pad
            )  # vis_polygon(cropped_image, polygons[0])

            # 4) split polygon to goods positions
            polygons = self.split_area(cropped_image, polygons)

            # 5) convert local coordinates to global
            polygons = self.upscale_to_world_coordinates(polygons, cropp_coordinate, pad)
            all_poly_coordinates.extend(polygons)

        vis_contours(image=image_orig, contours=all_poly_coordinates, show_contours=True)
        return json.dumps(polygons), 1  # bytes, 1

    def process(self, image_object, anno_object, anno_format='yolo_output', pad=None):
        if pad is None:
            pad = [50, 50, 50, 0]
        polygons = self.body(image_object, anno_object, anno_format=anno_format, pad=pad)
        # try:
        #     polygons = self.body(self, image_object, anno_object, anno_format=anno_format, pad=pad)
        #     return json.dumps(polygons), 1
        # except:
        #     return None, 0


def get_toy_data(anno_frmt='yolo_output', index=None):  # ['yolo_output', 'preprocessed', 'source_data']
    if anno_frmt == 'yolo_output':
        dataset_path = 'examples/input'
        filenames = [Path(i) for i in glob.glob(dataset_path + '/*.npz')]
    elif anno_frmt == 'preprocessed':
        NotImplemented
    elif anno_frmt == 'source_data':
        # dataset_path = '../datasets/DATA3/NOT_DELETE/all_jsons'
        dataset_path = '../datasets/DATA3'
        filenames = [Path(p) for p in glob.glob(dataset_path + '/**/*.npz')]
    else:
        NotImplemented
    npz_path = str(filenames[index])
    # get image and convert it to 'bytes'
    with open(npz_path, 'rb') as stream:
        image_bytes = stream.read()

    # specify annotation json path
    if anno_frmt == 'yolo_output':
        json_path = npz_path.replace('input', 'output').replace('npz', 'json')
    elif anno_frmt == 'preprocessed':
        raise NotImplemented
    elif anno_frmt == 'source_data':
        json_path = npz_path.replace('.npz', '_predict.json')
    else:
        NotImplemented
    # get annotation and convert it to 'bytes'
    with open(json_path, 'rb') as stream:
        json_data = json.load(stream)
    json_bytes = json.dumps(json_data).encode('utf-8')
    return image_bytes, json_bytes


def main(anno_frmt='yolo_output'):
    # 0) get image index
    index = 2

    # 1) initialize image pool
    image_bytes, json_bytes = get_toy_data(anno_frmt, index)

    # 2) initialize detector class
    contour = EdgeDetector(
        get_crop_coordinate_=get_coordinates_to_crop,
        get_image_crop_=get_cropped_image,
        get_area_=get_area,
        split_area_=split_area
    )

    # 3) loop through images
    contour.process(image_bytes, json_bytes, anno_format=anno_frmt, pad=[50, 50, 50, 0])


if __name__ == '__main__':
    anno_frmt = 'yolo_output'
    # anno_frmt = 'source_data'
    main(anno_frmt)
