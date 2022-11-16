from functools import partial
from multiprocessing import Pool
from tqdm import tqdm

# custom imports
from src.get_data import get_image, get_cropped_image, get_annotation, get_coordinates_to_crop
from src.get_mask import get_mask
from src.split_mask import split_mask


class EdgeDetector:
    def __init__(self, get_crop_coordinate_, get_image_crop_, get_mask_, split_mask_):
        self.get_crop_coordinate = get_crop_coordinate_
        self.get_image_crop = get_image_crop_
        self.get_mask = get_mask_
        self.split_mask = split_mask_

    def process(self, image_object, anno_object):
        # get crop coordinates
        cropp_coordinates = self.get_crop_coordinate(anno_object)
        # crop image by given coordinates
        cropped_image = self.get_image_crop(image_object, cropp_coordinates, pad=50)
        # get mask of given image
        mask = self.get_mask(
            cropped_image,
            return_type='mask',
            blur_mode='gaussian_blur',
            ksize=(7, 7),
            quantile=0.5
        )
        # split mask to goods positions
        polygons = self.split_mask(mask)
        return polygons


def main():
    # 1) initialize image pool
    name = '20210331STM0104727'
    npz_path = f'examples/input/{name}.npz'
    np_image = get_image(npz_path)
    # vis_image(np_image)
    json_path = f'examples/output/{name}.json'
    json_data = get_annotation(json_path)

    # 2) initialize detector class
    contour = EdgeDetector(
        get_crop_coordinate_=get_coordinates_to_crop,
        get_image_crop_=get_cropped_image,
        get_mask_=get_mask,
        split_mask_=split_mask
    )

    # 3) loop through images
    polygons = contour.process(np_image, json_data)
    print('Done')


if __name__ == '__main__':
    main()
