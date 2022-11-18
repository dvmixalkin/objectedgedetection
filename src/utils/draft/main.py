import cv2
import json
import numpy as np
from src.threshold_filter_cotourer.utils import get_image, get_annotation, vis_image
from src.threshold_filter_cotourer.utils import eliminate_holes_and_tiny_objects, vis_contours
from src.utils.draft.contour_refiner.geometry.polygon import MyPolygon
import rasterio
from rasterio import features
import shapely
from pathlib import Path


def auto_thresholding(image_orig, step=17):
    target_square = image_orig.shape[0] * image_orig.shape[1] * 0.9
    mask_candidates = []
    im_mean = image_orig.mean()
    image = image_orig - im_mean
    # image = np.invert(image_mean > 0)

    for threshold in range(0, 256, step):
        ret, thresh = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)
        max_area = 0
        target_poly = None
        for shape, value in features.shapes(thresh.astype(np.uint8), mask=(thresh > 0),
                                            transform=rasterio.Affine(1.0, 0, 0, 0, 1.0, 0)):
            poly = shapely.geometry.shape(shape)
            if poly.area > max_area:
                target_poly = poly
                max_area = poly.area
        if target_poly is not None:
            mask_candidates.append([target_poly.area, ret, thresh])
    tmp = [abs(triplet[0] - target_square) for triplet in mask_candidates]
    nearest_square_idx = np.argmin(tmp)
    nearest_ret = int(mask_candidates[nearest_square_idx][1])

    mask_candidates = []
    for threshold in range(nearest_ret - 8, nearest_ret + 8, 2):
        ret, thresh = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)
        max_area = 0
        target_poly = None
        for shape, value in features.shapes(thresh.astype(np.uint8), mask=(thresh > 0),
                                            transform=rasterio.Affine(1.0, 0, 0, 0, 1.0, 0)):
            poly = shapely.geometry.shape(shape)
            if poly.area > max_area:
                target_poly = poly
                max_area = poly.area
        if target_poly is not None:
            mask_candidates.append([target_poly.area, ret, thresh])
    tmp = [abs(triplet[0] - target_square) for triplet in mask_candidates]
    nearest_square_idx = np.argmin(tmp)
    ret = mask_candidates[nearest_square_idx][1]
    thresh = mask_candidates[nearest_square_idx][2]
    return ret, thresh


def get_object_contour(img_crop, return_type='mask', blur_mode='gaussian_blur', ksize=(7, 7), debug=False, quantile=0.5):
    assert blur_mode in ['simple_blur', 'gaussian_blur', 'no_blur']
    height, width = img_crop.shape
    if blur_mode == 'simple_blur':
        image = cv2.blur(img_crop, ksize)
    elif blur_mode == 'gaussian_blur':
        image = cv2.GaussianBlur(img_crop, ksize, 0)
    else:
        image = img_crop

    # im_mean = image.mean()
    # qaN = np.quantile(image, quantile)
    # image_to_process = (image - qaN).astype(np.uint8)
    #
    # ret, thresh = cv2.threshold(image_to_process,
    #                             int(np.quantile(image_to_process, quantile)),  # 40,  # point_2,
    #                             255,
    #                             cv2.THRESH_BINARY_INV)  # | cv2.THRESH_OTSU
    # img_blurred = cv2.medianBlur(image_to_process, 5)
    # img_blurred = cv2.GaussianBlur(image_to_process, (5, 5), 0)
    # # find normalized_histogram, and its cumulative distribution function
    # hist = cv2.calcHist([img_blurred], [0], None, [256], [0, 256])
    # hist_norm = hist.ravel() / hist.sum()
    # Q = hist_norm.cumsum()
    # bins = np.arange(256)
    # fn_min = np.inf
    # thresh = -1
    # for i in range(1, 256):
    #     p1, p2 = np.hsplit(hist_norm, [i])  # probabilities
    #     q1, q2 = Q[i], Q[255] - Q[i]  # cum sum of classes
    #     if q1 < 1.e-6 or q2 < 1.e-6:
    #         continue
    #     b1, b2 = np.hsplit(bins, [i])  # weights
    #     # finding means and variances
    #     m1, m2 = np.sum(p1 * b1) / q1, np.sum(p2 * b2) / q2
    #     v1, v2 = np.sum(((b1 - m1) ** 2) * p1) / q1, np.sum(((b2 - m2) ** 2) * p2) / q2
    #     # calculates the minimization function
    #     fn = v1 * q1 + v2 * q2
    #     if fn < fn_min:
    #         fn_min = fn
    #         thresh = i
    # # find otsu's threshold value with OpenCV function
    # ret, otsu = cv2.threshold(img_blurred, 0, 255, cv2.THRESH_BINARY_INV)  #  + cv2.THRESH_OTSU
    # th3 = cv2.adaptiveThreshold(img_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # @TODO Automated thresholding
    ret, thresh = auto_thresholding(image)
    cleared_mask = np.array(
        eliminate_holes_and_tiny_objects(thresh, width, height, eps=None, return_type=return_type))

    if debug:
        vis_image(cleared_mask)
        vis_contours(img_crop, cleared_mask)
    return cleared_mask


def crop_image_by_bb_coordinates(img_crop, full_size_minimals, pad=50, return_type='polygon', debug=False, quantile=0.5):
    # return_type = 'polygon'  # binary_mask  polygon
    height, width = img_crop.shape
    x_min, y_min = full_size_minimals
    refined_mask = get_object_contour(img_crop, blur_mode='gaussian_blur', ksize=(7, 7), quantile=quantile)
    refined_wo_holes = eliminate_holes_and_tiny_objects(refined_mask, width, height,
                                                        eps=None, return_type=return_type)
    if debug:
        if return_type == 'polygon':
            vis_contours(img_crop, refined_wo_holes)
        elif return_type == 'mask':
            vis_image(refined_wo_holes, True)
        else:
            raise NotImplemented

    if return_type == 'polygon':
        detected_object_polygons = []
        for polygon in refined_wo_holes:
            local_polygon = polygon.copy()
            local_polygon[:, 0] += x_min
            local_polygon[:, 1] += y_min - pad
            detected_object_polygons.append(local_polygon)
        return detected_object_polygons
    else:
        raise NotImplemented


def get_prepared_data():
    pass


def parse_json(annotations):
    try:
        # priority_labeler_list = [
        #     'КОМП', 'Александр', 'kayla', 'Сергей', 'Пользователь', 'Сергей', '2002kh', '2002k', 'user', 'asus',
        #     'HOMEr', 'kiyh', 'Крылова Виктория', 'Home', 'банан228', 'Jokerit55', 'user', 'marw1xx',
        #     'vladimir', 'Светлана', 'Бедолага', 'marys', 'olgal', 'artem', 'ольга', 'Ольга', 'SPECIALIST'
        # ]
        # test version
        data = annotations['regions']
        coordinates = []
        for obj_ in data:
            if obj_['tnved'] is not None:  # data['user_role'],
                points = np.array(obj_['points'])
                x_min, y_min = points[:, 0].min(), points[:, 1].min()
                x_max, y_max = points[:, 0].max(), points[:, 1].max()
                coordinates.append([x_min, y_min, x_max, y_max])
    except:
        # yolov5 version
        pass
    return np.array(coordinates)


def get_test_data(index=0):
    prefix = '../../../../'
    json_path = f'{prefix}data/all_jsons/all_jsons_test.json'
    with open(json_path, 'r') as stream:
        json_data = json.load(stream)

    # get image
    original_npz = json_data[index]['original_npz']
    image_path = f'{prefix}{original_npz}'
    np_image = get_image(image_path)
    # converted_image = json_data[index]['converted_image']

    # get annotation
    anno_path = original_npz.replace('.npz', '_predict.json')
    annotations = get_annotation(f'{prefix}{anno_path}')
    boxes = parse_json(annotations)
    return image_path, np_image, boxes


def substract_intersected_polygons(world_coordinate_polygons):
    poly_objects = [MyPolygon(world_coordinate_polygon) for world_coordinate_polygon in world_coordinate_polygons]
    num_polygons = len(poly_objects)
    for i_idx in range(num_polygons):
        for j_idx in range(num_polygons):
            if i_idx > j_idx:
                intersected_poly = poly_objects[i_idx].intersect_with(poly_objects[j_idx])
                if intersected_poly is not None:
                    if poly_objects[i_idx].get_area() > poly_objects[j_idx].get_area():
                        poly_objects[i_idx] = poly_objects[i_idx].sub(poly_objects[j_idx])
                    else:
                        poly_objects[j_idx] = poly_objects[j_idx].sub(poly_objects[i_idx])
    return poly_objects


def single_image_processing(idx):
    try:
        # get list of paths to images
        image_path, np_image, boxes = get_test_data(index=idx)

        # start preprocessing
        pad = 0
        world_coordinate_polygons = []
        box_coordinates = []

        prepared_images, prepared_polygons = [], []

        # boxes = [[boxes[:, 0].min(), boxes[:, 1].min(), boxes[:, 2].max(), boxes[:, 3].max()]]
        boxes = [
            [
                int(boxes[:, 0].min()),
                int(boxes[:, 1].min()),
                int(boxes[:, 2].max()),
                int(boxes[:, 3].max())
            ]
        ]

        for box in boxes:
            x_min, y_min, x_max, y_max = box
            img_crop = np_image[max(0, y_min - pad):y_max, x_min:x_max]
            np_mean = round(np_image.mean(), 5)
            for quantile in (i/100 for i in range(85, 94)):
                polygon = crop_image_by_bb_coordinates(img_crop, [x_min, y_min], pad=pad, debug=False, quantile=quantile)  # pad=50
                vis_contours(
                    np_image, [polygon],
                    show_contours=False,
                    save_path=(f'experiments/quantiles_{quantile}', f'{Path(image_path).stem}_mean_{np_mean}.png')
                )
            # prepared_polygons.append(polygon)
    except:
        pass
    # poly_objects = substract_intersected_polygons([i[0] for i in prepared_polygons])
    # vis_contours(np_image, [f.get_coordinates() for f in poly_objects], save_path=None)
    # vis_contours(np_image, [f for f in prepared_polygons], save_path=None)
    # print('Done')


if __name__ == '__main__':
    # main()
    # num_cores = 12
    # num_images = 50
    # with Pool(num_cores) as p:
    #     list(
    #         tqdm(
    #             p.imap_unordered(
    #                 single_image_processing, range(num_images)
    #             ), total=len(range(num_images))
    #         )
    #     )
    single_image_processing(40)
