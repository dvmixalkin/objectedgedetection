import json
import numpy as np


def original_jsons(anno_object):
    try:
        # priority_labeler_list = [
        #     'КОМП', 'Александр', 'kayla', 'Сергей', 'Пользователь', 'Сергей', '2002kh', '2002k', 'user', 'asus',
        #     'HOMEr', 'kiyh', 'Крылова Виктория', 'Home', 'банан228', 'Jokerit55', 'user', 'marw1xx',
        #     'vladimir', 'Светлана', 'Бедолага', 'marys', 'olgal', 'artem', 'ольга', 'Ольга', 'SPECIALIST'
        # ]
        # test version
        data = anno_object['regions']
        cargo_coordiantes = []
        for item in data:
            if item['user_create'] != 'EXPERT' and item['label'] == 'Кузов':
                all_points = np.array(item['points'])
                x_min, y_min = all_points[:, 0].min(), all_points[:, 1].min()
                x_max, y_max = all_points[:, 0].max(), all_points[:, 1].max()
                cargo_coordiantes.append([x_min, y_min, x_max, y_max])

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
    return np.array(cargo_coordiantes), np.array(coordinates)


def has50_intersection(good_box, cargo_box):
    x_min = np.maximum(good_box[:, 0], cargo_box[0])
    y_min = np.maximum(good_box[:, 1], cargo_box[1])
    x_max = np.minimum(good_box[:, 2], cargo_box[2])
    y_max = np.minimum(good_box[:, 3], cargo_box[3])
    mask = ((x_max - x_min) * (y_max - y_min)) > 0
    return mask


def get_coordinates_to_crop(image, json_data, frmt='yolo_output'):
    coordinates = []
    assert frmt in ['yolo_output', 'preprocessed', 'source_data'], 'Select correct format.'
    all_boxes = []
    if frmt == 'yolo_output':
        for box in json_data['prediction']:
            all_boxes.append(box['coord'])
        np_all_boxes = np.array(all_boxes)
        x_min, y_min = np_all_boxes[:, 0].min(), np_all_boxes[:, 1].min()
        x_max, y_max = np_all_boxes[:, 2].max(), np_all_boxes[:, 3].max()
        coordinates = [[x_min, y_min, x_max, y_max]]
    elif frmt == 'source_data':
        cargo_coordiantes, all_coordinates = original_jsons(json_data)
        per_cargo_coordinates = []
        for cargo_coordinate in cargo_coordiantes:
            mask = has50_intersection(all_coordinates, cargo_coordinate)
            # left, top, right, bot = cargo_coordinate
            # mask = (all_coordinates[:, 0] > left) * \
            #        (all_coordinates[:, 1] > top) * \
            #        (all_coordinates[:, 2] < right) * \
            #        (all_coordinates[:, 3] < bot)
            try:
                coordinates.append(
                    [
                        all_coordinates[mask, 0].min(),
                        all_coordinates[mask, 1].min(),
                        all_coordinates[mask, 2].max(),
                        all_coordinates[mask, 3].max()
                    ]
                )
            except:
                pass

    else:
        raise NotImplemented
    return np.array(coordinates)


def get_annotation(name):
    with open(name, 'r', encoding='utf-8') as stream:
        annotations = json.load(stream)
    return annotations


def vis_boxes(image, boxes):
    import cv2
    import random
    from PIL import Image
    image_to_draw = np.array(Image.fromarray(image).convert('RGB'))
    for box in boxes:
        start_point = box[:2]
        end_point = box[2:]
        color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
        thickness = 5
        image_to_draw = cv2.rectangle(image_to_draw, start_point, end_point, color, thickness)
    Image.fromarray(image_to_draw).show()
