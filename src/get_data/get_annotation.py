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


def get_coordinates_to_crop(json_data, frmt='yolo_output'):
    coordinates = []
    assert frmt in ['yolo_output', 'preprocessed', 'source_data'], 'Select correct format.'
    all_boxes = []
    if frmt == 'yolo_output':
        for box in json_data['prediction']:
            all_boxes.append(box['coord'])
        np_all_boxes = np.array(all_boxes)
        x_min, y_min = np_all_boxes[:, 0].min(), np_all_boxes[:, 1].min()
        x_max, y_max = np_all_boxes[:, 2].max(), np_all_boxes[:, 3].max()
        coordinates = [x_min, y_min, x_max, y_max]
    elif frmt == 'source_data':
        all_coordinates = original_jsons(json_data)
        coordinates = [
            all_coordinates[:, 0].min(),
            all_coordinates[:, 1].min(),
            all_coordinates[:, 2].max(),
            all_coordinates[:, 3].max()
        ]
    else:
        raise NotImplemented
    return coordinates


def get_annotation(name):
    with open(name, 'r', encoding='utf-8') as stream:
        annotations = json.load(stream)
    return annotations
