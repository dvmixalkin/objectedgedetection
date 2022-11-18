import json
import numpy as np


def original_jsons(anno_object):
    raise NotImplemented
    # # START - for original jsons
    # import cv2
    # import random
    # from PIL import Image
    # rgb_np_imge = np.array(Image.fromarray(image_object).convert('RGB'))
    # all_boxes = []
    # im = rgb_np_imge.copy()
    # for item in anno_object['regions']:
    #     if item['tnved'] is None and item['label'] == 'Кузов':  # len(item['points']) > 2:
    #         polygon = np.array(item['points'])
    #         x_min, x_max = polygon[:, 0].min(), polygon[:, 0].max()
    #         y_min, y_max = polygon[:, 1].min(), polygon[:, 1].max()
    #         all_boxes.append([x_min, x_max, y_min ,y_max])
    #         im = cv2.polylines(
    #             img=im,
    #             pts=[np.array(item['points'])],
    #             isClosed=True,
    #             color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
    #             thickness=5
    #         )
    # Image.fromarray(im).show()
    #
    # np_all_boxes = np.array(all_boxes)
    #
    # x_min, y_min = np_all_boxes[:, 0].min(), np_all_boxes[:, 1].min()
    # x_max, y_max = np_all_boxes[:, 0].max(), np_all_boxes[:, 1].max()
    # coordinates = [x_min, y_min, x_max, y_max]
    # return coordinates


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
        coordinates = original_jsons(json_data)
    else:
        raise NotImplemented
    return coordinates


def get_annotation(name):
    with open(name, 'r', encoding='utf-8') as stream:
        annotations = json.load(stream)
    return annotations
