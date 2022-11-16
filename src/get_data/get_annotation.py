import json
import numpy as np


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
    elif frmt == 'preprocessed':
        pass
    else:
        pass

    return coordinates


def get_annotation(name):
    with open(name, 'r') as stream:
        annotations = json.load(stream)
    return annotations
