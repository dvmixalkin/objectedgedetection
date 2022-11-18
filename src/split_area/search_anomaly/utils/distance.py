import numpy as np


def get_manhattan_distance(point1, point2):
    x = abs(point1[0] - point2[0])
    y = abs(point1[1] - point2[1])
    return x + y


def filter_through_manhattan_distance(polygon, min_distance=5):
    polygon_points = []
    for point in polygon[0]:
        if polygon_points:
            distance = get_manhattan_distance(point, polygon_points[-1])
            if distance >= min_distance:
                polygon_points.append(point)
        else:
            polygon_points.append(point)

    return np.array([polygon_points])
