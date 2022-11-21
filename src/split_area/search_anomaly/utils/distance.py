import numpy as np
from shapely.geometry import Polygon
from .get_np_points_from_polygon import get_np_points_from_polygon


def get_manhattan_distance(point1, point2):
    x = abs(point1[0] - point2[0])
    y = abs(point1[1] - point2[1])
    return x + y


def filter_through_manhattan_distance(polygon, min_distance='auto'):
    def fn(min_distance, polygon):
        polygon_points = []
        for point in polygon[0]:
            if polygon_points:
                distance = get_manhattan_distance(point, polygon_points[-1])
                if distance >= min_distance:
                    polygon_points.append(point)
            else:
                polygon_points.append(point)
        return polygon_points

    if isinstance(min_distance, str):
        orig_poly = Polygon(polygon[0])
        local_polygons = []
        start_points = 5
        for dist in range(start_points, 20):
            local_polygons.append(Polygon(fn(dist, polygon)))
        areas = [abs(polygon.area - orig_poly.area) for polygon in local_polygons]
        min_difference = min(areas)
        min_difference_element_index = areas.index(min_difference)
        print(f'min_distance = {min_difference_element_index + start_points}')
        the_chosen_one = local_polygons[min_difference_element_index]

        return get_np_points_from_polygon(the_chosen_one)
    elif isinstance(min_distance, int):
        polygon_points = []
        for point in polygon[0]:
            if polygon_points:
                distance = get_manhattan_distance(point, polygon_points[-1])
                if distance >= min_distance:
                    polygon_points.append(point)
            else:
                polygon_points.append(point)

        return np.array([polygon_points])
    else:
        raise NotImplemented
