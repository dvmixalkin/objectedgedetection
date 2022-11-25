import numpy as np

from .stats import get_stats
from .distance import filter_through_manhattan_distance
from .filters import filter_polygon_by_indices
from .polygon_rupture import find_rapture_by_axis


def prepare_intervals(polygon_original, step_size, pix_min_dist=None,
                      pre_filter_dist=None, post_filter_dist=None, image=None):

    polygon = polygon_original.copy()
    if pre_filter_dist is not None:
        polygon = filter_through_manhattan_distance(polygon, min_distance=pre_filter_dist)

    if pix_min_dist is None:
        if pre_filter_dist is not None:
            pix_min_dist = pre_filter_dist
        else:
            pix_min_dist = 1
    else:
        if pre_filter_dist is not None:
            if pre_filter_dist > pix_min_dist:
                print('pix_min_dist could not be less than pre_filter_dist, fixing')
                pix_min_dist = pre_filter_dist

    # polygon = point_holes_eliminator(polygon, interval=pix_min_dist)
    # polygon = point_holes_eliminator(polygon, interval='3')

    if post_filter_dist is not None:
        polygon = filter_through_manhattan_distance(polygon, min_distance=post_filter_dist)

    x_hists = get_stats(
        version=3,
        coordinates=polygon,
        interval=step_size,
        k=1.1,
        image=image
    )  # [bound_offshore_size:-bound_offshore_size]

    y_hists = None
    return x_hists, y_hists


def merge_intersected_intervals(peaks, hard_threshold):
    new_areas = []
    tmp = []
    start_point = 0
    end_point = 0
    points = 0

    for peak_idx in range(peaks.shape[0] - 1):
        if tmp == []:
            start_point = peaks[peak_idx][0]
            end_point = peaks[peak_idx][1]
            points = peaks[peak_idx][2]
            tmp = [start_point, end_point, points]

        if peaks[peak_idx][1] >= peaks[peak_idx + 1][0]:
            end_point = peaks[peak_idx + 1][1]
            points += peaks[peak_idx + 1][2]
        else:
            new_areas.append([start_point, end_point, points])
            tmp = []
            if peak_idx == peaks.shape[0] - 2:
                new_areas.append(peaks[-1].tolist())

    if tmp != []:
        new_areas.append([start_point, end_point, points])
        tmp = []
    peaks = np.array(new_areas)
    return peaks[peaks[:, 2] > hard_threshold]


def get_activations_for_interval(indexes_inner):
    # Look for True holes meaning current interval points
    dense_points_inner = []
    dense_points_outer = []
    sub_list_inner = []
    sub_list_outer = []
    for idx, element in enumerate(indexes_inner):
        if element:
            if len(sub_list_outer) >= 1:
                dense_points_outer.append(sub_list_outer)
                sub_list_outer = []
            sub_list_inner.append(idx)
        else:
            if len(sub_list_inner) >= 1:
                dense_points_inner.append(sub_list_inner)
                sub_list_inner = []
            sub_list_outer.append(idx)

    if sub_list_outer:
        dense_points_outer.append(sub_list_outer)
    if sub_list_inner:
        dense_points_inner.append(sub_list_inner)
    del sub_list_outer, sub_list_inner
    return dense_points_outer, dense_points_inner


def prepare_intervals_v3(polygon_original, step_size, image=None):

    polygon = polygon_original.copy()
    return get_stats(
        version=3,
        coordinates=polygon,
        interval=step_size,
        k=1.1,
        image=image
    )


def get_area_center_point(working_data, tmp, image=None):
    print(working_data.shape, [tmp[:, 0].min(), tmp[:, 1].max()])
    # vis_polygon(
    #     image, working_data,
    #     x_bounds_coordinates=None,
    #     y_bounds_coordinates=None,
    #     hole_point_coordinate=None
    # )

    # берем ломаные на зхаданном интервале
    tmp_working_data = filter_polygon_by_indices(
        working_data, interval=[tmp[:, 0].min(), tmp[:, 1].max()])[0]  # working_data[0, indexes_inner]
    # vis_polygon(
    #     image, tmp_working_data,
    #     x_bounds_coordinates=None,
    #     y_bounds_coordinates=None,
    #     hole_point_coordinate=None
    # )

    # ищем минимум по Y на отфильтрованном интервале (индекс для Y-ов)
    y_min, y_max, index = find_rapture_by_axis(array=tmp_working_data, axis=1)
    # vis_polygon(
    #     image, tmp_working_data,
    #     x_bounds_coordinates=None,
    #     y_bounds_coordinates=[y_min, y_max],
    #     hole_point_coordinate=None
    # )

    # берем верхнюю ломаную на интервале
    upper_poly_line = tmp_working_data[tmp_working_data[:, 1] <= y_min]

    # ищем на верхней ломаной миксимальную точку по Y
    upper_high_peak_points = upper_poly_line[upper_poly_line[:, 1] == y_min]

    # проверка на количество точек с минимальным значением по Y. Если их несколько, считаем расстояние между ними
    mean_value = np.abs(upper_high_peak_points[:, 0] - np.median(upper_high_peak_points[:, 0]))
    # берем связку точек с минмальным расстоянием
    min_difference = mean_value.min()
    top_candidates = upper_high_peak_points[mean_value == min_difference]

    # point_to_draw точка минимума на верхней части полигона, относительно нее и будет формироваться альфа-область
    # для разделения
    if top_candidates.shape[0] > 1:
        idx = np.random.choice(range(top_candidates.shape[0]), 1)[0]
        point_to_draw = top_candidates[idx]
    else:
        point_to_draw = top_candidates[0]
    return point_to_draw  # [point_to_draw[0] - alpha, point_to_draw[0] + alpha]
