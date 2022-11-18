import numpy as np
from PIL import Image
import cv2
import random
from collections import Counter
from .polygon.peaks import get_peaks_v1, get_peaks_v2, get_peaks_v3, get_actual_intervals
from .polygon import point_holes_eliminator, vis_polygon, vis_contours
from .utils import (get_stats,
                    get_area_center_point,
                    get_manhattan_distance,
                    filter_through_manhattan_distance,
                    get_activations_for_interval)  # prepare_intervals, get_thresholds


def filterPolygon_by_interval(polygon, interval):
    # фильтруем точки полигона на заданном интервале -> индексы точек
    indexes_inner = ((polygon[0, :, 0] >= interval[0]) *
                     (polygon[0, :, 0] <= interval[1])).astype(bool)
    assert polygon.shape[1] == indexes_inner.shape[0], 'Size mismatch! utils_v1.py: line 15'
    return polygon[:, indexes_inner]


def eliminate_borders(polygon, border_fraction=0.15, return_data='polygon'):
    x, y = polygon[0, :, 0], polygon[0, :, 1]
    width = x.max() - x.min()
    separate_threshold = min(int(width * border_fraction), 30)
    separate_x_min = x.min() + separate_threshold
    separate_x_max = x.max() - separate_threshold
    interval = [separate_x_min, separate_x_max]
    if return_data == 'interval':
        return interval

    elif return_data == 'polygon':
        new_polygon = filterPolygon_by_interval(polygon, interval)
        return new_polygon
    else:
        raise NotImplemented


def get_y_stats(filtered_poly_lines):
    y_stats = list(Counter(filtered_poly_lines[:, 1]).keys())
    y_stats.sort()
    tmp_stats = []
    for i in range(len(y_stats) - 1):
        tmp_stats.append(abs(y_stats[i] - y_stats[i + 1]))
    try:
        max_value = max(tmp_stats)
    except:
        return None
    max_idx = tmp_stats.index(max_value)
    y_max_threshold = min(y_stats[max_idx] + 1, y_stats[-1])
    upper_y_coordinates = filtered_poly_lines[filtered_poly_lines[:, 1] < y_max_threshold]
    return upper_y_coordinates


def get_threshold(polygon, x_hists):
    separate_x_min, separate_x_max = eliminate_borders(polygon, return_data='interval')

    interval_bounds = [[np.array(i)[:, 0].min(), np.array(i)[:, 1].max()] for i in x_hists]
    x = polygon[0, :, 0]
    masks = np.zeros_like(x).astype(bool)
    poly_splits = []
    for interval in interval_bounds:
        left_bound = interval[0]
        right_bound = interval[1]
        if interval[0] < separate_x_min:
            left_bound = separate_x_min
            if interval[1] < separate_x_min:
                continue
        if interval[1] > separate_x_max:
            right_bound = separate_x_min
            if interval[0] > separate_x_max:
                continue
        mask = (x > left_bound) * (x < right_bound)
        masks += mask
        poly_splits.append(mask)
    filtered_poly_lines = polygon[0, masks]
    filtered_poly_lines_list = []
    for mask in poly_splits:
        filtered_poly = polygon[0, mask]
        y_stat = get_y_stats(filtered_poly)
        if y_stat is not None:
            filtered_poly_lines_list.append(y_stat)
    return filtered_poly_lines, filtered_poly_lines_list


def prepare_intervals_v3(polygon_original, step_size, image=None):

    polygon = polygon_original.copy()
    return get_stats(
        version=3,
        coordinates=polygon,
        interval=step_size,
        k=1.1,
        image=image
    )


def search_anomaly_refined_v3(image, polygon, step_size=20, debug=False):
    # убрать дырки между точками полигона(сильное уплотнение - 1+1)
    dense_polygon = point_holes_eliminator(polygon.copy(), interval=1)
    # обрезка полигона по краям (чтобы не было статистики на основе боковых линий) 15% ширины слева и справа
    polygon_wo_borders = eliminate_borders(polygon, border_fraction=0.15)
    # получить интервалы
    polygon_wo_borders_5 = filter_through_manhattan_distance(polygon_wo_borders, min_distance=5)
    x_hists = prepare_intervals_v3(
        polygon_wo_borders_5,
        step_size,
        image=image
    )
    new_intervals = []
    for inter_ in x_hists:
        point_ = get_area_center_point(polygon, np.array(inter_))
        alpha = 25
        x_min, x_max = point_[0] - alpha, point_[0] + alpha

        # ДЛЯ ОТРИСОВКИ ВО ВРЕМЯ ДЕБАГГИНГА
        # polygon1 = filterPolygon_by_interval(polygon, interval=[x_min, x_max])
        # y_min, y_max = polygon1[0, :, 1].min(), point_[1]
        # vis_polygon(image, polygon1,
        #             is_closed=False,
        #             x_bounds_coordinates=[x_min, x_max],
        #             y_bounds_coordinates=[y_min, y_max],
        #             hole_point_coordinate=point_
        #             )
        new_intervals.append([x_min, x_max])

    working_data = dense_polygon[0].copy()
    resulting_poly_list = []
    # image_to_draw = image.copy()
    image_to_draw = np.array(Image.fromarray(image).convert('RGB'))
    for peak in new_intervals:  # peaks:
        x_min, x_max = peak
        if isinstance(working_data, list):
            working_data = np.ascontiguousarray(working_data)

        # vis_polygon(image, dense_polygon, x_bounds_coordinates=peak[:2])

        indexes_inner = ((working_data[:, 0] >= x_min).astype(np.int) *
                         (working_data[:, 0] <= x_max).astype(np.int)).astype(bool)
        dense_points_outer, dense_points_inner = get_activations_for_interval(indexes_inner)
        if len(dense_points_outer) != 3 or len(dense_points_inner) != 2:
            continue
            # raise NotImplemented
            # vis_polygon(image, working_data[dense_points_inner[0]])

        # верхняя линия точек
        upper_points = working_data[dense_points_inner[0]]
        # нижняя линия точек
        lower_points = working_data[dense_points_inner[1]]
        distance_matrix = np.zeros((upper_points.shape[0], lower_points.shape[0]))  # lines - columns
        for i_x, upper_point in enumerate(upper_points):
            for i_y, lower_point in enumerate(lower_points):
                distance_matrix[i_x, i_y] = get_manhattan_distance(upper_point, lower_point)

        # попытка найти 2 точки для разделния по прямой
        # кандидат с верхней гряды точек для разделения
        top = np.argmin(distance_matrix.min(axis=1))
        # кандидат с нижней гряды точек для разделения
        bot = np.argmin(distance_matrix[top])
        dilation = 1
        upper_left, upper_right = upper_points[:top - dilation], upper_points[top + dilation:]
        lower_right, lower_left = lower_points[:bot - dilation], lower_points[bot + dilation:]

        left_bound_top = working_data[dense_points_outer[0]].tolist()  # dense_points_outer_new
        left_bound_bot = working_data[dense_points_outer[2]].tolist()  # dense_points_outer_new
        right_bound_mid = working_data[dense_points_outer[1]].tolist()

        first_polygon = left_bound_top + upper_left.tolist() + lower_left.tolist() + left_bound_bot
        x_f = np.array(first_polygon)[:, 0]
        y_f = np.array(first_polygon)[:, 1]
        area_f = (x_f.max() - x_f.min()) * (y_f.max() - y_f.min())
        second_polygon = upper_right.tolist() + right_bound_mid + lower_right.tolist()
        x_s = np.array(second_polygon)[:, 0]
        y_s = np.array(second_polygon)[:, 1]
        area_s = (x_s.max() - x_s.min()) * (y_s.max() - y_s.min())

        filtered_poly = first_polygon if area_f < area_s else second_polygon
        working_data = first_polygon if area_f >= area_s else second_polygon
        resulting_poly_list.append(filtered_poly)

        # image_to_draw = cv2.polylines(
        #     img=image_to_draw.astype(np.uint8),
        #     pts=[np.asarray(filtered_poly)],
        #     isClosed=True,
        #     color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
        #     thickness=3)
    if isinstance(working_data, np.ndarray):
        if len(working_data.shape) == 3:
            working_data = working_data[0]
        working_data = working_data.tolist()
    image_to_draw = cv2.polylines(
        img=image_to_draw.astype(np.uint8),
        pts=[np.asarray(working_data)],
        isClosed=True,
        color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
        thickness=3)
    Image.fromarray(image_to_draw).show()
    resulting_poly_list.append(working_data)
    return resulting_poly_list


def main():
    pass


if __name__ == '__main__':
    main()
