import numpy as np
from PIL import Image
import cv2
import random
from collections import Counter
from .polygon.peaks import get_peaks_v1, get_peaks_v2, get_peaks_v3, get_actual_intervals
from .polygon import point_holes_eliminator, vis_polygon, vis_contours


def get_manhattan_distance(point1, point2):
    x = abs(point1[0] - point2[0])
    y = abs(point1[1] - point2[1])
    return x + y


def get_stats(version, coordinates, interval=100, k=1.1, image=None):
    if version == 1:
        return get_peaks_v1(coordinates, interval=100, k=1.1)
    elif version == 2:
        return get_peaks_v2(coordinates)
    elif version == 3:
        return get_peaks_v3(coordinates, image)


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


def search_anomaly_refined(image, polygon, debug=False):
    step_size = 20
    multiplier = 5
    pix_min_dist = 10
    # polygon = filter_through_manhattan_distance(polygon, min_distance=pix_min_dist)
    polygon = point_holes_eliminator(polygon, interval=int(pix_min_dist/2))
    # polygon = filter_through_manhattan_distance(polygon, min_distance=pix_min_dist)
    x, y = polygon[0, :, 0], polygon[0, :, 1]
    width, heigh = x.max() - x.min(), y.max() - y.min()
    bound_offshore_size = 1 if width <= step_size * multiplier else 3

    x_hists = get_peaks_v1(
        polygon[0, :, 0],
        interval=step_size
    )[bound_offshore_size:-bound_offshore_size]

    if debug:
        vis_polygon(image, polygon, is_closed=True)
    # @TODO Check for exclude
    if x_hists.shape[0] == 0:
        return [polygon[0].tolist()]

    high_peaks_x = x_hists[x_hists[:, 2] > 2]
    if high_peaks_x.shape[0] == 0:
        return [polygon[0].tolist()]

    try:
        avg_without_max = (high_peaks_x[:, 2].sum() - high_peaks_x[:, 2].max()) / (high_peaks_x.shape[0] - 1)
    except:
        print('message utils avg_without_max')
    weighted = high_peaks_x[:, 2].min() + (high_peaks_x[:, 2].max() - high_peaks_x[:, 2].min()) * 0.5
    if high_peaks_x[:, 2].max() / avg_without_max < 1.5:
        # if high_peaks_x[:, 2].min() / high_peaks_x[:, 2].mean() > 0.5:
        return [polygon[0].tolist()]

    resulting_poly_list = []
    working_data = polygon.copy()
    peaks = high_peaks_x[high_peaks_x[:, 2] > avg_without_max]
    # peaks = peaks1[(peaks1[:, 2] - peaks1[:, 2].mean()) / peaks1[:, 2].std() > 2]
    if peaks.shape[0] > 1:
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
                tmp = start_point, end_point, points

            if peaks[peak_idx][1] >= peaks[peak_idx + 1][0]:
                end_point = peaks[peak_idx + 1][1]
                points += peaks[peak_idx + 1][2]
            else:
                new_areas.append([start_point, end_point, points])
                tmp = []
        if tmp != []:
            new_areas.append([start_point, end_point, points])
            tmp = []
        peaks = np.array(new_areas)
        peaks = peaks[peaks[:, 2] > weighted]
    if peaks.shape[0] == 0:
        return [polygon[0].tolist()]

    for line_idx, line in enumerate(peaks):
        xmin, xmax, _ = line
        if isinstance(working_data, list):
            working_data = np.expand_dims(np.ascontiguousarray(working_data), axis=0)
        indexes_inner = ((working_data[0, :, 0] >= xmin).astype(np.int) *
                         (working_data[0, :, 0] <= xmax).astype(np.int)).astype(bool)

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

        if len(dense_points_outer) <= len(dense_points_inner):
            l = []
            for set_ in range(len(dense_points_inner) - 1):
                left = dense_points_inner[set_]
                right = dense_points_inner[set_ + 1]
                l.append(abs(left[-1] - right[0]))
            holes = [i for i, d in enumerate(l) if d < 30]
            holes_to_clamp = []
            local_holes = []
            for idx_h, h in enumerate(holes):
                if local_holes == []:
                    local_holes.append(h)
                else:
                    if abs(local_holes[-1] - h) > 1:
                        if len(local_holes) > 1:
                            holes_to_clamp.append(local_holes)
                        local_holes = []
                    else:
                        local_holes.append(h)
                if idx_h == len(holes) - 1:
                    holes_to_clamp.append(local_holes)
                    local_holes = []

            for dense_hole in holes_to_clamp:
                # dense_points_inner
                start_idx = dense_points_inner[dense_hole[0]][0]

                finish_idx = dense_points_inner[dense_hole[-1] + 1][-1] + 1
                idxs = [i for i in range(start_idx, finish_idx)]
                dense_points_inner[dense_hole[0]] = idxs

                num = (dense_hole[-1] + 2) - (dense_hole[0] + 1)
                for i in range(num):
                    dense_points_inner.pop(dense_hole[0] + 1)

        dense_points_outer_new = []
        s = None
        e = None
        for poly_idx, poly_line in enumerate(dense_points_inner):
            if poly_idx == 0:
                dense_points_outer_new.append([i for i in range(poly_line[0])])
                e = poly_line[-1]
            else:
                s = e + 1
                e = poly_line[0]
                dense_points_outer_new.append([i for i in range(s, e + 1)])
        s = poly_line[-1] + 1
        e = working_data.shape[1]
        dense_points_outer_new.append([i for i in range(s, e)])

        try:
            upper_points = working_data[0, dense_points_inner[0]]
            lower_points = working_data[0, dense_points_inner[1]]
        except:
            print('utils - message')
            # @TODO Fix this
            return [polygon[0].tolist()]
        distance_matrix = np.zeros((upper_points.shape[0], lower_points.shape[0]))  # lines - columns
        for i_x, upper_point in enumerate(upper_points):
            for i_y, lower_point in enumerate(lower_points):
                distance_matrix[i_x, i_y] = get_manhattan_distance(upper_point, lower_point)

        dilation = 1
        top = np.argmin(distance_matrix.min(axis=1))
        bot = np.argmin(distance_matrix[top])
        upper_left, upper_right = upper_points[:top - dilation], upper_points[top + dilation:]
        lower_right, lower_left = lower_points[:bot - dilation], lower_points[bot + dilation:]
        left_bound_top = working_data[0, dense_points_outer_new[0]].tolist()
        try:
            left_bound_bot = working_data[0, dense_points_outer_new[2]].tolist()
        except:
            print('message')
            return [working_data[0].tolist()]

        right_bound_mid = working_data[0, dense_points_outer_new[1]].tolist()
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
        if debug:
            Image.fromarray(
                cv2.polylines(
                    img=image.astype(np.uint8).copy(),
                    pts=[np.asarray([filtered_poly])],
                    isClosed=True,
                    color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
                    thickness=5
                )).show()
    resulting_poly_list.append(working_data)
    return resulting_poly_list


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


def get_thresholds(polygon, x_hists, y_hists, vertical_multiplier=1.5):
    if isinstance(x_hists, np.ndarray):
        if x_hists.shape[0] == 1:
            soft_threshold = x_hists[:, 2].mean()
        else:
            soft_threshold = (x_hists[:, 2].sum() - x_hists[:, 2].max()) / (x_hists.shape[0] - 1)
        hard_threshold = x_hists[:, 2].min() + (x_hists[:, 2].max() - x_hists[:, 2].min()) * 0.5

    elif isinstance(x_hists, list):
        x, y = polygon[0, :, 0], polygon[0, :, 1]
        width = x.max() - x.min()
        separate_threshold = min(int(width * 0.15), 30)
        separate_x_min = x.min() + separate_threshold
        separate_x_max = x.max() - separate_threshold

        interval_bounds = [[np.array(i)[:, 0].min(), np.array(i)[:, 1].max()] for i in x_hists]
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
        y_filter_threshold = np.quantile(filtered_poly_lines_list[0][:, 1], 0.85)
        return filtered_poly_lines, filtered_poly_lines_list
    else:
        raise NotImplemented
    return soft_threshold, hard_threshold


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


def find_rapture_by_axis(array, axis):
    if len(array.shape) == 3:
        if array.shape[0] == 1:
            array = array[0]
        else:
            raise NotImplemented('Incorrect array format')

    stats = list(Counter(array[:, axis]).keys())
    stats.sort()
    tmp_stats = []
    for i in range(len(stats) - 1):
        tmp_stats.append(abs(stats[i] - stats[i + 1]))

    max_value = max(tmp_stats)
    max_idx = tmp_stats.index(max_value)
    y_min_threshold = min(stats[max_idx], stats[-1])
    y_max_threshold = min(stats[max_idx+1], stats[-1])
    return y_min_threshold, y_max_threshold, max_idx


def filterPolygon_by_indices(polygon, interval):
    # фильтруем точки полигона на заданном интервале -> индексы точек
    indexes_inner = ((polygon[0, :, 0] >= interval[0]).astype(np.int) *
                     (polygon[0, :, 0] <= interval[1]).astype(np.int)).astype(bool)
    assert polygon.shape[1] == indexes_inner.shape[0], 'Size mismatch! utils.py: line 457'
    return polygon[:, indexes_inner]


def get_area_center_point(working_data, tmp):
    print(working_data.shape, [tmp[:, 0].min(), tmp[:, 1].max()])
    tmp_working_data = filterPolygon_by_indices(
        working_data, interval=[tmp[:, 0].min(), tmp[:, 1].max()])[0]  # working_data[0, indexes_inner]

    # ищем минимум по Y на отфильтрованном интервале
    y_min, y_max, index = find_rapture_by_axis(array=tmp_working_data, axis=1)
    upper_poly_line = tmp_working_data[tmp_working_data[:, 1] <= y_min]
    upper_high_peak_points = upper_poly_line[upper_poly_line[:, 1] == y_min]
    mean_value = np.abs(upper_high_peak_points[:, 0] - np.median(upper_high_peak_points[:, 0]))
    min_difference = mean_value.min()
    top_candidates = upper_high_peak_points[mean_value == min_difference]
    # point_to_draw точка минимума на верхней части полигона, относительно нее и будет формироваться альфа-область для разделения
    if top_candidates.shape[0] > 1:
        idx = np.random.choice(range(top_candidates.shape[0]), 1)[0]
        point_to_draw = top_candidates[idx]
    else:
        point_to_draw = top_candidates[0]
    return point_to_draw  # [point_to_draw[0] - alpha, point_to_draw[0] + alpha]


def search_anomaly_refined_v2(image, polygon, step_size=20, debug=False):
    x_hists, y_hists = prepare_intervals(
        polygon, step_size, pix_min_dist=2, pre_filter_dist=None, post_filter_dist=None,
        image=image
    )

    if isinstance(x_hists, np.ndarray):
        if x_hists.shape[0] == 0 or x_hists[x_hists[:, 2] > 2].shape[0] == 0:
            return [polygon[0].tolist()]
    else:
        # raise NotImplemented
        if len(x_hists) == 1:
            return [polygon[0].tolist()]
    soft_threshold, hard_threshold = get_thresholds(polygon, x_hists, y_hists)

    working_data = point_holes_eliminator(polygon.copy(), interval=1)

    resulting_poly_list = []
    if not isinstance(soft_threshold, np.ndarray):
        peaks = x_hists[x_hists[:, 2] > soft_threshold]
    else:
        peaks = get_stats(
            version=3, coordinates=np.expand_dims(soft_threshold, axis=0),
            interval=step_size, k=1.1,
            image=image
        )
        try:
            peaks = get_actual_intervals(hard_threshold)  # peaks
        except:
            if peaks is None:
                return [polygon[0].tolist()]

    if not isinstance(peaks, list):
        if peaks.shape[0] > 1:
            peaks = merge_intersected_intervals(peaks, hard_threshold)
    try:
        if peaks.shape[0] == 0:
            return [polygon[0].tolist()]
    except:
        if len(peaks) == 0:
            return [polygon[0].tolist()]

    for line_idx, line in enumerate(peaks):
        if isinstance(working_data, list):
            working_data = np.expand_dims(np.ascontiguousarray(working_data), axis=0)
        # переводим в формат Numpy-массива
        tmp = np.array(line)
        if len(tmp.shape) < 2:
            # raise NotImplemented
            continue
            # return [polygon[0].tolist()] # ЗАГЛУШКА-КОСТЫЛЬ
        # получение точки полигона с минимальным значением по Y(центр интервала) для дальнеёшего разделения
        peak_point = get_area_center_point(working_data, tmp)
        alpha = 25
        x_min, x_max = peak_point[0] - alpha, peak_point[0] + alpha
        polygon1 = filterPolygon_by_indices(working_data, interval=[x_min, x_max])
        y_min, y_max = polygon1[0, :, 1].min(), peak_point[1]
        vis_polygon(image, polygon1,
                    is_closed=True,
                    x_bounds_coordinates=[x_min, x_max],
                    y_bounds_coordinates=[y_min, y_max],
                    hole_point_coordinate=peak_point
                    )

        indexes_inner = ((working_data[0, :, 0] >= x_min).astype(np.int) *
                         (working_data[0, :, 0] <= x_max).astype(np.int)).astype(bool)

        # points indexes in current area
        dense_points_outer, dense_points_inner = get_activations_for_interval(indexes_inner)

        if len(dense_points_outer) <= len(dense_points_inner):
            l = []
            for set_ in range(len(dense_points_inner) - 1):
                left = dense_points_inner[set_]
                right = dense_points_inner[set_ + 1]
                l.append(abs(left[-1] - right[0]))
            holes = [i for i, d in enumerate(l) if d < 30]
            holes_to_clamp = []
            local_holes = []
            for idx_h, h in enumerate(holes):
                if local_holes == []:
                    local_holes.append(h)
                else:
                    if abs(local_holes[-1] - h) > 1:
                        if len(local_holes) > 1:
                            holes_to_clamp.append(local_holes)
                        local_holes = []
                    else:
                        local_holes.append(h)
                if idx_h == len(holes) - 1:
                    holes_to_clamp.append(local_holes)
                    local_holes = []

            for dense_hole in holes_to_clamp:
                # dense_points_inner
                start_idx = dense_points_inner[dense_hole[0]][0]

                finish_idx = dense_points_inner[dense_hole[-1] + 1][-1] + 1
                idxs = [i for i in range(start_idx, finish_idx)]
                dense_points_inner[dense_hole[0]] = idxs

                num = (dense_hole[-1] + 2) - (dense_hole[0] + 1)
                for i in range(num):
                    dense_points_inner.pop(dense_hole[0] + 1)

        try:
            upper_points = working_data[0, dense_points_inner[0]]
            lower_points = working_data[0, dense_points_inner[1]]
        except:
            continue
            # print('utils - message')
            # # @TODO Fix this
            # return [polygon[0].tolist()]
        distance_matrix = np.zeros((upper_points.shape[0], lower_points.shape[0]))  # lines - columns
        for i_x, upper_point in enumerate(upper_points):
            for i_y, lower_point in enumerate(lower_points):
                distance_matrix[i_x, i_y] = get_manhattan_distance(upper_point, lower_point)

        dilation = 1
        top = np.argmin(distance_matrix.min(axis=1))
        bot = np.argmin(distance_matrix[top])
        upper_left, upper_right = upper_points[:top - dilation], upper_points[top + dilation:]
        lower_right, lower_left = lower_points[:bot - dilation], lower_points[bot + dilation:]
        left_bound_top = working_data[0, dense_points_outer[0]].tolist()  # dense_points_outer_new
        try:
            left_bound_bot = working_data[0, dense_points_outer[2]].tolist()  # dense_points_outer_new
        except:
            print('message')
            return [working_data[0].tolist()]

        right_bound_mid = working_data[0, dense_points_outer[1]].tolist()  #dense_points_outer_new
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
        if debug:
            vis_polygon(image, filtered_poly)

    if isinstance(working_data, np.ndarray):
        if len(working_data.shape) == 3:
            working_data = working_data[0]
        working_data = working_data.tolist()

    resulting_poly_list.append(working_data)
    return resulting_poly_list


def search_anomaly_refined_v3(image, polygon, step_size=20, debug=False):
    dense_polygon = point_holes_eliminator(polygon.copy(), interval=1)
    x_hists, y_hists = prepare_intervals(
        dense_polygon,
        step_size,
        pix_min_dist=2,
        pre_filter_dist=None,
        post_filter_dist=None,
        image=image
    )

    if isinstance(x_hists, np.ndarray):
        if x_hists.shape[0] == 0 or x_hists[x_hists[:, 2] > 2].shape[0] == 0:
            return [polygon[0].tolist()]
    else:
        if len(x_hists) == 1:
            return [polygon[0].tolist()]

    soft_threshold, hard_threshold = get_thresholds(polygon, x_hists, y_hists)
    peaks = get_actual_intervals(hard_threshold)
    resulting_poly_list = []


def main():
    pass


if __name__ == '__main__':
    main()
