import numpy as np
from .utils.distance import get_manhattan_distance
from .utils.filters import filter_polygon_by_indices
from .utils.intervals import prepare_intervals, merge_intersected_intervals, \
    get_area_center_point, get_activations_for_interval
from .utils.point_hole_eliminator import point_holes_eliminator
from .utils.stats import get_stats
from .utils.thresholds import get_thresholds
from .utils.polygon.peaks import get_actual_intervals


def search_anomaly_v2(image, polygon, step_size=20, debug=False):
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
        polygon1 = filter_polygon_by_indices(working_data, interval=[x_min, x_max])
        y_min, y_max = polygon1[0, :, 1].min(), peak_point[1]
        # vis_polygon(image, polygon1,
        #             is_closed=True,
        #             x_bounds_coordinates=[x_min, x_max],
        #             y_bounds_coordinates=[y_min, y_max],
        #             hole_point_coordinate=peak_point
        #             )

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

    if isinstance(working_data, np.ndarray):
        if len(working_data.shape) == 3:
            working_data = working_data[0]
        working_data = working_data.tolist()

    resulting_poly_list.append(working_data)
    return resulting_poly_list
