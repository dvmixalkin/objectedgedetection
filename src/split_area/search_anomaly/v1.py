import numpy as np
from src.split_area.search_anomaly.utils.polygon.peaks import get_peaks_v1
from .utils.point_hole_eliminator import point_holes_eliminator
from .utils.distance import get_manhattan_distance


def search_anomaly_v1(image, polygon, debug=False):
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

    # if debug:
    #     vis_polygon(image, polygon, is_closed=True)
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
        # if debug:
        #     Image.fromarray(
        #         cv2.polylines(
        #             img=image.astype(np.uint8).copy(),
        #             pts=[np.asarray([filtered_poly])],
        #             isClosed=True,
        #             color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
        #             thickness=5
        #         )).show()
    resulting_poly_list.append(working_data)
    return resulting_poly_list
