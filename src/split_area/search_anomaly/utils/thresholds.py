import numpy as np
from .eliminate_borders import eliminate_borders
from .stats import get_y_stats


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
