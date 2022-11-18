import numpy as np


def get_poly_stats(coordinates, sector_len):
    x, y = coordinates[0, :, 0], coordinates[0, :, 1]
    bins = np.ceil((x.max() - x.min()) / sector_len).astype(int)
    n, bin_edges = np.histogram(x, bins=bins)
    bin_edges = bin_edges.astype(int)
    intervals = []
    for idx in range(len(n)):
        intervals.append([bin_edges[idx], bin_edges[idx+1]])
    return intervals, n, bin_edges


def get_intersected_intervals(intervals, n, sub_interval_len, mean_multiplier=1.2):
    intersected_interval_stats = []
    for triplet_start_point in range(len(intervals)-(sub_interval_len-1)):
        intersected_interval_stats.append(
            [
                intervals[triplet_start_point][0],
                intervals[triplet_start_point + (sub_interval_len-1)][1],
                n[triplet_start_point: triplet_start_point + sub_interval_len].sum()
            ]
        )
    if intersected_interval_stats == []:
        return None

    np_intersected_interval_stats = np.array(intersected_interval_stats)
    mean = np_intersected_interval_stats[:, 2].mean() * 1.2
    activated_intervals = np_intersected_interval_stats[np_intersected_interval_stats[:, 2] > mean].tolist()
    return activated_intervals


def get_separated_intervals(activated_intervals):
    divided_activated_intervals = []
    right_bound = None
    for idx in range(len(activated_intervals)):
        if idx == 0:
            interval = [activated_intervals[idx]]
            right_bound = activated_intervals[idx][1]
        # elif idx == len(activated_intervals) - 2:
        else:
            if right_bound >= activated_intervals[idx][0]:  # INTERSECTION
                interval.append(activated_intervals[idx])
                right_bound = activated_intervals[idx][1]
            else:  # NO INTERSECTION
                divided_activated_intervals.append(interval)
                interval = [activated_intervals[idx]]
                right_bound = activated_intervals[idx][1]
            if idx == len(activated_intervals)-1:
                divided_activated_intervals.append(interval)
                interval = None
                right_bound = None
    return divided_activated_intervals


def get_actual_intervals(divided_activated_intervals):
    actual_intervals = []
    for element in divided_activated_intervals:
        np_element = np.array(element)
        max_points_idx = 0
        if np_element.shape[0] > 1:
            max_value = np_element[:, 2].max()
            max_points_idx = np_element[:, 2].tolist().index(max_value)
        actual_intervals.append(element[max_points_idx])
    return np.array(actual_intervals)


def get_peaks_v3(coordinates, image=None, sector_len=5, sub_interval_len=7):
    assert sub_interval_len > 0, 'sub_interval_len should be greater than 0'
    print(f'Sub interval range = {sector_len*sub_interval_len}')

    intervals, n, bin_edges = get_poly_stats(coordinates, sector_len)
    activated_intervals = get_intersected_intervals(intervals, n, sub_interval_len, mean_multiplier=1.2)
    try:
        divided_activated_intervals = get_separated_intervals(activated_intervals)
    except:
        return None
    # actual_intervals = get_actual_intervals(divided_activated_intervals)

    # if image is not None:
    #     visualize_polygon(image, coordinates[0])
    return divided_activated_intervals  # np.array(actual_intervals)


def main():
    pass


if __name__ == '__main__':
    main()
