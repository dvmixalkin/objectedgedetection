from collections import Counter
from .polygon.peaks import get_peaks_v1, get_peaks_v2, get_peaks_v3


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


def get_stats(version, coordinates, interval=100, k=1.1, image=None):
    if version == 1:
        return get_peaks_v1(coordinates, interval=100, k=1.1)
    elif version == 2:
        return get_peaks_v2(coordinates)
    elif version == 3:
        return get_peaks_v3(coordinates, image, sector_len=7, sub_interval_len=5)
