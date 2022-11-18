from collections import Counter


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
