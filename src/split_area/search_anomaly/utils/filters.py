import numpy as np


def filter_polygon_by_indices(polygon, interval):
    # фильтруем точки полигона на заданном интервале -> индексы точек
    indexes_inner = ((polygon[0, :, 0] >= interval[0]).astype(np.int) *
                     (polygon[0, :, 0] <= interval[1]).astype(np.int)).astype(bool)
    assert polygon.shape[1] == indexes_inner.shape[0], 'Size mismatch! utils.py: line 457'
    return polygon[:, indexes_inner]
