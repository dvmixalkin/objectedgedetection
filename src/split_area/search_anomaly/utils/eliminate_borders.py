from .filters import filter_polygon_by_indices


def eliminate_borders(polygon, border_fraction=0.15, return_data='polygon', pixel_limit=30):
    x, y = polygon[0, :, 0], polygon[0, :, 1]
    width = x.max() - x.min()
    separate_threshold = min(int(width * border_fraction), pixel_limit)
    separate_x_min = x.min() + separate_threshold
    separate_x_max = x.max() - separate_threshold
    interval = [separate_x_min, separate_x_max]
    if return_data == 'interval':
        return interval

    elif return_data == 'polygon':
        new_polygon = filter_polygon_by_indices(polygon, interval)
        return new_polygon
    else:
        raise NotImplemented
