from collections import Counter


def find_rapture_by_axis(array, axis):
    if len(array.shape) == 3:
        if array.shape[0] == 1:
            array = array[0]
        else:
            raise NotImplemented('Incorrect array format')

    # take all pixel Y coordinates
    stats = list(Counter(array[:, axis]).keys())
    stats.sort()

    # ищем разницу между соседними значениями координат пикселей
    tmp_stats = []
    for i in range(len(stats) - 1):
        tmp_stats.append(abs(stats[i] - stats[i + 1]))

    # максимальное значение разрыва между координатами
    max_value = max(tmp_stats)

    # получение индекса минимальной координаты по Y у верхней полилинии
    max_idx = tmp_stats.index(max_value)

    # получение пиковых точек верхней и нижней полилиний(выборс в сторону друг друга)
    y_min_threshold = stats[max_idx]
    y_max_threshold = stats[max_idx + 1]
    return y_min_threshold, y_max_threshold, max_idx
