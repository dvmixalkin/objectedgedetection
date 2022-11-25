import numpy as np


def get_poly_stats(coordinates, sector_len):
    # получить значения координат по X и Y
    x, y = coordinates[0, :, 0], coordinates[0, :, 1]

    # подсчет количества интервалов
    bins = np.ceil((x.max() - x.min()) / sector_len).astype(int)

    # подсчёт статистики(количества точек полигона) на каждом интервале (шириной в sector_len)
    # bin_edges - границы интервалов, n - количество точек на интервале
    # bin_edges.shape = n.shape + 1
    n, bin_edges = np.histogram(x, bins=bins)
    bin_edges = bin_edges.astype(int)
    intervals = []

    #  запаковка интевалов в читаемый вид, в соответствии с статистикой(количеством точек на интервале)
    for idx in range(len(n)):
        intervals.append([bin_edges[idx], bin_edges[idx+1]])
    return intervals, n, bin_edges


def get_intersected_intervals(intervals, n, sub_interval_len, mean_multiplier=1.2,
                              return_type='np.ndarray', threshold_type='quantile80'):
    assert return_type in ['list', 'np.ndarray'], 'Please, specify return_type.'
    intersected_interval_stats = []
    # итерируемся по интервалам и складываем их мини батчами(по sub_interval_len штук в одном батче)
    for triplet_start_point in range(len(intervals)-(sub_interval_len-1)):
        intersected_interval_stats.append(
            [
                # левая граница интервала
                intervals[triplet_start_point][0],
                # правая граница интервала
                intervals[triplet_start_point + (sub_interval_len-1)][1],
                # количество точек на интервале (+ 1 тестовая версия)
                n[triplet_start_point: triplet_start_point + sub_interval_len + 1].sum()
            ]
        )

    # Если интервалов нет - нет маски
    if intersected_interval_stats == []:
        if intervals != []:
            for element_idx in range(len(intervals)-1):
                pass
        else:
            return None

    # подобрать порог для выбора интервалов
    np_intersected_interval_stats = np.array(intersected_interval_stats)
    if threshold_type == 'mean':
        threshold = np.mean(np_intersected_interval_stats[:, 2]) * mean_multiplier
    elif threshold_type == 'median':
        threshold = np.median(np_intersected_interval_stats[:, 2])
    elif threshold_type == 'quantile50':
        threshold = np.quantile(np_intersected_interval_stats[:, 2], 0.5)
    elif threshold_type == 'quantile80':
        threshold = np.quantile(np_intersected_interval_stats[:, 2], 0.8)
    elif threshold_type == 'quantile95':
        threshold = np.quantile(np_intersected_interval_stats[:, 2], 0.9)
    else:
        raise NotImplementedError

    # отобрать те батчи, где количество точек полигона в батче больше определенного порога.
    activated_intervals = np_intersected_interval_stats[np_intersected_interval_stats[:, 2] > threshold]
    if return_type == 'list':
        activated_intervals = activated_intervals.tolist()
    return activated_intervals


def get_separated_intervals(activated_intervals):
    divided_activated_intervals = []
    right_bound = None
    for idx in range(len(activated_intervals)):

        # если нулевой элемент списка - автоматом добавляем в итоговый список
        if idx == 0:
            interval = [activated_intervals[idx]]
            right_bound = activated_intervals[idx][1]
        else:
            # если правая граница итогового списка больше левой границы следующего элемента из проверяемого списка -
            if right_bound >= activated_intervals[idx][0]:  # INTERSECTION
                # есть пересечение между интервалами и необходимо обновить правую границу итогового списка
                interval.append(activated_intervals[idx])  # итоговый список
                right_bound = activated_intervals[idx][1]  # обновленная правая граница итогового списка

            else:  # NO INTERSECTION
                # нет пересечения между интервалами, добавляем новый интервал в итоговый список,
                # обновляем правую границу нового интервала
                divided_activated_intervals.append(interval)
                interval = [activated_intervals[idx]]
                right_bound = activated_intervals[idx][1]

            # если последний элемент из списка - добавить в итоговый список
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

    # получение интервалов и количества точек на каждом интервале
    intervals, n, bin_edges = get_poly_stats(coordinates, sector_len)

    # получить объединенные интервалы и общее количество точек на каждом из них
    activated_intervals = get_intersected_intervals(intervals, n, sub_interval_len, mean_multiplier=1.2, threshold_type='mean')
    try:
        # проверка на пересекающиеся интервалы. Если они пересекаются -
        divided_activated_intervals = get_separated_intervals(activated_intervals)
    except:
        return None
    # actual_intervals = get_actual_intervals(divided_activated_intervals)

    return divided_activated_intervals  # np.array(actual_intervals)


def main():
    pass


if __name__ == '__main__':
    main()
