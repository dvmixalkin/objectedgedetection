import numpy as np
from collections import Counter, OrderedDict


def get_peaks_v2(coordinates):
    x = Counter(coordinates)
    x_new = OrderedDict()
    keys = list(x.keys())
    keys.sort()
    for key in keys:
        # if x[key] >= 2:
        #     x_new[key] = x[key]
        x_new[key] = x[key]

    def collapse_dense_points(array):
        dense_list = []

        interval = [None, None, None]
        last_value = None

        for k, v in array.items():
            if last_value is not None:
                if abs(k - last_value) > 1:
                    if interval[1] is not None:
                        if interval[1] - interval[0] >= 1:
                            dense_list.append(interval)
                    interval = [k, None, v]
                else:
                    interval[1] = k
                    interval[2] += v
                last_value = k
            else:
                last_value = k
                interval = [k, None, v]

        np_dense_list = np.array(dense_list)
        threshold = np.quantile(np_dense_list[:, 2], 0.975)
        collapsed_list = np_dense_list[np_dense_list[:, 2] >= threshold]

        return collapsed_list

    result = collapse_dense_points(x_new)

    return result  # coordinates


def main():
    pass


if __name__ == '__main__':
    main()
