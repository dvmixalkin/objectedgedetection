import numpy as np


def get_peaks_v1(coordinates, interval=100, k=1.1):
    # split into bins and get statistic by coordinates
    min_ = coordinates.min()
    max_ = coordinates.max()
    difference = max_ - min_
    n_bins = difference // interval
    bins_with_activations = []
    for i in range(int(n_bins * k)+1):
        interval_min = min_ + i * int(interval/k)
        activated_min = coordinates > interval_min
        interval_max = interval_min + interval
        if i == n_bins - 1:
            interval_max = max_
        activated_max = coordinates < interval_max
        activated_num = (activated_min * activated_max).sum()
        bins_with_activations.append([interval_min, interval_max, activated_num])
    return np.array(bins_with_activations)


def main():
    pass


if __name__ == '__main__':
    main()
