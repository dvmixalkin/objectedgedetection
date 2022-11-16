import numpy as np
import random


def point_holes_eliminator(data_orig, interval=5):
    new_polygon = []
    data = data_orig[0]
    if isinstance(interval, str):
        try:
            interval = int(interval)
        except:
            interval = 5
    for pp in range(data.shape[0] - 1):
        x = abs(data[pp][0] - data[pp + 1][0])
        y = abs(data[pp][1] - data[pp + 1][1])
        interval_value = interval if isinstance(interval, int) else random.randint(1, interval)
        add_points_x = x // interval_value
        add_points_y = y // interval_value

        if add_points_x > 1 or add_points_y > 1:
            if add_points_x > add_points_y:
                x_arr = [i for i in range(add_points_x)]
                y_arr = [int(add_points_y * (i / add_points_x)) for i in range(add_points_x)]
            elif add_points_x < add_points_y:
                x_arr = [int(add_points_x * (i / add_points_y)) for i in range(add_points_y)]
                y_arr = [i for i in range(add_points_y)]
            else:
                x_arr = [i for i in range(add_points_x)]
                y_arr = [i for i in range(add_points_y)]

            for x_i, y_i in zip(x_arr, y_arr):
                if data[pp][0] < data[pp + 1][0]:
                    tmp_x = data[pp][0] + x_i * interval
                else:
                    tmp_x = data[pp][0] - x_i * interval

                if data[pp][1] < data[pp + 1][1]:
                    tmp_y = data[pp][1] + y_i * interval
                else:
                    tmp_y = data[pp][1] - y_i * interval

                # print(tmp_x, tmp_y)
                new_polygon.append(np.array([tmp_x, tmp_y]))
        else:
            new_polygon.append(data[pp])
        # print(new_polygon)
    return np.stack([new_polygon])
