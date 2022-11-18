import numpy as np


def get_np_points_from_polygon(polygon):
    x, y = polygon.exterior.xy
    np_poly = np.expand_dims(
        np.vstack(
            [
                np.array(x.tolist()),
                np.array(y.tolist())
            ]
        ).transpose(), axis=0
    ).astype(int)
    return np_poly
