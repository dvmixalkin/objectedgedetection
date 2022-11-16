import cv2
import numpy as np
import rasterio
import shapely
from rasterio import features
from shapely.geometry import Polygon


def eliminate_holes_and_tiny_objects(target_mask, width, height, eps=None, store_single=True, return_type='polygon',
                                     debug=False):
    assert return_type in ['polygon', 'mask'], 'Specify correct return type.'
    all_polygons = []
    for shape, value in features.shapes(target_mask.astype(np.uint8), mask=(target_mask > 0),
                                        transform=rasterio.Affine(1.0, 0, 0, 0, 1.0, 0)):
        all_polygons.append(shapely.geometry.shape(shape))

    if store_single and all_polygons != []:
        actual_polygon = None
        previous_max_area = 0
        for polygon in all_polygons:
            if polygon.area > previous_max_area:
                actual_polygon = polygon
                previous_max_area = polygon.area
        all_polygons = [actual_polygon]

    new_polygons = []
    for polygon in all_polygons:
        if polygon.area < 1000:
            continue
        list_interiors = []
        if eps is not None:
            for interior in polygon.interiors:
                p = Polygon(interior)
                if p.area > eps:
                    list_interiors.append(interior)
        new_polygon = Polygon(polygon.exterior.coords, holes=list_interiors)
        new_polygons.append(new_polygon)
    if return_type == 'polygon':
        result = []
        for new_polygon in new_polygons:
            x, y = new_polygon.exterior.xy
            coordinates = np.vstack([np.array(x.tolist()), np.array(y.tolist())]).transpose().astype(int)
            result.append(coordinates)
        # result = new_polygons
    else:
        result = (rasterio.features.rasterize(
            new_polygons,
            out_shape=(height, width)
        ) * 255).astype(np.uint8)

    return result