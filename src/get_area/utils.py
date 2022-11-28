import numpy as np
import rasterio
import shapely
from rasterio import features
from shapely.geometry import Polygon
from src.split_area import get_np_points_from_polygon


def mask2poly(mask):
    all_polygons = []
    for shape, value in features.shapes(mask.astype(np.uint8), mask=(mask > 0),
                                        transform=rasterio.Affine(1.0, 0, 0, 0, 1.0, 0)):
        all_polygons.append(shapely.geometry.shape(shape))
    return all_polygons


def mask2coordinates(mask):
    polygon = mask2poly(mask)
    coordinates = get_np_points_from_polygon(polygon)
    return coordinates


def poly2mask(polygons, shape):
    binary_mask = rasterio.features.rasterize(
        polygons,
        out_shape=shape
    )
    return (binary_mask * 255).astype(np.uint8)


def poly2coordinate(polygon):
    return get_np_points_from_polygon(polygon)


def coordinate2polygon(coordinate):
    return Polygon(coordinate)


def coordinate2mask(coordinate):
    polygon = coordinate2polygon(coordinate)
    return poly2mask(polygon)
