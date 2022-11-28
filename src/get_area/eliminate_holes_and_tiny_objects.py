import numpy as np
from shapely.geometry import Polygon
from .utils import mask2poly, poly2mask, poly2coordinate


def convert2target_format(new_polygons, shape=None, return_type='polygon'):
    if return_type == 'polygon':
        return new_polygons
    elif return_type == 'mask':
        if shape is None:
            raise 'Specify mask size.'
        return poly2mask(new_polygons, shape)
    else:
        return poly2coordinate(new_polygons)


def eliminate_holes(polygons, shape, eps=None, return_type='polygon'):
    assert return_type in ['polygon', 'mask', 'coordinate'], 'Specify return type.'
    if isinstance(polygons, np.ndarray):
        polygons = mask2poly(mask=polygons)

    new_polygons = []
    for polygon in polygons:
        list_interiors = []
        if eps is not None:
            for interior in polygon.interiors:
                p = Polygon(interior)
                if p.area > eps:
                    list_interiors.append(interior)
        new_polygon = Polygon(polygon.exterior.coords, holes=list_interiors)
        new_polygons.append(new_polygon)

    return convert2target_format(new_polygons, shape, return_type)


def eliminate_tiny_objects(polygons, shape, min_size=1000, return_type='polygon'):
    assert return_type in ['polygon', 'mask', 'coordinate'], 'Specify return type.'
    if isinstance(polygons, np.ndarray):
        polygons = mask2poly(mask=polygons)

    new_polygons = []
    for polygon in polygons:
        if polygon.area < min_size:
            continue
        new_polygons.append(polygon)

    return convert2target_format(new_polygons, shape, return_type)


def eliminate_holes_and_tiny_objects(
        target_mask, eps=None, store_single=True, return_type='polygon'):

    assert return_type in ['polygon', 'mask', 'coordinates'], 'Specify correct return type.'
    all_polygons = mask2poly(mask=target_mask)

    if all_polygons != []:
        areas = [polygon.area for polygon in all_polygons]
        if store_single:
            all_polygons = all_polygons[areas.index(max(areas))]
        else:
            max_area = max(areas)
            all_polygons = [all_polygons[idx] for idx, area in enumerate(areas) if area / max_area > 0.005]
    else:
        return  # target_mask

    new_polygons = eliminate_tiny_objects(all_polygons, target_mask.shape, min_size=1000)
    new_polygons = eliminate_holes(new_polygons, target_mask.shape, eps=eps)
    return convert2target_format(new_polygons, target_mask.shape, return_type)
