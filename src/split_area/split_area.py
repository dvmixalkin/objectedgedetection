from .search_anomaly import search_anomaly_v1, search_anomaly_v2, search_anomaly_v3
from .search_anomaly.utils import MyPolygon, get_np_points_from_polygon
from src.ideas.visualize import vis_image, vis_polygon,  vis_contours

# priority_labeler_list = [
#     'КОМП', 'Александр', 'kayla', 'Сергей', 'Пользователь', 'Сергей', '2002kh', '2002k', 'user', 'asus',
#     'HOMEr', 'kiyh', 'Крылова Виктория', 'Home', 'банан228', 'Jokerit55', 'user', 'marw1xx',
#     'vladimir', 'Светлана', 'Бедолага', 'marys', 'olgal', 'artem', 'ольга', 'Ольга', 'SPECIALIST'
# ]


def substract_intersected_polygons(world_coordinate_polygons):
    poly_objects = [MyPolygon(world_coordinate_polygon) for world_coordinate_polygon in world_coordinate_polygons]
    num_polygons = len(poly_objects)
    for i_idx in range(num_polygons):
        for j_idx in range(num_polygons):
            if i_idx > j_idx:
                intersected_poly = poly_objects[i_idx].intersect_with(poly_objects[j_idx])
                if intersected_poly is not None:
                    if poly_objects[i_idx].get_area() > poly_objects[j_idx].get_area():
                        poly_objects[i_idx] = poly_objects[i_idx].sub(poly_objects[j_idx])
                    else:
                        poly_objects[j_idx] = poly_objects[j_idx].sub(poly_objects[i_idx])
    return poly_objects


def refine_polygons(image, polygons, debug=False, version=3):
    if not isinstance(polygons, list):
        polygons = [polygons]
    res_poly_list = []
    for idx, polygon_ in enumerate(polygons):
        if polygon_.area < 1000:
            continue

        # @TODO Waiting
        poly = get_np_points_from_polygon(polygon_)
        if version == 1:
            # @TODO Waiting
            poly = search_anomaly_v1(image, poly, debug=debug)
        elif version == 2:
            # @TODO Waiting
            poly = search_anomaly_v2(image, poly, debug=debug)
        elif version == 3:
            # @TODO WIP
            poly = search_anomaly_v3(image, poly, debug=debug)
            # poly = search_anomaly_v3(image, polygon_, debug=debug)
        else:
            raise NotImplemented

        res_poly_list.extend(poly)
    return res_poly_list


def convert_mask_to_polygon(mask):
    raise NotImplemented
    polygons = 1
    return polygons


def split_area(image, polygons):
    # vis_contours(image, polygons)
    poly_list = refine_polygons(image, polygons, debug=False, version=3)
    return poly_list
