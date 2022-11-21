import numpy as np
from PIL import Image
from .utils.point_hole_eliminator import point_holes_eliminator
from .utils.eliminate_borders import eliminate_borders
from .utils.distance import filter_through_manhattan_distance, get_manhattan_distance
from .utils.intervals import prepare_intervals_v3, get_area_center_point, get_activations_for_interval
from .utils.filters import filter_polygon_by_indices
from src.ideas.visualize import vis_polygon


def scroll_poly_to_zero_point(polygon):

    # все точки полигона слева до 30го пикселя
    left30 = polygon[polygon[:, 0] <= polygon[:, 0].min() + 30]

    # все точки слева до 30 пикселей и сверху до 30 пикселей
    top30 = left30[left30[:, 1] <=left30[:, 1].min() + 30]

    # расстояние от точки (0, 0) до каждой из точек в выделенном прямоугольнике
    strange_metric = top30[:, 0] + top30[:, 1]

    # индекс ближайшей до (0, 0) точки
    min_value = np.argmin(strange_metric)

    # индекс ближайшей к (0, 0) из большого полигона
    start_index = np.where((polygon[:, 0] == top30[min_value][0]) * (polygon[:, 1] == top30[min_value][1]))[0][0]

    # прокрученный полигон"
    scrolled_polygon = np.vstack([polygon[start_index:], polygon[:start_index]])
    return scrolled_polygon


def search_anomaly_v3(image, polygon, step_size=20, debug=False):
    # убрать дырки между точками полигона(сильное уплотнение - 1+1)
    dense_polygon = point_holes_eliminator(polygon.copy(), interval=1)
    filtered_polygon = filter_through_manhattan_distance(dense_polygon, min_distance=5)  # 'auto'
    # обрезка полигона по краям (чтобы не было статистики на основе боковых линий) 15% ширины слева и справа(максимум 30 пикселей)
    polygon_wo_borders = eliminate_borders(dense_polygon, border_fraction=0.15, pixel_limit=30)

    # получить интервалы
    polygon_wo_borders_threshold5 = filter_through_manhattan_distance(polygon_wo_borders, min_distance=5)  # 'auto'
    x_hists = prepare_intervals_v3(
        polygon_wo_borders_threshold5,
        step_size,
        image=image
    )
    alpha = 25
    new_intervals = []
    for inter_ in x_hists:
        point_ = get_area_center_point(polygon_wo_borders_threshold5, np.array(inter_), image=image)
        x_min, x_max = point_[0] - alpha, point_[0] + alpha
        # vis_polygon(image, polygon, x_bounds_coordinates=[x_min, x_max], hole_point_coordinate=point_)

        # ДЛЯ ОТРИСОВКИ ВО ВРЕМЯ ДЕБАГГИНГА
        # polygon1 = filter_polygon_by_indices(polygon_wo_borders_threshold5, interval=[x_min, x_max])
        # y_min, y_max = polygon1[0, :, 1].min(), point_[1]
        # vis_polygon(
        #     image, polygon1,
        #     is_closed=False,
        #     x_bounds_coordinates=[x_min, x_max],
        #     y_bounds_coordinates=[y_min, y_max],
        #     hole_point_coordinate=point_
        # )
        new_intervals.append([x_min, x_max])

    working_data = filtered_polygon[0].copy()
    # working_data = scroll_poly_to_zero_point(working_data)
    resulting_poly_list = []
    # image_to_draw = image.copy()
    image_to_draw = np.array(Image.fromarray(image).convert('RGB'))
    for peak in new_intervals:  # peaks:
        x_min, x_max = peak
        if isinstance(working_data, list):
            working_data = np.ascontiguousarray(working_data)
        working_data = scroll_poly_to_zero_point(working_data)
        # vis_polygon(image, dense_polygon, x_bounds_coordinates=peak[:2])

        indexes_inner = ((working_data[:, 0] >= x_min).astype(np.int) *
                         (working_data[:, 0] <= x_max).astype(np.int)).astype(bool)
        dense_points_outer, dense_points_inner = get_activations_for_interval(indexes_inner)
        if len(dense_points_outer) != 3 or len(dense_points_inner) != 2:
            continue
            # raise NotImplemented
            vis_polygon(image, working_data[dense_points_inner[0]])

        # верхняя линия точек
        upper_points = working_data[dense_points_inner[0]]
        # нижняя линия точек
        lower_points = working_data[dense_points_inner[1]]
        distance_matrix = np.zeros((upper_points.shape[0], lower_points.shape[0]))  # lines - columns
        for i_x, upper_point in enumerate(upper_points):
            for i_y, lower_point in enumerate(lower_points):
                distance_matrix[i_x, i_y] = get_manhattan_distance(upper_point, lower_point)

        # попытка найти 2 точки для разделния по прямой
        # кандидат с верхней гряды точек для разделения
        top = np.argmin(distance_matrix.min(axis=1))
        # кандидат с нижней гряды точек для разделения
        bot = np.argmin(distance_matrix[top])
        dilation = 1
        upper_left, upper_right = upper_points[:top - dilation], upper_points[top + dilation:]
        lower_right, lower_left = lower_points[:bot - dilation], lower_points[bot + dilation:]

        left_bound_top = working_data[dense_points_outer[0]].tolist()  # dense_points_outer_new
        left_bound_bot = working_data[dense_points_outer[2]].tolist()  # dense_points_outer_new
        right_bound_mid = working_data[dense_points_outer[1]].tolist()

        first_polygon = left_bound_top + upper_left.tolist() + lower_left.tolist() + left_bound_bot
        x_f = np.array(first_polygon)[:, 0]
        y_f = np.array(first_polygon)[:, 1]
        area_f = (x_f.max() - x_f.min()) * (y_f.max() - y_f.min())
        second_polygon = upper_right.tolist() + right_bound_mid + lower_right.tolist()
        x_s = np.array(second_polygon)[:, 0]
        y_s = np.array(second_polygon)[:, 1]
        area_s = (x_s.max() - x_s.min()) * (y_s.max() - y_s.min())

        filtered_poly = first_polygon if area_f < area_s else second_polygon
        working_data = first_polygon if area_f >= area_s else second_polygon
        resulting_poly_list.append(filtered_poly)
        # vis_polygon(image, filtered_poly)
        # image_to_draw = cv2.polylines(
        #     img=image_to_draw.astype(np.uint8),
        #     pts=[np.asarray(filtered_poly)],
        #     isClosed=True,
        #     color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
        #     thickness=3)

    if isinstance(working_data, np.ndarray):
        if len(working_data.shape) == 3:
            working_data = working_data[0]
        working_data = working_data.tolist()
    # image_to_draw = cv2.polylines(
    #     img=image_to_draw.astype(np.uint8),
    #     pts=[np.asarray(working_data)],
    #     isClosed=True,
    #     color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
    #     thickness=3)
    # Image.fromarray(image_to_draw).show()
    resulting_poly_list.append(working_data)
    return resulting_poly_list
