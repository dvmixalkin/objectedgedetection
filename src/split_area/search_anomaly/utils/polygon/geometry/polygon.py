import numpy as np
import rasterio
from rasterio import features
import shapely
from shapely.geometry import Polygon
from PIL import Image


def import_from_Polygon(polygon: Polygon):
    x, y = polygon.exterior.xy
    x = np.array(x)
    y = np.array(y)
    return np.stack([x, y], axis=1).astype(int)


class MyPolygon:
    def __init__(self, coordinates=None):
        if coordinates is not None:
            if not isinstance(coordinates, np.ndarray):
                try:
                    coordinates = np.asarray(coordinates)
                except:
                    raise NotImplemented
            self.coordinates = coordinates
            self.polygon = self.get_polygon(source='coordinates')
            self.mask = self.get_mask(source='polygon')
        else:
            self.coordinates = None
            self.polygon = None
            self.mask = None

    def get_area(self):
        return self.polygon.area

    def get_coordinates(self, source='coordinates'):
        assert source in ['coordinates', 'polygon'], 'Specify correct return type: polygon or mask.'
        if source == 'coordinates':
            return self.coordinates
        else:
            return import_from_Polygon(self.polygon)
            # x, y = self.polygon.exterior.xy
            # x = np.array(x)
            # y = np.array(y)
            # return np.stack([x, y], axis=1).astype(int)

    def get_polygon(self, source='coordinates'):
        assert source in ['coordinates', 'polygon', 'mask'], 'Specify correct return type: polygon or mask.'
        if source == 'coordinates':
            return Polygon(self.coordinates)
        elif source == 'polygon':
            raise NotImplemented
        else:
            all_polygons = []
            for shape, value in features.shapes(self.mask.astype(np.uint8), mask=(self.mask > 0),
                                                transform=rasterio.Affine(1.0, 0, 0, 0, 1.0, 0)):
                all_polygons.append(shapely.geometry.shape(shape))
            try:
                areas = [f.area for f in all_polygons]
                index = areas.index(max(areas))
                return MyPolygon(import_from_Polygon(all_polygons[index]))
            except:
                return None

    def get_mask(self, source='coordinates', get_shape=False):
        assert source in ['coordinates', 'polygon', 'mask'], 'Specify correct return type: polygon or mask.'
        coordinates = self.get_coordinates()
        width = coordinates[:, 0].max()
        height = coordinates[:, 1].max()
        polygon = rasterio.features.rasterize(
            [Polygon(shapes) for shapes in [coordinates]], out_shape=(height, width))

        if get_shape:
            return polygon, (width, height)
        else:
            return polygon

    def intersect_with(self, poly, mode='mask', return_type='polygon'):
        assert mode in ['coordinates', 'polygon', 'mask'], 'Specify correct return type: polygon or mask.'
        assert return_type in ['coordinates', 'polygon', 'mask'], 'Specify correct return type: polygon or mask.'
        if mode == 'coordinates':
            raise NotImplemented
        elif mode == 'polygon':
            raise NotImplemented
        else:
            poly_1, (x1_max, y1_max) = poly.get_mask(source='polygon', get_shape=True)
            poly_2, (x2_max, y2_max) = self.get_mask(source='polygon', get_shape=True)
            width = max(x1_max, x2_max)
            height = max(y1_max, y2_max)

            mask1 = np.zeros(shape=(height, width))
            mask1[:y1_max, :x1_max] = poly_1
            mask2 = np.zeros(shape=(height, width))
            mask2[:y2_max, :x2_max] = poly_2

            intersection = mask1 * mask2
            if return_type == 'mask':
                return intersection

            # x, y = self.get_polygon(source='mask').exterior.xy
            new_poly = MyPolygon()
            new_poly.mask = intersection
            polygon_from_mask = new_poly.get_polygon(source='mask')
            if polygon_from_mask is not None:
                coordinates = polygon_from_mask.get_coordinates()
                # coordinates = import_from_Polygon(polygon_from_mask)
                # x, y = polygon_from_mask.exterior.xy
                # x = np.array(x)
                # y = np.array(y)
                # coordinates = np.stack([x, y], axis=1).astype(int)
                if return_type == 'coordinates':
                    return coordinates
                return MyPolygon(coordinates)
            else:
                return None

    def reshape_as(self, target_polygon):
        poly_height, poly_width = target_polygon.shape
        self_height, self_width = self.mask.shape
        draft_mask = np.zeros(
            shape=(
                max(poly_height, self_height),  # height,
                max(poly_width, self_width),  # width
            )
        )
        h, w = self.mask.shape
        draft_mask[:h, :w] = self.mask
        self.mask = draft_mask

    # @TODO WIP
    def sub(self, poly, return_type='polygon'):
        # вычесть poly из self
        assert return_type in ['polygon', 'mask'], 'Specify correct return type: polygon or mask.'
        intersection = self.intersect_with(poly, mode='mask', return_type='mask')

        poly_height, poly_width = intersection.shape
        self_height, self_width = self.mask.shape
        shape = (
            max(poly_height, self_height),  # height,
            max(poly_width, self_width),  # width
        )
        tmp_mask1 = np.zeros(shape)
        height, width = intersection.shape
        tmp_mask1[:height, :width] = intersection
        inverted_intersection_mask = np.invert(tmp_mask1.astype(bool))

        tmp_mask2 = np.zeros(shape)
        height, width = self.mask.shape
        tmp_mask2[:height, :width] = self.mask
        self.mask = tmp_mask2.astype(bool)

        substracted_mask = (self.mask * inverted_intersection_mask).astype(np.uint8)
        if return_type == 'mask':
            return substracted_mask
        substracted_object = MyPolygon()
        substracted_object.mask = substracted_mask
        if return_type == 'polygon':
            return substracted_object.get_polygon(source='mask')
        else:
            return substracted_object.get_coordinates(source='polygon')


def save(polygon, save_path):
    polygons = [Polygon(shapes) for shapes in [polygon]]
    gt = rasterio.features.rasterize(polygons, out_shape=(100, 100))
    Image.fromarray(gt * 255).save(save_path)


if __name__ == '__main__':
    poly1 = MyPolygon([[0, 0], [10, 10], [20, 20], [30, 30], [40, 40], [50, 30], [60, 20], [70, 10], [80, 0]])
    poly2 = MyPolygon([[0, 40], [10, 30], [20, 20], [30, 10], [40, 0], [50, 10], [60, 20], [70, 30], [80, 40]])
    poly3 = MyPolygon([[0, 40], [40, 0], [80, 40]])
    poly4 = MyPolygon([[30, 70], [40, 50], [60, 60], [70, 80], [60, 90], [40, 80]])
    poly5 = MyPolygon([[60, 50], [80, 50], [80, 100], [60, 100]])
    poly6 = MyPolygon([[70, 60], [80, 40], [90, 50], [100, 70], [70, 90], [80, 80], [70, 70]])

    save(poly4.intersect_with(poly5), 'intersect_res.png')
    save(poly5.sub(poly4), 'sub_res54.png')
    save(poly4.sub(poly5), 'sub_res45.png')
