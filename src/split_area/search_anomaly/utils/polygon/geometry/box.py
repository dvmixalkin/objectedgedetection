class Box:
    def __init__(self, coordinates):
        assert coordinates[2] > coordinates[0], 'x_max should be greater than x_min'
        assert coordinates[3] > coordinates[1], 'y_max should be greater than y_min'
        self.x_min = coordinates[0]
        self.y_min = coordinates[1]
        self.x_max = coordinates[2]
        self.y_max = coordinates[3]

    def get_coordinates(self):
        return [self.x_min, self.y_min, self.x_max, self.y_max]

    def get_width(self):
        return self.x_max - self.x_min

    def get_height(self):
        return self.y_max - self.y_min

    def get_area(self):
        return self.get_width() * self.get_height()

    def intersect_with(self, box):
        x_min, y_min, x_max, y_max = box.get_coordinates()
        x1 = min(self.x_max, x_max)
        y1 = min(self.y_max, y_max)
        x2 = max(self.x_min, x_min)
        y2 = max(self.y_min, y_min)
        intersection = max(0, x1 - x2) * max(0, y1 - y2)
        return intersection

    def union_with(self, box):
        x_max, y_max, x_min, y_min = box.get_coordinates()
        area1 = (self.x_max - self.x_min) * (self.y_max - self.y_min)
        area2 = (x_max - x_min) * (y_max - y_min)
        intersect_area = self.intersect_with(box)
        return area1 + area2 - intersect_area


def print_status(method, result):
    if result == 'passed':
        print(f'{method} successfully passed.')
    else:
        print(f'{method} failed.')


if __name__ == '__main__':
    box1 = Box([1, 2, 3, 4])
    box2 = Box([1, 2, 3, 4])
    if box1.intersect_with(box2) == 4:
        print_status('intersection', 'passed')
    else:
        print_status('intersection', 'failed')
