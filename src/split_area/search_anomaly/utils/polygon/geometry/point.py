from collections import namedtuple

Point = namedtuple('Point', ['x', 'y'])


# WIP
class MyPoint:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.coord = Point(x=x, y=y)

    def get_coordinates(self):
        return self.coord
