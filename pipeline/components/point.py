class Point(tuple):
    def __new__(cls, x, y=None):
        if y is not None:
            return Point.__new__(cls, (x, y))
        return tuple.__new__(cls, x)

    @property
    def x(self):
        return self[0]

    @property
    def y(self):
        return self[1]