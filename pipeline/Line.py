


@jitclass(spec=[
            ("x0", nb.types.float32), ("y0", nb.types.float32), ("x1", nb.types.float32), ("y1", nb.types.float32), 
            ("dx", nb.types.float32), ("dy", nb.types.float32),
            ("dead", nb.types.boolean),
            ("length", nb.types.float32),
            ("angle", nb.types.float32),
            ("midpoint", nb.types.UniTuple(nb.types.float32, 2)),
            ])
class Line:
    def __init__(self, x0, y0, x1, y1):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1

        self.dead = False

        self.dx = self.x1 - self.x0
        self.dy = self.y1 - self.y0
        self.length = math.sqrt(self.dx * self.dx + self.dy * self.dy)
        self.angle = line_angle(x0, y0, x1, y1)
        self.midpoint = ((x0 + x1) / 2, (y0 + y1) / 2)

    def __getitem__(self, i):
        return self.data[i]

    def __len__(self):
        return len(self.data)

    @property
    def point_a(self):
        return (self.x0, self.y0)

    @property
    def point_b(self):
        return (self.x1, self.y1)

    def draw(self, img, color=(255,50,255,255), thickness=2):
        cv2.line(img, self.point_a, self.point_b, color, thickness)    

