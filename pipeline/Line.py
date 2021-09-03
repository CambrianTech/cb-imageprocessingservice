import math
import numpy as np
import numba as nb
from numba.experimental import jitclass

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


@nb.jit(nopython=True)
def line_angle_difference(x, y): #minimum angle between lines segments cannot differ by more than 90 degrees
    diff = abs(math.atan2(math.sin(x-y), math.cos(x-y)))
    if diff > 0.5 * math.pi:
        diff = math.pi - diff

    return diff

@nb.jit(nopython=True)
def line_angle(x0, y0, x1, y1):
    #return np.arctan2(y1 - y0, x1 - x0)
    return math.atan2(float(y1 - y0), float(x1 - x0))

@nb.jit(nopython=True)
def bounding_box(line, width, length_multiplier=1.0):
    return (line.midpoint, (line.length * length_multiplier, width), np.degrees(line.angle))

@nb.jit(nopython=True)
def merge_line_pair(line_a, line_b):

    ax = line_a[0][0]
    ay = line_a[0][1]
    bx = line_a[1][0]
    by = line_a[1][1]

    cx = line_b[0][0]
    cy = line_b[0][1]
    dx = line_b[1][0]
    dy = line_b[1][1]
    
    thi = 0.0; 
    thj = 0.0; 
    thr = 0.0;

    dlix = (bx - ax);
    dliy = (by - ay);
    dljx = (dx - cx);
    dljy = (dy - cy);

    li = math.sqrt((dlix * dlix) + (dliy * dliy));
    lj = math.sqrt((dljx * dljx) + (dljy * dljy));

    xg = (li * (ax + bx) + lj * (cx + dx)) / (2.0 * (li + lj));
    yg = (li * (ay + by) + lj * (cy + dy)) / (2.0 * (li + lj));

    if (dlix == 0.0): thi = math.pi / 2.0;
    else: thi = math.atan(dliy / dlix);

    if (dljx == 0.0): thj = math.pi / 2.0;
    else: thj = math.atan(dljy / dljx);

    if abs(thi - thj) <= math.pi / 2.0:
        thr = (li * thi + lj * thj) / (li + lj);
    else:
        tmp = thj - math.pi * (thj / abs(thj));
        thr = li * thi + lj * tmp;
        thr /= (li + lj);

    sin_thr = math.sin(thr)
    cos_thr = math.cos(thr)

    axg = (ay - yg) * sin_thr + (ax - xg) * cos_thr;
    bxg = (by - yg) * sin_thr + (bx - xg) * cos_thr;
    cxg = (cy - yg) * sin_thr + (cx - xg) * cos_thr;
    dxg = (dy - yg) * sin_thr + (dx - xg) * cos_thr;

    delta1xg = min(axg,min(bxg,min(cxg,dxg)));
    delta2xg = max(axg,max(bxg,max(cxg,dxg)));

    delta1x = delta1xg * cos_thr + xg;
    delta1y = delta1xg * sin_thr + yg;
    delta2x = delta2xg * cos_thr + xg;
    delta2y = delta2xg * sin_thr + yg;

    return (delta1x, delta1y), (delta2x, delta2y)  

