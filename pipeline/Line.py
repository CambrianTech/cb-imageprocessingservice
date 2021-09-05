import math
import numpy as np
import numba as nb
import cv2
from numba.experimental import jitclass
from collections.abc import Sequence
from scipy.spatial import distance
from bisect import bisect_left, bisect_right
from cambrian.LineFunctions import LineFunctions

# @jitclass(spec=[
#             ("x0", nb.types.float32), ("y0", nb.types.float32), ("x1", nb.types.float32), ("y1", nb.types.float32), 
#             ("dx", nb.types.float32), ("dy", nb.types.float32),
#             ("dead", nb.types.boolean),
#             ("length", nb.types.float32),
#             ("angle", nb.types.float32),
#             ("midpoint", nb.types.UniTuple(nb.types.float32, 2)),
#             ])

class Line(Sequence):
    def __init__(self, data, sx=1, sy=1):
        super().__init__()
        self.data = data
        self.data[0] *= sx
        self.data[2] *= sx
        self.data[1] *= sy
        self.data[3] *= sy
        self.recalculate()

        self.dead = False

    def __getitem__(self, i):
        return self.data[i]

    def __len__(self):
        return len(self.data)

    def __lt__(self, other):
        return self.angle < other.angle
    
    def __eq__(self, other):
        return self.angle == other.angle

    @property
    def point_a(self):
        return (self.data[0], self.data[1])

    @property
    def point_b(self):
        return (self.data[2], self.data[3])

    def draw(self, img, color=(255,50,255,255), thickness=2, sx=1.0, sy=1.0):
        cv2.line(img, (int(self.point_a[0] * sx), int(self.point_a[1] * sy)), (int(self.point_b[0] * sx), int(self.point_b[1] * sy)), color, thickness)

    def reshape(self, *args):
        return self.data.reshape(*args)

    def recalculate(self):
        self.midpoint = ((self.point_a[0] + self.point_b[0]) / 2, (self.point_a[1] + self.point_b[1]) / 2)
        self.length = distance.euclidean(self.point_a, self.point_b)
        self.angle = LineFunctions.line_angle(self.point_a[0], self.point_a[1], self.point_b[0], self.point_b[1])

    def bounding_box(self, width, length_multiplier=1.0):
        return (self.midpoint, (self.length * length_multiplier, width), np.degrees(self.angle))

    @classmethod
    def draw_all(cls, lines, img, color=(255,50,255,255), thickness=2, sx=1.0, sy=1.0):
        [line.draw(img, color=color, thickness=thickness, sx=sx, sy=sy) for line in lines]

    # def bounding_box_points(self, width, length_multiplier=1.0):
    #     return rotated_rects_points(self.midpoint, (self.length * length_multiplier, width), self.angle)

#todo: write in C or lambda
def merge(lines, search_width, search_length=1.01, angle_threshold=math.radians(3)):

    #lines = sorted(lines)
    min_dist_sq = search_width * search_width

    for i in range(len(lines)):
        line_a = lines[i]

        if line_a.dead: continue

        rect_a = line_a.bounding_box(search_width, length_multiplier=search_length)
        data = (line_a.point_a, line_a.point_b)

        for j in range(len(lines)):
            if i == j: continue

            line_b = lines[j]

            #Optimization possible: line_angle_difference should not be required by bisect methods above returning only angles in range
            if line_b.dead or LineFunctions.line_angle_difference(line_a.angle, line_b.angle) > angle_threshold: continue

            dist_sq = distance.sqeuclidean(line_a.midpoint, line_b.midpoint)

            if dist_sq <= min_dist_sq:
                result = 1
            else:
                rect_b = line_b.bounding_box(search_width, length_multiplier=search_length)
                result, _ = cv2.rotatedRectangleIntersection(rect_a, rect_b)

            if result != 0:
                line_a.dead = True
                line_b.dead = True
                data = LineFunctions.merge_lines(data, (line_b.point_a, line_b.point_b))


        if line_a.dead:
            lines[i] = Line(np.array([(data[0][0], data[0][1], data[1][0], data[1][1])], dtype=np.int).reshape(4))

    return list(filter(lambda x: not x.dead, lines))


@nb.jit(nopython=True)
def line_angle(x0, y0, x1, y1):
    return math.atan2(float(y1 - y0), float(x1 - x0))

def draw_line(line, img, color=(255,50,255,255), thickness=2):
    cv2.line(img, (int(line.point_a[0]), int(line.point_a[1])), (int(line.point_b[0]), int(line.point_b[1])), color, thickness)

@nb.jit(nopython=True)
def line_angle_difference(x, y): #minimum angle between lines segments cannot differ by more than 90 degrees
    diff = abs(math.atan2(math.sin(x-y), math.cos(x-y)))
    if diff > 0.5 * math.pi:
        diff = math.pi - diff

    return diff

def line_angle(x0, y0, x1, y1):
    #return np.arctan2(y1 - y0, x1 - x0)
    return math.atan2(float(y1 - y0), float(x1 - x0))

@nb.jit(nopython=True)
def bounding_box(line, width, length_multiplier=1.0):
    return (line.midpoint, (line.length * length_multiplier, width), np.degrees(line.angle))

@nb.jit(nopython=True)
def sqeuclidean(point_a, point_b):
    dx = point_b[0] - point_a[0]
    dy = point_b[1] - point_a[1]
    return dx * dx + dy * dy

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

#(line.midpoint, (line.length * length_multiplier, width), np.degrees(line.angle))
@nb.jit(nopython=True)
def rotated_rects_points(center, size, angle):
    b = math.cos(angle)
    a = math.sin(angle)

    pt = np.empty((4,2))
    width = size[0]
    height = size[1]

    pt[0][0] = center[0] - a*height - b*width
    pt[0][1] = center[1] + b*height - a*width
    pt[1][0] = center[0] + a*height - b*width
    pt[1][1] = center[1] - b*height - a*width
    pt[2][0] = 2*center[0] - pt[0][0]
    pt[2][1] = 2*center[1] - pt[0][1]
    pt[3][0] = 2*center[0] - pt[1][0]
    pt[3][1] = 2*center[1] - pt[1][1]
    return pt

@nb.jit(nopython=True)
def rotated_rects_intersect(pts1, area1, pts2, area2):
    # L2 metric const float samePointEps = std::max(1e-16f, 1e-6f * (float)std::max(rect1.size.area(), rect2.size.area()));
    samePointEps = max(1e-16, 1e-6 * max(area1, area2));

    same = True;
    for i in range(4):
        if (abs(pts1[i][0] - pts2[i][0]) > samePointEps or (abs(pts1[i][1] - pts2[i][1]) > samePointEps) ):
            same = False;
            break;        
    
    if same: 
        return True

    vec1 = np.empty((4,2))
    vec2 = np.empty((4,2))

    # Line vector
    # A line from p1 to p2 is: p1 + (p2-p1)*t, t=[0,1]
    for i in range(4):
        _i = (i+1)%4
        test = pts1[_i][0] - pts1[i][0]
        vec1[i][0] = pts1[_i][0] - pts1[i][0];
        vec1[i][1] = pts1[_i][1] - pts1[i][1];

        vec2[i][0] = pts2[_i][0] - pts2[i][0];
        vec2[i][1] = pts2[_i][1] - pts2[i][1];

    # Line test - test all line combos for intersection
    for i in range(4):
        for j in range(4):
            #// Solve for 2x2 Ax=b
            x21 = pts2[j][0] - pts1[i][0];
            y21 = pts2[j][1] - pts1[i][1];

            vx1 = vec1[i][0];
            vy1 = vec1[i][1];

            vx2 = vec2[j][0];
            vy2 = vec2[j][1];

            det = vx2*vy1 - vx1*vy2;

            if (det == 0):
                continue

            t1 = (vx2*y21 - vy2*x21) / det
            t2 = (vx1*y21 - vy1*x21) / det

            if (np.isnan(t1) or np.isnan(t2)):
                continue
            
            #// This takes care of parallel lines
            if ( t1 >= 0.0 and t1 <= 1.0 and t2 >= 0.0 and t2 <= 1.0 ):
                xi = pts1[i][0] + vec1[i][0]*t1
                return True

    return verts_inside(pts2, pts1, vec2) or verts_inside(pts1, pts2, vec1)

@nb.jit(nopython=True)
def verts_inside(pts1, pts2, vec1):
    for i in range(4):
        #// We do a sign test to see which side the point lies.
        #// If the point all lie on the same sign for all 4 sides of the rect,
        #// then there's an intersection
        posSign = 0;
        negSign = 0;

        x = pts2[i][0];
        y = pts2[i][1];

        for j in range(4):
            #// line equation: Ax + By + C = 0
            #// see which side of the line this point is at
            A = -vec1[j][1];
            B = vec1[j][0];
            C = -(A*pts1[j][0] + B*pts1[j][1]);

            s = A*x + B*y + C;

            if ( s >= 0 ): posSign+=1;
            else: negSign+=1;

        if ( posSign == 4 or negSign == 4 ):
            return True

    return False


