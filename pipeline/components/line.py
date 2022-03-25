import math
import numpy as np
import numba as nb
import cv2
import uuid
from numba.experimental import jitclass

from scipy.spatial import distance
from bisect import bisect_left, bisect_right
from pipeline.misc.utils import normalize
from pipeline.data.logging import Timer

def out_of_range(x, y, width, height):
    return x < 0 or y < 0 or x >= width or y >= height

@jitclass(spec=[
            ("data", nb.types.float32[:]),
            ("point_a", nb.types.UniTuple(nb.types.int32, 2)),
            ("point_b", nb.types.UniTuple(nb.types.int32, 2)), 
            ("dy", nb.types.float32),
            ("dx", nb.types.float32),
            ("length", nb.types.float32),
            ("midpoint", nb.types.UniTuple(nb.types.float32, 2)),
            ("angle", nb.types.float32),
            ("degrees", nb.types.float32),
            ("direction", nb.types.float32[:]),
            ("dead", nb.types.boolean),
            ])
class Line():
    def __init__(self, ax, ay, bx, by):
        self.data = np.array((ax, ay, bx, by), dtype=nb.types.float32)

        self.point_a = int(ax), int(ay)
        self.point_b = int(bx), int(by)
        
        self.dx = self.data[2] - self.data[0]
        self.dy = self.data[3] - self.data[1]
        self.length = euclidean(self.point_a, self.point_b)

        self.midpoint = ((ax + bx) / 2.0, (ay + by) / 2.0)
        self.angle = math.atan2(self.dy, self.dx)
        self.degrees = np.degrees(self.angle)
        self.direction = np.array((self.dy / self.length, self.dx / self.length))

        #for tracking
        self.dead = False

    def get_intersection(self, other):
        return get_line_intersection(self.point_a[0], self.point_a[1], self.point_b[0], self.point_b[1], \
                                     other.point_a[0], other.point_a[1], other.point_b[0], other.point_b[1])

    def intersects(self, other):
        return self.get_intersection(other) is not None

    def equals(self, other, epsilon=0):
        if epsilon == 0:
            return np.array_equal(self.data, other.data)
        else:
            result = np.linalg.norm(self.data - other.data)
            return result < epsilon

    def get_points(self, width, height, num_points=None):
        if num_points is None:
            num_points = int(math.ceil(self.length))

        return list(filter(lambda p: not out_of_range(p[0], p[1], width, height), np.linspace(self.point_b, self.point_a, num_points)))

    @property
    def normal_a(self):
        return np.array((-self.direction[1], self.direction[0]))

    @property
    def normal_b(self):
        return np.array((self.direction[1], -self.direction[0]))

    def closest_point(self, point):
        return closest_line_point(self.point_a[0], self.point_a[1], self.point_b[0], self.point_b[1], point[0], point[1])

    def draw(self, img, color=(255,50,255,255), thickness=1, scale=1.0, lineType=cv2.LINE_8):
        draw_line(line, img, (int(self.point_a[0] * sx), int(self.point_a[1] * scale)), (int(self.point_b[0] * sx), int(self.point_b[1] * scale)), color, thickness=thickness, lineType=lineType)

    def bounding_box(self, width, length_multiplier=1.0):
        return (self.midpoint, (self.length * length_multiplier, width), self.degrees)

    def extended(self, ratio=1.1, from_a=True, from_b=True):

        if not from_a and not from_b:
            return self

        direction = self.direction
        amount = self.length * ratio
        data = self.data.copy()

        if from_a:
            data[0] = self.midpoint[0] - direction[0] * amount
            data[1] = self.midpoint[1] - direction[1] * amount

        if from_b:
            data[2] = self.midpoint[0] + direction[0] * amount
            data[3] = self.midpoint[1] + direction[1] * amount

        return Line(data[0], data[1], data[2], data[3])        

    def copy(self):
        #todo: ineffcient
        return Line(self.data[0], self.data[1], self.data[2], self.data[3])

#jit functions, unused:
@nb.jit(nopython=True)
def line_angle(x0, y0, x1, y1):
    return math.atan2(float(y1 - y0), float(x1 - x0))

@nb.jit(nopython=True)
def line_angle_difference(x, y): #minimum angle between lines segments cannot differ by more than 90 degrees
    diff = abs(math.atan2(math.sin(x-y), math.cos(x-y)))
    if diff > 0.5 * math.pi:
        diff = math.pi - diff

    return diff

@nb.jit(nopython=True)
def closest_line_point(x0, y0, x1, y1, px, py): #minimum angle between lines segments cannot differ by more than 90 degrees
    dx, dy = x1-x0, y1-y0
    det = dx*dx + dy*dy
    a = (dy*(py-y0)+dx*(px-x0))/det
    return (x0+a*dx), (y0+a*dy)

@nb.jit(nopython=True)
def bounding_box(line, width, length_multiplier=1.0):
    return (line.midpoint, (line.length * length_multiplier, width), line.degrees)

@nb.jit(nopython=True)
def euclidean(point_a, point_b):
    dx = point_b[0] - point_a[0]
    dy = point_b[1] - point_a[1]
    return math.sqrt(dx * dx + dy * dy)

@nb.jit(nopython=True)
def sqeuclidean(point_a, point_b):
    dx = point_b[0] - point_a[0]
    dy = point_b[1] - point_a[1]
    return dx * dx + dy * dy

@nb.jit(nopython=True)
def point_on_image_edge(point, image, min_distance=3):
    edge = 0x0
    if point[0] <= min_distance:
        edge |= 0x0001
    if point[0] >= image.shape[1] - min_distance - 1:
        edge |= 0x0010
    if point[1] <= min_distance:
        edge |= 0x0100
    if point[1] >= image.shape[0] - min_distance - 1:
        edge |= 0x1000
    return edge

@nb.jit(nopython=True)
def line_on_image_edge(point_a, point_b, image, min_distance=3):
    return point_on_image_edge(point_a, image, min_distance) and point_on_image_edge(point_a, image, min_distance) & point_on_image_edge(point_b, image, min_distance) > 0

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


EPSILON = np.finfo(float).eps

#https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect
@nb.jit(nopython=True)
def get_line_intersection(p0_x, p0_y, p1_x, p1_y, p2_x, p2_y, p3_x, p3_y):

    s1_x = p1_x - p0_x;     s1_y = p1_y - p0_y;
    s2_x = p3_x - p2_x;     s2_y = p3_y - p2_y;

    det = (-s2_x * s1_y + s1_x * s2_y)

    if abs(det) < EPSILON: return None #sufficiently parallel

    s = (-s1_y * (p0_x - p2_x) + s1_x * (p0_y - p2_y)) / det;
    if (s < 0 or s > 1): return None # No collision

    t = ( s2_x * (p0_y - p2_y) - s2_y * (p0_x - p2_x)) / det;
    if (t < 0 or t > 1): return None # No collision

    #Collision detected
    return p0_x + (t * s1_x), p0_y + (t * s1_y)

@nb.jit(nopython=False)
def draw_line(line, img, color=(255,50,255,255), thickness=1, scale=1.0, lineType=cv2.LINE_8):
    cv2.line(img, (int(line.point_a[0] * sx), int(line.point_a[1] * scale)), (int(line.point_b[0] * sx), int(line.point_b[1] * scale)), color, thickness=thickness, lineType=lineType)

def draw_lines(img, lines, color=(255,50,255,255), thickness=1, scale=1.0, lineType=cv2.LINE_8):
    
    pts = np.array([line.data for line in lines])

    if scale != 1.0:
        pts = pts * scale

    #Important! keep this, very fast: 0.0007 for 512 lines vs iterating is 0.63 seconds, a thousand times slower
    pts = pts.astype(np.int32).reshape((-1,2,2))

    cv2.drawContours(img, pts, -1, color, thickness=thickness, lineType=lineType)

@nb.jit(nopython=True)
def merge_line_pair(ax, ay, bx, by, cx, cy, dx, dy, dljx, dljy):

    dlix = bx - ax
    dliy = by - ay
    #dljx = dx - cx
    #dljy = dy - cy

    li = math.sqrt((dlix * dlix) + (dliy * dliy))
    lj = math.sqrt((dljx * dljx) + (dljy * dljy))

    xg = (li * (ax + bx) + lj * (cx + dx)) / (2.0 * (li + lj))
    yg = (li * (ay + by) + lj * (cy + dy)) / (2.0 * (li + lj))

    if (dlix == 0.0): thi = math.pi / 2.0
    else: thi = math.atan(dliy / dlix)

    if (dljx == 0.0): thj = math.pi / 2.0
    else: thj = math.atan(dljy / dljx)

    if abs(thi - thj) <= math.pi / 2.0:
        thr = (li * thi + lj * thj) / (li + lj)
    else:
        tmp = thj - math.pi * (thj / abs(thj))
        thr = li * thi + lj * tmp
        thr /= (li + lj)

    sin_thr = math.sin(thr)
    cos_thr = math.cos(thr)

    axg = (ay - yg) * sin_thr + (ax - xg) * cos_thr
    bxg = (by - yg) * sin_thr + (bx - xg) * cos_thr
    cxg = (cy - yg) * sin_thr + (cx - xg) * cos_thr
    dxg = (dy - yg) * sin_thr + (dx - xg) * cos_thr

    delta1xg = min(axg, min(bxg, min(cxg,dxg)))
    delta2xg = max(axg, max(bxg, max(cxg,dxg)))

    return  delta1xg * cos_thr + xg, \
            delta1xg * sin_thr + yg, \
            delta2xg * cos_thr + xg, \
            delta2xg * sin_thr + yg

def merge_lines(lines, search_width, search_length=1.01, angle_threshold=math.radians(3)):

    #lines = [line.copy() for line in lines]

    min_dist_sq = search_width * search_width

    timer = Timer("merge_lines")
    timer.disable()

    for i in range(len(lines)):
        
        line_a = lines[i]
        if line_a.dead: continue

        rect_a = bounding_box(line_a, width=search_width, length_multiplier=search_length)
        data = line_a.data

        for line_b in lines:

            if line_angle_difference(line_a.angle, line_b.angle) > angle_threshold or line_b.dead or line_a == line_b:
                continue

            dist_sq = sqeuclidean(line_a.midpoint, line_b.midpoint)

            if dist_sq <= min_dist_sq:
                result = 1
            else:
                timer.reset()
                rect_b = bounding_box(line_b, width=search_width, length_multiplier=search_length)
                result, _ = cv2.rotatedRectangleIntersection(rect_a, rect_b)
                timer.time_event("rotatedRectangleIntersection")

            if result != 0:
                line_a.dead = True
                line_b.dead = True
                timer.reset()
                data = merge_line_pair(data[0], data[1], data[2], data[3], line_b.data[0], line_b.data[1], line_b.data[2], line_b.data[3], line_b.dx, line_b.dy)
                timer.time_event("merge_line_pair")

        if line_a.dead:
            lines[i] = Line(data[0], data[1], data[2], data[3])

    timer.log_all_events()

    return list(filter(lambda x: not x.dead, lines))

