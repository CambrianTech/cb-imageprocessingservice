import math
import numpy as np
import numba as nb
import cv2
import uuid
from enum import IntEnum

from scipy.spatial import distance
from bisect import bisect_left, bisect_right
from pipeline.misc.utils import normalize
from cambrian.LineFunctions import LineFunctions

class Line():

    def __init__(self, ax, ay, bx, by, cluster=None):
        self.data = np.array([ax, ay, bx, by], dtype=np.float32)
        
        self.dx = self.data[2] - self.data[0]
        self.dy = self.data[3] - self.data[1]

        self.length = math.hypot(self.dx, self.dy)

        self.midpoint = np.array([self.data[0] + self.data[2], self.data[1] + self.data[3]]) / 2.0

        self.angle = math.atan2(self.dy, self.dx)
        self.degrees = np.degrees(self.angle)
        self.direction = np.array([self.dx, self.dy]) / self.length

        #for tracking
        self.dead = False
        self.cluster = cluster

    def __del__(self):
        del self.data
        del self.midpoint
        del self.direction

    @property
    def point_a(self):
        return np.array([self.data[0], self.data[1]])

    @property
    def point_b(self):
        return np.array([self.data[2], self.data[3]])

    def get_intersection(self, other):
        return get_line_intersection(self.data[0], self.data[1], self.data[2], self.data[3], other.data[0], other.data[1], other.data[2], other.data[3])

    def intersects(self, other):
        return self.get_intersection(other) is not None

    def equals(self, other, epsilon=0):
        if epsilon == 0:
            return np.array_equal(self.data, other.data)
        else:
            result = np.linalg.norm(self.data - other.data)
            return result < epsilon

    @property
    def normal_a(self):
        return np.array((-self.direction[1], self.direction[0]))

    @property
    def normal_b(self):
        return np.array((self.direction[1], -self.direction[0]))

    def closest_point(self, point):
        return closest_line_point(self.data[0], self.data[1], self.data[2], self.data[3], point[0], point[1])

    def draw(self, img, color=(255,50,255,255), thickness=1, scale=1.0, lineType=cv2.LINE_8):
        draw_line(line, img, (int(self.data[0] * sx), int(self.data[1] * scale)), (int(self.data[2] * sx), int(self.data[3] * scale)), color, thickness=thickness, lineType=lineType)

    def bounding_box(self, width, length_multiplier=1.0):
        return ((self.midpoint[0], self.midpoint[1]), [self.length * length_multiplier, width], self.degrees)

    def extended(self, ratio=1.1, from_a=True, from_b=True, vanishing_point=None):

        if not from_a and not from_b:
            return self

        if vanishing_point is None:
            direction = self.direction
        else:
            direction = vanishing_point - self.midpoint
            direction /= np.linalg.norm(direction)

        amount = self.length * ratio
        data = self.data.copy()

        if from_a:
            data[0] = self.midpoint[0] - direction[0] * amount
            data[1] = self.midpoint[1] - direction[1] * amount

        if from_b:
            data[2] = self.midpoint[0] + direction[0] * amount
            data[3] = self.midpoint[1] + direction[1] * amount

        return Line(data[0], data[1], data[2], data[3], self.cluster)      

    def copy(self):
        #todo: ineffcient
        return Line(self.data[0], self.data[1], self.data[2], self.data[3])

EPSILON = np.finfo(float).eps

@nb.jit(nopython=True)
def closest_line_point(x0, y0, x1, y1, px, py): #minimum angle between lines segments cannot differ by more than 90 degrees
    dx, dy = x1-x0, y1-y0
    det = dx*dx + dy*dy
    a = (dy*(py-y0)+dx*(px-x0))/det
    return (x0+a*dx), (y0+a*dy)

class ImageEdge(IntEnum):
    Top = 0x0001
    Right = 0x0010
    Bottom = 0x0100
    Left = 0x1000

@nb.jit(nopython=True)
def point_on_image_edge(point, width, height, min_distance=3):
    edge = 0x0
    if point[1] <= min_distance:
        edge |= ImageEdge.Top

    if point[0] >= width - min_distance - 1:
        edge |= ImageEdge.Right

    if point[1] >= height - min_distance - 1:
        edge |= ImageEdge.Bottom

    if point[0] <= min_distance:
        edge |= ImageEdge.Left

    return edge

@nb.jit(nopython=True)
def line_on_image_edge(point_a, point_b,  width, height, min_distance=3):
    #if A is on the edge and also one of those edges is the same as B is on
    return point_on_image_edge(point_a, width, height, min_distance) \
        and (point_on_image_edge(point_a,  width, height, min_distance) & point_on_image_edge(point_b, width, height, min_distance)) > 0

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

def draw_line(line, img, color=(255,50,255,255), thickness=1, scale=1.0, lineType=cv2.LINE_8):
    cv2.line(img, (int(line.data[0] * scale), int(line.data[1] * scale)), (int(line.data[2] * scale), int(line.data[3] * scale)), color, thickness=thickness, lineType=lineType)

def draw_lines(img, lines, color=(255,50,255,255), thickness=1, scale=1.0, lineType=cv2.LINE_8):
    
    pts = np.array([line.data for line in lines])

    if scale != 1.0:
        pts = pts * scale

    #Important! keep this, very fast: 0.0007 for 512 lines vs iterating is 0.63 seconds, a thousand times slower
    pts = pts.astype(np.int32).reshape((-1,2,2))

    cv2.drawContours(img, pts, -1, color, thickness=thickness, lineType=lineType)

# @nb.jit(nopython=True)
# def merge_line_pair(ax, ay, bx, by, cx, cy, dx, dy, dljx, dljy):

#     dlix = bx - ax
#     dliy = by - ay
#     #dljx = dx - cx
#     #dljy = dy - cy

#     li = math.sqrt((dlix * dlix) + (dliy * dliy))
#     lj = math.sqrt((dljx * dljx) + (dljy * dljy))

#     xg = (li * (ax + bx) + lj * (cx + dx)) / (2.0 * (li + lj))
#     yg = (li * (ay + by) + lj * (cy + dy)) / (2.0 * (li + lj))

#     if (dlix == 0.0): thi = math.pi / 2.0
#     else: thi = math.atan(dliy / dlix)

#     if (dljx == 0.0): thj = math.pi / 2.0
#     else: thj = math.atan(dljy / dljx)

#     if abs(thi - thj) <= math.pi / 2.0:
#         thr = (li * thi + lj * thj) / (li + lj)
#     else:
#         tmp = thj - math.pi * (thj / abs(thj))
#         thr = li * thi + lj * tmp
#         thr /= (li + lj)

#     sin_thr = math.sin(thr)
#     cos_thr = math.cos(thr)

#     axg = (ay - yg) * sin_thr + (ax - xg) * cos_thr
#     bxg = (by - yg) * sin_thr + (bx - xg) * cos_thr
#     cxg = (cy - yg) * sin_thr + (cx - xg) * cos_thr
#     dxg = (dy - yg) * sin_thr + (dx - xg) * cos_thr

#     delta1xg = min(axg, min(bxg, min(cxg,dxg)))
#     delta2xg = max(axg, max(bxg, max(cxg,dxg)))

#     return  delta1xg * cos_thr + xg, \
#             delta1xg * sin_thr + yg, \
#             delta2xg * cos_thr + xg, \
#             delta2xg * sin_thr + yg

def merge_lines(lines, search_width, search_length=1.05, angle_threshold=math.radians(3)):

    min_dist_sq = search_width * search_width

    cluster_index = 0

    for i in range(len(lines)):
        
        line_a = lines[i]
        if line_a.dead: continue

        if line_a.cluster is None:
            line_a.cluster = cluster_index

        rect_a = line_a.bounding_box(width=search_width, length_multiplier=search_length)
        data = line_a.data.copy()

        for line_b in lines:

            if LineFunctions.line_angle_difference(line_a.angle, line_b.angle) > angle_threshold or line_b.dead or line_a == line_b:
                continue

            if distance.sqeuclidean(line_a.midpoint, line_b.midpoint) <= min_dist_sq:
                result = 1
            else:
                rect_b = line_b.bounding_box(width=search_width, length_multiplier=search_length)
                result, _ = cv2.rotatedRectangleIntersection(rect_a, rect_b)

            if result != 0:
                line_a.dead = True
                line_b.dead = True
                data = LineFunctions.merge_line_pair(data[0], data[1], data[2], data[3], \
                                                     line_b.data[0], line_b.data[1], line_b.data[2], line_b.data[3], \
                                                     line_b.dx, line_b.dy)

        if line_a.dead:
            lines[i] = Line(data[0], data[1], data[2], data[3], cluster=line_a.cluster)

        cluster_index += 1

    return list(filter(lambda x: not x.dead, lines))
