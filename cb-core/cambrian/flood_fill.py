import numpy as np
import cv2
from collections import deque

def flood_fill(img, mask):

    unit = max(img.shape[0],img.shape[1]) / 200
    unit_size = (int(unit), int(unit))

    grid_shape = (int(img.shape[0]/unit_size[0]), int(img.shape[1]/unit_size[1]))
    grid = cv2.resize(mask, (grid_shape[1], grid_shape[0]))
    max_count = cv2.countNonZero(grid) * 1.5

    trans = cv2.distanceTransform(grid, cv2.DIST_L1, 5)
    grid = np.uint8(cv2.threshold(trans, 0.2 * trans.max(), 255, 0)[1])

    #return cv2.resize(grid, (mask.shape[1], mask.shape[0]))

    sampler = np.uint8(cv2.threshold(trans, 0.6 * trans.max(), 255, 0)[1])

    if cv2.countNonZero(sampler) < 50:
        sampler = grid.copy()

    sampler = cv2.resize(sampler, (mask.shape[1], mask.shape[0]))

    mean, std = cv2.meanStdDev(img, mask=sampler)
    sum_std = sum(std[0]);

    contours, hierarchy = cv2.findContours(grid, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    points = deque()

    #add in all mask edges
    for contour in contours:
        for p in contour:
            points.append(((p[0][1], p[0][0]), None, 0))

    total_count = 0
    
    while len(points) > 0 and total_count < max_count:
        
        point, last_point, count = points.pop()

        
        if not last_point is None:

            _y = unit_size[0] * point[0]
            _x = unit_size[1] * point[1]
            img_a = img[_y:_y+unit_size[0], _x:_x+unit_size[1]]
            mean_a, std_a = cv2.meanStdDev(img_a)


            _y = unit_size[0] * last_point[0]
            _x = unit_size[1] * last_point[1]
            img_b = img[_y:_y+unit_size[0], _x:_x+unit_size[1]]
            mean_b, std_b = cv2.meanStdDev(img_b)

            #compare now
            diff = sum(abs(mean_a - mean_b))[0]
            diff_std = sum(abs(std_a - std_b))[0]

            if diff > 0.3 * sum_std:
                continue

            # if std_a[1] < 5.0 or diff < 10.0:
            #     continue

            count += diff_std

            if count > 2.0 * sum_std: continue
            

            # if s_diff[0] > std[0] or s_diff[1] > std[1] or s_diff[2] > std[2]:
            #     continue

        def get_neighbors(pt):
            return [(point[0]-1, point[1]), 
                   (point[0]+1, point[1]), 
                   (point[0], point[1]-1), 
                   (point[0], point[1]+1),
                   (point[0]-1, point[1]-1),
                   (point[0]-1, point[1]+1),
                   (point[0]+1, point[1]+1),
                   (point[0]+1, point[1]-1)]

        def is_valid_point(pt):
            return 0 <= pt[0] < grid.shape[0] and 0 <= pt[1] < grid.shape[1]

        neigbors = get_neighbors(point)

        for pt in neigbors:
            if is_valid_point(pt) and grid[pt[0], pt[1]] != 255:
                total_count += 1
                points.append((pt, point, count))

        grid[point[0], point[1]] = 255

    return cv2.resize(grid, (mask.shape[1], mask.shape[0]))
