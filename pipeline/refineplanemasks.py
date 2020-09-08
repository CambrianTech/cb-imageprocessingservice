from pipeline.core import PipelineStep
import cv2
import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.morphology import watershed, disk
from cambrian import image_processing as ip
from skimage import filters, img_as_float
from skimage.filters import threshold_multiotsu, frangi
import os
import pickle

IM_LOGGING_ENABLED = True


def _log_image(name, image):
    if IM_LOGGING_ENABLED:
        cv2.imwrite('logging/' + name, image)


def _log_ply(image, masks, plane_XYZ, file_path='logging/3D.ply', write_occlusion=False, mult=1.0):
    if IM_LOGGING_ENABLED:
        image = cv2.resize(image, (mult * 160, mult * 120))
        width = image.shape[1]
        height = image.shape[0]
        faces = []
        points = []
        planes = np.zeros(shape=(len(masks), height, width, 3))
        new_masks = np.zeros(shape=(len(masks), height, width))
        for i in range(len(masks)):
            new_masks[i] = cv2.resize(masks[i], (width, height))
            planes[i] = cv2.resize(plane_XYZ[i], (width, height))
        plane_XYZ = planes
        masks = new_masks

        plane_depths = plane_XYZ[:, :, :, 1] * masks + 10 * (1 - masks)
        segmentation = np.argmin(plane_depths, axis=0)
        _log_image("logging/ply_segmention.png",cv2.normalize(segmentation, None, 0, 255, cv2.NORM_MINMAX))

        for mask_index, (mask, XYZ) in enumerate(zip(masks, plane_XYZ)):
            indices = np.nonzero(np.logical_and(segmentation == mask_index, masks[mask_index] > 0))
            for y, x in zip(indices[0], indices[1]):
                if y == height - 1 or x == width - 1:
                    continue
                validNeighborPixels = []
                for neighborPixel in [(x, y + 1), (x + 1, y), (x + 1, y + 1)]:
                    # if mask[neighborPixel[1], neighborPixel[0]] > 0.5:
                    validNeighborPixels.append(neighborPixel)
                    # pass
                    continue
                if len(validNeighborPixels) == 3:
                    faces.append([len(points) + c for c in range(3)])
                    points += [(XYZ[pixel[1], pixel[0]], pixel, segmentation[pixel[1], pixel[0]] == mask_index) for
                               pixel in
                               [(x, y), (x + 1, y + 1), (x + 1, y)]]
                    faces.append([len(points) + c for c in range(3)])
                    points += [(XYZ[pixel[1], pixel[0]], pixel, segmentation[pixel[1], pixel[0]] == mask_index) for
                               pixel in
                               [(x, y), (x, y + 1), (x + 1, y + 1)]]
                elif len(validNeighborPixels) == 2:
                    faces.append([len(points) + c for c in range(3)])
                    points += [(XYZ[pixel[1], pixel[0]], pixel, segmentation[pixel[1], pixel[0]] == mask_index) for
                               pixel in
                               [(x, y), (validNeighborPixels[0][0], validNeighborPixels[0][1]),
                                (validNeighborPixels[1][0], validNeighborPixels[1][1])]]
                    pass
                continue
            continue

        imageFilename = "textureless"
        with open(file_path, 'w') as f:
            header = """ply
        format ascii 1.0"""
            header += imageFilename
            header += """
        element vertex """
            header += str(len(points))
            header += """
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        element face """
            header += str(len(faces))
            header += """
        property list uchar int vertex_indices
        end_header
        """
            f.write(header)
            for point in points:
                X = point[0][0]
                Y = point[0][1]
                Z = point[0][2]
                if not write_occlusion or point[2]:
                    color = image[point[1][1], point[1][0]]
                else:
                    color = (128, 128, 128)
                    pass
                f.write(str(X) + ' ' + str(Z) + ' ' + str(-Y) + ' ' + str(color[2]) + ' ' + str(color[1]) + ' ' + str(
                    color[0]) + '\n')
                continue

            for face in faces:
                valid = True
                f.write('3 ')
                for c in face:
                    f.write(str(c) + ' ')
                    continue
                f.write('\n')
                continue
            f.close()
            pass
    return


a = 1.0
b = 1.0

METADATA = np.array([a * 571.87, b * 571.87, 320, 240, 640, 480, 0, 0, 0, 0])

IMAGE_MAX_DIM = 640
IMAGE_MIN_DIM = 480


def calcPlaneXYZ(planes, width=IMAGE_MIN_DIM, height=IMAGE_MAX_DIM, camera=METADATA, max_depth=10):
    urange = (np.arange(width, dtype=np.float32) / (width) * (camera[4]) - camera[2]) / camera[0]
    urange = urange.reshape(1, -1).repeat(height, 0)

    vrange = (np.arange(height, dtype=np.float32) / (height) * (camera[5]) - camera[3]) / camera[1]
    vrange = vrange.reshape(-1, 1).repeat(width, 1)

    ranges = np.stack([urange, np.ones(urange.shape), -vrange], axis=-1)

    planeOffsets = np.linalg.norm(planes, axis=-1, keepdims=True)
    planeNormals = planes / np.maximum(planeOffsets, 1e-4)

    normalXYZ = np.dot(ranges, planeNormals.transpose())

    # _log_image("n.png", cv2.normalize(normalXYZ.transpose(2,0,1)[0], None, 0, 255, cv2.NORM_MINMAX))

    normalXYZ[normalXYZ == 0] = 1e-4

    planeDepths = planeOffsets.squeeze(-1) / normalXYZ
    if max_depth > 0:
        planeDepths = np.clip(planeDepths, 0, max_depth)
        pass
    XYZ = (np.expand_dims(planeDepths, -1) * np.expand_dims(ranges, 2))

    return XYZ.transpose(2, 0, 1, 3), planeDepths.transpose(2, 0, 1)



def get_XYZ_from_depth(depth, width=IMAGE_MIN_DIM, height=IMAGE_MAX_DIM, camera=METADATA, max_depth=10):
    height = depth.shape[0]
    width = depth.shape[1]

    urange = (np.arange(width, dtype=np.float32) / (width) * (camera[4]) - camera[2]) / camera[0]
    urange = urange.reshape(1, -1).repeat(height, 0)

    vrange = (np.arange(height, dtype=np.float32) / (height) * (camera[5]) - camera[3]) / camera[1]
    vrange = vrange.reshape(-1, 1).repeat(width, 1)

    X = depth * urange
    Y = depth
    Z = -depth * vrange
    XYZ = np.dstack((X, Y, Z))

    return XYZ


def get_segmentation_image(labels, image, avg=False, resize=True):
    if True:
        img_seg = image
        if resize:
            img_seg = cv2.resize(image, (labels.shape[1], labels.shape[0]))

        for label in range(1, np.amax(labels) + 1):
            color = np.random.randint([0, 0, 10], [254, 254, 235])
            if avg: color = np.mean(img_seg[labels == label], axis=0)
            img_seg[labels == label] = color
        return img_seg


def find_floor_indices(floor_mask, plane_masks, plane_normals):
    floor_mask = cv2.resize(floor_mask, (plane_masks[0].shape[1], plane_masks[0].shape[0]))

    floor_intersections = []
    dots = []
    for d in range(len(plane_masks)):
        floor_intersections.append(cv2.countNonZero(np.uint8(plane_masks[d][floor_mask > 255./3.] > .5)))

    scores = np.int32(floor_intersections)
    floor_indices = np.nonzero(scores > np.mean(scores))[0]
    floor_indices = floor_indices[np.argsort(scores[floor_indices])[::-1]]

    for d in range(len(plane_masks)):
        dots.append(np.dot(plane_normals[floor_indices[0]], plane_normals[d]))

    angs = np.arccos(np.clip(dots,-1.0,1.0))*180/np.pi

    horiz_indices = np.nonzero(np.abs(angs)<15)[0]
    return floor_indices, horiz_indices, angs


def find_wall_indices(wall_mask, plane_masks, floor_normal, plane_normals):
    wall_mask = cv2.resize(wall_mask, (plane_masks[0].shape[1], plane_masks[0].shape[0]))

    wall_intersections = []
    dots = []
    for d in range(len(plane_masks)):
        wall_intersections.append(cv2.countNonZero(np.uint8(plane_masks[d][wall_mask > 255./3.] > .5)))
        dots.append(np.dot(floor_normal, plane_normals[d]))

    angs = np.arccos(np.clip(dots, -1.0, 1.0)) * 180 / np.pi
    vert_indices = np.nonzero(np.abs(90 - angs) < 15)[0]

    scores = np.int32(wall_intersections)
    wall_indices = np.nonzero(np.logical_and(scores > np.mean(scores), np.abs(90 - angs) < 15))[0]

    return wall_indices, vert_indices, angs


def proj(points,plane):
    plane_offset = np.linalg.norm(plane)
    plane_normal = plane/plane_offset
    s = np.dot(points, plane_normal) - plane_offset
    return s


class PipelineRefinePlaneMasks(PipelineStep):
    @property
    def required_keys(self) -> list:
        return ["image", "semantic_probs", "hed", "mask"]

    @property
    def output_keys(self) -> list:
        return []

    def run(self, data):

        img = data["image"]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_rs = cv2.resize(img, (640, 480))
        hed = data["hed"]

        w, h = hed.shape
        # shape = (h, w)
        edges_hed = cv2.Canny(hed, 10, 250)

        _log_image('edges_hed.png', edges_hed)
        hed = cv2.resize(hed, (640, 480))
        edges_hed = cv2.resize(edges_hed, (640, 480))
        edges_hed = cv2.dilate(255 * np.uint8(edges_hed > 0),
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

        planes_data = data["planes"]

        detection_parameters = np.array(data["planes"]["detection"], dtype=np.float32)
        plane_parameters = np.array(detection_parameters[:, 6:9], dtype=np.float32)
        plane_offsets = np.linalg.norm(plane_parameters, axis=-1, keepdims=True)
        plane_normals = plane_parameters / np.maximum(plane_offsets, 1e-4)

        plane_clusters = np.array(detection_parameters[:, 4], dtype=np.int32)
        roi = np.array(detection_parameters[:, :4], dtype=np.int32)
        roi = np.minimum(roi, 639)
        roi[:,0] = np.minimum(roi[:,0], 479)
        roi[:,2] = np.minimum(roi[:,2], 479)
        # print("roi", roi)

        plane_masks = planes_data["masks"]
        number_planes = len(plane_masks)
        print("number of planes: ", number_planes)

        depth_or = planes_data["depth_np"][0][80:-80, :]
        max_depth = np.amax(depth_or)
        plane_XYZ = planes_data["plane_XYZ"][:,:,80:-80,:].transpose(0,2,3,1)

        _log_ply(img_rs, data["planes"]["masks"], plane_XYZ, mult=1,file_path='logging/3D1.ply')

        semantic_mask = data["semantic_probs"]
        print("sm", np.amax(semantic_mask))
        # semantic_mask = cv2.resize(semantic_mask, (640, 480))
        # floor_mask = semantic_mask[:, :, 0]
        print(np.amax(semantic_mask))
        for k in range(len(semantic_mask)):
            _log_image(str(k)+'_mask.png', 225.*semantic_mask[k])
        floor_mask = np.uint8(255.*(semantic_mask[3]+semantic_mask[28]))
        wall_mask = np.uint8(255.*semantic_mask[0])
        # other_mask = semantic_mask[:, :, 2]

        # _log_image('semantic_mask.png', semantic_mask)
        _log_image('floor_mask.png', floor_mask)
        _log_image('wall_mask.png', wall_mask)
        # _log_image('other_mask.png', other_mask)

        floor_indices, horiz_indices, floor_angs = find_floor_indices(floor_mask, plane_masks, plane_normals)

        print("floor indices are: ", floor_indices, horiz_indices)

        floor_index = floor_indices[0]

        plane_masks[floor_index] = np.sqrt(floor_mask * 3. / 2. / 255. * plane_masks[floor_index])

        floor_normal = plane_normals[floor_index]
        wall_indices, vert_indices, vert_angs = find_wall_indices(wall_mask, plane_masks, plane_normals[floor_index],
                                                                  plane_normals)
        print("wall indices are: ", wall_indices, vert_indices, vert_angs)

        for i in range(number_planes):
            if i in vert_indices:
                mass = np.sum(plane_masks[i])
                if mass>0:
                    plane_normals[i] = np.cross(floor_normal, np.cross(floor_normal, plane_normals[i]))

                    plane_parameters[i] = plane_normals[i] * np.sum(
                        plane_masks[i] * np.dot(plane_XYZ[i], plane_normals[i])) / mass

            if i in horiz_indices:
                mass = np.sum(plane_masks[i])
                if mass>0:
                    plane_normals[i] = floor_normal
                    plane_parameters[i] = plane_normals[i] * np.sum(
                        plane_masks[i] * np.dot(plane_XYZ[i], plane_normals[i])) / np.sum(plane_masks[i])

        plane_XYZ, plane_depth = calcPlaneXYZ(plane_parameters, width=640, height=480, max_depth=10)

        for i in np.unique(plane_clusters):

            cluster = np.nonzero(plane_clusters == i)[0]

            for j in cluster:
                if j not in wall_indices: continue

                for k in cluster:
                    if k <= j: continue
                    rect = roi[j]
                    plane_rect = [plane_XYZ[j, rect[0], rect[1]], plane_XYZ[j, rect[0], rect[3]],
                                  plane_XYZ[j, rect[2], rect[3]], plane_XYZ[j, rect[2], rect[1]]]
                    p = np.abs(proj(plane_rect, plane_parameters[k]))

                    if len(p)>0 and (np.mean(p) < 0.3):
                        plane_parameters[k] = plane_parameters[j]

        plane_XYZ, plane_depth = calcPlaneXYZ(plane_parameters, width=640, height=480, max_depth=10)


        for d in range(number_planes):
            mask = np.uint8(plane_depth[d] > 0)
            local_maxi = peak_local_max(np.abs(plane_masks[d] - .5), min_distance=5, indices=False,
                                        labels=mask, exclude_border=False)
            markers = ndimage.label(local_maxi)[0]

            labels_new = watershed(hed, markers=markers, mask=mask)
            plane_mask = plane_masks[d].copy()
            plane_mask[edges_hed > 0] = 0
            plane_masks[d] = get_segmentation_image(np.uint8(labels_new + 1), plane_masks[d], avg=True)
            _log_image('distance_masks_segmentation' + str(d) + '.png', 255. * plane_masks[d])

        plane_masks[plane_masks < .05] = 0

        plus = np.concatenate(([np.zeros_like(plane_masks[0])], plane_masks))
        depth_segmentation = np.int32(np.argmax(plus, axis=0))

        _log_image('depth_segmentation.png', get_segmentation_image(np.uint8(depth_segmentation), img_rs, avg=False))

        data["planes"]["masks"] = np.uint8(np.zeros_like(plane_masks))

        data["planes"]["detection"] = np.zeros((number_planes, 10))
        data["planes"]["detection"][:,:9] = detection_parameters

        for d in range(number_planes):
            final_mask = np.zeros_like(plane_masks[d], dtype=np.uint8)

            contours, hierarchy = cv2.findContours(255*np.uint8(depth_segmentation==d+1), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            areas = [cv2.contourArea(cnt) for cnt in contours]

            max_cnt = np.argmax(areas)

            cv2.drawContours(final_mask, [contours[max_cnt]], 0, 255, -1, cv2.LINE_AA)
            cv2.drawContours(final_mask, [contours[max_cnt]], 0, 255, 2, cv2.LINE_AA)

            data["planes"]["masks"][d] = final_mask

            rect = cv2.boundingRect(np.uint8(data["planes"]["masks"][d]))
            data["planes"]["detection"][d, 0:4] = [
                rect[1], rect[0], rect[1] + rect[3], rect[0] + rect[2]
            ]

            if d in floor_indices:
                data["planes"]["detection"][d, 9] = 1
                continue

            if d in wall_indices:
                data["planes"]["detection"][d, 9] = 2
                continue

            if d in horiz_indices:
                data["planes"]["detection"][d, 9] = 3
                continue

            if d in vert_indices:
                data["planes"]["detection"][d, 9] = 4


        _log_ply(img_rs, data["planes"]["masks"], plane_XYZ, mult=2, file_path='logging/3D2.ply')