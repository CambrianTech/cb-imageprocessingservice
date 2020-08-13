from pipeline.core import PipelineStep
import cv2
import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.morphology import watershed, disk
from cambrian import image_processing as ip
from skimage import filters, color
from skimage.morphology.extrema import local_minima
from skimage.future import graph
from skimage.filters import threshold_multiotsu, frangi
import os
import pickle

IM_LOGGING_ENABLED = True
print("logging enabled")


def _log_image(name, image):
    if IM_LOGGING_ENABLED:
        cv2.imwrite('logging/' + name, image)


def _log_ply(image, masks, plane_XYZ, file_path='logging/3D.ply', write_occlusion=False, mult=1.0):
    if IM_LOGGING_ENABLED:
        # image = cv2.resize(image, (mult * 160, mult * 120))
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
        # with open('results/data.pickle', 'wb') as handle:
        #     pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        img = data["image"]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_sobel = filters.sobel(color.rgb2gray(img))
        img_sobel = cv2.resize(img_sobel, (640, 480))
        img_rs = cv2.resize(img, (640, 480))

        hed = data["hed"]

        w, h = hed.shape
        # shape = (h, w)
        edges_hed = cv2.Canny(hed, 10, 250)
        _log_image('edges_hed.png', edges_hed)
        hed_min = local_minima(hed)
        hed_min = cv2.resize(np.uint8(hed_min), (640, 480))
        _log_image('hed_min.png', 255. * hed_min)

        hed = cv2.resize(hed, (640, 480))
        edges_hed = cv2.resize(edges_hed, (640, 480))
        # edges_hed = cv2.dilate(255 * np.uint8(edges_hed > 0),
        #                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

        normals = np.uint8(data["normals"])
        normals = cv2.resize(normals, (640, 480))

        _log_image("normals_nn.png", normals)



        planes_data = data["planes"]

        # for key, value in planes_data.items():
        #     print(key, value.shape, value.dtype)


        detection_parameters = np.array(data["planes"]["detection"], dtype=np.float32)
        plane_parameters = np.array(detection_parameters[:, 6:9], dtype=np.float32)
        plane_offsets = np.linalg.norm(plane_parameters, axis=-1, keepdims=True)
        plane_normals = plane_parameters / np.maximum(plane_offsets, 1e-4)

        plane_clusters = np.array(detection_parameters[:, 4], dtype=np.int32)
        roi = np.array(detection_parameters[:, :4], dtype=np.int32)
        a=32

        roi[:,0] = np.clip(roi[:,0]-a,0,479)
        roi[:,1] = np.clip(roi[:,1]-a,0,639)
        roi[:,2] = np.clip(roi[:,2]+a,0, 479)
        roi[:,3] = np.clip(roi[:,3]+a,0, 639)
        plane_masks = planes_data["masks"]
        number_planes = len(plane_masks)
        print("number of planes: ", number_planes)

        plane_XYZ = planes_data["plane_XYZ"][:,:,80:-80,:].transpose(0,2,3,1)

        _log_ply(img_rs, data["planes"]["masks"], plane_XYZ, mult=1,file_path='logging/3D1.ply')

        semantic_mask = np.uint8(255 * data["semantic_probs"])
        semantic_mask = cv2.resize(semantic_mask, (640, 480))
        floor_mask = semantic_mask[:, :, 0]
        wall_mask = semantic_mask[:, :, 1]
        other_mask = semantic_mask[:, :, 2]

        _log_image('semantic_mask.png', semantic_mask)
        _log_image('floor_mask.png', floor_mask)
        _log_image('wall_mask.png', wall_mask)
        _log_image('other_mask.png', other_mask)

        floor_indices, horiz_indices, floor_angs = find_floor_indices(floor_mask, plane_masks, plane_normals)

        print("floor indices are: ", floor_indices, horiz_indices)

        floor_index = floor_indices[0]

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

        plane_XYZ, plane_depth = calcPlaneXYZ(plane_parameters, camera=planes_data["camera"], width=640, height=480, max_depth=10)

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

        plane_XYZ, plane_depth = calcPlaneXYZ(plane_parameters, camera=planes_data["camera"], width=640, height=480, max_depth=10)

        not_plane = np.sum(plane_masks, axis=0)
        not_plane = np.amax(not_plane) - not_plane
        not_plane_mask = np.int32(np.ones_like(plane_masks[0]))
        not_plane_mask[hed > 0] = 0
        not_plane_mask[wall_mask > 255 / 3.] = 0
        not_plane_mask[floor_mask > 255 / 3.] = 0

        core_labels = np.int32(np.zeros_like(plane_masks[0]))

        for i in range(number_planes):
            box = roi[i]
            print(box)
            # mask was upsampled so expand roi to deal with border issues
            thresh_o = filters.threshold_multiotsu(plane_masks[i][box[0]: box[2], box[1]: box[3]], classes=4)
            print(thresh_o)
            not_plane_mask[plane_masks[i] > min(.05, thresh_o[0])] = 0

            mask = np.ones_like(plane_masks[i])
            mask[box[0]: box[2], box[1]: box[3]] = 0

            plane_masks[i][mask > 0] = 0

            core_mask = plane_masks[i] > thresh_o[2]
            components, output, stats, centroids = cv2.connectedComponentsWithStats(np.uint8(core_mask), connectivity=4)

            sizes = stats[:, -1]
            sizes[0] = 0
            largest = np.argmax(sizes)
            core_labels[output == largest] = i + 1
        _log_image('img.png', img_rs)
        _log_image('not_plane_mask.png', not_plane_mask)

        components, output, stats, centroids = cv2.connectedComponentsWithStats(np.uint8(not_plane_mask),
                                                                                connectivity=4)
        sizes = stats[:, -1]
        sizes[0] = 0
        large = np.nonzero(sizes > 16)[0]
        print(large)
        for l in large:
            core_labels[output == l] = np.amax(core_labels) + 1

        # detect lines in image
        length_threshold = 16
        distance_threshold = 1.41421356
        canny_th1 = 50.0
        canny_th2 = 50.0
        canny_aperture_size = 5
        do_merge = False
        fld = cv2.ximgproc.createFastLineDetector(length_threshold,
                                                  distance_threshold, canny_th1, canny_th2, canny_aperture_size,
                                                  do_merge)
        lines1 = fld.detect(cv2.cvtColor(img_rs, cv2.COLOR_BGR2GRAY))

        # detect lines in normals
        length_threshold = 32
        distance_threshold = 1.41421356
        canny_th1 = 50.0
        canny_th2 = 250.0
        canny_aperture_size = 3
        do_merge = True
        fld = cv2.ximgproc.createFastLineDetector(length_threshold,
                                                  distance_threshold, canny_th1, canny_th2, canny_aperture_size,
                                                  do_merge)

        lines2 = fld.detect(np.uint8(cv2.cvtColor(normals, cv2.COLOR_BGR2GRAY)))

        # Draw lines on the image
        img_lines = fld.drawSegments(np.zeros_like(hed), lines1, )[:, :, 2]
        # _log_image('img_lines.png',fld.drawSegments(img_rs, lines1))

        normal_lines = fld.drawSegments(np.zeros_like(hed), lines2)[:, :, 2]
        # _log_image('hed_lines.png',fld.drawSegments(img_rs, lines2))

        # We use the hed network as a starting point for watershed

        line_mask = img_lines == 0
        line_mask[normal_lines > 0] = 0

        conn = ndimage.generate_binary_structure(line_mask.ndim, 2)
        line_mask = ndimage.grey_erosion(line_mask, footprint=conn)

        _log_image('line_mask.png', 255. * (line_mask))

        core_labels[line_mask == 0] = 0
        core_labels = ip.refine_mask_watershed(None, img_rs, core_labels, None, erode=0, distance=0.1,
                                              max_value=np.amax(core_labels) + 1)
        _log_image('core_labels.png', get_segmentation_image(core_labels, img_rs, avg=False))


        components, output, stats, centroids = cv2.connectedComponentsWithStats(np.uint8(hed_min), connectivity=4)

        sizes = stats[:, -1]
        sizes[0] = 0
        large = np.nonzero(sizes > 16)[0]

        hed_large_components = np.zeros_like(hed_min)
        for comp in large:
            hed_large_components[output == comp] = 1

        hed_min[hed_large_components > 0] = 0
        _log_image('hed_min.png', 255. * (hed_min))

        hed_large_components = ndimage.grey_erosion(hed_large_components, footprint=conn)

        _log_image('hed_large_components.png', 255. * (hed_large_components))

        hed_min[hed_large_components > 0] = 1

        distances = cv2.distanceTransform(np.uint8(hed_min), cv2.DIST_L2, 5)

        hed_min_markers = ndimage.label(hed_min)[0]

        # If we need to break the hed so that pieces are naturally compact

        local_maxi = peak_local_max(distances, indices=False,
                                    labels=hed_min_markers, num_peaks_per_label=20)
        hed_min_markers = ndimage.label(local_maxi)[0]

        _log_image('hed_min_markers.png', get_segmentation_image(hed_min_markers, img_rs, avg=False))

        labels_hed = watershed(img_sobel, markers=hed_min_markers, mask=line_mask)

        _log_image('watershed_hed.png', get_segmentation_image(labels_hed, img_rs, avg=False))

        line_masks = plane_masks.copy()
        line_masks[:, line_mask == 0] = -1000
        not_plane[line_mask == 0] = -1000
        line_masks[line_masks < .01] = 0
        floor_mask[floor_mask < wall_mask] = 0
        floor_mask[floor_mask < .05] = 0
        g = ip.rag_mean_color(
            np.dstack((line_masks.transpose(1, 2, 0), not_plane, floor_mask / 255., img_rs / 255.)), labels_hed)

        labels_graph = graph.merge_hierarchical(labels_hed, g, thresh=0.5, rag_copy=False,
                                                in_place_merge=True,
                                                merge_func=ip.merge_mean_color,
                                                weight_func=ip.weight_mean_color)

        print("graph_reduction", np.amax(labels_graph), np.amax(labels_hed))

        labels_graph[line_mask == 0] = 0

        _log_image('watershed_graph.png', get_segmentation_image(labels_graph, img_rs, avg=False))

        labels_hed = labels_graph
        label_number = len(np.unique(labels_hed))
        means = np.zeros((label_number, number_planes), dtype=np.float32)
        plane_masks[:, line_mask == 0] = 0
        for i in range(number_planes):
            _log_image('plane_masks' + str(i) + '.png', 255. * plane_masks[i])
            plane_masks[i][core_labels == i + 1] = 1
            means[:, i] = ndimage.mean(plane_masks[i], labels=labels_hed, index=np.unique(labels_hed))

        means_plus = np.zeros((label_number, number_planes + 3), dtype=np.float32)
        means_plus[:, 3:number_planes + 3] = means
        means_plus[means_plus < .25] = 0
        floor_mask[floor_mask < .5] = 0
        means_plus[:1] = np.mean(floor_mask / 255.)
        arg = np.argmax(means_plus, axis=-1)

        labels_arg = labels_hed.copy() + number_planes + np.amax(arg)

        #
        # labels_arg = np.zeros_like(labels_hed)
        for i in range(label_number):
            if arg[i] > 0:
                labels_arg[labels_hed == np.unique(labels_hed)[i]] = arg[i]

        _log_image('watershed_arg.png', get_segmentation_image(labels_arg, img_rs, avg=False))
        labels_arg[line_mask == 0] = 0

        line_mask = img_lines == 0
        line_mask[normal_lines > 0] = 0
        labels_arg = watershed(img_sobel, markers=labels_arg, mask=line_mask, compactness=.003)
        labels_arg[labels_arg < 0] = 0

        for label in np.unique(labels_arg):
            mask = np.uint8(labels_arg == label)
            labels_arg[mask > 0] = 0
            isolated = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
            ret, isolated = cv2.threshold(isolated, min(10, isolated.max() - 2), 1, 0)
            labels_arg[np.uint8(isolated) > 0] = np.int32(label)

        labels_arg = cv2.watershed(img_rs, markers=labels_arg)
        _log_image('watershed_refine.png', get_segmentation_image(labels_arg, np.zeros_like(img_rs), avg=False))



        data["planes"]["masks"] = np.uint8(np.zeros_like(plane_masks))

        data["planes"]["detection"] = np.zeros((number_planes, 10))
        data["planes"]["detection"][:,:9] = detection_parameters

        for d in range(number_planes):
            final_mask = np.zeros_like(plane_masks[d], dtype=np.uint8)

            contours, hierarchy = cv2.findContours(255*np.uint8(labels_arg==d+1), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            areas = [cv2.contourArea(cnt) for cnt in contours]

            max_cnt = np.argmax(areas)

            cv2.drawContours(final_mask, [contours[max_cnt]], 0, 255, -1, cv2.LINE_AA)

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