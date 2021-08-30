import cv2
import os.path
import numpy as np
from enum import IntFlag

class LogLevel(IntFlag):
    Nothing = 0
    Images = 0x1 << 0
    Segmentation = 0x1 << 1
    Lines = 0x1 << 2
    Models = 0x1 << 3

    All = 0xff

_logging_index = 0

def get_unique_id(data:dict):
    return data["unique_id"]

def make_log_path(data:dict, name:str, extension=".jpg"):
    global _logging_index
    directory = get_logging_dir(data)
    if not os.path.exists(directory):
        os.makedirs(directory)

    filename = "%d - %s%s" % (_logging_index, name, extension)
    return os.path.join(directory, filename)

def im_logging_enabled(data:dict, level=LogLevel.All):
    logging_step = get_logging_step(data)
    if get_logging_dir(data) is None or (logging_step is not None and get_current_step(data) != get_logging_step(data)):
        return False

    return level & get_logging_level(data)

def get_logging_dir(data:dict):
    return data["logging_dir"] if "logging_dir" in data else None

def set_logging_dir(data:dict, directory):
    data["logging_dir"] = directory

def get_logging_step(data:dict):
    return data["logging_step"] if "logging_step" in data else LogLevel.Nothing

def set_logging_step(data:dict, step:int, current_step:int):
    global _logging_index
    _logging_index = 0
    data["logging_step"] = step
    data["step"] = current_step

def get_current_step(data:dict):
    return data["step"] if "step" in data else -1

def get_logging_level(data:dict):
    return data["logging_level"] if "logging_level" in data else LogLevel.Nothing

def set_logging_level(data:dict, level:int):
    data["logging_level"] = level

def log_data(data:dict):
    data_filename = os.path.join(get_logging_dir(data), 'data.pickle')
    print("Saving data pickle to " + data_filename)
    with open(data_filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
def log_image(data:dict, name:str, image, extension=".jpg"):
    if im_logging_enabled(data, LogLevel.Images):
        _log_image(data, name, image.astype(np.uint8), extension)

def _log_image(data:dict, name:str, image, extension=".jpg"):
    global _logging_index
    path = make_log_path(data, name, extension)
    print("Saving image", path)
    cv2.imwrite(path, image)
    _logging_index += 1

def get_segmentation_image(labels, image, avg=False, resize=True):
    img_seg = image
    if resize:
        img_seg = cv2.resize(image, (labels.shape[1], labels.shape[0]))

    for label in range(1, np.amax(labels) + 1):
        color = np.random.randint([0, 0, 10], [254, 254, 235])
        if avg: color = np.mean(img_seg[labels == label], axis=0)
        img_seg[labels == label] = color
    return img_seg

def log_segmentation_image(data:dict, name, segmentation, image, avg=False, extension=".jpg"):
    if im_logging_enabled(data, LogLevel.Segmentation):
        _log_image(data, name, get_segmentation_image(segmentation, image, avg), extension)

def log_ply(data:dict, name, image, masks, plane_XYZ, write_occlusion=False, mult=1.0):
    global _logging_index
    if im_logging_enabled(data, LogLevel.Models):
        file_path = make_log_path(data, name, ".ply")
        print("Saving model", file_path)
        _logging_index += 1

        image = cv2.resize(image, (int(mult * 160), int(mult * 120)))
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

        for mask_index, (mask, XYZ) in enumerate(zip(masks, plane_XYZ)):
            indices = np.nonzero(np.logical_and(segmentation == mask_index, masks[mask_index] > 0.01))
            # indices = np.nonzero(masks[mask_index] > .05)
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
