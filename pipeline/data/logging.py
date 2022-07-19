import cv2
import os.path
import numpy as np
import pickle
from enum import IntFlag
from termcolor import colored
from time import time
import sys

from pipeline.misc.utils import random_color
from .ade20k import ADE20K

class Timer():

    def __init__(self, prefix=None, color="cyan"):
        self.prefix = prefix
        self.color = color
        self.enabled = True
        self.counters = {}
        self.total_elapsed = {}
        self.start = time()
        self.checktime = self.start

    def reset(self):
        self.checktime = time()

    def time_event(self, name):
        self.log_elapsed(name, every=1000000)

    def log_event(self, name): #print timing accumulation and mean, and then trigger reset
        self.log_elapsed(name, every=1) #trigger print and reset now

    def log_all_events(self):
        names = list(self.counters.keys())
        [self.log_elapsed(name, every=1, disabled_prefix=True) for name in names]
        if len(names) != 1: #if not a single timer output the total:
            print(colored("%s took %.4f seconds total" % (self.prefix, time() - self.start), self.color, attrs=['bold'] if len(names) > 1 else None))

    def log_elapsed(self, name, every=None, description=None, disabled_prefix=False):
        # if not self.enabled: return

        elapsed = time() - self.checktime
        self.reset()

        if self.prefix is not None and not disabled_prefix:
            name = self.prefix + "." + name
               
        self.total_elapsed[name] = elapsed + self.total_elapsed[name] if name in self.total_elapsed else elapsed
        elapsed = self.total_elapsed[name]

        if every is not None:
            self.counters[name] = 1 + self.counters[name] if name in self.counters else 1

            if self.counters[name] % every != 0:
                return

            every = self.counters[name] - 1
            self.counters[name] = 0
            iterations_text = " for %d iterations, avg: %.4f" % (every, elapsed / every) if every > 1 else ""
            print(colored("%s took a total of %.4f seconds%s" % (name, elapsed, iterations_text), self.color))
        else:
            print(colored("%s took %.4f seconds" % (name, elapsed), self.color))

        self.total_elapsed[name] = 0
        
        if description is not None:
            print("--", colored(description, "grey"))

    def disable(self):
        self.enabled = False

    def enable(self):
        self.enabled = True

class LogLevel(IntFlag):
    Nothing =       0
    Images =        0x1 << 0
    Segmentation =  0x1 << 1
    Markers =       0x1 << 2
    Lines =         0x1 << 3
    Models =        0x1 << 4

    Default =       Images | Segmentation | Lines
    All =           0xff

def get_unique_id(data:dict):
    return data["unique_id"]

def make_log_path(data:dict, name:str, extension=".jpg"):
    directory = get_logging_dir(data)
    if not os.path.exists(directory):
        #print("Creating directory" + directory)
        os.makedirs(directory)

    filename = "%s%s" % (name, extension)
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

def log_mask(data:dict, name:str, mask, background=None, extension=".jpg", primary=False):
    if im_logging_enabled(data, LogLevel.Images) or (im_logging_enabled(data) and primary):
        image = np.zeros_like(mask)
        image[mask > 0] = 255
        if background is not None:
            image = overlay_image(image, background)
        _log_image(data, name, image, extension)
                
def log_image(data:dict, name:str, image, extension=".jpg", primary=False):
    if im_logging_enabled(data, LogLevel.Images) or (im_logging_enabled(data) and primary):
        _log_image(data, name, image, extension)

def _log_image(data:dict, name:str, image, extension=".jpg", quality=95):
    parts = os.path.splitext(name)
    if len(parts)==2 and len(parts[1]) > 2:
        name = parts[0]
        extension = parts[1]
    path = make_log_path(data, name, extension)
    #print("Save image %s" % path)
    success = cv2.imwrite(path, cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else image.astype(np.uint8), [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not success:
        print("Could not save image", path)

def draw_legend(data:dict, debug:np.ndarray, legend:tuple):
    #draw legend
    font_scale = min(max(0.5, debug.shape[0] / 1000), 2)
    thickness = max(int(font_scale * 2), 1)
    radius = int(12 * font_scale)
    padding = int(12 * font_scale)
    line_height = int(40 * font_scale)
    text_color = (50,50,50)
    
    font = cv2.FONT_HERSHEY_SIMPLEX

    text_height = cv2.getTextSize(text=str("Just Some Text"), fontFace=font, fontScale=font_scale, thickness=thickness)[0][1]

    start_location = (padding * 2 + radius, padding * 2 + line_height // 2)
    
    x = start_location[0]
    y = start_location[1]
    
    for label, color in legend:
        cv2.circle(debug, (x+radius, y+radius), radius, color, cv2.FILLED) 
        cv2.circle(debug, (x+radius, y+radius), radius, text_color, min(thickness, 2))
        x += 2 * radius + padding
        text_y = y + text_height + int(2 * font_scale)
        cv2.putText(debug, label, (x, text_y), font, font_scale, text_color, thickness, cv2.LINE_AA)

        x = start_location[0]
        y += line_height

def get_markers_image(data:dict, markers, mask=None, num_labels=None):
    if num_labels is None:
        num_labels = markers.max()

    alpha = 255 * (num_labels + 1) / (num_labels + 2)
    debug = markers * alpha

    if mask is not None:
        debug[mask == 0] = 255

    debug[markers == -1] = 128
    return debug

def overlay_image(foreground, background, opacity=0.5):
    _foreground = foreground.astype(np.uint8)
    _background = background.astype(np.uint8)

    if len(_foreground.shape) != len(_background.shape):
        if len(_foreground.shape) == 2:
            _foreground = cv2.cvtColor(_foreground, cv2.COLOR_GRAY2RGB)
        elif len(_background.shape) == 2:
            _background = cv2.cvtColor(_background, cv2.COLOR_GRAY2RGB)
            
    return cv2.addWeighted(_background, opacity, _foreground, 1.0 - opacity, 0)

def log_markers(data:dict, name, markers, mask=None, num_labels=None, primary=False):
    if im_logging_enabled(data, LogLevel.Markers) or (primary and im_logging_enabled(data)):
        debug = get_markers_image(data, markers, mask, num_labels)
        _log_image(data, name, debug)

def get_segmentation_image(labels, image, avg=False, resize=True, min_matches=100, labelset=None):
    if resize:
        img_seg = cv2.resize(image, (labels.shape[1], labels.shape[0]))
    else:
        img_seg = image.copy()

    legend = []

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]

    for label in range(0, np.amax(labels) + 1):
        color = colors[label] if label < len(colors) else random_color()
        
        if avg: color = np.mean(img_seg[labels == label], axis=0)
        img_seg[labels == label] = color

        if labelset is not None and len(img_seg[labels == label]) > min_matches:
            color_value =  (int(color[0]), int(color[1]), int(color[2]))
            if type(labelset) == dict or type(labelset) == list:
                text = labelset[label] if label in labelset else "other"
                legend.append((text, color_value))
            elif type(labelset) == list:
                legend.append((labelset[label], (int(color[0]), int(color[1]), int(color[2]))))
            elif label <= labelset.max_index():
                legend.append((labelset(label+labelset.value_offset()).name, (int(color[0]), int(color[1]), int(color[2]))))

    if len(legend) > 0:
        return img_seg, legend
    return img_seg

def log_segmentation_image(data:dict, name, segmentation, image, avg=False, extension=".jpg", labelset=ADE20K,  opacity=0.5, get_image=False, min_matches=100, primary=False):
    
    if get_image or im_logging_enabled(data, LogLevel.Segmentation) or (primary and im_logging_enabled(data)):
        
        debug = get_segmentation_image(segmentation, image, avg, labelset=labelset, min_matches=min_matches)

        if labelset:
            debug, legend = debug

        if debug.shape != image.shape:
            debug = cv2.resize(debug, (image.shape[1], image.shape[0]))
            
        debug = cv2.addWeighted(debug, opacity, image, 1.0 - opacity, 0)

        if labelset:
            draw_legend(data, debug, legend)

        if get_image:
            return debug

        _log_image(data, name, debug, extension)

def log_model(data:dict, name, image, masks, plane_XYZ, write_occlusion=False, mult=1.0):
    if im_logging_enabled(data, LogLevel.Models):
        file_path = make_log_path(data, name, ".ply")
        print("Saving model", file_path)

        image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB)
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
