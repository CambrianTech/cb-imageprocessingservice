import cv2
import numpy as np
import random
from scipy.spatial import distance

def partition(pred, iterable):
    trues = []
    falses = []
    for item in iterable:
        if pred(item):
            trues.append(item)
        else:
            falses.append(item)
    return trues, falses

def multi_filter(fs, l):
    if not fs:
        return l
    return multi_filter(fs[1:], (x for x in l if fs[0](x)))

def random_color():
    random.seed()
    haystack = np.arange(80, 255, 10)
    random.shuffle(haystack)
    return (int(haystack[0]), int(haystack[1]), int(haystack[2]))

def resize_array(array, shape):
    length = len(array)

    dim=1
    array_lr = np.zeros((length, shape[1], shape[0]))

    if array[0].ndim>2:
        dim = array.shape[-1]
        array_lr = np.zeros((length, shape[1], shape[0], dim))

    for k in range(length):
        array_lr[k] = cv2.resize(array[k], shape)

    return array_lr

def adjust_mask(func, mask, size=5, iterations=1, scale=0.5, maintain_size=True, interpolation=cv2.INTER_NEAREST):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(size,size))
    shape = mask.shape
    result = cv2.resize(mask, (int(shape[1] * scale), int(shape[0] * scale)), interpolation)
    result = func(result, kernel, iterations=iterations)
    if maintain_size:
        result = cv2.resize(result, (shape[1], shape[0]), interpolation)
    return result

def sample_at_point(img, point, size=20):
    half_size = size // 2
    x1 = max(point[0] - half_size, 0)
    x2 = min(x1 + half_size, img.shape[1]-1)

    y1 = max(point[1] - half_size, 0)
    y2 = min(y1 + half_size, img.shape[0]-1)

    return img[y1:y2, x1:x2]

def get_segmentation_image(labels, image, avg=False, resize=True, min_matches=100, labelset=None):
    if resize:
        img_seg = cv2.resize(image, (labels.shape[1], labels.shape[0]))
    else:
        img_seg = image.copy()

    legend = []

    for label in range(0, np.amax(labels) + 1):
        color = np.random.randint([0, 0, 10], [254, 254, 235])
        if avg: color = np.mean(img_seg[labels == label], axis=0)
        img_seg[labels == label] = color

        if labelset is not None and len(img_seg[labels == label]) > min_matches:
            if type(labelset) == list:
                legend.append((labelset[label], (int(color[0]), int(color[1]), int(color[2]))))
            elif label <= labelset.max_index():
                legend.append((labelset(label+labelset.value_offset()).name, (int(color[0]), int(color[1]), int(color[2]))))

    if len(legend) > 0:
        return img_seg, legend
    return img_seg

def fov_to_focal(fov, length):
    return length / (2 * np.tan(np.radians(fov) / 2))

def focal_to_fov(focal_length, length):
    return 2 * np.arctan2(length,(2 * focal_length))

def camera_fov_res_to_intrinsics(fov: float, res: np.ndarray):

    c = res / 2
    f = c[0] / np.tan(np.radians(fov) / 2)
    K = np.array([f, f, c[0], c[1], res[0], res[1]], dtype=np.float32)

    return K, f

def camera_fov_to_intrinsic_matrix(fov,w,h):
    K = np.eye(3)

    K[0, :] = [fov_to_focal(fov, w),  w/2,0]
    K[1, :] = [0, h/2, -fov_to_focal(fov, w)]
    K[2, :] = [0, 1, 0]
    return K

def camera_focal_length_to_intrinsic_matrix(focal_length, w, h):
    K = np.eye(3)

    K[0, :] = [focal_length,  w/2,0]
    K[1, :] = [0, h/2, -focal_length]
    K[2, :] = [0, 1, 0]
    return K

def calculate_plane_xyz(planes, width, height, camera, max_depth=10):
    urange = (np.arange(width, dtype=np.float32) / (width) * (camera[4]) - camera[2]) / camera[0]
    urange = urange.reshape(1, -1).repeat(height, 0)

    vrange = (np.arange(height, dtype=np.float32) / (height) * (camera[5]) - camera[3]) / camera[1]
    vrange = vrange.reshape(-1, 1).repeat(width, 1)

    ranges = np.stack([urange, np.ones(urange.shape), -vrange], axis=-1)

    planeOffsets = np.linalg.norm(planes, axis=-1, keepdims=True)
    planeNormals = planes / np.maximum(planeOffsets, 1e-4)

    normalXYZ = np.dot(ranges, planeNormals.transpose())
    # normalXYZ = np.round(normalXYZ,2)


    normalXYZ[normalXYZ == 0] = 1e-4

    planeDepths = planeOffsets.squeeze(-1) / normalXYZ
    if max_depth > 0:
        planeDepths = np.clip(planeDepths, 0, max_depth)
        pass
    XYZ = (np.expand_dims(planeDepths, -1) * np.expand_dims(ranges, 2))
    
    return XYZ.transpose(2, 0, 1, 3), planeDepths.transpose(2, 0, 1)

def overlay_mask(img, mask, hue=None, saturation=255, darkest_value=80):
    
    overlay = cv2.resize(mask, (img.shape[1], img.shape[0]))
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL) #range 0-180

    grey = img_hsv[:, :, 2].copy()
    grey[grey<darkest_value] = darkest_value

    if hue is None:
        hue = random.randint(0,360)

    img_hsv[:, :, 0][overlay>0] = hue
    img_hsv[:, :, 1][overlay>0] = saturation
    img_hsv[:, :, 2][overlay>0] = grey[overlay>0] 

    out = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR_FULL)

    return out

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

def scale_contour(cnt, scale, moments=None):

    if moments is None:
        moments = cv2.moments(cnt)

    cx = int(moments['m10']/moments['m00'])
    cy = int(moments['m01']/moments['m00'])

    cnt_norm = cnt - [cx, cy]
    cnt_scaled = cnt_norm * scale
    cnt_scaled = cnt_scaled + [cx, cy]
    cnt_scaled = cnt_scaled.astype(np.int32)

    return cnt_scaled

def overlay_image_alpha(img, img_overlay, x, y, alpha_mask=None):
    """Overlay `img_overlay` onto `img` at (x, y) and blend using optional `alpha_mask`.

    `alpha_mask` must have same HxW as `img_overlay` and values in range [0, 1].
    """
    # Image ranges

    if y < 0 or y + img_overlay.shape[0] > img.shape[0] or x < 0 or x + img_overlay.shape[1] > img.shape[1]:
        y_origin = 0 if y > 0 else -y
        y_end = img_overlay.shape[0] if y < 0 else min(img.shape[0] - y, img_overlay.shape[0])

        x_origin = 0 if x > 0 else -x
        x_end = img_overlay.shape[1] if x < 0 else min(img.shape[1] - x, img_overlay.shape[1])

        img_overlay_crop = img_overlay[y_origin:y_end, x_origin:x_end]
        alpha = alpha_mask[y_origin:y_end, x_origin:x_end] if alpha_mask is not None else None
    else:
        img_overlay_crop = img_overlay
        alpha = alpha_mask

    y1 = max(y, 0)
    y2 = min(img.shape[0], y1 + img_overlay_crop.shape[0])

    x1 = max(x, 0)
    x2 = min(img.shape[1], x1 + img_overlay_crop.shape[1])

    img_crop = img[y1:y2, x1:x2]
    img_crop[:] = alpha * img_overlay_crop + (1.0 - alpha) * img_crop if alpha is not None else img_overlay_crop

def convert_color(color, conversion):
    #return tuple(int(i) for i in cv2.cvtColor(img, conversion).flatten())
    
    #bug in opencv 4.5.4 incorrectly asserting on width or height parameter instead of channels
    #return tuple(int(i) for i in cv2.cvtColor(img, conversion).flatten())
    img = np.zeros([3,3,3],dtype=np.uint8)
    img[0,0] = color
    converted = cv2.cvtColor(img, conversion)[0,0]
    return (int(converted[0]), int(converted[1]), int(converted[2]))

def put_text(img, text, origin, color, shadow_offset=(1,1), font=cv2.FONT_HERSHEY_SIMPLEX, size=1, thickness=1, line_type=cv2.LINE_AA, shadow=False, highlights=False):

    dimensions = cv2.getTextSize(text, font, size, thickness)[0]
    origin = (int(origin[0]), int(origin[1]))

    loc = min(max(origin[0], 10), img.shape[1] - dimensions[0] - 10), min(max(origin[1], 10 + dimensions[1] // 2), img.shape[0] - dimensions[1] // 2 - 10)

    if highlights or shadow:
        hsv_color = convert_color(color, cv2.COLOR_RGB2HSV_FULL)

        if highlights:
            glow_color = convert_color((hsv_color[0], hsv_color[1] // 3, 255), cv2.COLOR_HSV2RGB_FULL)
            cv2.putText(img, text, (loc[0] - shadow_offset[0], loc[1] - shadow_offset[1]), font, size, glow_color, thickness, line_type)
            
        if shadow:
            shadow_color = convert_color((hsv_color[0], hsv_color[1] // 3, 50), cv2.COLOR_HSV2RGB_FULL)
            cv2.putText(img, text, (loc[0] + shadow_offset[0], loc[1] + shadow_offset[1]), font, size, shadow_color, thickness, line_type)

    cv2.putText(img, text, loc, font, size, color, thickness, line_type)

    return (loc[0], loc[1] + 3 * dimensions[1] // 2)

