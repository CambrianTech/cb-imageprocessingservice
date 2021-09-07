import cv2
import numpy as np
from pipeline.ade20k import ADE20K

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

def get_segmentation_image(labels, image, avg=False, resize=True, get_legend=False, min_matches=100, labelset=ADE20K):
    if resize:
        img_seg = cv2.resize(image, (labels.shape[1], labels.shape[0]))
    else:
        img_seg = image.copy()

    legend = []

    for label in range(0, np.amax(labels) + 1):
        color = np.random.randint([0, 0, 10], [254, 254, 235])
        if avg: color = np.mean(img_seg[labels == label], axis=0)
        img_seg[labels == label] = color

        if get_legend and label <= labelset.max_index() and len(img_seg[labels == label]) > min_matches:
            legend.append((labelset(label+labelset.value_offset()), (int(color[0]), int(color[1]), int(color[2]))))
    if get_legend:
        return img_seg, legend
    return img_seg