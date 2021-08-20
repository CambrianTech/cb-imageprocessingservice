import cv2
import numpy as np

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