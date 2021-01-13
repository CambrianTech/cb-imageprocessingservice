import numpy as np
import cv2
from skimage.segmentation import watershed
from skimage import filters

from . import diagnostics as d

from gluoncv.utils.viz.segmentation import adepallete

_adepallete = np.array(adepallete).reshape(len(adepallete)//3, 3)

def overlay_mask(img, mask, hue=None, saturation=255, darkest_value=80):
    
    overlay = cv2.resize(mask, (img.shape[1], img.shape[0]))
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #range 0-180

    grey = img_hsv[:, :, 2].copy()
    grey[grey<darkest_value] = darkest_value

    if hue is None:
        hue = random.randint(0,360)

    img_hsv[:, :, 0][overlay>0] = int(hue) / 2.0
    img_hsv[:, :, 1][overlay>0] = saturation
    img_hsv[:, :, 2][overlay>0] = grey[overlay>0] 

    out = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

    return out

def rough_dilate_erode(is_dilate, mask, size=5, iterations=1, scale=0.5, maintain_size=True, interpolation=cv2.INTER_NEAREST):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(size,size))
    shape = mask.shape
    mask = cv2.resize(mask, (int(shape[1] * scale), int(shape[0] * scale)), interpolation)
    mask = cv2.dilate(mask, kernel, iterations=iterations) if is_dilate else cv2.erode(mask, kernel, iterations=iterations)
    if maintain_size:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation)
    return mask

def convert_color(color, conversion):
    return tuple(int(i) for i in cv2.cvtColor(np.uint8([[color]]), conversion).flatten())

def get_label_color(label, palette=_adepallete):
    color = palette[label]
    return int(color[0]), int(color[1]), int(color[2])

def get_random_color():
    color = np.random.randint(0, 255, size=(3, ))
    return ( int (color [ 0 ]), int (color [ 1 ]), int (color [ 2 ]))

def soft_light(img_in, img_layer, opacity):
    """
    Apply soft light blending mode of a layer on an image.

    Find more information on `Wikipedia <https://en.wikipedia.org/w/index.php?title=Blend_modes&oldid=747749280#Soft_Light>`__.

    Example::

        import cv2, numpy
        from blend_modes import blend_modes
        img_in = cv2.imread('./orig.png', -1).astype(float)
        img_layer = cv2.imread('./layer.png', -1).astype(float)
        img_out = soft_light(img_in, img_layer, 0.5)
        cv2.imshow('window', img_out.astype(numpy.uint8))
        cv2.waitKey()

    :param img_in: Image to be blended upon
    :type img_in: 3-dimensional numpy array of floats (r/g/b/a) in range 0-255.0
    :param img_layer: Layer to be blended with image
    :type img_layer: 3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0
    :param opacity: Desired opacity of layer for blending
    :type opacity: float
    :return: Blended image
    :rtype: 3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0
    """

    # sanity check of inputs
    assert img_in.dtype.kind == 'f', 'Input variable img_in should be of numpy.float type.'
    assert img_layer.dtype.kind == 'f', 'Input variable img_layer should be of numpy.float type.'
    assert img_in.shape[2] == 4, 'Input variable img_in should be of shape [:, :,4].'
    assert img_layer.shape[2] == 4, 'Input variable img_layer should be of shape [:, :,4].'
    assert 0.0 <= opacity <= 1.0, 'Opacity needs to be between 0.0 and 1.0.'

    img_in /= 255.0
    img_layer /= 255.0

    ratio = _compose_alpha(img_in, img_layer, opacity)

    # The following code does this:
    #   multiply = img_in[:, :, :3] * img_layer[:, :, :3]
    #   screen = 1.0 - (1.0-img_in[:, :, :3])*(1.0-img_layer[:, :, :3])
    #   comp = (1.0 - img_in[:, :, :3]) * multiply + img_in[:, :, :3] * screen
    #   ratio_rs = np.reshape(np.repeat(ratio, 3), comp.shape)
    #   img_out = comp * ratio_rs + img_in[:, :, :3] * (1.0-ratio_rs)

    comp = (1.0 - img_in[:, :, :3]) * img_in[:, :, :3] * img_layer[:, :, :3] \
           + img_in[:, :, :3] * (1.0 - (1.0-img_in[:, :, :3])*(1.0-img_layer[:, :, :3]))

    ratio_rs = np.reshape(np.repeat(ratio, 3), [comp.shape[0], comp.shape[1], comp.shape[2]])
    img_out = comp*ratio_rs + img_in[:, :, :3] * (1.0-ratio_rs)
    img_out = np.nan_to_num(np.dstack((img_out, img_in[:, :, 3])))  # add alpha channel and replace nans
    return img_out * 255.0


def lighten_only(img_in, img_layer, opacity):
    """
    Apply lighten only blending mode of a layer on an image.

    Find more information on `Wikipedia <https://en.wikipedia.org/w/index.php?title=Blend_modes&oldid=747749280#Lighten_Only>`__.

    Example::

        import cv2, numpy
        from blend_modes import blend_modes
        img_in = cv2.imread('./orig.png', -1).astype(float)
        img_layer = cv2.imread('./layer.png', -1).astype(float)
        img_out = lighten_only(img_in, img_layer, 0.5)
        cv2.imshow('window', img_out.astype(numpy.uint8))
        cv2.waitKey()

    :param img_in: Image to be blended upon
    :type img_in: 3-dimensional numpy array of floats (r/g/b/a) in range 0-255.0
    :param img_layer: Layer to be blended with image
    :type img_layer: 3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0
    :param opacity: Desired opacity of layer for blending
    :type opacity: float
    :return: Blended image
    :rtype: 3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0
    """

    # sanity check of inputs
    assert img_in.dtype.kind == 'f', 'Input variable img_in should be of numpy.float type.'
    assert img_layer.dtype.kind == 'f', 'Input variable img_layer should be of numpy.float type.'
    assert img_in.shape[2] == 4, 'Input variable img_in should be of shape [:, :,4].'
    assert img_layer.shape[2] == 4, 'Input variable img_layer should be of shape [:, :,4].'
    assert 0.0 <= opacity <= 1.0, 'Opacity needs to be between 0.0 and 1.0.'

    img_in /= 255.0
    img_layer /= 255.0

    ratio = _compose_alpha(img_in, img_layer, opacity)

    comp = np.maximum(img_in[:, :, :3], img_layer[:, :, :3])

    ratio_rs = np.reshape(np.repeat(ratio, 3), [comp.shape[0], comp.shape[1], comp.shape[2]])
    img_out = comp*ratio_rs + img_in[:, :, :3] * (1.0-ratio_rs)
    img_out = np.nan_to_num(np.dstack((img_out, img_in[:, :, 3])))  # add alpha channel and replace nans
    return img_out * 255.0


def screen(img_in, img_layer, opacity):
    """
    Apply screen blending mode of a layer on an image.

    Find more information on `Wikipedia <https://en.wikipedia.org/w/index.php?title=Blend_modes&oldid=747749280#Screen>`__.

    Example::

        import cv2, numpy
        from blend_modes import blend_modes
        img_in = cv2.imread('./orig.png', -1).astype(float)
        img_layer = cv2.imread('./layer.png', -1).astype(float)
        img_out = screen(img_in, img_layer, 0.5)
        cv2.imshow('window', img_out.astype(numpy.uint8))
        cv2.waitKey()

    :param img_in: Image to be blended upon
    :type img_in: 3-dimensional numpy array of floats (r/g/b/a) in range 0-255.0
    :param img_layer: Layer to be blended with image
    :type img_layer: 3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0
    :param opacity: Desired opacity of layer for blending
    :type opacity: float
    :return: Blended image
    :rtype: 3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0
    """

    # sanity check of inputs
    assert img_in.dtype.kind == 'f', 'Input variable img_in should be of numpy.float type.'
    assert img_layer.dtype.kind == 'f', 'Input variable img_layer should be of numpy.float type.'
    assert img_in.shape[2] == 4, 'Input variable img_in should be of shape [:, :,4].'
    assert img_layer.shape[2] == 4, 'Input variable img_layer should be of shape [:, :,4].'
    assert 0.0 <= opacity <= 1.0, 'Opacity needs to be between 0.0 and 1.0.'

    img_in /= 255.0
    img_layer /= 255.0

    ratio = _compose_alpha(img_in, img_layer, opacity)

    comp = 1.0 - (1.0 - img_in[:, :, :3]) * (1.0 - img_layer[:, :, :3])

    ratio_rs = np.reshape(np.repeat(ratio, 3), [comp.shape[0], comp.shape[1], comp.shape[2]])
    img_out = comp*ratio_rs + img_in[:, :, :3] * (1.0-ratio_rs)
    img_out = np.nan_to_num(np.dstack((img_out, img_in[:, :, 3])))  # add alpha channel and replace nans
    return img_out * 255.0


def dodge(img_in, img_layer, opacity):
    """
    Apply dodge blending mode of a layer on an image.

    Find more information on `Wikipedia <https://en.wikipedia.org/w/index.php?title=Blend_modes&oldid=747749280#Dodge_and_burn>`__.

    Example::

        import cv2, numpy
        from blend_modes import blend_modes
        img_in = cv2.imread('./orig.png', -1).astype(float)
        img_layer = cv2.imread('./layer.png', -1).astype(float)
        img_out = dodge(img_in, img_layer, 0.5)
        cv2.imshow('window', img_out.astype(numpy.uint8))
        cv2.waitKey()

    :param img_in: Image to be blended upon
    :type img_in: 3-dimensional numpy array of floats (r/g/b/a) in range 0-255.0
    :param img_layer: Layer to be blended with image
    :type img_layer: 3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0
    :param opacity: Desired opacity of layer for blending
    :type opacity: float
    :return: Blended image
    :rtype: 3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0
    """

    # sanity check of inputs
    assert img_in.dtype.kind == 'f', 'Input variable img_in should be of numpy.float type.'
    assert img_layer.dtype.kind == 'f', 'Input variable img_layer should be of numpy.float type.'
    assert img_in.shape[2] == 4, 'Input variable img_in should be of shape [:, :,4].'
    assert img_layer.shape[2] == 4, 'Input variable img_layer should be of shape [:, :,4].'
    assert 0.0 <= opacity <= 1.0, 'Opacity needs to be between 0.0 and 1.0.'

    img_in /= 255.0
    img_layer /= 255.0

    ratio = _compose_alpha(img_in, img_layer, opacity)

    comp = np.minimum(img_in[:, :, :3]/(1.0 - img_layer[:, :, :3]), 1.0)

    ratio_rs = np.reshape(np.repeat(ratio, 3), [comp.shape[0], comp.shape[1], comp.shape[2]])
    img_out = comp*ratio_rs + img_in[:, :, :3] * (1.0-ratio_rs)
    img_out = np.nan_to_num(np.dstack((img_out, img_in[:, :, 3])))  # add alpha channel and replace nans
    return img_out * 255.0


def addition(img_in, img_layer, opacity):
    """
    Apply addition blending mode of a layer on an image.

    Find more information on `Wikipedia <https://en.wikipedia.org/w/index.php?title=Blend_modes&oldid=747749280#Addition>`__.

    Example::

        import cv2, numpy
        from blend_modes import blend_modes
        img_in = cv2.imread('./orig.png', -1).astype(float)
        img_layer = cv2.imread('./layer.png', -1).astype(float)
        img_out = addition(img_in, img_layer, 0.5)
        cv2.imshow('window', img_out.astype(numpy.uint8))
        cv2.waitKey()

    :param img_in: Image to be blended upon
    :type img_in: 3-dimensional numpy array of floats (r/g/b/a) in range 0-255.0
    :param img_layer: Layer to be blended with image
    :type img_layer: 3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0
    :param opacity: Desired opacity of layer for blending
    :type opacity: float
    :return: Blended image
    :rtype: 3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0
    """

    # sanity check of inputs
    assert img_in.dtype.kind == 'f', 'Input variable img_in should be of numpy.float type.'
    assert img_layer.dtype.kind == 'f', 'Input variable img_layer should be of numpy.float type.'
    assert img_in.shape[2] == 4, 'Input variable img_in should be of shape [:, :,4].'
    assert img_layer.shape[2] == 4, 'Input variable img_layer should be of shape [:, :,4].'
    assert 0.0 <= opacity <= 1.0, 'Opacity needs to be between 0.0 and 1.0.'

    img_in /= 255.0
    img_layer /= 255.0

    ratio = _compose_alpha(img_in, img_layer, opacity)

    comp = img_in[:, :, :3] + img_layer[:, :, :3]

    ratio_rs = np.reshape(np.repeat(ratio, 3), [comp.shape[0], comp.shape[1], comp.shape[2]])
    img_out = np.clip(comp*ratio_rs + img_in[:, :, :3] * (1.0-ratio_rs), 0.0, 1.0)
    img_out = np.nan_to_num(np.dstack((img_out, img_in[:, :, 3])))  # add alpha channel and replace nans
    return img_out * 255.0


def darken_only(img_in, img_layer, opacity):
    """
    Apply darken only blending mode of a layer on an image.

    Find more information on `Wikipedia <https://en.wikipedia.org/w/index.php?title=Blend_modes&oldid=747749280#Darken_Only>`__.

    Example::

        import cv2, numpy
        from blend_modes import blend_modes
        img_in = cv2.imread('./orig.png', -1).astype(float)
        img_layer = cv2.imread('./layer.png', -1).astype(float)
        img_out = darken_only(img_in, img_layer, 0.5)
        cv2.imshow('window', img_out.astype(numpy.uint8))
        cv2.waitKey()

    :param img_in: Image to be blended upon
    :type img_in: 3-dimensional numpy array of floats (r/g/b/a) in range 0-255.0
    :param img_layer: Layer to be blended with image
    :type img_layer: 3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0
    :param opacity: Desired opacity of layer for blending
    :type opacity: float
    :return: Blended image
    :rtype: 3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0
    """

    # sanity check of inputs
    assert img_in.dtype.kind == 'f', 'Input variable img_in should be of numpy.float type.'
    assert img_layer.dtype.kind == 'f', 'Input variable img_layer should be of numpy.float type.'
    assert img_in.shape[2] == 4, 'Input variable img_in should be of shape [:, :,4].'
    assert img_layer.shape[2] == 4, 'Input variable img_layer should be of shape [:, :,4].'
    assert 0.0 <= opacity <= 1.0, 'Opacity needs to be between 0.0 and 1.0.'

    img_in /= 255.0
    img_layer /= 255.0

    ratio = _compose_alpha(img_in, img_layer, opacity)

    comp = np.minimum(img_in[:, :, :3], img_layer[:, :, :3])

    ratio_rs = np.reshape(np.repeat(ratio, 3), [comp.shape[0], comp.shape[1], comp.shape[2]])
    img_out = comp*ratio_rs + img_in[:, :, :3] * (1.0-ratio_rs)
    img_out = np.nan_to_num(np.dstack((img_out, img_in[:, :, 3])))  # add alpha channel and replace nans
    return img_out * 255.0


def multiply(img_in, img_layer, opacity):
    """
    Apply multiply blending mode of a layer on an image.

    Find more information on `Wikipedia <https://en.wikipedia.org/w/index.php?title=Blend_modes&oldid=747749280#Multiply>`__.

    Example::

        import cv2, numpy
        from blend_modes import blend_modes
        img_in = cv2.imread('./orig.png', -1).astype(float)
        img_layer = cv2.imread('./layer.png', -1).astype(float)
        img_out = multiply(img_in, img_layer, 0.5)
        cv2.imshow('window', img_out.astype(numpy.uint8))
        cv2.waitKey()

    :param img_in: Image to be blended upon
    :type img_in: 3-dimensional numpy array of floats (r/g/b/a) in range 0-255.0
    :param img_layer: Layer to be blended with image
    :type img_layer: 3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0
    :param opacity: Desired opacity of layer for blending
    :type opacity: float
    :return: Blended image
    :rtype: 3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0
    """

    # sanity check of inputs
    assert img_in.dtype.kind == 'f', 'Input variable img_in should be of numpy.float type.'
    assert img_layer.dtype.kind == 'f', 'Input variable img_layer should be of numpy.float type.'
    assert img_in.shape[2] == 4, 'Input variable img_in should be of shape [:, :,4].'
    assert img_layer.shape[2] == 4, 'Input variable img_layer should be of shape [:, :,4].'
    assert 0.0 <= opacity <= 1.0, 'Opacity needs to be between 0.0 and 1.0.'

    img_in /= 255.0
    img_layer /= 255.0

    ratio = _compose_alpha(img_in, img_layer, opacity)

    comp = np.clip(img_layer[:, :, :3] * img_in[:, :, :3], 0.0, 1.0)

    ratio_rs = np.reshape(np.repeat(ratio, 3), [comp.shape[0], comp.shape[1], comp.shape[2]])
    img_out = comp * ratio_rs + img_in[:, :, :3] * (1.0 - ratio_rs)
    img_out = np.nan_to_num(np.dstack((img_out, img_in[:, :, 3])))  # add alpha channel and replace nans
    return img_out * 255.0


def hard_light(img_in, img_layer, opacity):
    """
    Apply hard light blending mode of a layer on an image.

    Find more information on `Wikipedia <https://en.wikipedia.org/w/index.php?title=Blend_modes&oldid=747749280#Hard_Light>`__.

    Example::

        import cv2, numpy
        from blend_modes import blend_modes
        img_in = cv2.imread('./orig.png', -1).astype(float)
        img_layer = cv2.imread('./layer.png', -1).astype(float)
        img_out = hard_light(img_in, img_layer, 0.5)
        cv2.imshow('window', img_out.astype(numpy.uint8))
        cv2.waitKey()

    :param img_in: Image to be blended upon
    :type img_in: 3-dimensional numpy array of floats (r/g/b/a) in range 0-255.0
    :param img_layer: Layer to be blended with image
    :type img_layer: 3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0
    :param opacity: Desired opacity of layer for blending
    :type opacity: float
    :return: Blended image
    :rtype: 3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0
    """

    # sanity check of inputs
    assert img_in.dtype.kind == 'f', 'Input variable img_in should be of numpy.float type.'
    assert img_layer.dtype.kind == 'f', 'Input variable img_layer should be of numpy.float type.'
    assert img_in.shape[2] == 4, 'Input variable img_in should be of shape [:, :,4].'
    assert img_layer.shape[2] == 4, 'Input variable img_layer should be of shape [:, :,4].'
    assert 0.0 <= opacity <= 1.0, 'Opacity needs to be between 0.0 and 1.0.'

    img_in /= 255.0
    img_layer /= 255.0

    ratio = _compose_alpha(img_in, img_layer, opacity)

    comp = np.greater(img_layer[:, :, :3], 0.5)*np.minimum(1.0-((1.0 - img_in[:, :, :3])
                                                             * (1.0 - (img_layer[:, :, :3] - 0.5) * 2.0)), 1.0) \
           + np.logical_not(np.greater(img_layer[:, :, :3], 0.5))*np.minimum(img_in[:, :, :3]
                                                                            * (img_layer[:, :, :3] * 2.0), 1.0)

    ratio_rs = np.reshape(np.repeat(ratio, 3), [comp.shape[0], comp.shape[1], comp.shape[2]])
    img_out = comp*ratio_rs + img_in[:, :, :3] * (1.0-ratio_rs)
    img_out = np.nan_to_num(np.dstack((img_out, img_in[:, :, 3])))  # add alpha channel and replace nans
    return img_out * 255.0


def difference(img_in, img_layer, opacity):
    """
    Apply difference blending mode of a layer on an image.

    Find more information on `Wikipedia <https://en.wikipedia.org/w/index.php?title=Blend_modes&oldid=747749280#Difference>`__.

    Example::

        import cv2, numpy
        from blend_modes import blend_modes
        img_in = cv2.imread('./orig.png', -1).astype(float)
        img_layer = cv2.imread('./layer.png', -1).astype(float)
        img_out = difference(img_in, img_layer, 0.5)
        cv2.imshow('window', img_out.astype(numpy.uint8))
        cv2.waitKey()

    :param img_in: Image to be blended upon
    :type img_in: 3-dimensional numpy array of floats (r/g/b/a) in range 0-255.0
    :param img_layer: Layer to be blended with image
    :type img_layer: 3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0
    :param opacity: Desired opacity of layer for blending
    :type opacity: float
    :return: Blended image
    :rtype: 3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0
    """

    # sanity check of inputs
    assert img_in.dtype.kind == 'f', 'Input variable img_in should be of numpy.float type.'
    assert img_layer.dtype.kind == 'f', 'Input variable img_layer should be of numpy.float type.'
    assert img_in.shape[2] == 4, 'Input variable img_in should be of shape [:, :,4].'
    assert img_layer.shape[2] == 4, 'Input variable img_layer should be of shape [:, :,4].'
    assert 0.0 <= opacity <= 1.0, 'Opacity needs to be between 0.0 and 1.0.'

    img_in /= 255.0
    img_layer /= 255.0

    ratio = _compose_alpha(img_in, img_layer, opacity)

    comp = img_in[:, :, :3] - img_layer[:, :, :3]
    comp[comp < 0.0] *= -1.0

    ratio_rs = np.reshape(np.repeat(ratio, 3), [comp.shape[0], comp.shape[1], comp.shape[2]])
    img_out = comp * ratio_rs + img_in[:, :, :3] * (1.0 - ratio_rs)
    img_out = np.nan_to_num(np.dstack((img_out, img_in[:, :, 3])))  # add alpha channel and replace nans
    return img_out * 255.0


def subtract(img_in, img_layer, opacity):
    """
    Apply subtract blending mode of a layer on an image.

    Find more information on `Wikipedia <https://en.wikipedia.org/w/index.php?title=Blend_modes&oldid=747749280#Subtract>`__.

    Example::

        import cv2, numpy
        from blend_modes import blend_modes
        img_in = cv2.imread('./orig.png', -1).astype(float)
        img_layer = cv2.imread('./layer.png', -1).astype(float)
        img_out = subtract(img_in, img_layer, 0.5)
        cv2.imshow('window', img_out.astype(numpy.uint8))
        cv2.waitKey()

    :param img_in: Image to be blended upon
    :type img_in: 3-dimensional numpy array of floats (r/g/b/a) in range 0-255.0
    :param img_layer: Layer to be blended with image
    :type img_layer: 3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0
    :param opacity: Desired opacity of layer for blending
    :type opacity: float
    :return: Blended image
    :rtype: 3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0
    """

    # sanity check of inputs
    assert img_in.dtype.kind == 'f', 'Input variable img_in should be of numpy.float type.'
    assert img_layer.dtype.kind == 'f', 'Input variable img_layer should be of numpy.float type.'
    assert img_in.shape[2] == 4, 'Input variable img_in should be of shape [:, :,4].'
    assert img_layer.shape[2] == 4, 'Input variable img_layer should be of shape [:, :,4].'
    assert 0.0 <= opacity <= 1.0, 'Opacity needs to be between 0.0 and 1.0.'

    img_in /= 255.0
    img_layer /= 255.0

    ratio = _compose_alpha(img_in, img_layer, opacity)

    comp = img_in[:, :, :3] - img_layer[:, :, :3]

    ratio_rs = np.reshape(np.repeat(ratio, 3), [comp.shape[0], comp.shape[1], comp.shape[2]])
    img_out = np.clip(comp * ratio_rs + img_in[:, :, :3] * (1.0 - ratio_rs), 0.0, 1.0)
    img_out = np.nan_to_num(np.dstack((img_out, img_in[:, :, 3])))  # add alpha channel and replace nans
    return img_out * 255.0


def grain_extract(img_in, img_layer, opacity):
    """
    Apply grain extract blending mode of a layer on an image.

    Find more information on the `KDE UserBase Wiki <https://userbase.kde.org/Krita/Manual/Blendingmodes#Grain_Extract>`__.

    Example::

        import cv2, numpy
        from blend_modes import blend_modes
        img_in = cv2.imread('./orig.png', -1).astype(float)
        img_layer = cv2.imread('./layer.png', -1).astype(float)
        img_out = grain_extract(img_in, img_layer, 0.5)
        cv2.imshow('window', img_out.astype(numpy.uint8))
        cv2.waitKey()

    :param img_in: Image to be blended upon
    :type img_in: 3-dimensional numpy array of floats (r/g/b/a) in range 0-255.0
    :param img_layer: Layer to be blended with image
    :type img_layer: 3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0
    :param opacity: Desired opacity of layer for blending
    :type opacity: float
    :return: Blended image
    :rtype: 3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0
    """

    # sanity check of inputs
    assert img_in.dtype.kind == 'f', 'Input variable img_in should be of numpy.float type.'
    assert img_layer.dtype.kind == 'f', 'Input variable img_layer should be of numpy.float type.'
    assert img_in.shape[2] == 4, 'Input variable img_in should be of shape [:, :,4].'
    assert img_layer.shape[2] == 4, 'Input variable img_layer should be of shape [:, :,4].'
    assert 0.0 <= opacity <= 1.0, 'Opacity needs to be between 0.0 and 1.0.'

    img_in /= 255.0
    img_layer /= 255.0

    ratio = _compose_alpha(img_in, img_layer, opacity)

    comp = np.clip(img_in[:, :, :3] - img_layer[:, :, :3] + 0.5, 0.0, 1.0)

    ratio_rs = np.reshape(np.repeat(ratio, 3), [comp.shape[0], comp.shape[1], comp.shape[2]])
    img_out = comp * ratio_rs + img_in[:, :, :3] * (1.0 - ratio_rs)
    img_out = np.nan_to_num(np.dstack((img_out, img_in[:, :, 3])))  # add alpha channel and replace nans
    return img_out * 255.0


def grain_merge(img_in, img_layer, opacity):
    """
    Apply grain merge blending mode of a layer on an image.

    Find more information on the `KDE UserBase Wiki <https://userbase.kde.org/Krita/Manual/Blendingmodes#Grain_Merge>`__.

    Example::

        import cv2, numpy
        from blend_modes import blend_modes
        img_in = cv2.imread('./orig.png', -1).astype(float)
        img_layer = cv2.imread('./layer.png', -1).astype(float)
        img_out = grain_merge(img_in, img_layer, 0.5)
        cv2.imshow('window', img_out.astype(numpy.uint8))
        cv2.waitKey()

    :param img_in: Image to be blended upon
    :type img_in: 3-dimensional numpy array of floats (r/g/b/a) in range 0-255.0
    :param img_layer: Layer to be blended with image
    :type img_layer: 3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0
    :param opacity: Desired opacity of layer for blending
    :type opacity: float
    :return: Blended image
    :rtype: 3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0
    """

    # sanity check of inputs
    assert img_in.dtype.kind == 'f', 'Input variable img_in should be of numpy.float type.'
    assert img_layer.dtype.kind == 'f', 'Input variable img_layer should be of numpy.float type.'
    assert img_in.shape[2] == 4, 'Input variable img_in should be of shape [:, :,4].'
    assert img_layer.shape[2] == 4, 'Input variable img_layer should be of shape [:, :,4].'
    assert 0.0 <= opacity <= 1.0, 'Opacity needs to be between 0.0 and 1.0.'

    img_in /= 255.0
    img_layer /= 255.0

    ratio = _compose_alpha(img_in, img_layer, opacity)

    comp = np.clip(img_in[:, :, :3] + img_layer[:, :, :3] - 0.5, 0.0, 1.0)

    ratio_rs = np.reshape(np.repeat(ratio, 3), [comp.shape[0], comp.shape[1], comp.shape[2]])
    img_out = comp * ratio_rs + img_in[:, :, :3] * (1.0 - ratio_rs)
    img_out = np.nan_to_num(np.dstack((img_out, img_in[:, :, 3])))  # add alpha channel and replace nans
    return img_out * 255.0


def divide(img_in, img_layer, opacity):
    """
    Apply divide blending mode of a layer on an image.

    Find more information on `Wikipedia <https://en.wikipedia.org/w/index.php?title=Blend_modes&oldid=747749280#Divide>`__.

    Example::

        import cv2, numpy
        from blend_modes import blend_modes
        img_in = cv2.imread('./orig.png', -1).astype(float)
        img_layer = cv2.imread('./layer.png', -1).astype(float)
        img_out = divide(img_in, img_layer, 0.5)
        cv2.imshow('window', img_out.astype(numpy.uint8))
        cv2.waitKey()

    :param img_in: Image to be blended upon
    :type img_in: 3-dimensional numpy array of floats (r/g/b/a) in range 0-255.0
    :param img_layer: Layer to be blended with image
    :type img_layer: 3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0
    :param opacity: Desired opacity of layer for blending
    :type opacity: float
    :return: Blended image
    :rtype: 3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0
    """

    # sanity check of inputs
    assert img_in.dtype.kind == 'f', 'Input variable img_in should be of numpy.float type.'
    assert img_layer.dtype.kind == 'f', 'Input variable img_layer should be of numpy.float type.'
    assert img_in.shape[2] == 4, 'Input variable img_in should be of shape [:, :,4].'
    assert img_layer.shape[2] == 4, 'Input variable img_layer should be of shape [:, :,4].'
    assert 0.0 <= opacity <= 1.0, 'Opacity needs to be between 0.0 and 1.0.'

    img_in /= 255.0
    img_layer /= 255.0

    ratio = _compose_alpha(img_in, img_layer, opacity)

    comp = np.minimum((256.0 / 255.0 * img_in[:, :, :3]) / (1.0 / 255.0 + img_layer[:, :, :3]), 1.0)

    ratio_rs = np.reshape(np.repeat(ratio, 3), [comp.shape[0], comp.shape[1], comp.shape[2]])
    img_out = comp * ratio_rs + img_in[:, :, :3] * (1.0 - ratio_rs)
    img_out = np.nan_to_num(np.dstack((img_out, img_in[:, :, 3])))  # add alpha channel and replace nans
    return img_out * 255.0


def overlay(img_in, img_layer, opacity):
    """
    Apply overlay blending mode of a layer on an image.

    Find more information on `https://docs.gimp.org/en/gimp-concepts-layer-modes.html`

    Example::

        import cv2, numpy
        from blend_modes import blend_modes
        img_in = cv2.imread('./orig.png', -1).astype(float)
        img_layer = cv2.imread('./layer.png', -1).astype(float)
        img_out = overlay(img_in, img_layer, 0.5)
        cv2.imshow('window', img_out.astype(numpy.uint8))
        cv2.waitKey()

    :param img_in: Image to be blended upon
    :type img_in: 3-dimensional numpy array of floats (r/g/b/a) in range 0-255.0
    :param img_layer: Layer to be blended with image
    :type img_layer: 3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0
    :param opacity: Desired opacity of layer for blending
    :type opacity: float
    :return: Blended image
    :rtype: 3-dimensional numpy array of floats (r/g/b/a) in range 0.0-255.0
    """

    # sanity check of inputs
    assert img_in.dtype.kind == 'f', 'Input variable img_in should be of numpy.float type.'
    assert img_layer.dtype.kind == 'f', 'Input variable img_layer should be of numpy.float type.'
    assert img_in.shape[2] == 4, 'Input variable img_in should be of shape [:, :,4].'
    assert img_layer.shape[2] == 4, 'Input variable img_layer should be of shape [:, :,4].'
    assert 0.0 <= opacity <= 1.0, 'Opacity needs to be between 0.0 and 1.0.'

    img_in /= 255.0
    img_layer /= 255.0

    ratio = _compose_alpha(img_in, img_layer, opacity)

    comp = img_in[:,:,:3] * (img_in[:,:,:3] + (2 * img_layer[:,:,:3]) * (1 - img_in[:,:,:3]))

    ratio_rs = np.reshape(np.repeat(ratio, 3), [comp.shape[0], comp.shape[1], comp.shape[2]])
    img_out = comp * ratio_rs + img_in[:, :, :3] * (1.0 - ratio_rs)
    img_out = np.nan_to_num(np.dstack((img_out, img_in[:, :, 3])))  # add alpha channel and replace nans
    return img_out * 255.0


def _compose_alpha(img_in, img_layer, opacity):
    """
    Calculate alpha composition ratio between two images.
    """
    comp_alpha = np.minimum(img_in[:, :, 3], img_layer[:, :, 3]) * opacity
    new_alpha = img_in[:, :, 3] + (1.0 - img_in[:, :, 3]) * comp_alpha
    np.seterr(divide='ignore', invalid='ignore')
    ratio = comp_alpha / new_alpha
    ratio[ratio == np.NAN] = 0.0
    return ratio


def constrain_image(image, width=None, height=None, inter=None):
    """Constrains image to be within width and height and returns the result"""
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    if inter is None:
        if len(image.shape) == 2:
            inter = cv2.INTER_NEAREST
        else:
            inter = cv2.INTER_AREA

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def crop_image(img, margin=30):
    """Crops the image using the given scalar margin and returns the result"""
    h, w = img.shape[:2]

    result = img[margin:h-margin, margin:w-margin]
    
    return result


def refine_mask_watershed(args, rgb, mask, image_name, distance=0.0, erode=0, max_value=151, gradient=False,
                          background=True, watershed_mask=None):
    """Runs the watershed algorithm on rgb and returns markers"""
    if gradient:
        markers = np.zeros(mask.shape, dtype=np.int32)

        for i in range(max_value + 1):
            num_elements = (mask == i).sum()

            if num_elements > 50:
                isolated = np.zeros(mask.shape, dtype=np.uint8)
                isolated[mask == i] = i + 1  # add one for 0 label, all values are one higher

                if distance == 0.0:
                    erode_amount = erode
                    if erode == 0:
                        erode_amount = int(min(15.0, np.sqrt(num_elements) / 10.0))  # calculate

                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_amount, erode_amount))
                    isolated = cv2.erode(isolated, kernel)
                else:
                    isolated = cv2.distanceTransform(isolated, cv2.DIST_L2, 5)
                    ret, isolated = cv2.threshold(isolated, distance * isolated.max(), i + 1, 0)

                markers += np.uint8(isolated)

        # d.save_diagnostics_image(args, markers, image_name, "markers", verbose=True)

        if len(rgb.shape) < 3:
            # print("watershed used on bw")
            base = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)
            markers = np.int32(watershed(rgb, markers, mask=watershed_mask))
    else:

        markers = np.zeros(mask.shape, dtype=np.int32)

        for i in range(max_value + 1):
            num_elements = (mask == i).sum()
            if num_elements > 50:
                isolated = np.zeros(mask.shape, dtype=np.uint8)
                isolated[mask == i] = i + 1  # add one for 0 label, all values are one higher

                if distance == 0.0:
                    erode_amount = erode
                    if erode == 0:
                        erode_amount = int(min(15.0, np.sqrt(num_elements) / 10.0))  # calculate

                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_amount, erode_amount))
                    isolated = cv2.erode(isolated, kernel)
                else:
                    isolated = cv2.distanceTransform(isolated, cv2.DIST_L2, 5)
                    ret, isolated = cv2.threshold(isolated, distance * isolated.max(), i + 1, 0)

                markers += np.uint8(isolated)

        # d.save_diagnostics_image(args, markers, image_name, "markers", verbose=True)

        if len(rgb.shape) == 2:
            base = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)
        else:
            base = rgb

        markers = cv2.watershed(base, markers)
        markers[markers<0] = 0

        # # replace border walls
        # markers[markers > max_value] = 0
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        # markers = cv2.dilate(markers, kernel)

    return markers



def kmeans_image(rgb, k=None, max_k=11, cutoff=1.01):
    """Runs k-means on rgb and returns a result image, labels and the centers"""
    img = rgb.copy()
    dim = 1
    if img.ndim > 2:
        dim = img.shape[2]
    z = img.reshape((-1, dim))

    # convert to np.float32
    z = np.float32(z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    if k is None:
        last_deviation = 0
        for test_k in range(3, max_k):
            
            compactness, labels, centers = cv2.kmeans(z, test_k, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
            values = centers[labels.flatten()]
            deviation = np.mean(np.std(values, axis=0))
            
            if deviation < last_deviation * 1.01:
                break

            k = test_k
            last_deviation = deviation

    compactness, labels, centers = cv2.kmeans(z, k, None, criteria, 9, cv2.KMEANS_PP_CENTERS)
    values = centers[labels.flatten()]
    
    # Now convert back into uint8, and make original image
    image = values.reshape((img.shape)).astype("uint8")

    return image, labels, centers



def isolate_color(rgbImg, rgb_color):
    """
    Creates a mask based on rgbImg where its color is rgb_color and
    returns it
    """
    mask = np.zeros((rgbImg.shape[0], rgbImg.shape[1]), dtype=np.uint8)

    mask[np.where((rgbImg == rgb_color).all(axis = 2))] = 255

    return mask 


def adjust_contrast(img, alpha=1.5, beta=None):
    """Adjusts the contrast on img and returns the result"""
    new_img = img.copy()
                     
    # multiply every pixel value by alpha
    cv2.multiply(new_img, alpha, new_img)

    # add a beta value to every pixel 
    if not beta is None:
        cv2.add(new_img, beta, new_img)

    return new_img

def generate_shadows_and_highlights(args, rgb):
    """Generates shadows and highlights on rgb and returns the result"""
    base = cv2.pyrMeanShiftFiltering(rgb, 9, 9)
    base = cv2.cvtColor(base, cv2.COLOR_RGB2GRAY)
    base = cv2.GaussianBlur(base, (5, 5), 0)
    base = cv2.bilateralFilter(base, 31, 31, 61)
    # base = cv2.bilateralFilter(base, 27, 31, 61)
    # base = cv2.pyrMeanShiftFiltering(base, 31, 21)

    # base = cv2.cvtColor(base, cv2.COLOR_RGB2GRAY)

    # base[np.where(np.logical_and(base > 50, base < 140))] = 127

    # base = cv2.equalizeHist(base)
    # clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(8, 8))
    # base = clahe.apply(base)

    # base = cv2.GaussianBlur(base, (5, 5), 0)

    return base

def find_lines(args, rgb, image_name):
    """Finds lines in an image and saves the image to image_name."""
    grey = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    lsd = cv2.createLineSegmentDetector()
    lines = lsd.detect(grey)[0]
    drawn_img = lsd.drawSegments(rgb,lines)
    # d.save_diagnostics_image(args, drawn_img, image_name, "lines", verbose=True)

def get_grayscale_image(img):
    """Returns the input image as grayscale."""
    if len(img.shape) < 3:
        return img
    else:
        if img.shape[2] == 3:
            img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        else:
            img_gray = cv2.cvtColor(img,cv2.COLOR_BGRA2GRAY)

    return img_gray

def find_defects(img, debug_image):
    """Finds defects in img, draws them to debug_image and returns it"""
    img_gray = get_grayscale_image(img)    

    ret, thresh = cv2.threshold(img_gray, 127, 255, 0)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]

    hull = cv2.convexHull(cnt, returnPoints = False)
    defects = cv2.convexityDefects(cnt, hull)

    if defects is not None:
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(cnt[s][0])
            end = tuple(cnt[e][0])
            far = tuple(cnt[f][0])
            cv2.line(debug_image, start, end, [0, 255, 0], 2)
            cv2.circle(debug_image, far, 5, [0, 0, 255], -1)

    return debug_image

def generate_gabor_filters(radians_inc=np.pi/16, ksize=31, sigma=4.0):
    """Returns gabor filters with given parameters"""
    filters = []
    ksize = 31
    for theta in np.arange(0, np.pi, radians_inc):
        kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        kern /= 1.5 * kern.sum()
        filters.append(kern)
    return filters

def filter_gabor(img, radians_inc=np.pi/16, ksize=31, sigma=4.0):
    """Filters img using gabor filters with given parameters"""
    filters = generate_gabor_filters(radians_inc=radians_inc, ksize=ksize, sigma=sigma)

    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        np.maximum(accum, fimg, accum)
    return accum

def remove_grooves(image, mask=None):
    inpaint_mask = filters.sobel(image)

    inpaint_mask[inpaint_mask < 0.05] = 0.
    inpaint_mask[inpaint_mask > 0.] = 255.

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    inpaint_mask = cv2.dilate(inpaint_mask, kernel, iterations = 1)

    inpaint_mask = inpaint_mask.astype('uint8')
    if not mask is None:
        inpaint_mask &= mask.astype('uint8')

    image = cv2.inpaint(image, inpaint_mask,3,cv2.INPAINT_TELEA) # or INPAINT_TELEA

    return image

def alpha_blend(img1, img2, mask):
    """ alphaBlend img1 and img 2 (of CV_8UC3) with mask (CV_8UC1 or CV_8UC3)
    """
    if img1.ndim != img2.ndim == 2:
        raise ValueError('img1 and img2 must have same dimensions') 

    if (mask.ndim == 3 and mask.shape[-1] == 3) or img1.ndim == 2:
        alpha = mask/255.0
    else:
        alpha = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)/255.0
    blended = cv2.convertScaleAbs(img1*(1-alpha) + img2*alpha)
    return blended

