
from math import sqrt
import numpy as np
from skimage import img_as_float
from scipy.ndimage import convolve

def frei_chen(image,mask=None):
    """Find the edge magnitude using FreiChen Filter.

    Parameters
----------
    image : 2-D array
        Image to process.
    mask : 2-D array, optional
        An optional mask to limit the application to a certain area.
        Note that pixels surrounding masked regions are also masked to
        prevent masked regions from affecting the result.

    Returns
    -------
    output : ndarray
        The FreiChen edge .
    """
    return np.sqrt(freiChen_edge1(image, mask)**2 +\
                   freiChen_edge2(image, mask)**2 +\
          freiChen_edge3(image, mask)**2 +\
                   freiChen_edge4(image, mask)**2 +\
                   freiChen_line1(image, mask)**2 +\
                   freiChen_line2(image, mask)**2 +\
                   freiChen_line3(image, mask)**2 +\
                   freiChen_line4(image, mask)**2 )
                   #freiChen_average(image, mask)**2)

def _mask_filter_result(result, mask):
    """Return result after masking.

    Input masks are eroded so that mask areas in the original image don't
    affect values in the result.
    """
    if mask is None:
        result[0, :] = 0
        result[-1, :] = 0
        result[:, 0] = 0
        result[:, -1] = 0
        return result
    else:
        mask = binary_erosion(mask, EROSION_SELEM, border_value=0)
        return result * mask


def freiChen_edge1(image, mask):
    """Find the horizontal edges of an image using the FreiChen operator.

    The kernel is applied to the input image, to produce separate 
measurements
    of the gradient component one orientation.

    Parameters
    ----------
    image : 2-D array
        Image to process.
    mask : 2-D array, optional
        An optional mask to limit the application to a certain area.
        Note that pixels surrounding masked regions are also masked to
        prevent masked regions from affecting the result.

    Returns
    -------
    output : ndarray
        The FreiChen horizontal edge.

    Notes
    -----
    We use the following kernel and return the absolute value of the
    result at each point::

      1  sqrt(2)  1
      0       0   0
 -1 -sqrt(2) -1

    """
    image = img_as_float(image)
    result = np.abs(convolve(image,
                             np.array([[ 1, sqrt(2), 1],
                                       [ 0,       0, 0],
                                       [-1,-sqrt(2),-1]]).\
astype(float) / sqrt(8)))
    return _mask_filter_result(result, mask)


def freiChen_edge2(image, mask):
    """Find the vertical edges of an image using the FreiChen operator.

    The kernel is applied to the input image, to produce separate 
measurements
    of the gradient component one orientation.

    Parameters
    ----------
    image : 2-D array
        Image to process.
    mask : 2-D array, optional
        An optional mask to limit the application to a certain area.
        Note that pixels surrounding masked regions are also masked to
        prevent masked regions from affecting the result.

    Returns
    -------
    output : ndarray
        The FreiChan edges .

    Notes
    -----
    We use the following kernel and return the absolute value of the
    result at each point::

         1   0       -1
    sqrt(2)  0  -sqrt(2)
 1 0 -1
    """
    image = img_as_float(image)
    result = np.abs(convolve(image,
                             np.array([[      1, 0,      -1 ],
                                       [sqrt(2), 0, -sqrt(2)],
                                       [      1, 0,      -1]]).\
astype(float) / sqrt(8)))
    return _mask_filter_result(result, mask)

def freiChen_edge3(image, mask):
    """Find the edges in an image having slope 135 degree FreiChen operator.

    The kernel is applied to the input image, to produce separate 
measurements
    of the gradient component one orientation.

    Parameters
    ----------
    image : 2-D array
        Image to process.
    mask : 2-D array, optional
        An optional mask to limit the application to a certain area.
        Note that pixels surrounding masked regions are also masked to
        prevent masked regions from affecting the result.

    Returns
    -------
    output : ndarray
        The FreiChen edges.

    Notes
    -----
    We use the following kernel and return the absolute value of the
    result at each point::

          0   -1   sqrt(2)
          1    0       -1
-sqrt(2)   1        0

    """
    image = img_as_float(image)
    result = np.abs(convolve(image,
                             np.array([[       0,-1, sqrt(2)],
                                       [       1, 0,      -1],
                                       [-sqrt(2), 1,       0]]).\
astype(float) / sqrt(8)))
    return _mask_filter_result(result, mask)


def freiChen_edge4(image, mask):
    """Find the edges of an image having slope 45 degree using the FreiChen 
operator.

    The kernel is applied to the input image, to produce separate 
measurements
    of the gradient component one orientation.

    Parameters
    ----------
    image : 2-D array
        Image to process.
    mask : 2-D array, optional
        An optional mask to limit the application to a certain area.
        Note that pixels surrounding masked regions are also masked to
        prevent masked regions from affecting the result.

    Returns
    -------
    output : ndarray
        The Freichen edges.

    Notes
    -----
    We use the following kernel and return the absolute value of the
    result at each point::

      sqrt(2)   -1         0
          -1     0         1
   0     1   -sqrt(2)

    """
    image = img_as_float(image)
    result = np.abs(convolve(image,
                             np.array([[sqrt(2), -1,        0],
                                       [     -1,  0,        1],
                                       [      0,  1, -sqrt(2)]]).\
astype(float) / sqrt(8)))
    return _mask_filter_result(result, mask)


def freiChen_line1(image, mask):
    """Find the edge intersection in an image using the FreiChen operator.

    The kernel is applied to the input image, to produce separate 
measurements
    of the gradient component one orientation.

    Parameters
    ----------
    image : 2-D array
        Image to process.
    mask : 2-D array, optional
        An optional mask to limit the application to a certain area.
        Note that pixels surrounding masked regions are also masked to
        prevent masked regions from affecting the result.

    Returns
    -------
    output : ndarray
        The FreiChen lines.

    Notes
    -----
    We use the following kernel and return the absolute value of the
    result at each point::

      0   1   0
     -1   0  -1
      0   1   0

    """
    image = img_as_float(image)
    result = np.abs(convolve(image,
                             np.array([[ 0, 1,  0],
                                       [-1, 0, -1],
                                       [ 0, 1,  0]]).astype(float) / 2.0))
    return _mask_filter_result(result, mask)


def freiChen_line2(image, mask):
    """Find the intersection of edges in an image using the FreiChen 
operator.

    The kernel is applied to the input image, to produce separate 
measurements
    of the gradient component one orientation.

    Parameters
    ----------
    image : 2-D array
        Image to process.
    mask : 2-D array, optional
        An optional mask to limit the application to a certain area.
        Note that pixels surrounding masked regions are also masked to
        prevent masked regions from affecting the result.

    Returns
    -------
    output : ndarray
        The FreiChen lines.

    Notes
    -----
    We use the following kernel and return the absolute value of the
    result at each point::

      -1   0   1
       0   0   0
       1   0  -1

    """
    image = img_as_float(image)
    result = np.abs(convolve(image,
                             np.array([[-1, 0,  1 ],
                                       [ 0, 0,  0],
                                       [ 1, 0, -1]]).astype(float) / 2.0))
    return _mask_filter_result(result, mask)

def freiChen_line3(image, mask):
    """Find the lines of an image using the FreiChen operator.

    The kernel is applied to the input image, to produce separate 
measurements
    of the gradient component one orientation.

    Parameters
    ----------
    image : 2-D array
        Image to process.
    mask : 2-D array, optional
        An optional mask to limit the application to a certain area.
        Note that pixels surrounding masked regions are also masked to
        prevent masked regions from affecting the result.

    Returns
    -------
    output : ndarray
        The FreiChen lines.

    Notes
    -----
    We use the following kernel and return the absolute value of the
    result at each point::

      1   -2   1
     -2    4  -2
      1   -2   1

    """
    image = img_as_float(image)
    result = np.abs(convolve(image,
                             np.array([[ 1, -2,  1 ],
                                       [-2,  4, -2 ],
                                       [ 1, -2,  1]]).astype(float) / 6.0))
    return _mask_filter_result(result, mask)


def freiChen_line4(image, mask):
    """Find the lines of an image using the FreiChen operator.

    The kernel is applied to the input image, to produce separate 
measurements
    of the gradient component one orientation.

    Parameters
    ----------
    image : 2-D array
        Image to process.
    mask : 2-D array, optional
        An optional mask to limit the application to a certain area.
        Note that pixels surrounding masked regions are also masked to
        prevent masked regions from affecting the result.

    Returns
    -------
    output : ndarray
        The FreiChen lines.

    Notes
    -----
    We use the following kernel and return the absolute value of the
    result at each point::

      -2   1   -2
       1   4    1
      -2   1   -2

    """
    image = img_as_float(image)
    result = np.abs(convolve(image,
                             np.array([[ -2, 1, -2 ],
                                       [  1, 4,  1 ],
                                       [ -2, 1, -2 ]]).astype(float) / 6.0))
    return _mask_filter_result(result, mask)


def freiChen_average(image, mask):
    image = img_as_float(image)
    result = np.abs(convolve(image,
                             np.array([[ 1, 1, 1 ],
                                       [ 1, 1, 1 ],
                                       [ 1, 1, 1 ]]).astype(float) / 3.0))
    return _mask_filter_result(result, mask)