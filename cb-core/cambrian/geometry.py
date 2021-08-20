
import numpy as np
import math
from scipy.linalg import expm, norm

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

    >>> angle_between((1, 0, 0), (0, 1, 0))
    1.5707963267948966
    >>> angle_between((1, 0, 0), (1, 0, 0))
    0.0
    >>> angle_between((1, 0, 0), (-1, 0, 0))
    3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


#https://stackoverflow.com/questions/6802577/rotation-of-3d-vector
def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians::

    >>> v, axis, theta = [3,5,0], [4,4,1], 1.2
    >>> M0 = rotation_matrix(axis, theta)
    >>> print(dot(M0,v))
    [ 2.74911638  4.77180932  1.91629719]
    """
    return expm(np.cross(np.eye(3), axis/norm(axis)*theta))

def radians_to_degrees(radians):
    return 180.0 * radians / math.pi

def degrees_to_radians(degrees):
    return (degrees * math.pi) / 180.0



