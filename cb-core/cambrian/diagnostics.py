from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt

import cv2
from skimage import color
try:
    from imageio import imsave
except:
    from scipy.misc import imsave

try:
    from skimage.transform import resize as imresize
except:
    from scipy.misc import imresize

import os
import numpy as np

HUE_RED=0.0
HUE_GREEN=0.32
HUE_YELLOW=0.17

def save_diagnostics_image(args, img, image_name, result_name, verbose=False):
    if args is None or not args.debug or (verbose and not args.verbose):
        return

    filename = image_name + "_" + result_name + ".png"

    if not os.path.isdir(args.logging_path):
        os.mkdir(args.logging_path)

    path = os.path.join(args.logging_path, filename)

    print("Saving diagnostic image %s to %s" % (result_name, path))
    imsave(path, img)

def save_diagnostics_mask(args, mask, image_name, result_name, color_image=None, hue=HUE_YELLOW, verbose=False):
    if args is None or not args.debug:
        return

    if color_image is not None:
        out = overlay_mask(args, mask, color_image, hue)
    else: 
        out = np.zeros(mask.shape, dtype=np.uint8)
        out[mask>0] = 255

    save_diagnostics_image(args, out, image_name, result_name, verbose=verbose)

def overlay_mask(args, mask, color_image, hue):
    darkest_value = 0.3
    img_hsv = color.rgb2hsv(color_image)

    grey = img_hsv[:, :, 2]
    grey[grey<darkest_value] = darkest_value

    img_hsv[:, :, 0][mask>0] = hue
    img_hsv[:, :, 1][mask>0] = 1
    img_hsv[:, :, 2][mask>0] = grey[mask>0]
    out = color.hsv2rgb(img_hsv)

    return out

def label_mask(debug_image, mask, name):
    overlay = imresize(mask, debug_image.shape[:2])

    overlay[overlay > 0] = 255

    moments = cv2.moments(overlay)
    centroid = int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00'])

    # Point p1(m.m10/m.m00, m.m01/m.m00);

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    markers = cv2.erode(overlay, kernel, iterations=3)
    overlay = overlay - markers

    # markers = cv2.erode(markers, kernel, iterations=3)
    #debug_image = ip.overlay_mask(args, markers, debug_image, ip.HUE_YELLOW)
    debug_image[overlay == 255] = [255, 255, 255]

    alpha = 0.3
    overlay = cv2.cvtColor(markers, cv2.COLOR_GRAY2RGB)
    output = debug_image.copy()
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

    debug_image[overlay == 255] = output[overlay == 255]
    #debug_image += overlay

    cv2.circle(debug_image, centroid, 5, (0, 0, 0))
    
    cv2.putText(debug_image, name, centroid, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, (255, 255, 0), 2, cv2.LINE_AA)


    return debug_image

def figure_to_image(fig):
    fig.canvas.draw()

    # convert canvas to image
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return img

def to_plot_color(rgb_color):
    return (rgb_color[0] / 255.0, rgb_color[1] / 255.0, rgb_color[2] / 255.0, 1.0)

# matches https://academo.org/demos/3d-vector-plotter/
# colors: https://matplotlib.org/examples/color/named_colors.html
def plot_vectors(p0, p1, p2, colors=['b', 'r', 'orange']):
    plt.switch_backend('agg') #no display!

    origin = [0, 0, 0]
    X, Y, Z = zip(origin, origin, origin) 
    U, V, W = zip(p0, p1, p2)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.quiver(X, Y, Z, U, V, W, arrow_length_ratio=0.01, color=colors)

    ax.view_init(elev=10.0, azim=270)

    return figure_to_image(fig)

