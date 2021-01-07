import math
import os
import json
import cv2
import numpy as np
from pathlib import Path

import click
import time

import mxnet as mx
from mxnet.gluon.data.vision import transforms
import gluoncv
from gluoncv.utils.viz import get_color_pallete
import matplotlib.image as mpimg

import cambrian.image_processing as ip

import io
from cambrian.frei_chen import frei_chen
from cambrian.Line import Line
from cambrian.VanishingPointFinder import VanishingPointFinder

from modelutils import feed_image_batched, feed_images_batched, load_model

import random

#SEG_RES = None
#ADE_MODEL = 'deeplab_resnet101_ade'

ADE_MODEL = 'deeplab_resnest269_ade'
SEG_RES = 480

SAVE_DEBUG_IMAGES = True 

def segment_image(ctx, model, img):
    from gluoncv.data.transforms.presets.segmentation import test_transform
    
    #img = test_transform(img, ctx)
    if SEG_RES is None:
        img = test_transform(img, ctx)
    else:
        img = test_transform(mx.img.resize_short(img, SEG_RES),ctx)
    output = model.predict(img)
    predict = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy()

    return predict

def find_lines(img, output_path, segmented, normals):

    height, width = img.shape[:2]
    diagonal = np.hypot(width, height)
    print("image w,h", width, height)

    bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gabor_scale = 1500.0 / diagonal
    bw_res = cv2.resize(bw, (int(width * gabor_scale), int(height * gabor_scale)), cv2.INTER_CUBIC) if gabor_scale < 1.0 else bw

    def _gabor(theta, lambd, gamma = 0.0, psi = 0.0):
        ksize = lambd
        sigma = ksize * lambd
        result = cv2.filter2D(bw_res, cv2.CV_8UC1, cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F))
        return result

    v_gabor = _gabor(0, 7)
    h_gabor = _gabor(np.pi/2.0, 9)

    contours_src = cv2.addWeighted(v_gabor, 1.0, h_gabor, 1.0, 0)
    contours_src = cv2.resize(contours_src, (width, height), interpolation = cv2.INTER_CUBIC)
    
    edges = cv2.addWeighted(v_gabor, 3.0, h_gabor, 3.0, -20)
    edges = cv2.bilateralFilter(edges, 5, 5, 5)
    edges = cv2.resize(edges, (width, height), interpolation = cv2.INTER_CUBIC)

    clean_edges = frei_chen(bw)

    if SAVE_DEBUG_IMAGES:
        lines_a = img.copy()
        lines_b = img.copy()
        lines_c = img.copy()
        lines_d = img.copy()
        intersections = []
    else:
        lines_a = lines_b = lines_c = lines_d = None

    contours_src = cv2.adaptiveThreshold(contours_src, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, int(diagonal / 50) * 2 + 1, -30)
    contours_dilated = ip.rough_dilate_erode(True, contours_src, 3, scale=400/diagonal, interpolation=cv2.INTER_AREA)
    Line.prepare(img, contours_dilated)

    line_data = []

    def _add_lines(fld, image, min_confidence):
        lines = fld.detect(image)
        if lines is None: return

        for line in lines: 
            new_line = Line(line[0][0], line[0][1], line[0][2], line[0][3])
            if new_line.get_confidence() > min_confidence: 
                line_data.append(new_line)

    #find all lines in the edge image
    fld = cv2.ximgproc.createFastLineDetector(int(diagonal / 30.0), 1.41, 200, 240, 3, False)
    _add_lines(fld, edges, 0.2)

    aperture = 5 if diagonal > 1200 else 3
    fld = cv2.ximgproc.createFastLineDetector(int(diagonal / 50.0), 1.41, 200, 220, aperture, False)

    _add_lines(fld, bw - (clean_edges * 5.0).astype("uint8"), 0.15)
    _add_lines(fld, segmented, 0.15)

    fld = cv2.ximgproc.createFastLineDetector(int(diagonal / 50.0), 1.41, 200, 220, aperture, False)
    #normals_lines = cv2.resize(cv2.cvtColor(normals, cv2.COLOR_BGR2GRAY), (width, height), interpolation = cv2.INTER_CUBIC)
    normals_lines = cv2.addWeighted(img, 0.5, cv2.resize(normals, (width, height)), 0.5, 0)
    cv2.imwrite(os.path.join(output_path, "normals_lines.jpg"), normals_lines)
    #_add_lines(fld, normals_lines, 0.15)
    
    if lines_a is not None: Line.draw_all(line_data, lines_a)
    
    line_data = Line.merge(line_data, diagonal / 300.0, search_length=1.1, angle_threshold=math.radians(5.0), max_color_std=7.0)

    if lines_b is not None: Line.draw_all(line_data, lines_b)

    line_data = Line.merge(line_data, diagonal / 80.0, search_length=1.0, angle_threshold=math.radians(5.0), create_pairs=True)

    if lines_c is not None: Line.draw_all(line_data, lines_c)

    line_data, intersections = Line.find_corners(line_data, search_length=1.5, angle_threshold=math.radians(10.0), parallel_threshold=math.radians(4), \
            max_color_std=3.0, confidence_diff=0.4)

    line_data = Line.merge(line_data, diagonal / 300.0, search_length=1.07, angle_threshold=math.radians(5.0))

    #if lines_d is not None: Line.draw_all(line_data, lines_d)

    #vanishing points:
    vpf = VanishingPointFinder(line_data)
    vanishing_points = vpf.compute()
    
    if len(vanishing_points) > 0:
        for vp in vanishing_points:
            color = np.random.randint(0, 255, size=(3, ))
            color = ( int (color [ 0 ]), int (color [ 1 ]), int (color [ 2 ]))
            inliers = np.array(line_data)[vp.votes > 0]
            for line in inliers:
                line.draw(lines_d, color=color, thickness=3)

    if lines_d is not None:
        for intersection in intersections: cv2.circle(lines_c, (int(intersection[0]), int(intersection[1])), max(int(diagonal/100), 3), (255,0,0), 2)
        
        cont_img = np.vstack((np.hstack((lines_a, lines_b)), np.hstack((lines_c, lines_d))))
        cv2.imwrite(os.path.join(output_path, "cont.jpg"), cont_img)
        

def get_file_paths(input_dir, pattern):
    files = []
    if pattern:
        files.extend(Path(input_dir).glob('**/' + pattern))
    else: 
        extensions = ('.png', '.jpg', '.jpeg')
        for ext in extensions:
            files.extend(Path(input_dir).glob('**/*' + ext))
    return files

def parse_data(input_dir, output_dir, model_normals):

    files = get_file_paths(input_dir, "**/data_v2.json")
    index = 0

    ctx = mx.cpu(0) if mx.context.num_gpus() == 0 else mx.gpu(0)
    model = gluoncv.model_zoo.get_model(ADE_MODEL, pretrained=True)

    for path in files:
        data_path = str(path)
        dir_path = os.path.dirname(data_path)

        print("Parsing", path)
        with open(path) as f:
            data = json.load(f)
        
        image_path = os.path.join(dir_path, "background")

        if not os.path.exists(image_path):
            print("No image at path", image_path)
            continue

        surfaces = data['geometry']['surfaces']
        masks = []

        if len(surfaces) < 3:
            continue

        img = cv2.imread(image_path)

        hfov = data["camera"]["fov"]
        aspect = float(img.shape[1]) / float(img.shape[0])
        
        parts = dir_path.split(input_dir)
        if len(parts) > 1:
            file_name = parts[1][1:]
            output_path = os.path.join(output_dir, file_name)
        else:
            file_name = "image_" + str(index)
            output_path = output_dir

        surface_invalid = False

        #make output dir
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        #run segmentation:
        seg_path = os.path.join(output_path, "mask.png")

        segmented = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
        if segmented is None:
            print("Segmenting %s" % image_path)
            start = time.process_time()
            segmented = segment_image(ctx, model, mx.image.imread(image_path)).astype("uint8")
            end = time.process_time()
            print("segmentation took %.2f seconds" % (end-start))
            cv2.imwrite(seg_path, segmented)

            if SAVE_DEBUG_IMAGES:
                vis_segmented = np.array(get_color_pallete(segmented, 'ade20k').convert('RGB'))
                vis_segmented = cv2.resize(vis_segmented, (img.shape[1], img.shape[0]))
                vis_segmented = cv2.addWeighted(img,0.5,vis_segmented,0.5,0)
                cv2.imwrite(os.path.join(output_path, "segmented.png"), vis_segmented)

        normals_path = os.path.join(output_path, "normals.png")

        normals = cv2.imread(normals_path)
        if normals is None:
            normals = feed_image_batched(model_normals, [cv2.resize(img, (512, 512))])[0].astype("uint8")
            cv2.imwrite(os.path.join(output_path, "normals.png"), normals)

        #process each surface:
        for surface in surfaces:
            mask_url = surface['images']['mask']
            mask_path = os.path.join(dir_path, mask_url[mask_url.index('plane_masks'):])
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if mask is None:
                surface_invalid = True
                continue

            surface['mask'] = mask

        if surface_invalid:
            print("Invalid surface")
            continue

        find_lines(img, output_path, segmented, normals)

        index += 1
    return index

@click.command()
@click.argument("input_dir", default='input', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument("output_dir", default='output', type=click.Path(exists=False, file_okay=False, dir_okay=True))
def main(input_dir, output_dir):

    if not os.path.exists(input_dir):
        raise Exception('The directory does not exist at path {}'.format(input_dir)) 

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    start = time.time()

    normals_path = "models/normals"
    model_normals = load_model(normals_path)

    num_files = parse_data(input_dir, output_dir, model_normals)
    
    elapsed = (time.time() - start)
    print("Processing %d images took %.2f seconds (%.2fs per image)" % (num_files, elapsed, elapsed/num_files))

if __name__ == "__main__":
    main()
