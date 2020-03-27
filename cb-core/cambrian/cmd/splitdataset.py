import os
import numpy as np
import argparse
import glob
from string import ascii_lowercase
import cv2
from shutil import copyfile, rmtree

def get_images(path, expression):
    image_paths = []

    if not path is None:
        if expression is None: 
            image_paths = glob.glob(os.path.join(path, "*.png"))
            if len(image_paths) == 0:
                image_paths = glob.glob(os.path.join(path, "*.jpg"))
        else:
            image_paths = glob.glob(os.path.join(path, expression))

    return image_paths
        
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_path', type=str, required=True, help='Path to the input images')
    parser.add_argument('-e', '--expressions', type=str, default="*.*", help='Expression to match input images')
    parser.add_argument('-o', '--output_path', type=str, default="dataset", help='Path to the output images')

    parser.add_argument('--split', type=str, default="90,10", help='Percentages to split images by')

    args = parser.parse_args()

    print(args)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    
    if not args.input_path is None:
        if not os.path.exists(args.input_path):
            raise ValueError("Error: input_path '%s' not found" % args.input_path)
    else:
        raise ValueError("Either input_path is required")

    expressions = args.expressions.split(',')

    image_expression_paths = []
    for expression in expressions:
        image_paths = get_images(args.input_path, expression)
        image_paths.sort()
        image_expression_paths.append(image_paths)

    num_images = len(image_expression_paths[0])

    if num_images == 0:
        raise ValueError("No images found at input_path %s" % num_images)

    indices = np.arange(num_images)
    np.random.shuffle(indices)

    percentages = args.split.split(',')

    counts = []
    subpaths = []
    for i, percent in enumerate(percentages):
        subpaths.append("images_" + ascii_lowercase[i] + "_" + str(percent))
        counts.append(int(num_images * float(percent) / 100.0))

    print("Found", num_images, "images, separating into percentages of", args.split)

    start = 0
    processed = 0
    path_index = 0
    pct_completed = 0

    for path_index in range(0,len(counts)):
        count = counts[path_index]
        subpath = subpaths[path_index]
        start = processed
        end = min(start + count, num_images)
 
        output_path = os.path.join(args.output_path, subpath)

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        for i in range(start, end):
            index = indices[i]

            for image_paths in image_expression_paths:
                src_path = image_paths[index]
                filename = os.path.basename(image_paths[index])
                dest_path = os.path.join(output_path, filename)
                copyfile(src_path, dest_path)
                print("(%.1f%% complete) %d Copied %s " % (pct_completed, index, filename))

            processed = processed + 1
            pct_completed = 100.0 * (float(processed)/ float(num_images))
            
        # print(processed, num_images)

    print("Done.")

if __name__ == "__main__":
    main()
