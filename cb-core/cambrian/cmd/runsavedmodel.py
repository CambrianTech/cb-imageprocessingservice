import os
import argparse

import glob
import cv2

import tensorflow as tf
import numpy as np

from tqdm import tqdm

from cambrian.nn import get_input_name, get_output_name

def load_image(path, crop_size):
    img = cv2.imread(path, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (crop_size, crop_size))
    img = img.astype(np.float32) / 255.0
    return img

def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def process_images(args, predict_fn):
    src_paths = [sorted(glob.glob(path)) for path in args.input_paths.split(",")]

    path_count = len(src_paths[0])

    assert all([len(paths) == path_count for paths in src_paths])

    os.makedirs(args.output_path, exist_ok=True)

    for paths in tqdm(zip(*src_paths), total=path_count):
        name = os.path.basename(paths[0])
        image_name = os.path.splitext(name)[0]
        result_path = os.path.join(args.output_path, image_name + args.suffix + ".png")

        if not os.path.isfile(result_path):
            inputs = {args.input_keys.split(",")[i]: [load_image(path, args.crop_size)] for i, path in enumerate(paths)}
            output = predict_fn(inputs)

            if args.softmax:
                output = softmax(output)

            output = output * 255

            # Append an empty channel if we have 2 channels as we can only save 1/3/4 channels
            if len(output.shape) == 3 and output.shape[-1] == 2:
                output = np.concatenate((output, np.zeros((output.shape[0], output.shape[1], 1), dtype=output.dtype)), axis=-1)

            # Convert color-space if the output is not grayscale
            if len(output.shape) == 3 and output.shape[-1] > 1:
                output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

            cv2.imwrite(result_path, output)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str, help="Model to use")
    parser.add_argument("-s", "--suffix", type=str, default="", help="Output suffix")
    parser.add_argument("-i", "--input_paths", required=True, type=str, help="Input directories seperated by comma. Can contain wildcards (passed directly to glob).")
    parser.add_argument("-ik", "--input_keys", type=str, help="Model input dictionary keys seperated by comma. Uses cambrian.nn.get_input_name(i) if None.")
    parser.add_argument("-ok", "--output_key", type=str, help="Model output dictionary key. Uses cambrian.nn.get_output_name(0) if None.")
    parser.add_argument("-o", "--output_path", required=True, type=str, help="Output directory")
    parser.add_argument("-c", "--crop_size", type=int, default=512, help="Crop width and height")
    parser.add_argument("-sm", "--softmax", action="store_true")
    
    args = parser.parse_args()
    
    print(args)

    if args.input_keys is None:
        args.input_keys = ",".join(map(get_input_name, range(len(args.input_paths.split(",")))))

    if args.output_key is None:
        args.output_key = get_output_name(0)

    print("Loading model from", args.model)

    # Load the model from a saved model directory.
    # The resulting predict_fn takes as argument a dictionary
    predictor = tf.contrib.predictor.from_saved_model(args.model)
    predict_fn = lambda inputs: predictor(inputs)[args.output_key][0]

    process_images(args, predict_fn)
    
    print("Done.")


if __name__ == "__main__":
    main()
