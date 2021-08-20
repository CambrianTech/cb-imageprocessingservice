import tensorflow as tf
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str, help="Path to the saved model")
    args = parser.parse_args()

    predictor = tf.contrib.predictor.from_saved_model(args.input_path)
    print("Input tensors:", predictor.feed_tensors)
    print("Output tensors:", predictor.fetch_tensors)