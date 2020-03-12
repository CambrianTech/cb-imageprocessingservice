import numpy as np
import cv2
import tensorflow as tf


def get_session_config(use_gpu=True, dynamic_gpu_memory=True) -> tf.ConfigProto:
    # Allow GPU memory growth so tensorflow doesn't allocate all memory
    config = tf.ConfigProto(device_count={ "GPU": 1 if use_gpu else 0 })
    config.gpu_options.allow_growth = dynamic_gpu_memory
    return config


def feed_images_batched(model, images_batch: list) -> list:
    inputs = {}

    for images in images_batch:
        # Resize and normalize the images
        for key, feed_tensor in model.feed_tensors.items():
            assert key in images, "Required input image %s not found in passed images" % key

            img = images[key]

            shape = feed_tensor.shape

            # Resize to target size. This needs to be
            # done before potentially expanding the
            # channels dimension as it removes it again.
            img = cv2.resize(img, tuple(shape[1:3]))

            # Make sure we have the channels dimension
            # for 1-channel images.
            if len(img.shape) == 2:
                img = np.expand_dims(img, -1)

            assert img.shape == shape[1:], "Shape not equal to target shape, shape: %s target: %s" % (
                img.shape, shape[1:])

            input_image = img.astype(np.float32) / 255.0

            if not key in inputs:
                inputs[key] = []

            inputs[key].append(input_image)

    inference = model(inputs)

    # Process all outputs
    outputs = []

    for _ in range(len(images_batch)):
        output_dict = {}
        for key in model.fetch_tensors.keys():
            output = inference[key][0] * 255.0

            # Remove single channel dimension if any
            if len(output.shape) == 3 and output.shape[-1] == 1:
                output = np.squeeze(output, axis=-1)

            output_dict[key] = output
        outputs.append(output_dict)

    return outputs

def feed_image_batched(model, image_batch: np.ndarray) -> np.ndarray:
    assert len(model.feed_tensors) == 1 and len(
        model.fetch_tensors) == 1, "Tried to use feed_image with more than one input or output in the model"

    input_key = next(iter(model.feed_tensors.keys()))
    output_key = next(iter(model.fetch_tensors.keys()))

    outputs = feed_images_batched(model, [{input_key: image} for image in image_batch])

    return [output[output_key] for output in outputs]

def feed_images(model, images: dict) -> dict:
    # Send in all inputs
    inputs = {}

    # Resize and normalize the images
    for key, feed_tensor in model.feed_tensors.items():
        assert key in images, "Required input image %s not found in passed images" % key

        img = images[key]

        shape = feed_tensor.shape

        # Resize to target size. This needs to be
        # done before potentially expanding the
        # channels dimension as it removes it again.
        img = cv2.resize(img, tuple(shape[1:3]))

        # Make sure we have the channels dimension
        # for 1-channel images.
        if len(img.shape) == 2:
            img = np.expand_dims(img, -1)

        assert img.shape == shape[1:], "Shape not equal to target shape, shape: %s target: %s" % (
            img.shape, shape[1:])

        input_image = img.astype(np.float32) / 255.0

        inputs[key] = [input_image]

    inference = model(inputs)

    # Process all outputs
    outputs = {}
    for key in model.fetch_tensors.keys():
        output = inference[key][0] * 255.0

        # Remove single channel dimension if any
        if len(output.shape) == 3 and output.shape[-1] == 1:
            output = np.squeeze(output, axis=-1)

        outputs[key] = output

    return outputs


def feed_image(model, image: np.ndarray) -> np.ndarray:
    assert len(model.feed_tensors) == 1 and len(
        model.fetch_tensors) == 1, "Tried to use feed_image with more than one input or output in the model"

    input_key = next(iter(model.feed_tensors.keys()))
    output_key = next(iter(model.fetch_tensors.keys()))

    outputs = feed_images(model, {input_key: image})

    return outputs[output_key]


def load_model(model_path: str, session_config=None):
    if session_config is None:
        session_config = get_session_config()

    print("Loading model from", model_path)
    model = tf.contrib.predictor.from_saved_model(model_path, config=session_config)
    input_keys = ", ".join(model.feed_tensors.keys())
    output_keys = ", ".join(model.fetch_tensors.keys())
    print("Loaded model with inputs", input_keys, "and outputs", output_keys)
    return model


def load_models(model_paths: dict) -> dict:
    return {model_name: load_model(model_path) for model_name, model_path in model_paths}
