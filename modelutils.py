import numpy as np
import cv2


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

    input_key = next(model.feed_tensors.items())
    output_key = next(model.fetch_tensors.items())

    outputs = feed_images(model, {input_key: image})

    return outputs[output_key]


def load_model(model_path: str):
    print("Loading model from", model_path)
    model = tf.contrib.predictor.from_saved_model(model_path)
    input_keys = ",".join(model.feed_tensors.keys())
    output_keys = ",".join(model.fetch_tensors.keys())
    print("Loaded model with inputs", input_keys, "and outputs", output_keys)
    return model


def load_models(model_paths: dict) -> dict:
    return {model_name: load_model(model_path) for model_name, model_path in model_paths}
