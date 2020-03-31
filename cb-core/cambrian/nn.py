import tensorflow as tf
import numpy as np
from abc import ABC, abstractmethod
from . import utils

def get_input_name(index):
    """Returns the input name for a given input index"""
    return "input_%d" % index

def get_output_name(index):
    """Returns the output name for a given output index"""
    return "output_%d" % index

def get_input_names(count):
    """Returns a list of input names from 0 to count-1"""
    return [get_input_name(i) for i in range(count)]

def get_output_names(count):
    """Returns a list of output names from 0 to count-1"""
    return [get_output_name(i) for i in range(count)]

def load_prediction_fn(saved_model_path, num_inputs=1, batched=False):
    """
    Loads a saved model from a path and returns
    a function that can process images.

    saved_model_path: Path to the saved model
    num_inputs: How many different inputs there are
    batched: Whether we want to input batches of images

    Possible signatures:
    fn(image) when called with num_inputs=1, batched=False
    fn(image_batch) when called with num_inputs=1, batched=True
    fn([image_a, image_b, ...]) when called with num_inputs>1, batched=False
    fn([image_batch_a, image_batch_b, ...]) when called with num_inputs>1, batched=True
    """
    assert num_inputs > 0

    predictor = tf.contrib.predictor.from_saved_model(saved_model_path)

    def predict(inputs):
        assert inputs is not None

        if num_inputs > 1:
            assert len(inputs) == num_inputs

        if batched:
            assert len(inputs) >= 0

        if num_inputs > 1:
            inputs_dict = {
                get_input_name(i): [inp] if batched else inp
                for i, inp in enumerate(inputs)
            }
        else:
            inputs_dict = {
                get_input_name(0): [inputs] if batched else inputs
            }
        
        predictions = predictor(inputs_dict)

        if batched:
            return predictions
        else:
            return predictions[0]

    return predict

class IOSpecification:
    def __init__(self, index, start_channel, match_path, channels, scale_size, crop_size, dtype=tf.float32):
        self.index = index
        self.start_channel = start_channel
        self.match_path = match_path
        self.channels = channels
        self.scale_size = scale_size
        self.crop_size = crop_size
        self.dtype=dtype

    def __repr__(self):
        return "IOSpecification index: %d, start_channel: %d, path: %s, channels: %d, scale_size: %d, crop_size: %d, dtype: %s" % (
            self.index, self.start_channel, self.match_path, self.channels, self.scale_size, self.crop_size, self.dtype
        )

    __str__ = __repr__

class InputFnArgs:
    def __init__(self, batch_size, epochs, shuffle, augment, random_flip):
        self.batch_size = batch_size
        self.epochs = epochs
        self.shuffle = shuffle
        self.augment = augment
        self.random_flip = random_flip

    @staticmethod
    def train(batch_size, epochs, random_flip=False):
        return InputFnArgs(batch_size, epochs, shuffle=True, augment=True, random_flip=random_flip)

    @staticmethod
    def eval(batch_size, epochs, random_flip=False):
        return InputFnArgs(batch_size, epochs, shuffle=False, augment=False, random_flip=random_flip)

def augment_image(image):
    original_shape = image.shape

    ops = [
        lambda im: tf.image.random_brightness(im, 0.2),
        lambda im: tf.image.random_contrast(im, 0.7, 1.2),
        lambda im: tf.image.random_hue(im, 0.05),
        lambda im: tf.image.random_saturation(im, 0.7, 1.2),
        lambda im: im + tf.random.normal(im.shape, stddev=0.02)
    ]

    np.random.shuffle(ops)

    for augment_op in ops:
        image = tf.clip_by_value(augment_op(image), 0, 1)

    image = tf.image.random_jpeg_quality(image, 60, 100)
    image.set_shape(original_shape)

    return image

def get_parse_image_ab_fn(input_specs, output_specs, augment=True, random_flip=False):
    def parse_image_ab(*file_names):
        num_inputs = len(input_specs)
        num_outputs = len(output_specs)
        assert len(file_names) == num_inputs + num_outputs

        def _parse(file_name, spec):
            image_data = tf.read_file(file_name)
            image = tf.image.convert_image_dtype(tf.image.decode_png(image_data, channels=spec.channels), spec.dtype)
            image = tf.image.resize_images(image, [spec.scale_size, spec.scale_size], method=tf.image.ResizeMethod.AREA) #reduces artifacts, consider as part of specs
            return image        
            
        specs = input_specs + output_specs

        images = [_parse(file_name, spec) for file_name, spec in zip(file_names, specs)]

        # Randomly flip horizontally. Can currently only be enabled when
        # augmentation is enabled. Perhaps combine this into a class that has
        # more information about what kind of augmentation should be used.
        if augment and random_flip:
            flip_seed = np.random.randint(10000000)
            images = [tf.image.random_flip_left_right(image, seed=flip_seed) for image in images]

        # Augment the input images if enabled
        input_images = images[:num_inputs]
        if augment:
            input_images = [augment_image(image) for image in input_images]

        output_images = images[num_inputs:]

        input_dict = {get_input_name(i): img for i, img in enumerate(input_images)}
        output_dict = {get_output_name(i): img for i, img in enumerate(output_images)}

        return input_dict, output_dict
    return parse_image_ab

def get_input_fn_ab(a_specs, b_specs, input_fn_args, parse_image_fn=None):
    def input_fn():
        nonlocal parse_image_fn

        if parse_image_fn is None:
            parse_image_fn = get_parse_image_ab_fn(a_specs, b_specs, augment=input_fn_args.augment, random_flip=input_fn_args.random_flip)

        all_match_paths = [spec.match_path for spec in a_specs + b_specs]

        file_seed = np.random.randint(10000000)
        all_files = tuple(tf.data.Dataset.list_files(match_path, seed=file_seed)
                            for match_path in all_match_paths)

        dataset = tf.data.Dataset.zip(all_files)

        if input_fn_args.shuffle:
            dataset = dataset.shuffle(buffer_size=100000)

        dataset = dataset.map(parse_image_fn, num_parallel_calls=4)
        dataset = dataset.repeat(input_fn_args.epochs)
        dataset = dataset.batch(input_fn_args.batch_size)
        dataset = dataset.prefetch(2)

        return dataset

    return input_fn

class ModelBase(ABC):
    def __init__(self):
        self._inputs = None
        self._targets = None
        self._outputs = None
        self._train_op = None
        self._loss = None
        self._metrics = {}
        self._summary_op = None

    @property
    def inputs(self):
        return self._inputs

    @property
    def targets(self):
        return self._targets

    @property
    def outputs(self):
        return self._outputs

    @property
    def loss(self):
        return self._loss

    @property
    def train_op(self):
        return self._train_op

    @property
    def metrics(self):
        return self._metrics

    @property
    def summary_op(self):
        return self._summary_op

    def set_inputs(self, inputs):
        self._inputs = inputs

    def set_targets(self, targets):
        self._targets = targets

def get_model_fn_ab(model_class, a_specs, b_specs, **model_kw_args):
    def model_fn(features, labels, mode, params, config):
        assert len(features) == len(a_specs)
        assert labels is None or len(labels) == len(b_specs)

        # Construct the model
        model = model_class(**model_kw_args)
        
        # Setup inputs
        inputs = [features[input_name] for input_name in get_input_names(len(features))]
        inputs = tf.concat(inputs, axis=-1)
        model.set_inputs(inputs)

        # Setup targets if any
        if labels is not None:
            targets = [labels[output_name] for output_name in get_output_names(len(labels))]
            targets = tf.concat(targets, axis=-1)
            model.set_targets(targets)

        # Extract outputs from model
        outputs = []
        for spec in b_specs:
            outputs.append(model.outputs[:, :, :, spec.start_channel:spec.start_channel+spec.channels])

        predictions = {get_output_name(i): output for i, output in enumerate(outputs)}
        export_outputs = {name: tf.estimator.export.PredictOutput(output) for name, output in predictions.items()}

        # Set a default output (needed when exporting multiple outputs)
        export_outputs[tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = export_outputs[get_output_name(0)]

        # Log metrics
        logging_hook = [tf.train.LoggingTensorHook(model.metrics, every_n_iter=100)]

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=model.loss,
            train_op=model.train_op,
            export_outputs=export_outputs,
            training_hooks=logging_hook,
            evaluation_hooks=logging_hook,
            prediction_hooks=logging_hook,
        )

    return model_fn

def get_distribution_strategy(num_gpus):
    if num_gpus <= 1:
        return None
    return tf.contrib.distribute.MirroredStrategy(num_gpus=num_gpus)

def get_serving_input_receiver_fn(input_specs):
	def serving_input_receiver_fn():
		inputs = {
            get_input_name(i): tf.placeholder(spec.dtype, (None, spec.crop_size, spec.crop_size, spec.channels))
            for i, spec in enumerate(input_specs)
        }
		return tf.estimator.export.ServingInputReceiver(inputs, inputs)
	return serving_input_receiver_fn