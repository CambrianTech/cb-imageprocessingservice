import numpy as np
from scipy.special import softmax
import cv2
import tensorflow as tf
from modelutils import feed_image_batched, feed_images_batched, load_model
from pipeline.core import PipelineStep
import os
from tensorpack import *
from tensorpack.dataflow import dataset
from tensorpack.tfutils import gradproc, optimizer
from tensorpack.tfutils.summary import add_moving_summary, add_param_summary
from tensorpack.utils.gpu import get_num_gpu
from tensorpack.utils import logger


def class_balanced_sigmoid_cross_entropy(logits, label, name='cross_entropy_loss'):
    """
    The class-balanced cross entropy loss,
    as in `Holistically-Nested Edge Detection
    <http://arxiv.org/abs/1504.06375>`_.

    Args:
        logits: of shape (b, ...).
        label: of the same shape. the ground truth in {0,1}.
    Returns:
        class-balanced cross entropy loss.
    """
    with tf.name_scope('class_balanced_sigmoid_cross_entropy'):
        y = tf.cast(label, tf.float32)

        count_neg = tf.reduce_sum(1. - y)
        count_pos = tf.reduce_sum(y)
        beta = count_neg / (count_neg + count_pos)

        pos_weight = beta / (1 - beta)
        cost = tf.nn.weighted_cross_entropy_with_logits(logits=logits, targets=y, pos_weight=pos_weight)
        cost = tf.reduce_mean(cost * (1 - beta))
        zero = tf.equal(count_pos, 0.0)
    return tf.where(zero, 0.0, cost, name=name)


@layer_register(log_shape=True)
def CaffeBilinearUpSample(x, shape):
    """
    Deterministic bilinearly-upsample the input images.
    It is implemented by deconvolution with "BilinearFiller" in Caffe.
    It is aimed to mimic caffe behavior.

    Args:
        x (tf.Tensor): a NCHW tensor
        shape (int): the upsample factor

    Returns:
        tf.Tensor: a NCHW tensor.
    """
    inp_shape = x.shape.as_list()
    ch = inp_shape[1]
    assert ch == 1, "This layer only works for channel=1"
    # for a version that supports >1 channels, see:
    # https://github.com/tensorpack/tensorpack/issues/1040#issuecomment-452798180

    shape = int(shape)
    filter_shape = 2 * shape

    def bilinear_conv_filler(s):
        """
        s: width, height of the conv filter
        https://github.com/BVLC/caffe/blob/99bd99795dcdf0b1d3086a8d67ab1782a8a08383/include/caffe/filler.hpp#L219-L268
        """
        f = np.ceil(float(s) / 2)
        c = float(2 * f - 1 - f % 2) / (2 * f)
        ret = np.zeros((s, s), dtype='float32')
        for x in range(s):
            for y in range(s):
                ret[x, y] = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
        return ret

    w = bilinear_conv_filler(filter_shape)
    w = np.repeat(w, ch * ch).reshape((filter_shape, filter_shape, ch, ch))

    weight_var = tf.constant(w, tf.float32,
                             shape=(filter_shape, filter_shape, ch, ch),
                             name='bilinear_upsample_filter')
    x = tf.pad(x, [[0, 0], [0, 0], [shape - 1, shape - 1], [shape - 1, shape - 1]], mode='SYMMETRIC')
    out_shape = tf.shape(x) * tf.constant([1, 1, shape, shape], tf.int32)
    deconv = tf.nn.conv2d_transpose(x, weight_var, out_shape,
                                    [1, 1, shape, shape], 'SAME', data_format='NCHW')
    edge = shape * (shape - 1)
    deconv = deconv[:, :, edge:-edge, edge:-edge]

    if inp_shape[2]:
        inp_shape[2] *= shape
    if inp_shape[3]:
        inp_shape[3] *= shape
    deconv.set_shape(inp_shape)
    return deconv


class Model(ModelDesc):
    def inputs(self):
        return [tf.TensorSpec([None, None, None, 3], tf.float32, 'image'),
                tf.TensorSpec([None, None, None], tf.int32, 'edgemap')]

    def build_graph(self, image, edgemap):
        image = image - tf.constant([104, 116, 122], dtype='float32')
        image = tf.transpose(image, [0, 3, 1, 2])
        edgemap = tf.expand_dims(edgemap, 3, name='edgemap4d')

        def branch(name, l, up):
            with tf.variable_scope(name):
                l = Conv2D('convfc', l, 1, kernel_size=1, activation=tf.identity,
                           use_bias=True,
                           kernel_initializer=tf.constant_initializer())
                while up != 1:
                    l = CaffeBilinearUpSample('upsample{}'.format(up), l, 2)
                    up = up // 2
                return l

        with argscope(Conv2D, kernel_size=3, activation=tf.nn.relu), \
                argscope([Conv2D, MaxPooling], data_format='NCHW'):
            l = Conv2D('conv1_1', image, 64)
            l = Conv2D('conv1_2', l, 64)
            b1 = branch('branch1', l, 1)
            l = MaxPooling('pool1', l, 2)

            l = Conv2D('conv2_1', l, 128)
            l = Conv2D('conv2_2', l, 128)
            b2 = branch('branch2', l, 2)
            l = MaxPooling('pool2', l, 2)

            l = Conv2D('conv3_1', l, 256)
            l = Conv2D('conv3_2', l, 256)
            l = Conv2D('conv3_3', l, 256)
            b3 = branch('branch3', l, 4)
            l = MaxPooling('pool3', l, 2)

            l = Conv2D('conv4_1', l, 512)
            l = Conv2D('conv4_2', l, 512)
            l = Conv2D('conv4_3', l, 512)
            b4 = branch('branch4', l, 8)
            l = MaxPooling('pool4', l, 2)

            l = Conv2D('conv5_1', l, 512)
            l = Conv2D('conv5_2', l, 512)
            l = Conv2D('conv5_3', l, 512)
            b5 = branch('branch5', l, 16)

            final_map = Conv2D('convfcweight',
                               tf.concat([b1, b2, b3, b4, b5], 1), 1, kernel_size=1,
                               kernel_initializer=tf.constant_initializer(0.2),
                               use_bias=False, activation=tf.identity)
        costs = []
        for idx, b in enumerate([b1, b2, b3, b4, b5, final_map]):
            b = tf.transpose(b, [0, 2, 3, 1])
            output = tf.nn.sigmoid(b, name='output{}'.format(idx + 1))
            xentropy = class_balanced_sigmoid_cross_entropy(
                b, edgemap,
                name='xentropy{}'.format(idx + 1))
            costs.append(xentropy)

        # some magic threshold
        pred = tf.cast(tf.greater(output, 0.5), tf.int32, name='prediction')
        wrong = tf.cast(tf.not_equal(pred, edgemap), tf.float32)
        wrong = tf.reduce_mean(wrong, name='train_error')

        wd_w = tf.train.exponential_decay(2e-4, get_global_step_var(),
                                          80000, 0.7, True)
        wd_cost = tf.multiply(wd_w, regularize_cost('.*/W', tf.nn.l2_loss), name='wd_cost')
        costs.append(wd_cost)

        add_param_summary(('.*/W', ['histogram']))   # monitor W
        total_cost = tf.add_n(costs, name='cost')
        add_moving_summary(wrong, total_cost, *costs)
        return total_cost

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=3e-5, trainable=False)
        opt = tf.train.AdamOptimizer(lr, epsilon=1e-3)
        return optimizer.apply_grad_processors(
            opt, [gradproc.ScaleGradient(
                [('convfcweight.*', 0.1), ('conv5_.*', 5)])])

class PipelineRunModels(PipelineStep):

    def __init__(self, semantic_path: str, normals_path: str, unlit_path: str,
                 elevation_path: str, lighting_path: str):
        self.model_semantic = load_model(semantic_path)
        self.model_normals = load_model(normals_path)
        self.model_unlit = load_model(unlit_path)
        self.model_elevation = load_model(elevation_path)
        self.model_lighting = load_model(lighting_path)

        model_path = os.path.sep.join(["hed_model", "HED_pretrained_bsds.npz"])
        pred_config = PredictConfig(
            model=Model(),
            session_init=SmartInit(model_path),
            input_names=['image'],
            output_names=['output' + str(k) for k in range(1, 7)])
        self.model_hed = OfflinePredictor(pred_config)


    @property
    def required_keys(self) -> list:
        return ["image"]

    @property
    def output_keys(self) -> list:
        return ["image", "semantic", "semantic_probs", "normals", "elevation", "lighting", "normals_latents", "hed"]

    @property
    def is_batched(self) -> bool:
        return True

    def run(self, data):
        images = [datum["image"] for datum in data]
        
        def _run_single(model, key):
            results = feed_image_batched(model, images)
            for datum, result in zip(data, results):
                datum[key] = result

        _run_single(self.model_elevation, "elevation")
        _run_single(self.model_lighting, "lighting")
        _run_single(self.model_unlit, "unlit")
        
        semantic_input = [{ "image": datum["image"], "unlit": datum["unlit"] } for datum in data]
        semantic_results = [s["output"] for s in feed_images_batched(self.model_semantic, semantic_input)]
        for datum, result in zip(data, semantic_results):
            datum["semantic"] = result
            datum["semantic_probs"] = softmax(result/255, axis=-1)

        # Normals output with latents
        input_tensor = list(self.model_normals.feed_tensors.values())[0]
        latent_tensors = self.model_normals.graph.get_tensor_by_name(
            "generator/decoder_8/conv2d_transpose/BiasAdd:0")
        output_tensor = list(self.model_normals.fetch_tensors.values())[0]
        normals_latents, normals = self.model_normals.session.run([latent_tensors, output_tensor], feed_dict={
            input_tensor: [cv2.resize(datum["image"], (512, 512)).astype(np.float32)/255 for datum in data]})
        normals_latents = normals_latents[:, :, :, :128].reshape(len(data), 1, -1)

        for datum, nl, n in zip(data, normals_latents, normals):
            datum["normals"] = n
            datum["normals_latents"] = nl

        images = [cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (1024, 1024)).astype('float32') for img in images]
        outputs = self.model_hed(images)

        for datum, hed in zip(data, outputs[5]):
            datum["hed"] = (255 * hed[:,:,0]).astype("uint8")