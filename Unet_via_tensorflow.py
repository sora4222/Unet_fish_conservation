import tensorflow as tf
from typing import Tuple

PADDING: str = "valid"
IMG_WIDTH: int = 768
IMG_HEIGHT: int = 768
BATCH_SIZE: int = 16

def crop_and_concat(to_crop: tf.Tensor, layer_to_concatenate_to: tf.Tensor) -> tf.Tensor:
    """
    Obtained from
    https://github.com/jakeret/tf_unet/blob/master/tf_unet/layers.py
    slices the to_crop layer to be the same size as the layer_to_concatenate_to layer
    :param to_crop:
    :param layer_to_concatenate_to:
    :return:
    """
    with tf.name_scope("crop_and_concat"):
        layer_to_crop_shape = tf.shape(to_crop)
        layer_shape = tf.shape(layer_to_concatenate_to)
        # offsets for the top left corner of the crop
        offsets = [0, (layer_to_crop_shape[1] - layer_shape[1]) // 2,
                   (layer_to_crop_shape[2] - layer_shape[2]) // 2, 0]
        size = [-1, layer_shape[1], layer_shape[2], -1]
        x1_crop = tf.slice(to_crop, offsets, size)
        return tf.concat([x1_crop, layer_to_concatenate_to], 3)


def add_downsample_layer(layer: tf.Tensor,
                         input_layer_size: int,
                         layer_name: str) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Gives the layers going down a U-net
    :param input_layer_size: The size of the starting input
    :param layer_name: The name to assign to the scope
    :return Two tensors [output layer downsized, ouputlayer]
    """
    with tf.name_scope(layer_name):
        non_pooled_layer: tf.Tensor = tf.layers.Conv2D(input_layer_size,
                                                       kernel_size=(3, 3),
                                                       strides=(1, 1),
                                                       padding=PADDING,
                                                       activation="relu")(layer)
        non_pooled_layer = tf.layers.BatchNormalization()(non_pooled_layer)
        non_pooled_layer = tf.layers.Conv2D(input_layer_size,
                                            kernel_size=(3, 3),
                                            strides=(1, 1),
                                            padding=PADDING,
                                            activation="relu",
                                            name="Second convolution")(non_pooled_layer)
        non_pooled_layer_norm: tf.Tensor = tf.layers.BatchNormalization()(non_pooled_layer)
        pooled_layer: tf.Tensor = tf.layers.MaxPooling2D(pool_size=(2, 2),
                                                         strides=(2, 2),
                                                         padding=PADDING,
                                                         name="pooling")(non_pooled_layer_norm)
        return pooled_layer, non_pooled_layer


def add_expansive_layer(layer: tf.Tensor,
                        downsample_input_layer: tf.Tensor,
                        filters_to_end_with: int,
                        layer_name: str) -> tf.Tensor:
    with tf.name_scope(layer_name):
        upsample_layer = tf.layers.Conv2DTranspose(filters_to_end_with,
                                                   kernel_size=(2, 2),
                                                   strides=(2, 2),
                                                   name="Upsampling")(layer)

        concatenation: tf.Tensor = crop_and_concat(downsample_input_layer, upsample_layer)
        conv: tf.Tensor = tf.layers.BatchNormalization()(concatenation)
        conv = tf.layers.Conv2D(filters_to_end_with // 2,
                                kernel_size=(3, 3),
                                activation="relu",
                                padding=PADDING)(conv)
        return conv


if __name__ == '__main__':
    INITIAL_SIZE: int = 64
    input_layer = tf.reshape()

    with tf.name_scope("input image processing"):
        Input = tf.placeholder("", shape=[None, IMG_HEIGHT, IMG_WIDTH, 3])