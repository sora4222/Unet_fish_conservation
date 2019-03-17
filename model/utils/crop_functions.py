import keras as K
import tensorflow as tf
from typing import Tuple, Union
import logging


def correct_dimension(dimension_size: int) -> Tuple[int, int]:
    """
    Corrects the dimensions for an image that has an odd height or width
    :param dimension_size:
    :return:
    """
    assert dimension_size >= 0, "The dimension size is less than 0"
    if dimension_size % 2 != 0:
        logging.debug("Odd number of dimensions")
        return int(dimension_size / 2), int(dimension_size / 2) + 1
    else:
        logging.debug("Even number of dimensions")
        return int(dimension_size / 2), int(dimension_size / 2)


def get_crop_dimensions(to_crop: Union[tf.Tensor, tf.layers.Layer],
                        layer_to_crop_to: Union[tf.Tensor, tf.layers.Layer]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Obtained from
    https://github.com/zizhaozhang/unet-tensorflow-keras/blob/master/model.py
    slices the to_crop layer to be the same size as the layer_to_concatenate_to layer.
    :type to_crop: tf.Tensor
    :param to_crop: The Tensor to crop.
    :param layer_to_crop_to: The Tensor to match the dimensions of.
    :return: The number of rows and columns to remove in the format keras.layers.Cropping2D uses
    """
    logging.debug("get_crop_dimensions")
    with tf.name_scope("crop_and_concat"):
        logging.debug(f"{__name__}: to_crop: {to_crop.get_shape()[2]}")
        logging.debug(f"{__name__}: layer_to_crop: {layer_to_crop_to.get_shape()[2]}")
        change_width: int = (to_crop.get_shape()[2] - layer_to_crop_to.get_shape()[2]).value
        change_height: int = (to_crop.get_shape()[1] - layer_to_crop_to.get_shape()[1]).value
        logging.debug(f"Change width {change_width}")
        logging.debug(f"Change height {change_height}")
        corrected_width: Tuple[int, int] = correct_dimension(change_width)
        corrected_height: Tuple[int, int] = correct_dimension(change_height)
        return corrected_height, corrected_width
