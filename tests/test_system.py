from model.generate_u_net import unet, IMAGE_SIZE
from tensorflow.python import keras
import numpy as np
import tensorflow as tf
import pytest


@pytest.fixture(scope="function")
def model() -> keras.models:
    """
    Gives back the unet model
    """
    return unet()


def test_zeros_and_ones(model):
    """
    Test the model compiles and can be trained on a mask of ones.
    """

    model_typed: keras.Model = model

    # need to generate an input of zeros
    input_zeros: np.ndarray = np.zeros(
        shape=(1, IMAGE_SIZE[0], IMAGE_SIZE[1], IMAGE_SIZE[2]))

    # need to generate an output of ones
    output_labels: np.ndarray = np.ones(
        shape=(1, IMAGE_SIZE[0], IMAGE_SIZE[1], 1))

    optimizer = keras.optimizers.Adam()

    # Initialize, predict, loss function
    # need to use model
    model_typed.compile(optimizer=optimizer,
                        loss=keras.losses.binary_crossentropy)

    model_typed.fit(x=input_zeros,
                    y=output_labels,
                    batch_size=1)
