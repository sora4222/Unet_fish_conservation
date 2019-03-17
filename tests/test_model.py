from model.generate_u_net import unet, IMAGE_SIZE
from tensorflow.python import keras
import pytest


@pytest.fixture(scope="session")
def model() -> keras.models:
    return unet()


def test_model_output_shape(model):
    assert model.output_shape == (None, IMAGE_SIZE[0], IMAGE_SIZE[1], 1)


def test_model_input_shape(model):
    assert model.input_shape == (None, IMAGE_SIZE[0], IMAGE_SIZE[1], IMAGE_SIZE[2])

