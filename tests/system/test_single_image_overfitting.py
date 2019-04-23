import logging
import pathlib

import pytest
import tensorflow as tf
from tensorflow.python import keras

from image_genarators.input_output_generators import generate_image_data, save_image_location
from model.generate_u_net import unet
from model.utils.loss_function import dice_coef_loss, jaccard_distance_loss

resource_location: pathlib.PurePath = pathlib.PurePath(__file__).parents[1].joinpath("resources")

logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=logging.DEBUG)


@pytest.fixture(scope="class")
def unet_model():
    return unet()


def test_train_on_alb_img_00019_sparse_crossentropy_compiles_and_gives_results(unet_model):
    """
    This will test that the UNET can run on a single image generating
    I expect that if I give a single image, and many epochs that the
    mask will be given back perfectly.
    """
    model = unet_model

    optimizer = keras.optimizers.Adam(lr=0.01)
    model.compile(optimizer=optimizer, loss=keras.losses.categorical_crossentropy)

    validation_data = generate_image_data(1,
                                          str(resource_location.joinpath("images")),
                                          glob_pattern="ALB_img_00019.tif")
    EPOCHS = 100
    model.fit_generator(
        validation_data,
        steps_per_epoch=1,
        epochs=EPOCHS)
    image, mask = next(validation_data)
    assert model.evaluate(image, mask) == pytest.approx(0, abs=1e-3)


def test_train_on_alb_img_00019_jaccard_distance_loss_compiles_and_gives_results(unet_model):
    """
    This will test that the UNET can run on a single image generating
    I expect that if I give a single image, and many epochs that the
    mask will be given back perfectly.
    """
    model = unet_model
    tf.set_random_seed(1)

    optimizer = keras.optimizers.Adam(lr=4)

    model.compile(optimizer=optimizer, loss=jaccard_distance_loss(10000, 10))

    validation_data = generate_image_data(1,
                                          str(resource_location.joinpath("images")),
                                          glob_pattern="ALB_img_00019.tif")
    EPOCHS = 20
    model.fit_generator(
        validation_data,
        steps_per_epoch=1,
        epochs=EPOCHS)
    image, mask = next(validation_data)
    mask_pred = model.predict(image)
    save_image_location(".pytest_cache/")(mask_pred[0], 1, mask=True)
    assert model.evaluate(validation_data, steps=1) == pytest.approx(0, abs=5e-2)


def test_train_on_alb_img_00019_dice_coef_loss_compiles_and_gives_results(unet_model):
    """
    This will test that the UNET can run on a single image generating
    I expect that if I give a single image, and many epochs that the
    mask will be given back almost perfectly.
    """
    model = unet_model
    tf.set_random_seed(1)

    optimizer = keras.optimizers.Adam(lr=0.03)

    model.compile(optimizer=optimizer, loss=dice_coef_loss(100, 1e-10))

    validation_data = generate_image_data(1,
                                          str(resource_location.joinpath("images")),
                                          glob_pattern="ALB_img_00019.tif")
    EPOCHS = 50
    model.fit_generator(
        validation_data,
        steps_per_epoch=1,
        epochs=EPOCHS)
    image, mask = next(validation_data)
    mask_pred = model.predict(image)
    save_image_location(".pytest_cache/")(mask_pred[0], 1, mask=True)
    assert model.evaluate(validation_data, steps=1) == pytest.approx(0, abs=5e-2)
