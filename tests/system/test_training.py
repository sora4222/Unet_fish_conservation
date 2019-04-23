import logging
import pathlib

import pytest
from tensorflow.python import keras

from image_genarators.input_output_generators import generate_image_data
from model.generate_u_net import unet
from model.utils.loss_function import dice_coef_loss, jaccard_distance_loss

resource_location: pathlib.PurePath = pathlib.PurePath(__file__).parents[1].joinpath("resources")
logging.basicConfig(filemode="w+", filename="prediction_logs.txt", level=logging.DEBUG,
                    format="[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s")


@pytest.fixture(scope="class")
def unet_model():
    return unet()


def test_train_on_alb_img_00019_categorical_crossentropy_compiles_and_runs(unet_model):
    """
    Checks this will compile and run one of the images through the sparse cross entropy
    loss function
    """
    model = unet_model

    optimizer = keras.optimizers.Adam(lr=0.001)
    model.compile(optimizer=optimizer, loss=keras.losses.categorical_crossentropy)

    validation_data = generate_image_data(1,
                                          str(resource_location.joinpath("images")),
                                          glob_pattern="ALB_img_00019.tif")
    EPOCHS = 20
    model.fit_generator(
        validation_data,
        steps_per_epoch=1,
        epochs=EPOCHS)


def test_train_on_alb_img_00019_jaccard_distance_loss_compiles_and_runs(unet_model):
    """
    Checks this will compile and run one of the images through the jaccard distance
    loss function
    """
    model = unet_model

    optimizer = keras.optimizers.Adam(lr=0.001)
    model.compile(optimizer=optimizer, loss=jaccard_distance_loss())

    validation_data = generate_image_data(1,
                                          str(resource_location.joinpath("images")),
                                          glob_pattern="ALB_img_00019.tif")
    EPOCHS = 20
    model.fit_generator(
        validation_data,
        steps_per_epoch=1,
        epochs=EPOCHS)


def test_train_on_alb_img_00019_dice_coef_loss_compiles_and_runs(unet_model):
    """
    Checks this will compile and run one of the images through the sparse cross entropy
    loss function
    """
    model = unet_model

    optimizer = keras.optimizers.Adam(lr=0.001)
    model.compile(optimizer=optimizer, loss=dice_coef_loss())

    validation_data = generate_image_data(1,
                                          str(resource_location.joinpath("images")),
                                          glob_pattern="ALB_img_00019.tif")
    EPOCHS = 20
    model.fit_generator(
        validation_data,
        steps_per_epoch=1,
        epochs=EPOCHS)


@pytest.mark.skip(reason="Not implemented yet")
def test_train_on_resource_images(unet_model):
    """
    Trains the Unet on the 3 images in the Images folder in resource
    for many epochs, I expect that the masks will given by the model will
    have a very low loss functional value.
    :return:
    """
    pass
