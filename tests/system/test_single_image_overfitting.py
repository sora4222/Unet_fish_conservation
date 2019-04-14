import pathlib

import pytest
from tensorflow.python import keras

from image_genarators.input_output_generators import generate_image_data
from model.generate_u_net import unet
from model.utils.loss_function import dice_coef_loss, jaccard_distance_loss

resource_location: pathlib.PurePath = pathlib.PurePath(__file__).parents[1].joinpath("resources")

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
    model.compile(optimizer=optimizer, loss=keras.losses.sparse_categorical_crossentropy)

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

    optimizer = keras.optimizers.Adam(lr=0.1)
    model.compile(optimizer=optimizer, loss=jaccard_distance_loss())

    validation_data = generate_image_data(1,
                                          str(resource_location.joinpath("images")),
                                          glob_pattern="ALB_img_00019.tif")
    EPOCHS = 30
    model.fit_generator(
        validation_data,
        steps_per_epoch=1,
        epochs=EPOCHS)
    image, mask = next(validation_data)
    assert model.evaluate(image, mask) == pytest.approx(0, abs=1e-3)


def test_train_on_alb_img_00019_dice_coef_loss_compiles_and_gives_results(unet_model):
    """
    This will test that the UNET can run on a single image generating
    I expect that if I give a single image, and many epochs that the
    mask will be given back almost perfectly.
    """
    model = unet_model

    optimizer = keras.optimizers.Adam(lr=0.001)
    model.compile(optimizer=optimizer, loss=dice_coef_loss())

    validation_data = generate_image_data(1,
                                          str(resource_location.joinpath("images")),
                                          glob_pattern="ALB_img_00019.tif")
    EPOCHS = 30
    model.fit_generator(
        validation_data,
        steps_per_epoch=1,
        epochs=EPOCHS)
    image, mask = next(validation_data)
    mask_predict = model.predict(image)

    print(f"Mask prediction greater than 2: {mask_predict[mask_predict > 2]}")
    print(f"Mask prediction less than 0: {mask_predict[mask_predict < 0]}")
    print(f"Mask prediction less than 0: {mask_predict[mask_predict < 0]}")
    assert model.evaluate(image, mask) == pytest.approx(0, abs=1e-3)


def test_check():
    validation_data = generate_image_data(1,
                                          str(resource_location.joinpath("images")),
                                          glob_pattern="ALB_img_00019.tif")
    image, mask = next(validation_data)
    print(mask[mask > 1])
