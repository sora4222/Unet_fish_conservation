from tensorflow.python import keras
import pytest

from image_genarators.input_output_generators import generate_image_data, SEGMENTATION_DIRECTORY_LOCATION
from model.generate_u_net import unet


@pytest.fixture(scope="class")
def unet_model():
    return unet()


@pytest.mark.skip(reason="Unimplemented.")
def test_train_on_alb_img_00019(unet_model):
    """
    This will test that the UNET can run on a single image generating
    I expect that if I give a single image, and many epochs that the
    mask will be given back perfectly.
    """
    model = unet_model

    optimizer = keras.optimizers.Adam(lr=0.01)
    model.compile(optimizer=optimizer, loss=keras.losses.sparse_categorical_crossentropy)

    # TODO: Add validation data and histogram=x where x > 0
    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=r'./logs',
        batch_size=1,
        histogram_freq=0,
        write_graph=True,
        write_images=False)

    reduce_learning_rate_callback = keras.callbacks.ReduceLROnPlateau(patience=3,
                                                                      mode="min")

    validation_generator = generate_image_data(50, SEGMENTATION_DIRECTORY_LOCATION + "\\train")

    validation_data = next(validation_generator)
    EPOCHS = 20
    NUMBER_OF_TRAINING_IMAGES = 137
    model.fit_generator(
        generate_image_data(BATCH_SIZE,
                            SEGMENTATION_DIRECTORY_LOCATION + "\\train",
                            deg_range=180,
                            save_transformed_images=False),
        steps_per_epoch=58,
        epochs=EPOCHS,
        callbacks=[tensorboard_callback, reduce_learning_rate_callback],
        validation_data=validation_data)


@pytest.mark.skip(reason="Not implemented yet")
def test_train_on_resource_images(unet_model):
    """
    Trains the Unet on the 3 images in the Images folder in resource
    for many epochs, I expect that the masks will given by the model will
    have a very low loss functional value.
    :return:
    """
