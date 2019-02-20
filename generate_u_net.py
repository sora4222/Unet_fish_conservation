from typing import List, Tuple

from tensorflow.python import keras
from tensorflow.python.keras import layers
from input_output_generators import generate_image_data, SEGMENTATION_DIRECTORY_LOCATION, save_image
import logging
import numpy as np
from model.utils.crop_functions import get_crop_dimensions

BATCH_SIZE: int = 2
IMAGE_SIZE: List[int] = [768, 768, 3]
PADDING: str = "same"
NUM_CLASSES: int = 1


def add_downsample_layer(layer: keras.layers.Layer, filters: int):
    """
    Adds a downsample set of Convolutional2D layers.
    :param model:
    :param filters:
    :return: the downsampled layer, the conv2d layer just before
    """
    # padding "valid" means no padding
    cnn_layer_1 = layers.Conv2D(filters,
                                kernel_size=(3, 3),
                                strides=(1, 1),
                                padding=PADDING,
                                activation="relu",
                                data_format="channels_last")(layer)
    logging.debug(f"add_downsample_layer: cnn_layer_1 shape: {cnn_layer_1.get_shape()}")
    cnn_layer_1_norm = layers.BatchNormalization()(cnn_layer_1)
    cnn_layer_2 = layers.Conv2D(filters,
                                kernel_size=(3, 3),
                                strides=(1, 1),
                                padding=PADDING,
                                activation="relu",
                                data_format="channels_last")(cnn_layer_1_norm)
    logging.debug(f"add_downsample_layer: cnn_layer_2 shape (skip connection): {cnn_layer_2.get_shape()[2]}")
    cnn_layer_2_norm = layers.BatchNormalization()(cnn_layer_2)
    sample_down_layer = layers.MaxPool2D(pool_size=(2, 2),
                                         strides=(2, 2),
                                         padding=PADDING,
                                         data_format="channels_last")(cnn_layer_2_norm)
    return sample_down_layer, cnn_layer_2


def add_expansive_layer(input_layer, filters_to_end_with: int,
                        jump_connections) -> layers.Layer:
    logging.debug(f"filters: {filters_to_end_with}")
    # Upsample
    logging.debug(f"add_expansive_layer : input_layer layer shape: {input_layer.get_shape()}")
    upsample_layer = layers.Conv2DTranspose(filters_to_end_with,
                                            kernel_size=(2, 2),
                                            padding=PADDING,
                                            data_format="channels_last",
                                            strides=(2, 2))(input_layer)
    logging.debug(f"add_expansive_layer : Upsample layer shape: {upsample_layer.get_shape()}")

    crop_dimensions: Tuple[Tuple[int], Tuple[int]] = get_crop_dimensions(to_crop=jump_connections,
                                                                         layer_to_crop_to=upsample_layer)
    logging.info(f"cropping dimensions: {crop_dimensions}")
    crop_layer = layers.Cropping2D(cropping=crop_dimensions)(jump_connections)
    concatenation_layer = layers.concatenate([upsample_layer, crop_layer])
    logging.debug(f"Come out of concat with: {keras.backend.int_shape(concatenation_layer)}")
    cnn_layer_1 = layers.Conv2D(filters=filters_to_end_with // 2,
                                kernel_size=(3, 3),
                                activation="relu",
                                data_format="channels_last",
                                padding=PADDING)(concatenation_layer)
    cnn_layer_1_norm = layers.BatchNormalization()(cnn_layer_1)
    cnn_layer_2 = layers.Conv2D(filters=filters_to_end_with // 2,
                                kernel_size=(3, 3),
                                activation="relu",
                                data_format="channels_last",
                                padding=PADDING)(cnn_layer_1_norm)
    cnn_layer_2_norm: layers.Layer = layers.BatchNormalization()(cnn_layer_2)
    return cnn_layer_2_norm


# try: https://www.kaggle.com/keegil/keras-u-net-starter-lb-0-277
def log_shape(layer_name: str, layer) -> None:
    logging.info(f"{layer_name} has shape: {layer.get_shape()}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format='%(name)s - %(levelname)s - %(message)s',
                        filename="logs.txt",
                        filemode="w+")
    INITIAL_SIZE = 64
    input_tensor: layers.Layer = layers.Input(shape=IMAGE_SIZE, name="Input_tensor")

    logging.debug("Going down the U-net:...")

    input_normal = layers.BatchNormalization()(input_tensor)
    layer_1_downsampled, layer_1 = add_downsample_layer(input_normal, INITIAL_SIZE)
    logging.debug(f"layer 1 downsampled: {keras.backend.int_shape(layer_1_downsampled)}")
    logging.debug(f"layer 1: {keras.backend.int_shape(layer_1)}")
    log_shape("layer_1_downsampled", layer_1_downsampled)
    log_shape("layer_1", layer_1)

    layer_2_downsampled, layer_2 = add_downsample_layer(layer_1_downsampled, INITIAL_SIZE * 2)
    layer_2_downsampled_norm = layers.BatchNormalization()(layer_2_downsampled)
    logging.debug(f"layer 2 downsampled: {keras.backend.int_shape(layer_2_downsampled)}")
    logging.debug(f"layer 2: {keras.backend.int_shape(layer_2)}")
    log_shape("layer_2_downsampled", layer_2_downsampled)
    log_shape("layer_2", layer_2)

    layer_3_downsampled, layer_3 = add_downsample_layer(layer_2_downsampled_norm, INITIAL_SIZE * 4)
    layer_3_downsampled_norm = layers.BatchNormalization()(layer_3_downsampled)
    logging.debug(f"layer 3 downsampled: {keras.backend.int_shape(layer_3_downsampled)}")
    logging.debug(f"layer 3: {keras.backend.int_shape(layer_3)}")
    log_shape("layer_3_downsampled", layer_3_downsampled)
    log_shape("layer_3", layer_3)

    layer_4_downsampled, layer_4 = add_downsample_layer(layer_3_downsampled_norm, INITIAL_SIZE * 8)
    layer_4_downsampled_norm = layers.BatchNormalization()(layer_4_downsampled)
    logging.debug(f"layer 4 downsampled: {keras.backend.int_shape(layer_4_downsampled)}")
    logging.debug(f"layer 4: {keras.backend.int_shape(layer_4)}")
    log_shape("layer_4_downsampled", layer_4_downsampled)
    log_shape("layer_4", layer_4)

    layer_5 = layers.Conv2D(INITIAL_SIZE * 16, (3, 3), padding=PADDING)(layer_4_downsampled_norm)
    layer_5_norm = layers.BatchNormalization()(layer_5)
    logging.debug(f"layer 5 norm {keras.backend.int_shape(layer_5_norm)}")

    logging.debug("Going back up the U-net")
    logging.debug("layer 6")
    layer_6_upsampled = add_expansive_layer(layer_5_norm, INITIAL_SIZE * 8, layer_4)
    logging.debug("layer 7")
    layer_7_upsampled = add_expansive_layer(layer_6_upsampled, INITIAL_SIZE * 4, layer_3)
    logging.debug("layer 8")
    layer_8_upsampled = add_expansive_layer(layer_7_upsampled, INITIAL_SIZE * 2, layer_2)
    layer_9_upsampled = add_expansive_layer(layer_8_upsampled, INITIAL_SIZE, layer_1)

    ch, cw = get_crop_dimensions(input_tensor, layer_9_upsampled)
    layer_10 = layers.Conv2D(2, (3,3), padding=PADDING)
    layer_11_padding = layers.ZeroPadding2D(padding=(ch, cw))(layer_9_upsampled)
    layer_11_conv = layers.Conv2D(NUM_CLASSES, (1, 1))(layer_11_padding)

    model: keras.Model = keras.Model(inputs=[input_tensor], outputs=[layer_11_conv])
    model.summary()

    logging.debug(f"Model {model.output.get_shape()}")

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

    model.save(f"Fish_unet_{EPOCHS}_{NUMBER_OF_TRAINING_IMAGES}.cnn",
               overwrite=True)
    prediction: np.ndarray = model.predict(np.reshape(validation_data[0][0], (1, 768, 768, 3)))

    save_image(prediction[0], 1, mask=True)
