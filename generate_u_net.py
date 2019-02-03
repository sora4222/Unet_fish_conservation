from typing import List

import keras
from keras import Model
from keras import layers as layers
from keras.losses import binary_crossentropy
from keras.optimizers import Adam
from input_output_generators import generate_image_data, SEGMENTATION_DIRECTORY_LOCATION, save_image
import logging
import numpy as np

BATCH_SIZE: int = 2
IMAGE_SIZE: List[int] = [768, 768, 3]


def add_downsample_layer(layer: keras.layers.Layer, filters: int):
    """

    :param model:
    :param filters:
    :return: model, the second cnn network
    """
    # padding "valid" means no padding
    cnn_layer_1 = layers.Conv2D(filters, kernel_size=(3, 3), strides=(1, 1),
                                padding="same", activation="relu", data_format="channels_last")(layer)
    cnn_layer_1_norm = layers.BatchNormalization()(cnn_layer_1)
    cnn_layer_2 = layers.Conv2D(filters, kernel_size=(3, 3), strides=(1, 1),
                                padding="same", activation="relu", data_format="channels_last")(cnn_layer_1_norm)
    cnn_layer_2_norm = layers.BatchNormalization()(cnn_layer_2)
    sample_down_layer = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same",
                                         data_format="channels_last")(cnn_layer_2_norm)
    return sample_down_layer, cnn_layer_2


def add_expansive_layer(input_layer, filters_to_end_with: int,
                        downsample_input_layer) -> layers.Layer:
    print("filters: ", filters_to_end_with)
    # Upsample
    upsample_layer = layers.Conv2DTranspose(filters_to_end_with, kernel_size=(2, 2),
                                            data_format="channels_last",
                                            strides=(2, 2))(input_layer)

    print("Upsample layer: ", keras.backend.int_shape(upsample_layer))
    print("Input layer to concatenate: ", keras.backend.int_shape(downsample_input_layer))
    concatenation_layer = layers.concatenate([upsample_layer, downsample_input_layer])
    cnn_layer_1 = layers.Conv2D(filters=filters_to_end_with // 2,
                                kernel_size=(3, 3), activation="relu",
                                data_format="channels_last",
                                padding="same")(concatenation_layer)
    cnn_layer_1_norm = layers.BatchNormalization()(cnn_layer_1)
    cnn_layer_2 = layers.Conv2D(filters=filters_to_end_with // 2,
                                kernel_size=(3, 3),
                                activation="relu",
                                data_format="channels_last",
                                padding="same")(cnn_layer_1_norm)
    cnn_layer_2_norm: layers.Layer = layers.BatchNormalization()(cnn_layer_2)
    return cnn_layer_2_norm


# try: https://www.kaggle.com/keegil/keras-u-net-starter-lb-0-277
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
    INITIAL_SIZE = 64
    input_tensor: layers.Layer = layers.Input(shape=IMAGE_SIZE, name="Input_tensor")

    print("")
    print("Going down the U-net:...")

    input_normal = layers.BatchNormalization()(input_tensor)
    layer_1_downsampled, layer_1 = add_downsample_layer(input_normal, INITIAL_SIZE)
    print("layer 1 downsampled:", keras.backend.int_shape(layer_1_downsampled))
    print("layer 1: ", keras.backend.int_shape(layer_1))

    layer_2_downsampled, layer_2 = add_downsample_layer(layer_1_downsampled, INITIAL_SIZE * 2)
    layer_2_downsampled_norm = layers.BatchNormalization()(layer_2_downsampled)
    print("layer 2 downsampled:", keras.backend.int_shape(layer_2_downsampled))
    print("layer 2: ", keras.backend.int_shape(layer_2))

    layer_3_downsampled, layer_3 = add_downsample_layer(layer_2_downsampled_norm, INITIAL_SIZE * 4)
    layer_3_downsampled_norm = layers.BatchNormalization()(layer_3_downsampled)
    print("layer 3 downsampled:", keras.backend.int_shape(layer_3_downsampled))
    print("layer 3: ", keras.backend.int_shape(layer_3))

    layer_4_downsampled, layer_4 = add_downsample_layer(layer_3_downsampled_norm, INITIAL_SIZE * 8)
    layer_4_downsampled_norm = layers.BatchNormalization()(layer_4_downsampled)
    print("layer 4 downsampled:", keras.backend.int_shape(layer_4_downsampled))
    print("layer 4: ", keras.backend.int_shape(layer_4))

    layer_5 = layers.Conv2D(INITIAL_SIZE * 16, (3, 3), padding="same")(layer_4_downsampled_norm)
    layer_5_norm = layers.BatchNormalization()(layer_5)
    print("layer 5 norm", keras.backend.int_shape(layer_5_norm))

    print()
    print("Going back up the U-net")
    layer_6_upsampled = add_expansive_layer(layer_5_norm, INITIAL_SIZE * 8, layer_4)
    layer_7_upsampled = add_expansive_layer(layer_6_upsampled, INITIAL_SIZE * 4, layer_3)
    layer_8_upsampled = add_expansive_layer(layer_7_upsampled, INITIAL_SIZE * 2, layer_2)
    layer_9_upsampled = add_expansive_layer(layer_8_upsampled, INITIAL_SIZE, layer_1)

    # activation softmax has higher loss but lower val_loss
    output_layer = layers.Conv2D(1, (1, 1), activation="sigmoid", padding="same")(layer_9_upsampled)

    model: Model = Model(inputs=[input_tensor], outputs=[output_layer])
    model.summary()

    optimizer = Adam(lr=0.001)
    model.compile(optimizer=optimizer, loss=binary_crossentropy)

    # TODO: Add validation data and histogram=x where x > 0
    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=r'./logs',
        batch_size=1,
        histogram_freq=0,
        write_graph=True,
        write_images=False,
        update_freq='batch')

    reduce_learning_rate_callback = keras.callbacks.ReduceLROnPlateau(patience=3,
                                                                      mode="min")

    validation_generator = generate_image_data(50, SEGMENTATION_DIRECTORY_LOCATION + "\\validation")

    validation_data = next(validation_generator)
    EPOCHS = 10
    NUMBER_OF_TRAINING_IMAGES = 137
    model.fit_generator(generate_image_data(BATCH_SIZE, SEGMENTATION_DIRECTORY_LOCATION + "\\train", deg_range=180,
                                            save_transformed_images=False),
                        steps_per_epoch=58,
                        epochs=EPOCHS,
                        callbacks=[tensorboard_callback, reduce_learning_rate_callback],
                        validation_data=validation_data)

    model.save(f"Fish_unet_{EPOCHS}_{NUMBER_OF_TRAINING_IMAGES}.cnn",
               overwrite=True,
               include_optimizer=True)
    prediction: np.ndarray = model.predict(np.reshape(validation_data[0][0], (1, 768, 768, 3)))

    save_image(prediction[0], 1, mask=True)
