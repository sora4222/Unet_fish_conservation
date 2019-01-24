from typing import List

import keras
from keras import Sequential
from keras import layers as layers

BATCH_SIZE: int = 1
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
    sample_down_layer = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="valid",
                                         data_format="channels_last")(cnn_layer_2_norm)
    return sample_down_layer, cnn_layer_2


def add_expansive_layer(input_layer, filters_to_end_with: int,
                        downsample_input_layer) -> Sequential:
    print("filters: ", filters_to_end_with)
    # Upsample
    upsample_layer = layers.Conv2DTranspose(filters_to_end_with, kernel_size=(2, 2),
                                            data_format="channels_last")(input_layer)

    # Crop the downsample_input_layer down to concatenate with the previous
    # crop_height, crop_width = get_crop_shape(downsample_input_layer, upsample_layer)
    print("Upsample layer: ", upsample_layer)
    print("Input layer to concatenate: ", downsample_input_layer)
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
    cnn_layer_2_norm = layers.BatchNormalization()(cnn_layer_2)
    return cnn_layer_2_norm


# try: https://www.kaggle.com/keegil/keras-u-net-starter-lb-0-277
if __name__ == '__main__':
    INITIAL_SIZE = 64
    input_tensor = layers.Input(shape=IMAGE_SIZE, name="Input_tensor")

    input_normal = layers.BatchNormalization()(input_tensor)
    layer_1_downsampled, layer_1 = add_downsample_layer(input_normal, INITIAL_SIZE)
    print("layer 1: ", keras.backend.int_shape(layer_1))
    print("layer 1 downsampled:", keras.backend.int_shape(layer_1_downsampled))

    layer_2_downsampled, layer_2 = add_downsample_layer(layer_1_downsampled, INITIAL_SIZE*2)
    layer_2_downsampled_norm = layers.BatchNormalization()(layer_2_downsampled)
    print("layer 2: ", keras.backend.int_shape(layer_2))
    print("layer 2 downsampled:", keras.backend.int_shape(layer_2_downsampled))

    layer_3 = layers.Conv2D(INITIAL_SIZE*4, (3, 3), padding="same")(layer_2_downsampled_norm)
    layer_3_norm = layers.BatchNormalization()(layer_3)
    print("layer 3: ", keras.backend.int_shape(layer_2))

    layer_3 = layers.Conv2D(INITIAL_SIZE*4, (3, 3), padding="same")(layer_3_norm)
    layer_3_norm = layers.BatchNormalization()(layer_3)
    print("layer 3 norm", keras.backend.int_shape(layer_3_norm))

    layer_4_upsampled = add_expansive_layer(layer_3_norm, INITIAL_SIZE * 2, layer_2)
    layer_5_upsampled = add_expansive_layer(layer_4_upsampled, INITIAL_SIZE, layer_1)

    output_layer = layers.Conv2D(1, (1, 1), activation="sigmoid", padding="same")(layer_5_upsampled)

    model: Sequential = Sequential(layers=output_layer)
    model.compile()
    model.summary()


