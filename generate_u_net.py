from keras import Sequential
from keras import layers as layers
from typing import List

BATCH_SIZE: int = 1
IMAGE_SIZE: List[int] = [768, 768, 3]


def add_downsample_layer(model: Sequential, filters: int):
    """

    :param model:
    :param filters:
    :return: model, the second cnn network
    """
    # padding "valid" means no padding
    cnn_layer_1 = layers.Conv2D(filters, kernel_size=(3, 3), strides=(1, 1),
                                padding="valid", activation="relu")
    cnn_layer_2 = layers.Conv2D(filters, kernel_size=(3, 3), strides=(1, 1),
                                padding="valid", activation="relu")
    sample_down_layer = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="valid")
    model.add(cnn_layer_1)
    model.add(cnn_layer_2)
    model.add(sample_down_layer)
    return model, cnn_layer_2


def add_expansive_layer(model: Sequential, filters_to_begin_with: int,
                        downsample_input_layer: layers.Conv2D) -> Sequential:
    # Upsample
    upsample_layer = layers.Conv2DTranspose(filters_to_begin_with, kernel_size=(2, 2))
    concatenation_layer = layers.concatenate([upsample_layer, downsample_input_layer])
    cnn_layer_1 = layers.Conv2D(filters=filters_to_begin_with / 2, kernel_size=(3, 3), activation="relu")
    cnn_layer_2 = layers.Conv2D(filters=filters_to_begin_with / 2, kernel_size=(3, 3), activation="relu")
    model.add(upsample_layer)
    model.add(concatenation_layer)
    model.add(cnn_layer_1)
    model.add(cnn_layer_2)
    return model


if __name__ == '__main__':
    model: Sequential = Sequential()
    INITIAL_SIZE = 64
    model.add(layers.InputLayer(input_shape=IMAGE_SIZE))
    model, layer_1 = add_downsample_layer(model, INITIAL_SIZE)
    model, layer_2 = add_downsample_layer(model, INITIAL_SIZE*2)

    model.add(layers.Conv2D(INITIAL_SIZE*4))
    model.add(layers.Conv2D(INITIAL_SIZE*4))

    model = add_expansive_layer(model, INITIAL_SIZE*4, layer_2)
    model = add_expansive_layer(model, INITIAL_SIZE*2, layer_1)
