from functools import lru_cache
from pathlib import Path
from typing import Union, Tuple, List, Generator

import numpy as np
import logging
import cv2

# The classes to classify things as.
Fish: List[int] = [255, 255, 255]
Unlabeled: List[int] = [0, 0, 0]
COLOR_DICT: np.ndarray = np.array([Fish, Unlabeled])


def one_hot_encode(mask_image: np.ndarray) -> np.ndarray:
    """
    Performs one-hot encoding on a masking image
    :param mask_image: A mask image to use to obtain classes
    :return: An array reshaped to be a column vector
    """
    new_mask: np.ndarray = mask_image[:, :, 0]

    # Reshapes into a column vector
    mask_reshaped: np.ndarray = np.reshape(new_mask,
                                           newshape=(mask_image.shape[0] * mask_image.shape[1], 1))
    new_mask = mask_reshaped / 255

    # Currently only accounts for one class
    new_mask[np.where(new_mask > 0.5)] = 1
    return new_mask.astype(np.int8)


def load_mask_image(location: str) -> np.ndarray:
    """
    Loads the output mask image as a grayscale image with one hot
    encoding for each of the categories.
    :param location: The absolute path to the image
    :return: The mask with one hot encoding for the final channel
    """
    # Reads in the image
    logging.debug(f"{__name__}: loading mask image")
    image = cv2.imread(location)

    image_converted: np.ndarray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_converted: np.ndarray = np.reshape(image_converted,
                                             newshape=image.shape)
    image_one_hot_encoded: np.ndarray = one_hot_encode(image_converted)
    logging.debug(f"image converted: {image_one_hot_encoded.shape}")
    return image_one_hot_encoded


@lru_cache(maxsize=None)
def cached_read_image_mask(image_location: str, mask_location: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    A cached function to read both the image and the mask
    :param image_location: The location to load the image
    :param mask_location:  The location to load the corresponding mask
    :return: the image and the mask in that order
    """
    logging.debug("Requesting {}, and {}".format(image_location, mask_location))
    image: np.ndarray = cv2.imread(image_location)
    mask: np.ndarray = load_mask_image(mask_location)
    return image, mask


def get_images_list(location_to_train: str, glob_pattern: Union[str, None]) -> Tuple[List[str], List[str]]:
    """
    Walks through the image and mask folders to obtain two lists one of masks and one of images.
    :param location_to_train: The location to find Fish and Mask folders
    :param glob_pattern: The pattern to use in folders to obtain images
    :return: two lists of the locations of images and masks
    """
    logging.debug("get_images_list ")

    image_locations: List = []
    mask_locations: List = []

    # Walk through from the location
    location_path: Path = Path(location_to_train)
    image_location: Path = location_path.joinpath("Fish")
    mask_location: Path = location_path.joinpath("Mask")

    assert image_location.exists(), "The 'Fish' folder doesn't exists"
    assert mask_location.exists(), "The 'Mask' folder doesn't exists"

    # Check for what to start with
    # noinspection PyUnusedLocal
    image_location_dir: Generator
    if glob_pattern is not None:
        image_location_dir = image_location.glob(glob_pattern)
        assert image_location_dir is not None, "The image_location_dir with glob results in None"
    else:
        image_location_dir = image_location.iterdir()

    # noinspection PyUnusedLocal
    file_or_directory: Path
    for file_or_directory in image_location_dir:
        if not file_or_directory.is_file():
            continue
        # Check the same file exists in the "Fish" folder
        logging.debug("File found: " + file_or_directory.name)
        if mask_location.joinpath(file_or_directory.name).exists():
            logging.debug("Matching mask found: " + file_or_directory.name)
            mask_locations.append(str(mask_location.joinpath(file_or_directory.name).resolve()))
            image_locations.append(str(image_location.joinpath(file_or_directory.name).resolve()))
            logging.debug("A matching mask was found for file: " + file_or_directory.name)
        else:
            logging.debug("A matching mask was not found for file: " + file_or_directory.name)
    return image_locations, mask_locations
