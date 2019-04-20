import logging
from random import shuffle
from typing import List, Tuple, Optional

import cv2
import numpy as np

from image_genarators.utils.image_loading import COLOR_DICT
from image_genarators.utils.image_loading import cached_read_image_mask, get_images_list
from image_genarators.utils.transforms import rotate

SEGMENTATION_DIRECTORY_LOCATION: str = r'E:\Downloads\fish_conservation\segmentation'
# Set this to where you want the images saved
OUTPUT_DIRECTORY_LOCATION: str = r'E:\Downloads\fish_conservation\segmentation\output\\'
IMAGE_OUTPUT_SIZE: Tuple[int, int, int] = (768, 768, 3)


def generate_image_data(batch_size: int,
                        location: str,
                        number_of_images: Optional[int] = None,
                        glob_pattern: Optional[str] = None,
                        deg_range: Optional[int] = None,
                        save_transformed_images: Optional[bool] = False):
    """
    Generates image data for a fit_generator method of Keras. This is the main function of this module.
    :param deg_range: The range degrees the images can be rotated within randomly
    :param batch_size: The number of images to include in a single batch.
    :param number_of_images: This will allow you to select how many of the images to loop on.
    :param location: The location of the test folder to use, this must have Mask and Fish folders inside.
    :param glob_pattern: A pattern using the common linux glob pattern.
    :param save_transformed_images: Determines whether the transformed images will be saved to a
    separate folder.
    :return: The images and masks
    """
    if glob_pattern is None:
        glob_pattern = "*"
    batch_images: np.ndarray = np.zeros((batch_size, IMAGE_OUTPUT_SIZE[0], IMAGE_OUTPUT_SIZE[1], 3))
    batch_masks: np.ndarray = np.zeros((batch_size, IMAGE_OUTPUT_SIZE[0], IMAGE_OUTPUT_SIZE[1], len(COLOR_DICT)))

    # Load in all the image locations that this is going to use as a list of strings
    # noinspection PyUnusedLocal
    list_of_image_locations: List[str]
    # noinspection PyUnusedLocal
    list_of_mask_locations: List[str]
    list_of_image_locations, list_of_mask_locations = obtain_images_locations(glob_pattern, location, number_of_images)

    # Join the two lists to ensure shuffling will keep the mask with the image
    image_mask_location: List[Tuple[str, str]] = list(zip(list_of_image_locations,
                                                          list_of_mask_locations))

    # Count the number of times the loop has progressed
    number_of_loops: int = 0

    logging.debug("Entering while loop")
    while True:
        # Shuffle the two image lists together.
        # Stops the order from being memorized.
        logging.debug("Shuffling data")
        shuffle(image_mask_location)

        # Marks the next image container that will be written over.
        placeholder: int = 0

        # Take each of these place them into a batch and yield them
        # noinspection PyUnusedLocal
        image_location: str
        # noinspection PyUnusedLocal
        mask_location: str
        number_of_images_processed: int = 0
        yield from yield_images(batch_images, batch_masks, batch_size, deg_range, image_mask_location,
                                number_of_images_processed, number_of_loops, placeholder, save_transformed_images)
        # placeholder is reset at the top of the loop
        logging.debug("A loop has completed.")
        number_of_loops += 1


def yield_images(batch_images, batch_masks, batch_size, deg_range, image_mask_location, number_of_images_processed,
                 number_of_loops, placeholder, save_transformed_images):
    """
    This is an internal generator to obtain the images and yield them

    :param number_of_loops: The number of loops this generator has been  involved in
    :param placeholder: Marks the next image container of the numpy array to write over
    :param number_of_images_processed:
    :param image_mask_location:
    :param batch_masks:
    :param batch_images:
    :param deg_range: The range degrees the images can be rotated within randomly
    :param batch_size: The number of images to include in a single batch.
    :param save_transformed_images: Determines whether the transformed images will be saved to a
    separate folder.
    :return:
    """
    for image_location, mask_location in image_mask_location:

        # Read the image and mask in
        # noinspection PyUnusedLocal
        image: np.ndarray
        # noinspection PyUnusedLocal
        mask: np.ndarray
        image, mask = cached_read_image_mask(image_location, mask_location)

        # Only performs transformations after the first loop
        if number_of_loops > 0:
            logging.debug("Transforming data")
            if deg_range is not None:
                image, mask = rotate(image, mask, deg_range)
            if save_transformed_images:
                # Save the images
                save_image(image, number_of_images_processed)
                save_image(mask, number_of_images_processed, mask=True)
                number_of_images_processed += 1

        batch_images[placeholder, :, :, :] = image
        batch_masks[placeholder, :, :, :] = mask

        placeholder += 1
        logging.debug(f"placeholder: {placeholder}")

        if placeholder == batch_size:
            placeholder = 0
            yield batch_images, batch_masks
    # This is to handle the final batch that could be shorter than the others
    if placeholder > 0:
        logging.debug("A small batch is being created.")

        batch_images_small: np.ndarray = batch_images[0:placeholder, :, :, :]
        batch_masks_small: np.ndarray = batch_masks[0:placeholder, :, :, :]

        yield batch_images_small, batch_masks_small


def obtain_images_locations(glob_pattern: str, location: str, number_of_images: int) -> Tuple[List[str], List[str]]:
    list_of_image_locations, list_of_mask_locations = get_images_list(location, glob_pattern)
    # Obtain a slice of the image LOCATIONS available if a number_of_images has been specified
    if number_of_images is not None:
        assert len(list_of_image_locations) >= number_of_images, "The number of images  needs " \
                                                                 "to be equal or less than the " \
                                                                 "number in the mask directory."
    # number_of_images=None means that all of them will be returned
    list_of_image_locations = list_of_image_locations[:number_of_images]
    list_of_mask_locations = list_of_mask_locations[:number_of_images]
    logging.info(f"list_of_image_locations: {list_of_image_locations}")
    logging.info(f"list_of_mask_locations: {list_of_mask_locations}")
    return list_of_image_locations, list_of_mask_locations


def save_image(image: np.ndarray, number, mask: Optional[bool] = False) -> None:
    # TODO: Deal with the new mask arrays
    # noinspection PyUnusedLocal
    name: str
    if mask:
        name = f"{number}_mask.png"
    else:
        name = f"{number}.png"

    logging.debug(f"image numbers to save: \n{image}")

    cv2.imwrite(OUTPUT_DIRECTORY_LOCATION + name, image)
    logging.info(f"Wrote image to {OUTPUT_DIRECTORY_LOCATION + name}")


def save_image_location(location: str = ""):
    def save_image_inner(image: np.ndarray, number, mask: Optional[bool] = False) -> None:
        # TODO: Deal with the new mask

        # noinspection PyUnusedLocal
        logging.info(f"Image size before processing {image.shape}")
        image_to_write: np.ndarray = np.zeros(image.shape)
        name: str
        if mask:
            name = f"{number}_mask.png"
            logging.debug(f"Sum: {np.sum(image)}")
            image_to_write[image > 0.8] = 255
        else:
            name = f"{number}.png"
            image_to_write = image

        logging.debug(f"image numbers to save: \n{image_to_write}")

        cv2.imwrite(location + name, image_to_write)
        logging.info(f"Wrote image to {location + name}")

    return save_image_inner
