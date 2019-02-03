import cv2
import numpy as np
import logging
from typing import List, Tuple, Union, Generator, Optional
from random import shuffle, uniform
from pathlib import Path
from scipy import ndimage
from functools import lru_cache

SEGMENTATION_DIRECTORY_LOCATION: str = r'E:\Downloads\fish_conservation\segmentation'
# Set this to where you want the images saved
OUTPUT_DIRECTORY_LOCATION: str = r'E:\Downloads\fish_conservation\segmentation\output\\'


def load_output_image(location: str) -> np.ndarray:
    """
    Loads the output image as a grayscale image with one
    channel.
    :param location: The absolute path to the image
    :return:
    """
    image = cv2.imread(location)
    logging.debug(f"image shape: {image.shape}")

    image_converted: np.ndarray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image_converted: np.ndarray = np.reshape(image_converted, (image_converted.shape[0], image_converted.shape[1], 1))
    logging.debug(f"image converted: {image_converted.shape}")
    return image_converted


def scale_mask(mask: np.ndarray) -> np.ndarray:
    return (mask > 30).astype(int)


def rotate(image: np.ndarray, mask: np.ndarray, deg_range: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Randomly rotates within the deg_range clockwise or counter-clockwise.
    :param image: image to rotate
    :param mask: mask to rotate the same amount
    :param deg_range:
    :return: (image rotated, mask rotated)
    """
    if deg_range is None:
        return image, mask

    rotation_amount: int = uniform(-deg_range, deg_range)
    image_new = ndimage.rotate(image, rotation_amount, reshape=False)
    mask_new = ndimage.rotate(mask, rotation_amount, reshape=False)
    return image_new, mask_new


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
    mask: np.ndarray = load_output_image(mask_location)
    return image, mask


def generate_image_data(batch_size: int,
                        location: str,
                        number_of_images: Optional[int] = None,
                        starting_with: Optional[str] = None,
                        deg_range: Optional[int] = None,
                        save_transformed_images: Optional[bool] = False):
    """
    Generates image data for a fit_generator method of Keras.
    :param deg_range: The range degrees the images can be rotated within randomly
    :param batch_size: The number of images to include in a single batch.
    :param number_of_images: This will allow you to select how many of the images to loop on.
    :param location: The location of the test folder to use, this must have Mask and Fish folders inside.
    :param starting_with: The string the images are expected to start with.
    :return: The images currently
    """

    batch_images: np.ndarray = np.zeros((batch_size, 768, 768, 3))
    batch_masks: np.ndarray = np.zeros((batch_size, 768, 768, 1))

    list_of_image_locations: List[str]
    list_of_mask_locations: List[str]
    list_of_image_locations, list_of_mask_locations = get_images_list(location, starting_with)

    if number_of_images is not None:
        assert len(list_of_image_locations) >= number_of_images, "The number of images  needs " \
                                                                 "to be equal or less than the " \
                                                                 "number in the mask directory."
    list_of_image_locations = list_of_image_locations[:number_of_images]
    list_of_mask_locations = list_of_mask_locations[:number_of_images]
    logging.info(f"list_of_image_locations: {list_of_image_locations}")
    logging.info(f"list_of_mask_locations: {list_of_mask_locations}")

    image_mask_location: List[Tuple[str, str]] = list(zip(list_of_image_locations, list_of_mask_locations))

    # Count the number of times the loop has progressed
    number_of_loops: int = 0

    while True:
        logging.debug("While loop entered.")

        # Shuffle the two image lists together.
        # Stops the order from being memorized.
        logging.debug("Shuffling data")
        shuffle(image_mask_location)

        # Marks the next image container that will be written over.
        placeholder: int = 0

        # Take each of these place them into a batch and yield them
        image_location: str
        mask_location: str
        number_of_images_processed: int = 0
        for image_location, mask_location in image_mask_location:

            # Read the image and mask in
            image: np.ndarray
            mask: np.ndarray
            image, mask = cached_read_image_mask(image_location, mask_location)

            # Only performs transformations after the first loop
            if number_of_loops > 0:
                logging.debug("Rotating data")
                image, mask = rotate(image, mask, deg_range)
                if save_transformed_images:
                    # Save the images
                    save_image(image, number_of_images_processed)
                    save_image(mask, number_of_images_processed, mask=True)
                    number_of_images_processed += 1

            mask: np.ndarray = scale_mask(mask)
            batch_images[placeholder, :, :, :] = image
            batch_masks[placeholder, :, :, :] = mask


            placeholder += 1
            logging.debug(f"placeholder: {placeholder}")

            if placeholder == batch_size:
                placeholder = 0
                yield batch_images, batch_masks

        # This is to handle the final batch that could be shorter than the others
        if placeholder > 0:
            logging.debug("A final small batch is being created.")

            batch_images_small: np.ndarray = batch_images[0:placeholder, :, :, :]
            batch_masks_small: np.ndarray = batch_masks[0:placeholder, :, :, :]

            yield batch_images_small, batch_masks_small

            # Reset for the next set of batches
            placeholder = 0
        number_of_loops += 1


def get_images_list(location_to_train: str, starting_with: Union[str, None]) -> Tuple[List[str], List[str]]:
    """
    Walks through the image and mask folders to obtain two lists of masks and images.
    :param location_to_train:
    :param starting_with:
    :return:
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
    image_location_dir: Generator
    if starting_with is not None:
        image_location_dir = image_location.glob(starting_with)
        assert image_location_dir is not None, "The image_location_dir with glob results in None"
    else:
        image_location_dir = image_location.iterdir()

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


def save_image(image: np.ndarray, number, mask: Optional[bool] = False) -> None:
    image_out: np.ndarray
    name: str
    if mask:
        image_out = image * 255.0
        name = f"{number}_mask.png"
    else:
        image_out = image
        name = f"{number}.png"

    cv2.imwrite(OUTPUT_DIRECTORY_LOCATION + name, image_out)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')
    loop = 0
    for image_batch, mask_batch in generate_image_data(10, SEGMENTATION_DIRECTORY_LOCATION + "\\train", deg_range=180):
        # assert image_batch.shape == (10, 768, 768, 3), f"{str(image.shape)}"
        # assert mask.shape == (10, 768, 768, 1)

        # Using a warning to take note of it.
        logging.info("Completed loop")

        for i in range(image_batch.shape[0]):
            save_image(image_batch[i], i + loop)
            save_image(mask_batch[i], i + loop, mask=True)
        loop +=1
        # Save the image
        # save_image(image_batch[0])
