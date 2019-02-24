from random import uniform
from typing import Tuple

import numpy as np
from scipy import ndimage


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
