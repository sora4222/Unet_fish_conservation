import numpy as np

import image_genarators.utils.image_loading as image_loading
from image_genarators.utils.image_loading import one_hot_encode


def test_one_hot_encode():
    image_loading.COLOR_DICT = [[0, 0, 0], [255, 255, 255]]
    image: np.ndarray = np.array([
        [[0, 0, 0], [255, 255, 255]],
        [[255, 255, 255], [0, 0, 0]]
    ])
    expected_mask: np.ndarray = np.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]]], dtype=np.float32)
    assert len(image_loading.COLOR_DICT) == 2
    result = one_hot_encode(image)
    print(f"Here: {result}")
    assert result.shape == (2, 2, 2)
    assert np.all(expected_mask == result)
