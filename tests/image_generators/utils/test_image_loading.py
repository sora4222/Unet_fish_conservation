import numpy as np

from image_genarators.utils.image_loading import one_hot_encode


def test_one_hot_encode():
    image: np.ndarray = np.array([
        [[0, 0, 0], [255, 255, 255]],
        [[255, 255, 255], [0, 0, 0]]
    ])
    expected_mask: np.ndarray = np.array([[[0], [1]], [[1], [0]]], dtype=np.float32)
    result = one_hot_encode(image)
    print(f"Here: {result}")
    assert result.shape == (2, 2, 1)
    assert np.all(expected_mask == result)
