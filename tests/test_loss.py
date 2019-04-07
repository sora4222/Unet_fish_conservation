import numpy as np
from pytest import approx
from tensorflow.python import keras

from model.utils.loss_function import dice_coef, jaccard_distance, jaccard_distance_loss


def test_dice_coef_zeros():
    """
    Tests that the dice coefficient give a zero result
    """

    smooth = 1e-12

    zeros = np.zeros((2, 2))
    zeros_tf = keras.backend.variable(zeros, dtype='float64')
    result = dice_coef(smooth)(zeros_tf, zeros_tf)

    evaluate_tf = keras.backend.eval

    assert evaluate_tf(result) == approx(1)


def test_dice_coef_perfect():
    """
    Tests that the dice coefficient as a perfect match up.
    """
    smooth = 1e-12

    ones_k = keras.backend.variable(np.ones((1, 1)))
    result_ones = dice_coef(smooth)(ones_k, ones_k)

    evaluate_tf = keras.backend.eval

    assert evaluate_tf(result_ones) == approx(1), "The value of the "

    ones_on_left = keras.backend.variable(
        [
            [1, 0],
            [1, 0]
        ]
    )

    assert evaluate_tf(dice_coef(smooth)(ones_on_left, ones_on_left)) == approx(1)


def test_jaccard_distance():
    """
    Tests some pre calculated values for the jaccard distance
    """

    smooth = 1e-12
    abs_error = 1e-7

    true_mask = keras.backend.variable(
        [
            [1, 0],
            [1, 0]
        ]
    )

    all_zeros = keras.backend.variable(
        [
            [0, 0],
            [0, 0]
        ]
    )
    ones_right_side = keras.backend.variable(
        [
            [0, 1],
            [0, 1]
        ]
    )
    identity_matrix = keras.backend.variable(
        [
            [1, 0],
            [0, 1]
        ]
    )
    all_ones = keras.backend.variable(
        [
            [1, 1],
            [1, 1]
        ]
    )

    flatten = keras.backend.flatten
    evaluate_tf = keras.backend.eval

    assert evaluate_tf(jaccard_distance(smooth)(flatten(true_mask),
                                                flatten(all_zeros))) == approx(0, abs=abs_error)

    assert evaluate_tf(jaccard_distance(smooth)(flatten(true_mask),
                                                flatten(ones_right_side))) == approx(0, abs=abs_error)

    assert evaluate_tf(jaccard_distance(smooth)(flatten(true_mask),
                                                flatten(identity_matrix))) == approx(1 / 3, abs=abs_error)

    assert evaluate_tf(jaccard_distance(smooth)(flatten(true_mask),
                                                flatten(true_mask))) == approx(1, abs=abs_error)

    assert evaluate_tf(jaccard_distance(smooth)(flatten(true_mask),
                                                flatten(all_ones))) == approx(1 / 2, abs=abs_error)


def test_jaccard_distance_loss():
    smooth_internal = 1e-12
    smooth = 1
    abs_error = 1e-7

    true_mask = keras.backend.variable(
        [
            [1, 0],
            [1, 0]
        ]
    )

    all_zeros = keras.backend.variable(
        [
            [0, 0],
            [0, 0]
        ]
    )
    ones_right_side = keras.backend.variable(
        [
            [0, 1],
            [0, 1]
        ]
    )
    identity_matrix = keras.backend.variable(
        [
            [1, 0],
            [0, 1]
        ]
    )
    all_ones = keras.backend.variable(
        [
            [1, 1],
            [1, 1]
        ]
    )

    flatten = keras.backend.flatten
    evaluate_tf = keras.backend.eval
    assert evaluate_tf(jaccard_distance_loss(smooth, smooth_internal)(flatten(true_mask),
                                                                      flatten(all_zeros))) == approx(1, abs=abs_error)

    assert evaluate_tf(jaccard_distance_loss(smooth, smooth_internal)(flatten(true_mask),
                                                                      flatten(ones_right_side))) == approx(1,
                                                                                                           abs=abs_error)

    assert evaluate_tf(jaccard_distance_loss(smooth, smooth_internal)(flatten(true_mask),
                                                                      flatten(identity_matrix))) == approx(2 / 3,
                                                                                                           abs=abs_error)

    assert evaluate_tf(jaccard_distance_loss(smooth, smooth_internal)(flatten(true_mask),
                                                                      flatten(true_mask))) == approx(0, abs=abs_error)

    assert evaluate_tf(jaccard_distance_loss(smooth, smooth_internal)(flatten(true_mask),
                                                                      flatten(all_ones))) == approx(1 / 2,
                                                                                                    abs=abs_error)
