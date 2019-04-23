import numpy as np
from pytest import approx
from tensorflow.python import keras

from model.utils.loss_function import dice_coef

tensor_gen = lambda ls: keras.backend.variable(np.array(ls))
evaluate_tf = keras.backend.eval


def test_dice_coefficient_single_class():
    """
    This test should come back with the exact dice_loss
    results as before
    """
    smooth = 1e-12

    zeros = np.zeros((2, 2, 1))
    zeros_tf = keras.backend.variable(zeros)
    result_zeros = dice_coef(smooth)(zeros_tf, zeros_tf)

    ones_k = keras.backend.variable(np.ones((2, 2, 1)))
    result_zeros_ones = dice_coef(smooth)(zeros_tf, ones_k)

    assert evaluate_tf(result_zeros) == approx(1)
    assert evaluate_tf(result_zeros_ones) == approx(0)


def test_dice_coefficient_two_classes_all_background():
    """
    This will be the test of two separate classes that will

    """
    smooth = 1e-12
    all_background = tensor_gen([[[1, 0], [1, 0]]])
    all_fish = tensor_gen([[[0, 1], [0, 1]]])
    half_background_half_fish = tensor_gen([
        [[0, 1], [1, 0]],
        [[0, 1], [1, 0]]
    ])

    result_background = dice_coef(smooth)(all_background, all_background)

    assert evaluate_tf(result_background) == approx(1)


def test_dice_coefficient_two_classes_all_fish():
    """
    This will be the test of two separate classes that will
    """
    smooth = 1e-12
    all_fish = tensor_gen([[[0, 1], [0, 1]]])

    result_background = dice_coef(smooth)(all_fish, all_fish)

    assert evaluate_tf(result_background) == approx(1)


def test_dice_coefficient_two_classes_pred_no_fish_true_all_fish():
    smooth = 1e-12

    all_fish = tensor_gen([[[0, 1], [0, 1]]])

    result_background = dice_coef(smooth)(all_fish, all_fish)

    assert evaluate_tf(result_background) == approx(1)


def test_dice_coefficient_two_classes_half_and_half_matching():
    """
    This will be the test of two separate classes that will
    """
    smooth = 1e-12
    half_background_half_fish = tensor_gen([
        [[0, 1], [1, 0]],
        [[0, 1], [1, 0]]
    ])

    result_background = dice_coef(smooth)(half_background_half_fish, half_background_half_fish)

    evaluate_tf = keras.backend.eval
    assert evaluate_tf(result_background) == approx(1)
