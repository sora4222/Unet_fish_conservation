import logging
from typing import Union, Optional

from keras import backend as K


def dice_coef(smooth):
    logging.debug(f"smooth constant: {smooth}")

    def dice_coef_internal(y_true, y_pred):
        """
        This is the dice coefficient intersection over union implementation
        :param y_true:
        :param y_pred:
        :return:
        """
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / \
               (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    return dice_coef_internal


def dice_coef_loss(smooth: Optional[Union[float, int]] = 1,
                   smooth_internal: Optional[Union[float, int]] = 1e-10):
    logging.info(f"smooth constant: {smooth}")
    logging.info(f"smooth_internal constant: {smooth_internal}")

    def dice_coef_loss_internal(y_true, y_pred):
        """
        Gives the dice coefficient as a loss function translated
        to have a range of [0, 1] with 0 being a perfect prediction
        :param y_true:
        :param y_pred:
        :return:
        """
        print("Sends off")
        logging.info(f"y_true shape:{y_true.get_shape()}, y_pred shape {y_pred.get_shape()}")
        return (1 - dice_coef(smooth_internal)(y_true, y_pred)) * smooth

    return dice_coef_loss_internal


def jaccard_distance(smooth):
    logging.info(f"smooth variable: {smooth}")

    def jaccard_distance_internal(y_true, y_pred):
        """
        Gives the Jaccard distance values. Useful to
        unbalanced datasets.

        Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
                = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
        @author: wassname
        :param y_true: The expected values
        :param y_pred: The predicted values
        :return: A floating point number an element of [0, 1]
        """
        logging.info(f"y_true shape:{y_true.get_shape()}, y_pred shape {y_pred.get_shape()}")
        intersection = K.sum(y_true * y_pred, axis=-1)
        sum_ = K.sum(y_true + y_pred, axis=-1)
        return (intersection + smooth) / (sum_ - intersection + smooth)

    return jaccard_distance_internal


def jaccard_distance_loss(smooth: Union[int, float] = 1,
                          smooth_internal: Union[int, float] = 1e-12):
    logging.info(f"smooth variable {smooth}, smooth internal {smooth_internal}")

    def jaccard_distance_loss_internal(y_true, y_pred):
        """

        This has been shifted so it converges on 0 and is smoothed to avoid exploding or
        disappearing gradient.

        Ref: https://en.wikipedia.org/wiki/Jaccard_index

        @url: https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96
        @author: wassname
        """
        logging.info(f"y_true shape:{y_true.get_shape()}, y_pred shape {y_pred.get_shape()}")
        jaccard_dist = jaccard_distance(smooth_internal)(y_true, y_pred)
        return (1 - jaccard_dist) * smooth

    return jaccard_distance_loss_internal
