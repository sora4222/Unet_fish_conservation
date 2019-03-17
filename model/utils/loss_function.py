from keras import backend as K

smooth = 1e-12


def dice_coef(y_true, y_pred):
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


def dice_coef_loss(y_true, y_pred):
    """
    Gives the dice coefficient as a loss function translated
    to have a range of [0, 1] with 0 being a perfect prediction
    :param y_true:
    :param y_pred:
    :return:
    """
    return 1-dice_coef(y_true, y_pred)


def jaccard_distance(y_true, y_pred):
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
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    print(f"intersection size: {K.get_variable_shape(intersection)}")
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    return (intersection + smooth) / (sum_ - intersection + smooth)


def jaccard_distance_loss(y_true, y_pred):
    """

    This has been shifted so it converges on 0 and is smoothed to avoid exploding or disapearing
    gradient.

    Ref: https://en.wikipedia.org/wiki/Jaccard_index

    @url: https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96
    @author: wassname
    """
    return (1 - jaccard_distance(y_true, y_pred)) * smooth
