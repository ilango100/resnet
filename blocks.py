from tensorflow.keras.layers import BatchNormalization, Conv2D, Activation, add
from tensorflow.keras.regularizers import l2


def plain(inputs, filters, reduce=False, reg=None, cs=True):
    if reg:
        x = Conv2D(filters, 3, 2 if reduce else 1, "same",
                   kernel_regularizer=l2(reg))(inputs)
    else:
        x = Conv2D(filters, 3, 2 if reduce else 1, "same")(inputs)
    x = BatchNormalization(center=cs, scale=cs)(x)
    x = Activation("relu")(x)

    if reg:
        x = Conv2D(filters, 3, 1, "same", kernel_regularizer=l2(reg))(x)
    else:
        x = Conv2D(filters, 3, 1, "same")(x)
    x = BatchNormalization(center=cs, scale=cs)(x)
    x = Activation("relu")(x)

    return x


def plain_prebn(inputs, filters, reduce=False, reg=None, cs=True):
    x = BatchNormalization(center=cs, scale=cs)(inputs)
    if reg:
        x = Conv2D(filters, 3, 2 if reduce else 1, "same",
                   activation="relu", kernel_regularizer=l2(reg))(x)
    else:
        x = Conv2D(filters, 3, 2 if reduce else 1,
                   "same", activation="relu")(x)

    x = BatchNormalization(center=cs, scale=cs)(x)
    if reg:
        x = Conv2D(filters, 3, 1, "same", activation="relu",
                   kernel_regularizer=l2(reg))(x)
    else:
        x = Conv2D(filters, 3, 1, "same", activation="relu")(x)

    return x


def resblock(inputs, filters, reduce=False, reg=None, cs=True):

    if reg:
        x = Conv2D(filters, 3, 2 if reduce else 1, "same",
                   kernel_regularizer=l2(reg))(inputs)
    else:
        x = Conv2D(filters, 3, 2 if reduce else 1, "same")(inputs)

    x = BatchNormalization(center=cs, scale=cs)(x)
    x = Activation("relu")(x)

    if reg:
        x = Conv2D(filters, 3, 1, "same", kernel_regularizer=l2(reg))(x)
    else:
        x = Conv2D(filters, 3, 1, "same")(x)
    x = BatchNormalization(center=cs, scale=cs)(x)

    if reduce:
        inputs = Conv2D(filters, 1, 2, "same")(inputs)

    x = add([inputs, x])

    x = Activation("relu")(x)

    return x


def resblock_prebn(inputs, filters, reduce=False, reg=None, cs=True):

    x = BatchNormalization(center=cs, scale=cs)(inputs)
    if reg:
        x = Conv2D(filters, 3, 2 if reduce else 1, "same",
                   activation="relu", kernel_regularizer=l2(reg))(x)
    else:
        x = Conv2D(filters, 3, 2 if reduce else 1,
                   "same", activation="relu")(x)

    x = BatchNormalization(center=cs, scale=cs)(x)
    if reg:
        x = Conv2D(filters, 3, 1, "same", kernel_regularizer=l2(reg))(x)
    else:
        x = Conv2D(filters, 3, 1, "same")(x)

    if reduce:
        inputs = Conv2D(filters, 1, 2, "same")(inputs)

    x = add([inputs, x])

    x = Activation("relu")(x)

    return x


def resblockv2(inputs, filters, reduce=False, reg=None, cs=True):

    x = BatchNormalization(center=cs, scale=cs)(inputs)
    x = Activation("relu")(x)
    if reg:
        x = Conv2D(filters, 3, 2 if reduce else 1, "same",
                   kernel_regularizer=l2(reg))(x)
    else:
        x = Conv2D(filters, 3, 2 if reduce else 1, "same")(x)

    x = BatchNormalization(center=cs, scale=cs)(x)
    x = Activation("relu")(x)

    if reg:
        x = Conv2D(filters, 3, 1, "same", kernel_regularizer=l2(reg))(x)
    else:
        x = Conv2D(filters, 3, 1, "same")(x)

    if reduce:
        inputs = Conv2D(filters, 1, 2, "same")(inputs)

    x = add([inputs, x])

    return x


def resblockv2_prebn(inputs, filters, reduce=False, reg=None, cs=True):

    x = Activation("relu")(inputs)
    x = BatchNormalization(center=cs, scale=cs)(x)
    if reg:
        x = Conv2D(filters, 3, 2 if reduce else 1, "same",
                   activation="relu", kernel_regularizer=l2(reg))(x)
    else:
        x = Conv2D(filters, 3, 2 if reduce else 1,
                   "same", activation="relu")(x)

    x = BatchNormalization(center=cs, scale=cs)(x)
    if reg:
        x = Conv2D(filters, 3, 1, "same", kernel_regularizer=l2(reg))(x)
    else:
        x = Conv2D(filters, 3, 1, "same")(x)

    if reduce:
        inputs = Conv2D(filters, 1, 2, "same")(inputs)

    x = add([inputs, x])

    return x


__all__ = ["plain",  "plain_prebn", "resblock",
           "resblock_prebn", "resblockv2", "resblockv2_prebn"]
