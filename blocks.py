from tensorflow import Tensor
from tensorflow.keras.layers import BatchNormalization, Conv2D, Activation, add


def plain(inputs: Tensor, filters: int, reduce=False, cs=True) -> Tensor:
    x = Conv2D(filters, 3, 2 if reduce else 1, "same")(inputs)
    x = BatchNormalization(center=cs, scale=cs)(x)
    x = Activation("relu")(x)

    x = Conv2D(filters, 3, 1, "same")(x)
    x = BatchNormalization(center=cs, scale=cs)(x)
    x = Activation("relu")(x)

    return x


def plain_prebn(inputs: Tensor, filters: int, reduce=False, cs=True) -> Tensor:
    x = BatchNormalization(center=cs, scale=cs)(inputs)
    x = Conv2D(filters, 3, 2 if reduce else 1, "same", activation="relu")(x)

    x = BatchNormalization(center=cs, scale=cs)(x)
    x = Conv2D(filters, 3, 1, "same", activation="relu")(x)

    return x


def resblock(inputs: Tensor, filters: int, reduce=False, cs=True) -> Tensor:

    x = Conv2D(filters, 3, 2 if reduce else 1, "same")(inputs)
    x = BatchNormalization(center=cs, scale=cs)(x)
    x = Activation("relu")(x)

    x = Conv2D(filters, 3, 1, "same")(x)
    x = BatchNormalization(center=cs, scale=cs)(x)

    if reduce:
        inputs = Conv2D(filters, 1, 2, "same")(inputs)

    x = add([inputs, x])

    x = Activation("relu")(x)

    return x


def resblock_prebn(inputs: Tensor, filters: int, reduce=False, cs=True) -> Tensor:

    x = BatchNormalization(center=cs, scale=cs)(inputs)
    x = Conv2D(filters, 3, 2 if reduce else 1, "same", activation="relu")(x)

    x = BatchNormalization(center=cs, scale=cs)(x)
    x = Conv2D(filters, 3, 1, "same")(x)

    if reduce:
        inputs = Conv2D(filters, 1, 2, "same")(inputs)

    x = add([inputs, x])

    x = Activation("relu")(x)

    return x


__all__ = ["plain",  "plain_prebn", "resblock", "resblock_prebn"]
