import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Conv2D, Activation, add


def plain(inputs: tf.Tensor, filters: int, reduce=False) -> tf.Tensor:
    stride = 2 if reduce else 1
    x = Conv2D(filters, 3, stride, "same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(filters, 3, 1, "same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x


def plain_wocs(inputs: tf.Tensor, filters: int, reduce=False) -> tf.Tensor:
    stride = 2 if reduce else 1
    x = Conv2D(filters, 3, stride, "same")(inputs)
    x = BatchNormalization(center=False, scale=False)(x)
    x = Activation("relu")(x)

    x = Conv2D(filters, 3, 1, "same")(x)
    x = BatchNormalization(center=False, scale=False)(x)
    x = Activation("relu")(x)

    return x


def plain_prebn(inputs: tf.Tensor, filters: int, reduce=False) -> tf.Tensor:
    stride = 2 if reduce else 1
    x = BatchNormalization()(inputs)
    x = Conv2D(filters, 3, stride, "same", activation="relu")(x)

    x = BatchNormalization()(x)
    x = Conv2D(filters, 3, 1, "same", activation="relu")(x)

    return x


def plain_prebn_wocs(inputs: tf.Tensor, filters: int, reduce=False) -> tf.Tensor:
    stride = 2 if reduce else 1
    x = BatchNormalization(center=False, scale=False)(inputs)
    x = Conv2D(filters, 3, stride, "same", activation="relu")(x)

    x = BatchNormalization(center=False, scale=False)(x)
    x = Conv2D(filters, 3, 1, "same", activation="relu")(x)

    return x


def resblock(inputs: tf.Tensor, filters: int, reduce=False) -> tf.Tensor:
    stride = 2 if reduce else 1

    x = Conv2D(filters, 3, stride, "same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(filters, 3, 1, "same")(x)
    x = BatchNormalization()(x)

    if reduce:
        inputs = Conv2D(filters, 3, stride, "same")(inputs)

    x = add([inputs, x])

    x = Activation("relu")(x)

    return x


def resblock_prebn(inputs: tf.Tensor, filters: int, reduce=False) -> tf.Tensor:
    stride = 2 if reduce else 1
    x = BatchNormalization()(inputs)
    x = Conv2D(filters, 3, stride, "same", activation="relu")(x)

    x = BatchNormalization()(x)
    x = Conv2D(filters, 3, 1, "same")(x)

    if reduce:
        inputs = Conv2D(filters, 3, stride, "same")(inputs)

    x = add([inputs, x])

    x = Activation("relu")(x)

    return x


def resblock_prebn_wocs(inputs: tf.Tensor, filters: int, reduce=False) -> tf.Tensor:
    stride = 2 if reduce else 1
    x = BatchNormalization(center=False, scale=False)(inputs)
    x = Conv2D(filters, 3, stride, "same", activation="relu")(x)

    x = BatchNormalization(center=False, scale=False)(x)
    x = Conv2D(filters, 3, 1, "same")(x)

    if reduce:
        inputs = Conv2D(filters, 3, stride, "same")(inputs)

    x = add([inputs, x])

    x = Activation("relu")(x)

    return x


__all__ = ["plain", "plain_wocs", "plain_prebn", "plain_prebn_wocs",
           "resblock", "resblock_prebn", "resblock_prebn_wocs"]
