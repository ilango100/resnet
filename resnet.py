import tensorflow as tf


def resnet(block, filters=[64, 128, 256], blocks=2, conv_layers=2, cs=True, classes=10) -> tf.keras.Model:
    inp = tf.keras.Input(shape=(None, None, 3))

    x = tf.keras.layers.Conv2D(
        filters[0], 5, 2, "same", activation="relu")(inp)

    for _ in range(blocks):
        x = block(x, filters[0], conv_layers, cs)

    for filt in filters[1:]:
        x = block(x, filt, conv_layers, cs, reduce=True)
        for _ in range(blocks):
            x = block(x, filt, conv_layers, cs)

    x = tf.keras.layers.GlobalAvgPool2D()(x)
    x = tf.keras.layers.Dense(classes, activation="softmax")(x)

    return tf.keras.Model(inputs=inp, outputs=x)


__all__ = ["resnet"]
