from tensorflow import keras


def stack(block, filters=[64, 128, 256], blocks=2, conv_layers=2, cs=True, classes=10) -> keras.Model:
    inp = keras.Input(shape=(None, None, 3))

    x = keras.layers.Conv2D(filters[0], 5, 2, "same", activation="relu")(inp)

    for _ in range(blocks):
        x = block(x, filters[0], conv_layers, cs)

    for filt in filters[1:]:
        x = block(x, filt, conv_layers, cs, reduce=True)
        for _ in range(blocks):
            x = block(x, filt, conv_layers, cs)

    x = keras.layers.GlobalAvgPool2D()(x)
    x = keras.layers.Dense(classes, activation="softmax")(x)

    return keras.Model(inputs=inp, outputs=x)


__all__ = ["stack"]
