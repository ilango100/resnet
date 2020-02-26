from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, GlobalAvgPool2D, Dense, add
from tensorflow.keras.regularizers import l2
from tensorflow.keras import Model


class Network(object):
    def __init__(self, filters, nblocks, classes, reg=None, cs=True):
        self.filters = filters
        self.nblocks = nblocks
        self.classes = classes
        self.reg = reg
        self.cs = cs

    def block(self, inputs, filters, reduce=False, reg=None, cs=True):
        raise NotImplementedError

    def __call__(self):
        # Return a keras model on call

        inputs = x = Input(shape=(None, None, 3))

        # preact has to be set by subclass
        if self.preact:
            x = Conv2D(self.filters[0], 5, 2, "same", kernel_regularizer=l2(self.reg))(x)
        else:
            x = Conv2D(self.filters[0], 5, 2, "same", kernel_regularizer=l2(self.reg))(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)

        for filt in self.filters:
            x = self.block(x, filt, reduce=True, reg=self.reg, cs=self.cs)
            for _ in range(self.nblocks-1):
                x = self.block(x, filt, reg=self.reg, cs=self.cs)

        if self.preact:
            x = BatchNormalization()(x)
            x = Activation("relu")(x)
        x = GlobalAvgPool2D()(x)
        x = Dense(self.classes, activation="softmax")(x)

        model = Model(inputs, x)
        return model


class PlainNet(Network):
    def __init__(self, filters, nblocks, classes, reg=None, cs=True):
        super().__init__(filters, nblocks, classes, reg=reg, cs=cs)
        self.preact = False

    def block(self, inputs, filters, reduce=False, reg=None, cs=True):
        x = Conv2D(filters, 3, 2 if reduce else 1, "same",
                   kernel_regularizer=l2(reg))(inputs)
        x = BatchNormalization(center=cs, scale=cs)(x)
        x = Activation("relu")(x)

        x = Conv2D(filters, 3, 1, "same", kernel_regularizer=l2(reg))(x)
        x = BatchNormalization(center=cs, scale=cs)(x)
        x = Activation("relu")(x)

        return x


class ResNet(Network):
    def __init__(self, filters, nblocks, classes, reg=None, cs=True):
        super().__init__(filters, nblocks, classes, reg=reg, cs=cs)
        self.preact = False

    def block(self, inputs, filters, reduce=False, reg=None, cs=True):
        x = Conv2D(filters, 3, 2 if reduce else 1, "same",
                   kernel_regularizer=l2(reg))(inputs)
        x = BatchNormalization(center=cs, scale=cs)(x)
        x = Activation("relu")(x)

        x = Conv2D(filters, 3, 1, "same", kernel_regularizer=l2(reg))(x)
        x = BatchNormalization(center=cs, scale=cs)(x)

        if reduce:
            inputs = Conv2D(filters, 1, 2, "same")(inputs)

        x = add([inputs, x])

        x = Activation("relu")(x)

        return x


class ResNetV2(Network):
    def __init__(self, filters, nblocks, classes, reg=None, cs=True):
        super().__init__(filters, nblocks, classes, reg=reg, cs=cs)
        self.preact = True

    def block(self, inputs, filters, reduce=False, reg=None, cs=True):
        x = BatchNormalization(center=cs, scale=cs)(inputs)
        x = Activation("relu")(x)

        if reduce:
            inputs = Conv2D(filters, 1, 2, "same")(x)

        x = Conv2D(filters, 3, 2 if reduce else 1,
                   "same", kernel_regularizer=l2(reg))(x)

        x = BatchNormalization(center=cs, scale=cs)(x)
        x = Activation("relu")(x)
        x = Conv2D(filters, 3, 1, "same", kernel_regularizer=l2(reg))(x)

        x = add([inputs, x])

        return x


__all__ = ["PlainNet", "ResNet", "ResNetV2"]
