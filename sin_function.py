import math
import tensorflow as tf
import matplotlib.pyplot as plt


def sin():
    while True:
        x = tf.random.normal((32, 10), math.pi, 1.5)
        y = tf.math.sin(x)
        yield (x, y)


def build_model(layers):
    x = inputs = tf.keras.Input((10, ))

    for _ in range(layers):
        x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dense(10)(x)

    model = tf.keras.Model(inputs, x)
    model.compile("adam", "mse")
    return model


def resblock(inputs):
    x = tf.keras.layers.Dense(32, activation='relu')(inputs)
    x = tf.keras.layers.Dense(32)(x)
    x = x + inputs
    return tf.keras.layers.Activation('relu')(x)


def build_resnet_model(layers):
    x = inputs = tf.keras.Input((10, ))
    x = tf.keras.layers.Dense(32, activation='relu')(x)

    # Each resblock includes 2 layers
    for _ in range((layers-2)//2):
        x = resblock(x)

    x = tf.keras.layers.Dense(10)(x)

    model = tf.keras.Model(inputs, x)
    model.compile("adam", "mse")
    return model


dts = tf.data.Dataset.from_generator(sin, (tf.float32, tf.float32),
                                     ((32, 10), (32, 10)))

layers = 30
losses = []
print("PlainNet")
for t in range(1, layers+1):
    plain = build_model(t)
    hist = plain.fit(dts, steps_per_epoch=20, epochs=20*t, verbose=0)
    losses.append(hist.history["loss"][-1])
    print(losses[-1])

plt.plot(range(1, layers+1), losses)
plt.title("sin function with NN")
plt.xlabel("Layers")
plt.ylabel("Final loss")
plt.show()

res_losses = []
print("ResNet")
for t in range(2, layers+1, 2):
    resnet = build_resnet_model(t)
    hist = resnet.fit(dts, steps_per_epoch=20, epochs=20*t, verbose=0)
    res_losses.append(hist.history["loss"][-1])
    print(res_losses[-1])

plt.plot(range(2, layers+1, 2), res_losses)
plt.title("sin function with ResNet")
plt.xlabel("Layers")
plt.ylabel("Final loss")
plt.show()

plt.plot(range(1, layers+1), losses, label="PlainNet")
plt.plot(range(2, layers+1, 2), res_losses, label="ResNet")
plt.title("sin function with NN")
plt.xlabel("Layers")
plt.ylabel("Final loss")
plt.legend()
plt.show()

