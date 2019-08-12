import argparse
import blocks
from data import get_cifar10
from network import stack
from tensorflow import keras
from utils import *
from os.path import join

assert __name__ == "__main__", "Not intended to be imported. Run as script."

argp = argparse.ArgumentParser()
argp.add_argument("block", type=str)
argp.add_argument("-f", "--filters", nargs="+",
                  type=int, default=[32, 64, 128])
argp.add_argument("-n", "--nblocks", type=int, default=2)
argp.add_argument("-c", "--conv", type=int, default=2)
argp.add_argument("-e", "--epochs", type=int, default=500)

args = argp.parse_args()


# Get block definition
assert args.block in blocks.__all__, "Block {} not defined. Please specify one of following: {}".format(
    args.block, blocks.__all__)
block = getattr(blocks, args.block)

# Get data
train, val, test, trsteps, valsteps, testeps = get_cifar10()

# Build model
model = stack(block, args.filters, args.nblocks, args.conv)

# Compile model
model.compile("adam", "sparse_categorical_crossentropy", metrics=[
    keras.metrics.SparseCategoricalAccuracy("acc")
])

# Train the model
hist = model.fit(train, steps_per_epoch=trsteps, epochs=args.epochs, callbacks=[
    keras.callbacks.EarlyStopping("val_acc", patience=25),
    keras.callbacks.TensorBoard(join("logs", "{}{}x{}x{}".format(args.block, args.filters,
                                                                 args.nblocks, args.conv)))
], validation_data=val, validation_steps=valsteps)

# Evaluate the model
loss, acc = model.evaluate(test, steps=testeps)

print("Test loss, accuracy:", loss, acc)
