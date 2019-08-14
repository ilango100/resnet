import argparse
import blocks
import data
from network import stack
from tensorflow import keras
from os.path import join, expanduser

assert __name__ == "__main__", "Not intended to be imported. Run as script."

argp = argparse.ArgumentParser()
argp.add_argument("block", type=str)
argp.add_argument("-f", "--filters", nargs="+",
                  type=int, default=[32, 64, 128])
argp.add_argument("-n", "--nblocks", type=int, default=2)
argp.add_argument("-c", "--conv", type=int, default=2)
argp.add_argument("-e", "--epochs", type=int, default=500)
argp.add_argument("-d", "--dataset", default="cifar10")
argp.add_argument("-p", "--path",
                  default=join(expanduser("~"), "tensorflow_datasets"))
argp.add_argument("-b", "--batch", type=int, default=1024)

args = argp.parse_args()


# Get block definition
assert args.block in blocks.__all__, "Block {} not defined. Please specify one of following: {}".format(
    args.block, blocks.__all__)
block = getattr(blocks, args.block)

# Get data
assert args.dataset in data.__all__, "Dataset {} not defined. Please specify one of following: {}".format(
    args.dataset, data.__all__)
train, val, test, trsteps, valsteps, testeps, classes = getattr(
    data, args.dataset)(args.path, args.batch)
print("Dataset {} loaded from {}".format(args.dataset, args.path))

# Build model
model = stack(block, args.filters, args.nblocks, args.conv, classes)

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
