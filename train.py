import argparse
import blocks
import data
from stack import *
from tensorflow import keras
from os.path import join, expanduser

assert __name__ == "__main__", "Not intended to be imported. Run as script."

argp = argparse.ArgumentParser()
argp.add_argument("block", type=str)
argp.add_argument("-f", "--filters", nargs="+",
                  type=int, default=[32, 64, 128])
argp.add_argument("-n", "--nblocks", type=int, default=2)
argp.add_argument("-e", "--epochs", type=int, default=500)
argp.add_argument("-d", "--dataset", default="cifar10")
argp.add_argument("-p", "--path",
                  default=join(expanduser("~"), "tensorflow_datasets"))
argp.add_argument("-b", "--batch", type=int, default=1024)
argp.add_argument("-o", "--optimizer", type=str, default="sgd")
argp.add_argument("-lr", "--learning_rate", type=float, default=0.1)
argp.add_argument("-r", "--regularization", type=float, default=0.0001)

args = argp.parse_args()


# Get block definition
assert args.block in blocks.__all__, "Block {} not defined. Please specify one of following: {}".format(
    args.block, blocks.__all__)
block = getattr(blocks, args.block)

# Get optimizer
optimizers = {
    "sgd": keras.optimizers.SGD,
    "adam": keras.optimizers.Adam
}
assert args.optimizer in optimizers, "Optimizer {} not defined. Please specify one of following: {}".format(
    args.optimizer, list(optimizers.keys()))
opt = optimizers[args.optimizer](args.learning_rate)

# Get data
assert args.dataset in data.__all__, "Dataset {} not defined. Please specify one of following: {}".format(
    args.dataset, data.__all__)
train, val, test, trsteps, valsteps, testeps, classes = getattr(
    data, args.dataset)(args.path, args.batch)
print("Dataset {} loaded from {}".format(args.dataset, args.path))

# Run name
run_name = "{}{}x{}(lr{}r{})".format(args.block, args.filters,
                                     args.nblocks, args.learning_rate, args.regularization)
print("Run Name:", run_name)

# Build model
# TODO: Make network.py to build different networks
preact = "v2" in args.block
inputs = keras.Input(shape=(None, None, 3))
x = start_stack(inputs, args.filters[0],
                preact=preact, reg=args.regularization)
x = stack(x, block, args.filters, args.nblocks, reg=args.regularization)
x = end_stack(x, classes, preact=preact)

model = keras.Model(inputs, x)

# Print model summary
model.summary()

# Compile model
model.compile(opt, "sparse_categorical_crossentropy", metrics=[
    keras.metrics.SparseCategoricalAccuracy("acc")
])


def lr_schedule(epoch, lr):
    if epoch < 100:
        return 0.1
    elif epoch < 250:
        return 0.01
    elif epoch < 400:
        return 0.001
    elif epoch % 100 == 0:
        return lr/10
    else:
        return lr


# Train the model
try:
    hist = model.fit(train, steps_per_epoch=trsteps, epochs=args.epochs, callbacks=[
        keras.callbacks.TerminateOnNaN(),
        keras.callbacks.LearningRateScheduler(lr_schedule),
        # keras.callbacks.ReduceLROnPlateau(patience=10, verbose=1, cooldown=2, min_lr=1e-7)
        keras.callbacks.TensorBoard(join("logs", args.dataset, run_name),
                                    profile_batch=0)  # To avoid the bug: https://github.com/tensorflow/tensorboard/issues/2084
    ], validation_data=val, validation_steps=valsteps)
except KeyboardInterrupt:
    print("\n\nTraining stopped, Evaluating Model...")

# Evaluate the model
loss, acc = model.evaluate(test, steps=testeps)

# Write evaluation results
with open(join("logs", "results.csv"), "a") as f:
    f.write("{},{},{},{}\n".format(args.dataset,
                                   run_name.replace(",", ""), loss, acc))

# Save the model
model.save(join("models", args.dataset, run_name))

print("Test loss, accuracy:", loss, acc)
