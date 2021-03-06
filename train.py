import argparse
import data
import network
from schedule import lr_schedule
import tensorflow as tf
from tensorflow import keras
from os.path import join

assert __name__ == "__main__", "Not intended to be imported. Please run as script."

argp = argparse.ArgumentParser()
argp.add_argument("network", type=str)
argp.add_argument("-f", "--filters", nargs="+",
                  type=int, default=[8, 16, 32])
argp.add_argument("-n", "--nblocks", type=int, default=2)
argp.add_argument("-e", "--epochs", type=int, default=300)
argp.add_argument("-d", "--dataset", default="cifar10")
argp.add_argument("-b", "--batch", type=int, default=64)
argp.add_argument("-o", "--optimizer", type=str, default="sgd")
argp.add_argument("-lr", "--learning_rate", type=float, default=0.01)
argp.add_argument("-r", "--regularization", type=float, default=0.0001)
argp.add_argument("--tpu_name", type=str, default="None")
argp.add_argument("--tpu_zone", type=str, default="None")

args = argp.parse_args()

strategy = tf.distribute.get_strategy()

if args.tpu_name != "None":
    print("TPU Configured, using TPUStrategy")
    tpu_cluster = tf.distribute.cluster_resolver.TPUClusterResolver(
        args.tpu_name, args.tpu_zone)
    tf.config.experimental_connect_to_cluster(tpu_cluster)
    tf.tpu.experimental.initialize_tpu_system(tpu_cluster)
    strategy = tf.distribute.experimental.TPUStrategy(tpu_cluster)
elif len(tf.config.experimental.list_physical_devices("GPU")) > 1:
    # Multiple GPU's in single worker
    print("Multi GPU system found, using MirroredStrategy")
    num_gpus = len(tf.config.experimental.list_physical_devices("GPU"))
    args.batch_size = num_gpus * args.batch
    strategy = tf.distribute.MirroredStrategy()


# Get network definition
assert args.network in network.__all__, f"Network {args.network} not defined. Please specify one of: {network.__all__}"
net = getattr(network, args.network)

# Get optimizer
optimizers = {
    "sgd": keras.optimizers.SGD,
    "adam": keras.optimizers.Adam
}
assert args.optimizer in optimizers, f"Optimizer {args.optimizer} not defined. Please specify one of: {list(optimizers.keys())}"
opt = optimizers[args.optimizer](args.learning_rate)

# Get data
assert args.dataset in data.__all__, f"Dataset {args.dataset} not defined. Please specify one of: {data.__all__}"
train, val, test, trsteps, valsteps, testeps, classes = getattr(
    data, args.dataset)(args.batch)
print("Dataset {} loaded".format(args.dataset))

# Run name
run_name = f"{args.network}{args.filters}x{args.nblocks}"
print("Run Name:", run_name)

# Build model
with strategy.scope():
    model = net(args.filters, args.nblocks, classes, reg=args.regularization)()

    model.compile(opt, "sparse_categorical_crossentropy", metrics=[
        keras.metrics.SparseCategoricalAccuracy("acc")
    ])

model.summary()

# Train the model
try:
    hist = model.fit(train, steps_per_epoch=trsteps, epochs=args.epochs, callbacks=[
        keras.callbacks.TerminateOnNaN(),
        keras.callbacks.LearningRateScheduler(lr_schedule),
        # keras.callbacks.ReduceLROnPlateau(patience=10, verbose=1, cooldown=2, min_lr=1e-7)
        keras.callbacks.TensorBoard(join("logs", args.dataset, run_name))
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
