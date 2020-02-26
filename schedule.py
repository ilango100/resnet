import tensorflow as tf
import math


def step_lr_schedule(epoch, lr):
    schedule = {
        200: 0.1,
        400: 0.05,
        450: 0.01,
        500: 0.001
    }
    for se, slr in schedule.items():
        if epoch < se:
            return slr
    if epoch % 200 == 0:
        return lr/3
    return lr


def cos_lr_schedule(epoch, lr):
    # Half cosine
    cycles = 490
    max_lr = 0.1
    min_lr = 0.001
    if epoch <= cycles:
        cosval = (math.cos((math.pi*epoch)/cycles) + 1)/2
        return cosval * (max_lr-min_lr) + min_lr
    return min_lr


def lr_schedule(epoch, lr):
    return cos_lr_schedule(epoch, lr)


if __name__ == "__main__":
    lrs = []
    lr = 0
    for i in range(1000):
        lr = lr_schedule(i, lr)
        lrs.append(lr)

    import matplotlib.pyplot as plt
    plt.plot(lrs)
    # plt.yscale("log")
    plt.show()
