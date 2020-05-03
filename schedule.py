import math


def step_lr_schedule(epoch, lr):
    schedule = {
        200: 0.01,
        250: 0.001,
        300: 0.0001,
    }
    for se, slr in schedule.items():
        if epoch < se:
            return slr
    return 0.0001


def cos_lr_schedule(epoch, lr):
    # Half cosine
    cycles = 490
    max_lr = 0.01
    min_lr = 0.0005
    if epoch <= cycles:
        cosval = (math.cos((math.pi*epoch)/cycles) + 1)/2
        return cosval * (max_lr-min_lr) + min_lr
    return min_lr


def lr_schedule(epoch, lr):
    return step_lr_schedule(epoch, lr)


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

