
schedule = {
    200: 0.1,
    400: 0.05,
    450: 0.01,
    500: 0.001
}


def lr_schedule(e, lr):
    global schedule
    for se, slr in schedule.items():
        if e < se:
            return slr
    if e % 200 == 0:
        return lr/3
    return lr


if __name__ == "__main__":
    lrs = []
    lr = 0
    for i in range(1000):
        lr = lr_schedule(i, lr)
        lrs.append(lr)

    import matplotlib.pyplot as plt
    plt.plot(lrs)
    plt.yscale("log")
    plt.show()
