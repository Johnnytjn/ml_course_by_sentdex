from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')


def create_dataset(n, variance, step=2, correlation=False):
    val = 1
    ys = []
    for i in range(n):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == '+':
            val += step
        elif correlation and correlation == '-':
            val -= step
    return np.arange(n, dtype=np.float64), np.array(ys, dtype=np.float64)


def best_fit(in_xs, in_ys):
    mean_xs, mean_ys = mean(in_xs), mean(in_ys)
    slope = ((mean_xs * mean_ys) - mean(in_xs * in_ys)) / (mean_xs ** 2 - mean(in_xs ** 2))
    intercept = mean_ys - mean_xs * slope
    return slope, intercept


def squared_error(ys_in, ys_calc):
    return np.sum((ys_calc - ys_in) ** 2)


def r_coefficient(ys_in, ys_calc):
    return 1 - squared_error(ys_in, ys_calc)/squared_error(ys_in, mean(ys_in))

xs, ys = create_dataset(40, 10, 2, '+')
a, b = best_fit(xs, ys)
new_ys = a * xs + b
print('Slope is {0:.6f} and intercept is {1:.2f}'.format(a, b))
r_squared = r_coefficient(ys, new_ys)
print(r_squared)
plt.scatter(xs, ys)
plt.plot(xs, new_ys)
plt.show()


