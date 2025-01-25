import numpy as np
import matplotlib.pyplot as plt


def main():
    linspace = np.linspace(-5, 5, 10)
    xs, ys = np.meshgrid(linspace, linspace)
    mean = (xs + ys) / 2
    x_centered, y_centered = xs - mean, ys - mean
    plt.scatter(xs, ys)
    plt.scatter(x_centered, y_centered)
    plt.show()


if __name__ == "__main__":
    main()
