import numpy as np
import matplotlib.pyplot as plt


def plot_rul_bounds(
    actual_RUL: np.array,
    expected: np.array,
    lower_bound: np.array,
    upper_bound: np.array,
):
    x_axis = np.arange(len(actual_RUL))
    plt.figure()
    plt.plot(x_axis, expected, label="Prognostic", color="b")
    plt.fill_between(x_axis, lower_bound, upper_bound, alpha=0.2, color="b")
    plt.plot(x_axis, actual_RUL, color="r", label="Actual RUL")
    plt.ylabel("RUL")
    plt.xlabel("Timestep")
    plt.legend()
    plt.show()


def plot_rul_bounds_multiple(actual_RUL: np.array, prediction: list):
    x_axis = np.arange(len(actual_RUL))
    plt.figure()
    plt.plot(x_axis, actual_RUL, color="r", label="Actual RUL")
    for i, prd in enumerate(prediction, 1):
        expected, lower_bound, upper_bound = prd
        plt.plot(x_axis, expected, label=f"Model {i}")
        plt.fill_between(x_axis, lower_bound, upper_bound, alpha=0.2)

    plt.ylabel("RUL")
    plt.xlabel("Timestep")
    plt.legend()
    plt.show()
