import numpy as np


def mae(expected: np.array, actual_RUL: np.array):
    return abs(expected - actual_RUL).mean()


def pinaw(lower_bound: np.array, upper_bound: np.array):
    return (upper_bound - lower_bound).mean()


def picp(lower_bound: np.array, upper_bound: np.array, actual_RUL: np.array):
    coverage = np.logical_and(
        np.less_equal(lower_bound, actual_RUL), np.less_equal(actual_RUL, upper_bound)
    )
    return coverage.sum() / len(coverage)
