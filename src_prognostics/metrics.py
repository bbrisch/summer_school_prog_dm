import numpy as np


def mae(expected: np.array, actual_RUL: np.array):
    return abs(expected - actual_RUL).mean()


def crps(prognostic_pdf: np.array, actual_RUL: np.array):
    cdf = prognostic_pdf.cumsum(1)
    l, s = cdf.shape
    axis = np.arange(s).reshape(1, -1).repeat(l, 0)
    heavyside = np.heaviside(axis - actual_RUL.reshape(-1, 1), 0)
    return ((cdf - heavyside) ** 2).sum(1).mean()


def pinaw(lower_bound: np.array, upper_bound: np.array):
    return (upper_bound - lower_bound).mean()


def picp(lower_bound: np.array, upper_bound: np.array, actual_RUL: np.array):
    coverage = np.logical_and(
        np.less_equal(lower_bound, actual_RUL), np.less_equal(actual_RUL, upper_bound)
    )
    return coverage.sum() / len(coverage)
