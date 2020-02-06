from sklearn.preprocessing import StandardScaler
import numpy as np
from enum import Enum

DEFAULT_RANDOM_DISTRIBUTION_SIZE = 10
DEFAULT_POLYNOMIAL_DEGREE = 2
DEFAULT_GAMMA_RANGE = [-1, 1]
DEFAULT_R_RANGE = [1, 3]
DEFAULT_EXPONENT = 5
DEFAULT_SIGMOID_COEFFICIENT_RANGE = [-1, 0]


class Normalization(Enum):
    STANDARD = 1
    ABSOLUTE = 2
    NEGATIVE = 3
    NONE = 4


def normalize(x, normalization_method):
    if normalization_method == Normalization.NEGATIVE:
        return 2. * (x - np.min(x)) / np.ptp(x) - 1
    if normalization_method == Normalization.ABSOLUTE:
        return x / np.max(np.abs(x), axis=0)
    if normalization_method == Normalization.STANDARD:
        return StandardScaler().fit_transform(x)
    if normalization_method == Normalization.NONE:
        return x