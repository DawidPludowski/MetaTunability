import numpy as np

from typing import Dict, Callable
from statsmodels.tsa.stattools import acf, pacf


def NumberOfFeatures(X, y=None) -> float:
    return X.shape[1]


def NumberOfInstances(X, y=None) -> float:
    return X.shape[0]


def NumberOfNumericFeatures(X, y=None) -> float:
    return X.select_dtypes(include=np.number).shape[1]


def NumberOfBinaryFeatures(X, y=None) -> float:
    number_of_unique = np.unique(X, axis=1)
    number_of_binary = number_of_unique[number_of_unique == 2].shape[0]
    return number_of_binary


def _AutoCorrelationFeatures(X, y=None) -> float:
    autocorr = acf(np.concatenate([X, y], axis=1))
    return autocorr


def MinAutoCorrelation(X, y=None) -> float:
    pass


meta_extractors: Dict[str, Callable[..., float]] = {
    "NumberOfFeatures": NumberOfFeatures,
    "NumberOfInstances": NumberOfInstances,
    "NumberOfNumericFeatures": NumberOfNumericFeatures,
    "NumberOfBinaryFeatures": NumberOfBinaryFeatures,
}
