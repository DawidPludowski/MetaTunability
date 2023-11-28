import numpy as np
import pytest

from copy import deepcopy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from meta_tuner.searchers.hpo_searchers import RandomSearch
from meta_tuner.searchers.preprocessors import wrap_model_with_preprocessing
from meta_tuner.searchers.early_stopping import NoImprovementEarlyStopping
from sklearn.metrics import accuracy_score


def test_get_cv_func():
    search = RandomSearch("model_obj", "grid")
    size: int = 100
    folds = search._get_cv_indexes(size, 5)

    chosen_idx = np.empty(0)

    for fold in folds:
        chosen_idx = np.concatenate([chosen_idx, fold[0]])
        total_idx = np.concatenate([fold[0], fold[1]])

        assert np.unique(total_idx).shape[0] == size
        assert (fold[0].shape[0] + fold[1].shape[0]) // fold[0].shape[0] == 5

    assert chosen_idx.shape[0] == np.unique(chosen_idx).shape[0]
    assert len(folds) == 5


def test_override_model_hpo():
    class _Model:
        def __init__(self, a) -> None:
            self.a = a

    model = _Model(1)
    search = RandomSearch(model, "grid")

    model_new = search._override_model_hpo({"a": 2})

    assert model_new.a == 2
    assert model.a == 1
    assert search.model.a == 1


def test_random_search_search(test_datasets, logistic_grid):
    dataset = test_datasets[0]
    X, y = dataset.iloc[:, :-1], dataset.iloc[:, [-1]]
    model = LogisticRegression()
    model_wrapper = wrap_model_with_preprocessing(model)
    grid = deepcopy(logistic_grid)

    search = RandomSearch(model_wrapper, grid)

    search.search(X, y, accuracy_score, n_iter=10)

    assert len(search.search_results["mean_score"]) == 10
