import pytest
from copy import copy

from meta_tuner.searchers.early_stopping import (
    DummyEarlyStopping,
    NoImprovementEarlyStopping,
)


def test_dummy():
    dummy = DummyEarlyStopping()
    assert not dummy.is_stop({})


def test_not_improvement_check():
    best_score = [100]
    scores_imp = [i for i in range(10)]
    scores_not_imp = [-i for i in range(10)]
    scores = copy(scores_imp) + copy(scores_not_imp)

    stopper = NoImprovementEarlyStopping(n_iteration=5, lowest_best=False)
    assert not stopper.is_stop({"mean_score": scores_imp[:6]})
    assert stopper.is_stop({"mean_score": scores_not_imp[:6]})
    assert stopper.is_stop({"mean_score": best_score + scores[5:11]})
    assert stopper.is_stop({"mean_score": best_score + scores_imp[:5]})


def test_not_improvement_not_enough_data():
    scores_imp = [i for i in range(6)]
    stopper = NoImprovementEarlyStopping(n_iteration=100, lowest_best=False)
    assert not stopper.is_stop({"mean_score": scores_imp})
