import pytest

from meta_tuner.extractors.tunability import TunabilityExtractor


def test_wrong_init():
    dir_1 = {"hpo": [{"a": 1}, {"a": 2}], "mean_score": [100, 99]}
    dir_2 = {"hpo": [{"a": 1}], "mean_score": [0, 0]}
    dir_3 = {"hpo": [{"a": 1}, {"a": 2}], "mean_score": [4]}

    with pytest.raises(AssertionError):
        _ = TunabilityExtractor([dir_1, dir_2, dir_3], lowest_best=True)


def test_extract_default_hpo(search_result):
    extractor = TunabilityExtractor(search_result, lowest_best=True)
    best_hpo = extractor.extract_default_hpo()

    assert best_hpo["a"] == 2

    extractor = TunabilityExtractor(search_result, lowest_best=False)
    best_hpo = extractor.extract_default_hpo()

    assert best_hpo["a"] == 1


def test_extract_gains(search_result):
    extractor = TunabilityExtractor(search_result, lowest_best=True)
    _ = extractor.extract_default_hpo()
    gains = extractor.extract_gains()

    assert gains[0] == 0
    assert gains[1] == 10
    assert gains[2] == 0

    extractor = TunabilityExtractor(search_result, lowest_best=False)
    _ = extractor.extract_default_hpo()
    gains = extractor.extract_gains()

    assert gains[0] == 0
    assert gains[1] == 10
    assert gains[2] == 0


def test_raise_error_when_default_not_set(search_result):
    extractor = TunabilityExtractor(search_result, lowest_best=True)
    with pytest.raises(ValueError):
        _ = extractor.extract_gains()
