import pandas as pd
import pytest

from meta_tuner.extractors.metadata import MetaDataExtractor


def test_download_single_qualities():
    extractor = MetaDataExtractor()
    metadata = extractor.get_from_openml(3)

    assert isinstance(metadata, dict)


def test_download_multiple_qualities():
    extractor = MetaDataExtractor()
    metadata = extractor.get_from_openml([3, 31])

    assert isinstance(metadata, list)
    assert all([isinstance(item, dict) for item in metadata])


def test_get_metadata(resource_path):
    df = pd.read_csv(resource_path / "credit-g.csv")
    X, y = df.iloc[:, :-1], df.iloc[:, -1]

    extractor = MetaDataExtractor()
    metadata = extractor.get_metadata(X, y)

    assert isinstance(metadata, dict)
    for value in metadata.values():
        assert value is not None
        assert isinstance(value, (float, int))


def test_get_missing_metadata(resource_path):
    df = pd.read_csv(resource_path / "credit-g.csv")
    X, y = df.iloc[:, :-1], df.iloc[:, -1]

    meta_not_completed = {
        "NumberOfFeatures": -123,
        "NotExists": -123,
        "NumberOfInstances": None,
    }

    extractor = MetaDataExtractor()
    metadata = extractor.get_missing_metadata(X, y, meta_not_completed)

    assert isinstance(metadata, dict)
    assert metadata["NumberOfFeatures"] == -123
    assert metadata["NumberOfInstances"] is not None

    with pytest.raises(KeyError) as e:
        metadata["NotExists"]

    for value in metadata.values():
        assert value is not None
        assert isinstance(value, (float, int))


def test_create_empty_dict():
    extractor = MetaDataExtractor(load_default=False)

    assert len(list(extractor.meta_extractors.keys())) == 0


def test_add_new_extractor(resource_path):
    extractor = MetaDataExtractor(load_default=False)
    extractor.add_extractor("new_extractor", lambda X, y: 999)

    df = pd.read_csv(resource_path / "credit-g.csv")
    X, y = df.iloc[:, :-1], df.iloc[:, -1]

    metadata = extractor.get_metadata(X, y)

    assert len(list(extractor.meta_extractors.keys())) == 1
    assert hasattr(extractor.meta_extractors["new_extractor"], "__call__")
    assert metadata["new_extractor"] == 999
