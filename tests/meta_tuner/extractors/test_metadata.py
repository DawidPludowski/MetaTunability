import pandas as pd
import pytest

from meta_tuner.extractors.metadata import MetaDataExtractor


def test_download_single_qualities():
    metadata = MetaDataExtractor.get_from_openml(3)

    assert isinstance(metadata, dict)


def test_download_multiple_qualities():
    metadata = MetaDataExtractor.get_from_openml([3, 31])

    assert isinstance(metadata, list)
    assert all([isinstance(item, dict) for item in metadata])


def test_get_metadata(resource_path):
    df = pd.read_csv(resource_path / "credit-g.csv")
    X, y = df.iloc[:, :-1], df.iloc[:, -1]

    metadata = MetaDataExtractor.get_metadata(X, y)

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

    metadata = MetaDataExtractor.get_missing_metadata(X, y, meta_not_completed)

    assert isinstance(metadata, dict)
    assert metadata["NumberOfFeatures"] == -123
    assert metadata["NumberOfInstances"] is not None

    with pytest.raises(KeyError) as e:
        metadata["NotExists"]

    for value in metadata.values():
        assert value is not None
        assert isinstance(value, (float, int))
