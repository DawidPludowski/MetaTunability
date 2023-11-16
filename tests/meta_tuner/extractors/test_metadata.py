from meta_tuner.extractors.metadata import MetaDataExtractor


def test_download_single_qualities():
    metadata = MetaDataExtractor.get_from_openml(3)

    assert isinstance(metadata, dict)


def test_download_multiple_qualities():
    metadata = MetaDataExtractor.get_from_openml([3, 31])

    assert isinstance(metadata, list)
    assert all([isinstance(item, dict) for item in metadata])


def test_removing_not_allowed_meta():
    raise NotImplementedError
