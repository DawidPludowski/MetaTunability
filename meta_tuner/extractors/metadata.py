import pandas as pd

from openml import datasets
from typing import List, Dict, Callable

meta_extractors: Dict[str, Callable[..., float]] = {}


class MetaDataExtractor:
    @classmethod
    def get_from_openml(cls, ids: int | List[int]) -> Dict[str, float]:
        """
        Download available metadata from OpenML. Filter out
        metadata that is not in meta_extractor keys.

        Args:
            ids (int | List[int]): id or ids of dataset(s).
            If list provided, metadata is returned in list with
            order as in ids list.

        Returns:
            dir[str, float]: metadata
        """
        if isinstance(ids, int):
            ids = [ids]

        dataset_list = datasets.get_datasets(
            dataset_ids=ids, download_qualities=True, download_data=False
        )

        qualities = [dataset.qualities for dataset in dataset_list]

        for quality in qualities:
            cls.__remove_not_allowed_meta(quality)

        if len(qualities) == 1:
            qualities = qualities[0]

        return qualities

    @classmethod
    def get_metadata(cls, df: pd.DataFrame) -> Dict[str, float]:
        """
        Extract metadata from pandas dataframe. Metadata
        is specified in meta_extractors dictionary.

        Args:
            df (pd.DataFrame): dataframe from which
                metadata is extracted

        Returns:
            dict[str, float]: metadata
        """
        metadata = {}

        for extractor_name, extractor in meta_extractors.items():
            metadata[extractor_name] = extractor(df)

        return metadata

    @classmethod
    def get_missing_metadata(
        cls, df: pd.DataFrame, metadata: Dict[str, float]
    ) -> Dict[str, float]:
        """_summary_

        Args:
            df (pd.DataFrame): _description_
            metadata (dir[str, float]): _description_

        Returns:
            dir[str, float]: _description_
        """
        filled_metadata = {}

        for metadata_name, metadata_value in metadata.items():
            if metadata_name in set(meta_extractors.keys()):
                if metadata_value is None:
                    extractor = meta_extractors[metadata_name]
                    extracted_metadata_value = extractor(df)
                    filled_metadata[metadata_name] = extracted_metadata_value
                else:
                    filled_metadata[metadata_name] = metadata_value

        missing_metadata = set(meta_extractors.keys()) - set(metadata.keys())

        for missing_metadata_name in missing_metadata:
            extractor = meta_extractors[missing_metadata_name]
            extracted_metadata_value = extractor(df)
            filled_metadata[missing_metadata_name] = extracted_metadata_value

        return filled_metadata

    @classmethod
    def __remove_not_allowed_meta(cls, metadata: Dict[str, float]) -> Dict[str, float]:
        allowed_qualities = set(meta_extractors.keys())
        meta_to_delete = set()
        for metadata_item in metadata.keys():
            if metadata_item not in allowed_qualities:
                meta_to_delete.add(metadata_item)

        for quality in meta_to_delete:
            metadata.pop(quality)

        return metadata
