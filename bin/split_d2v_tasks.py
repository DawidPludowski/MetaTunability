import pickle as pkl
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split


def yield_task(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    data_files = path.glob("*.csv")
    for data_file in data_files:
        df = pd.read_csv(data_file)
        yield df, data_file.stem


def main(d2v_data_path: str | Path = "data/warmstart_d2v_data/train") -> None:

    d2v_data_path = Path(d2v_data_path)
    (d2v_data_path / "train_split").mkdir(exist_ok=True)

    with open("data/warmstart_d2v_data/logistic_scores.pkl", "rb") as f:
        obj = pkl.load(f)
    already_calc = list(
        obj["<class 'sklearn.linear_model._logistic.LogisticRegression'>"].keys()
    )

    for df, name in yield_task(d2v_data_path):

        if name in already_calc:
            continue

        (d2v_data_path / "train_split" / name).mkdir(exist_ok=True)
        df_train, df_test = train_test_split(
            df, test_size=0.3, random_state=123, stratify=df.iloc[:, -1]
        )
        df_train.to_csv(d2v_data_path / "train_split" / name / "train.csv", index=False)
        df_test.to_csv(d2v_data_path / "train_split" / name / "test.csv", index=False)
        logger.info(f"Task {name} splitted and saved")


if __name__ == "__main__":
    main()
