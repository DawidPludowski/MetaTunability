import numpy as np
import pandas as pd

from abc import ABC, abstractmethod
from typing import Dict, Tuple, List, Callable

from meta_tuner.searchers.search_grid import RandomGrid


class HPOSearch(ABC):
    def __init__(self, ModelCls: type, random_grid: RandomGrid) -> None:
        self.ModelCls = ModelCls
        self.random_grid = random_grid
        self.seach_results = {"scores": [], "hpo": []}

    @abstractmethod
    def search(
        X: pd.DataFrame,
        y: pd.DataFrame,
        scoring: Callable[..., float],
        n_iter: int = 100,
        cv: int = 5,
        preprocessor_X: Callable[..., np.ndarray] = None,
        preprocessor_y: Callable[..., np.ndarray] = None,
        encode_y: bool = False,
    ) -> Dict[str, any]:
        ...

    def _get_cv_indexes(
        self, size: int, cv: int
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        all_indexes = np.arange(size)
        fold_size = size // cv
        folds = []

        remain_indexes = np.copy(all_indexes)

        for _ in range(cv):
            fold = np.random.choice(remain_indexes, fold_size, replace=False)
            remain_indexes = np.setdiff1d(remain_indexes, fold)
            folds.append((fold, np.delete(all_indexes, fold)))

        return folds

    def _convert_data(self, X: any, y: any):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if isinstance(y, pd.DataFrame):
            y = y.to_numpy()

        if y.shape[1] == 1:
            y = y.ravel()

        return X, y

    def _prepare_X(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        preprocessor: Callable[..., np.ndarray] = None,
    ) -> Tuple[np.ndarray]:
        if preprocessor:
            X_train_ = preprocessor.fit_transform(X_train)
            X_test_ = preprocessor.transform(X_test)
        else:
            X_train_, X_test_ = X_train.to_numpy(), X_test.to_numpy()

        return X_train_, X_test_

    def _encode_y(self, y: pd.Series) -> pd.DataFrame:
        y = y.astype("category")
        y = pd.get_dummies(y, drop_first=True)

        return y

    def _prepare_y(
        self,
        y_train: pd.Series,
        y_test: pd.Series,
        preprocessor: Callable[..., np.ndarray] = None,
    ) -> pd.DataFrame:
        if preprocessor:
            y_train_ = preprocessor.fit_transform(y_train_)
            y_test_ = preprocessor.transform(y_test)
        else:
            y_train_, y_test_ = y_train.to_numpy(), y_test.to_numpy()

        if y_train_.shape[1] == 1:
            y_train_, y_test_ = y_train_.ravel(), y_test_.ravel()

        return y_train_, y_test_


class RandomSearch(HPOSearch):
    """
    Implementation of random search. Contains informaiton about searching process
    in dictionary `search results`. Can be run multiple times without removing
    information about previous runs.
    """

    def __init__(self, ModelCls: type, random_grid: RandomGrid) -> None:
        super().__init__(ModelCls, random_grid)

    def search(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        scoring: Callable[..., float],
        n_iter: int = 100,
        cv: int = 5,
        preprocessor_X: Callable[..., np.ndarray] = None,
        preprocessor_y: Callable[..., np.ndarray] = None,
        encode_y: bool = False,
    ) -> None:
        """
        Evaluate model on provided data with randomly
        selected hyperparameters.

        Args:
            X (pd.DataFrame): dataframe with features.
            y (pd.DataFrame): dataframe with target.
            scoring (Callable[..., float]): scoring function. Should take
                array of y_hat, y as input and return single value.
            n_iter (int, optional): Number of iteration. Defaults to 100.
            cv (int, optional): Number of cross validaiton folds. Defaults to 5.
            preprocessor_X (Callable[..., np.ndarray], optional): Preprocessing
                object. Sholud have fit(), transform() and fit_transform()
                functions. Defaults to None.
            preprocessor_y (Callable[..., np.ndarray], optional): preprocessing
                object. Sholud have fit(), transform() and fit_transform()
                functions. Defaults to None. Defaults to None.
            encode_y (bool, optional): if set as true, y will be hot-one
                encoded before all operations. Defaults to False.
        """
        if encode_y:
            y = super()._encode_y(y)
        for _ in range(n_iter):
            hpo = self.random_grid.pick()
            folds = super()._get_cv_indexes(X.shape[0], cv)
            scores = []
            for fold in folds:
                train_X, test_X = X.iloc[fold[0], :], X.iloc[fold[1], :]
                train_y, test_y = y.iloc[fold[0], :], y.iloc[fold[1], :]

                train_X, test_X = super()._prepare_X(train_X, test_X, preprocessor_X)
                train_y, test_y = super()._prepare_y(train_y, test_y, preprocessor_y)

                model = self.ModelCls(**hpo)
                model.fit(train_X, train_y)
                pred_y = model.predict(test_X)

                score = scoring(test_y, pred_y)
                scores.append(score)

            self.seach_results["scores"].append(scores)
            self.seach_results["hpo"].append(hpo)
