"""File for Evaluate model machine learning"""

from typing import Union, List, Tuple
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    KFold,
    StratifiedKFold,
    ShuffleSplit,
    TimeSeriesSplit
)
from abc import ABC, abstractmethod


class DataSplitter:
    """
    A class for splitting data using different techniques.

    Args:
        X (array-like): The feature matrix.
        y (array-like): The target variable.

    Methods:
        train_test_split_data(test_size=0.2, random_state=None):
            Split the data using train-test split technique.
        kfold_split_data(n_splits=5):
            Split the data using K-fold cross-validation technique.
        stratified_kfold_split_data(n_splits=5):
            Split the data using stratified K-fold cross-validation technique.
        shuffle_split_data(n_splits=5, test_size=0.2, random_state=None):
            Split the data using shuffle split technique.
        time_series_split_data(n_splits=5):
            Split the data using time series split technique.

    Returns:
        The split data: X_train, X_test, y_train, y_test

    Examples:
        >>> splitter = DataSplitter(X, y)
        >>> X_train, X_test, y_train, y_test = splitter.train_test_split_data(test_size=0.2)
        >>> kf_splits = splitter.kfold_split_data(n_splits=5)
        >>> skf_splits = splitter.stratified_kfold_split_data(n_splits=5)
        >>> shuffle_splits = splitter.shuffle_split_data(n_splits=5, test_size=0.2)
        >>> tscv_splits = splitter.time_series_split_data(n_splits=5)
    """

    def __init__(self,
        X: Union[pd.DataFrame, pd.Series, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> None:
        """
        X (array-like): The feature matrix.
        y (array-like): The target variable.
        """
        self.X = X
        self.y = y

    def train_test_split_data(self, test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """
        Split the data using train-test split technique.

        Args:
            test_size (float, optional): The proportion of the data to include in the test split.
                Default is 0.2 (20%).
            random_state (int, optional): The seed used by the random number generator.
                Default is None.

        Returns:
            Tuple: The split data: X_train, X_test, y_train, y_test
        """
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        return X_train, X_test, y_train, y_test

    def kfold_split_data(self, n_splits: int = 5) -> List[Tuple]:
        """
        Split the data using K-fold cross-validation technique.

        Args:
            n_splits (int, optional): The number of folds to create. Default is 5.

        Returns:
            List[Tuple]: List of tuples containing the split data:
                X_train, X_test, y_train, y_test for each fold.
        """
        kfold = KFold(n_splits=n_splits)
        splits = []
        for train_index, test_index in kfold.split(self.X):
            X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
            y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]
            splits.append((X_train, X_test, y_train, y_test))
        return splits

    def stratified_kfold_split_data(self, n_splits: int = 5) -> List[Tuple]:
        """
        Split the data using stratified K-fold cross-validation technique.

        Args:
            n_splits (int, optional): The number of folds to create. Default is 5.

        Returns:
            List[Tuple]: List of tuples containing the split data:
                X_train, X_test, y_train, y_test for each fold.
        """
        skf = StratifiedKFold(n_splits=n_splits)
        splits = []
        for train_index, test_index in skf.split(self.X, self.y):
            X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
            y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]
            splits.append((X_train, X_test, y_train, y_test))
        return splits

    def shuffle_split_data(self,
        n_splits: int = 5, test_size: float = 0.2,
        random_state: int = 42
    ) -> List[Tuple]:
        """
        Split the data using shuffle split technique.

        Args:
            n_splits (int, optional): The number of re-shuffling and splitting iterations.
                Default is 5.
            test_size (float, optional): The proportion of the data to include in the test split.
                Default is 0.2 (20%).
            random_state (int, optional): The seed used by the random number generator.
                Default is None.

        Returns:
            List[Tuple]: List of tuples containing the split data:
                X_train, X_test, y_train, y_test for each iteration.
        """
        shflts = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
        splits = []
        for train_index, test_index in shflts.split(self.X):
            X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
            y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]
            splits.append((X_train, X_test, y_train, y_test))
        return splits

    def time_series_split_data(self, n_splits: int = 5) -> List[Tuple]:
        """
        Split the data using time series split technique.

        Args:
            n_splits (int, optional): The number of splits. Default is 5.

        Returns:
            List[Tuple]: List of tuples containing the split data:
                X_train, X_test, y_train, y_test for each split.
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        splits = []
        for train_index, test_index in tscv.split(self.X):
            X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
            y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]
            splits.append((X_train, X_test, y_train, y_test))
        return splits


class IDataPartitioner(ABC):
    """Abstract class for DataPartitioner"""
    @abstractmethod
    def partition_data(self,
        data: pd.DataFrame, **kwargs
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        pass

class DataPartitioner(IDataPartitioner):
    """
    A class for partitioning data into two parts.

    Args:
        data (array-like): The input data.

    Methods:
        partition_data(test_size=0.2, random_state=None):
            Split the data into two parts using train-test split technique.
    Returns:
        The split data: part1, part2

    Examples:
        >>> partitioner = DataPartitioner(data)
        >>> part1, part2 = partitioner.partition_data(test_size=0.2)
    """

    def __init__(self) -> None:
        """Init class DataPartitioner"""

    def partition_data(self, data: pd.DataFrame, **kwargs
        ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split the data into two parts using train-test split technique.

        Args:
            test_size (float, optional): The proportion of the data to include in the test split.
                Default is 0.2 (20%).
            random_state (int, optional): The seed used by the random number generator.
                Default is None.

        Returns:
            Tuple: The split data: train, test
        """
        train, test = train_test_split(
            data,
            **kwargs
        )
        return train, test
