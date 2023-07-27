"""
class for ingestion data
"""
import os
import pandas as pd
from abc import ABC, abstractmethod
from src.utils.logger import logging
from src.utils.splitter import IDataPartitioner
from src.utils.injection import container

class IDataIngestion(ABC):
    """
    Abstract class for data Ingestion

    Methods:
        load_data(self, data_format: str = 'csv', **kwargs)
        split_data(self, test_size: float = 0.2, random_state: int = 42)
    """
    @abstractmethod
    def load_data(self, data_source: str, **kwargs) -> None:
        pass

    @abstractmethod
    def split_data(self, test_size: float = 0.2, random_state: int = 42) -> None:
        pass

    @abstractmethod
    def store_data(self, output_dir: str) -> None:
        pass




class DataIngestion(IDataIngestion):
    """
    A class for data ingestion in a data science project.

    Args:
        data_source (str): The path to the data source.

    Attributes:
        data (pd.DataFrame): The loaded data.
        X_train (pd.DataFrame): The training feature matrix.
        X_test (pd.DataFrame): The testing feature matrix.
        y_train (pd.Series): The training target variable.
        y_test (pd.Series): The testing target variable.

    Methods:
        load_data(data_format): Load the data from a data source.
        split_data(test_size=0.2, random_state=None): Split the data into training and testing sets.
        store_split_data(output_dir): Store the split data in separate folders.

    Example Usage:
        >>> ingestion = DataIngestion('data.csv')
        >>> ingestion.load_data()
        >>> ingestion.split_data(test_size=0.2, random_state=42)
        >>> ingestion.store_split_data('split_data')
    """

    def __init__(self, partitioner: IDataPartitioner):
        self.partitioner = partitioner
        self.data = None
        self.data_train = None
        self.data_test = None

    def load_data(self, data_source: str, **kwargs) -> None:
        """
        Load the data from a data source.

        Args:
            data_source (str): The path of the data. Default is 'csv'.
            **kwargs: Additional keyword arguments to pass to the pandas read function.

        Raises:
            ValueError: If the data format is not supported.
        """
        _, data_format = os.path.splitext(data_source)
        if data_format == '.csv':
            self.data = pd.read_csv(data_source, **kwargs)
        elif data_format == '.json':
            self.data = pd.read_json(data_source, **kwargs)
        elif data_format == '.excel':
            self.data = pd.read_excel(data_source, **kwargs)
        else:
            raise ValueError(f"The data format '{data_format}' is not supported.")
        logging.info(f"data is loaded from {data_source}")

    def split_data(self, test_size: float = 0.2, random_state: int = 42) -> None:
        """
        Split the data into training and testing sets.

        Args:
            test_size (float, optional): The proportion of the data to include in the test split.
                Default is 0.2 (20%).
            random_state (int, optional): The seed used by the random number generator.
                Default is 42.
        """
        self.data_train, self.data_test = self.partitioner.partition_data(
            data=self.data, test_size=test_size, random_state=random_state
        )
        logging.info(
            f"data is splited X_train: {self.data_train.shape}, X_test: {self.data_test.shape}"
        )

    def store_data(self, output_dir: str) -> None:
        """
        Store the split data in separate folders.

        Args:
            output_dir (str): The directory to store the split data.
        """

        os.makedirs(output_dir, exist_ok=True)
        for data, name in zip([self.data_train, self.data_test], ['train_data', 'test_data']):
            data.to_csv(os.path.join(output_dir, name), index=False)


        logging.info("data is stored")
4
