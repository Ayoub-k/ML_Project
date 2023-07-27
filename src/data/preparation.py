"""file for assure data preparation in model ML with pandas"""

from typing import Union, Tuple, Dict, List
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.impute import KNNImputer
from sklearn.preprocessing import FunctionTransformer, StandardScaler, MinMaxScaler, RobustScaler
from src.utils.logger import logging
from scipy.stats import boxcox
from sklearn.base import BaseEstimator, TransformerMixin
from enum import Enum
#import datawig

class ParamsConstraints(Enum):
    """Class for store data for parametres_contraints
    """
    STRATEGY_KEY: str = "strategy"
    TRANSFORMER_KEY: str = "transformer"
    STATISTICS_KEY: str = "statistics"
    PARAM_GRID_KEY: str = "param_grid"
    STRATEGY_TRANSFORMER: str = "transformer"
    STRATEGY_STATISTICS: str = "statistics"
    STRATEGY_RANDOM_FOREST: str = "random_forest"
    STRATEGY_ZSCORE: str = "zscore"
    STRATEGY_REMOVE: str = "remove"
    TRANSFORMER_LOG: str = "log"
    TRANSFORMER_SQRT: str = "sqrt"
    TRANSFORMER_BOXCOX: str = "boxcox"
    STATISTICS_MEDIAN: str = "median"
    STATISTICS_TRIMMED_MEAN: str = "trimmed_mean"
    STATISTICS_WINSORIZATION: str = "winsorization"
    PARAM_GRID_N_ESTIMATORS: str = "n_estimators"
    PARAM_GRID_MAX_DEPTH: str = "max_depth"
    PARAM_GRID_MIN_SAMPLES_SPLIT: str = "min_samples_split"

class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Class for handling outliers.

    Parameters:
        strategy : str, optional
            The strategy for handling outliers. Possible values are:
                - "transformer": Apply a transformer method.
                - "statistics": Apply a statistical method.
                - "random_forest": Apply a random forest method.
                - "zscore": Apply the Z-score method.
                - "remove": Remove outliers based on quantile range.
            Default is "transformer".

        transformer : str, optional
            The transformer method to use when strategy is "transformer". Possible values are:
                - "log": Apply the logarithmic transformation.
                - "sqrt": Apply the square root transformation.
            Default is "log".

        statistics : str, optional
            The statistical method to use when strategy is "statistics". Possible values are:
                - "median": Replace outliers with the median value.
                - "trimmed_mean": Replace outliers with the trimmed mean value.
                - "winsorization": Apply winsorization to the data.
            Default is None.

        threshold : float, optional
            The threshold value to use when strategy is "zscore".
            Values above this threshold will be considered outliers.
            Default is 0.0.

        quantile_range : tuple(float, float), optional
            The quantile range to use when strategy is "remove".
            Data points outside this range will be considered outliers
            and will be removed from the data.
            Default is (0.05, 0.95).

        param_grid : dict[str, list[Union[int, float]]] or None, optional
            The parameter grid to use when strategy is "random_forest".
                The grid should contain parameter names as keys
            and a list of possible parameter values as values.
                This grid will be used for hyperparameter tuning of the
            random forest model.
            Default is None.
    """
    _parameter_constraints: Dict[str, List[str]] = {
        ParamsConstraints.STRATEGY_KEY.value: [
            ParamsConstraints.STRATEGY_TRANSFORMER.value,
            ParamsConstraints.STRATEGY_STATISTICS.value,
            ParamsConstraints.STRATEGY_RANDOM_FOREST.value,
            ParamsConstraints.STRATEGY_ZSCORE.value,
            ParamsConstraints.STRATEGY_REMOVE.value,
        ],
        ParamsConstraints.TRANSFORMER_KEY.value: [
            ParamsConstraints.TRANSFORMER_LOG.value,
            ParamsConstraints.TRANSFORMER_SQRT.value,
            ParamsConstraints.TRANSFORMER_BOXCOX.value
        ],
        ParamsConstraints.STATISTICS_KEY.value: [
            ParamsConstraints.STATISTICS_MEDIAN.value,
            ParamsConstraints.STATISTICS_TRIMMED_MEAN.value,
            ParamsConstraints.STATISTICS_WINSORIZATION.value,
        ],
        ParamsConstraints.PARAM_GRID_KEY.value: [
            ParamsConstraints.PARAM_GRID_N_ESTIMATORS.value,
            ParamsConstraints.PARAM_GRID_MAX_DEPTH.value,
            ParamsConstraints.PARAM_GRID_MIN_SAMPLES_SPLIT.value,
        ],
    }


    def __init__(
        self,
        strategy: str = "transformer",
        transformer: str = "log",
        statistics:str = None,
        threshold:float = 0.0,
        quantile_range: Tuple[float, float] = (0.05, 0.95),
        param_grid: Union[Dict[str, List[Union[int, float]]], None] = None
    ) -> None:
        """
        Initialize the HandleOutlier class.

        Args:
            strategy (str, optional): The strategy for handling outliers. Defaults to "transformer".
            transformer (str, optional): The transformer method to use. Defaults to "log".
            statistics (str, optional): The statistical method to use. Defaults to None.
            threshold (float, optional): The threshold value for the Z-score method. Defaults to 0.0
            quantile_range (Tuple[float, float], optional): The quantile range for the remove method
                Defaults to (0.05, 0.95).
            param_grid (Union[Dict[str, List[Union[int, float]]], None], optional):
                The parameter grid for the random forest method. Defaults to None.
        """
        self.strategy = strategy
        self.transformer = transformer
        self.statistics = statistics
        self.threshold = threshold
        self.quantile_range = quantile_range
        self.param_grid = param_grid
        self._validate_parameters()


    def _validate_input(self, **kwargs):
        """For validate input"""
        # validate data we will transfrom
        data = kwargs.get("data")
        if data is None or (
            isinstance(data, (pd.DataFrame, pd.Series)) and data.empty
        )   or (
            isinstance(data, np.ndarray) and data.size == 0
        ):
            raise ValueError("Invalid data. Data is None or empty")

        # validate and check data for the method `"random_forest"`
        if self.strategy == ParamsConstraints.STRATEGY_RANDOM_FOREST.value:
            target = kwargs.get("target")
            if target is None:
                raise ValueError("Invalid target. Traget is None")
            if (isinstance(target, pd.Series) and target.empty) or (
                isinstance(data, np.ndarray) and data.size == 0
            ):
                raise ValueError("Invalid target. Traget is empty")
            if not isinstance(target, (np.ndarray, pd.Series)):
                raise ValueError("Invalid data. Target is not a 1D array or a Pandas Series.")
            param_grid = kwargs.get(param_grid)
            if param_grid is not None:
                if not isinstance(param_grid, dict):
                    raise ValueError("Invalid param_grid. param_grid must be dict")
                if set(param_grid.keys()) != \
                    set(self._parameter_constraints[ParamsConstraints.PARAM_GRID_KEY.value]):
                    valid_statist = ", ".join(
                        self._parameter_constraints[ParamsConstraints.PARAM_GRID_KEY.value]
                    )
                    raise ValueError(f"Invalid param_grid: {param_grid}. Supported {valid_statist}")
                for value in param_grid.values():
                    if not isinstance(value, (list, int)):
                        raise ValueError(
                            f"Invalid param_grid: {param_grid} type of values isn't int or list")


    def _validate_parameters(self, **kwargs):
        """Valid & check the parameters for invalid values.

        Args:
            **kwargs: Additional keyword arguments.

        Raises:
            ValueError: If any of the parameters is invalid or missing.
        """

        # Validate strategy
        if self.strategy not in self._parameter_constraints[ParamsConstraints.STRATEGY_KEY.value]:
            valid_strategies = ", "\
                .join(self._parameter_constraints[ParamsConstraints.STRATEGY_KEY.value])
            raise ValueError(f"Invalid strategy: '{self.strategy}'. Supported: {valid_strategies}")

        # Validate transformer
        if self.strategy == ParamsConstraints.TRANSFORMER_KEY.value:
            if self.transformer not in self.\
                _parameter_constraints[ParamsConstraints.TRANSFORMER_KEY.value]:
                vld_tr = ", ".join(
                    self._parameter_constraints[ParamsConstraints.TRANSFORMER_KEY.value]
                )
                raise ValueError(f"Invalid transformer: '{self.transformer}'. Supported: {vld_tr}")

        # Validate statistics
        if self.strategy == ParamsConstraints.STATISTICS_KEY.value:
            if self.statistics not in self.\
                _parameter_constraints[ParamsConstraints.STATISTICS_KEY.value]:
                vald_stst = ", "\
                    .join(self._parameter_constraints[ParamsConstraints.STATISTICS_KEY.value])
                raise ValueError(f"Invalid statistics: '{self.statistics}'. Supported: {vald_stst}")
        # validate z-score
        if self.strategy == ParamsConstraints.STRATEGY_ZSCORE.value:
            threshold = kwargs.get("threshold")
            nim_threshold = 0
            max_threshold = 10
            if not nim_threshold < threshold < max_threshold:
                raise ValueError(f"Threshold must be between {nim_threshold} and {max_threshold}.")
        # validate range quantiles
        if self.strategy == ParamsConstraints.STRATEGY_REMOVE.value:
            min_q_range = 0.0 # max_q_range
            max_q_range = 1
            lower_quantile, upper_quantile = self.quantile_range
            if not min_q_range < lower_quantile < upper_quantile < max_q_range:
                raise ValueError(f"Quantile_range must be between {min_q_range} and {max_q_range}")


    def fit(
        self, data: pd.DataFrame,
        target: pd.Series=None
    ):
        """ fit
        """
        self._validate_input(data=data, target=target)
        return self


    def transform(
        self,
        data: Union[pd.DataFrame, pd.Series],
        target: pd.Series=None
    ) -> pd.DataFrame:
        """
        Apply outlier handling methods to the input DataFrame.

        Args:
            data (pd.DataFrame): The input DataFrame.
            target (pd.Series): The input DataFrame.

        Returns:
            pd.DataFrame: The DataFrame with outliers handled.
        """
        logging.info(f"Transform data for handling outlier with {self.strategy}")
        dataframe = data.copy()
        self._validate_input(data=dataframe, target=target)
        if self.strategy == ParamsConstraints.STRATEGY_TRANSFORMER.value:
            return self._transform_with_transformer(dataframe)
        if self.strategy == ParamsConstraints.STRATEGY_STATISTICS.value:
            return self._transform_with_statistics(dataframe)
        if self.strategy == ParamsConstraints.STRATEGY_RANDOM_FOREST.value:
            return self._transform_with_random_forest(dataframe, target)
        if self.strategy == ParamsConstraints.STRATEGY_ZSCORE.value:
            return self._transform_with_zscore(dataframe)
        if self.strategy == ParamsConstraints.STRATEGY_REMOVE.value:
            return self._transform_with_remove(dataframe)
        raise ValueError("The strategy doesn't exist")

    def _transform_with_transformer(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data using a specified method.

        Args:
            data (pd.DataFrame): Data Frame.

        Returns:
            pd.DataFrame: The transformed DataFrame.
        """

        transformer = FunctionTransformer()
        if self.transformer == ParamsConstraints.TRANSFORMER_LOG.value:
            transformer.func = np.log
        elif self.transformer == ParamsConstraints.TRANSFORMER_SQRT.value:
            transformer.func = np.sqrt
        elif self.transformer == ParamsConstraints.TRANSFORMER_BOXCOX.value:
            transformer.func = boxcox
        else:
            raise ValueError("Invalid transformation method.")

        if self.transformer == ParamsConstraints.TRANSFORMER_BOXCOX.value:
            transformed_data, _ = transformer.transform(data)
        else:
            transformed_data = transformer.transform(data)

        if isinstance(data, pd.DataFrame):
            transformed_data = pd.DataFrame(transformed_data, columns=data.columns)
        if isinstance(data, pd.Series):
            transformed_data = pd.Series(
                    transformed_data, index=data.index
                ).to_frame(name=data.name)
        return transformed_data

    def _transform_with_statistics(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply robust statistical methods to handle outliers.
            - Supported: 'median', 'trimmed_mean', 'winsorization'.

        Args:
            data (pd.DataFrame): The DataFrame from which outliers will be removed.

        Returns:
            pd.DataFrame: The DataFrame with outliers handled using the specified method.
        """
        if isinstance(data, pd.DataFrame):
            if self.statistics == ParamsConstraints.STATISTICS_MEDIAN.value:
                data = data.fillna(data.median())
            elif self.statistics == ParamsConstraints.STATISTICS_TRIMMED_MEAN.value:
                quantiles = data.quantile([0.05, 0.95])
                for column in data.columns:
                    lower_quantile = quantiles.loc[0.05, column]
                    upper_quantile = quantiles.loc[0.95, column]
                    mask = (
                        (data[column] < lower_quantile) \
                            | (data[column] > upper_quantile)
                    )
                    data.loc[mask, column] = data[column].median()
            elif self.statistics == ParamsConstraints.STATISTICS_WINSORIZATION.value:
                winsorized_data = data.copy()
                quantiles = winsorized_data.quantile([0.01, 0.99])
                for column in data.columns:
                    lower_quantile = quantiles.loc[0.01, column]
                    upper_quantile = quantiles.loc[0.99, column]
                    mask = (
                        (winsorized_data[column] < lower_quantile) \
                            | (winsorized_data[column] > upper_quantile)
                    )
                    winsorized_data.loc[mask, column] = lower_quantile
                data = winsorized_data
        elif isinstance(data, pd.Series):
            if self.statistics == ParamsConstraints.STATISTICS_MEDIAN.value:
                data = data.fillna(data.median()).to_frame(name=data.name)
            elif self.statistics == ParamsConstraints.STATISTICS_TRIMMED_MEAN.value:
                lower_quantile, upper_quantile = data.quantile([0.05, 0.95])
                mask = (data < lower_quantile) | (data > upper_quantile)
                data.loc[mask] = data.median()
                data = data.to_frame(name=data.name)
            elif self.statistics == ParamsConstraints.STATISTICS_WINSORIZATION.value:
                winsorized_data = data.copy()
                lower_quantile, upper_quantile = winsorized_data.quantile([0.01, 0.99])
                mask = (winsorized_data < lower_quantile) | (winsorized_data > upper_quantile)
                winsorized_data.loc[mask] = lower_quantile
                data = winsorized_data.to_frame(name=data.name)
        else:
            raise ValueError("Invalid data type. Expected pd.DataFrame or pd.Series.")

        return data

    def _transform_with_random_forest(
        self, data: pd.DataFrame,
        target:pd.Series,
    ) -> pd.DataFrame:
        """
        Apply random forest algorithm to handle outliers.

        Args:
            data (pd.DataFrame): The DataFrame from which outliers will be removed.
            target (pd.Series): The target variable Series.

        Returns:
            pd.DataFrame: The DataFrame with outliers handled using random forest.
        """
        target_na_mask = data.isna()

        x_train = data[~target_na_mask]
        y_train = target[~target_na_mask]
        x_test = data[target_na_mask]

        rdf = RandomForestRegressor()
        self.param_grid = self.param_grid or {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, None],
            'min_samples_split': [2, 5, 10]
        }
        grid_search = GridSearchCV(rdf, self.param_grid, scoring='neg_mean_squared_error', cv=5)
        grid_search.fit(x_train, y_train)
        best_rf = grid_search.best_estimator_
        predicted_values = best_rf.predict(x_test)
        data.loc[target_na_mask, data.columns] = predicted_values

        return data

    def _transform_with_zscore(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the Z-score method to handle outliers.

        Args:
            data (pd.DataFrame): The DataFrame from which outliers will be removed.

        Returns:
            pd.DataFrame: The DataFrame with outliers handled using the Z-score method.
        """
        z_scores = (data - data.mean()) / data.std()
        outlier_mask = (z_scores.abs() > self.threshold)
        data[outlier_mask] = np.nan

        return data

    def _transform_with_remove(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Remove outliers from the DataFrame using Quantiles.

        Args:
            data (pd.DataFrame): The DataFrame from which outliers will be removed.

        Returns:
            pd.DataFrame: The DataFrame with outliers removed.
        """
        if isinstance(data, pd.DataFrame):
            lower_quantile, upper_quantile = self.quantile_range
            for column in data.select_dtypes("number").columns:
                low, high = data[column].quantile([lower_quantile, upper_quantile])
                mask_outliers = data[column].between(low, high)
                data = data[mask_outliers]
        elif isinstance(data, pd.Series):
            lower_quantile, upper_quantile = self.quantile_range
            low, high = data.quantile([lower_quantile, upper_quantile])
            mask_outliers = data.between(low, high)
            data = data[mask_outliers].to_frame(name=data.name)
        else:
            raise ValueError("Invalid data type. Expected pd.DataFrame or pd.Series.")
        return data

class MissingImputer(BaseEstimator, TransformerMixin):
    """
    Class to impute missing values in a DataFrame.

    This class provides various strategies for imputing missing values in a DataFrame,
    such as mean imputation, median imputation, mode imputation, forward fill, backward fill,
    value imputation, regression imputation, KNN imputation, and dropping columns/rows.

    Parameters:
        strategy (str): The imputation strategy to use. Available options are:
            - "mean": Impute missing values with the mean of the column.
            - "median": Impute missing values with the median of the column.
            - "most_frequent": Impute missing values with the most frequent value in the column.
            - "forward_fill": Impute missing values with the previous non-null value.
            - "backward_fill": Impute missing values with the next non-null value.
            - "value": Impute missing values with a specified value
                (provided via the 'value' parameter).
            - "regression": Impute missing values using regression imputation.
            - "knn": Impute missing values using K-Nearest Neighbors imputation.
            - "drop_columns": Drop columns with a high percentage of missing values.
            - "drop_rows": Drop rows with a threshold number of missing values.

        value (str, int, or float): The value to be used for imputing missing values
            when the strategy is set to "value". If None, an error will be raised.

        n_neighbors (int): The number of neighbors to consider for KNN imputation.
            This parameter is used when the strategy is set to "knn".
            If None, an error will be raised.

        threshold (float): The threshold for dropping columns with null values.
            Columns with a null percentage greater than or equal to this threshold will be dropped
            when the strategy is set to "drop_columns".

        thresh (int): The threshold for dropping rows with null values.
            Rows with a null count greater than or equal to this threshold will be dropped
            when the strategy is set to "drop_rows".

        param_grid (Dict[str, List[Union[int, float]]] or None): The parameter grid for grid search
            in regression imputation. This is a dictionary with hyperparameter names as keys
            and a list of values to search as the corresponding values. If None, a default
            parameter grid will be used for grid search.
    """
    class ImputationStrategy(Enum):
        """To define all startegy for impution missing values"""
        MEAN = "mean"
        MEDIAN = "median"
        MOST_FREQUENT = "most_frequent"
        FORWARD_FILL = "forward_fill"
        BACKWARD_FILL = "backward_fill"
        VALUE = "value"
        REGRESSION = "regression"
        KNN = "knn"
        DROP_COLUMNS = "drop_columns"
        DROP_ROWS = "drop_rows"

    def __init__(
        self,
        strategy: str = "mean",
        value: Union[str, int, float] = None,
        n_neighbors: int = None,
        threshold: float = None,
        thresh: int = None,
        param_grid: Dict[str, List[Union[int, float]]] = None,
    ) -> None:
        """Init class MissingImputer

        Args:
            strategy (str, optional): _description_. Defaults to "mean".
            value (Union[str, int, float], optional): _description_. Defaults to None.
            n_neighbors (int, optional): _description_. Defaults to None.
            threshold (float, optional): _description_. Defaults to None.
            thresh (int, optional): _description_. Defaults to None.
            param_grid (Dict[str, List[Union[int, float]]], optional): _description_.
        """
        self.strategy = self.ImputationStrategy(strategy)
        self.value = value
        self.n_neighbors = n_neighbors
        self.threshold = threshold
        self.thresh = thresh
        self.param_grid = param_grid
        self._validate_params()


    def _validate_params(self) -> None:
        """For validate params
        """
        # Check and valid value
        if type(self.value) not in (int, float, str) and (
            self.strategy == self.ImputationStrategy.VALUE
        ):
            raise ValueError("Invalid Value Type. suported types are (int, float, str)")

        # Check and valid n_neighbors
        if self.n_neighbors <= 0  and self.strategy == self.ImputationStrategy.KNN:
            raise ValueError(f"Invalid n_neibgbors {self.n_neighbors}. its value must be positive")

        if self.strategy == self.ImputationStrategy.DROP_COLUMNS and (
            not 0 < self.threshold < 1
        ):
            raise ValueError("Invalid threshold. value must be between 0 & 1")


    def _validate_input(self, **kwargs) -> None:
        """For check and validate inputs like data, target and others"""
        data = kwargs.get("data")
        if data is None or (
            isinstance(data, (pd.DataFrame, pd.Series)) and data.empty
        )   or (
            isinstance(data, np.ndarray) and data.size == 0
        ):
            raise ValueError("Invalid data. Data is None or empty")


    def fit(self, data: pd.DataFrame, target: pd.Series=None):
        """fit data

        Args:
            data (pd.DataFrame): _description_
            target (pd.Series, optional): _description_. Defaults to None.
        """
        self._validate_input(data=data, target=target)
        return self


    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Perform missing value imputation on the DataFrame.

        Args:
            dataframe (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: The DataFrame with missing values imputed.

        """
        self._validate_input(data=dataframe)
        if self.strategy == self.ImputationStrategy.MEAN:
            return self.mean_imputation(dataframe)
        if self.strategy == self.ImputationStrategy.MEDIAN:
            return self.median_imputation(dataframe)
        if self.strategy == self.ImputationStrategy.MOST_FREQUENT:
            return self.mode_imputation(dataframe)
        if self.strategy == self.ImputationStrategy.FORWARD_FILL:
            return self.forward_fill_imputation(dataframe)
        if self.strategy == self.ImputationStrategy.BACKWARD_FILL:
            return self.backward_fill_imputation(dataframe)
        if self.strategy == self.ImputationStrategy.VALUE:
            return self.value_imputation(dataframe)
        if self.strategy == self.ImputationStrategy.REGRESSION:
            return self.regression_imputation(dataframe)
        if self.strategy == self.ImputationStrategy.KNN:
            return self.knn_imputation(dataframe)
        if self.strategy == self.ImputationStrategy.DROP_COLUMNS:
            return self.drop_null_columns(dataframe)
        if self.strategy == self.ImputationStrategy.DROP_ROWS:
            return self.drop_null_rows(dataframe)
        raise ValueError("Invalid strategy.")

    def mean_imputation(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Perform mean imputation for missing values in the DataFrame.

        Args:
            dataframe (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: The DataFrame with missing values imputed using mean imputation.

        """
        imputer = SimpleImputer(strategy = self.strategy)
        dataframe = pd.DataFrame(imputer.fit_transform(dataframe), columns=dataframe.columns)
        return dataframe

    def median_imputation(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Perform median imputation for missing values in the DataFrame.

        Args:
            dataframe (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: The DataFrame with missing values imputed using median imputation.

        """
        imputer = SimpleImputer(strategy = self.strategy)
        dataframe = pd.DataFrame(imputer.fit_transform(dataframe), columns=dataframe.columns)
        return dataframe


    def mode_imputation(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Perform mode imputation for missing values in the DataFrame.

        Args:
            dataframe (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: The DataFrame with missing values imputed using mode imputation.

        """
        imputer = SimpleImputer(strategy = self.strategy)
        dataframe = pd.DataFrame(imputer.fit_transform(dataframe), columns=dataframe.columns)
        return dataframe


    def forward_fill_imputation(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Perform forward fill imputation for missing values in the DataFrame.

        Args:
            dataframe (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: The DataFrame with missing values imputed using forward fill imputation.

        """
        return dataframe.ffill()


    def backward_fill_imputation(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Perform backward fill imputation for missing values in the DataFrame.

        Args:
            dataframe (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: The DataFrame with missing values imputed using backward fill imputation.

        """
        return dataframe.bfill()


    def value_imputation(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Perform value imputation for missing values in the DataFrame.

        Args:
            dataframe (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: The DataFrame with missing values imputed using value imputation.

        """
        if self.value is None:
            raise ValueError("The 'value' parameter is required.")

        if not isinstance(self.value, (int, float, str)):
            raise TypeError("Invalid value type. Value should be an int, float, or str.")

        if isinstance(self.value, (int, float)):
            columns_to_impute = dataframe.select_dtypes(include=[np.number]).columns
        elif isinstance(self.value, str):
            columns_to_impute = dataframe.select_dtypes(include=[object]).columns

        dataframe[columns_to_impute] = dataframe[columns_to_impute].fillna(self.value)

        return dataframe


    def regression_imputation(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Perform regression imputation for missing values in the DataFrame.

        Args:
            dataframe (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: The DataFrame with missing values imputed using regression imputation.

        """
        try:
            target_column = dataframe.columns[dataframe.isna().any()].tolist()
            dataframe_missing = dataframe[dataframe[target_column].isnull()]
            dataframe_not_missing = dataframe[~dataframe[target_column].isnull()]

            if dataframe_missing.empty:
                logging.info("No missing values found in the target column.")
                return dataframe

            x_train = dataframe_not_missing.drop(target_column, axis=1)
            y_train = dataframe_not_missing[target_column]

            self.param_grid = self.param_grid or {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, None],
                'min_samples_split': [2, 5, 10]
            }

            rdf = RandomForestRegressor()
            grid_search = GridSearchCV(rdf, self.param_grid, scoring='neg_mean_squared_error', cv=5)
            grid_search.fit(x_train, y_train)

            best_rf = grid_search.best_estimator_

            x_missing = dataframe_missing.drop(target_column, axis=1)
            dataframe_missing_imputed = dataframe_missing.copy()
            dataframe_missing_imputed[target_column] = best_rf.predict(x_missing)

            dataframe_imputed = pd.concat([dataframe_not_missing, dataframe_missing_imputed])
            return dataframe_imputed

        except KeyError as error:
            logging.error(f"The target column '{target_column}' does not exist in the DataFrame.")
            raise Exception(error) from KeyError

        except Exception as error:
            logging.error("An error occurred during regression imputation.")
            logging.error(str(error))
            raise Exception(error)


    def knn_imputation(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Perform KNN imputation for missing values in the DataFrame.

        Args:
            dataframe (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: The DataFrame with missing values imputed using KNN imputation.

        """
        try:
            imputer = KNNImputer(n_neighbors=self.n_neighbors)
            dataframe_imputed = imputer.fit_transform(dataframe)
            dataframe_imputed = pd.DataFrame(dataframe_imputed, columns=dataframe.columns)
            return dataframe_imputed
        except Exception as e:
            logging.error("An error occurred during KNN imputation.")
            logging.error(str(e))
            return dataframe


    def drop_null_columns(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Drop columns with a greater percentage of null values than the specified threshold.

        Args:
            dataframe (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: The DataFrame with null columns dropped.

        """
        null_percentages = dataframe.isnull().mean()
        columns_to_drop = null_percentages[null_percentages >= self.threshold].index
        dataframe = dataframe.drop(columns=columns_to_drop)
        return dataframe


    def drop_null_rows(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Drop rows with missing values based on the threshold.

        Args:
            dataframe (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: The DataFrame with null rows dropped.

        """
        return dataframe.dropna(thresh=self.thresh, axis=0)

class Scaler(BaseEstimator, TransformerMixin):
    """
    Class for scaling numerical features in a dataset.

    Parameters:
        strategy (str): The scaling strategy to use. Options: 'standard', 'minmax', 'robust'.
    """
    class ScalerOptions(Enum):
        """For all options of the scaler"""
        STANDARD = "standard"
        MINMAX = "minmax"
        ROBUST = "robust"

    def __init__(self, strategy='standard'):
        self.strategy = self.ScalerOptions(strategy)
        self.scaler = None

    def fit(self, X, y=None):
        """
        Fit the scaler to the data.

        Args:
            X (array-like): The input data.
            y (array-like, optional): The target values. Defaults to None.

        Returns:
            self
        """
        self.scaler = self._get_scaler()
        self.scaler.fit(X, y)
        return self

    def transform(self, X):
        """
        Apply scaling to the data.

        Args:
            X (array-like): The input data.

        Returns:
            array-like: The scaled data.
        """
        if self.scaler is None:
            raise ValueError("Scaler has not been fitted.")

        return self.scaler.transform(X)

    def fit_transform(self, X, y=None):
        """
        Fit the scaler to the data and apply scaling.

        Args:
            X (array-like): The input data.
            y (array-like, optional): The target values. Defaults to None.

        Returns:
            array-like: The scaled data.
        """
        self.fit(X, y)
        return self.transform(X)

    def _get_scaler(self):
        """
        Get the appropriate scaler based on the selected strategy.

        Returns:
            scaler: The scaler object.
        """
        if self.strategy == self.ScalerOptions.STANDARD:
            return StandardScaler()
        if self.strategy ==  self.ScalerOptions.MINMAX:
            return MinMaxScaler()
        if self.strategy == self.ScalerOptions.ROBUST:
            return RobustScaler()
        supported = ", ".join([option.value for option in self.ScalerOptions])
        raise ValueError(f"Invalid scaling strategy. Supported: {supported}.")
