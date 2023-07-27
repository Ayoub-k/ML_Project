"""Evaluate file"""
from typing import Tuple, List, Union
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
from enum import Enum
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report,
    ConfusionMatrixDisplay
)
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score
)

class MetricRegs(Enum):
    """List of Metrics Regression"""
    MSE = 'MSE'
    MAE = 'MAE'
    R2S = 'R2S'
    EVS = 'EVS'
    RMSE = 'RMSE'
    AR2S = 'AR2S'

    @staticmethod
    def get_metric_list():
        """Return a list of metric values"""
        return [metric.value for metric in MetricRegs]

class IEvaluator(ABC):
    """
    Base abstract class for model evaluators.
    """

    @abstractmethod
    def print_metrics_table(self, **kwargs) -> Union[pd.DataFrame, List[Tuple]]:
        """
        Print a table of all evaluation metrics.

        Args:
            **kwargs: Additional keyword arguments for computing metrics.
        """
        pass

    @abstractmethod
    def metrics(self, **kwargs) -> Union[pd.DataFrame, List[Tuple]]:
        """
        Print a table of all evaluation metrics.

        Args:
            **kwargs: Additional keyword arguments for computing metrics.
        """
        pass

class RegressionEvaluator(IEvaluator):
    """
    A class for evaluating regression models using various metrics.

    Args:
        y_true (array-like): The true target values.
        y_pred (array-like): The predicted target values.

    Methods:
        mean_squared_error(): Compute the mean squared error.
        mean_absolute_error(): Compute the mean absolute error.
        r2_score(): Compute the coefficient of determination (R^2 score).
        explained_variance_score(): Compute the explained variance score.
        root_mean_squared_error(): Compute the root mean squared error.
        adjusted_r_squared(): Compute the adjusted R^2 score.
        print_metrics_table():
            Print a table of all evaluation metrics.

    Example Usage:
        >>> y_true = [2.5, 1.5, 3.2, 4.0, 2.8]
        >>> y_pred = [2.0, 1.8, 3.5, 3.9, 2.6]
        >>> evaluator = RegressionEvaluator(y_true, y_pred)
        >>> print("Mean Squared Error:", evaluator.mean_squared_error())
        >>> print("Mean Absolute Error:", evaluator.mean_absolute_error())
        >>> print("R^2 Score:", evaluator.r2_score())
        >>> print("Explained Variance Score:", evaluator.explained_variance_score())
        >>> print("Root Mean Squared Error:", evaluator.root_mean_squared_error())
        >>> print("Adjusted R^2 Score:", evaluator.adjusted_r_squared())
        >>> evaluator.print_metrics_table()
    """

    def __init__(self, y_true: np.ndarray=None, y_pred: np.ndarray=None):
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)

    def mean_squared_error(self) -> float:
        """
        Compute the mean squared error.

        Returns:
            Mean squared error (float): The computed mean squared error.
        """
        return mean_squared_error(self.y_true, self.y_pred)

    def mean_absolute_error(self) -> float:
        """
        Compute the mean absolute error.

        Returns:
            Mean absolute error (float): The computed mean absolute error.
        """
        return mean_absolute_error(self.y_true, self.y_pred)

    def r2_score(self) -> float:
        """
        Compute the coefficient of determination (R^2 score).

        Returns:
            R^2 score (float): The computed coefficient of determination (R^2 score).
        """
        return r2_score(self.y_true, self.y_pred)

    def explained_variance_score(self) -> float:
        """
        Compute the explained variance score.

        Returns:
            Explained variance score (float): The computed explained variance score.
        """
        return explained_variance_score(self.y_true, self.y_pred)

    def root_mean_squared_error(self) -> float:
        """
        Compute the root mean squared error.

        Returns:
            Root mean squared error (float): The computed root mean squared error.
        """
        mse = self.mean_squared_error()
        return np.sqrt(mse)

    def adjusted_r_squared(self, num_features: int=None) -> float:
        """
        Compute the adjusted R^2 score.

        Returns:
            Adjusted R^2 score (float): The computed adjusted R^2 score.
        """
        if num_features is None:
            return None
        adj_r2 = self.r2_score()
        length = len(self.y_true)
        adjusted_r2 = 1 - (1 - adj_r2) * (length - 1) / (length - num_features - 1)
        return adjusted_r2

    def print_metrics_table(self, **kwargs) -> List[Tuple]:
        """
        Print a table of all evaluation metrics.
        """
        num_features: int=kwargs.get('num_features') if kwargs.get('num_features') else None
        self.y_true = kwargs.get('y_true') if not kwargs.get('y_true').empty else self.y_true
        self.y_pred = kwargs.get('y_pred') if not kwargs.get('y_pred').empty else self.y_pred

        metrics = [
            (MetricRegs.MSE.value, self.mean_squared_error()),
            (MetricRegs.MAE.value, self.mean_absolute_error()),
            (MetricRegs.R2S.value, self.r2_score()),
            (MetricRegs.EVS.value, self.explained_variance_score()),
            (MetricRegs.RMSE.value, self.root_mean_squared_error()),
            (MetricRegs.AR2S.value, self.adjusted_r_squared(num_features)),
        ]
        headers = ["Metric", "Value"]
        print(tabulate(metrics, headers=headers, tablefmt='fancy_grid'))

    def metrics(self, **kwargs) -> List[Tuple]:
        """_summary_

        Returns:
            List[Tuple]: _description_
        """
        num_features: int=kwargs.get('num_features') if kwargs.get('num_features') else None
        y_true = np.array(kwargs.get('y_true'))
        y_pred = np.array(kwargs.get('y_pred'))

        self.y_true = y_true if y_true.size != 0 else self.y_true
        self.y_pred = y_pred if y_pred.size != 0 else self.y_pred

        metrics = [
            (MetricRegs.MSE.value, self.mean_squared_error()),
            (MetricRegs.MAE.value, self.mean_absolute_error()),
            (MetricRegs.R2S.value, self.r2_score()),
            (MetricRegs.EVS.value, self.explained_variance_score()),
            (MetricRegs.RMSE.value, self.root_mean_squared_error()),
            (MetricRegs.AR2S.value, self.adjusted_r_squared(num_features)),
        ]
        return metrics

class ClassificationEvaluator(IEvaluator):
    """
    A class for evaluating classification models using various metrics.

    Args:
        y_true (array-like): The true labels.
        y_pred (array-like): The predicted labels.

    Methods:
        accuracy(): Compute the accuracy score.
        precision(average=None): Compute the precision score.
        recall(average=None): Compute the recall score.
        f1_score(average=None): Compute the F1 score.
        confusion_matrix(): Compute the confusion matrix.
        roc_auc(average='macro'): Compute the ROC AUC score.
        print_metrics_table(average=None):
            Print a table of all evaluation metrics.
        plot_confusion_matrix(normalize=False, cmap='Blues'):
            Plot the confusion matrix.

    Example Usage:
        >>> y_true = [0, 1, 0, 1, 1]
        >>> y_pred = [0, 0, 1, 1, 1]
        >>> evaluator = ClassificationEvaluator(y_true, y_pred)
        >>> print("Accuracy:", evaluator.accuracy())
        >>> print("Precision:", evaluator.precision())
        >>> print("Recall:", evaluator.recall())
        >>> print("F1 Score:", evaluator.f1_score())
        >>> evaluator.plot_confusion_matrix()
        >>> evaluator.print_metrics_table()
    """

    def __init__(self, y_true: np.ndarray=None, y_pred: np.ndarray=None):
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)

    def accuracy(self) -> float:
        """
        Compute the accuracy score.

        Returns:
            Accuracy score (float): The computed accuracy score.
        """
        return accuracy_score(self.y_true, self.y_pred)

    def precision(self, average=None) -> float:
        """
        Compute the precision score.

        Args:
            average (str or None, optional):
                Determines the type of averaging to perform for multi-class problems.
                Possible values: None, 'micro', 'macro', 'weighted'.
                Default is None for binary classification.

        Returns:
            Precision score (float): The computed precision score.
        """
        return precision_score(self.y_true, self.y_pred, average=average)

    def recall(self, average=None) -> float:
        """
        Compute the recall score.

        Args:
            average (str or None, optional):
                Determines the type of averaging to perform for multi-class problems.
                Possible values: None, 'micro', 'macro', 'weighted'.
                Default is None for binary classification.

        Returns:
            Recall score (float): The computed recall score.
        """
        return recall_score(self.y_true, self.y_pred, average=average)

    def f1_score(self, average=None) -> float:
        """
        Compute the F1 score.

        Args:
            average (str or None, optional):
                Determines the type of averaging to perform for multi-class problems.
                Possible values: None, 'micro', 'macr   o', 'weighted'.
                Default is None for binary classification.

        Returns:
            F1 score (float): The computed F1 score.
        """
        return f1_score(self.y_true, self.y_pred, average=average)

    def confusion_matrix(self) -> np.ndarray:
        """
        Compute the confusion matrix.

        Returns:
            Confusion matrix (np.ndarray): The computed confusion matrix.
        """
        return confusion_matrix(self.y_true, self.y_pred)

    def roc_auc(self, average='macro') -> float:
        """
        Compute the Area Under the Receiver Operating Characteristic Curve (ROC AUC) score.

        Args:
            average (str, optional):
                Determines the type of averaging to perform for multi-class problems.
                Possible values: None, 'micro', 'macro', 'weighted'. Default is 'macro'.

        Returns:
            ROC AUC score (float): The computed ROC AUC score.
        """
        return roc_auc_score(self.y_true, self.y_pred, average=average)

    def print_metrics_table(self, **kwargs) -> pd.DataFrame:
        """
        Print a table of all evaluation metrics.

        Args:
            average (str or None, optional):
                Determines the type of averaging to perform for multi-class problems.
                Possible values: None, 'micro', 'macro', 'weighted'.
                Default is None for binary classification.
        """
        average= kwargs.get('average') if kwargs.get('average') else None
        y_true = np.array(kwargs.get('y_true'))
        y_pred = np.array(kwargs.get('y_pred'))

        self.y_true = y_true if y_true.size != 0 else self.y_true
        self.y_pred = y_pred if y_pred.size != 0 else self.y_pred

        metrics = {
            'Accuracy': self.accuracy(),
            'Precision': self.precision(average=average),
            'Recall': self.recall(average=average),
            'F1 Score': self.f1_score(average=average),
            'ROC AUC': self.roc_auc(average=average)
        }
        print(pd.DataFrame(metrics))


    def metrics(self,  **kwargs) -> pd.DataFrame:
        """
        Print a table of all evaluation metrics.

        Args:
            average (str or None, optional):
                Determines the type of averaging to perform for multi-class problems.
                Possible values: None, 'micro', 'macro', 'weighted'.
                Default is None for binary classification.
        return:
            retrun dict of key-value (metric and value)
        """
        average= kwargs.get('average') if kwargs.get('average') else None
        y_true = np.array(kwargs.get('y_true'))
        y_pred = np.array(kwargs.get('y_pred'))

        self.y_true = y_true if y_true.size != 0 else self.y_true
        self.y_pred = y_pred if y_pred.size != 0 else self.y_pred

        metrics = {
            'Accuracy': self.accuracy(),
            'Precision': self.precision(average=average),
            'Recall': self.recall(average=average),
            'F1 Score': self.f1_score(average=average),
            'ROC AUC': self.roc_auc(average=average)
        }
        return metrics

    def plot_confusion_matrix(self, cmap: str='Blues') -> None:
        """
        Plot the confusion matrix.

        Args:
            cmap (str, optional): The colormap to use for the plot. Default is 'Blues'.
        """
        cme = self.confusion_matrix()
        display = ConfusionMatrixDisplay(
            confusion_matrix=cme,
            display_labels=np.unique(self.y_true)
        )
        display.plot(
            include_values=True,
            cmap=cmap, ax=plt.gca(),
            xticks_rotation='horizontal'
        )
        plt.show()

    def classification_report(self) -> str:
        """
        Generate a classification report.

        Args:
            average (str or None, optional):
                Determines the type of averaging to perform for multi-class problems.
                Possible values: None, 'micro', 'macro', 'weighted'.
                Default is None for binary classification.
        """
        report = classification_report(self.y_true, self.y_pred)
        print(report)
        return report
