"""Class for traing a model form a yml file
"""

"""File For create a transform pipeline using Sklearn PipeLine"""

import importlib
from typing import List, Dict, Union, Tuple
from sklearn.model_selection import GridSearchCV
from src.utils.evaluate import IEvaluator, MetricRegs
from dataclasses import dataclass
from pandas import DataFrame


@dataclass
class ReportModels:
    """Class for reports"""
    name: str = ''
    model: object = object()
    report: Union[DataFrame, List[Tuple]] = None


class Train:
    """
    Class for creating a transform pipeline using Sklearn's ColumnTransformer.

    Args:
        config (List[Dict]): The configuration for the pipeline.

    Methods:
        create_pipeline: Create and return the ColumnTransformer object based on the configuration.
    """

    def __init__(self, config: List[Dict], evaluator: IEvaluator) -> None:
        """Initiate Train class

        Args:
            config (List[Dict]): Configuration
            evaluator (IEvaluator): Class Evaluator for evalaute models.
                The evaluator must be (Classfier or Regresser)
        """
        self.config = config
        self.models = self._create_models()
        self.evaluator = evaluator
        self.report: List[ReportModels] = None

    def _create_models(self):
        """
        Create and return the ColumnTransformer object based on the configuration.

            ColumnTransformer: The composed ColumnTransformer object.
        """
        models = []
        for step_config in self.config:
            step_class = step_config['model']
            step_params = step_config.get('params', {})
            step_name = step_config.get('name')

            module_name, class_name = step_class.rsplit('.', 1)
            module = importlib.import_module(module_name)
            step_class = getattr(module, class_name)

            models.append(
                (step_name, step_class(), step_params)
            )

        self.models = models
        return self.models

    def train_models(self, X_train, y_train, X_test, y_test, **kwargs
    ) -> List[ReportModels]:
        """_summary_

        Args:
            X_train (_type_): _description_
            y_train (_type_): _description_
            X_test (_type_): _description_
            y_test (_type_): _description_

        Returns:
            List[ReportModels]: _description_
        """
        self.report = []
        for name, model, params in self.models:
            cross = kwargs.get("cv") if kwargs.get("cv") else 5
            gscv = GridSearchCV(model, params, cv=cross)
            gscv.fit(X_train, y_train)
            best_params = gscv.best_params_
            model.set_params(**best_params)
            model.fit(X_train, y_train)

            #y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            params = {
                "y_true": y_test,
                "y_pred": y_test_pred,
                "num_features": X_train.shape[1]
            }

            self.report.append(
                ReportModels(
                    name = name,
                    model = model,
                    report = self.evaluator.metrics(**params)
                )
            )
        return self.report
    def get_best_model(self, metric :str) -> List[ReportModels]:
        """
        Get the best model based on the specified evaluation metric.

        Args:
            report_models (List[ReportModels]): List of ReportModels objects.
            metric (str): The evaluation metric to consider
                (e.g., 'Mean Squared Error', 'R^2 Score').

        Returns:
            List[ReportModels]: List of the best models based on the specified metric.
        """
        if metric not in MetricRegs.get_metric_list():
            supported_metrics = ', '.join(MetricRegs.get_metric_list())
            raise Exception(f"The metrics supported are: {supported_metrics}")
        best_models = []
        max_metric_value = None

        for report_model in self.report:
            for metric_name, metric_value in report_model.report:
                if metric_name == metric:
                    if max_metric_value is None or metric_value > max_metric_value:
                        max_metric_value = metric_value
                        best_models = [report_model]
                    elif metric_value == max_metric_value:
                        best_models.append(report_model)
                    break

        return best_models
