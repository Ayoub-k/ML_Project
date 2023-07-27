"""File For create a transform pipeline using Sklearn PipeLine"""

import importlib
from typing import List, Dict
from sklearn.compose import ColumnTransformer
from src.utils.utils import PickleObject
from sklearn.pipeline import Pipeline

class SklearnPipelineCreator:
    """
    Class for creating a transform pipeline using Sklearn's ColumnTransformer.

    Args:
        config (List[Dict]): The configuration for the pipeline.

    Methods:
        create_pipeline: Create and return the ColumnTransformer object based on the configuration.
    """

    def __init__(self, config: List[Dict]):
        self.config = config
        self.transformer = self._create_pipeline()

    def _create_pipeline(self) -> ColumnTransformer:
        """
        Create and return the ColumnTransformer object based on the configuration.

        Returns:
            ColumnTransformer: The composed ColumnTransformer object.
        """
        column_transformers = []
        previous_columns = None
        pipelines = {}

        for step_config in self.config:
            step_class = step_config['step']
            step_params = step_config.get('params', {})
            step_columns = tuple(step_config.get('columns', previous_columns))
            step_name = step_config.get('name')
            module_name, class_name = step_class.rsplit('.', 1)
            module = importlib.import_module(module_name)
            step_class = getattr(module, class_name)

            if step_columns:
                if step_columns not in pipelines:
                    pipelines[step_columns] = (
                        Pipeline(steps=[(step_name, step_class(**step_params))])
                    )
                else:
                    pipelines[step_columns].steps.append((step_name, step_class(**step_params)))

            previous_columns = step_columns

        for columns, pipeline in pipelines.items():
            column_transformers.append((pipeline.steps[0][0], pipeline, list(columns)))

        self.transformer = ColumnTransformer(transformers=column_transformers)
        return self.transformer

    def load_pipeline(self, file_path: str) -> ColumnTransformer:
        """Get pipeline from .pkl file and load it to the class

        Args:
            file_path (str): file path for .pkl file

        Returns:
            ColumnTransformer: The composed ColumnTransformer object.
        """
        self.transformer = PickleObject.deserialized(file_path)
        return self.transformer

    def save_pipeline(self, file_path: str) -> None:
        """for save the the pipeline into a .pkl file

        Args:
            file_path (str): file path for file .pkl
        """
        PickleObject.serialized(
            self._create_pipeline(),
            file_path
        )
