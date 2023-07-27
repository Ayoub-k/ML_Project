"""Test
"""

# from src.data.ingestion import DataIngestion, configure_dependency_injection
from src.utils.injection import container
from src.utils.utils import PickleObject

import yaml
from sklearn.pipeline import Pipeline
import importlib
from sklearn.compose import ColumnTransformer
from src.utils.transformer import SklearnPipelineCreator
 
def create_sklearn_pipeline(config):
    column_transformers = []
    previous_columns = None
    count = 1
    for step_config in config:
        step_class = step_config['step']
        step_params = step_config.get('params', {})
        step_columns = step_config.get('columns', previous_columns)
        step_name = step_config.get('name')

        module_name, class_name = step_class.rsplit('.', 1)
        module = importlib.import_module(module_name)
        step_class = getattr(module, class_name)
        print(step_params, step_class)
        column_transformers.append((step_name, step_class(**step_params), step_columns))

        count+=1

    column_transformer = ColumnTransformer(transformers=column_transformers)
    return column_transformer

def load_pipeline_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def create_object(config):
    """
    Create and return the ColumnTransformer object based on the configuration.

    Returns:
        ColumnTransformer: The composed ColumnTransformer object.
    """
    models = []
    for step_config in config:
        step_class = step_config['model']
        step_params = step_config.get('params', {})
        step_name = step_config.get('name')

        module_name, class_name = step_class.rsplit('.', 1)
        module = importlib.import_module(module_name)
        step_class = getattr(module, class_name)

        models.append(
            (step_name, step_class(), step_params)
        )

    return models


if __name__ == '__main__':
    # configure_dependency_injection()
    # ingestion = container.get(DataIngestion)
    # ingestion.load_data(
    #     data_source="/home/ay/FullStackMLDataScience/Projects/ML_Project/data/placement.csv"
    # )
    # ingestion.split_data()
    # ingestion.store_split_data(
    #     output_dir="/home/ay/FullStackMLDataScience/Projects/ML_Project/articfacts"
    # )

  

    # Specify the path to the YAML file
    yaml_file_path = 'config/pipeline_transfrom.yml'

    # Load the pipeline configuration from the YAML file
    pipeline_config = load_pipeline_config(yaml_file_path)

    # Create the sklearn pipeline
    pipeline = create_sklearn_pipeline(pipeline_config)
    PickleObject.serialized(pipeline, "articfacts/models/transform_pipeline.pkl")
    # Use the pipeline for training, testing, or prediction
    # ...
    print(pipeline)

    yaml_file_path = 'config/train_model.yml'

    # Load the pipeline configuration from the YAML file
    pipeline_config = load_pipeline_config(yaml_file_path)

    print(create_object(pipeline_config))
    from src.utils.configure import Config
    print(Config.params_config())
