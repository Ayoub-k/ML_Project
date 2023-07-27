"""Train Models"""

from src.utils.train import Train
from src.utils.configure import Config
from src.utils.injection import container
from src.data.ingestion import(
    DataIngestion, configure_dependency_injection
)


configure_dependency_injection()
ingestion = container.get(DataIngestion)
PATH="/home/ay/FullStackMLDataScience/Projects/ML_Project/data/placement.csv"
ingestion.load_data(data_source=PATH)
ingestion.split_data()
PATH = "/home/ay/FullStackMLDataScience/Projects/ML_Project/articfacts"
ingestion.store_split_data(output_dir=PATH)



params_models = Config.params_config("train_model.yml")
models = Train(params_models)

print(models)
