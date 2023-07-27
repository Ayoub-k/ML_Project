"""This file for configuration"""
from datetime import datetime
import yaml
from src.utils.paths import Paths
from src.utils.constants import PathFolder
from dotenv import load_dotenv

class Config:
    """Class for getting configuration"""

    @staticmethod
    def config_file(file_name: str='config.yml', encoding: str = 'utf-8') -> dict:
        """Read config form yml file in root of project config.yml

        Raises:
            Exception: _description_
            Exception: _description_

        Returns:
            dict: dict
        """
        try:
            config_path = Paths.get_file_path(f"config/{file_name}")
            with open(config_path, 'r', encoding=encoding) as config_file:
                config = yaml.safe_load(config_file)
                return config
        except FileNotFoundError:
            raise Exception(f"File {file_name} not found.") from FileNotFoundError
        except yaml.YAMLError as error:
            raise Exception(f"Error reading config file {file_name}: {error}") from error


    @staticmethod
    def get_config(config_path: str, encoding: str = 'utf-8') -> dict:
        """
        Read configuration from a YAML file.

        Args:
            config_path (str): The path to the YAML file.
            encoding (str): The encoding of the file. Default is UTF-8.

        Returns:
            dict: The configuration dictionary.
        """
        try:
            with open(config_path, 'r', encoding=encoding) as config_file:
                config = yaml.safe_load(config_file)
                return config
        except FileNotFoundError:
            raise Exception(f"File {config_path} not found.") from FileNotFoundError
        except yaml.YAMLError as error:
            raise Exception(f"Error reading config file {config_path}: {error}") from error


    @staticmethod
    def get_config_yml(section_key: str = None) -> dict:
        """Get the configuration settings from the YAML file.

        Args:
            section_key (str): The section of the YAML file to retrieve.

        Returns:
            dict: The dictionary of configuration settings.
        """
        config_path = Paths.get_file_path(PathFolder.CONFIG_YAML.value)
        config_dict = Config.get_config(config_path)
        return config_dict.get(section_key) if section_key else config_dict

    @staticmethod
    def load_config():
        """Load .env"""
        load_dotenv()

class TimeFormatter:
    """TimeFormatter for formatting time"""

    @staticmethod
    def format_dttime_now(pattern: str) -> str:
        """
        Format the current datetime as a string using a specific pattern.

        Args:
            pattern (str): The format string to use for the datetime.

        Returns:
            str: The formatted datetime string.

        Raises:
            ValueError: If an invalid datetime format string is provided.
        """
        try:
            formatted_time = datetime.now().strftime(pattern)
        except ValueError as error:
            raise ValueError(f"Invalid datetime format string: {pattern}") from error
        return formatted_time
