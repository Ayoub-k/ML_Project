"""This file for configuration"""

from datetime import datetime
import yaml
from paths import Paths
from constants import PathFolder
from dotenv import load_dotenv

class Config:
    """Class for getting configuration"""

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
            print(f"File {config_path} not found.")
            return {}
        except yaml.YAMLError as error:
            print(f"Error reading configuration file {config_path}: {error}")
            return {}


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
        """
        try:
            formatted_time = datetime.now().strftime(pattern)
        except ValueError as error:
            raise ValueError(f"Invalid datetime format string: {pattern}") from error
        return formatted_time
