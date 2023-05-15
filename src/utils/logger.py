"""Logger
"""

import os
import logging
import logging.config
from paths import Paths
from constants import FileType, PathFolder, DateFormat
from configure import TimeFormatter, Config


class Logger:
    """Logger class for logging infos
    """
    def __init__(self, name: str=__name__):

        self.__setup_logging()
        self.logger = logging.getLogger(name)

    def __setup_logging(self):
        """Set up configuration logger
        """
        # Creating folder logs if not exist
        log_dir = Paths.get_project_root() / PathFolder.LOGS.value
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        date_now = TimeFormatter.format_dttime_now(DateFormat.DATE_FILE_FORMAT.value)
        file_name = f"{date_now}{FileType.LOG.value}"
        log_filename = os.path.join(log_dir, file_name)
        # Config logging
        config = Config.get_config_yml('logging')
        config['handlers']['fileHandler']['filename'] = log_filename
        logging.config.dictConfig(config)

    def info(self, message:str):
        """log infos

        Args:
            message (str): message we want logged
        """
        self.logger.info(message)

    def warning(self, message:str):
        """log warnings

        Args:
            message (str): message we want logged
        """
        self.logger.warning(message)

    def error(self, message:str):
        """log errors

        Args:
            message (str): message we want logged
        """
        self.logger.error(message)

    def critical(self, message:str):
        """log criticals

        Args:
            message (str): message we want logged
        """
        self.logger.critical(message)

    def debug(self, message:str):
        """log debugs

        Args:
            message (str): message we want logged
        """
        self.logger.debug(message)
