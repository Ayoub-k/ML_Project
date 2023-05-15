"""File for storing constants
"""

from enum import Enum

class DateFormat(Enum):
    """For format date
    """
    DATE_FILE_FORMAT = '%Y-%m-%d'

class FileType(Enum):
    """file type 'extenstion of file'
    """
    LOG = '.log'
    CSV     = '.csv'
    PARQUET = '.parquet'

class PathFolder(Enum):
    """This class for storing paths folders and files
    """
    LOGS = 'logs'
    CONFIG_YAML = 'config/config.yml'


class TimeFilter(Enum):
    """Storing time filter
    """
    HOUR = 'hour'
    DAY = 'day'
    WEEK = 'week'
    MONTH = 'month'
    YEAR = 'year'
    ALL = 'all'
