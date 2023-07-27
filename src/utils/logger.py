"""Logger
"""

import os
import logging
import logging.config
from src.utils.paths import Paths
from src.utils.constants import FileType, PathFolder, DateFormat
from src.utils.configure import TimeFormatter, Config


def setup_logging() -> dict:
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
    return config

logging.config.dictConfig(setup_logging())
