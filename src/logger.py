"""
Logging Module.

Provides logging setup with timestamped log files for traceability.
It provides a standardized way to initialize logging and start recording log entries.

Example:
    from src.logger import logging

    if __name__ == '__main__':
        logging.info("Logging has started")
"""

import logging
import os
from datetime import datetime

from src.utils.get_root import get_root

LOG_FILE = f"{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log"
logs_path = os.path.join(get_root(), "logs", LOG_FILE)
os.makedirs(logs_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
