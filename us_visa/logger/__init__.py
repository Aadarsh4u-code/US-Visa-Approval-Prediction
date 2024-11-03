import logging
import os
from datetime import datetime

# Format to how log file is created
LOG_FILE = f"{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log"

log_dir = 'logs'

# Path for a log file
logs_path = os.path.join(os.getcwd(), log_dir, LOG_FILE)

# Even there is file keep on appending
os.makedirs(log_dir, exist_ok=True)  

logging.basicConfig(
    filename=logs_path,
    filemode='w',
    level=logging.DEBUG,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
)

# Log message with different severity levels
'''
    logging.debug("This is dubug message")
    logging.info("This is info message")
    logging.warning("This is warning message")
    logging.error("This is error message")
    logging.critical("This is critical message")
'''