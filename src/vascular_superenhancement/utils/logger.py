import logging
import sys
from pathlib import Path
from tqdm import tqdm
from vascular_superenhancement.utils.path_config import load_path_config

class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)

def setup_sync_logger() -> logging.Logger:
    """Set up a logger specifically for sync operations.
    Note: File logging is handled by the calling shell script."""
    logger = logging.getLogger("sync_to_nas")
    logger.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

def setup_dataset_logger(name: str = "vascular_superenhancement", level: int = logging.INFO) -> logging.Logger:
    """
    Set up a logger that works with tqdm and logs to both console and file.
    Logs are written to working_dir.
    
    Args:
        name: Name of the logger
        level: Logging level (default: INFO)
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Get path configuration
    path_config = load_path_config()
    
    # Create and configure file handler for working_dir
    log_dir = path_config.working_dir / "logs"
    log_dir.mkdir(exist_ok=True, parents=True)
    file_handler = logging.FileHandler(log_dir / f"{name}.log")
    file_handler.setLevel(level)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Create and configure console handler that works with tqdm
    console_handler = TqdmLoggingHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)
    
    # Add console handler to logger
    logger.addHandler(console_handler)
    
    return logger 