import logging
import sys
from pathlib import Path
from tqdm import tqdm
from vascular_superenhancement.utils.path_config import load_path_config

class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)
        self._last_progress_bar = None

    def emit(self, record):
        try:
            msg = self.format(record)
            # If there's an active progress bar, clear it first
            if self._last_progress_bar is not None:
                self._last_progress_bar.clear()
            
            # Print the log message
            print(msg, file=sys.stderr)
            
            # If there was a progress bar, redraw it
            if self._last_progress_bar is not None:
                self._last_progress_bar.refresh()
        except Exception:
            self.handleError(record)

    def set_last_progress_bar(self, pbar):
        """Set the last active progress bar to be managed by this handler."""
        self._last_progress_bar = pbar

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

def setup_patient_logger(patient_id: str, name: str = "vascular_superenhancement", level: int = logging.INFO) -> logging.Logger:
    """
    Set up a logger specifically for a patient that works with tqdm and logs to both console and file.
    Logs are written to working_dir/logs/patients/patient_id.log.
    
    Args:
        patient_id: ID of the patient
        name: Name of the logger (default: "vascular_superenhancement")
        level: Logging level (default: INFO)
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logger with patient-specific name
    logger = logging.getLogger(f"{name}.patient.{patient_id}")
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
    
    # Create and configure file handler for patient-specific log
    log_dir = path_config.working_dir / "logs" / "patients"
    log_dir.mkdir(exist_ok=True, parents=True)
    log_file = log_dir / f"{patient_id}.log"
    print(f"Patient {patient_id} log file: {log_file.absolute()}")
    
    file_handler = logging.FileHandler(log_file)
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
    log_file = log_dir / f"{name}.log"
    print(f"Main log file: {log_file.absolute()}")
    
    file_handler = logging.FileHandler(log_file)
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