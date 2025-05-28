from datetime import datetime
import logging
import os
import sys


def setup_logging(run_name: str = "run", level=logging.INFO) -> None:
    """
    Setup logging to both console and a uniquely named file
    inside the 'logs/' directory.

    Parameters
    ----------
    level : int
        Logging level (e.g., logging.INFO).
    run_name : str
        Custom prefix for the log filename (e.g. 'training', 'tuning').
    """
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # Create log directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)

    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"logs/{run_name}_{timestamp}.log"

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(filename)s - %(message)s"
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # File handler
    file_handler = logging.FileHandler(log_filename, mode="w")
    file_handler.setFormatter(formatter)

    # Add handlers
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    root_logger.setLevel(level)
