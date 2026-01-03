"""
Centralized logging configuration for RL's Razor experiment.

Following SafetyNet pattern: https://github.com/MaheepChaudhary/safetynet
Reference: https://arxiv.org/pdf/2509.04259

Usage:
    from logger import get_logger
    logger = get_logger(__name__)
    logger.info("Message")
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path


def get_logger(name: str, log_dir: str = "logs", level: int = logging.INFO) -> logging.Logger:
    """
    Get or create a logger with file and console handlers.
    
    Args:
        name: Logger name (typically __name__)
        log_dir: Directory for log files
        level: Logging level (default: INFO)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    # Create logs directory
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler (INFO and above)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (DEBUG and above)
    log_filename = name.replace('.', '_') + '.log'
    file_handler = logging.FileHandler(
        os.path.join(log_dir, log_filename),
        mode='a',
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


def configure_root_logger(log_dir: str = "logs", level: int = logging.INFO):
    """
    Configure the root logger for the entire application.
    
    Args:
        log_dir: Directory for log files
        level: Logging level
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_handler = logging.FileHandler(
        os.path.join(log_dir, f'experiment_{timestamp}.log'),
        mode='w',
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    return root_logger
