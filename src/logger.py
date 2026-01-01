"""
Centralized logging configuration for the RL's Razor experiment.

This module provides a consistent logging setup across the entire project,
with both console and file handlers for different log levels.
"""

import logging
import os
from pathlib import Path
from typing import Optional


class LoggerConfig:
    """Configuration class for setting up project-wide logging."""

    DEFAULT_FORMAT = (
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    DETAILED_FORMAT = (
        "%(asctime)s - %(name)s - %(levelname)s - "
        "[%(filename)s:%(lineno)d] - %(message)s"
    )

    def __init__(
        self,
        log_dir: Optional[str] = None,
        level: int = logging.INFO,
        detailed: bool = False,
    ):
        """
        Initialize logger configuration.

        Args:
            log_dir: Directory for log files. Defaults to 'logs/' in project root.
            level: Logging level (default: INFO).
            detailed: If True, include filename and line numbers in logs.
        """
        self.log_dir = Path(log_dir or "logs")
        self.level = level
        self.detailed = detailed
        self.format_string = (
            self.DETAILED_FORMAT if detailed else self.DEFAULT_FORMAT
        )

        # Create log directory if it doesn't exist
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def get_logger(self, name: str) -> logging.Logger:
        """
        Get a configured logger instance.

        Args:
            name: Logger name (typically __name__).

        Returns:
            Configured logger instance.
        """
        logger = logging.getLogger(name)

        # Avoid duplicate handlers
        if logger.handlers:
            return logger

        logger.setLevel(self.level)

        # Create formatter
        formatter = logging.Formatter(self.format_string)

        # Console handler (INFO and above)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(max(self.level, logging.INFO))
        console_handler.setFormatter(formatter)

        # File handler (DEBUG and above)
        log_file = self.log_dir / f"{name.replace('.', '_')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        # Add handlers
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

        return logger


# Global logger configuration instance
_logger_config: Optional[LoggerConfig] = None


def configure_logging(
    log_dir: Optional[str] = None,
    level: int = logging.INFO,
    detailed: bool = False,
) -> None:
    """
    Configure the global logging setup.

    Should be called once at the start of the application.

    Args:
        log_dir: Directory for log files.
        level: Logging level.
        detailed: Include detailed information in logs.
    """
    global _logger_config
    _logger_config = LoggerConfig(log_dir=log_dir, level=level, detailed=detailed)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the global configuration.

    Args:
        name: Logger name (typically __name__).

    Returns:
        Configured logger instance.
    """
    if _logger_config is None:
        configure_logging()

    return _logger_config.get_logger(name)
