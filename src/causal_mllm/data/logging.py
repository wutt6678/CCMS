"""Structured logging for the causal_mllm pipeline."""

from __future__ import annotations

import logging
import sys
from typing import Optional


_LOG_FORMAT = "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_initialized = False


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    name: str = "causal_mllm",
) -> logging.Logger:
    """Configure structured logging for the pipeline.

    Call once at the start of any CLI entry point.

    Args:
        level: Logging level (default INFO).
        log_file: Optional path to a log file. If given, logs go to both
                  stderr and the file.
        name: Root logger name for the package.

    Returns:
        The configured logger instance.
    """
    global _initialized

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if _initialized:
        return logger

    # Console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    logger.addHandler(console_handler)

    # Optional file handler
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        logger.addHandler(file_handler)

    # Prevent double-logging from root
    logger.propagate = False

    _initialized = True
    return logger


def get_logger(name: str = "causal_mllm") -> logging.Logger:
    """Get a child logger for a specific module.

    Args:
        name: Logger name, typically __name__ of the calling module.

    Returns:
        A logging.Logger instance.
    """
    return logging.getLogger(name)
