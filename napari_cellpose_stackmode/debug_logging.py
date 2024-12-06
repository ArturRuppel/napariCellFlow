# optimized_logging.py

import logging
from functools import wraps
from typing import Callable

# Configure logging for console only
logger = logging.getLogger('cell_tracking')
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# Only add handler if none exists
if not logger.handlers:
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)


def log_operation(func: Callable) -> Callable:
    """Lightweight decorator for logging operations"""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if logger.level <= logging.DEBUG:
            logger.debug(f"Entering {func.__name__}")

        result = func(self, *args, **kwargs)

        if logger.level <= logging.DEBUG:
            logger.debug(f"Exiting {func.__name__}")
        return result

    return wrapper


def enable_debug():
    """Temporarily enable debug logging"""
    logger.setLevel(logging.DEBUG)


def disable_debug():
    """Disable debug logging"""
    logger.setLevel(logging.INFO)


class DebugLogging:
    """Context manager for temporary debug logging"""

    def __enter__(self):
        enable_debug()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        disable_debug()