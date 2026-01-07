"""Logging utilities."""

import logging
from typing import Optional


def setup_logging(level: int = logging.INFO,
                  format: str = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'):
    """Setup logging configuration."""
    logging.basicConfig(level=level, format=format)


def get_logger(name: str = 'storm_logos') -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)
