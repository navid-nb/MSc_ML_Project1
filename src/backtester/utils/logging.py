"""Loguru logger configuration."""

from loguru import logger


def setup_logger(
    level: str = "INFO",
    fmt: str = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | {message}",
) -> None:
    """Configure loguru logger.

    Parameters
    ----------
    level : str
        Log level.
    fmt : str
        Format string.
    """
    logger.remove()
    logger.add(lambda msg: print(msg, end=""), level=level, format=fmt)
