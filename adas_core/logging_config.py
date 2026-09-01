import logging
import sys


def setup_logging(
    level: int = logging.INFO,
    log_file: str | None = None,
    format_str: str | None = None,
) -> logging.Logger:
    """Configures and returns the root framework logger."""
    if format_str is None:
        format_str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    formatter = logging.Formatter(format_str)
    logger = logging.getLogger("adas")
    logger.setLevel(level)

    # Avoid adding multiple handlers if setup_logging is called repeatedly
    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if log_file:
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "adas") -> logging.Logger:
    """Retrieves a logger instance under the framework namespace."""
    if not name.startswith("adas") and name != "adas":
        name = f"adas.{name}"
    return logging.getLogger(name)
