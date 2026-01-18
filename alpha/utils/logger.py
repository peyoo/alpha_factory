import sys
from loguru import logger
from alpha.utils.config import settings

def setup_logger():
    """Sets up the logger configuration."""
    # Remove any existing handlers
    logger.remove()

    # Get the log level from the settings, default to 'DEBUG'
    log_level = settings.LOG_LEVEL if hasattr(settings, 'LOG_LEVEL') else 'DEBUG'

    # Add a new handler with the specified log level
    logger.add(sys.stdout, level=log_level, format="{time} {level} {message}")

    # You can add more handlers here (e.g., file handler) if needed

# Call the setup_logger function to configure the logger
setup_logger()
