"""
Logging configuration for the Agentic AI System
"""

import logging
import logging.handlers
from .settings import SYSTEM_LOG_FILE, ERROR_LOG_FILE, LOG_FORMAT, LOG_LEVEL

def setup_logging():
    """Set up logging configuration for the application"""
    
    # Create formatters
    formatter = logging.Formatter(LOG_FORMAT)
    
    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, LOG_LEVEL, "INFO"))
    
    # Avoid adding handlers multiple times
    if root_logger.handlers:
        return root_logger

    # Create file handler for system logs
    system_handler = logging.handlers.RotatingFileHandler(
        SYSTEM_LOG_FILE,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    system_handler.setFormatter(formatter)
    root_logger.addHandler(system_handler)
    
    # Create file handler for error logs
    error_handler = logging.handlers.RotatingFileHandler(
        ERROR_LOG_FILE,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    root_logger.addHandler(error_handler)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    return root_logger

# NOTE: removed automatic `setup_logging()` call on import to avoid duplicate handlers.