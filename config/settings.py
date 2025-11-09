"""
Application settings for the Agentic AI System
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.parent

# Data directories
DATA_DIR = BASE_DIR / "data"
CONTACTS_DIR = DATA_DIR / "contacts"
MEETINGS_DIR = DATA_DIR / "meetings"
LOGS_DIR = DATA_DIR / "logs"

# Ensure directories exist
for directory in [DATA_DIR, CONTACTS_DIR, MEETINGS_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True)

# File paths
CONTACTS_FILE = CONTACTS_DIR / "contacts.csv"
MEETING_HISTORY_FILE = MEETINGS_DIR / "meeting_history.json"
SYSTEM_LOG_FILE = LOGS_DIR / "system.log"
ERROR_LOG_FILE = LOGS_DIR / "errors.log"

# API endpoints
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta"
GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"

# Agent settings
AGENT_TIMEOUT = 30  # seconds
MAX_RETRY_ATTEMPTS = 3
RETRY_DELAY = 1  # second

# Calendar settings
DEFAULT_MEETING_DURATION = 60  # minutes
TIMEZONE = "UTC"

# Email settings
EMAIL_SUBJECT_PREFIX = "[AI Assistant]"
EMAIL_SIGNATURE = "\n\n---\nSent by your AI Assistant"

# UI settings
CLI_PROMPT = "> "
WEB_PORT = 5000
WEB_HOST = "127.0.0.1"
DEBUG_MODE = True

# Logging settings
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"