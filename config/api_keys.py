"""
API key management for the Agentic AI System
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class APIKeys:
    """Class to manage API keys from environment variables"""
    
    @staticmethod
    def get_gemini_api_key():
        """Get Gemini API key from environment"""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        return api_key
    
    @staticmethod
    def get_google_credentials_path():
        """Get path to Google credentials file"""
        credentials_path = os.getenv("GOOGLE_CREDENTIALS_PATH")
        if not credentials_path:
            # Default path
            return os.path.join(os.path.dirname(__file__), "..", "credentials", "google_credentials.json")
        return credentials_path
    
    @staticmethod
    def get_email_credentials():
        """Get email credentials from environment"""
        email = os.getenv("EMAIL_ADDRESS")
        password = os.getenv("EMAIL_PASSWORD")
        
        if not email or not password:
            raise ValueError("Email credentials not found in environment variables")
        
        return {
            "email": email,
            "password": password
        }