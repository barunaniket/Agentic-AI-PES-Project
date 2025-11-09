"""
Gmail API integration wrapper
"""

import base64
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# We can reuse the authentication logic from the calendar wrapper
from integrations.google_calendar import SCOPES, TOKEN_PATH, BASE_DIR

# Add Gmail scope to the existing list
SCOPES.append('https://www.googleapis.com/auth/gmail.send')


class GmailAPI:
    """
    A wrapper for the Gmail API to simplify sending emails.
    """
    def __init__(self, credentials_path: str):
        """
        Initializes the API wrapper and handles authentication.
        Note: This will reuse or create the same token.json as the Calendar API.
        """
        self.logger = logging.getLogger("gmail_api")
        self.creds = None
        self.service = None
        self.credentials_path = credentials_path
        #self._authenticate()

    def _authenticate(self):
        """Handles the OAuth 2.0 flow. This is largely the same as the calendar auth."""
        if os.path.exists(TOKEN_PATH):
            self.creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
        
        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                self.creds.refresh(Request())
            else:
                if not os.path.exists(self.credentials_path):
                    raise FileNotFoundError(f"Google credentials file not found at {self.credentials_path}.")
                    
                flow = InstalledAppFlow.from_client_secrets_file(self.credentials_path, SCOPES)
                try:
                    self.creds = flow.run_local_server(port=0)
                except Exception:
                    print("⚠️ Browser authentication failed, falling back to console mode.")
                    self.creds = flow.run_console()
            
            with open(TOKEN_PATH, 'w') as token:
                token.write(self.creds.to_json())

        try:
            self.service = build('gmail', 'v1', credentials=self.creds)
            self.logger.info("Successfully authenticated with Gmail API.")
        except Exception as e:
            self.logger.error(f"Failed to build Gmail service: {e}")
            raise

    def send_email(self, to_email: str, subject: str, body_text: str, body_html: str = None):
        """
        Sends an email.

        Args:
            to_email (str): Recipient's email address.
            subject (str): Subject of the email.
            body_text (str): Plain text body of the email.
            body_html (str, optional): HTML body of the email.

        Returns:
            dict: The sent message object, or None if failed.
        """
        message = MIMEMultipart("alternative")
        message["to"] = to_email
        message["subject"] = subject

        # Attach parts
        message.attach(MIMEText(body_text, "plain"))
        if body_html:
            message.attach(MIMEText(body_html, "html"))

        # Encode the message
        encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
        
        create_message = {"raw": encoded_message}
        
        try:
            sent_message = self.service.users().messages().send(
                userId="me", 
                body=create_message
            ).execute()
            self.logger.info(f"Email sent to {to_email}, Message ID: {sent_message['id']}")
            return sent_message
        except HttpError as e:
            self.logger.error(f"An error occurred while sending the email: {e}")
            return None