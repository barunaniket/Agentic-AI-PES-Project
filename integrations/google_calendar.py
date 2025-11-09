"""
Google Calendar API integration wrapper
"""

import os.path
import logging
from datetime import datetime, timedelta

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from config.settings import BASE_DIR

# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/calendar']
TOKEN_PATH = BASE_DIR / "credentials" / "token.json"

class GoogleCalendarAPI:
    """
    A wrapper for the Google Calendar API to simplify common operations.
    """
    def __init__(self, credentials_path: str):
        """
        Initializes the API wrapper and handles authentication.
        
        Args:
            credentials_path (str): Path to the Google Cloud 'credentials.json' file.
        """
        self.logger = logging.getLogger("google_calendar_api")
        self.creds = None
        self.service = None
        self.credentials_path = credentials_path
        #self._authenticate()

    def _authenticate(self):
        """Handles the OAuth 2.0 flow."""
        # The file token.json stores the user's access and refresh tokens
        if os.path.exists(TOKEN_PATH):
            self.creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
        
        # If there are no (valid) credentials available, let the user log in.
        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                self.creds.refresh(Request())
            else:
                if not os.path.exists(self.credentials_path):
                    raise FileNotFoundError(f"Google credentials file not found at {self.credentials_path}. Please download it from your Google Cloud Console.")
                    
                flow = InstalledAppFlow.from_client_secrets_file(self.credentials_path, SCOPES)
                try:
                    self.creds = flow.run_local_server(port=0)
                except Exception:
                    print("⚠️ Browser authentication failed, falling back to console mode.")
                    self.creds = flow.run_console()
            
            # Save the credentials for the next run
            os.makedirs(os.path.dirname(TOKEN_PATH), exist_ok=True)
            with open(TOKEN_PATH, 'w') as token:
                token.write(self.creds.to_json())

        try:
            self.service = build('calendar', 'v3', credentials=self.creds)
            self.logger.info("Successfully authenticated with Google Calendar API.")
        except Exception as e:
            self.logger.error(f"Failed to build Google Calendar service: {e}")
            raise

    def create_event(self, title: str, start_time: str, end_time: str, attendees: list, description: str = None, timezone: str = 'UTC'):
        """
        Creates a new event in the primary calendar.
        
        Args:
            title (str): The title of the event.
            start_time (str): ISO 8601 formatted start time.
            end_time (str): ISO 8601 formatted end time.
            attendees (list): List of attendee email addresses.
            description (str, optional): Event description.
            timezone (str, optional): Timezone for the event.
        
        Returns:
            dict: The created event object, or None if failed.
        """
        event_body = {
            'summary': title,
            'location': '',
            'description': description,
            'start': {
                'dateTime': start_time,
                'timeZone': timezone,
            },
            'end': {
                'dateTime': end_time,
                'timeZone': timezone,
            },
            'attendees': [{'email': email} for email in attendees],
            'reminders': {
                'useDefault': False,
                'overrides': [
                    {'method': 'email', 'minutes': 24 * 60},
                    {'method': 'popup', 'minutes': 10},
                ],
            },
        }
        try:
            event = self.service.events().insert(calendarId='primary', body=event_body, sendUpdates='all').execute()
            self.logger.info(f"Event created: {event.get('htmlLink')}")
            return event
        except HttpError as e:
            self.logger.error(f"An error occurred while creating the event: {e}")
            return None

    def list_events(self, query: str = None, time_min: str = None, max_results: int = 10):
        """
        Lists events from the primary calendar.
        
        Args:
            query (str, optional): Search terms for the event.
            time_min (str, optional): ISO 8601 formatted lower bound for event's start time.
            max_results (int, optional): Maximum number of events to return.
        
        Returns:
            list: A list of event objects.
        """
        try:
            now = datetime.utcnow().isoformat() + 'Z'  # 'Z' indicates UTC time
            events_result = self.service.events().list(
                calendarId='primary', 
                timeMin=time_min or now,
                maxResults=max_results, 
                singleEvents=True,
                orderBy='startTime',
                q=query
            ).execute()
            events = events_result.get('items', [])
            return events
        except HttpError as e:
            self.logger.error(f"An error occurred while listing events: {e}")
            return []

    def update_event(self, event_id: str, new_start_time: str, new_end_time: str, timezone: str = 'UTC'):
        """
        Updates the time of an existing event.
        
        Args:
            event_id (str): The ID of the event to update.
            new_start_time (str): ISO 8601 formatted new start time.
            new_end_time (str): ISO 8601 formatted new end time.
            timezone (str, optional): Timezone for the event.
        
        Returns:
            dict: The updated event object, or None if failed.
        """
        try:
            # First, get the existing event to preserve other details
            event = self.service.events().get(calendarId='primary', eventId=event_id).execute()
            
            # Update the time
            event['start']['dateTime'] = new_start_time
            event['start']['timeZone'] = timezone
            event['end']['dateTime'] = new_end_time
            event['end']['timeZone'] = timezone
            
            updated_event = self.service.events().update(
                calendarId='primary', 
                eventId=event_id, 
                body=event,
                sendUpdates='all'
            ).execute()
            self.logger.info(f"Event updated: {updated_event.get('htmlLink')}")
            return updated_event
        except HttpError as e:
            self.logger.error(f"An error occurred while updating the event: {e}")
            return None