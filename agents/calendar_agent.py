"""
Calendar Agent for managing calendar events in the Agentic AI System
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional

from core.base_agent import BaseAgent, AgentMessage, AgentStatus
from integrations.google_calendar import GoogleCalendarAPI
from config.api_keys import APIKeys
from config.settings import TIMEZONE

class CalendarAgent(BaseAgent):
    """
    Agent responsible for managing calendar events.
    It can schedule, reschedule, and check for meeting availability.
    """

    def __init__(self):
        super().__init__("calendar_agent")
        self.calendar_api: Optional[GoogleCalendarAPI] = None
        self._pending_contact_lookups = {} # To store futures for contact lookups

    async def on_start(self):
        """Initialize the Google Calendar API when the agent starts."""
        try:
            credentials_path = APIKeys.get_google_credentials_path()
            # 1. Create the instance (it's lightweight now)
            self.calendar_api = GoogleCalendarAPI(credentials_path)
            
            # 2. Run the blocking _authenticate method in a separate thread
            await asyncio.to_thread(self.calendar_api._authenticate)
            
            self.logger.info("CalendarAgent started and Google Calendar API initialized.")
        except Exception as e:
            self.logger.error(f"Failed to initialize Google Calendar API: {e}", exc_info=True)
            self.status = AgentStatus.ERROR
            raise

    async def handle_message(self, message: AgentMessage) -> Optional[Dict]:
        """Handle incoming messages for calendar operations."""
        if message.data.get("type") != "task":
            return {"status": "error", "message": "Invalid message type."}

        action = message.data.get("action")
        params = message.data.get("parameters", {})

        if action == "schedule_meeting":
            return await self._schedule_meeting(params)
        elif action == "reschedule_meeting":
            return await self._reschedule_meeting(params)
        elif action == "select_meeting":
            return await self._select_meeting_for_reschedule(params)
        elif action == "check_availability":
            return await self._check_availability(params)
        elif action == "list_upcoming_meetings": # <-- ADD THIS
            return await self._list_upcoming_meetings(params)
        elif action == "cancel_meeting":
            return await self._cancel_meeting(params)
        
        else:
            return {"status": "error", "message": f"Unknown action: {action}"}

    async def _schedule_meeting(self, params: Dict) -> Dict:
        """Schedules a new meeting after resolving attendee emails."""
        attendee_identifiers = params.get("attendees", [])
        title = params.get("title", "Meeting")
        start_time = params.get("start_time")
        end_time = params.get("end_time")
        description = params.get("description", "")

        if not all([attendee_identifiers, title, start_time, end_time]):
            return {"status": "error", "message": "Missing required parameters for scheduling."}

        # Resolve all identifiers to emails
        attendee_emails = await self._resolve_identifiers_to_emails(attendee_identifiers)
        if not attendee_emails:
            return {"status": "error", "message": "Could not resolve any attendees to valid emails."}

        event = self.calendar_api.create_event(
            title=title,
            start_time=start_time,
            end_time=end_time,
            attendees=attendee_emails,
            description=description,
            timezone=TIMEZONE,
        )

        if event:
            return {"status": "success", "data": {"event_id": event['id'], "link": event['htmlLink']}}
        else:
            return {"status": "error", "message": "Failed to create event in Google Calendar."}

    async def _cancel_meeting(self, params: Dict) -> Dict:
        """
        Finds and cancels a meeting based on time range and/or attendee.
        Handles ambiguity if multiple meetings are found.
        """
        start_time = params.get("start_time")
        end_time = params.get("end_time")
        attendee_identifier = params.get("attendee") # Optional

        if not start_time:
            # We at least need a time to search
            return {"status": "error", "message": "Missing required 'start_time' to find meeting."}

        # Find meetings:
        # If we have an attendee, search using their email (more specific)
        search_query = None
        if attendee_identifier:
            email = await self._get_contact_email(attendee_identifier)
            if email:
                search_query = email

        # Find all meetings in the specified time range
        # We give a small buffer (e.g., 2 hours) if no end_time is given
        if not end_time:
            from datetime import datetime, timedelta
            end_time_dt = datetime.fromisoformat(start_time) + timedelta(hours=2)
            end_time = end_time_dt.isoformat()

        meetings = self.calendar_api.list_events(
            query=search_query,
            time_min=start_time,
            time_max=end_time
        )

        if not meetings:
            return {"status": "error", "message": f"No meetings found matching your criteria to cancel."}

        elif len(meetings) == 1:
            # Only one meeting, proceed with deletion
            meeting = meetings[0]
            success = self.calendar_api.delete_event(event_id=meeting['id'])
            if success:
                return {"status": "success", "message": f"Successfully cancelled meeting: {meeting.get('summary', 'Untitled Meeting')}"}
            else:
                return {"status": "error", "message": "Failed to delete the event from Google Calendar."}

        else:
            # Multiple meetings found, ask for clarification
            formatted_meetings = [
                {"id": m['id'], "title": m.get('summary', 'No Title'), "start_time": m['start'].get('dateTime')} for m in meetings
            ]
            return {
                "status": "ambiguous",
                "message": f"Found multiple meetings. Please specify which one to cancel:",
                "action_to_perform": "cancel_meeting_by_id", # We need a new follow-up action
                "meetings": formatted_meetings
            }

    async def _list_upcoming_meetings(self, params: Dict) -> Dict:
        """Lists all upcoming events."""
        # We don't pass time_min so it defaults to 'now'
        meetings = self.calendar_api.list_events() 
        if meetings:
            return {"status": "success", "data": {"meetings": meetings}}
        else:
            return {"status": "success", "message": "No upcoming meetings found."}

    async def _reschedule_meeting(self, params: Dict) -> Dict:
        """Initial reschedule request that checks for ambiguity."""
        attendee_identifier = params.get("attendee")
        new_start_time = params.get("new_start_time")
        new_end_time = params.get("new_end_time")

        if not all([attendee_identifier, new_start_time, new_end_time]):
            return {"status": "error", "message": "Missing required parameters for rescheduling."}

        # Find all meetings with this attendee
        meetings = await self._find_meetings_with_attendee(attendee_identifier)

        if not meetings:
            return {"status": "error", "message": f"No meetings found with '{attendee_identifier}'."}
        elif len(meetings) == 1:
            # Only one meeting, proceed with rescheduling
            meeting = meetings[0]
            updated_event = self.calendar_api.update_event(
                event_id=meeting['id'],
                new_start_time=new_start_time,
                new_end_time=new_end_time,
                timezone=TIMEZONE,
            )
            if updated_event:
                return {"status": "success", "data": {"event_id": updated_event['id'], "link": updated_event['htmlLink']}}
            else:
                return {"status": "error", "message": "Failed to update event."}
        else:
            # Multiple meetings, ask for clarification
            formatted_meetings = [
                {"id": m['id'], "title": m['summary'], "start_time": m['start']['dateTime']} for m in meetings
            ]
            return {
                "status": "ambiguous",
                "message": f"Found multiple meetings with '{attendee_identifier}'. Please specify which one:",
                "meetings": formatted_meetings,
                "new_start_time": new_start_time,
                "new_end_time": new_end_time
            }

    async def _select_meeting_for_reschedule(self, params: Dict) -> Dict:
        """Reschedules a specific meeting after user selection."""
        meeting_id = params.get("meeting_id")
        new_start_time = params.get("new_start_time")
        new_end_time = params.get("new_end_time")

        if not all([meeting_id, new_start_time, new_end_time]):
            return {"status": "error", "message": "Missing required parameters for selection."}

        updated_event = self.calendar_api.update_event(
            event_id=meeting_id,
            new_start_time=new_start_time,
            new_end_time=new_end_time,
            timezone=TIMEZONE,
        )
        
        if updated_event:
            return {"status": "success", "data": {"event_id": updated_event['id'], "link": updated_event['htmlLink']}}
        else:
            return {"status": "error", "message": "Failed to update the selected event."}

    async def _check_availability(self, params: Dict) -> Dict:
        """Checks if attendees are free during a specific time slot."""
        attendee_identifiers = params.get("attendees", [])
        start_time = params.get("start_time")
        end_time = params.get("end_time")

        if not all([attendee_identifiers, start_time, end_time]):
            return {"status": "error", "message": "Missing required parameters for availability check."}
        
        # For a simple check, we can see if any events exist in that time slot.
        # A more advanced check would query free/busy endpoints.
        conflicting_events = self.calendar_api.list_events(time_min=start_time, time_max=end_time)

        return {
            "status": "success",
            "data": {
                "available": len(conflicting_events) == 0,
                "conflicts": conflicting_events
            }
        }

    async def _resolve_identifiers_to_emails(self, identifiers: List[str]) -> List[str]:
        """Helper to resolve a list of identifiers to a list of emails."""
        emails = []
        for identifier in identifiers:
            email = await self._get_contact_email(identifier)
            if email:
                emails.append(email)
        return emails

    async def _cancel_meeting_by_id(self, params: Dict) -> Dict:
        """Cancels a specific meeting by its ID after user selection."""
        meeting_id = params.get("meeting_id")
        if not meeting_id:
            return {"status": "error", "message": "Missing 'meeting_id'."}

        success = self.calendar_api.delete_event(event_id=meeting_id)
        if success:
            return {"status": "success", "message": f"Successfully cancelled the selected meeting."}
        else:
            return {"status": "error", "message": "Failed to delete the selected event."}

    async def _get_contact_email(self, identifier: str) -> Optional[str]:
        """Helper to ask the contact agent for an email."""
        correlation_id = f"contact_lookup_{identifier}_{self.name}_{asyncio.get_event_loop().time()}"
        await self.send_message(
            recipient="contact_agent",
            data={"type": "task", "action": "find_contact", "parameters": {"identifier": identifier}},
            correlation_id=correlation_id
        )
        
        # Wait for the response
        try:
            response = await self._wait_for_response(correlation_id, timeout=10)
            if response.get("status") == "success" and "data" in response and "email" in response["data"]:
                return response["data"]["email"]
        except asyncio.TimeoutError:
            self.logger.error(f"Timeout waiting for contact info for '{identifier}'")
        
        return None

    async def _find_meetings_with_attendee(self, identifier: str) -> List[Dict]:
        """Finds all future meetings that include a specific attendee."""
        email = await self._get_contact_email(identifier)
        if not email:
            return []
        
        # Use the email as a search query
        return self.calendar_api.list_events(query=email)

    async def _wait_for_response(self, correlation_id: str, timeout: int) -> Dict[str, Any]:
        """
        Wait for a specific response by correlation ID.
        This is a simplified version for this agent.
        """
        start_time = asyncio.get_event_loop().time()
        while (asyncio.get_event_loop().time() - start_time) < timeout:
            # Check the message queue for a matching response
            # This is a naive implementation. A more robust system would use futures.
            try:
                message = self.message_queue.get_nowait()
                if message.correlation_id == correlation_id and message.message_type == "task_response":
                    return message.data
                else:
                    # Put it back if it's not the one we're waiting for
                    await self.message_queue.put(message)
            except asyncio.QueueEmpty:
                pass
            await asyncio.sleep(0.1)
        raise asyncio.TimeoutError()