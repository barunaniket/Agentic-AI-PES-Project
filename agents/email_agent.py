"""
Email Agent for sending emails in the Agentic AI System
"""

import asyncio
import logging
from typing import Dict, Any, List

from core.base_agent import BaseAgent, AgentMessage, AgentStatus
from .gmail_api import GmailAPI
from config.api_keys import APIKeys
from config.settings import EMAIL_SUBJECT_PREFIX, EMAIL_SIGNATURE

class EmailAgent(BaseAgent):
    """
    Agent responsible for sending emails.
    """

    def __init__(self):
        super().__init__("email_agent")
        self.gmail_api: GmailAPI = None

    async def on_start(self):
        """Initialize the Gmail API when the agent starts."""
        try:
            credentials_path = APIKeys.get_google_credentials_path()
            # 1. Create the instance
            self.gmail_api = GmailAPI(credentials_path)

            # 2. Run the blocking _authenticate method in a separate thread
            await asyncio.to_thread(self.gmail_api._authenticate)

            self.logger.info("EmailAgent started and Gmail API initialized.")
        except Exception as e:
            self.logger.error(f"Failed to initialize Gmail API: {e}", exc_info=True)
            self.status = AgentStatus.ERROR
            raise

    async def handle_message(self, message: AgentMessage) -> Dict:
        """Handle incoming messages for email operations."""
        if message.data.get("type") != "task":
            return {"status": "error", "message": "Invalid message type."}

        action = message.data.get("action")
        params = message.data.get("parameters", {})

        if action == "send_email":
            return await self._send_email(params)
        else:
            return {"status": "error", "message": f"Unknown action: {action}"}

    async def _send_email(self, params: Dict) -> Dict:
        """Sends an email to specified recipients."""
        recipients = params.get("recipients", [])
        subject = params.get("subject", "Notification")
        body = params.get("body", "")

        if not recipients:
            return {"status": "error", "message": "No recipients specified."}

        # Add prefix and signature
        full_subject = f"{EMAIL_SUBJECT_PREFIX} {subject}"
        full_body = f"{body}{EMAIL_SIGNATURE}"

        successful_sends = []
        failed_sends = []

        for recipient in recipients:
            self.logger.info(f"Sending email to {recipient}...")
            result = self.gmail_api.send_email(
                to_email=recipient,
                subject=full_subject,
                body_text=full_body
            )
            if result:
                successful_sends.append(recipient)
            else:
                failed_sends.append(recipient)

        if not failed_sends:
            return {
                "status": "success", 
                "message": f"Email successfully sent to {len(successful_sends)} recipient(s)."
            }
        else:
            return {
                "status": "partial_success",
                "message": f"Email sent to {len(successful_sends)} recipient(s). Failed to send to {len(failed_sends)}: {', '.join(failed_sends)}."
            }