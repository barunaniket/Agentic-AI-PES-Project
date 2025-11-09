"""
Contact Agent for managing contact information in the Agentic AI System
"""

import pandas as pd
import logging
from typing import Dict, Any, Optional

from core.base_agent import BaseAgent, AgentMessage
from config.settings import CONTACTS_FILE

class ContactAgent(BaseAgent):
    """
    Agent responsible for managing and retrieving contact information.
    It loads contacts from a CSV file and can search by name, email, SRN, or PRN.
    """

    def __init__(self):
        super().__init__("contact_agent")
        self.contacts_df: pd.DataFrame = pd.DataFrame()
        self._load_contacts()

    def _load_contacts(self):
        """
        Load contacts from the CSV file specified in the settings.
        """
        self.logger.info(f"Attempting to load contacts from: {CONTACTS_FILE}")
        try:
            # Check if file exists to avoid a confusing pandas error
            if not CONTACTS_FILE.exists():
                self.logger.warning(f"Contacts file not found at {CONTACTS_FILE}. Starting with no contacts.")
                self.contacts_df = pd.DataFrame(columns=["Name", "Email", "SRN", "PRN"])
                return

            self.contacts_df = pd.read_csv(CONTACTS_FILE)
            # Standardize column names to lowercase to avoid case sensitivity issues
            self.contacts_df.columns = [col.strip().lower() for col in self.contacts_df.columns]
            
            # Ensure essential columns exist, fill with empty strings if not
            required_columns = ['name', 'email', 'srn', 'prn']
            for col in required_columns:
                if col not in self.contacts_df.columns:
                    self.contacts_df[col] = ''
                    self.logger.warning(f"Column '{col}' not found in contacts.csv. It has been added with empty values.")

            self.logger.info(f"Successfully loaded {len(self.contacts_df)} contacts.")

        except Exception as e:
            self.logger.error(f"Failed to load contacts from {CONTACTS_FILE}: {e}", exc_info=True)
            # Start with an empty DataFrame on error
            self.contacts_df = pd.DataFrame(columns=["name", "email", "srn", "prn"])

    async def handle_message(self, message: AgentMessage) -> Optional[Dict]:
        """
        Handle incoming messages, primarily for finding contacts.
        """
        if message.data.get("type") == "task":
            action = message.data.get("action")
            
            if action == "find_contact":
                identifier = message.data.get("parameters", {}).get("identifier")
                if not identifier:
                    return {"status": "error", "message": "Missing 'identifier' in parameters."}
                
                contact = self._find_contact(identifier)
                if contact:
                    self.logger.info(f"Found contact for identifier '{identifier}': {contact['name']}")
                    return {"status": "success", "data": contact}
                else:
                    self.logger.warning(f"No contact found for identifier: '{identifier}'")
                    return {"status": "error", "message": f"No contact found with identifier '{identifier}'."}
            
            elif action == "get_all_contacts":
                # A utility action, could be useful for a UI
                return {
                    "status": "success", 
                    "data": self.contacts_df.to_dict('records')
                }

        # If it's not a task we recognize, we return an error
        return {"status": "error", "message": f"Unknown action '{message.data.get('action')}'."}

    def _find_contact(self, identifier: str) -> Optional[Dict[str, Any]]:
        """
        Find a contact by name, email, SRN, or PRN.
        The search is case-insensitive.
        """
        if self.contacts_df.empty:
            return None

        identifier_lower = str(identifier).lower()

        # Create a boolean mask for rows where any of the fields match the identifier
        mask = (
            self.contacts_df['name'].str.lower().str.contains(identifier_lower, na=False) |
            self.contacts_df['email'].str.lower().str.contains(identifier_lower, na=False) |
            self.contacts_df['srn'].astype(str).str.lower().str.contains(identifier_lower, na=False) |
            self.contacts_df['prn'].astype(str).str.lower().str.contains(identifier_lower, na=False)
        )

        # Get the first matching row
        matched_rows = self.contacts_df[mask]
        
        if not matched_rows.empty:
            # Convert the first matching row to a dictionary
            contact_dict = matched_rows.iloc[0].to_dict()
            # Clean up NaN values by converting them to empty strings
            return {k: (v if pd.notna(v) else '') for k, v in contact_dict.items()}
        
        return None

    async def on_start(self):
        """Called when the agent starts."""
        self.logger.info("ContactAgent started and ready.")

    async def on_stop(self):
        """Called when the agent stops."""
        self.logger.info("ContactAgent stopped.")