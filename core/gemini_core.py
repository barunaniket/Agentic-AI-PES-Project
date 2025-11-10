"""
Gemini Core - The central orchestrator for the Agentic AI System
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, List, Optional

import google.generativeai as genai

from .base_agent import BaseAgent, AgentMessage, AgentStatus
from .agent_registry import AgentRegistry
from config.api_keys import APIKeys
from config.settings import AGENT_TIMEOUT

class GeminiCore(BaseAgent):
    """
    The central orchestrator agent that uses Gemini to understand user requests
    and coordinate other agents to fulfill them.
    """

    def __init__(self):
        super().__init__("gemini_core")
        self.model = None
        self.active_tasks = {}  # To track ongoing tasks and their responses
        self.agent_registry = AgentRegistry() # Get the singleton instance
        self.history = []

    async def on_start(self):
        """Initialize the Gemini model when the agent starts."""
        try:
            api_key = APIKeys.get_gemini_api_key()
            genai.configure(api_key=api_key)
            # Using gemini-pro for text-based tasks
            self.model = genai.GenerativeModel('gemini-2.5-flash')
            self.chat = self.model.start_chat(history=[])
            self.logger.info("Gemini model initialized successfully.")
        except Exception as e:
            self.logger.error(f"Failed to initialize Gemini model: {e}")
            self.status = AgentStatus.ERROR
            # Prevent the agent from starting if the model fails
            raise

    async def process_user_request(self, request: str) -> str:
        """
        Public method to process a user request from the UI (e.g., CLI).
        
        Args:
            request (str): The natural language request from the user.
            
        Returns:
            str: The final response to be shown to the user.
        """
        self.logger.info(f"Processing user request: '{request}'")
        self.history.append({"role": "user", "parts": request})
        try:
            # Step 1: Generate a structured task plan from the user request
            task_plan = await self._generate_task_plan(self.history)
            self.logger.debug(f"Generated task plan: {json.dumps(task_plan, indent=2)}")

            if not task_plan or "steps" not in task_plan:
                return "I'm sorry, I couldn't understand how to fulfill that request."

            # Step 2: Execute the plan by coordinating with other agents
            execution_results = await self._execute_task_plan(task_plan)

            # Step 3: Generate a natural language response based on the results
            final_response = await self._generate_response(self.history, execution_results)
            self.history.append({"role": "model", "parts": final_response})
            return final_response

        except Exception as e:
            self.logger.error(f"Error processing user request: {e}", exc_info=True)
            response = f"An unexpected error occurred: {str(e)}"
            self.history.append({{"role": "model", "parts": response}}) # <--- ADD THIS
            return response

    async def _generate_task_plan(self, history: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Use Gemini to create a structured task plan from a natural language request,
        using the conversation history for context.
        """
        
        # --- NEW: Get current time to provide context to the model ---
        from datetime import datetime
        current_time = datetime.now().isoformat()
        
        # --- NEW: Extract current request and format history ---
        current_request = history[-1]["parts"]
        # Format all messages *except* the latest one (which is the prompt)
        formatted_history = "\n".join([f"{msg['role']}: {msg['parts']}" for msg in history[:-1]])

        # --- UPDATED: A more robust and detailed prompt ---
        prompt = f"""
        You are a meticulous AI orchestrator. Your task is to convert a user's natural language request into a precise JSON array of steps for specialized agents to execute, using the conversation history as context. You must follow all rules exactly.

        --- Conversation History (for context) ---
        {formatted_history}
        
        --- Rules ---
        1.  **Current Time**: The current date and time is: {current_time}. You MUST use this as a reference for any relative times (e.g., "tomorrow", "in 1 hour", "at 2pm").
        1.5. **User Timezone**: The user's timezone is IST ('Asia/Kolkata').
        2.  **MANDATORY UTC CONVERSION**: You MUST interpret all user times (e.g., "2pm", "tomorrow morning") in the user's timezone (IST) and then **convert them to UTC (Coordinated Universal Time)** for the final JSON output. All output times MUST be in ISO 8601 format and end with a 'Z' to signify UTC.
            * **Example Conversion:** User says "tomorrow at 8am". IST is UTC+5:30. 8:00 AM IST is 02:30 AM UTC. The output MUST be "YYYY-MM-DDT02:30:00Z".
            * **Example Conversion:** User says "5pm". 5:00 PM IST (17:00) is 11:30 AM UTC. The output MUST be "YYYY-MM-DDT11:30:00Z".
        3.  **Dependencies & Placeholders**: You MUST identify dependencies.
            * If a request involves one or more people (e.g., "Kishan", "Sehal"), you MUST create a *separate* `contact_agent.find_contact` step for *each* person.
            * When a `find_contact` step uses an identifier like "Sehal", you MUST assume the resulting email will be available in a placeholder named `"$sehal_email"` (lowercase identifier + "_email").
            * Subsequent steps (like `schedule_meeting` or `send_email`) MUST then use these placeholders (e.g., `"$sehal_email"`, `"$kishan_email"`) in their parameter lists.
        4.  **Parameter Integrity**: All parameters for an action must be correctly formatted. Do not omit required parameters.
        5.  **Output Format**: You MUST return ONLY the raw JSON object, starting with {{{{ and ending with }}}}. Do NOT include "```json", "Here is the plan:", or any other text, greetings, or explanations.
        6.  **Contextual Awareness**: You MUST use the "Conversation History" to resolve ambiguous requests. For example, if the user says "reschedule it to 5pm", you must look at the history to find which meeting "it" refers to. If they say "at the same time", you must find the original time.

        --- Available Agents and Actions ---
        - "contact_agent":
            - "find_contact": Finds a person by name, email, SRN, or PRN.
              Parameters: {{"identifier": "string"}}
              Returns: {{"name": "...", "email": "...", "srn": "...", "prn": "..."}}
        - "calendar_agent":
            - "schedule_meeting": Schedules a new meeting.
              Parameters: {{"attendees": ["list_of_emails"], "title": "string", "start_time": "ISO 8601 datetime", "end_time": "ISO 8601 datetime", "description": "string (optional)"}}
            - "reschedule_meeting": Reschedules an existing meeting. **This action finds the meeting using the attendee's name or email.**
              Parameters: {{"attendee": "string, the name or email of a person in the meeting (e.g., 'Kishan')", "new_start_time": "ISO 8601 datetime", "new_end_time": "ISO 8601 datetime"}}
            - "check_availability": Checks if attendees are free.
              Parameters: {{"attendees": ["list_of_emails"], "start_time": "ISO 8601 datetime", "end_time": "ISO 8601 datetime"}}
            - "list_upcoming_meetings": Lists all upcoming meetings.
              Parameters: {{}}
            - "cancel_meeting": Cancels a meeting. Finds meetings by time and/or attendee.
              Parameters: {{"start_time": "ISO 8601 datetime (start of search window)", "end_time": "ISO 8601 datetime (end of search window, optional)", "attendee": "string_identifier (optional)"}}
        - "email_agent":
            - "send_email": Sends an email.
              Parameters: {{"recipients": ["list_of_emails"], "subject": "string", "body": "string"}}

        --- Examples (based on current time: {current_time}) ---
        
        User Request: "Schedule a meeting with Kishan Bhardwaj tomorrow at 2 PM for project updates. The meeting should last 1 hour."
        {{
            "steps": [
                {{
                    "agent": "contact_agent",
                    "action": "find_contact",
                    "parameters": {{"identifier": "Kishan Bhardwaj"}}
                }},
                {{
                    "agent": "calendar_agent",
                    "action": "schedule_meeting",
                    "parameters": {{
                        "attendees": ["$contact_agent.email"],
                        "title": "Project Updates",
                        "start_time": "2025-11-11T08:30:00Z", // 2:00 PM IST converted to UTC
                        "end_time": "2025-11-11T09:30:00Z", // 3:00 PM IST converted to UTC
                        "description": "Project updates meeting."
                    }}
                }}
            ]
        }}

        User Request: "Email Sehal and Kishan about the new deadline. Subject: Deadlines."
        {{
            "steps": [
                {{
                    "agent": "contact_agent",
                    "action": "find_contact",
                    "parameters": {{"identifier": "Sehal"}}
                }},
                {{
                    "agent": "contact_agent",
                    "action": "find_contact",
                    "parameters": {{"identifier": "Kishan"}}
                }},
                {{
                    "agent": "email_agent",
                    "action": "send_email",
                    "parameters": {{
                        "recipients": ["$sehal_email", "$kishan_email"],
                        "subject": "Deadlines",
                        "body": "Hi both, the new deadline is next Friday."
                    }}
                }}
            ]
        }}
        
        User Request: "Reschedule my meeting with Alice Johnson to next Monday at 10am."
        {{
            "steps": [
                {{
                    "agent": "calendar_agent",
                    "action": "reschedule_meeting",
                    "parameters": {{
                        "attendee": "Alice Johnson",
                        "new_start_time": "2025-11-17T04:30:00Z", // 10:00 AM IST converted to UTC
                        "new_end_time": "2025-11-17T05:30:00Z"  // 11:00 AM IST converted to UTC
                    }}
                }}
            ]
        }}

        --- Current User Request ---
        "{current_request}"

        --- JSON Plan ---
        """
        
        # Note: Using gemini-pro might be more reliable than gemini-2.5-flash for this complex JSON task
        # self.model = genai.GenerativeModel('gemini-pro') 
        response = self.model.generate_content(prompt)
        
        try:
            # Clean up the response text to ensure it's valid JSON
            response_text = response.text.strip()
            
            # Remove markdown fences if they still appear
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
                
            response_text = response_text.strip() # Final strip
            
            return json.loads(response_text)
        except (json.JSONDecodeError, AttributeError) as e:
            self.logger.error(f"Failed to parse Gemini response as JSON. Error: {e}")
            self.logger.debug(f"Raw model response text: {response.text}")
            return {{}} # Return empty dict on failure

    async def _execute_task_plan(self, task_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the task plan by sending messages to the appropriate agents.
        """
        results = {}
        context = {}  # To store data passed between steps, e.g., {"contact_agent.email": "persona@example.com"}

        for i, step in enumerate(task_plan.get("steps", [])):
            agent_name = step["agent"]
            action = step["action"]
            parameters = step["parameters"]

            # Resolve parameter dependencies (e.g., "$contact_agent.email")
            resolved_parameters = self._resolve_parameters(parameters, context)
            
            correlation_id = f"{agent_name}_{action}_{int(time.time())}_{i}"
            self.logger.info(f"Executing step {i+1}: {agent_name}.{action} with params {resolved_parameters}")

            # Send the task to the target agent
            await self.send_message(
                recipient=agent_name,
                data={
                    "type": "task",
                    "action": action,
                    "parameters": resolved_parameters
                },
                correlation_id=correlation_id
            )

            # Wait for the response
            try:
                response_data = await self._wait_for_response(correlation_id, timeout=AGENT_TIMEOUT)
                
                # Store the result for potential use in subsequent steps
                # We use a generic key like "email" or "event_id" which should be consistent in agent responses
                if response_data.get("status") == "success" and "data" in response_data:
                    # Find a suitable key to store the result (e.g., 'email', 'event_id')
                    result_key = self._extract_result_key(response_data["data"])
                    
                    if result_key: # We must have a result to store
                        result_value = response_data["data"][result_key]
                        
                        # -- NEW CONTEXT LOGIC --
                        # Check if this was a contact_agent search
                        if agent_name == "contact_agent" and action == "find_contact" and result_key == "email":
                            # Use the identifier to create a unique context key
                            # This matches the model's observed behavior of creating keys like '$sehal_email'
                            try:
                                identifier = resolved_parameters.get("identifier", "").lower()
                                # Clean the identifier (basic cleaning, removes spaces/symbols)
                                cleaned_identifier = ''.join(e for e in identifier if e.isalnum())
                                
                                if cleaned_identifier:
                                    custom_key = f"{cleaned_identifier}_email"
                                    context[custom_key] = result_value
                                    self.logger.debug(f"Stored contact result in context: {custom_key} = {result_value}")
                                else:
                                    self.logger.warning("Contact agent identifier was empty, cannot create unique context key.")
                            except Exception as e:
                                self.logger.error(f"Error creating unique context key: {e}")
                        # -- END NEW CONTEXT LOGIC --

                        # Fallback for other agents OR if custom key creation failed
                        # We also *still* store the default key.
                        # This allows single contacts to work and provides a fallback.
                        context_key = f"{agent_name}.{result_key}"
                        if context_key not in context: # Only set if not already set by a more specific key
                            context[context_key] = result_value
                            self.logger.debug(f"Stored default result in context: {context_key} = {result_value}")

                results[f"step_{i+1}"] = response_data

                # Handle ambiguity (e.g., multiple meetings found)
                if response_data.get("status") == "ambiguous":
                    self.logger.info(f"Ambiguity detected in step {i+1}. Asking for clarification.")
                    clarification_data = await self._handle_ambiguity(response_data)
                    if clarification_data:
                        # Re-execute the step with the clarified information
                        # This is a simplified approach; a more robust system might have a dedicated "clarify_and_retry" action
                        await self.send_message(
                            recipient=agent_name,
                            data={
                                "type": "task",
                                "action": "select_meeting", # Assuming agents have this for disambiguation
                                "parameters": clarification_data
                            },
                            correlation_id=f"{correlation_id}_clarified"
                        )
                        clarified_response = await self._wait_for_response(f"{correlation_id}_clarified", timeout=AGENT_TIMEOUT)
                        results[f"step_{i+1}_clarified"] = clarified_response
                    else:
                        # User couldn't clarify, so we fail this step
                        results[f"step_{i+1}_clarified"] = {"status": "failed", "message": "Could not resolve ambiguity."}


            except asyncio.TimeoutError:
                self.logger.error(f"Timeout waiting for response from {agent_name} for correlation_id {correlation_id}")
                results[f"step_{i+1}"] = {"status": "timeout", "message": f"Agent {agent_name} did not respond in time."}
            except Exception as e:
                self.logger.error(f"Error executing step {i+1}: {e}")
                results[f"step_{i+1}"] = {"status": "error", "message": str(e)}

        return {"results": results, "context": context}

    async def _wait_for_response(self, correlation_id: str, timeout: int) -> Dict[str, Any]:
        """
        Wait for a response with a specific correlation ID.
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            if correlation_id in self.active_tasks:
                response = self.active_tasks.pop(correlation_id)
                return response
            await asyncio.sleep(0.1)  # Small delay to prevent busy-waiting
        raise asyncio.TimeoutError()

    async def _handle_ambiguity(self, ambiguous_response: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Handle ambiguous responses from agents by asking the user for clarification.
        This method would typically interact with a UI. For now, we'll use the CLI.
        """
        # This is a placeholder for UI interaction
        # In a real app, this would trigger a UI prompt
        print("\n--- CLARIFICATION NEEDED ---")
        print(ambiguous_response.get("message", "Please clarify your request."))
        
        meetings = ambiguous_response.get("meetings", [])
        if not meetings:
            print("No options provided to clarify.")
            return None

        for i, meeting in enumerate(meetings, 1):
            print(f"{i}. {meeting.get('title', 'No Title')} on {meeting.get('start_time', 'Unknown Time')}")

        try:
            choice = input("Please select an option (number): ")
            selection_index = int(choice) - 1
            if 0 <= selection_index < len(meetings):
                selected_meeting = meetings[selection_index]
                # We need to pass back the ID of the selected meeting and the new time
                return {
                    "meeting_id": selected_meeting.get("id"),
                    "new_start_time": ambiguous_response.get("new_start_time"),
                    "new_end_time": ambiguous_response.get("new_end_time")
                }
            else:
                print("Invalid selection.")
                return None
        except (ValueError, TypeError):
            print("Invalid input.")
            return None

    async def _generate_response(self, history: List[Dict[str, str]], execution_results: Dict[str, Any]) -> str:
        """
        Generate a final, natural language response based on the execution results.
        """
        # Format the full history for the model
        history_string = "\n".join([f"{msg['role']}: {msg['parts']}" for msg in history])
        
        prompt = f"""
        You are a helpful AI assistant. Based on the *entire* conversation history and the results of the *latest* action, generate a helpful and natural-sounding response for the user.

        --- Conversation History ---
        {history_string}

        --- Execution Results (for the latest user request) ---
        {json.dumps(execution_results, indent=2)}

        Analyze the results. If there were errors, explain them simply. If everything was successful, confirm what was done.
        Do not mention "steps", "agents", or "JSON". Just provide a clear, user-friendly summary.
        Be concise and directly answer the user's last message.
        """
        response = self.model.generate_content(prompt)
        return response.text.strip()

    def _resolve_parameters(self, parameters: Dict, context: Dict) -> Dict:
        """Replace placeholders like '$agent.key' with actual values from the context."""
        resolved = {}
        for key, value in parameters.items():
            if isinstance(value, str) and value.startswith("$"):
                # Simple placeholder replacement
                resolved[key] = context.get(value[1:], value) # Look up key without '$'
            elif isinstance(value, list):
                # Handle lists of placeholders
                # Look up each item's key without the '$'
                resolved[key] = [context.get(item[1:], item) if isinstance(item, str) and item.startswith("$") else item for item in value]
            else:
                resolved[key] = value
        return resolved

    def _extract_result_key(self, data: Dict) -> Optional[str]:
        """Try to find a sensible key to store a result under (e.g., 'email', 'event_id')."""
        # Priority order for keys to extract
        priority_keys = ["email", "event_id", "contact", "meeting_id"]
        for key in priority_keys:
            if key in data:
                return key
        # If no priority key, just return the first key in the data dict
        return next(iter(data.keys())) if data else None

    async def handle_message(self, message: AgentMessage) -> Optional[Dict]:
        """
        Handle incoming messages, typically responses from other agents.
        """
        if message.message_type == "task_response":
            # Store the response correlated by ID
            self.active_tasks[message.correlation_id] = message.data
            self.logger.debug(f"Received response for {message.correlation_id}")
            # No need to send a reply back to the agent
            return None
        
        # Handle other message types if necessary
        self.logger.warning(f"Received unhandled message type: {message.message_type}")
        return {"status": "error", "message": "Unhandled message type"}