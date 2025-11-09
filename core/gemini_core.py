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
        
    async def on_start(self):
        """Initialize the Gemini model when the agent starts."""
        try:
            api_key = APIKeys.get_gemini_api_key()
            genai.configure(api_key=api_key)
            # Using gemini-pro for text-based tasks
            self.model = genai.GenerativeModel('gemini-2.5-flash')
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
        try:
            # Step 1: Generate a structured task plan from the user request
            task_plan = await self._generate_task_plan(request)
            self.logger.debug(f"Generated task plan: {json.dumps(task_plan, indent=2)}")

            if not task_plan or "steps" not in task_plan:
                return "I'm sorry, I couldn't understand how to fulfill that request."

            # Step 2: Execute the plan by coordinating with other agents
            execution_results = await self._execute_task_plan(task_plan)

            # Step 3: Generate a natural language response based on the results
            final_response = await self._generate_response(request, execution_results)
            
            return final_response

        except Exception as e:
            self.logger.error(f"Error processing user request: {e}", exc_info=True)
            return f"An unexpected error occurred: {str(e)}"

    async def _generate_task_plan(self, request: str) -> Dict[str, Any]:
        """
        Use Gemini to create a structured task plan from a natural language request.
        """
        # This is a crucial prompt. It needs to be very clear about the available agents
        # and the expected JSON structure.
        prompt = f"""
        You are an AI orchestrator. Your task is to break down a user's request into a sequence of steps for specialized agents to execute.

        Available agents and their actions:
        - "contact_agent":
            - "find_contact": Finds a person by name, email, SRN, or PRN. Parameters: {{"identifier": "string"}}
        - "calendar_agent":
            - "schedule_meeting": Schedules a new meeting. Parameters: {{"attendees": ["email1", "email2"], "title": "string", "start_time": "ISO 8601 datetime", "end_time": "ISO 8601 datetime", "description": "string (optional)"}}
            - "reschedule_meeting": Reschedules an existing meeting. Parameters: {{"attendee": "string", "new_start_time": "ISO 8601 datetime", "new_end_time": "ISO 8601 datetime"}}. Note: This might be ambiguous if there are multiple meetings with the attendee.
            - "check_availability": Checks if attendees are free. Parameters: {{"attendees": ["email1", "email2"], "start_time": "ISO 8601 datetime", "end_time": "ISO 8601 datetime"}}
        - "email_agent":
            - "send_email": Sends an email. Parameters: {{"recipients": ["email1", "email2"], "subject": "string", "body": "string"}}

        User Request: "{request}"

        Create a JSON plan with an array of steps. Each step should specify the agent, action, and parameters. Steps will be executed in order.
        If an action depends on the result of a previous step, you can note that, but for now, we'll keep it simple.

        Example for "Schedule a meeting with person A tomorrow at 2 PM for project updates":
        {{
            "steps": [
                {{"agent": "contact_agent", "action": "find_contact", "parameters": {{"identifier": "person A"}}}},
                {{"agent": "calendar_agent", "action": "schedule_meeting", "parameters": {{"attendees": ["$contact_agent.email"], "title": "Project Updates", "start_time": "2023-10-27T14:00:00", "end_time": "2023-10-27T15:00:00"}}}}
            ]
        }}
        Note: Use "$agent_name.result_key" to pass data between steps. For the example above, "$contact_agent.email" would be replaced by the email found in the first step.

        Now, create the plan for the user's request. Return ONLY the JSON object.
        """
        
        response = self.model.generate_content(prompt)
        try:
            # Clean up the response text to ensure it's valid JSON
            response_text = response.text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse Gemini response as JSON: {response.text}. Error: {e}")
            return {}

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
                    # Find a suitable key to store the result. Let's assume agents return a primary result.
                    result_key = self._extract_result_key(response_data["data"])
                    if result_key:
                        context[f"{agent_name}.{result_key}"] = response_data["data"][result_key]

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

    async def _generate_response(self, original_request: str, execution_results: Dict[str, Any]) -> str:
        """
        Generate a final, natural language response based on the execution results.
        """
        prompt = f"""
        Based on the original request and the execution results, generate a helpful and natural-sounding response for the user.

        Original Request: "{original_request}"
        Execution Results: {json.dumps(execution_results, indent=2)}

        Analyze the results. If there were errors, explain them simply. If everything was successful, confirm what was done.
        Do not mention "steps", "agents", or "JSON". Just provide a clear, user-friendly summary.
        """
        response = self.model.generate_content(prompt)
        return response.text.strip()

    def _resolve_parameters(self, parameters: Dict, context: Dict) -> Dict:
        """Replace placeholders like '$agent.key' with actual values from the context."""
        resolved = {}
        for key, value in parameters.items():
            if isinstance(value, str) and value.startswith("$"):
                # Simple placeholder replacement
                resolved[key] = context.get(value, value) # Fallback to original if not found
            elif isinstance(value, list):
                # Handle lists of placeholders
                resolved[key] = [context.get(item, item) if isinstance(item, str) and item.startswith("$") else item for item in value]
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