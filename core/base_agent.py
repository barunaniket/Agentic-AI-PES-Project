"""
Base Agent class for the Agentic AI System
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Optional, Callable

class AgentStatus(Enum):
    """Enumeration for agent status"""
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    INITIALIZING = "initializing"

@dataclass
class AgentMessage:
    """Message structure for inter-agent communication"""
    sender: str
    recipient: str
    message_type: str
    data: Dict[str, Any]
    correlation_id: str
    timestamp: float

class BaseAgent(ABC):
    """
    Abstract base class for all agents in the system.
    
    This class provides the basic structure and functionality for agents,
    including message handling, status management, and lifecycle management.
    """
    
    def __init__(self, name: str):
        """
        Initialize the agent
        
        Args:
            name (str): Unique name for the agent
        """
        self.name = name
        self.status = AgentStatus.INITIALIZING
        self.message_queue = asyncio.Queue()
        self.response_handlers = {}
        self.logger = logging.getLogger(f"agent.{name}")
        self._running = False
        self._message_handler_task = None
        
    async def start(self):
        """Start the agent's message processing loop"""
        if self._running:
            self.logger.warning(f"Agent {self.name} is already running")
            return
            
        self.logger.info(f"Starting agent {self.name}")
        self._running = True
        self.status = AgentStatus.IDLE
        
        # Start the message processing task
        self._message_handler_task = asyncio.create_task(self._process_messages())
        
        # Call the agent's custom initialization
        await self.on_start()
        
    async def stop(self):
        """Stop the agent"""
        if not self._running:
            return
            
        self.logger.info(f"Stopping agent {self.name}")
        self._running = False
        
        # Cancel the message processing task
        if self._message_handler_task:
            self._message_handler_task.cancel()
            try:
                await self._message_handler_task
            except asyncio.CancelledError:
                pass
                
        # Call the agent's custom cleanup
        await self.on_stop()
        
        self.status = AgentStatus.IDLE
        
    async def _process_messages(self):
        """Process incoming messages in a continuous loop"""
        while self._running:
            try:
                # Wait for a message with a timeout to allow periodic checks
                message = await asyncio.wait_for(
                    self.message_queue.get(), 
                    timeout=1.0
                )
                
                # Process the message
                await self._handle_message(message)
                
            except asyncio.TimeoutError:
                # Timeout is expected, just continue the loop
                pass
            except asyncio.CancelledError:
                # Task was cancelled, exit gracefully
                break
            except Exception as e:
                self.logger.error(f"Error processing message in {self.name}: {str(e)}")
                self.status = AgentStatus.ERROR
                
    async def _handle_message(self, message: AgentMessage):
        """Handle an incoming message"""
        self.logger.debug(f"Agent {self.name} received message from {message.sender}: {message.message_type}")
        
        try:
            self.status = AgentStatus.BUSY
            
            # Check if there's a specific handler for this message type
            if message.message_type in self.response_handlers:
                handler = self.response_handlers[message.message_type]
                response = await handler(message)
            else:
                # Use the default handler
                response = await self.handle_message(message)
                
            # Normalize response to include a message "type" so recipients can identify it
            if response and isinstance(response, dict):
                # If a handler returned its own type, keep it. Otherwise, mark it as a task response.
                if "type" not in response:
                    response["type"] = "task_response"
                
            # Send response if needed and the message wasn't from the system
            if response and message.sender != "system":
                await self.send_message(message.sender, response, message.correlation_id)
                
        except Exception as e:
            self.logger.error(f"Error handling message in {self.name}: {str(e)}")
            self.status = AgentStatus.ERROR
            
            # Send error response (ensure it has task_response type)
            error_response = {
                "status": "error",
                "message": f"Error in {self.name}: {str(e)}",
                "correlation_id": message.correlation_id,
                "type": "task_response"
            }
            await self.send_message(
                message.sender,
                error_response,
                message.correlation_id
            )
        finally:
            if self.status != AgentStatus.ERROR:
                self.status = AgentStatus.IDLE
                
    @abstractmethod
    async def handle_message(self, message: AgentMessage) -> Optional[Dict]:
        """
        Handle an incoming message. Must be implemented by subclasses.
        
        Args:
            message (AgentMessage): The incoming message
            
        Returns:
            Optional[Dict]: Response data, or None if no response is needed
        """
        pass
        
    async def send_message(self, recipient: str, data: Dict, correlation_id: str = None):
        """
        Send a message to another agent
        
        Args:
            recipient (str): Name of the recipient agent
            data (Dict): Message data
            correlation_id (str): ID to correlate request/response
        """
        if not correlation_id:
            correlation_id = f"{self.name}_{asyncio.get_event_loop().time()}"
            
        message = AgentMessage(
            sender=self.name,
            recipient=recipient,
            message_type=data.get("type", "generic"),
            data=data,
            correlation_id=correlation_id,
            timestamp=asyncio.get_event_loop().time()
        )
        
        # Import here to avoid circular imports
        from .message_bus import MessageBus
        # Use the singleton instance and call its send_message method
        await MessageBus().send_message(message)
        
    def register_handler(self, message_type: str, handler: Callable):
        """
        Register a custom handler for a specific message type
        
        Args:
            message_type (str): Type of message to handle
            handler (Callable): Handler function
        """
        self.response_handlers[message_type] = handler
        
    async def on_start(self):
        """
        Called when the agent starts. Can be overridden by subclasses.
        """
        pass
        
    async def on_stop(self):
        """
        Called when the agent stops. Can be overridden by subclasses.
        """
        pass
        
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the agent
        
        Returns:
            Dict[str, Any]: Status information
        """
        return {
            "name": self.name,
            "status": self.status.value,
            "running": self._running,
            "queue_size": self.message_queue.qsize()
        }