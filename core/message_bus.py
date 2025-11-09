"""
Message Bus for inter-agent communication in the Agentic AI System
"""

import asyncio
import logging
from typing import Dict, List
from .base_agent import AgentMessage

class MessageBus:
    """
    Central message bus for agent communication.
    
    This class manages the routing of messages between agents in the system.
    It's implemented as a singleton to ensure there's only one message bus.
    """
    
    _instance = None
    _lock = asyncio.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MessageBus, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
        
    def __init__(self):
        if self._initialized:
            return
            
        self.logger = logging.getLogger("message_bus")
        self.message_queues: Dict[str, asyncio.Queue] = {}
        self.subscribers: Dict[str, List[str]] = {}
        self._initialized = True
        
    async def register_agent(self, agent_name: str):
        """
        Register an agent with the message bus
        
        Args:
            agent_name (str): Name of the agent to register
        """
        async with self._lock:
            if agent_name not in self.message_queues:
                self.message_queues[agent_name] = asyncio.Queue()
                self.logger.info(f"Registered agent: {agent_name}")
                
    async def unregister_agent(self, agent_name: str):
        """
        Unregister an agent from the message bus
        
        Args:
            agent_name (str): Name of the agent to unregister
        """
        async with self._lock:
            if agent_name in self.message_queues:
                del self.message_queues[agent_name]
                self.logger.info(f"Unregistered agent: {agent_name}")
                
    async def send_message(self, message: AgentMessage):
        """
        Send a message to a specific agent
        
        Args:
            message (AgentMessage): The message to send
        """
        async with self._lock:
            if message.recipient not in self.message_queues:
                self.logger.warning(f"Attempted to send message to unknown agent: {message.recipient}")
                return False
                
            # Add the message to the recipient's queue
            await self.message_queues[message.recipient].put(message)
            self.logger.debug(f"Message sent from {message.sender} to {message.recipient}")
            return True
            
    async def broadcast_message(self, message: AgentMessage, exclude_sender: bool = True):
        """
        Broadcast a message to all agents
        
        Args:
            message (AgentMessage): The message to broadcast
            exclude_sender (bool): Whether to exclude the sender from the broadcast
        """
        async with self._lock:
            for agent_name, queue in self.message_queues.items():
                if exclude_sender and agent_name == message.sender:
                    continue
                    
                # Create a copy of the message for each recipient
                broadcast_message = AgentMessage(
                    sender=message.sender,
                    recipient=agent_name,
                    message_type=message.message_type,
                    data=message.data,
                    correlation_id=message.correlation_id,
                    timestamp=message.timestamp
                )
                
                await queue.put(broadcast_message)
                
            self.logger.debug(f"Message broadcast from {message.sender} to all agents")
            
    async def subscribe_to_topic(self, agent_name: str, topic: str):
        """
        Subscribe an agent to a topic
        
        Args:
            agent_name (str): Name of the agent
            topic (str): Topic to subscribe to
        """
        async with self._lock:
            if topic not in self.subscribers:
                self.subscribers[topic] = []
                
            if agent_name not in self.subscribers[topic]:
                self.subscribers[topic].append(agent_name)
                self.logger.info(f"Agent {agent_name} subscribed to topic: {topic}")
                
    async def unsubscribe_from_topic(self, agent_name: str, topic: str):
        """
        Unsubscribe an agent from a topic
        
        Args:
            agent_name (str): Name of the agent
            topic (str): Topic to unsubscribe from
        """
        async with self._lock:
            if topic in self.subscribers and agent_name in self.subscribers[topic]:
                self.subscribers[topic].remove(agent_name)
                self.logger.info(f"Agent {agent_name} unsubscribed from topic: {topic}")
                
    async def publish_to_topic(self, topic: str, message: AgentMessage):
        """
        Publish a message to all subscribers of a topic
        
        Args:
            topic (str): Topic to publish to
            message (AgentMessage): The message to publish
        """
        async with self._lock:
            if topic not in self.subscribers:
                self.logger.warning(f"No subscribers for topic: {topic}")
                return
                
            for agent_name in self.subscribers[topic]:
                # Create a copy of the message for each subscriber
                topic_message = AgentMessage(
                    sender=message.sender,
                    recipient=agent_name,
                    message_type=message.message_type,
                    data=message.data,
                    correlation_id=message.correlation_id,
                    timestamp=message.timestamp
                )
                
                await self.message_queues[agent_name].put(topic_message)
                
            self.logger.debug(f"Message published to topic {topic} from {message.sender}")
            
    async def get_agent_queue(self, agent_name: str) -> asyncio.Queue:
        """
        Get the message queue for an agent
        
        Args:
            agent_name (str): Name of the agent
            
        Returns:
            asyncio.Queue: The agent's message queue
        """
        async with self._lock:
            if agent_name not in self.message_queues:
                await self.register_agent(agent_name)
                
            return self.message_queues[agent_name]
            
    def get_status(self) -> Dict[str, any]:
        """
        Get the current status of the message bus
        
        Returns:
            Dict[str, any]: Status information
        """
        queue_sizes = {
            name: queue.qsize() 
            for name, queue in self.message_queues.items()
        }
        
        return {
            "registered_agents": list(self.message_queues.keys()),
            "queue_sizes": queue_sizes,
            "topics": list(self.subscribers.keys()),
            "subscriptions": self.subscribers
        }