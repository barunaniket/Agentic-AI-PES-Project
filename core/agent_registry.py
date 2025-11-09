"""
Agent Registry for managing agents in the Agentic AI System
"""

import asyncio
import logging
from typing import Dict, List, Optional
from .base_agent import BaseAgent, AgentStatus
from .message_bus import MessageBus

class AgentRegistry:
    """
    Registry for managing all agents in the system.
    
    This class provides a centralized way to register, start, stop, and manage agents.
    It's implemented as a singleton to ensure there's only one registry.
    """
    
    _instance = None
    _lock = asyncio.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AgentRegistry, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
        
    def __init__(self):
        if self._initialized:
            return
            
        self.logger = logging.getLogger("agent_registry")
        self.agents: Dict[str, BaseAgent] = {}
        self.message_bus = MessageBus()
        self._initialized = True
        
    async def register_agent(self, agent: BaseAgent):
        """
        Register an agent with the registry
        
        Args:
            agent (BaseAgent): The agent to register
        """
        async with self._lock:
            if agent.name in self.agents:
                self.logger.warning(f"Agent {agent.name} is already registered")
                return False
                
            self.agents[agent.name] = agent
            await self.message_bus.register_agent(agent.name)
            
            # Connect the agent to the message bus
            agent.message_queue = await self.message_bus.get_agent_queue(agent.name)
            
            self.logger.info(f"Registered agent: {agent.name}")
            return True
            
    async def unregister_agent(self, agent_name: str):
        """
        Unregister an agent from the registry
        
        Args:
            agent_name (str): Name of the agent to unregister
        """
        # Acquire lock briefly to check and remove reference
        async with self._lock:
            if agent_name not in self.agents:
                self.logger.warning(f"Agent {agent_name} is not registered")
                return False
            agent = self.agents[agent_name]
            # Remove from registry immediately so other callers don't see it
            del self.agents[agent_name]
        
        # Stop the agent and unregister from message bus outside the lock
        try:
            await agent.stop()
        except Exception as e:
            self.logger.error(f"Error stopping agent {agent_name} during unregister: {e}")
        
        await self.message_bus.unregister_agent(agent_name)
        self.logger.info(f"Unregistered agent: {agent_name}")
        return True
            
    async def start_agent(self, agent_name: str):
        """
        Start a specific agent
        
        Args:
            agent_name (str): Name of the agent to start
        """
        async with self._lock:
            if agent_name not in self.agents:
                self.logger.error(f"Agent {agent_name} is not registered")
                return False
            agent = self.agents[agent_name]
        
        # Start outside the lock to avoid deadlocks
        await agent.start()
        self.logger.info(f"Started agent: {agent_name}")
        return True
            
    async def stop_agent(self, agent_name: str):
        """
        Stop a specific agent
        
        Args:
            agent_name (str): Name of the agent to stop
        """
        async with self._lock:
            if agent_name not in self.agents:
                self.logger.error(f"Agent {agent_name} is not registered")
                return False
            agent = self.agents[agent_name]
        
        # Stop outside the lock
        await agent.stop()
        self.logger.info(f"Stopped agent: {agent_name}")
        return True
            
    async def start_all_agents(self):
        """Start all registered agents"""
        # Copy the agent names under the lock and then start them outside the lock
        async with self._lock:
            agent_names = list(self.agents.keys())
        for agent_name in agent_names:
            await self.start_agent(agent_name)
                
    async def stop_all_agents(self):
        """Stop all registered agents"""
        # Copy the agent names under the lock and then stop them outside the lock
        async with self._lock:
            agent_names = list(self.agents.keys())
        for agent_name in agent_names:
            await self.stop_agent(agent_name)
                
    def get_agent(self, agent_name: str) -> Optional[BaseAgent]:
        """
        Get a specific agent by name
        
        Args:
            agent_name (str): Name of the agent
            
        Returns:
            Optional[BaseAgent]: The agent if found, None otherwise
        """
        return self.agents.get(agent_name)
        
    def get_all_agents(self) -> List[BaseAgent]:
        """
        Get all registered agents
        
        Returns:
            List[BaseAgent]: List of all registered agents
        """
        return list(self.agents.values())
        
    def get_agents_by_status(self, status: AgentStatus) -> List[BaseAgent]:
        """
        Get all agents with a specific status
        
        Args:
            status (AgentStatus): The status to filter by
            
        Returns:
            List[BaseAgent]: List of agents with the specified status
        """
        return [agent for agent in self.agents.values() if agent.status == status]
        
    def get_status(self) -> Dict[str, any]:
        """
        Get the current status of all agents
        
        Returns:
            Dict[str, any]: Status information for all agents
        """
        agent_statuses = {
            name: agent.get_status() 
            for name, agent in self.agents.items()
        }
        
        return {
            "total_agents": len(self.agents),
            "agent_statuses": agent_statuses,
            "message_bus": self.message_bus.get_status()
        }