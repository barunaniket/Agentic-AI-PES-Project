import asyncio
import logging
import os
import sys
from signal import SIGINT, SIGTERM

# Import core components and agents
from core.gemini_core import GeminiCore
from core.agent_registry import AgentRegistry
from agents.calendar_agent import CalendarAgent
from agents.email_agent import EmailAgent
from agents.contact_agent import ContactAgent

# Import configuration
from config.logging_config import setup_logging
from config.api_keys import APIKeys

# Set up logging for the entire application
logger = setup_logging()

def check_environment():
    """Check if all required environment variables are set."""
    if not os.path.exists(".env"):
        logger.error(".env file not found! Please copy .env.example to .env and fill in your API keys.")
        sys.exit(1)
    
    try:
        # This will raise an error if the key is not found
        APIKeys.get_gemini_api_key()
    except ValueError as e:
        logger.error(f"Environment check failed: {e}")
        sys.exit(1)
    
    logger.info("Environment variables are configured correctly.")

async def initialize_system():
    """
    Initialize all agents and the agent registry.
    
    Returns:
        tuple: A tuple containing the GeminiCore instance and the AgentRegistry instance.
    """
    logger.info("Initializing Agentic AI System...")
    
    # Get the singleton instance of the AgentRegistry
    agent_registry = AgentRegistry()
    
    # Create instances of all agents
    gemini_core = GeminiCore()
    calendar_agent = CalendarAgent()
    email_agent = EmailAgent()
    contact_agent = ContactAgent()
    
    # Register all agents with the registry
    await agent_registry.register_agent(gemini_core)
    await agent_registry.register_agent(calendar_agent)
    await agent_registry.register_agent(email_agent)
    await agent_registry.register_agent(contact_agent)
    
    logger.info("All agents registered. Starting them now...")
    
    # Start all agents
    await agent_registry.start_all_agents()
    
    logger.info("System initialization complete.")
    return gemini_core, agent_registry

async def shutdown_system(agent_registry: AgentRegistry):
    """Gracefully shut down all agents."""
    logger.info("Shutting down the system...")
    await agent_registry.stop_all_agents()
    logger.info("All agents stopped. System shutdown complete.")

async def main():
    """Main function to run the system."""
    # Perform initial checks
    check_environment()
    
    gemini_core = None
    agent_registry = None
    
    try:
        # Initialize the system
        gemini_core, agent_registry = await initialize_system()
        
        # Main command-line interface loop
        print("\n" + "="*50)
        print("   Agentic AI System is Ready!")
        print("   Type 'exit' or press Ctrl+C to quit.")
        print("="*50 + "\n")
        
        while True:
            try:
                # Get user input
                user_input = input("> ")
                
                # Check for exit command
                if user_input.lower() in ['exit', 'quit']:
                    print("Shutting down...")
                    break
                
                if not user_input.strip():
                    continue
                
                # Process the request through the Gemini Core
                response = await gemini_core.process_user_request(user_input)
                print(f"\n{response}\n")

            except EOFError: # Handle Ctrl+D
                print("\nShutting down...")
                break
            except Exception as e:
                logger.error(f"An error occurred in the main loop: {e}", exc_info=True)
                print(f"Sorry, I encountered an error: {e}")

    except Exception as e:
        logger.critical(f"A critical error occurred during system startup: {e}", exc_info=True)
        print(f"Failed to start the system: {e}")
    
    finally:
        # Ensure graceful shutdown
        if agent_registry:
            await shutdown_system(agent_registry)

if __name__ == "__main__":
    try:
        # Run the main async function
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        # Handle Ctrl+C gracefully
        print("\nInterrupt received. Shutting down...")
        logger.info("Program interrupted by user.")
    except Exception as e:
        # Catch any other unexpected errors
        logger.critical(f"An unhandled exception occurred: {e}", exc_info=True)
        print(f"A critical error occurred: {e}")