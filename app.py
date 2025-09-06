# app.py
# Main entry point for the command-line NL-to-SQL assistant.
# This file initializes all components and runs the user interaction loop.

import os
import logging
from dotenv import load_dotenv
from termcolor import cprint

# Import our custom modules
import mcp_handler
from vector import VectorStore, prime_vector_store
from query_processor import QueryProcessor

# --- Load Environment Variables ---
load_dotenv()
LLM_MODEL = os.getenv("LLM_MODEL", "llama3")

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - APP - %(levelname)s - %(message)s')

def print_welcome_message():
    """Prints a welcome message to the user."""
    cprint("=====================================================", 'cyan')
    cprint(" Welcome to the Natural Language to SQL Assistant!", 'white', attrs=['bold'])
    cprint("=====================================================", 'cyan')
    cprint("\nThis application connects to your database to answer questions.", 'yellow')
    cprint("Type your question in plain English and press Enter.", 'yellow')
    cprint("Type 'exit' or 'quit' to close the application.\n", 'yellow')

def main():
    """
    Main function to initialize and run the application.
    """
    try:
        # 1. Start the secure MCP SQL server
        cprint("Starting the secure SQL server...", 'green')
        mcp_handler.start_mcp_server()
        # Verify connection
        health = mcp_handler.health_check()
        if not health or health.get('status') != 'success':
            cprint("Error: Could not establish connection with the MCP SQL server.", 'red')
            logging.critical("MCP server health check failed on startup.")
            return
        cprint("SQL server connection is healthy.", 'green')

        # 2. Initialize the Vector Store
        cprint("\nInitializing vector store for RAG...", 'green')
        vector_store = VectorStore()

        # Prime the vector store if it's empty
        if vector_store.index.ntotal == 0:
            cprint("Vector store is empty. Priming with database schema and sample data...", 'yellow')
            cprint("This might take a moment...", 'yellow')
            prime_vector_store(vector_store)
            cprint("Vector store priming complete.", 'green')
        else:
            cprint("Loaded existing vector store.", 'green')

        # 3. Initialize the Query Processor
        cprint(f"\nInitializing Query Processor with LLM: {LLM_MODEL}...", 'green')
        processor = QueryProcessor(vector_store_instance=vector_store, llm_model_name=LLM_MODEL)

        # 4. Start the main interaction loop
        print_welcome_message()

        while True:
            try:
                user_input = input("> ")
                if user_input.lower() in ['exit', 'quit']:
                    cprint("Exiting application. Goodbye!", 'yellow')
                    break
                if not user_input.strip():
                    continue

                cprint("\nThinking...", 'cyan')
                result = processor.process_query(user_input)

                cprint("\n--- Assistant's Response ---", 'green', attrs=['bold'])
                cprint(result, 'white')
                cprint("--------------------------\n", 'green', attrs=['bold'])

            except KeyboardInterrupt:
                cprint("\nInterrupted by user. Exiting...", 'yellow')
                break
            except Exception as e:
                cprint(f"\nAn unexpected error occurred in the main loop: {e}", 'red')
                logging.error(f"Error in main loop: {e}", exc_info=True)

    except Exception as e:
        cprint(f"\nAn critical error occurred during startup: {e}", 'red')
        logging.critical(f"Application startup failed: {e}", exc_info=True)
    finally:
        # The atexit hook in mcp_handler will take care of stopping the server
        cprint("\nShutting down.", 'yellow')

if __name__ == "__main__":
    main()
