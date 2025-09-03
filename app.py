"""
The main CLI application entry point for the NL-to-SQL assistant.

This script provides an interactive command-line interface for users to ask
natural language questions about their database.
"""

import argparse
import sys
from tqdm import tqdm
import time

# It's good practice to handle potential import errors.
try:
    from query_processor import QueryProcessor
except ImportError:
    print("Error: The 'query_processor.py' file was not found. Please ensure it is in the same directory.")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during import: {e}")
    sys.exit(1)


def main():
    """
    Main function to run the interactive CLI application.
    """
    parser = argparse.ArgumentParser(
        description="Natural Language to SQL Assistant CLI.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  python app.py
  python app.py --model llama3:70b
"""
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="The name of the Ollama model to use (e.g., 'llama3.2'). Overrides LLM_MODEL env var."
    )
    args = parser.parse_args()

    try:
        # Pass the model from CLI args to the processor.
        # The processor will handle falling back to the environment variable.
        print("Initializing the assistant...")
        processor = QueryProcessor(model_name=args.model)
        print(f"Initialization complete. Using model: {processor.llm_model}")
        print("Type your question and press Enter. Type 'exit' or 'quit' to end.")
    except Exception as e:
        print(f"\nFATAL: Could not initialize the QueryProcessor: {e}", file=sys.stderr)
        print("Please ensure your MYSQL_URL is correctly set in your .env file or environment.", file=sys.stderr)
        print("Also, check that the Ollama service is running and the specified model is available.", file=sys.stderr)
        sys.exit(1)

    while True:
        try:
            question = input("\n> ")
            if question.lower().strip() in ["exit", "quit"]:
                break
            if not question.strip():
                continue

            # A simple progress bar to show the user something is happening
            with tqdm(total=100, desc="Thinking...", bar_format='{desc}: {bar}|') as pbar:
                # Simulate progress as the LLM thinks
                for i in range(80):
                    pbar.update(1)
                    time.sleep(0.02)

                response = processor.process_question(question)

                # Finish the progress bar
                pbar.update(100 - pbar.n)

            print("\n--- Answer ---")
            print(response)
            print("--------------")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}", file=sys.stderr)
            print("Please try rephrasing your question or check the logs.", file=sys.stderr)

    print("\nGoodbye!")


if __name__ == "__main__":
    main()
