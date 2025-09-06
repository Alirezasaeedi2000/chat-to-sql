# query_processor.py
# The brain of the application. It orchestrates the process from
# natural language query to a formatted, multi-mode response.

import json
import logging
import pandas as pd
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import matplotlib.pyplot as plt
import os

# Import our custom modules
import mcp_handler
import vector

# --- Configuration ---
MAX_RETRIES = 2

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - QUERY_PROC - %(levelname)s - %(message)s')

class QueryProcessor:
    def __init__(self, vector_store_instance, llm_model_name='llama3'):
        """
        Initializes the QueryProcessor with the LLM, vector store, and MCP handler.
        """
        self.vector_store = vector_store_instance
        self.llm = ChatOllama(model=llm_model_name, format="json")
        self.analytical_llm = ChatOllama(model=llm_model_name) # For markdown text generation

        # Ensure the mcp server is running
        mcp_handler.start_mcp_server()
        logging.info("QueryProcessor initialized.")

    def _format_context(self, context_docs):
        """Formats the retrieved context documents into strings."""
        schema_info = [doc['content'] for doc in context_docs if doc['type'] == 'schema']
        sample_rows = [doc['content'] for doc in context_docs if doc['type'] == 'sample_rows']
        past_queries = [doc['content'] for doc in context_docs if doc['type'] == 'past_query']

        return {
            "schema_info": "\n".join(schema_info) or "No schema info found.",
            "sample_rows": "\n".join(sample_rows) or "No sample rows found.",
            "past_queries": "\n".join(past_queries) or "No past queries found."
        }

    def _generate_visualization(self, df, user_query, sql_query):
        """Generates a plot from a DataFrame and returns the file path."""
        if df.empty or len(df.columns) < 2:
            return "Could not generate visualization: The query result is empty or has fewer than two columns.", None

        os.makedirs("outputs/charts", exist_ok=True)

        plt.figure(figsize=(10, 6))

        x_axis = df.columns[0]
        y_axis = df.columns[1]

        # Simple plot logic: bar for categorical, line for potential time series
        if pd.api.types.is_numeric_dtype(df[y_axis]):
            if pd.api.types.is_string_dtype(df[x_axis]) or pd.api.types.is_categorical_dtype(df[x_axis]):
                 # Take top 15 for readability
                subset = df.head(15)
                plt.bar(subset[x_axis], subset[y_axis])
                plt.xticks(rotation=45, ha='right')
            else:
                plt.plot(df[x_axis], df[y_axis])
        else:
             return "Could not generate visualization: The second column is not numeric.", None

        plt.title(f"Visualization for: {user_query[:50]}...")
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        plt.tight_layout()

        chart_filename = f"chart_{hash(user_query) & 0xffffffff}.png"
        filepath = os.path.join("outputs/charts", chart_filename)
        plt.savefig(filepath)
        plt.close()

        summary = f"Generated a plot of '{y_axis}' vs '{x_axis}'. Chart saved to: {filepath}\n"
        summary += df.to_markdown(index=False)

        return summary, filepath


    def process_query(self, user_query):
        """
        Processes a natural language query through the full RAG and execution pipeline.
        """
        logging.info(f"Processing query: '{user_query}'")

        for attempt in range(MAX_RETRIES + 1):
            try:
                # 1. Retrieve Context
                context_docs = self.vector_store.retrieve_context(user_query, k=5)
                formatted_context = self._format_context(context_docs)

                # 2. Construct Prompt and Invoke LLM for SQL
                system_prompt = mcp_handler.get_system_prompt()
                user_prompt_template = mcp_handler.get_user_prompt_template()

                prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("user", user_prompt_template)
                ])

                chain = prompt | self.llm | StrOutputParser()

                llm_response_str = chain.invoke({
                    "user_query": user_query,
                    **formatted_context
                })

                # Parse the JSON response from the LLM
                llm_response = json.loads(llm_response_str)
                thought = llm_response.get("thought", "No thought process provided.")
                mode = llm_response.get("response_mode", "UNKNOWN")
                sql_query = llm_response.get("sql_query", "")

                logging.info(f"LLM generated thought: {thought}")
                logging.info(f"LLM determined mode: {mode}, SQL: '{sql_query}'")

                if mode == "UNKNOWN" or not sql_query:
                    return f"I'm sorry, I could not understand the request or it is beyond my capabilities. Please try rephrasing.\nLLM thought: {thought}"

                # 3. Execute SQL
                sql_result = mcp_handler.run_sql(sql_query)
                if sql_result.get("status") == "error":
                    # This could be a SQL syntax error. Let's retry.
                    raise ValueError(f"SQL execution failed: {sql_result.get('message')}")

                # 4. Add successful query to history
                vector.add_query_to_history(self.vector_store, user_query, sql_query)

                # 5. Format Output based on Mode
                data = sql_result.get("data", [])
                if not data:
                    return "(No rows returned from query)"

                df = pd.DataFrame(data)

                if mode == "TABLE":
                    return df.to_markdown(index=False)

                elif mode == "SHORT_ANSWER":
                    # Return the first value of the first row
                    return str(df.iloc[0, 0])

                elif mode == "VISUALIZATION":
                    summary, filepath = self._generate_visualization(df, user_query, sql_query)
                    return summary

                elif mode == "ANALYTICAL":
                    logging.info("Performing analytical step...")
                    analytical_prompt_template = mcp_handler.get_analytical_prompt_template()

                    analytical_prompt = ChatPromptTemplate.from_template(analytical_prompt_template)
                    analytical_chain = analytical_prompt | self.analytical_llm | StrOutputParser()

                    analysis = analytical_chain.invoke({
                        "user_query": user_query,
                        "sql_query": sql_query,
                        "data_json": df.to_json(orient="records")
                    })
                    return analysis

                return "Successfully executed query, but response mode was unclear."

            except (json.JSONDecodeError, KeyError) as e:
                logging.warning(f"Attempt {attempt + 1}: Failed to parse LLM response. Error: {e}. Retrying...")
                if attempt >= MAX_RETRIES:
                    return f"Error: The LLM returned a malformed response after {MAX_RETRIES + 1} attempts."
                continue # Retry the loop

            except (ValueError, ConnectionError, TimeoutError) as e:
                 logging.error(f"Attempt {attempt + 1}: An error occurred. Error: {e}. Retrying...")
                 if attempt >= MAX_RETRIES:
                    return f"Error: An unrecoverable error occurred after {MAX_RETRIES + 1} attempts: {e}"
                 mcp_handler.stop_mcp_server() # Restart handler on error
                 mcp_handler.start_mcp_server()
                 continue # Retry the loop

        return "An unexpected error occurred after all retries."
