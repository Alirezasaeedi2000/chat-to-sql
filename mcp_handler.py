# mcp_handler.py
# Client-side handler for the Mission Control Proxy (MCP) server.
# It starts the MCP server as a subprocess and provides a Python API
# for interacting with it. Also contains prompt templates for the LLM.

import subprocess
import json
import atexit
import logging
import sys
from threading import Thread
from queue import Queue, Empty

# --- Globals ---
mcp_process = None
mcp_queue = Queue()

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - MCP_HANDLER - %(levelname)s - %(message)s')

def _enqueue_output(pipe, queue):
    """Thread target to read from a pipe and put lines into a queue."""
    try:
        for line in iter(pipe.readline, ''):
            queue.put(line)
    except Exception as e:
        logging.error(f"Error in reader thread: {e}")
    finally:
        pipe.close()


def start_mcp_server():
    """Starts the mcp_sql_server.py as a subprocess."""
    global mcp_process, mcp_queue
    if mcp_process and mcp_process.poll() is None:
        logging.info("MCP server is already running.")
        return

    try:
        # We use sys.executable to ensure we use the same python interpreter
        mcp_process = subprocess.Popen(
            [sys.executable, 'mcp_sql_server.py'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1, # Line-buffered
            encoding='utf-8'
        )
        logging.info(f"MCP server started with PID: {mcp_process.pid}")

        # Start a thread to read from stdout without blocking
        stdout_thread = Thread(target=_enqueue_output, args=(mcp_process.stdout, mcp_queue), daemon=True)
        stdout_thread.start()

        # Start a thread to log stderr for debugging
        stderr_queue = Queue()
        stderr_thread = Thread(target=_enqueue_output, args=(mcp_process.stderr, stderr_queue), daemon=True)
        stderr_thread.start()

        def log_stderr():
            while mcp_process.poll() is None or not stderr_queue.empty():
                try:
                    line = stderr_queue.get_nowait()
                    logging.info(f"[MCP_SERVER_STDERR] {line.strip()}")
                except Empty:
                    pass

        stderr_log_thread = Thread(target=log_stderr, daemon=True)
        stderr_log_thread.start()


    except FileNotFoundError:
        logging.error("mcp_sql_server.py not found. Make sure it's in the same directory.")
        raise
    except Exception as e:
        logging.error(f"Failed to start MCP server: {e}")
        raise

def stop_mcp_server():
    """Stops the MCP server process."""
    global mcp_process
    if mcp_process and mcp_process.poll() is None:
        logging.info("Stopping MCP server...")
        mcp_process.terminate()
        try:
            mcp_process.wait(timeout=5)
            logging.info("MCP server stopped.")
        except subprocess.TimeoutExpired:
            logging.warning("MCP server did not terminate gracefully. Forcing kill.")
            mcp_process.kill()
    mcp_process = None

# Register the cleanup function to run on exit
atexit.register(stop_mcp_server)

def _send_command(action, params=None):
    """Sends a command to the MCP server and gets the response."""
    if not mcp_process or mcp_process.poll() is not None:
        raise ConnectionError("MCP server is not running.")

    command = {"action": action, "params": params or {}}
    try:
        mcp_process.stdin.write(json.dumps(command) + '\n')
        mcp_process.stdin.flush()

        # Read from the queue with a timeout
        response_line = mcp_queue.get(timeout=20)
        response = json.loads(response_line)

        if response.get("status") == "error":
            logging.error(f"MCP server returned an error for action '{action}': {response.get('message')}")
            # Return the error response for the caller to handle
            return response

        return response
    except Empty:
        logging.error(f"Timeout waiting for response from MCP server for action '{action}'.")
        raise TimeoutError("No response from MCP server.")
    except (IOError, json.JSONDecodeError) as e:
        logging.error(f"Failed to communicate with MCP server: {e}")
        # Attempt to restart or handle error
        stop_mcp_server()
        start_mcp_server()
        raise ConnectionAbortedError("Communication with MCP server failed.")


# --- Public API for MCP Interaction ---

def get_schema(schema_name=None):
    """Retrieves the database schema."""
    params = {"schema": schema_name} if schema_name else {}
    return _send_command("get_schema", params)

def get_sample_rows(table_name):
    """Retrieves sample rows for a given table."""
    return _send_command("get_sample_rows", {"table": table_name})

def run_sql(sql_query):
    """Executes a safe SQL query."""
    return _send_command("run_sql", {"sql": sql_query})

def health_check():
    """Checks if the MCP server is alive and responding."""
    return _send_command("health_check")


# --- LLM PROMPT ENGINEERING SECTION ---
# Contains few-shot examples and templates for guiding the LLM.

def get_system_prompt():
    """
    Returns the main system prompt for the NL-to-SQL agent.
    This prompt defines the agent's persona, capabilities, and constraints.
    """
    return """You are a world-class AI assistant specialized in converting natural language questions into executable SQL queries for a MySQL database.

Your primary goal is to be safe, accurate, and helpful.

**Your operational directives are:**

1.  **Analyze the User's Intent**: Carefully determine what the user is asking for. Classify the intent into one of the following response modes:
    *   `TABLE`: The user wants to see a set of data (e.g., "list all users").
    *   `SHORT_ANSWER`: The user is asking for a single, specific value (e.g., "how many products do we have?").
    *   `ANALYTICAL`: The user needs an interpretation of the data, including insights, trends, or recommendations (e.g., "analyze sales by region").
    *   `VISUALIZATION`: The user wants a chart or graph (e.g., "plot monthly revenue").
    *   `UNKNOWN`: The request is unclear, ambiguous, or not related to the database.

2.  **Generate Safe SQL**: If the intent requires data, generate a single, syntactically correct, `SELECT`-only MySQL query.
    *   **NEVER** generate DML (INSERT, UPDATE, DELETE) or DDL (CREATE, DROP, ALTER) statements.
    *   Use the provided schema and context to ensure correct table and column names.
    *   If a query seems complex or requires joins, think step-by-step to construct it accurately.

3.  **Format Your Response**: You MUST respond with a single JSON object containing the following keys:
    *   `"thought"`: A brief, step-by-step reasoning of how you interpreted the query and constructed the SQL.
    *   `"response_mode"`: One of the modes listed above (`TABLE`, `SHORT_ANSWER`, `ANALYTICAL`, `VISUALIZATION`, `UNKNOWN`).
    *   `"sql_query"`: The generated SQL query. This should be an empty string if no query is needed (e.g., for `UNKNOWN` mode).

**IMPORTANT CONTEXT YOU WILL BE GIVEN:**

*   **Database Schema**: A description of the tables and columns available.
*   **Sample Rows**: Example data from relevant tables to understand data formats and values.
*   **Past Queries**: Examples of similar questions and their corresponding correct SQL queries.

Use this context to improve the accuracy of your generated SQL.
"""

def get_user_prompt_template():
    """
    This function returns a template for the user prompt, which will be filled with dynamic context.
    """
    return """
Please process the following user request.

**User's Question:**
"{user_query}"

---

**Database Context:**

**1. Schema Information:**
```json
{schema_info}
```

**2. Relevant Sample Rows:**
```json
{sample_rows}
```

**3. Similar Past Queries (Question -> SQL):**
```json
{past_queries}
```
---
**Few-shot Examples (User Question -> Your JSON Response):**

**Example 1: TABLE mode**
*   **User Question**: "Show me the names and emails of the last 5 customers who signed up"
*   **Your JSON Response**:
    ```json
    {{
      "thought": "The user wants a list of the 5 most recent customers. I need to select the name and email columns from the 'customers' table, order by the creation date in descending order, and limit the result to 5.",
      "response_mode": "TABLE",
      "sql_query": "SELECT name, email FROM customers ORDER BY created_at DESC LIMIT 5;"
    }}
    ```

**Example 2: SHORT_ANSWER mode**
*   **User Question**: "What is the total number of products in stock?"
*   **Your JSON Response**:
    ```json
    {{
      "thought": "The user is asking for a single number: the count of all products. I will use COUNT(*) on the 'products' table.",
      "response_mode": "SHORT_ANSWER",
      "sql_query": "SELECT COUNT(*) FROM products;"
    }}
    ```

**Example 3: ANALYTICAL mode**
*   **User Question**: "Analyze the monthly sales trend for the last 6 months. Are we growing?"
*   **Your JSON Response**:
    ```json
    {{
      "thought": "The user wants an analysis of sales trends. First, I need to get the data. I will group sales by month and sum the revenue for the last 6 months. The analysis itself will be performed in a subsequent step after the data is fetched.",
      "response_mode": "ANALYTICAL",
      "sql_query": "SELECT DATE_FORMAT(order_date, '%Y-%m') AS month, SUM(total_amount) AS monthly_revenue FROM orders WHERE order_date >= DATE_SUB(CURDATE(), INTERVAL 6 MONTH) GROUP BY month ORDER BY month ASC;"
    }}
    ```

**Example 4: VISUALIZATION mode**
*   **User Question**: "Can you create a bar chart showing the number of orders per city?"
*   **Your JSON Response**:
    ```json
    {{
      "thought": "The user wants a bar chart of orders by city. I need to count the number of orders and group them by the city. This data is perfect for a bar chart.",
      "response_mode": "VISUALIZATION",
      "sql_query": "SELECT city, COUNT(order_id) AS number_of_orders FROM orders GROUP BY city ORDER BY number_of_orders DESC LIMIT 10;"
    }}
    ```
---

Now, based on all the provided context and examples, process the user's question at the top of this prompt and provide your response in the required JSON format.
"""

def get_analytical_prompt_template():
    """
    Returns a prompt template for generating the analytical narrative after data has been fetched.
    """
    return """
You are a data analyst AI. Your task is to provide a clear, concise, and insightful analysis of the provided data.

**User's Original Question:**
"{user_query}"

**SQL Query Used to Fetch Data:**
```sql
{sql_query}
```

**Data Result (in JSON format):**
```json
{data_json}
```

---

**Your Task:**

Based on the user's question and the data, provide a structured analytical report in Markdown format.

The report should include the following sections:
1.  **`## Executive Summary`**: A brief, high-level summary of the key findings.
2.  **`## Key Insights`**: Bullet points detailing the most important insights derived from the data.
3.  **`## Data Overview`**: A short description of the data that was analyzed.
4.  **`## Recommendations`** (Optional): If applicable, suggest 1-2 actionable recommendations based on the insights.

**Example Response:**

**`## Executive Summary`**
Monthly sales have shown a consistent upward trend over the past six months, with a notable 25% increase in the most recent month.

**`## Key Insights`**
*   Revenue grew from $15,000 to $18,750 over the six-month period.
*   The largest month-over-month growth occurred between April and May.
*   The average monthly revenue is approximately $17,000.

**`## Data Overview`**
The analysis is based on the total revenue per month, calculated from the `orders` table for the last six months.

**`## Recommendations`**
*   Investigate the marketing campaigns that ran in the last two months to replicate the successful growth.
---

Now, generate the analytical report for the provided data.
"""
