# Production-Ready NL-to-SQL Assistant

This project provides a full-stack, production-ready Natural Language to SQL assistant for MySQL. It leverages advanced features like a secure read-only SQL execution server, a multi-mode response system, and a RAG pipeline with vector embeddings to provide accurate, context-aware answers.

## Features

*   **Safe MCP (Mission Control Proxy) Server**: All SQL queries are executed in a sandboxed environment that only allows `SELECT` statements, automatically enforces `LIMIT` clauses, and prevents any DDL/DML operations.
*   **Multi-mode Response System**: The assistant can intelligently determine the user's intent and provide responses in various formats:
    *   **TABLE**: For data retrieval queries.
    *   **SHORT_ANSWER**: For queries asking for a single value.
    *   **ANALYTICAL**: For queries requiring interpretation and insights from the data.
    *   **VISUALIZATION**: For queries that are best represented with a chart or graph.
    *   **COMBO**: A combination of the above.
*   **Vector Embeddings (RAG)**: Utilizes `mxbai-embed-large` to create vector embeddings of the database schema, sample rows, and historical queries. This provides rich, dynamic context to the LLM for more accurate SQL generation.
*   **Pluggable LLM**: Uses `langchain_ollama` with `llama3` by default, but can be easily configured to use other powerful LLMs.
*   **Scalable Architecture**: The codebase is modular, maintainable, and designed for production use, with clear separation of concerns.
*   **Robustness**: Includes features like logging, query retries, and fallback strategies for errors or empty results.

## Project Structure

```
.
├── app.py                  # Main CLI application entry point
├── query_processor.py      # Core logic for NLQ processing, SQL generation, and output formatting
├── mcp_handler.py          # Client-side handler for the MCP server and LLM prompt management
├── mcp_sql_server.py       # Standalone, secure SQL execution stdio server
├── vector.py               # Handles vector store, embeddings, and RAG context retrieval
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── .env.example            # Example environment variables file
└── outputs/
    ├── synonyms.json       # Example synonyms file for schema mapping
    ├── export_template.csv # Example of an exported CSV file
    └── export_template.json# Example of an exported JSON file
```

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-folder>
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up the environment variables:**
    *   Copy the `.env.example` file to `.env`:
        ```bash
        cp .env.example .env
        ```
    *   Edit the `.env` file with your MySQL database credentials and other settings. The `DATABASE_URI` should follow the SQLAlchemy format for MySQL:
        ```
        # .env
        DATABASE_URI="mysql+pymysql://user:password@host:port/database"
        LLM_MODEL="llama3" # Or any other model available in Ollama
        EMBEDDING_MODEL="mxbai-embed-large"
        ```

5.  **Ensure you have Ollama running** with the specified model (e.g., `llama3`). You can pull the model with:
    ```bash
    ollama pull llama3
    ```

## How to Run

1.  **Make sure your MySQL server is running.**

2.  **Run the application:**
    ```bash
    python app.py
    ```

3.  The application will first initialize the vector store, which may take a moment. It will fetch schema information and sample rows from your database.

4.  Once initialized, you can start asking questions in natural language at the prompt.

## Example Queries

Here are some examples of queries you can try, demonstrating the different response modes:

*   **TABLE Mode:**
    > "Show me the last 5 users who signed up"
    > "List all products in the 'Electronics' category"

*   **SHORT_ANSWER Mode:**
    > "How many orders were placed yesterday?"
    > "What is the total revenue from the last month?"

*   **ANALYTICAL Mode:**
    > "Analyze the sales performance by product category. What are the key insights and recommendations?"
    > "What is the customer churn rate? Identify potential risks and suggest mitigation strategies."

*   **VISUALIZATION Mode:**
    > "Visualize the number of orders per month for the last year"
    > "Plot a bar chart of sales by region"

## How It Works

1.  **Input**: The `app.py` CLI takes a user's natural language query.
2.  **Context Retrieval (RAG)**: `query_processor.py` calls `vector.py` to search the vector store for relevant context (schema, sample rows, similar past queries) based on the user's query.
3.  **Intent & SQL Generation**: The query and retrieved context are sent to the LLM via `mcp_handler.py`. The LLM determines the user's intent (response mode) and generates the appropriate SQL query. The prompts are carefully engineered with few-shot examples to guide the LLM.
4.  **Safe SQL Execution**: The generated SQL is sent to the `mcp_sql_server.py` through the `mcp_handler.py`. The server validates the query (SELECT-only, no DDL/DML), injects a `LIMIT` clause, executes it, and returns the result.
5.  **Output Formatting**: `query_processor.py` receives the data and the response mode. It then formats the output as a markdown table, a single answer, a detailed analytical report, or generates a visualization, which is then printed to the console in `app.py`.
