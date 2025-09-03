# Natural Language to SQL Assistant

This project provides a production-ready, AI-powered assistant that converts natural language questions into read-only MySQL `SELECT` queries. It features a multi-mode response system to deliver answers in the most appropriate format (table, single value, or analytical narrative).

The system is delivered as both an interactive CLI application and a standalone MCP (Model Context Protocol) server for programmatic access to database tools.

## Key Features

- **NL-to-SQL Conversion**: Translates user questions into safe, read-only MySQL queries using an LLM (powered by Ollama).
- **Multi-Mode Response**: Automatically classifies user intent to provide:
    - `TABLE`: For structured lists of results.
    - `SHORT_ANSWER`: For concise, single-value answers (e.g., a count or a name).
    - `ANALYTICAL`: For narrative-style insights and recommendations.
- **Strict Safety Guardrails**:
    - Enforces `SELECT`-only queries, blocking all DDL and DML operations.
    - Automatically injects a `LIMIT` clause to prevent resource exhaustion.
    - Validates generated SQL for correctness and safety before execution.
- **Interactive CLI**: A user-friendly command-line interface for asking questions.
- **MCP Stdio Server**: A separate server that exposes a rich set of database tools over standard I/O for programmatic use.

---

## Requirements

- Python 3.11+
- A running MySQL instance.
- A running [Ollama](https://ollama.com/) instance with a model downloaded (e.g., `llama3.2`).

---

## Setup Instructions

First, clone the repository to your local machine:
```sh
git clone <repository_url>
cd <repository_directory>
```

### 1. Environment Setup (Linux/macOS)

```sh
# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 1. Environment Setup (Windows - PowerShell)

```powershell
# Create a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate-ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

The application is configured via environment variables. Create a file named `.env` in the root of the project directory and add the following variables.

```env
# .env file

# Connection string for your MySQL database.
# Format: mysql+pymysql://<user>:<password>@<host>:<port>/<database>
MYSQL_URL="mysql+pymysql://root:password@localhost:3306/test_db"

# The default Ollama model to use. Can be overridden by the --model CLI flag.
LLM_MODEL="llama3.2"

# The default row limit automatically applied to all queries.
DEFAULT_LIMIT=100

# The directory to store outputs from the 'export' tool and other files.
OUTPUT_DIR="./outputs"
```

---

## Usage

### Running the CLI Application

To start the interactive assistant, run:
```sh
python app.py
```

You can specify a different Ollama model using the `--model` flag:
```sh
python app.py --model llama3:70b
```

**Example Questions:**
- `how many employees in sales department?`
- `skills for ID 1079`
- `top 5 highest salaries with departments`
- `performance summary for Omid Shahbazi`
- `recommend a course of action for the HR department`

### Running the MCP Server

The MCP server provides programmatic access to a suite of database tools. It runs over standard I/O.

To start the server:
```sh
python mcp_sql_server.py
```

You can interact with it using an MCP client like `mcp-cli`.

**Example MCP Tool Calls (as JSON):**
```json
// Get the database schema
{"tool":"get_schema"}

// Describe a table
{"tool":"describe_table","params":{"table_name":"employees"}}

// Find tables matching a keyword
{"tool":"find_tables","params":{"keyword":"depart"}}

// Get distinct values from a column
{"tool":"distinct_values","params":{"table_name":"departments","column_name":"name","limit":50}}

// Run a safe, limited SQL query
{"tool":"run_sql","params":{"sql":"SELECT COUNT(*) FROM employees"}}

// Export query results to a CSV file
{"tool":"export","params":{"sql":"SELECT * FROM employees WHERE salary > 80000", "format":"csv"}}
```

---

## Safety and Limitations

This assistant is designed with safety as a primary concern.
- **Read-Only**: The query processor and MCP server strictly enforce that only `SELECT` statements can be executed. Any attempt to run a `CREATE`, `ALTER`, `DROP`, `INSERT`, `UPDATE`, or `DELETE` statement will be blocked.
- **Automatic LIMIT**: Every query executed through the system has a `LIMIT` clause automatically added to it (default is 100) to prevent accidentally fetching large amounts of data.
- **LLM Imperfection**: While the system has several layers of validation and correction, the underlying language model can still generate incorrect or nonsensical SQL. Always verify critical results.
- **Context-Dependent**: The quality of the generated SQL is highly dependent on a well-defined database schema with clear table and column names.
