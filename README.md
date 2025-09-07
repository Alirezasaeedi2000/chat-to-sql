## Natural Language → SQL Assistant (MySQL) with MCP, RAG, and Multi‑Mode Output

This project provides a production‑ready Python 3.11+ assistant that converts natural language into safe MySQL `SELECT` queries, executes them with strict guards, and returns multi‑mode responses (TABLE, SHORT_ANSWER, ANALYTICAL, VISUALIZATION/COMBO). It includes a secure MCP stdio server, vector embeddings (mxbai-embed-large) for schema/sample/past queries (RAG), and a CLI app.

### Features
- Safe SQL execution: SELECT‑only, single statement, automatic LIMIT injection and clamping
- MCP stdio server with useful tools (`get_schema`, `describe_table`, `find_tables`, `find_columns`, `distinct_values`, `run_sql`, `explain_sql`, `export`, `save_query`, `run_saved_query`, `health_check`)
- Vector store with Chroma and `mxbai-embed-large` embeddings for schema, sample rows, and past queries
- LLM via `langchain_ollama` (default `llama3.2`) optimized prompts and few‑shots
- Multi‑mode outputs: TABLE, SHORT_ANSWER, ANALYTICAL, VISUALIZATION/COMBO with optional matplotlib charts
- Logging, retries, error handling, and export helpers

### Requirements
- Python 3.11+
- MySQL reachable from your environment
- Ollama installed with required models:
  - `ollama pull llama3.2`
  - `ollama pull mxbai-embed-large`

Install Python dependencies:
```bash
pip install -r requirements.txt
```

Set your database URL (example):
```bash
export DATABASE_URL='mysql+pymysql://user:pass@host:3306/yourdb'
```

### Build the Vector Index (Schema + Samples)
```bash
python app.py --db-url "$DATABASE_URL" --build-index
```

### Run CLI (One‑shot)
```bash
python app.py --db-url "$DATABASE_URL" --query "Show top 10 customers by total purchase amount last year" --mode TABLE --export csv
```

### Run CLI (Interactive)
```bash
python app.py --db-url "$DATABASE_URL"
```

### Run MCP Server (stdio)
```bash
python app.py --server
```

Your MCP client should set `DATABASE_URL` in the environment. Tools are SELECT‑only and auto‑LIMIT guarded.

### Output Modes
- TABLE: prints a Markdown table; falls back to `(no rows)`
- SHORT_ANSWER: prints a concise scalar value
- ANALYTICAL: structured narrative (Insights, Gaps, Risks, Recommendations)
- VISUALIZATION/COMBO: saves a PNG chart under `outputs/plots/` and shows path

### File Structure
- `app.py`: CLI entrypoint and MCP server runner
- `query_processor.py`: RAG, intent detection, SQL generation, execution, formatting
- `mcp_handler.py`: Few‑shot prompts and builders
- `mcp_sql_server.py`: MCP stdio server and safe tools
- `vector.py`: Vector index manager and query expansion
- `outputs/`: exports, plots, logs, saved queries, synonyms

### Example Queries
- TABLE: "List the 20 most recent orders with customer name"
- SHORT_ANSWER: "How many orders were placed in 2023?"
- ANALYTICAL: "Compare Q2 vs Q1 revenue; highlight drivers and risks"
- VISUALIZATION: "Plot monthly sales trend for the past 12 months"

### Safety and Limits
- The system strictly rejects non‑SELECT queries and multiple statements
- Missing LIMIT is injected (default 50) and any LIMIT is clamped (max 1000)
- Timeouts and retries are enabled for robust execution

### Notes
- If the LLM hallucinated a column/table, first run `--build-index` and use MCP tools like `describe_table` to verify names
- You can add domain synonyms in `outputs/synonyms.json` to improve retrieval

# chat-to-sql
:)
