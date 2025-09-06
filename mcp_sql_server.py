#!/usr/bin/env python
# mcp_sql_server.py
# Standalone Mission Control Proxy (MCP) for safe SQL execution.
# This server communicates via stdio using JSON messages.
# It ensures that only read-only, SELECT queries are executed.

import os
import sys
import json
import logging
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.exc import SQLAlchemyError
import sqlparse
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()
DATABASE_URI = os.getenv("DATABASE_URI")
SQL_AUTO_LIMIT = int(os.getenv("SQL_AUTO_LIMIT", 100))

# --- Logging ---
# Log to stderr to avoid interfering with stdout JSON communication
logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                    format='%(asctime)s - MCP_SERVER - %(levelname)s - %(message)s')

# --- Database Engine ---
engine = None

def init_engine():
    """Initializes the SQLAlchemy engine."""
    global engine
    if not DATABASE_URI:
        logging.error("DATABASE_URI environment variable not set.")
        raise ValueError("DATABASE_URI is not configured.")
    try:
        engine = create_engine(DATABASE_URI)
        # Test connection
        with engine.connect() as connection:
            logging.info("Database engine initialized and connection successful.")
    except SQLAlchemyError as e:
        logging.error(f"Failed to create database engine or connect: {e}")
        raise

def is_select_only(sql_query):
    """
    Validates if the query is a SELECT-only statement using sqlparse.
    """
    if not sql_query.strip():
        return False

    # Check for DDL/DML keywords explicitly for extra safety
    ddl_dml_keywords = [
        'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'DROP', 'ALTER', 'TRUNCATE',
        'GRANT', 'REVOKE', 'COMMIT', 'ROLLBACK', 'MERGE'
    ]

    upper_query = sql_query.upper()
    for keyword in ddl_dml_keywords:
        if keyword in upper_query:
            return False

    parsed = sqlparse.parse(sql_query)
    for stmt in parsed:
        if stmt.get_type() != 'SELECT':
            return False
    return True

def inject_limit(sql_query, limit):
    """
    Injects a LIMIT clause into the SQL query if it doesn't already have one.
    """
    # Normalize query to handle different casings and spacing
    normalized_query = ' '.join(sql_query.strip().split()).upper()
    if 'LIMIT' in normalized_query:
        return sql_query # Return original query if LIMIT is present

    # Simple injection; might not cover all edge cases like comments or complex subqueries
    # For production, a more robust SQL parser/rewriter might be needed
    if sql_query.strip().endswith(';'):
        return f"{sql_query.strip()[:-1]} LIMIT {limit};"
    return f"{sql_query.strip()} LIMIT {limit}"

def process_command(command_data):
    """Processes a single command and returns a result dictionary."""
    action = command_data.get("action")
    params = command_data.get("params", {})

    if not action:
        return {"status": "error", "message": "No action specified."}

    try:
        with engine.connect() as connection:
            inspector = inspect(engine)

            if action == "get_schema":
                schema_name = params.get("schema", None)
                tables = inspector.get_table_names(schema=schema_name)
                schema_info = {}
                for table in tables:
                    schema_info[table] = [
                        {"name": col["name"], "type": str(col["type"])}
                        for col in inspector.get_columns(table, schema=schema_name)
                    ]
                return {"status": "success", "data": schema_info}

            elif action == "get_sample_rows":
                table_name = params.get("table")
                if not table_name:
                    return {"status": "error", "message": "Table name not provided."}

                # Use text() for literal SQL execution
                query = text(f"SELECT * FROM {table_name} LIMIT 5")
                result = connection.execute(query)
                rows = [dict(row) for row in result.mappings()]
                return {"status": "success", "data": rows}

            elif action == "run_sql":
                sql_query = params.get("sql")
                if not sql_query:
                    return {"status": "error", "message": "SQL query not provided."}

                if not is_select_only(sql_query):
                    logging.warning(f"Blocked non-SELECT query: {sql_query}")
                    return {"status": "error", "message": "Query is not a SELECT statement. Only SELECT queries are allowed."}

                safe_sql = inject_limit(sql_query, SQL_AUTO_LIMIT)
                logging.info(f"Executing safe SQL: {safe_sql}")

                result = connection.execute(text(safe_sql))

                if result.returns_rows:
                    columns = result.keys()
                    data = [dict(zip(columns, row)) for row in result.fetchall()]
                    return {"status": "success", "data": data}
                else:
                    return {"status": "success", "data": [], "message": "Query executed successfully, but returned no rows."}

            elif action == "health_check":
                return {"status": "success", "message": "MCP server is running."}

            else:
                return {"status": "error", "message": f"Unknown action: {action}"}

    except SQLAlchemyError as e:
        logging.error(f"Database error on action '{action}': {e}")
        return {"status": "error", "message": f"Database error: {e}"}
    except Exception as e:
        logging.error(f"An unexpected error occurred on action '{action}': {e}")
        return {"status": "error", "message": f"An unexpected error occurred: {e}"}

def main_loop():
    """Main loop to read commands from stdin and write results to stdout."""
    logging.info("MCP SQL server started. Listening on stdin for JSON commands.")
    for line in sys.stdin:
        try:
            command = json.loads(line)
            logging.info(f"Received command: {command.get('action')}")
            result = process_command(command)
        except json.JSONDecodeError:
            result = {"status": "error", "message": "Invalid JSON format."}
            logging.error("Failed to decode JSON from stdin.")
        except Exception as e:
            result = {"status": "error", "message": f"Server-side exception: {e}"}
            logging.error(f"Exception in main loop: {e}")

        # Write result to stdout
        sys.stdout.write(json.dumps(result) + '\n')
        sys.stdout.flush()

if __name__ == "__main__":
    try:
        init_engine()
        main_loop()
    except ValueError as e:
        logging.critical(f"Server initialization failed: {e}")
        sys.exit(1)
    except Exception as e:
        logging.critical(f"A critical error caused the server to stop: {e}")
        sys.exit(1)
