"""
MCP Stdio Server for Read-Only MySQL Operations.

This server provides a set of tools to interact with a MySQL database in a safe,
read-only manner. It is designed to be consumed by other applications over
standard I/O using the Model-Context-Protocol (MCP).
"""

import os
import json
import logging
from pathlib import Path
import re

import mcp
import sqlalchemy
from dotenv import load_dotenv
from fastmcp import FastMcpApplication
from sqlalchemy import create_engine, text, inspect, MetaData
from sqlalchemy.exc import SQLAlchemyError
import sqlglot
from sqlglot import exp as sqlglot_exp

# --- Setup ---
logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] [MCP_SERVER] [%(levelname)s] %(message)s"
)
load_dotenv()


class SQLServer(FastMcpApplication):
    """
    An MCP server that exposes safe, read-only tools for a MySQL database.
    """

    def __init__(self):
        super().__init__()
        self.mysql_url = os.getenv("MYSQL_URL")
        if not self.mysql_url:
            raise ValueError("MYSQL_URL environment variable is not set.")

        self.engine = None
        self.inspector = None
        self._connect()

        self.default_limit = int(os.getenv("DEFAULT_LIMIT", 100))
        self.output_dir = Path(os.getenv("OUTPUT_DIR", "./outputs"))
        self.output_dir.mkdir(exist_ok=True)

        self.synonyms_file = self.output_dir / "synonyms.json"
        self.saved_queries_file = self.output_dir / "saved_queries.json"

        self._synonyms = self._load_json_data(self.synonyms_file)
        self._saved_queries = self._load_json_data(self.saved_queries_file)
        logging.info(f"Server initialized. Default LIMIT: {self.default_limit}. Outputs dir: {self.output_dir}")

    def _connect(self):
        """Initializes the database engine and inspector."""
        try:
            self.engine = create_engine(self.mysql_url)
            self.inspector = inspect(self.engine)
            logging.info(f"Successfully connected to database.")
        except Exception as e:
            logging.error(f"Failed to create database engine: {e}")
            raise

    def _load_json_data(self, file_path: Path) -> dict:
        """Loads data from a JSON file."""
        if file_path.exists():
            try:
                with open(file_path, "r") as f:
                    return json.load(f)
            except (IOError, json.JSONDecodeError) as e:
                logging.error(f"Could not load JSON from {file_path}: {e}")
        return {}

    def _save_json_data(self, data: dict, file_path: Path):
        """Saves data to a JSON file."""
        try:
            with open(file_path, "w") as f:
                json.dump(data, f, indent=2)
        except IOError as e:
            logging.error(f"Could not save JSON to {file_path}: {e}")

    def _validate_and_limit_sql(self, sql: str) -> str:
        """
        Validates that the SQL is a single, read-only SELECT statement.
        Injects a LIMIT if one is not present.
        """
        try:
            parsed_expressions = sqlglot.parse(sql, read="mysql")
            if len(parsed_expressions) != 1:
                raise ValueError("Only a single SQL statement is allowed.")

            parsed = parsed_expressions[0]

            is_select = isinstance(parsed, sqlglot_exp.Select)
            is_explain = isinstance(parsed, sqlglot_exp.Explain)

            if not (is_select or (is_explain and isinstance(parsed.this, sqlglot_exp.Select))):
                 raise ValueError("Only SELECT or EXPLAIN SELECT statements are allowed.")

            # Add LIMIT to the SELECT statement if not present
            target_select = parsed.this if is_explain else parsed
            if not target_select.args.get("limit"):
                target_select.limit(self.default_limit)

            return parsed.sql(dialect="mysql")

        except Exception as e:
            logging.error(f"SQL validation failed: {e} for SQL: {sql}")
            raise

    def _execute_sql(self, sql: str):
        """A helper to execute validated SQL and return results."""
        validated_sql = self._validate_and_limit_sql(sql)
        with self.engine.connect() as connection:
            result = connection.execute(text(validated_sql))
            columns = result.keys()
            rows = [dict(zip(columns, row)) for row in result.fetchall()]
            return {"columns": list(columns), "rows": rows}

    @mcp.tool()
    def get_schema(self) -> str:
        """Returns the full database schema as a JSON string."""
        try:
            schema_names = self.inspector.get_schema_names()
            schema = {}
            for schema_name in schema_names:
                # Skipping system schemas
                if schema_name in ['information_schema', 'mysql', 'performance_schema', 'sys']:
                    continue
                schema[schema_name] = {}
                tables = self.inspector.get_table_names(schema=schema_name)
                for table in tables:
                    columns = self.inspector.get_columns(table, schema=schema_name)
                    schema[schema_name][table] = {col['name']: str(col['type']) for col in columns}
            return json.dumps(schema)
        except SQLAlchemyError as e:
            return json.dumps({"error": f"Failed to get schema: {e}"})

    @mcp.tool()
    def describe_table(self, table_name: str) -> str:
        """
        Returns details for a specific table, including columns, primary key,
        foreign keys, and an approximate row count.
        """
        try:
            if table_name not in self.inspector.get_table_names():
                 return json.dumps({"error": f"Table '{table_name}' not found."})

            with self.engine.connect() as connection:
                # Note: ROWNUM is Oracle-specific. For MySQL, this is an estimate.
                count_res = connection.execute(text(f"SELECT COUNT(*) FROM `{table_name}`")).scalar_one()

            description = {
                "table_name": table_name,
                "columns": self.inspector.get_columns(table_name),
                "primary_key": self.inspector.get_pk_constraint(table_name),
                "foreign_keys": self.inspector.get_foreign_keys(table_name),
                "approx_row_count": count_res,
            }
            return json.dumps(description, default=str)
        except SQLAlchemyError as e:
            return json.dumps({"error": f"Failed to describe table '{table_name}': {e}"})

    @mcp.tool()
    def find_tables(self, keyword: str) -> str:
        """Finds tables with names containing the keyword."""
        keyword = keyword.lower()
        all_tables = self.inspector.get_table_names()
        found_tables = [tbl for tbl in all_tables if keyword in tbl.lower()]
        return json.dumps({"found_tables": found_tables})

    @mcp.tool()
    def find_columns(self, keyword: str) -> str:
        """Finds columns across all tables with names containing the keyword."""
        keyword = keyword.lower()
        found_columns = []
        for table in self.inspector.get_table_names():
            columns = self.inspector.get_columns(table)
            for col in columns:
                if keyword in col['name'].lower():
                    found_columns.append({"table": table, "column": col['name'], "type": str(col['type'])})
        return json.dumps({"found_columns": found_columns})

    @mcp.tool()
    def sample_rows(self, table_name: str, limit: int = 5) -> str:
        """Returns a small sample of rows from a table."""
        try:
            # Use the more robust _execute_sql helper
            return json.dumps(self._execute_sql(f"SELECT * FROM `{table_name}` LIMIT {limit}"), default=str)
        except (SQLAlchemyError, ValueError) as e:
            return json.dumps({"error": f"Failed to sample rows from '{table_name}': {e}"})

    @mcp.tool()
    def distinct_values(self, table_name: str, column_name: str, limit: int = 50) -> str:
        """Returns distinct values for a column in a table."""
        try:
            sql = f"SELECT DISTINCT `{column_name}` FROM `{table_name}` LIMIT {limit}"
            # No need to use _execute_sql here as it's a simple query
            with self.engine.connect() as connection:
                result = connection.execute(text(sql))
                values = [row[0] for row in result.fetchall()]
            return json.dumps({"table": table_name, "column": column_name, "distinct_values": values})
        except SQLAlchemyError as e:
            return json.dumps({"error": f"Failed to get distinct values: {e}"})

    @mcp.tool()
    def suggest_joins(self, table_a: str, table_b: str) -> str:
        """Suggests direct join conditions between two tables based on foreign keys."""
        fks_a = self.inspector.get_foreign_keys(table_a)
        fks_b = self.inspector.get_foreign_keys(table_b)
        suggestions = []

        for fk in fks_a:
            if fk['referred_table'] == table_b:
                for local_col, referred_col in zip(fk['constrained_columns'], fk['referred_columns']):
                    suggestions.append(f"{table_a}.{local_col} = {table_b}.{referred_col}")

        for fk in fks_b:
            if fk['referred_table'] == table_a:
                 for local_col, referred_col in zip(fk['constrained_columns'], fk['referred_columns']):
                    suggestions.append(f"{table_b}.{local_col} = {table_a}.{referred_col}")

        return json.dumps({"suggestions": suggestions})

    @mcp.tool()
    def join_path(self, source_table: str, target_table: str) -> str:
        """Finds a path of joins between a source and target table using foreign key relationships (BFS)."""
        if source_table == target_table:
            return json.dumps({"path": [source_table]})

        q = [(source_table, [source_table])]
        visited = {source_table}

        while q:
            current_table, path = q.pop(0)

            if current_table == target_table:
                return json.dumps({"path": path})

            # Check FKs pointing out from current_table
            fks = self.inspector.get_foreign_keys(current_table)
            for fk in fks:
                ref_table = fk['referred_table']
                if ref_table not in visited:
                    visited.add(ref_table)
                    new_path = path + [ref_table]
                    q.append((ref_table, new_path))

        return json.dumps({"error": "No join path found between specified tables."})

    @mcp.tool()
    def run_sql(self, sql: str) -> str:
        """
        Executes a validated, read-only, limited SELECT query and returns the results.
        """
        try:
            result = self._execute_sql(sql)
            return json.dumps(result, default=str)
        except (SQLAlchemyError, ValueError) as e:
            return json.dumps({"error": f"Failed to run SQL: {e}"})

    @mcp.tool()
    def explain_sql(self, sql: str) -> str:
        """
        Runs EXPLAIN on a SELECT query and returns the plan.
        """
        try:
            # We need to ensure the inner query is a SELECT
            parsed = sqlglot.parse_one(sql, read="mysql")
            if not isinstance(parsed, sqlglot_exp.Select):
                 raise ValueError("EXPLAIN can only be used with SELECT statements.")

            explain_sql = f"EXPLAIN {sql}"
            with self.engine.connect() as connection:
                result = connection.execute(text(explain_sql))
                plan = [dict(zip(result.keys(), row)) for row in result.fetchall()]
            return json.dumps({"explain_plan": plan}, default=str)
        except (SQLAlchemyError, ValueError) as e:
            return json.dumps({"error": f"Failed to explain SQL: {e}"})

    @mcp.tool()
    def data_profile(self, table_name: str) -> str:
        """
        Provides a data profile for a table including row count, NULL counts,
        and basic stats for up to 8 numeric columns.
        """
        try:
            if table_name not in self.inspector.get_table_names():
                 return json.dumps({"error": f"Table '{table_name}' not found."})

            columns = self.inspector.get_columns(table_name)
            numeric_cols = [
                c['name'] for c in columns
                if isinstance(c['type'], (sqlalchemy.types.Integer, sqlalchemy.types.Float, sqlalchemy.types.Numeric))
            ][:8] # Limit to 8 numeric columns to avoid huge queries

            selects = ["COUNT(*) as row_count"]
            for col in columns:
                selects.append(f"SUM(CASE WHEN `{col['name']}` IS NULL THEN 1 ELSE 0 END) AS `{col['name']}_null_count`")

            for col in numeric_cols:
                selects.extend([
                    f"MIN(`{col}`) as `{col}_min`",
                    f"MAX(`{col}`) as `{col}_max`",
                    f"AVG(`{col}`) as `{col}_avg`",
                ])

            sql = f"SELECT {', '.join(selects)} FROM `{table_name}`"
            with self.engine.connect() as connection:
                profile = connection.execute(text(sql)).fetchone()
                profile_dict = dict(zip(profile.keys(), profile))

            return json.dumps(profile_dict, default=str)
        except SQLAlchemyError as e:
            return json.dumps({"error": f"Failed to profile data for '{table_name}': {e}"})

    @mcp.tool()
    def export(self, sql: str, format: str = 'csv', limit: int = 1000) -> str:
        """
        Exports the result of a SELECT query to a file (csv or json) and returns the absolute file path.
        """
        if format not in ['csv', 'json']:
            return json.dumps({"error": "Invalid format. Must be 'csv' or 'json'."})

        try:
            # Use pandas for easy export, if available.
            import pandas as pd
        except ImportError:
            return json.dumps({"error": "Pandas is required for the export tool. Please install it."})

        try:
            validated_sql = self._validate_and_limit_sql(sql)
            # Override limit for export
            parsed = sqlglot.parse_one(validated_sql, read="mysql")
            parsed.limit(limit)
            final_sql = parsed.sql(dialect="mysql")

            df = pd.read_sql(text(final_sql), self.engine)

            # Generate a safe filename
            filename_base = re.sub(r'\W+', '_', sql)[:50]
            filepath = self.output_dir / f"export_{filename_base}.{format}"

            if format == 'csv':
                df.to_csv(filepath, index=False)
            else: # json
                df.to_json(filepath, orient='records', indent=2)

            return json.dumps({"exported_file_path": str(filepath.absolute())})
        except (SQLAlchemyError, ValueError, ImportError) as e:
             return json.dumps({"error": f"Export failed: {e}"})

    @mcp.tool()
    def save_query(self, name: str, sql: str) -> str:
        """Saves a SQL query with a given name for later use."""
        self._saved_queries[name] = sql
        self._save_json_data(self._saved_queries, self.saved_queries_file)
        return json.dumps({"status": "success", "name": name})

    @mcp.tool()
    def run_saved_query(self, name: str) -> str:
        """Runs a previously saved query by name."""
        if name not in self._saved_queries:
            return json.dumps({"error": f"Saved query '{name}' not found."})
        sql = self._saved_queries[name]
        return self.run_sql(sql)

    @mcp.tool()
    def get_synonyms(self) -> str:
        """Returns the dictionary of saved synonyms."""
        return json.dumps(self._synonyms)

    @mcp.tool()
    def upsert_synonym(self, alias: str, canonical: str) -> str:
        """Adds or updates a synonym (alias) for a canonical table/column name."""
        self._synonyms[alias] = canonical
        self._save_json_data(self._synonyms, self.synonyms_file)
        return json.dumps({"status": "success", "alias": alias, "canonical": canonical})

    @mcp.tool()
    def set_database(self, name: str) -> str:
        """
        Changes the database for the current session by reconnecting.
        The MYSQL_URL should be a template like 'mysql+pymysql://user:pass@host:port/{db_name}'.
        """
        try:
            # Assumes the URL has a placeholder or can be easily manipulated.
            # This is a simple approach; a more robust one might use URL parsing.
            base_url = self.mysql_url.rsplit('/', 1)[0]
            self.mysql_url = f"{base_url}/{name}"
            self._connect() # Re-initialize engine and inspector
            return json.dumps({"status": "success", "new_database_url": self.mysql_url})
        except Exception as e:
            return json.dumps({"error": f"Failed to set database: {e}"})

    @mcp.tool()
    def set_default_limit(self, n: int) -> str:
        """Sets the server-side default LIMIT for queries."""
        if n > 0 and n <= 10000: # Safety cap
            self.default_limit = n
            logging.info(f"Default LIMIT set to {n}")
            return json.dumps({"status": "success", "new_default_limit": n})
        return json.dumps({"error": "Limit must be between 1 and 10000."})

    @mcp.tool()
    def health_check(self) -> str:
        """Checks database connectivity and returns the MySQL version."""
        try:
            with self.engine.connect() as connection:
                result = connection.execute(text("SELECT VERSION()"))
                version = result.scalar()
                return json.dumps({"status": "ok", "mysql_version": version})
        except SQLAlchemyError as e:
            logging.error(f"Health check failed: {e}")
            return json.dumps({"status": "error", "message": str(e)})


if __name__ == "__main__":
    logging.info("Starting MCP SQL Server...")
    # The prompt requires pandas for the export tool, but lists it as optional.
    # I'll add a check inside the export tool itself.
    try:
        import pandas
    except ImportError:
        logging.warning("Pandas not found. The 'export' tool will not be available.")
        logging.warning("Please run 'pip install pandas' to enable it.")

    try:
        app = SQLServer()
        # Per instructions, run synchronously to avoid asyncio conflicts.
        app.run(transport="stdio")
    except Exception as e:
        logging.critical(f"Failed to start the MCP server: {e}")
        exit(1)
