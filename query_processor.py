"""
Core logic for the NL-to-SQL assistant.

This module contains the QueryProcessor class, which orchestrates the entire
process of converting a natural language question into a SQL query, executing it
safely, and formatting the results.
"""

import os
import re
import logging
import json

from dotenv import load_dotenv
from langchain_ollama.llms import OllamaLLM
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import SQLAlchemyError
import sqlglot
from sqlglot import exp as sqlglot_exp
from tabulate import tabulate

import mcp_handler

# --- Setup ---
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s')
load_dotenv()

class QueryProcessor:
    """
    Handles the end-to-end process of NL -> SQL -> Formatted Result.
    """
    def __init__(self, model_name: str = None):
        self.mysql_url = os.getenv("MYSQL_URL")
        if not self.mysql_url:
            raise ValueError("MYSQL_URL environment variable not set.")

        self.llm_model = model_name or os.getenv("LLM_MODEL", "llama3.2")

        try:
            self.llm = OllamaLLM(model=self.llm_model)
        except Exception as e:
            logging.error(f"Failed to initialize OllamaLLM with model '{self.llm_model}': {e}")
            raise

        self.engine = create_engine(self.mysql_url)
        self.inspector = inspect(self.engine)
        self.default_limit = int(os.getenv("DEFAULT_LIMIT", 100))

        logging.info(f"QueryProcessor initialized with model: {self.llm_model}")

    def _get_schema_representation(self) -> str:
        """Generates a string representation of the database schema."""
        schema_info = []
        tables = self.inspector.get_table_names()
        for table in tables:
            columns = self.inspector.get_columns(table)
            col_defs = ", ".join([f"{c['name']} ({c['type']})" for c in columns])
            schema_info.append(f"Table `{table}`: {col_defs}")
        return "\n".join(schema_info)

    def _get_value_hints(self, max_hints=10) -> str:
        """Fetches distinct values from key columns to provide as hints to the LLM."""
        hints = []
        hint_candidates = {
            "departments": "name",
            "employees": "first_name",
        }
        with self.engine.connect() as connection:
            for table, column in hint_candidates.items():
                try:
                    query = text(f"SELECT DISTINCT `{column}` FROM `{table}` LIMIT {max_hints}")
                    result = connection.execute(query).fetchall()
                    values = [row[0] for row in result]
                    if values:
                        hints.append(f"Sample values for `{table}`.`{column}`: {values}")
                except SQLAlchemyError:
                    continue
        return "\n".join(hints)

    def _classify_mode(self, question: str) -> str:
        """Classifies the user's question into TABLE, SHORT_ANSWER, or ANALYTICAL."""
        prompt = mcp_handler.create_mode_classification_prompt(question)
        try:
            response = self.llm.invoke(prompt).strip().upper()
            if response in ["TABLE", "SHORT_ANSWER", "ANALYTICAL"]:
                logging.info(f"[Mode] Classified as: {response}")
                return response
        except Exception as e:
            logging.error(f"LLM-based mode classification failed: {e}")

        q_lower = question.lower()
        if any(w in q_lower for w in ["recommend", "summarize", "insights", "why", "suggest"]):
            return "ANALYTICAL"
        if any(w in q_lower for w in ["how many", "what is", "who is", "count"]):
            return "SHORT_ANSWER"
        return "TABLE"

    def _extract_sql(self, llm_response: str) -> str | None:
        """Extracts a SQL query from a markdown code block."""
        match = re.search(r"```sql\n(.*?)\n```", llm_response, re.DOTALL)
        if match:
            return match.group(1).strip()
        if "SELECT" in llm_response.upper():
            return llm_response.strip()
        return None

    def _validate_sql(self, sql: str) -> str | None:
        """Validates SQL using sqlglot for safety and correctness."""
        try:
            parsed = sqlglot.parse_one(sql, read="mysql")
            if not isinstance(parsed, sqlglot_exp.Select):
                raise ValueError("Query is not a SELECT statement.")
            if not parsed.args.get("from"):
                raise ValueError("Query must have a FROM clause.")

            is_constant_only = all(isinstance(c, sqlglot_exp.Literal) for c in parsed.selects)
            if is_constant_only and not (parsed.args.get("where") or parsed.args.get("from")):
                 raise ValueError("Query selects only constants without a meaningful FROM/WHERE clause.")

            if not parsed.args.get("limit"):
                parsed.limit(self.default_limit)

            return parsed.sql(dialect="mysql")
        except Exception as e:
            logging.error(f"SQL validation failed: {e} for SQL: {sql}")
            return None

    def _execute_sql(self, sql: str) -> dict | None:
        """Executes a validated SQL query and returns the results."""
        try:
            with self.engine.connect() as connection:
                connection.execute(text("SET SESSION MAX_EXECUTION_TIME=5000;")) # 5s timeout
                result = connection.execute(text(sql))
                columns = list(result.keys())
                rows = [dict(zip(columns, row)) for row in result.fetchall()]
                return {"columns": columns, "rows": rows}
        except SQLAlchemyError as e:
            logging.error(f"[Error] {e} for SQL: {sql}")
            raise

    def _format_response(self, result: dict, mode: str) -> str:
        """Formats the query result based on the response mode."""
        if not result or not result["rows"]:
            return "(no rows)"

        if mode == "TABLE":
            return tabulate(result["rows"], headers="keys", tablefmt="pipe")

        if mode == "SHORT_ANSWER":
            first_row = result["rows"][0]
            if "first_name" in first_row and "last_name" in first_row:
                return f"{first_row['first_name']} {first_row['last_name']}"
            for key in ["name", "email", "title"]:
                if key in first_row:
                    return str(first_row[key])
            return str(list(first_row.values())[0])

        return json.dumps(result, indent=2)

    def _get_analytical_fallback(self, question: str, schema: str) -> str:
        """Generates an analytical fallback response."""
        logging.info("Generating analytical fallback response.")
        prompt = mcp_handler.create_analytical_prompt(question, schema)
        try:
            return self.llm.invoke(prompt)
        except Exception as e:
            logging.error(f"Failed to generate analytical fallback: {e}")
            return "Could not generate an analytical response due to an internal error."

    def process_question(self, question: str) -> str:
        """The main workflow for processing a user's question."""
        mode = self._classify_mode(question)
        schema = self._get_schema_representation()
        value_hints = self._get_value_hints()

        if mode == "ANALYTICAL":
            sql_result, _ = self._run_sql_generation_flow(question, schema, value_hints)
            data_samples = self._format_response(sql_result, "TABLE") if sql_result else ""
            analytical_prompt = mcp_handler.create_analytical_prompt(question, schema, data_samples)
            try:
                return self.llm.invoke(analytical_prompt)
            except Exception as e:
                return f"Failed to generate an analytical response: {e}"

        sql_result, last_sql = self._run_sql_generation_flow(question, schema, value_hints)

        if sql_result and not sql_result["rows"]:
            logging.info("Query returned 0 rows. Attempting LIKE rewrite.")
            rewritten_sql = self._rewrite_query_with_like(last_sql)
            if rewritten_sql:
                validated_sql = self._validate_sql(rewritten_sql)
                if validated_sql:
                    try:
                        sql_result = self._execute_sql(validated_sql)
                    except SQLAlchemyError:
                        pass

        if not sql_result or not sql_result["rows"]:
            logging.info("No data found, providing analytical suggestion.")
            return self._get_analytical_fallback(
                f"The user asked '{question}', but the query returned no results. What are the likely causes and what could they try instead?",
                schema
            )

        return self._format_response(sql_result, mode)

    def _rewrite_query_with_like(self, sql: str) -> str | None:
        """Rewrites a simple 'col = "value"' query to 'col LIKE "%value%"'."""
        if not sql: return None
        try:
            expression = sqlglot.parse_one(sql, read="mysql")
            # Find all equality checks with string literals
            for eq in expression.find_all(sqlglot_exp.EQ):
                if isinstance(eq.right, sqlglot_exp.Literal) and eq.right.is_string:
                    # Transform A = 'B' into A LIKE '%B%'
                    eq.this.set('sql', 'LIKE')
                    eq.right.set('sql', f"'%{eq.right.this}%'")
                    logging.info(f"Rewrote SQL with LIKE: {expression.sql()}")
                    return expression.sql()
        except Exception:
            return None # If parsing/rewriting fails, do nothing
        return None

    def _pattern_based_fallbacks(self, question: str) -> str | None:
        """Generates SQL from simple, hardcoded regex patterns as a last resort."""
        patterns = {
            r"skills for ID (\d+)": r"SELECT s.skill_name, es.skill_level FROM skills s JOIN employee_skills es ON s.id = es.skill_id WHERE es.employee_id = \1;",
            r"top (\d+) highest salaries": r"SELECT e.first_name, e.last_name, e.salary, d.name AS department_name FROM employees e JOIN departments d ON e.department_id = d.id ORDER BY e.salary DESC LIMIT \1;"
        }
        for pattern, sql_template in patterns.items():
            match = re.search(pattern, question, re.IGNORECASE)
            if match:
                sql = re.sub(pattern, sql_template, question, flags=re.IGNORECASE)
                logging.info(f"Pattern-based fallback triggered. SQL: {sql}")
                return sql

        # Special case for "performance summary for <name>"
        match = re.search(r"performance summary for ([\w\s]+)", question, re.IGNORECASE)
        if match:
            name_parts = match.group(1).split()
            first_name = name_parts[0]
            last_name = name_parts[-1]
            sql = f"SELECT AVG(pr.rating) as average_rating FROM performance_reviews pr JOIN employees e ON pr.employee_id = e.id WHERE e.first_name LIKE '%{first_name}%' AND e.last_name LIKE '%{last_name}%'"
            logging.info(f"Pattern-based fallback triggered. SQL: {sql}")
            return sql

        return None

    def _run_sql_generation_flow(self, question: str, schema: str, value_hints: str) -> (dict | None, str | None):
        """Handles the logic of generating, validating, and executing SQL with retries."""
        sql_attempts = []
        last_error = None
        current_sql = None

        gen_prompt = mcp_handler.create_sql_generation_prompt(question, schema, value_hints)
        try:
            llm_response = self.llm.invoke(gen_prompt)
            current_sql = self._extract_sql(llm_response)
        except Exception as e:
            last_error = e

        if current_sql:
            sql_attempts.append(current_sql)
            logging.info(f"[SQL Attempts initial] {current_sql}")
            validated_sql = self._validate_sql(current_sql)
            if validated_sql:
                try:
                    return self._execute_sql(validated_sql), validated_sql
                except SQLAlchemyError as e:
                    last_error = e

        for i in range(2):
            if not sql_attempts or not last_error: break
            logging.info(f"Attempting SQL correction, retry #{i+1}")
            prompt = mcp_handler.create_sql_correction_prompt(question, schema, sql_attempts[-1], str(last_error))
            try:
                llm_response = self.llm.invoke(prompt)
                current_sql = self._extract_sql(llm_response)
            except Exception as e:
                last_error = e
                continue

            if current_sql:
                sql_attempts.append(current_sql)
                logging.info(f"[SQL Attempts retry {i+1}] {current_sql}")
                validated_sql = self._validate_sql(current_sql)
                if validated_sql:
                    try:
                        return self._execute_sql(validated_sql), validated_sql
                    except SQLAlchemyError as e:
                        last_error = e

        logging.error("All LLM-based SQL generation and correction attempts failed.")

        logging.info("Trying pattern-based fallbacks.")
        pattern_sql = self._pattern_based_fallbacks(question)
        if pattern_sql:
            validated_sql = self._validate_sql(pattern_sql)
            if validated_sql:
                try:
                    return self._execute_sql(validated_sql), validated_sql
                except SQLAlchemyError:
                    pass

        return None, sql_attempts[-1] if sql_attempts else None
