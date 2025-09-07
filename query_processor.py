import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, cast

import pandas as pd
import sqlparse
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from langchain_ollama import ChatOllama

from vector import VectorStoreManager, RetrievedContext
import mcp_handler


LOGGER = logging.getLogger(__name__)


def ensure_dirs() -> None:
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("outputs/exports", exist_ok=True)
    os.makedirs("outputs/plots", exist_ok=True)
    os.makedirs("outputs/logs", exist_ok=True)


def create_engine_from_url(db_url: str) -> Engine:
    connect_args: Dict[str, Any] = {}
    if db_url.startswith("mysql+"):
        connect_args["connect_timeout"] = 10
        connect_args["charset"] = "utf8mb4"
    engine = create_engine(
        db_url,
        pool_pre_ping=True,
        pool_recycle=1800,
        pool_size=5,
        max_overflow=10,
        connect_args=connect_args,
    )
    return engine


def create_engine_from_env() -> Engine:
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL environment variable is not set.")
    return create_engine_from_url(db_url)


def _strip_code_fences(text_value: str) -> str:
    fenced = re.sub(r"^```(?:sql|json|\w+)?\n|\n```$", "", text_value.strip(), flags=re.IGNORECASE | re.MULTILINE)
    if fenced != text_value:
        return fenced.strip()
    return text_value.strip()


def _extract_sql_from_text(text_value: str) -> Optional[str]:
    pattern = re.compile(r"```sql\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)
    m = pattern.search(text_value)
    if m:
        return m.group(1).strip()
    # fallback: look for SELECT start
    start = re.search(r"\bSELECT\b", text_value, flags=re.IGNORECASE)
    if start:
        return text_value[start.start():].strip().rstrip("`")
    return None


def _parse_json_block(text_value: str) -> Optional[Dict[str, Any]]:
    try:
        cleaned = _strip_code_fences(text_value)
        return json.loads(cleaned)
    except Exception:
        return None


def _stringify_llm_content(value: Any) -> str:
    """Convert LLM response content into a plain string for parsing.

    Handles cases where the content can be a list of chunks or a dict.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: List[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if "text" in item and isinstance(item["text"], str):
                    parts.append(item["text"]) 
                elif "json" in item:
                    try:
                        parts.append(json.dumps(item["json"]))
                    except Exception:
                        parts.append(str(item["json"]))
                else:
                    parts.append(str(item))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    if isinstance(value, dict):
        if "text" in value and isinstance(value["text"], str):
            return value["text"]
        try:
            return json.dumps(value)
        except Exception:
            return str(value)
    # Fallback
    return str(value)


class SqlValidationError(Exception):
    pass


class SafeSqlExecutor:
    """Guards and executes SELECT-only SQL with LIMIT enforcement and timeouts."""

    def __init__(self, engine: Engine, default_limit: int = 50, max_limit: int = 1000, timeout_secs: int = 30) -> None:
        self.engine = engine
        self.default_limit = default_limit
        self.max_limit = max_limit
        self.timeout_secs = timeout_secs

    def validate_select_only(self, sql: str) -> None:
        stripped = sql.strip().rstrip(";")
        if ";" in stripped:
            raise SqlValidationError("Multiple statements are not allowed.")
        parsed = sqlparse.parse(stripped)
        if not parsed:
            raise SqlValidationError("Invalid SQL.")
        if len(parsed) != 1:
            raise SqlValidationError("Only a single SELECT statement is allowed.")
        stmt = parsed[0]
        # Allow WITH ... SELECT
        tokens = [t for t in stmt.tokens if not t.is_whitespace]
        keywords = " ".join([t.value.upper() for t in tokens if t.ttype in (sqlparse.tokens.Keyword, sqlparse.tokens.DML)])
        forbidden = [
            "INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER", "TRUNCATE", "REPLACE",
            "MERGE", "GRANT", "REVOKE", "CALL", "USE", "SET", "SHOW", "DESCRIBE", "EXPLAIN ",
        ]
        if any(f in keywords for f in forbidden):
            raise SqlValidationError("Only SELECT queries are allowed.")
        if not (" SELECT " in f" {keywords} " or keywords.strip().startswith("SELECT") or "WITH" in keywords):
            raise SqlValidationError("Query must be a SELECT.")

    def _clamp_or_inject_limit(self, sql: str) -> str:
        # crude but effective LIMIT detection and clamping
        limit_regex = re.compile(r"\bLIMIT\s+(\d+)(?:\s*,\s*(\d+))?\b", flags=re.IGNORECASE)
        def repl(m: re.Match) -> str:
            first = int(m.group(1))
            second = m.group(2)
            if second is not None:
                second_val = int(second)
                second_val = min(second_val, self.max_limit)
                return f"LIMIT {first}, {second_val}"
            clamped = min(first, self.max_limit)
            return f"LIMIT {clamped}"
        if limit_regex.search(sql):
            return limit_regex.sub(repl, sql)
        # inject default limit at the end
        sql_no_semicolon = sql.strip().rstrip(";")
        return f"{sql_no_semicolon} LIMIT {self.default_limit}"

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=0.5, min=0.5, max=2.0), reraise=True, retry=retry_if_exception_type(SQLAlchemyError))
    def execute_select(self, sql: str) -> Tuple[pd.DataFrame, str]:
        self.validate_select_only(sql)
        safe_sql = self._clamp_or_inject_limit(sql)
        LOGGER.info("Executing SQL: %s", safe_sql)
        with self.engine.connect() as conn:
            df = pd.read_sql(text(safe_sql), conn)
        return df, safe_sql

    def explain(self, sql: str) -> pd.DataFrame:
        self.validate_select_only(sql)
        explain_sql = f"EXPLAIN {sql.strip().rstrip(';')}"
        with self.engine.connect() as conn:
            df = pd.read_sql(text(explain_sql), conn)
        return df

    def get_schema_summary(self) -> Dict[str, Any]:
        from sqlalchemy import inspect
        inspector = inspect(self.engine)
        result: Dict[str, Any] = {"tables": []}
        for t in inspector.get_table_names():
            cols = inspector.get_columns(t)
            result["tables"].append({
                "name": t,
                "columns": [{"name": c.get("name"), "type": str(c.get("type"))} for c in cols],
            })
        return result

    def describe_table(self, table_name: str) -> Dict[str, Any]:
        from sqlalchemy import inspect
        inspector = inspect(self.engine)
        cols = inspector.get_columns(table_name)
        return {
            "name": table_name,
            "columns": [{"name": c.get("name"), "type": str(c.get("type"))} for c in cols],
        }

    def find_tables(self, pattern: str) -> List[str]:
        from sqlalchemy import inspect
        inspector = inspect(self.engine)
        names = inspector.get_table_names()
        regex = re.compile(pattern, re.IGNORECASE)
        return [n for n in names if regex.search(n)]

    def find_columns(self, pattern: str) -> List[Tuple[str, str]]:
        from sqlalchemy import inspect
        inspector = inspect(self.engine)
        regex = re.compile(pattern, re.IGNORECASE)
        matches: List[Tuple[str, str]] = []
        for t in inspector.get_table_names():
            for c in inspector.get_columns(t):
                name = c.get("name")
                if name and regex.search(name):
                    matches.append((t, name))
        return matches

    def distinct_values(self, table: str, column: str, limit: int = 50) -> List[Any]:
        sql = f"SELECT DISTINCT `{column}` AS val FROM `{table}` LIMIT :lim"
        with self.engine.connect() as conn:
            rows = conn.execute(text(sql), {"lim": min(limit, self.max_limit)}).fetchall()
        return [r[0] for r in rows]


@dataclass
class NL2SQLOutput:
    mode: str
    sql: Optional[str]
    table_markdown: Optional[str]
    short_answer: Optional[str]
    analysis: Optional[str]
    visualization_path: Optional[str]
    metadata: Dict[str, Any]


class QueryProcessor:
    """Coordinates RAG retrieval, intent detection, SQL generation, execution, and multi-mode formatting."""

    def __init__(
        self,
        engine: Engine,
        vector_manager: VectorStoreManager,
        model_name: str = "llama3.2",
        temperature: float = 0.0,
        default_limit: int = 50,
        max_limit: int = 1000,
        timeout_secs: int = 30,
    ) -> None:
        ensure_dirs()
        self.engine = engine
        self.vector_manager = vector_manager
        self.llm = ChatOllama(model=model_name, temperature=temperature)
        self.safe_exec = SafeSqlExecutor(engine, default_limit=default_limit, max_limit=max_limit, timeout_secs=timeout_secs)

    def _detect_mode(self, user_query: str, context: RetrievedContext, prefer_mode: Optional[str]) -> str:
        if prefer_mode:
            return prefer_mode
        sys_prompt, few_shots = mcp_handler.build_intent_prompt()
        input_block = {
            "question": user_query,
            "context": context.texts[:6],
        }
        prompt = sys_prompt + "\n\nExamples:\n" + "\n\n".join(few_shots) + "\n\nUser:\n" + json.dumps(input_block, ensure_ascii=False)
        try:
            resp = self.llm.invoke(prompt)
            raw = resp.content if hasattr(resp, "content") else resp
            txt: str = _stringify_llm_content(raw)
            data = _parse_json_block(txt)
            if data and "mode" in data:
                return data["mode"].strip().upper()
        except Exception as exc:
            LOGGER.warning("Mode detection failed: %s", exc)
        # heuristics fallback
        lower = user_query.lower()
        if any(w in lower for w in ["plot", "chart", "graph", "trend"]):
            return "VISUALIZATION"
        if any(w in lower for w in ["how many", "count", "total number", "avg", "average", "sum", "max", "min"]):
            return "SHORT_ANSWER"
        if any(w in lower for w in ["why", "insight", "explain", "compare", "versus", "vs "]):
            return "ANALYTICAL"
        return "TABLE"

    def _generate_sql(self, user_query: str, context: RetrievedContext, mode: str) -> Optional[str]:
        sys_prompt = mcp_handler.build_sql_prompt()
        ctx_snippets = "\n\n".join(context.texts[:8])
        payload = {
            "question": user_query,
            "mode": mode,
            "context": ctx_snippets,
        }
        prompt = sys_prompt + "\n\n" + json.dumps(payload, ensure_ascii=False)
        try:
            resp = self.llm.invoke(prompt)
            raw = resp.content if hasattr(resp, "content") else resp
            txt: str = _stringify_llm_content(raw)
            sql = _extract_sql_from_text(txt)
            return sql
        except Exception as exc:
            LOGGER.error("SQL generation failed: %s", exc)
            return None

    def _format_table(self, df: pd.DataFrame) -> str:
        if df is None or df.empty:
            return "(no rows)"
        return df.to_markdown(index=False)

    def _short_answer_from_df(self, df: pd.DataFrame) -> str:
        if df is None or df.empty:
            return "No result"
        # prioritize a column named value or count
        for name in df.columns:
            if name.lower() in {"value", "count", "total", "sum", "avg", "average"}:
                return str(df[name].iloc[0])
        return str(df.iloc[0, 0])

    def _maybe_visualize(self, df: pd.DataFrame, title: str) -> Optional[str]:
        if df is None or df.empty or df.shape[1] < 2:
            return None
        try:
            import matplotlib.pyplot as plt
            x_col = df.columns[0]
            y_candidates = [c for c in df.columns[1:] if pd.api.types.is_numeric_dtype(df[c])]
            if not y_candidates:
                return None
            y_col = y_candidates[0]
            fig, ax = plt.subplots(figsize=(8, 4.5))
            # choose line for datetime-like x
            if pd.api.types.is_datetime64_any_dtype(df[x_col]) or "date" in x_col.lower():
                ax.plot(df[x_col], df[y_col], marker="o")
            else:
                ax.bar(df[x_col].astype(str), df[y_col])
                ax.tick_params(axis='x', labelrotation=45)
            ax.set_title(title)
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            plt.tight_layout()
            path = os.path.join("outputs", "plots", f"viz_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}.png")
            plt.savefig(path)
            plt.close(fig)
            return path
        except Exception as exc:
            LOGGER.warning("Visualization failed: %s", exc)
            return None

    def _analysis_narrative(self, user_query: str, df: pd.DataFrame, sql: str, context: RetrievedContext) -> str:
        sys_prompt = mcp_handler.build_analytical_prompt()
        summary = {
            "question": user_query,
            "sql": sql,
            "columns": list(df.columns) if df is not None else [],
            "row_count": int(len(df)) if df is not None else 0,
            "sample_rows": df.head(5).to_dict(orient="records") if df is not None else [],
            "context": context.texts[:4],
        }
        prompt = sys_prompt + "\n\n" + json.dumps(summary, ensure_ascii=False)
        try:
            resp = self.llm.invoke(prompt)
            raw = resp.content if hasattr(resp, "content") else resp
            text_out: str = _stringify_llm_content(raw)
            return text_out
        except Exception as exc:
            LOGGER.warning("Analytical narration failed: %s", exc)
            return "Unable to generate analytical summary."

    def process(self, user_query: str, prefer_mode: Optional[str] = None, export: Optional[str] = None) -> NL2SQLOutput:
        context = self.vector_manager.similarity_search(user_query, top_k=8)
        mode = self._detect_mode(user_query, context, prefer_mode)
        sql = self._generate_sql(user_query, context, mode)
        df: Optional[pd.DataFrame] = None
        executed_sql = None
        if sql:
            try:
                df, executed_sql = self.safe_exec.execute_select(sql)
            except Exception as exc:
                LOGGER.warning("Initial SQL failed, attempting repair: %s", exc)
                # single-shot repair attempt
                repair_prompt = mcp_handler.build_sql_repair_prompt()
                payload = {
                    "question": user_query,
                    "failed_sql": sql,
                    "error": str(exc),
                    "context": context.texts[:6],
                }
                try:
                    resp = self.llm.invoke(repair_prompt + "\n\n" + json.dumps(payload, ensure_ascii=False))
                    raw = resp.content if hasattr(resp, "content") else resp
                    txt: str = _stringify_llm_content(raw)
                    repaired_sql = _extract_sql_from_text(txt)
                    if repaired_sql:
                        df, executed_sql = self.safe_exec.execute_select(repaired_sql)
                        sql = repaired_sql
                except Exception as exc2:
                    LOGGER.error("SQL repair failed: %s", exc2)

        table_md = self._format_table(df) if df is not None else None
        short_answer = self._short_answer_from_df(df) if df is not None else None
        viz_path: Optional[str] = None
        analysis: Optional[str] = None

        if mode in ("VISUALIZATION", "COMBO") and df is not None:
            viz_path = self._maybe_visualize(df, title=user_query)
        if mode in ("ANALYTICAL", "COMBO"):
            analysis = self._analysis_narrative(user_query, df if df is not None else pd.DataFrame(), sql or "", context)

        if export and df is not None:
            try:
                ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
                base = os.path.join("outputs", "exports", f"export_{ts}")
                if export.lower() == "csv":
                    df.to_csv(base + ".csv", index=False)
                elif export.lower() == "json":
                    df.to_json(base + ".json", orient="records")
            except Exception as exc:
                LOGGER.warning("Export failed: %s", exc)

        return NL2SQLOutput(
            mode=mode,
            sql=executed_sql or sql,
            table_markdown=table_md,
            short_answer=short_answer,
            analysis=analysis,
            visualization_path=viz_path,
            metadata={
                "row_count": int(len(df)) if df is not None else 0,
            },
        )


