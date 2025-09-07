import asyncio
import json
import logging
import os
from typing import Any, Dict, Optional, Callable

from mcp.server import Server
from mcp.server.stdio import stdio_server
# Version-tolerant imports without symbol lookups to satisfy Pylance
try:
    import mcp.server.models as _mcp_models  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    _mcp_models = None  # type: ignore[assignment]
try:
    import mcp.types as _mcp_types  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    _mcp_types = None  # type: ignore[assignment]

# Resolve classes dynamically (may be None if not present)
InitializationOptions = (
    getattr(_mcp_models, "InitializationOptions", None) if _mcp_models else None
) or (
    getattr(_mcp_types, "InitializationOptions", None) if _mcp_types else None
)
NotificationOptions = (
    getattr(_mcp_models, "NotificationOptions", None) if _mcp_models else None
) or (
    getattr(_mcp_types, "NotificationOptions", None) if _mcp_types else None
)

from query_processor import SafeSqlExecutor, create_engine_from_env
from vector import VectorStoreManager


LOGGER = logging.getLogger(__name__)


def build_executor() -> SafeSqlExecutor:
    engine = create_engine_from_env()
    return SafeSqlExecutor(engine)


def build_vector() -> VectorStoreManager:
    return VectorStoreManager()


server = Server("mysql-nl2sql")


# ---- Core tool implementations (no decorators) ----
async def tool_health_check(_: Dict[str, Any]) -> Dict[str, Any]:
    return {"status": "ok"}


async def tool_get_schema(_: Dict[str, Any]) -> Dict[str, Any]:
    execu = build_executor()
    return execu.get_schema_summary()


async def tool_describe_table(args: Dict[str, Any]) -> Dict[str, Any]:
    table = args.get("table")
    if not isinstance(table, str) or not table:
        return {"error": "Missing required string argument: table"}
    execu = build_executor()
    return execu.describe_table(table)


async def tool_find_tables(args: Dict[str, Any]) -> Dict[str, Any]:
    pattern = args.get("pattern", ".*")
    if not isinstance(pattern, str):
        pattern = str(pattern)
    execu = build_executor()
    return {"matches": execu.find_tables(pattern)}


async def tool_find_columns(args: Dict[str, Any]) -> Dict[str, Any]:
    pattern = args.get("pattern", ".*")
    if not isinstance(pattern, str):
        pattern = str(pattern)
    execu = build_executor()
    return {"matches": execu.find_columns(pattern)}


async def tool_distinct_values(args: Dict[str, Any]) -> Dict[str, Any]:
    table = args.get("table")
    column = args.get("column")
    if not isinstance(table, str) or not isinstance(column, str) or not table or not column:
        return {"error": "Arguments 'table' and 'column' must be non-empty strings"}
    limit = int(args.get("limit", 50))
    execu = build_executor()
    vals = execu.distinct_values(table, column, limit)
    return {"values": vals}


async def tool_run_sql(args: Dict[str, Any]) -> Dict[str, Any]:
    query = args.get("query")
    if not isinstance(query, str) or not query:
        return {"error": "Missing required string argument: query"}
    execu = build_executor()
    df, safe_sql = execu.execute_select(query)
    rows = df.to_dict(orient="records")
    return {"sql": safe_sql, "row_count": len(df), "rows": rows[:50]}


async def tool_explain_sql(args: Dict[str, Any]) -> Dict[str, Any]:
    query = args.get("query")
    if not isinstance(query, str) or not query:
        return {"error": "Missing required string argument: query"}
    execu = build_executor()
    df = execu.explain(query)
    return {"rows": df.to_dict(orient="records")}


async def tool_export(args: Dict[str, Any]) -> Dict[str, Any]:
    query = args.get("query")
    fmt = args.get("fmt", "csv")
    if not isinstance(query, str) or not query:
        return {"error": "Missing required string argument: query"}
    if not isinstance(fmt, str):
        fmt = str(fmt)
    execu = build_executor()
    df, safe_sql = execu.execute_select(query)
    os.makedirs("outputs/exports", exist_ok=True)
    ts = os.path.getmtime(__file__)  # deterministic but fine
    base = os.path.join("outputs", "exports", f"export_{int(ts)}")
    path: Optional[str] = None
    if fmt.lower() == "csv":
        path = base + ".csv"
        df.to_csv(path, index=False)
    elif fmt.lower() == "json":
        path = base + ".json"
        df.to_json(path, orient="records")
    else:
        return {"error": f"Unsupported format: {fmt}"}
    return {"sql": safe_sql, "path": path, "row_count": len(df)}


SAVED_QUERIES = os.path.join("outputs", "saved_queries.json")


def _load_saved() -> Dict[str, Dict[str, Any]]:
    if not os.path.isfile(SAVED_QUERIES):
        return {}
    try:
        with open(SAVED_QUERIES, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
            # migrate from old list format if any
            if isinstance(data, list):
                return {item.get("name", f"q{i}"): item for i, item in enumerate(data)}
    except Exception:
        pass
    return {}


def _save_saved(data: Dict[str, Dict[str, Any]]) -> None:
    os.makedirs("outputs", exist_ok=True)
    with open(SAVED_QUERIES, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


async def tool_save_query(args: Dict[str, Any]) -> Dict[str, Any]:
    name = args.get("name")
    query = args.get("query")
    description = args.get("description")
    if not isinstance(name, str) or not name:
        return {"error": "Missing required string argument: name"}
    if not isinstance(query, str) or not query:
        return {"error": "Missing required string argument: query"}
    execu = build_executor()
    execu.validate_select_only(query)
    data = _load_saved()
    data[name] = {"query": query, "description": description or ""}
    _save_saved(data)
    try:
        VectorStoreManager().add_past_query(query, result_summary=description)
    except Exception:
        pass
    return {"status": "saved", "name": name}


async def tool_run_saved_query(args: Dict[str, Any]) -> Dict[str, Any]:
    name = args.get("name")
    if not isinstance(name, str) or not name:
        return {"error": "Missing required string argument: name"}
    data = _load_saved()
    item = data.get(name)
    if not item:
        return {"error": f"No saved query named {name}"}
    execu = build_executor()
    df, safe_sql = execu.execute_select(item["query"])
    return {"sql": safe_sql, "row_count": len(df), "rows": df.to_dict(orient="records")[:50]}


# ---- Registry and compatibility layers ----
TOOLS: Dict[str, Callable[[Dict[str, Any]], Any]] = {
    "health_check": tool_health_check,
    "get_schema": tool_get_schema,
    "describe_table": tool_describe_table,
    "find_tables": tool_find_tables,
    "find_columns": tool_find_columns,
    "distinct_values": tool_distinct_values,
    "run_sql": tool_run_sql,
    "explain_sql": tool_explain_sql,
    "export": tool_export,
    "save_query": tool_save_query,
    "run_saved_query": tool_run_saved_query,
}


def _list_tools_payload() -> Dict[str, Any]:
    return {
        "tools": [
            {"name": "health_check", "description": "Health probe", "inputSchema": {"type": "object", "properties": {}, "required": []}},
            {"name": "get_schema", "description": "Return schema summary", "inputSchema": {"type": "object", "properties": {}, "required": []}},
            {"name": "describe_table", "description": "Describe a table's columns", "inputSchema": {"type": "object", "properties": {"table": {"type": "string"}}, "required": ["table"]}},
            {"name": "find_tables", "description": "Find tables matching a pattern", "inputSchema": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]}},
            {"name": "find_columns", "description": "Find columns matching a pattern", "inputSchema": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]}},
            {"name": "distinct_values", "description": "List distinct values of a column", "inputSchema": {"type": "object", "properties": {"table": {"type": "string"}, "column": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["table", "column"]}},
            {"name": "run_sql", "description": "Run a safe SELECT query", "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
            {"name": "explain_sql", "description": "Explain a safe SELECT query", "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
            {"name": "export", "description": "Export query results to file", "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}, "fmt": {"type": "string", "enum": ["csv", "json"]}}, "required": ["query"]}},
            {"name": "save_query", "description": "Save a named SELECT query", "inputSchema": {"type": "object", "properties": {"name": {"type": "string"}, "query": {"type": "string"}, "description": {"type": "string"}}, "required": ["name", "query"]}},
            {"name": "run_saved_query", "description": "Execute a saved query by name", "inputSchema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}},
        ]
    }


if hasattr(server, "method"):
    @server.method("tools/list")  # type: ignore[attr-defined]
    async def _tools_list() -> Dict[str, Any]:
        return _list_tools_payload()

    @server.method("tools/call")  # type: ignore[attr-defined]
    async def _tools_call(name: str, arguments: Optional[Dict[str, Any]] = None, **_: Any) -> Dict[str, Any]:
        func = TOOLS.get(name)
        if func is None:
            return {"content": [{"type": "text", "text": f"Unknown tool: {name}"}], "is_error": True}
        result = await func(arguments or {})
        return {"content": [{"type": "json", "json": result}], "is_error": False}


async def main() -> None:
    async with stdio_server() as (read, write):
        init_opts: Any
        # Build capabilities robustly across versions
        caps: Any = None
        try:
            if hasattr(server, "get_capabilities"):
                if NotificationOptions is not None:
                    caps = server.get_capabilities(notification_options=NotificationOptions(), experimental_capabilities={})
                else:
                    caps = server.get_capabilities()
        except Exception:
            caps = None
        if caps is None:
            caps = {}

        if InitializationOptions is not None:
            try:
                init_opts = InitializationOptions(server_name="mysql-nl2sql", server_version="0.1.0", capabilities=caps)
            except Exception:
                init_opts = {"server_name": "mysql-nl2sql", "server_version": "0.1.0", "capabilities": caps}
        else:
            init_opts = {"server_name": "mysql-nl2sql", "server_version": "0.1.0", "capabilities": caps}

        await server.run(read, write, init_opts)  # type: ignore[arg-type]


if __name__ == "__main__":
    asyncio.run(main())

