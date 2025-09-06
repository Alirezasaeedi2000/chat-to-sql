import asyncio
import json
import logging
import os
from typing import Any, Dict, Optional

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import JSONResponse

from query_processor import SafeSqlExecutor, create_engine_from_env
from vector import VectorStoreManager


LOGGER = logging.getLogger(__name__)


def build_executor() -> SafeSqlExecutor:
    engine = create_engine_from_env()
    return SafeSqlExecutor(engine)


def build_vector() -> VectorStoreManager:
    return VectorStoreManager()


server = Server("mysql-nl2sql")


@server.tool()
async def health_check() -> JSONResponse:
    return JSONResponse(content={"status": "ok"})


@server.tool()
async def get_schema() -> JSONResponse:
    execu = build_executor()
    return JSONResponse(content=execu.get_schema_summary())


@server.tool()
async def describe_table(table: str) -> JSONResponse:
    execu = build_executor()
    return JSONResponse(content=execu.describe_table(table))


@server.tool()
async def find_tables(pattern: str) -> JSONResponse:
    execu = build_executor()
    return JSONResponse(content={"matches": execu.find_tables(pattern)})


@server.tool()
async def find_columns(pattern: str) -> JSONResponse:
    execu = build_executor()
    return JSONResponse(content={"matches": execu.find_columns(pattern)})


@server.tool()
async def distinct_values(table: str, column: str, limit: int = 50) -> JSONResponse:
    execu = build_executor()
    vals = execu.distinct_values(table, column, limit)
    return JSONResponse(content={"values": vals})


@server.tool()
async def run_sql(query: str) -> JSONResponse:
    execu = build_executor()
    df, safe_sql = execu.execute_select(query)
    rows = df.to_dict(orient="records")
    return JSONResponse(content={"sql": safe_sql, "row_count": len(df), "rows": rows[:50]})


@server.tool()
async def explain_sql(query: str) -> JSONResponse:
    execu = build_executor()
    df = execu.explain(query)
    return JSONResponse(content={"rows": df.to_dict(orient="records")})


@server.tool()
async def export(query: str, fmt: str = "csv") -> JSONResponse:
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
        return JSONResponse(content={"error": f"Unsupported format: {fmt}"})
    return JSONResponse(content={"sql": safe_sql, "path": path, "row_count": len(df)})


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


@server.tool()
async def save_query(name: str, query: str, description: Optional[str] = None) -> JSONResponse:
    execu = build_executor()
    execu.validate_select_only(query)
    data = _load_saved()
    data[name] = {"query": query, "description": description or ""}
    _save_saved(data)
    # also add to vector store for future retrieval
    try:
        VectorStoreManager().add_past_query(query, result_summary=description)
    except Exception:
        pass
    return JSONResponse(content={"status": "saved", "name": name})


@server.tool()
async def run_saved_query(name: str) -> JSONResponse:
    data = _load_saved()
    item = data.get(name)
    if not item:
        return JSONResponse(content={"error": f"No saved query named {name}"})
    execu = build_executor()
    df, safe_sql = execu.execute_select(item["query"])
    return JSONResponse(content={"sql": safe_sql, "row_count": len(df), "rows": df.to_dict(orient="records")[:50]})


async def main() -> None:
    async with stdio_server() as (read, write):
        await server.run(read, write)


if __name__ == "__main__":
    asyncio.run(main())

