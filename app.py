import argparse
import json
import logging
import os
import sys
from typing import Optional

from query_processor import QueryProcessor, create_engine_from_url
from vector import VectorStoreManager


def setup_logging() -> None:
    os.makedirs("outputs/logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("outputs/logs/nl2sql.log", encoding="utf-8"),
        ],
    )


def run_cli(db_url: str, query: Optional[str], mode: Optional[str], export: Optional[str], build_index: bool) -> int:
    setup_logging()
    logger = logging.getLogger("cli")

    engine = create_engine_from_url(db_url)
    vector = VectorStoreManager()
    if build_index:
        added = vector.upsert_schema_and_samples(engine, sample_rows_per_table=5)
        logger.info("Indexed %d items into vector store", added)

    qp = QueryProcessor(engine, vector)

    if query:
        result = qp.process(query, prefer_mode=mode, export=export)
        print("Mode:", result.mode)
        if result.sql:
            print("SQL:")
            print(result.sql)
        if result.table_markdown and result.mode in ("TABLE", "COMBO", "VISUALIZATION"):
            print("\nTable:")
            print(result.table_markdown)
        if result.short_answer and result.mode in ("SHORT_ANSWER", "COMBO"):
            print("\nShort Answer:")
            print(result.short_answer)
        if result.analysis and result.mode in ("ANALYTICAL", "COMBO"):
            print("\nAnalytical:")
            print(result.analysis)
        if result.visualization_path:
            print("\nVisualization saved to:", result.visualization_path)
        return 0

    print("Interactive mode. Type 'exit' to quit.")
    while True:
        try:
            user_q = input("NL> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user_q.lower() in {"exit", "quit"}:
            break
        if not user_q:
            continue
        res = qp.process(user_q, prefer_mode=mode, export=export)
        print("Mode:", res.mode)
        if res.sql:
            print("SQL:\n", res.sql)
        if res.table_markdown and res.mode in ("TABLE", "COMBO", "VISUALIZATION"):
            print("\nTable:\n", res.table_markdown)
        if res.short_answer and res.mode in ("SHORT_ANSWER", "COMBO"):
            print("\nShort Answer:\n", res.short_answer)
        if res.analysis and res.mode in ("ANALYTICAL", "COMBO"):
            print("\nAnalytical:\n", res.analysis)
        if res.visualization_path:
            print("\nVisualization saved to:", res.visualization_path)
    return 0


def run_server() -> int:
    setup_logging()
    import subprocess
    # Run the MCP stdio server in this process
    from mcp_sql_server import main as mcp_main
    import asyncio
    asyncio.run(mcp_main())
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Natural Language to SQL Assistant for MySQL")
    parser.add_argument("--db-url", help="SQLAlchemy DB URL, e.g., mysql+pymysql://user:pass@host:3306/db", required=False)
    parser.add_argument("--query", help="One-shot NL query", required=False)
    parser.add_argument("--mode", help="Force mode: TABLE|SHORT_ANSWER|ANALYTICAL|VISUALIZATION|COMBO", required=False)
    parser.add_argument("--export", help="Export format: csv|json", required=False)
    parser.add_argument("--build-index", help="Build or refresh the vector index", action="store_true")
    parser.add_argument("--server", help="Run MCP stdio server", action="store_true")
    args = parser.parse_args()

    if args.server:
        return run_server()

    db_url = args.db_url or os.environ.get("DATABASE_URL")
    if not db_url:
        print("Please provide --db-url or set DATABASE_URL.")
        return 2

    return run_cli(db_url, args.query, args.mode, args.export, args.build_index)


if __name__ == "__main__":
    raise SystemExit(main())

