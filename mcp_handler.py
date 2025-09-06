from typing import List, Tuple


def build_intent_prompt() -> Tuple[str, List[str]]:
    system = (
        "You are an expert MySQL analyst. Detect the user's intent and output ONLY a JSON object with keys: "
        "mode (one of: TABLE, SHORT_ANSWER, ANALYTICAL, VISUALIZATION, COMBO) and reason."
    )
    few_shots = [
        # TABLE
        '{"mode": "TABLE", "reason": "User wants a tabular listing or detail."}',
        # SHORT_ANSWER
        '{"mode": "SHORT_ANSWER", "reason": "User asks for a scalar or count/sum/avg."}',
        # ANALYTICAL
        '{"mode": "ANALYTICAL", "reason": "User seeks insights, comparisons, or causal analysis."}',
        # VISUALIZATION
        '{"mode": "VISUALIZATION", "reason": "User requests a chart/plot/graph/trend."}',
        # COMBO
        '{"mode": "COMBO", "reason": "User wants multiple views: table + analysis or plot."}',
    ]
    return system, few_shots


def build_sql_prompt() -> str:
    return (
        "You convert a question into a single, safe MySQL SELECT query.\n"
        "Rules:\n"
        "- Use only SELECT statements. No DDL/DML/transactions.\n"
        "- Prefer explicit column lists.\n"
        "- Always include LIMIT if not provided.\n"
        "- Do not guess columns; rely on provided context.\n"
        "- Use backticks around identifiers.\n"
        "- Output ONLY the SQL inside ```sql fences.\n"
    )


def build_sql_repair_prompt() -> str:
    return (
        "The previous SQL failed. Produce a corrected MySQL SELECT.\n"
        "Constraints:\n"
        "- Single statement, SELECT-only.\n"
        "- Keep semantics close to the question.\n"
        "- Use backticks around identifiers.\n"
        "- Output ONLY the SQL inside ```sql fences.\n"
    )


def build_analytical_prompt() -> str:
    examples = (
        "Examples of analytical outputs (short, structured):\n\n"
        "Insights:\n- Revenue grew MoM except in Feb.\n- Top 3 customers contributed 48% of sales.\n\n"
        "Gaps:\n- No data for refunds.\n\n"
        "Risks:\n- Concentration risk in two regions.\n\n"
        "Recommendations:\n- Investigate Feb dip; diversify regions; upsell to long-tail customers.\n"
    )
    return (
        "You are a concise data analyst. Given a SQL result summary and context, write a short, structured analysis.\n"
        "Structure: Insights, Gaps, Risks, Recommendations.\n"
        "Avoid hallucination and clearly state when data is insufficient.\n\n" + examples
    )


