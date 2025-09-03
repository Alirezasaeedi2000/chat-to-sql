"""
Prompt Engineering for the NL-to-SQL Assistant.

This module contains functions that construct specific prompts for different
stages of the query generation process:
1.  Mode Classification: Determines if the user wants a table, a single answer, or an analysis.
2.  SQL Generation: Creates the initial SQL query from the user's question.
3.  SQL Correction: Attempts to fix a failed SQL query given an error message.
4.  Analytical Response: Generates a narrative summary when SQL is not suitable.
"""

def create_mode_classification_prompt(question: str) -> str:
    """
    Creates a prompt to classify the user's question into one of three modes.
    """
    return f"""
Classify the user's question into one of three categories: TABLE, SHORT_ANSWER, or ANALYTICAL.
- TABLE: For questions that expect a list of items, multiple rows/columns. E.g., "list all employees", "show me skills for user X".
- SHORT_ANSWER: For questions that expect a single value, like a number, a name, or a date. E.g., "how many employees?", "what is the CEO's name?".
- ANALYTICAL: For questions that are broad, ask for "recommendations", "insights", "summaries", or "why". E.g., "recommend a department for a new project", "summarize sales performance".

Respond with *only* the category name and nothing else.

Question: "how many people in sales?"
Mode: SHORT_ANSWER

Question: "list all projects and their budgets"
Mode: TABLE

Question: "give me a summary of Q3 performance for the engineering team"
Mode: ANALYTICAL

Question: "who is the highest paid employee?"
Mode: SHORT_ANSWER

Question: "show me details for the highest paid employee"
Mode: TABLE

Question: "recommend a course of action for the marketing department"
Mode: ANALYTICAL

Question: "{question}"
Mode:"""


def create_sql_generation_prompt(question: str, schema: str, value_hints: str = "", task_hints: str = "") -> str:
    """
    Creates the main few-shot prompt for generating a SQL query.
    """
    prompt = f"""
You are a MySQL expert. Your task is to convert a user's question into a single, read-only MySQL `SELECT` statement.
Pay close attention to the provided database schema and value hints.

**Rules:**
1.  **Single Statement:** You MUST generate a single `SELECT` statement.
2.  **Read-Only:** You MUST NOT generate any DDL (CREATE, ALTER, DROP) or DML (INSERT, UPDATE, DELETE) statements.
3.  **FROM Clause:** Your query MUST include a `FROM` clause. Do not generate queries like `SELECT 123;`.
4.  **LIMIT Clause:** A `LIMIT` will be added automatically. You may add your own `LIMIT` if a specific number is requested (e.g., "top 5").
5.  **Fuzzy Matching:** Use `LIKE` with '%' for case-insensitive matching on text columns (names, departments) unless an exact ID is provided.
6.  **Safe Aggregates:** Use `COALESCE` around aggregate functions like `COUNT`, `SUM`, `AVG` to ensure `0` is returned instead of `NULL` for empty results.
7.  **Output Format:** Wrap the final SQL query in a single markdown code block: ````sql ... ````.

**Database Schema:**
```
{schema}
```

**Value Hints (sample values from key columns):**
```
{value_hints}
```

**Task Hints (optional context):**
```
{task_hints}
```

---
**Few-Shot Examples:**

**Question:** how many employees work in the sales department?
**SQL:**
````sql
SELECT COALESCE(COUNT(e.id), 0) FROM employees e JOIN departments d ON e.department_id = d.id WHERE d.name LIKE '%sales%'
````

**Question:** tell me about employee ID 1079 and their skills
**SQL:**
````sql
SELECT e.first_name, e.last_name, e.email, s.skill_name, es.skill_level FROM employees e LEFT JOIN employee_skills es ON e.id = es.employee_id LEFT JOIN skills s ON es.skill_id = s.id WHERE e.id = 1079
````

**Question:** list omid shahbazi's details
**SQL:**
````sql
SELECT * FROM employees WHERE first_name LIKE '%omid%' AND last_name LIKE '%shahbazi%'
````

**Question:** show me the top 5 highest salaries with department names
**SQL:**
````sql
SELECT e.first_name, e.last_name, e.salary, d.name AS department_name FROM employees e JOIN departments d ON e.department_id = d.id ORDER BY e.salary DESC LIMIT 5
````

**Question:** what is the performance summary for Omid Shahbazi?
**SQL:**
````sql
SELECT AVG(pr.rating) AS average_rating FROM performance_reviews pr JOIN employees e ON pr.employee_id = e.id WHERE e.first_name LIKE '%Omid%' AND e.last_name LIKE '%Shahbazi%'
````

**Question:** For HR, which departments have the lowest average performance ratings?
**SQL:**
````sql
SELECT d.name, AVG(pr.rating) AS avg_rating FROM departments d JOIN employees e ON d.id = e.department_id JOIN performance_reviews pr ON e.id = pr.employee_id GROUP BY d.name ORDER BY avg_rating ASC LIMIT 5
````

**Question:** how many projects are in the 'New Ventures' department?
**SQL:**
````sql
SELECT COALESCE(COUNT(p.id), 0) FROM projects p JOIN departments d ON p.department_id = d.id WHERE d.name LIKE '%New Ventures%'
````
---

**New Task:**

**Question:** {question}
**SQL:**
"""
    return prompt.strip()


def create_sql_correction_prompt(question: str, schema: str, sql_attempt: str, error_message: str) -> str:
    """
    Creates a few-shot prompt to correct a SQL query that failed.
    """
    prompt = f"""
You are a SQL debugging expert. The user's question and the database schema are provided below.
A previous SQL query attempt failed. Your task is to analyze the error message and the query, then provide a corrected version.

**Rules:**
1.  The corrected query must be a single, read-only `SELECT` statement.
2.  Fix the error based on the provided message (e.g., incorrect column/table names, syntax errors, bad joins).
3.  Wrap the final corrected SQL in a single markdown code block: ````sql ... ````.

**Database Schema:**
```
{schema}
```

**User Question:** {question}

---
**Few-Shot Examples:**

**Failed SQL:** `SELECT e.name FROM employees e JOIN departments d ON e.depart_id = d.id`
**Error Message:** `(MySQLdb.OperationalError) (1054, "Unknown column 'e.depart_id' in 'on clause'")`
**Corrected SQL:**
````sql
SELECT e.first_name, e.last_name FROM employees e JOIN departments d ON e.department_id = d.id
````

**Failed SQL:** `SELECT name FROM employes WHERE salary > 50000`
**Error Message:** `(MySQLdb.ProgrammingError) (1146, "Table 'test_01.employes' doesn't exist")`
**Corrected SQL:**
````sql
SELECT first_name, last_name FROM employees WHERE salary > 50000
````

**Failed SQL:** `SELECT d.name, COUNT(e.id) FROM departments d JOIN employees e ON d.id = e.department_id`
**Error Message:** `(MySQLdb.OperationalError) (1140, "In aggregated query without GROUP BY, expression #1 of SELECT list contains nonaggregated column 'test_01.d.name'; this is incompatible with sql_mode=only_full_group_by")`
**Corrected SQL:**
````sql
SELECT d.name, COUNT(e.id) FROM departments d JOIN employees e ON d.id = e.department_id GROUP BY d.name
````
---

**New Task:**

**Failed SQL:** `{sql_attempt}`
**Error Message:** `{error_message}`
**Corrected SQL:**
"""
    return prompt.strip()


def create_analytical_prompt(question: str, schema: str, data_samples: str = "") -> str:
    """
    Creates a prompt to generate a consultant-style analytical response.
    """
    prompt = f"""
You are a professional business consultant and data analyst.
You are given a user's question, the database schema, and optionally some sample data.
Your task is to provide a structured, insightful narrative that directly addresses the user's question.

**Rules:**
1.  **Do not** output SQL code.
2.  **Do not** just describe the data or schema. Provide actionable interpretations.
3.  Structure your response using the following markdown headings:
    - `### Insights`: Key findings and observations from the data or schema.
    - `### Potential Gaps`: What information seems to be missing to give a complete answer? What data would be needed?
    - `### Risks & Mitigations`: Potential risks highlighted by the analysis and how to address them.
    - `### Recommendations`: Actionable next steps for the user (e.g., for HR, for a department manager).

**Database Schema:**
```
{schema}
```

**Sample Data (if available):**
```
{data_samples}
```

**User Question:** {question}

**Your Analysis:**
"""
    return prompt.strip()
