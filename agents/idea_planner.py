"""
Idea Planner Agent
Converts a user's idea into a system architecture graph using Ollama.

Model: qwen2.5:7b-instruct (configurable via config.py)
"""

import json
import logging

import httpx

from config import OLLAMA_BASE_URL, PLANNER_MODEL
from models.schemas import SystemGraph

logger = logging.getLogger(__name__)

# ── Few-shot example for reliable JSON output ─────────────────────────
# (Per Refinements §13: prevent AI from inventing invalid node types)

FEW_SHOT_EXAMPLE = """\
Example input: "Build a SaaS for managing ideas with AI"

Example output:
```json
{
  "nodes": [
    {"id": "idea", "type": "idea", "label": "Idea Manager SaaS"},
    {"id": "frontend", "type": "frontend", "label": "Web Dashboard", "config": {"framework": "nextjs", "port": 3000}},
    {"id": "backend", "type": "backend", "label": "API Server", "config": {"framework": "fastapi", "language": "python", "port": 8000}},
    {"id": "database", "type": "database", "label": "PostgreSQL", "config": {"engine": "postgres", "port": 5432}}
  ],
  "edges": [
    {"source": "idea", "target": "frontend"},
    {"source": "frontend", "target": "backend"},
    {"source": "backend", "target": "database"}
  ]
}
```"""

SYSTEM_PROMPT = f"""\
You are a System Architect AI. Your job is to convert a user's idea into a \
system architecture graph in JSON format.

RULES:
1. Always output ONLY valid JSON (no markdown, no explanation).
2. Node "type" must be one of: idea, frontend, backend, database, api, service, cache, queue.
3. Every graph must have at least one "idea" node connected to the system.
4. Backend nodes must connect to at least one database or service node.
5. Include realistic "config" with framework, language, engine, or port as appropriate.
6. Use descriptive "label" values.

{FEW_SHOT_EXAMPLE}

Now generate a system graph for the user's idea. Output ONLY the JSON object."""


async def generate_graph(idea: str) -> SystemGraph:
    """
    Send the idea to Ollama and parse the response into a SystemGraph.

    Args:
        idea: The user's project idea as free-form text.

    Returns:
        A validated SystemGraph object.

    Raises:
        httpx.HTTPStatusError: If Ollama returns a non-200 status.
        json.JSONDecodeError: If AI output is not valid JSON.
        pydantic.ValidationError: If JSON doesn't match the schema.
    """
    logger.info(f"Generating graph for idea: {idea[:80]}...")

    payload = {
        "model": PLANNER_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": idea},
        ],
        "stream": False,
        "format": "json",          # Ollama JSON mode
        "options": {
            "temperature": 0.3,     # Low temp for structured output
            "num_predict": 2048,
        },
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
        )
        response.raise_for_status()

    result = response.json()
    raw_content = result["message"]["content"]
    logger.debug(f"Raw AI response: {raw_content[:500]}")

    # Parse and validate
    graph_data = json.loads(raw_content)
    graph = SystemGraph.model_validate(graph_data)

    logger.info(f"Graph generated: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
    return graph
