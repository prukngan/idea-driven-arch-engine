"""
Graph Planner Agent
Converts a system graph into an ordered list of implementation tasks.

Model: qwen2.5:7b-instruct (configurable via config.py)
"""

import json
import logging

import httpx

from config import OLLAMA_BASE_URL, PLANNER_MODEL
from models.schemas import SystemGraph, TaskPlan

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a Project Planner AI. Given a system architecture graph (JSON), \
generate an ordered list of implementation tasks.

RULES:
1. Output ONLY valid JSON matching this schema:
   {"tasks": [{"id": "task_1", "description": "...", "node_id": "...", "dependencies": ["task_0"]}]}
2. Each task must reference a valid "node_id" from the graph.
3. Tasks must be ordered by dependency — infrastructure first, then services, then frontend.
4. Use clear, actionable descriptions (e.g., "Create PostgreSQL schema with users and ideas tables").
5. Dependencies must reference previous task IDs only.
6. Every node in the graph should have at least one task.

Output ONLY the JSON object."""


async def generate_tasks(graph: SystemGraph) -> TaskPlan:
    """
    Convert a system graph into an implementation plan.

    The full graph JSON is included in the prompt so the AI has
    global context about all nodes (per Refinements §13).
    """
    logger.info(f"Generating tasks for graph with {len(graph.nodes)} nodes...")

    # Send full graph as context
    graph_json = graph.model_dump_json(indent=2)

    payload = {
        "model": PLANNER_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Generate implementation tasks for this system graph:\n\n{graph_json}"},
        ],
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.3,
            "num_predict": 2048,
        },
    }

    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
        )
        response.raise_for_status()

    result = response.json()
    raw_content = result["message"]["content"]
    logger.debug(f"Raw AI response: {raw_content[:500]}")

    plan_data = json.loads(raw_content)
    plan = TaskPlan.model_validate(plan_data)

    logger.info(f"Plan generated: {len(plan.tasks)} tasks")
    return plan
