"""
Code Generator Agent
Converts implementation tasks into actual code files using Ollama.

Model: qwen2.5-coder (configurable via config.py)

Key design decision (per Refinements §13):
- The FULL system graph is always sent as context so the AI knows about
  other nodes (e.g., Backend knows Frontend's port).
- Output goes to a Virtual File System (VFS) buffer, NOT directly to disk.
"""

import json
import logging

import httpx

from config import OLLAMA_BASE_URL, CODER_MODEL
from models.schemas import (
    GeneratedFile,
    SystemGraph,
    Task,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a Code Generator AI. Given a task and the full system architecture graph, \
generate the necessary code files.

RULES:
1. Output ONLY valid JSON matching this schema:
   {"files": [{"path": "relative/path/to/file.ext", "content": "file content here"}]}
2. Use the GLOBAL GRAPH context to ensure compatibility between components \
   (e.g., matching ports, API endpoints, database schemas).
3. Generate production-ready, well-commented code.
4. Use the framework/language specified in the node config.
5. Include necessary configuration files (package.json, requirements.txt, etc.).
6. Paths must be relative to the project root.

Output ONLY the JSON object."""


async def generate_code_for_task(
    task: Task,
    graph: SystemGraph,
) -> list[GeneratedFile]:
    """
    Generate code files for a single task, with full graph context.

    Returns a list of GeneratedFile objects (VFS buffer).
    Files are NOT written to disk — the caller decides when to persist.
    """
    logger.info(f"Generating code for task: {task.id} — {task.description}")

    graph_json = graph.model_dump_json(indent=2)
    task_json = task.model_dump_json(indent=2)

    payload = {
        "model": CODER_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"GLOBAL SYSTEM GRAPH (for context):\n{graph_json}\n\n"
                    f"TASK TO IMPLEMENT:\n{task_json}\n\n"
                    f"Generate the code files for this task."
                ),
            },
        ],
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.2,     # Low temp for code accuracy
            "num_predict": 4096,
        },
    }

    async with httpx.AsyncClient(timeout=600.0) as client:
        response = await client.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
        )
        response.raise_for_status()

    result = response.json()
    raw_content = result["message"]["content"]
    logger.debug(f"Raw AI response: {raw_content[:500]}")

    code_data = json.loads(raw_content)
    files_raw = code_data.get("files", [])
    files = [GeneratedFile.model_validate(f) for f in files_raw]

    logger.info(f"Generated {len(files)} files for task {task.id}")
    return files
