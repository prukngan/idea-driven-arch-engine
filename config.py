"""
Application configuration for the AI System Builder backend.
All settings are local-first — no cloud dependencies.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Ollama Configuration ─────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Model used for idea → system graph planning
PLANNER_MODEL = os.getenv("PLANNER_MODEL", "qwen2.5:7b-instruct")

# Model used for task → code generation
CODER_MODEL = os.getenv("CODER_MODEL", "qwen2.5-coder")

# ── Server Configuration ─────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# ── Safety / Sandbox ─────────────────────────────────────────────────
# Root directory for generated project files (sandboxed)
OUTPUT_ROOT = os.getenv("OUTPUT_ROOT", os.path.join(os.getcwd(), "generated_projects"))
