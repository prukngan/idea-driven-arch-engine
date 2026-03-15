"""
AI System Builder — Backend API
================================
FastAPI application serving the Idea → Graph → Tasks → Code pipeline.

Run with:
    uvicorn main:app --reload --port 8000
"""

import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from config import OLLAMA_BASE_URL, API_HOST, API_PORT
from models.schemas import (
    IdeaRequest,
    GenerateGraphResponse,
    GenerateTasksRequest,
    GenerateTasksResponse,
    GenerateCodeRequest,
    GenerateCodeResponse,
    GeneratedFile,
)
from agents import idea_planner, graph_planner, code_generator
from tools.file_writer import VirtualFileSystem

# ── Logging ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-20s │ %(levelname)-7s │ %(message)s",
)
logger = logging.getLogger("idea-platform")

# ── Global VFS instance ──────────────────────────────────────────────
vfs = VirtualFileSystem()


# ── Lifespan (startup/shutdown) ──────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Check Ollama connectivity on startup."""
    logger.info("🚀 AI System Builder starting up...")
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            models = [m["name"] for m in resp.json().get("models", [])]
            logger.info(f"✅ Ollama connected — {len(models)} models available: {models}")
    except Exception as e:
        logger.warning(f"⚠️  Ollama not reachable at {OLLAMA_BASE_URL}: {e}")
        logger.warning("   Backend will start, but AI features require Ollama.")
    yield
    logger.info("👋 AI System Builder shutting down.")


# ── App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI System Builder",
    description="Convert ideas into software systems: Idea → Graph → Tasks → Code",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — allow Vue dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",    # Vite default
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health Check ──────────────────────────────────────────────────────

@app.get("/api/health")
async def health_check():
    """Health check + Ollama connectivity status."""
    ollama_ok = False
    models = []
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            models = [m["name"] for m in resp.json().get("models", [])]
            ollama_ok = True
    except Exception:
        pass

    return {
        "status": "healthy",
        "ollama": {
            "connected": ollama_ok,
            "url": OLLAMA_BASE_URL,
            "models": models,
        },
    }


# ── Pipeline Endpoints ───────────────────────────────────────────────

@app.post("/api/generate-graph", response_model=GenerateGraphResponse)
async def generate_graph(request: IdeaRequest):
    """
    Step 1-2: Convert an idea into a system architecture graph.
    """
    try:
        graph = await idea_planner.generate_graph(request.idea)
        return GenerateGraphResponse(graph=graph)
    except httpx.HTTPStatusError as e:
        logger.error(f"Ollama HTTP error: {e}")
        raise HTTPException(status_code=502, detail="AI engine returned an error")
    except Exception as e:
        logger.error(f"Graph generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate-tasks", response_model=GenerateTasksResponse)
async def generate_tasks(request: GenerateTasksRequest):
    """
    Step 4: Convert a system graph into implementation tasks.
    """
    try:
        plan = await graph_planner.generate_tasks(request.graph)
        return GenerateTasksResponse(plan=plan)
    except Exception as e:
        logger.error(f"Task generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate-code", response_model=GenerateCodeResponse)
async def generate_code(request: GenerateCodeRequest):
    """
    Step 5: Generate code for all tasks in the plan.

    Files are buffered in VFS — not written to disk yet.
    Use POST /api/flush-files to persist.
    """
    vfs.clear()
    all_files: list[GeneratedFile] = []

    try:
        for task in request.plan.tasks:
            files = await code_generator.generate_code_for_task(
                task=task,
                graph=request.graph,  # Full graph context (Refinements §13)
            )
            for f in files:
                vfs.add_file(f.path, f.content)
            all_files.extend(files)

        return GenerateCodeResponse(
            files=all_files,
            message=f"Generated {len(all_files)} files (buffered in VFS, not yet persisted)",
        )
    except Exception as e:
        logger.error(f"Code generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/vfs/files")
async def list_vfs_files():
    """List all files currently buffered in the VFS."""
    return {"files": vfs.list_files(), "count": len(vfs)}


@app.post("/api/vfs/flush")
async def flush_vfs():
    """
    Persist all VFS-buffered files to disk.
    This is the user's explicit "confirm write" action.
    """
    try:
        written = vfs.flush()
        vfs.clear()
        return {"written": written, "count": len(written)}
    except Exception as e:
        logger.error(f"VFS flush failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ── Run ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=API_HOST, port=API_PORT, reload=True)
