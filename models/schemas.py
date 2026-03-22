"""
Pydantic models for the AI System Builder.
These schemas define the data structures used across the entire pipeline:
  Idea → Graph → Tasks → Code
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ── Enums ─────────────────────────────────────────────────────────────

class NodeType(str, Enum):
    """Allowed node types in a system graph."""
    IDEA = "idea"
    FRONTEND = "frontend"
    BACKEND = "backend"
    DATABASE = "database"
    API = "api"
    SERVICE = "service"
    CACHE = "cache"
    QUEUE = "queue"


class NodeStatus(str, Enum):
    """Visual status of a node on the canvas."""
    IDLE = "idle"
    THINKING = "thinking"        # 🟡 AI is processing
    GENERATING = "generating"    # 🔵 Code is being generated
    COMPLETE = "complete"        # 🟢 Done
    ERROR = "error"              # 🔴 Failed


class TaskStatus(str, Enum):
    """Status of an implementation task."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    FAILED = "failed"


# ── Graph Models ──────────────────────────────────────────────────────

class NodeConfig(BaseModel):
    """Flexible configuration for a graph node."""
    framework: str | None = None
    engine: str | None = None
    language: str | None = None
    port: int | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class GraphNode(BaseModel):
    """A single node in the system graph."""
    id: str
    type: NodeType
    label: str | None = None
    config: NodeConfig = Field(default_factory=NodeConfig)
    status: NodeStatus = NodeStatus.IDLE

    # Canvas position (for Vue Flow)
    position_x: float = 0.0
    position_y: float = 0.0


class GraphEdge(BaseModel):
    """A directed edge between two nodes."""
    source: str
    target: str
    label: str | None = None


class SystemGraph(BaseModel):
    """The complete system architecture graph."""
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)


# ── Task Models ───────────────────────────────────────────────────────

class Task(BaseModel):
    """A single implementation task derived from the graph."""
    id: str
    description: str
    node_id: str              # Which graph node this task belongs to
    dependencies: list[str] = Field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING


class TaskPlan(BaseModel):
    """The full implementation plan (ordered list of tasks)."""
    tasks: list[Task] = Field(default_factory=list)


# ── Request / Response Models ─────────────────────────────────────────

class IdeaRequest(BaseModel):
    """Request body for idea submission."""
    idea: str = Field(..., min_length=3, max_length=2000,
                      examples=["Build a SaaS for managing ideas with AI"])


class GenerateGraphResponse(BaseModel):
    """Response from the graph generation endpoint."""
    graph: SystemGraph
    message: str = "Graph generated successfully"


class GenerateTasksRequest(BaseModel):
    """Request body for task generation."""
    graph: SystemGraph


class GenerateTasksResponse(BaseModel):
    """Response from the task generation endpoint."""
    plan: TaskPlan
    message: str = "Tasks generated successfully"


class GenerateCodeRequest(BaseModel):
    """Request body for code generation."""
    plan: TaskPlan
    graph: SystemGraph          # Global context (per Refinements §13)


class GeneratedFile(BaseModel):
    """A single file produced by the code generator (VFS buffer)."""
    path: str
    content: str


class GenerateCodeResponse(BaseModel):
    """Response from the code generation endpoint."""
    files: list[GeneratedFile] = Field(default_factory=list)
    message: str = "Code generated successfully"

class FlushRequest(BaseModel):
    """Request body for flushing VFS."""
    project_name: str | None = None
