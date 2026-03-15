"""
Sandboxed Command Runner

Executes shell commands within a restricted directory scope.
Implements the safety recommendation from Refinements §13.
"""

import asyncio
import logging
import os
from pathlib import Path

from config import OUTPUT_ROOT

logger = logging.getLogger(__name__)

# Commands that are never allowed
BLOCKED_COMMANDS = {"rm", "rmdir", "del", "format", "mkfs", "dd", "shutdown", "reboot"}


async def run_command(
    command: str,
    cwd: str | None = None,
    timeout: float = 60.0,
) -> dict:
    """
    Run a shell command in a sandboxed directory.

    Args:
        command: The command to execute.
        cwd: Working directory (must be within OUTPUT_ROOT).
        timeout: Max execution time in seconds.

    Returns:
        Dict with 'stdout', 'stderr', and 'returncode'.
    """
    work_dir = Path(cwd or OUTPUT_ROOT).resolve()
    safe_root = Path(OUTPUT_ROOT).resolve()

    # Safety check: ensure cwd is within the sandbox
    if not str(work_dir).startswith(str(safe_root)):
        raise PermissionError(
            f"Command execution outside sandbox is not allowed. "
            f"cwd={work_dir} is not inside {safe_root}"
        )

    # Check for blocked commands
    cmd_name = command.strip().split()[0].lower() if command.strip() else ""
    if cmd_name in BLOCKED_COMMANDS:
        raise PermissionError(f"Command '{cmd_name}' is blocked for safety reasons.")

    work_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Running command: {command} (cwd={work_dir})")

    try:
        process = await asyncio.create_subprocess_shell(
            command,
            cwd=str(work_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            process.communicate(), timeout=timeout
        )
    except asyncio.TimeoutError:
        process.kill()
        return {
            "stdout": "",
            "stderr": f"Command timed out after {timeout}s",
            "returncode": -1,
        }

    result = {
        "stdout": stdout.decode("utf-8", errors="replace"),
        "stderr": stderr.decode("utf-8", errors="replace"),
        "returncode": process.returncode,
    }
    logger.info(f"Command finished with code {process.returncode}")
    return result
