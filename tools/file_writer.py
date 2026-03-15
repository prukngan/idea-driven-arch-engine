"""
Virtual File System (VFS) Writer

Buffers generated files in memory and writes to disk only on user confirmation.
This implements the safety recommendation from Refinements §13.
"""

import logging
import os
from pathlib import Path

from config import OUTPUT_ROOT

logger = logging.getLogger(__name__)


class VirtualFileSystem:
    """
    In-memory buffer for generated project files.

    Files are stored as {path: content} pairs and only written
    to the real filesystem when `flush()` is called.
    """

    def __init__(self):
        self._files: dict[str, str] = {}

    def add_file(self, path: str, content: str) -> None:
        """Add or overwrite a file in the VFS buffer."""
        # Normalize and prevent path traversal
        clean_path = os.path.normpath(path).lstrip(os.sep)
        if ".." in clean_path.split(os.sep):
            raise ValueError(f"Path traversal detected: {path}")
        self._files[clean_path] = content
        logger.debug(f"VFS: buffered {clean_path} ({len(content)} chars)")

    def list_files(self) -> list[str]:
        """List all buffered file paths."""
        return list(self._files.keys())

    def get_file(self, path: str) -> str | None:
        """Read a buffered file's content."""
        clean_path = os.path.normpath(path).lstrip(os.sep)
        return self._files.get(clean_path)

    def flush(self, output_dir: str | None = None) -> list[str]:
        """
        Write all buffered files to disk.

        Args:
            output_dir: Root directory for output. Defaults to OUTPUT_ROOT from config.

        Returns:
            List of absolute paths of written files.
        """
        root = Path(output_dir or OUTPUT_ROOT)
        written: list[str] = []

        for rel_path, content in self._files.items():
            full_path = root / rel_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content, encoding="utf-8")
            written.append(str(full_path))
            logger.info(f"VFS: wrote {full_path}")

        logger.info(f"VFS: flushed {len(written)} files to {root}")
        return written

    def clear(self) -> None:
        """Clear all buffered files."""
        self._files.clear()

    def __len__(self) -> int:
        return len(self._files)
