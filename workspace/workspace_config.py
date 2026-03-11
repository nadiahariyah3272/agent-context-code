"""Workspace mode configuration and file-to-mode routing.

This module defines which processing pipeline applies to each file:

- **coding** — AST-aware chunking, relational graph extraction, and vector
  embeddings.  Designed for source code where structural relationships
  (imports, class hierarchies, call graphs) are valuable.

- **writing** — Merkle DAG change tracking with text-oriented chunking and
  vector embeddings.  Designed for documentation, prose, and tabular data
  where structural parsing adds no value.

Configuration priority (highest → lowest):
1. Per-file extension overrides in ``.agent-context-code.json``
2. Project-level ``workspace_mode`` in ``.agent-context-code.json``
3. ``install_config.json`` global defaults
4. Built-in defaults (code extensions → coding; doc extensions → writing)
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Set

logger = logging.getLogger(__name__)

# ── Built-in extension classification ────────────────────────────────────
# These are the sensible defaults.  Users can override via config.

# Extensions that are unambiguously source code — processed via the
# tree-sitter AST pipeline + relational graph + vector embeddings.
CODING_EXTENSIONS: Set[str] = {
    ".py", ".js", ".ts", ".tsx", ".jsx",
    ".go", ".java", ".rs", ".c", ".cpp", ".cs",
    ".kt", ".svelte",
}

# Extensions that are documentation or structured data — processed via the
# Merkle DAG text chunking pipeline + vector embeddings (no graph).
WRITING_EXTENSIONS: Set[str] = {
    ".md", ".txt", ".csv", ".rst", ".adoc",
    ".json", ".yaml", ".yml", ".toml", ".xml", ".html",
}

# The two recognised mode names.  Using a simple str enum keeps config
# files human-readable while still being validatable.
MODE_CODING = "coding"
MODE_WRITING = "writing"
VALID_MODES = {MODE_CODING, MODE_WRITING}


class WorkspaceConfig:
    """Resolves the processing mode for every file in a workspace.

    The config is loaded once per indexing session and cached.  It merges
    built-in defaults with user overrides from ``.agent-context-code.json``:

    .. code-block:: json

        {
            "workspace_mode": {
                "default_mode": "coding",
                "extension_overrides": {
                    ".md": "writing",
                    ".proto": "coding"
                }
            }
        }

    Parameters
    ----------
    project_config : dict, optional
        The parsed contents of ``.agent-context-code.json``.  When *None*,
        only built-in defaults are used.
    """

    def __init__(self, project_config: Optional[Dict] = None):
        self._project_config = project_config or {}

        # Merge user overrides on top of built-in defaults.
        ws_cfg = self._project_config.get("workspace_mode", {})
        self._default_mode: str = ws_cfg.get("default_mode", MODE_CODING)
        if self._default_mode not in VALID_MODES:
            logger.warning(
                "Invalid default_mode '%s' in workspace config; falling back to '%s'",
                self._default_mode,
                MODE_CODING,
            )
            self._default_mode = MODE_CODING

        # Build the effective extension → mode mapping.
        # Start with built-in classification …
        self._ext_map: Dict[str, str] = {}
        for ext in CODING_EXTENSIONS:
            self._ext_map[ext] = MODE_CODING
        for ext in WRITING_EXTENSIONS:
            self._ext_map[ext] = MODE_WRITING

        # … then layer user overrides on top so they always win.
        user_overrides = ws_cfg.get("extension_overrides", {})
        for ext, mode in user_overrides.items():
            normalised = ext if ext.startswith(".") else f".{ext}"
            normalised = normalised.lower()
            if mode not in VALID_MODES:
                logger.warning(
                    "Ignoring invalid mode '%s' for extension '%s'",
                    mode,
                    normalised,
                )
                continue
            self._ext_map[normalised] = mode

    # ── Public API ───────────────────────────────────────────────────────

    def get_mode(self, file_path: str) -> str:
        """Return the processing mode for *file_path*.

        Resolution order:
        1. Exact extension match in the merged ext→mode map.
        2. The configured ``default_mode``.

        Returns ``"coding"`` or ``"writing"``.
        """
        suffix = Path(file_path).suffix.lower()
        return self._ext_map.get(suffix, self._default_mode)

    def is_coding(self, file_path: str) -> bool:
        """Convenience: True when the file should use the AST/graph pipeline."""
        return self.get_mode(file_path) == MODE_CODING

    def is_writing(self, file_path: str) -> bool:
        """Convenience: True when the file should use Merkle/text pipeline."""
        return self.get_mode(file_path) == MODE_WRITING

    @property
    def default_mode(self) -> str:
        """The workspace-level fallback mode."""
        return self._default_mode

    @property
    def extension_map(self) -> Dict[str, str]:
        """A *copy* of the effective extension → mode mapping."""
        return dict(self._ext_map)

    def get_coding_extensions(self) -> Set[str]:
        """Return all extensions currently mapped to coding mode."""
        return {ext for ext, mode in self._ext_map.items() if mode == MODE_CODING}

    def get_writing_extensions(self) -> Set[str]:
        """Return all extensions currently mapped to writing mode."""
        return {ext for ext, mode in self._ext_map.items() if mode == MODE_WRITING}

    def to_dict(self) -> Dict:
        """Serialise the active config for persistence / debugging."""
        return {
            "default_mode": self._default_mode,
            "extension_overrides": {
                ext: mode
                for ext, mode in self._ext_map.items()
                # Only include entries that differ from built-in defaults
                # so the serialised form stays concise.
                if (ext in CODING_EXTENSIONS and mode != MODE_CODING)
                or (ext in WRITING_EXTENSIONS and mode != MODE_WRITING)
                or (ext not in CODING_EXTENSIONS and ext not in WRITING_EXTENSIONS)
            },
        }

    def __repr__(self) -> str:  # pragma: no cover
        coding = sorted(self.get_coding_extensions())
        writing = sorted(self.get_writing_extensions())
        return (
            f"WorkspaceConfig(default={self._default_mode!r}, "
            f"coding={coding}, writing={writing})"
        )
