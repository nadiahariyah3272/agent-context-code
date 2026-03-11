"""Workspace configuration and dual-mode routing.

This package provides configurable workspace modes that determine how files
are processed during indexing:

- **coding** mode: Uses tree-sitter AST parsing + SQLite relational graph +
  LanceDB vector embeddings.  Returns structural relationships (imports,
  call graphs, class hierarchies) alongside semantic search results.

- **writing** mode: Retains the existing Merkle DAG state-tracking and
  text-oriented chunking for documentation files (.md, .txt, .csv, etc.).

Users configure modes per-workspace or per-extension via the project-level
``.agent-context-code.json`` file or the ``workspace_mode`` key in
``install_config.json``.
"""
