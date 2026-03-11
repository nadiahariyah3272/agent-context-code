"""Lightweight relational graph for code structure.

This package provides a SQLite-backed graph database that stores structural
relationships extracted from AST parsing: function definitions, class
hierarchies, import dependencies, and call-site references.

The graph complements the LanceDB vector index — semantic similarity finds
*what* is relevant while the graph reveals *how* those results connect
(caller/callee, parent/child, importer/importee).
"""
