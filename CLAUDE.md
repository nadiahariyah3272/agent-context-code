# CLAUDE.md

> **Scope:** This file is for the **source checkout / development copy** of
> `agent-context-local`. If you're troubleshooting a **PyPI install**
> (`uv tool install agent-context-local`), see the [README](README.md) instead.

## Quick Reference

```bash
uv sync                            # install deps
uv run python tests/run_tests.py   # run all tests
uv run python -m pytest tests/unit/test_cli.py -v
uv run python -m pytest tests/test_lancedb_schema.py -v
```

## Code Search

This project is indexed. Use `search_code` before reading files when
exploring unfamiliar areas, investigating bugs, or scoping changes.

| Area | What lives here |
|------|----------------|
| `chunking/` | Tree-sitter AST chunking — `base_chunker.py`, `multi_language_chunker.py`, `code_chunk.py` |
| `embeddings/` | Model loading and encoding — `sentence_transformer.py`, `model_catalog.py` |
| `search/` | LanceDB indexing and retrieval — `indexer.py`, `searcher.py`, `incremental_indexer.py` |
| `graph/` | SQLite relational graph — `code_graph.py` |
| `mcp_server/` | MCP tool surface — `code_search_server.py`, `code_search_mcp.py`, `strings.yaml` |
| `reranking/` | Opt-in reranker — `reranker.py`, `reranker_catalog.py` |
| `scripts/` | Install, uninstall, CLI — `install.sh`, `install.ps1`, `cli.py` |

Proven queries:

```
search_code("tree sitter chunk extract metadata")        → chunking/base_chunker.py
search_code("resolve cross file edges inheritance")      → graph/code_graph.py
search_code("incremental index merkle snapshot")         → search/incremental_indexer.py
search_code("MCP tool registration setup")               → mcp_server/code_search_mcp.py
search_code("embedding model load device dtype float16") → embeddings/sentence_transformer.py
search_code("reranker causal LM prompt build")           → reranking/reranker.py
```

After finding a chunk, use `get_graph_context(chunk_id)` to map sibling
methods without reading the file. Use `find_similar_code(chunk_id)` to
find other implementations of the same interface.

Scores >= 0.80 are reliable. Below 0.40, rephrase using method/class names
instead of natural language descriptions.

## Architecture

Two stores, one pipeline:

- **LanceDB** (`search/indexer.py`) — vector embeddings + BM25 FTS for hybrid search.
- **SQLite** (`graph/code_graph.py`) — relational graph: `contains`, `inherits`, `calls` edges.
  `imports` is reserved but not yet extracted.

Key design rules:
- `search_code` includes lightweight graph hints. `get_graph_context` does deep BFS traversal.
  Keep this two-tier model — don't merge them.
- Snapshot metadata only advances when both vector and graph stores succeed (consistency barrier).
- Graph enrichment is non-mutating — never creates graph DB files on read paths.

## Storage

All data under `~/.agent_code_search` (or `CODE_SEARCH_STORAGE`):
`models/`, `install_config.json`, `projects/{name}_{hash}/`, `merkle/`.
Never move DB files into the target workspace.

## Models

- CPU default: `mixedbread-ai/mxbai-embed-xsmall-v1` (384-dim)
- GPU default: `Qwen/Qwen3-Embedding-0.6B` (1024-dim)
- Reranker: opt-in only, never auto-enabled.

## MCP Registration

Install script auto-registers. Manual:

```bash
# macOS/Linux
claude mcp add code-search --scope user -- uv run --directory ~/.local/share/agent-context-code python mcp_server/server.py
# Windows PowerShell
claude mcp add code-search --scope user -- uv run --directory "$env:LOCALAPPDATA\agent-context-code" python mcp_server/server.py
```

GPU machines: add `--extra cu128` (or `cu126`) after `uv run`.

## Working Rules

- Run tests after each logical change. Don't batch all changes and test at the end.
- `mcp_server/strings.yaml` is deliberately tight — read `mcp_server/AGENTS.md` before editing.
  Every sentence must pass: "Would removing this cause an agent to make a mistake?"
- Keep docs and installer messaging aligned with actual behavior.
- When modifying setup flow, update `README.md`, installers, and `scripts/cli.py` together.
- Prefer compatibility-preserving changes over path/command renames.
