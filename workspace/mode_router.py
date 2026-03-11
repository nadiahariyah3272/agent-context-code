"""Mode-aware file routing for the dual-mode indexing pipeline.

This module sits between the ``IncrementalIndexer`` and the underlying
chunking / graph-extraction logic.  For each file it:

1. Asks ``WorkspaceConfig`` whether the file is in *coding* or *writing* mode.
2. Delegates to the appropriate processing pipeline:
   - **coding** → tree-sitter AST chunking + relational graph extraction
   - **writing** → existing text-oriented chunking (Merkle DAG tracked)
3. Returns a uniform ``List[CodeChunk]`` so the downstream embedding and
   LanceDB insertion logic remains unchanged.

This design ensures the mode split is invisible to the embedder and indexer —
only the *chunking* and *graph population* steps differ.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

from chunking.code_chunk import CodeChunk
from chunking.multi_language_chunker import MultiLanguageChunker
from workspace.workspace_config import WorkspaceConfig, MODE_CODING, MODE_WRITING

logger = logging.getLogger(__name__)


class ModeRouter:
    """Routes files through the correct processing pipeline based on workspace mode.

    The router wraps an existing ``MultiLanguageChunker`` and augments it with
    mode awareness.  In *coding* mode it additionally feeds parsed chunks into
    a ``CodeGraph`` (if one is provided).  In *writing* mode it passes files
    through unchanged.

    Parameters
    ----------
    chunker : MultiLanguageChunker
        The existing multi-language chunker (tree-sitter + structured data).
    workspace_config : WorkspaceConfig
        Resolved workspace mode configuration.
    code_graph : optional
        A ``CodeGraph`` instance.  When provided, coding-mode chunks will
        have their relationships (imports, class hierarchies, calls)
        extracted and stored in the graph.  When *None*, coding-mode files
        are still chunked via tree-sitter but no graph is built.
    """

    def __init__(
        self,
        chunker: MultiLanguageChunker,
        workspace_config: WorkspaceConfig,
        code_graph=None,
    ):
        self.chunker = chunker
        self.config = workspace_config
        self.code_graph = code_graph

        # Counters for reporting — reset each indexing session.
        self._coding_files: int = 0
        self._writing_files: int = 0

    # ── Public API ───────────────────────────────────────────────────────

    def route_file(self, file_path: str) -> List[CodeChunk]:
        """Chunk a single file using the appropriate mode pipeline.

        Parameters
        ----------
        file_path : str
            Absolute path to the source file.

        Returns
        -------
        list[CodeChunk]
            Chunks produced by the mode-appropriate pipeline.
        """
        if not self.chunker.is_supported(file_path):
            return []

        mode = self.config.get_mode(file_path)

        if mode == MODE_CODING:
            return self._process_coding(file_path)
        else:
            return self._process_writing(file_path)

    def route_files(self, file_paths: List[str]) -> List[CodeChunk]:
        """Chunk multiple files, routing each through the correct pipeline.

        This is a convenience batch wrapper around :meth:`route_file`.
        """
        all_chunks: List[CodeChunk] = []
        for fp in file_paths:
            try:
                all_chunks.extend(self.route_file(fp))
            except Exception as exc:
                logger.warning("Failed to route %s: %s", fp, exc)
        return all_chunks

    def get_routing_stats(self) -> Dict[str, int]:
        """Return counters for the current session."""
        return {
            "coding_files_processed": self._coding_files,
            "writing_files_processed": self._writing_files,
        }

    def reset_stats(self) -> None:
        """Reset session counters (call at the start of each indexing run)."""
        self._coding_files = 0
        self._writing_files = 0

    # ── Private pipeline methods ─────────────────────────────────────────

    def _process_coding(self, file_path: str) -> List[CodeChunk]:
        """Process a source-code file: AST chunking + optional graph extraction.

        The chunking itself is delegated to the existing ``MultiLanguageChunker``
        which already uses tree-sitter for programming languages.  After
        chunking, if a ``CodeGraph`` is available, we extract relationships
        from the chunks and store them.
        """
        chunks = self.chunker.chunk_file(file_path)
        self._coding_files += 1

        # Populate the relational graph when available.
        if self.code_graph is not None and chunks:
            try:
                self.code_graph.index_file_chunks(file_path, chunks)
            except Exception as exc:
                # Graph population is best-effort — don't block indexing.
                logger.warning(
                    "Graph extraction failed for %s: %s", file_path, exc
                )

        return chunks

    def _process_writing(self, file_path: str) -> List[CodeChunk]:
        """Process a documentation / text file: standard text chunking.

        Uses the same ``MultiLanguageChunker`` (which routes structured data
        through ``StructuredDataChunker`` and Markdown through tree-sitter
        Markdown grammar).  No graph extraction is performed.
        """
        chunks = self.chunker.chunk_file(file_path)
        self._writing_files += 1
        return chunks
