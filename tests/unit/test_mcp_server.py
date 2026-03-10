"""Unit tests for MCP server module surface."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


class TestMCPServerImport:
    """The MCP server modules must be importable without errors."""

    def test_code_search_server_importable(self):
        import mcp_server.code_search_server  # noqa: F401

    def test_code_search_mcp_importable(self):
        # mcp_server.code_search_mcp may import fastmcp; skip if not installed
        try:
            import mcp_server.code_search_mcp  # noqa: F401
        except ImportError as exc:
            pytest.skip(f"Optional MCP dependency not installed: {exc}")


class TestCodeSearchServerPublicSurface:
    """CodeSearchServer must expose the expected public methods."""

    def test_search_code_method_exists(self):
        from mcp_server.code_search_server import CodeSearchServer
        assert callable(getattr(CodeSearchServer, "search_code", None))

    def test_index_directory_method_exists(self):
        from mcp_server.code_search_server import CodeSearchServer
        assert callable(getattr(CodeSearchServer, "index_directory", None))

    def test_list_projects_method_exists(self):
        from mcp_server.code_search_server import CodeSearchServer
        assert callable(getattr(CodeSearchServer, "list_projects", None))

    def test_switch_project_method_exists(self):
        from mcp_server.code_search_server import CodeSearchServer
        assert callable(getattr(CodeSearchServer, "switch_project", None))

    def test_get_index_status_method_exists(self):
        from mcp_server.code_search_server import CodeSearchServer
        assert callable(getattr(CodeSearchServer, "get_index_status", None))

    def test_clear_index_method_exists(self):
        from mcp_server.code_search_server import CodeSearchServer
        assert callable(getattr(CodeSearchServer, "clear_index", None))

    def test_find_similar_code_method_exists(self):
        from mcp_server.code_search_server import CodeSearchServer
        assert callable(getattr(CodeSearchServer, "find_similar_code", None))


class TestSearchCodeReturnSchema:
    """search_code must always return valid JSON with a predictable schema."""

    def test_empty_query_returns_json(self):
        from mcp_server.code_search_server import CodeSearchServer
        server = CodeSearchServer()
        raw = server.search_code("")
        result = json.loads(raw)
        assert isinstance(result, dict)

    def test_empty_query_has_error_key(self):
        from mcp_server.code_search_server import CodeSearchServer
        server = CodeSearchServer()
        result = json.loads(server.search_code(""))
        assert "error" in result

    def test_k_out_of_range_returns_error_json(self):
        from mcp_server.code_search_server import CodeSearchServer
        server = CodeSearchServer()
        result = json.loads(server.search_code("auth", k=0))
        assert "error" in result

    def test_search_response_query_key_matches_input(self):
        from mcp_server.code_search_server import CodeSearchServer
        server = CodeSearchServer()
        server._current_project = "/some/project"

        mock_searcher = MagicMock()
        mock_searcher.search.return_value = []
        mock_searcher.index_manager = MagicMock()
        mock_searcher.index_manager.get_stats.return_value = {"total_chunks": 0}

        with patch.object(server, "get_searcher", return_value=mock_searcher), \
             patch.object(server, "embedder", return_value=MagicMock()):
            result = json.loads(server.search_code("my query", auto_reindex=False))

        assert result.get("query") == "my query"


class TestIndexDirectoryReturnSchema:
    """index_directory must always return valid JSON."""

    def test_nonexistent_path_returns_json(self):
        from mcp_server.code_search_server import CodeSearchServer
        server = CodeSearchServer()
        raw = server.index_directory("/this/path/does/not/exist_xyz")
        result = json.loads(raw)
        assert isinstance(result, dict)

    def test_nonexistent_path_has_error_key(self):
        from mcp_server.code_search_server import CodeSearchServer
        server = CodeSearchServer()
        result = json.loads(server.index_directory("/this/path/does/not/exist_xyz"))
        assert "error" in result

    def test_file_path_instead_of_dir_has_error_key(self, tmp_path):
        from mcp_server.code_search_server import CodeSearchServer
        f = tmp_path / "somefile.py"
        f.write_text("x = 1")
        server = CodeSearchServer()
        result = json.loads(server.index_directory(str(f)))
        assert "error" in result
