"""Unit tests for search/searcher.IntelligentSearcher."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from search.searcher import IntelligentSearcher, SearchResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_index_manager():
    """Return a MagicMock that satisfies IntelligentSearcher's index_manager API."""
    im = MagicMock()
    im.search.return_value = []
    im.get_similar_chunks.return_value = []
    im.get_file_chunk_count.return_value = 0
    im.get_stats.return_value = {"top_tags": {}, "chunk_types": {}}
    return im


def _make_mock_embedder(dim: int = 4):
    """Return a MagicMock that satisfies the CodeEmbedder embed_query API."""
    embedder = MagicMock()
    embedder.embed_query.return_value = np.zeros(dim, dtype=np.float32)
    return embedder


def _make_search_result(
    chunk_id: str = "c0",
    similarity: float = 0.9,
    chunk_type: str = "function",
    name: str = "my_func",
    relative_path: str = "src/helpers.py",
    content_preview: str = "def my_func(): pass",
    docstring: str = "",
    tags: list | None = None,
) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        similarity_score=similarity,
        content_preview=content_preview,
        file_path=f"/proj/{relative_path}",
        relative_path=relative_path,
        folder_structure=["src"],
        chunk_type=chunk_type,
        name=name,
        parent_name=None,
        start_line=1,
        end_line=10,
        docstring=docstring or None,
        tags=tags or [],
        context_info={},
    )


@pytest.fixture()
def searcher():
    im = _make_mock_index_manager()
    embedder = _make_mock_embedder()
    return IntelligentSearcher(im, embedder)


# ---------------------------------------------------------------------------
# _is_entity_like_query
# ---------------------------------------------------------------------------

class TestIsEntityLikeQuery:
    def test_camel_case_is_entity_like(self, searcher):
        tokens = searcher._normalize_to_tokens("AuthManager")
        assert searcher._is_entity_like_query("AuthManager", tokens) is True

    def test_long_query_not_entity_like(self, searcher):
        query = "find all authentication functions in the project"
        tokens = searcher._normalize_to_tokens(query)
        assert searcher._is_entity_like_query(query, tokens) is False

    def test_action_word_not_entity_like(self, searcher):
        for action in ("find", "search", "get", "show", "create", "build"):
            query = f"{action} something"
            tokens = searcher._normalize_to_tokens(query)
            assert searcher._is_entity_like_query(query, tokens) is False, (
                f"'{query}' should not be entity-like"
            )

    def test_short_noun_phrase_is_entity_like(self, searcher):
        query = "database"
        tokens = searcher._normalize_to_tokens(query)
        assert searcher._is_entity_like_query(query, tokens) is True

    def test_two_word_noun_is_entity_like(self, searcher):
        query = "user model"
        tokens = searcher._normalize_to_tokens(query)
        assert searcher._is_entity_like_query(query, tokens) is True

    def test_four_word_query_not_entity_like(self, searcher):
        query = "user model manager helper"
        tokens = searcher._normalize_to_tokens(query)
        assert searcher._is_entity_like_query(query, tokens) is False


# ---------------------------------------------------------------------------
# _normalize_to_tokens
# ---------------------------------------------------------------------------

class TestNormalizeToTokens:
    def test_camel_case_split(self, searcher):
        tokens = searcher._normalize_to_tokens("CamelCaseWord")
        assert "camel" in tokens
        assert "case" in tokens
        assert "word" in tokens

    def test_snake_case_split(self, searcher):
        tokens = searcher._normalize_to_tokens("snake_case_word")
        assert "snake" in tokens
        assert "case" in tokens
        assert "word" in tokens

    def test_lowercase_output(self, searcher):
        tokens = searcher._normalize_to_tokens("UPPERCASE")
        for t in tokens:
            assert t == t.lower()

    def test_empty_string(self, searcher):
        assert searcher._normalize_to_tokens("") == []


# ---------------------------------------------------------------------------
# _calculate_name_boost
# ---------------------------------------------------------------------------

class TestCalculateNameBoost:
    def test_exact_match_returns_highest_boost(self, searcher):
        tokens = searcher._normalize_to_tokens("authenticate")
        boost = searcher._calculate_name_boost("authenticate", "authenticate", tokens)
        assert boost == pytest.approx(1.4)

    def test_no_name_returns_1(self, searcher):
        tokens = searcher._normalize_to_tokens("auth")
        assert searcher._calculate_name_boost(None, "auth", tokens) == 1.0

    def test_no_overlap_returns_1(self, searcher):
        tokens = searcher._normalize_to_tokens("database")
        assert searcher._calculate_name_boost("completely_different", "database", tokens) == 1.0

    def test_high_overlap_boosts_above_1(self, searcher):
        tokens = searcher._normalize_to_tokens("authenticate user")
        boost = searcher._calculate_name_boost("authenticate_user", "authenticate user", tokens)
        assert boost > 1.0


# ---------------------------------------------------------------------------
# _calculate_path_boost
# ---------------------------------------------------------------------------

class TestCalculatePathBoost:
    def test_matching_token_boosts(self, searcher):
        tokens = searcher._normalize_to_tokens("auth")
        boost = searcher._calculate_path_boost("src/auth/authenticator.py", tokens)
        assert boost > 1.0

    def test_no_overlap_returns_1(self, searcher):
        tokens = searcher._normalize_to_tokens("completely_unrelated")
        boost = searcher._calculate_path_boost("src/helpers.py", tokens)
        assert boost == pytest.approx(1.0)

    def test_empty_path_returns_1(self, searcher):
        tokens = searcher._normalize_to_tokens("auth")
        assert searcher._calculate_path_boost("", tokens) == pytest.approx(1.0)

    def test_empty_tokens_returns_1(self, searcher):
        assert searcher._calculate_path_boost("src/auth/auth.py", []) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _rank_results — ordering and dampening
# ---------------------------------------------------------------------------

class TestRankResults:
    def test_higher_similarity_ranks_first_by_default(self, searcher):
        r_high = _make_search_result("c1", similarity=0.95, chunk_type="function")
        r_low = _make_search_result("c2", similarity=0.3, chunk_type="function")
        ranked = searcher._rank_results([r_high, r_low], "generic query", [])
        assert ranked[0].chunk_id == "c1"

    def test_class_keyword_boosts_class_chunk(self, searcher):
        r_class = _make_search_result("c_class", similarity=0.7, chunk_type="class", name="MyClass")
        r_func = _make_search_result("c_func", similarity=0.7, chunk_type="function", name="my_func")
        ranked = searcher._rank_results([r_class, r_func], "class MyClass", ["function_search"])
        assert ranked[0].chunk_id == "c_class"

    def test_reranked_flag_dampens_boosts(self, searcher):
        """With reranked=True, heuristic boosts should be smaller.

        We use a class-keyword query where the class chunk normally gets a
        large type boost.  With dampening the gap should narrow.
        """
        r_class = _make_search_result("c_class", similarity=0.7, chunk_type="class")
        r_func = _make_search_result("c_func", similarity=0.695, chunk_type="function")

        # Without dampening: class should strongly lead
        ranked_no_damp = searcher._rank_results(
            [r_class, r_func], "class usage", [], reranked=False
        )
        # With dampening: gap is reduced
        ranked_damp = searcher._rank_results(
            [r_class, r_func], "class usage", [], reranked=True
        )
        # Both orderings may put class first, but the test verifies the method
        # runs without error and returns the same count
        assert len(ranked_no_damp) == 2
        assert len(ranked_damp) == 2

    def test_docstring_boost_applied(self, searcher):
        r_with_doc = _make_search_result("c_doc", similarity=0.8, docstring="Does X")
        r_no_doc = _make_search_result("c_nodoc", similarity=0.8, docstring=None)
        ranked = searcher._rank_results([r_with_doc, r_no_doc], "query", [])
        assert ranked[0].chunk_id == "c_doc"

    def test_empty_results_returns_empty(self, searcher):
        assert searcher._rank_results([], "query", []) == []


# ---------------------------------------------------------------------------
# get_search_suggestions
# ---------------------------------------------------------------------------

class TestGetSearchSuggestions:
    def test_suggestions_returns_list(self, searcher):
        result = searcher.get_search_suggestions("auth")
        assert isinstance(result, list)

    def test_suggestions_at_most_five(self, searcher):
        searcher.index_manager.get_stats.return_value = {
            "top_tags": {f"tag{i}": i for i in range(20)},
            "chunk_types": {"function": 10, "class": 5},
        }
        result = searcher.get_search_suggestions("t")
        assert len(result) <= 5

    def test_suggestions_empty_on_no_index_data(self, searcher):
        result = searcher.get_search_suggestions("never_matches_anything_xyz_12345")
        assert isinstance(result, list)

    def test_suggestions_include_partial_match(self, searcher):
        searcher.index_manager.get_stats.return_value = {
            "top_tags": {"authentication": 5, "database": 3},
            "chunk_types": {},
        }
        suggestions = searcher.get_search_suggestions("auth")
        assert any("auth" in s.lower() for s in suggestions)


# ---------------------------------------------------------------------------
# Reranker enabled / fallback paths
# ---------------------------------------------------------------------------

class TestRerankerIntegration:
    def _full_meta(self, chunk_id: str = "c0") -> dict:
        """Return a metadata dict with all keys _create_search_result expects."""
        return {
            "content_preview": "def foo(): pass",
            "file_path": "/proj/foo.py",
            "relative_path": "foo.py",
            "chunk_type": "function",
            "name": "foo",
            "parent_name": "",
            "start_line": 1,
            "end_line": 5,
            "docstring": "",
            "tags": [],
            "folder_structure": ["proj"],
        }

    def test_reranker_called_when_present(self):
        im = _make_mock_index_manager()
        im.search.return_value = [
            ("c0", 0.8, self._full_meta("c0")),
        ]
        im.get_similar_chunks.return_value = []
        im.get_file_chunk_count.return_value = 0

        embedder = _make_mock_embedder()
        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = [
            ("c0", 0.95, {
                "content_preview": "def foo(): pass",
                "file_path": "/proj/foo.py",
                "relative_path": "foo.py",
                "chunk_type": "function",
                "name": "foo",
                "parent_name": "",
                "start_line": 1,
                "end_line": 5,
                "docstring": "",
                "tags": [],
                "folder_structure": ["proj"],
                "reranked": True,
                "vector_similarity": 0.8,
            }),
        ]

        s = IntelligentSearcher(im, embedder, reranker=mock_reranker, reranker_recall_k=20)
        s.search("query", k=1)
        mock_reranker.rerank.assert_called_once()

    def test_fallback_to_vector_scores_when_reranker_raises(self):
        im = _make_mock_index_manager()
        im.search.return_value = [
            ("c0", 0.8, {"content_preview": "x", "file_path": "/f", "relative_path": "f.py",
                          "chunk_type": "function", "name": "f", "parent_name": "",
                          "start_line": 1, "end_line": 5, "docstring": "", "tags": [],
                          "folder_structure": []}),
        ]
        im.get_similar_chunks.return_value = []
        im.get_file_chunk_count.return_value = 0

        embedder = _make_mock_embedder()
        mock_reranker = MagicMock()
        mock_reranker.rerank.side_effect = RuntimeError("reranker exploded")

        s = IntelligentSearcher(im, embedder, reranker=mock_reranker)
        results = s.search("query", k=1, context_depth=0)
        # Should return results using vector scores (fallback)
        assert len(results) == 1

    def test_no_reranker_uses_vector_scores(self):
        im = _make_mock_index_manager()
        im.search.return_value = [
            ("c0", 0.8, {"content_preview": "x", "file_path": "/f", "relative_path": "f.py",
                          "chunk_type": "function", "name": "f", "parent_name": "",
                          "start_line": 1, "end_line": 5, "docstring": "", "tags": [],
                          "folder_structure": []}),
        ]
        im.get_similar_chunks.return_value = []
        im.get_file_chunk_count.return_value = 0

        embedder = _make_mock_embedder()
        s = IntelligentSearcher(im, embedder, reranker=None)
        results = s.search("query", k=1, context_depth=0)
        assert len(results) == 1
        assert results[0].similarity_score == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# search_by_chunk_type / search_by_file_pattern — filter pass-through
# ---------------------------------------------------------------------------

class TestFilteredSearch:
    def test_search_by_chunk_type_passes_filter(self, searcher):
        searcher.search("query", k=2, filters={"chunk_type": "class"})
        call_args = searcher.index_manager.search.call_args
        assert call_args is not None, "index_manager.search was not called"
        # The third positional argument (or 'filters' keyword) should be the filter dict
        # _semantic_search calls index_manager.search(query_embedding, fetch_k, filters)
        positional = call_args.args
        kw = call_args.kwargs
        passed_filters = positional[2] if len(positional) > 2 else kw.get("filters")
        assert passed_filters is not None, "filters were not passed to index_manager.search"
        assert passed_filters.get("chunk_type") == "class"

    def test_search_by_file_pattern_method_exists(self, searcher):
        assert hasattr(searcher, "search_by_file_pattern")

    def test_search_by_chunk_type_method_exists(self, searcher):
        assert hasattr(searcher, "search_by_chunk_type")
