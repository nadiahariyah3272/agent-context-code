# Code-Search MCP Tool — Test Findings

Date: 2026-03-12
Branch: revision-1.6 (Session C)

---

## Index State at Test Time

- **Chunks:** 1697
- **Files:** 158
- **Embedding dim:** 1024 (from prior Qwen3-Embedding-0.6B model)
- **Sync status:** synced
- **Reranker:** loaded (Qwen3-Reranker-0.6B)

---

## Overall Quality

Top results are consistently correct. Vector embeddings carry the semantic
load effectively — even CamelCase entity queries like `"getUserById"` find
semantically relevant code (score 0.98-0.99) without relying on BM25 keyword
splitting.

Exact-name queries work excellently: `"CodeSearchServer"` returned all 5
results at score 1.0.

---

## Observations

### 1. Duplicate Results Bug

Seen 3 times during testing — the same `chunk_id` appearing twice in the
result set. Examples:

- `docs/user-add-ons.md:358-383` duplicated
- `docs/user-add-ons.md:459-493` duplicated

**Likely cause:** RRF fusion merging results from both the BM25 and vector
pipelines when both return the same chunk. The deduplication step in the
merge may not be catching all cases, or the same chunk appears with slightly
different metadata from each pipeline.

**Impact:** Low — results are still correct, just repeated. Wastes one result
slot per duplicate.

### 2. Score Cliff on Generic Queries

Query `"refine_factor configuration search"` returned:

| Rank | Score | Chunk |
|------|-------|-------|
| 1 | 1.00 | (highly relevant) |
| 2 | 0.02 | (barely relevant) |
| 3 | 0.02 | ... |
| 4 | 0.02 | ... |
| 5 | 0.02 | ... |

**Analysis:** When query terms are generic (common English words like
"configuration", "search"), the top BM25 hit dominates via RRF while the
remaining results have only marginal vector similarity. This is expected
behavior — the score cliff reflects genuine quality differences.

### 3. CamelCase Entity Queries

Queries like `"getUserById"` and `"IncrementalIndexResult"` find semantically
relevant code even without BM25 splitting. The vector embedding model handles
compound identifiers well.

However, BM25 cannot match `"getUserById"` against content containing
`"get_user_by_id"` or `"get user by id"` because it treats the CamelCase
string as a single token. Session C's `_preprocess_bm25_query()` addresses
this by expanding the BM25 query: `"getUserById"` becomes
`"getUserById get User By Id"`, allowing Tantivy to match individual terms.

### 4. Exact Name Queries

`"CodeSearchServer"` → all 5 results at score 1.0. Both BM25 (exact string
match) and vector (semantic similarity) agree, producing perfect RRF scores.

This is the ideal case and confirms the hybrid search architecture works as
designed.

---

## Recommendations Applied in Session C

1. **BM25 query preprocessing** — CamelCase/snake_case/kebab-case splitting
   added to `search/searcher.py` via `_preprocess_bm25_query()`. Vector
   embedding path unchanged (models handle compound terms natively).

2. **BM25 parameter tuning** — researched and documented. Tantivy hardcodes
   k1=1.2 and b=0.75; not configurable. Defaults are acceptable given our
   hybrid architecture. See `search/indexer.py` `_ensure_fts_index()` docstring.

3. **Code-specific model documentation** — `sfr-code-400m` option documented
   in `README.md` with tradeoff table.
