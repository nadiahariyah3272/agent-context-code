"""Two-stage reranker using Qwen3-Reranker causal LM.

Scores each (query, passage) pair via binary yes/no classification on the
last generated token.  Designed for lazy loading — the model is not
instantiated until the first ``rerank()`` call.

Input/output contract:
- ``rerank()`` accepts the same ``List[Tuple[str, float, Dict]]`` format
  returned by ``CodeIndexManager.search()``.
- It returns a re-sorted list in the same format, with the ``float``
  replaced by the reranker relevance score (0-1).
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

from reranking.reranker_catalog import RerankerModelConfig, get_reranker_config

logger = logging.getLogger(__name__)


class CodeReranker:
    """Lazy-loaded two-stage reranker using Qwen3-Reranker causal LM."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Reranker-4B",
        cache_dir: Optional[str] = None,
        device: str = "auto",
    ):
        self._model_name = model_name
        self._cache_dir = cache_dir
        self._device = device
        self._config: RerankerModelConfig = get_reranker_config(model_name)

        # Lazily initialised
        self._model = None
        self._tokenizer = None
        self._yes_token_id: Optional[int] = None
        self._no_token_id: Optional[int] = None

    def _ensure_loaded(self) -> None:
        """Import transformers and load the model + tokenizer on first use."""
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading reranker model: %s", self._model_name)

        # Resolve device — AMD ROCm GPUs: PyTorch's ROCm build makes
        # torch.cuda.is_available() return True, so the "cuda" path works
        # for both NVIDIA and AMD GPUs without special handling.
        if self._device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        else:
            device = self._device

        # float16 on CUDA (works for both NVIDIA and AMD ROCm consumer GPUs),
        # float32 on CPU and MPS (MPS does not support bfloat16).
        dtype = torch.float16 if device == "cuda" else torch.float32

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name,
            cache_dir=self._cache_dir,
            padding_side="left",
        )

        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_name,
            cache_dir=self._cache_dir,
            torch_dtype=dtype,
        ).to(device).eval()

        # Cache yes/no token IDs for logit extraction
        self._yes_token_id = self._tokenizer.convert_tokens_to_ids("yes")
        self._no_token_id = self._tokenizer.convert_tokens_to_ids("no")

        logger.info(
            "Reranker loaded on %s (dtype=%s, yes_id=%s, no_id=%s)",
            device, dtype, self._yes_token_id, self._no_token_id,
        )

    def _build_prompt(self, query: str, document: str) -> str:
        """Build the official Qwen3-Reranker chat-template prompt.

        Format follows the Qwen3-Reranker reference implementation:
        system message with judge instruction, then user message with
        <Instruct>, <Query>, <Document> tags, then assistant prefix
        with empty think block.
        """
        instruction = self._config.instruction
        return (
            "<|im_start|>system\n"
            "Judge whether the document is relevant to the search query. "
            "Answer only \"yes\" or \"no\".<|im_end|>\n"
            "<|im_start|>user\n"
            f"<Instruct>: {instruction}\n"
            f"<Query>: {query}\n"
            f"<Document>: {document}<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n\n</think>\n"
        )

    def rerank(
        self,
        query: str,
        passages: List[Tuple[str, float, Dict[str, Any]]],
        top_k: Optional[int] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Rerank passages by relevance to the query.

        Args:
            query: The search query string.
            passages: List of (chunk_id, similarity_score, metadata) tuples
                     as returned by ``CodeIndexManager.search()``.
            top_k: Maximum number of results to return. If None, returns all.

        Returns:
            Re-sorted list in the same format, with scores replaced by
            reranker relevance probabilities (0-1).  Each metadata dict
            gains a ``"reranked": True`` flag and the original vector
            similarity is preserved as ``"vector_similarity"``.
        """
        if not passages:
            return []

        self._ensure_loaded()

        import torch

        # Build prompts for all passages
        prompts = []
        for chunk_id, score, metadata in passages:
            # Use content from metadata for reranking
            content = metadata.get("content_preview", "") or metadata.get("content", "")
            prompts.append(self._build_prompt(query, content))

        # Batch tokenize
        inputs = self._tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self._config.max_length,
            return_tensors="pt",
        ).to(self._model.device)

        # Forward pass — no gradient needed
        with torch.no_grad():
            outputs = self._model(**inputs)

        # Extract last-token logits and compute yes/no scores
        scores = []
        for i in range(len(passages)):
            # Get logits for the last non-padding token
            logits = outputs.logits[i, -1, :]
            yes_logit = logits[self._yes_token_id].float().item()
            no_logit = logits[self._no_token_id].float().item()
            # Numerically stable softmax over yes/no logits.
            # Subtracting max(yes, no) prevents math.exp() overflow for large
            # logit values while keeping the ratio identical (the max cancels).
            shift = max(yes_logit, no_logit)
            yes_exp = math.exp(yes_logit - shift)
            no_exp = math.exp(no_logit - shift)
            score = yes_exp / (yes_exp + no_exp)
            # Clamp to [0, 1] as a safety net against floating-point edge cases.
            score = max(0.0, min(1.0, score))
            scores.append(score)

        # Build results with reranker scores
        reranked = []
        for (chunk_id, original_score, metadata), rerank_score in zip(passages, scores):
            enriched_meta = dict(metadata)
            enriched_meta["reranked"] = True
            enriched_meta["vector_similarity"] = original_score
            reranked.append((chunk_id, rerank_score, enriched_meta))

        # Sort by reranker score descending
        reranked.sort(key=lambda x: x[1], reverse=True)

        if top_k is not None:
            reranked = reranked[:top_k]

        return reranked

    def cleanup(self) -> None:
        """Release model resources."""
        if self._model is not None:
            import torch

            del self._model
            del self._tokenizer
            self._model = None
            self._tokenizer = None
            self._yes_token_id = None
            self._no_token_id = None

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("Reranker model unloaded")

    def get_model_info(self) -> Dict[str, Any]:
        """Return information about the reranker model."""
        return {
            "model_name": self._model_name,
            "short_name": self._config.short_name,
            "loaded": self._model is not None,
            "device": str(self._model.device) if self._model is not None else None,
            "max_length": self._config.max_length,
            "description": self._config.description,
        }
