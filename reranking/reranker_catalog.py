"""Reranker model presets and configuration."""

from dataclasses import dataclass
from typing import Optional


RERANKER_INSTRUCTION = (
    "Given a code search query, does the following code chunk "
    "answer or relate to the query?"
)


@dataclass(frozen=True)
class RerankerModelConfig:
    """Reranker model configuration used by the download script and runtime."""

    model_name: str
    short_name: str
    instruction: str
    max_length: int = 8192
    description: str = ""
    recommended_for: str = ""
    vram_requirement_gb: float = 8.0
    cpu_feasible: bool = True


RERANKER_CATALOG = {
    "Qwen/Qwen3-Reranker-4B": RerankerModelConfig(
        model_name="Qwen/Qwen3-Reranker-4B",
        short_name="qwen-reranker-4b",
        instruction=RERANKER_INSTRUCTION,
        max_length=8192,
        description="Qwen3-Reranker-4B causal LM for two-stage code search reranking.",
        recommended_for="High-precision reranking on GPU; feasible on CPU with higher latency.",
        vram_requirement_gb=8.0,
        cpu_feasible=True,
    ),
}

# Short-name reverse lookup: maps e.g. "qwen-reranker-4b" → "Qwen/Qwen3-Reranker-4B"
RERANKER_SHORT_NAMES = {
    config.short_name: name
    for name, config in RERANKER_CATALOG.items()
    if config.short_name
}


def get_reranker_config(model_name: str) -> RerankerModelConfig:
    """Resolve a reranker config by full HuggingFace name or short name.

    Raises ``KeyError`` if the model is not in the catalog.
    """
    # Try direct lookup first
    if model_name in RERANKER_CATALOG:
        return RERANKER_CATALOG[model_name]

    # Try short-name reverse lookup
    full_name = RERANKER_SHORT_NAMES.get(model_name)
    if full_name:
        return RERANKER_CATALOG[full_name]

    raise KeyError(
        f"Unknown reranker model: '{model_name}'. "
        f"Available: {', '.join(RERANKER_CATALOG.keys())} "
        f"(short names: {', '.join(RERANKER_SHORT_NAMES.keys())})"
    )
