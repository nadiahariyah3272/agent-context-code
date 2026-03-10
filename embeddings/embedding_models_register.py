"""Embedding models registry."""
from embeddings.gemma import GemmaEmbeddingModel

# Registry mapping model IDs to custom EmbeddingModel subclasses.
# Models NOT listed here fall back to SentenceTransformerModel automatically.
AVAILABLE_MODELS = {
    "google/embeddinggemma-300m": GemmaEmbeddingModel,
}
