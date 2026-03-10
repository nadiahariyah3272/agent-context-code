"""Abstract base class for embedding models."""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np
import torch

logger = logging.getLogger(__name__)


class EmbeddingModel(ABC):
    """Abstract base class for embedding models."""

    def __init__(self, device: str):
        """Initialize with device resolution."""
        self._device = self._resolve_device(device)

    @abstractmethod
    def encode(self, texts: List[str], **kwargs) -> np.ndarray:
        """Encode texts to embeddings.

        Args:
            texts: List of texts to encode
            **kwargs: Additional model-specific arguments

        Returns:
            Array of embeddings with shape (len(texts), embedding_dim)
        """
        pass

    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        pass

    @abstractmethod
    def cleanup(self):
        """Clean up model resources."""
        pass

    def __del__(self):
        """Ensure cleanup when object is destroyed."""
        try:
            self.cleanup()
        except Exception:
            pass

    def _resolve_device(self, requested: Optional[str]) -> str:
        """Resolve device string.

        AMD ROCm GPUs: PyTorch's ROCm build makes torch.cuda.is_available()
        return True, so the "cuda" path works for both NVIDIA and AMD GPUs.
        No special handling is needed here.
        """
        req = (requested or "auto").lower()
        if req in ("auto", "none", ""):
            if torch.cuda.is_available():
                return "cuda"
            try:
                if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    return "mps"
            except Exception:
                pass
            return "cpu"
        if req.startswith("cuda"):
            if torch.cuda.is_available():
                return "cuda"
            logger.warning("CUDA requested but not available, falling back to CPU")
            return "cpu"
        if req == "mps":
            try:
                if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    return "mps"
            except Exception:
                pass
            logger.warning("MPS requested but not available, falling back to CPU")
            return "cpu"
        return "cpu"
