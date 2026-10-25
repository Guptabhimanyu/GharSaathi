"""Embeddings via fastembed (ONNX, no PyTorch): much lighter on RAM and install size, which matters on free hosting."""
import numpy as np

MODEL = "BAAI/bge-small-en-v1.5"
ID = f"fastembed:{MODEL}"  # stored with cached vectors so a model change invalidates old caches
_model = None


def embed(texts) -> np.ndarray:
    global _model
    if _model is None:
        from fastembed import TextEmbedding
        _model = TextEmbedding(MODEL)
    vecs = np.array(list(_model.embed(list(texts))), dtype="float32")
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return vecs / norms
