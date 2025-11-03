# source_code/eval/metrics.py
# Drop-in semantic similarity utilities for RAG evaluation.
# - Sentence-Transformers cosine similarity (HF encoders)
# - OpenAI (or any HF) embedding cosine similarity
#
# No sklearn dependency: cosine similarity with NumPy.

from typing import List, Dict, Any, Optional, Tuple
import os
import numpy as np
from transformers import pipeline
import nltk
# Lazy imports to keep optional deps optional
def _maybe_import(name: str):
    try:
        return __import__(name)
    except Exception:
        return None

_sentence_transformers = _maybe_import("sentence_transformers")
_openai = _maybe_import("openai")


# --------------------------
# Core math: cosine similarity
# --------------------------
def _l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True) + eps
    return x / n

def _pairwise_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    a: [N, D], b: [N, D]  (paired rows)
    returns: [N] cosine similarity per pair
    """
    a = _l2_normalize(a, axis=1)
    b = _l2_normalize(b, axis=1)
    return np.sum(a * b, axis=1)


# --------------------------
# Sentence-Transformers STS
# --------------------------
def sts_similarity_hf(
    preds: List[str],
    refs: List[str],
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
    batch_size: int = 64,
    show_mismatches_below: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Compute per-example cosine similarity using a Sentence-Transformers encoder.
    Returns:
      {
        "mean": float,
        "per_example": List[float],
        "below_threshold": Optional[List[Tuple[int, float, str, str]]],  # (idx, score, pred, ref)
      }
    """
    if _sentence_transformers is None:
        raise RuntimeError("sentence-transformers not installed. `pip install sentence-transformers`")

    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(model_name)

    # Encode with normalization for stable cosines
    ref_vecs = st.encode(refs, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True)
    pred_vecs = st.encode(preds, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True)

    scores = np.sum(ref_vecs * pred_vecs, axis=1)  # cosine since already normalized
    result: Dict[str, Any] = {
        "mean": float(np.mean(scores)),
        "per_example": scores.tolist(),
    }

    if show_mismatches_below is not None:
        below = []
        for i, s in enumerate(scores):
            if s < show_mismatches_below:
                below.append((i, float(s), preds[i], refs[i]))
        result["below_threshold"] = below
    return result


# --------------------------
# Embedding cosine similarity (OpenAI or HF)
# --------------------------
def embeddings_similarity(
    preds: List[str],
    refs: List[str],
    backend: str = "openai",                # "openai" | "hf"
    model_name: str = "text-embedding-3-large",  # or any SentenceTransformer model if backend="hf"
    batch_size: int = 64,
    api_base: Optional[str] = None,         # for self-hosted OpenAI-compatible endpoints
    show_mismatches_below: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Compute cosine similarity between embeddings from either OpenAI or HF (Sentence-Transformers).
    Returns structure same as sts_similarity_hf.
    """
    if backend not in {"openai", "hf"}:
        raise ValueError("backend must be 'openai' or 'hf'")

    if backend == "openai":
        if _openai is None:
            raise RuntimeError("openai library not installed. `pip install openai` (>=1.40 recommended)")
        # OpenAI v1 client
        from openai import OpenAI
        os.environ["OPENAI_API_KEY"] = "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"
        client = OpenAI(base_url=api_base or None)

        # Batch: we’ll embed preds then refs
        def _embed_batch(texts: List[str]) -> np.ndarray:
            # OpenAI embeddings API supports batching; but we keep it simple and chunk
            out = []
            for i in range(0, len(texts), batch_size):
                chunk = texts[i:i + batch_size]
                resp = client.embeddings.create(model=model_name, input=chunk)
                vecs = [np.array(e.embedding, dtype=np.float32) for e in resp.data]
                out.append(np.stack(vecs, axis=0))
            return np.concatenate(out, axis=0) if out else np.zeros((0, 0), dtype=np.float32)

        ref_vecs = _embed_batch(refs)
        pred_vecs = _embed_batch(preds)

    else:  # backend == "hf"
        if _sentence_transformers is None:
            raise RuntimeError("sentence-transformers not installed. `pip install sentence-transformers`")
        from sentence_transformers import SentenceTransformer
        st = SentenceTransformer(model_name)
        ref_vecs = st.encode(refs, batch_size=batch_size, convert_to_numpy=True)
        pred_vecs = st.encode(preds, batch_size=batch_size, convert_to_numpy=True)

    # Cosine similarity per pair
    scores = _pairwise_cosine(ref_vecs, pred_vecs)
    result: Dict[str, Any] = {
        "mean": float(np.mean(scores)),
        "per_example": scores.tolist(),
    }

    if show_mismatches_below is not None:
        below = []
        for i, s in enumerate(scores):
            if s < show_mismatches_below:
                below.append((i, float(s), preds[i], refs[i]))
        result["below_threshold"] = below

    return result
