# source_code/indexing/index_builder.py
import os, json, glob
from typing import List, Dict, Any, Optional
import numpy as np

try:
    import faiss
except Exception:
    faiss = None

# Lazy import sentence-transformers to keep optional
_ST = None
def _get_st_model(name: str):
    global _ST
    if _ST is None:
        from sentence_transformers import SentenceTransformer
        _ST = SentenceTransformer(name, local_files_only=True)
    elif getattr(_ST, "model_card_data", None) and getattr(_ST.model_card_data, "modelId", "") != name:
        from sentence_transformers import SentenceTransformer
        _ST = SentenceTransformer(name, local_files_only=True)
    return _ST

# -----------------------------
# Corpus ingestion
# -----------------------------
def load_corpus_from_folder(corpus_dir: str) -> List[Dict[str, Any]]:
    """
    Reads a folder of text-like files (.txt, .md, .jsonl).
    For .jsonl, expects {"text": "..."} (additional fields kept in meta).
    Returns: [{"id": str, "text": str, "path": str, "meta": {...}}, ...]
    """
    os.makedirs(corpus_dir, exist_ok=True)
    out = []
    # text-like files
    '''
    for ext in ("*.txt", "*.md"):
        for p in glob.glob(os.path.join(corpus_dir, ext)):
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                txt = f.read().strip()
            if not txt:
                continue
            out.append({"id": os.path.relpath(p, corpus_dir), "text": txt, "path": p, "meta": {}})
    '''
    # jsonl files
    for p in glob.glob(os.path.join(corpus_dir, "*.jsonl")):
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                txt = obj.get("chunk") or obj.get("content") or ""
                if not txt:
                    print("no txt")
                    continue
                meta = {k:v for k,v in obj.items() if k not in {"chunk", "id", "content"}}
                id_ = obj.get("id")
                out.append({"id": id_, "text": txt, "path": p, "meta": meta})
    return out

# -----------------------------
# FAISS: build / persist / load
# -----------------------------
def _normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / n

def _paths(index_dir: str):
    return (
        os.path.join(index_dir, "index.faiss"),
        os.path.join(index_dir, "meta.jsonl"),
        os.path.join(index_dir, "embed_model.txt"),
    )

def build_or_load_faiss_index(
    corpus_dir: str,
    index_dir: str,
    embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    rebuild_if_model_mismatch: bool = True,
):
    """
    If index exists and matches embed model → load.
    Else builds from scratch using files under corpus_dir.
    Returns: (faiss.Index, List[dict] meta)
    """
    if faiss is None:
        raise RuntimeError("faiss is not installed. Please install faiss-cpu or faiss-gpu.")

    os.makedirs(index_dir, exist_ok=True)
    index_path, meta_path, model_tag_path = _paths(index_dir)

    # Quick reuse path
    if os.path.exists(index_path) and os.path.exists(meta_path) and os.path.exists(model_tag_path):
        with open(model_tag_path, "r", encoding="utf-8") as f:
            model_tag = f.read().strip()
        if (model_tag == embed_model_name) or (not rebuild_if_model_mismatch):
            index = faiss.read_index(index_path)
            meta = [json.loads(l) for l in open(meta_path, "r", encoding="utf-8")]
            return index, meta

    # (Re)build
    corpus = load_corpus_from_folder(corpus_dir)
    if not corpus:
        raise ValueError(f"No documents found under corpus_dir={corpus_dir}")

    # embed
    st = _get_st_model(embed_model_name)
    texts = [d["text"] for d in corpus]
    embs = st.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
    embs = _normalize(embs.astype("float32"))

    # index (cosine via inner product on normalized vectors)
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs)

    # persist
    faiss.write_index(index, index_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        for d in corpus:
            f.write(json.dumps(d) + "\n")
    with open(model_tag_path, "w", encoding="utf-8") as f:
        f.write(embed_model_name)

    return index, corpus

class FaissRetriever:
    """
    Callable retriever:
      docs = FaissRetriever(...)(query)
    Returns a list of dicts with .text and .meta
    """
    def __init__(self, index, meta: List[Dict[str, Any]], embed_model_name: str):
        self.index = index
        self.meta = meta
        self.embed_model_name = embed_model_name
        self._st = _get_st_model(embed_model_name)

    def __call__(self, query: str, top_k: int = 25) -> List[Dict[str, Any]]:
        q = self._st.encode([query], convert_to_numpy=True).astype("float32")
        q = _normalize(q)
        scores, idxs = self.index.search(q, top_k)
        docs = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1: continue
            d = self.meta[idx]
            docs.append({
                "id": d["id"],
                "text": d["text"],
                "path": d["path"],
                "meta": d.get("meta", {}),
                "score": float(score),
            })
        return docs

# -----------------------------
# Rankify corpus export helper
# -----------------------------
def export_corpus_jsonl_for_rankify(corpus_dir: str, out_jsonl: str):
    """
    Writes a simple JSONL corpus file with fields {id,text,path}.
    Rankify can consume its own corpus/index format; this gives it a plaintext corpus.
    """
    corpus = load_corpus_from_folder(corpus_dir)
    os.makedirs(os.path.dirname(out_jsonl), exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for d in corpus:
            f.write(json.dumps({"id": d["id"], "text": d["text"], "path": d["path"]}) + "\n")
    return out_jsonl
