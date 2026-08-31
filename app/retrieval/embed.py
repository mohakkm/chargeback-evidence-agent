"""
Evidence document embedding module.

Loads multilingual-e5-large from sentence-transformers and indexes all
evidence docs from train.jsonl + holdout.jsonl into the Qdrant
'evidence_docs' collection.

Embedding format for multilingual-e5-large:
  - Query:    "query: <text>"
  - Passage:  "passage: <text>"
"""

import json
import sys
import uuid
from pathlib import Path
from typing import List, Dict, Any

# Ensure project root is first on sys.path and script dir is removed to prevent module name shadowing
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = str(Path(__file__).resolve().parent)
if SCRIPT_DIR in sys.path:
    sys.path.remove(SCRIPT_DIR)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from app.retrieval.qdrant_client import (
    COLLECTION_NAME,
    ensure_collection,
    get_qdrant_client,
)

MODEL_NAME = "intfloat/multilingual-e5-large"
DATASETS_DIR = Path(__file__).parents[1] / "data" / "datasets"
BATCH_SIZE = 32  # sentence-transformers batch size


def _load_model():
    """Lazy-load the SentenceTransformer model to avoid import-time overhead."""
    try:
        from sentence_transformers import SentenceTransformer
        print(f"[Embed] Loading model: {MODEL_NAME}")
        model = SentenceTransformer(MODEL_NAME)
        print(f"[Embed] Model loaded — embedding dim: {model.get_sentence_embedding_dimension()}")
        return model
    except ImportError as e:
        raise ImportError(
            "sentence-transformers is required. Install with: pip install sentence-transformers"
        ) from e


def embed_texts(texts: List[str], model=None) -> List[List[float]]:
    """
    Embeds a list of passage texts using multilingual-e5-large.
    Prepends 'passage: ' prefix per the model's recommended usage.
    """
    if model is None:
        model = _load_model()
    prefixed = [f"passage: {t}" for t in texts]
    embeddings = model.encode(prefixed, batch_size=BATCH_SIZE, show_progress_bar=True, normalize_embeddings=True)
    return embeddings.tolist()


def embed_query(query_text: str, model=None) -> List[float]:
    """
    Embeds a single query string for retrieval.
    Prepends 'query: ' prefix per the model's recommended usage.
    """
    if model is None:
        model = _load_model()
    embedding = model.encode(
        f"query: {query_text}",
        normalize_embeddings=True,
    )
    return embedding.tolist()


def _extract_evidence_docs(jsonl_path: Path) -> List[Dict[str, Any]]:
    """Reads all evidence doc objects from a JSONL dataset file."""
    docs = []
    if not jsonl_path.exists():
        print(f"[Embed] WARNING: Dataset not found at {jsonl_path} — skipping.")
        return docs

    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            case = json.loads(line)
            for ev in case.get("_evidence_docs_obj", []):
                docs.append(ev)
    return docs


def _stable_point_id(evidence_id: str) -> str:
    """
    Derives a stable UUID v5 from the evidence_id string so re-indexing
    the same doc always upserts to the same Qdrant point ID.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, evidence_id))


def index_evidence_docs(
    client: QdrantClient | None = None,
    force_reindex: bool = True,
) -> int:
    """
    Embeds all evidence documents from train.jsonl and holdout.jsonl and
    upserts them into the Qdrant 'evidence_docs' collection.

    Args:
        client: Optional pre-created QdrantClient (creates one if not provided).
        force_reindex: If True, drops and recreates the collection first.

    Returns:
        Total number of points indexed.
    """
    if client is None:
        client = get_qdrant_client()

    if force_reindex:
        print("[Embed] Force reindex requested — clearing points for fresh reindex.")
        try:
            from qdrant_client.models import Filter
            client.delete(collection_name=COLLECTION_NAME, points_selector=Filter())
        except Exception as e:
            print(f"[Embed] Warning clearing points: {e}")
        ensure_collection(client)
    else:
        ensure_collection(client)

    # Load evidence docs from both splits
    all_docs: List[Dict[str, Any]] = []
    for split in ("train.jsonl", "holdout.jsonl"):
        docs = _extract_evidence_docs(DATASETS_DIR / split)
        print(f"[Embed] Loaded {len(docs)} evidence docs from {split}")
        all_docs.extend(docs)

    if not all_docs:
        print("[Embed] No evidence docs found. Run generate_synthetic.py first.")
        return 0

    # Deduplicate by evidence_id
    seen: set[str] = set()
    unique_docs = []
    for doc in all_docs:
        eid = doc["evidence_id"]
        if eid not in seen:
            seen.add(eid)
            unique_docs.append(doc)

    print(f"[Embed] Total unique evidence docs to index: {len(unique_docs)}")

    # Embed in batches
    model = _load_model()
    texts = [doc["content"] for doc in unique_docs]
    vectors = embed_texts(texts, model=model)

    # Build Qdrant points
    points = []
    for doc, vector in zip(unique_docs, vectors):
        point_id = _stable_point_id(doc["evidence_id"])
        points.append(
            PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "evidence_id": doc["evidence_id"],
                    "transaction_id": doc["transaction_id"],
                    "doc_type": doc["doc_type"],
                    "quality": doc["quality"],
                    "content": doc["content"],
                },
            )
        )

    # Upsert in batches
    total_upserted = 0
    for i in range(0, len(points), BATCH_SIZE):
        batch = points[i : i + BATCH_SIZE]
        client.upsert(collection_name=COLLECTION_NAME, points=batch)
        total_upserted += len(batch)
        print(f"[Embed] Upserted {total_upserted}/{len(points)} points...")

    print(f"[Embed] [OK] Indexing complete. {total_upserted} evidence docs in '{COLLECTION_NAME}'.")
    return total_upserted


if __name__ == "__main__":
    indexed = index_evidence_docs(force_reindex=True)
    print(f"Done. {indexed} evidence documents indexed into Qdrant.")
