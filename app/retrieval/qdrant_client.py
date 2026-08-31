"""
Qdrant local/embedded client wrapper.

Creates and manages a single Qdrant collection 'evidence_docs' backed by
a local on-disk database at ./qdrant_data (no server / Docker required).

Note: Payload indexes are created for forward-compatibility with a Qdrant
server deployment; they have no effect in local/embedded mode and the
associated UserWarning is suppressed below.
"""

import warnings
from pathlib import Path
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PayloadSchemaType,
)

COLLECTION_NAME = "evidence_docs"
VECTOR_SIZE = 1024  # multilingual-e5-large output dimension
QDRANT_DATA_PATH = str(Path(__file__).parents[2] / "qdrant_data")


def get_qdrant_client() -> QdrantClient:
    """
    Returns a QdrantClient connected to the local embedded store.
    Creates the directory if it does not exist.
    """
    Path(QDRANT_DATA_PATH).mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=QDRANT_DATA_PATH)
    return client


def ensure_collection(client: Optional[QdrantClient] = None) -> QdrantClient:
    """
    Ensures the 'evidence_docs' collection exists with correct vector config.
    Creates or recreates the collection if the vector size has changed.
    Returns the client for convenience.
    """
    if client is None:
        client = get_qdrant_client()

    existing = {c.name for c in client.get_collections().collections}

    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE,
            ),
        )
        # Create payload indexes for fast filtered retrieval (server mode)
        # Suppressed: local/embedded mode emits a harmless UserWarning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="transaction_id",
                field_schema=PayloadSchemaType.KEYWORD,
            )
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="doc_type",
                field_schema=PayloadSchemaType.KEYWORD,
            )
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="quality",
                field_schema=PayloadSchemaType.KEYWORD,
            )
        print(f"[Qdrant] Created collection '{COLLECTION_NAME}' at {QDRANT_DATA_PATH}")
    else:
        # Validate vector size matches — re-create if stale
        info = client.get_collection(COLLECTION_NAME)
        existing_size = info.config.params.vectors.size  # type: ignore[union-attr]
        if existing_size != VECTOR_SIZE:
            client.recreate_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=VECTOR_SIZE,
                    distance=Distance.COSINE,
                ),
            )
            print(f"[Qdrant] Recreated collection '{COLLECTION_NAME}' (size mismatch: {existing_size} → {VECTOR_SIZE})")
        else:
            print(f"[Qdrant] Collection '{COLLECTION_NAME}' already exists ({info.points_count} points)")

    return client
