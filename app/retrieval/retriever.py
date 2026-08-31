"""
Evidence retrieval module per Phase 2 of CHECKLIST.md.

Retrieves top-k evidence documents from Qdrant filtered to transaction_id.
Strips ground-truth 'quality' evaluation fields before returning payloads to the agent.
Computes low_coverage flag for the bounded action gate if retrieved docs < 2.
"""

from typing import List, Dict, Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from app.retrieval.qdrant_client import COLLECTION_NAME, ensure_collection, get_qdrant_client
from app.retrieval.embed import embed_query, _load_model

# Reason-code -> domain hints to augment vector query representation
REASON_CODE_HINTS: Dict[str, str] = {
    "goods_not_received": "delivery confirmation shipping tracking proof of delivery carrier receipt",
    "goods_defective": "product condition defect return communication customer support email ticket",
    "duplicate_charge": "order details authentication log duplicate invoice billing statement",
    "credit_not_processed": "refund policy store credit email acknowledgement return receipt",
    "subscription_canceled_but_charged": "cancellation confirmation subscription policy billing log",
    "unauthorized_transaction": "authentication log IP address 3DS OTP device fingerprint MAC address",
}

DEFAULT_TOP_K = 5


def strip_evidence_quality(evidence_doc: Dict[str, Any]) -> Dict[str, Any]:
    """
    Strips evaluation-only ground truth 'quality' field from evidence payload
    so it never reaches the decision agent pipeline.
    """
    clean_doc = evidence_doc.copy()
    clean_doc.pop("quality", None)
    return clean_doc


class EvidenceRetriever:
    """
    Retrieves evidence documents from Qdrant for a dispute case.
    Filters by transaction_id and strips ground-truth evaluation fields.
    """

    def __init__(self, client: Optional[QdrantClient] = None):
        self._client = client or get_qdrant_client()
        ensure_collection(self._client)
        self._model = _load_model()

    def build_query_text(self, dispute: Dict[str, Any]) -> str:
        """
        Constructs query string from dispute_reason_code + customer_claim_text
        plus domain vocabulary hints.
        """
        reason_code = dispute.get("dispute_reason_code", "")
        claim_text = dispute.get("customer_claim_text", "")
        hint = REASON_CODE_HINTS.get(reason_code, "")
        return f"{reason_code} {claim_text} {hint}".strip()

    def retrieve(
        self,
        dispute: Dict[str, Any],
        top_k: int = DEFAULT_TOP_K,
        filter_by_transaction: bool = True,
        strip_quality: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Embeds query, searches Qdrant filtered to transaction_id, and returns top-k docs.
        """
        query_text = self.build_query_text(dispute)
        query_vector = embed_query(query_text, model=self._model)

        qdrant_filter = None
        if filter_by_transaction:
            txn_id = dispute.get("transaction_id")
            if txn_id:
                qdrant_filter = Filter(
                    must=[
                        FieldCondition(
                            key="transaction_id",
                            match=MatchValue(value=txn_id),
                        )
                    ]
                )

        response = self._client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        )

        evidence_list = []
        for hit in response.points:
            payload = dict(hit.payload or {})
            payload["_score"] = round(hit.score, 4)

            if strip_quality:
                payload = strip_evidence_quality(payload)

            evidence_list.append(payload)

        return evidence_list

    def retrieve_evidence_for_dispute(
        self,
        dispute: Dict[str, Any],
        top_k: int = DEFAULT_TOP_K,
    ) -> Dict[str, Any]:
        """
        Primary entrypoint for dispute evidence retrieval.

        Returns:
            Dict containing:
              - retrieved_evidence: List[Dict] (quality field stripped)
              - low_coverage: bool (True if transaction-scoped docs < 2)
              - retrieval_count: int
        """
        # Search Qdrant filtered strictly to transaction_id
        scoped_docs = self.retrieve(
            dispute=dispute,
            top_k=top_k,
            filter_by_transaction=True,
            strip_quality=True,
        )

        scoped_count = len(scoped_docs)
        low_coverage = scoped_count < 2
        retrieved_docs = list(scoped_docs)

        # Fallback to global search if case-scoped docs are missing/sparse (< 2)
        if low_coverage:
            global_docs = self.retrieve(
                dispute=dispute,
                top_k=top_k,
                filter_by_transaction=False,
                strip_quality=True,
            )
            seen_ids = {doc.get("evidence_id") for doc in retrieved_docs}
            for doc in global_docs:
                if doc.get("evidence_id") not in seen_ids:
                    retrieved_docs.append(doc)
                    seen_ids.add(doc.get("evidence_id"))

        return {
            "retrieved_evidence": retrieved_docs,
            "low_coverage": low_coverage,
            "retrieval_count": len(retrieved_docs),
            "scoped_count": scoped_count,
        }

    def retrieve_for_dispute(
        self,
        dispute: Dict[str, Any],
        top_k: int = DEFAULT_TOP_K,
    ) -> List[Dict[str, Any]]:
        """
        Returns list of evidence payload dicts (quality stripped) for direct caller usage.
        """
        result = self.retrieve_evidence_for_dispute(dispute, top_k=top_k)
        return result["retrieved_evidence"]


if __name__ == "__main__":
    retriever = EvidenceRetriever()
    sample_dispute = {
        "transaction_id": "txn_0001_2824",
        "dispute_reason_code": "goods_not_received",
        "customer_claim_text": "I ordered the product two weeks ago and never received it. Tracking has not updated.",
    }
    output = retriever.retrieve_evidence_for_dispute(sample_dispute, top_k=3)
    print("Retrieval output keys:", list(output.keys()))
    print("Retrieved count:", output["retrieval_count"])
    print("Low coverage flag:", output["low_coverage"])
    if output["retrieved_evidence"]:
        print("Sample doc keys (quality stripped):", list(output["retrieved_evidence"][0].keys()))
