"""
Synthetic Dispute + Evidence Dataset Generator.

Generates labeled dispute cases and associated evidence documents according to Phase 1 of CHECKLIST.md.
Supports both Claude API batch generation and local deterministic fallback.
"""

import json
import os
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple
from pydantic import BaseModel, Field

# Constants & Paths
DATASETS_DIR = Path(__file__).parent / "datasets"

REASON_CODES = [
    "goods_not_received",
    "goods_defective",
    "duplicate_charge",
    "credit_not_processed",
    "subscription_canceled_but_charged",
    "unauthorized_transaction"
]

MERCHANT_CATEGORIES = [
    "ecommerce",
    "subscription",
    "services"
]

DOC_TYPES = [
    "delivery_confirmation",
    "communication_log",
    "order_details",
    "refund_policy",
    "shipping_tracking",
    "auth_log"
]

QUALITIES = ["strong", "weak", "missing"]


class EvidenceDoc(BaseModel):
    evidence_id: str
    transaction_id: str
    doc_type: str
    content: str
    quality: str


class DisputeCase(BaseModel):
    case_id: str
    transaction_id: str
    merchant_category: str
    dispute_reason_code: str
    dispute_amount: float
    dispute_raised_date: str
    response_deadline: str
    customer_claim_text: str
    evidence_doc_ids: List[str]
    # Evaluation-only fields — must be stripped before passing to retrieval/agent pipeline
    label_winnable: bool
    ground_truth_rationale: str


def strip_eval_fields(case_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Strips evaluation-only fields (label_winnable and ground_truth_rationale)
    from a dispute case dictionary before passing to retrieval/agent pipeline.
    """
    clean_case = case_dict.copy()
    clean_case.pop("label_winnable", None)
    clean_case.pop("ground_truth_rationale", None)
    return clean_case


class SyntheticDataGenerator:
    """
    Generates synthetic dispute cases and corresponding evidence documents.
    """

    def __init__(self, seed: int = 42):
        random.seed(seed)
        DATASETS_DIR.mkdir(parents=True, exist_ok=True)
        self.api_key = os.environ.get("ANTHROPIC_API_KEY")

    def _generate_fallback_evidence(
        self,
        case_id: str,
        txn_id: str,
        reason_code: str,
        merchant_cat: str,
        winnable: bool,
        is_ambiguous: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Generates 2-4 evidence documents consistent with winnable label and reason code.
        Enforces auth_log for unauthorized_transaction.
        """
        num_docs = random.randint(2, 4)
        evidence_docs = []

        if reason_code == "unauthorized_transaction":
            # Mandatory auth_log for unauthorized transactions
            doc_types = ["auth_log"]
            secondary_pool = ["communication_log", "order_details", "refund_policy"]
            random.shuffle(secondary_pool)
            doc_types.extend(secondary_pool[: num_docs - 1])
        else:
            if reason_code in ["goods_not_received", "goods_defective"]:
                primary_pool = ["delivery_confirmation", "shipping_tracking", "order_details", "communication_log"]
            elif reason_code in ["subscription_canceled_but_charged", "credit_not_processed"]:
                primary_pool = ["communication_log", "refund_policy", "order_details"]
            elif reason_code == "duplicate_charge":
                primary_pool = ["order_details", "communication_log", "auth_log"]
            else:
                primary_pool = ["order_details", "communication_log", "refund_policy"]

            # Exclude auth_log unless duplicate_charge or explicitly chosen
            random.shuffle(primary_pool)
            doc_types = primary_pool[:num_docs]

        for i, doc_type in enumerate(doc_types):
            ev_id = f"EVD-{case_id.replace('CB-', '')}-{i+1:02d}"

            if is_ambiguous:
                quality = "weak" if i == 0 else random.choice(["strong", "missing"])
            elif winnable:
                quality = "strong" if i < 2 else random.choice(["strong", "weak"])
            else:
                quality = "weak" if i == 0 else random.choice(["weak", "missing"])

            content = self._build_doc_content(doc_type, reason_code, winnable, quality, txn_id)

            evidence_docs.append({
                "evidence_id": ev_id,
                "transaction_id": txn_id,
                "doc_type": doc_type,
                "content": content,
                "quality": quality
            })

        return evidence_docs

    def _build_doc_content(self, doc_type: str, reason_code: str, winnable: bool, quality: str, txn_id: str) -> str:
        if doc_type == "auth_log":
            if winnable and quality == "strong":
                return (
                    f"Authentication Log [{txn_id}]: IP Address 198.51.100.45 matches customer's registered home network. "
                    "2FA OTP successfully verified via SMS (+91-9876543210). Device Fingerprint: Chrome 127.0 on macOS. "
                    "3DS v2.2 Challenge Completed successfully with frictionless authentication."
                )
            elif quality == "weak":
                return (
                    f"Authentication Log [{txn_id}]: Single-factor web transaction from IP 203.0.113.12 (VPN endpoint). "
                    "3DS Status: Attempted / Not Enrolled. No device fingerprint on record."
                )
            else: # missing
                return f"Authentication Log [{txn_id}]: Standard card-not-present transaction record. 3DS metadata unavailable."

        elif doc_type in ["delivery_confirmation", "shipping_tracking"]:
            if winnable and quality == "strong":
                return (
                    f"Shipping & Delivery Proof [{txn_id}]: Carrier BlueDart Tracking #BD98765432. "
                    "Status: DELIVERED on 2026-08-12 at 14:22 IST. Delivery location: Customer door. "
                    "Proof of Delivery: OTP confirmed by recipient at doorstep."
                )
            elif quality == "weak":
                return (
                    f"Shipping & Delivery Proof [{txn_id}]: Carrier Delhivery Tracking #DL11223344. "
                    "Status: IN_TRANSIT / Delayed at hub. Last scanned 2026-08-11. Delivery status unconfirmed."
                )
            else:
                return f"Shipping & Delivery Proof [{txn_id}]: Tracking number generated. No carrier scan data available."

        elif doc_type == "communication_log":
            if winnable and quality == "strong":
                return (
                    f"Customer Communication Log [{txn_id}]: Email thread on 2026-08-14. Customer support sent user manual "
                    "and replacement confirmation. Customer replied: 'Thank you, received and working fine now!'"
                )
            elif quality == "weak":
                return (
                    f"Customer Communication Log [{txn_id}]: Support ticket #TCK-4091. Customer inquired about refund status on 2026-08-11. "
                    "Support agent replied with automated response: 'Your ticket is being reviewed by finance.'"
                )
            else:
                return f"Customer Communication Log [{txn_id}]: No customer communication logs found in CRM."

        elif doc_type == "refund_policy":
            return (
                f"Merchant Refund & Cancellation Policy [{txn_id}]: Digital subscriptions must be canceled 48 hours prior to renewal. "
                "Non-refundable after billing date unless requested within 24 hours of first sign-up. Physical goods eligible for return within 15 days."
            )

        elif doc_type == "order_details":
            return (
                f"Order Details [{txn_id}]: Order Placed: 2026-08-09. Item: Premium Wireless Headphones x 1. "
                "Billing Address matches Shipping Address. Customer Account Age: 14 months (12 successful past orders)."
            )

        return f"Document Record [{txn_id}]: General evidence documentation record."

    def _generate_fallback_case(self, case_num: int, is_ambiguous: bool = False) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        case_id = f"CB-{case_num:05d}"
        txn_id = f"txn_{case_num:04d}_{random.randint(1000, 9999)}"
        reason_code = random.choice(REASON_CODES)
        merchant_cat = random.choice(MERCHANT_CATEGORIES)
        amount = round(random.uniform(299.0, 4999.0), 2)

        # Pre-assign winnable (~55% winnable, ~45% unwinnable)
        if is_ambiguous:
            winnable = False  # Edge case for confidence gate testing
        else:
            winnable = random.random() < 0.55

        claim_texts = {
            "goods_not_received": "I ordered the product two weeks ago and never received it. Tracking has not updated.",
            "goods_defective": "The item arrived damaged and unusable. Support refused to send a replacement.",
            "duplicate_charge": "My card was charged twice for the same single purchase on the website.",
            "credit_not_processed": "I returned the item per instructions but never received the agreed store credit/refund.",
            "subscription_canceled_but_charged": "I canceled my recurring subscription last week, but was charged again today.",
            "unauthorized_transaction": "I do not recognize this transaction on my card statement. It was made without my permission."
        }

        customer_claim = claim_texts.get(reason_code, "Customer disputes transaction amount.")

        if is_ambiguous:
            rationale = f"Ambiguous case: Evidence for {reason_code} contains conflicting records and partial documentation."
        elif winnable:
            rationale = f"Merchant has conclusive proof (strong quality evidence) refuting the {reason_code} claim."
        else:
            rationale = f"Merchant lacks sufficient proof (weak/missing evidence) to disprove customer's {reason_code} claim."

        evidence_docs = self._generate_fallback_evidence(case_id, txn_id, reason_code, merchant_cat, winnable, is_ambiguous)
        ev_ids = [d["evidence_id"] for d in evidence_docs]

        case_obj = {
            "case_id": case_id,
            "transaction_id": txn_id,
            "merchant_category": merchant_cat,
            "dispute_reason_code": reason_code,
            "dispute_amount": amount,
            "dispute_raised_date": "2026-08-10",
            "response_deadline": "2026-08-24",
            "customer_claim_text": customer_claim,
            "evidence_doc_ids": ev_ids,
            "label_winnable": winnable,
            "ground_truth_rationale": rationale
        }

        return case_obj, evidence_docs

    def generate_batch_claude(self, batch_size: int, start_idx: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Attempts generation using Anthropic Claude API.
        Falls back to local generator if API is unavailable or fails.
        """
        if not self.api_key:
            return self._generate_batch_fallback(batch_size, start_idx)

        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)

            prompt = f"""Generate {batch_size} realistic synthetic chargeback dispute cases starting from ID index {start_idx}.
Reason codes taxonomy: {json.dumps(REASON_CODES)}
Merchant categories: {json.dumps(MERCHANT_CATEGORIES)}

Rules:
1. Assign "label_winnable" (boolean) FIRST (~55% true, ~45% false).
2. Generate 2-4 evidence documents per case consistent with "label_winnable".
3. For "unauthorized_transaction", evidence MUST use "auth_log" doc_type (IP, 3DS, device fingerprint, OTP), NOT delivery/shipping.
4. Each evidence doc must have "quality" in ["strong", "weak", "missing"].
5. Output valid JSON array containing objects with keys: "case" and "evidence_docs".

Respond ONLY with valid JSON."""

            response = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=4000,
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = response.content[0].text
            parsed = json.loads(response_text)

            cases = []
            all_evidence = []
            for item in parsed:
                cases.append(item["case"])
                all_evidence.extend(item["evidence_docs"])

            return cases, all_evidence

        except Exception as e:
            print(f"[WARN] Claude API call failed ({e}). Using deterministic fallback generator.")
            return self._generate_batch_fallback(batch_size, start_idx)

    def _generate_batch_fallback(self, batch_size: int, start_idx: int, ambiguous_count: int = 0) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        cases = []
        all_evidence = []

        for i in range(batch_size):
            case_num = start_idx + i
            is_ambiguous = (i < ambiguous_count)
            case_obj, ev_docs = self._generate_fallback_case(case_num, is_ambiguous=is_ambiguous)
            cases.append(case_obj)
            all_evidence.extend(ev_docs)

        return cases, all_evidence

    def generate_full_dataset(self, total_cases: int = 120, train_ratio: float = 0.75) -> Dict[str, Any]:
        """
        Generates 120 total dispute cases, splits 75/25 into train.jsonl and holdout.jsonl,
        seeds 4-5 ambiguous cases into holdout, and saves files.
        """
        num_train = int(total_cases * train_ratio)  # 90
        num_holdout = total_cases - num_train      # 30

        print(f"Generating {total_cases} dispute cases ({num_train} train, {num_holdout} holdout)...")

        # Generate train cases (90 cases)
        train_cases, train_ev = self._generate_batch_fallback(num_train, start_idx=1)

        # Generate holdout cases (30 cases, with 5 seeded ambiguous cases)
        holdout_cases, holdout_ev = self._generate_batch_fallback(num_holdout, start_idx=num_train + 1, ambiguous_count=5)

        # Merge evidence docs into cases or separate if needed
        # Save datasets to disk
        train_file = DATASETS_DIR / "train.jsonl"
        holdout_file = DATASETS_DIR / "holdout.jsonl"

        self._save_jsonl(train_file, train_cases, train_ev)
        self._save_jsonl(holdout_file, holdout_cases, holdout_ev)

        print(f"Successfully generated dataset:")
        print(f"  - Train cases: {len(train_cases)} -> {train_file}")
        print(f"  - Holdout cases: {len(holdout_cases)} (including 5 ambiguous cases) -> {holdout_file}")

        return {
            "train_cases_count": len(train_cases),
            "holdout_cases_count": len(holdout_cases),
            "train_file": str(train_file),
            "holdout_file": str(holdout_file)
        }

    def _save_jsonl(self, filepath: Path, cases: List[Dict[str, Any]], evidence: List[Dict[str, Any]]):
        """
        Saves dispute cases along with embedded evidence docs list to JSONL format.
        """
        ev_map = {}
        for ev in evidence:
            ev_map[ev["evidence_id"]] = ev

        with open(filepath, "w", encoding="utf-8") as f:
            for case in cases:
                # Include resolved evidence objects for convenient evaluation record-keeping
                record = case.copy()
                record["_evidence_docs_obj"] = [
                    ev_map[eid] for eid in case["evidence_doc_ids"] if eid in ev_map
                ]
                f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    generator = SyntheticDataGenerator()
    generator.generate_full_dataset(total_cases=120, train_ratio=0.75)
