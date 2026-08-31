"""
Decision Agent Module (Phase 3).

Evaluates dispute cases and retrieved evidence documents using Groq LLM (openai/gpt-oss-120b).
Outputs structured JSON decisions:
- decision: "contest" | "no_contest"
- confidence: float (0.0 to 1.0)
- rebuttal_draft: formal merchant evidence packet text (if contesting)
- reasoning_summary: 2-3 sentence internal audit note explaining rationale
- used_fallback: bool - True when heuristic reasoner was used instead of Groq

STRICT SECURITY RULE:
The decision agent NEVER receives evaluation-only ground-truth fields:
- label_winnable
- ground_truth_rationale
- quality

These fields are explicitly stripped at the function signature / sanitization level.

CONFIDENCE CALIBRATION RULE:
Post-validation safeguard: if low_coverage=True, returned confidence is capped at 0.45
regardless of the model's raw output. This is confidence calibration only - it never
changes the decision itself. Action-gate hard-blocking lives downstream.

RATE LIMIT & PACING RULE:
Paces requests using GROQ_MIN_REQUEST_INTERVAL_SECONDS (default 16s).
On HTTP 429 rate limit errors, retries up to GROQ_MAX_RATE_LIMIT_RETRIES (default 3)
with delay parsed from error text (X + 1s buffer) or bounded exponential backoff.
"""

import json
import os
import re
import time
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field

from app.config import GROQ_API_KEY, GROQ_MODEL

# Maximum confidence allowed when low_coverage=True (calibration only, not action-gate logic)
_LOW_COVERAGE_CONFIDENCE_CAP = 0.45


class CleanDisputeInput(BaseModel):
    """Sanitized dispute case input - guaranteed free of ground-truth evaluation fields."""
    dispute_reason_code: str
    customer_claim_text: str
    dispute_amount: Optional[float] = 0.0
    merchant_category: Optional[str] = "ecommerce"
    dispute_raised_date: Optional[str] = ""


class CleanEvidenceInput(BaseModel):
    """Sanitized evidence document input - guaranteed free of quality evaluation field."""
    evidence_id: str
    doc_type: str
    content: str
    score: Optional[float] = None


class DecisionAgentResponse(BaseModel):
    """Structured response schema returned by the decision agent."""
    decision: Literal["contest", "no_contest"] = Field(
        ..., description="Recommended dispute action: contest or no_contest"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score from 0.0 to 1.0"
    )
    rebuttal_draft: str = Field(
        ..., description="Drafted formal merchant rebuttal letter (populated if contesting)"
    )
    reasoning_summary: str = Field(
        ..., description="2-3 sentence internal rationale summary for audit logging"
    )
    used_fallback: bool = Field(
        ..., description="True if heuristic fallback reasoner was used instead of Groq LLM"
    )


def sanitize_dispute_input(dispute: Dict[str, Any]) -> CleanDisputeInput:
    """
    Explicitly strips ground-truth evaluation fields (label_winnable, ground_truth_rationale, quality)
    from a dispute dictionary before passing to the agent.
    """
    clean_dict = dispute.copy()
    clean_dict.pop("label_winnable", None)
    clean_dict.pop("ground_truth_rationale", None)
    clean_dict.pop("quality", None)
    return CleanDisputeInput(
        dispute_reason_code=str(clean_dict.get("dispute_reason_code", "")),
        customer_claim_text=str(clean_dict.get("customer_claim_text", "")),
        dispute_amount=float(clean_dict.get("dispute_amount", 0.0)),
        merchant_category=str(clean_dict.get("merchant_category", "ecommerce")),
        dispute_raised_date=str(clean_dict.get("dispute_raised_date", "")),
    )


def sanitize_evidence_input(evidence_list: List[Dict[str, Any]]) -> List[CleanEvidenceInput]:
    """
    Explicitly strips ground-truth evaluation fields (quality) from evidence payloads.
    """
    clean_items = []
    for item in evidence_list:
        clean_item = item.copy()
        clean_item.pop("quality", None)
        clean_items.append(CleanEvidenceInput(
            evidence_id=str(clean_item.get("evidence_id", "")),
            doc_type=str(clean_item.get("doc_type", "")),
            content=str(clean_item.get("content", "")),
            score=clean_item.get("_score"),
        ))
    return clean_items


def _apply_confidence_calibration(confidence: float, low_coverage: bool) -> float:
    """
    Post-validation confidence calibration safeguard.
    If low_coverage=True, caps confidence at _LOW_COVERAGE_CONFIDENCE_CAP (0.45).
    Never changes the decision - purely a confidence calibration step.
    """
    if low_coverage:
        return min(confidence, _LOW_COVERAGE_CONFIDENCE_CAP)
    return confidence


class DecisionAgent:
    """
    LLM Decision Agent leveraging Groq to reason over dispute claims and retrieved evidence.
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass

        from app.config import GROQ_API_KEY as CONFIG_API_KEY, GROQ_MODEL as CONFIG_MODEL

        self.api_key = api_key or os.environ.get("GROQ_API_KEY", "") or CONFIG_API_KEY
        self.model = model or os.environ.get("GROQ_MODEL", "") or CONFIG_MODEL or "openai/gpt-oss-120b"
        self._client = None
        self._last_request_time: Optional[float] = None

        if self.api_key and self.api_key not in ("your-anthropic-api-key-here", "dummy"):
            try:
                from groq import Groq
                self._client = Groq(api_key=self.api_key)
            except Exception as e:
                print(f"[DecisionAgent] Warning: Groq client initialization failed ({e}). Using fallback reasoner.")

    def _apply_pacing(self) -> None:
        """
        Enforces a minimum interval between requests (default 16 seconds)
        using time.monotonic().
        """
        min_interval_str = os.environ.get("GROQ_MIN_REQUEST_INTERVAL_SECONDS", "16.0")
        try:
            min_interval = float(min_interval_str)
        except ValueError:
            min_interval = 16.0

        if min_interval <= 0:
            self._last_request_time = time.monotonic()
            return

        now = time.monotonic()
        if self._last_request_time is not None:
            elapsed = now - self._last_request_time
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)

        self._last_request_time = time.monotonic()

    @staticmethod
    def _is_rate_limit_error(e: Exception) -> bool:
        """
        Detects if an exception represents a 429 / Rate Limit error.
        """
        status_code = getattr(e, "status_code", None)
        if status_code == 429:
            return True

        err_type = type(e).__name__
        if err_type in ("RateLimitError", "APIStatusError") and status_code == 429:
            return True

        err_str = str(e).lower()
        if "429" in err_str or "rate limit" in err_str or "rate_limit" in err_str or "try again in" in err_str:
            return True

        return False

    @staticmethod
    def _parse_retry_delay(e: Exception, attempt: int) -> float:
        """
        Parses retry delay from error message if available (adding 1 sec buffer),
        otherwise uses bounded exponential backoff [5.0, 10.0, 20.0].
        """
        err_msg = str(e)
        patterns = [
            r"try again in ([\d\.]+)\s*s",
            r"try again in ([\d\.]+)\s*second",
            r"in ([\d\.]+)\s*s",
            r"in ([\d\.]+)\s*second",
        ]
        for pattern in patterns:
            match = re.search(pattern, err_msg, re.IGNORECASE)
            if match:
                try:
                    parsed_val = float(match.group(1))
                    return max(parsed_val + 1.0, 1.0)
                except ValueError:
                    pass

        default_backoffs = [5.0, 10.0, 20.0]
        idx = min(attempt - 1, len(default_backoffs) - 1)
        return default_backoffs[max(0, idx)]

    def evaluate_dispute(
        self,
        dispute: CleanDisputeInput,
        evidence: List[CleanEvidenceInput],
        low_coverage: bool = False
    ) -> DecisionAgentResponse:
        """
        Evaluates dispute case alongside evidence documents to produce structured decision.

        Args:
            dispute: CleanDisputeInput (sanitized, zero eval fields)
            evidence: List[CleanEvidenceInput] (sanitized, zero quality fields)
            low_coverage: bool (True if retrieved evidence < 2 docs)

        Returns:
            DecisionAgentResponse (decision, confidence, rebuttal_draft, reasoning_summary, used_fallback)
        """
        if self._client:
            try:
                response = self._evaluate_with_groq(dispute, evidence, low_coverage)
                # Apply post-validation confidence calibration
                response.confidence = _apply_confidence_calibration(response.confidence, low_coverage)
                return response
            except Exception as e:
                print(f"[DecisionAgent] Groq API call failed ({e}). Falling back to heuristic reasoner.")

        response = self._evaluate_heuristic(dispute, evidence, low_coverage)
        # Apply post-validation confidence calibration on fallback path too
        response.confidence = _apply_confidence_calibration(response.confidence, low_coverage)
        return response

    def evaluate_dispute_dict(
        self,
        dispute_dict: Dict[str, Any],
        retrieval_output: Dict[str, Any]
    ) -> DecisionAgentResponse:
        """
        Convenience entrypoint accepting raw dicts.
        Enforces strict field isolation by sanitizing inputs before evaluation.
        """
        clean_dispute = sanitize_dispute_input(dispute_dict)
        raw_evidence = retrieval_output.get("retrieved_evidence", [])
        clean_evidence = sanitize_evidence_input(raw_evidence)
        low_coverage = retrieval_output.get("low_coverage", False)

        return self.evaluate_dispute(
            dispute=clean_dispute,
            evidence=clean_evidence,
            low_coverage=low_coverage
        )

    def _evaluate_with_groq(
        self,
        dispute: CleanDisputeInput,
        evidence: List[CleanEvidenceInput],
        low_coverage: bool
    ) -> DecisionAgentResponse:
        """
        Executes structured JSON reasoning via Groq Chat Completions API with rate-limit retries & pacing.
        Sets used_fallback=False on all successful Groq responses.
        """
        system_prompt = (
            "You are a Senior Chargeback Dispute Operations Specialist at Razorpay.\n"
            "Your task is to analyze credit card dispute claims against retrieved evidence documentation and decide whether the merchant should CONTEST the chargeback or submit NO CONTEST.\n\n"
            "RULES:\n"
            "1. Analyze the dispute reason code and card network rules (Visa/Mastercard/Amex/Discover).\n"
            "2. Evaluate whether the provided evidence documents prove customer authorization, proper delivery, non-defective condition, terms compliance, or valid processing.\n"
            "3. If evidence is compelling and refutes the customer claim, decide 'contest'.\n"
            "4. If evidence is missing, weak, or supports the customer claim, decide 'no_contest'.\n"
            "5. CONFIDENCE CALIBRATION PRINCIPLE: Confidence must reflect the amount and strength of evidence actually available, not merely whether a plausible decision can be suggested. "
            "If the available evidence is insufficient to make a confident determination in either direction, output low confidence regardless of which way you lean on the decision. "
            "Do not output high confidence simply because you can form a plausible argument - high confidence requires strong, specific, directly relevant evidence.\n"
            "6. low_coverage=True in the user prompt means fewer than 2 evidence documents were retrieved. "
            "Factor this into your raw confidence score naturally (sparse evidence -> lower confidence). "
            "Do not use low_coverage to override a 'contest' decision to 'no_contest' - make the best decision the evidence supports regardless.\n"
            "7. When decision is 'contest', write a formal, highly professional, point-by-point Merchant Rebuttal Letter (rebuttal_draft) addressed to the acquiring bank dispute committee. "
            "Cite specific order IDs, tracking numbers, timestamps, IP addresses, 2FA/3DS proof, or policy clauses from the evidence documents.\n"
            "8. When decision is 'no_contest', set rebuttal_draft to an empty string ''.\n"
            "9. Provide a concise 2-3 sentence reasoning_summary for internal audit logging.\n"
            "10. Output ONLY a valid JSON object matching the requested schema."
        )

        formatted_evidence = ""
        if not evidence:
            formatted_evidence = "No evidence documents retrieved for this dispute."
        else:
            for idx, ev in enumerate(evidence, start=1):
                formatted_evidence += f"\n[Doc #{idx}] ID: {ev.evidence_id} | Type: {ev.doc_type}\nContent: {ev.content}\n"

        user_prompt = (
            f"DISPUTE CASE DETAILS:\n"
            f"- Reason Code: {dispute.dispute_reason_code}\n"
            f"- Amount: ${dispute.dispute_amount:.2f}\n"
            f"- Merchant Category: {dispute.merchant_category}\n"
            f"- Customer Claim Text: \"{dispute.customer_claim_text}\"\n\n"
            f"RETRIEVED EVIDENCE DOCUMENTS ({len(evidence)} docs, low_coverage={low_coverage}):\n"
            f"{formatted_evidence}\n\n"
            f"Respond strictly with a JSON object containing keys:\n"
            f'{{"decision": "contest" | "no_contest", "confidence": <float 0-1>, "rebuttal_draft": "<string>", "reasoning_summary": "<string>"}}'
        )

        max_retries_str = os.environ.get("GROQ_MAX_RATE_LIMIT_RETRIES", "3")
        try:
            max_retries = int(max_retries_str)
        except ValueError:
            max_retries = 3

        attempt = 0
        response = None

        while True:
            self._apply_pacing()
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.1,
                    max_tokens=1024,
                )
                break
            except Exception as e:
                if self._is_rate_limit_error(e) and attempt < max_retries:
                    attempt += 1
                    wait_seconds = self._parse_retry_delay(e, attempt)
                    print(
                        f"[DecisionAgent] Groq rate limit hit (429). "
                        f"Waiting {wait_seconds:.1f}s before retry {attempt}/{max_retries}..."
                    )
                    time.sleep(wait_seconds)
                    self._last_request_time = time.monotonic()
                else:
                    raise e

        raw_json = response.choices[0].message.content
        data = json.loads(raw_json)

        decision = str(data.get("decision", "no_contest")).lower()
        if decision not in ("contest", "no_contest"):
            decision = "no_contest"

        confidence = float(data.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))

        rebuttal_draft = str(data.get("rebuttal_draft", "")) if decision == "contest" else ""
        reasoning_summary = str(data.get("reasoning_summary", "Dispute evaluated based on evidence availability."))

        return DecisionAgentResponse(
            decision=decision,
            confidence=confidence,
            rebuttal_draft=rebuttal_draft,
            reasoning_summary=reasoning_summary,
            used_fallback=False,
        )

    def _evaluate_heuristic(
        self,
        dispute: CleanDisputeInput,
        evidence: List[CleanEvidenceInput],
        low_coverage: bool
    ) -> DecisionAgentResponse:
        """
        Deterministic fallback heuristic when Groq API key is unavailable or offline.
        Analytically checks evidence availability and strength for dispute reason code.
        Always sets used_fallback=True.
        """
        reason_code = dispute.dispute_reason_code
        doc_types = [e.doc_type for e in evidence]

        # Key evidence doc requirements per reason code
        strong_evidence_found = False
        if reason_code in ("goods_not_received", "goods_defective"):
            if "delivery_confirmation" in doc_types and "order_details" in doc_types:
                strong_evidence_found = True
            elif any("DELIVERED" in e.content or "signature captured" in e.content for e in evidence):
                strong_evidence_found = True
        elif reason_code in ("subscription_canceled_but_charged", "credit_not_processed"):
            if "refund_policy" in doc_types or "communication_log" in doc_types:
                if any("canceled" in e.content.lower() or "working fine now" in e.content.lower() or "terms" in e.content.lower() for e in evidence):
                    strong_evidence_found = True
        elif reason_code in ("unauthorized_transaction", "duplicate_charge"):
            if "auth_log" in doc_types:
                if any("3DS" in e.content or "2FA OTP" in e.content or "verified" in e.content.lower() for e in evidence):
                    strong_evidence_found = True

        if strong_evidence_found and len(evidence) >= 2:
            decision = "contest"
            confidence = 0.88
            evidence_summary = "\n".join([f"- [{e.doc_type.upper()}] {e.content}" for e in evidence[:3]])
            rebuttal_draft = (
                f"REBUTTAL STATEMENT FOR DISPUTE {dispute.dispute_reason_code.upper()}\n"
                f"Merchant Category: {dispute.merchant_category.capitalize()}\n"
                f"Disputed Amount: ${dispute.dispute_amount:.2f}\n\n"
                f"Dear Dispute Resolution Committee,\n\n"
                f"We are formally contesting the chargeback claim initiated for transaction of ${dispute.dispute_amount:.2f}. "
                f"The cardholder alleges: '{dispute.customer_claim_text}'.\n\n"
                f"However, merchant records and verifiable evidence prove that the transaction was fully authorized and fulfilled:\n"
                f"{evidence_summary}\n\n"
                f"Per card network operating regulations, the submitted documentation refutes the cardholder's claim. "
                f"We respectfully request that the disputed funds be returned to the merchant."
            )
            reasoning_summary = (
                f"Strong evidence documents ({', '.join(doc_types)}) confirm authorization and fulfillment for {reason_code}. "
                f"Dispute is winnable; contesting with full evidence packet."
            )
        elif len(evidence) >= 1 and not low_coverage:
            decision = "contest"
            confidence = 0.58
            rebuttal_draft = (
                f"REBUTTAL STATEMENT FOR DISPUTE {dispute.dispute_reason_code.upper()}\n"
                f"Disputed Amount: ${dispute.dispute_amount:.2f}\n\n"
                f"The merchant presents available transaction records refuting the cardholder claim of '{dispute.customer_claim_text}'. "
                f"Available evidence details: {evidence[0].content}."
            )
            reasoning_summary = (
                f"Moderate evidence present for {reason_code}. Contesting with moderate confidence score."
            )
        elif len(evidence) >= 1 and low_coverage:
            if strong_evidence_found:
                decision = "contest"
                confidence = 0.70
                evidence_summary = evidence[0].content
                rebuttal_draft = (
                    f"REBUTTAL STATEMENT FOR DISPUTE {dispute.dispute_reason_code.upper()}\n"
                    f"Disputed Amount: ${dispute.dispute_amount:.2f}\n\n"
                    f"The merchant presents strong primary evidence refuting the cardholder claim. "
                    f"Key evidence: {evidence_summary}."
                )
                reasoning_summary = (
                    f"Single strong evidence document available for {reason_code} (low_coverage=True). "
                    f"Contesting based on available evidence; confidence capped due to sparse retrieval."
                )
            else:
                decision = "no_contest"
                confidence = 0.35
                rebuttal_draft = ""
                reasoning_summary = (
                    f"One weak evidence document retrieved for {reason_code} (low_coverage=True). "
                    f"Evidence insufficient to form a credible rebuttal; recommending no_contest."
                )
        else:
            decision = "no_contest"
            confidence = 0.25
            rebuttal_draft = ""
            reasoning_summary = (
                f"No evidence retrieved for {reason_code} claim (low_coverage={low_coverage}, docs={len(evidence)}). "
                f"Recommending no_contest to prevent representment fees without supporting documentation."
            )

        return DecisionAgentResponse(
            decision=decision,
            confidence=confidence,
            rebuttal_draft=rebuttal_draft,
            reasoning_summary=reasoning_summary,
            used_fallback=True,
        )
