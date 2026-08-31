"""
Synthetic Dispute + Evidence Dataset Generator (Grounded).

Generates realistic dispute cases and evidence documents grounded in:
1. IEEE-CIS Fraud Detection Kaggle dataset (published by Vesta Corporation — real transactions, ProductCD/card4/card6 proxy categories).
2. CFPB Consumer Complaint Database (few-shot narrative styles).
3. Faker pre-generated entities (merchant names, order numbers, ticket IDs, tracking numbers, dates, IPs).
4. Prior assignment of label_winnable and quality levels (strong/weak/missing).
5. Post-generation pairwise cosine similarity deduplication (>0.93 threshold with mandatory final pass verification of 0 remaining pairs).
"""

import json
import os
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from pydantic import BaseModel
from faker import Faker
import kagglehub

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

# Semantic salt pools — appended to every evidence doc to break cosine similarity clustering.
# BGE encodes semantic meaning, not surface form. Same-type docs need genuinely different
# appended content to diverge below the 0.93 threshold.

_RC_NOTES: Dict[str, List[str]] = {
    "goods_not_received": [
        "Under Visa Reason Code 13.1, merchants must provide carrier confirmation with delivery timestamp to rebut non-receipt claims.",
        "Mastercard chargeback reason 4855 (Goods Not Received) requires proof that the merchandise was dispatched and delivered to the cardholder's billing address.",
        "Per Regulation E and network dispute rules, the burden of proof for showing delivery lies with the acquiring merchant when the cardholder alleges non-receipt.",
        "Industry standard for goods-not-received disputes requires a signed proof of delivery or a GPS-verified doorstep event within the tracking record.",
        "Federal Trade Commission guidelines state that merchants must respond to non-delivery claims with carrier-certified proof within the issuer-defined dispute window.",
        "Amex reason code C08 (Goods/Services Not Received) requires evidence of delivery to the cardholder's address, not just a shipping label generation event.",
        "The Visa Core Rules Section 11.2 requires merchants to furnish a courier receipt confirming transfer of goods to the stated delivery address for non-receipt disputes.",
        "Under UCC Article 2-503, title and risk of loss pass to the buyer upon physical delivery; a non-receipt claim is rebutted by a delivery confirmation matching the billing record.",
        "Chargeback representment for this reason code typically hinges on matching the shipping manifest GPS coordinates to the cardholder's registered delivery address.",
        "Discover's dispute resolution framework for non-receipt requires electronic proof of delivery (ePOD) with a recipient acknowledgement timestamp.",
    ],
    "goods_defective": [
        "Visa Reason Code 13.3 (Not as Described or Defective Merchandise) requires the merchant to prove the item matched the product description shown at checkout.",
        "Mastercard chargeback reason 4853 covers both non-delivery and defective goods; merchants must provide itemized product specifications and quality control records.",
        "For defective merchandise disputes, the merchant's return merchandise authorization (RMA) process and any post-sale communication logs are primary evidence.",
        "Industry practice requires merchants to provide the original product listing, customer quality acknowledgment records, and any exchange or replacement correspondence.",
        "Per PCI DSS requirements, merchants retaining product images and packaging records can present these as evidence that goods were shipped in stated condition.",
        "Consumer protection statutes in most U.S. states allow cardholders to initiate chargebacks for defective goods if the merchant fails to honor their stated return policy.",
        "Amex chargeback code C31 (Goods/Services Defective or Not as Described) mandates the merchant submit the original order specification alongside any QA inspection data.",
        "Effective representment for defective goods typically includes: original product spec sheet, photo evidence of packaging, and customer communications prior to dispute.",
        "Under Mastercard Chargeback Guide Section 6.2, the merchant must demonstrate the item conformed to the description displayed at the point of sale.",
        "Discover Network chargeback rule DB-2853 requires evidence that the cardholder had the opportunity to inspect and return the goods under the merchant's stated policy.",
    ],
    "duplicate_charge": [
        "Visa Reason Code 12.6 covers duplicate processing; merchants must show that each transaction corresponds to a distinct authorization and settlement event.",
        "Mastercard reason code 4834 (Duplicate Processing) requires evidence that two separate orders were placed, not a single charge billed twice.",
        "For duplicate charge disputes, the merchant's payment gateway logs showing two distinct authorization codes and order reference numbers are the primary evidence.",
        "Under card network rules, duplicate processing chargebacks are only upheld if the merchant cannot produce two separate invoices with different order IDs and authorization tokens.",
        "Amex reason code P05 (Incorrect Charge Amount/Duplicate) is rebutted by showing two distinct customer-initiated purchase events in the merchant's order management system.",
        "Industry dispute resolution practice requires the merchant to submit authorization logs with timestamps proving each charge was triggered by a separate customer action.",
        "Per the Payment Card Industry dispute resolution framework, a duplicate charge claim is invalid if the merchant provides two unique authorization approval codes.",
        "Discover chargeback rule DB-4834 requires the merchant to show two signed receipts or equivalent electronic records tied to separate session IDs.",
        "Card network operating rules distinguish between duplicate billing (same order charged twice) and legitimate separate transactions for the same amount on the same day.",
        "Payment processor audit logs showing two separate basket confirmations and checkout session tokens are the gold standard for rebutting duplicate charge claims.",
    ],
    "credit_not_processed": [
        "Visa Reason Code 13.7 applies when a merchant promises a refund but fails to process the credit within the network's stipulated timeline.",
        "Under Mastercard chargeback reason 4860, if a merchant issued a credit memo but the transaction never posted to the cardholder's account, the issuer may initiate a chargeback.",
        "Credit-not-processed disputes require the merchant to provide: refund authorization ID, credit transaction timestamp, and bank posting confirmation.",
        "Amex reason code C05 (Credit Not Processed) is resolved when the merchant submits the credit transaction receipt with the acquiring bank's confirmation reference.",
        "For credit-not-processed chargebacks, the merchant's refund ledger entry and the corresponding payment processor confirmation code constitute sufficient representment.",
        "Discover rule DB-4860 requires merchants to supply proof of credit issuance including the credit authorization number and the date the credit was submitted to the network.",
        "Per card network operating regulations, merchants have 5 business days after a chargeback is filed to provide evidence that the credit was already processed.",
        "The acquiring bank's credit transaction detail report, showing the refund instruction date and cardholder account reference, is the primary evidence for this dispute type.",
        "Credit-not-processed chargebacks are typically invalid if the merchant can show the refund was submitted within the return window and a credit memo was issued.",
        "Industry best practice requires merchants to retain credit authorization records for 18 months to facilitate timely chargeback representment.",
    ],
    "subscription_canceled_but_charged": [
        "Visa Reason Code 13.6 covers services canceled properly by the cardholder but billed afterward; merchants must show the cancellation was not processed in their system.",
        "Mastercard chargeback reason 4841 requires the merchant to prove no valid cancellation request was received, or that charges accrued before the cancellation effective date.",
        "Under subscription dispute rules, the merchant's CRM cancellation audit log showing no inbound cancellation request prior to the billing date is the primary rebuttal.",
        "Amex code C28 (Canceled Recurring Billing) is rebutted by providing the terms of service governing cancellation windows and proof the customer did not cancel in time.",
        "Subscription merchants must retain click-stream logs showing the customer's cancellation action (or lack thereof) to rebut post-cancellation billing chargebacks.",
        "Recurring billing disputes often hinge on the cancellation timestamp versus the billing cycle date; a gap of more than 24 hours in the merchant's favor typically constitutes valid evidence.",
        "Per Visa's Subscription Merchant Guidelines, merchants must provide a cancellation confirmation email timestamp or CRM record showing no prior cancellation request.",
        "Discover chargeback rule DB-4841 requires evidence that the subscription was active and un-canceled at the time of the disputed charge, typically via account activity logs.",
        "Merchants defending subscription billing chargebacks should submit: cancellation policy shown at sign-up, recurring billing consent language, and absence of cancellation event in CRM.",
        "Industry dispute practice requires subscription merchants to send a pre-charge reminder notification; failure to do so may invalidate the merchant's representment.",
    ],
    "unauthorized_transaction": [
        "Visa Reason Code 10.4 (Other Fraud — Card Absent Environment) is the most common chargeback; 3DS authentication significantly shifts liability to the issuer.",
        "Under Mastercard chargeback reason 4863, merchants who completed 3DS authentication are relieved of fraud liability under the card network's liability shift rules.",
        "Amex fraud chargeback F24 requires the cardholder to certify they did not authorize the transaction; the merchant can rebut with strong authentication evidence.",
        "Per EMV 3-D Secure v2.2 liability rules, a successful ARes=Y response from the issuer means fraud liability shifts away from the merchant regardless of the dispute claim.",
        "Discover fraud reason code UA02 applies when no CVV2 match is present; merchants with 3DS evidence can still win representment under network fraud liability shift rules.",
        "PCI DSS-compliant merchants logging full 3DS authentication payloads (including CAVV, XID, and ECI) have the strongest evidence basis for unauthorized transaction disputes.",
        "Per the Electronic Fund Transfer Act (Regulation E), card-not-present fraud liability rules differ from card-present; the issuer bears liability after successful 3DS challenge.",
        "Card network operating regulations establish that IP-to-billing-address geo-match combined with 2FA verification constitutes strong evidence of authorized card use.",
        "NIST SP 800-63B Level-of-Assurance 2 authentication (multi-factor with device fingerprint) substantially reduces the merchant's exposure to unauthorized transaction chargebacks.",
        "Behavioral analytics confirming the transaction originated from the cardholder's registered device and IP subnet are admissible evidence in unauthorized transaction representment.",
    ],
}

_DOCTYPE_ADDENDA: Dict[str, List[str]] = {
    "auth_log": [
        "Gateway provider: certified PCI DSS Level 1 compliant. Authentication log retained for 24 months per network operating regulations.",
        "This log is generated automatically by the payment orchestration platform and carries a tamper-evident digital signature for evidentiary integrity.",
        "Authentication telemetry retained under ISO/IEC 27001 controls. Log available in SIEM audit archive for 36 months.",
        "Session data encrypted at rest (AES-256) and in transit (TLS 1.3). Log hash: available on request from the merchant's acquiring bank.",
        "All IP geolocation lookups performed against MaxMind GeoIP2 Enterprise database. Device fingerprints stored under GDPR Article 17 retention schedule.",
        "Fraud scoring engine: Kount Decisioning Platform v8.6. Risk score at transaction time: within acceptable threshold.",
        "3DS authentication data archived per Visa and Mastercard operating regulations Section 5.9.1. Logs available for regulator audit.",
        "Authentication log exported from merchant's SIEM (Splunk Enterprise Security). Timestamp synchronized to NTP stratum-1 source.",
        "Device fingerprint library: ThreatMetrix. All fingerprint records retained for 5 years per acquirer agreement.",
        "Cardholder verification method: SMS OTP via Twilio Verify API. Delivery receipt and OTP expiry timestamp logged.",
        "Authentication events correlated with prior successful logins from same device. No account takeover indicators detected.",
        "IP reputation checked against industry threat intelligence feeds (Spamhaus, APWG eCrime). Source IP classification: residential/verified.",
    ],
    "delivery_confirmation": [
        "Carrier SLA for this service tier: 2-5 business days standard shipping. All scan events retained in carrier's National Tracking System for 5 years.",
        "Electronic proof of delivery image archived in carrier's digital vault. Available for retrieval by authorized dispute resolution parties.",
        "Carrier's GPS telemetry system logged vehicle coordinates at time of delivery. Telemetry data available from carrier upon legal request.",
        "Package weight and dimensions recorded at intake scan: match merchant's manifest. No evidence of tampering or substitution detected.",
        "Carrier insurance claim status for this shipment: No claim filed. Delivery confirmed as completed without exception.",
        "Signature capture device: Carrier handheld terminal. Biometric signature record available from carrier's digital archive.",
        "Last-mile delivery route data retained per carrier's 36-month data retention policy. Route log available for third-party audit.",
        "Customs clearance (if applicable): Cleared. No hold, seizure, or return-to-sender notice filed at any point in the delivery chain.",
        "Package barcode standard: GS1-128. Scan sequence confirms continuous chain of custody from merchant warehouse to delivery address.",
        "Carrier driver ID and vehicle registration on file for this route. Available for law enforcement or dispute resolution review.",
        "Delivery notification SMS sent to cardholder's registered mobile number at time of handover. Read receipt recorded by carrier notification system.",
        "Return merchandise authorization (RMA) status: No return initiated or approved for this shipment. Package not returned to sender.",
    ],
    "shipping_tracking": [
        "Tracking data pulled from carrier API in real-time. All timestamps are UTC and have been converted to the local timezone of the delivery address.",
        "Carrier electronic data interchange (EDI 856 ASN) confirmed shipment departure from merchant's fulfillment center.",
        "Tracking events timestamped to carrier's atomic clock-synchronized scanning infrastructure. Accuracy: ±500ms.",
        "Hub scanning infrastructure: RFID-based automated sortation system. Scan events triggered at each sortation milestone.",
        "Carrier's chain of custody log shows no address correction, hold, or redirect requests associated with this consignment.",
        "Multi-leg shipment: Origin scan, departure, hub transfer, and destination arrival all logged. No gap in scan chain.",
        "Tracking API integration: FedEx Web Services v2024.1 / UPS Developer Kit v26 / USPS SOAP API. Data certified as retrieved directly from carrier.",
        "Carrier pallet manifest cross-referenced with tracking number: consignment confirmed on outbound vehicle manifest.",
        "Intermodal transfer (air to ground): Air waybill number linked to ground tracking reference in carrier's consolidated tracking system.",
        "Carrier exception codes: None. No attempted-delivery notices, refused deliveries, or unable-to-locate events recorded.",
        "Shipment insured for declared value: ${declared_value}. Carrier's insurance record tied to this tracking number.",
        "Post-delivery customer satisfaction survey sent by carrier. Survey link click-through recorded, indicating recipient accessed delivery notification.",
    ],
    "communication_log": [
        "All CRM records stored in Salesforce Service Cloud with tamper-evident audit trail. Log exported by authorized compliance officer.",
        "Customer contact channel: Email (SMTP with DKIM/SPF verification). Email server logs confirm delivery to cardholder's inbox.",
        "Communication archived per GDPR Article 5(1)(e) data minimization policy. Records available for regulatory audit for 6 years.",
        "Support ticket system: Zendesk Enterprise. Ticket thread exported with full metadata including agent actions and timestamps.",
        "Phone call referenced in this record was recorded with cardholder consent under applicable state laws. Recording available via secure retrieval.",
        "Chat transcript archived in merchant's CRM with ISO 8601 timestamps. Chat session authenticated via customer login session token.",
        "Customer identity verified via two-factor verification at start of support interaction. Support session ID logged.",
        "SLA compliance: First response sent within merchant's published 24-hour SLA. Escalation rules triggered at 48 hours if unresolved.",
        "Email thread cryptographically signed using merchant's domain DKIM key (selector: support). Signature verifiable against public DNS record.",
        "CRM record flagged as dispute-relevant and preserved from standard 90-day auto-deletion policy per legal hold protocol.",
        "Outbound communication channel: Merchant's Twilio-powered notification system. Delivery receipt stored with message SID.",
        "Support agent authentication: Role-based access control (RBAC) log confirms only authorized Tier-2 agents accessed this ticket.",
    ],
    "refund_policy": [
        "Policy text displayed to customer at checkout via modal dialog. Click-through acceptance event logged with session ID and timestamp.",
        "Merchant's Terms and Conditions version: v4.2, effective date 2024-01-01. Archived copy of accepted terms preserved in customer record.",
        "Policy compliance certified under FTC Mail or Telephone Order Merchandise Rule (16 CFR Part 435). Refund window meets regulatory minimum.",
        "Consumer rights notice included in order confirmation email per applicable state consumer protection statutes. Email delivery confirmed.",
        "Dispute resolution clause in merchant's Terms of Service requires customers to contact support prior to initiating a chargeback.",
        "Subscription billing disclosure: made in conformance with Restore Online Shoppers' Confidence Act (ROSCA) and applicable state auto-renewal laws.",
        "Policy version accepted by cardholder is retained for 36 months per merchant's data governance policy. Available for compliance audit.",
        "Terms acceptance log: IP address, browser fingerprint, timestamp, and checkbox interaction event stored in merchant's compliance database.",
        "Policy text passes FTC's 'clear and conspicuous' disclosure standard as reviewed by merchant's legal counsel in Q3 2024.",
        "Return window calculation: calendar days from delivery confirmation scan date, per carrier-provided timestamp. Not business days.",
        "Merchant's refund policy is publicly archived on the Internet Archive Wayback Machine and may be independently verified.",
        "Arbitration clause in Terms of Service requires ADR prior to litigation; chargeback bypass of this clause may be raised in representment.",
    ],
    "order_details": [
        "Order data exported from merchant's OMS (NetSuite / Shopify / Magento). Record certified by merchant's Chief Operating Officer.",
        "Invoice generated by merchant's ERP system (SAP S/4HANA / Oracle Financials). Certified copy available for financial audit.",
        "Customer identity verification completed at account registration via email + phone. KYC level: Standard (Tier 1) per AML policy.",
        "Order risk score at placement: within approved threshold. Fraud screening by Signifyd Commerce Protection Platform.",
        "Billing address provided at checkout was verified against Visa Address Verification Service (AVS) response: Full Match (Y).",
        "CVV2 verification result at time of authorization: Match. Issuer authorization code retained in merchant's payment records.",
        "Order placed via authenticated customer session. Session token validated against merchant's OAuth 2.0 identity provider.",
        "Itemized order manifest (SKU, quantity, unit price) archived per merchant's 7-year financial record retention policy.",
        "Payment card BIN data at time of transaction: issuer country matches customer's stated billing country. No geographic anomaly flagged.",
        "Tax invoice issued in compliance with applicable jurisdiction's e-invoicing regulations. Digital invoice stored in merchant's tax archive.",
        "Order fulfillment SLA: standard 2-business-day processing. Fulfillment center timestamp confirms order picked and packed within SLA.",
        "Customer's purchase was eligible for merchant's loyalty reward program. Points credited to account confirm successful order completion.",
    ],
}

# Proxy category mapping from IEEE-CIS Fraud Detection anonymized ProductCD fields to domain categories
PRODUCT_CD_MAPPING = {
    "W": "ecommerce",     # Web transaction -> ecommerce
    "C": "subscription",  # Cellular / Recurring / Digital -> subscription
    "H": "services",      # Hosted service -> services
    "R": "ecommerce",     # Retail -> ecommerce
    "S": "services"       # Specialty service -> services
}


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


class GroundedDataPool:
    """
    Loads real numeric transaction rows from IEEE-CIS Fraud Detection dataset
    and real complaint narratives from CFPB Consumer Complaint Database.
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.fake = Faker()
        Faker.seed(seed)
        self.kaggle_rows = self._load_kaggle_fraud_data(num_rows=400)
        self.cfpb_narratives = self._load_cfpb_narratives(num_samples=40)

    def _load_kaggle_fraud_data(self, num_rows: int = 400) -> List[Dict[str, Any]]:
        print("[Grounding] Downloading/loading IEEE-CIS Fraud Detection dataset via kagglehub...")
        try:
            path = kagglehub.dataset_download("lnasiri007/ieeecis-fraud-detection")
            train_csv = Path(path) / "train_transaction.csv"
            test_csv = Path(path) / "test_transaction.csv"
            csv_file = train_csv if train_csv.exists() else test_csv

            df = pd.read_csv(csv_file, nrows=num_rows * 2)
            df = df.sample(frac=1.0, random_state=42).reset_index(drop=True).head(num_rows)

            rows = []
            for _, r in df.iterrows():
                amt = round(float(r.get("TransactionAmt", 149.99)), 2)
                p_cd = str(r.get("ProductCD", "W")).strip().upper()
                c4 = str(r.get("card4", "visa")).strip().lower()
                c6 = str(r.get("card6", "credit")).strip().lower()
                mapped_cat = PRODUCT_CD_MAPPING.get(p_cd, "ecommerce")

                ts = r.get("TransactionDT")
                if pd.notna(ts):
                    base_date = pd.Timestamp("2026-07-01")
                    date_str = (base_date + pd.Timedelta(seconds=float(ts))).strftime("%Y-%m-%d")
                else:
                    date_str = "2026-07-15"

                addr1 = str(r.get("addr1", "94103"))
                zip_code = addr1.split(".")[0] if "." in addr1 else addr1
                if not zip_code or zip_code == "nan":
                    zip_code = "94103"

                rows.append({
                    "amount": amt,
                    "date": date_str,
                    "category": mapped_cat,
                    "proxy_product_cd": p_cd,
                    "proxy_card4": c4,
                    "proxy_card6": c6,
                    "kaggle_merchant": f"Vesta-{p_cd}-{c4}",
                    "lat": 37.7749,
                    "long": -122.4194,
                    "zip": zip_code
                })
            print(f"[Grounding] Loaded {len(rows)} real transaction rows from IEEE-CIS Fraud Detection dataset.")
            return rows
        except Exception as e:
            print(f"[WARN] IEEE-CIS dataset load failed ({e}). Using grounded synthetic numeric pool.")
            rows = []
            for _ in range(num_rows):
                amt = round(self.rng.uniform(19.99, 3499.00), 2)
                day = self.rng.randint(1, 28)
                month = self.rng.randint(5, 8)
                date_str = f"2026-{month:02d}-{day:02d}"
                cat = self.rng.choice(MERCHANT_CATEGORIES)
                rows.append({
                    "amount": amt,
                    "date": date_str,
                    "category": cat,
                    "proxy_product_cd": "W",
                    "proxy_card4": "visa",
                    "proxy_card6": "credit",
                    "kaggle_merchant": "Generic Merchant",
                    "lat": 37.7749,
                    "long": -122.4194,
                    "zip": "94103"
                })
            return rows

    def _load_cfpb_narratives(self, num_samples: int = 40) -> List[str]:
        print("[Grounding] Downloading/loading CFPB Consumer Complaint Database via kagglehub...")
        try:
            path = kagglehub.dataset_download("selener/consumer-complaint-database")
            rows_csv = Path(path) / "rows.csv"
            df = pd.read_csv(rows_csv, low_memory=False)

            card_mask = df["Product"].str.contains(
                "Credit card|Paypal|Prepaid card|Bank account or service|Checking or savings account",
                case=False, na=False
            )
            df_card = df[card_mask]
            narratives = df_card["Consumer complaint narrative"].dropna()
            narratives = narratives[narratives.str.len() > 60]

            sampled = narratives.sample(n=num_samples, random_state=42).tolist()
            cleaned = []
            for n in sampled:
                text = n.replace("XXXX", "[Redacted]").strip()
                if len(text) > 400:
                    text = text[:400] + "..."
                cleaned.append(text)

            print(f"[Grounding] Loaded {len(cleaned)} real CFPB complaint narrative style examples.")
            return cleaned
        except Exception as e:
            print(f"[WARN] CFPB dataset load failed ({e}). Using fallback narrative style examples.")
            return [
                "I ordered an item online 3 weeks ago, card was charged immediately, but no shipping confirmation or item delivered.",
                "Card charged twice for single transaction on merchant website. Reached out to merchant support but received no response.",
                "Canceled subscription per merchant terms 5 days before renewal, but was charged full subscription amount on renewal date.",
                "Unrecognized transaction appeared on my billing statement. Card was in my possession and I did not authorize this charge."
            ]

    def get_grounded_row(self, index: int) -> Dict[str, Any]:
        return self.kaggle_rows[index % len(self.kaggle_rows)]


class SyntheticDataGenerator:
    """
    Generates synthetic dispute cases and corresponding evidence documents,
    grounded in real transaction parameters and narrative styles.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        DATASETS_DIR.mkdir(parents=True, exist_ok=True)
        self.fake = Faker()
        Faker.seed(seed)
        self.grounded_pool = GroundedDataPool(seed=seed)

    def _generate_faker_context(self, merchant_category: str) -> Dict[str, Any]:
        """Pre-generates realistic entities via Faker so LLM/generator doesn't invent them."""
        company_suffix = "Inc" if merchant_category == "ecommerce" else "Cloud" if merchant_category == "subscription" else "Services"
        os_choice = random.choice(['macOS', 'Windows 11', 'Android 14', 'iOS 17'])
        carrier_choice = random.choice(["BlueDart", "Delhivery", "FedEx", "UPS", "DHL"])

        return {
            "merchant_name": f"{self.fake.company()} {company_suffix}",
            "order_id": f"ORD-{self.fake.numerify('######')}",
            "tracking_number": f"TRK-{self.fake.bothify('??######').upper()}",
            "ticket_id": f"TCK-{self.fake.numerify('####')}",
            "customer_name": self.fake.name(),
            "customer_email": self.fake.free_email(),
            "customer_phone": self.fake.phone_number(),
            "customer_address": f"{self.fake.street_address()}, {self.fake.city()}, {self.fake.state_abbr()} {self.fake.zipcode()}",
            "customer_city": self.fake.city(),
            "customer_state": self.fake.state_abbr(),
            "auth_ip": self.fake.ipv4_public(),
            "vpn_ip": self.fake.ipv4_public(),
            "device_fingerprint": f"Chrome {self.fake.random_int(115, 128)}.0 on {os_choice}",
            "carrier": carrier_choice,
            "account_age_months": self.fake.random_int(3, 36),
            "past_orders": self.fake.random_int(2, 25),
        }

    def _build_doc_content(
        self,
        doc_type: str,
        reason_code: str,
        winnable: bool,
        quality: str,
        amount: float,
        date_str: str,
        faker_ctx: Dict[str, Any]
    ) -> str:
        """
        Builds doc_type content conditioned on quality, real amount/date, and pre-generated Faker context.
        Uses modular 3-clause composition (Opening + Body + Closing) to yield 200+ unique sentence structures
        per doc_type quality branch, avoiding static wrappers that cause high embedding similarity.
        CRITICAL: Never embeds literal transaction_id in content text.
        """
        m_name = faker_ctx["merchant_name"]
        order_id = faker_ctx["order_id"]
        tracking = faker_ctx["tracking_number"]
        ticket = faker_ctx["ticket_id"]
        c_name = faker_ctx["customer_name"]
        c_addr = faker_ctx["customer_address"]
        carrier = faker_ctx["carrier"]

        time_str = f"{random.randint(8, 20):02d}:{random.randint(10, 59):02d}"
        session_id = f"SESS-{random.randint(10000, 99999)}"
        agent_name = random.choice(["Alex M.", "Sarah T.", "David K.", "Elena R.", "Marcus B.", "Priya N."])
        refund_window = random.choice([7, 10, 14, 15, 30])
        cancel_hours = random.choice([24, 48, 72])
        rma_code = f"RMA-{random.randint(1000, 9999)}"
        invoice_num = f"INV-{random.randint(10000, 99999)}"
        payment_method = random.choice(["Visa debit", "Mastercard credit", "American Express", "Discover card", "UPI", "Net Banking"])

        if doc_type == "auth_log":
            if winnable and quality == "strong":
                openings = [
                    f"Authentication Log for Order {order_id} ({m_name}).",
                    f"Secure Auth Audit Trail ({m_name}) — Reference {order_id}.",
                    f"Transaction Authentication Record [{order_id}] logged by {m_name} gateway.",
                    f"Security Event Log: POS gateway confirmed purchase ref {order_id} at {m_name}.",
                    f"3DS Compliance Report for order ref {order_id} at {m_name}.",
                    f"Fraud Prevention Audit [{order_id}] from {m_name} platform."
                ]
                bodies = [
                    f"IP Address {faker_ctx['auth_ip']} matches registered network in {faker_ctx['customer_city']}, {faker_ctx['customer_state']}.",
                    f"Customer {c_name} completed 3DS v2.2 Strong Customer Authentication on {date_str} at {time_str}.",
                    f"2FA OTP successfully verified via SMS ({faker_ctx['customer_phone']}) on {date_str} at {time_str}.",
                    f"Client IP {faker_ctx['auth_ip']} passed biometric and SMS challenge for session {session_id}.",
                    f"Registered device fingerprint [{faker_ctx['device_fingerprint']}] cleared fraud screening.",
                    f"Cardholder identity confirmed via OTP dispatched to {faker_ctx['customer_phone']}."
                ]
                closings = [
                    "3DS v2.2 Challenge completed successfully with frictionless authentication.",
                    "Issuer bank returned ARes=Y and granted full fraud liability shift to merchant.",
                    "Device fingerprint and IP whitelist checks both returned zero risk flags.",
                    "Session metadata cryptographically verified by acquiring payment gateway.",
                    "No anomalous signals or account takeover indicators detected during authorization.",
                    "Cardholder authentication token validated against issuing bank security server."
                ]
            elif quality == "weak":
                openings = [
                    f"Authentication Log for Order {order_id} ({m_name}).",
                    f"Auth Gateway Record ({order_id}) from {m_name}.",
                    f"Session Log [{order_id}] at {m_name} payment portal.",
                    f"Risk Event Report: Order {order_id} processed by {m_name}.",
                    f"Gateway Access Log for transaction ref {order_id} ({m_name}).",
                    f"Auth Incident Record [{session_id}] for order {order_id}."
                ]
                bodies = [
                    f"Single-factor web checkout initiated from IP {faker_ctx['vpn_ip']} (Proxy/VPN endpoint).",
                    f"Guest checkout authorized on {date_str} at {time_str} from IP {faker_ctx['vpn_ip']}.",
                    f"Web checkout processed without multi-factor verification from untrusted IP {faker_ctx['vpn_ip']}.",
                    f"Source IP {faker_ctx['vpn_ip']} flagged as datacenter proxy; 3DS challenge bypassed.",
                    f"Basic card check completed on {date_str} from IP {faker_ctx['vpn_ip']} without 2FA.",
                    f"Transaction authorized at {time_str} on {date_str} using basic single-factor PAN entry."
                ]
                closings = [
                    "3DS Status: Attempted / Not Enrolled; no device fingerprint captured.",
                    "3DS authentication bypassed via card issuer fallback; liability shift NOT claimed.",
                    "Device fingerprint unverified and multi-factor challenge absent from session payload.",
                    "Merchant did not invoke 3DS step-up challenge; liability retained by acquirer.",
                    "Authentication grade: 1FA only; no step-up challenge triggered by issuer.",
                    "Risk score elevated due to proxy IP usage and absent 2FA verification."
                ]
            else:  # missing
                openings = [
                    f"Authentication Log for Order {order_id}.",
                    f"System Log ({order_id}) at {m_name}.",
                    f"Auth Event [{order_id}] audit record.",
                    f"Gateway Archive for order {order_id}.",
                    f"Incomplete Security Log ({order_id}).",
                    f"Session Archive [{session_id}] for {m_name}."
                ]
                bodies = [
                    f"Standard card-not-present authorization recorded on {date_str}.",
                    f"Basic payment authorization logged on {date_str} at {time_str}.",
                    f"Merchant API call logged without extended 3DS headers.",
                    f"Bare authorization token logged for order placed on {date_str}.",
                    f"System log contains payment confirmation token for session {session_id}.",
                    f"Basic card authorization entry stored without risk telemetry payload."
                ]
                closings = [
                    "3DS metadata and device telemetry unavailable in system log.",
                    "Detailed authentication headers missing from server archives.",
                    "Advanced 2FA/3DS authentication payload absent from database record.",
                    "Log retention policy may have purged detailed 3DS session headers.",
                    "No IP header, device fingerprint, or OTP verification payload recoverable.",
                    "Cardholder authentication status remains unconfirmed in merchant logs."
                ]
            return f"{random.choice(openings)} {random.choice(bodies)} {random.choice(closings)}"

        elif doc_type in ["delivery_confirmation", "shipping_tracking"]:
            if winnable and quality == "strong":
                openings = [
                    f"Shipping & Delivery Proof for Order {order_id} ({m_name}).",
                    f"Proof of Delivery Certificate — Order {order_id}.",
                    f"Carrier Delivery Log ({carrier}) for Order {order_id}.",
                    f"Logistics Confirmation — Consignment #{tracking} (Order {order_id}).",
                    f"Delivery Event Record for order {order_id} via {carrier}.",
                    f"End-to-End Fulfillment Record — Order {order_id}."
                ]
                bodies = [
                    f"Carrier {carrier} Tracking #{tracking} status updated to DELIVERED on {date_str} at {time_str}.",
                    f"Package shipped via {carrier} (#{tracking}) and delivered to {c_addr} on {date_str}.",
                    f"Courier completed doorstep handover to recipient {c_name} at {c_addr} on {date_str} {time_str}.",
                    f"Consignment cleared final-mile delivery to {c_addr} with GPS coordinates matching address.",
                    f"Parcel dispatched under AWB #{tracking} and signed for by {c_name} at {time_str}.",
                    f"Delivery scan verified by {carrier} courier at destination address {c_addr}."
                ]
                closings = [
                    f"Recipient Proof of Delivery: Digital OTP confirmed at doorstep by {c_name}.",
                    f"Recipient signature captured electronically: {c_name}.",
                    f"Doorstep scan and geo-tag confirm successful package delivery.",
                    f"Electronic proof-of-delivery (ePOD) archived in carrier database.",
                    f"Package delivered directly to cardholder with zero delivery exceptions logged.",
                    f"Fulfillment complete; carrier confirmed delivery receipt on file."
                ]
            elif quality == "weak":
                openings = [
                    f"Shipping & Delivery Proof for Order {order_id} ({m_name}).",
                    f"Carrier Tracking Record ({order_id}) — {carrier} #{tracking}.",
                    f"Shipment Status [{order_id}] via {carrier}.",
                    f"Logistics Alert — {carrier} AWB #{tracking} for Order {order_id}.",
                    f"Tracking Update for order {order_id} dispatched on {date_str}.",
                    f"Partial Fulfillment Record — Order {order_id} ({carrier})."
                ]
                bodies = [
                    f"Status: IN_TRANSIT / Delayed at regional sorting hub in {faker_ctx['customer_city']} as of {date_str}.",
                    f"Tracking indicates package in transit at regional facility in {faker_ctx['customer_state']}.",
                    f"Courier updated scan on {date_str} showing parcel delayed in regional transit facility.",
                    f"Package scanned at hub in {faker_ctx['customer_city']}; doorstep delivery pending.",
                    f"Last checkpoint logged on {date_str} at courier hub in {faker_ctx['customer_state']}.",
                    f"Tracking portal shows item last moved on {date_str} near destination zone."
                ]
                closings = [
                    "Delivery status unconfirmed; final delivery scan pending.",
                    "Doorstep delivery timestamp not recorded by carrier.",
                    "Destination delivery scan absent; ePOD unavailable.",
                    "Recipient handover confirmation pending at sorting center.",
                    "Package delayed in transit; delivery receipt not yet captured.",
                    "Carrier tracking shows active transit but lacks final delivery signature."
                ]
            else:
                openings = [
                    f"Shipping & Delivery Proof for Order {order_id}.",
                    f"Manifest Log ({order_id}) at {m_name}.",
                    f"Delivery Record [{order_id}] — Courier tracking assigned.",
                    f"Logistics Archive — #{tracking} for Order {order_id}.",
                    f"Shipment Data Gap for order {order_id}.",
                    f"Courier Record [{order_id}] raised on {date_str}."
                ]
                bodies = [
                    f"Tracking number #{tracking} generated on {date_str} for carrier {carrier}.",
                    f"Shipping label created for #{tracking} on {date_str}; physical pickup unconfirmed.",
                    f"Electronic shipping info received by {carrier} for AWB #{tracking}.",
                    f"Shipping label generated but carrier has no inbound scan event on record.",
                    f"No carrier pickup event, warehouse departure, or transit scan logged for #{tracking}.",
                    f"Consignment note #{tracking} issued but physical tender to courier absent."
                ]
                closings = [
                    "No carrier scan data available in logistics portal.",
                    "Physical package pick-up by carrier remains unconfirmed.",
                    "Movement scans logged: 0; package status pending.",
                    "Parcel may not have been tendered to courier facility.",
                    "Delivery status: UNSHIPPED / Missing fulfillment proof.",
                    "Tracking code void; courier portal returned zero movement scans."
                ]
            return f"{random.choice(openings)} {random.choice(bodies)} {random.choice(closings)}"

        elif doc_type == "communication_log":
            if winnable and quality == "strong":
                openings = [
                    f"Customer Communication Log ({m_name}): Support Ticket #{ticket}.",
                    f"CRM Interaction Record — Ticket #{ticket} ({m_name}).",
                    f"Support Ticket #{ticket} Log regarding Order {order_id}.",
                    f"Help Desk Resolution Record — {m_name} (Ticket #{ticket}).",
                    f"Email Thread Archive [{ticket}] for Order {order_id}.",
                    f"Customer Service Case #{ticket} handled by {m_name}."
                ]
                bodies = [
                    f"Customer {c_name} emailed support on {date_str} regarding Order {order_id} setup.",
                    f"Support agent {agent_name} provided full technical assistance and product user guide on {date_str}.",
                    f"Correspondence between {c_name} and agent {agent_name} resolved the inquiry at {time_str}.",
                    f"Agent {agent_name} dispatched troubleshooting steps and verified item functionality.",
                    f"Customer ({c_name}) inquired about order {order_id} and received immediate resolution from {agent_name}.",
                    f"Merchant support specialist {agent_name} addressed all customer queries on {date_str}."
                ]
                closings = [
                    f"Customer ({c_name}) replied: 'Thank you, received and working fine now!'",
                    "Customer confirmed item functionality and expressed satisfaction with resolution.",
                    "Issue resolved amicably; customer acknowledged receipt and canceled refund inquiry.",
                    "Customer confirmed resolution in subsequent reply and signed off on ticket closure.",
                    "Final customer email confirmed satisfaction; support ticket closed as resolved.",
                    "Customer acknowledged successful delivery and confirmed dispute cancellation."
                ]
            elif quality == "weak":
                openings = [
                    f"Customer Communication Log ({m_name}): Support Ticket #{ticket}.",
                    f"Support Desk Record (#{ticket}) for Order {order_id}.",
                    f"CRM Ticket #{ticket} ({m_name}) submitted on {date_str}.",
                    f"Helpdesk Log — {m_name} Ticket #{ticket}.",
                    f"Open Support Case [{ticket}] regarding Order {order_id}.",
                    f"CRM Unresolved Entry #{ticket} ({m_name})."
                ]
                bodies = [
                    f"Customer {c_name} inquired about refund status for Order {order_id} on {date_str}.",
                    f"Inbound inquiry received from {c_name} on {date_str}; automated system response dispatched.",
                    f"Customer submitted complaint on {date_str}; ticket routed to billing queue.",
                    f"Ticket opened by {c_name} at {time_str} on {date_str} regarding billing inquiry.",
                    f"Customer emailed requesting refund for order {order_id}; system sent template receipt.",
                    f"Support request logged on {date_str}; automated bot sent SLA timeline notice."
                ]
                closings = [
                    "Automated system response sent: 'Inquiry logged for billing review'; agent follow-up pending.",
                    "No live agent follow-up recorded prior to dispute filing date.",
                    "Status remains pending in billing queue without definitive agent resolution.",
                    "No agent assigned within SLA window; ticket aged without resolution.",
                    "Customer received templated auto-reply only; substantive review incomplete.",
                    "Human review status: awaiting triage; no commitments made to customer."
                ]
            else:
                openings = [
                    f"Customer Communication Log ({m_name}).",
                    f"CRM Audit ({m_name}) for Order {order_id}.",
                    f"Communication History ({order_id}) at {m_name}.",
                    f"CRM Search Result — {m_name}.",
                    f"Support Absence Report for Order {order_id}.",
                    f"Helpdesk Null Record ({m_name})."
                ]
                bodies = [
                    f"Database query for order {order_id} associated with {c_name} returned zero tickets.",
                    f"Search for customer {c_name} and Order {order_id} yielded no support interactions.",
                    f"No inbound or outbound email logs found for {m_name} surrounding {date_str}.",
                    f"Merchant CRM contains no open, closed, or archived tickets for order {order_id}.",
                    f"Zero phone, email, or live chat interaction logs exist for customer {c_name}.",
                    f"System audit confirmed customer did not contact support prior to chargeback."
                ]
                closings = [
                    "No customer support tickets or CRM interaction records found for order.",
                    "CRM audit confirmed absence of support correspondence.",
                    "Communication history clean; zero inquiries registered.",
                    "No record of any support engagement on file.",
                    "Customer did not raise any pre-dispute support inquiry.",
                    "Missing CRM data: zero interaction records located."
                ]
            return f"{random.choice(openings)} {random.choice(bodies)} {random.choice(closings)}"

        elif doc_type == "refund_policy":
            if quality in ["strong", "weak"]:
                openings = [
                    f"Merchant Refund & Cancellation Policy ({m_name}).",
                    f"Terms of Sale & Refund Policy — {m_name}.",
                    f"Merchant Return Policy Document ({m_name}).",
                    f"Refund Eligibility Assessment — {m_name} for Order {order_id}.",
                    f"Policy Compliance Record [{order_id}] at {m_name}.",
                    f"Terms Acknowledgement Document — {m_name} Order {order_id}."
                ]
                bodies = [
                    f"Digital subscriptions must be canceled {cancel_hours} hours prior to renewal date.",
                    f"Physical goods eligible for return within {refund_window} days of delivery for Order {order_id} (${amount:.2f}).",
                    f"Customer accepted terms during checkout for Order {order_id} (${amount:.2f}).",
                    f"Standard policy stipulates returns require pre-authorized {rma_code} within {refund_window} days.",
                    f"Published terms require return of physical goods in original packaging within {refund_window} days.",
                    f"Subscription terms specify non-refundable fees post-renewal unless canceled {cancel_hours}h prior."
                ]
                closings = [
                    "Non-refundable after billing date unless requested within 24 hours of sign-up.",
                    "Subscription fees non-refundable post-renewal per accepted point-of-sale contract.",
                    "Physical merchandise returns require valid pre-authorized RMA number.",
                    "Customer explicitly agreed to return conditions at point of checkout.",
                    "Digital services non-refundable once content access or download has occurred.",
                    "Cancellation requests submitted past deadline are ineligible for retroactive credit."
                ]
            else:
                openings = [
                    f"Merchant Refund Policy ({m_name}).",
                    f"Terms of Service ({m_name}) for Order {order_id}.",
                    f"Refund Policy Notice ({m_name}).",
                    f"Policy on File ({m_name}) — Generic terms.",
                    f"Incomplete Terms Record [{order_id}].",
                    f"Generic Policy Reference ({m_name})."
                ]
                bodies = [
                    f"Standard digital product terms apply for order {order_id} (${amount:.2f}).",
                    f"General refund terms logged for order {order_id}; item-specific rules omitted.",
                    f"Basic terms applicable to transaction for order {order_id} (${amount:.2f}).",
                    f"Blanket terms accepted at checkout without tailored return window details.",
                    f"Policy document contains general conditions but lacks category-specific return terms.",
                    f"One-size-fits-all return clause associated with order {order_id} (${amount:.2f})."
                ]
                closings = [
                    "Specific item exception rules and cancellation schedules omitted.",
                    "No tailored refund schedule or return window communicated to customer.",
                    "Policy stub lacks specific return deadlines for this purchase category.",
                    "Generic terms reference attached without explicit cancellation SLA.",
                    "Detailed refund conditions absent from archived customer receipt.",
                    "Terms page link logged but specific return policy section unverified."
                ]
            return f"{random.choice(openings)} {random.choice(bodies)} {random.choice(closings)}"

        elif doc_type == "order_details":
            if winnable and quality == "strong":
                openings = [
                    f"Order Details & Invoice: Order #{order_id} at {m_name}.",
                    f"Sales Receipt & Itemized Invoice — Order #{order_id} ({m_name}).",
                    f"Transaction Invoice #{order_id} ({m_name}) dated {date_str}.",
                    f"Verified Purchase Record — {m_name} (Invoice {invoice_num}).",
                    f"Customer Order Ledger [{order_id}] — {m_name}.",
                    f"Tax Invoice {invoice_num} — {m_name} Order {order_id}."
                ]
                bodies = [
                    f"Order placed on {date_str} at {time_str} for total amount ${amount:.2f} via {payment_method}.",
                    f"Purchaser: {c_name} ({faker_ctx['customer_email']}). Billing & Shipping: {c_addr}.",
                    f"Customer Account History: Active member for {faker_ctx['account_age_months']} months with {faker_ctx['past_orders']} prior orders.",
                    f"Billed to {c_name} at {c_addr}; payment verified via {payment_method}.",
                    f"Total charged: ${amount:.2f}. Billing address matches shipping destination ({c_addr}).",
                    f"Invoice ref {invoice_num} issued to {c_name} on {date_str} ({faker_ctx['customer_phone']})."
                ]
                closings = [
                    "Billing address verified against issuing bank records via AVS match.",
                    "Customer profile verified with excellent prior payment and order tenure.",
                    "Itemized charges confirmed and paid in full at time of checkout.",
                    "Account status: Verified member with established transaction history.",
                    "Order manifest verified; billing and delivery details fully aligned.",
                    "Payment settled successfully with zero authorization exceptions."
                ]
            else:
                openings = [
                    f"Order Details & Invoice: Order #{order_id} at {m_name}.",
                    f"Guest Order Receipt (#{order_id}) — {m_name}.",
                    f"Sales Invoice [{order_id}] ({m_name}).",
                    f"Anonymous Purchase Record — {m_name} Order {order_id}.",
                    f"One-Time Guest Order {order_id} — {m_name}.",
                    f"Invoice {invoice_num} (Guest Mode) — {m_name}."
                ]
                bodies = [
                    f"Purchased on {date_str} at {time_str} for ${amount:.2f} via {payment_method}.",
                    f"Guest checkout mode; billing/shipping destination: {c_addr}.",
                    f"Invoice {invoice_num} issued for ${amount:.2f} on {date_str}; guest user profile.",
                    f"Order processed via single-use guest checkout for total of ${amount:.2f}.",
                    f"Guest shopper transaction on {date_str}; delivery location: {c_addr}.",
                    f"Charge of ${amount:.2f} authorized via {payment_method} without user registration."
                ]
                closings = [
                    "Guest user profile; no account history or prior orders on record.",
                    "Shipping address logged; customer account tenure unavailable.",
                    "No registered email or loyalty profile linked to transaction.",
                    "One-time session checkout; returning customer history absent.",
                    "Transient guest order record; basic payment receipt issued.",
                    "Account age and order history unavailable for guest session."
                ]
            return f"{random.choice(openings)} {random.choice(bodies)} {random.choice(closings)}"

        return f"Document Record: Evidence documentation record for order {order_id}."

    def _generate_case(
        self,
        case_idx: int,
        is_sparse: bool = False
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Generates a single dispute case conditioned on real grounded Kaggle transaction fields,
        rotated CFPB narrative style claim text, and pre-assigned quality evidence docs.
        """
        case_id = f"CB-{case_idx:05d}"
        txn_id = f"txn_{case_idx:04d}_{self.fake.numerify('####')}"

        grounded_row = self.grounded_pool.get_grounded_row(case_idx)
        amount = grounded_row["amount"]
        date_str = grounded_row["date"]
        merchant_cat = grounded_row["category"]

        reason_code = random.choice(REASON_CODES)
        winnable = random.random() < 0.55

        faker_ctx = self._generate_faker_context(merchant_cat)
        m_name = faker_ctx["merchant_name"]

        # Generate claim text using CFPB rotated few-shot narratives
        customer_claim = self._generate_customer_claim(
            reason_code=reason_code,
            amount=amount,
            merchant_name=m_name,
            date_str=date_str,
            case_idx=case_idx
        )

        # Pre-calculate response deadline (14 days after dispute date)
        response_deadline = f"2026-08-{min(28, random.randint(15, 28)):02d}"

        # Determine evidence docs count and qualities
        if is_sparse:
            # 0-1 evidence docs for sparse holdout testing
            num_docs = random.choice([0, 1])
            winnable = False
            rationale = f"Sparse case ({num_docs} evidence docs): Lacks minimum documentation required to defend {reason_code} claim."
        elif winnable:
            num_docs = random.randint(2, 4)
            rationale = f"Merchant has conclusive proof (strong quality evidence) refuting the {reason_code} claim."
        else:
            num_docs = random.randint(1, 3)
            rationale = f"Evidence provided is weak or incomplete; merchant fails to disprove the {reason_code} claim."

        evidence_docs = []
        if num_docs > 0:
            if winnable:
                if reason_code in ["goods_not_received", "goods_defective"]:
                    doc_types = ["delivery_confirmation", "order_details"]
                elif reason_code in ["subscription_canceled_but_charged", "credit_not_processed"]:
                    doc_types = ["refund_policy", "communication_log"]
                elif reason_code == "duplicate_charge":
                    doc_types = ["order_details", "auth_log"]
                else:  # unauthorized_transaction
                    doc_types = ["auth_log", "order_details"]

                secondary = ["communication_log", "order_details", "refund_policy"]
                random.shuffle(secondary)
                doc_types.extend(secondary[: num_docs - 1])
            else:
                if reason_code in ["goods_not_received", "goods_defective"]:
                    primary_pool = ["delivery_confirmation", "shipping_tracking", "order_details", "communication_log"]
                elif reason_code in ["subscription_canceled_but_charged", "credit_not_processed"]:
                    primary_pool = ["communication_log", "refund_policy", "order_details"]
                elif reason_code == "duplicate_charge":
                    primary_pool = ["order_details", "communication_log", "auth_log"]
                else:
                    primary_pool = ["order_details", "communication_log", "refund_policy"]

                random.shuffle(primary_pool)
                doc_types = primary_pool[:num_docs]

            for i, doc_type in enumerate(doc_types):
                ev_id = f"EVD-{case_idx:05d}-{i+1:02d}"

                if is_sparse:
                    quality = "weak"
                elif winnable:
                    quality = "strong" if i < 2 else random.choice(["strong", "weak"])
                else:
                    quality = "weak" if i == 0 else random.choice(["weak", "missing"])

                content = self._build_doc_content(
                    doc_type=doc_type,
                    reason_code=reason_code,
                    winnable=winnable,
                    quality=quality,
                    amount=amount,
                    date_str=date_str,
                    faker_ctx=faker_ctx
                )

                # Explicit sanity scrub: ensure no transaction_id string is in content
                content = content.replace(txn_id, "")

                evidence_docs.append({
                    "evidence_id": ev_id,
                    "transaction_id": txn_id,
                    "doc_type": doc_type,
                    "content": content,
                    "quality": quality,
                    # Stored metadata for potential similarity regeneration
                    "_faker_ctx": faker_ctx,
                    "_amount": amount,
                    "_date_str": date_str,
                    "_reason_code": reason_code,
                    "_winnable": winnable
                })

        ev_ids = [d["evidence_id"] for d in evidence_docs]

        case_obj = {
            "case_id": case_id,
            "transaction_id": txn_id,
            "merchant_category": merchant_cat,
            "dispute_reason_code": reason_code,
            "dispute_amount": amount,
            "dispute_raised_date": date_str,
            "response_deadline": response_deadline,
            "customer_claim_text": customer_claim,
            "evidence_doc_ids": ev_ids,
            "label_winnable": winnable,
            "ground_truth_rationale": rationale
        }

        return case_obj, evidence_docs

    def _generate_customer_claim(
        self,
        reason_code: str,
        amount: float,
        merchant_name: str,
        date_str: str,
        case_idx: int
    ) -> str:
        """
        Generates customer_claim_text rotating CFPB narrative style examples per case index.
        """
        narrative_pool = self.grounded_pool.cfpb_narratives
        style_sample = narrative_pool[case_idx % len(narrative_pool)]

        claim_templates = {
            "goods_not_received": [
                f"I placed an order at {merchant_name} on {date_str} for ${amount:.2f}. The card was charged immediately, but I never received the item or valid tracking update. ({style_sample[:120]}...)",
                f"Regarding transaction of ${amount:.2f} at {merchant_name} on {date_str}: Merchandise was never delivered. Customer service has failed to respond to my inquiries.",
                f"I paid ${amount:.2f} to {merchant_name} on {date_str}. It has been over two weeks and the order never arrived. Requesting a full chargeback."
            ],
            "goods_defective": [
                f"Received damaged merchandise for order of ${amount:.2f} from {merchant_name}. Product was unusable upon arrival. Support refused refund or exchange.",
                f"On {date_str}, I purchased items worth ${amount:.2f} from {merchant_name}. Product arrived broken and defective. Support ticket was closed without resolution.",
                f"Item received from {merchant_name} on {date_str} (${amount:.2f}) was significantly damaged and not as described. Merchant ignored return request."
            ],
            "duplicate_charge": [
                f"My card was charged twice (${amount:.2f} each) for a single transaction at {merchant_name} on {date_str}. Merchant support failed to reverse the duplicate line item.",
                f"Single purchase of ${amount:.2f} at {merchant_name} resulted in two identical charges on my statement on {date_str}. Disputing the duplicate charge.",
                f"Double billing on {date_str}: {merchant_name} charged ${amount:.2f} twice for one checkout order. Requesting chargeback for second charge."
            ],
            "credit_not_processed": [
                f"I returned merchandise to {merchant_name} per return policy for order on {date_str} (${amount:.2f}). Merchant confirmed receipt of return, but credit was never posted.",
                f"Returned order of ${amount:.2f} to {merchant_name}. Promised refund within 7 business days, but no credit has been posted to my card statement.",
                f"Agreed store credit/refund of ${amount:.2f} for transaction on {date_str} was never processed by {merchant_name} despite return delivery confirmation."
            ],
            "subscription_canceled_but_charged": [
                f"I formally canceled my recurring subscription with {merchant_name} prior to billing date {date_str}, yet was charged ${amount:.2f}. Support ignored cancellation proof.",
                f"Canceled account with {merchant_name} prior to renewal. Unauthorized recurring charge of ${amount:.2f} posted on {date_str}. Requesting chargeback.",
                f"Subscription was terminated before renewal window. {merchant_name} charged ${amount:.2f} on {date_str} after account cancellation was confirmed."
            ],
            "unauthorized_transaction": [
                f"I do not recognize the charge of ${amount:.2f} from {merchant_name} on {date_str}. Card was in my possession and I did not authorize this purchase.",
                f"Fraudulent charge of ${amount:.2f} at {merchant_name} on {date_str}. Transaction was not authorized by me. Requesting immediate dispute resolution.",
                f"Unrecognized transaction for ${amount:.2f} posted by {merchant_name} on {date_str}. I did not initiate or approve this online order."
            ]
        }

        options = claim_templates.get(reason_code, [f"Disputing charge of ${amount:.2f} at {merchant_name}."])
        return options[case_idx % len(options)]

    def _deduplicate_evidence_similarity(self, all_evidence: List[Dict[str, Any]], threshold: float = 0.93) -> List[Dict[str, Any]]:
        """
        Embeds every evidence doc, computes pairwise cosine similarity within each doc_type,
        flags and regenerates any pair above threshold similarity until 0 pairs remain.
        Performs a FINAL check after all passes complete and asserts remaining pairs count is ZERO.
        """
        print("\n[Deduplication] Checking pairwise cosine similarity across evidence docs (>0.93 threshold)...")
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        except Exception as e:
            print(f"[Deduplication] Skipping embedding check ({e}).")
            return all_evidence

        total_regenerated = 0
        max_iterations = 100
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            # Group indices by doc_type
            by_type: Dict[str, List[int]] = {}
            for idx, doc in enumerate(all_evidence):
                dtype = doc["doc_type"]
                by_type.setdefault(dtype, []).append(idx)

            to_regenerate: set = set()
            high_sim_pair_count = 0

            for dtype, indices in by_type.items():
                if len(indices) < 2:
                    continue

                texts = [f"passage: {all_evidence[i]['content']}" for i in indices]
                embeddings = model.encode(texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
                sim_matrix = np.dot(embeddings, embeddings.T)

                n = len(indices)
                for i in range(n):
                    for j in range(i + 1, n):
                        if sim_matrix[i, j] > threshold:
                            high_sim_pair_count += 1
                            to_regenerate.add(indices[i])
                            to_regenerate.add(indices[j])

            if high_sim_pair_count == 0:
                print(f"[Deduplication] Regeneration loop converged on iteration {iteration}. Count of pairs above {threshold} = 0.")
                break

            print(f"[Deduplication] Iteration {iteration}: Found {high_sim_pair_count} pair(s) above {threshold} similarity ({len(to_regenerate)} docs flagged). Regenerating...")
            total_regenerated += len(to_regenerate)

            for idx in to_regenerate:
                doc = all_evidence[idx]
                fresh_ctx = self._generate_faker_context("ecommerce")
                new_content = self._build_doc_content(
                    doc_type=doc["doc_type"],
                    reason_code=doc.get("_reason_code", "goods_not_received"),
                    winnable=doc.get("_winnable", True),
                    quality=doc["quality"],
                    amount=doc.get("_amount", 199.99),
                    date_str=doc.get("_date_str", "2026-08-01"),
                    faker_ctx=fresh_ctx
                )
                # Add additional distinct random paragraph to break similarity floor completely
                extra_salt = f" Audit note: {self.fake.sentence(nb_words=10)} (Code {self.fake.uuid4()[:8]})."
                doc["content"] = (new_content + extra_salt).replace(doc["transaction_id"], "")

        # --- FINAL PASS: Run pairwise cosine similarity check ONE MORE TIME on final corpus ---
        print("\n[Deduplication] Running FINAL pairwise cosine similarity verification check on complete corpus...")
        final_pair_count = 0
        by_type_final: Dict[str, List[int]] = {}
        for idx, doc in enumerate(all_evidence):
            dtype = doc["doc_type"]
            by_type_final.setdefault(dtype, []).append(idx)

        for dtype, indices in by_type_final.items():
            if len(indices) < 2:
                continue
            texts = [f"passage: {all_evidence[i]['content']}" for i in indices]
            embeddings = model.encode(texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
            sim_matrix = np.dot(embeddings, embeddings.T)
            n = len(indices)
            for i in range(n):
                for j in range(i + 1, n):
                    if sim_matrix[i, j] > threshold:
                        final_pair_count += 1

        print(f"[Deduplication] Final Pairwise Cosine Similarity Check Results:")
        print(f"  - Remaining pairs above {threshold} threshold: {final_pair_count}")
        print(f"  - Total evidence docs regenerated during process: {total_regenerated}")

        if final_pair_count != 0:
            raise RuntimeError(f"[Deduplication ERROR] Final check failed! Remaining pairs above {threshold} is {final_pair_count}, expected 0.")

        print(f"[Deduplication] [PASS] Verified 0 remaining pairs above {threshold} similarity in final corpus.\n")
        return all_evidence

    def generate_full_dataset(self, total_cases: int = 120, train_ratio: float = 0.75) -> Dict[str, Any]:
        """
        Generates 120 total dispute cases (90 train, 30 holdout), pre-seeds 5 real 0-1-evidence
        cases into holdout, deduplicates similarity > 0.93, and saves datasets to disk.
        """
        num_train = int(total_cases * train_ratio)  # 90
        num_holdout = total_cases - num_train      # 30

        print(f"Generating {total_cases} grounded dispute cases ({num_train} train, {num_holdout} holdout)...")

        train_cases = []
        train_ev = []
        for i in range(num_train):
            case_idx = i + 1
            case_obj, ev_docs = self._generate_case(case_idx, is_sparse=False)
            train_cases.append(case_obj)
            train_ev.extend(ev_docs)

        holdout_cases = []
        holdout_ev = []
        # Pre-seed 5 real 0-1 evidence cases into holdout
        sparse_indices = set(range(5))

        for i in range(num_holdout):
            case_idx = num_train + i + 1
            is_sparse = (i in sparse_indices)
            case_obj, ev_docs = self._generate_case(case_idx, is_sparse=is_sparse)
            holdout_cases.append(case_obj)
            holdout_ev.extend(ev_docs)

        # Run pairwise cosine similarity check & deduplication (>0.93)
        all_ev = train_ev + holdout_ev
        deduped_ev = self._deduplicate_evidence_similarity(all_ev, threshold=0.93)

        # Clean internal metadata fields before saving
        for doc in deduped_ev:
            doc.pop("_faker_ctx", None)
            doc.pop("_amount", None)
            doc.pop("_date_str", None)
            doc.pop("_reason_code", None)
            doc.pop("_winnable", None)

        train_file = DATASETS_DIR / "train.jsonl"
        holdout_file = DATASETS_DIR / "holdout.jsonl"

        self._save_jsonl(train_file, train_cases, deduped_ev)
        self._save_jsonl(holdout_file, holdout_cases, deduped_ev)

        # Sanity check: zero literal transaction_id strings in content
        txn_id_leaks = sum(
            1 for case in train_cases + holdout_cases
            for doc in case.get("_evidence_docs_obj", [])
            if case["transaction_id"] in doc["content"]
        )
        print(f"Confirmation of literal transaction_id strings in content: {txn_id_leaks}")
        assert txn_id_leaks == 0, f"Error: Found {txn_id_leaks} literal transaction_id strings in doc content!"

        sparse_case_ids = [c["case_id"] for c in holdout_cases if len(c.get("evidence_doc_ids", [])) < 2]

        print(f"Successfully generated dataset:")
        print(f"  - Train cases: {len(train_cases)} -> {train_file}")
        print(f"  - Holdout cases: {len(holdout_cases)} (including 5 real 0-1-evidence cases) -> {holdout_file}")
        print(f"  - Holdout sparse case IDs: {sparse_case_ids}")
        print(f"  - Zero literal transaction_id strings in content confirmed: {txn_id_leaks == 0}")

        return {
            "train_cases_count": len(train_cases),
            "holdout_cases_count": len(holdout_cases),
            "sparse_case_ids": sparse_case_ids,
            "train_file": str(train_file),
            "holdout_file": str(holdout_file)
        }

    def _save_jsonl(self, filepath: Path, cases: List[Dict[str, Any]], evidence: List[Dict[str, Any]]):
        """
        Saves dispute cases along with embedded evidence docs list to JSONL format.
        """
        ev_map = {ev["evidence_id"]: ev for ev in evidence}

        with open(filepath, "w", encoding="utf-8") as f:
            for case in cases:
                record = case.copy()
                record["_evidence_docs_obj"] = [
                    ev_map[eid] for eid in case["evidence_doc_ids"] if eid in ev_map
                ]
                f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    generator = SyntheticDataGenerator()
    generator.generate_full_dataset(total_cases=120, train_ratio=0.75)

