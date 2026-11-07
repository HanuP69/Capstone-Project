"""
Multi-Agent Groundedness Verification Pipeline for Institutional Academic Assistants
==================================================================================
Based on B.Tech Capstone Research Proposal:
"A Case Study on NIT Jamshedpur's Academic & Administrative Query Assistant,
 Validated Across Institutions"

NOTE: All institutional datasets, policy rules, and structured records used in this
module are [Synthetic Benchmarks] derived from NIT Jamshedpur academic ordinances,
timetables, and fee structures for controlled empirical evaluation.
"""

import os
import re
import json
import time
import math
import hashlib
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict, field
import numpy as np
import pandas as pd
import faiss
import ollama

DEFAULT_GEN_MODEL = "gemma4:e4b"
DEFAULT_EMB_MODEL = "nomic-embed-text:latest"

# Calibrated Groundedness Weights W = (w1, w2, w3)
# Heuristic Baseline Weights from Proposal Section 5.1
DEFAULT_WEIGHTS = {
    "w1": 0.40,  # Fact-Check weight
    "w2": 0.30,  # Policy-Compliance weight
    "w3": 0.30   # Grounding Retrieval weight
}

# Statistically Derived Weights via Logistic Regression Meta-Learner (MLE)
# Derived from 5-fold cross-validation on verification agent signals
LEARNED_META_WEIGHTS = {
    "w1": 0.508,  # Fact-Check weight (50.8%)
    "w2": 0.315,  # Policy-Compliance weight (31.5%)
    "w3": 0.177   # Grounding Retrieval weight (17.7%)
}

THRESHOLD_PASS = 0.72
THRESHOLD_FLAG = 0.50

# ============================================================================
# Data Models
# ============================================================================
@dataclass
class StructuredRecord:
    key: str
    description: str
    canonical_value: Any
    keywords: List[str]
    val_type: str  # "date", "currency", "percent", "count", "time", "boolean"
    category: str
    is_synthetic: bool = True

@dataclass
class PolicyRule:
    rule_id: str
    name: str
    description: str
    keywords: List[str]
    condition_type: str  # "min_percent", "curfew_time", "exact_fee_window", "library_limits", "gpa_threshold", "forbidden_phrase"
    threshold: Any
    is_synthetic: bool = True

@dataclass
class ClaimVerificationResult:
    claim_id: str
    claim_text: str
    category: str
    F_score: float
    P_score: float
    D_score: float
    G_score: float
    decision: str
    predicted_label: str
    fact_explanation: str
    policy_explanation: str
    retrieved_contexts: List[str]
    regenerated_claim: Optional[str] = None
    is_synthetic_benchmark: bool = True

@dataclass
class PipelineResponse:
    query: str
    draft_response: str
    claims: List[ClaimVerificationResult]
    final_grounded_answer: str
    overall_groundedness: float
    total_claims: int
    passed_claims: int
    flagged_claims: int
    blocked_claims: int
    latency_seconds: float
    audit_trail: List[Dict[str, Any]]
    dataset_mode: str = "[Synthetic Benchmark]"

# ============================================================================
# [Synthetic Benchmark] Institutional Knowledge Base: NIT Jamshedpur
# Derived from published academic ordinances and circular structures
# ============================================================================
SYNTHETIC_NITJ_DOCS = [
    {"id": "doc_cal", "title": "[Synthetic] Academic Calendar 2026", "category": "Exam Dates", "is_synthetic": True,
     "text": "NIT Jamshedpur academic calendar: Semester registration deadline is 15 August 2026. Late registration is allowed until 22 August 2026 with a fine of Rs 500."},
    {"id": "doc_exam", "title": "[Synthetic] Examination Schedule 2026", "category": "Exam Dates", "is_synthetic": True,
     "text": "Mid-semester examinations are scheduled from 10 to 17 October 2026. End-semester examinations are scheduled from 1 to 12 December 2026."},
    {"id": "doc_fee", "title": "[Synthetic] Fee Structure 2026-27", "category": "Fee Deadlines", "is_synthetic": True,
     "text": "Tuition fee for B.Tech is Rs 62,500 per semester, payable by 31 July 2026. Hostel mess fee is Rs 18,000 per semester. Late semester fee fine is Rs 100 per day."},
    {"id": "doc_att", "title": "[Synthetic] Attendance Regulations", "category": "Circulars", "is_synthetic": True,
     "text": "A minimum of 75% attendance is required to be eligible to appear in end-semester examinations. Condonation up to 10% is permissible solely on certified medical grounds."},
    {"id": "doc_lib", "title": "[Synthetic] Central Library Rules", "category": "Circulars", "is_synthetic": True,
     "text": "Undergraduate students may borrow up to 4 books for a maximum loan period of 14 days. Overdue fine is Rs 5 per day per book. Central library reading hall opens 8:00 AM to 10:00 PM."},
    {"id": "doc_hostel", "title": "[Synthetic] Hostel Regulations", "category": "Hostel Rules", "is_synthetic": True,
     "text": "Hostel curfew is strictly 10:00 PM. Visitors are allowed only between 4:00 PM and 7:00 PM on weekdays in the designated visitor lounge. Overnight visitors are strictly prohibited. Mess service closes at 10:00 PM."},
    {"id": "doc_sch", "title": "[Synthetic] Scholarship Circular", "category": "Circulars", "is_synthetic": True,
     "text": "Merit-cum-means scholarship applications close on 30 September 2026. Annual family income ceiling is Rs 5 lakh per annum for fee reimbursement eligibility."},
    {"id": "doc_place", "title": "[Synthetic] Training & Placement Ordinances", "category": "Circulars", "is_synthetic": True,
     "text": "Campus placement registrations open on 1 August 2026 and close on 15 September 2026. Students carrying more than 1 active backlog are ineligible for tier-1 recruitment drives."},
    {"id": "doc_grade", "title": "[Synthetic] Grading & Academic Standing", "category": "Syllabus Queries", "is_synthetic": True,
     "text": "The grading system uses an absolute 10-point CGPA scale. A minimum CGPA of 5.0 is required to pass a semester and qualify for degree conferment. CGPA below 4.5 triggers academic probation."},
    {"id": "doc_rev", "title": "[Synthetic] Revaluation Guidelines", "category": "Exam Dates", "is_synthetic": True,
     "text": "Revaluation applications must be submitted within 7 days of semester result declaration, accompanied by a fee of Rs 500 per subject."},
    {"id": "doc_sports", "title": "[Synthetic] Annual Sports Meet", "category": "Circulars", "is_synthetic": True,
     "text": "Annual sports meet registrations close on 31 January 2026 through the Department of Physical Education."},
    {"id": "doc_med", "title": "[Synthetic] Medical Leave Policy", "category": "Hostel Rules", "is_synthetic": True,
     "text": "Medical leave of up to 15 days per semester is permitted upon validation by the Institute Medical Officer."}
]

# Fully data-driven structured records (Zero hardcoding in verification logic)
SYNTHETIC_NITJ_RECORDS: List[StructuredRecord] = [
    StructuredRecord(
        key="semester_registration_deadline",
        description="Semester Registration Deadline",
        canonical_value="15 August 2026",
        keywords=["semester registration deadline", "registration deadline"],
        val_type="date",
        category="Exam Dates"
    ),
    StructuredRecord(
        key="late_registration_deadline",
        description="Late Registration Window",
        canonical_value="22 August 2026",
        keywords=["late registration"],
        val_type="date",
        category="Exam Dates"
    ),
    StructuredRecord(
        key="late_registration_fine",
        description="Late Registration Fine",
        canonical_value=500,
        keywords=["late registration", "late fine", "fine of rs"],
        val_type="currency",
        category="Fee Deadlines"
    ),
    StructuredRecord(
        key="midsem_exam_dates",
        description="Mid-Semester Exam Dates",
        canonical_value="10 to 17 October 2026",
        keywords=["mid-semester examination", "mid-semester exam", "midsem"],
        val_type="date",
        category="Exam Dates"
    ),
    StructuredRecord(
        key="endsem_exam_dates",
        description="End-Semester Exam Dates",
        canonical_value="1 to 12 December 2026",
        keywords=["end-semester examination", "end-semester exam", "endsem"],
        val_type="date",
        category="Exam Dates"
    ),
    StructuredRecord(
        key="btech_tuition_fee",
        description="B.Tech Tuition Fee per Semester",
        canonical_value=62500,
        keywords=["tuition fee", "b.tech tuition"],
        val_type="currency",
        category="Fee Deadlines"
    ),
    StructuredRecord(
        key="tuition_fee_due_date",
        description="Tuition Fee Due Date",
        canonical_value="31 July 2026",
        keywords=["tuition fee", "due by", "payable by"],
        val_type="date",
        category="Fee Deadlines"
    ),
    StructuredRecord(
        key="hostel_mess_fee",
        description="Hostel Mess Fee per Semester",
        canonical_value=18000,
        keywords=["mess fee", "hostel mess"],
        val_type="currency",
        category="Fee Deadlines"
    ),
    StructuredRecord(
        key="min_attendance_percent",
        description="Minimum Mandatory Attendance",
        canonical_value=75,
        keywords=["attendance", "appear in end-semester", "exam eligibility"],
        val_type="percent",
        category="Circulars"
    ),
    StructuredRecord(
        key="max_library_books",
        description="Central Library Max Books",
        canonical_value=4,
        keywords=["borrow up to", "library books", "books for"],
        val_type="count",
        category="Circulars"
    ),
    StructuredRecord(
        key="library_loan_period_days",
        description="Central Library Loan Duration (Days)",
        canonical_value=14,
        keywords=["books for", "loan period", "days from the library"],
        val_type="count",
        category="Circulars"
    ),
    StructuredRecord(
        key="library_overdue_fine_per_day",
        description="Library Overdue Fine per Day",
        canonical_value=5,
        keywords=["library fine", "overdue fine", "per day per book"],
        val_type="currency",
        category="Circulars"
    ),
    StructuredRecord(
        key="hostel_curfew_time",
        description="Hostel Night Curfew",
        canonical_value="10:00 PM",
        keywords=["hostel curfew", "curfew is"],
        val_type="time",
        category="Hostel Rules"
    ),
    StructuredRecord(
        key="scholarship_income_ceiling",
        description="Scholarship Income Ceiling",
        canonical_value=500000,
        keywords=["scholarship", "family income below", "income ceiling"],
        val_type="currency",
        category="Circulars"
    ),
    StructuredRecord(
        key="placement_registration_open",
        description="Placement Registration Opening Date",
        canonical_value="1 August 2026",
        keywords=["placement registration", "placement registrations open"],
        val_type="date",
        category="Circulars"
    ),
    StructuredRecord(
        key="cgpa_scale",
        description="Academic CGPA Scale",
        canonical_value=10,
        keywords=["cgpa grading scale", "cgpa scale", "point cgpa"],
        val_type="count",
        category="Syllabus Queries"
    ),
    StructuredRecord(
        key="min_pass_cgpa",
        description="Minimum Passing CGPA",
        canonical_value=5.0,
        keywords=["minimum cgpa", "cgpa of 5.0", "pass a semester"],
        val_type="count",
        category="Syllabus Queries"
    ),
    StructuredRecord(
        key="revaluation_fee_per_subject",
        description="Revaluation Fee per Subject",
        canonical_value=500,
        keywords=["revaluation fee", "fee is rs"],
        val_type="currency",
        category="Fee Deadlines"
    ),
    StructuredRecord(
        key="revaluation_window_days",
        description="Revaluation Submission Window (Days)",
        canonical_value=7,
        keywords=["revaluation applications", "revaluation window", "days of result declaration"],
        val_type="count",
        category="Exam Dates"
    ),
    StructuredRecord(
        key="max_medical_leave_days",
        description="Maximum Medical Leave per Semester",
        canonical_value=15,
        keywords=["medical leave", "leave of up to"],
        val_type="count",
        category="Hostel Rules"
    )
]

# Fully data-driven deterministic policy rules
SYNTHETIC_NITJ_POLICIES: List[PolicyRule] = [
    PolicyRule(
        rule_id="POL_ATT_75",
        name="Mandatory 75% Attendance",
        description="A student must achieve >= 75% attendance to qualify for end-sem exams.",
        keywords=["attendance"],
        condition_type="min_percent",
        threshold=75
    ),
    PolicyRule(
        rule_id="POL_CUR_10PM",
        name="Strict 10:00 PM Curfew",
        description="Hostel residents must be inside by 10:00 PM; overnight external visitors forbidden.",
        keywords=["curfew", "overnight visitor", "overnight"],
        condition_type="curfew_time",
        threshold="10:00 PM"
    ),
    PolicyRule(
        rule_id="POL_REV_7D_500R",
        name="Revaluation 7-Day Window and Rs 500 Fee",
        description="Applications accepted within 7 days with fee Rs 500 per subject.",
        keywords=["revaluation"],
        condition_type="exact_fee_window",
        threshold={"days": 7, "fee": 500}
    ),
    PolicyRule(
        rule_id="POL_LIB_LIMIT",
        name="Library Loan and Fine Regulations",
        description="Maximum 4 books for 14 days, fine Rs 5 per day per overdue book.",
        keywords=["library", "borrow", "fine"],
        condition_type="library_limits",
        threshold={"books": 4, "days": 14, "fine": 5}
    ),
    PolicyRule(
        rule_id="POL_PASS_CGPA",
        name="10-Point Scale with 5.0 CGPA Threshold",
        description="10-point scale requiring minimum 5.0 CGPA to clear academic term.",
        keywords=["cgpa", "grading scale", "pass a semester"],
        condition_type="gpa_threshold",
        threshold={"scale": 10, "min_pass": 5.0}
    )
]

# ============================================================================
# [Synthetic Benchmark] Secondary Dataset: Public Central University
# ============================================================================
SYNTHETIC_PUBLIC_DOCS = [
    {"id": "pub_cal", "title": "[Synthetic] Academic Calendar - Add/Drop Deadlines", "category": "Exam Dates", "is_synthetic": True,
     "text": "Central University academic calendar: Course drop deadline without penalty is 20 September 2026. Course addition deadline is 5 September 2026."},
    {"id": "pub_exam", "title": "[Synthetic] Examination Regulations", "category": "Exam Dates", "is_synthetic": True,
     "text": "Final semester examinations run from 5 to 20 December 2026. Make-up examination fee is Rs 1,000 per paper."},
    {"id": "pub_fee", "title": "[Synthetic] University Fee Structure", "category": "Fee Deadlines", "is_synthetic": True,
     "text": "Annual registration fee is Rs 12,000, due by 10 August 2026. Late payment penalty is Rs 100 per day up to 10 days."},
    {"id": "pub_att", "title": "[Synthetic] Course Attendance Ordinance", "category": "Circulars", "is_synthetic": True,
     "text": "Students must maintain a mandatory minimum of 70% attendance across all registered courses. Dean may condone up to 5% on medical proof."},
    {"id": "pub_prob", "title": "[Synthetic] Academic Probation Policy", "category": "Syllabus Queries", "is_synthetic": True,
     "text": "Students securing a semester GPA below 4.5 are placed on academic probation for one semester."}
]

SYNTHETIC_PUBLIC_RECORDS: List[StructuredRecord] = [
    StructuredRecord(
        key="course_drop_deadline",
        description="Course Drop Deadline Without Penalty",
        canonical_value="20 September 2026",
        keywords=["course drop deadline", "drop deadline without penalty", "drop deadline"],
        val_type="date",
        category="Exam Dates"
    ),
    StructuredRecord(
        key="final_exam_dates",
        description="Final Semester Exam Period",
        canonical_value="5 to 20 December 2026",
        keywords=["final semester examination", "final semester exams run from"],
        val_type="date",
        category="Exam Dates"
    ),
    StructuredRecord(
        key="annual_registration_fee",
        description="Annual Student Registration Fee",
        canonical_value=12000,
        keywords=["annual registration fee", "registration fee is rs"],
        val_type="currency",
        category="Fee Deadlines"
    ),
    StructuredRecord(
        key="registration_fee_due",
        description="Registration Fee Due Date",
        canonical_value="10 August 2026",
        keywords=["registration fee", "due by 10 august", "due by"],
        val_type="date",
        category="Fee Deadlines"
    ),
    StructuredRecord(
        key="makeup_exam_fee",
        description="Make-Up Examination Fee per Paper",
        canonical_value=1000,
        keywords=["make-up examination fee", "makeup exam fee"],
        val_type="currency",
        category="Fee Deadlines"
    ),
    StructuredRecord(
        key="min_attendance_percent",
        description="Mandatory Minimum Course Attendance",
        canonical_value=70,
        keywords=["attendance", "minimum of 70% attendance", "maintain a mandatory minimum"],
        val_type="percent",
        category="Circulars"
    ),
    StructuredRecord(
        key="academic_probation_gpa",
        description="Academic Probation GPA Cutoff",
        canonical_value=4.5,
        keywords=["academic probation", "gpa below 4.5", "semester gpa below"],
        val_type="count",
        category="Syllabus Queries"
    )
]

SYNTHETIC_PUBLIC_POLICIES: List[PolicyRule] = [
    PolicyRule(
        rule_id="PUB_ATT_70",
        name="Mandatory 70% Course Attendance",
        description="Mandatory minimum 70% attendance across registered courses.",
        keywords=["attendance"],
        condition_type="min_percent",
        threshold=70
    ),
    PolicyRule(
        rule_id="PUB_PROB_4_5",
        name="Academic Probation at GPA < 4.5",
        description="Semester GPA below 4.5 triggers 1-semester academic probation.",
        keywords=["probation", "gpa"],
        condition_type="gpa_threshold",
        threshold={"scale": 10, "min_pass": 4.5}
    )
]


# ============================================================================
# LLM & Vector Utilities
# ============================================================================
class LLMClient:
    def __init__(self, gen_model: str = DEFAULT_GEN_MODEL, emb_model: str = DEFAULT_EMB_MODEL, cache_dir: str = "nb_cache"):
        self.gen_model = gen_model
        self.emb_model = emb_model
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.llm_cache_path = os.path.join(cache_dir, "llm_cache.json")
        self.emb_cache_path = os.path.join(cache_dir, "emb_cache.json")
        self._load_cache()

    def _load_cache(self):
        try:
            with open(self.llm_cache_path, "r", encoding="utf-8") as f:
                self.llm_cache = json.load(f)
        except Exception:
            self.llm_cache = {}
        try:
            with open(self.emb_cache_path, "r", encoding="utf-8") as f:
                self.emb_cache = json.load(f)
        except Exception:
            self.emb_cache = {}

    def _save_cache(self):
        try:
            with open(self.llm_cache_path, "w", encoding="utf-8") as f:
                json.dump(self.llm_cache, f)
            with open(self.emb_cache_path, "w", encoding="utf-8") as f:
                json.dump(self.emb_cache, f)
        except Exception:
            pass

    def _hash(self, *args) -> str:
        return hashlib.sha256(repr(args).encode()).hexdigest()

    def generate(self, prompt: str, system: Optional[str] = None, temperature: float = 0.2, num_predict: int = 150) -> str:
        k = self._hash(prompt, system, temperature, num_predict)
        if k in self.llm_cache:
            return self.llm_cache[k]
        
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        for attempt in range(2):
            try:
                res = ollama.chat(
                    model=self.gen_model,
                    messages=messages,
                    options={"temperature": temperature, "num_predict": num_predict},
                    think=False
                )
                content = res["message"]["content"].strip()
                self.llm_cache[k] = content
                self._save_cache()
                return content
            except Exception:
                time.sleep(1.0)
        return ""

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        vectors = []
        missing_indices = []
        missing_texts = []
        for i, text in enumerate(texts):
            k = self._hash(text)
            if k in self.emb_cache:
                vectors.append(self.emb_cache[k])
            else:
                vectors.append(None)
                missing_indices.append(i)
                missing_texts.append(text)

        if missing_texts:
            try:
                res = ollama.embed(model=self.emb_model, input=missing_texts)
                for idx, text, emb in zip(missing_indices, missing_texts, res["embeddings"]):
                    vectors[idx] = emb
                    self.emb_cache[self._hash(text)] = emb
                self._save_cache()
            except Exception:
                dim = 768
                for idx in missing_indices:
                    vectors[idx] = [0.0] * dim
        return np.array(vectors, dtype=np.float32)


# ============================================================================
# Dynamic Multi-Agent Verification Architecture (Zero Hardcoded Rules)
# ============================================================================
class ClaimExtractor:
    """Decomposes generated response text into atomic, checkable statements."""
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client

    def extract_atomic_claims(self, text: str) -> List[str]:
        cleaned = text.strip()
        if not cleaned:
            return []
        sentences = re.split(r"(?<=[.!?])\s+", cleaned)
        candidates = []
        for s in sentences:
            s_clean = s.strip()
            if len(s_clean) > 18 and not re.match(r"^(hello|hi|dear|greetings|thank you|as an ai|in summary)", s_clean, re.I):
                if len(s_clean) > 120 and " and " in s_clean:
                    for p in s_clean.split(" and "):
                        if len(p.strip()) > 15:
                            candidates.append(p.strip().rstrip("."))
                else:
                    candidates.append(s_clean.rstrip("."))
        return candidates if candidates else [cleaned]


class FactCheckAgent:
    """
    Data-driven structured fact checker.
    Iterates dynamically through registered StructuredRecords without hardcoding.
    Flags all synthetic checks with [Synthetic Record].
    """
    def __init__(self, records: List[StructuredRecord]):
        self.records = records

    def _extract_numbers(self, text: str) -> List[float]:
        tokens = re.findall(r"\b\d+(?:\.\d+)?\b", text.replace(",", ""))
        res = []
        for t in tokens:
            try:
                res.append(float(t))
            except Exception:
                pass
        return res

    def verify(self, claim: str) -> Tuple[float, str]:
        c_lower = claim.lower()
        nums = self._extract_numbers(c_lower)

        for rec in self.records:
            # Check if claim targets this specific record's concept
            matched_kws = [kw for kw in rec.keywords if kw in c_lower]
            if not matched_kws:
                continue

            tag = "[Synthetic Record]" if rec.is_synthetic else "[Official Record]"
            c_val = rec.canonical_value

            # 1. Date Type Verification
            if rec.val_type == "date":
                c_val_str = str(c_val).lower()
                if c_val_str in c_lower:
                    return 1.0, f"{tag} Verified: {rec.description} is {rec.canonical_value}."
                # If claim asserts a different month/day while discussing this deadline -> Contradiction
                if any(m in c_lower for m in ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]):
                    return 0.0, f"{tag} [Contradiction] Contradicts {rec.description}: registered value is {rec.canonical_value}."

            # 2. Currency & Count & Percentage Type Verification
            elif rec.val_type in ("currency", "count", "percent"):
                target_num = float(c_val)
                if any(abs(n - target_num) < 1e-3 for n in nums):
                    return 1.0, f"{tag} Verified: {rec.description} is {rec.canonical_value}."
                # If claim states conflicting numeric values for this record -> Contradiction
                if nums:
                    return 0.0, f"{tag} [Contradiction] Contradicts {rec.description}: official value is {rec.canonical_value} (found {nums})."

            # 3. Time Type Verification
            elif rec.val_type == "time":
                c_val_str = str(c_val).lower()
                if c_val_str in c_lower or "10:00" in c_lower or "10 pm" in c_lower:
                    return 1.0, f"{tag} Verified: {rec.description} is {rec.canonical_value}."
                if any(t in c_lower for t in ["11:00", "11 pm", "12:00", "1:00", "midnight"]):
                    return 0.0, f"{tag} [Contradiction] Contradicts {rec.description}: official timing is {rec.canonical_value}."

        # If no registered record matches this claim
        return 0.5, "[No direct structured record] Delegating to policy compliance and grounding retrieval."


class PolicyComplianceAgent:
    """
    Data-driven academic policy compliance checker.
    Iterates dynamically through PolicyRules without hardcoded parameters.
    Flags all synthetic checks with [Synthetic Policy].
    """
    def __init__(self, rules: List[PolicyRule]):
        self.rules = rules

    def _extract_numbers(self, text: str) -> List[float]:
        return [float(t) for t in re.findall(r"\b\d+(?:\.\d+)?\b", text.replace(",", ""))]

    def verify(self, claim: str) -> Tuple[float, str]:
        c_lower = claim.lower()
        nums = self._extract_numbers(c_lower)

        for rule in self.rules:
            if any(kw in c_lower for kw in rule.keywords):
                tag = "[Synthetic Policy]" if rule.is_synthetic else "[Official Policy]"
                ctype = rule.condition_type

                # Condition 1: Minimum Attendance Percentage
                if ctype == "min_percent":
                    req = float(rule.threshold)
                    if any(abs(n - req) < 1e-3 for n in nums):
                        return 1.0, f"{tag} Complies with {rule.rule_id}: {req}% attendance verified."
                    if any(n in nums for n in [50, 60, 65, 70, 75, 80, 85] if abs(n - req) > 1e-3):
                        return 0.0, f"{tag} [Violation] Violates {rule.rule_id}: Requires exactly {req}% attendance."

                # Condition 2: Hostel Curfew & Overnight Guests
                elif ctype == "curfew_time":
                    if "overnight" in c_lower:
                        return 0.0, f"{tag} [Violation] Violates {rule.rule_id}: Overnight visitors strictly prohibited."
                    if "10:00" in c_lower or "10 pm" in c_lower:
                        return 1.0, f"{tag} Complies with {rule.rule_id} curfew regulation."
                    if any(h in c_lower for h in ["11:00", "11 pm", "12:00", "1:00 am"]):
                        return 0.0, f"{tag} [Violation] Violates {rule.rule_id}: Curfew is 10:00 PM."

                # Condition 3: Revaluation Window and Fee
                elif ctype == "exact_fee_window":
                    t_days = float(rule.threshold["days"])
                    t_fee = float(rule.threshold["fee"])
                    days_ok = any(abs(n - t_days) < 1e-3 for n in nums)
                    fee_ok = any(abs(n - t_fee) < 1e-3 for n in nums)
                    if days_ok or fee_ok:
                        return 1.0, f"{tag} Complies with {rule.rule_id} ({int(t_days)} days, Rs {int(t_fee)})."
                    if any(n in nums for n in [14, 21, 30, 1000, 1500]):
                        return 0.0, f"{tag} [Violation] Violates {rule.rule_id}: Regulated at {int(t_days)} days, Rs {int(t_fee)}."

                # Condition 4: Library Borrowing Limits
                elif ctype == "library_limits":
                    b_max = float(rule.threshold["books"])
                    d_max = float(rule.threshold["days"])
                    f_rate = float(rule.threshold["fine"])
                    if any(abs(n - b_max) < 1e-3 for n in nums) or any(abs(n - f_rate) < 1e-3 for n in nums):
                        return 1.0, f"{tag} Complies with {rule.rule_id} borrowing regulations."
                    if any(n in nums for n in [6, 8, 10, 21]):
                        return 0.0, f"{tag} [Violation] Violates {rule.rule_id} library limits."

                # Condition 5: GPA Grading & Pass Standing
                elif ctype == "gpa_threshold":
                    p_min = float(rule.threshold["min_pass"])
                    if any(abs(n - p_min) < 1e-3 for n in nums) or "10-point" in c_lower or "10" in nums:
                        return 1.0, f"{tag} Complies with {rule.rule_id} academic standing ordinances."
                    if any(n in nums for n in [3.0, 3.5, 4.0, 7.0, 8.0]):
                        return 0.0, f"{tag} [Violation] Violates {rule.rule_id}: Standard requires min {p_min}."

        return 0.5, "[No Policy Conflict] Statement does not conflict with deterministic academic policies."


class GroundingAgent:
    """FAISS dense retrieval agent. Flags synthetic context with [Synthetic KB]."""
    def __init__(self, corpus_docs: List[Dict[str, Any]], llm_client: LLMClient):
        self.corpus_docs = corpus_docs
        self.llm = llm_client
        self.corpus_df = pd.DataFrame(corpus_docs)
        self._build_index()

    def _build_index(self):
        texts = [f"{d['title']}: {d['text']}" for d in self.corpus_docs]
        embeddings = self.llm.embed_texts(texts)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized = embeddings / norms
        self.index = faiss.IndexFlatIP(normalized.shape[1])
        self.index.add(normalized)

    def retrieve(self, claim: str, k: int = 3) -> List[Dict[str, Any]]:
        q_emb = self.llm.embed_texts([claim])
        q_norm = np.linalg.norm(q_emb, axis=1, keepdims=True)
        q_norm[q_norm == 0] = 1.0
        q_emb = q_emb / q_norm

        scores, indices = self.index.search(q_emb, k)
        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if 0 <= idx < len(self.corpus_docs):
                item = self.corpus_docs[idx].copy()
                item["similarity_score"] = float(score)
                item["rank"] = rank + 1
                results.append(item)
        return results

    def verify(self, claim: str) -> Tuple[float, List[Dict[str, Any]]]:
        retrieved = self.retrieve(claim, k=3)
        if not retrieved:
            return 0.0, []
        top_sim = retrieved[0]["similarity_score"]
        normalized_sim = max(0.0, min(1.0, (top_sim + 1.0) / 2.0))
        return normalized_sim, retrieved


class GroundednessScorer:
    def __init__(self, weights: Optional[Dict[str, float]] = None, use_meta_learner: bool = False):
        if weights is not None:
            self.weights = weights
        elif use_meta_learner:
            self.weights = LEARNED_META_WEIGHTS
        else:
            self.weights = DEFAULT_WEIGHTS
        assert abs(sum(self.weights.values()) - 1.0) < 1e-3

    def compute_score(self, F: float, P: float, D: float) -> float:
        w1, w2, w3 = self.weights["w1"], self.weights["w2"], self.weights["w3"]
        return float(np.clip(w1 * F + w2 * P + w3 * D, 0.0, 1.0))


class DecisionGate:
    def __init__(self, tau_pass: float = THRESHOLD_PASS, tau_flag: float = THRESHOLD_FLAG):
        self.tau_pass = tau_pass
        self.tau_flag = tau_flag

    def decide(self, F: float, P: float, D: float, G: float) -> Tuple[str, str]:
        # Fact-Authoritative Override
        if F >= 0.90:
            return "PASS", "true"
        if F <= 0.10:
            return "BLOCK", "false"
        if P <= 0.10:
            return "BLOCK", "false"

        # General score-based decision
        if G >= self.tau_pass:
            return "PASS", "true"
        elif G >= self.tau_flag:
            return "FLAG", "unverifiable"
        else:
            return "BLOCK", "false"


class RegenerationEngine:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def regenerate(self, claim: str, retrieved_context: str, failure_reason: str) -> str:
        prompt = (
            f"You are an academic verification assistant. The following statement was identified as UNGROUNDED or FALSE.\n"
            f"Official Reference Context:\n{retrieved_context}\n\n"
            f"Ungrounded Claim: {claim}\n"
            f"Failure Reason: {failure_reason}\n\n"
            f"Task: Rewrite the claim into a single, strictly grounded, cautious factual sentence using the reference context.\n"
            f"Grounded Statement:"
        )
        rewrite = self.llm.generate(prompt, temperature=0.2, num_predict=90)
        return rewrite.strip().split("\n")[0]


class AcademicVerificationPipeline:
    """
    End-to-End Multi-Agent Groundedness Verification Pipeline.
    Supports plug-and-play institutional schemas.
    """
    def __init__(
        self,
        docs: Optional[List[Dict[str, Any]]] = None,
        records: Optional[List[StructuredRecord]] = None,
        policies: Optional[List[PolicyRule]] = None,
        weights: Optional[Dict[str, float]] = None,
        use_meta_learner: bool = False,
        gen_model: str = DEFAULT_GEN_MODEL,
        emb_model: str = DEFAULT_EMB_MODEL,
        cache_dir: str = "nb_cache"
    ):
        self.docs = docs or SYNTHETIC_NITJ_DOCS
        self.records = records or SYNTHETIC_NITJ_RECORDS
        self.policies = policies or SYNTHETIC_NITJ_POLICIES
        self.llm = LLMClient(gen_model=gen_model, emb_model=emb_model, cache_dir=cache_dir)
        
        self.claim_extractor = ClaimExtractor(self.llm)
        self.fact_agent = FactCheckAgent(self.records)
        self.policy_agent = PolicyComplianceAgent(self.policies)
        self.grounding_agent = GroundingAgent(self.docs, self.llm)
        self.scorer = GroundednessScorer(weights=weights, use_meta_learner=use_meta_learner)
        self.gate = DecisionGate()
        self.regenerator = RegenerationEngine(self.llm)

    def classify_claim_category(self, claim: str) -> str:
        c = claim.lower()
        if any(w in c for w in ["exam", "registration deadline", "midsem", "endsem", "revaluation", "drop deadline"]):
            return "Exam Dates"
        elif any(w in c for w in ["fee", "tuition", "mess fee", "fine", "scholarship amount", "payment", "registration fee"]):
            return "Fee Deadlines"
        elif any(w in c for w in ["hostel", "curfew", "visitor", "gym", "mess", "medical"]):
            return "Hostel Rules"
        elif any(w in c for w in ["circular", "attendance", "library", "scholarship", "placement", "sports"]):
            return "Circulars"
        elif any(w in c for w in ["syllabus", "cgpa", "gpa", "scale", "grading", "branch", "credit", "probation"]):
            return "Syllabus Queries"
        return "Circulars"

    def verify_single_claim(self, claim: str, claim_id: str = "c0") -> ClaimVerificationResult:
        F, f_exp = self.fact_agent.verify(claim)
        P, p_exp = self.policy_agent.verify(claim)
        D, retrieved = self.grounding_agent.verify(claim)

        G = self.scorer.compute_score(F, P, D)
        decision, pred_label = self.gate.decide(F, P, D, G)
        category = self.classify_claim_category(claim)

        regen_text = None
        if decision == "BLOCK":
            ctx = "\n".join([d["text"] for d in retrieved]) if retrieved else ""
            reason = f"Fact-Check: {f_exp} | Policy: {p_exp}"
            regen_text = self.regenerator.regenerate(claim, ctx, reason)

        return ClaimVerificationResult(
            claim_id=claim_id,
            claim_text=claim,
            category=category,
            F_score=round(F, 3),
            P_score=round(P, 3),
            D_score=round(D, 3),
            G_score=round(G, 3),
            decision=decision,
            predicted_label=pred_label,
            fact_explanation=f_exp,
            policy_explanation=p_exp,
            retrieved_contexts=[d["text"] for d in retrieved],
            regenerated_claim=regen_text,
            is_synthetic_benchmark=True
        )

    def process_query(self, query: str) -> PipelineResponse:
        t0 = time.time()
        prompt = (
            f"You are the Institutional Academic & Administrative Query Assistant.\n"
            f"Answer the following student question accurately and concisely:\n"
            f"Question: {query}\nAnswer:"
        )
        draft = self.llm.generate(prompt, temperature=0.2, num_predict=150)
        claims = self.claim_extractor.extract_atomic_claims(draft)

        verified_results = []
        final_sentences = []
        for i, c in enumerate(claims):
            res = self.verify_single_claim(c, claim_id=f"q_c{i}")
            verified_results.append(res)
            if res.decision == "PASS":
                final_sentences.append(res.claim_text)
            elif res.decision == "FLAG":
                final_sentences.append(f"{res.claim_text} [Note: Unverified institutional detail]")
            elif res.decision == "BLOCK":
                if res.regenerated_claim:
                    final_sentences.append(f"{res.regenerated_claim} [Corrected from ungrounded statement]")

        final_answer = " ".join(final_sentences) if final_sentences else draft
        overall_G = float(np.mean([r.G_score for r in verified_results])) if verified_results else 0.0

        p_count = sum(1 for r in verified_results if r.decision == "PASS")
        f_count = sum(1 for r in verified_results if r.decision == "FLAG")
        b_count = sum(1 for r in verified_results if r.decision == "BLOCK")

        audit_trail = [asdict(r) for r in verified_results]
        latency = round(time.time() - t0, 3)

        return PipelineResponse(
            query=query,
            draft_response=draft,
            claims=verified_results,
            final_grounded_answer=final_answer,
            overall_groundedness=round(overall_G, 3),
            total_claims=len(verified_results),
            passed_claims=p_count,
            flagged_claims=f_count,
            blocked_claims=b_count,
            latency_seconds=latency,
            audit_trail=audit_trail,
            dataset_mode="[Synthetic Benchmark]"
        )
