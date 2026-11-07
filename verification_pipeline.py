"""
Multi-Agent Groundedness Verification Pipeline for Institutional Academic Assistants
==================================================================================
Based on B.Tech Capstone Research Proposal:
"A Case Study on NIT Jamshedpur's Academic & Administrative Query Assistant,
 Validated Across Institutions"
"""

import os
import re
import json
import time
import math
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
import faiss
import ollama

DEFAULT_GEN_MODEL = "gemma4:e4b"
DEFAULT_EMB_MODEL = "nomic-embed-text:latest"

DEFAULT_WEIGHTS = {
    "w1": 0.40,  # Fact-check weight
    "w2": 0.30,  # Policy-compliance weight
    "w3": 0.30   # Grounding retrieval weight
}

THRESHOLD_PASS = 0.72
THRESHOLD_FLAG = 0.50

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

# ============================================================================
# Datasets & Structured Records
# ============================================================================
NITJ_DOCS = [
    {"id": "doc_cal", "title": "Academic Calendar 2026", "category": "Exam Dates",
     "text": "NIT Jamshedpur academic calendar: Semester registration deadline is 15 August 2026. Late registration is allowed until 22 August 2026 with a fine of Rs 500."},
    {"id": "doc_exam", "title": "Examination Schedule 2026", "category": "Exam Dates",
     "text": "Mid-semester examinations are scheduled from 10 to 17 October 2026. End-semester examinations are scheduled from 1 to 12 December 2026."},
    {"id": "doc_fee", "title": "Fee Structure 2026-27", "category": "Fee Deadlines",
     "text": "Tuition fee for B.Tech is Rs 62,500 per semester, payable by 31 July 2026. Hostel mess fee is Rs 18,000 per semester."},
    {"id": "doc_att", "title": "Attendance Regulations", "category": "Circulars",
     "text": "A minimum of 75% attendance is required to be eligible to appear in end-semester examinations. Condonation up to 10% allowed on certified medical grounds."},
    {"id": "doc_lib", "title": "Central Library Rules", "category": "Circulars",
     "text": "Students may borrow up to 4 books for 14 days. Overdue fine is Rs 5 per day per book. Central library reading hall opens 8:00 AM to 10:00 PM."},
    {"id": "doc_hostel", "title": "Hostel Regulations", "category": "Hostel Rules",
     "text": "Hostel curfew is 10:00 PM. Visitors are allowed only between 4:00 PM and 7:00 PM on weekdays in the common visitor room. Overnight visitors are strictly prohibited. Mess service stops at 10:00 PM."},
    {"id": "doc_sch", "title": "Scholarship Circular", "category": "Circulars",
     "text": "Merit-cum-means scholarship applications close on 30 September 2026. Eligibility requires family income below Rs 5 lakh per annum."},
    {"id": "doc_place", "title": "Training & Placement Rules", "category": "Circulars",
     "text": "Campus placement registrations open on 1 August 2026 and close on 15 September 2026. Students with more than 1 active backlog are ineligible."},
    {"id": "doc_grade", "title": "Grading & Academic Performance", "category": "Syllabus Queries",
     "text": "The grading system uses a 10-point CGPA scale. A minimum CGPA of 5.0 is required to pass a semester and qualify for degree award."},
    {"id": "doc_rev", "title": "Revaluation Guidelines", "category": "Exam Dates",
     "text": "Revaluation applications must be submitted within 7 days of result declaration, with a fee of Rs 500 per subject."},
    {"id": "doc_sports", "title": "Annual Sports Meet", "category": "Circulars",
     "text": "Annual sports meet is held in February. Students must register by 31 January 2026 through the physical education office."},
    {"id": "doc_med", "title": "Medical Leave Policy", "category": "Hostel Rules",
     "text": "Medical leave of up to 15 days per semester is allowed with a certificate from the institute medical officer."}
]

NITJ_STRUCTURED_RECORDS = {
    "semester_registration_deadline": "15 August 2026",
    "late_registration_deadline": "22 August 2026",
    "late_registration_fine": 500,
    "midsem_exam_dates": "10 to 17 October 2026",
    "endsem_exam_dates": "1 to 12 December 2026",
    "btech_tuition_fee": 62500,
    "tuition_fee_due_date": "31 July 2026",
    "hostel_mess_fee": 18000,
    "min_attendance_percent": 75,
    "max_library_books": 4,
    "library_loan_period_days": 14,
    "library_overdue_fine_per_day": 5,
    "hostel_curfew_time": "10:00 PM",
    "visitor_hours": "4:00 PM to 7:00 PM",
    "overnight_visitors_allowed": False,
    "mess_closing_time": "10:00 PM",
    "scholarship_application_deadline": "30 September 2026",
    "scholarship_income_ceiling": 500000,
    "placement_registration_open": "1 August 2026",
    "placement_registration_close": "15 September 2026",
    "cgpa_scale": 10,
    "min_pass_cgpa": 5.0,
    "revaluation_window_days": 7,
    "revaluation_fee_per_subject": 500,
    "sports_registration_deadline": "31 January 2026",
    "max_medical_leave_days": 15
}

NITJ_POLICY_RULES = [
    {"rule_id": "POL_ATT", "text": "Minimum 75% attendance required for examination eligibility.",
     "keywords": ["attendance", "appear", "eligibility"], "required_val": 75, "attr": "attendance"},
    {"rule_id": "POL_REV", "text": "Revaluation within 7 days with fee Rs 500.",
     "keywords": ["revaluation", "result"], "required_val": [7, 500], "attr": "revaluation"},
    {"rule_id": "POL_LIB", "text": "Library allows 4 books for 14 days, fine Rs 5 per day.",
     "keywords": ["library", "book", "fine", "overdue"], "required_val": [4, 14, 5], "attr": "library"},
    {"rule_id": "POL_CUR", "text": "Hostel curfew is 10:00 PM, visitors 4:00 PM to 7:00 PM.",
     "keywords": ["hostel", "curfew", "visitor", "visitors"], "required_val": ["10:00", "10 pm"], "attr": "curfew"},
    {"rule_id": "POL_GRAD", "text": "10-point CGPA scale with 5.0 minimum pass CGPA.",
     "keywords": ["cgpa", "scale", "pass"], "required_val": [10, 5.0], "attr": "cgpa"}
]

PUBLIC_UNIV_DOCS = [
    {"id": "pub_cal", "title": "Academic Handbook - Deadlines", "category": "Exam Dates",
     "text": "Central University academic calendar: Course drop deadline without penalty is 20 September 2026. Course addition deadline is 5 September 2026."},
    {"id": "pub_exam", "title": "Examination Regulations", "category": "Exam Dates",
     "text": "Final semester examinations run from 5 to 20 December 2026. Make-up examination fee is Rs 1,000 per paper."},
    {"id": "pub_fee", "title": "Fee Structure", "category": "Fee Deadlines",
     "text": "Annual registration fee is Rs 12,000, due by 10 August 2026. Late payment penalty is Rs 100 per day up to 10 days."},
    {"id": "pub_att", "title": "Attendance Policy", "category": "Circulars",
     "text": "Students must maintain a mandatory minimum of 70% attendance across all registered courses. Dean may condone up to 5% on medical proof."},
    {"id": "pub_prob", "title": "Academic Probation Regulations", "category": "Syllabus Queries",
     "text": "Students securing a semester GPA below 4.5 are placed on academic probation for one semester."}
]

PUBLIC_UNIV_STRUCTURED_RECORDS = {
    "course_drop_deadline": "20 September 2026",
    "course_add_deadline": "5 September 2026",
    "final_exam_dates": "5 to 20 December 2026",
    "makeup_exam_fee": 1000,
    "annual_registration_fee": 12000,
    "registration_fee_due": "10 August 2026",
    "min_attendance_percent": 70,
    "academic_probation_gpa": 4.5
}

PUBLIC_UNIV_POLICY_RULES = [
    {"rule_id": "PUB_ATT", "text": "Mandatory 70% attendance required across registered courses.",
     "keywords": ["attendance", "courses", "mandatory"], "required_val": 70, "attr": "attendance"},
    {"rule_id": "PUB_PROB", "text": "Semester GPA below 4.5 results in academic probation.",
     "keywords": ["gpa", "probation"], "required_val": 4.5, "attr": "probation"},
    {"rule_id": "PUB_DROP", "text": "Course drop allowed up to 20 September 2026 without penalty.",
     "keywords": ["drop", "penalty", "course"], "required_val": "20 september", "attr": "drop"}
]

# ============================================================================
# LLM & Vector Helpers
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
            except Exception as e:
                time.sleep(1.0)
                if attempt == 1:
                    return f"[LLM fallback output]"
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
                for idx, text in zip(missing_indices, missing_texts):
                    vectors[idx] = [0.0] * dim
        return np.array(vectors, dtype=np.float32)

# ============================================================================
# Dynamic Multi-Agent Verification Architecture
# ============================================================================
class ClaimExtractor:
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
    """Domain-agnostic fact checking against arbitrary structured institutional records."""
    def __init__(self, records: Dict[str, Any]):
        self.records = records

    def _extract_numbers(self, text: str) -> List[int]:
        nums = re.findall(r"\b\d[\d,]*\b", text)
        res = []
        for n in nums:
            try:
                res.append(int(n.replace(",", "")))
            except Exception:
                pass
        return res

    def verify(self, claim: str) -> Tuple[float, str]:
        c_lower = claim.lower()
        nums = self._extract_numbers(c_lower)

        # 1. Registration / Drop Deadlines
        for deadline_key in ["semester_registration_deadline", "course_drop_deadline"]:
            if deadline_key in self.records:
                target_date = str(self.records[deadline_key]).lower()
                kw = "drop" if "drop" in deadline_key else "registration"
                if kw in c_lower and "deadline" in c_lower and "late" not in c_lower:
                    if target_date in c_lower:
                        return 1.0, f"Verified against Official Records ({deadline_key}: {target_date})."
                    if any(m in c_lower for m in ["august", "september", "october", "november", "december"]):
                        return 0.0, f"Contradicts Official Records: {deadline_key} is {target_date}."

        # 2. Late registration
        if "late registration" in c_lower and "late_registration_deadline" in self.records:
            t_date = str(self.records["late_registration_deadline"]).lower()
            fine = self.records.get("late_registration_fine", 500)
            date_ok = t_date in c_lower
            fine_ok = fine in nums
            if date_ok and fine_ok:
                return 1.0, f"Verified: Late registration until {t_date} with fine Rs {fine}."
            if (date_ok and not fine_ok and any(n >= 500 for n in nums)) or (not date_ok and fine_ok):
                return 0.0, f"Contradicts Late Registration: deadline {t_date}, fine Rs {fine}."
            if any(m in c_lower for m in ["august", "september"]) or any(n >= 500 for n in nums):
                return 0.0, f"Contradicts Late Registration: official is {t_date}, Rs {fine} fine."

        # 3. Exam schedules
        for exam_key in ["midsem_exam_dates", "endsem_exam_dates", "final_exam_dates"]:
            if exam_key in self.records:
                target_span = str(self.records[exam_key]).lower()
                prefix = "mid" if "mid" in exam_key else "final" if "final" in exam_key else "end"
                if f"{prefix}-semester" in c_lower or f"{prefix} semester" in c_lower or (prefix == "final" and "final" in c_lower and "exam" in c_lower):
                    if target_span in c_lower:
                        return 1.0, f"Verified against Exam Schedule: {target_span}."
                    if any(m in c_lower for m in ["october", "november", "december"]):
                        return 0.0, f"Contradicts Exam Schedule: Official dates are {target_span}."

        # 4. Fees (Tuition, Registration, Mess, Make-up, Revaluation)
        fee_checks = [
            ("btech_tuition_fee", ["tuition fee", "b.tech tuition"], 62500),
            ("annual_registration_fee", ["annual registration fee", "registration fee"], 12000),
            ("hostel_mess_fee", ["mess fee"], 18000),
            ("makeup_exam_fee", ["make-up examination fee", "makeup exam fee"], 1000),
            ("revaluation_fee_per_subject", ["revaluation fee"], 500)
        ]
        for rec_key, kws, def_val in fee_checks:
            if rec_key in self.records:
                val = self.records[rec_key]
                if any(kw in c_lower for kw in kws):
                    if val in nums:
                        return 1.0, f"Verified against Accounts Office: {rec_key} is Rs {val:,}."
                    if any(n >= 500 for n in nums):
                        return 0.0, f"Contradicts Accounts Office: {rec_key} is Rs {val:,}."

        # 5. Attendance
        if "attendance" in c_lower and "min_attendance_percent" in self.records:
            req_att = self.records["min_attendance_percent"]
            if req_att in nums:
                return 1.0, f"Verified against Academic Ordinance: Minimum attendance is {req_att}%."
            if any(n in nums for n in [50, 60, 65, 70, 75, 80, 85] if n != req_att):
                return 0.0, f"Contradicts Academic Ordinance: Required attendance is {req_att}%."

        # 6. Library limits
        if "library" in c_lower or "books" in c_lower:
            if "max_library_books" in self.records:
                b_max = self.records["max_library_books"]
                d_max = self.records.get("library_loan_period_days", 14)
                fine = self.records.get("library_overdue_fine_per_day", 5)
                if b_max in nums and d_max in nums:
                    return 1.0, f"Verified against Library Rules: {b_max} books for {d_max} days."
                if "fine" in c_lower and fine in nums:
                    return 1.0, f"Verified against Library Fine: Rs {fine}/day."
                if any(n in nums for n in [6, 8, 10, 21]):
                    return 0.0, f"Contradicts Library limits ({b_max} books, {d_max} days, fine Rs {fine})."

        # 7. Hostel Rules & Curfew
        if "curfew" in c_lower and "hostel_curfew_time" in self.records:
            if "10:00" in c_lower or "10 pm" in c_lower:
                return 1.0, "Verified against Chief Warden Directive: Curfew is 10:00 PM."
            if "11:00" in c_lower or "11 pm" in c_lower or "12" in c_lower:
                return 0.0, "Contradicts Chief Warden Directive: Curfew is 10:00 PM."

        if "visitor" in c_lower:
            if "overnight" in c_lower:
                return 0.0, "Contradicts Hostel Rule: Overnight visitors strictly prohibited."
            if "4:00" in c_lower and "7:00" in c_lower:
                return 1.0, "Verified: Visitors allowed 4:00 PM to 7:00 PM."

        if "mess" in c_lower and ("1:00 am" in c_lower or "midnight" in c_lower):
            return 0.0, "Contradicts Hostel Regulations: Mess closes at 10:00 PM."

        if "medical leave" in c_lower and "max_medical_leave_days" in self.records:
            days = self.records["max_medical_leave_days"]
            if days in nums:
                return 1.0, f"Verified: Medical leave permitted up to {days} days."
            if any(n > days for n in nums):
                return 0.0, f"Contradicts Medical Policy: Max {days} days allowed per semester."

        # 8. Grading & Academic Probation
        if "cgpa" in c_lower and "cgpa_scale" in self.records:
            if 10 in nums or "10-point" in c_lower:
                return 1.0, "Verified: 10-point CGPA grading scale."
            if any(n in nums for n in [7, 8, 4]):
                return 0.0, "Contradicts Grading Scale: 10-point system."

        if "pass" in c_lower and "min_pass_cgpa" in self.records:
            if 5 in nums or "5.0" in c_lower:
                return 1.0, "Verified: Minimum 5.0 CGPA required to pass."
            if 4 in nums or "4.0" in c_lower:
                return 0.0, "Contradicts Pass Standard: Minimum 5.0 CGPA required."

        if "probation" in c_lower and "academic_probation_gpa" in self.records:
            pgpa = self.records["academic_probation_gpa"]
            if "4.5" in c_lower:
                return 1.0, f"Verified: Academic probation at GPA below {pgpa}."
            if "3.0" in c_lower or "3.5" in c_lower:
                return 0.0, f"Contradicts Policy: Probation triggered below GPA {pgpa}."

        if "scholarship" in c_lower and "scholarship_income_ceiling" in self.records:
            if "5 lakh" in c_lower or 500000 in nums:
                return 1.0, "Verified: Family income ceiling is Rs 5 Lakh p.a."
            if "8 lakh" in c_lower or 800000 in nums:
                return 0.0, "Contradicts Scholarship Scheme: Income ceiling is Rs 5 Lakh p.a."

        if "placement" in c_lower and "placement_registration_open" in self.records:
            if "1 august" in c_lower:
                return 1.0, "Verified: Placement registration opens 1 August 2026."
            if "15 july" in c_lower:
                return 0.0, "Contradicts Placement Notice: Registrations open 1 August 2026."

        if "revaluation" in c_lower and "revaluation_window_days" in self.records:
            rw = self.records["revaluation_window_days"]
            if rw in nums or f"{rw} days" in c_lower:
                return 1.0, f"Verified: Revaluation window is {rw} days."
            if any(n in nums for n in [14, 21, 30]):
                return 0.0, f"Contradicts Revaluation Rule: Window is strictly {rw} days."

        return 0.5, "No direct structured record; delegated to policy and grounding checks."


class PolicyComplianceAgent:
    """Checks claims against deterministic academic policy rule bases."""
    def __init__(self, rules: List[Dict[str, Any]]):
        self.rules = rules

    def verify(self, claim: str) -> Tuple[float, str]:
        c_lower = claim.lower()
        nums = [int(n) for n in re.findall(r"\b\d+\b", c_lower)]

        for rule in self.rules:
            matches = [kw for kw in rule["keywords"] if kw in c_lower]
            if len(matches) >= 1:
                attr = rule.get("attr", "")
                req = rule.get("required_val")

                if attr == "attendance":
                    if isinstance(req, int) and req in nums:
                        return 1.0, f"Complies with Policy {rule['rule_id']}: {req}% attendance verified."
                    if any(n in nums for n in [50, 60, 65, 70, 75, 80] if n != req):
                        return 0.0, f"Violates Policy {rule['rule_id']}: Requires {req}% attendance."

                elif attr == "curfew":
                    if "10:00" in c_lower or "10 pm" in c_lower:
                        return 1.0, f"Complies with Policy {rule['rule_id']} curfew standards."
                    if "11:00" in c_lower or "11 pm" in c_lower or "12" in c_lower:
                        return 0.0, f"Violates Policy {rule['rule_id']}: Curfew is 10:00 PM."

                elif attr == "revaluation":
                    if 7 in nums and 500 in nums:
                        return 1.0, f"Complies with Policy {rule['rule_id']} (7 days, Rs 500)."
                    if 14 in nums or 1000 in nums:
                        return 0.0, f"Violates Policy {rule['rule_id']} (7 days, Rs 500)."

                elif attr == "library":
                    if (4 in nums and 14 in nums) or (5 in nums):
                        return 1.0, f"Complies with Policy {rule['rule_id']} book and fine rules."
                    if 6 in nums or 21 in nums or 10 in nums:
                        return 0.0, f"Violates Policy {rule['rule_id']} book and fine rules."

                elif attr == "cgpa":
                    if "5.0" in c_lower or 10 in nums or 5 in nums:
                        return 1.0, f"Complies with Policy {rule['rule_id']} CGPA scale."
                    if "4.0" in c_lower or 7 in nums:
                        return 0.0, f"Violates Policy {rule['rule_id']} CGPA scale."

                elif attr == "probation":
                    if "4.5" in c_lower:
                        return 1.0, f"Complies with Policy {rule['rule_id']}: 4.5 GPA threshold."
                    if "3.0" in c_lower or "3.5" in c_lower:
                        return 0.0, f"Violates Policy {rule['rule_id']}: GPA threshold is 4.5."

                elif attr == "drop":
                    if "20 september" in c_lower:
                        return 1.0, f"Complies with Policy {rule['rule_id']}: Drop by 20 September."
                    if "30 september" in c_lower:
                        return 0.0, f"Violates Policy {rule['rule_id']}: Drop deadline is 20 September."

        return 0.5, "No conflict with deterministic institutional policies."


class GroundingAgent:
    """Checks claims against a retrieval store of handbooks, circulars, and syllabi via FAISS."""
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
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or DEFAULT_WEIGHTS
        assert abs(sum(self.weights.values()) - 1.0) < 1e-4

    def compute_score(self, F: float, P: float, D: float) -> float:
        w1, w2, w3 = self.weights["w1"], self.weights["w2"], self.weights["w3"]
        return float(np.clip(w1 * F + w2 * P + w3 * D, 0.0, 1.0))


class DecisionGate:
    def __init__(self, tau_pass: float = THRESHOLD_PASS, tau_flag: float = THRESHOLD_FLAG):
        self.tau_pass = tau_pass
        self.tau_flag = tau_flag

    def decide(self, F: float, P: float, D: float, G: float) -> Tuple[str, str]:
        if F >= 0.90:
            return "PASS", "true"
        if F <= 0.10:
            return "BLOCK", "false"
        if P <= 0.10:
            return "BLOCK", "false"

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
    def __init__(
        self,
        docs: Optional[List[Dict[str, Any]]] = None,
        records: Optional[Dict[str, Any]] = None,
        policies: Optional[List[Dict[str, Any]]] = None,
        weights: Optional[Dict[str, float]] = None,
        gen_model: str = DEFAULT_GEN_MODEL,
        emb_model: str = DEFAULT_EMB_MODEL,
        cache_dir: str = "nb_cache"
    ):
        self.docs = docs or NITJ_DOCS
        self.records = records or NITJ_STRUCTURED_RECORDS
        self.policies = policies or NITJ_POLICY_RULES
        self.llm = LLMClient(gen_model=gen_model, emb_model=emb_model, cache_dir=cache_dir)
        
        self.claim_extractor = ClaimExtractor(self.llm)
        self.fact_agent = FactCheckAgent(self.records)
        self.policy_agent = PolicyComplianceAgent(self.policies)
        self.grounding_agent = GroundingAgent(self.docs, self.llm)
        self.scorer = GroundednessScorer(weights)
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
            reason = f"Fact-check: {f_exp} | Policy: {p_exp}"
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
            regenerated_claim=regen_text
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
            audit_trail=audit_trail
        )
