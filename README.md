# Multi-Agent Groundedness Verification Pipeline for Institutional Academic Assistants

> **A Case Study on NIT Jamshedpur's Academic & Administrative Query Assistant, Validated Across Institutions**  
> *B.Tech Capstone Project — Department of Computer Science & Engineering, National Institute of Technology Jamshedpur*  
> **Supervisor:** Dr. Dilip Kumar  

> [!NOTE]
> **Dataset Status — [Synthetic Benchmark]:**  
> Current empirical evaluations in this repository are conducted on **[Synthetic Benchmarks]** (150 total claims across institutions). These datasets were curated from published NIT Jamshedpur academic ordinances, fee schedules, examination timetables, and hostel rules to create controlled, ground-truth labeled test sets (True, Controlled False Hallucinations, and Out-of-KB Unverifiable). Ingestion and parsing of live raw institutional PDF circulars from `nitjsr.ac.in` is designated as the subsequent data engineering milestone.

---

## 📌 Abstract & Overview

Large Language Models (LLMs) deployed as academic assistants routinely hallucinate critical operational facts—incorrect exam dates, altered fee deadlines, and fabricated institutional policies. This project presents a **domain-agnostic, multi-agent groundedness verification pipeline** that:
1. Decomposes candidate LLM answers into atomic, checkable claims.
2. Evaluates each claim in parallel across three specialized verification agents:
   - **Fact-Check Agent**: Checks structured key-value institutional records (exam schedules, fee tables, deadlines). Tagged as `[Synthetic Record]` in benchmark mode.
   - **Policy-Compliance Agent**: Validates claims against deterministic academic policy rule bases (attendance minimums, curfew, grading scales, probation). Tagged as `[Synthetic Policy]`.
   - **Grounding Agent**: Performs dense vector retrieval over circulars, handbooks, and ordinances via FAISS and `nomic-embed-text`. Tagged as `[Synthetic Context]`.
3. Computes a unified groundedness score:
   $$G(c) = w_1 \cdot F(c) + w_2 \cdot P(c) + w_3 \cdot D(c)$$
4. Routes each claim through a **Decision Gate** (`PASS`, `FLAG`, `BLOCK`), triggering an automated **Regeneration Feedback Loop** that rewrites ungrounded or false statements into cautious, grounded facts using verified institutional context.
5. Demonstrates cross-institution weight transfer across NIT Jamshedpur and a secondary Public University corpus without re-tuning.

---

## 🏛️ System Architecture

```mermaid
flowchart TD
    Q["Student Query"] --> GEN["LLM Response Generator (Drafts Answer)"]
    GEN --> EXT["Atomic Claim Extractor"]
    
    subgraph MultiAgent ["Parallel Verification Layer"]
        EXT --> FC["Fact-Check Agent ([Synthetic Record] KV Records)"]
        EXT --> PC["Policy-Compliance Agent ([Synthetic Policy] Rules)"]
        EXT --> GA["Grounding Agent ([Synthetic Context] FAISS Vector Store)"]
    end
    
    FC --> SC["Groundedness Scoring Engine G(c) = w1*F + w2*P + w3*D"]
    PC --> SC
    GA --> SC
    
    SC --> GATE{"Decision Gate"}
    GATE -- "G >= 0.72 (Grounded)" --> PASS["PASS (Keep Claim)"]
    GATE -- "0.50 <= G < 0.72" --> FLAG["FLAG (Attach Institutional Caveat)"]
    GATE -- "G < 0.50 or F <= 0.10" --> BLOCK["BLOCK (Regenerate / Remove)"]
    
    BLOCK --> REGEN["Regeneration Feedback Loop (Context-Aware Rewrite)"]
    REGEN --> FINAL["Final Grounded Response (With Citations & Audit Trail)"]
    PASS --> FINAL
    FLAG --> FINAL
```

---

## 📊 Empirical Results on Full Expanded Sample (150 Claims)

The evaluation was executed on the full expanded benchmark sample:
* **Primary (NIT Jamshedpur [Synthetic Benchmark]):** 120 claims (40 True, 40 False, 40 Unverifiable) distributed evenly across 5 proposal categories.
* **Secondary (Public Central University [Synthetic Benchmark]):** 30 claims (10 True, 10 False, 10 Unverifiable).

### 1. Benchmark Comparison Across Conditions
| Condition | Dataset | Sample Size | Accuracy | Precision | Recall | F1 Score | Missed Falsehood Rate |
|---|---|---|---|---|---|---|---|
| **Baseline A: Plain LLM** | NIT Jamshedpur `[Synthetic]` | 120 | 33.3% | 0.111 | 0.333 | 0.167 | 100.0% |
| **Baseline B: RAG-only** | NIT Jamshedpur `[Synthetic]` | 120 | 65.0% | 0.680 | 0.650 | 0.660 | 35.0% |
| **Baseline C: Self-Consistency** | NIT Jamshedpur `[Synthetic]` | 120 | 33.3% | 0.111 | 0.333 | 0.167 | 100.0% |
| **Proposed Full Pipeline** | **NIT Jamshedpur `[Synthetic]`** | **120** | **91.7%** | **0.965** | **0.917** | **0.958** | **2.5%** |
| **Proposed Full Pipeline** | **Public University `[Synthetic]`** | **30** | **100.0%** | **1.000** | **1.000** | **1.000** | **0.0%** |

### 2. Per-Category Hallucination Reduction (Proposal Figure 8)
Comparing Plain LLM vs Proposed Multi-Agent Pipeline on 120 claims:
* **Exam Dates**: $100.0\% \to 0.0\%$ (100% reduction)
* **Fee Deadlines**: $100.0\% \to 0.0\%$ (100% reduction)
* **Hostel Rules**: $100.0\% \to 12.5\%$ (87.5% reduction)
* **Circulars**: $100.0\% \to 0.0\%$ (100% reduction)
* **Syllabus Queries**: $100.0\% \to 0.0\%$ (100% reduction)

### 3. Human Annotator Agreement Study (80 Sampled Claims)
* Sample Size: 80 balanced claims (True, False, Unverifiable)
* **Cohen's Kappa ($\kappa$)**: **0.812** (Target $\ge 0.70$ Met)
* **Inter-Annotator Raw Agreement**: **87.5%**
* **Pipeline-to-Human Consensus Agreement**: **91.2%**

### 4. Real-World Retrieval & QA Stress-Test: MFAQ Benchmark (Full Run: 250 FAQ Pairs)
Evaluated on the English subset of the `clips/mfaq` real-world FAQ corpus to validate the dense RAG layer (`nomic-embed-text` + FAISS):
* **Dense Retrieval Layer (250 Real-World FAQ Pairs)**:
  * **Recall@1**: **78.0%**
  * **Recall@3**: **91.6%**
  * **Recall@5**: **94.8%**
  * **Recall@10**: **96.8%**
  * **Mean Reciprocal Rank (MRR)**: **0.851**
* **QA Answering Layer (Token-F1 on 30 Diverse FAQ Questions)**:
  * **Plain LLM Token-F1**: **0.176**
  * **RAG-Augmented Token-F1**: **0.566** (+221.5% relative improvement over ungrounded generation)

### 5. Multi-Agent Verification Pipeline on MFAQ (All 4 Proposal Conditions)
All 4 conditions were also evaluated on atomic claims extracted from real-world MFAQ queries:
| Condition | Dataset | Accuracy | Precision | Recall | F1 Score | Missed Falsehood Rate | Latency |
|---|---|---|---|---|---|---|---|
| **Baseline A: Plain LLM** | MFAQ | 52.7% | 0.241 | 0.307 | 0.270 | 100.0% | 0.85s |
| **Baseline B: RAG-only** | MFAQ | 70.9% | 0.455 | 0.630 | 0.526 | 100.0% | 1.10s |
| **Baseline C: Self-Consistency** | MFAQ | 54.5% | 0.248 | 0.316 | 0.278 | 100.0% | 2.75s |
| **Proposed Multi-Agent Pipeline** | **MFAQ** | **78.2%** | **0.850** | **0.704** | **0.670** | **83.3%** | **1.45s** |

---

## 🛠️ Repository Structure

```text
├── capstone_verification.ipynb    # Main end-to-end research notebook with outputs & charts
├── verification_pipeline.py       # Modular, zero-hardcoding Python AI verification library
├── run_complete_ai_evaluation.py  # Automated benchmark suite on expanded 150-claim dataset
├── run_full_mfaq_benchmark.py     # Full MFAQ retrieval & QA evaluation runner (250 pairs)
├── run_mfaq_full_agentic_pipeline.py # Full 4-condition agentic verification pipeline on MFAQ
├── app.py                         # Production-grade FastAPI backend microservice
├── primary_audit_trail.csv        # Detailed audit logs for 120 claims (NIT Jamshedpur [Synthetic])
├── secondary_audit_trail.csv      # Cross-institution audit logs for 30 claims (Public University [Synthetic])
├── mfaq_agentic_audit_trail.csv   # Multi-agent verification audit logs on MFAQ claims
├── master_evaluation_metrics.csv  # Benchmark summary metrics
├── mfaq_benchmark_results.csv     # Full MFAQ retrieval & QA metrics table
├── mfaq_4condition_comparison.csv # All 4 conditions comparison table on MFAQ
├── results_table.tex              # Paper-ready LaTeX table export
├── human_eval_report.json         # Inter-annotator agreement & Cohen's Kappa report
├── evaluation_results/            # Generated high-resolution publication charts
│   ├── roc_curve_comparison.png
│   ├── confusion_matrix_groundedness.png
│   ├── per_category_hallucination.png
│   ├── ablation_study_chart.png
│   ├── mfaq_retrieval_performance.png
│   ├── mfaq_qa_comparison.png
│   └── mfaq_4condition_comparison.png
├── Capstone Project_Proposal.pdf  # Approved capstone project proposal
└── README.md                      # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites
* Python 3.10+
* [Ollama](https://ollama.ai) installed and running locally with models:
  ```bash
  ollama pull gemma4:e4b
  ollama pull nomic-embed-text
  ```

### Installation
```bash
pip install faiss-cpu scikit-learn pandas numpy matplotlib seaborn fastapi uvicorn pydantic requests
```

### Running the Full Evaluation Suite
To re-run the benchmark on the expanded 150-claim sample:
```bash
python run_complete_ai_evaluation.py
```

### Running the FastAPI Backend Microservice
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Interactive API documentation:
* Swagger UI: `http://localhost:8000/docs`
* ReDoc: `http://localhost:8000/redoc`

#### Example API Request
```bash
curl -X POST "http://localhost:8000/api/verify-claim" \
     -H "Content-Type: application/json" \
     -d '{"claim": "The semester registration deadline at NIT Jamshedpur is 15 August 2026.", "institution": "NITJ"}'
```

---

## 👥 Capstone Team

| Student Name | Roll Number | Primary Responsibility |
|---|---|---|
| Sujal Kumar Yadav | 2024UGCS033 | Verification Agent Architecture & Orchestration |
| Tanishk S Pandey | 2024UGCS037 | LLM Response Generator & Retrieval (RAG) Layer |
| Mohit Anand | 2024UGCS072 | Groundedness Scoring Algorithm & Evaluation |
| Ankit Sharma | 2024UGCS063 | Fact-Check & Policy-Compliance Agents, Backend API |
| Dhirendra Pratap Dindoure | 2024UGCS084 | Web Dashboard & Results Visualization |
| Himanshu Gupta | 2024UGCS113 | Dataset Curation (NITJ + Public), Baselines & Human Evaluation |
