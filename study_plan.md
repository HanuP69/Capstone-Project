# 🎓 Comprehensive Capstone Defense & Interview Master Study Plan
## *Multi-Agent Groundedness Verification Pipeline for Institutional Academic Assistants*
**Project:** B.Tech Capstone Project — Department of Computer Science & Engineering, National Institute of Technology Jamshedpur  
**Supervisor:** Dr. Dilip Kumar | **Semester:** 5th Semester, B.Tech CSE  
**Team Members & Rolls:**  
1. Sujal Kumar Yadav (`2024UGCS033`) — Verification Agent Architecture & Orchestration  
2. Tanishk S Pandey (`2024UGCS037`) — LLM Generator & Retrieval (RAG) Layer  
3. Mohit Anand (`2024UGCS072`) — Groundedness Scoring Algorithm & Evaluation  
4. Ankit Sharma (`2024UGCS063`) — Fact-Check & Policy-Compliance Agents, Backend API  
5. Dhirendra Pratap Dindoure (`2024UGCS084`) — Web Dashboard & Results Visualization  
6. Himanshu Gupta (`2024UGCS113`) — Dataset Curation (NITJ + Public), Baselines & Human Evaluation  

---

## 📌 PART 1: The Core Research Problem & Proposed Plan Alignment

### 1.1 Why Does This Problem Exist?
Academic assistants powered by off-the-shelf Large Language Models (LLMs) suffer from **Operational Hallucination**:
- They blend genuinely retrieved procedural advice with plausible-sounding but fabricated deadlines, altered tuition fee figures, or invented hostel regulations.
- In academic and administrative domains, a single hallucination (e.g., claiming a fee can be paid in cash at the security desk, or that attendance is optional) carries severe academic and financial penalties for students.

### 1.2 The Two Fatal Gaps in Existing Literature (Page 1 & 2 of Proposal)
1. **The Technical Gap (Response-Level vs. Claim-Level Verification):**
   - Existing RAG-based systems verify or generate answers at the *full response level*. If a 100-word paragraph has 4 accurate facts and 1 subtle hallucinated date, the entire answer is either passed or rejected.
   - **Our Solution:** **Atomic Claim Decomposition** (`ClaimExtractor`). We break candidate drafts into independent, checkable atomic claims $c_i$ and verify each claim individually.
2. **The Generalizability Gap (Single-Institution vs. Cross-Institution):**
   - Most published educational chatbots evaluate only on their own university's FAQ, making reviewers question whether the method overfits one institution's vocabulary.
   - **Our Solution:** Validated across **two independent datasets**:
     - **Primary:** National Institute of Technology Jamshedpur (NITJSR) — 2,241 legacy PDFs, 18,853 chunks, 6,094 Markdown tables.
     - **Secondary:** Public Central University benchmark + 250 real-world QA pairs from the **MFAQ dataset**.

---

## 🏛️ PART 2: System Architecture — Component-by-Component Deep Dive

```
                             [Student Query]
                                    │
                                    ▼
                      [LLM Response Generator]
                       (Ollama gemma4:e4b draft)
                                    │
                                    ▼
                          [Claim Extractor]
                     (Decomposes into atomic {c_i})
                                    │
               ┌────────────────────┼────────────────────┐
               ▼                    ▼                    ▼
       [Fact-Check Agent]   [Policy Agent]       [Grounding Agent]
      (Structured Records:  (Deterministic Rules: (ChromaDB + BM25 RRF:
         Exam/Fee DB)         Attendance/Curfew)   Circulars/Tables)
          Score: F(c)          Score: P(c)          Score: D(c)
               └────────────────────┬────────────────────┘
                                    │
                                    ▼
                      [Groundedness Scorer G(c)]
                    (MLE Logistic Regression Weights:
                       w1=0.508, w2=0.315, w3=0.177)
                                    │
                                    ▼
                             [Decision Gate]
               ┌────────────────────┼────────────────────┐
               ▼                    ▼                    ▼
            [PASS]               [FLAG]               [BLOCK]
         (G >= 0.72)       (0.50 <= G < 0.72)        (G < 0.50)
         Keep Claim           Add Warning          Regenerate Loop
               │                    │                    │
               ▼                    ▼                    ▼
                               [Final Grounded Answer]
```

### Component 1: LLM Response Generator & Claim Extractor
- **Generator:** Uses local Ollama instance (`gemma4:e4b`) with low temperature ($T=0.1$) to synthesize initial draft responses.
- **Claim Extractor:** Decomposes candidate text into independent atomic propositions.
  - *Example Draft:* "Tuition fee for B.Tech is payable at the start of each semester via SBI Collect, and students can pay in cash at the hostel office."
  - *Decomposed Claims:*
    1. $c_1$: "Tuition fee for B.Tech is payable at the start of each semester via SBI Collect."
    2. $c_2$: "Students can pay B.Tech tuition fee in cash at the hostel office."
  - *Why?* If evaluated together, $c_1$'s high retrieval score masks $c_2$'s falsehood. Evaluating them separately allows passing $c_1$ and blocking $c_2$.

### Component 2: The Tri-Agent Verification Layer
Each claim $c_i$ is evaluated concurrently by three heterogeneous agents:

#### 1. Fact-Check Agent ($F(c) \in [0, 1]$)
- **Role:** Checks structured key-value institutional records (canonical fee tables, exam reporting dates, caution deposit terms).
- **Logic:** Extracts entities and numbers from the claim. If exact match with canonical record $\implies F=1.0$. If direct numerical contradiction (e.g., claiming revaluation is Rs 1,500 when official record is Rs 500) $\implies F=0.0$. If no structured record exists $\implies F=0.5$ (neutral prior).

#### 2. Policy-Compliance Agent ($P(c) \in [0, 1]$)
- **Role:** Deterministic symbolic rule engine enforcing institutional ordinances.
- **Core Rules Codified:**
  - `POL_ATT_75`: 75% minimum mandatory attendance. Flags "no minimum", "zero attendance", "40%", "50%" as direct violations ($P=0.0$).
  - `POL_CUR_10PM`: Strict 10:00 PM hostel curfew. Flags "zero curfew", "unrestricted 24 hours", "unauthorized overnight guests" ($P=0.0$).
  - `POL_REV_7D_500R`: Revaluation within 7 days of result with Rs 500 fee.
  - `POL_PASS_CGPA`: 10-point CGPA scale with 5.0 minimum passing threshold.
  - `POL_FEE_PORTAL`: Online fee payment exclusively via SBI Collect / QFix. Cash payments flagged as violation ($P=0.0$).

#### 3. Grounding Agent ($D(c) \in [0, 1]$)
- **Role:** Unstructured and tabular semantic retrieval over official university circulars.
- **Persistent Vector Database:** **ChromaDB** (`./nitjsr_chroma_db`) with HNSW indexing in Cosine Space (`metadata={"hnsw:space": "cosine"}`).
- **Hybrid Reciprocal Rank Fusion (RRF):**
  - Dense Vector Search (`nomic-embed-text`, 768-dim) captures semantic intent.
  - Sparse Lexical Search (BM25Okapi) captures exact keyword/number matches ("Notice 194", "Rs 62,500").
  - RRF Formula:
    $$\text{RRF}(d) = \frac{w_{\text{dense}}}{k_{\text{rrf}} + \text{rank}_{\text{dense}}(d) + 1} + \frac{w_{\text{bm25}}}{k_{\text{rrf}} + \text{rank}_{\text{bm25}}(d) + 1}$$
    Calibrated parameters: $k_{\text{rrf}} = 20, w_{\text{dense}} = 2.0, w_{\text{bm25}} = 1.0$.
  - Normalized Evidence Score:
    $$D(c) = \frac{\text{sim}_{\text{top}} + 1.0}{2.0} \in [0, 1]$$

---

## 📐 PART 3: Mathematical Formulations & Calibration

### 3.1 The Groundedness Scoring Function
The unified confidence score for each claim is defined as:
$$G(c) = w_1 \cdot F(c) + w_2 \cdot P(c) + w_3 \cdot D(c)$$
subject to:
$$\sum_{i=1}^3 w_i = 1, \quad w_i \ge 0$$

### 3.2 Weight Calibration: Heuristic vs. MLE Logistic Regression Meta-Learner
- **Default Heuristic Weights:** $w_1 = 0.40, w_2 = 0.35, w_3 = 0.25$.
- **Statistically Learned MLE Weights (`meta_learner_grounding.py`):**
  We framed groundedness scoring as a Maximum Likelihood Estimation (MLE) problem using a Logistic Regression Meta-Learner trained on labeled claim triples $(F, P, D)$:
  $$P(y = \text{True} \mid F, P, D) = \sigma(\beta_0 + \beta_1 F + \beta_2 P + \beta_3 D)$$
  Normalizing the positive learned coefficients yields:
  $$w_1 = \frac{\beta_1}{\sum \beta_i} = 0.508, \quad w_2 = \frac{\beta_2}{\sum \beta_i} = 0.315, \quad w_3 = \frac{\beta_3}{\sum \beta_i} = 0.177$$
- **Insight:** The Fact-Check Agent receives the highest weight ($0.508$) because structured database records provide zero-variance ground truth, whereas Grounding ($0.177$) carries higher variance due to semantic retrieval noise.

### 3.3 Decision Gate Calibration
- **Thresholds:**
  $$\tau_{\text{pass}} = 0.72, \quad \tau_{\text{flag}} = 0.50$$
- **Authoritative Priority Overrides:**
  1. **Fact Override:** If $F \ge 0.90 \implies \text{PASS}$ (`true`). If $F \le 0.10 \implies \text{BLOCK}$ (`false`).
  2. **Policy Override:** If $P \le 0.10 \implies \text{BLOCK}$ (`false`). If $P \ge 0.90 \implies \text{PASS}$ (`true`).
  3. **Neutral Prior Override (for open document corpora):** When $F = 0.5, P = 0.5$, decisions are driven by retrieval evidence $D$:
     $$\text{Decision} = \begin{cases} \text{PASS} (\text{true}) & \text{if } D \ge 0.84 \\ \text{FLAG} (\text{unverifiable}) & \text{if } 0.70 \le D < 0.84 \\ \text{BLOCK} (\text{false}) & \text{if } D < 0.70 \end{cases}$$

---

## 🗄️ PART 4: Real NIT Jamshedpur Data Engineering Milestone

### 4.1 Corpus Audit (`scan_documents_metadata.py`)
- Total Archive: **2,241 PDF documents (2.52 GB)**.
- **Digital Searchable PDFs:** 1,056 files (47.1%).
- **Scanned Image PDFs:** 1,185 files (52.9%) — deferred to OCR to maintain high vector fidelity.

### 4.2 Structure-Aware Ingestion & PyMuPDF Tables (`nitjsr_ingestion_engine.py`)
- **The Problem:** Standard recursive text splitters slice table rows, disconnecting fee amounts from fee heads (e.g., tuition fee Rs 62,500 gets placed in chunk 1, while header "Tuition Fee" is in chunk 2).
- **Our Solution:**
  1. PyMuPDF `find_tables()` detects coordinate borders and converts grids into **GitHub-flavored Markdown tables**.
  2. Tables are stored as atomic, unsliced chunks.
- **Database Schema (`nitjsr_documents.db` — 35.04 MB):**
  - **18,853 total chunks:** 6,094 Markdown tables + 12,759 narrative prose chunks.
  - Metadata: `chunk_id`, `document_name`, `category`, `issue_date`, `academic_session`, `ref_number`, `page_number`, `chunk_type`, `content`.

### 4.3 Temporal System Prompt Grounding & Recency Decay
- Academic circulars span 2022 to 2024. Stale notices (e.g. Autumn 2022 fee schedule) conflict with newer circulars.
- **Dynamic Context Injection:** System prompt automatically injects:
  `Current Wall-Clock Time: Monday, 23 November 2026 | Active Session: 2026-2027 Autumn`
- **Temporal Weight Kernel:**
  $$\omega_{\text{temporal}}(d) = \exp(-\lambda \cdot \Delta t(d))$$
  Ensures active circulars supersede archived historical notices.

---

## 📊 PART 5: Empirical Results & Verification Benchmarks

### 5.1 Supervised 4-Condition Benchmark (100 Real NITJSR Claims)
Benchmarked across 5 domains: Exam Dates, Fee Deadlines, Hostel Rules, Circulars, Syllabus Queries.

| Experimental Condition | Accuracy | Precision | Recall | Macro-F1 | Missed Falsehood Rate |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Baseline A: Plain LLM (Ungrounded)** | 40.0% | 0.400 | 1.000 | 0.571 | 100.0% |
| **Baseline B: Standard Dense RAG** | 69.0% | 0.563 | 1.000 | 0.721 | 77.5% |
| **Baseline C: Self-Consistency ($T=3$)** | 60.0% | 0.500 | 1.000 | 0.667 | 100.0% |
| **Proposed: Multi-Agent Pipeline (Ours)** | **91.0%** | **0.867** | **0.975** | **0.918** | **12.5%** |

#### Why Did Dense RAG Fail?
- Standard Dense RAG had a **77.5% Missed Falsehood Rate**.
- *Reason:* Plausible falsehoods ("B.Tech tuition fee can be paid in cash at security desk") retrieve the genuine fee notice. The top-1 vector cosine similarity remains high ($\sim 0.82$). Dense RAG assumes high similarity means truth! Without a symbolic Policy/Fact agent, Dense RAG cannot detect that "cash" violates the SBI Collect rule.

### 5.2 Unsupervised Reference-Free Evaluation (RAG Triad on 30 Real Campus Queries)
Evaluates uncurated administrative queries where human gold answers do not exist:
1. **Faithfulness / Groundedness ($0.936$ / 93.6%):** 93.6% of generated response statements are logically entailed by retrieved official circulars.
2. **Context Relevance / Precision ($0.906$ / 90.6%):** Hybrid RRF eliminates noisy distractors, ensuring retrieved chunks directly address the student query.
3. **Answer Relevance ($0.920$ / 92.0%):** Generated answers directly and concisely address the user prompt without tangent confabulation.

---

## 🎯 PART 6: Top 10 High-Stakes Interview / Viva Questions & Model Answers

### Q1: "Why use a Multi-Agent architecture? Why not just fine-tune Gemma or write a better RAG prompt?"
**Model Answer:**
> *"Fine-tuning adjusts parametric weights, but institutional data (exam dates, fee amounts, circulars) changes on a weekly basis. Re-training an LLM weekly is computationally impractical and does not guarantee factual zero-hallucination. Prompt engineering alone fails because LLMs are stochastic token predictors lacking verification capability; our benchmark showed Dense RAG still missed 77.5% of plausible falsehoods.  
> Our Multi-Agent approach decouples generation from deterministic verification: the Fact Agent queries ground-truth key-value tables, the Policy Agent enforces non-negotiable boolean rules (attendance $\ge 75\%$), and the Grounding Agent retrieves semantic evidence from ChromaDB. Combining them yields an auditable, transparent verification decision."*

---

### Q2: "How did you arrive at the weights $w_1=0.508, w_2=0.315, w_3=0.177$? Why not equal weights $0.33/0.33/0.33$?"
**Model Answer:**
> *"We started with an intuitive heuristic ($0.40/0.35/0.25$), but to make it scientifically rigorous, we formulated weight calibration as a Maximum Likelihood Estimation problem. We trained a Logistic Regression meta-learner on ground-truth labeled claim triples $(F, P, D)$.  
> The empirical weights reflect the signal reliability of each agent: Fact-Check records are zero-variance structured facts, giving it the highest weight ($0.508$). Policy checks are deterministic boolean rules ($0.315$). Grounding ($0.177$) carries semantic retrieval variance. We proved that these weights transfer to Public University and MFAQ benchmarks without re-tuning, confirming cross-institution generalization."*

---

### Q3: "What is the difference between FAISS and ChromaDB? Why did you migrate to ChromaDB?"
**Model Answer:**
> *"FAISS is a high-performance vector search library/index in C++ that operates in-memory on raw embedding arrays. However, it lacks native database capabilities like persistent storage, collection management, and metadata filtering.  
> We migrated to ChromaDB because it is a true persistent Vector Database. It maintains an on-disk HNSW index in Cosine Space (`nitjsr_chroma_db/chroma.sqlite3`) and allows relational metadata filtering (e.g. `where={'category': 'Fee Deadlines', 'chunk_type': 'table'}`). This allows us to prune irrelevant documents at the database level before running Hybrid Reciprocal Rank Fusion."*

---

### Q4: "Why did you use Reciprocal Rank Fusion (RRF) instead of simply adding BM25 and Dense scores?"
**Model Answer:**
> *"BM25 produces unbounded lexical scores (typically $0$ to $40+$ depending on query length and term frequency), whereas dense vector search produces cosine similarities bounded between $-1$ and $+1$. Normalizing them directly (e.g. min-max scaling) is fragile because score distributions vary drastically across queries.  
> Reciprocal Rank Fusion (RRF) bypasses score calibration entirely by operating purely on rank positions:
> $$\text{RRF}(d) = \sum_{m \in \{\text{dense}, \text{bm25}\}} \frac{w_m}{k + \text{rank}_m(d) + 1}$$
> Dense search captures broad semantic concepts, while BM25 captures exact numbers ('Rs 62,500', 'Notice 194'). On MFAQ, our calibrated RRF ($k=20, w_d=2.0, w_b=1.0$) outperformed pure BM25 by $+12.4\%$ Recall@1 and pure Dense by $+8.0\%$ Recall@1."*

---

### Q5: "How do you handle tables in PDFs? Most chunking strategies ruin tables."
**Model Answer:**
> *"Standard character or word splitters slice text arbitrarily, splitting columns across chunk boundaries so a fee value is separated from its header row.  
> In `nitjsr_ingestion_engine.py`, we implemented structure-aware extraction using PyMuPDF's `find_tables()` API. It detects coordinate vector lines and cell bounding boxes, converting table grids into clean GitHub-flavored Markdown tables. Each table is stored as an atomic, unsliced chunk in SQLite and ChromaDB. Because Markdown preserves row-column topological relations, the embedding model (`nomic-embed-text`) and BM25 can match both the row title ('General Category Tuition Fee') and the value ('Rs 62,500') within the exact same context window."*

---

### Q6: "How does your Decision Gate prevent False Positives (rejecting true statements) and False Negatives (accepting false statements)?"
**Model Answer:**
> *"We designed a two-tiered decision mechanism:
> 1. General Thresholding: $G(c) \ge 0.72 \implies \text{PASS}$, $0.50 \le G(c) < 0.72 \implies \text{FLAG}$, $G(c) < 0.50 \implies \text{BLOCK}$.
> 2. Authoritative Overrides: If the Fact-Check agent confirms an official entity ($F \ge 0.90$), it overrides minor retrieval noise and passes. Conversely, if a claim violates an institutional policy (e.g. $P \le 0.10$ for 40% attendance), it is blocked immediately regardless of how high retrieval cosine similarity might be.  
> On our 100-claim NITJSR benchmark, this dual-layer gate achieved 86.7% precision, 97.5% recall, and dropped missed falsehoods from 77.5% (Dense RAG) down to 12.5%."*

---

### Q7: "Tell me about a difficult technical bug you faced and how you resolved it."
**Model Answer:**
> *"During our initial evaluation on the 18,853 chunks, our Multi-Agent Pipeline accuracy unexpectedly dropped to 45% and every claim was being flagged or blocked.  
> Upon inspecting `nitjsr_audit_trail.csv`, I discovered that the Grounding score $D(c)$ was exactly $0.500$ for all 100 claims. Tracing into `LLMClient.embed_texts()`, we found that Ollama's local HTTP server was returning an HTTP 400 'Connection Refused' error because we were sending 10,957 document strings in a single POST request payload. The generic exception handler was catching this error silently and falling back to zero-vectors (`[0.0] * 768`). Cosine similarity against zero vectors returned $0.0$, which normalized to $(0 + 1) / 2 = 0.500$.  
> I re-architected `embed_texts()` to process embeddings in micro-batches of 64 with disk caching in `emb_cache.json`. This fixed vector indexing completely, and pipeline accuracy immediately jumped to 91.0%."*

---

### Q8: "Why do you have both a Supervised Benchmark and an Unsupervised RAG Triad?"
**Model Answer:**
> *"In real institutional deployments, administrators do not have human annotators labeling thousands of legacy PDF notices with True/False tags.  
> Therefore, we evaluated along two orthogonal axes:
> 1. Supervised Benchmark (100 claims): Balanced test set (40 True, 40 False, 20 Unverifiable) to evaluate rigorous classification metrics (Accuracy 91.0%, Macro-F1 0.918, ROC AUC 0.942).
> 2. Unsupervised RAG Triad (30 real student queries): Reference-free evaluation measuring Faithfulness (93.6%), Context Relevance (90.6%), and Answer Relevance (92.0%). This allows an institution to continuously monitor answer quality on uncurated live query streams without manual labeling."*

---

### Q9: "What did each team member contribute according to the proposal?"
**Model Answer:**
> *"Our project strictly aligns with the division of responsibilities on Page 1 of the Capstone Proposal:
> - **Sujal Kumar Yadav:** Designed the Multi-Agent Verification Architecture and orchestrator pipeline in `verification_pipeline.py`.
> - **Tanishk S Pandey:** Engineered the LLM response generator and the hybrid retrieval layer combining ChromaDB, FAISS, and BM25Okapi via RRF.
> - **Mohit Anand:** Formulated the Groundedness Scoring algorithm $G(c) = w_1 F + w_2 P + w_3 D$ and trained the Logistic Regression Meta-Learner.
> - **Ankit Sharma:** Developed the Fact-Check & Policy-Compliance agents and built the production FastAPI backend microservice (`app.py`).
> - **Dhirendra Pratap Dindoure:** Built the interactive Web Dashboard (`dashboard.html`) and generated all 6 publication charts.
> - **Himanshu Gupta:** Curated the NITJ primary dataset (2,241 PDFs), secondary public dataset, MFAQ benchmark, and conducted the human inter-annotator agreement study (Cohen's Kappa = 0.887)."*

---

### Q10: "What happens when a claim is BLOCKED? Explain the closed-loop feedback."
**Model Answer:**
> *"Traditional hallucination detectors merely detect errors and drop the response or say 'I don't know'.  
> Our architecture implements a closed-loop **Regeneration Engine** (`RegenerationEngine`). When the Decision Gate flags `BLOCK`, it packages:
> 1. The original hallucinated claim,
> 2. The retrieved verified context from ChromaDB, and
> 3. The exact agent failure explanation (e.g., 'Violates POL_CUR_10PM: Curfew is strictly 10:00 PM').
> It prompts the LLM to rewrite the statement into a cautious, verified factual sentence. For instance, 'Visitors can stay overnight in hostels' is automatically regenerated into: 'As per Chief Warden regulations, overnight stay of unauthorized guests in student hostel rooms is strictly prohibited.' This closes the loop from detection directly to correction."*

---

## ⚡ PART 7: 2-Minute Elevator Pitch / Defense Opening

> *"Good morning respected evaluators and supervisor Dr. Dilip Kumar.  
> Today we present our B.Tech Capstone Project: **A Multi-Agent Groundedness Verification Pipeline for Institutional Academic Assistants**.  
> 
> Deploying LLMs as university assistants is dangerous because models routinely hallucinate critical operational facts—inventing exam dates, altering fee amounts, or misrepresenting attendance rules. Standard Dense RAG fails to fix this because plausible falsehoods share high cosine similarity with genuine notices, missing 77.5% of falsehoods in our tests.  
> 
> To solve this, we introduce three technical novelties:  
> 1. **Atomic Claim-Level Verification:** We decompose candidate answers into atomic statements and verify each individually.  
> 2. **Tri-Agent Heterogeneous Verification:** We verify claims concurrently against structured database records (Fact Agent), deterministic academic ordinances (Policy Agent), and a persistent **ChromaDB** vector store with BM25 Hybrid Reciprocal Rank Fusion (Grounding Agent).  
> 3. **Statistically Calibrated Scoring & Closed-Loop Regeneration:** We fuse agent signals using MLE Logistic Regression weights ($w_1=0.508, w_2=0.315, w_3=0.177$) into a Decision Gate that passes, flags, or automatically rewrites ungrounded statements.  
> 
> We validated our system on the real legacy archives of NIT Jamshedpur—ingesting **2,241 PDFs into 18,853 chunks with 6,094 Markdown-preserved tables**. Our proposed pipeline achieved **91.0% accuracy, a 0.918 Macro-F1 score, and eliminated 87.5% of baseline missed falsehoods**, while maintaining **93.6% Faithfulness** on real student campus queries.  
> The system is fully deployed with an interactive Web Dashboard and a production FastAPI microservice. Thank you, and we welcome your questions."*
