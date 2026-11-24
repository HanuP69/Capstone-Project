import hashlib
import json
import pickle
import re
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from datetime import date

import chromadb
import numpy as np
import ollama
from rank_bm25 import BM25Okapi

from config import (ANSWER_CTX_CHAR, ANSWER_NUM_PREDICT, ATOMIC_MIN_PART_WORDS, ATOMIC_SPLIT,
                    BEST_SENT_MIN_WORDS, CACHE, DATA, DEFAULT_WEIGHTS, DENSE_WEIGHT,
                    DRAFT_NUM_PREDICT, EMBED_BATCH, EMBED_CHAR_LIMIT, EMB_MODEL,
                    EVIDENCE_CHAR_LIMIT, EVIDENCE_SNIPPET_CHAR, FACT_BLOCK, FACT_PASS,
                    GEN_MODEL, GROUND_PENALTY, JUDGE_LLM_NUM_PREDICT, JUDGE_TOP,
                    LLM_CTX_CHAR, LLM_NUM_CTX, LLM_NUM_PREDICT, LLM_TEMPERATURE,
                    NLI_MODEL, NUMERIC_TOL_DAYS, NUMERIC_TOL_MONEY, NUMERIC_TOL_PCT,
                    OVERLAP_GATE, RECENCY_BASE, RECENCY_DEFAULT_AGE, RECENCY_LAMBDA,
                    RECENCY_SPAN, RECORD_MIN_SUPPORT, REGEN_NUM_PREDICT, REGEN_ROUNDS,
                    REGEN_TEMPERATURE, RERANK_CHAR_LIMIT, RERANK_MODEL, RERANK_TOP,
                    RETRIEVE_K, ROOT, RRF_K, SEED, SPARSE_WEIGHT, TAU_CONTRA, TAU_PASS,
                    UNSUP_CLAIM_MIN_WORDS, WIDE_SENT_LIMIT)
from pipeline_text import (age_years, entities, overlap_terms, parse_dates, parse_money,
                           parse_percent, reference_date, sentences, tokenize)

INDEX_DIR = ROOT / "index"
COLLECTION = "nitj_chunks"
TOPIC_STOP = set("the a an and or of to in on at by for with from is are was were be been being this that "
                 "these those it its as if then than not no yes fee fees amount rs inr rupees".split())


def content_terms(text):
    return {token for token in tokenize(text) if token not in TOPIC_STOP and not token.isdigit()}

@dataclass
class ClaimResult:
    claim: str
    category: str
    F: float
    P: float
    D: float
    C: float
    G: float
    decision: str
    label: str
    fact_note: str
    policy_note: str
    evidence: list = field(default_factory=list)
    regen: str = ""
    judge_backend: str = "nli"

@dataclass
class AnswerResult:
    query: str
    draft: str
    answer: str
    claims: list
    latency: float
    abstained: bool = False
    citations: list = field(default_factory=list)
    evaluated_claims: list = field(default_factory=list)
    retrieval_stats: dict = field(default_factory=dict)


def load_chunks():
    return [json.loads(l) for l in open(DATA / "chunks.jsonl", encoding="utf-8")]


class Cache:
    def __init__(self, path=CACHE / "llm.sqlite"):
        self.con = sqlite3.connect(path, check_same_thread=False)
        self.con.execute("create table if not exists kv (k text primary key, v text, at real)")
        self.con.commit()

    def key(self, model, prompt, temperature, num_predict, seed, num_ctx):
        raw = f"{model}|{temperature}|{num_predict}|{num_ctx}|{seed}|{prompt}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def legacy_key(self, model, prompt, temperature, num_predict, seed):
        raw = f"{model}|{temperature}|{num_predict}|{seed}|{prompt}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def get(self, k):
        row = self.con.execute("select v from kv where k=?", (k,)).fetchone()
        return row[0] if row else None

    def put(self, k, v):
        self.con.execute("insert or replace into kv values (?,?,?)", (k, v, time.time()))
        self.con.commit()


class LLM:
    def __init__(self, model=GEN_MODEL, cache=None):
        self.model = model
        self.cache = cache or Cache()

    def call(self, prompt, temperature=LLM_TEMPERATURE, num_predict=LLM_NUM_PREDICT, seed=SEED):
        k = self.cache.key(self.model, prompt, temperature, num_predict, seed, LLM_NUM_CTX)
        hit = self.cache.get(k)
        if hit is None:
            hit = self.cache.get(self.cache.legacy_key(self.model, prompt, temperature, num_predict, seed))
        if hit is not None:
            return hit
        last = None
        for _ in range(2):
            try:
                res = ollama.chat(model=self.model, messages=[{"role": "user", "content": prompt}],
                                  options={"temperature": temperature, "num_predict": num_predict,
                                           "seed": seed, "num_ctx": LLM_NUM_CTX}, think=False)
                text = (res.message.content or "").strip()
                if text:
                    self.cache.put(k, text)
                    return text
                last = "empty generation"
            except Exception as exc:
                last = repr(exc)
        raise RuntimeError(f"generation failed for {self.model}: {last}")

    def embed(self, texts):
        out = []
        for i in range(0, len(texts), EMBED_BATCH):
            part = [f"search_query: {t[:EMBED_CHAR_LIMIT]}" for t in texts[i:i + EMBED_BATCH]]
            res = ollama.embed(model=EMB_MODEL, input=part)
            out.extend(res["embeddings"])
        arr = np.array(out, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms


class Judge:
    def __init__(self, llm=None):
        self.backend = "nli"
        self.llm = llm
        self.model = None
        self.load_error = ""
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(NLI_MODEL, max_length=512, local_files_only=True, device="cpu")
            self.labels = {str(v).lower(): k for k, v in self.model.model.config.id2label.items()}
        except Exception as exc:
            self.backend = "llm"
            self.load_error = repr(exc)

    def probs(self, pairs):
        if self.model is None:
            return None
        raw = self.model.predict(pairs, apply_softmax=True)
        out = []
        for row in raw:
            out.append({
                "entail": float(row[self.labels.get("entailment", 1)]),
                "contra": float(row[self.labels.get("contradiction", 0)]),
                "neutral": float(row[self.labels.get("neutral", 2)]),
            })
        return out

    def entail(self, claim, evidence):
        if not evidence:
            return 0.0, 0.0, "nli"
        pairs = [(e["text"][:EVIDENCE_CHAR_LIMIT], claim) for e in evidence]
        scored = self.probs(pairs)
        if scored is None:
            ent, contra = self._llm_judge(claim, evidence)
            return ent, contra, "llm"
        best = max(zip(scored, evidence), key=lambda x: x[0]["entail"])
        contra = max(s["contra"] for s in scored)
        return best[0]["entail"], contra, "nli"

    def _llm_judge(self, claim, evidence):
        ctx = "\n\n".join(f"Passage {i + 1}: {e['text'][:LLM_CTX_CHAR]}" for i, e in enumerate(evidence[:3]))
        prompt = (f"Passages are excerpts from different NIT Jamshedpur notices. A claim is supported when at least one "
                  f"passage states it, even if other passages are about a different semester, year or department.\n\n{ctx}\n\n"
                  f"Claim: {claim}\n\n"
                  f"Answer with one word only: SUPPORTED if a passage states the claim, CONTRADICTED if a passage states a "
                  f"different value for the same fact, or NEUTRAL if no passage addresses it.")
        out = self.llm.call(prompt, temperature=0.0, num_predict=JUDGE_LLM_NUM_PREDICT).upper()
        if "CONTRADICT" in out:
            return 0.0, 1.0
        if "SUPPORT" in out:
            return 1.0, 0.0
        return 0.0, 0.0


class Reranker:
    def __init__(self):
        self.model = None
        self.error = ""
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(RERANK_MODEL, max_length=512, local_files_only=True, device="cpu")
        except Exception as exc:
            self.error = repr(exc)

    def score(self, query, docs):
        if self.model is None or not docs:
            return [0.0] * len(docs)
        return [float(x) for x in self.model.predict([(query, d["text"][:RERANK_CHAR_LIMIT]) for d in docs])]


class Retriever:
    def __init__(self, llm=None):
        self.llm = llm or LLM()
        self.chunks = load_chunks()
        self.by_id = {c["chunk_id"]: c for c in self.chunks}
        with open(INDEX_DIR / "bm25.pkl", "rb") as f:
            blob = pickle.load(f)
        self.bm25 = blob["bm25"]
        self.ids = blob["ids"]
        client = chromadb.PersistentClient(path=str(INDEX_DIR))
        self.col = client.get_collection(COLLECTION)
        self.reranker = Reranker()
        self.stats = {"dense_calls": 0, "rerank_calls": 0, "retrieve_calls": 0}

    def _dense(self, query, k):
        q = self.llm.embed([query])
        res = self.col.query(query_embeddings=q.tolist(), n_results=min(k, self.col.count()))
        out = []
        for rank, cid in enumerate(res["ids"][0]):
            meta = res["metadatas"][0][rank] or {}
            out.append({"chunk_id": cid, "score": 1.0 - float(res["distances"][0][rank]),
                        "rank": rank + 1, "meta": meta, "text": res["documents"][0][rank]})
        self.stats["dense_calls"] += 1
        return out

    def _sparse(self, query, k):
        scores = self.bm25.get_scores(tokenize(query))
        order = np.argsort(-scores)[:k]
        return [{"chunk_id": self.ids[i], "score": float(scores[i]), "rank": r + 1} for r, i in enumerate(order)]

    def retrieve(self, query, k=RETRIEVE_K, rerank=True):
        self.stats["retrieve_calls"] += 1
        dense = self._dense(query, k)
        sparse = self._sparse(query, k)
        fused = {}
        for lst, w in ((dense, DENSE_WEIGHT), (sparse, SPARSE_WEIGHT)):
            for row in lst:
                cid = row["chunk_id"]
                chunk = self.by_id.get(cid, {})
                age = age_years(chunk.get("issue_date", "")) or RECENCY_DEFAULT_AGE
                recency = RECENCY_BASE + RECENCY_SPAN * float(np.exp(-RECENCY_LAMBDA * age))
                fused[cid] = fused.get(cid, 0.0) + w * recency / (RRF_K + row["rank"])
        ranked = sorted(fused.items(), key=lambda x: -x[1])[:max(k, RERANK_TOP * 3)]
        docs = []
        for cid, score in ranked:
            chunk = self.by_id[cid]
            docs.append({"chunk_id": cid, "text": chunk["text"], "doc_name": chunk["doc_name"],
                         "category": chunk["category"], "issue_date": chunk["issue_date"],
                         "ref_number": chunk["ref_number"], "page": chunk["page"],
                         "rrf": score, "dense": next((d["score"] for d in dense if d["chunk_id"] == cid), 0.0)})
        if rerank:
            scores = self.reranker.score(query, docs)
            self.stats["rerank_calls"] += 1
            docs = sorted(zip(docs, scores), key=lambda x: -x[1])[:RERANK_TOP]
            docs = [{**d, "rerank": s} for d, s in docs]
        return docs

    def best_sentences(self, claim, docs, limit=JUDGE_TOP):
        scored = []
        for d in docs:
            for s in sentences(d["text"]):
                if len(s.split()) < BEST_SENT_MIN_WORDS:
                    continue
                ov = overlap_terms(claim, s)
                if ov <= 0:
                    continue
                scored.append((ov, s, d))
        scored.sort(key=lambda x: -x[0])
        out = []
        seen = set()
        for ov, s, d in scored:
            key = re.sub(r"\W+", " ", s.lower()).strip()
            if key in seen:
                continue
            seen.add(key)
            out.append({"text": s, "doc_name": d["doc_name"], "chunk_id": d["chunk_id"],
                        "issue_date": d["issue_date"], "overlap": ov})
            if len(out) >= limit:
                break
        return out


class NumericCheck:
    def __init__(self, tol_money=NUMERIC_TOL_MONEY, tol_pct=NUMERIC_TOL_PCT, records=None):
        self.tol_money = tol_money
        self.tol_pct = tol_pct
        self.records = records or []

    def _record_result(self, claim, evidence_sentences):
        ce = entities(claim)
        wide = evidence_sentences or []
        ev_ids = {e.get("chunk_id") for e in wide}
        low = claim.lower()
        for rec in self.records:
            if int(rec.get("support", 0)) < RECORD_MIN_SUPPORT:
                continue
            if rec.get("source_chunk_id") not in ev_ids:
                continue
            aliases = rec.get("aliases") or [rec["key"].replace("_", " ")]
            if not any(alias.lower() in low for alias in aliases):
                continue
            rt = rec.get("val_type")
            kind = {"currency": "money", "percent": "percent", "count_days": "days",
                    "date": "dates", "count_books": "books", "number": "cgpa",
                    "time": "times"}.get(rt)
            if kind is None or not ce.get(kind):
                continue
            other_kinds = [name for name in ("money", "percent", "dates", "days", "cgpa", "times")
                           if name != kind and ce.get(name)]
            if other_kinds:
                continue
            rv = rec["value"]
            if rt in ("date", "time"):
                if str(rv) in ce[kind]:
                    return 1.0, f"record {rec['key']} matches [{rec['source_doc']}]"
                return 0.0, f"record {rec['key']} {rv} vs {ce[kind]} [{rec['source_doc']}]"
            try:
                rvn = float(rv)
            except (TypeError, ValueError):
                continue
            if kind == "money":
                matched = any(abs(cv - rvn) <= self.tol_money * max(1.0, abs(rvn)) for cv in ce[kind])
            elif kind in ("percent", "cgpa"):
                matched = any(abs(cv - rvn) <= self.tol_pct for cv in ce[kind])
            else:
                matched = any(abs(cv - rvn) <= NUMERIC_TOL_DAYS for cv in ce[kind])
            if matched:
                return 1.0, f"record {rec['key']} matches [{rec['source_doc']}]"
            return 0.0, f"record {rec['key']} {rec.get('operator', '=')} {rv} vs {ce[kind]} [{rec['source_doc']}]"
        return None

    def compare(self, claim, evidence_sentences):
        rr = self._record_result(claim, evidence_sentences)
        if rr is not None:
            return rr
        ce = entities(claim)
        kinds = [k for k in ("money", "percent", "dates", "days", "cgpa", "times") if ce[k]]
        if not kinds:
            return 0.5, "no checkable value in claim"
        best = None
        claim_terms = content_terms(claim)
        for es in evidence_sentences:
            if overlap_terms(claim, es["text"]) < OVERLAP_GATE:
                continue
            if claim_terms and not (claim_terms & content_terms(es["text"])):
                continue
            ee = entities(es["text"])
            hits = 0
            misses = 0
            detail = []
            for k in kinds:
                if not ce[k]:
                    continue
                if not ee[k]:
                    continue
                for cv in ce[k]:
                    if k == "money":
                        matched = any(abs(cv - ev) <= self.tol_money * max(1.0, abs(ev)) for ev in ee[k])
                    elif k in ("percent", "cgpa"):
                        matched = any(abs(cv - ev) <= self.tol_pct for ev in ee[k])
                    elif k == "days":
                        matched = any(abs(cv - ev) <= NUMERIC_TOL_DAYS for ev in ee[k])
                    else:
                        matched = cv in ee[k]
                    if matched:
                        hits += 1
                        detail.append(f"{k} {cv} matches")
                    else:
                        misses += 1
                        detail.append(f"{k} {cv} vs {ee[k][:3]}")
            if hits or misses:
                score = (hits - misses) / max(1, hits + misses)
                cand = (round(overlap_terms(claim, es["text"]), 2), score, "; ".join(detail), es)
                if best is None or cand[:2] > best[:2]:
                    best = cand
        if best is None:
            return 0.5, "no overlapping evidence value"
        _, score, note, es = best
        if score >= 0.5:
            return 1.0, f"{note} [{es['doc_name']}]"
        if score <= -0.5:
            return 0.0, f"{note} [{es['doc_name']}]"
        return 0.5, f"{note} [{es['doc_name']}]"


class PolicyCheck:
    def __init__(self, path=DATA / "policies.json"):
        self.rules = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []

    def check(self, claim, ce):
        for rule in self.rules:
            if not rule.get("enabled"):
                continue
            if not any(k in claim.lower() for k in rule["keywords"]):
                continue
            vals = ce.get(rule["field"], [])
            if not vals:
                continue
            op, ref = rule["op"], float(rule["value"])
            for v in vals:
                ok = {"<=": v <= ref, ">=": v >= ref, "=": abs(v - ref) < 1e-6}[op]
                if not ok:
                    return 0.0, f"violates {rule['rule_id']} ({op} {rule['value']})"
            return 1.0, f"consistent with {rule['rule_id']} ({op} {rule['value']})"
        return 0.5, "no applicable policy rule"


class Gate:
    def __init__(self, weights=None, tau_pass=None, tau_contra=None):
        cfg = json.loads((DATA / "thresholds.json").read_text(encoding="utf-8")) if (DATA / "thresholds.json").exists() else {}
        self.w = weights or cfg.get("weights") or DEFAULT_WEIGHTS
        self.tau_pass = tau_pass if tau_pass is not None else cfg.get("tau_pass", TAU_PASS)
        self.tau_contra = tau_contra if tau_contra is not None else cfg.get("tau_contra", TAU_CONTRA)

    def score(self, F, P, D):
        return float(np.clip(self.w["w_fact"] * F + self.w["w_policy"] * P + self.w["w_ground"] * D, 0, 1))

    def decide(self, F, P, D, C):
        G = self.score(F, P, D)
        if F <= FACT_BLOCK or P <= FACT_BLOCK:
            return G, "BLOCK", "false"
        if C >= self.tau_contra and D < self.tau_pass:
            return G, "BLOCK", "false"
        if D >= self.tau_pass:
            return G, "PASS", "true"
        return G, "FLAG", "unverifiable"


def current_session(ref=None):
    """Academic session for a date; a session starts with the autumn admission in July."""
    ref = ref or reference_date()
    start = ref.year if ref.month >= 7 else ref.year - 1
    return f"{start}-{start + 1}"


def context_block(docs, by_id):
    """Render retrieved chunks for the answer prompt, carrying each chunk's session."""
    parts = []
    for d in docs:
        session = by_id.get(d.get("chunk_id"), {}).get("session", "")
        stamp = " | ".join(x for x in (d["doc_name"], d["issue_date"], session, d["ref_number"]) if x)
        parts.append(f"[{stamp}]\n{d['text'][:ANSWER_CTX_CHAR]}")
    return "\n\n".join(parts)


def answer_prompt(query, ctx, today=None):
    today = today or reference_date()
    session = current_session(today)
    return (
        f"You are the NIT Jamshedpur academic assistant. Use only the context.\n"
        f"Today is {today.isoformat()}. The current academic session is {session}.\n\n"
        f"Rules:\n"
        f"1. Prefer the newest notice for the topic; a newer circular supersedes an older one.\n"
        f"2. Every context entry shows its issue date and session. If the newest evidence for the "
        f"question is older than session {session}, say so plainly: state the date and session of the "
        f"latest available notice and that no {session} notice is in the corpus. Never present an "
        f"older notice's dates as the current schedule.\n"
        f"3. Answer in at most three sentences. If the context does not contain the answer, say you "
        f"do not know.\n\n"
        f"Context:\n{ctx}\n\n"
        f"Question: {query}\nAnswer:"
    )


ANSWER_ABBREV = re.compile(
    r"\b(Rs|No|Nos|Dr|Mr|Mrs|Ms|Prof|Sr|Jr|vs|etc|viz|cf|Fig|Figs|Ref|Vol|Sec|Dept|"
    r"i\.e|e\.g|Ph\.D|M\.Tech|B\.Tech|M\.Sc|B\.Sc)\.",
    re.IGNORECASE)
ANSWER_DOT = "\x01"


def answer_sentences(text):
    """Split a draft answer into sentences without cutting at abbreviations.

    A plain split on '. ' turns "the hostel fee was Rs. 15,000/-" into
    "the hostel fee was Rs." plus a detached "15,000/-", which drops the amount
    from the claim and lets it pass unverified.
    """
    protected = ANSWER_ABBREV.sub(lambda m: m.group(1) + ANSWER_DOT, text)
    return [part.replace(ANSWER_DOT, ".") for part in sentences(protected)]


class Pipeline:
    def __init__(self, weights=None, judge=None, retriever=None, llm=None):
        self.llm = llm or LLM()
        self.retriever = retriever or Retriever(self.llm)
        self.judge = judge or Judge(self.llm)
        try:
            import json as _j
            _recs = _j.loads((DATA / "records.json").read_text(encoding="utf-8"))
        except Exception:
            _recs = []
        self.numeric = NumericCheck(records=_recs)
        self.policy = PolicyCheck()
        self.gate = Gate(weights=weights)

    def verify(self, claim, weights=None):
        gate = Gate(weights=weights) if weights else self.gate
        parts = [p.strip() for p in re.split(ATOMIC_SPLIT, claim) if p.strip()]
        if len(parts) >= 2 and all(len(p.split()) >= ATOMIC_MIN_PART_WORDS for p in parts):
            subs = [self.verify(p, weights=weights) for p in parts]
            F = min(s.F for s in subs)
            P = min(s.P for s in subs)
            D = min(s.D for s in subs)
            C = max(s.C for s in subs)
            G, decision, label = gate.decide(F, P, D, C)
            ev = []
            for s in subs:
                ev.extend(s.evidence)
            seen = set()
            dedup = []
            for e in ev:
                if e["chunk_id"] in seen:
                    continue
                seen.add(e["chunk_id"])
                dedup.append(e)
            return ClaimResult(
                claim=claim, category=subs[0].category,
                F=round(F, 3), P=round(P, 3), D=round(D, 3), C=round(C, 3), G=round(G, 3),
                decision=decision, label=label,
                fact_note=" | ".join(s.fact_note for s in subs),
                policy_note=" | ".join(s.policy_note for s in subs),
                evidence=dedup[:6], judge_backend=subs[0].judge_backend)
        docs = self.retriever.retrieve(claim)
        ev = self.retriever.best_sentences(claim, docs, limit=JUDGE_TOP)
        wide = self.retriever.best_sentences(claim, docs, limit=WIDE_SENT_LIMIT)
        ce = entities(claim)
        fnum, fnote = self.numeric.compare(claim, wide or ev)
        ent, contra, backend = self.judge.entail(claim, ev or docs[:2])
        D = float(np.clip(ent - GROUND_PENALTY * contra, 0, 1))
        P, pnote = self.policy.check(claim, ce)
        G, decision, label = gate.decide(fnum, P, D, contra)
        return ClaimResult(
            claim=claim, category=docs[0]["category"] if docs else "General",
            F=round(fnum, 3), P=round(P, 3), D=round(D, 3), C=round(float(contra), 3), G=round(G, 3),
            decision=decision, label=label, fact_note=fnote, policy_note=pnote,
            evidence=[{"doc_name": e["doc_name"], "chunk_id": e["chunk_id"],
                       "date": e["issue_date"], "text": e["text"][:EVIDENCE_SNIPPET_CHAR]} for e in ev],
            judge_backend=backend)

    def answer(self, query):
        t0 = time.time()
        docs = self.retriever.retrieve(query)
        ctx = context_block(docs, self.retriever.by_id)
        prompt = answer_prompt(query, ctx)
        draft = self.llm.call(prompt, temperature=0.0, num_predict=DRAFT_NUM_PREDICT)
        claims = []
        for s in answer_sentences(draft):
            if len(s.split()) < UNSUP_CLAIM_MIN_WORDS:
                continue
            claims.append(self.verify(s.rstrip(".")))
        kept = []
        for c in claims:
            if c.decision == "PASS":
                kept.append(c)
                continue
            if c.decision != "BLOCK":
                continue
            cur = c
            for _ in range(REGEN_ROUNDS):
                evtext = "\n".join(e["text"][:EVIDENCE_SNIPPET_CHAR] for e in cur.evidence)
                rw = self.llm.call(
                    f"Rewrite the following claim as ONE factual sentence using only the evidence. "
                    f"If the evidence does not support any version, reply UNVERIFIABLE.\n"
                    f"Claim: {cur.claim}\nEvidence:\n{evtext}\nRewrite:", REGEN_TEMPERATURE, REGEN_NUM_PREDICT)
                rw = rw.strip().splitlines()[0].strip() if rw.strip() else ""
                if not rw or "unverifiable" in rw.lower():
                    break
                nxt = self.verify(rw.rstrip("."))
                if nxt.decision == "PASS":
                    nxt.regen = rw
                    kept.append(nxt)
                    break
                cur = nxt
        seen = set()
        final = []
        for c in kept:
            key = re.sub(r"\W+", " ", c.claim.lower()).strip()
            if key in seen:
                continue
            seen.add(key)
            final.append(c)
        abstained = not final
        answer = " ".join(f"{c.claim}." for c in final) if final else "I could not verify an answer from the available circulars."
        cites = sorted({e["doc_name"] for c in final for e in c.evidence})
        return AnswerResult(query=query, draft=draft, answer=answer, claims=final,
                            latency=round(time.time() - t0, 2), abstained=abstained, citations=cites,
                            evaluated_claims=claims, retrieval_stats=dict(self.retriever.stats))

    def health(self):
        return {
            "judge_backend": self.judge.backend,
            "judge_error": self.judge.load_error,
            "reranker_enabled": self.retriever.reranker.model is not None,
            "reranker_error": self.retriever.reranker.error,
            "records": len(self.numeric.records),
            "thresholds": {"tau_pass": self.gate.tau_pass, "tau_contra": self.gate.tau_contra},
            "retrieval_stats": dict(self.retriever.stats),
        }


if __name__ == "__main__":
    p = Pipeline()
    for q in ["What is the minimum attendance required for end semester examinations?",
              "What is the tuition fee for B.Tech first semester?"]:
        r = p.answer(q)
        print("\nQ:", q, "|", r.latency, "s")
        print("draft:", r.draft[:300])
        print("answer:", r.answer[:300])
        for c in r.claims:
            print(f"  [{c.decision}] F={c.F} P={c.P} D={c.D} G={c.G} {c.fact_note[:80]} :: {c.claim[:90]}")
