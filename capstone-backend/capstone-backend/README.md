# Groundedness Verification Backend (FastAPI)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload      # docs at http://localhost:8000/docs
pytest
```

## Flow
query -> `ResponseGenerator` -> `ClaimExtractor` -> 3 agents in parallel (fact / policy / grounding)
-> `scoring.groundedness` -> `gate.decide` -> regenerate blocked claims (max `MAX_REGENERATIONS`)

## Who plugs in where
| Module | Owner | Status |
|---|---|---|
| `agents/team.py` (+ `agents/adapters.py`) | Ankit | placeholders: plug real agents here; falls back to `agents/starter/` |
| `api/` | Ankit | working |
| `pipeline/generator.py`, grounding agent via `agents/team.py` | Tanishk | lexical starter in `agents/starter/` | |
| `pipeline/scoring.py` | Mohit | fixed weights, needs calibration |
| `pipeline/claim_extractor.py` | Sujal | sentence splitter placeholder |
| `data/*.json` | Himanshu | placeholder data, replace with real NITJ records |
| `/api/v1/*` consumed by | Dhirendra | dashboard |

Endpoints: `POST /api/v1/query`, `POST /api/v1/verify` (score baseline outputs/datasets), `GET /api/v1/queries[/{id}]`.
