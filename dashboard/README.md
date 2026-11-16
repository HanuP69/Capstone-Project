# Capstone Project Dashboard

A local React + TypeScript research site for the NIT Jamshedpur groundedness-verification capstone. The site includes a project overview, a query demo, pipeline description, evaluation results, audit trail and team information.

## Run locally

Requirements: Node.js LTS and npm.

On Windows, double-click `start-dashboard.cmd` to start Vite in an interactive console and open the demo after the local server responds. Keep that console open while using the site; press Ctrl+C in it to stop the server. Alternatively, use the manual commands below.

```powershell
cd dashboard
Copy-Item .env.example .env
npm install
npm run dev
```

Open the Vite URL printed by the dev server (normally `http://127.0.0.1:5173`). The default `.env.example` sets `USE_MOCK=true`; the mock verification layer uses the repository’s existing `primary_audit_trail.csv` records, and evaluation data is generated from the repository CSV/JSON files whenever the app starts or builds.

To use the actual FastAPI query endpoint, set `USE_MOCK=false` in `dashboard/.env`, then run the backend in a second terminal from the repository root:

```powershell
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

The Vite `/api` proxy forwards to `http://127.0.0.1:8000`. Verification uses the current `POST /api/verify-query` request contract (`query`, `institution`, `use_meta_learner`, `retrieval_mode`) and response fields defined by root `app.py`. Other benchmark views read the local CSV/JSON snapshots.

## Build

```powershell
npm run build
npm run preview
```

## Data note

Institutional benchmark records shown in mock mode are synthetic and are based on published NIT Jamshedpur material. Per-claim latency is not included in the source claim records, so it is not inferred. The app remains plain local; no hosted preview or Manus services are configured.
