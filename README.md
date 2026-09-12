# Pehchaan — AI-Based Fake Identity & Document Screening System

**SIH Problem Statement 26188** · Ministry of Home Affairs · Sashastra Seema
Bal (SSB), Police II Division · Category: Software · Theme: Blockchain &
Cybersecurity

A working prototype that screens passports, visas, national IDs, driving
licences, and permits in seconds — extracting fields via OCR, validating
them against real document standards (ICAO 9303 checksum math for
passports), detecting tampering with digital-forensics techniques, and
verifying the presenter's face against the document photo. Everything
rolls up into one transparent, explainable risk score an officer can act
on immediately.

Full problem statement: [docs/PROBLEM_STATEMENT.md](docs/PROBLEM_STATEMENT.md)
Architecture & design rationale: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
API reference: [docs/API.md](docs/API.md)

---

## Quick start

You need two terminals — one for the API, one for the UI.

```bash
# Terminal 1 — backend (Python 3.10+)
cd backend
python -m venv .venv
.venv\Scripts\activate          # Windows — use `source .venv/bin/activate` on macOS/Linux
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# Terminal 2 — frontend (Node 18+)
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173**. Full setup notes (including the one manual
step — installing the Tesseract OCR binary — and how the system degrades
gracefully without it) are in [backend/README.md](backend/README.md).

Run the backend test suite (validates the real ICAO 9303 check-digit math
against the canonical worked example) with `pytest tests/ -v` from `backend/`.

---

## How a screening works, end to end

```
Officer uploads document (+ optional live selfie)
        │
        ▼
Module 1 — OCR Extraction        →  raw text, MRZ lines, per-doc-type fields
        │
        ▼
Module 2 — Document Validation   →  ICAO checksum verification, expiry/date
        │                             logic, cross-field consistency
        ▼
Module 3 — Tampering Detection    →  Error Level Analysis, copy-move
        │                             forensics, metadata inspection
        ▼
Module 4 — Face Verification      →  document photo vs. live capture
        │
        ▼
Risk Engine  →  0–100 score, CLEAR / REVIEW / REJECT, evidence list
        │
        ▼
Digital trail  →  every scan persisted for audit & investigation
```

---

## Project structure

```
ps 26188/
│
├── docs/                          Problem statement, architecture, API reference
│   ├── PROBLEM_STATEMENT.md         The original SIH brief, preserved verbatim
│   ├── ARCHITECTURE.md              Why this stack, module-by-module design
│   └── API.md                        Every endpoint, request/response shape
│
├── backend/                        FastAPI service — all four AI modules
│   ├── app/
│   │   ├── main.py                    App entrypoint, CORS, router wiring
│   │   ├── config.py                  Every tunable value (thresholds, weights), env-driven
│   │   │
│   │   ├── api/                       HTTP route handlers (thin — logic lives in modules/)
│   │   │   ├── routes_documents.py       POST /scan, GET /scans, GET /stats
│   │   │   ├── routes_face.py             Standalone face-verify endpoint
│   │   │   └── routes_health.py           System status for the frontend's live badges
│   │   │
│   │   ├── modules/                   The four problem-statement modules, one folder each
│   │   │   ├── ocr/                       Module 1 — OCR Extraction
│   │   │   │   ├── engine.py                 Tesseract wrapper, degrades gracefully if absent
│   │   │   │   ├── mrz_parser.py              ICAO 9303 machine-readable-zone decoder +
│   │   │   │   │                                check-digit algorithm (the one deterministic,
│   │   │   │   │                                mathematically-provable module in the pipeline)
│   │   │   │   └── field_extractors.py         Regex/heuristic field pulls for non-MRZ docs
│   │   │   │                                    (visa, national ID, driving licence, permit)
│   │   │   │
│   │   │   ├── validation/                Module 2 — Document Validation
│   │   │   │   ├── schemas.py                 Official-standard field shapes per doc type
│   │   │   │   └── rules_engine.py             Checksum/expiry/date-logic/name-consistency rules,
│   │   │   │                                    each with a human-readable reason
│   │   │   │
│   │   │   ├── tampering/                 Module 3 — Tampering Detection (core AI innovation)
│   │   │   │   ├── ela.py                      Error Level Analysis (JPEG re-compression forensics)
│   │   │   │   ├── copy_move.py                 ORB self-similarity matching (clone-stamp detection)
│   │   │   │   ├── metadata_analysis.py          EXIF / editor-signature inspection
│   │   │   │   └── tampering_engine.py            Aggregates the three signals into one verdict
│   │   │   │
│   │   │   ├── face/                       Module 4 — Face Verification
│   │   │   │   ├── detector.py                 Face localization (OpenCV)
│   │   │   │   └── verifier.py                  Similarity scoring, pluggable backend
│   │   │   │                                     (lightweight by default; swap in
│   │   │   │                                     `face_recognition`/dlib for stronger accuracy)
│   │   │   │
│   │   │   └── risk/                       Combines Modules 2–4 into the final decision
│   │   │       └── risk_engine.py              Weighted, explainable 0–100 score + verdict
│   │   │
│   │   ├── models/schemas.py           API-facing response contracts (wire format)
│   │   ├── storage/file_store.py        The "digital trail" — every scan, audit-ready
│   │   └── utils/                        Shared image/logging helpers
│   │
│   ├── tests/                          pytest suite — proves the MRZ checksum math, validation
│   │                                     rules, and face-similarity logic are actually correct
│   ├── requirements.txt
│   ├── Dockerfile
│   └── README.md                        Setup, OCR/face backend notes, test instructions
│
├── frontend/                        React 18 + TypeScript + Vite + Tailwind + Framer Motion
│   ├── src/
│   │   ├── pages/                      One file per screen
│   │   │   ├── LandingPage.tsx             Command Overview — hero, live stats, module explainer
│   │   │   ├── ScanDashboard.tsx            New Screening — doc-type picker, upload, progress
│   │   │   ├── ScanResult.tsx                Screening Report — risk gauge + full module breakdown
│   │   │   ├── AuditLog.tsx                   Digital trail table, click-through to any past report
│   │   │   └── NotFound.tsx
│   │   │
│   │   ├── components/
│   │   │   ├── layout/                     AppShell, Sidebar, Topbar (live OCR/face status badges)
│   │   │   ├── upload/                      DocumentTypeSelect, DocumentDropzone (drag-and-drop)
│   │   │   ├── risk/                         RiskGauge (animated), RiskBadge, ScoreBar
│   │   │   └── ui/                            Button, Card, Badge, Spinner — the design-system primitives
│   │   │
│   │   ├── lib/                          api.ts (typed fetch client), utils.ts (color/format helpers)
│   │   └── types/                          TypeScript mirrors of the backend's response schemas
│   │
│   ├── tailwind.config.ts              Design tokens — the navy/cyan/amber/rose/emerald palette
│   └── README.md                        Setup, design-system notes, build instructions
│
├── sample-data/                     Space for demo document images (kept out of git)
└── .gitignore
```

---

## What's genuinely real vs. what's a documented prototype shortcut

Being upfront about this matters for a security tool:

| Piece | Status |
|---|---|
| ICAO 9303 MRZ check-digit math | **Real.** Implements the published weight-7/3/1 algorithm exactly; verified against the canonical ICAO worked example in `tests/test_mrz_parser.py`, including a test that deliberately corrupts one digit and confirms the parser catches it. |
| Error Level Analysis, copy-move detection, metadata forensics | **Real, classic digital-forensics techniques** — no training data needed, run offline, generalize to any image. |
| OCR text extraction | Real via Tesseract, but the **Tesseract binary is a one-time manual install** (not pip-installable) — the system detects its absence and degrades gracefully with a clear warning rather than crashing. |
| Face verification | Real comparison logic (verified via unit tests), but the **default backend is a lightweight OpenCV histogram/ORB comparator**, not a trained face-embedding model — swap in `face_recognition` (dlib) for production-grade accuracy; the code already supports it as a drop-in. |
| Audit trail / digital trail | **Functional, in-memory** for this prototype — swap `app/storage/file_store.py` for Postgres/SQLite for persistence across restarts. |
| Risk scoring | A transparent, weighted rule engine — deliberately not a black-box model, since a border officer needs to see *why* a score was assigned. |

---

## Expected impact (per the problem statement)

- Verification time: minutes → seconds
- Standardized, explainable screening decisions across checkpoints
- Machine-verifiable forgery detection instead of purely manual inspection
- A full digital trail for investigations and intelligence analysis
