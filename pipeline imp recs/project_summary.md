# AI-Based Fake Identity & Document Screening System — Project Summary

**Organization:** Ministry of Home Affairs — Sashastra Seema Bal (SSB), Police II Division
**Category:** Software | **Theme:** Blockchain & Cybersecurity

---

## 1. Problem & Vision

Border checkpoints process thousands of identity/travel documents daily (passports, visas, national IDs, permits). Current verification relies on manual inspection and basic database lookups, which is:
- Slow (minutes per document), causing passenger-volume bottlenecks
- Error-prone and unable to reliably catch sophisticated forgery
- Not standardized across checkpoints/officers
- Not investigable after the fact (no systematic digital trail)

**Threats to detect specifically:** fake passports/visas, altered photographs, modified dates of birth, tampered visa stamps, identity impersonation, multiple identities per person, expired/blacklisted documents.

**Vision:** an AI-powered screening platform that extracts document data, validates it, detects tampering/forgery, verifies the presenting person's identity, and produces an **explainable risk score** to assist (not replace) border security personnel — reducing verification time from minutes to seconds while creating an immutable digital trail for investigations.

**Core design principle carried through every decision below:** the system augments human judgment, it never auto-decides. Every scoring/flagging mechanism is built to be explainable to an officer and auditable after the fact, because border decisions can be legally challenged.

---

## 2. High-Level Architecture & Why

**Pattern chosen: event-driven microservices connected by Apache Kafka.**

Why not a monolith or simple synchronous request-response API: the four AI modules (OCR, Tampering Detection, Face Verification, Validation) have very different compute profiles — GPU-heavy CNN inference (tampering, face) vs CPU-bound rule/OCR logic. A monolith or sync pipeline couples them, so one slow module blocks the officer's counter, and you can't scale or upgrade one module independently. Kafka decouples producers/consumers, absorbs passenger-volume bursts (explicitly named pain point), and gives replayability per document for debugging/investigation — which a sync system can't offer for free.

**Pipeline (top to bottom):**

```
Document Capture → Preprocessing
        ↓ (fan-out via Kafka, 3 parallel consumer groups)
   OCR Extraction   Tampering Detection   Face Verification
        ↓                                        ↓
   Document Validation ─────────────────────────┘
        ↓
   Risk Fusion Engine (joins all 4 signal streams per doc_id)
        ↓
   Decision Dashboard (officer) → Audit Ledger (permissioned blockchain)
```

---

## 3. Kafka Backbone — Topics & Reasoning

| Topic | Producer | Consumer(s) | Key |
|---|---|---|---|
| `doc.ingested` | Ingestion gateway | Preprocessing | `doc_id` |
| `doc.preprocessed` | Preprocessing | OCR, Tampering, Face (separate consumer groups) | `doc_id` |
| `ocr.completed` | OCR service | Validation, Risk Fusion | `doc_id` |
| `validation.completed` | Validation service | Risk Fusion | `doc_id` |
| `tampering.completed` | Tampering service | Risk Fusion | `doc_id` |
| `face.completed` | Face service | Risk Fusion | `doc_id` |
| `risk.assessed` | Risk Fusion engine | Decision dashboard | `doc_id` |
| `decision.logged` | Decision dashboard | Audit ledger writer | `doc_id` |
| `audit.committed` | Audit ledger writer | monitoring/investigation tooling | `doc_id` |

**Why `doc_id` as partition key everywhere:** guarantees all events for one document land in the same partition, so the Risk Fusion aggregator can do an ordered windowed join per document without cross-document race conditions. Each downstream service is its own consumer group so OCR/Tampering/Face genuinely run in parallel off the same `doc.preprocessed` event rather than competing for messages.

**Failure handling:** if a module hasn't reported within the SLA window, Risk Fusion fuses on whatever signals are available and flags `partial_assessment` rather than blocking the officer indefinitely.

---

## 4. Module-by-Module Logic

### 4.0 Preprocessing (shared, before fan-out)
Deskew, denoise, contrast-normalize, auto-crop to document boundary; classify document type (passport/visa/ID/license) via a lightweight CNN, since downstream field templates differ per type. Must preserve a copy of the image *prior* to any JPEG recompression, because ELA (used later in tampering detection) is sensitive to recompression artifacts.

### 4.1 Module 1 — OCR Extraction
**Approaches considered:** generic OCR (Tesseract) — brittle on structured docs with MRZ/holograms; cloud Vision APIs — accurate but a data-sovereignty non-starter for MHA/SSB (sends passport data off-prem); **chosen: layout-aware OCR pipeline, on-prem.**

**Pipeline:** zone detection (CRAFT/DBNet) locates MRZ/text/stamp regions → per-zone recognition (PaddleOCR) to avoid confusing similar fields (passport no. vs visa no.) → dedicated MRZ parser exploiting ICAO 9303's self-validating checksum → LayoutLM-style model maps text spans to semantic fields using text + spatial position. Low-confidence fields or failed MRZ checksum route to manual correction rather than silently continuing.
**Output → `ocr.completed`:** `{field: {value, confidence, bbox}}`, `mrz_checksum_valid`.

### 4.2 Module 2 — Document Validation
**Approaches considered:** pure ML classifier (unauditable black box, legally weak) vs pure rules (deterministic but misses unanticipated anomalies) vs **chosen: hybrid — rules as primary gate, ML anomaly layer as soft signal.**

**Pipeline:** format regex + date logic (`dob < issue_date < expiry_date`) + independent MRZ checksum re-verification + blacklist/watchlist DB lookup + isolation-forest anomaly scoring over field statistics (font metrics, position drift vs known-good templates).
**Output → `validation.completed`:** `{rule_violations[], watchlist_hit, anomaly_score}`.

### 4.3 Module 3 — Tampering Detection (core AI innovation)
**Key constraint driving the design:** no large labeled forged-passport dataset exists (can't legally mass-produce forgeries to train on). Solution: don't ask one model to learn "forged vs real" from raw pixels — decompose tampering into physically-grounded weak signals, and only let a small model learn to *combine* them.

**Approaches considered:** metadata-only forensics (trivially defeated by re-saving) — insufficient alone; ELA alone — false-positives on legitimately rescanned documents; end-to-end CNN/GAN forgery classifier alone — needs a forged-document dataset that doesn't exist at scale; **chosen: ensemble of ELA + PRNU noise-residual analysis + metadata check, fused by a small CNN, plus two targeted checks.**

**Pipeline:** ELA (recompression-artifact heatmap) + PRNU/noise-residual (sensor-fingerprint discontinuity detects splicing) + metadata analysis (editing-software traces, missing scanner signatures) → stack as channels → small fusion CNN → `tamper_score` + region heatmap. Two targeted checks run as OR-gates on top of the fused score (catch the two threats explicitly named in the problem statement even if general fusion is moderate): photo-zone check (edge/compression signature vs rest of document, catches photo replacement) and stamp-zone template match (catches stamp forgery).
**Output → `tampering.completed`:** `{tamper_score, tamper_regions[], flags[]}`.

### 4.4 Module 4 — Face Verification
**Key constraint driving the design:** identity impersonation and photo replacement are explicit threats. A pure similarity score is vulnerable to presentation attacks (printed photo, screen replay) — a high similarity score against a spoofed capture is a false assurance, not a safety signal.

**Approaches considered:** raw pixel similarity (fails under pose/lighting/aging) vs classical methods (LBPH, weak on degraded document photos) vs **chosen: liveness/anti-spoof gate first (hard sequential dependency, not parallel) → ArcFace deep embeddings + cosine similarity, chosen for robustness to pose/lighting/aging.**

**Pipeline:** liveness check on live capture (short-circuits with `liveness_passed: false` if it fails, no similarity computed) → face detection + alignment applied identically to document photo crop and live capture → ArcFace embeddings for both → cosine similarity → age-adjusted acceptance threshold using document issue date from OCR (passports valid up to 10 years, so a flat threshold either over-rejects long-time holders or under-catches impostors on fresh documents).
**Output → `face.completed`:** `{similarity_score, liveness_passed, match}`.

---

## 5. Risk Fusion Engine & Fake-ID Scoring (implemented)

**Approaches considered:** simple weighted sum vs black-box neural fusion vs **chosen: explainable weighted/rule-based scorecard with hard overrides**, because an officer denying entry needs a *reason*, not just a black-box number — this directly serves the "standardize screening decisions" and "digital trail" goals.

**Logic (already implemented in Python, `fake_id_scorer.py`):**
- **Hard overrides bypass weighting** for near-certain fraud signals (can't be diluted by otherwise-clean scores):
  - `watchlist_hit` → floor score 95
  - `liveness_passed = False` → floor score 92
- **Otherwise, weighted blend** of four normalized (0–100) components:
  - Tampering: weight 0.35 (highest — core AI innovation)
  - Face mismatch: weight 0.25
  - Validation (rule violations + anomaly): weight 0.20
  - Upstream risk_score (aggregate context): weight 0.20
- **Officer decision = bounded post-hoc adjuster**, not a raw weighted input: accept −10, hold +10, escalate +20. This lets a human catch what the pipeline under-weighted without silently overwriting the audited machine score.
- Output includes `tier` (low/medium/high) and **top 3 contributing factors** for explainability — this is returned to both the officer's dashboard and the audit record.

---

## 6. Decision Dashboard + Audit Ledger

Officer sees: document image, extracted fields, flagged tamper regions overlaid, face-match result, fake-ID score, tier, and top factors — then makes the final accept/hold/escalate call. **The system never auto-denies.**

**Approaches considered for the audit layer:** public blockchain (wrong — PII/biometric data should never sit on a permissionless public ledger; also cost/speed issues) vs plain RDBMS with hash-chaining (admin-editable in practice, weak for investigations) vs **chosen: permissioned blockchain (Hyperledger Fabric).**

A hash of `{ocr_output, validation_result, tamper_result, face_result, risk_score, officer_decision}` is committed on-chain — immutable, timestamped, queryable for investigations. Actual PII/biometric data stays in encrypted off-chain storage, referenced only by hash, keeping the ledger itself privacy-safe while satisfying the "Blockchain & Cybersecurity" theme.

---

## 7. Feasibility & Viability

**Technical feasibility:**
- Every component is proven existing technology (PaddleOCR/LayoutLM, ELA/PRNU forensics, ArcFace, Kafka, Hyperledger Fabric) — no unproven research bets
- The tampering-detection data-scarcity problem is solved architecturally (physically-grounded weak signals need no forged-document training set), not by assuming more data will appear
- Modular design allows phased rollout — OCR + Validation can ship and deliver value before Tampering/Face modules are fully tuned
- On-prem/air-gapped deployability satisfies MHA/SSB data-sovereignty requirements

**Operational viability:**
- Kafka-based scaling directly addresses the stated high-passenger-volume problem
- Human-in-the-loop by design — realistic to deploy without changing legal decision authority
- Explainability (rules kept separate from ML, top-3 factors always returned) supports legal defensibility of flags

**Key risks & mitigations:**
- Document format variability across issuers → extensible per-issuer templates, not hardcoded
- GPU demand at every checkpoint → centralize GPU-heavy inference at regional hubs, run lightweight steps at the edge
- False positives from ELA on legitimately rescanned documents → ensemble design means no single weak signal can trigger a hard flag alone; officer remains final gate
- Legacy checkpoint camera/scanner quality → preprocessing stage explicitly normalizes/quality-checks before AI modules see the image

---

## 8. Impact & Benefits

- Verification time: minutes → seconds per document
- Improved detection of sophisticated forgery beyond manual-inspection capability
- Standardized, repeatable screening decisions across checkpoints/officers
- Shift to data-driven, quantified risk assessment while keeping a human as final decision-maker
- Full investigable digital trail (Kafka replay + immutable blockchain audit record)
- Raises practical difficulty of successful document fraud (multi-signal, harder to predict/defeat than manual checks)
- Immutable audit trail protects both travelers and officers via independent verifiability
- Throughput benefit holds up under peak-load conditions, where manual screening currently degrades most

---

## 9. Current Implementation Status

- ✅ Fake-ID scoring module implemented and tested in Python (`fake_id_scorer.py`): dataclasses for `ValidationResult`, `TamperResult`, `FaceResult`; `compute_fake_id_score()` function with hard overrides, weighted blend, officer adjustment, and full `ScoreBreakdown` output (component scores + top factors) for auditability.
- 📝 Full architecture, Kafka topic map, and module-level data flow have been designed and documented (not yet implemented as running services).
- 📝 Presentation content outlined slide-by-slide for OCR, Validation, Tampering Detection, Face Verification, Kafka backbone, Risk Fusion, Decision/Audit — condensed further into three summary slides: Technical Approach, Feasibility and Viability, Impact and Benefits.

**Not yet decided / open for the next session:** concrete model choices/fine-tuning data sources for LayoutLM and the tampering fusion CNN, exact Hyperledger Fabric channel/chaincode design, UI/UX for the officer dashboard, and load-testing targets for Kafka partition/consumer-group sizing under realistic checkpoint volume.
