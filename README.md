# Waypoint: Study-Abroad Intervention Risk

Waypoint is a portfolio-ready decision-support project for a fictional study-abroad consultancy. It estimates whether an in-progress application may need staff intervention within 30 days, explains the score, and records staff-verified outcomes.

> **Evidence boundary:** the included dataset is synthetic. Metrics produced by this repository are demonstration metrics, not consultancy results. This project does **not** claim that manual reviews fell from 20 to 8—or any other operational impact—without verified historical evidence.

## What this demonstrates

- Snapshot-time feature design and a chronological 80/20 evaluation split to prevent future information leaking into training.
- An interpretable, regularized logistic regression baseline, with the review threshold tuned separately from probability estimation.
- Optional XGBoost comparison. It is selected only when out-of-time average precision improves by at least 0.02; otherwise the simpler baseline remains.
- ROC AUC, average precision, Brier score, calibration preview, and an operating threshold selected for at least 70% recall on the holdout period.
- Per-prediction reason codes, an API, a staff verification workflow, and SQLite feedback storage.
- Explicit product guardrails: prioritization only, human review, no protected attributes, and no automated eligibility decision.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -e ".[dev,xgboost]"
python -m src.synthetic
python -m src.train
uvicorn src.api:app --reload
```

Open `http://localhost:8000`. API documentation is at `http://localhost:8000/docs`.

To run without XGBoost: `python -m src.train --no-xgboost`.

## Data contract

Each row must represent what was known on a single `snapshot_date`. The target, `needs_intervention_30d`, is observed only after that snapshot. Never include later staff outcomes, final payment status, cancellation status, or post-snapshot communications as input features.

The synthetic generator creates fictional IDs and values plus `data_provenance=synthetic_demo_only`. For real anonymized data, map fields to the schema in `src/config.py`, document each field's availability time, remove direct identifiers, define the intervention outcome with operations staff, and train with:

```bash
python -m src.train --data path/to/anonymized_snapshots.csv
```

Before operational use, add group-level performance and calibration audits using legally and ethically approved audit attributes kept out of model features; validate drift, capacity-aware thresholds, adverse impacts, consent, retention, access control, and escalation procedures.

## Tests

```bash
pytest
ruff check .
```

## Deployment

```bash
docker build -t waypoint-risk .
docker run --rm -p 8000:8000 -v waypoint-feedback:/app waypoint-risk
```

For production, store model artifacts and feedback in managed services, add authentication and role-based access, encrypt data in transit and at rest, avoid logging student records, pin an approved model version, add monitoring/alerts, and deploy behind a managed HTTPS gateway. The demo is intentionally not production-ready for student data.

## Portfolio walkthrough

1. Explain why the outcome is an intervention need, not enrollment eligibility.
2. Show the temporal split and discuss leakage risks.
3. Compare model metrics and justify the selection rule.
4. Score a case, read the reason codes, and record a staff outcome.
5. Discuss how feedback would support error analysis, drift monitoring, and retraining—not automatic online learning.


