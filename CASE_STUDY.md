# Case study: prioritizing student outreach without automating eligibility

## The operating problem

A study-abroad consultancy may have many applications moving through document, payment, and departure deadlines at once. Staff need a consistent way to identify cases that could benefit from outreach, but an opaque score should not determine whether a student can participate.

Waypoint frames the task as **intervention prioritization**: estimate which in-progress cases may need staff attention in the next 30 days, show why, and let staff verify the outcome.

This repository is a portfolio demonstration built with synthetic data. It is not a deployed client system, and it does not claim a reduction from 20 reviews to eight.

## My approach

### 1. Define a snapshot-time prediction target

Each training row represents what staff knew on a specific date. The outcome records whether intervention was needed during the following 30 days. Final application status, later payment activity, and post-snapshot communications are excluded from model inputs.

### 2. Evaluate on the future

The data is sorted by snapshot date. The earliest 80% of dates form the training period and the most recent 20% form the holdout period. The boundary is strict: a date cannot appear in both sets.

### 3. Start with an interpretable baseline

The primary model is regularized logistic regression with numeric preprocessing and one-hot encoding inside a single scikit-learn pipeline. The operating threshold is selected separately from probability estimation to reach at least 70% recall on the holdout period.

XGBoost is an optional challenger. It replaces logistic regression only if it improves out-of-time average precision by at least 0.02, preventing unnecessary complexity for a negligible gain.

### 4. Build for a staff decision

The FastAPI service returns a probability, a review/monitor priority, and the strongest feature-level reason codes. The interface makes the model limitation visible and captures a staff outcome in SQLite for later error analysis and retraining.

## Synthetic demonstration results

The seeded synthetic holdout contains approximately 1,000 cases with an intervention prevalence of about 4.6%. The logistic baseline produces:

| Measure | Synthetic holdout result |
|---|---:|
| ROC AUC | 0.685 |
| Average precision | 0.114 |
| Brier score | 0.043 |
| Recall at review threshold | 0.783 |
| Precision at review threshold | 0.066 |

These values demonstrate an end-to-end evaluation workflow. They are not evidence of performance on real applications. The low precision is also operationally important: most flagged synthetic cases would not become positive outcomes, so a real deployment would need a capacity-aware threshold and a cost/benefit study.

## Safeguards and limitations

- The score prioritizes outreach; it cannot approve, reject, or rank student eligibility.
- Protected attributes are not model inputs.
- Staff verify context before acting and record the outcome.
- Feedback is stored for monitoring; it does not trigger automatic online learning.
- Real deployment would require privacy review, access controls, retention limits, subgroup performance audits, drift monitoring, and prospective validation.
- Synthetic data cannot establish real-world accuracy, fairness, or business impact.

## What I would do with real anonymized data

1. Interview operations staff to define a consistent intervention outcome.
2. Audit every field for its exact availability time and remove leakage.
3. Measure missingness, label consistency, seasonality, and policy changes.
4. Use rolling temporal validation and compare calibration across cohorts.
5. Select a threshold based on staff capacity and the relative costs of missed and unnecessary outreach.
6. Run a monitored pilot before making any operational-impact claim.

## Technical stack

Python, pandas, scikit-learn, optional XGBoost, FastAPI, Pydantic, SQLite, pytest, Ruff, Docker, HTML, CSS, and JavaScript.


