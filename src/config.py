from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"
STATIC_DIR = ROOT / "static"
MODEL_PATH = ARTIFACT_DIR / "model.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
FEEDBACK_DB = ROOT / "feedback.db"

FEATURES = [
    "days_to_departure",
    "application_completion_pct",
    "documents_missing",
    "days_since_last_contact",
    "payment_completion_pct",
    "payment_attempts_failed",
    "prior_staff_contacts",
    "program_cost_usd",
    "destination_region",
    "application_channel",
]
TARGET = "needs_intervention_30d"
TIME_COLUMN = "snapshot_date"


