from __future__ import annotations

import json
import sqlite3
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.config import FEEDBACK_DB, METRICS_PATH, MODEL_PATH, STATIC_DIR
from src.modeling import top_reasons
from src.schemas import ApplicationCase, Feedback


def bundle():
    global _bundle
    if _bundle is None:
        if not MODEL_PATH.exists():
            raise HTTPException(503, "Model artifact missing. Run: python -m src.train")
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def init_db():
    with sqlite3.connect(FEEDBACK_DB) as db:
        db.execute("""CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY, case_id TEXT, predicted_probability REAL,
            staff_outcome TEXT, note TEXT, reviewer TEXT, created_at TEXT
        )""")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    yield


app = FastAPI(title="Waypoint Intervention Risk API", version="0.1.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
_bundle = None


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "model_ready": MODEL_PATH.exists()}


@app.get("/model-card")
def model_card():
    if not METRICS_PATH.exists():
        raise HTTPException(404, "Train the model first")
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


@app.post("/predict")
def predict(case: ApplicationCase):
    loaded = bundle()
    row = pd.DataFrame([case.model_dump()])
    probability = float(loaded["pipeline"].predict_proba(row)[:, 1][0])
    threshold = loaded["metadata"]["threshold"]
    return {
        "case_id": case.case_id,
        "intervention_probability": round(probability, 4),
        "priority": "review" if probability >= threshold else "monitor",
        "threshold": round(threshold, 4),
        "reason_codes": top_reasons(loaded["pipeline"], row),
        "decision_notice": "Decision support only. Staff must verify context before outreach.",
        "data_notice": "Demo model trained on synthetic data; not validated for operational use.",
    }


@app.post("/feedback", status_code=201)
def feedback(item: Feedback):
    with sqlite3.connect(FEEDBACK_DB) as db:
        cursor = db.execute(
            "INSERT INTO feedback(case_id,predicted_probability,staff_outcome,note,reviewer,created_at) VALUES(?,?,?,?,?,?)",
            (item.case_id, item.predicted_probability, item.staff_outcome, item.note,
             item.reviewer, datetime.now(UTC).isoformat()),
        )
    return {"id": cursor.lastrowid, "status": "recorded"}


