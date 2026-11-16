from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import ARTIFACT_DIR, FEATURES, METRICS_PATH, MODEL_PATH, TARGET, TIME_COLUMN

NUMERIC = FEATURES[:8]
CATEGORICAL = FEATURES[8:]


@dataclass
class Evaluation:
    model: str
    train_rows: int
    test_rows: int
    train_end: str
    test_start: str
    prevalence: float
    roc_auc: float
    average_precision: float
    brier_score: float
    threshold: float
    recall_at_threshold: float
    precision_at_threshold: float


def temporal_split(df: pd.DataFrame, fraction: float = 0.8):
    ordered = df.sort_values(TIME_COLUMN).reset_index(drop=True)
    unique_dates = pd.Series(ordered[TIME_COLUMN].drop_duplicates().sort_values().to_list())
    boundary = unique_dates.iloc[max(1, int(len(unique_dates) * fraction))]
    return ordered[ordered[TIME_COLUMN] < boundary].copy(), ordered[ordered[TIME_COLUMN] >= boundary].copy()


def preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                            ("encode", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", numeric, NUMERIC), ("cat", categorical, CATEGORICAL)])


def _threshold_for_recall(y_true, probabilities, target_recall: float = 0.70) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    valid = np.where(recall[:-1] >= target_recall)[0]
    if not len(valid):
        return 0.5
    # Of thresholds meeting recall, select the one with best precision.
    idx = valid[np.argmax(precision[valid])]
    return float(thresholds[idx])


def train(df: pd.DataFrame, try_xgboost: bool = True):
    train_df, test_df = temporal_split(df)
    candidates: dict[str, object] = {
        "logistic_regression": LogisticRegression(max_iter=1500, C=0.5)
    }
    if try_xgboost:
        try:
            from xgboost import XGBClassifier
            candidates["xgboost"] = XGBClassifier(
                n_estimators=250, max_depth=3, learning_rate=0.04, subsample=0.8,
                colsample_bytree=0.8, eval_metric="logloss", random_state=42,
            )
        except ImportError:
            pass

    evaluations = []
    fitted = {}
    for name, estimator in candidates.items():
        pipe = Pipeline([("prep", preprocessor()), ("model", estimator)])
        pipe.fit(train_df[FEATURES], train_df[TARGET])
        probs = pipe.predict_proba(test_df[FEATURES])[:, 1]
        threshold = _threshold_for_recall(test_df[TARGET], probs)
        preds = probs >= threshold
        tp = int(((preds == 1) & (test_df[TARGET] == 1)).sum())
        evaluations.append(Evaluation(
            model=name, train_rows=len(train_df), test_rows=len(test_df),
            train_end=str(train_df[TIME_COLUMN].max().date()),
            test_start=str(test_df[TIME_COLUMN].min().date()),
            prevalence=float(test_df[TARGET].mean()),
            roc_auc=float(roc_auc_score(test_df[TARGET], probs)),
            average_precision=float(average_precision_score(test_df[TARGET], probs)),
            brier_score=float(brier_score_loss(test_df[TARGET], probs)), threshold=threshold,
            recall_at_threshold=tp / max(1, int((test_df[TARGET] == 1).sum())),
            precision_at_threshold=tp / max(1, int(preds.sum())),
        ))
        fitted[name] = pipe

    # XGBoost is justified only if its out-of-time AP improves by >= 0.02.
    best_name = "logistic_regression"
    scores = {e.model: e.average_precision for e in evaluations}
    if scores.get("xgboost", -1) >= scores["logistic_regression"] + 0.02:
        best_name = "xgboost"
    selected = fitted[best_name]
    selected_eval = next(e for e in evaluations if e.model == best_name)
    metadata = {
        "selected_model": best_name,
        "selection_rule": "XGBoost selected only for >=0.02 out-of-time average-precision gain; otherwise logistic regression.",
        "synthetic_data_notice": "All reported metrics are from fictional synthetic data and are not business results.",
        "evaluations": [asdict(e) for e in evaluations],
        "threshold": selected_eval.threshold,
        "calibration_preview": _calibration_preview(test_df[TARGET], selected.predict_proba(test_df[FEATURES])[:, 1]),
    }
    ARTIFACT_DIR.mkdir(exist_ok=True)
    joblib.dump({"pipeline": selected, "metadata": metadata}, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return selected, metadata


def _calibration_preview(y, probabilities):
    observed, predicted = calibration_curve(y, probabilities, n_bins=5, strategy="quantile")
    return [{"predicted": float(p), "observed": float(o)} for p, o in zip(predicted, observed)]


def top_reasons(pipeline: Pipeline, row: pd.DataFrame, limit: int = 4):
    prep = pipeline.named_steps["prep"]
    model = pipeline.named_steps["model"]
    transformed = prep.transform(row[FEATURES])
    names = prep.get_feature_names_out()
    if hasattr(model, "coef_"):
        contributions = np.asarray(transformed.toarray() if hasattr(transformed, "toarray") else transformed)[0] * model.coef_[0]
    elif hasattr(model, "feature_importances_"):
        contributions = np.asarray(transformed.toarray() if hasattr(transformed, "toarray") else transformed)[0] * model.feature_importances_
    else:
        return []
    order = np.argsort(np.abs(contributions))[::-1][:limit]
    return [{"feature": names[i].replace("num__", "").replace("cat__", ""),
             "direction": "raises" if contributions[i] > 0 else "lowers",
             "contribution": round(float(contributions[i]), 3)} for i in order]


