from src.modeling import temporal_split, train
from src.synthetic import generate


def test_temporal_split_has_no_overlap():
    before, after = temporal_split(generate(200))
    assert before.snapshot_date.max() < after.snapshot_date.min()
    assert len(before) + len(after) == 200


def test_training_produces_probabilities(tmp_path, monkeypatch):
    from src import modeling
    monkeypatch.setattr(modeling, "ARTIFACT_DIR", tmp_path)
    monkeypatch.setattr(modeling, "MODEL_PATH", tmp_path / "model.joblib")
    monkeypatch.setattr(modeling, "METRICS_PATH", tmp_path / "metrics.json")
    model, metadata = train(generate(600), try_xgboost=False)
    probabilities = model.predict_proba(generate(10)[modeling.FEATURES])[:, 1]
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    assert metadata["selected_model"] == "logistic_regression"


