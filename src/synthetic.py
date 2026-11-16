"""Synthetic data generator. Values are fictional and not performance evidence."""

import numpy as np
import pandas as pd

from src.config import DATA_DIR


def generate(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.to_datetime("2023-01-01") + pd.to_timedelta(rng.integers(0, 730, n), unit="D")
    days_to_departure = rng.integers(14, 240, n)
    completion = np.clip(rng.beta(5, 2, n) * 100, 5, 100).round(1)
    missing = np.clip(np.round((100 - completion) / 20 + rng.normal(0, 0.8, n)), 0, 6).astype(int)
    last_contact = np.clip(rng.gamma(2.2, 5, n), 0, 60).round().astype(int)
    payment = np.clip(rng.beta(3, 2, n) * 100, 0, 100).round(1)
    failed = np.clip(rng.poisson(0.45, n), 0, 5)
    contacts = np.clip(rng.poisson(1.4, n), 0, 8)
    cost = np.clip(rng.normal(11200, 3800, n), 2500, 28000).round(0)
    regions = rng.choice(["Europe", "Asia", "Latin America", "Oceania", "Africa"], n,
                         p=[0.38, 0.22, 0.16, 0.14, 0.10])
    channels = rng.choice(["Advisor referral", "Web", "University partner", "Event"], n,
                          p=[0.28, 0.35, 0.25, 0.12])

    # Outcome uses only information available at the snapshot. A small temporal drift
    # deliberately makes the out-of-time evaluation more realistic.
    drift = ((dates - pd.Timestamp("2023-01-01")).days / 730) * 0.35
    logit = (
        -2.25 + 0.045 * missing + 0.026 * last_contact + 0.42 * failed
        - 0.016 * payment - 0.010 * completion - 0.0022 * days_to_departure
        + 0.08 * contacts + drift + rng.normal(0, 0.65, n)
    )
    probability = 1 / (1 + np.exp(-logit))
    target = rng.binomial(1, probability)
    frame = pd.DataFrame({
        "case_id": [f"SYN-{i:06d}" for i in range(n)],
        "snapshot_date": dates,
        "days_to_departure": days_to_departure,
        "application_completion_pct": completion,
        "documents_missing": missing,
        "days_since_last_contact": last_contact,
        "payment_completion_pct": payment,
        "payment_attempts_failed": failed,
        "prior_staff_contacts": contacts,
        "program_cost_usd": cost,
        "destination_region": regions,
        "application_channel": channels,
        "needs_intervention_30d": target,
        "data_provenance": "synthetic_demo_only",
    }).sort_values("snapshot_date").reset_index(drop=True)
    return frame


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / "synthetic_cases.csv"
    generate().to_csv(path, index=False)
    print(f"Wrote clearly labeled synthetic data to {path}")


if __name__ == "__main__":
    main()


