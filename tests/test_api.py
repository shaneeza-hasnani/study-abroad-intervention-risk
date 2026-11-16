import pytest
from pydantic import ValidationError

from src.api import health
from src.schemas import ApplicationCase


def test_health():
    assert health()["status"] == "ok"


def test_validation_rejects_bad_percent():
    with pytest.raises(ValidationError):
        ApplicationCase(
            case_id="BAD-1", snapshot_date="2025-01-01", days_to_departure=30,
            application_completion_pct=150, documents_missing=2,
            days_since_last_contact=5, payment_completion_pct=50,
            payment_attempts_failed=0, prior_staff_contacts=1,
            program_cost_usd=10000, destination_region="Europe",
            application_channel="Web",
        )


