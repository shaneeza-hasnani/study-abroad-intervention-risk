from datetime import date
from typing import Literal

from pydantic import BaseModel, Field


class ApplicationCase(BaseModel):
    case_id: str = Field(min_length=1, max_length=80)
    snapshot_date: date
    days_to_departure: int = Field(ge=0, le=730)
    application_completion_pct: float = Field(ge=0, le=100)
    documents_missing: int = Field(ge=0, le=20)
    days_since_last_contact: int = Field(ge=0, le=365)
    payment_completion_pct: float = Field(ge=0, le=100)
    payment_attempts_failed: int = Field(ge=0, le=20)
    prior_staff_contacts: int = Field(ge=0, le=50)
    program_cost_usd: float = Field(ge=0, le=100000)
    destination_region: Literal["Europe", "Asia", "Latin America", "Oceania", "Africa"]
    application_channel: Literal["Advisor referral", "Web", "University partner", "Event"]


class Feedback(BaseModel):
    case_id: str
    predicted_probability: float = Field(ge=0, le=1)
    staff_outcome: Literal["intervention_needed", "no_intervention_needed", "follow_up_later"]
    note: str = Field(default="", max_length=1000)
    reviewer: str = Field(min_length=1, max_length=100)


