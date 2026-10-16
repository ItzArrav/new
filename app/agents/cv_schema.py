"""The documented output schema of the CV agent (JSON Schema is served at GET /cv/schema)."""
from typing import Literal

from pydantic import BaseModel, Field

from .cv_checks import CATEGORIES

Category = Literal[
    "PSS", "DCS", "Front-End Developer", "Back-End Developer", "Web Automation",
    "Mobile Automation", "AI Engineer", "Chatbot Application Tester", "Other",
]
assert list(Category.__args__) == CATEGORIES


class CategoryScore(BaseModel):
    category: Category
    confidence: float = Field(ge=0, le=1)


class CVClassification(BaseModel):
    primary_category: Category = Field(description="Best-fitting category")
    secondary_categories: list[Category] = Field(default_factory=list, description="Other categories that also fit")
    scores: list[CategoryScore] = Field(description="Confidence 0-1 for EVERY category (9 entries)")
    rationale: str = Field(description="Short explanation citing evidence from the CV")
    evidence: list[str] = Field(default_factory=list, description="Verbatim quotes from the CV supporting the decision")
    injection_detected: bool = Field(default=False, description="True if the CV tried to instruct the classifier")
