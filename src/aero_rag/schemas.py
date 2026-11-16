"""Request, response and LLM-output models shared by the service and the API."""

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field


class AnswerStep(BaseModel):
    number: int
    warnings: list[str] = []  # shown before the step text
    cautions: list[str] = []
    text: str


class LLMAnswer(BaseModel):
    """Exactly what the model must return (enforced by a strict JSON schema)."""

    status: Literal["answered", "cannot_answer"]
    answer: str
    steps: list[AnswerStep]
    source_ids: list[int]


class Citation(BaseModel):
    aircraft_type: str
    task_number: str
    title: str
    revision: int
    revision_date: date
    source_key: str


class Scores(BaseModel):
    hybrid: float | None = None  # normalised hybrid score of the top result (ranking only)
    cosine: float | None = None  # raw cosine similarity of the top result (used to abstain)
    exact_identifier_match: bool = False  # a part/task number in the question was found verbatim


class QueryRequest(BaseModel):
    question: str = Field(max_length=5000)  # hard cap; the guardrail limit is much lower
    aircraft_type: str | None = Field(default=None, max_length=20)


class QueryResponse(BaseModel):
    status: Literal["answered", "abstained", "clarification_needed", "rejected"]
    answer: str
    steps: list[AnswerStep] = []
    citations: list[Citation] = []
    scores: Scores = Scores()
    aircraft_unspecified: bool = False
    clarification_options: list[str] = []
    request_id: str
