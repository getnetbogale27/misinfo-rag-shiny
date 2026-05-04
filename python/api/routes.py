"""API routes for the misinformation detection MVP."""

from fastapi import APIRouter
from pydantic import BaseModel, Field

from rag.pipeline import run_pipeline

router = APIRouter()


class AnalyzeRequest(BaseModel):
    """Request payload for claim analysis."""

    claim: str


class AnalyzeResponse(BaseModel):
    verdict: str
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: str
    evidence: list[str]
    language: str


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    """Run the RAG pipeline for the provided claim."""

    result = run_pipeline(request.claim)
    return AnalyzeResponse(**result)
