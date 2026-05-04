"""API routes for the misinformation detection MVP."""

from fastapi import APIRouter
from pydantic import BaseModel

from rag.pipeline import run_pipeline

router = APIRouter()


class AnalyzeRequest(BaseModel):
    """Request payload for claim analysis."""

    claim: str


@router.post("/analyze")
def analyze(request: AnalyzeRequest):
    """Run the RAG pipeline for the provided claim."""

    return run_pipeline(request.claim)
