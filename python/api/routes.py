"""API routes for the misinformation detection MVP."""

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class AnalyzeRequest(BaseModel):
    """Request payload for claim analysis."""

    claim: str


@router.post("/analyze")
def analyze(request: AnalyzeRequest):
    """Return a mock analysis response for the provided claim."""

    # MVP scaffold: static response until full RAG is implemented.
    return {
        "verdict": "Likely False",
        "confidence": 0.75,
        "explanation": "This claim is not supported by reliable sources.",
        "evidence": [
            "WHO states no evidence for this claim",
            "Scientific studies contradict it",
        ],
    }
