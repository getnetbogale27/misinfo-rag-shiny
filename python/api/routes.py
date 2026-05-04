"""API routes for the misinformation detection MVP."""

from fastapi import APIRouter
from pydantic import BaseModel, Field

from dataset.stats import dataset_statistics

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


class EvaluateRequest(BaseModel):
    """Request payload for dataset evaluation."""

    dataset_path: str = "data/evaluation_dataset.json"


@router.post("/analyze")
def analyze(request: AnalyzeRequest) -> dict:
    """Run the RAG pipeline for the provided claim with safe fallback output."""

    try:
        from rag.pipeline import run_pipeline

        result = run_pipeline(request.claim)
        return AnalyzeResponse(**result).model_dump()
    except Exception as exc:
        return {"error": str(exc), "fallback": True}


@router.post("/evaluate")
def evaluate(request: EvaluateRequest) -> dict:
    """Run the evaluation pipeline against a labeled dataset."""

    from evaluation.evaluator import run_evaluation

    return run_evaluation(request.dataset_path)


class DatasetStatsRequest(BaseModel):
    dataset_path: str = "data/misinformation_dataset.json"


@router.post("/dataset/stats")
def dataset_stats(request: DatasetStatsRequest) -> dict:
    """Return summary statistics for a misinformation dataset."""

    return dataset_statistics(request.dataset_path)
