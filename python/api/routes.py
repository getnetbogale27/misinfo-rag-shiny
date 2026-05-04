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


def _safe_response(*, result: dict | list | None = None, error: str | None = None) -> dict:
    return {
        "status": "failed" if error else "success",
        "result": result,
        "error": error,
    }


@router.post("/analyze")
def analyze(request: AnalyzeRequest) -> dict:
    """Run the RAG pipeline for the provided claim with safe fallback output."""

    try:
        from rag.pipeline import run_pipeline

        result = run_pipeline(request.claim)
        payload = AnalyzeResponse(**result).model_dump()
        return _safe_response(result=payload, error=None)
    except Exception as exc:
        return _safe_response(result=None, error=str(exc))


@router.post("/evaluate")
def evaluate(request: EvaluateRequest) -> dict:
    """Run the evaluation pipeline against a labeled dataset."""

    try:
        from evaluation.evaluator import run_evaluation

        result = run_evaluation(request.dataset_path)
        return _safe_response(result=result, error=None)
    except Exception as exc:
        return _safe_response(result=None, error=str(exc))


class DatasetStatsRequest(BaseModel):
    dataset_path: str = "data/misinformation_dataset.json"


@router.post("/dataset/stats")
def dataset_stats(request: DatasetStatsRequest) -> dict:
    """Return summary statistics for a misinformation dataset."""

    try:
        result = dataset_statistics(request.dataset_path)
        return _safe_response(result=result, error=None)
    except Exception as exc:
        return _safe_response(result=None, error=str(exc))
