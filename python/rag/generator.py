"""Generation layer for RAG claim analysis."""

from __future__ import annotations

import os
from typing import Dict, List


def _template_answer(claim: str, evidence_text: str) -> Dict[str, str]:
    """Simple fallback answer when no LLM is configured."""

    return {
        "verdict": "Uncertain",
        "explanation": (
            "Based on the retrieved evidence, the claim cannot be conclusively "
            "verified or falsified automatically. Review the evidence snippets "
            "for final judgment."
        ),
    }


def generate_answer(claim: str, retrieved_chunks: List[str]) -> Dict[str, str]:
    """Generate structured claim analysis from claim + evidence."""

    evidence_text = "\n\n".join(f"- {chunk}" for chunk in retrieved_chunks)

    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

    if api_key:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
            prompt = (
                "Given the claim and evidence below, determine if the claim is "
                "True, False, or Uncertain and explain why briefly.\n\n"
                f"Claim:\n{claim}\n\n"
                f"Evidence:\n{evidence_text}\n\n"
                "Return strictly as JSON with keys: verdict, explanation."
            )
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a factual verification assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or ""

            import json

            parsed = json.loads(content)
            verdict = str(parsed.get("verdict", "Uncertain"))
            explanation = str(parsed.get("explanation", "No explanation provided."))
            return {"verdict": verdict, "explanation": explanation}
        except Exception:
            pass

    return _template_answer(claim, evidence_text)
