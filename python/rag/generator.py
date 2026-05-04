"""Generation layer for RAG claim analysis."""

from __future__ import annotations

import json
import os
from typing import Dict, List


def _template_answer(language: str) -> Dict[str, str | float]:
    """Simple fallback answer when no LLM is configured."""

    if language == "am":
        return {
            "verdict": "ያልታወቀ",
            "confidence": 0.5,
            "explanation": "በተመለሱ ማስረጃዎች ላይ ብቻ ተመርኮዞ ይህን ጥያቄ በሙሉ ለማረጋገጥ ወይም ለማስተባበል አይቻልም።",
        }

    return {
        "verdict": "Uncertain",
        "confidence": 0.5,
        "explanation": "Based on retrieved evidence only, the claim cannot be conclusively verified or falsified.",
    }


def generate_answer(claim: str, retrieved_chunks: List[str], language: str = "en") -> Dict[str, str | float]:
    """Generate structured claim analysis from claim + evidence."""

    evidence_text = "\n\n".join(f"- {chunk}" for chunk in retrieved_chunks)

    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

    if api_key:
        try:
            from openai import OpenAI

            response_language = "Amharic" if language == "am" else "English"
            prompt = (
                "You are a fact-checking assistant. Use retrieved evidence to evaluate the claim. "
                "Do not hallucinate or add facts not supported by evidence.\n\n"
                f"Respond in {response_language}.\n"
                "Return strict JSON with keys: verdict, explanation, confidence.\n"
                "Confidence must be a number between 0 and 1.\n\n"
                f"Claim:\n{claim}\n\n"
                f"Evidence:\n{evidence_text}"
            )

            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a fact-checking assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content or "{}"
            parsed = json.loads(content)

            verdict = str(parsed.get("verdict", "Uncertain"))
            explanation = str(parsed.get("explanation", "No explanation provided."))
            confidence = float(parsed.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))

            return {"verdict": verdict, "explanation": explanation, "confidence": confidence}
        except Exception:
            pass

    return _template_answer(language)
