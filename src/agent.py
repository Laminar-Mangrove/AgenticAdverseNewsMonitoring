"""
Agentic pipeline for adverse media screening.
Orchestrates: data collection -> LLM scoring -> ANI output.
"""
from dataclasses import dataclass, field
from typing import Optional

from .collectors import run_full_screening, ScreeningResult
from .llm_scorer import compute_ani_score


@dataclass
class AdverseNewsReport:
    """Final adverse news screening report."""
    entity_name: str
    entity_type: str
    ani_score: float
    risk_level: str
    justification: str
    screening: ScreeningResult
    ani_details: dict = field(default_factory=dict)


def _score_to_risk(score: float) -> str:
    """Map ANI score to risk level."""
    if score < 0.2:
        return "Low"
    if score < 0.5:
        return "Moderate"
    if score < 0.8:
        return "High"
    return "Critical"


def run_adverse_news_screening(
    entity_name: str,
    entity_type: str = "person",
    use_ollama: bool = True,
) -> AdverseNewsReport:
    """
    Run the full agentic adverse media screening pipeline.
    
    1. Collect data from web, news, social media, PEP list, sanction list
    2. Compute Adverse News Index via LLM (or rule-based fallback)
    3. Return structured report
    """
    # Step 1: Data collection
    screening = run_full_screening(entity_name, entity_type)
    
    # Step 2: LLM scoring
    ani_details = compute_ani_score(screening, use_ollama=use_ollama)
    score = ani_details["ani_score"]
    
    return AdverseNewsReport(
        entity_name=entity_name,
        entity_type=entity_type,
        ani_score=score,
        risk_level=_score_to_risk(score),
        justification=ani_details["justification"],
        screening=screening,
        ani_details=ani_details,
    )
