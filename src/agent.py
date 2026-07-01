"""
Agentic pipeline for adverse media screening.
Orchestrates: data collection -> LLM scoring -> ANI output.
"""
from dataclasses import dataclass, field
from typing import Optional

from .collectors import run_full_screening, ScreeningResult
from .config import USE_AGENTIC_DEFAULT
from .llm_scorer import compute_ani_score

try:
    from .graph import LANGGRAPH_AVAILABLE, run_graph_screening
except Exception:  # pragma: no cover - import guard
    LANGGRAPH_AVAILABLE = False
    run_graph_screening = None


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


def agentic_available() -> bool:
    """True when the LangGraph agentic RAG pipeline can be used."""
    return bool(LANGGRAPH_AVAILABLE and run_graph_screening is not None)


def run_adverse_news_screening(
    entity_name: str,
    entity_type: str = "person",
    use_ollama: bool = True,
    use_agentic: Optional[bool] = None,
) -> AdverseNewsReport:
    """
    Run the adverse media screening pipeline.

    When `use_agentic` is enabled (and LangGraph is installed) this runs the
    LangGraph agentic RAG pipeline: it fetches full article text, retrieves the
    most risk-relevant passages, lets an LLM decide whether to search again, and
    scores from that grounded evidence. Otherwise it runs the simpler
    collect -> snippet-score pipeline. The agentic path falls back to the simple
    one automatically if anything goes wrong.
    """
    if use_agentic is None:
        use_agentic = USE_AGENTIC_DEFAULT

    if use_agentic and agentic_available():
        try:
            out = run_graph_screening(entity_name, entity_type, use_ollama=use_ollama)
            score = out["ani_score"]
            return AdverseNewsReport(
                entity_name=entity_name,
                entity_type=entity_type,
                ani_score=score,
                risk_level=_score_to_risk(score),
                justification=out["justification"],
                screening=out["screening"],
                ani_details=out["ani_details"],
            )
        except Exception as e:  # noqa: BLE001 - degrade to the simple pipeline
            screening = run_full_screening(entity_name, entity_type)
            screening.errors.append(f"Agentic pipeline failed ({e}); used simple pipeline.")
            ani_details = compute_ani_score(screening, use_ollama=use_ollama)
            ani_details["pipeline"] = "simple_fallback"
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

    # Simple pipeline: collect everything, then score from snippets.
    screening = run_full_screening(entity_name, entity_type)
    ani_details = compute_ani_score(screening, use_ollama=use_ollama)
    ani_details.setdefault("pipeline", "simple")
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
