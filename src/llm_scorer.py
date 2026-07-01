"""
LLM-based Adverse News Index (ANI) scoring.
Supports Ollama (free, local) and OpenRouter (API-based).
Based on the agentic framework from the AML compliance paper.
"""
import json
import re
from typing import Optional

import httpx

from .config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_MODEL,
)
from .collectors import ScreeningResult, SearchResult


def _format_context_for_llm(screening: ScreeningResult) -> str:
    """Format screening results into context for the LLM."""
    lines = [
        f"Entity: {screening.entity_name}",
        f"Type: {screening.entity_type}",
        "",
        "=== PEP List Check ===",
        f"Match: {'YES - Politically Exposed Person' if screening.pep_match else 'No match'}",
    ]
    if screening.pep_match:
        lines.append(f"Details: {json.dumps(screening.pep_match)}")
    
    lines.extend([
        "",
        "=== Sanction List Check ===",
        f"Match: {'YES - On sanction/watchlist' if screening.sanction_match else 'No match'}",
    ])
    if screening.sanction_match:
        lines.append(f"Details: {json.dumps(screening.sanction_match)}")
    
    lines.extend([
        "",
        "=== Web, News, and Social Media Results ===",
    ])
    
    for r in screening.results[:20]:  # Limit context size
        lines.append(f"\n[{r.source}] {r.title}")
        lines.append(f"URL: {r.url}")
        lines.append(f"Snippet: {r.snippet[:300]}...")
    
    return "\n".join(lines)


def _call_ollama(prompt: str, system_prompt: str) -> str:
    """Call local Ollama API (free)."""
    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }
    with httpx.Client(timeout=120.0) as client:
        resp = client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "")


def _call_openrouter(prompt: str, system_prompt: str) -> str:
    """Call OpenRouter API (requires API key)."""
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not set. Use Ollama for free local inference.")
    
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }
    with httpx.Client(timeout=120.0) as client:
        resp = client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


def call_llm(prompt: str, system_prompt: str, use_ollama: bool = True) -> str:
    """
    Public dispatcher used by the agentic graph. Routes to Ollama (local, free)
    or OpenRouter (API). Raises on failure so callers can decide how to degrade.
    """
    if use_ollama:
        return _call_ollama(prompt, system_prompt)
    return _call_openrouter(prompt, system_prompt)


def extract_json(response: str) -> dict:
    """
    Best-effort extraction of a JSON object from an LLM response, tolerating
    markdown code fences and surrounding prose. Raises ValueError if none found.
    """
    if not response:
        raise ValueError("empty response")
    # Try the largest balanced {...} block first.
    candidates = re.findall(r"\{.*\}", response, re.DOTALL)
    for cand in sorted(candidates, key=len, reverse=True):
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            continue
    return json.loads(response)  # last resort, raises on failure


SYSTEM_PROMPT = """You are an expert AML (Anti-Money Laundering) compliance analyst specializing in adverse media screening.
Your task is to analyze information about an entity (person or company) and produce an Adverse News Index (ANI) score.

The ANI score is a number between 0 and 1 where:
- 0.0 - 0.2: Low risk (clean record, no adverse findings)
- 0.2 - 0.5: Moderate risk (some concerning mentions, PEP status, minor issues)
- 0.5 - 0.8: High risk (significant adverse media, regulatory actions)
- 0.8 - 1.0: Critical risk (sanctioned, major fraud/crime, severe adverse media)

Consider: PEP status, sanction list presence, negative news, fraud/corruption mentions, regulatory penalties.
Apply entity disambiguation: only count findings that clearly refer to the SAME entity, not different people with similar names.

You MUST respond with a valid JSON object in this exact format, nothing else:
{"ani_score": <number 0-1>, "justification": "<2-4 sentence explanation>"}"""


def compute_ani_score(screening: ScreeningResult, use_ollama: bool = True) -> dict:
    """
    Compute Adverse News Index using LLM.
    Returns: {"ani_score": float, "justification": str, "raw_response": str}
    """
    context = _format_context_for_llm(screening)
    prompt = f"""Analyze the following adverse media screening results and compute the Adverse News Index.

{context}

Provide your assessment as JSON: {{"ani_score": <0-1>, "justification": "<explanation>"}}"""

    try:
        if use_ollama:
            response = _call_ollama(prompt, SYSTEM_PROMPT)
        else:
            response = _call_openrouter(prompt, SYSTEM_PROMPT)
    except Exception as e:
        # Fallback: rule-based scoring when LLM unavailable
        return _rule_based_fallback(screening, str(e))

    # Parse JSON from response
    try:
        # Extract JSON block if wrapped in markdown
        json_match = re.search(r"\{[^{}]*\"ani_score\"[^{}]*\}", response, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
        else:
            parsed = json.loads(response)
        
        score = float(parsed.get("ani_score", 0))
        score = max(0, min(1, score))  # Clamp to [0, 1]
        return {
            "ani_score": round(score, 3),
            "justification": parsed.get("justification", "No justification provided."),
            "raw_response": response,
        }
    except (json.JSONDecodeError, ValueError) as e:
        return _rule_based_fallback(screening, f"LLM parse error: {e}. Response: {response[:200]}")


def _rule_based_fallback(screening: ScreeningResult, error_msg: str) -> dict:
    """
    Rule-based ANI when LLM is unavailable.
    Sanction match = 0.9, PEP match = 0.4, adverse results = 0.3 base + 0.1 per result.
    """
    score = 0.0
    reasons = []
    
    if screening.sanction_match:
        score = max(score, 0.9)
        reasons.append("Entity appears on sanction/watchlist")
    if screening.pep_match:
        score = max(score, 0.4)
        reasons.append("Politically Exposed Person")
    
    # Count adverse-looking results (simple heuristic)
    adverse_keywords = ["fraud", "corruption", "money laundering", "arrest", "convicted", "fine", "penalty", "scandal"]
    adverse_count = 0
    for r in screening.results:
        text = (r.title + " " + r.snippet).lower()
        if any(kw in text for kw in adverse_keywords):
            adverse_count += 1
    
    if adverse_count > 0:
        score = min(1.0, score + 0.2 + adverse_count * 0.1)
        reasons.append(f"{adverse_count} potentially adverse media mentions")
    
    if not reasons:
        reasons.append("No significant adverse findings")
    
    return {
        "ani_score": round(min(1.0, score), 3),
        "justification": f"[Rule-based fallback - LLM unavailable: {error_msg}] " + "; ".join(reasons),
        "raw_response": "",
    }
