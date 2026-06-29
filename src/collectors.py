"""
Data collectors for adverse media screening.
All sources use FREE APIs - no paid subscriptions required.
"""
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Optional
from urllib.parse import quote_plus
import csv
import html
import random
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests
from duckduckgo_search import DDGS

# Retry/backoff tuning for DuckDuckGo (which rate-limits automated queries).
DDG_RETRIES = 3
DDG_BACKOFF_BASE = 2.0  # seconds: 2, 4, 8 ...
DDG_INTER_CALL_DELAY = 1.0  # polite pause between sequential DDG calls

from .config import (
    MAX_WEB_RESULTS,
    MAX_NEWS_RESULTS,
    MAX_SOCIAL_RESULTS,
    OPENSANCTIONS_API_KEY,
    OPENSANCTIONS_API_URL,
    OPENSANCTIONS_DIR,
    is_opensanctions_api_configured,
)


@dataclass
class SearchResult:
    """Single search result from any source."""
    source: str  # web, news, pep, sanctions, social
    title: str
    url: str
    snippet: str
    date: Optional[str] = None
    relevance_score: float = 1.0


@dataclass
class ScreeningResult:
    """Aggregated screening results for an entity."""
    entity_name: str
    entity_type: str  # person or company
    results: list[SearchResult] = field(default_factory=list)
    pep_match: Optional[dict] = None
    sanction_match: Optional[dict] = None
    errors: list[str] = field(default_factory=list)  # non-fatal source warnings


def _is_ratelimit(err: Exception) -> bool:
    msg = str(err).lower()
    return "ratelimit" in msg or "403" in msg or "429" in msg or "202" in msg


def _ddgs_call(method: str, query: str, max_results: int) -> list[dict]:
    """
    Call a DuckDuckGo search method with retry + exponential backoff.
    `method` is "text" or "news". Raises the last exception on failure.
    """
    last_err: Optional[Exception] = None
    for attempt in range(DDG_RETRIES):
        try:
            with DDGS() as ddgs:
                fn = getattr(ddgs, method)
                return list(fn(query, max_results=max_results))
        except Exception as e:  # noqa: BLE001 - we re-raise after retries
            last_err = e
            if attempt < DDG_RETRIES - 1 and _is_ratelimit(e):
                # backoff with jitter
                time.sleep(DDG_BACKOFF_BASE * (2 ** attempt) + random.uniform(0, 1))
                continue
            break
    if last_err:
        raise last_err
    return []


def search_web(query: str, max_results: int = MAX_WEB_RESULTS) -> list[SearchResult]:
    """Search the web using DuckDuckGo (free, no API key)."""
    results = []
    try:
        for r in _ddgs_call("text", query, max_results):
            results.append(SearchResult(
                source="web",
                title=r.get("title", ""),
                url=r.get("href", ""),
                snippet=r.get("body", ""),
            ))
    except Exception as e:
        raise RuntimeError(f"web: {e}") from e
    return results


def _google_news_rss(query: str, max_results: int) -> list[SearchResult]:
    """
    Free, key-less news fallback via Google News RSS. More reliable than DDG
    news under rate-limiting.
    """
    url = (
        "https://news.google.com/rss/search?q="
        + quote_plus(query)
        + "&hl=en-US&gl=US&ceid=US:en"
    )
    headers = {"User-Agent": "Mozilla/5.0 (compatible; adverse-news-classifier/1.0)"}
    resp = requests.get(url, headers=headers, timeout=20)
    resp.raise_for_status()
    root = ET.fromstring(resp.content)
    results = []
    for item in root.iter("item"):
        title = item.findtext("title") or ""
        link = item.findtext("link") or ""
        pub = item.findtext("pubDate") or ""
        desc = item.findtext("description") or ""
        # strip HTML tags from the description snippet
        desc = re.sub(r"<[^>]+>", " ", html.unescape(desc)).strip()
        results.append(SearchResult(
            source="news", title=title, url=link, snippet=desc, date=pub,
        ))
        if len(results) >= max_results:
            break
    return results


def search_news(query: str, max_results: int = MAX_NEWS_RESULTS) -> list[SearchResult]:
    """
    Search news. Tries DuckDuckGo News first, then falls back to Google News
    RSS (both free, no API key).
    """
    try:
        results = []
        for r in _ddgs_call("news", query, max_results):
            results.append(SearchResult(
                source="news",
                title=r.get("title", ""),
                url=r.get("url", ""),
                snippet=r.get("body", ""),
                date=r.get("date", ""),
            ))
        if results:
            return results
    except Exception:
        pass  # fall through to RSS fallback

    # Fallback: Google News RSS
    return _google_news_rss(query, max_results)


def search_social_media(query: str, max_results: int = MAX_SOCIAL_RESULTS) -> list[SearchResult]:
    """
    Search social media mentions via DuckDuckGo.
    Searches: Reddit, LinkedIn, X/Twitter (public posts), Mastodon.
    No paid API - uses DuckDuckGo's index of public content.
    """
    results = []
    social_sites = "site:reddit.com OR site:linkedin.com OR site:x.com OR site:twitter.com OR site:mastodon.social"
    full_query = f"{query} ({social_sites})"
    try:
        for r in _ddgs_call("text", full_query, max_results):
            source = "social"
            href = r.get("href", "")
            if "reddit.com" in href:
                source = "social_reddit"
            elif "linkedin.com" in href:
                source = "social_linkedin"
            elif "x.com" in href or "twitter.com" in href:
                source = "social_twitter"
            results.append(SearchResult(
                source=source,
                title=r.get("title", ""),
                url=href,
                snippet=r.get("body", ""),
            ))
    except Exception as e:
        raise RuntimeError(f"social: {e}") from e
    return results


def _normalize_name(name: str) -> str:
    """Normalize name for fuzzy matching."""
    return re.sub(r"\s+", " ", name.lower().strip())


def _get_entity_names(entity: dict) -> list[str]:
    """Return all names (primary + aliases) for a loaded OpenSanctions entity."""
    return entity.get("names", [])


# Module-level cache so the (potentially large) CSV files are parsed only once
# per process, not on every screening call.
_OS_CACHE: Optional[tuple[list[dict], list[dict]]] = None


def _parse_targets_csv(fpath: Path) -> list[dict]:
    """
    Parse an OpenSanctions `targets.simple.csv` file into lightweight entity
    dicts. Columns: id, schema, name, aliases (";"-separated), ..., dataset.
    """
    entities: list[dict] = []
    try:
        with open(fpath, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                names = []
                if row.get("name"):
                    names.append(row["name"])
                if row.get("aliases"):
                    names.extend(a for a in row["aliases"].split(";") if a.strip())
                if not names:
                    continue
                entities.append({
                    "id": row.get("id"),
                    "schema": row.get("schema"),
                    "names": names,
                    "dataset": row.get("dataset", ""),
                    "sanctions": row.get("sanctions", ""),
                })
    except Exception:
        pass
    return entities


def _load_opensanctions_data() -> tuple[list[dict], list[dict]]:
    """
    Load OpenSanctions data from local CSV files (cached after first call).

    Download with: python scripts/download_opensanctions.py
    Files: pep_targets.csv (PEPs) and *_targets.csv / sanction_targets.csv
    (sanctions). Each collection is already topic-filtered, so the file a row
    came from determines its category.
    """
    global _OS_CACHE
    if _OS_CACHE is not None:
        return _OS_CACHE

    pep_entities: list[dict] = []
    sanction_entities: list[dict] = []

    if not OPENSANCTIONS_DIR.exists():
        OPENSANCTIONS_DIR.mkdir(parents=True, exist_ok=True)
        _OS_CACHE = (pep_entities, sanction_entities)
        return _OS_CACHE

    for fpath in OPENSANCTIONS_DIR.glob("*.csv"):
        entities = _parse_targets_csv(fpath)
        if "pep" in fpath.name.lower():
            pep_entities.extend(entities)
        else:
            sanction_entities.extend(entities)

    _OS_CACHE = (pep_entities, sanction_entities)
    return _OS_CACHE


# Minimum confidence (0-1) required to report a watchlist hit. Tune higher for
# fewer false positives, lower to catch more spelling variants.
MATCH_THRESHOLD = 0.85


def _name_tokens(name_norm: str) -> set[str]:
    """Tokenize a normalized name, dropping punctuation and 1-char tokens."""
    cleaned = re.sub(r"[^\w\s]", " ", name_norm)
    return {t for t in cleaned.split() if len(t) > 1}


def _match_score(query_norm: str, query_tokens: set[str], alias_norm: str) -> float:
    """
    Return a 0-1 confidence that `alias_norm` refers to the same entity as the
    query. Conservative: requires full token containment or very high string
    similarity, so unrelated names score ~0.
    """
    if not alias_norm:
        return 0.0

    alias_tokens = _name_tokens(alias_norm)
    # Require full names on BOTH sides: single-token names (e.g. "Mohammed")
    # are far too ambiguous to flag in AML screening.
    if len(query_tokens) < 2 or len(alias_tokens) < 2:
        return 0.0

    if query_norm == alias_norm:
        return 1.0

    str_ratio = SequenceMatcher(None, query_norm, alias_norm).ratio()

    # Every query token appears in the alias (e.g. "john smith" vs
    # "john a. smith") -> strong match.
    if query_tokens.issubset(alias_tokens):
        return max(0.9, str_ratio)
    # Alias fully contained in the query the same way.
    if alias_tokens.issubset(query_tokens):
        return max(0.9, str_ratio)

    # Otherwise fall back to overall string similarity (catches transliteration
    # / minor spelling differences) but only when tokens actually overlap.
    if query_tokens & alias_tokens:
        return str_ratio
    return 0.0


def _find_watchlist_match(name: str, entities: list[dict], category: str) -> Optional[dict]:
    """
    Find the best matching entity on a watchlist. Returns the highest-confidence
    match at/above MATCH_THRESHOLD, or None.
    """
    query_norm = _normalize_name(name)
    query_tokens = _name_tokens(query_norm)
    # Need a full name (>=2 tokens) to screen meaningfully.
    if len(query_tokens) < 2:
        return None

    best = None
    best_score = 0.0
    for entity in entities:
        for alias in _get_entity_names(entity):
            alias_norm = _normalize_name(alias)
            # Cheap pre-filter: skip aliases sharing no token with the query.
            if not (query_tokens & _name_tokens(alias_norm)):
                continue
            score = _match_score(query_norm, query_tokens, alias_norm)
            if score > best_score:
                best_score = score
                best = {
                    "matched_name": alias_norm,
                    "match_confidence": round(score, 3),
                    "entity_id": entity.get("id"),
                    "category": category,
                    "dataset": entity.get("dataset", ""),
                }
                if category == "sanction":
                    best["sanctions"] = entity.get("sanctions", "")
                if score >= 0.999:  # exact match, can't do better
                    return best
    return best if best_score >= MATCH_THRESHOLD else None


def _opensanctions_api_match(name: str, entity_type: str = "person") -> tuple[Optional[dict], Optional[dict]]:
    """
    Query the OpenSanctions matching API and return (pep_match, sanction_match).
    A single API call to the `default` dataset covers both PEP and sanctions.
    Costs EUR 0.10 per successful call. Requires OPENSANCTIONS_API_KEY.

    API docs: https://www.opensanctions.org/api/
    """
    schema = "Person" if entity_type == "person" else "Organization"
    payload = {
        "queries": {
            "q1": {
                "schema": schema,
                "properties": {"name": [name]},
            }
        }
    }
    headers = {
        "Authorization": f"ApiKey {OPENSANCTIONS_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        with requests.Session() as s:
            resp = s.post(
                f"{OPENSANCTIONS_API_URL}/match/default",
                json=payload,
                headers=headers,
                timeout=20,
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        raise RuntimeError(f"OpenSanctions API error: {e}") from e

    results = data.get("responses", {}).get("q1", {}).get("results", [])

    pep_match: Optional[dict] = None
    sanction_match: Optional[dict] = None

    for hit in results:
        if not hit.get("match"):
            continue

        score = float(hit.get("score", 0))
        topics = hit.get("properties", {}).get("topics", [])
        datasets = hit.get("datasets", [])
        entity_id = hit.get("id", "")
        caption = hit.get("caption", name)

        base = {
            "matched_name": caption,
            "match_confidence": round(score, 3),
            "entity_id": entity_id,
            "dataset": ", ".join(datasets),
        }

        is_pep = any("pep" in t.lower() for t in topics)
        is_sanction = any(
            t.lower() in ("sanction", "debarment", "wanted")
            for t in topics
        ) or any(
            kw in d.lower() for d in datasets
            for kw in ("sanction", "ofac", "eu_", "un_", "sdn")
        )

        if is_pep and pep_match is None:
            pep_match = {**base, "category": "pep"}

        if is_sanction and sanction_match is None:
            sanction_match = {
                **base,
                "category": "sanction",
                "sanctions": ", ".join(
                    hit.get("properties", {}).get("program", [])
                ),
            }

        if pep_match and sanction_match:
            break

    return pep_match, sanction_match


def check_pep_list(name: str, entity_type: str = "person") -> Optional[dict]:
    """
    Check if entity appears on PEP list.
    Uses OpenSanctions API when configured, otherwise local CSV bulk data.
    """
    if is_opensanctions_api_configured():
        pep_match, _ = _opensanctions_api_match(name, entity_type)
        return pep_match
    pep_entities, _ = _load_opensanctions_data()
    return _find_watchlist_match(name, pep_entities, "pep")


def check_sanction_list(name: str, entity_type: str = "person") -> Optional[dict]:
    """
    Check if entity appears on a sanctions/watchlist.
    Uses OpenSanctions API when configured, otherwise local CSV bulk data.
    """
    if is_opensanctions_api_configured():
        _, sanction_match = _opensanctions_api_match(name, entity_type)
        return sanction_match
    _, sanction_entities = _load_opensanctions_data()
    return _find_watchlist_match(name, sanction_entities, "sanction")


def run_full_screening(entity_name: str, entity_type: str = "person") -> ScreeningResult:
    """
    Run full adverse media screening across all sources.
    Watchlist checks use the OpenSanctions API when OPENSANCTIONS_API_KEY is set,
    falling back to local CSV files otherwise.
    """
    result = ScreeningResult(entity_name=entity_name, entity_type=entity_type)

    # Each free-search source is attempted independently; a rate-limited source
    # is recorded as a warning rather than failing the whole screening. A short
    # pause between DuckDuckGo calls reduces the chance of being throttled.
    searches = [
        ("Web", search_web),
        ("News", search_news),
        ("Social media", search_social_media),
    ]
    for i, (label, fn) in enumerate(searches):
        try:
            result.results.extend(fn(entity_name))
        except Exception as e:  # noqa: BLE001
            result.errors.append(f"{label} search unavailable ({e}).")
        if i < len(searches) - 1:
            time.sleep(DDG_INTER_CALL_DELAY)

    # 4 & 5. PEP + sanctions — single API call when key configured, two CSV
    # lookups otherwise. Errors are non-fatal: screening proceeds without matches.
    if is_opensanctions_api_configured():
        try:
            result.pep_match, result.sanction_match = _opensanctions_api_match(
                entity_name, entity_type
            )
        except Exception as e:
            result.errors.append(f"OpenSanctions API unavailable ({e}). Watchlist checks skipped.")
    else:
        result.pep_match = check_pep_list(entity_name, entity_type)
        result.sanction_match = check_sanction_list(entity_name, entity_type)
    
    return result
