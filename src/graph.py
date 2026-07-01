"""
Agentic adverse-media screening as a LangGraph state machine.

Flow:
    watchlist -> search -> fetch+index(RAG) -> retrieve -> decide ─┐
                   ▲                                               │
                   └────────────── refine_search ◄── search_deeper │
                                                                   ▼
                                                                 score -> END

The `decide` node is where genuine agency lives: an LLM inspects the retrieved
evidence and chooses whether to finalise the assessment or reformulate the query
and search again (bounded by AGENT_MAX_ITERATIONS). Scoring is RAG-grounded:
the model reasons over retrieved passages and cites them, mirroring (and
extending) the identity + negativity cross-check from the Luxembourg AMI paper.
"""
from __future__ import annotations

from typing import Optional, TypedDict

from .collectors import (
    SearchResult,
    ScreeningResult,
    check_pep_list,
    check_sanction_list,
    search_news,
    search_social_media,
    search_web,
    _opensanctions_api_match,
)
from .config import (
    AGENT_MAX_ITERATIONS,
    RAG_TOP_K,
    is_opensanctions_api_configured,
)
from .llm_scorer import call_llm, extract_json
from .rag import (
    Chunk,
    Document,
    adverse_probe_queries,
    build_chunks,
    fetch_documents,
    get_retriever,
)

try:
    from langgraph.graph import END, StateGraph

    LANGGRAPH_AVAILABLE = True
except Exception:  # pragma: no cover - import guard
    LANGGRAPH_AVAILABLE = False


class ScreeningState(TypedDict, total=False):
    entity_name: str
    entity_type: str
    use_ollama: bool

    current_query: str
    queries_tried: list[str]
    raw_results: list  # list[SearchResult]

    chunks: list  # list[Chunk]
    retriever_backend: str
    retrieved: list  # list[Chunk]

    pep_match: Optional[dict]
    sanction_match: Optional[dict]

    iteration: int
    decision: str
    decision_reason: str

    identity_score: float
    negativity_score: float
    ani_score: float
    justification: str

    agent_trace: list  # list[str]
    errors: list  # list[str]


# --------------------------------------------------------------------------- #
# Node implementations
# --------------------------------------------------------------------------- #
def watchlist_node(state: ScreeningState) -> dict:
    """Deterministic PEP + sanctions screening (OpenSanctions API or CSV)."""
    name = state["entity_name"]
    etype = state.get("entity_type", "person")
    trace = list(state.get("agent_trace", []))
    errors = list(state.get("errors", []))
    pep = sanc = None
    try:
        if is_opensanctions_api_configured():
            pep, sanc = _opensanctions_api_match(name, etype)
        else:
            pep = check_pep_list(name, etype)
            sanc = check_sanction_list(name, etype)
    except Exception as e:  # noqa: BLE001
        errors.append(f"Watchlist check unavailable ({e}).")
    hits = []
    if pep:
        hits.append("PEP")
    if sanc:
        hits.append("sanctions")
    trace.append(f"Watchlist screened — {'hit: ' + ', '.join(hits) if hits else 'no hits'}.")
    return {"pep_match": pep, "sanction_match": sanc, "agent_trace": trace, "errors": errors}


def _adverse_query(entity: str) -> str:
    """Adverse-focused query to surface risk content on the first pass."""
    return (
        f"{entity} fraud OR corruption OR sanctions OR investigation "
        "OR lawsuit OR arrest OR scandal OR misconduct"
    )


def search_node(state: ScreeningState) -> dict:
    """Run free web/news/social searches for the current query and accumulate."""
    entity = state["entity_name"]
    query = state.get("current_query") or entity
    trace = list(state.get("agent_trace", []))
    errors = list(state.get("errors", []))
    existing = list(state.get("raw_results", []))
    queries_tried = list(state.get("queries_tried", []))
    first_pass = not queries_tried

    # (label, search_fn, query). On the first pass, add an adverse-focused web
    # search so dirty entities surface evidence immediately — this stabilises
    # scoring and reduces reliance on the deeper-search loop.
    tasks = [
        ("web", search_web, query),
        ("news", search_news, query),
        ("social", search_social_media, query),
    ]
    if first_pass:
        tasks.insert(1, ("web-adverse", search_web, _adverse_query(entity)))

    seen_urls = {getattr(r, "url", "") for r in existing}
    found = 0
    ran_queries: list[str] = []
    for label, fn, q in tasks:
        if q not in ran_queries and q not in queries_tried:
            ran_queries.append(q)
        try:
            for r in fn(q):
                if r.url and r.url in seen_urls:
                    continue
                seen_urls.add(r.url)
                existing.append(r)
                found += 1
        except Exception as e:  # noqa: BLE001
            errors.append(f"{label} search unavailable ({e}).")

    queries_tried.extend(ran_queries)
    trace.append(f'Searched {ran_queries} — {found} new result(s).')
    return {
        "raw_results": existing,
        "queries_tried": queries_tried,
        "agent_trace": trace,
        "errors": errors,
    }


def fetch_index_node(state: ScreeningState) -> dict:
    """
    Build the retrieval corpus. Full article text is fetched where possible, but
    many news sites block bots — so we ALWAYS also index the search-result
    snippets. This guarantees the scorer is grounded in real retrieved text
    (never zero passages -> never parametric guessing) even when fetch fails.
    """
    results = state.get("raw_results", [])
    trace = list(state.get("agent_trace", []))
    errors = list(state.get("errors", []))

    documents, warns = fetch_documents(results)
    errors.extend(warns)
    fetched_urls = {d.url for d in documents}

    # Snippet fallback: index title+snippet for every result we did NOT fetch.
    snippet_docs: list[Document] = []
    for r in results:
        url = getattr(r, "url", "") or ""
        if url and url in fetched_urls:
            continue
        text = f"{getattr(r, 'title', '')}. {getattr(r, 'snippet', '')}".strip(". ").strip()
        if len(text) < 20:
            continue
        snippet_docs.append(
            Document(url=url, title=getattr(r, "title", ""),
                     source=getattr(r, "source", "web"), text=text)
        )

    chunks = build_chunks(documents + snippet_docs)
    trace.append(
        f"Indexed {len(documents)} full article(s) + {len(snippet_docs)} "
        f"snippet(s) into {len(chunks)} passage(s)."
    )
    return {"chunks": chunks, "agent_trace": trace, "errors": errors}


def retrieve_node(state: ScreeningState) -> dict:
    """Retrieve the passages most relevant to adverse-risk signals."""
    chunks: list[Chunk] = state.get("chunks", [])
    trace = list(state.get("agent_trace", []))
    retriever, backend = get_retriever(chunks, use_ollama=state.get("use_ollama", False))
    retrieved: list[Chunk] = []
    if retriever is not None:
        retrieved = retriever.retrieve(adverse_probe_queries(state["entity_name"]), RAG_TOP_K)
    trace.append(
        f"Retrieved {len(retrieved)} relevant passage(s) via {backend}."
    )
    return {"retrieved": retrieved, "retriever_backend": backend, "agent_trace": trace}


# Adverse-signal lexicon shared by the "should we dig deeper?" gate and the
# rule-based fallback scorer.
ADVERSE_KEYWORDS = [
    "fraud", "corruption", "money laundering", "launder", "bribery", "bribe",
    "arrest", "arrested", "convicted", "conviction", "charged", "indicted",
    "guilty", "fine", "penalty", "sanction", "sanctioned", "scandal",
    "investigation", "investigated", "lawsuit", "sued", "embezzle",
    "terror", "trafficking", "smuggling", "ponzi", "misconduct", "allegation",
]


def _adverse_signal_count(retrieved) -> int:
    """How many retrieved passages contain an adverse-risk keyword."""
    n = 0
    for c in retrieved:
        text = c.text.lower()
        if any(k in text for k in ADVERSE_KEYWORDS):
            n += 1
    return n


DECIDE_SYSTEM = (
    "You are an AML analyst deciding whether to keep investigating. Search "
    "DEEPER ONLY to chase a SPECIFIC lead already present in the evidence — e.g. "
    "to confirm/deny a named allegation, or to disambiguate the entity from a "
    "namesake. Do NOT search deeper merely hoping to find adverse content that "
    "the current evidence does not suggest. If the evidence shows no credible "
    "adverse signal, FINALIZE. Respond ONLY with JSON."
)


def decide_node(state: ScreeningState) -> dict:
    """
    Agentic control point: LLM decides finalize vs. search again, and may
    reformulate the query. Bounded by AGENT_MAX_ITERATIONS. Falls back to a
    heuristic if the LLM is unavailable.
    """
    iteration = state.get("iteration", 0)
    trace = list(state.get("agent_trace", []))
    retrieved: list[Chunk] = state.get("retrieved", [])
    entity = state["entity_name"]

    # Hard stop: never exceed the iteration budget.
    if iteration >= AGENT_MAX_ITERATIONS:
        trace.append("Decision: finalize (iteration budget reached).")
        return {"decision": "finalize", "decision_reason": "iteration budget reached", "agent_trace": trace}

    evidence_preview = "\n".join(
        f"[{i+1}] ({c.source}) {c.text[:220]}" for i, c in enumerate(retrieved[:6])
    ) or "(no passages retrieved yet)"
    wl = []
    if state.get("pep_match"):
        wl.append("PEP match")
    if state.get("sanction_match"):
        wl.append("sanctions match")
    wl_str = ", ".join(wl) if wl else "none"

    prompt = f"""Entity under screening: "{entity}" (type: {state.get('entity_type','person')})
Watchlist findings: {wl_str}
Queries already run: {state.get('queries_tried', [])}
Retrieved evidence passages:
{evidence_preview}

Is this enough to assign an adverse-media risk score with confidence? If a
specific follow-up search would clearly help (e.g. to disambiguate identity or
chase a named company/allegation), request it.

Respond as JSON:
{{"action": "finalize" | "search_deeper",
  "reason": "<one sentence>",
  "refined_query": "<new search query, only if action is search_deeper>"}}"""

    decision = "finalize"
    reason = ""
    refined = ""
    try:
        raw = call_llm(prompt, DECIDE_SYSTEM, use_ollama=state.get("use_ollama", False))
        parsed = extract_json(raw)
        decision = "search_deeper" if str(parsed.get("action")) == "search_deeper" else "finalize"
        reason = str(parsed.get("reason", ""))[:300]
        refined = str(parsed.get("refined_query", "")).strip()
    except Exception:
        # Heuristic fallback (LLM unavailable): we only reach here when an
        # adverse signal exists. Finalize on what we have rather than looping.
        decision, reason = "finalize", "adverse evidence present; scoring (heuristic)."

    if decision == "search_deeper" and not refined:
        refined = f"{entity} fraud OR investigation OR sanction OR lawsuit"

    trace.append(f"Decision: {decision} — {reason}")
    out = {"decision": decision, "decision_reason": reason, "agent_trace": trace}
    if decision == "search_deeper":
        out["current_query"] = refined
    return out


def refine_node(state: ScreeningState) -> dict:
    """Increment the iteration counter before looping back to search."""
    iteration = state.get("iteration", 0) + 1
    trace = list(state.get("agent_trace", []))
    trace.append(f'Refined search (iteration {iteration}): "{state.get("current_query","")}"')
    return {"iteration": iteration, "agent_trace": trace}


SCORE_SYSTEM = """You are an expert AML adverse-media analyst. Using ONLY the\
 provided evidence passages and watchlist findings, assess an entity.

Work in three steps and return all of them:
1. identity: how confident are you the adverse passages refer to THIS entity
   (not a namesake)? score 0-1.
2. negativity: how severe/credible is the adverse content? score 0-1.
3. ani_score: the final Adverse News Index 0-1, reconciling identity and
   negativity (low identity confidence must cap the ANI).

Scale: 0.0-0.2 low, 0.2-0.5 moderate, 0.5-0.8 high, 0.8-1.0 critical.
A sanctions hit alone warrants >=0.8; a PEP-only profile is moderate.

Respond ONLY with JSON:
{"identity": {"score": <0-1>, "justification": "<text>"},
 "negativity": {"score": <0-1>, "justification": "<text>"},
 "ani_score": <0-1>,
 "justification": "<2-4 sentences citing passage numbers like [2]>"}"""


def _build_score_context(state: ScreeningState) -> str:
    retrieved: list[Chunk] = state.get("retrieved", [])
    lines = [
        f'Entity: "{state["entity_name"]}" (type: {state.get("entity_type","person")})',
        f"PEP watchlist: {'MATCH ' + str(state['pep_match']) if state.get('pep_match') else 'no match'}",
        f"Sanctions watchlist: {'MATCH ' + str(state['sanction_match']) if state.get('sanction_match') else 'no match'}",
        "",
        "Evidence passages (cite by number):",
    ]
    if retrieved:
        for i, c in enumerate(retrieved):
            lines.append(f"[{i+1}] ({c.source}, {c.url})\n{c.text[:600]}")
    else:
        lines.append("(no article passages retrieved — rely on watchlist findings only)")
    return "\n".join(lines)


def _rule_based_score(state: ScreeningState) -> dict:
    """Grounded fallback when the LLM is unavailable or unparseable."""
    retrieved: list[Chunk] = state.get("retrieved", [])
    score = 0.0
    reasons = []
    if state.get("sanction_match"):
        score = max(score, 0.9)
        reasons.append("entity on a sanctions/watchlist")
    if state.get("pep_match"):
        score = max(score, 0.4)
        reasons.append("politically exposed person")
    adverse_hits = _adverse_signal_count(retrieved)
    if adverse_hits:
        score = min(1.0, score + 0.2 + adverse_hits * 0.08)
        reasons.append(f"{adverse_hits} adverse passage(s) retrieved")
    if not reasons:
        reasons.append("no significant adverse findings in retrieved evidence")
    return {
        "identity_score": 0.5,
        "negativity_score": round(min(1.0, score), 3),
        "ani_score": round(min(1.0, score), 3),
        "justification": "[Rule-based fallback — LLM unavailable] " + "; ".join(reasons),
    }


def score_node(state: ScreeningState) -> dict:
    """RAG-grounded scoring: identity + negativity cross-check -> final ANI."""
    trace = list(state.get("agent_trace", []))
    retrieved: list[Chunk] = state.get("retrieved", [])
    has_watchlist = bool(state.get("pep_match") or state.get("sanction_match"))

    # Never let the LLM invent a score from parametric memory: if we have no
    # retrieved evidence AND no watchlist hit, the sources were unavailable.
    if not retrieved and not has_watchlist:
        trace.append("No evidence retrieved and no watchlist hit — insufficient evidence.")
        return {
            "identity_score": 0.0,
            "negativity_score": 0.0,
            "ani_score": 0.0,
            "justification": (
                "INSUFFICIENT EVIDENCE: no watchlist match and no media passages "
                "could be retrieved (search sources may be rate-limited or "
                "unavailable). This is NOT a clean bill of health — re-run the "
                "screening to obtain coverage."
            ),
            "agent_trace": trace,
        }

    context = _build_score_context(state)
    prompt = f"{context}\n\nProduce the JSON assessment now."
    try:
        raw = call_llm(prompt, SCORE_SYSTEM, use_ollama=state.get("use_ollama", False))
        parsed = extract_json(raw)
        identity = float(parsed.get("identity", {}).get("score", 0.5))
        negativity = float(parsed.get("negativity", {}).get("score", 0.0))
        ani = float(parsed.get("ani_score", 0.0))
        ani = max(0.0, min(1.0, ani))
        justification = str(parsed.get("justification", "No justification provided."))
        trace.append("Scored entity via RAG-grounded LLM assessment.")
        return {
            "identity_score": round(max(0.0, min(1.0, identity)), 3),
            "negativity_score": round(max(0.0, min(1.0, negativity)), 3),
            "ani_score": round(ani, 3),
            "justification": justification,
            "agent_trace": trace,
        }
    except Exception as e:  # noqa: BLE001
        trace.append(f"LLM scoring unavailable ({e}); used rule-based fallback.")
        fb = _rule_based_score(state)
        fb["agent_trace"] = trace
        return fb


# --------------------------------------------------------------------------- #
# Graph assembly
# --------------------------------------------------------------------------- #
def _route_after_decide(state: ScreeningState) -> str:
    return "refine_search" if state.get("decision") == "search_deeper" else "score"


def build_graph():
    """Compile and return the LangGraph screening agent."""
    if not LANGGRAPH_AVAILABLE:
        raise RuntimeError("langgraph is not installed")
    g = StateGraph(ScreeningState)
    g.add_node("watchlist", watchlist_node)
    g.add_node("search", search_node)
    g.add_node("fetch_index", fetch_index_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("decide", decide_node)
    g.add_node("refine_search", refine_node)
    g.add_node("score", score_node)

    g.set_entry_point("watchlist")
    g.add_edge("watchlist", "search")
    g.add_edge("search", "fetch_index")
    g.add_edge("fetch_index", "retrieve")
    g.add_edge("retrieve", "decide")
    g.add_conditional_edges(
        "decide", _route_after_decide, {"refine_search": "refine_search", "score": "score"}
    )
    g.add_edge("refine_search", "search")
    g.add_edge("score", END)
    return g.compile()


_COMPILED = None


def _get_compiled():
    global _COMPILED
    if _COMPILED is None:
        _COMPILED = build_graph()
    return _COMPILED


def run_graph_screening(
    entity_name: str, entity_type: str = "person", use_ollama: bool = False
) -> dict:
    """
    Execute the agentic RAG pipeline and return a dict compatible with
    AdverseNewsReport construction:
        {"screening": ScreeningResult, "ani_score": float,
         "justification": str, "ani_details": dict}
    """
    app = _get_compiled()
    initial: ScreeningState = {
        "entity_name": entity_name,
        "entity_type": entity_type,
        "use_ollama": use_ollama,
        "current_query": entity_name,
        "queries_tried": [],
        "raw_results": [],
        "iteration": 0,
        "agent_trace": [],
        "errors": [],
    }
    # Recursion limit guards against any unexpected loop; budget * nodes + slack.
    final: ScreeningState = app.invoke(
        initial, config={"recursion_limit": 50}
    )

    screening = ScreeningResult(
        entity_name=entity_name,
        entity_type=entity_type,
        results=final.get("raw_results", []),
        pep_match=final.get("pep_match"),
        sanction_match=final.get("sanction_match"),
        errors=final.get("errors", []),
    )
    retrieved: list[Chunk] = final.get("retrieved", [])
    ani_details = {
        "ani_score": final.get("ani_score", 0.0),
        "justification": final.get("justification", ""),
        "identity_score": final.get("identity_score"),
        "negativity_score": final.get("negativity_score"),
        "retriever_backend": final.get("retriever_backend", "none"),
        "iterations": final.get("iteration", 0),
        "queries_tried": final.get("queries_tried", []),
        "agent_trace": final.get("agent_trace", []),
        "retrieved_passages": [
            {"source": c.source, "url": c.url, "title": c.title,
             "score": c.score, "text": c.text}
            for c in retrieved
        ],
        "pipeline": "agentic_rag",
    }
    return {
        "screening": screening,
        "ani_score": final.get("ani_score", 0.0),
        "justification": final.get("justification", ""),
        "ani_details": ani_details,
    }
