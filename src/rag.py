"""
Retrieval-Augmented Generation (RAG) layer for adverse media screening.

Unlike snippet-only approaches, this module fetches the *full* article text
behind each search result, splits it into overlapping chunks, vectorises the
chunks, and retrieves the passages most relevant to adverse-risk signals. Those
grounded passages (with source URLs) are what the LLM scorer reasons over.

Retrieval is tiered and cost-aware:
  * TF-IDF (scikit-learn) by default — free, lightweight, works on Streamlit Cloud.
  * Optional dense embeddings via local Ollama — semantic retrieval when available.
"""
from __future__ import annotations

import concurrent.futures
import math
import re
from dataclasses import dataclass
from typing import Optional

import requests
from bs4 import BeautifulSoup

from .config import (
    OLLAMA_BASE_URL,
    OLLAMA_EMBED_MODEL,
    RAG_CHUNK_OVERLAP,
    RAG_CHUNK_WORDS,
    RAG_FETCH_TIMEOUT,
    RAG_MAX_DOCS,
    RAG_TOP_K,
)

_FETCH_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; adverse-news-classifier/1.0; +https://opensanctions.org)"
}
_MAX_BYTES = 2_000_000  # skip very large pages to stay fast and memory-safe


@dataclass
class Chunk:
    """A retrievable passage of source text with provenance."""
    text: str
    url: str
    title: str
    source: str
    score: float = 0.0


@dataclass
class Document:
    """Full extracted text of one fetched source."""
    url: str
    title: str
    source: str
    text: str


# --------------------------------------------------------------------------- #
# 1. Document fetching + extraction
# --------------------------------------------------------------------------- #
def fetch_article(url: str, title: str, source: str, timeout: float) -> Optional[Document]:
    """
    Fetch a URL and extract readable article text. Returns None on any failure
    (paywall, binary content, timeout, etc.) — failures are expected and benign.
    """
    if not url or not url.startswith(("http://", "https://")):
        return None
    try:
        resp = requests.get(
            url, headers=_FETCH_HEADERS, timeout=timeout, stream=True
        )
        resp.raise_for_status()
        ctype = resp.headers.get("content-type", "").lower()
        if "html" not in ctype and "text" not in ctype:
            return None
        # Bounded read to avoid pulling huge pages into memory.
        raw = resp.raw.read(_MAX_BYTES, decode_content=True)
        html_text = raw.decode(resp.encoding or "utf-8", errors="ignore")
    except Exception:
        return None

    text = extract_main_text(html_text)
    if len(text) < 200:  # too little usable content to be worth indexing
        return None
    return Document(url=url, title=title, source=source, text=text)


def extract_main_text(html_text: str) -> str:
    """Strip boilerplate and return concatenated paragraph text."""
    try:
        soup = BeautifulSoup(html_text, "html.parser")
    except Exception:
        return ""
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form", "noscript"]):
        tag.decompose()

    # Prefer <article>/<main> if present, else fall back to all paragraphs.
    container = soup.find("article") or soup.find("main") or soup
    parts = [p.get_text(" ", strip=True) for p in container.find_all("p")]
    text = "\n".join(p for p in parts if len(p) > 40)
    if len(text) < 200:  # fall back to whole-body text
        text = soup.get_text(" ", strip=True)
    return re.sub(r"\s+", " ", text).strip()


def fetch_documents(
    results: list,
    max_docs: int = RAG_MAX_DOCS,
    timeout: float = RAG_FETCH_TIMEOUT,
) -> tuple[list[Document], list[str]]:
    """
    Concurrently fetch the top unique URLs from a list of SearchResult-like
    objects (need .url, .title, .source). Returns (documents, warnings).
    """
    seen: set[str] = set()
    targets = []
    for r in results:
        url = getattr(r, "url", "") or ""
        if not url or url in seen:
            continue
        seen.add(url)
        targets.append((url, getattr(r, "title", ""), getattr(r, "source", "web")))
        if len(targets) >= max_docs:
            break

    documents: list[Document] = []
    warnings: list[str] = []
    if not targets:
        return documents, warnings

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(6, len(targets))) as ex:
        futures = {
            ex.submit(fetch_article, url, title, source, timeout): url
            for (url, title, source) in targets
        }
        for fut in concurrent.futures.as_completed(futures):
            doc = fut.result()
            if doc:
                documents.append(doc)

    fetched = len(documents)
    if fetched < len(targets):
        warnings.append(
            f"Fetched full text for {fetched}/{len(targets)} sources "
            "(some blocked fetching or had no extractable article body)."
        )
    return documents, warnings


# --------------------------------------------------------------------------- #
# 2. Chunking
# --------------------------------------------------------------------------- #
def chunk_text(text: str, size_words: int, overlap_words: int) -> list[str]:
    """Split text into overlapping word windows."""
    words = text.split()
    if not words:
        return []
    if len(words) <= size_words:
        return [" ".join(words)]
    step = max(1, size_words - overlap_words)
    chunks = []
    for start in range(0, len(words), step):
        window = words[start : start + size_words]
        if window:
            chunks.append(" ".join(window))
        if start + size_words >= len(words):
            break
    return chunks


def build_chunks(
    documents: list[Document],
    size_words: int = RAG_CHUNK_WORDS,
    overlap_words: int = RAG_CHUNK_OVERLAP,
) -> list[Chunk]:
    """Turn fetched documents into a flat list of retrievable chunks."""
    chunks: list[Chunk] = []
    for doc in documents:
        for piece in chunk_text(doc.text, size_words, overlap_words):
            chunks.append(
                Chunk(text=piece, url=doc.url, title=doc.title, source=doc.source)
            )
    return chunks


# --------------------------------------------------------------------------- #
# 3. Retrieval
# --------------------------------------------------------------------------- #
class TfidfRetriever:
    """Sparse vector retrieval via TF-IDF + cosine similarity (no API cost)."""

    def __init__(self, chunks: list[Chunk]):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.chunks = chunks
        self._vectorizer = TfidfVectorizer(stop_words="english", max_features=20000)
        self._matrix = self._vectorizer.fit_transform([c.text for c in chunks])

    def retrieve(self, queries: list[str], top_k: int) -> list[Chunk]:
        from sklearn.metrics.pairwise import cosine_similarity

        q_matrix = self._vectorizer.transform(queries)
        sims = cosine_similarity(q_matrix, self._matrix)  # (n_queries, n_chunks)
        # A chunk's relevance = its best match against ANY probe query.
        best = sims.max(axis=0)
        ranked = sorted(range(len(self.chunks)), key=lambda i: best[i], reverse=True)
        out = []
        for i in ranked[:top_k]:
            if best[i] <= 0:
                continue
            c = self.chunks[i]
            out.append(Chunk(c.text, c.url, c.title, c.source, score=round(float(best[i]), 4)))
        return out


class DenseRetriever:
    """Dense semantic retrieval via local Ollama embeddings (free, local only)."""

    def __init__(self, chunks: list[Chunk], embeddings: list[list[float]]):
        self.chunks = chunks
        self._embeddings = embeddings

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb) if na and nb else 0.0

    def retrieve(self, queries: list[str], top_k: int) -> list[Chunk]:
        q_embs = embed_texts(queries)
        if not q_embs:
            return []
        scored = []
        for c, emb in zip(self.chunks, self._embeddings):
            best = max(self._cosine(qe, emb) for qe in q_embs)
            scored.append((best, c))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [
            Chunk(c.text, c.url, c.title, c.source, score=round(float(s), 4))
            for s, c in scored[:top_k]
            if s > 0
        ]


def embed_texts(texts: list[str]) -> Optional[list[list[float]]]:
    """Embed texts with local Ollama. Returns None if Ollama is unavailable."""
    try:
        import httpx

        vectors: list[list[float]] = []
        with httpx.Client(timeout=30.0) as client:
            for t in texts:
                resp = client.post(
                    f"{OLLAMA_BASE_URL}/api/embeddings",
                    json={"model": OLLAMA_EMBED_MODEL, "prompt": t},
                )
                resp.raise_for_status()
                vectors.append(resp.json()["embedding"])
        return vectors
    except Exception:
        return None


def get_retriever(chunks: list[Chunk], use_ollama: bool = False):
    """
    Build the best available retriever for the given chunks.
    Tries dense Ollama embeddings when use_ollama is set, else TF-IDF.
    Returns (retriever, backend_name) or (None, "none") if no chunks.
    """
    if not chunks:
        return None, "none"
    if use_ollama:
        embs = embed_texts([c.text for c in chunks])
        if embs:
            return DenseRetriever(chunks, embs), f"ollama:{OLLAMA_EMBED_MODEL}"
    try:
        return TfidfRetriever(chunks), "tfidf"
    except Exception:
        return None, "none"


def adverse_probe_queries(entity_name: str) -> list[str]:
    """Risk-focused probe queries used to pull the most relevant passages."""
    return [
        entity_name,
        f"{entity_name} fraud corruption bribery money laundering",
        f"{entity_name} investigation lawsuit charges arrest convicted",
        f"{entity_name} sanction penalty fine regulatory violation scandal",
    ]
