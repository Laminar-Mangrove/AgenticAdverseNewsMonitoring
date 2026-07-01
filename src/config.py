"""Configuration for Adverse News Classifier."""
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
OPENSANCTIONS_DIR = DATA_DIR / "opensanctions"

# LLM Configuration - supports multiple backends
# Option 1: Ollama (free, local) - set OLLAMA_BASE_URL if different
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

# Option 2: OpenRouter (requires API key) - for GPT, Gemini, etc.
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

# Local embedding model for dense RAG retrieval (Ollama only; pull with
# `ollama pull nomic-embed-text`). Ignored when running TF-IDF retrieval.
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# --- Agentic RAG pipeline (LangGraph) tunables -------------------------------
# Enable the LangGraph agentic+RAG pipeline by default (falls back to the simple
# pipeline automatically if langgraph isn't installed).
USE_AGENTIC_DEFAULT = os.getenv("USE_AGENTIC_DEFAULT", "true").lower() == "true"
# How many source URLs to fetch full text for, per search pass.
RAG_MAX_DOCS = int(os.getenv("RAG_MAX_DOCS", "6"))
# Per-document fetch timeout (seconds).
RAG_FETCH_TIMEOUT = float(os.getenv("RAG_FETCH_TIMEOUT", "8"))
# Chunking: window size and overlap, in words.
RAG_CHUNK_WORDS = int(os.getenv("RAG_CHUNK_WORDS", "180"))
RAG_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "40"))
# How many retrieved passages to feed the scorer.
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "8"))
# Max agent reasoning iterations (each extra loop = 1 refined search + LLM call).
# Default 1: dig one level deeper on a real lead, then finalize. The decide node
# already fast-finalizes clean entities (no adverse signal) without any loop.
AGENT_MAX_ITERATIONS = int(os.getenv("AGENT_MAX_ITERATIONS", "1"))

# Stripe
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "")
STRIPE_PUBLISHABLE_KEY = os.getenv("STRIPE_PUBLISHABLE_KEY", "")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")

# Supabase (email auth + per-user credits)
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

# OpenSanctions API (commercial use; get key at https://www.opensanctions.org/api/)
# When set, the app uses the API instead of local CSV files.
# Free 30-day trial available with a business email.
OPENSANCTIONS_API_KEY = os.getenv("OPENSANCTIONS_API_KEY", "")
OPENSANCTIONS_API_URL = "https://api.opensanctions.org"

# Cloudflare Turnstile CAPTCHA (optional but recommended for production)
TURNSTILE_SITE_KEY = os.getenv("TURNSTILE_SITE_KEY", "")
TURNSTILE_SECRET_KEY = os.getenv("TURNSTILE_SECRET_KEY", "")

FREE_CREDITS = 1


def is_opensanctions_api_configured() -> bool:
    return bool(OPENSANCTIONS_API_KEY)


def is_supabase_configured() -> bool:
    return bool(SUPABASE_URL and SUPABASE_ANON_KEY and SUPABASE_SERVICE_ROLE_KEY)


def is_turnstile_configured() -> bool:
    return bool(TURNSTILE_SITE_KEY and TURNSTILE_SECRET_KEY)


def hydrate_from_streamlit_secrets() -> None:
    """Load Streamlit Cloud secrets into os.environ before reading config."""
    try:
        import streamlit as st

        for key, value in st.secrets.items():
            if isinstance(value, str) and not os.getenv(key):
                os.environ[key] = value
    except Exception:
        pass


# Search limits (to avoid rate limits)
MAX_WEB_RESULTS = 10
MAX_NEWS_RESULTS = 5
MAX_SOCIAL_RESULTS = 5
