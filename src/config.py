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

# Stripe
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "")
STRIPE_PUBLISHABLE_KEY = os.getenv("STRIPE_PUBLISHABLE_KEY", "")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")

# Search limits (to avoid rate limits)
MAX_WEB_RESULTS = 10
MAX_NEWS_RESULTS = 5
MAX_SOCIAL_RESULTS = 5
