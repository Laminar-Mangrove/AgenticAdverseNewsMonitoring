# Adverse News Classifier

An **agentic AI system** for adverse media screening in AML (Anti-Money Laundering) compliance. Based on the framework from *"An Agentic LLM Framework for Adverse Media Screening in AML Compliance"* (Chernakov et al., University of Luxembourg).

## Features

- **Multi-source screening** (all free, no paid APIs):
  - Web search (DuckDuckGo)
  - News (DuckDuckGo News)
  - Social media (Reddit, LinkedIn, X/Twitter, Mastodon)
  - PEP list (OpenSanctions bulk data)
  - Sanction list (OpenSanctions bulk data)
- **Adverse News Index (ANI)** – 0–1 risk score with LLM-generated justifications
- **Streamlit dashboard** with dark theme
- **Stripe payment gateway** for credit-based access

## Quick Start

### 1. Install dependencies

```bash
cd "Adverse News Classifier"
pip install -r requirements.txt
```

### 2. LLM setup (choose one)

**Option A: Ollama (free, local)**  
Install [Ollama](https://ollama.ai) and run:
```bash
ollama pull llama3.2
```

**Option B: OpenRouter**  
Set `OPENROUTER_API_KEY` in `.env` for GPT/Gemini.

### 3. OpenSanctions data (optional but recommended)

Download PEP and sanction lists:
```bash
python scripts/download_opensanctions.py
```

Or manually from [OpenSanctions](https://www.opensanctions.org/datasets/default/) (free for non-commercial use).

### 4. Run the app

```bash
streamlit run app.py
```

### 5. Stripe (optional)

1. Create a [Stripe](https://stripe.com) account
2. Create products and prices for credit packs
3. Copy `.env.example` to `.env` and add your keys

## Project Structure

```
Adverse News Classifier/
├── app.py                 # Streamlit dashboard
├── requirements.txt
├── src/
│   ├── agent.py           # Agentic pipeline
│   ├── collectors.py      # Web, news, social, PEP, sanctions
│   ├── config.py
│   ├── llm_scorer.py      # ANI scoring (Ollama/OpenRouter)
│   └── stripe_payments.py # Stripe integration
├── scripts/
│   └── download_opensanctions.py
└── data/
    └── opensanctions/     # PEP & sanction lists (after download)
```

## ANI Score Interpretation

| Score  | Risk Level | Description                          |
|--------|------------|--------------------------------------|
| 0.0–0.2 | Low       | Clean record, no adverse findings   |
| 0.2–0.5 | Moderate  | PEP status, minor concerns           |
| 0.5–0.8 | High     | Significant adverse media            |
| 0.8–1.0 | Critical | Sanctioned, major fraud/crime        |

## License

MIT. OpenSanctions data: free for non-commercial use; commercial use requires a [license](https://www.opensanctions.org/licensing/).
