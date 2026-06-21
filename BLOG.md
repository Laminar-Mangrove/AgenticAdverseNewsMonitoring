# I Built an Agentic AI That Hunts for Adverse News — Here's How It Works

*Screening people and companies against the web, news, social media, PEP lists, and sanctions — with zero paid APIs.*

## The problem nobody enjoys

If you've ever worked in anti-money-laundering (AML) compliance, you know the drill: before onboarding a customer, someone has to check whether that person or company shows up in bad news — fraud, corruption, sanctions, criminal charges. It's called **adverse media screening**, and regulators increasingly require it.

The traditional tools have two problems. The expensive ones (World-Check, Dow Jones, LexisNexis) are "black boxes" — you get a flag but no transparent reasoning. The cheap ones are keyword matchers that drown analysts in false positives. Searching "John Smith fraud" and flagging *every* John Smith is not screening; it's noise.

So I built an **MVP of an agentic AI** that does this differently — inspired by a recent University of Luxembourg paper, *"An Agentic LLM Framework for Adverse Media Screening in AML Compliance."*

## What it does

You type in a name — a person or a company. The agent then, on its own:

1. **Searches the open web** for mentions
2. **Pulls recent news** about them
3. **Scans social media** (Reddit, LinkedIn, X, Mastodon)
4. **Checks PEP lists** (Politically Exposed Persons — ~750k records)
5. **Checks global sanctions & watchlists** (~90k records)

Then it reads everything, reasons about it, and produces a single number: the **Adverse News Index (ANI)** — a 0-to-1 risk score — *plus a plain-English justification* explaining why.

> **0.0–0.2** Low · **0.2–0.5** Moderate · **0.5–0.8** High · **0.8–1.0** Critical

The justification is the part that matters. A reviewer doesn't just see "0.87 — Critical." They see *"Entity appears on the OFAC SDN list under the Global Magnitsky program; multiple news sources report..."* — with clickable links to the source records.

## Why "agentic," not just "a script"

The word *agentic* gets thrown around a lot, so here's what it concretely means in this app:

- **Autonomy** — you give it a name; it decides what to search and gathers its own evidence.
- **Tool use** — it calls five different data sources like an analyst using different databases.
- **Reasoning over evidence (RAG)** — instead of keyword-matching, a language model reads the retrieved material and *judges* relevance, including **entity disambiguation** (is this the *same* John Smith, or a different one?).
- **Pluggable brains** — it runs on a free local model (Ollama) or a hosted one (via OpenRouter), and **degrades gracefully** to rule-based scoring if no model is available.

That last point is the difference between a demo and a tool: it keeps working when things break.

## The "no paid API" constraint

I deliberately built this without a single paid API, to prove it's possible:

- **Search** → DuckDuckGo (with a Google News RSS fallback when rate-limited)
- **PEP & Sanctions data** → [OpenSanctions](https://www.opensanctions.org) bulk data (free for non-commercial use)
- **The LLM** → Ollama, running locally for free

It's wrapped in a **Streamlit dashboard** with a **Stripe** payment gateway and a credit system — so it's a working product skeleton, not just a notebook.

## Being honest about an MVP

This is a v1, and I'd rather tell you what it *can't* do yet:

- **Free search gets rate-limited.** DuckDuckGo throttles bots; the news fallback helps, but heavy use needs a more robust search layer.
- **It scores on snippets, not full articles.** The research version does deeper document retrieval and embeddings; mine currently reasons over search summaries.
- **Name matching is good, not perfect.** It now requires full names and a confidence threshold (so "Mohammed" alone won't flag a sanctioned person), but date-of-birth and nationality disambiguation would make it sharper.
- **Not regulatory-grade — yet.** This is a transparent, hackable foundation, not a certified compliance product.

## Why I think this matters

The interesting shift isn't "AI replaces compliance analysts." It's that an agentic system can do the **tedious first pass** — gather, disambiguate, summarize, and *explain* — while a human makes the judgment call with full visibility into the reasoning. Transparency, not automation, is the real upgrade over both legacy black boxes and dumb keyword tools.

## What's next

On my list: multi-step reasoning (separate identity-matching and negativity-scoring passes that cross-check each other), true full-text RAG, and date-of-birth/nationality disambiguation.

*If you work in AML, fintech, or applied AI — I'd love your feedback on what would make this genuinely useful. Reply or drop a comment.*
