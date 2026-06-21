"""
Adverse News Classifier - Streamlit Dashboard
Agentic AML screening with Stripe payment gateway.
"""
import streamlit as st

from src.agent import run_adverse_news_screening, AdverseNewsReport
from src.stripe_payments import (
    create_checkout_session,
    get_publishable_key,
    is_stripe_configured,
    get_credits_from_session,
    CREDIT_PACKS,
    FREE_CREDITS,
)

# Page config
st.set_page_config(
    page_title="Adverse News Classifier",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Space+Grotesk:wght@400;500;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif !important;
        color: #e8e8e8 !important;
    }
    
    .ani-score-card {
        background: linear-gradient(145deg, #1e1e3f 0%, #2d2d5a 100%);
        border-radius: 16px;
        padding: 24px;
        margin: 16px 0;
        border: 1px solid rgba(99, 102, 241, 0.3);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    
    .risk-low { color: #22c55e !important; }
    .risk-moderate { color: #eab308 !important; }
    .risk-high { color: #f97316 !important; }
    .risk-critical { color: #ef4444 !important; }
    
    .source-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 2px;
    }
    .source-web { background: #3b82f6; color: white; }
    .source-news { background: #8b5cf6; color: white; }
    .source-social { background: #06b6d4; color: white; }
    .source-pep { background: #f59e0b; color: black; }
    .source-sanction { background: #ef4444; color: white; }
    
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #0f0f23 100%);
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.5rem 1.5rem !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(90deg, #4f46e5 0%, #7c3aed 100%) !important;
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.4) !important;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state."""
    if "credits" not in st.session_state:
        st.session_state.credits = FREE_CREDITS
    if "report" not in st.session_state:
        st.session_state.report = None
    if "processing" not in st.session_state:
        st.session_state.processing = False


def handle_stripe_callback():
    """Handle Stripe success redirect."""
    session_id = st.query_params.get("session_id")
    if session_id and is_stripe_configured():
        credits = get_credits_from_session(session_id)
        if credits:
            st.session_state.credits += credits
            st.query_params.clear()
            st.success(f"✅ Payment successful! {credits} credits added.")
            st.rerun()


def main():
    init_session_state()
    handle_stripe_callback()

    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/search.png", width=80)
        st.title("🔍 Adverse News Classifier")
        st.caption("Agentic AML Screening")
        st.divider()

        st.metric("Credits", st.session_state.credits)
        if st.session_state.credits < 1:
            st.warning("No credits left. Purchase more below.")
        
        st.divider()
        st.subheader("Data Sources")
        st.markdown("""
        - 🌐 **Web** (DuckDuckGo)
        - 📰 **News** (DuckDuckGo News)
        - 📱 **Social Media** (Reddit, LinkedIn, X)
        - 👔 **PEP List** (OpenSanctions)
        - ⚠️ **Sanctions** (OpenSanctions)
        """)
        
        st.divider()
        use_ollama = st.checkbox("Use Ollama (local, free)", value=True, help="Uncheck to use OpenRouter API")
        
        st.divider()
        st.subheader("💳 Purchase Credits")
        if is_stripe_configured():
            for pack_id, pack in CREDIT_PACKS.items():
                if st.button(f"{pack_id.title()}: {pack['credits']} credits", key=f"buy_{pack_id}"):
                    base_url = st.secrets.get("BASE_URL", "http://localhost:8501") if hasattr(st, "secrets") else "http://localhost:8501"
                    url = create_checkout_session(pack_id, base_url, base_url)
                    if url:
                        st.link_button("Complete Purchase →", url)
        else:
            st.info("Set STRIPE_SECRET_KEY and STRIPE_PUBLISHABLE_KEY to enable payments.")

    # Main content
    st.title("Adverse News Index")
    st.markdown("Screen individuals or companies against adverse media, PEP lists, and sanctions. **No paid APIs** — uses DuckDuckGo, OpenSanctions bulk data, and local/OpenRouter LLM.")

    col1, col2 = st.columns([2, 1])
    with col1:
        entity_name = st.text_input("Entity name (person or company)", placeholder="e.g. John Smith, Acme Corp")
    with col2:
        entity_type = st.selectbox("Type", ["person", "company"])

    if st.button("Run Screening", type="primary", use_container_width=True):
        if not entity_name.strip():
            st.error("Please enter an entity name.")
        elif st.session_state.credits < 1:
            st.error("Insufficient credits. Please purchase more in the sidebar.")
        else:
            st.session_state.credits -= 1
            st.session_state.processing = True

    if st.session_state.processing:
        with st.spinner("Running agentic screening (web, news, social, PEP, sanctions)..."):
            try:
                report = run_adverse_news_screening(
                    entity_name.strip(),
                    entity_type,
                    use_ollama=use_ollama,
                )
                st.session_state.report = report
            except Exception as e:
                st.error(f"Screening failed: {e}")
                st.session_state.credits += 1  # Refund on error
            finally:
                st.session_state.processing = False
        st.rerun()

    # Display report
    report: AdverseNewsReport = st.session_state.report
    if report:
        risk_class = f"risk-{report.risk_level.lower()}"
        
        st.markdown("---")
        st.subheader("📊 Adverse News Index")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("ANI Score", f"{report.ani_score:.2f}", f"Risk: {report.risk_level}")
        with c2:
            st.metric("Risk Level", report.risk_level, "")
        with c3:
            st.metric("Sources Checked", len(report.screening.results) + 2, "web+news+social+PEP+sanctions")
        
        st.markdown(f"""
        <div class="ani-score-card">
            <h4>Justification</h4>
            <p>{report.justification}</p>
        </div>
        """, unsafe_allow_html=True)

        # PEP / Sanction watchlist checks (OpenSanctions)
        st.subheader("🛡️ Watchlist Screening")
        wl1, wl2 = st.columns(2)

        def _os_link(entity_id):
            return f"https://www.opensanctions.org/entities/{entity_id}/" if entity_id else None

        with wl1:
            pep = report.screening.pep_match
            if pep:
                conf = pep.get("match_confidence")
                st.warning("⚠️ **PEP Match** — Politically Exposed Person")
                st.markdown(f"**Matched name:** {pep.get('matched_name', '—').title()}")
                if conf is not None:
                    st.markdown(f"**Match confidence:** {conf:.0%}")
                sources = pep.get("dataset") or "—"
                st.markdown(f"**Source list(s):** {sources}")
                link = _os_link(pep.get("entity_id"))
                if link:
                    st.markdown(f"**OpenSanctions record:** [{pep.get('entity_id')}]({link})")
            else:
                st.success("✅ **PEP:** No match found")

        with wl2:
            sanc = report.screening.sanction_match
            if sanc:
                conf = sanc.get("match_confidence")
                st.error("🚫 **Sanction Match** — On sanction/watchlist")
                st.markdown(f"**Matched name:** {sanc.get('matched_name', '—').title()}")
                if conf is not None:
                    st.markdown(f"**Match confidence:** {conf:.0%}")
                sources = sanc.get("dataset") or "—"
                st.markdown(f"**Source list(s):** {sources}")
                if sanc.get("sanctions"):
                    st.markdown(f"**Program(s):** {sanc.get('sanctions')}")
                link = _os_link(sanc.get("entity_id"))
                if link:
                    st.markdown(f"**OpenSanctions record:** [{sanc.get('entity_id')}]({link})")
            else:
                st.success("✅ **Sanctions:** No match found")

        # Warn if the local watchlist data hasn't been downloaded
        from src.collectors import _load_opensanctions_data
        _peps, _sanctions = _load_opensanctions_data()
        if not _peps and not _sanctions:
            st.info(
                "ℹ️ No local OpenSanctions data loaded — PEP/Sanction checks were skipped. "
                "Run `python3 scripts/download_opensanctions.py` to enable them."
            )
        else:
            st.caption(f"Screened against {len(_peps):,} PEP and {len(_sanctions):,} sanction records (OpenSanctions).")

        # Non-fatal source warnings (e.g. DuckDuckGo rate-limiting)
        if report.screening.errors:
            with st.expander(f"⚠️ {len(report.screening.errors)} source warning(s)"):
                for err in report.screening.errors:
                    st.caption(err)
                st.caption(
                    "Free search sources (DuckDuckGo) occasionally rate-limit automated "
                    "queries. Other sources still ran; try again in a moment for full coverage."
                )

        # Results by source
        real_results = [r for r in report.screening.results if r.title]
        st.subheader(f"📑 Retrieved Results ({len(real_results)})")
        if not real_results:
            st.info("No web/news/social results retrieved (sources may be rate-limited). PEP/sanction screening above is unaffected.")
        for r in real_results:
            title = r.title[:70] + ("..." if len(r.title) > 70 else "")
            with st.expander(f"[{r.source}] {title}"):
                if r.url:
                    st.markdown(f"**URL:** {r.url}")
                if r.date:
                    st.caption(f"Date: {r.date}")
                st.write(r.snippet)
                st.caption(f"Source: {r.source}")


if __name__ == "__main__":
    main()
