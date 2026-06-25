"""
Adverse News Classifier - Streamlit Dashboard
Agentic AML screening with email login, CAPTCHA, and Stripe payments.
"""
import os

import streamlit as st

# Load Streamlit Cloud secrets before other app modules read config.
try:
    for key, value in st.secrets.items():
        if isinstance(value, str) and not os.getenv(key):
            os.environ[key] = value
except Exception:
    pass

from components.turnstile import turnstile_widget
from src.agent import run_adverse_news_screening, AdverseNewsReport
from src.captcha import captcha_passed, new_math_challenge
from src.config import TURNSTILE_SITE_KEY, is_turnstile_configured
from src.stripe_payments import (
    create_checkout_session,
    get_paid_session_details,
    is_stripe_configured,
    CREDIT_PACKS,
)
from src.user_store import (
    AuthUser,
    deduct_credit,
    get_credits,
    is_auth_configured,
    record_stripe_session,
    refund_credit,
    sign_in,
    sign_up,
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
    if "user" not in st.session_state:
        st.session_state.user = None
    if "credits" not in st.session_state:
        st.session_state.credits = 0
    if "report" not in st.session_state:
        st.session_state.report = None
    if "processing" not in st.session_state:
        st.session_state.processing = False
    if "math_question" not in st.session_state:
        question, answer = new_math_challenge()
        st.session_state.math_question = question
        st.session_state.math_answer = answer


def refresh_user_credits(user: AuthUser) -> None:
    st.session_state.credits = get_credits(user.id)


def handle_stripe_callback(user: AuthUser) -> None:
    session_id = st.query_params.get("session_id")
    if not session_id:
        return

    # Keep session_id in state so it survives reruns even if URL is cleared
    st.session_state["pending_session_id"] = session_id

    if not is_stripe_configured():
        st.warning("Stripe not configured — cannot verify payment automatically.")
        st.query_params.clear()
        return

    details, error = get_paid_session_details(session_id)

    if not details:
        st.error(f"Payment verification failed: {error}")
        st.caption(f"Session ID: `{session_id}`")
        st.query_params.clear()
        return

    if details["user_id"] != user.id:
        st.error(
            f"Payment was made under a different account (expected `{details['user_id'][:8]}…`). "
            "Sign in with the account used at checkout."
        )
        st.query_params.clear()
        return

    try:
        applied = record_stripe_session(session_id, user.id, details["credits"])
    except Exception as e:
        st.error(f"Credits could not be saved to database: {e}. Session ID: `{session_id}`")
        st.query_params.clear()
        return

    refresh_user_credits(user)
    st.query_params.clear()
    st.session_state.pop("pending_session_id", None)

    if applied:
        st.success(f"Payment confirmed — {details['credits']} credits added!")
    else:
        st.info("Payment already processed. Your credits are up to date.")
    st.rerun()


def render_auth_page() -> None:
    st.title("Sign in to continue")
    st.caption("One free screening per account. Credits are tied to your email — not your browser session.")

    if not is_auth_configured():
        st.error(
            "Authentication is not configured. Set `SUPABASE_URL`, `SUPABASE_ANON_KEY`, and "
            "`SUPABASE_SERVICE_ROLE_KEY` in `.env` or Streamlit secrets, then run "
            "`scripts/supabase_schema.sql` in Supabase."
        )
        return

    tab_sign_in, tab_sign_up = st.tabs(["Sign in", "Create account"])

    def auth_form(mode: str) -> None:
        with st.form(f"{mode}_form", clear_on_submit=False):
            email = st.text_input("Email", placeholder="you@company.com")
            password = st.text_input("Password", type="password")
            st.markdown(f"**Security check:** {st.session_state.math_question}")
            math_input = st.number_input(
                "Answer",
                min_value=0,
                max_value=100,
                step=1,
                value=0,
                label_visibility="collapsed",
            )

            turnstile_token = None
            if is_turnstile_configured():
                st.caption("Complete the CAPTCHA below:")
                turnstile_token = turnstile_widget(TURNSTILE_SITE_KEY, key=f"turnstile_{mode}")

            submitted = st.form_submit_button(
                "Sign in" if mode == "sign_in" else "Create account",
                use_container_width=True,
            )

        if not submitted:
            return

        ok, message = captcha_passed(
            st.session_state.math_answer,
            math_input,
            turnstile_token,
        )
        if not ok:
            st.error(message)
            question, answer = new_math_challenge()
            st.session_state.math_question = question
            st.session_state.math_answer = answer
            st.rerun()

        if mode == "sign_in":
            user, error = sign_in(email, password, turnstile_token)
        else:
            user, error = sign_up(email, password, turnstile_token)

        if error and not user:
            st.error(error)
            question, answer = new_math_challenge()
            st.session_state.math_question = question
            st.session_state.math_answer = answer
            st.rerun()

        if user:
            st.session_state.user = user
            refresh_user_credits(user)
            st.success("Welcome back!" if mode == "sign_in" else "Account ready — you have 1 free credit.")
            st.rerun()

        if error:
            st.info(error)

    with tab_sign_in:
        auth_form("sign_in")
    with tab_sign_up:
        auth_form("sign_up")


def require_auth() -> AuthUser:
    init_session_state()
    user = st.session_state.user
    if user:
        return user
    render_auth_page()
    st.stop()


def get_base_url() -> str:
    return os.getenv("BASE_URL", "http://localhost:8501")


def main():
    user = require_auth()
    handle_stripe_callback(user)

    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/search.png", width=80)
        st.title("Adverse News Classifier")
        st.caption("Agentic AML Screening")
        st.divider()

        st.caption(f"Signed in as **{user.email}**")
        if st.button("Sign out", use_container_width=True):
            st.session_state.user = None
            st.session_state.report = None
            st.rerun()

        st.metric("Credits", st.session_state.credits)
        if st.button("Refresh credits", use_container_width=True):
            refresh_user_credits(user)
            st.rerun()
        if st.session_state.credits < 1:
            st.warning("No credits left. Purchase more below.")

        st.divider()
        st.subheader("Data Sources")
        st.markdown("""
        - **Web** (DuckDuckGo)
        - **News** (DuckDuckGo News)
        - **Social Media** (Reddit, LinkedIn, X)
        - **PEP List** (OpenSanctions)
        - **Sanctions** (OpenSanctions)
        """)

        st.divider()
        use_ollama = st.checkbox(
            "Use Ollama (local, free)",
            value=False,
            help="Only works when Streamlit runs on the same machine as Ollama. Use OpenRouter when hosted.",
        )

        st.divider()
        with st.expander("Already paid? Apply manually"):
            manual_sid = st.text_input(
                "Stripe session ID",
                placeholder="cs_test_...",
                key="manual_session_id",
            )
            if st.button("Apply credits", use_container_width=True):
                if manual_sid.strip():
                    details, error = get_paid_session_details(manual_sid.strip())
                    if not details:
                        st.error(f"Could not apply: {error}")
                    elif details["user_id"] != user.id:
                        st.error("Session belongs to a different account.")
                    else:
                        try:
                            applied = record_stripe_session(manual_sid.strip(), user.id, details["credits"])
                            refresh_user_credits(user)
                            if applied:
                                st.success(f"{details['credits']} credits applied!")
                            else:
                                st.info("Already applied. Credits refreshed.")
                            st.rerun()
                        except Exception as e:
                            st.error(str(e))
                else:
                    st.warning("Paste your Stripe session ID above.")

        st.divider()
        st.subheader("Purchase Credits")
        if is_stripe_configured():
            for pack_id, pack in CREDIT_PACKS.items():
                if st.button(f"{pack_id.title()}: {pack['credits']} credits", key=f"buy_{pack_id}"):
                    url = create_checkout_session(
                        pack_id,
                        get_base_url(),
                        get_base_url(),
                        customer_email=user.email,
                        user_id=user.id,
                    )
                    if url:
                        st.link_button("Complete Purchase", url)
                    else:
                        st.error("Could not start checkout. Configure Stripe price IDs for logged-in purchases.")
        else:
            st.info("Set Stripe keys to enable payments.")

    st.title("Adverse News Index")
    st.markdown(
        "Screen individuals or companies against adverse media, PEP lists, and sanctions. "
        "**No paid search APIs** — uses DuckDuckGo, OpenSanctions bulk data, and OpenRouter/Ollama."
    )

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
            ok, new_balance = deduct_credit(user.id)
            if not ok:
                st.session_state.credits = get_credits(user.id)
                st.error("Insufficient credits. Please purchase more in the sidebar.")
            else:
                st.session_state.credits = new_balance
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
                st.session_state.credits = refund_credit(user.id)
            finally:
                st.session_state.processing = False
        st.rerun()

    report: AdverseNewsReport = st.session_state.report
    if report:
        st.markdown("---")
        st.subheader("Adverse News Index")

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

        st.subheader("Watchlist Screening")
        wl1, wl2 = st.columns(2)

        def _os_link(entity_id):
            return f"https://www.opensanctions.org/entities/{entity_id}/" if entity_id else None

        with wl1:
            pep = report.screening.pep_match
            if pep:
                conf = pep.get("match_confidence")
                st.warning("**PEP Match** — Politically Exposed Person")
                st.markdown(f"**Matched name:** {pep.get('matched_name', '—').title()}")
                if conf is not None:
                    st.markdown(f"**Match confidence:** {conf:.0%}")
                sources = pep.get("dataset") or "—"
                st.markdown(f"**Source list(s):** {sources}")
                link = _os_link(pep.get("entity_id"))
                if link:
                    st.markdown(f"**OpenSanctions record:** [{pep.get('entity_id')}]({link})")
            else:
                st.success("**PEP:** No match found")

        with wl2:
            sanc = report.screening.sanction_match
            if sanc:
                conf = sanc.get("match_confidence")
                st.error("**Sanction Match** — On sanction/watchlist")
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
                st.success("**Sanctions:** No match found")

        from src.collectors import _load_opensanctions_data
        _peps, _sanctions = _load_opensanctions_data()
        if not _peps and not _sanctions:
            st.info(
                "No local OpenSanctions data loaded — PEP/Sanction checks were skipped. "
                "Run `python3 scripts/download_opensanctions.py` to enable them."
            )
        else:
            st.caption(f"Screened against {len(_peps):,} PEP and {len(_sanctions):,} sanction records (OpenSanctions).")

        if report.screening.errors:
            with st.expander(f"{len(report.screening.errors)} source warning(s)"):
                for err in report.screening.errors:
                    st.caption(err)
                st.caption(
                    "Free search sources (DuckDuckGo) occasionally rate-limit automated "
                    "queries. Other sources still ran; try again in a moment for full coverage."
                )

        real_results = [r for r in report.screening.results if r.title]
        st.subheader(f"Retrieved Results ({len(real_results)})")
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
