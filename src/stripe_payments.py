"""
Stripe payment integration for Adverse News Classifier.
Supports: one-time credit packs via Checkout or Payment Links.
"""
import os
from typing import Optional

import stripe

from .config import FREE_CREDITS, STRIPE_PUBLISHABLE_KEY, STRIPE_SECRET_KEY

# Initialize Stripe
if STRIPE_SECRET_KEY:
    stripe.api_key = STRIPE_SECRET_KEY

# Credit packs - use STRIPE_LINK_* for simple Payment Links, or STRIPE_PRICE_* for Checkout
CREDIT_PACKS = {
    "starter": {
        "credits": 10,
        "price_id": os.getenv("STRIPE_PRICE_STARTER"),
        "payment_link": os.getenv("STRIPE_LINK_STARTER"),
    },
    "professional": {
        "credits": 50,
        "price_id": os.getenv("STRIPE_PRICE_PRO"),
        "payment_link": os.getenv("STRIPE_LINK_PRO"),
    },
    "enterprise": {
        "credits": 200,
        "price_id": os.getenv("STRIPE_PRICE_ENTERPRISE"),
        "payment_link": os.getenv("STRIPE_LINK_ENTERPRISE"),
    },
}

def create_checkout_session(
    pack_id: str,
    success_url: str,
    cancel_url: str,
    customer_email: Optional[str] = None,
    user_id: Optional[str] = None,
) -> Optional[str]:
    """
    Create Stripe Checkout session or return Payment Link URL.
    Returns checkout URL or None if Stripe not configured.
    """
    pack = CREDIT_PACKS.get(pack_id)
    if not pack:
        return None

    # Payment Links cannot attach user_id metadata — use Checkout for logged-in users.
    if pack.get("payment_link") and not user_id:
        return pack["payment_link"]

    # Create Checkout session (requires price_id)
    if not STRIPE_SECRET_KEY or not pack.get("price_id"):
        return None

    try:
        session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            line_items=[{"price": pack["price_id"], "quantity": 1}],
            mode="payment",
            success_url=success_url + "?session_id={CHECKOUT_SESSION_ID}",
            cancel_url=cancel_url,
            customer_email=customer_email,
            metadata={
                "pack_id": pack_id,
                "credits": str(pack["credits"]),
                **({"user_id": user_id} if user_id else {}),
            },
        )
        return session.url
    except Exception:
        return None


def get_publishable_key() -> str:
    """Return Stripe publishable key for frontend."""
    return STRIPE_PUBLISHABLE_KEY or ""


def is_stripe_configured() -> bool:
    """Check if Stripe is properly configured (API keys or payment links)."""
    has_api = bool(STRIPE_SECRET_KEY and STRIPE_PUBLISHABLE_KEY)
    has_links = any(p.get("payment_link") for p in CREDIT_PACKS.values())
    return has_api or has_links


def get_paid_session_details(session_id: str) -> Optional[dict]:
    """Return paid session metadata needed to grant credits to a user."""
    if not STRIPE_SECRET_KEY:
        return None
    try:
        session = stripe.checkout.Session.retrieve(session_id)
        if session.payment_status != "paid":
            return None
        credits = int(session.metadata.get("credits", 0))
        user_id = session.metadata.get("user_id")
        if credits <= 0 or not user_id:
            return None
        return {"credits": credits, "user_id": user_id}
    except Exception:
        return None
