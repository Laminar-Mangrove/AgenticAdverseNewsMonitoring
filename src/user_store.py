"""Supabase email auth and server-side credit balances."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from supabase import Client, create_client

from .config import (
    FREE_CREDITS,
    SUPABASE_ANON_KEY,
    SUPABASE_SERVICE_ROLE_KEY,
    SUPABASE_URL,
    is_supabase_configured,
)


@dataclass
class AuthUser:
    id: str
    email: str
    access_token: str


def is_auth_configured() -> bool:
    return is_supabase_configured()


def _anon_client() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_ANON_KEY)


def _service_client() -> Client:
    if not SUPABASE_SERVICE_ROLE_KEY:
        raise RuntimeError("SUPABASE_SERVICE_ROLE_KEY is required for credit updates.")
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


def sign_up(email: str, password: str, captcha_token: Optional[str] = None) -> tuple[Optional[AuthUser], str]:
    """Register a new user. Trigger creates user_credits row with FREE_CREDITS."""
    if not is_auth_configured():
        return None, "Supabase is not configured."

    try:
        options: dict[str, Any] = {}
        if captcha_token:
            options["captcha_token"] = captcha_token

        client = _anon_client()
        result = client.auth.sign_up(
            {
                "email": email.strip().lower(),
                "password": password,
                "options": options,
            }
        )
        user = result.user
        session = result.session
        if not user:
            return None, "Sign up failed. Please try again."

        if not session:
            return None, (
                "Account created. Check your email to confirm your address, "
                "then sign in."
            )

        _ensure_user_credits(user.id, user.email or email)
        return AuthUser(
            id=user.id,
            email=user.email or email,
            access_token=session.access_token,
        ), ""
    except Exception as exc:
        return None, str(exc)


def sign_in(email: str, password: str, captcha_token: Optional[str] = None) -> tuple[Optional[AuthUser], str]:
    if not is_auth_configured():
        return None, "Supabase is not configured."

    try:
        payload: dict[str, Any] = {
            "email": email.strip().lower(),
            "password": password,
        }
        if captcha_token:
            payload["options"] = {"captcha_token": captcha_token}

        client = _anon_client()
        result = client.auth.sign_in_with_password(payload)
        user = result.user
        session = result.session
        if not user or not session:
            return None, "Invalid email or password."

        _ensure_user_credits(user.id, user.email or email)
        return AuthUser(
            id=user.id,
            email=user.email or email,
            access_token=session.access_token,
        ), ""
    except Exception as exc:
        return None, str(exc)


def _ensure_user_credits(user_id: str, email: str) -> None:
    """Create a credits row if the auth trigger did not run yet."""
    service = _service_client()
    existing = (
        service.table("user_credits")
        .select("user_id")
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )
    if existing.data:
        return

    service.table("user_credits").insert(
        {
            "user_id": user_id,
            "email": email.strip().lower(),
            "credits": FREE_CREDITS,
        }
    ).execute()


def get_credits(user_id: str) -> int:
    service = _service_client()
    result = (
        service.table("user_credits")
        .select("credits")
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )
    if not result.data:
        return FREE_CREDITS
    return int(result.data[0]["credits"])


def deduct_credit(user_id: str) -> tuple[bool, int]:
    """Atomically deduct one credit. Returns (success, new_balance)."""
    service = _service_client()
    result = service.rpc("deduct_credit", {"p_user_id": user_id}).execute()
    new_balance = int(result.data)
    if new_balance < 0:
        return False, 0
    return True, new_balance


def refund_credit(user_id: str) -> int:
    service = _service_client()
    result = service.rpc("add_credits", {"p_user_id": user_id, "p_amount": 1}).execute()
    return int(result.data)


def add_credits(user_id: str, amount: int) -> int:
    if amount <= 0:
        return get_credits(user_id)

    service = _service_client()
    result = service.rpc("add_credits", {"p_user_id": user_id, "p_amount": amount}).execute()
    return int(result.data)


def record_stripe_session(session_id: str, user_id: str, credits: int) -> bool:
    """
    Idempotently record a paid Stripe session.
    Returns True only when credits were newly applied.
    """
    service = _service_client()
    existing = (
        service.table("stripe_sessions")
        .select("session_id")
        .eq("session_id", session_id)
        .limit(1)
        .execute()
    )
    if existing.data:
        return False

    service.table("stripe_sessions").insert(
        {
            "session_id": session_id,
            "user_id": user_id,
            "credits": credits,
        }
    ).execute()
    add_credits(user_id, credits)
    return True
