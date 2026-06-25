"""CAPTCHA helpers: math challenge (always) + optional Cloudflare Turnstile."""
import random
from typing import Optional

import httpx

from .config import TURNSTILE_SECRET_KEY, is_turnstile_configured


def new_math_challenge() -> tuple[str, int]:
    """Return a human-readable question and the expected numeric answer."""
    a = random.randint(2, 12)
    b = random.randint(2, 12)
    return f"What is {a} + {b}?", a + b


def verify_math_answer(expected: int, submitted: Optional[int]) -> bool:
    return submitted is not None and int(submitted) == int(expected)


def verify_turnstile(token: Optional[str], remote_ip: Optional[str] = None) -> bool:
    """Verify a Cloudflare Turnstile token server-side."""
    if not is_turnstile_configured():
        return True
    if not token:
        return False

    payload = {"secret": TURNSTILE_SECRET_KEY, "response": token}
    if remote_ip:
        payload["remoteip"] = remote_ip

    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(
                "https://challenges.cloudflare.com/turnstile/v0/siteverify",
                data=payload,
            )
            resp.raise_for_status()
            return bool(resp.json().get("success"))
    except Exception:
        return False


def captcha_passed(
    math_expected: int,
    math_submitted: Optional[int],
    turnstile_token: Optional[str] = None,
) -> tuple[bool, str]:
    """
    Validate CAPTCHA layers.
    Math challenge is always required; Turnstile is required when configured.
    """
    if not verify_math_answer(math_expected, math_submitted):
        return False, "Incorrect CAPTCHA answer. Please try again."

    if is_turnstile_configured() and not verify_turnstile(turnstile_token):
        return False, "CAPTCHA verification failed. Please complete the challenge and try again."

    return True, ""
