"""Streamlit wrapper for Cloudflare Turnstile."""
import os

import streamlit.components.v1 as components

_component = components.declare_component(
    "turnstile",
    path=os.path.join(os.path.dirname(__file__), "frontend"),
)


def turnstile_widget(site_key: str, key: str | None = None) -> str | None:
    """Render Turnstile and return the response token, or None until completed."""
    if not site_key:
        return None
    return _component(site_key=site_key, key=key, default=None)
