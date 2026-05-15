"""
shared/config.py — App registry, paths, and JSON persistence for Lora Training Suite 2.0
Designed by: Zero  |  Built by: Jarvis (v2.0)
"""

import os
import json
import socket
from urllib.parse import urlparse

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE      = os.path.dirname(os.path.abspath(__file__))
SUITE_ROOT = os.path.dirname(_HERE)
APP_FILE   = os.path.join(SUITE_ROOT, "launcher", "apps.json")
HF_CFG     = os.path.join(SUITE_ROOT, "launcher", "hf_config.json")
TAGS_CFG   = os.path.join(SUITE_ROOT, "tags", "tagger_config.json")
CALC_CFG   = os.path.join(SUITE_ROOT, "calculator", "calc_config.json")
HEALTH_CFG = os.path.join(SUITE_ROOT, "health",     "health_config.json")

# ── Default web apps ──────────────────────────────────────────────────────────
DEFAULT_APPS: list[dict] = [
    {"name": "SD.next",    "url": "http://localhost:7000"},
    {"name": "AI-Toolkit", "url": "http://192.168.1.64:8675"},
    {"name": "Kohya",      "url": "http://localhost:7862"},
]


# ── App registry I/O ──────────────────────────────────────────────────────────
def load_apps() -> list[dict]:
    """Load saved app list from apps.json, falling back to defaults."""
    if os.path.exists(APP_FILE):
        try:
            with open(APP_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
        except Exception:
            pass
    return [dict(a) for a in DEFAULT_APPS]


def save_apps(apps: list[dict]) -> None:
    """Persist app list to apps.json."""
    os.makedirs(os.path.dirname(APP_FILE), exist_ok=True)
    with open(APP_FILE, "w", encoding="utf-8") as f:
        json.dump(apps, f, indent=2)


# ── Generic config I/O ────────────────────────────────────────────────────────
def load_json(path: str, defaults: dict) -> dict:
    """Load a JSON config file, merging with defaults for missing keys."""
    cfg = dict(defaults)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                stored = json.load(f)
                if isinstance(stored, dict):
                    cfg.update(stored)
        except Exception:
            pass
    return cfg


def save_json(path: str, data: dict) -> None:
    """Persist a dict to a JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ── HuggingFace token ─────────────────────────────────────────────────────────
def load_hf_token() -> str:
    """Return the saved HuggingFace token, or empty string."""
    try:
        if os.path.exists(HF_CFG):
            with open(HF_CFG, "r", encoding="utf-8") as f:
                return json.load(f).get("token", "")
    except Exception:
        pass
    return ""


def save_hf_token(token: str) -> None:
    """Persist HuggingFace token to disk."""
    os.makedirs(os.path.dirname(HF_CFG), exist_ok=True)
    with open(HF_CFG, "w", encoding="utf-8") as f:
        json.dump({"token": token.strip()}, f)


def apply_hf_token(token: str) -> bool:
    """
    Set HF_TOKEN env var and call huggingface_hub.login().
    Returns True on success, False if token is empty or login fails.
    """
    token = token.strip()
    if not token:
        return False
    os.environ["HF_TOKEN"] = token
    try:
        from huggingface_hub import login
        login(token=token, add_to_git_credential=False)
        return True
    except Exception:
        return False


# ── Network helpers ───────────────────────────────────────────────────────────
def check_port(url: str) -> bool:
    """TCP-connect check. Returns True if the host:port is listening."""
    try:
        p      = urlparse(url)
        host   = p.hostname or "localhost"
        scheme = (p.scheme or "http").lower()
        port   = p.port or (443 if scheme == "https" else 80)
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except Exception:
        return False
