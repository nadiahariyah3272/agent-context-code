"""Utilities for Hugging Face authentication discovery and error messaging."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional

HF_TOKEN_ENV_VARS = (
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
)
AUTH_ERROR_INDICATORS = (
    "401",
    "unauthorized",
    "gated repo",
    "restricted",
    "access to model",
)


def _unique_paths(paths: Iterable[Path]) -> list[Path]:
    """Return paths in order without duplicates."""
    seen: set[str] = set()
    unique: list[Path] = []
    for path in paths:
        normalized = str(path)
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append(path)
    return unique


def _candidate_home_dirs() -> list[Path]:
    """Return possible home directories across Linux, macOS, cmd, PowerShell, and Git Bash."""
    candidates: list[Path] = [Path.home()]

    for env_var in ("HOME", "USERPROFILE"):
        value = os.environ.get(env_var)
        if value:
            candidates.append(Path(value).expanduser())

    home_drive = os.environ.get("HOMEDRIVE")
    home_path = os.environ.get("HOMEPATH")
    if home_drive and home_path:
        candidates.append(Path(f"{home_drive}{home_path}").expanduser())

    return _unique_paths(candidates)


def _candidate_token_paths() -> list[Path]:
    """Return likely Hugging Face token file locations."""
    candidates: list[Path] = []

    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        candidates.append(Path(hf_home) / "token")

    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache_home:
        candidates.append(Path(xdg_cache_home) / "huggingface" / "token")

    for home_dir in _candidate_home_dirs():
        candidates.extend([
            home_dir / ".cache" / "huggingface" / "token",
            home_dir / ".huggingface" / "token",
            home_dir / "AppData" / "Local" / "huggingface" / "token",
        ])

    return _unique_paths(candidates)


def _read_token_file(token_path: Path) -> Optional[str]:
    """Read a token from a known Hugging Face token file if present."""
    try:
        token = token_path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return token or None


def get_huggingface_token() -> Optional[str]:
    """Return the first available Hugging Face token from env, CLI cache, or fallback paths."""
    for env_var in HF_TOKEN_ENV_VARS:
        token = os.environ.get(env_var, "").strip()
        if token:
            return token

    try:
        from huggingface_hub import get_token

        token = get_token()
        if token:
            return token.strip()
    except Exception:
        pass

    for token_path in _candidate_token_paths():
        token = _read_token_file(token_path)
        if token:
            return token

    return None


def configure_huggingface_auth() -> Optional[str]:
    """Populate standard Hugging Face env vars when a token can be discovered."""
    token = get_huggingface_token()
    if not token:
        return None

    for env_var in HF_TOKEN_ENV_VARS:
        os.environ.setdefault(env_var, token)

    return token


def build_huggingface_auth_error(model_name: str, exc: Exception) -> str:
    """Create a more actionable error message for gated or unauthorized model downloads."""
    error_text = str(exc)
    lower_error = error_text.lower()

    if not any(term in lower_error for term in AUTH_ERROR_INDICATORS):
        return error_text

    return (
        f"{error_text}\n\n"
        f"Unable to download gated model '{model_name}'. To fix this:\n"
        f"1. Visit https://huggingface.co/{model_name} and accept the model terms.\n"
        "2. Authenticate with the current Hugging Face CLI:\n"
        "   hf auth login\n"
        "   hf auth whoami\n"
        "3. If you are running the installer from a different shell than the one you used for login,\n"
        "   export the token in the same shell before retrying:\n"
        "   - PowerShell: $env:HF_TOKEN='hf_xxx'\n"
        "   - cmd.exe: set HF_TOKEN=hf_xxx\n"
        "   - bash/zsh: export HF_TOKEN=hf_xxx\n"
        "   The legacy HUGGING_FACE_HUB_TOKEN variable is also supported.\n"
        "4. On Windows with Git Bash/MSYS, a different HOME directory can hide the token cached by\n"
        "   cmd.exe or PowerShell. Setting HF_TOKEN explicitly avoids that mismatch."
    )
