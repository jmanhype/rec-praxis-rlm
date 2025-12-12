"""Shared constants for agents and CLI."""

from __future__ import annotations

from typing import Final

from rec_praxis_rlm.types import Severity

# Severity ordering helpers (higher = more severe).
SEVERITY_ORDER: Final[dict[Severity, int]] = {
    Severity.INFO: 0,
    Severity.LOW: 1,
    Severity.MEDIUM: 2,
    Severity.HIGH: 3,
    Severity.CRITICAL: 4,
}

SEVERITY_ORDER_BY_NAME: Final[dict[str, int]] = {sev.name: order for sev, order in SEVERITY_ORDER.items()}

# Emoji icons for human-readable output.
SEVERITY_ICONS: Final[dict[Severity, str]] = {
    Severity.CRITICAL: "🔴",
    Severity.HIGH: "🟠",
    Severity.MEDIUM: "🟡",
    Severity.LOW: "🟢",
    Severity.INFO: "ℹ️",
}

SEVERITY_ICONS_BY_NAME: Final[dict[str, str]] = {
    sev.name: icon for sev, icon in SEVERITY_ICONS.items()
}

