"""Administrator access decisions for the dashboard.

Administrator screens (operations monitoring, release readiness and the full
WM-811K experiment log) appear only in the local development mode:
``SHAPGPT_ADMIN_MODE=1`` while the server listens on a loopback address,
which remote viewers cannot reach. Everyone else sees the public screens, and
links to administrator sections fall back to the default screen.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
LOOPBACK_ADDRESSES = frozenset({"localhost", "127.0.0.1", "::1"})
LOCAL_ADMIN_ENV = "SHAPGPT_ADMIN_MODE"

SECOM_ADMIN_SECTIONS = frozenset({"monitoring", "release"})
WM_ADMIN_SECTIONS = frozenset({"research_log"})
ADMIN_BASIS_LABELS = {
    "local_development": "로컬 개발 모드(루프백 주소 전용)",
    "none": "없음",
}


@dataclass(frozen=True)
class AccessContext:
    is_admin: bool
    admin_basis: str


def env_flag(env: Mapping[str, str], name: str) -> bool:
    return str(env.get(name, "")).strip().lower() in TRUE_VALUES


def resolve_access(
    *, env: Mapping[str, str], server_address: str | None
) -> AccessContext:
    """Decide whether this server shows the administrator screens."""
    address = str(server_address or "").strip().lower()
    if env_flag(env, LOCAL_ADMIN_ENV) and address in LOOPBACK_ADDRESSES:
        return AccessContext(True, "local_development")
    return AccessContext(False, "none")


def section_allowed(
    section: str, admin_sections: frozenset[str], access: AccessContext
) -> bool:
    return section not in admin_sections or access.is_admin
