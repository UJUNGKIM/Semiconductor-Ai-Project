"""Login and administrator access decisions for the dashboard.

Administrator screens are granted per signed-in user from identity-token
claims matched against allowlists kept outside the code, in environment
variables or the ``[shapgpt]`` section of Streamlit secrets. Nobody is an
administrator by default. Without login, the local development flag grants
administrator screens only while the server listens on a loopback address,
which remote viewers cannot reach.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass


TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
LOOPBACK_ADDRESSES = frozenset({"localhost", "127.0.0.1", "::1"})
REQUIRE_LOGIN_ENV = "SHAPGPT_REQUIRE_LOGIN"
LOCAL_ADMIN_ENV = "SHAPGPT_ADMIN_MODE"
ADMIN_EMAILS_ENV = "SHAPGPT_ADMIN_EMAILS"
ADMIN_SUBJECTS_ENV = "SHAPGPT_ADMIN_SUBJECTS"
SECRETS_SECTION = "shapgpt"

SECOM_ADMIN_SECTIONS = frozenset({"monitoring", "release"})
WM_ADMIN_SECTIONS = frozenset({"research_log"})
ADMIN_BASIS_LABELS = {
    "verified_email_allowlist": "인증된 이메일 허용 목록",
    "subject_allowlist": "발급자·주체(iss|sub) 허용 목록",
    "local_development": "로컬 개발 모드(루프백 주소 전용)",
    "none": "없음",
}
# st.login needs these in [auth]; the identity provider's settings live either
# in [auth] itself or in a named section such as [auth.google].
AUTH_SHARED_KEYS = ("redirect_uri", "cookie_secret")
AUTH_PROVIDER_KEYS = ("client_id", "client_secret", "server_metadata_url")
GOOGLE_PROVIDER = "google"


@dataclass(frozen=True)
class AccessContext:
    login_required: bool
    logged_in: bool
    is_admin: bool
    admin_basis: str
    user_label: str | None = None
    notice: str | None = None
    display_name: str | None = None
    email: str | None = None

    @property
    def role_label(self) -> str:
        return "관리자" if self.is_admin else "일반 사용자"


def env_flag(env: Mapping[str, str], name: str) -> bool:
    return str(env.get(name, "")).strip().lower() in TRUE_VALUES


def _split_entries(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return value.replace(";", ",").replace("\n", ",").split(",")
    if isinstance(value, Iterable):
        return [str(item) for item in value]
    raise TypeError("관리자 허용 목록은 문자열 또는 문자열 목록이어야 합니다.")


def parse_admin_emails(value: object) -> frozenset[str]:
    return frozenset(
        entry.strip().lower() for entry in _split_entries(value) if "@" in entry
    )


def parse_admin_subjects(value: object) -> frozenset[str]:
    """Parse ``issuer|subject`` pairs; OIDC subjects are case-sensitive."""
    return frozenset(
        entry.strip() for entry in _split_entries(value) if entry.count("|") == 1
        and all(part.strip() for part in entry.split("|"))
    )


def admin_allowlists(
    env: Mapping[str, str], secrets: Mapping[str, object] | None
) -> tuple[frozenset[str], frozenset[str]]:
    section = (secrets or {}).get(SECRETS_SECTION) or {}
    if not isinstance(section, Mapping):
        section = {}
    emails = parse_admin_emails(env.get(ADMIN_EMAILS_ENV)) | parse_admin_emails(
        section.get("admin_emails")
    )
    subjects = parse_admin_subjects(env.get(ADMIN_SUBJECTS_ENV)) | parse_admin_subjects(
        section.get("admin_subjects")
    )
    return emails, subjects


def _claim_is_true(value: object) -> bool:
    return value is True or (isinstance(value, str) and value.strip().lower() == "true")


def resolve_access(
    *,
    env: Mapping[str, str],
    claims: Mapping[str, object] | None,
    secrets: Mapping[str, object] | None,
    server_address: str | None,
) -> AccessContext:
    """Decide login and administrator access for the current viewer."""
    claims = claims or {}
    login_required = env_flag(env, REQUIRE_LOGIN_ENV)
    local_flag = env_flag(env, LOCAL_ADMIN_ENV)
    logged_in = _claim_is_true(claims.get("is_logged_in"))

    if login_required:
        if not logged_in:
            return AccessContext(True, False, False, "none")
        emails, subjects = admin_allowlists(env, secrets)
        email = str(claims.get("email") or "").strip().lower()
        issuer = str(claims.get("iss") or "").strip()
        subject = str(claims.get("sub") or "").strip()
        if issuer and subject and f"{issuer}|{subject}" in subjects:
            basis = "subject_allowlist"
        elif email and _claim_is_true(claims.get("email_verified")) and email in emails:
            basis = "verified_email_allowlist"
        else:
            basis = "none"
        notice = (
            f"{LOCAL_ADMIN_ENV}는 로그인 필수 모드에서 무시됩니다." if local_flag else None
        )
        name = str(claims.get("name") or "").strip()
        label = email or name or "로그인 사용자"
        return AccessContext(
            True,
            True,
            basis != "none",
            basis,
            label,
            notice,
            display_name=name or None,
            email=email or None,
        )

    address = str(server_address or "").strip().lower()
    if local_flag and address in LOOPBACK_ADDRESSES:
        return AccessContext(False, False, True, "local_development")
    notice = (
        f"{LOCAL_ADMIN_ENV}는 로그인 없는 로컬 실행(server.address가 루프백 주소)에서만 "
        "적용됩니다."
        if local_flag
        else None
    )
    return AccessContext(False, False, False, "none", notice=notice)


def section_allowed(
    section: str, admin_sections: frozenset[str], access: AccessContext
) -> bool:
    return section not in admin_sections or access.is_admin


def current_user_claims() -> dict[str, object]:
    """Read identity-token claims from Streamlit without failing when absent."""
    import streamlit as st

    try:
        return dict(st.user)
    except Exception:
        return {}


def streamlit_secrets() -> dict[str, object]:
    """Return only the dashboard's own secrets section, or nothing."""
    import streamlit as st

    try:
        section = st.secrets.get(SECRETS_SECTION)
    except Exception:
        return {}
    if section is None:
        return {}
    try:
        return {SECRETS_SECTION: dict(section)}
    except Exception:
        return {}


def _filled(section: Mapping[str, object], keys: tuple[str, ...]) -> bool:
    return all(str(section.get(key) or "").strip() for key in keys)


def login_provider(auth_section: object) -> str | None:
    """Return the ``st.login`` provider name for a complete ``[auth]`` section.

    ``"google"`` means a filled ``[auth.google]`` section and ``""`` the
    unnamed provider configured directly in ``[auth]``. ``None`` means login
    cannot work, so the dashboard must not offer it.
    """
    if not isinstance(auth_section, Mapping) or not _filled(auth_section, AUTH_SHARED_KEYS):
        return None
    google = auth_section.get(GOOGLE_PROVIDER)
    if isinstance(google, Mapping) and _filled(google, AUTH_PROVIDER_KEYS):
        return GOOGLE_PROVIDER
    if _filled(auth_section, AUTH_PROVIDER_KEYS):
        return ""
    return None


def configured_login_provider() -> str | None:
    """Read ``[auth]`` from Streamlit secrets without failing when absent."""
    import streamlit as st

    try:
        return login_provider(st.secrets.get("auth"))
    except Exception:
        return None


def auth_configured() -> bool:
    """True only when Streamlit secrets hold a complete login configuration."""
    return configured_login_provider() is not None


def start_login() -> None:
    """Button callback that starts OIDC login, or does nothing without setup."""
    import streamlit as st

    provider = configured_login_provider()
    if provider is None:
        return
    st.login(provider or None)
