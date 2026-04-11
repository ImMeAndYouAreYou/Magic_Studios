from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from sales_bot.exceptions import ConfigurationError


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ConfigurationError(f"Missing required environment variable: {name}")
    return value


def _optional_int(name: str) -> int | None:
    raw = os.getenv(name, "").strip()
    return int(raw) if raw else None


def _optional_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class Settings:
    discord_token: str
    discord_client_id: int
    discord_client_secret: str
    owner_user_id: int
    vouch_channel_id: int
    roblox_client_id: str
    roblox_client_secret: str
    roblox_redirect_uri: str
    roblox_entry_link: str
    roblox_privacy_policy_url: str
    roblox_terms_url: str
    public_base_url: str
    paypal_webhook_token: str
    web_host: str
    web_port: int
    sqlite_path: Path
    data_dir: Path
    log_level: str
    sync_commands_on_startup: bool
    dev_guild_id: int | None

    @classmethod
    def from_env(cls) -> "Settings":
        base_dir = Path(__file__).resolve().parent.parent
        sqlite_path = Path(os.getenv("SQLITE_PATH", "data/bot.sqlite3"))
        if not sqlite_path.is_absolute():
            sqlite_path = base_dir / sqlite_path

        data_dir = sqlite_path.parent

        settings = cls(
            discord_token=_require_env("DISCORD_TOKEN"),
            discord_client_id=int(_require_env("DISCORD_CLIENT_ID")),
            discord_client_secret=_require_env("DISCORD_CLIENT_SECRET"),
            owner_user_id=int(os.getenv("OWNER_USER_ID", "1204103872348557372")),
            vouch_channel_id=int(os.getenv("VOUCH_CHANNEL_ID", "1492468162372046908")),
            roblox_client_id=_require_env("ROBLOX_CLIENT_ID"),
            roblox_client_secret=_require_env("ROBLOX_CLIENT_SECRET"),
            roblox_redirect_uri=_require_env("ROBLOX_REDIRECT_URI"),
            roblox_entry_link=_require_env("ROBLOX_ENTRY_LINK"),
            roblox_privacy_policy_url=_require_env("ROBLOX_PRIVACY_POLICY_URL"),
            roblox_terms_url=_require_env("ROBLOX_TERMS_URL"),
            public_base_url=os.getenv("PUBLIC_BASE_URL", "http://localhost:8080").rstrip("/"),
            paypal_webhook_token=_require_env("PAYPAL_WEBHOOK_TOKEN"),
            web_host=os.getenv("WEB_HOST", "0.0.0.0"),
            web_port=int(os.getenv("WEB_PORT", "8080")),
            sqlite_path=sqlite_path,
            data_dir=data_dir,
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
            sync_commands_on_startup=_optional_bool("SYNC_COMMANDS_ON_STARTUP", True),
            dev_guild_id=_optional_int("DEV_GUILD_ID"),
        )

        settings.data_dir.mkdir(parents=True, exist_ok=True)
        (settings.data_dir / "systems").mkdir(parents=True, exist_ok=True)
        return settings
