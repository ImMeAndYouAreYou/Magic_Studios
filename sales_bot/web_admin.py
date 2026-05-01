from __future__ import annotations

import html
import logging
from itertools import zip_longest
from typing import TYPE_CHECKING, Any

import discord
from aiohttp import web

from sales_bot.exceptions import ConfigurationError, PermissionDeniedError, SalesBotError
from sales_bot.models import PollOption

if TYPE_CHECKING:
    from sales_bot.bot import SalesBot


LOGGER = logging.getLogger(__name__)
SUPPORTED_WEBSITE_CURRENCIES = ("ILS", "USD")
DEFAULT_WEBSITE_CURRENCY = "ILS"

POLL_FORM_SCRIPT = """
<script>
function escapeHtml(value) {
    return value
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#39;');
}

function buildOptionRow(label = '', emoji = '') {
    return `
        <div class="option-row">
            <input type="text" name="option_emoji" maxlength="32" placeholder="😀" value="${escapeHtml(emoji)}" required>
            <input type="text" name="option_label" maxlength="120" placeholder="טקסט האפשרות" value="${escapeHtml(label)}" required>
            <button type="button" class="ghost-button danger" onclick="removeOptionRow(this)">הסר</button>
        </div>
    `;
}

function addOptionRow(label = '', emoji = '') {
    const container = document.getElementById('poll-options');
    container.insertAdjacentHTML('beforeend', buildOptionRow(label, emoji));
}

function removeOptionRow(button) {
    const container = document.getElementById('poll-options');
    if (container.children.length <= 2) {
        return;
    }
    button.closest('.option-row').remove();
}
</script>
"""


def admin_html_response(title: str, body: str) -> web.Response:
    content = f"""
    <!doctype html>
    <html lang="he" dir="rtl">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>{html.escape(title)}</title>
        <script>
            (() => {{
                const match = document.cookie.match(/(?:^|; )magic_admin_theme=([^;]+)/);
                if (!match) {{
                    return;
                }}
                const theme = decodeURIComponent(match[1] || '').trim().toLowerCase();
                if (theme === 'default' || theme === 'dark' || theme === 'light') {{
                    document.documentElement.dataset.theme = theme;
                }}
            }})();
        </script>
        <style>
            :root {{
                color-scheme: dark;
                --bg-radial-1: rgba(48, 198, 255, 0.15);
                --bg-radial-2: rgba(109, 167, 255, 0.16);
                --bg-start: #2f445d;
                --bg-mid: #334a64;
                --bg-end: #2b4059;
                --panel: rgba(35, 51, 71, 0.88);
                --panel-border: rgba(173, 201, 228, 0.18);
                --surface-card: rgba(45, 64, 88, 0.78);
                --surface-strong: rgba(31, 46, 64, 0.9);
                --surface-soft: rgba(208, 226, 245, 0.06);
                --surface-border: rgba(177, 205, 231, 0.14);
                --surface-border-strong: rgba(177, 205, 231, 0.24);
                --surface-hero-start: rgba(58, 82, 111, 0.92);
                --surface-hero-end: rgba(43, 61, 84, 0.86);
                --text: #f8fbff;
                --muted: #cad7e5;
                --accent: #29c8ff;
                --accent-strong: #138fd0;
                --accent-soft: rgba(41, 200, 255, 0.13);
                --accent-border: rgba(41, 200, 255, 0.3);
                --danger: #ff8579;
                --danger-soft: rgba(255, 133, 121, 0.12);
                --danger-border: rgba(255, 133, 121, 0.24);
                --warning-soft: rgba(255, 210, 112, 0.14);
                --warning-border: rgba(255, 210, 112, 0.3);
                --warning-text: #ffefc2;
                --success-soft: rgba(84, 228, 190, 0.14);
                --success-border: rgba(84, 228, 190, 0.24);
                --success-text: #e6fff8;
                --field: rgba(30, 45, 62, 0.92);
                --button-text: #f8fcff;
                --shadow-lg: 0 28px 80px rgba(8, 16, 28, 0.28);
                --shadow-md: 0 18px 42px rgba(8, 16, 28, 0.18);
            }}
            html[data-theme="dark"] {{
                color-scheme: dark;
                --bg-radial-1: rgba(45, 211, 255, 0.16);
                --bg-radial-2: rgba(77, 127, 244, 0.18);
                --bg-start: #141f2c;
                --bg-mid: #172535;
                --bg-end: #111a26;
                --panel: rgba(20, 31, 45, 0.92);
                --panel-border: rgba(146, 179, 212, 0.18);
                --surface-card: rgba(26, 38, 54, 0.84);
                --surface-strong: rgba(18, 28, 41, 0.92);
                --surface-soft: rgba(173, 196, 218, 0.06);
                --surface-border: rgba(146, 179, 212, 0.16);
                --surface-border-strong: rgba(146, 179, 212, 0.24);
                --surface-hero-start: rgba(31, 47, 68, 0.96);
                --surface-hero-end: rgba(20, 30, 43, 0.92);
                --field: rgba(15, 24, 35, 0.92);
            }}
            html[data-theme="light"] {{
                color-scheme: light;
                --bg-radial-1: rgba(48, 198, 255, 0.12);
                --bg-radial-2: rgba(109, 167, 255, 0.14);
                --bg-start: #eef4fb;
                --bg-mid: #e7eef7;
                --bg-end: #dde6f2;
                --panel: rgba(250, 252, 255, 0.92);
                --panel-border: rgba(88, 120, 154, 0.16);
                --surface-card: rgba(255, 255, 255, 0.86);
                --surface-strong: rgba(241, 246, 252, 0.96);
                --surface-soft: rgba(54, 90, 128, 0.06);
                --surface-border: rgba(88, 120, 154, 0.14);
                --surface-border-strong: rgba(88, 120, 154, 0.22);
                --surface-hero-start: rgba(248, 251, 255, 0.98);
                --surface-hero-end: rgba(232, 239, 247, 0.94);
                --text: #1f3146;
                --muted: #536a83;
                --accent: #0f8fc4;
                --accent-strong: #0a6f9c;
                --accent-soft: rgba(15, 143, 196, 0.1);
                --accent-border: rgba(15, 143, 196, 0.2);
                --danger: #b83c32;
                --danger-soft: rgba(184, 60, 50, 0.08);
                --danger-border: rgba(184, 60, 50, 0.18);
                --warning-soft: rgba(201, 143, 35, 0.1);
                --warning-border: rgba(201, 143, 35, 0.18);
                --warning-text: #93620f;
                --success-soft: rgba(11, 132, 109, 0.1);
                --success-border: rgba(11, 132, 109, 0.18);
                --success-text: #0d6758;
                --field: rgba(248, 251, 255, 0.96);
                --button-text: #f8fcff;
                --shadow-lg: 0 28px 72px rgba(45, 71, 100, 0.16);
                --shadow-md: 0 16px 34px rgba(45, 71, 100, 0.12);
            }}
            * {{ box-sizing: border-box; }}
            html {{ min-height: 100%; }}
            body {{
                margin: 0;
                min-height: 100vh;
                font-family: "Segoe UI", Arial, sans-serif;
                color: var(--text);
                background:
                    radial-gradient(circle at 18% 10%, var(--bg-radial-1), transparent 34%),
                    radial-gradient(circle at 82% 2%, var(--bg-radial-2), transparent 30%),
                    linear-gradient(180deg, var(--bg-start) 0%, var(--bg-mid) 52%, var(--bg-end) 100%);
                background-attachment: fixed;
            }}
            body::before,
            body::after {{
                content: "";
                position: fixed;
                inset: auto;
                width: 28rem;
                height: 28rem;
                border-radius: 999px;
                pointer-events: none;
                filter: blur(12px);
                opacity: 0.65;
                z-index: 0;
            }}
            body::before {{
                top: -8rem;
                left: -7rem;
                background: radial-gradient(circle, var(--bg-radial-1) 0%, transparent 72%);
            }}
            body::after {{
                top: 10rem;
                right: -10rem;
                background: radial-gradient(circle, var(--bg-radial-2) 0%, transparent 72%);
            }}
            main {{
                width: min(100%, 1680px);
                margin: 0 auto;
                padding: 22px 24px 72px;
                position: relative;
                z-index: 1;
            }}
            .page-utility {{
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 16px;
                flex-wrap: wrap;
                margin-bottom: 18px;
            }}
            .brand-mark {{
                display: inline-flex;
                align-items: center;
                gap: 12px;
                padding: 12px 16px;
                border-radius: 18px;
                border: 1px solid var(--panel-border);
                background: var(--panel);
                color: var(--text);
                text-decoration: none;
                box-shadow: var(--shadow-md);
                backdrop-filter: blur(14px);
            }}
            .brand-mark__dot {{
                width: 12px;
                height: 12px;
                border-radius: 999px;
                background: linear-gradient(135deg, var(--accent) 0%, var(--accent-strong) 100%);
                box-shadow: 0 0 0 6px var(--accent-soft);
                flex: 0 0 auto;
            }}
            .brand-mark span {{
                display: flex;
                flex-direction: column;
                gap: 2px;
                font-weight: 700;
            }}
            .brand-mark small {{
                color: var(--muted);
                font-size: 0.72rem;
                letter-spacing: 0.12em;
            }}
            .utility-actions {{
                display: inline-flex;
                align-items: center;
                gap: 12px;
                flex-wrap: wrap;
            }}
            .utility-label {{
                color: var(--muted);
                font-size: 0.8rem;
                letter-spacing: 0.12em;
            }}
            .theme-toolbar {{
                display: inline-flex;
                align-items: center;
                gap: 8px;
                padding: 8px;
                border-radius: 18px;
                border: 1px solid var(--panel-border);
                background: var(--panel);
                box-shadow: var(--shadow-md);
                backdrop-filter: blur(14px);
            }}
            .theme-pill,
            button.theme-pill {{
                border: 1px solid transparent;
                border-radius: 999px;
                padding: 10px 15px;
                background: transparent;
                color: var(--muted);
                font: inherit;
                font-weight: 700;
                cursor: pointer;
                box-shadow: none;
                transition: background 0.18s ease, color 0.18s ease, border-color 0.18s ease, transform 0.18s ease;
            }}
            .theme-pill:hover,
            button.theme-pill:hover {{
                color: var(--text);
                background: var(--surface-soft);
                border-color: var(--surface-border);
                transform: translateY(-1px);
            }}
            .theme-pill.is-active,
            button.theme-pill.is-active {{
                color: var(--button-text);
                background: linear-gradient(135deg, var(--accent) 0%, var(--accent-strong) 100%);
                border-color: transparent;
                box-shadow: 0 12px 24px rgba(19, 143, 208, 0.22);
            }}
            .shell {{
                position: relative;
                overflow: hidden;
                background: linear-gradient(180deg, rgba(255, 255, 255, 0.02) 0%, rgba(255, 255, 255, 0) 100%), var(--panel);
                border: 1px solid var(--panel-border);
                border-radius: 26px;
                padding: 34px 34px 40px 54px;
                box-shadow: var(--shadow-lg);
                backdrop-filter: blur(18px);
            }}
            .shell::before {{
                content: "";
                position: absolute;
                top: 24px;
                bottom: 24px;
                left: 20px;
                width: 4px;
                border-radius: 999px;
                background: linear-gradient(180deg, transparent 0%, var(--accent) 16%, var(--accent) 84%, transparent 100%);
                opacity: 0.95;
            }}
            .shell > * {{
                position: relative;
            }}
            .eyebrow {{
                margin: 0 0 10px;
                color: var(--accent);
                text-transform: uppercase;
                letter-spacing: 0.18em;
                font-size: 0.76rem;
            }}
            h1 {{
                margin: 0 0 10px;
                font-size: clamp(2.25rem, 4vw, 3.45rem);
                line-height: 1.08;
                letter-spacing: -0.03em;
            }}
            h2,
            h3 {{
                margin: 0 0 10px;
                color: var(--text);
            }}
            p, li, label, small {{
                color: var(--muted);
                line-height: 1.65;
            }}
            strong {{ color: var(--text); }}
            a {{
                color: var(--accent);
                text-decoration-thickness: 0.08em;
                text-underline-offset: 0.18em;
            }}
            a:hover {{ color: var(--text); }}
            form {{ margin-top: 24px; }}
            .grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
                gap: 18px;
            }}
            .field {{ display: flex; flex-direction: column; gap: 8px; }}
            .field-wide {{ grid-column: 1 / -1; }}
            input, select, textarea {{
                width: 100%;
                border: 1px solid var(--surface-border);
                border-radius: 16px;
                background: var(--field);
                color: var(--text);
                padding: 14px 16px;
                font: inherit;
                box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.03);
                transition: border-color 0.18s ease, box-shadow 0.18s ease, background 0.18s ease;
            }}
            input:focus,
            select:focus,
            textarea:focus {{
                outline: none;
                border-color: var(--accent-border);
                box-shadow: 0 0 0 4px var(--accent-soft);
            }}
            textarea {{ min-height: 150px; resize: vertical; }}
            .option-list {{ display: flex; flex-direction: column; gap: 12px; }}
            .option-row {{ display: grid; grid-template-columns: 120px 1fr auto; gap: 12px; align-items: center; }}
            .actions {{ display: flex; gap: 12px; flex-wrap: wrap; margin-top: 26px; }}
            button, .link-button {{
                border: 1px solid transparent;
                border-radius: 999px;
                padding: 13px 19px;
                background: linear-gradient(135deg, var(--accent) 0%, var(--accent-strong) 100%);
                color: var(--button-text);
                font: inherit;
                font-weight: 700;
                text-decoration: none;
                cursor: pointer;
                box-shadow: 0 16px 28px rgba(19, 143, 208, 0.2);
                transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease, background 0.18s ease, color 0.18s ease;
            }}
            button:hover,
            .link-button:hover {{
                transform: translateY(-1px);
                box-shadow: 0 18px 34px rgba(19, 143, 208, 0.24);
            }}
            .ghost-button {{
                background: var(--surface-soft);
                color: var(--text);
                border: 1px solid var(--surface-border);
                box-shadow: none;
            }}
            .danger {{ color: var(--danger); }}
            .notice {{
                border-radius: 20px;
                padding: 15px 18px;
                margin-top: 18px;
                background: var(--danger-soft);
                border: 1px solid var(--danger-border);
                color: var(--danger);
            }}
            .success {{
                background: var(--success-soft);
                border-color: var(--success-border);
                color: var(--success-text);
            }}
            .meta-card {{
                margin-top: 20px;
                padding: 18px;
                border-radius: 20px;
                background: var(--surface-card);
                border: 1px solid var(--surface-border);
            }}
            .marketing-shell {{ display: flex; flex-direction: column; gap: 22px; }}
            .marketing-hero {{
                padding: 28px 30px;
                border-radius: 22px;
                background: linear-gradient(135deg, var(--surface-hero-start) 0%, var(--surface-hero-end) 100%);
                border: 1px solid var(--surface-border-strong);
            }}
            .marketing-panel {{
                padding: 28px 30px;
                border-radius: 22px;
                background: var(--surface-card);
                border: 1px solid var(--surface-border);
                box-shadow: var(--shadow-md);
            }}
            .copy-stack {{ display: flex; flex-direction: column; gap: 16px; }}
            .doc-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 18px; }}
            .doc-card {{
                padding: 22px;
                border-radius: 20px;
                background: var(--surface-strong);
                border: 1px solid var(--surface-border);
            }}
            .doc-list {{ margin: 0; padding-inline-start: 20px; }}
            .inline-link-grid {{ display: flex; gap: 12px; flex-wrap: wrap; margin-top: 10px; }}
            @media (max-width: 700px) {{
                main {{ padding: 18px 16px 52px; }}
                .page-utility {{ align-items: stretch; }}
                .utility-actions {{ justify-content: space-between; width: 100%; }}
                .theme-toolbar {{ width: 100%; justify-content: space-between; }}
                .theme-pill,
                button.theme-pill {{ flex: 1 1 0; justify-content: center; }}
                .shell {{ padding: 22px; border-radius: 22px; }}
                .shell::before {{ display: none; }}
                .marketing-hero,
                .marketing-panel {{ padding: 22px; }}
                .option-row {{ grid-template-columns: 1fr; }}
            }}
        </style>
    </head>
    <body>
        <main>
            <div class="page-utility">
                <a class="brand-mark" href="/">
                    <span class="brand-mark__dot"></span>
                    <span>Magic Studio's<small>דף בית</small></span>
                </a>
            </div>
            <section class="shell">
                {body}
            </section>
        </main>
        <script>
            (() => {{
                const themes = new Set(['default', 'dark', 'light']);
                const root = document.documentElement;
                const buttons = Array.from(document.querySelectorAll('[data-theme-mode]'));

                const readTheme = () => {{
                    const match = document.cookie.match(/(?:^|; )magic_admin_theme=([^;]+)/);
                    if (!match) {{
                        return 'default';
                    }}
                    const theme = decodeURIComponent(match[1] || '').trim().toLowerCase();
                    return themes.has(theme) ? theme : 'default';
                }};

                const syncButtons = (theme) => {{
                    buttons.forEach((button) => {{
                        const isActive = button.dataset.themeMode === theme;
                        button.classList.toggle('is-active', isActive);
                        button.setAttribute('aria-pressed', isActive ? 'true' : 'false');
                    }});
                }};

                const applyTheme = (theme) => {{
                    const safeTheme = themes.has(theme) ? theme : 'default';
                    root.dataset.theme = safeTheme;
                    document.cookie = `magic_admin_theme=${{encodeURIComponent(safeTheme)}}; Max-Age=31536000; Path=/; SameSite=Lax`;
                    syncButtons(safeTheme);
                }};

                const initialTheme = themes.has(root.dataset.theme) ? root.dataset.theme : readTheme();
                root.dataset.theme = initialTheme;
                syncButtons(initialTheme);
                buttons.forEach((button) => {{
                    button.addEventListener('click', () => applyTheme(button.dataset.themeMode || 'default'));
                }});
            }})();
        </script>
    </body>
    </html>
    """
    return web.Response(text=content, content_type="text/html")


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _error_response(title: str, message: str, *, status: int) -> web.Response:
    body = f"""
    <p class="eyebrow">פאנל ניהול</p>
    <h1>{_escape(title)}</h1>
    <div class="notice">{_escape(message)}</div>
    """
    response = admin_html_response(title, body)
    response.set_status(status)
    return response


async def _authorize_panel_request(
    request: web.Request,
    *,
    panel_type: str,
    target_id: int | None = None,
) -> tuple["SalesBot", int]:
    bot: SalesBot = request.app["bot"]
    token = request.query.get("token", "").strip()
    if token:
        session = await bot.services.panels.get_session(token, expected_panel_type=panel_type)
        if target_id is None and session.target_id is not None:
            raise PermissionDeniedError("קישור פאנל הניהול הזה שייך לרשומה אחרת.")
        if target_id is not None and session.target_id != target_id:
            raise PermissionDeniedError("קישור פאנל הניהול הזה לא תואם לרשומה שביקשת.")
        if not await bot.services.admins.is_admin(session.admin_user_id):
            raise PermissionDeniedError("קישור פאנל הניהול הזה כבר לא משויך לאדמין של הבוט.")
        return bot, session.admin_user_id

    website_token = request.cookies.get(bot.services.web_auth.cookie_name, "").strip()
    if not website_token:
        raise PermissionDeniedError("חסר טוקן גישה לפאנל הניהול או שלא בוצעה התחברות לאתר.")

    website_session = await bot.services.web_auth.get_session(website_token)
    if not await bot.services.admins.is_admin(website_session.discord_user_id):
        raise PermissionDeniedError("רק אדמינים של הבוט יכולים לפתוח את פאנל הניהול הזה.")
    return bot, website_session.discord_user_id


async def _list_text_channels(bot: "SalesBot") -> list[discord.TextChannel]:
    if bot.settings.primary_guild_id is None:
        raise ConfigurationError("חובה להגדיר PRIMARY_GUILD_ID לפני שימוש בפאנלי הניהול.")

    try:
        guild = bot.get_guild(bot.settings.primary_guild_id)
        if guild is None:
            guild = await bot.fetch_guild(bot.settings.primary_guild_id)
            channels = await guild.fetch_channels()
        else:
            channels = guild.channels
            if not channels:
                channels = await guild.fetch_channels()
    except discord.Forbidden as exc:
        raise PermissionDeniedError(
            "לבוט אין הרשאה לטעון ערוצים מתוך PRIMARY_GUILD_ID."
        ) from exc
    except discord.HTTPException as exc:
        raise ConfigurationError(
            "לא הצלחתי לטעון ערוצים מתוך PRIMARY_GUILD_ID. בדקו שהמזהה נכון ושהבוט נמצא בשרת הזה."
        ) from exc

    text_channels = [channel for channel in channels if isinstance(channel, discord.TextChannel)]
    if not text_channels:
        raise ConfigurationError("לא נמצאו ערוצי טקסט בשרת הראשי שהוגדר.")

    return sorted(text_channels, key=lambda channel: (channel.position, channel.name.casefold()))


def _render_channel_options(channels: list[discord.TextChannel], selected_channel_id: int | None) -> str:
    option_lines: list[str] = []
    for channel in channels:
        selected = " selected" if selected_channel_id == channel.id else ""
        option_lines.append(
            f'<option value="{channel.id}"{selected}>#{_escape(channel.name)}</option>'
        )
    return "\n".join(option_lines)


def _render_duration_unit_options(selected_unit: str) -> str:
    unit_labels = {
        "minutes": "דקות",
        "hours": "שעות",
        "days": "ימים",
        "weeks": "שבועות",
    }
    return "\n".join(
        f'<option value="{unit}"{" selected" if unit == selected_unit else ""}>{label}</option>'
        for unit, label in unit_labels.items()
    )


def _message_link(bot: "SalesBot", channel_id: int, message_id: int | None) -> str | None:
    if bot.settings.primary_guild_id is None or message_id is None:
        return None
    return f"https://discord.com/channels/{bot.settings.primary_guild_id}/{channel_id}/{message_id}"


def _render_success_body(title: str, message: str, *, record_id: int, message_url: str | None) -> str:
    link_html = (
        f'<a class="link-button" href="{_escape(message_url)}" target="_blank" rel="noreferrer">פתח את ההודעה בדיסקורד</a>'
        if message_url
        else ""
    )
    return f"""
    <p class="eyebrow">פאנל ניהול</p>
    <h1>{_escape(title)}</h1>
    <div class="notice success">{_escape(message)}</div>
    <div class="meta-card">
        <p><strong>מזהה שמור:</strong> {_escape(record_id)}</p>
        <p>אפשר להשתמש במזהה הזה אחר כך עם פקודת העריכה המתאימה.</p>
        <div class="actions">{link_html}</div>
    </div>
    """


def _poll_form_defaults() -> dict[str, Any]:
    return {
        "question": "",
        "channel_id": None,
        "duration_value": 1,
        "duration_unit": "hours",
        "options": [{"label": "", "emoji": ""}, {"label": "", "emoji": ""}],
    }


def _poll_values_from_record(poll: Any) -> dict[str, Any]:
    return {
        "question": poll.question,
        "channel_id": poll.channel_id,
        "duration_value": poll.duration_value,
        "duration_unit": poll.duration_unit,
        "options": [{"label": option.label, "emoji": option.emoji} for option in poll.options],
    }


def _extract_poll_form_values(post_data: Any) -> dict[str, Any]:
    labels = post_data.getall("option_label", [])
    emojis = post_data.getall("option_emoji", [])
    options = [
        {"label": str(label), "emoji": str(emoji)}
        for label, emoji in zip_longest(labels, emojis, fillvalue="")
    ]
    return {
        "question": str(post_data.get("question", "")),
        "channel_id": int(str(post_data.get("channel_id", "0")) or 0),
        "duration_value": int(str(post_data.get("duration_value", "0")) or 0),
        "duration_unit": str(post_data.get("duration_unit", "hours")),
        "options": options,
    }


def _build_poll_options(form_values: dict[str, Any]) -> list[PollOption]:
    return [
        PollOption(emoji=str(item.get("emoji", "")), label=str(item.get("label", "")))
        for item in form_values["options"]
    ]


def _render_poll_form(
    *,
    mode_label: str,
    channels: list[discord.TextChannel],
    values: dict[str, Any],
    error_text: str | None = None,
) -> str:
    option_rows = "\n".join(
        f"""
        <div class="option-row">
            <input type="text" name="option_emoji" maxlength="32" placeholder="😀" value="{_escape(option['emoji'])}" required>
            <input type="text" name="option_label" maxlength="120" placeholder="טקסט האפשרות" value="{_escape(option['label'])}" required>
            <button type="button" class="ghost-button danger" onclick="removeOptionRow(this)">הסר</button>
        </div>
        """
        for option in values["options"]
    )
    error_html = f'<div class="notice">{_escape(error_text)}</div>' if error_text else ""
    return f"""
    <p class="eyebrow">פאנל סקרים</p>
    <h1>{_escape(mode_label)} סקר</h1>
    <p>צרו או ערכו סקר בסגנון דיסקורד עם כמה אפשרויות שתרצו. לכל אפשרות חייב להיות אימוג'י משלה.</p>
    {error_html}
    <form method="post">
        <div class="grid">
            <label class="field field-wide">
                <span>שאלה</span>
                <textarea name="question" required>{_escape(values['question'])}</textarea>
            </label>
            <label class="field">
                <span>שלח לערוץ</span>
                <select name="channel_id" required>
                    {_render_channel_options(channels, values['channel_id'])}
                </select>
            </label>
            <label class="field">
                <span>משך זמן</span>
                <input type="number" min="1" name="duration_value" value="{_escape(values['duration_value'])}" required>
            </label>
            <label class="field">
                <span>יחידת זמן</span>
                <select name="duration_unit" required>
                    {_render_duration_unit_options(str(values['duration_unit']))}
                </select>
            </label>
            <div class="field field-wide">
                <span>אפשרויות הסקר</span>
                <div class="option-list" id="poll-options">
                    {option_rows}
                </div>
                <div class="actions">
                    <button type="button" class="ghost-button" onclick="addOptionRow()">הוסף אפשרות</button>
                </div>
            </div>
        </div>
        <div class="actions">
            <button type="submit">{_escape(mode_label)} סקר</button>
        </div>
    </form>
    {POLL_FORM_SCRIPT}
    """


def _giveaway_form_defaults() -> dict[str, Any]:
    return {
        "title": "",
        "description": "",
        "requirements": "",
        "channel_id": None,
        "winner_count": 1,
        "duration_value": 1,
        "duration_unit": "hours",
    }


def _giveaway_values_from_record(giveaway: Any) -> dict[str, Any]:
    return {
        "title": giveaway.title,
        "description": giveaway.description or "",
        "requirements": giveaway.requirements or "",
        "channel_id": giveaway.channel_id,
        "winner_count": giveaway.winner_count,
        "duration_value": giveaway.duration_value,
        "duration_unit": giveaway.duration_unit,
    }


def _extract_giveaway_form_values(post_data: Any) -> dict[str, Any]:
    return {
        "title": str(post_data.get("title", "")),
        "description": str(post_data.get("description", "")),
        "requirements": str(post_data.get("requirements", "")),
        "channel_id": int(str(post_data.get("channel_id", "0")) or 0),
        "winner_count": int(str(post_data.get("winner_count", "0")) or 0),
        "duration_value": int(str(post_data.get("duration_value", "0")) or 0),
        "duration_unit": str(post_data.get("duration_unit", "hours")),
    }


def _render_giveaway_form(
    *,
    mode_label: str,
    channels: list[discord.TextChannel],
    values: dict[str, Any],
    error_text: str | None = None,
) -> str:
    error_html = f'<div class="notice">{_escape(error_text)}</div>' if error_text else ""
    return f"""
    <p class="eyebrow">פאנל הגרלות</p>
    <h1>{_escape(mode_label)} הגרלה</h1>
    <p>צרו או ערכו הגרלה, בחרו לאיזה ערוץ לשלוח אותה, כמה זמן היא תימשך, כמה זוכים יהיו ומה יוצג בתנאים.</p>
    {error_html}
    <form method="post">
        <div class="grid">
            <label class="field field-wide">
                <span>כותרת ההגרלה</span>
                <input type="text" maxlength="180" name="title" value="{_escape(values['title'])}" required>
            </label>
            <label class="field field-wide">
                <span>תיאור</span>
                <textarea name="description">{_escape(values['description'])}</textarea>
            </label>
            <label class="field field-wide">
                <span>דרישות</span>
                <textarea name="requirements">{_escape(values['requirements'])}</textarea>
            </label>
            <label class="field">
                <span>שלח לערוץ</span>
                <select name="channel_id" required>
                    {_render_channel_options(channels, values['channel_id'])}
                </select>
            </label>
            <label class="field">
                <span>מספר זוכים</span>
                <input type="number" min="1" name="winner_count" value="{_escape(values['winner_count'])}" required>
            </label>
            <label class="field">
                <span>משך זמן</span>
                <input type="number" min="1" name="duration_value" value="{_escape(values['duration_value'])}" required>
            </label>
            <label class="field">
                <span>יחידת זמן</span>
                <select name="duration_unit" required>
                    {_render_duration_unit_options(str(values['duration_unit']))}
                </select>
            </label>
        </div>
        <div class="actions">
            <button type="submit">{_escape(mode_label)} הגרלה</button>
        </div>
    </form>
    """


def _event_form_defaults() -> dict[str, Any]:
    return {
        "title": "",
        "description": "",
        "reward": "",
        "channel_id": None,
        "duration_value": 1,
        "duration_unit": "hours",
    }


def _event_values_from_record(event: Any) -> dict[str, Any]:
    return {
        "title": event.title,
        "description": event.description or "",
        "reward": event.reward,
        "channel_id": event.channel_id,
        "duration_value": event.duration_value,
        "duration_unit": event.duration_unit,
    }


def _extract_event_form_values(post_data: Any) -> dict[str, Any]:
    return {
        "title": str(post_data.get("title", "")),
        "description": str(post_data.get("description", "")),
        "reward": str(post_data.get("reward", "")),
        "channel_id": int(str(post_data.get("channel_id", "0")) or 0),
        "duration_value": int(str(post_data.get("duration_value", "0")) or 0),
        "duration_unit": str(post_data.get("duration_unit", "hours")),
    }


def _render_event_form(
    *,
    mode_label: str,
    channels: list[discord.TextChannel],
    values: dict[str, Any],
    error_text: str | None = None,
) -> str:
    error_html = f'<div class="notice">{_escape(error_text)}</div>' if error_text else ""
    return f"""
    <p class="eyebrow">פאנל אירועים</p>
    <h1>{_escape(mode_label)} אירוע</h1>
    <p>צרו או ערכו אירוע עם פרס, תיאור, ערוץ ומשך זמן. ההודעה תפורסם עם ריאקשן כוכב להשתתפות.</p>
    {error_html}
    <form method="post">
        <div class="grid">
            <label class="field field-wide">
                <span>כותרת האירוע</span>
                <input type="text" maxlength="180" name="title" value="{_escape(values['title'])}" required>
            </label>
            <label class="field field-wide">
                <span>תיאור</span>
                <textarea name="description">{_escape(values['description'])}</textarea>
            </label>
            <label class="field field-wide">
                <span>פרס</span>
                <input type="text" maxlength="220" name="reward" value="{_escape(values['reward'])}" required>
            </label>
            <label class="field">
                <span>שלח לערוץ</span>
                <select name="channel_id" required>
                    {_render_channel_options(channels, values['channel_id'])}
                </select>
            </label>
            <label class="field">
                <span>משך זמן</span>
                <input type="number" min="1" name="duration_value" value="{_escape(values['duration_value'])}" required>
            </label>
            <label class="field">
                <span>יחידת זמן</span>
                <select name="duration_unit" required>
                    {_render_duration_unit_options(str(values['duration_unit']))}
                </select>
            </label>
        </div>
        <div class="actions">
            <button type="submit">{_escape(mode_label)} אירוע</button>
        </div>
    </form>
    """


def _system_values_from_record(system: Any) -> dict[str, Any]:
    return {
        "name": system.name,
        "description": system.description,
        "paypal_link": system.paypal_link or "",
        "roblox_gamepass": system.roblox_gamepass_id or "",
        "website_price": system.website_price or "",
        "website_currency": system.website_currency or DEFAULT_WEBSITE_CURRENCY,
        "is_visible_on_website": system.is_visible_on_website,
        "is_for_sale": system.is_for_sale,
        "is_in_stock": system.is_in_stock,
        "is_special_system": getattr(system, "is_special_system", False),
        "replace_images": False,
        "clear_image": False,
    }


def _extract_system_form_values(post_data: Any) -> dict[str, Any]:
    return {
        "name": str(post_data.get("name", "")),
        "description": str(post_data.get("description", "")),
        "paypal_link": str(post_data.get("paypal_link", "")),
        "roblox_gamepass": str(post_data.get("roblox_gamepass", "")),
        "website_price": str(post_data.get("website_price", "")),
        "website_currency": str(post_data.get("website_currency", DEFAULT_WEBSITE_CURRENCY)),
        "is_visible_on_website": str(post_data.get("is_visible_on_website", "")).lower() in {"1", "true", "yes", "on"},
        "is_for_sale": str(post_data.get("is_for_sale", "")).lower() in {"1", "true", "yes", "on"},
        "is_in_stock": str(post_data.get("is_in_stock", "")).lower() in {"1", "true", "yes", "on"},
        "is_special_system": str(post_data.get("is_special_system", "")).lower() in {"1", "true", "yes", "on"},
        "replace_images": str(post_data.get("replace_images", "")).lower() in {"1", "true", "yes", "on"},
        "clear_image": str(post_data.get("clear_image", "")).lower() in {"1", "true", "yes", "on"},
    }


def _extract_upload(field: Any) -> tuple[str, bytes] | None:
    if not isinstance(field, web.FileField) or not field.filename:
        return None
    payload = field.file.read()
    if not payload:
        return None
    return field.filename, payload


def _extract_uploads(fields: list[Any], *, image_only: bool = False) -> list[tuple[str, bytes, str | None]]:
    uploads: list[tuple[str, bytes, str | None]] = []
    for field in fields:
        if not isinstance(field, web.FileField) or not field.filename:
            continue
        if image_only and field.content_type and not field.content_type.startswith("image/"):
            raise PermissionDeniedError("The uploaded image must be a valid image file.")
        payload = field.file.read()
        if not payload:
            continue
        uploads.append((field.filename, payload, field.content_type))
    return uploads

def _website_currency_options(selected_currency: str | None) -> str:
    normalized = str(selected_currency or DEFAULT_WEBSITE_CURRENCY).strip().upper()
    if normalized not in SUPPORTED_WEBSITE_CURRENCIES:
        normalized = DEFAULT_WEBSITE_CURRENCY
    return "".join(
        f'<option value="{currency}"{" selected" if currency == normalized else ""}>{currency}</option>'
        for currency in SUPPORTED_WEBSITE_CURRENCIES
    )


def _render_system_form(
    *,
    system: Any,
    values: dict[str, Any],
    error_text: str | None = None,
) -> str:
    error_html = f'<div class="notice">{_escape(error_text)}</div>' if error_text else ""
    current_image = f"<p>תמונה ראשית נוכחית: {_escape(system.image_path)}</p>" if system.image_path else "<p>אין כרגע תמונה שמורה.</p>"
    return f"""
    <p class="eyebrow">פאנל מערכות</p>
    <h1>עריכת מערכת #{_escape(system.id)}</h1>
    <p>אפשר לעדכן כאן את הטקסט, ההעלאות, גלריית התמונות והגדרות החשיפה של המערכת באתר.</p>
    {error_html}
    <div class="meta-card">
        <p><strong>קובץ נוכחי:</strong> {_escape(system.file_path)}</p>
        {current_image}
    </div>
    <form method="post" enctype="multipart/form-data">
        <div class="grid">
            <label class="field field-wide">
                <span>שם</span>
                <input type="text" maxlength="180" name="name" value="{_escape(values['name'])}" required>
            </label>
            <label class="field field-wide">
                <span>תיאור</span>
                <textarea name="description" required>{_escape(values['description'])}</textarea>
            </label>
            <label class="field">
                <span>קישור פייפאל ישיר (ישן, אופציונלי)</span>
                <input type="url" name="paypal_link" value="{_escape(values['paypal_link'])}">
            </label>
            <label class="field">
                <span>מזהה או קישור גיימפאס רובקס</span>
                <input type="text" name="roblox_gamepass" value="{_escape(values['roblox_gamepass'])}">
            </label>
            <label class="field">
                <span>מחיר באתר / לקופת פייפאל</span>
                <input type="text" name="website_price" inputmode="decimal" placeholder="19.99" value="{_escape(values['website_price'])}">
            </label>
            <label class="field">
                <span>מטבע</span>
                <select name="website_currency">{_website_currency_options(values['website_currency'])}</select>
            </label>
            <label class="field">
                <span>החלפת קובץ מערכת</span>
                <input type="file" name="file">
            </label>
            <label class="field">
                <span>הוספת תמונות לגלריה</span>
                <input type="file" name="images" accept="image/*" multiple>
            </label>
            <label class="meta-card check-card">
                <span class="check-line">
                    <input type="checkbox" name="is_visible_on_website" value="true"{' checked' if values['is_visible_on_website'] else ''}>
                    <strong>להציג את המערכת באתר</strong>
                </span>
            </label>
            <label class="meta-card check-card">
                <span class="check-line">
                    <input type="checkbox" name="is_for_sale" value="true"{' checked' if values['is_for_sale'] else ''}>
                    <strong>המערכת מוצעת כרגע למכירה</strong>
                </span>
            </label>
            <label class="meta-card check-card">
                <span class="check-line">
                    <input type="checkbox" name="is_in_stock" value="true"{' checked' if values['is_in_stock'] else ''}>
                    <strong>המערכת במלאי</strong>
                </span>
            </label>
            <label class="meta-card check-card">
                <span class="check-line">
                    <input type="checkbox" name="is_special_system" value="true"{' checked' if values['is_special_system'] else ''}>
                    <strong>מערכת מיוחדת</strong>
                </span>
            </label>
            <label class="meta-card check-card">
                <span class="check-line">
                    <input type="checkbox" name="replace_images" value="true"{' checked' if values['replace_images'] else ''}>
                    <strong>להחליף את כל גלריית התמונות הקיימת</strong>
                </span>
            </label>
            <label class="field field-wide">
                <span>
                    <input type="checkbox" name="clear_image" value="true"{' checked' if values['clear_image'] else ''}>
                    למחוק לגמרי את כל תמונות המערכת
                </span>
            </label>
        </div>
        <div class="actions">
            <button type="submit">שמור שינויים</button>
        </div>
    </form>
    """


async def poll_create_page(request: web.Request) -> web.Response:
    return await _handle_poll_form(request, poll_id=None, panel_type="poll-create")


async def poll_edit_page(request: web.Request) -> web.Response:
    return await _handle_poll_form(
        request,
        poll_id=int(request.match_info["poll_id"]),
        panel_type="poll-edit",
    )


async def giveaway_create_page(request: web.Request) -> web.Response:
    return await _handle_giveaway_form(request, giveaway_id=None, panel_type="giveaway-create")


async def giveaway_edit_page(request: web.Request) -> web.Response:
    return await _handle_giveaway_form(
        request,
        giveaway_id=int(request.match_info["giveaway_id"]),
        panel_type="giveaway-edit",
    )


async def event_create_page(request: web.Request) -> web.Response:
    return await _handle_event_form(request, event_id=None, panel_type="event-create")


async def event_edit_page(request: web.Request) -> web.Response:
    return await _handle_event_form(
        request,
        event_id=int(request.match_info["event_id"]),
        panel_type="event-edit",
    )


async def system_edit_page(request: web.Request) -> web.Response:
    system_id = int(request.match_info["system_id"])
    try:
        bot, _ = await _authorize_panel_request(request, panel_type="system-edit", target_id=system_id)
        system = await bot.services.systems.get_system(system_id)
        values = _system_values_from_record(system)

        if request.method == "POST":
            form_data = await request.post()
            values = _extract_system_form_values(form_data)
            try:
                image_uploads = _extract_uploads(list(form_data.getall("images", [])), image_only=True)

                updated_system = await bot.services.systems.update_system(
                    system_id,
                    name=values["name"],
                    description=values["description"],
                    paypal_link=values["paypal_link"] or None,
                    roblox_gamepass_reference=values["roblox_gamepass"] or None,
                    website_price=values["website_price"] or None,
                    website_currency=values["website_currency"] or DEFAULT_WEBSITE_CURRENCY,
                    is_visible_on_website=bool(values["is_visible_on_website"]),
                    is_for_sale=bool(values["is_for_sale"]),
                    is_in_stock=bool(values["is_in_stock"]),
                    is_special_system=bool(values["is_special_system"]),
                    file_upload=_extract_upload(form_data.get("file")),
                    image_uploads=image_uploads,
                    replace_images=bool(values["replace_images"]),
                    clear_image=bool(values["clear_image"]),
                )
                return admin_html_response(
                    "המערכת עודכנה",
                    _render_success_body(
                        f"מערכת #{updated_system.id} עודכנה",
                        f"{updated_system.name} עודכנה בהצלחה.",
                        record_id=updated_system.id,
                        message_url=None,
                    ),
                )
            except SalesBotError as exc:
                return admin_html_response(
                    f"עריכת מערכת #{system.id}",
                    _render_system_form(system=system, values=values, error_text=str(exc)),
                )

        return admin_html_response(
            f"עריכת מערכת #{system.id}",
            _render_system_form(system=system, values=values),
        )
    except SalesBotError as exc:
        return _error_response("System Editor", str(exc), status=400)
    except Exception:
        LOGGER.exception("Unexpected system edit panel failure")
        return _error_response("System Editor", "An unexpected error occurred while loading the system editor.", status=500)


async def _handle_poll_form(
    request: web.Request,
    *,
    poll_id: int | None,
    panel_type: str,
) -> web.Response:
    try:
        bot, admin_user_id = await _authorize_panel_request(request, panel_type=panel_type, target_id=poll_id)
        channels = await _list_text_channels(bot)
        values = _poll_form_defaults()
        if poll_id is not None:
            values = _poll_values_from_record(await bot.services.polls.get_editable_poll(poll_id))

        if request.method == "POST":
            form_data = await request.post()
            values = _extract_poll_form_values(form_data)
            try:
                poll_options = _build_poll_options(values)
                if poll_id is None:
                    saved_poll = await bot.services.polls.create_poll(
                        bot,
                        created_by=admin_user_id,
                        channel_id=values["channel_id"],
                        question=values["question"],
                        options=poll_options,
                        duration_value=values["duration_value"],
                        duration_unit=values["duration_unit"],
                    )
                    success_title = f"הסקר #{saved_poll.id} נוצר"
                    success_message = "הסקר פורסם בהצלחה."
                else:
                    saved_poll = await bot.services.polls.update_poll(
                        bot,
                        poll_id,
                        channel_id=values["channel_id"],
                        question=values["question"],
                        options=poll_options,
                        duration_value=values["duration_value"],
                        duration_unit=values["duration_unit"],
                    )
                    success_title = f"הסקר #{saved_poll.id} עודכן"
                    success_message = "הודעת הסקר עודכנה בהצלחה."

                return admin_html_response(
                    success_title,
                    _render_success_body(
                        success_title,
                        success_message,
                        record_id=saved_poll.id,
                        message_url=_message_link(bot, saved_poll.channel_id, saved_poll.message_id),
                    ),
                )
            except SalesBotError as exc:
                return admin_html_response(
                    "פאנל סקרים",
                    _render_poll_form(
                        mode_label="ערוך" if poll_id is not None else "צור",
                        channels=channels,
                        values=values,
                        error_text=str(exc),
                    ),
                )

        return admin_html_response(
            "פאנל סקרים",
            _render_poll_form(
                mode_label="ערוך" if poll_id is not None else "צור",
                channels=channels,
                values=values,
            ),
        )
    except SalesBotError as exc:
        return _error_response("פאנל סקרים", str(exc), status=400)
    except Exception:
        LOGGER.exception("Unexpected poll panel failure")
        return _error_response("פאנל סקרים", "אירעה שגיאה לא צפויה בזמן טעינת פאנל הסקרים.", status=500)


async def _handle_giveaway_form(
    request: web.Request,
    *,
    giveaway_id: int | None,
    panel_type: str,
) -> web.Response:
    try:
        bot, admin_user_id = await _authorize_panel_request(request, panel_type=panel_type, target_id=giveaway_id)
        channels = await _list_text_channels(bot)
        values = _giveaway_form_defaults()
        if giveaway_id is not None:
            values = _giveaway_values_from_record(await bot.services.giveaways.get_editable_giveaway(giveaway_id))

        if request.method == "POST":
            form_data = await request.post()
            values = _extract_giveaway_form_values(form_data)
            try:
                if giveaway_id is None:
                    saved_giveaway = await bot.services.giveaways.create_giveaway(
                        bot,
                        created_by=admin_user_id,
                        channel_id=values["channel_id"],
                        title=values["title"],
                        description=values["description"],
                        requirements=values["requirements"],
                        winner_count=values["winner_count"],
                        duration_value=values["duration_value"],
                        duration_unit=values["duration_unit"],
                    )
                    success_title = f"ההגרלה #{saved_giveaway.id} נוצרה"
                    success_message = "ההגרלה פורסמה בהצלחה."
                else:
                    saved_giveaway = await bot.services.giveaways.update_giveaway(
                        bot,
                        giveaway_id,
                        channel_id=values["channel_id"],
                        title=values["title"],
                        description=values["description"],
                        requirements=values["requirements"],
                        winner_count=values["winner_count"],
                        duration_value=values["duration_value"],
                        duration_unit=values["duration_unit"],
                    )
                    success_title = f"ההגרלה #{saved_giveaway.id} עודכנה"
                    success_message = "הודעת ההגרלה עודכנה בהצלחה."

                return admin_html_response(
                    success_title,
                    _render_success_body(
                        success_title,
                        success_message,
                        record_id=saved_giveaway.id,
                        message_url=_message_link(bot, saved_giveaway.channel_id, saved_giveaway.message_id),
                    ),
                )
            except SalesBotError as exc:
                return admin_html_response(
                    "פאנל הגרלות",
                    _render_giveaway_form(
                        mode_label="ערוך" if giveaway_id is not None else "צור",
                        channels=channels,
                        values=values,
                        error_text=str(exc),
                    ),
                )

        return admin_html_response(
            "פאנל הגרלות",
            _render_giveaway_form(
                mode_label="ערוך" if giveaway_id is not None else "צור",
                channels=channels,
                values=values,
            ),
        )
    except SalesBotError as exc:
        return _error_response("פאנל הגרלות", str(exc), status=400)
    except Exception:
        LOGGER.exception("Unexpected giveaway panel failure")
        return _error_response("פאנל הגרלות", "אירעה שגיאה לא צפויה בזמן טעינת פאנל ההגרלות.", status=500)


async def _handle_event_form(
    request: web.Request,
    *,
    event_id: int | None,
    panel_type: str,
) -> web.Response:
    try:
        bot, admin_user_id = await _authorize_panel_request(request, panel_type=panel_type, target_id=event_id)
        channels = await _list_text_channels(bot)
        values = _event_form_defaults()
        if event_id is not None:
            values = _event_values_from_record(await bot.services.events.get_editable_event(event_id))

        if request.method == "POST":
            form_data = await request.post()
            values = _extract_event_form_values(form_data)
            try:
                if event_id is None:
                    saved_event = await bot.services.events.create_event(
                        bot,
                        created_by=admin_user_id,
                        channel_id=values["channel_id"],
                        title=values["title"],
                        description=values["description"],
                        reward=values["reward"],
                        duration_value=values["duration_value"],
                        duration_unit=values["duration_unit"],
                    )
                    success_title = f"האירוע #{saved_event.id} נוצר"
                    success_message = "האירוע פורסם בהצלחה."
                else:
                    saved_event = await bot.services.events.update_event(
                        bot,
                        event_id,
                        channel_id=values["channel_id"],
                        title=values["title"],
                        description=values["description"],
                        reward=values["reward"],
                        duration_value=values["duration_value"],
                        duration_unit=values["duration_unit"],
                    )
                    success_title = f"האירוע #{saved_event.id} עודכן"
                    success_message = "הודעת האירוע עודכנה בהצלחה."

                return admin_html_response(
                    success_title,
                    _render_success_body(
                        success_title,
                        success_message,
                        record_id=saved_event.id,
                        message_url=_message_link(bot, saved_event.channel_id, saved_event.message_id),
                    ),
                )
            except SalesBotError as exc:
                return admin_html_response(
                    "פאנל אירועים",
                    _render_event_form(
                        mode_label="ערוך" if event_id is not None else "צור",
                        channels=channels,
                        values=values,
                        error_text=str(exc),
                    ),
                )

        return admin_html_response(
            "פאנל אירועים",
            _render_event_form(
                mode_label="ערוך" if event_id is not None else "צור",
                channels=channels,
                values=values,
            ),
        )
    except SalesBotError as exc:
        return _error_response("פאנל אירועים", str(exc), status=400)
    except Exception:
        LOGGER.exception("Unexpected event panel failure")
        return _error_response("פאנל אירועים", "אירעה שגיאה לא צפויה בזמן טעינת פאנל האירועים.", status=500)
