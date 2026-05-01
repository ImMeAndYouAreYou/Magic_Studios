from __future__ import annotations

import asyncio
from datetime import datetime
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import mimetypes
import logging
from io import BytesIO
import time
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

import discord

def _public_robux_calculator_html() -> str:
    return """
        <div class=\"robux-tool\" data-robux-tool>
            <button type=\"button\" class=\"robux-tool-toggle\" data-robux-toggle aria-expanded=\"false\">מחשבון Robux</button>
            <section class=\"robux-tool-panel\" data-robux-panel hidden>
                <div class=\"robux-tool-header\">
                    <div>
                        <h2 class=\"robux-tool-title\">Robux -> USD / ILS</h2>
                        <p>מחשבון מהיר לערך ברוטו ונטו אחרי עמלת Roblox.</p>
                    </div>
                    <button type=\"button\" class=\"ghost-button robux-tool-close\" data-robux-close>סגור</button>
                </div>
                <div class=\"robux-tool-grid\">
                    <label class=\"field field-wide\">
                        <span>כמות Robux</span>
                        <input type=\"number\" min=\"0\" step=\"1\" value=\"1000\" inputmode=\"numeric\" data-robux-input>
                    </label>
                    <label class=\"field\">
                        <span>עמלת Roblox %</span>
                        <input type=\"number\" min=\"0\" max=\"100\" step=\"0.01\" value=\"30\" inputmode=\"decimal\" data-robux-fee>
                    </label>
                    <label class=\"field\">
                        <span>USD לכל 1 Robux</span>
                        <input type=\"number\" min=\"0\" step=\"0.0001\" value=\"0.0035\" inputmode=\"decimal\" data-usd-rate>
                    </label>
                    <label class=\"field field-wide\">
                        <span>ILS לכל 1 USD</span>
                        <input type=\"number\" min=\"0\" step=\"0.01\" value=\"3.65\" inputmode=\"decimal\" data-ils-rate>
                    </label>
                </div>
                <div class=\"robux-result-grid\">
                    <div class=\"robux-result-card\">
                        <strong data-robux-gross-usd>0.00 USD</strong>
                        <span>ערך ברוטו בדולר</span>
                    </div>
                    <div class=\"robux-result-card\">
                        <strong data-robux-gross-ils>0.00 ILS</strong>
                        <span>ערך ברוטו בשקל</span>
                    </div>
                    <div class=\"robux-result-card\">
                        <strong data-robux-net>0 Robux</strong>
                        <span>Robux נטו אחרי עמלה</span>
                    </div>
                    <div class=\"robux-result-card\">
                        <strong data-robux-net-usd>0.00 USD</strong>
                        <span>ערך נטו בדולר</span>
                    </div>
                    <div class=\"robux-result-card\">
                        <strong data-robux-net-ils>0.00 ILS</strong>
                        <span>ערך נטו בשקל</span>
                    </div>
                </div>
                <p class=\"muted robux-tool-footer\">ברירת המחדל מבוססת על שער DevEx של 0.0035 USD לכל Robux, ואפשר לשנות כל שדה לפי החישוב שאתה צריך.</p>
            </section>
        </div>
    """

from aiohttp import web

from sales_bot.exceptions import (
    ConfigurationError,
    ExternalServiceError,
    NotFoundError,
    PermissionDeniedError,
    SalesBotError,
)
from sales_bot.models import (
    BlacklistEntry,
    CartItemRecord,
    CheckoutOrderItemRecord,
    CheckoutOrderRecord,
    DiscountCodeRecord,
    NotificationRecord,
    OrderRequestImageRecord,
    OrderRequestRecord,
    RedeemCodeRecord,
    RobloxGamePassRecord,
    RobloxLinkRecord,
    SpecialOrderRequestRecord,
    SpecialSystemImageRecord,
    SpecialSystemRecord,
    SystemGalleryImageRecord,
    SystemRecord,
    WebsiteSessionRecord,
)
from sales_bot.web_admin import (
    _error_response,
    _escape,
    _list_text_channels,
    _message_link,
    _render_channel_options,
    _website_currency_options,
    admin_html_response,
)

if TYPE_CHECKING:
    from sales_bot.bot import SalesBot


LOGGER = logging.getLogger(__name__)

DISCORD_USER_LABEL_CACHE_TTL_SECONDS = 15 * 60
DISCORD_USER_LABEL_CACHE_MAX_ENTRIES = 4096
_DISCORD_USER_LABEL_CACHE: dict[int, tuple[float, str]] = {}

THEME_COOKIE_NAME = "magic_admin_theme"
THEME_LABELS = {
    "default": "ברירת מחדל",
    "dark": "כהה",
    "light": "בהיר",
}

ADMIN_NAV_SECTIONS = (
    (
        "ראשי",
        (
            {"label": "לוח ניהול", "href": "/admin", "matches": ("/admin",)},
            {"label": "אדמינים", "href": "/admin/admins", "matches": ("/admin/admins",)},
        ),
    ),
    (
        "יצירה",
        (
            {"label": "מערכות", "href": "/admin/systems", "matches": ("/admin/systems",)},
            {"label": "גיימפאסים", "href": "/admin/gamepasses", "matches": ("/admin/gamepasses",)},
            {"label": "מערכות מיוחדות", "href": "/admin/special-systems", "matches": ("/admin/special-systems",)},
        ),
    ),
    (
        "הזמנות",
        (
            {"label": "קופות אתר", "href": "/admin/checkouts", "matches": ("/admin/checkouts",)},
            {"label": "הזמנות אישיות", "href": "/admin/custom-orders", "matches": ("/admin/custom-orders",)},
            {"label": "הזמנות מיוחדות", "href": "/admin/special-orders", "matches": ("/admin/special-orders",)},
        ),
    ),
    (
        "לקוחות",
        (
            {"label": "בלאקליסט", "href": "/admin/blacklist", "matches": ("/admin/blacklist",)},
            {"label": "קודי הנחה", "href": "/admin/discount-codes", "matches": ("/admin/discount-codes",)},
            {"label": "קודי מימוש", "href": "/admin/redeem-codes", "matches": ("/admin/redeem-codes",)},
            {"label": "התראות", "href": "/admin/notifications", "matches": ("/admin/notifications",)},
        ),
    ),
    (
        "אירועים",
        (
            {"label": "הגרלות", "href": "/admin/giveaways/new", "matches": ("/admin/giveaways",)},
            {"label": "סקרים", "href": "/admin/polls/new", "matches": ("/admin/polls",)},
            {"label": "אירועים", "href": "/admin/events/new", "matches": ("/admin/events",)},
        ),
    ),
    (
        "אחר",
        (
            {"label": "הגדרות", "href": "/admin/settings", "matches": ("/admin/settings",)},
            {"label": "התנתק", "href": "/auth/logout", "matches": (), "danger": True},
        ),
    ),
)

PUBLIC_NAV_ITEMS = (
    ("דף הבית", "/"),
    ("מערכות", "/systems"),
    ("הזמנות אישיות", "/custom-orders"),
    ("מערכות מיוחדות", "/special-systems"),
    ("דירוגים", "/vouches"),
    ("מידע", "/info"),
)


PORTAL_STYLE = """
<style>
.portal-root { display: flex; flex-direction: column; gap: 24px; }
.top-strip { display: flex; justify-content: space-between; gap: 14px; align-items: center; flex-wrap: wrap; }
.user-chip { display: inline-flex; align-items: center; gap: 12px; padding: 10px 16px; border-radius: 18px; background: var(--surface-strong); border: 1px solid var(--surface-border); box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04); max-width: 100%; }
.user-chip div { min-width: 0; }
.user-chip img { width: 42px; height: 42px; border-radius: 14px; object-fit: cover; border: 1px solid var(--surface-border-strong); }
.account-link { color: inherit; text-decoration: none; }
.account-link:hover { color: inherit; }
.account-cluster { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
.account-shortcuts { display: inline-flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.shortcut-pill { display: inline-flex; align-items: center; justify-content: center; min-height: 42px; padding: 0 16px; border-radius: 999px; border: 1px solid var(--surface-border); background: var(--surface-card); box-shadow: var(--shadow-md); color: var(--text); text-decoration: none; font-weight: 700; transition: background 0.18s ease, color 0.18s ease, transform 0.18s ease, border-color 0.18s ease; }
.shortcut-pill:hover { color: var(--text); background: var(--surface-soft); border-color: var(--accent-border); transform: translateY(-1px); }
.shortcut-pill.is-active { color: var(--button-text); background: linear-gradient(135deg, var(--accent) 0%, var(--accent-strong) 100%); border-color: transparent; box-shadow: 0 14px 24px rgba(19, 143, 208, 0.2); }
.public-shell-top { display: flex; flex-direction: column; gap: 16px; }
.public-shell-actions { display: flex; align-items: center; justify-content: space-between; gap: 12px; flex-wrap: wrap; }
.public-heading { display: flex; flex-direction: column; gap: 10px; }
.public-site-nav { display: inline-flex; align-items: center; gap: 8px; padding: 8px; border-radius: 18px; border: 1px solid var(--surface-border); background: var(--surface-card); box-shadow: var(--shadow-md); }
.public-site-nav a { display: inline-flex; align-items: center; justify-content: center; min-height: 42px; padding: 0 16px; border-radius: 999px; color: var(--muted); text-decoration: none; font-weight: 700; transition: background 0.18s ease, color 0.18s ease, transform 0.18s ease; }
.public-site-nav a:hover { color: var(--text); background: var(--surface-soft); transform: translateY(-1px); }
.public-site-nav a.is-active { color: var(--button-text); background: linear-gradient(135deg, var(--accent) 0%, var(--accent-strong) 100%); box-shadow: 0 14px 24px rgba(19, 143, 208, 0.2); }
.admin-shell { gap: 22px; }
.admin-topbar { display: flex; justify-content: flex-start; direction: ltr; }
.user-chip-profile { direction: rtl; }
.admin-layout { display: grid; grid-template-columns: 1fr; grid-template-areas: "sidebar" "main"; gap: 20px; align-items: start; direction: rtl; }
.admin-sidebar { grid-area: sidebar; direction: rtl; }
.admin-sidebar-card { display: flex; flex-wrap: wrap; align-items: flex-start; gap: 18px; padding: 18px 20px; border-radius: 22px; background: var(--surface-card); border: 1px solid var(--surface-border); position: sticky; top: 18px; box-shadow: var(--shadow-md); }
.admin-sidebar-card p { margin: 0; }
.sidebar-copy { display: flex; flex-direction: column; gap: 8px; flex: 0 1 280px; min-width: 220px; }
.sidebar-copy .eyebrow { margin-bottom: 0; }
.sidebar-sections { display: flex; flex-wrap: wrap; gap: 12px 16px; flex: 1 1 680px; }
.nav-section { display: flex; flex-direction: column; gap: 8px; min-width: 180px; flex: 1 1 180px; }
.nav-section-title { color: var(--muted); font-size: 0.76rem; letter-spacing: 0.16em; text-transform: uppercase; }
.admin-main { grid-area: main; min-width: 0; display: flex; flex-direction: column; gap: 22px; direction: rtl; }
.admin-hero { position: relative; overflow: hidden; padding: 30px 32px 38px; border-radius: 24px; background: linear-gradient(135deg, var(--surface-hero-start) 0%, var(--surface-hero-end) 100%); border: 1px solid var(--surface-border-strong); box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04); }
.admin-hero::after { content: ""; position: absolute; inset: auto 26px 20px; height: 2px; border-radius: 999px; background: linear-gradient(90deg, transparent 0%, var(--accent) 50%, transparent 100%); opacity: 0.9; }
.admin-hero h1 { margin-bottom: 12px; }
.admin-hero p:last-child { margin-bottom: 0; }
.nav-links { display: flex; flex-wrap: wrap; gap: 8px; }
.nav-links a { padding: 11px 16px; border-radius: 999px; background: var(--surface-soft); border: 1px solid var(--surface-border); text-decoration: none; color: var(--text); font-weight: 700; transition: background 0.18s ease, border-color 0.18s ease, transform 0.18s ease, color 0.18s ease, box-shadow 0.18s ease; }
.nav-links a:hover { background: var(--surface-strong); border-color: var(--accent-border); transform: translateY(-1px); }
.nav-links a.is-active { color: var(--button-text); background: linear-gradient(135deg, var(--accent) 0%, var(--accent-strong) 100%); border-color: transparent; box-shadow: 0 14px 24px rgba(19, 143, 208, 0.2); }
.nav-links a.danger-link { color: var(--danger); background: var(--danger-soft); border-color: var(--danger-border); }
.nav-links a.danger-link:hover { background: rgba(255, 133, 121, 0.18); border-color: rgba(255, 133, 121, 0.34); }
.hero-grid, .stat-grid, .split-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 18px; }
.card { padding: 24px; border-radius: 22px; background: var(--surface-card); border: 1px solid var(--surface-border); box-shadow: var(--shadow-md); }
.card h2, .card h3 { margin-top: 0; margin-bottom: 10px; }
.stat-value { font-size: clamp(2rem, 4vw, 2.75rem); font-weight: 700; color: var(--text); }
.table-wrap { overflow-x: auto; border-radius: 20px; border: 1px solid var(--surface-border); background: var(--surface-strong); }
table { width: 100%; border-collapse: collapse; }
th, td { padding: 15px 17px; text-align: right; border-bottom: 1px solid var(--surface-border); vertical-align: top; }
th { color: var(--text); font-size: 0.95rem; background: rgba(255, 255, 255, 0.02); }
tbody tr:hover { background: rgba(255, 255, 255, 0.02); }
td strong { color: var(--text); }
.inline-form { display: inline-flex; gap: 10px; flex-wrap: wrap; align-items: center; margin: 0; }
.inline-form input, .inline-form select { width: auto; min-width: 120px; }
.stack { display: flex; flex-direction: column; gap: 14px; }
.badge { display: inline-flex; align-items: center; gap: 6px; padding: 6px 10px; border-radius: 999px; background: var(--success-soft); border: 1px solid var(--success-border); color: var(--success-text); font-size: 0.9rem; }
.badge.pending { background: var(--warning-soft); border-color: var(--warning-border); color: var(--warning-text); }
.badge.rejected { background: var(--danger-soft); border-color: var(--danger-border); color: var(--danger); }
.price-list { display: flex; flex-direction: column; gap: 10px; }
.price-item { display: flex; justify-content: space-between; gap: 10px; padding: 14px 16px; border-radius: 18px; background: var(--surface-strong); border: 1px solid var(--surface-border); }
.gallery { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 14px; }
.gallery img { width: 100%; height: 180px; object-fit: cover; border-radius: 18px; border: 1px solid var(--surface-border); background: var(--surface-strong); }
.media-slider { position: relative; overflow: hidden; border-radius: 20px; border: 1px solid var(--surface-border); background: var(--surface-strong); }
.media-slider.is-compact { aspect-ratio: 16 / 10; }
.media-slider.is-feature { aspect-ratio: 16 / 9; }
.slider-track { position: relative; width: 100%; height: 100%; }
.slider-slide { display: none; width: 100%; height: 100%; object-fit: cover; background: var(--surface-strong); }
.slider-slide.is-active { display: block; }
.media-slider.is-compact .slider-slide { aspect-ratio: 16 / 10; }
.media-slider.is-feature .slider-slide { aspect-ratio: 16 / 9; }
.slider-empty { display: flex; align-items: center; justify-content: center; width: 100%; height: 100%; min-height: 220px; color: var(--muted); font-weight: 700; }
.slider-arrow { position: absolute; top: 50%; transform: translateY(-50%); width: 42px; height: 42px; border-radius: 999px; border: 1px solid rgba(255, 255, 255, 0.14); background: rgba(12, 20, 34, 0.72); color: #fff; font-size: 1.35rem; line-height: 1; display: inline-flex; align-items: center; justify-content: center; cursor: pointer; transition: transform 0.18s ease, background 0.18s ease, border-color 0.18s ease; }
.slider-arrow:hover { transform: translateY(-50%) scale(1.04); background: rgba(19, 143, 208, 0.84); border-color: transparent; }
.slider-arrow.prev { right: 14px; }
.slider-arrow.next { left: 14px; }
.slider-count { position: absolute; left: 14px; bottom: 14px; padding: 6px 12px; border-radius: 999px; background: rgba(12, 20, 34, 0.72); color: #fff; font-size: 0.88rem; font-weight: 700; }
.gallery-section { margin-top: 18px; }
.upload-slot-list { display: flex; flex-direction: column; gap: 12px; }
.upload-slot { display: flex; flex-direction: column; gap: 8px; padding: 14px 16px; border-radius: 18px; background: var(--surface-strong); border: 1px dashed var(--surface-border-strong); }
.upload-slot.is-hidden { display: none; }
.upload-slot strong { color: var(--text); font-size: 0.95rem; }
.catalog-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 18px; }
.catalog-card { display: flex; flex-direction: column; gap: 16px; padding: 22px; border-radius: 22px; background: var(--surface-card); border: 1px solid var(--surface-border); box-shadow: var(--shadow-md); }
.catalog-media { width: 100%; aspect-ratio: 16 / 10; border-radius: 18px; border: 1px solid var(--surface-border); background: var(--surface-strong); object-fit: cover; }
.catalog-placeholder { display: flex; align-items: center; justify-content: center; font-weight: 700; color: var(--muted); }
.catalog-meta { display: flex; flex-direction: column; gap: 10px; }
.catalog-badges { display: flex; flex-wrap: wrap; gap: 8px; }
.catalog-badge { display: inline-flex; align-items: center; padding: 7px 11px; border-radius: 999px; background: var(--surface-strong); border: 1px solid var(--surface-border); color: var(--text); font-size: 0.88rem; }
.catalog-badge.warn { color: var(--warning-text); border-color: var(--warning-border); background: var(--warning-soft); }
.hero-banner { display: grid; grid-template-columns: minmax(0, 1.25fr) minmax(240px, 0.75fr); gap: 18px; }
.hero-banner-card { padding: 28px 30px; border-radius: 24px; background: linear-gradient(135deg, var(--surface-hero-start) 0%, var(--surface-hero-end) 100%); border: 1px solid var(--surface-border-strong); }
.hero-side-card { padding: 24px; border-radius: 24px; background: var(--surface-card); border: 1px solid var(--surface-border); box-shadow: var(--shadow-md); }
.profile-summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 14px; }
.summary-tile { padding: 18px 20px; border-radius: 18px; background: var(--surface-strong); border: 1px solid var(--surface-border); }
.summary-tile strong { display: block; font-size: 1.6rem; margin-bottom: 6px; }
.system-detail-grid { display: grid; grid-template-columns: minmax(0, 1.15fr) minmax(300px, 0.85fr); gap: 18px; }
.system-preview { padding: 22px; border-radius: 22px; background: var(--surface-card); border: 1px solid var(--surface-border); box-shadow: var(--shadow-md); }
.system-preview img { width: 100%; max-height: 360px; object-fit: cover; border-radius: 18px; border: 1px solid var(--surface-border); background: var(--surface-strong); }
.system-download-list { display: flex; flex-direction: column; gap: 12px; }
.empty-card { padding: 26px; border-radius: 22px; background: var(--surface-card); border: 1px dashed var(--surface-border-strong); text-align: center; }
.vouch-list { display: flex; flex-direction: column; gap: 14px; }
.vouch-card { padding: 20px 22px; border-radius: 22px; background: var(--surface-card); border: 1px solid var(--surface-border); box-shadow: var(--shadow-md); }
.stars { color: #ffd778; font-size: 1.05rem; letter-spacing: 0.12em; }
.check-card { display: flex; flex-direction: column; gap: 10px; }
.check-line { display: flex; gap: 10px; align-items: center; color: var(--text); }
.check-line input { width: auto; }
.warning-note { color: #ff8579; font-weight: 700; }
.muted { color: var(--muted); }
.mono { font-family: Consolas, "Cascadia Mono", monospace; }
.table-actions { display: flex; flex-wrap: wrap; gap: 8px; }
.table-actions form { margin: 0; }
.profile-grid { display: grid; grid-template-columns: minmax(0, 1.15fr) minmax(280px, 0.85fr); gap: 18px; }
.profile-hero { display: flex; gap: 18px; align-items: center; }
.profile-avatar { width: 74px; height: 74px; border-radius: 22px; object-fit: cover; border: 1px solid var(--surface-border-strong); background: var(--surface-strong); }
.settings-list { display: flex; flex-direction: column; gap: 12px; }
.setting-hint { margin: 0; }
.robux-tool { position: fixed; right: 24px; bottom: 24px; z-index: 30; display: flex; flex-direction: column; align-items: flex-end; gap: 12px; }
.robux-tool-toggle { padding: 12px 18px; border-radius: 999px; box-shadow: 0 18px 38px rgba(8, 16, 28, 0.24); }
.robux-tool-panel { width: min(360px, calc(100vw - 32px)); padding: 18px; border-radius: 24px; background: rgba(20, 31, 45, 0.94); border: 1px solid var(--surface-border-strong); box-shadow: 0 24px 70px rgba(8, 16, 28, 0.3); backdrop-filter: blur(18px); }
html[data-theme="light"] .robux-tool-panel { background: rgba(250, 252, 255, 0.96); }
.robux-tool-panel[hidden] { display: none; }
.robux-tool-header { display: flex; align-items: flex-start; justify-content: space-between; gap: 12px; margin-bottom: 14px; }
.robux-tool-header p { margin: 6px 0 0; font-size: 0.92rem; }
.robux-tool-title { margin: 0; font-size: 1.02rem; }
.robux-tool-close { padding: 10px 14px; box-shadow: none; }
.robux-tool-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
.robux-tool-grid .field { margin: 0; }
.robux-tool-grid .field-wide { grid-column: 1 / -1; }
.robux-result-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; margin-top: 16px; }
.robux-result-card { padding: 12px 14px; border-radius: 18px; background: var(--surface-strong); border: 1px solid var(--surface-border); }
.robux-result-card strong { display: block; margin-bottom: 6px; font-size: 1rem; }
.robux-result-card span { display: block; color: var(--muted); font-size: 0.88rem; }
.robux-tool-footer { margin: 14px 0 0; font-size: 0.84rem; }
@media (max-width: 1100px) {
    .admin-sidebar-card { position: static; }
    .profile-grid { grid-template-columns: 1fr; }
    .sidebar-copy { flex-basis: 100%; max-width: none; }
    .hero-banner, .system-detail-grid { grid-template-columns: 1fr; }
}
@media (max-width: 700px) {
    .top-strip { align-items: stretch; }
    .public-shell-actions { align-items: stretch; }
    .account-cluster { width: 100%; }
    .account-shortcuts { width: 100%; }
    .shortcut-pill { flex: 1 1 0; }
    .public-site-nav { width: 100%; justify-content: space-between; overflow-x: auto; }
    .admin-topbar { justify-content: stretch; }
    .user-chip-profile { width: 100%; }
    .admin-sidebar-card { padding: 16px; }
    .sidebar-sections { flex-direction: column; }
    .nav-section { width: 100%; }
    .nav-links { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .nav-links a { display: flex; align-items: center; justify-content: center; min-height: 46px; text-align: center; }
    .profile-hero { flex-direction: column; align-items: flex-start; }
    .price-item { flex-direction: column; }
    .admin-hero { padding: 24px 22px 30px; }
    .robux-tool { right: 16px; bottom: 16px; left: 16px; align-items: stretch; }
    .robux-tool-toggle { width: 100%; justify-content: center; }
    .robux-tool-panel { width: 100%; }
    .robux-tool-grid, .robux-result-grid { grid-template-columns: 1fr; }
}
</style>
"""

PORTAL_SCRIPT = """
<script>
(() => {
    const syncSlider = (slider, index) => {
        const slides = Array.from(slider.querySelectorAll('[data-slider-slide]'));
        if (!slides.length) {
            return;
        }
        const total = slides.length;
        const normalized = ((Number(index) % total) + total) % total;
        slider.dataset.index = String(normalized);
        slides.forEach((slide, slideIndex) => {
            slide.classList.toggle('is-active', slideIndex === normalized);
        });
        const counter = slider.querySelector('[data-slider-counter]');
        if (counter) {
            counter.textContent = `${normalized + 1}/${total}`;
        }
    };

    document.addEventListener('click', (event) => {
        const button = event.target.closest('[data-slider-step]');
        if (!button) {
            return;
        }
        const slider = button.closest('[data-slider]');
        if (!slider) {
            return;
        }
        const currentIndex = Number(slider.dataset.index || '0');
        const step = Number(button.getAttribute('data-slider-step') || '0');
        syncSlider(slider, currentIndex + step);
    });

    const syncUploadSequence = (container) => {
        const slots = Array.from(container.querySelectorAll('[data-upload-slot]'));
        let highestFilled = -1;
        slots.forEach((slot, index) => {
            const input = slot.querySelector('input[type="file"]');
            if (input && input.files && input.files.length > 0) {
                highestFilled = index;
            }
        });
        const visibleCount = Math.min(slots.length, Math.max(1, highestFilled + 2));
        slots.forEach((slot, index) => {
            slot.classList.toggle('is-hidden', index >= visibleCount);
        });
    };

    document.addEventListener('change', (event) => {
        const target = event.target;
        if (!(target instanceof HTMLInputElement) || target.type !== 'file') {
            return;
        }
        const container = target.closest('[data-upload-sequence]');
        if (!container) {
            return;
        }
        syncUploadSequence(container);
    });

    document.querySelectorAll('[data-slider]').forEach((slider) => {
        syncSlider(slider, Number(slider.dataset.index || '0'));
    });
    document.querySelectorAll('[data-upload-sequence]').forEach((container) => {
        syncUploadSequence(container);
    });

    document.querySelectorAll('[data-robux-tool]').forEach((tool) => {
        const toggleButton = tool.querySelector('[data-robux-toggle]');
        const panel = tool.querySelector('[data-robux-panel]');
        if (!(toggleButton instanceof HTMLButtonElement) || !(panel instanceof HTMLElement)) {
            return;
        }

        const robuxInput = panel.querySelector('[data-robux-input]');
        const feeInput = panel.querySelector('[data-robux-fee]');
        const usdRateInput = panel.querySelector('[data-usd-rate]');
        const ilsRateInput = panel.querySelector('[data-ils-rate]');
        const grossUsdOutput = panel.querySelector('[data-robux-gross-usd]');
        const grossIlsOutput = panel.querySelector('[data-robux-gross-ils]');
        const netRobuxOutput = panel.querySelector('[data-robux-net]');
        const netUsdOutput = panel.querySelector('[data-robux-net-usd]');
        const netIlsOutput = panel.querySelector('[data-robux-net-ils]');
        const storageKey = 'magic-admin-robux-calculator';

        const parseValue = (input, fallback) => {
            if (!(input instanceof HTMLInputElement)) {
                return fallback;
            }
            const parsed = Number.parseFloat(input.value);
            return Number.isFinite(parsed) ? parsed : fallback;
        };

        const formatNumber = (value, digits = 2) => new Intl.NumberFormat('en-US', {
            minimumFractionDigits: digits,
            maximumFractionDigits: digits,
        }).format(Number.isFinite(value) ? value : 0);

        const syncToggleState = (isOpen) => {
            toggleButton.setAttribute('aria-expanded', isOpen ? 'true' : 'false');
            panel.hidden = !isOpen;
        };

        const loadState = () => {
            try {
                const raw = window.localStorage.getItem(storageKey);
                if (!raw) {
                    return;
                }
                const state = JSON.parse(raw);
                if (robuxInput instanceof HTMLInputElement && typeof state.robux === 'string') {
                    robuxInput.value = state.robux;
                }
                if (feeInput instanceof HTMLInputElement && typeof state.fee === 'string') {
                    feeInput.value = state.fee;
                }
                if (usdRateInput instanceof HTMLInputElement && typeof state.usdRate === 'string') {
                    usdRateInput.value = state.usdRate;
                }
                if (ilsRateInput instanceof HTMLInputElement && typeof state.ilsRate === 'string') {
                    ilsRateInput.value = state.ilsRate;
                }
            } catch (_error) {
                return;
            }
        };

        const saveState = () => {
            try {
                window.localStorage.setItem(storageKey, JSON.stringify({
                    robux: robuxInput instanceof HTMLInputElement ? robuxInput.value : '',
                    fee: feeInput instanceof HTMLInputElement ? feeInput.value : '',
                    usdRate: usdRateInput instanceof HTMLInputElement ? usdRateInput.value : '',
                    ilsRate: ilsRateInput instanceof HTMLInputElement ? ilsRateInput.value : '',
                }));
            } catch (_error) {
                return;
            }
        };

        const updateResults = () => {
            const robux = Math.max(parseValue(robuxInput, 0), 0);
            const feePercent = Math.min(Math.max(parseValue(feeInput, 30), 0), 100);
            const usdPerRobux = Math.max(parseValue(usdRateInput, 0.0035), 0);
            const ilsPerUsd = Math.max(parseValue(ilsRateInput, 3.65), 0);
            const grossUsd = robux * usdPerRobux;
            const grossIls = grossUsd * ilsPerUsd;
            const netRobux = robux * (1 - (feePercent / 100));
            const netUsd = netRobux * usdPerRobux;
            const netIls = netUsd * ilsPerUsd;

            if (grossUsdOutput) {
                grossUsdOutput.textContent = `${formatNumber(grossUsd)} USD`;
            }
            if (grossIlsOutput) {
                grossIlsOutput.textContent = `${formatNumber(grossIls)} ILS`;
            }
            if (netRobuxOutput) {
                netRobuxOutput.textContent = `${formatNumber(netRobux, 0)} Robux`;
            }
            if (netUsdOutput) {
                netUsdOutput.textContent = `${formatNumber(netUsd)} USD`;
            }
            if (netIlsOutput) {
                netIlsOutput.textContent = `${formatNumber(netIls)} ILS`;
            }
            saveState();
        };

        loadState();
        updateResults();
        syncToggleState(false);

        toggleButton.addEventListener('click', () => {
            syncToggleState(panel.hidden);
        });

        panel.querySelectorAll('input').forEach((input) => {
            input.addEventListener('input', updateResults);
        });

        const closeButton = panel.querySelector('[data-robux-close]');
        if (closeButton instanceof HTMLButtonElement) {
            closeButton.addEventListener('click', () => {
                syncToggleState(false);
            });
        }

        document.addEventListener('keydown', (event) => {
            if (event.key === 'Escape') {
                syncToggleState(false);
            }
        });
    });
})();
</script>
"""

ORDER_STATUS_LABELS = {
    "pending": "ממתינה",
    "accepted": "התקבלה",
    "rejected": "נדחתה",
    "completed": "הושלמה",
    "cancelled": "בוטלה",
}

PAYMENT_METHOD_LABELS = {
    "card": "כרטיס אשראי",
    "paypal": "PayPal",
}

CHECKOUT_FULFILLMENT_LABELS = {
    "self": "עבורי",
    "gift": "לקנות בשביל חבר",
}

WEBSITE_CARD_CHECKOUT_ENABLED = False

PAYPAL_STATUS_LABELS = {
    "NOT-STARTED": "לא התחיל",
    "CREATED": "הזמנת PayPal נוצרה",
    "APPROVED": "אושר ב-PayPal",
    "COMPLETED": "הושלם",
    "CANCELLED": "בוטל",
    "VOIDED": "בוטל ב-PayPal",
    "DENIED": "נדחה",
    "DECLINED": "נדחה",
    "REFUNDED": "הוחזר",
}

CUSTOM_ORDER_MAX_IMAGES = 5
CUSTOM_ORDER_FORM_MAX_MB = 25
CUSTOM_ORDER_FORM_MAX_BYTES = CUSTOM_ORDER_FORM_MAX_MB * 1024 * 1024


def _page_response(title: str, body: str) -> web.Response:
    return admin_html_response(title, PORTAL_STYLE + body + PORTAL_SCRIPT)


def _custom_order_upload_limit_message() -> str:
    return (
        f"אפשר להעלות עד {CUSTOM_ORDER_MAX_IMAGES} תמונות להזמנה אישית, "
        f"וביחד עד {CUSTOM_ORDER_FORM_MAX_MB}MB. נסה להקטין את התמונות או להעלות פחות קבצים."
    )


def _theme_mode_from_request(request: web.Request) -> str:
    theme_mode = str(request.cookies.get(THEME_COOKIE_NAME, "default") or "default").strip().lower()
    if theme_mode not in THEME_LABELS:
        return "default"
    return theme_mode


def _set_theme_cookie(response: web.StreamResponse, theme_mode: str, *, secure: bool) -> None:
    response.set_cookie(
        THEME_COOKIE_NAME,
        theme_mode,
        max_age=365 * 24 * 60 * 60,
        httponly=False,
        secure=secure,
        samesite="Lax",
        path="/",
    )


def _session_label(session: WebsiteSessionRecord) -> str:
    global_name = (session.global_name or "").strip()
    username = session.username.strip()
    if global_name and username and global_name.casefold() != username.casefold():
        return f"{global_name} (@{username})"
    return global_name or f"@{username}"


def _session_avatar(session: WebsiteSessionRecord) -> str | None:
    if not session.avatar_hash:
        return None
    return f"https://cdn.discordapp.com/avatars/{session.discord_user_id}/{session.avatar_hash}.png?size=256"


def _nav_item_is_active(current_path: str, matches: tuple[str, ...]) -> bool:
    return any(current_path == value or current_path.startswith(f"{value}/") for value in matches)


def _admin_nav_html(current_path: str) -> str:
    sections: list[str] = []
    for section_label, items in ADMIN_NAV_SECTIONS:
        links: list[str] = []
        for item in items:
            classes: list[str] = []
            matches = tuple(item.get("matches", ()))
            if matches and _nav_item_is_active(current_path, matches):
                classes.append("is-active")
            if item.get("danger"):
                classes.append("danger-link")
            class_attr = f' class="{" ".join(classes)}"' if classes else ""
            links.append(f'<a href="{_escape(item["href"])}"{class_attr}>{_escape(item["label"])}</a>')
        sections.append(
            f"""
            <section class="nav-section">
                <span class="nav-section-title">{_escape(section_label)}</span>
                <div class="nav-links">{''.join(links)}</div>
            </section>
            """
        )
    return '<div class="sidebar-sections">' + ''.join(sections) + '</div>'


def _admin_rank_label(bot: "SalesBot", user_id: int) -> str:
    return "בעלים" if user_id == bot.settings.owner_user_id else "אדמין"


def _theme_options(selected_theme: str) -> str:
    return "\n".join(
        f'<option value="{value}"{" selected" if value == selected_theme else ""}>{_escape(label)}</option>'
        for value, label in THEME_LABELS.items()
    )



from __future__ import annotations
def _admin_shell(
    session: WebsiteSessionRecord,
    *,
    current_path: str,
    title: str,
    intro: str,
    content: str,
) -> str:
    avatar_url = _session_avatar(session)
    avatar_html = f'<img src="{_escape(avatar_url)}" alt="avatar">' if avatar_url else ""
    return f"""
    <div class="portal-root admin-shell" dir="rtl">
        <div class="admin-topbar">
            <a class="user-chip user-chip-profile account-link" href="/profile">
                {avatar_html}
                <div>
                    <strong>{_escape(_session_label(session))}</strong><br>
                    <span class="muted mono">{_escape(session.discord_user_id)}</span>
                </div>
            </a>
        </div>
        <div class="admin-layout">
            <aside class="admin-sidebar">
                <div class="admin-sidebar-card">
                    <div class="sidebar-copy">
                        <p class="eyebrow">ניווט מהיר</p>
                        <p>חלוקה לפי אזורים כדי לעבור בין ניהול, יצירה, הזמנות והגדרות בלי שורת כפתורים צפופה.</p>
                    </div>
                    {_admin_nav_html(current_path)}
                </div>
            </aside>
            <div class="admin-main">
                <div class="admin-hero">
                    <p class="eyebrow">אתר ניהול</p>
                    <h1>{_escape(title)}</h1>
                    <p>{_escape(intro)}</p>
                </div>
                {content}
            </div>
        </div>
        {_admin_robux_calculator_html()}
    </div>
    """


def _public_nav_html(current_path: str) -> str:
    links: list[str] = []
    for label, href in PUBLIC_NAV_ITEMS:
        is_active = current_path == href or (href != "/" and current_path.startswith(f"{href}/"))
        class_attr = ' class="is-active"' if is_active else ""
        links.append(f'<a href="{_escape(href)}"{class_attr}>{_escape(label)}</a>')
    return '<nav class="public-site-nav">' + ''.join(links) + '</nav>'


def _public_account_shortcuts(current_path: str) -> str:
    items = (("עגלה", "/cart"), ("התראות", "/inbox"), ("מימוש קוד", "/redeem"))
    links: list[str] = []
    for label, href in items:
        is_active = current_path == href or current_path.startswith(f"{href}/")
        class_attr = ' class="shortcut-pill is-active"' if is_active else ' class="shortcut-pill"'
        links.append(f'<a href="{_escape(href)}"{class_attr}>{_escape(label)}</a>')
    return '<div class="account-shortcuts">' + ''.join(links) + '</div>'


def _public_shell(
    session: WebsiteSessionRecord | None,
    *,
    current_path: str,
    title: str,
    intro: str,
    login_path: str,
    section_label: str = "מערכות מיוחדות",
    content: str,
    show_nav: bool = True,
) -> str:
    account_block = ""
    if session is None:
        account_block = (
            f'<div class="actions"><a class="link-button" href="/auth/discord/login?next={_escape(login_path)}">'
            "התחברות עם דיסקורד"
            "</a></div>"
        )
    else:
        avatar_url = _session_avatar(session)
        avatar_html = f'<img src="{_escape(avatar_url)}" alt="avatar">' if avatar_url else ""
        account_block = f"""
        <div class="account-cluster">
            {_public_account_shortcuts(current_path)}
            <a class="user-chip account-link" href="/profile">
                {avatar_html}
                <div>
                    <strong>{_escape(_session_label(session))}</strong><br>
                    <span class="muted mono">{_escape(session.discord_user_id)}</span>
                </div>
            </a>
        </div>
        """
    robux_calculator = "" if section_label == "דף הבית" else _public_robux_calculator_html()
    return f"""
    <div class="portal-root" dir="rtl">
        <div class="public-shell-top">
            <div class="public-shell-actions">
                {account_block}
                {_public_nav_html(current_path) if show_nav else ''}
            </div>
            <div class="top-strip">
                <div class="public-heading">
                <p class="eyebrow">{_escape(section_label)}</p>
                <h1>{_escape(title)}</h1>
                <p>{_escape(intro)}</p>
                </div>
            </div>
        </div>
        {content}
        {robux_calculator}
    </div>
    """


def _notice_html(message: str | None, *, success: bool) -> str:
    if not message:
        return ""
    classes = "notice success" if success else "notice"
    return f'<div class="{classes}">{_escape(message)}</div>'


def _money_decimal(raw_value: str | None) -> Decimal:
    try:
        return Decimal(str(raw_value or "0")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    except (InvalidOperation, ValueError):
        return Decimal("0.00")


def _money_label(amount: Decimal | str, currency: str) -> str:
    parsed = amount if isinstance(amount, Decimal) else _money_decimal(amount)
    return f"{format(parsed.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP), 'f')} {currency.upper()}"


def _checkout_method_label(method: str) -> str:
    return PAYMENT_METHOD_LABELS.get(method.strip().lower(), method)


def _checkout_fulfillment_label(mode: str) -> str:
    return CHECKOUT_FULFILLMENT_LABELS.get(mode.strip().lower(), mode)


def _checkout_fulfillment_options(selected_mode: str = "self") -> str:
    normalized_mode = selected_mode.strip().lower() if selected_mode else "self"
    return "".join(
        f'<option value="{value}"{" selected" if value == normalized_mode else ""}>{_escape(label)}</option>'
        for value, label in CHECKOUT_FULFILLMENT_LABELS.items()
    )


def _redeem_code_source_label(source: str) -> str:
    normalized = source.strip().lower()
    if normalized == "checkout-gift":
        return "קניה בשביל חבר"
    if normalized == "admin":
        return "נוצר על ידי אדמין"
    return source


def _redeem_code_state_label(code: RedeemCodeRecord) -> str:
    if not code.is_active:
        return '<span class="catalog-badge warn">כבוי</span>'
    if code.expires_at:
        try:
            expires_at = _parse_iso_datetime(code.expires_at)
            if expires_at <= datetime.now(expires_at.tzinfo):
                return '<span class="catalog-badge warn">פג תוקף</span>'
        except ValueError:
            pass
    if code.redeemed_count >= code.max_redemptions:
        return '<span class="catalog-badge warn">נוצל עד הסוף</span>'
    return '<span class="catalog-badge">פעיל</span>'


def _parse_iso_datetime(raw_value: str) -> datetime:
    cleaned = str(raw_value or "").strip()
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    return datetime.fromisoformat(cleaned)


def _paypal_status_label(status: str) -> str:
    normalized = status.strip().upper()
    return PAYPAL_STATUS_LABELS.get(normalized, normalized or "לא התחיל")


def _cart_currency(items: list[CartItemRecord]) -> str:
    if not items:
        return "ILS"
    currencies = {item.system.website_currency.upper() for item in items if item.system.website_currency}
    if len(currencies) > 1:
        raise PermissionDeniedError("כרגע אי אפשר לבצע קופה אחת למערכות עם כמה מטבעות שונים.")
    return next(iter(currencies), "ILS")


def _effective_system_price(system: SystemRecord, personal_discount_percent: int | None = None) -> Decimal:
    base_price = _money_decimal(system.website_price)
    if personal_discount_percent is None or personal_discount_percent <= 0:
        return base_price
    discounted = base_price * (Decimal("100") - Decimal(personal_discount_percent)) / Decimal("100")
    return discounted.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def _build_cart_pricing(
    items: list[CartItemRecord],
    *,
    personal_discounts: dict[int, int] | None = None,
) -> tuple[list[dict[str, Any]], Decimal, str]:
    pricing_rows: list[dict[str, Any]] = []
    subtotal = Decimal("0.00")
    currency = _cart_currency(items)
    discount_map = personal_discounts or {}
    for item in items:
        percent = discount_map.get(item.system.id)
        effective_price = _effective_system_price(item.system, percent)
        pricing_rows.append(
            {
                "item": item,
                "personal_discount_percent": percent,
                "base_price": _money_decimal(item.system.website_price),
                "effective_price": effective_price,
            }
        )
        subtotal += effective_price
    return pricing_rows, subtotal.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP), currency


def _checkout_items_html(items: list[CheckoutOrderItemRecord], currency: str) -> str:
    if not items:
        return '<div class="empty-card"><p>אין מערכות שמחוברות להזמנה הזאת.</p></div>'
    return ''.join(
        f'<div class="price-item"><strong>{_escape(item.system_name)}</strong><span>{_escape(_money_label(item.line_total, currency))}</span></div>'
        for item in items
    )


async def _send_optional_user_dm(
    bot: "SalesBot",
    *,
    user_id: int,
    title: str,
    body: str,
    link_path: str | None = None,
    message_override: str | None = None,
) -> bool:
    try:
        user = bot.get_user(user_id) or await bot.fetch_user(user_id)
        dm_channel = user.dm_channel or await user.create_dm()
        extra_line = ""
        if link_path:
            extra_line = f"\n{bot.settings.public_base_url}{link_path}"
        if message_override is not None:
            await dm_channel.send(f"{message_override}{extra_line}")
        else:
            await dm_channel.send(f"**{title}**\n{body}{extra_line}")
        return True
    except (discord.HTTPException, discord.Forbidden):
        return False


def _status_badge(status: str) -> str:
    normalized = status.strip().lower()
    extra_class = " pending" if normalized == "pending" else " rejected" if normalized == "rejected" else ""
    return f'<span class="badge{extra_class}">{_escape(ORDER_STATUS_LABELS.get(normalized, normalized))}</span>'


def _redirect_to_login(request: web.Request) -> None:
    next_path = quote(request.path_qs or request.path, safe="/?=&%")
    raise web.HTTPFound(f"/auth/discord/login?next={next_path}")


async def _current_site_session(request: web.Request) -> tuple["SalesBot", WebsiteSessionRecord | None]:
    bot: SalesBot = request.app["bot"]
    token = request.cookies.get(bot.services.web_auth.cookie_name, "").strip()
    if not token:
        return bot, None
    try:
        session = await bot.services.web_auth.get_session(token)
    except SalesBotError:
        return bot, None
    except Exception:
        LOGGER.warning("Ignoring invalid website session cookie during request to %s", request.path, exc_info=True)
        return bot, None
    return bot, session


async def _require_site_session(request: web.Request) -> tuple["SalesBot", WebsiteSessionRecord]:
    bot, session = await _current_site_session(request)
    if session is None:
        _redirect_to_login(request)
    assert session is not None
    return bot, session


async def _require_admin_session(request: web.Request) -> tuple["SalesBot", WebsiteSessionRecord]:
    bot, session = await _require_site_session(request)
    if not await bot.services.admins.is_admin(session.discord_user_id):
        raise PermissionDeniedError("רק אדמינים של הבוט יכולים לפתוח את האתר הזה.")
    return bot, session


async def _blacklist_entry_optional(bot: "SalesBot", user_id: int) -> BlacklistEntry | None:
    try:
        return await bot.services.blacklist.get_entry(user_id)
    except NotFoundError:
        return None


async def _ensure_site_session_allowed(bot: "SalesBot", session: WebsiteSessionRecord, *, allow_blacklisted: bool = False) -> BlacklistEntry | None:
    entry = await _blacklist_entry_optional(bot, session.discord_user_id)
    if entry is not None and not allow_blacklisted:
        raise web.HTTPFound("/blacklist-appeal")
    return entry


async def _require_active_site_session(request: web.Request) -> tuple["SalesBot", WebsiteSessionRecord]:
    bot, session = await _require_site_session(request)
    await _ensure_site_session_allowed(bot, session)
    return bot, session


def _parse_positive_int(raw_value: Any, field_label: str, *, allow_blank: bool = False) -> int | None:
    value = str(raw_value or "").strip()
    if not value and allow_blank:
        return None
    if not value:
        raise PermissionDeniedError(f"חסר ערך עבור {field_label}.")
    try:
        parsed = int(value)
    except ValueError as exc:
        raise PermissionDeniedError(f"{field_label} חייב להיות מספר תקין.") from exc
    if parsed <= 0:
        raise PermissionDeniedError(f"{field_label} חייב להיות גדול מ-0.")
    return parsed


def _parse_optional_bool(raw_value: Any) -> bool | None:
    value = str(raw_value or "").strip().lower()
    if not value:
        return None
    if value == "true":
        return True
    if value == "false":
        return False
    raise PermissionDeniedError("הערך הבוליאני שנשלח לא תקין.")


def _extract_file_upload(field: Any, *, image_only: bool = False) -> tuple[str, bytes, str | None] | None:
    if not isinstance(field, web.FileField) or not field.filename:
        return None
    if image_only and field.content_type and not field.content_type.startswith("image/"):
        raise PermissionDeniedError("הקובץ שנשלח חייב להיות תמונה.")
    payload = field.file.read()
    if not payload:
        return None
    return field.filename, payload, field.content_type


async def _discord_user_label(bot: "SalesBot", user_id: int) -> str:
    now = time.monotonic()
    cached = _DISCORD_USER_LABEL_CACHE.get(user_id)
    if cached is not None and cached[0] > now:
        return cached[1]

    user = bot.get_user(user_id)
    if user is None:
        try:
            user = await bot.fetch_user(user_id)
        except discord.HTTPException:
            label = str(user_id)
            _remember_discord_user_label(user_id, label, now=now)
            return label
    username = str(getattr(user, "name", "") or "").strip()
    global_name = str(getattr(user, "global_name", "") or "").strip()
    if global_name and username and global_name.casefold() != username.casefold():
        label = f"{global_name} (@{username})"
        _remember_discord_user_label(user_id, label, now=now)
        return label

    label = global_name or (f"@{username}" if username else str(user_id))
    _remember_discord_user_label(user_id, label, now=now)
    return label


def _remember_discord_user_label(user_id: int, label: str, *, now: float | None = None) -> None:
    current_time = now if now is not None else time.monotonic()
    _DISCORD_USER_LABEL_CACHE[user_id] = (current_time + DISCORD_USER_LABEL_CACHE_TTL_SECONDS, label)
    if len(_DISCORD_USER_LABEL_CACHE) <= DISCORD_USER_LABEL_CACHE_MAX_ENTRIES:
        return

    expired_user_ids = [cached_user_id for cached_user_id, (expires_at, _label) in _DISCORD_USER_LABEL_CACHE.items() if expires_at <= current_time]
    for cached_user_id in expired_user_ids:
        _DISCORD_USER_LABEL_CACHE.pop(cached_user_id, None)

    if len(_DISCORD_USER_LABEL_CACHE) <= DISCORD_USER_LABEL_CACHE_MAX_ENTRIES:
        return

    oldest_user_ids = sorted(
        _DISCORD_USER_LABEL_CACHE,
        key=lambda cached_user_id: _DISCORD_USER_LABEL_CACHE[cached_user_id][0],
    )[: len(_DISCORD_USER_LABEL_CACHE) - DISCORD_USER_LABEL_CACHE_MAX_ENTRIES]
    for cached_user_id in oldest_user_ids:
        _DISCORD_USER_LABEL_CACHE.pop(cached_user_id, None)


def _system_options(systems: list[SystemRecord], selected_system_id: int | None = None) -> str:
    options = ['<option value="">ללא</option>']
    for system in systems:
        selected = " selected" if selected_system_id == system.id else ""
        options.append(f'<option value="{system.id}"{selected}>{_escape(system.name)}</option>')
    return "\n".join(options)


def _gamepass_options(gamepasses: list[RobloxGamePassRecord], selected_gamepass_id: int | None = None) -> str:
    options = ['<option value="">בחר גיימפאס</option>']
    for gamepass in gamepasses:
        price = _gamepass_price_label(gamepass)
        selected = " selected" if selected_gamepass_id == gamepass.game_pass_id else ""
        label = f"{gamepass.name} ({gamepass.game_pass_id} | {price})"
        options.append(f'<option value="{gamepass.game_pass_id}"{selected}>{_escape(label)}</option>')
    return "\n".join(options)


def _bool_options(selected_value: str = "") -> str:
    options = {"": "ללא שינוי", "true": "כן", "false": "לא"}
    return "\n".join(
        f'<option value="{value}"{" selected" if value == selected_value else ""}>{label}</option>'
        for value, label in options.items()
    )


def _payment_method_editor(service: Any, selected_keys: set[str], prices: dict[str, str]) -> str:
    cards: list[str] = []
    for key, label in service.available_payment_methods():
        checked = " checked" if key in selected_keys else ""
        cards.append(
            f"""
            <label class="meta-card check-card">
                <span class="check-line">
                    <input type="checkbox" name="payment_method" value="{_escape(key)}"{checked}>
                    <strong>{_escape(label)}</strong>
                </span>
                <input type="text" name="price_{_escape(key)}" placeholder="מחיר ב{_escape(label)}" value="{_escape(prices.get(key, ''))}">
            </label>
            """
        )
    return "\n".join(cards)


def _payment_method_select_options(special_system: SpecialSystemRecord, selected_key: str | None = None) -> str:
    options = ['<option value="">בחר שיטת תשלום</option>']
    for method in special_system.payment_methods:
        selected = " selected" if method.key == (selected_key or "") else ""
        label = f"{method.label} | {method.price}"
        options.append(f'<option value="{_escape(method.key)}"{selected}>{_escape(label)}</option>')
    return "\n".join(options)


def _order_payment_method_select_options(order_service: Any, selected_key: str | None = None) -> str:
    normalized = (selected_key or "").strip()
    options = ['<option value="">בחר שיטת תשלום</option>']
    for key, label in order_service.available_payment_methods():
        selected = " selected" if normalized in {key, label} else ""
        options.append(f'<option value="{_escape(key)}"{selected}>{_escape(label)}</option>')
    return "\n".join(options)


def _yes_no_select_options(selected_value: str | None = None) -> str:
    normalized = (selected_value or "").strip().lower()
    options = ['<option value="">בחר</option>']
    for value, label in (("yes", "כן"), ("no", "לא")):
        selected = " selected" if normalized == value else ""
        options.append(f'<option value="{value}"{selected}>{label}</option>')
    return "\n".join(options)


def _special_system_url(bot: "SalesBot", special_system: SpecialSystemRecord) -> str:
    return f"{bot.settings.public_base_url}/special-systems/{special_system.slug}"


def _system_image_url(system: SystemRecord) -> str | None:
    if not system.image_path:
        return None
    return f"/system-images/{system.id}"


def _system_gallery_urls(system: SystemRecord, images: list[SystemGalleryImageRecord]) -> list[str]:
    urls = [f"/system-gallery-images/{image.id}" for image in images]
    if not urls and system.image_path:
        urls.append(f"/system-images/{system.id}")
    return urls


def _special_gallery_urls(images: list[SpecialSystemImageRecord]) -> list[str]:
    return [f"/special-system-images/{image.id}" for image in images]


def _custom_order_gallery_urls(images: list[OrderRequestImageRecord]) -> list[str]:
    return [f"/admin/custom-order-images/{image.id}" for image in images]


def _render_image_slider(
    image_urls: list[str],
    *,
    alt_text: str,
    compact: bool = False,
    empty_label: str = "אין תמונות תצוגה",
) -> str:
    slider_class = "is-compact" if compact else "is-feature"
    if not image_urls:
        return f'<div class="media-slider {slider_class}"><div class="slider-empty">{_escape(empty_label)}</div></div>'

    slides = ''.join(
        f'<img class="slider-slide{" is-active" if index == 0 else ""}" data-slider-slide src="{_escape(image_url)}" alt="{_escape(f"{alt_text} {index + 1}")}">'
        for index, image_url in enumerate(image_urls)
    )
    controls = ''
    if len(image_urls) > 1:
        controls = (
            '<button type="button" class="slider-arrow prev" data-slider-step="-1" aria-label="תמונה קודמת">&#8249;</button>'
            '<button type="button" class="slider-arrow next" data-slider-step="1" aria-label="תמונה הבאה">&#8250;</button>'
            f'<div class="slider-count" data-slider-counter>1/{len(image_urls)}</div>'
        )
    return f'<div class="media-slider {slider_class}" data-slider data-index="0"><div class="slider-track">{slides}</div>{controls}</div>'


def _catalog_badges_for_system(system: SystemRecord) -> str:
    badges: list[str] = []
    if system.website_price:
        badges.append(f'<span class="catalog-badge">{_escape(_money_label(system.website_price, system.website_currency))}</span>')
    if system.paypal_link:
        badges.append('<span class="catalog-badge">פייפאל</span>')
    if system.roblox_gamepass_id:
        badges.append('<span class="catalog-badge">רובקס</span>')
    if not badges:
        badges.append('<span class="catalog-badge warn">אין שיטת רכישה זמינה כרגע</span>')
    return ''.join(badges)


def _render_system_card(
    system: SystemRecord,
    *,
    image_urls: list[str] | None = None,
    owned: bool = False,
    discount_percent: int | None = None,
) -> str:
    image_html = _render_image_slider(image_urls or [], alt_text=system.name, compact=True, empty_label="אין תמונת תצוגה")
    extra_badges: list[str] = []
    if owned:
        extra_badges.append('<span class="catalog-badge">כבר בבעלותך</span>')
    if discount_percent is not None:
        extra_badges.append(f'<span class="catalog-badge">הנחה אישית {discount_percent}%</span>')
    add_to_cart_html = ""
    if not owned and system.website_price:
        add_to_cart_html = f'''<form method="post" action="/cart" class="inline-form"><input type="hidden" name="action" value="add"><input type="hidden" name="system_id" value="{system.id}"><button type="submit" class="ghost-button">הוסף לעגלה</button></form>'''
    return f"""
    <article class="catalog-card">
        {image_html}
        <div class="catalog-meta">
            <div>
                <h2>{_escape(system.name)}</h2>
                <p>{_escape(system.description)}</p>
            </div>
            <div class="catalog-badges">{_catalog_badges_for_system(system)}{''.join(extra_badges)}</div>
            <div class="actions"><a class="link-button" href="/systems/{system.id}">פתח עמוד מערכת</a>{add_to_cart_html}</div>
        </div>
    </article>
    """


def _render_special_system_card(special_system: SpecialSystemRecord, *, image_urls: list[str] | None = None) -> str:
    payment_summary = ', '.join(method.label for method in special_system.payment_methods) or 'לפי תיאום'
    return f"""
    <article class="catalog-card">
        {_render_image_slider(image_urls or [], alt_text=special_system.title, compact=True, empty_label="מערכת מיוחדת")}
        <div class="catalog-meta">
            <div>
                <h2>{_escape(special_system.title)}</h2>
                <p>{_escape(special_system.description)}</p>
            </div>
            <div class="catalog-badges"><span class="catalog-badge">{_escape(payment_summary)}</span></div>
            <div class="actions"><a class="link-button" href="/special-systems/{_escape(special_system.slug)}">פתח עמוד הזמנה</a></div>
        </div>
    </article>
    """


async def website_home_page(request: web.Request) -> web.Response:
    bot, session = await _require_active_site_session(request)
    systems = await bot.services.systems.list_public_systems()
    special_systems = await bot.services.special_systems.list_special_systems(active_only=True)
    content = f"""
    <div class="hero-banner">
        <div class="hero-banner-card">
            <p class="eyebrow">Magic Studio's</p>
            <h2>ברוכים הבאים לאתר המכירות</h2>
            <p>כאן אפשר לראות את המערכות, לשלוח הזמנה אישית, לעבור למסלולי רכישה, לעיין בדירוגים ולהיכנס לעמוד הפרופיל האישי שלך.</p>
            <div class="actions">
                <a class="link-button" href="/systems">מעבר למערכות</a>
                <a class="link-button ghost-button" href="/custom-orders">הזמנה אישית</a>
                <a class="link-button ghost-button" href="/special-systems">מערכות מיוחדות</a>
            </div>
        </div>
        <div class="hero-side-card">
            <p class="eyebrow">גישה מהירה</p>
            <p>כל רכישה באתר קשורה לחשבון דיסקורד המחובר, כדי שהמערכות, הדירוגים וההורדות יישמרו לך במקום אחד.</p>
        </div>
    </div>
    <div class="profile-summary-grid">
        <div class="summary-tile"><strong>{len(systems)}</strong><span>כמות מערכות שלנו</span></div>
        <div class="summary-tile"><strong>{len(special_systems)}</strong><span>מערכות מיוחדות פעילות</span></div>
        <div class="summary-tile"><strong>{'מחובר' if session else 'לא מחובר'}</strong><span>מצב החשבון באתר</span></div>
    </div>
    <div class="catalog-grid">
        <article class="catalog-card">
            <div class="catalog-meta">
                <div>
                    <h2>מערכות</h2>
                    <p>רשימת כל המערכות הרגילות, עם עמוד מפורט לכל מערכת ושיטות רכישה זמינות.</p>
                </div>
                <div class="actions"><a class="link-button" href="/systems">לכל המערכות</a></div>
            </div>
        </article>
        <article class="catalog-card">
            <div class="catalog-meta">
                <div>
                    <h2>מערכות מיוחדות</h2>
                    <p>הצעות מיוחדות עם טופס הזמנה ישיר ופרטי תשלום מותאמים.</p>
                </div>
                <div class="actions"><a class="link-button" href="/special-systems">למערכות המיוחדות</a></div>
            </div>
        </article>
        <article class="catalog-card">
            <div class="catalog-meta">
                <div>
                    <h2>הזמנות אישיות</h2>
                    <p>עמוד מסודר לשליחת בקשות מותאמות אישית עם פרטים, תקציב ותמונות לעיון האדמינים.</p>
                </div>
                <div class="actions"><a class="link-button" href="/custom-orders">לשליחת הזמנה אישית</a></div>
            </div>
        </article>
        <article class="catalog-card">
            <div class="catalog-meta">
                <div>
                    <h2>דירוגים</h2>
                    <p>כל הדירוגים שכבר נשמרו במערכת מוצגים גם באתר בעמוד אחד.</p>
                </div>
                <div class="actions"><a class="link-button" href="/vouches">לכל הדירוגים</a></div>
            </div>
        </article>
    </div>
    """
    body = _public_shell(
        session,
        current_path=request.path,
        title="אתר המכירות של Magic Studio's",
        intro="מרכז אחד למערכות, הזמנות אישיות, מערכות מיוחדות, דירוגים והעמוד האישי שלך.",
        login_path=request.path,
        section_label="דף הבית",
        content=content,
    )
    return _page_response("דף הבית", body)


async def website_info_page(request: web.Request) -> web.Response:
    _, session = await _require_active_site_session(request)
    invite_url = "https://discord.gg/xAf4YM9V3j"
    content = f"""
    <div class="hero-banner">
        <div class="hero-banner-card">
            <p class="eyebrow">מידע</p>
            <h2>ברוכים הבאים לאתר שלנו</h2>
            <p>באתר זה תוכלו לקנות את המערכות שמגיק מכין ומוכר בעזרת כסף אמיתי. לקניה ברובקס כנסו לשרת הדיסקורד וקנו מהמשחק מכירות.</p>
            <p>לעזרה ותמיכה כנסו לשרת דיסקורד.</p>
            <div class="actions"><a class="link-button" href="{invite_url}" target="_blank" rel="noreferrer">קישור לשרת הדיסקורד</a></div>
        </div>
        <div class="hero-side-card">
            <p class="eyebrow">מסמכים</p>
            <p>עמוד זה מרכז גם את מדיניות הפרטיות וגם את תנאי השימוש של האתר והשירות.</p>
        </div>
    </div>
    <div class="doc-grid">
        <article class="doc-card copy-stack">
            <h2 id="privacy">מדיניות פרטיות</h2>
            <ul class="doc-list">
                <li>ייתכן שיישמרו מזהי משתמש של דיסקורד, רשומות בעלות, בלאקליסט ודירוגים לצורך תפעול השירות.</li>
                <li>אם חשבון רובלוקס מחובר, ייתכן שיישמרו גם פרטי הזיהוי הציבוריים הדרושים לחיבור.</li>
                <li>קבצי מערכות נשמרים לצורך מסירה מאובטחת לרוכשים מורשים בלבד.</li>
            </ul>
            <div class="actions"><a class="link-button ghost-button" href="/privacy">לעמוד המדיניות המלא</a></div>
        </article>
        <article class="doc-card copy-stack">
            <h2 id="terms">תנאי שימוש</h2>
            <ul class="doc-list">
                <li>השימוש באתר ובבוט מיועד ללקוחות ולחברי השרת המורשים בלבד.</li>
                <li>ניסיון לנצל לרעה רכישות, חשבונות או תהליכי גישה עלול לגרום להסרת גישה.</li>
                <li>מסירת מערכות ורכישות כפופות לכללי השרת ולשיקול דעת הצוות.</li>
            </ul>
            <div class="actions"><a class="link-button ghost-button" href="/terms">לעמוד התנאים המלא</a></div>
        </article>
    </div>
    """
    body = _public_shell(
        session,
        current_path=request.path,
        title="מידע על האתר",
        intro="כאן נמצאים פרטי השירות, קישור התמיכה ומסמכי המדיניות של האתר.",
        login_path=request.path,
        section_label="מידע",
        content=content,
    )
    return _page_response("מידע", body)


async def public_systems_page(request: web.Request) -> web.Response:
    bot, session = await _require_active_site_session(request)
    systems = await bot.services.systems.list_public_systems()
    system_images_by_id = await bot.services.systems.list_system_images_for_systems(systems) if systems else {}
    owned_ids = {system.id for system in await bot.services.ownership.list_user_systems(session.discord_user_id)}
    discounts = {
        discount.system.id: discount.discount_percent
        for discount in await bot.services.discounts.list_user_discounts(session.discord_user_id)
    }
    content = (
        '<div class="catalog-grid">' + ''.join(
            _render_system_card(
                system,
                image_urls=_system_gallery_urls(system, system_images_by_id.get(system.id, [])),
                owned=system.id in owned_ids,
                discount_percent=discounts.get(system.id),
            )
            for system in systems
        ) + '</div>'
        if systems
        else '<div class="empty-card"><h2>אין מערכות זמינות כרגע</h2><p>נסו שוב מאוחר יותר.</p></div>'
    )
    body = _public_shell(
        session,
        current_path=request.path,
        title="מערכות",
        intro="כל המערכות שמחוברות לבוט, עם עמוד מפורט לכל מערכת ואפשרות רכישה או הורדה לבעלים קיימים.",
        login_path=request.path,
        section_label="קטלוג מערכות",
        content=content,
    )
    return _page_response("מערכות", body)


async def public_system_detail_page(request: web.Request) -> web.Response:
    bot, session = await _require_active_site_session(request)
    system = await bot.services.systems.get_system(int(request.match_info["system_id"]))
    owned = await bot.services.ownership.user_owns_system(session.discord_user_id, system.id)
    if (system.is_special_system or not system.is_visible_on_website or not system.is_for_sale or not system.is_in_stock) and not owned:
        raise NotFoundError("המערכת שביקשת לא זמינה כרגע לרכישה באתר.")
    discount = await bot.services.discounts.get_discount_optional(session.discord_user_id, system.id)
    image_urls = _system_gallery_urls(system, await bot.services.systems.list_system_images(system.id))
    image_html = _render_image_slider(image_urls, alt_text=system.name, empty_label="אין תמונת תצוגה")
    robux_url = bot.services.systems.gamepass_url_for_id(system.roblox_gamepass_id)
    actions: list[str] = []
    if owned:
        actions.append(f'<a class="link-button" href="/downloads/{system.id}">הורדה מהירה</a>')
    if not owned and system.website_price:
        actions.append(
            f'<form method="post" action="/cart" class="inline-form"><input type="hidden" name="action" value="add"><input type="hidden" name="system_id" value="{system.id}"><button type="submit" class="link-button">הוסף לעגלה</button></form>'
        )
    if system.paypal_link:
        actions.append(f'<a class="link-button ghost-button" href="/systems/{system.id}/buy/paypal">רכישה דרך פייפאל</a>')
    if robux_url:
        actions.append(f'<a class="link-button ghost-button" href="{_escape(robux_url)}" target="_blank" rel="noreferrer">רכישה ברובקס</a>')
    actions_html = ''.join(actions) if actions else '<span class="catalog-badge warn">אין שיטת רכישה זמינה כרגע</span>'
    discount_html = f'<div class="meta-card"><strong>הנחה אישית:</strong> {discount.discount_percent}% על המערכת הזאת.</div>' if discount is not None else ''
    ownership_html = '<div class="meta-card"><strong>מצב בעלות:</strong> המערכת כבר בבעלותך וניתנת להורדה ישירה.</div>' if owned else '<div class="meta-card"><strong>מצב בעלות:</strong> המערכת עדיין לא רשומה בבעלותך.</div>'
    price_html = (
        f'<div class="meta-card"><strong>מחיר באתר:</strong> {_escape(_money_label(system.website_price, system.website_currency))}</div>'
        if system.website_price
        else '<div class="meta-card"><strong>מחיר באתר:</strong> עדיין לא הוגדר מחיר עגלה למערכת הזאת.</div>'
    )
    content = f"""
    <div class="system-detail-grid">
        <div class="system-preview">{image_html}</div>
        <div class="card stack">
            <div>
                <h2>{_escape(system.name)}</h2>
                <p>{_escape(system.description)}</p>
            </div>
            <div class="catalog-badges">{_catalog_badges_for_system(system)}</div>
            {price_html}
            {discount_html}
            {ownership_html}
            <div class="actions">{actions_html}</div>
        </div>
    </div>
    """
    body = _public_shell(
        session,
        current_path="/systems",
        title=system.name,
        intro="עמוד המערכת כולל שיטות רכישה, סטטוס בעלות והורדה ישירה לחשבון המחובר.",
        login_path=request.path,
        section_label="מערכת",
        content=content,
    )
    return _page_response(system.name, body)


async def special_systems_page(request: web.Request) -> web.Response:
    bot, session = await _require_active_site_session(request)
    systems = await bot.services.special_systems.list_special_systems(active_only=True)
    system_images = await asyncio.gather(*(bot.services.special_systems.list_special_system_images(system.id) for system in systems)) if systems else []
    content = (
        '<div class="catalog-grid">' + ''.join(
            _render_special_system_card(system, image_urls=_special_gallery_urls(images))
            for system, images in zip(systems, system_images, strict=False)
        ) + '</div>'
        if systems
        else '<div class="empty-card"><h2>אין מערכות מיוחדות פעילות כרגע</h2><p>נסו שוב מאוחר יותר.</p></div>'
    )
    body = _public_shell(
        session,
        current_path=request.path,
        title="מערכות מיוחדות",
        intro="כאן תמצאו הצעות מיוחדות עם טופס הזמנה ישיר מתוך האתר.",
        login_path=request.path,
        section_label="קטלוג מערכות מיוחדות",
        content=content,
    )
    return _page_response("מערכות מיוחדות", body)


async def website_paypal_purchase_page(request: web.Request) -> web.Response:
    bot, session = await _require_active_site_session(request)
    system = await bot.services.systems.get_system(int(request.match_info["system_id"]))
    if system.is_special_system:
        raise NotFoundError("המערכת הזאת זמינה רק דרך עמוד המערכות המיוחדות.")
    if not system.paypal_link:
        raise PermissionDeniedError("למערכת הזאת אין כרגע קישור פייפאל פעיל.")
    purchase = await bot.services.payments.create_purchase(session.discord_user_id, system.id, system.paypal_link)
    content = f"""
    <div class="card stack">
        <h2>רכישה דרך פייפאל</h2>
        <p>נוצרה עבורך רשומת רכישה מספר <strong>#{purchase.id}</strong>. לחץ על הכפתור למטה כדי לעבור לקישור הפייפאל המחובר למערכת.</p>
        <div class="meta-card"><strong>מערכת:</strong> {_escape(system.name)}</div>
        <div class="actions">
            <a class="link-button" href="{_escape(system.paypal_link)}" target="_blank" rel="noreferrer">פתח את פייפאל</a>
            <a class="link-button ghost-button" href="/systems/{system.id}">חזרה לעמוד המערכת</a>
        </div>
    </div>
    """
    body = _public_shell(
        session,
        current_path="/systems",
        title="מעבר לתשלום",
        intro="האתר שמר עבורך רשומת רכישה לפני המעבר לקישור התשלום.",
        login_path=request.path,
        section_label="פייפאל",
        content=content,
    )
    return _page_response("פייפאל", body)


async def website_paypal_return_page(request: web.Request) -> web.Response:
    bot, session = await _require_active_site_session(request)
    paypal_order_token = str(request.query.get("token", "")).strip()
    order_id = _parse_positive_int(request.query.get("order_id"), "מזהה הזמנה", allow_blank=True)
    if order_id is None and not paypal_order_token:
        raise PermissionDeniedError("חסרים פרטים לחזרה מ-PayPal.")

    order = (
        await bot.services.payments.get_checkout_order(order_id)
        if order_id is not None
        else await bot.services.payments.get_checkout_order_by_paypal_order_id(paypal_order_token)
    )
    if order.user_id != session.discord_user_id:
        raise PermissionDeniedError("אי אפשר לצפות בהזמנה שלא שייכת לחשבון המחובר.")

    order = await bot.services.payments.capture_paypal_checkout(
        bot,
        order.id,
        paypal_order_id=paypal_order_token or None,
    )
    delivery_message = (
        "אם הכול עבר תקין, קודי המימוש כבר נוצרו עבורך ואפשר לראות אותם גם בעמוד המימושים."
        if order.fulfillment_mode == "gift"
        else "אם הכול עבר תקין, המערכות כבר נשלחו אוטומטית והעדכון נשמר גם במרכז ההתראות."
    )
    content = f"""
    <div class="card stack">
        <h2>התשלום הושלם</h2>
        <p>PayPal אישר את התשלום עבור הזמנה <strong>#{order.id}</strong>. {delivery_message}</p>
        <div class="price-list">
            <div class="price-item"><strong>סטטוס הזמנה</strong><span>{_status_badge(order.status)}</span></div>
            <div class="price-item"><strong>סטטוס PayPal</strong><span>{_escape(_paypal_status_label(order.paypal_status))}</span></div>
            <div class="price-item"><strong>מסלול מסירה</strong><span>{_escape(_checkout_fulfillment_label(order.fulfillment_mode))}</span></div>
            <div class="price-item"><strong>סה"כ</strong><span>{_escape(_money_label(order.total_amount, order.currency))}</span></div>
        </div>
        <div class="actions"><a class="link-button" href="{'/redeem' if order.fulfillment_mode == 'gift' else '/inbox'}">{'לעמוד המימושים' if order.fulfillment_mode == 'gift' else 'למרכז ההתראות'}</a><a class="link-button ghost-button" href="/systems">חזרה לחנות</a></div>
    </div>
    """
    body = _public_shell(
        session,
        current_path="/checkout",
        title="תשלום PayPal הושלם",
        intro="התשלום חזר מהשער המאובטח של PayPal ונבדק מול ההזמנה שלך.",
        login_path=request.path_qs or request.path,
        section_label="PayPal",
        content=content,
    )
    return _page_response("תשלום PayPal הושלם", body)


async def website_paypal_cancel_page(request: web.Request) -> web.Response:
    bot, session = await _require_active_site_session(request)
    order_id = _parse_positive_int(request.query.get("order_id"), "מזהה הזמנה", allow_blank=True)
    paypal_order_token = str(request.query.get("token", "")).strip()
    if order_id is None and not paypal_order_token:
        raise PermissionDeniedError("חסרים פרטים לביטול תשלום PayPal.")

    order = (
        await bot.services.payments.get_checkout_order(order_id)
        if order_id is not None
        else await bot.services.payments.get_checkout_order_by_paypal_order_id(paypal_order_token)
    )
    if order.user_id != session.discord_user_id:
        raise PermissionDeniedError("אי אפשר לנהל הזמנה שלא שייכת לחשבון המחובר.")

    cancel_reason = "הלקוח ביטל את תהליך התשלום ב-PayPal לפני אישור סופי."
    order = await bot.services.payments.mark_paypal_checkout_cancelled(order.id, cancel_reason)
    await bot.services.notifications.create_notification(
        user_id=order.user_id,
        title=f"הזמנה #{order.id} בוטלה",
        body=cancel_reason,
        link_path="/inbox",
        kind="checkout",
    )
    content = f"""
    <div class="card stack">
        <h2>התשלום בוטל</h2>
        <p>PayPal החזיר את ההזמנה <strong>#{order.id}</strong> כפעולה שבוטלה. לא נגבה ממך תשלום, והקופה המקומית סומנה כמבוטלת.</p>
        <div class="price-list">
            <div class="price-item"><strong>סטטוס הזמנה</strong><span>{_status_badge(order.status)}</span></div>
            <div class="price-item"><strong>סטטוס PayPal</strong><span>{_escape(_paypal_status_label(order.paypal_status))}</span></div>
        </div>
        <div class="actions"><a class="link-button" href="/systems">חזרה לחנות</a><a class="link-button ghost-button" href="/cart">לעגלה</a></div>
    </div>
    """
    body = _public_shell(
        session,
        current_path="/checkout",
        title="תשלום PayPal בוטל",
        intro="התשלום בוטל לפני אישור סופי, ולכן לא בוצעה מסירה של מערכות.",
        login_path=request.path_qs or request.path,
        section_label="PayPal",
        content=content,
    )
    return _page_response("תשלום PayPal בוטל", body)


async def website_cart_page(request: web.Request) -> web.Response:
    bot, session = await _require_active_site_session(request)
    if request.method == "POST":
        form = await request.post()
        action = str(form.get("action", "")).strip()
        if action == "add":
            system_id = _parse_positive_int(form.get("system_id"), "מזהה מערכת")
            assert system_id is not None
            system = await bot.services.systems.get_system(system_id)
            await bot.services.cart.add_system(session.discord_user_id, system)
            raise web.HTTPFound("/cart?saved=added")
        if action == "remove":
            system_id = _parse_positive_int(form.get("system_id"), "מזהה מערכת")
            assert system_id is not None
            await bot.services.cart.remove_system(session.discord_user_id, system_id)
            raise web.HTTPFound("/cart?saved=removed")
        if action == "clear":
            await bot.services.cart.clear_cart(session.discord_user_id)
            raise web.HTTPFound("/cart?saved=cleared")

    notice_map = {
        "added": "המערכת נוספה לעגלה.",
        "removed": "המערכת הוסרה מהעגלה.",
        "cleared": "העגלה נוקתה בהצלחה.",
    }
    saved_key = str(request.query.get("saved", "")).strip().lower()
    notice = notice_map.get(saved_key)
    items = await bot.services.cart.list_items(session.discord_user_id)
    personal_discounts = {
        record.system.id: record.discount_percent
        for record in await bot.services.discounts.list_user_discounts(session.discord_user_id)
    }
    pricing_rows, subtotal, currency = _build_cart_pricing(items, personal_discounts=personal_discounts)

    if pricing_rows:
        cart_rows = "".join(
            f'''
            <div class="price-item">
                <div>
                    <strong>{_escape(row["item"].system.name)}</strong><br>
                    <span class="muted">{_escape(row["item"].system.description[:120])}</span>
                    {f'<br><span class="muted">הנחה אישית {row["personal_discount_percent"]}%</span>' if row["personal_discount_percent"] else ''}
                </div>
                <div>
                    <strong>{_escape(_money_label(row["effective_price"], currency))}</strong>
                    {f'<br><span class="muted">מחיר רגיל {_escape(_money_label(row["base_price"], currency))}</span>' if row["personal_discount_percent"] else ''}
                </div>
                <form method="post" class="inline-form">
                    <input type="hidden" name="action" value="remove">
                    <input type="hidden" name="system_id" value="{row["item"].system.id}">
                    <button type="submit" class="ghost-button danger">הסר</button>
                </form>
            </div>
            '''
            for row in pricing_rows
        )
        summary_html = f'''
        <div class="card stack">
            <h2>סיכום העגלה</h2>
            <div class="price-list">
                <div class="price-item"><strong>כמות מערכות</strong><span>{len(pricing_rows)}</span></div>
                <div class="price-item"><strong>סכום ביניים</strong><span>{_escape(_money_label(subtotal, currency))}</span></div>
            </div>
            <div class="actions">
                <a class="link-button" href="/checkout">מעבר לקופה</a>
                <a class="link-button ghost-button" href="/systems">המשך קניה</a>
                <form method="post" class="inline-form"><input type="hidden" name="action" value="clear"><button type="submit" class="ghost-button danger">נקה עגלה</button></form>
            </div>
        </div>
        '''
        content = f'''
        {_notice_html(notice, success=True)}
        <div class="split-grid">
            <div class="card stack"><h2>הפריטים שלך</h2><div class="price-list">{cart_rows}</div></div>
            {summary_html}
        </div>
        '''
    else:
        content = _notice_html(notice, success=True) + '''
        <div class="empty-card">
            <h2>העגלה ריקה כרגע</h2>
            <p>אפשר לחזור לקטלוג, להוסיף כמה מערכות שתרצה, ואז להמשיך לקופה אחת משותפת.</p>
            <div class="actions"><a class="link-button" href="/systems">למעבר למערכות</a></div>
        </div>
        '''

    body = _public_shell(
        session,
        current_path="/cart",
        title="העגלה שלך",
        intro="כאן אפשר לרכז כמה מערכות להזמנה אחת, לראות הנחות אישיות לפני הקופה ולעבור להזמנת תשלום מרוכזת.",
        login_path=request.path,
        section_label="עגלה",
        content=content,
    )
    return _page_response("העגלה שלך", body)


async def website_checkout_page(request: web.Request) -> web.Response:
    notice: str | None = None
    success = True
    code_text = ""
    note = ""
    payment_method = "paypal"
    fulfillment_mode = "self"

    try:
        bot, session = await _require_active_site_session(request)
        items = await bot.services.cart.list_items(session.discord_user_id)
        personal_discounts = {
            record.system.id: record.discount_percent
            for record in await bot.services.discounts.list_user_discounts(session.discord_user_id)
        }
        pricing_rows, subtotal, currency = _build_cart_pricing(items, personal_discounts=personal_discounts)
        if not pricing_rows:
            content = '''
            <div class="empty-card">
                <h2>אין מה לשלוח לקופה</h2>
                <p>צריך להוסיף לפחות מערכת אחת לעגלה לפני יצירת הזמנה.</p>
                <div class="actions"><a class="link-button" href="/systems">למעבר למערכות</a></div>
            </div>
            '''
            body = _public_shell(
                session,
                current_path="/cart",
                title="קופה",
                intro="הקופה מחכה לפריטים מהעגלה שלך.",
                login_path=request.path,
                section_label="קופה",
                content=content,
            )
            return _page_response("קופה", body)

        code_record: DiscountCodeRecord | None = None
        code_discount_amount = Decimal("0.00")

        if request.method == "POST":
            form = await request.post()
            payment_method = str(form.get("payment_method", "paypal")).strip().lower() or "paypal"
            fulfillment_mode = str(form.get("fulfillment_mode", "self")).strip().lower() or "self"
            code_text = str(form.get("discount_code", "")).strip().upper()
            note = str(form.get("note", "")).strip()
            action = str(form.get("action", "preview")).strip().lower()

            if payment_method == "card" and not WEBSITE_CARD_CHECKOUT_ENABLED:
                raise PermissionDeniedError("תשלום בכרטיס אשראי עדיין לא זמין באתר.")

            if not WEBSITE_CARD_CHECKOUT_ENABLED and not bot.settings.paypal_checkout_enabled:
                raise ConfigurationError("אין כרגע אמצעי תשלום פעיל באתר. הפעל את PayPal כדי לאפשר קופה.")

            if payment_method == "paypal" and not bot.settings.paypal_checkout_enabled:
                raise ConfigurationError("PayPal עדיין לא מוגדר בשרת. עדכן PAYPAL_CLIENT_ID ו-PAYPAL_CLIENT_SECRET כדי להפעיל אותו.")

            if code_text:
                try:
                    code_record, discount_amount_text = await bot.services.discount_codes.preview_discount(
                        session.discord_user_id,
                        code_text,
                        items,
                    )
                    code_discount_amount = _money_decimal(discount_amount_text)
                except SalesBotError as exc:
                    notice = str(exc)
                    success = False
                    code_record = None
                    code_discount_amount = Decimal("0.00")

            total_amount = max(Decimal("0.00"), subtotal - code_discount_amount)
            if action == "submit" and success:
                effective_items = [
                    (row["item"].system, format(row["effective_price"], "f"))
                    for row in pricing_rows
                ]
                order = await bot.services.payments.create_checkout_order(
                    user_id=session.discord_user_id,
                    payment_method=payment_method,
                    fulfillment_mode=fulfillment_mode,
                    items=effective_items,
                    subtotal_amount=format(subtotal, "f"),
                    discount_amount=format(code_discount_amount, "f"),
                    total_amount=format(total_amount, "f"),
                    currency=currency,
                    note=note or None,
                    discount_code_id=code_record.id if code_record is not None else None,
                    discount_code_text=code_record.code if code_record is not None else None,
                )

                owner_lines = [
                    f"משתמש: {_session_label(session)} ({session.discord_user_id})",
                    f"מזהה הזמנה: #{order.id}",
                    f"שיטת תשלום: {_checkout_method_label(order.payment_method)}",
                    f"מסלול מסירה: {_checkout_fulfillment_label(order.fulfillment_mode)}",
                    f"סטטוס: {ORDER_STATUS_LABELS.get(order.status, order.status)}",
                    f"סכום ביניים: {_money_label(subtotal, currency)}",
                ]
                if code_record is not None:
                    owner_lines.append(f"קוד הנחה: {code_record.code} (-{_money_label(code_discount_amount, currency)})")
                owner_lines.append(f"סה\"כ: {_money_label(total_amount, currency)}")
                if note:
                    owner_lines.append(f"הערת לקוח: {note}")

                if payment_method == "paypal":
                    order = await bot.services.payments.start_paypal_checkout(bot, order.id)
                    if code_record is not None:
                        await bot.services.discount_codes.record_redemption(
                            code_record.id,
                            session.discord_user_id,
                            order.id,
                            format(code_discount_amount, "f"),
                        )
                    await bot.services.cart.clear_cart(session.discord_user_id)
                    raise web.HTTPFound(order.paypal_approval_url or f"/checkout/paypal/return?order_id={order.id}")

                if code_record is not None:
                    await bot.services.discount_codes.record_redemption(
                        code_record.id,
                        session.discord_user_id,
                        order.id,
                        format(code_discount_amount, "f"),
                    )

                await bot.services.cart.clear_cart(session.discord_user_id)

                owner_lines.append("מערכות:")
                owner_lines.extend(
                    f"- {row['item'].system.name}: {_money_label(row['effective_price'], currency)}"
                    for row in pricing_rows
                )
                owner_lines.append(f"קישור ניהול: {bot.settings.public_base_url}/admin/checkouts")
                await bot.services.payments.send_checkout_admin_notification(
                    bot,
                    title="הגיעה הזמנת קופה חדשה מהאתר",
                    body="\n".join(owner_lines),
                )

                summary_message = (
                    f"הזמנה #{order.id} נפתחה ונשמרה במערכת. הסכום הכולל הוא {_money_label(order.total_amount, order.currency)} "
                    f"בשיטת {_checkout_method_label(order.payment_method)}."
                )
                if order.fulfillment_mode == "gift":
                    summary_message += " אחרי אישור התשלום ייווצרו עבורך קודי מימוש שתוכל למסור לחבר דרך עמוד המימושים."
                else:
                    summary_message += " אחרי אישור התשלום המערכות יישלחו ישירות לחשבון שלך."
                await bot.services.notifications.create_notification(
                    user_id=session.discord_user_id,
                    title=f"הזמנה חדשה #{order.id}",
                    body=summary_message + " צוות האתר יעבור עליה ידנית ויעדכן אותך ברגע שיהיה שינוי.",
                    link_path="/inbox",
                    kind="checkout",
                )
                await _send_optional_user_dm(
                    bot,
                    user_id=session.discord_user_id,
                    title=f"הזמנה חדשה #{order.id}",
                    body=summary_message,
                    link_path="/inbox",
                )

                success_content = f'''
                <div class="card stack">
                    <h2>ההזמנה נפתחה</h2>
                    <p>הקופה נשמרה בהצלחה כמספר <strong>#{order.id}</strong>. כרגע האתר עובד עם אישור ידני לקופות מרובות פריטים, לכן הצוות יעבור על ההזמנה ויעדכן אותך דרך מרכז ההתראות.</p>
                    <p>{'אחרי אישור התשלום יופקו עבורך קודי מימוש למסירה לחבר.' if order.fulfillment_mode == 'gift' else 'אחרי אישור התשלום המערכות יישלחו ישירות לחשבון שלך.'}</p>
                    <div class="price-list">
                        <div class="price-item"><strong>שיטת תשלום</strong><span>{_escape(_checkout_method_label(order.payment_method))}</span></div>
                        <div class="price-item"><strong>מסלול מסירה</strong><span>{_escape(_checkout_fulfillment_label(order.fulfillment_mode))}</span></div>
                        <div class="price-item"><strong>סכום לתשלום</strong><span>{_escape(_money_label(order.total_amount, order.currency))}</span></div>
                        <div class="price-item"><strong>סטטוס</strong><span>{_status_badge(order.status)}</span></div>
                    </div>
                    <div class="actions"><a class="link-button" href="/inbox">למרכז ההתראות</a><a class="link-button ghost-button" href="{'/redeem' if order.fulfillment_mode == 'gift' else '/systems'}">{'לעמוד המימושים' if order.fulfillment_mode == 'gift' else 'חזור לחנות'}</a></div>
                </div>
                '''
                body = _public_shell(
                    session,
                    current_path="/cart",
                    title="הזמנה נוצרה",
                    intro="הקופה נשמרה ונשלחה לטיפול מנהל.",
                    login_path=request.path,
                    section_label="קופה",
                    content=success_content,
                )
                return _page_response("הזמנה נוצרה", body)

        else:
            total_amount = subtotal

        if request.method != "POST":
            total_amount = subtotal

        checkout_rows = "".join(
            f'''
            <div class="price-item">
                <div>
                    <strong>{_escape(row["item"].system.name)}</strong>
                    {f'<br><span class="muted">הנחה אישית {row["personal_discount_percent"]}%</span>' if row["personal_discount_percent"] else ''}
                </div>
                <span>{_escape(_money_label(row["effective_price"], currency))}</span>
            </div>
            '''
            for row in pricing_rows
        )
        if code_record is not None:
            total_amount = max(Decimal("0.00"), subtotal - code_discount_amount)
            code_notice = f'<div class="meta-card"><strong>קוד פעיל:</strong> {_escape(code_record.code)} | חיסכון {_escape(_money_label(code_discount_amount, currency))}</div>'
        else:
            code_notice = '<div class="meta-card"><strong>קוד הנחה:</strong> אפשר להזין קוד וללחוץ על בדיקה לפני שליחת הקופה.</div>'

        if fulfillment_mode == "gift":
            fulfillment_hint = "במסלול הזה, אחרי שהתשלום יאושר ייווצר קוד מימוש נפרד לכל מערכת שקנית. מי שמקבל את הקוד יוכל לממש אותו פעם אחת, ורק אם הוא עדיין לא מחזיק במערכת."
        else:
            fulfillment_hint = "במסלול הרגיל, אחרי אישור התשלום המערכות יישלחו ישירות לחשבון שמבצע את הקנייה."

        payment_options_html = '<option value="card" disabled>כרטיס אשראי (לא זמין כרגע)</option>'
        submit_disabled_attr = ''
        preview_disabled_attr = ''
        if bot.settings.paypal_checkout_enabled:
            payment_options_html += '<option value="paypal"' + (' selected' if payment_method == 'paypal' else '') + '>PayPal</option>'
            payment_hint = 'כרגע רק PayPal פעיל בקופה. כרטיס אשראי עדיין כבוי עד שתגדיר את המסלול הזה, ולכן התשלום יתבצע דרך PayPal בלבד.'
        else:
            payment_hint = 'כרגע כרטיס אשראי כבוי ו-PayPal עדיין לא מוגדר בשרת הזה, לכן אי אפשר לפתוח קופה חדשה עד שתשלים את הגדרת PayPal.'
            submit_disabled_attr = ' disabled'
            preview_disabled_attr = ' disabled'

        content = f'''
        {_notice_html(notice, success=success)}
        <div class="split-grid">
            <div class="card stack">
                <h2>פרטי הקופה</h2>
                <form method="post" class="stack">
                    <label class="field"><span>שיטת תשלום</span><select name="payment_method">{payment_options_html}</select></label>
                    <label class="field"><span>מסלול מסירה</span><select name="fulfillment_mode">{_checkout_fulfillment_options(fulfillment_mode)}</select></label>
                    <label class="field"><span>קוד הנחה</span><input type="text" name="discount_code" maxlength="32" value="{_escape(code_text)}" placeholder="SUMMER10"></label>
                    <label class="field field-wide"><span>הערה לצוות</span><textarea name="note" placeholder="למשל: עדיף לפנות אליי קודם בדיסקורד">{_escape(note)}</textarea></label>
                    <div class="actions"><button type="submit" name="action" value="preview" class="ghost-button"{preview_disabled_attr}>בדוק קוד וחשב מחדש</button><button type="submit" name="action" value="submit"{submit_disabled_attr}>שלח קופה</button></div>
                </form>
                {code_notice}
                <p class="muted">{_escape(payment_hint)}</p>
                <p class="muted">{_escape(fulfillment_hint)}</p>
            </div>
            <div class="card stack">
                <h2>סיכום חיוב</h2>
                <div class="price-list">{checkout_rows}</div>
                <div class="price-list">
                    <div class="price-item"><strong>סכום ביניים</strong><span>{_escape(_money_label(subtotal, currency))}</span></div>
                    <div class="price-item"><strong>הנחת קוד</strong><span>{_escape(_money_label(code_discount_amount, currency))}</span></div>
                    <div class="price-item"><strong>סה"כ</strong><span>{_escape(_money_label(total_amount, currency))}</span></div>
                    <div class="price-item"><strong>מסלול מסירה</strong><span>{_escape(_checkout_fulfillment_label(fulfillment_mode))}</span></div>
                </div>
            </div>
        </div>
        '''
        body = _public_shell(
            session,
            current_path="/cart",
            title="קופה",
            intro="בדוק את כל הפריטים, החל קוד הנחה אם יש, ואז פתח הזמנת תשלום אחת לכל המערכות יחד.",
            login_path=request.path,
            section_label="קופה",
            content=content,
        )
        return _page_response("קופה", body)
    except web.HTTPException:
        raise
    except SalesBotError as exc:
        notice = str(exc)
        success = False
        bot, session = await _require_active_site_session(request)
        content = _notice_html(notice, success=success) + '<div class="actions"><a class="link-button" href="/cart">חזרה לעגלה</a></div>'
        body = _public_shell(
            session,
            current_path="/cart",
            title="קופה",
            intro="אירעה שגיאה במהלך בדיקת הקופה.",
            login_path=request.path,
            section_label="קופה",
            content=content,
        )
        return _page_response("קופה", body)


async def website_inbox_page(request: web.Request) -> web.Response:
    bot, session = await _require_active_site_session(request)
    if request.method == "POST":
        form = await request.post()
        action = str(form.get("action", "")).strip()
        if action == "mark-read":
            notification_id = _parse_positive_int(form.get("notification_id"), "מזהה התראה")
            assert notification_id is not None
            await bot.services.notifications.mark_read(session.discord_user_id, notification_id)
            raise web.HTTPFound("/inbox?saved=read")
        if action == "mark-all-read":
            await bot.services.notifications.mark_all_read(session.discord_user_id)
            raise web.HTTPFound("/inbox?saved=all-read")

    notifications = await bot.services.notifications.list_notifications(session.discord_user_id)
    unread_count = await bot.services.notifications.unread_count(session.discord_user_id)
    orders = await bot.services.payments.list_user_checkout_orders(session.discord_user_id)
    notice_map = {
        "read": "ההתראה סומנה כנקראה.",
        "all-read": "כל ההתראות סומנו כנקראו.",
    }
    notice = notice_map.get(str(request.query.get("saved", "")).strip().lower())

    notifications_html = ''.join(
        f'''
        <div class="card stack">
            <div class="actions"><div>{'<span class="catalog-badge warn">לא נקראה</span>' if not record.is_read else '<span class="catalog-badge">נקראה</span>'}</div></div>
            <div>
                <h3>{_escape(record.title)}</h3>
                <p>{_escape(record.body)}</p>
                <p class="muted">נשלח ב-{_escape(record.created_at)}</p>
                {f'<p><a href="{_escape(record.link_path)}">פתח קישור קשור</a></p>' if record.link_path else ''}
            </div>
            <div class="actions">{'' if record.is_read else f'<form method="post" class="inline-form"><input type="hidden" name="action" value="mark-read"><input type="hidden" name="notification_id" value="{record.id}"><button type="submit" class="ghost-button">סמן כנקראה</button></form>'}</div>
        </div>
        '''
        for record in notifications
    ) or '<div class="empty-card"><p>עדיין אין התראות בחשבון שלך.</p></div>'

    orders_html = ''.join(
        f'''
        <div class="card stack">
            <div class="price-list">
                <div class="price-item"><strong>הזמנה #{order.id}</strong><span>{_status_badge(order.status)}</span></div>
                <div class="price-item"><strong>אמצעי תשלום</strong><span>{_escape(_checkout_method_label(order.payment_method))}</span></div>
                <div class="price-item"><strong>מסלול מסירה</strong><span>{_escape(_checkout_fulfillment_label(order.fulfillment_mode))}</span></div>
                <div class="price-item"><strong>סה"כ</strong><span>{_escape(_money_label(order.total_amount, order.currency))}</span></div>
                <div class="price-item"><strong>נפתחה ב</strong><span>{_escape(order.created_at)}</span></div>
                {f'<div class="price-item"><strong>סיבת ביטול</strong><span>{_escape(order.cancel_reason or "-")}</span></div>' if order.status == 'cancelled' else ''}
            </div>
        </div>
        '''
        for order in orders
    ) or '<div class="empty-card"><p>עדיין אין הזמנות קופה חדשות בחשבון שלך.</p></div>'

    content = f'''
    {_notice_html(notice, success=True)}
    <div class="profile-summary-grid">
        <div class="summary-tile"><strong>{unread_count}</strong><span>התראות שלא נקראו</span></div>
        <div class="summary-tile"><strong>{len(notifications)}</strong><span>סה"כ התראות</span></div>
        <div class="summary-tile"><strong>{len(orders)}</strong><span>הזמנות קופה</span></div>
    </div>
    <div class="split-grid">
        <div class="card stack">
            <div class="actions"><h2>התראות</h2><form method="post" class="inline-form"><input type="hidden" name="action" value="mark-all-read"><button type="submit" class="ghost-button">סמן הכל כנקרא</button></form></div>
            {notifications_html}
        </div>
        <div class="card stack">
            <h2>הזמנות קופה</h2>
            {orders_html}
        </div>
    </div>
    '''
    body = _public_shell(
        session,
        current_path="/inbox",
        title="מרכז ההתראות",
        intro="כאן יופיעו עדכוני צוות, תשובות להזמנות, ואישור או ביטול של קופות שנפתחו דרך האתר.",
        login_path=request.path,
        section_label="התראות",
        content=content,
    )
    return _page_response("מרכז ההתראות", body)


async def website_redeem_page(request: web.Request) -> web.Response:
    notice: str | None = None
    success = True
    code_text = ""
    bot, session = await _require_active_site_session(request)

    if request.method == "POST":
        form = await request.post()
        code_text = str(form.get("code", "")).strip()
        action = str(form.get("action", "redeem")).strip().lower()
        try:
            if action == "redeem":
                updated_code, system = await bot.services.redeem_codes.redeem_code(
                    bot,
                    user_id=session.discord_user_id,
                    code_text=code_text,
                )
                notice = f"הקוד {updated_code.code} מומש בהצלחה עבור {system.name}."
                code_text = ""
        except SalesBotError as exc:
            notice = str(exc)
            success = False

    systems = await bot.services.systems.list_systems()
    system_names = {system.id: system.name for system in systems}
    codes = await bot.services.redeem_codes.list_codes_for_user(session.discord_user_id, limit=120)
    code_cards = ''.join(
        f'''
        <div class="card stack">
            <div class="price-list">
                <div class="price-item"><strong>{_escape(code.code)}</strong><span>{_redeem_code_state_label(code)}</span></div>
                <div class="price-item"><strong>מערכת</strong><span>{_escape(system_names.get(code.system_id or -1, 'לא מחובר למערכת'))}</span></div>
                <div class="price-item"><strong>מקור</strong><span>{_escape(_redeem_code_source_label(code.source))}</span></div>
                <div class="price-item"><strong>שימושים</strong><span>{code.redeemed_count}/{code.max_redemptions}</span></div>
                <div class="price-item"><strong>תפוגה</strong><span>{_escape(code.expires_at or 'ללא')}</span></div>
                {f'<div class="price-item"><strong>הזמנת מקור</strong><span>#{code.checkout_order_id}</span></div>' if code.checkout_order_id is not None else ''}
            </div>
            <p class="muted">אפשר למסור את הקוד הזה לחבר. אם הוא כבר מחזיק במערכת או מימש את הקוד, המימוש ייחסם.</p>
        </div>
        '''
        for code in codes
    ) or '<div class="empty-card"><p>עדיין אין קודי מימוש בחשבון שלך.</p></div>'

    content = f'''
    {_notice_html(notice, success=success)}
    <div class="split-grid">
        <div class="card stack">
            <h2>מימוש קוד</h2>
            <form method="post" class="stack">
                <input type="hidden" name="action" value="redeem">
                <label class="field"><span>קוד מימוש</span><input type="text" name="code" maxlength="64" value="{_escape(code_text)}" placeholder="S12XABCDEFGH" required></label>
                <div class="actions"><button type="submit">ממש קוד</button></div>
            </form>
        </div>
        <div class="card stack">
            <h2>איך זה עובד</h2>
            <p>אם קיבלת קוד ממישהו שקנה בשבילך, מזינים אותו כאן. המערכת תבדוק שהקוד עדיין פעיל, לא פג תוקף, ויש בו מימוש פנוי.</p>
            <p>אי אפשר לממש קוד למערכת שכבר בבעלותך, ואחרי מימוש מוצלח המערכת תישלח אליך ב-DM ותופיע גם בעמוד הפרופיל.</p>
        </div>
    </div>
    <div class="card stack">
        <h2>הקודים שלי</h2>
        <p class="muted">כאן מופיעים כל קודי המתנה שנוצרו עבורך בקופות מסוג "לקנות בשביל חבר".</p>
        <div class="stack">{code_cards}</div>
    </div>
    '''
    body = _public_shell(
        session,
        current_path="/redeem",
        title="מימוש קודים",
        intro="כאן אפשר לממש קוד שקיבלת וגם לראות את כל קודי המתנה שיצאו מהקניות שלך.",
        login_path=request.path,
        section_label="מימוש קודים",
        content=content,
    )
    return _page_response("מימוש קודים", body)


async def website_profile_page(request: web.Request) -> web.Response:
    bot, session = await _require_active_site_session(request)
    notice: str | None = None
    is_admin = await bot.services.admins.is_admin(session.discord_user_id)
    avatar_url = _session_avatar(session)
    avatar_html = f'<img class="profile-avatar" src="{_escape(avatar_url)}" alt="avatar">' if avatar_url else '<div class="profile-avatar"></div>'
    try:
        roblox_link = await bot.services.oauth.get_link(session.discord_user_id)
    except SalesBotError:
        roblox_link = None
    ownerships = await bot.services.ownership.list_user_ownerships(session.discord_user_id)
    discounts = await bot.services.discounts.list_user_discounts(session.discord_user_id)
    owned_list = ''.join(
        f'<div class="price-item"><strong>{_escape(ownership.system.name)}</strong><span>{_escape(ownership.granted_at)}</span><a class="link-button ghost-button" href="/downloads/{ownership.system.id}">הורדה</a></div>'
        for ownership in ownerships
    ) or '<div class="empty-card"><p>עדיין אין מערכות בבעלותך.</p></div>'
    discount_list = ''.join(
        f'<div class="price-item"><strong>{_escape(record.system.name)}</strong><span>{record.discount_percent}% הנחה</span></div>'
        for record in discounts
    ) or '<div class="empty-card"><p>אין כרגע הנחות שמחוברות לחשבון שלך.</p></div>'
    roblox_block = '<div class="price-item"><strong>רובלוקס</strong><span>לא מחובר כרגע</span></div>'
    if roblox_link is not None:
        summary = ' | '.join(part for part in (roblox_link.roblox_display_name, roblox_link.roblox_username, roblox_link.roblox_sub) if part)
        profile_link_html = f'<a href="{_escape(roblox_link.profile_url or "")}" target="_blank" rel="noreferrer">פתח פרופיל</a>' if roblox_link.profile_url else 'אין קישור פרופיל'
        roblox_block = f'<div class="price-item"><strong>רובלוקס</strong><span>{_escape(summary)}</span></div><div class="price-item"><strong>פרופיל</strong><span>{profile_link_html}</span></div>'
    content = f"""
    {_notice_html(notice, success=True)}
    <div class="profile-grid">
        <div class="card stack">
            <div class="profile-hero">
                {avatar_html}
                <div>
                    <p class="eyebrow">החשבון שלך</p>
                    <h2>{_escape(_session_label(session))}</h2>
                    <p>כאן מרוכזים פרטי דיסקורד, החיבור לרובלוקס, המערכות שבבעלותך וההנחות שמחוברות לחשבון.</p>
                </div>
            </div>
            <div class="profile-summary-grid">
                <div class="summary-tile"><strong>{len(ownerships)}</strong><span>מערכות בבעלותך</span></div>
                <div class="summary-tile"><strong>{len(discounts)}</strong><span>הנחות פעילות</span></div>
                <div class="summary-tile"><strong>{_escape(_admin_rank_label(bot, session.discord_user_id) if is_admin else 'לקוח')}</strong><span>סוג חשבון</span></div>
            </div>
            <div class="price-list">
                <div class="price-item"><strong>דיסקורד</strong><span>{_escape(_session_label(session))}</span></div>
                <div class="price-item"><strong>מזהה משתמש</strong><span class="mono">{_escape(session.discord_user_id)}</span></div>
                {roblox_block}
            </div>
        </div>
    </div>
    <div class="split-grid">
        <div class="card stack">
            <h2>המערכות שלך</h2>
            <div class="system-download-list">{owned_list}</div>
        </div>
        <div class="card stack">
            <h2>הנחות על מערכות</h2>
            <div class="price-list">{discount_list}</div>
        </div>
    </div>
    """
    body = _public_shell(
        session,
        current_path=request.path,
        title="הפרופיל שלך",
        intro="כל המידע האישי, ההורדות וההעדפות שלך באתר מרוכזים כאן.",
        login_path=request.path,
        section_label="פרופיל",
        content=content,
    )
    return _page_response("הפרופיל שלך", body)


async def owned_system_download_page(request: web.Request) -> web.StreamResponse:
    bot, session = await _require_active_site_session(request)
    system_id = int(request.match_info["system_id"])
    if not await bot.services.ownership.user_owns_system(session.discord_user_id, system_id):
        raise PermissionDeniedError("המערכת שביקשת לא שייכת לחשבון המחובר.")

    system = await bot.services.systems.get_system(system_id)
    asset = await bot.services.systems.get_system_asset(system.id, asset_type=bot.services.systems.FILE_ASSET_TYPE)
    stored_path = bot.services.systems.resolve_storage_path(system.file_path)
    if stored_path is not None and stored_path.is_file():
        response = web.FileResponse(stored_path)
        response.headers["Content-Disposition"] = f'attachment; filename="{stored_path.name}"'
        return response
    if asset is not None:
        content_type = mimetypes.guess_type(asset.asset_name)[0] or "application/octet-stream"
        response = web.Response(body=asset.asset_bytes, content_type=content_type)
        response.headers["Content-Disposition"] = f'attachment; filename="{asset.asset_name}"'
        return response
    raise web.HTTPNotFound(text="קובץ המערכת לא נמצא כרגע להורדה.")


async def system_image_page(request: web.Request) -> web.StreamResponse:
    bot: SalesBot = request.app["bot"]
    system = await bot.services.systems.get_system(int(request.match_info["system_id"]))
    if not system.image_path:
        images = await bot.services.systems.list_system_images(system.id)
        if images:
            image = images[0]
            content_type = image.content_type or mimetypes.guess_type(image.asset_name)[0] or "application/octet-stream"
            return web.Response(body=image.asset_bytes, content_type=content_type)
        raise NotFoundError("לא נמצאה תמונת מערכת.")
    asset = await bot.services.systems.get_system_asset(system.id, asset_type=bot.services.systems.IMAGE_ASSET_TYPE)
    stored_path = bot.services.systems.resolve_storage_path(system.image_path)
    if stored_path is not None and stored_path.is_file():
        return web.FileResponse(stored_path)
    if asset is not None:
        content_type = mimetypes.guess_type(asset.asset_name)[0] or "application/octet-stream"
        return web.Response(body=asset.asset_bytes, content_type=content_type)
    raise NotFoundError("לא נמצאה תמונת מערכת.")


async def system_gallery_image_page(request: web.Request) -> web.Response:
    bot: SalesBot = request.app["bot"]
    try:
        image = await bot.services.systems.get_system_gallery_image(int(request.match_info["image_id"]))
        content_type = image.content_type or mimetypes.guess_type(image.asset_name)[0] or "application/octet-stream"
        return web.Response(body=image.asset_bytes, content_type=content_type)
    except SalesBotError as exc:
        return _error_response("תמונת מערכת", str(exc), status=404)


async def website_vouches_page(request: web.Request) -> web.Response:
    bot, session = await _require_active_site_session(request)
    is_admin = await bot.services.admins.is_admin(session.discord_user_id)
    notice: str | None = None
    if request.method == "POST":
        form = await request.post()
        action = str(form.get("action", "")).strip()
        if action == "delete":
            if not is_admin:
                raise PermissionDeniedError("רק אדמינים יכולים למחוק דירוגים מהאתר.")
            vouch_id = _parse_positive_int(form.get("vouch_id"), "מזהה דירוג")
            assert vouch_id is not None
            deleted_vouch = await bot.services.vouches.delete_vouch(vouch_id)
            channel = bot.get_channel(bot.settings.vouch_channel_id)
            if channel is None:
                try:
                    channel = await bot.fetch_channel(bot.settings.vouch_channel_id)
                except discord.HTTPException:
                    channel = None
            if deleted_vouch.posted_message_id is not None and channel is not None and hasattr(channel, "fetch_message"):
                try:
                    posted_message = await channel.fetch_message(deleted_vouch.posted_message_id)
                    await posted_message.delete()
                except (discord.NotFound, discord.Forbidden, discord.HTTPException):
                    pass
            notice = "הדירוג נמחק בהצלחה."
    admin_ids = await bot.services.admins.list_admin_ids()
    vouch_lists = await asyncio.gather(*(bot.services.vouches.list_vouches(admin_user_id) for admin_user_id in admin_ids))
    vouches = sorted((vouch for records in vouch_lists for vouch in records), key=lambda record: (record.created_at, record.id), reverse=True)
    label_user_ids = list({vouch.admin_user_id for vouch in vouches} | {vouch.author_user_id for vouch in vouches})
    label_values = await asyncio.gather(*(_discord_user_label(bot, user_id) for user_id in label_user_ids)) if label_user_ids else []
    labels_by_user_id = dict(zip(label_user_ids, label_values, strict=False))
    cards: list[str] = []
    for vouch in vouches:
        admin_label = labels_by_user_id.get(vouch.admin_user_id, str(vouch.admin_user_id))
        author_label = labels_by_user_id.get(vouch.author_user_id, str(vouch.author_user_id))
        delete_form = ""
        if is_admin:
            delete_form = f'<form method="post" class="inline-form"><input type="hidden" name="action" value="delete"><input type="hidden" name="vouch_id" value="{vouch.id}"><button type="submit" class="ghost-button danger">מחיקה</button></form>'
        cards.append(
            f"""
            <article class="vouch-card">
                <div class="stack">
                    <div><strong>{_escape(admin_label)}</strong><p class="muted">דירוג מאת {_escape(author_label)}</p></div>
                    <div class="stars">{'★' * vouch.rating}</div>
                    <p>{_escape(vouch.reason)}</p>
                    <div class="actions">{delete_form}</div>
                </div>
            </article>
            """
        )
    content = _notice_html(notice, success=True) + (
        '<div class="vouch-list">' + ''.join(cards) + '</div>'
        if cards
        else '<div class="empty-card"><h2>עדיין אין דירוגים</h2><p>כשהלקוחות יתחילו להשאיר דירוגים, הם יופיעו כאן.</p></div>'
    )
    body = _public_shell(
        session,
        current_path=request.path,
        title="דירוגים",
        intro="כל הדירוגים שנשמרו בדיסקורד מוצגים כאן גם באתר.",
        login_path=request.path,
        section_label="דירוגים",
        content=content,
    )
    return _page_response("דירוגים", body)


async def blacklist_appeal_page(request: web.Request) -> web.Response:
    notice_map = {
        "submitted": "הערעור נשלח בהצלחה לצוות האתר ויופיע עכשיו בפאנל הניהול.",
    }
    notice = notice_map.get(str(request.query.get("saved", "")).strip().lower())
    success = True
    bot, session = await _require_site_session(request)
    entry = await _ensure_site_session_allowed(bot, session, allow_blacklisted=True)
    if entry is None:
        raise PermissionDeniedError("עמוד הערעור זמין רק למשתמשים שנמצאים כרגע בבלאקליסט.")

    discord_name = _session_label(session)
    roblox_name = ""
    try:
        linked_account = await bot.services.oauth.get_link(session.discord_user_id)
    except SalesBotError:
        linked_account = None
    if linked_account is not None:
        roblox_name = linked_account.roblox_display_name or linked_account.roblox_username or linked_account.roblox_sub or ""

    pending_appeal = await bot.services.blacklist.get_pending_appeal_for_user(session.discord_user_id)
    appeal_reason = ""
    if request.method == "POST":
        try:
            form = await request.post()
            if pending_appeal is not None:
                raise PermissionDeniedError("כבר שלחת ערעור שממתין לבדיקה. חכה להחלטת הצוות לפני שליחה נוספת.")
            roblox_name = str(form.get("roblox_name", "")).strip()
            appeal_reason = str(form.get("appeal_reason", "")).strip()
            if not roblox_name or not appeal_reason:
                raise PermissionDeniedError("חובה למלא את כל שדות הערעור.")

            appeal = await bot.services.blacklist.create_appeal(
                session.discord_user_id,
                f"שם ברובלוקס: {roblox_name}\nשם בדיסקורד: {discord_name}",
                appeal_reason,
            )
            for admin_user_id in dict.fromkeys(await bot.services.admins.list_admin_ids()):
                await bot.services.notifications.create_notification(
                    user_id=admin_user_id,
                    title=f"ערעור בלאקליסט חדש #{appeal.id}",
                    body=(
                        f"{discord_name} ({session.discord_user_id}) שלח ערעור חדש. "
                        "אפשר לפתוח את דף הבלאקליסט באתר כדי לאשר או לדחות אותו."
                    ),
                    link_path="/admin/blacklist",
                    kind="admin-blacklist-appeal",
                )
            raise web.HTTPFound("/blacklist-appeal?saved=submitted")
        except SalesBotError as exc:
            notice = str(exc)
            success = False
            pending_appeal = await bot.services.blacklist.get_pending_appeal_for_user(session.discord_user_id)

    appeal_panel_html = """
    <form method="post">
        <div class="grid">
            <label class="field"><span>מה השם שלך ברובלוקס</span><input type="text" name="roblox_name" value="{roblox_name}" required></label>
            <label class="field"><span>מה השם שלך בדיסקורד</span><input type="text" value="{discord_name}" disabled></label>
            <label class="field field-wide"><span>למה אתה חושב שמגיע לך שנוריד לך בלאקליסט?</span><textarea name="appeal_reason" required>{appeal_reason}</textarea></label>
        </div>
        <div class="actions"><button type="submit">שלח ערעור</button></div>
    </form>
    """.format(
        roblox_name=_escape(roblox_name),
        discord_name=_escape(discord_name),
        appeal_reason=_escape(appeal_reason),
    )
    if pending_appeal is not None:
        appeal_panel_html = f"""
        <div class="stack">
            <div class="price-list">
                <div class="price-item"><strong>סטטוס</strong><span>{_status_badge(pending_appeal.status)}</span></div>
                <div class="price-item"><strong>נשלח בתאריך</strong><span>{_escape(pending_appeal.submitted_at)}</span></div>
                <div class="price-item"><strong>הפרטים שנשלחו</strong><span>{_escape(pending_appeal.answer_one)}</span></div>
                <div class="price-item"><strong>סיבת הערעור</strong><span>{_escape(pending_appeal.answer_two)}</span></div>
            </div>
            <p class="muted">יש לך כבר ערעור ממתין. אחרי שהצוות יקבל החלטה, תוכל לשלוח ערעור חדש רק אם יהיה בכך צורך.</p>
        </div>
        """

    content = f"""
    {_notice_html(notice, success=success)}
    <div class="split-grid">
        <div class="card stack">
            <div>
                <h2>הגישה לאתר חסומה</h2>
                <p>החשבון המחובר שלך נמצא כרגע בבלאקליסט ולכן אין גישה לשאר דפי האתר.</p>
            </div>
            <div class="meta-card"><strong>הסיבה:</strong> {_escape(entry.reason or 'לא נמסרה סיבה.')}</div>
            <div class="meta-card"><strong>שם בדיסקורד:</strong> {_escape(discord_name)}</div>
        </div>
        <div class="card">
            <h2>שליחת ערעור</h2>
            <p class="muted">זהו הדף היחיד שזמין עבורך כרגע באתר. כאן אפשר לשלוח ערעור, והאדמינים יבדקו אותו ישירות מתוך האתר.</p>
            {appeal_panel_html}
        </div>
    </div>
    """
    body = _public_shell(
        session,
        current_path="/blacklist-appeal",
        title="ערעור על בלאקליסט",
        intro="אם אתה חושב שהבלאקליסט צריך לרדת, שלח כאן ערעור מסודר לצוות.",
        login_path=request.path,
        section_label="בלאקליסט",
        content=content,
        show_nav=False,
    )
    return _page_response("ערעור על בלאקליסט", body)


def _custom_order_admin_url(bot: "SalesBot", order_id: int) -> str:
    return f"{bot.settings.public_base_url}/admin/custom-orders/{order_id}"


async def admin_blacklist_page(request: web.Request) -> web.Response:
    notice_map = {
        "added": "המשתמש נוסף לבלאקליסט בהצלחה.",
        "removed": "המשתמש הוסר מהבלאקליסט.",
        "appeal-accepted": "הערעור התקבל והמשתמש עודכן.",
        "appeal-rejected": "הערעור נדחה והמשתמש עודכן.",
    }
    try:
        bot, session = await _require_admin_session(request)
        if request.method == "POST":
            form = await request.post()
            action = str(form.get("action", "")).strip().lower()
            if action == "add":
                user_id = _parse_positive_int(form.get("user_id"), "מזהה משתמש בדיסקורד")
                assert user_id is not None
                reason = str(form.get("reason", "")).strip()
                if not reason:
                    raise PermissionDeniedError("חובה להזין סיבה לפני שמכניסים משתמש לבלאקליסט.")
                display_label = f"{await _discord_user_label(bot, user_id)} - {user_id}"
                await bot.services.blacklist.add_entry(user_id, display_label, reason, session.discord_user_id)
                await bot.services.delivery.purge_deliveries(bot, user_id=user_id)
                raise web.HTTPFound("/admin/blacklist?saved=added")
            if action == "remove":
                user_id = _parse_positive_int(form.get("user_id"), "מזהה משתמש בדיסקורד")
                assert user_id is not None
                await bot.services.blacklist.remove_entry(user_id)
                raise web.HTTPFound("/admin/blacklist?saved=removed")
            if action in {"accept-appeal", "reject-appeal"}:
                appeal_id = _parse_positive_int(form.get("appeal_id"), "מזהה ערעור")
                assert appeal_id is not None
                appeal = await bot.services.blacklist.get_appeal(appeal_id)
                accepted = action == "accept-appeal"
                if accepted and await bot.services.blacklist.is_blacklisted(appeal.user_id):
                    await bot.services.blacklist.remove_entry(appeal.user_id)
                appeal = await bot.services.blacklist.resolve_appeal(
                    appeal.id,
                    reviewer_id=session.discord_user_id,
                    status="accepted" if accepted else "rejected",
                )
                title = "הערעור שלך התקבל" if accepted else "הערעור שלך נדחה"
                body = (
                    "הצוות קיבל את הערעור שלך והבלאקליסט הוסר מהחשבון שלך."
                    if accepted
                    else "הצוות בדק את הערעור שלך והחליט לדחות אותו כרגע."
                )
                link_path = "/profile" if accepted else "/blacklist-appeal"
                await bot.services.notifications.create_notification(
                    user_id=appeal.user_id,
                    title=title,
                    body=body,
                    link_path=link_path,
                    kind="blacklist-appeal",
                    created_by=session.discord_user_id,
                )
                await _send_optional_user_dm(bot, user_id=appeal.user_id, title=title, body=body, link_path=link_path)
                saved_key = "appeal-accepted" if accepted else "appeal-rejected"
                raise web.HTTPFound(f"/admin/blacklist?saved={saved_key}")
            raise PermissionDeniedError("הפעולה שנשלחה לעמוד הבלאקליסט לא תקינה.")

        entries, pending_appeals = await asyncio.gather(
            bot.services.blacklist.list_entries(),
            bot.services.blacklist.list_pending_appeals(),
        )
        entry_map = {entry.user_id: entry for entry in entries}
        appeal_labels = (
            await asyncio.gather(*(_discord_user_label(bot, appeal.user_id) for appeal in pending_appeals))
            if pending_appeals
            else []
        )
        blacklist_rows = "".join(
            f"""
            <tr>
                <td>{_escape(entry.display_label)}<br><span class="mono">{entry.user_id}</span></td>
                <td>{_escape(entry.reason or 'לא נמסרה')}</td>
                <td>{_escape(entry.blacklisted_at)}</td>
                <td>
                    <form method="post" class="inline-form">
                        <input type="hidden" name="action" value="remove">
                        <input type="hidden" name="user_id" value="{entry.user_id}">
                        <button type="submit" class="ghost-button danger">הסר</button>
                    </form>
                </td>
            </tr>
            """
            for entry in entries
        ) or '<tr><td colspan="4">אין כרגע משתמשים בבלאקליסט.</td></tr>'
        appeals_html = "".join(
            f"""
            <div class="card stack">
                <div class="price-list">
                    <div class="price-item"><strong>ערעור #{appeal.id}</strong><span>{_status_badge(appeal.status)}</span></div>
                    <div class="price-item"><strong>משתמש</strong><span>{_escape(label)}<br><span class="mono">{appeal.user_id}</span></span></div>
                    <div class="price-item"><strong>סיבת בלאקליסט</strong><span>{_escape(entry_map.get(appeal.user_id).reason if entry_map.get(appeal.user_id) else 'לא נמצאה רשומת בלאקליסט פעילה')}</span></div>
                    <div class="price-item"><strong>נשלח בתאריך</strong><span>{_escape(appeal.submitted_at)}</span></div>
                    <div class="price-item"><strong>פרטי המשתמש</strong><span>{_escape(appeal.answer_one)}</span></div>
                    <div class="price-item"><strong>למה להסיר</strong><span>{_escape(appeal.answer_two)}</span></div>
                </div>
                <div class="actions">
                    <form method="post" class="inline-form"><input type="hidden" name="action" value="accept-appeal"><input type="hidden" name="appeal_id" value="{appeal.id}"><button type="submit">אשר ערעור</button></form>
                    <form method="post" class="inline-form"><input type="hidden" name="action" value="reject-appeal"><input type="hidden" name="appeal_id" value="{appeal.id}"><button type="submit" class="ghost-button danger">דחה ערעור</button></form>
                </div>
            </div>
            """
            for appeal, label in zip(pending_appeals, appeal_labels, strict=False)
        ) or '<div class="empty-card"><p>אין כרגע ערעורים שממתינים לטיפול.</p></div>'
        notice = notice_map.get(str(request.query.get("saved", "")).strip().lower())
        content = f"""
        {_notice_html(notice, success=True)}
        <div class="profile-summary-grid">
            <div class="summary-tile"><strong>{len(entries)}</strong><span>משתמשים בבלאקליסט</span></div>
            <div class="summary-tile"><strong>{len(pending_appeals)}</strong><span>ערעורים ממתינים</span></div>
        </div>
        <div class="split-grid">
            <div class="card stack">
                <h2>הכנסה לבלאקליסט</h2>
                <form method="post">
                    <input type="hidden" name="action" value="add">
                    <div class="grid">
                        <label class="field"><span>מזהה משתמש בדיסקורד</span><input type="number" min="1" name="user_id" required></label>
                        <label class="field field-wide"><span>סיבה</span><textarea name="reason" required></textarea></label>
                    </div>
                    <div class="actions"><button type="submit">הכנס לבלאקליסט</button></div>
                </form>
            </div>
            <div class="card stack">
                <h2>ערעורים ממתינים</h2>
                <p>כל הערעורים שנשלחו דרך האתר או דרך פקודת הערעור מרוכזים כאן לטיפול ישיר מתוך האתר.</p>
                {appeals_html}
            </div>
        </div>
        <div class="table-wrap"><table><thead><tr><th>משתמש</th><th>סיבה</th><th>הוכנס בתאריך</th><th>פעולות</th></tr></thead><tbody>{blacklist_rows}</tbody></table></div>
        """
        body = _admin_shell(
            session,
            current_path=request.path,
            title="בלאקליסט וערעורים",
            intro="ניהול מלא של משתמשי הבלאקליסט ושל ערעורי ההסרה מתוך האתר, בלי מעבר ל-DM של הבוט.",
            content=content,
        )
        return _page_response("בלאקליסט וערעורים", body)
    except web.HTTPException:
        raise
    except SalesBotError as exc:
        return _error_response("בלאקליסט וערעורים", str(exc), status=403)


def _special_system_embed(special_system: SpecialSystemRecord) -> discord.Embed:
    embed = discord.Embed(title=special_system.title, description=special_system.description, color=discord.Color.gold())
    embed.add_field(
        name="אמצעי תשלום",
        value="\n".join(f"• {method.label}: {method.price}" for method in special_system.payment_methods),
        inline=False,
    )
    return embed


def _special_system_files(images: list[SpecialSystemImageRecord]) -> tuple[list[discord.File], str | None]:
    attachments: list[discord.File] = []
    first_image_name: str | None = None
    for image in images:
        attachments.append(discord.File(BytesIO(image.asset_bytes), filename=image.asset_name))
        if first_image_name is None and (image.content_type or "").startswith("image/"):
            first_image_name = image.asset_name
    return attachments, first_image_name


def _gamepass_price_label(gamepass: RobloxGamePassRecord) -> str:
    return f"{gamepass.price_in_robux} Robux" if gamepass.price_in_robux is not None else "לא מתומחר"


async def _linked_system_for_gamepass(bot: "SalesBot", game_pass_id: int) -> SystemRecord | None:
    try:
        return await bot.services.systems.get_system_by_gamepass_id(str(game_pass_id))
    except NotFoundError:
        return None


def _gamepass_embed(
    gamepass: RobloxGamePassRecord,
    linked_system: SystemRecord | None,
    *,
    display_gamepass_name: str | None = None,
) -> discord.Embed:
    embed = discord.Embed(
        title=gamepass.name,
        description=gamepass.description or "אין כרגע תיאור לגיימפאס הזה.",
        color=discord.Color.blurple(),
    )
    embed.add_field(name="מזהה גיימפאס", value=str(gamepass.game_pass_id), inline=True)
    embed.add_field(name="מחיר", value=_gamepass_price_label(gamepass), inline=True)
    embed.add_field(name="למכירה", value="כן" if gamepass.is_for_sale else "לא", inline=True)
    embed.add_field(name="קישור רכישה", value=bot_gamepass_url(gamepass), inline=False)
    embed.add_field(name="מערכת מקושרת", value=linked_system.name if linked_system else "לא מקושר", inline=False)
    if display_gamepass_name:
        embed.add_field(name="שם תצוגה במשחק", value=display_gamepass_name, inline=False)
    return embed


def bot_gamepass_url(gamepass: RobloxGamePassRecord) -> str:
    return f"https://www.roblox.com/game-pass/{gamepass.game_pass_id}"


async def _resolve_gamepass_context(bot: "SalesBot", discord_user_id: int) -> tuple[int, int]:
    if bot.settings.primary_guild_id is None:
        raise ConfigurationError("כדי לנהל גיימפאסים דרך האתר צריך להגדיר PRIMARY_GUILD_ID.")
    link = await bot.services.roblox_creator.get_link(bot.settings.primary_guild_id)
    if link.discord_user_id != discord_user_id:
        raise PermissionDeniedError(
            "כדי לנהל גיימפאסים מהאתר צריך להתחבר עם חשבון דיסקורד שקישר את owner access דרך /linkasowner."
        )
    return bot.settings.primary_guild_id, discord_user_id


async def _owner_order_embed(
    special_system: SpecialSystemRecord,
    order: SpecialOrderRequestRecord,
) -> discord.Embed:
    embed = discord.Embed(title="יש בקשה לקניית מערכת מיוחדת חדשה", color=discord.Color.gold())
    embed.add_field(name="מערכת מיוחדת", value=special_system.title, inline=False)
    embed.add_field(name="משתמש דיסקורד", value=f"<@{order.user_id}>\n{order.discord_name}\n{order.user_id}", inline=False)
    embed.add_field(name="שם רובלוקס שנשלח", value=order.roblox_name, inline=False)
    embed.add_field(name="שיטת תשלום", value=f"{order.payment_method_label} | {order.payment_price}", inline=False)
    linked_label = "לא מחובר"
    if order.linked_roblox_sub:
        parts = [order.linked_roblox_display_name or "", order.linked_roblox_username or "", order.linked_roblox_sub]
        linked_label = " | ".join(part for part in parts if part)
    embed.add_field(name="חשבון רובלוקס מחובר", value=linked_label, inline=False)
    embed.add_field(name="סטטוס", value=ORDER_STATUS_LABELS.get(order.status, order.status), inline=False)
    embed.set_footer(text=f"בקשה #{order.id}")
    return embed


async def _owner_custom_order_embed(bot: "SalesBot", order: OrderRequestRecord) -> discord.Embed:
    requester_label = await _discord_user_label(bot, order.user_id)
    image_count = len(await bot.services.orders.list_request_images(order.id))
    embed = discord.Embed(title="יש הזמנה אישית חדשה", color=discord.Color.gold())
    embed.add_field(name="משתמש דיסקורד", value=f"<@{order.user_id}>\n{requester_label}\n{order.user_id}", inline=False)
    embed.add_field(name="מה אתה רוצה להזמין", value=order.requested_item, inline=False)
    embed.add_field(name="תוך כמה זמן אתה צריך את זה", value=order.required_timeframe, inline=False)
    embed.add_field(name="איך אתה משלם", value=order.payment_method, inline=False)
    embed.add_field(name="כמה אתה מוכן לשלם", value=order.offered_price, inline=False)
    embed.add_field(name="מה השם שלך ברובלוקס", value=order.roblox_username or "לא צוין", inline=False)
    if image_count:
        embed.add_field(name="תמונות שצורפו", value=str(image_count), inline=False)
    embed.add_field(name="סטטוס", value=ORDER_STATUS_LABELS.get(order.status, order.status), inline=False)
    if order.admin_reply:
        note_label = "סיבת דחייה" if order.status == "rejected" else "הודעת אדמין"
        embed.add_field(name=note_label, value=order.admin_reply, inline=False)
    embed.set_footer(text=f"הזמנה #{order.id}")
    return embed


async def _update_owner_order_message(
    bot: "SalesBot",
    special_system: SpecialSystemRecord,
    order: SpecialOrderRequestRecord,
) -> None:
    if order.owner_message_id is None:
        return
    try:
        owner = await bot.fetch_user(bot.settings.owner_user_id)
        owner_dm = owner.dm_channel or await owner.create_dm()
        message = await owner_dm.fetch_message(order.owner_message_id)
        embed = await _owner_order_embed(special_system, order)
        view = discord.ui.View()
        view.add_item(
            discord.ui.Button(
                label="פתח את הבקשה באתר",
                style=discord.ButtonStyle.link,
                url=f"{bot.settings.public_base_url}/admin/special-orders/{order.id}",
            )
        )
        await message.edit(content="עדכון סטטוס לבקשת מערכת מיוחדת", embed=embed, view=view)
    except discord.HTTPException:
        return


async def _update_owner_custom_order_message(bot: "SalesBot", order: OrderRequestRecord) -> None:
    if order.owner_message_id is None:
        return
    try:
        owner = await bot.fetch_user(bot.settings.owner_user_id)
        owner_dm = owner.dm_channel or await owner.create_dm()
        message = await owner_dm.fetch_message(order.owner_message_id)
        embed = await _owner_custom_order_embed(bot, order)
        view = discord.ui.View()
        view.add_item(
            discord.ui.Button(
                label="פתח את ההזמנה באתר",
                style=discord.ButtonStyle.link,
                url=_custom_order_admin_url(bot, order.id),
            )
        )
        await message.edit(content="עדכון סטטוס להזמנה אישית", embed=embed, view=view)
    except discord.HTTPException:
        return


async def _notify_custom_order_requester(
    bot: "SalesBot",
    order: OrderRequestRecord,
    *,
    admin_reply: str | None,
) -> None:
    try:
        requester = await bot.fetch_user(order.user_id)
    except discord.HTTPException:
        return

    if order.status == "accepted":
        message = "ההזמנה האישית שלך התקבלה. הבעלים יחזור אליך בהמשך."
    elif order.status == "rejected":
        message = "ההזמנה האישית שלך נדחתה."
    elif order.status == "completed":
        message = "ההזמנה שלך הושלמה בהצלחה. נשמח מאוד שתשאיר הוכחה באמצעות הפקודה: '/Vouch'. זה יוערך מאוד."
    else:
        return

    if admin_reply:
        if order.status == "rejected":
            message = f"{message}\n\nסיבה: {admin_reply}"
        else:
            message = f"{message}\n\n{admin_reply}"

    try:
        await requester.send(message)
    except discord.HTTPException:
        return


async def _send_custom_order_to_admins(bot: "SalesBot", order: OrderRequestRecord) -> tuple[int, int | None]:
    admin_ids = list(dict.fromkeys(await bot.services.admins.list_admin_ids()))
    image_count = len(await bot.services.orders.list_request_images(order.id))
    delivered_count = 0
    owner_message_id: int | None = None

    for admin_id in admin_ids:
        try:
            await bot.services.notifications.create_notification(
                user_id=admin_id,
                title=f"הזמנה אישית חדשה #{order.id}",
                body=f"נפתחה הזמנה אישית חדשה מ-{order.user_id}. אפשר לפתוח את הפרטים המלאים דרך דף הניהול.",
                link_path=f"/admin/custom-orders/{order.id}",
                kind="admin-custom-order",
            )
            admin_user = bot.get_user(admin_id) or await bot.fetch_user(admin_id)
            admin_dm = admin_user.dm_channel or await admin_user.create_dm()
            embed = await _owner_custom_order_embed(bot, order)
            if image_count:
                embed.description = f"צורפו להזמנה {image_count} תמונות לעיון בדף הניהול."
            view = discord.ui.View()
            view.add_item(
                discord.ui.Button(
                    label="פתח את ההזמנה באתר",
                    style=discord.ButtonStyle.link,
                    url=_custom_order_admin_url(bot, order.id),
                )
            )
            message = await admin_dm.send(content="יש הזמנה אישית חדשה", embed=embed, view=view)
            delivered_count += 1
            if admin_id == bot.settings.owner_user_id:
                owner_message_id = message.id
        except (discord.HTTPException, SalesBotError):
            continue

    return delivered_count, owner_message_id


async def _send_account_payment_submission_to_admins(
    bot: "SalesBot",
    *,
    session: WebsiteSessionRecord,
    roblox_username: str,
    roblox_password: str,
    profile_link: str | None,
    profile_image: tuple[str, bytes, str | None] | None,
    has_email: bool,
    has_phone: bool,
    has_two_factor: bool,
) -> int:
    admin_ids = list(dict.fromkeys(await bot.services.admins.list_admin_ids()))
    sender_label = _session_label(session)
    successful_deliveries = 0

    for admin_id in admin_ids:
        try:
            admin_user = bot.get_user(admin_id) or await bot.fetch_user(admin_id)
            admin_dm = admin_user.dm_channel or await admin_user.create_dm()
        except discord.HTTPException:
            continue

        embed = discord.Embed(title="נשלח משתמש רובלוקס בתור תשלום", color=discord.Color.orange())
        embed.add_field(name="שולח", value=f"{sender_label}\n{session.discord_user_id}", inline=False)
        embed.add_field(name="השם של המשתמש רובלוקס", value=roblox_username, inline=False)
        embed.add_field(name="סיסמא של המשתמש רובלוקס", value=roblox_password, inline=False)
        embed.add_field(name="קישור לפרופיל", value=profile_link or "לא נשלח", inline=False)
        embed.add_field(name="האם יש על המשתמש מייל", value="כן" if has_email else "לא", inline=True)
        embed.add_field(name="האם יש מספר טלפון על המשתמש", value="כן" if has_phone else "לא", inline=True)
        embed.add_field(name="האם יש אימות דו שלבי", value="כן" if has_two_factor else "לא", inline=True)
        embed.set_footer(text="המשתמש אישר שכל הפרטים נכונים ושהחשבון לא יחזור אליו לאחר מכן.")

        send_kwargs: dict[str, Any] = {"embed": embed}
        if profile_link:
            view = discord.ui.View()
            view.add_item(discord.ui.Button(label="פתח את הפרופיל", style=discord.ButtonStyle.link, url=profile_link))
            send_kwargs["view"] = view
        if profile_image is not None:
            image_name, image_bytes, _content_type = profile_image
            safe_name = image_name or "profile-image"
            send_kwargs["file"] = discord.File(BytesIO(image_bytes), filename=safe_name)
            embed.set_image(url=f"attachment://{safe_name}")

        try:
            await admin_dm.send(**send_kwargs)
            successful_deliveries += 1
        except discord.HTTPException:
            continue

    return successful_deliveries


async def _send_special_system_message(bot: "SalesBot", special_system: SpecialSystemRecord) -> discord.Message:
    images = await bot.services.special_systems.list_special_system_images(special_system.id)
    channel = bot.get_channel(special_system.channel_id) or await bot.fetch_channel(special_system.channel_id)
    if not isinstance(channel, discord.TextChannel):
        raise PermissionDeniedError("אפשר לפרסם מערכת מיוחדת רק לערוץ טקסט.")
    embed = _special_system_embed(special_system)
    files, first_image_name = _special_system_files(images)
    if first_image_name:
        embed.set_image(url=f"attachment://{first_image_name}")
    view = discord.ui.View()
    view.add_item(discord.ui.Button(label="קניה מיוחדת", style=discord.ButtonStyle.link, url=_special_system_url(bot, special_system)))
    send_kwargs: dict[str, Any] = {"embed": embed, "view": view}
    if files:
        send_kwargs["files"] = files
    return await channel.send(**send_kwargs)


async def _delete_special_system_message(bot: "SalesBot", special_system: SpecialSystemRecord) -> None:
    if special_system.message_id is None:
        return
    try:
        channel = bot.get_channel(special_system.channel_id) or await bot.fetch_channel(special_system.channel_id)
        if not isinstance(channel, discord.TextChannel):
            return
        message = await channel.fetch_message(special_system.message_id)
        await message.delete()
    except discord.HTTPException:
        return


async def _refresh_special_system_public_message(
    bot: "SalesBot",
    special_system: SpecialSystemRecord,
    *,
    previous_record: SpecialSystemRecord | None = None,
) -> SpecialSystemRecord:
    previous = previous_record or special_system
    if not special_system.is_active:
        await _delete_special_system_message(bot, previous)
        return await bot.services.special_systems.clear_public_message(special_system.id)

    message = await _send_special_system_message(bot, special_system)
    updated_system = await bot.services.special_systems.set_public_message(
        special_system.id,
        channel_id=special_system.channel_id,
        message_id=message.id,
    )
    if previous.message_id is not None and previous.message_id != message.id:
        await _delete_special_system_message(bot, previous)
    return updated_system


async def website_login(request: web.Request) -> web.Response:
    bot: SalesBot = request.app["bot"]
    next_path = request.query.get("next") or "/admin"
    try:
        state = await bot.services.web_auth.create_state(next_path)
        raise web.HTTPFound(bot.services.web_auth.build_authorization_url(state))
    except SalesBotError as exc:
        return _error_response("התחברות לאתר", str(exc), status=400)


async def website_callback(request: web.Request) -> web.Response:
    bot: SalesBot = request.app["bot"]
    state = request.query.get("state", "")
    code = request.query.get("code", "")
    if not state or not code:
        return _error_response("התחברות לאתר", "חסרים פרטי התחברות מהחזרה של דיסקורד.", status=400)
    try:
        next_path = await bot.services.web_auth.consume_state(state)
        tokens = await bot.services.web_auth.exchange_code(bot.http_session, code)
        identity = await bot.services.web_auth.fetch_identity(bot.http_session, str(tokens.get("access_token") or ""))
        session = await bot.services.web_auth.create_session(
            discord_user_id=int(str(identity.get("id") or "0")),
            username=str(identity.get("username") or "").strip(),
            global_name=str(identity.get("global_name") or "").strip() or None,
            avatar_hash=str(identity.get("avatar") or "").strip() or None,
        )
        response = web.HTTPFound(next_path)
        response.set_cookie(
            bot.services.web_auth.cookie_name,
            session.token,
            max_age=24 * 60 * 60,
            httponly=True,
            secure=bot.settings.public_base_url.startswith("https://"),
            samesite="Lax",
            path="/",
        )
        return response
    except SalesBotError as exc:
        return _error_response("התחברות לאתר", str(exc), status=400)


async def website_logout(request: web.Request) -> web.Response:
    bot, session = await _current_site_session(request)
    if session is not None:
        await bot.services.web_auth.delete_session(session.token)
    response = web.HTTPFound("/")
    response.del_cookie(bot.services.web_auth.cookie_name, path="/")
    return response


async def admin_settings_page(request: web.Request) -> web.Response:
    try:
        bot, session = await _require_admin_session(request)
        if request.method == "POST":
            form = await request.post()
            action = str(form.get("action", "")).strip().lower()
            if action != "save-theme":
                raise PermissionDeniedError("הפעולה שנשלחה להגדרות לא תקינה.")
            theme_mode = str(form.get("theme_mode", "default")).strip().lower()
            if theme_mode not in THEME_LABELS:
                raise PermissionDeniedError("מצב התצוגה שנבחר לא תקין.")
            response = web.HTTPFound("/admin/settings?saved=theme")
            _set_theme_cookie(response, theme_mode, secure=bot.settings.public_base_url.startswith("https://"))
            return response

        notice = "ערכת הנושא עודכנה בהצלחה." if request.query.get("saved") == "theme" else None
        theme_mode = _theme_mode_from_request(request)
        avatar_url = _session_avatar(session)
        avatar_html = f'<img class="profile-avatar" src="{_escape(avatar_url)}" alt="avatar">' if avatar_url else '<div class="profile-avatar"></div>'
        try:
            roblox_link = await bot.services.oauth.get_link(session.discord_user_id)
        except SalesBotError:
            roblox_link = None

        if roblox_link is None:
            roblox_profile = '<div class="price-item"><strong>רובלוקס</strong><span>אין חשבון רובלוקס מחובר כרגע.</span></div>'
        else:
            roblox_parts = [part for part in (roblox_link.roblox_display_name, roblox_link.roblox_username, roblox_link.roblox_sub) if part]
            roblox_summary = " | ".join(roblox_parts) if roblox_parts else roblox_link.roblox_sub
            profile_link_html = f'<a href="{_escape(roblox_link.profile_url)}" target="_blank" rel="noreferrer">פתח פרופיל</a>' if roblox_link.profile_url else 'אין קישור פרופיל'
            roblox_profile = f"""
            <div class="price-item"><strong>רובלוקס</strong><span>{_escape(roblox_summary)}</span></div>
            <div class="price-item"><strong>קישור</strong><span>{profile_link_html}</span></div>
            <div class="price-item"><strong>קושר בתאריך</strong><span>{_escape(roblox_link.linked_at)}</span></div>
            """

        content = f"""
        {_notice_html(notice, success=True)}
        <div class="profile-grid">
            <div class="card stack">
                <div class="profile-hero">
                    {avatar_html}
                    <div>
                        <p class="eyebrow">הפרופיל שלך</p>
                        <h2>{_escape(_session_label(session))}</h2>
                        <p class="muted">פרטי דיסקורד, חיבור רובלוקס קיים והדרגה של החשבון שמחובר כרגע לאתר.</p>
                    </div>
                </div>
                <div class="price-list">
                    <div class="price-item"><strong>דיסקורד</strong><span>{_escape(_session_label(session))}</span></div>
                    <div class="price-item"><strong>מזהה משתמש</strong><span class="mono">{_escape(session.discord_user_id)}</span></div>
                    <div class="price-item"><strong>דרגה</strong><span>{_escape(_admin_rank_label(bot, session.discord_user_id))}</span></div>
                    <div class="price-item"><strong>סשן נוצר</strong><span>{_escape(session.created_at)}</span></div>
                    <div class="price-item"><strong>נראה לאחרונה</strong><span>{_escape(session.last_seen_at)}</span></div>
                    {roblox_profile}
                </div>
            </div>
            <div class="card stack">
                <div>
                    <p class="eyebrow">מראה האתר</p>
                    <h2>ערכת נושא</h2>
                    <p class="setting-hint">ברירת מחדל שומרת על הסגנון הנוכחי, כהה מוסיפה יותר ניגודיות, ובהיר מתאים לעבודה ביום.</p>
                </div>
                <form method="post" class="settings-list">
                    <input type="hidden" name="action" value="save-theme">
                    <label class="field"><span>מצב תצוגה</span><select name="theme_mode">{_theme_options(theme_mode)}</select></label>
                    <div class="actions"><button type="submit">שמור העדפה</button></div>
                </form>
                <div class="price-list">
                    <div class="price-item"><strong>ברירת מחדל</strong><span>המראה הרגיל של האתר.</span></div>
                    <div class="price-item"><strong>כהה</strong><span>רקע עמוק יותר וניגודיות חזקה יותר.</span></div>
                    <div class="price-item"><strong>בהיר</strong><span>תצוגה בהירה לקריאה נוחה על מסכים מוארים.</span></div>
                </div>
            </div>
        </div>
        """
        body = _admin_shell(session, current_path=request.path, title="הגדרות", intro="ניהול פרטי החשבון המחובר והעדפת התצוגה של האתר.", content=content)
        return _page_response("הגדרות", body)
    except web.HTTPException:
        raise
    except SalesBotError as exc:
        return _error_response("הגדרות", str(exc), status=400)


async def admin_dashboard_page(request: web.Request) -> web.Response:
    try:
        bot, session = await _require_admin_session(request)
        (
            admin_ids,
            systems,
            blacklist_entries,
            pending_blacklist_appeals,
            pending_custom_orders,
            special_systems,
            rollable_events,
            pending_special_orders,
        ) = await asyncio.gather(
            bot.services.admins.list_admin_ids(),
            bot.services.systems.list_systems(),
            bot.services.blacklist.list_entries(),
            bot.services.blacklist.list_pending_appeals(),
            bot.services.orders.list_requests(statuses=("pending",)),
            bot.services.special_systems.list_special_systems(active_only=True),
            bot.services.events.list_rollable_events(),
            bot.services.special_systems.list_order_requests(statuses=("pending",)),
        )
        stats = f"""
        <div class="stat-grid">
            <div class="card"><h2>אדמינים</h2><div class="stat-value">{len(admin_ids)}</div></div>
            <div class="card"><h2>מערכות</h2><div class="stat-value">{len(systems)}</div></div>
            <div class="card"><h2>בלאקליסט</h2><div class="stat-value">{len(blacklist_entries)}</div></div>
            <div class="card"><h2>ערעורי בלאקליסט</h2><div class="stat-value">{len(pending_blacklist_appeals)}</div></div>
            <div class="card"><h2>הזמנות אישיות ממתינות</h2><div class="stat-value">{len(pending_custom_orders)}</div></div>
            <div class="card"><h2>מערכות מיוחדות</h2><div class="stat-value">{len(special_systems)}</div></div>
            <div class="card"><h2>אירועים פתוחים</h2><div class="stat-value">{len(rollable_events)}</div></div>
            <div class="card"><h2>בקשות ממתינות</h2><div class="stat-value">{len(pending_special_orders)}</div></div>
        </div>
        """
        quick_links = """
        <div class="hero-grid">
            <div class="card"><h3>ניהול אדמינים</h3><p>הוספה והסרה של צוות הניהול מתוך האתר.</p><div class="actions"><a class="link-button" href="/admin/admins">פתח</a></div></div>
            <div class="card"><h3>בלאקליסט וערעורים</h3><p>הכנסה והסרה של משתמשים מהבלאקליסט, יחד עם טיפול מלא בערעורים שנשלחו.</p><div class="actions"><a class="link-button" href="/admin/blacklist">פתח</a></div></div>
            <div class="card"><h3>קופות אתר</h3><p>רשימת כל הזמנות הסל, אישור מסירה, או ביטול עם הודעה חזרה ללקוח.</p><div class="actions"><a class="link-button" href="/admin/checkouts">פתח</a></div></div>
            <div class="card"><h3>הזמנות אישיות</h3><p>רשימת כל ההזמנות האישיות, צפייה בפרטים, אישור, דחייה וסימון כהושלמה.</p><div class="actions"><a class="link-button" href="/admin/custom-orders">פתח</a></div></div>
            <div class="card"><h3>מערכות רגילות</h3><p>יצירת מערכות, עריכה, מחיקה ומתן או הסרה לפי User ID.</p><div class="actions"><a class="link-button" href="/admin/systems">פתח</a></div></div>
            <div class="card"><h3>גיימפאסים</h3><p>יצירה, עדכון, קישור ושליחה של גיימפאסים ישירות מתוך האתר.</p><div class="actions"><a class="link-button" href="/admin/gamepasses">פתח</a></div></div>
            <div class="card"><h3>מערכות מיוחדות</h3><p>פרסום מערכת מיוחדת עם כפתור קניה, תמונות, מחירים ושיטות תשלום.</p><div class="actions"><a class="link-button" href="/admin/special-systems">פתח</a></div></div>
            <div class="card"><h3>בקשות מיוחדות</h3><p>רשימת כל הבקשות, צפייה בפרטים, אישור או דחייה עם הודעה חזרה.</p><div class="actions"><a class="link-button" href="/admin/special-orders">פתח</a></div></div>
            <div class="card"><h3>קודי הנחה</h3><p>יצירת קודים כלליים או קודים למערכת מסוימת, עם מגבלות שימוש ותוקף.</p><div class="actions"><a class="link-button" href="/admin/discount-codes">פתח</a></div></div>
            <div class="card"><h3>קודי מימוש</h3><p>יצירה, כיבוי ומחיקה של קודי מתנה, כולל קודים לקניה בשביל חבר והגרלות.</p><div class="actions"><a class="link-button" href="/admin/redeem-codes">פתח</a></div></div>
            <div class="card"><h3>התראות ללקוחות</h3><p>שליחת התראה ישירות לפי מזהה משתמש בדיסקורד, כולל שמירה במרכז ההתראות באתר.</p><div class="actions"><a class="link-button" href="/admin/notifications">פתח</a></div></div>
            <div class="card"><h3>כלי תוכן קיימים</h3><p>הפאנלים הקיימים של סקרים, הגרלות ואירועים נשארו זמינים גם דרך האתר.</p><div class="actions"><a class="link-button" href="/admin/polls/new">סקרים</a><a class="link-button ghost-button" href="/admin/giveaways/new">הגרלות</a><a class="link-button ghost-button" href="/admin/events/new">אירועים</a></div></div>
            <div class="card"><h3>הגדרות אישיות</h3><p>פרטי החשבון המחובר, הדרגה שלך והעדפת ערכת הנושא של האתר.</p><div class="actions"><a class="link-button" href="/admin/settings">פתח</a></div></div>
        </div>
        """
        config_html = f"""
        <div class="card">
            <h2>סיכום הגדרות ריצה</h2>
            <div class="price-list">
                <div class="price-item"><strong>PUBLIC_BASE_URL</strong><span class="mono">{_escape(bot.settings.public_base_url)}</span></div>
                <div class="price-item"><strong>PRIMARY_GUILD_ID</strong><span class="mono">{_escape(bot.settings.primary_guild_id or 'לא מוגדר')}</span></div>
                <div class="price-item"><strong>OWNER_USER_ID</strong><span class="mono">{_escape(bot.settings.owner_user_id)}</span></div>
                <div class="price-item"><strong>ORDER_CHANNEL_ID</strong><span class="mono">{_escape(bot.settings.order_channel_id)}</span></div>
            </div>
            <p class="muted">הגדרות סביבה עדיין מנוהלות דרך השרת וה-ENV, אבל כל הכלים התפעוליים של הבוט פתוחים מכאן.</p>
        </div>
        """
        body = _admin_shell(session, current_path=request.path, title="לוח ניהול ראשי", intro="כלי האתר מרוכזים כאן. כל דף משתמש באותם שירותים של פקודות הסלאש.", content=stats + quick_links + config_html)
        return _page_response("לוח ניהול", body)
    except web.HTTPException:
        raise
    except SalesBotError as exc:
        return _error_response("לוח ניהול", str(exc), status=403)


async def admin_admins_page(request: web.Request) -> web.Response:
    notice: str | None = None
    success = True
    try:
        bot, session = await _require_admin_session(request)
        if request.method == "POST":
            form = await request.post()
            action = str(form.get("action", "")).strip()
            if action == "add":
                user_id = _parse_positive_int(form.get("user_id"), "User ID")
                assert user_id is not None
                await bot.services.admins.add_admin(user_id, session.discord_user_id)
                notice = "האדמין נוסף בהצלחה."
            elif action == "remove":
                user_id = _parse_positive_int(form.get("user_id"), "User ID")
                assert user_id is not None
                await bot.services.admins.remove_admin(user_id)
                notice = "האדמין הוסר בהצלחה."
        admin_ids = await bot.services.admins.list_admin_ids()
        labels = await asyncio.gather(*(_discord_user_label(bot, user_id) for user_id in admin_ids))
        rows = "\n".join(
            f"""
            <tr>
                <td><strong>{_escape(label)}</strong><br><span class="mono">{user_id}</span></td>
                <td>{'בעלים' if user_id == bot.settings.owner_user_id else 'אדמין'}</td>
                <td>{'' if user_id == bot.settings.owner_user_id else f'<form method="post" class="inline-form"><input type="hidden" name="action" value="remove"><input type="hidden" name="user_id" value="{user_id}"><button type="submit" class="ghost-button danger">הסר</button></form>'}</td>
            </tr>
            """
            for user_id, label in zip(admin_ids, labels, strict=False)
        )
        content = f"""
        {_notice_html(notice, success=success)}
        <div class="split-grid">
            <div class="card">
                <h2>הוספת אדמין</h2>
                <form method="post">
                    <input type="hidden" name="action" value="add">
                    <div class="grid"><label class="field"><span>User ID</span><input type="number" min="1" name="user_id" required></label></div>
                    <div class="actions"><button type="submit">הוסף אדמין</button></div>
                </form>
            </div>
            <div class="card"><h2>הערה</h2><p>בעל הבוט המוגדר ב-ENV נשאר אדמין קבוע ואי אפשר להסיר אותו דרך האתר.</p></div>
        </div>
        <div class="table-wrap"><table><thead><tr><th>משתמש</th><th>סוג</th><th>פעולה</th></tr></thead><tbody>{rows}</tbody></table></div>
        """
        body = _admin_shell(session, current_path=request.path, title="ניהול אדמינים", intro="ניהול רשימת האדמינים של הבוט מתוך האתר.", content=content)
        return _page_response("ניהול אדמינים", body)
    except web.HTTPException:
        raise
    except SalesBotError as exc:
        return _error_response("ניהול אדמינים", str(exc), status=400)


async def admin_checkout_orders_page(request: web.Request) -> web.Response:
    notice: str | None = None
    success = True
    try:
        bot, session = await _require_admin_session(request)
        if request.method == "POST":
            form = await request.post()
            action = str(form.get("action", "")).strip()
            order_id = _parse_positive_int(form.get("order_id"), "מזהה הזמנה")
            assert order_id is not None
            if action == "complete":
                existing_order = await bot.services.payments.get_checkout_order(order_id)
                if existing_order.payment_method == "paypal":
                    raise PermissionDeniedError("הזמנות PayPal מושלמות אוטומטית אחרי אישור התשלום מפייפאל, ולא דרך הכפתור הידני הזה.")
                order = await bot.services.payments.complete_checkout_order(bot, order_id, session.discord_user_id)
                message, dm_sent = await bot.services.payments.send_completed_checkout_notifications(bot, order)
                notice = message + (" נשלחה גם הודעת DM ללקוח." if dm_sent else " נשמרה התראה באתר, אבל DM לא נשלח.")
            elif action == "cancel":
                cancel_reason = str(form.get("cancel_reason", "")).strip() or "ההזמנה בוטלה על ידי צוות האתר."
                order = await bot.services.payments.cancel_checkout_order(order_id, session.discord_user_id, cancel_reason)
                await bot.services.notifications.create_notification(
                    user_id=order.user_id,
                    title=f"הזמנה #{order.id} בוטלה",
                    body=cancel_reason,
                    link_path="/inbox",
                    kind="checkout",
                    created_by=session.discord_user_id,
                )
                dm_sent = await _send_optional_user_dm(
                    bot,
                    user_id=order.user_id,
                    title=f"הזמנה #{order.id} בוטלה",
                    body=cancel_reason,
                    link_path="/inbox",
                )
                notice = f"הזמנה #{order.id} בוטלה." + (" נשלחה גם הודעת DM ללקוח." if dm_sent else " ההתראה נשמרה באתר בלבד.")

        orders = await bot.services.payments.list_checkout_orders(limit=120)
        labels = await asyncio.gather(*(_discord_user_label(bot, order.user_id) for order in orders)) if orders else []
        item_lists_by_order = await bot.services.payments.list_checkout_order_items_for_orders([order.id for order in orders]) if orders else {}
        cards = "".join(
            f'''
            <div class="card stack">
                <div class="price-list">
                    <div class="price-item"><strong>הזמנה #{order.id}</strong><span>{_status_badge(order.status)}</span></div>
                    <div class="price-item"><strong>לקוח</strong><span>{_escape(label)}<br><span class="mono">{order.user_id}</span></span></div>
                    <div class="price-item"><strong>שיטת תשלום</strong><span>{_escape(_checkout_method_label(order.payment_method))}</span></div>
                    <div class="price-item"><strong>מסלול מסירה</strong><span>{_escape(_checkout_fulfillment_label(order.fulfillment_mode))}</span></div>
                    {f'<div class="price-item"><strong>סטטוס PayPal</strong><span>{_escape(_paypal_status_label(order.paypal_status))}</span></div>' if order.payment_method == 'paypal' else ''}
                    {f'<div class="price-item"><strong>PayPal Order ID</strong><span class="mono">{_escape(order.paypal_order_id or "-")}</span></div>' if order.payment_method == 'paypal' else ''}
                    {f'<div class="price-item"><strong>PayPal Capture ID</strong><span class="mono">{_escape(order.paypal_capture_id or "-")}</span></div>' if order.payment_method == 'paypal' else ''}
                    <div class="price-item"><strong>סכום ביניים</strong><span>{_escape(_money_label(order.subtotal_amount, order.currency))}</span></div>
                    <div class="price-item"><strong>הנחה</strong><span>{_escape(_money_label(order.discount_amount, order.currency))}</span></div>
                    <div class="price-item"><strong>סה"כ</strong><span>{_escape(_money_label(order.total_amount, order.currency))}</span></div>
                    <div class="price-item"><strong>קוד הנחה</strong><span>{_escape(order.discount_code_text or 'ללא')}</span></div>
                    <div class="price-item"><strong>נוצרה ב</strong><span>{_escape(order.created_at)}</span></div>
                    {f'<div class="price-item"><strong>הערה</strong><span>{_escape(order.note)}</span></div>' if order.note else ''}
                    {f'<div class="price-item"><strong>סיבת ביטול</strong><span>{_escape(order.cancel_reason or "-")}</span></div>' if order.status == 'cancelled' else ''}
                </div>
                <div>
                    <h3>מערכות בהזמנה</h3>
                    <div class="price-list">{_checkout_items_html(item_lists_by_order.get(order.id, []), order.currency)}</div>
                </div>
                {'' if order.status != 'pending' else (f'<div class="meta-card"><strong>PayPal:</strong> ההזמנה הזאת תושלם אוטומטית אחרי אישור וחזרה מ-PayPal או דרך הוובהוק, ואז תתבצע המסירה או יצירת קודי המימוש לפי המסלול שנבחר.</div><div class="actions"><form method="post" class="inline-form"><input type="hidden" name="action" value="cancel"><input type="hidden" name="order_id" value="{order.id}"><input type="text" name="cancel_reason" placeholder="סיבת ביטול ללקוח"><button type="submit" class="ghost-button danger">בטל הזמנה</button></form></div>' if order.payment_method == 'paypal' else f'<div class="actions"><form method="post" class="inline-form"><input type="hidden" name="action" value="complete"><input type="hidden" name="order_id" value="{order.id}"><button type="submit">סמן כהושלמה</button></form><form method="post" class="inline-form"><input type="hidden" name="action" value="cancel"><input type="hidden" name="order_id" value="{order.id}"><input type="text" name="cancel_reason" placeholder="סיבת ביטול ללקוח"><button type="submit" class="ghost-button danger">בטל הזמנה</button></form></div>')}
            </div>
            '''
            for order, label in zip(orders, labels, strict=False)
        ) or '<div class="empty-card"><p>עדיין אין הזמנות קופה שמורות במערכת.</p></div>'

        content = f'''
        {_notice_html(notice, success=success)}
        <div class="card stack">
            <h2>סקירת קופות אתר</h2>
            <p>העמוד הזה מרכז את כל הזמנות הסל שנפתחו דרך האתר. קופות כרטיס עדיין מטופלות ידנית, קופות PayPal מוצגות כאן עם סטטוס ההזמנה האמיתי מפייפאל, ומסלול המסירה מבהיר אם מדובר במסירה ישירה או ביצירת קודי מימוש.</p>
        </div>
        <div class="stack">{cards}</div>
        '''
        body = _admin_shell(session, current_path=request.path, title="קופות אתר", intro="כל הקופות המרוכזות שמגיעות מהאתר, עם אישור ידני לכרטיס ומעקב אוטומטי אחרי PayPal.", content=content)
        return _page_response("קופות אתר", body)
    except web.HTTPException:
        raise
    except SalesBotError as exc:
        return _error_response("קופות אתר", str(exc), status=400)


async def admin_discount_codes_page(request: web.Request) -> web.Response:
    notice: str | None = None
    success = True
    try:
        bot, session = await _require_admin_session(request)
        systems = await bot.services.systems.list_systems()
        if request.method == "POST":
            form = await request.post()
            action = str(form.get("action", "")).strip()
            if action == "create":
                system_id = _parse_positive_int(form.get("system_id"), "מזהה מערכת", allow_blank=True)
                max_redemptions = _parse_positive_int(form.get("max_redemptions"), "מספר מימושים", allow_blank=True)
                per_user_limit = _parse_positive_int(form.get("per_user_limit"), "מגבלת משתמש")
                assert per_user_limit is not None
                code = await bot.services.discount_codes.create_code(
                    code=str(form.get("code", "")),
                    description=str(form.get("description", "")).strip() or None,
                    discount_type=str(form.get("discount_type", "percent")),
                    amount=str(form.get("amount", "")),
                    currency=str(form.get("currency", "")).strip() or None,
                    system_id=system_id,
                    max_redemptions=max_redemptions,
                    per_user_limit=per_user_limit,
                    expires_at=str(form.get("expires_at", "")).strip() or None,
                    created_by=session.discord_user_id,
                )
                notice = f"קוד ההנחה {code.code} נוצר בהצלחה."
            elif action == "toggle":
                code_id = _parse_positive_int(form.get("code_id"), "מזהה קוד")
                assert code_id is not None
                next_state = str(form.get("next_state", "")).strip().lower() == "true"
                updated = await bot.services.discount_codes.set_active(code_id, next_state)
                notice = f"קוד ההנחה {updated.code} {'הופעל' if updated.is_active else 'הושבת'}."
            elif action == "delete":
                code_id = _parse_positive_int(form.get("code_id"), "מזהה קוד")
                assert code_id is not None
                code = await bot.services.discount_codes.get_code(code_id)
                await bot.services.discount_codes.delete_code(code_id)
                notice = f"קוד ההנחה {code.code} נמחק."

        codes = await bot.services.discount_codes.list_codes()
        system_names = {system.id: system.name for system in systems}
        code_cards = ''.join(
            f'''
            <div class="card stack">
                <div class="price-list">
                    <div class="price-item"><strong>{_escape(code.code)}</strong><span>{'<span class="catalog-badge">פעיל</span>' if code.is_active else '<span class="catalog-badge warn">מושבת</span>'}</span></div>
                    <div class="price-item"><strong>סוג</strong><span>{_escape('אחוזים' if code.discount_type == 'percent' else 'סכום קבוע')}</span></div>
                    <div class="price-item"><strong>ערך</strong><span>{_escape(code.amount + ('%' if code.discount_type == 'percent' else f' {code.currency or "USD"}'))}</span></div>
                    <div class="price-item"><strong>מוגבל למערכת</strong><span>{_escape(system_names.get(code.system_id, 'כל המערכות'))}</span></div>
                    <div class="price-item"><strong>שימושים כוללים</strong><span>{_escape(code.max_redemptions or 'ללא הגבלה')}</span></div>
                    <div class="price-item"><strong>שימושים למשתמש</strong><span>{code.per_user_limit}</span></div>
                    <div class="price-item"><strong>תפוגה</strong><span>{_escape(code.expires_at or 'ללא')}</span></div>
                </div>
                {f'<p class="muted">{_escape(code.description)}</p>' if code.description else ''}
                <div class="actions"><form method="post" class="inline-form"><input type="hidden" name="action" value="toggle"><input type="hidden" name="code_id" value="{code.id}"><input type="hidden" name="next_state" value="{'false' if code.is_active else 'true'}"><button type="submit" class="ghost-button">{'השבת' if code.is_active else 'הפעל'}</button></form><form method="post" class="inline-form"><input type="hidden" name="action" value="delete"><input type="hidden" name="code_id" value="{code.id}"><button type="submit" class="ghost-button danger">מחק</button></form></div>
            </div>
            '''
            for code in codes
        ) or '<div class="empty-card"><p>עדיין לא נוצרו קודי הנחה.</p></div>'

        content = f'''
        {_notice_html(notice, success=success)}
        <div class="split-grid">
            <div class="card stack">
                <h2>יצירת קוד חדש</h2>
                <form method="post">
                    <input type="hidden" name="action" value="create">
                    <div class="grid">
                        <label class="field"><span>קוד</span><input type="text" name="code" maxlength="32" required></label>
                        <label class="field"><span>סוג</span><select name="discount_type"><option value="percent">אחוזים</option><option value="fixed">סכום קבוע</option></select></label>
                        <label class="field"><span>ערך</span><input type="text" name="amount" inputmode="decimal" required></label>
                        <label class="field"><span>מטבע</span><select name="currency">{_website_currency_options("ILS")}</select></label>
                        <label class="field field-wide"><span>תיאור</span><textarea name="description"></textarea></label>
                        <label class="field"><span>מערכת ספציפית</span><select name="system_id">{_system_options(systems, None)}</select></label>
                        <label class="field"><span>מגבלת שימוש כוללת</span><input type="number" min="1" name="max_redemptions"></label>
                        <label class="field"><span>מגבלת שימוש למשתמש</span><input type="number" min="1" name="per_user_limit" value="1" required></label>
                        <label class="field"><span>תפוגה</span><input type="datetime-local" name="expires_at"></label>
                    </div>
                    <div class="actions"><button type="submit">צור קוד</button></div>
                </form>
            </div>
            <div class="card stack">
                <h2>איך זה עובד</h2>
                <p>קוד Percent מוריד אחוז מהפריטים הרלוונטיים בקופה. קוד Fixed מוריד סכום קבוע במטבע שתגדיר. אם בוחרים מערכת ספציפית, הקוד יופעל רק כשהיא נמצאת בעגלה.</p>
            </div>
        </div>
        <div class="stack">{code_cards}</div>
        '''
        body = _admin_shell(session, current_path=request.path, title="קודי הנחה", intro="יצירה וניהול של קודי הנחה שניתנים למימוש בקופה החדשה של האתר.", content=content)
        return _page_response("קודי הנחה", body)
    except web.HTTPException:
        raise
    except SalesBotError as exc:
        return _error_response("קודי הנחה", str(exc), status=400)


async def admin_redeem_codes_page(request: web.Request) -> web.Response:
    notice: str | None = None
    success = True
    bot, session = await _require_admin_session(request)
    systems = await bot.services.systems.list_systems()

    if request.method == "POST":
        form = await request.post()
        action = str(form.get("action", "")).strip().lower()
        try:
            if action == "create":
                system_id = _parse_positive_int(form.get("system_id"), "מזהה מערכת", allow_blank=True)
                max_redemptions = _parse_positive_int(form.get("max_redemptions"), "מספר מימושים", allow_blank=True) or 1
                code = await bot.services.redeem_codes.create_code(
                    code=str(form.get("code", "")).strip() or None,
                    system_id=system_id,
                    max_redemptions=max_redemptions,
                    expires_at=str(form.get("expires_at", "")).strip() or None,
                    created_by=session.discord_user_id,
                    issued_to_user_id=None,
                    checkout_order_id=None,
                    source=bot.services.redeem_codes.ADMIN_SOURCE,
                )
                notice = f"קוד המימוש {code.code} נוצר בהצלחה."
            elif action == "toggle":
                code_id = _parse_positive_int(form.get("code_id"), "מזהה קוד")
                assert code_id is not None
                next_state = str(form.get("next_state", "")).strip().lower() == "true"
                updated = await bot.services.redeem_codes.set_active(code_id, next_state)
                notice = f"קוד המימוש {updated.code} {'הופעל' if updated.is_active else 'הושבת'}."
            elif action == "delete":
                code_id = _parse_positive_int(form.get("code_id"), "מזהה קוד")
                assert code_id is not None
                code = await bot.services.redeem_codes.get_code(code_id)
                await bot.services.redeem_codes.delete_code(code_id)
                notice = f"קוד המימוש {code.code} נמחק."
        except SalesBotError as exc:
            notice = str(exc)
            success = False

    codes = await bot.services.redeem_codes.list_codes(limit=200)
    system_names = {system.id: system.name for system in systems}
    issued_user_ids = sorted({code.issued_to_user_id for code in codes if code.issued_to_user_id is not None})
    issued_user_labels = {}
    if issued_user_ids:
        labels = await asyncio.gather(*(_discord_user_label(bot, user_id) for user_id in issued_user_ids))
        issued_user_labels = dict(zip(issued_user_ids, labels, strict=False))

    code_cards = ''.join(
        f'''
        <div class="card stack">
            <div class="price-list">
                <div class="price-item"><strong>{_escape(code.code)}</strong><span>{_redeem_code_state_label(code)}</span></div>
                <div class="price-item"><strong>מערכת</strong><span>{_escape(system_names.get(code.system_id or -1, 'לא מחובר למערכת'))}</span></div>
                <div class="price-item"><strong>מקור</strong><span>{_escape(_redeem_code_source_label(code.source))}</span></div>
                <div class="price-item"><strong>שויך ללקוח</strong><span>{_escape(issued_user_labels.get(code.issued_to_user_id or 0, 'ללא שיוך'))}</span></div>
                <div class="price-item"><strong>שימושים</strong><span>{code.redeemed_count}/{code.max_redemptions}</span></div>
                <div class="price-item"><strong>תפוגה</strong><span>{_escape(code.expires_at or 'ללא')}</span></div>
                {f'<div class="price-item"><strong>הזמנת מקור</strong><span>#{code.checkout_order_id}</span></div>' if code.checkout_order_id is not None else ''}
            </div>
            <div class="actions"><form method="post" class="inline-form"><input type="hidden" name="action" value="toggle"><input type="hidden" name="code_id" value="{code.id}"><input type="hidden" name="next_state" value="{'false' if code.is_active else 'true'}"><button type="submit" class="ghost-button">{'השבת' if code.is_active else 'הפעל'}</button></form><form method="post" class="inline-form"><input type="hidden" name="action" value="delete"><input type="hidden" name="code_id" value="{code.id}"><button type="submit" class="ghost-button danger">מחק</button></form></div>
        </div>
        '''
        for code in codes
    ) or '<div class="empty-card"><p>עדיין לא נוצרו קודי מימוש.</p></div>'

    content = f'''
    {_notice_html(notice, success=success)}
    <div class="split-grid">
        <div class="card stack">
            <h2>יצירת קוד חדש</h2>
            <form method="post">
                <input type="hidden" name="action" value="create">
                <div class="grid">
                    <label class="field"><span>קוד</span><input type="text" name="code" maxlength="64" placeholder="השאר ריק ליצירה אוטומטית"></label>
                    <label class="field"><span>מערכת ספציפית</span><select name="system_id">{_system_options(systems, None)}</select></label>
                    <label class="field"><span>מקסימום מימושים</span><input type="number" min="1" name="max_redemptions" value="1"></label>
                    <label class="field"><span>תפוגה</span><input type="datetime-local" name="expires_at"></label>
                </div>
                <div class="actions"><button type="submit">צור קוד</button></div>
            </form>
        </div>
        <div class="card stack">
            <h2>מה אפשר לעשות כאן</h2>
            <p>קוד שמחובר למערכת אחת מתאים למתנות, פיצויים או הגרלות. אם משאירים את שדה הקוד ריק, המערכת תייצר קוד אוטומטי בסגנון המתאים.</p>
            <p>אפשר גם ליצור קוד עם כמה מימושים, לקבוע תפוגה, או להשבית קוד קיים בלי למחוק אותו. קודים שנוצרים אוטומטית מקופות "לקנות בשביל חבר" יופיעו כאן גם כן.</p>
        </div>
    </div>
    <div class="stack">{code_cards}</div>
    '''
    body = _admin_shell(
        session,
        current_path=request.path,
        title="קודי מימוש",
        intro="יצירה וניהול של קודי מתנה, קודי giveaway וקודי רכישה בשביל חבר.",
        content=content,
    )
    return _page_response("קודי מימוש", body)


async def admin_notifications_page(request: web.Request) -> web.Response:
    notice: str | None = None
    success = True
    try:
        bot, session = await _require_admin_session(request)
        if request.method == "POST":
            form = await request.post()
            action = str(form.get("action", "")).strip()
            if action == "send":
                user_id = _parse_positive_int(form.get("user_id"), "מזהה משתמש בדיסקורד")
                assert user_id is not None
                title = str(form.get("title", "")).strip()
                body_text = str(form.get("body", "")).strip()
                link_path = str(form.get("link_path", "")).strip() or None
                if link_path and not link_path.startswith("/"):
                    raise PermissionDeniedError("אם מצרפים קישור פנימי, הוא חייב להתחיל ב-/.")
                notification = await bot.services.notifications.create_notification(
                    user_id=user_id,
                    title=title,
                    body=body_text,
                    link_path=link_path,
                    kind="admin",
                    created_by=session.discord_user_id,
                )
                dm_sent = await _send_optional_user_dm(
                    bot,
                    user_id=user_id,
                    title=notification.title,
                    body=notification.body,
                    link_path=notification.link_path,
                    message_override="יש לך הודעה חדשה באתר. לך לבדוק אותה https://magicshubbot.onrender.com/inbox",
                )
                label = await _discord_user_label(bot, user_id)
                notice = f"ההתראה נשלחה אל {label}." + (" נשלח גם DM." if dm_sent else " נשמרה רק במרכז ההתראות באתר.")

        recent_notifications = await bot.services.notifications.list_recent_notifications(limit=80)
        labels = await asyncio.gather(*(_discord_user_label(bot, record.user_id) for record in recent_notifications)) if recent_notifications else []
        history_html = ''.join(
            f'''
            <div class="card stack">
                <div class="price-list">
                    <div class="price-item"><strong>{_escape(record.title)}</strong><span>{_escape(label)}</span></div>
                    <div class="price-item"><strong>סוג</strong><span>{_escape(record.kind)}</span></div>
                    <div class="price-item"><strong>נשלח ב</strong><span>{_escape(record.created_at)}</span></div>
                    <div class="price-item"><strong>סטטוס קריאה</strong><span>{'נקראה' if record.is_read else 'לא נקראה'}</span></div>
                    {f'<div class="price-item"><strong>קישור</strong><span>{_escape(record.link_path)}</span></div>' if record.link_path else ''}
                </div>
                <p>{_escape(record.body)}</p>
            </div>
            '''
            for record, label in zip(recent_notifications, labels, strict=False)
        ) or '<div class="empty-card"><p>עדיין אין התראות שנשלחו דרך האתר.</p></div>'

        content = f'''
        {_notice_html(notice, success=success)}
        <div class="split-grid">
            <div class="card stack">
                <h2>שליחת התראה לפי מזהה משתמש בדיסקורד</h2>
                <form method="post">
                    <input type="hidden" name="action" value="send">
                    <div class="grid">
                        <label class="field"><span>מזהה משתמש בדיסקורד</span><input type="number" min="1" name="user_id" required></label>
                        <label class="field field-wide"><span>כותרת</span><input type="text" name="title" maxlength="180" required></label>
                        <label class="field field-wide"><span>הודעה</span><textarea name="body" required></textarea></label>
                        <label class="field field-wide"><span>קישור פנימי אופציונלי</span><input type="text" name="link_path" placeholder="/inbox"></label>
                    </div>
                    <div class="actions"><button type="submit">שלח התראה</button></div>
                </form>
            </div>
            <div class="card stack">
                <h2>מה המשתמש רואה</h2>
                <p>כל התראה שנשלחת כאן נשמרת במרכז ההתראות באתר של המשתמש. אם ה-DM פתוח, תישלח בדיסקורד רק הודעה קצרה עם הטקסט "יש לך הודעה חדשה" וקישור ישיר לעמוד הרלוונטי.</p>
            </div>
        </div>
        <div class="stack">{history_html}</div>
        '''
        body = _admin_shell(session, current_path=request.path, title="התראות ללקוחות", intro="שליחת הודעות יזומות ללקוחות דרך מזהה משתמש בדיסקורד, יחד עם שמירה קבועה במרכז ההתראות באתר.", content=content)
        return _page_response("התראות ללקוחות", body)
    except web.HTTPException:
        raise
    except SalesBotError as exc:
        return _error_response("התראות ללקוחות", str(exc), status=400)


async def admin_systems_page(request: web.Request) -> web.Response:
    notice: str | None = None
    success = True
    try:
        bot, session = await _require_admin_session(request)
        if request.method == "POST":
            form = await request.post()
            action = str(form.get("action", "")).strip()
            if action == "create":
                file_upload = _extract_file_upload(form.get("file"))
                if file_upload is None:
                    raise PermissionDeniedError("חובה להעלות קובץ מערכת ראשי.")
                image_uploads = [
                    image
                    for image in (
                        _extract_file_upload(field, image_only=True)
                        for field in form.getall("images", [])
                    )
                    if image is not None
                ]
                created_system = await bot.services.systems.create_system_from_uploads(
                    name=str(form.get("name", "")),
                    description=str(form.get("description", "")),
                    file_upload=(file_upload[0], file_upload[1]),
                    image_uploads=image_uploads or None,
                    created_by=session.discord_user_id,
                    paypal_link=str(form.get("paypal_link", "")).strip() or None,
                    roblox_gamepass_reference=str(form.get("roblox_gamepass", "")).strip() or None,
                    website_price=str(form.get("website_price", "")).strip() or None,
                    website_currency=str(form.get("website_currency", "ILS")).strip() or "ILS",
                    is_visible_on_website=str(form.get("is_visible_on_website", "")).strip().lower() in {"1", "true", "yes", "on"},
                    is_for_sale=str(form.get("is_for_sale", "")).strip().lower() in {"1", "true", "yes", "on"},
                    is_in_stock=str(form.get("is_in_stock", "")).strip().lower() in {"1", "true", "yes", "on"},
                    is_special_system=str(form.get("is_special_system", "")).strip().lower() in {"1", "true", "yes", "on"},
                )
                notice = f"המערכת {created_system.name} נוצרה בהצלחה."
            elif action == "delete":
                system_id = _parse_positive_int(form.get("system_id"), "מזהה מערכת")
                assert system_id is not None
                deleted = await bot.services.systems.delete_system(system_id)
                notice = f"המערכת {deleted.name} נמחקה."
            elif action == "grant":
                system_id = _parse_positive_int(form.get("system_id"), "מזהה מערכת")
                user_id = _parse_positive_int(form.get("user_id"), "מזהה משתמש בדיסקורד")
                assert system_id is not None and user_id is not None
                system = await bot.services.systems.get_system(system_id)
                user = bot.get_user(user_id) or await bot.fetch_user(user_id)
                await bot.services.delivery.deliver_system(bot, user, system, source="grant", granted_by=session.discord_user_id)
                notice = f"המערכת {system.name} נשלחה למשתמש {user_id}."
            elif action == "revoke":
                system_id = _parse_positive_int(form.get("system_id"), "מזהה מערכת")
                user_id = _parse_positive_int(form.get("user_id"), "מזהה משתמש בדיסקורד")
                assert system_id is not None and user_id is not None
                system = await bot.services.systems.get_system(system_id)
                await bot.services.ownership.revoke_system(user_id, system_id)
                deleted_messages = await bot.services.delivery.purge_deliveries(bot, user_id=user_id, system_id=system_id)
                await bot.services.ownership.refresh_claim_role_membership(bot, user_id, sync_ownerships=False)
                notice = f"המערכת {system.name} הוסרה מ-{user_id}. נמחקו {deleted_messages} הודעות DM ישנות."
        systems = await bot.services.systems.list_systems()
        system_rows = "\n".join(
            f"""
            <tr>
                <td><strong>{_escape(system.name)}</strong><br><span class="muted">{_escape(system.description[:120])}</span></td>
                <td>{_escape(system.paypal_link or 'לא מוגדר')}</td>
                <td>{_escape(system.roblox_gamepass_id or 'לא מוגדר')}</td>
                <td>{_escape(f"{system.website_price} {system.website_currency}" if system.website_price else 'לא מוגדר')}</td>
                <td>{'כן' if system.is_visible_on_website else 'לא'}</td>
                <td>{'כן' if system.is_for_sale else 'לא'}</td>
                <td>{'כן' if system.is_in_stock else 'לא'}</td>
                <td>{'כן' if system.is_special_system else 'לא'}</td>
                <td>
                    <div class="actions">
                        <a class="link-button ghost-button" href="/admin/systems/{system.id}/edit">עריכה</a>
                        <form method="post" class="inline-form"><input type="hidden" name="action" value="delete"><input type="hidden" name="system_id" value="{system.id}"><button type="submit" class="ghost-button danger">מחיקה</button></form>
                    </div>
                </td>
            </tr>
            """
            for system in systems
        )
        content = f"""
        {_notice_html(notice, success=success)}
        <div class="split-grid">
            <div class="card">
                <h2>יצירת מערכת חדשה</h2>
                <form method="post" enctype="multipart/form-data">
                    <input type="hidden" name="action" value="create">
                    <div class="grid">
                        <label class="field field-wide"><span>שם</span><input type="text" name="name" required></label>
                        <label class="field field-wide"><span>תיאור</span><textarea name="description" required></textarea></label>
                        <label class="field"><span>קישור פייפאל ישיר (ישן, אופציונלי)</span><input type="url" name="paypal_link"></label>
                        <label class="field"><span>גיימפאס רובקס</span><input type="text" name="roblox_gamepass"></label>
                        <label class="field"><span>מחיר באתר / לקופת PayPal</span><input type="text" name="website_price" inputmode="decimal" placeholder="19.99"></label>
                        <label class="field"><span>מטבע</span><select name="website_currency">{_website_currency_options("ILS")}</select></label>
                        <label class="field"><span>קובץ מערכת</span><input type="file" name="file" required></label>
                        <label class="field"><span>תמונות</span><input type="file" name="images" accept="image/*" multiple></label>
                        <label class="meta-card check-card"><span class="check-line"><input type="checkbox" name="is_visible_on_website" value="true" checked><strong>להציג את המערכת באתר</strong></span></label>
                        <label class="meta-card check-card"><span class="check-line"><input type="checkbox" name="is_for_sale" value="true" checked><strong>להציג את המערכת למכירה</strong></span></label>
                        <label class="meta-card check-card"><span class="check-line"><input type="checkbox" name="is_in_stock" value="true" checked><strong>המערכת במלאי</strong></span></label>
                        <label class="meta-card check-card"><span class="check-line"><input type="checkbox" name="is_special_system" value="true"><strong>מערכת מיוחדת</strong></span></label>
                    </div>
                    <div class="actions"><button type="submit">צור מערכת</button></div>
                </form>
            </div>
            <div class="card stack">
                <div>
                    <h2>מתן מערכת לפי User ID</h2>
                    <form method="post">
                        <input type="hidden" name="action" value="grant">
                        <div class="grid">
                            <label class="field"><span>User ID</span><input type="number" min="1" name="user_id" required></label>
                            <label class="field"><span>מערכת</span><select name="system_id" required>{_system_options(systems, None)}</select></label>
                        </div>
                        <div class="actions"><button type="submit">שלח מערכת</button></div>
                    </form>
                </div>
                <div>
                    <h2>הסרת מערכת לפי User ID</h2>
                    <form method="post">
                        <input type="hidden" name="action" value="revoke">
                        <div class="grid">
                            <label class="field"><span>User ID</span><input type="number" min="1" name="user_id" required></label>
                            <label class="field"><span>מערכת</span><select name="system_id" required>{_system_options(systems, None)}</select></label>
                        </div>
                        <div class="actions"><button type="submit" class="ghost-button danger">הסר בעלות</button></div>
                    </form>
                </div>
            </div>
        </div>
        <div class="table-wrap"><table><thead><tr><th>מערכת</th><th>פייפאל</th><th>גיימפאס</th><th>מחיר</th><th>מוצג באתר</th><th>למכירה</th><th>במלאי</th><th>מיוחדת</th><th>פעולות</th></tr></thead><tbody>{system_rows}</tbody></table></div>
        """
        body = _admin_shell(session, current_path=request.path, title="ניהול מערכות", intro="יצירה, עריכה, מחיקה ומתן/הסרה של מערכות דרך האתר.", content=content)
        return _page_response("ניהול מערכות", body)
    except web.HTTPException:
        raise
    except SalesBotError as exc:
        return _error_response("ניהול מערכות", str(exc), status=400)


async def admin_gamepasses_page(request: web.Request) -> web.Response:
    notice: str | None = None
    success = True
    try:
        bot, session = await _require_admin_session(request)
        guild_id, discord_user_id = await _resolve_gamepass_context(bot, session.discord_user_id)
        systems = await bot.services.systems.list_systems()
        channels = await _list_text_channels(bot)
        if request.method == "POST":
            form = await request.post()
            action = str(form.get("action", "")).strip()
            if action == "create":
                price = _parse_positive_int(form.get("price"), "מחיר")
                assert price is not None
                image_upload = _extract_file_upload(form.get("image"), image_only=True)
                selected_system_id = _parse_positive_int(form.get("system_id"), "מערכת", allow_blank=True)
                created_gamepass = await bot.services.roblox_creator.create_gamepass(
                    bot,
                    guild_id,
                    discord_user_id,
                    name=str(form.get("name", "")),
                    description=str(form.get("description", "")).strip() or None,
                    price=price,
                    is_for_sale=str(form.get("for_sale", "")).lower() in {"1", "true", "yes", "on"},
                    is_regional_pricing_enabled=str(form.get("regional_pricing", "")).lower() in {"1", "true", "yes", "on"},
                    image_upload=image_upload,
                )
                if str(form.get("display_gamepass_name", "")).strip():
                    await bot.services.systems.set_gamepass_display_name(str(created_gamepass.game_pass_id), str(form.get("display_gamepass_name", "")).strip())
                if selected_system_id is not None:
                    await bot.services.systems.set_system_gamepass(selected_system_id, str(created_gamepass.game_pass_id))
                notice = f"הגיימפאס {created_gamepass.name} נוצר בהצלחה."
            elif action == "update":
                gamepass_id = _parse_positive_int(form.get("gamepass_id"), "מזהה גיימפאס")
                assert gamepass_id is not None
                image_upload = _extract_file_upload(form.get("image"), image_only=True)
                price = _parse_positive_int(form.get("price"), "מחיר", allow_blank=True)
                for_sale = _parse_optional_bool(form.get("for_sale_state"))
                regional_pricing = _parse_optional_bool(form.get("regional_pricing_state"))
                name = str(form.get("name", "")).strip() or None
                description = str(form.get("description", "")).strip() or None
                display_name = str(form.get("display_gamepass_name", "")).strip()
                clear_display_name = str(form.get("clear_display_gamepass_name", "")).lower() in {"1", "true", "yes", "on"}
                if any(value is not None for value in (name, description, price, for_sale, regional_pricing)) or image_upload is not None:
                    await bot.services.roblox_creator.update_gamepass(
                        bot,
                        guild_id,
                        discord_user_id,
                        game_pass_id=gamepass_id,
                        name=name,
                        description=description,
                        price=price,
                        is_for_sale=for_sale,
                        is_regional_pricing_enabled=regional_pricing,
                        image_upload=image_upload,
                    )
                if clear_display_name:
                    await bot.services.systems.set_gamepass_display_name(str(gamepass_id), None)
                elif display_name:
                    await bot.services.systems.set_gamepass_display_name(str(gamepass_id), display_name)
                elif not any(value is not None for value in (name, description, price, for_sale, regional_pricing)) and image_upload is None:
                    raise PermissionDeniedError("לא נשלח אף שדה לעדכון.")
                notice = f"הגיימפאס {gamepass_id} עודכן."
            elif action == "connect":
                gamepass_id = _parse_positive_int(form.get("gamepass_id"), "מזהה גיימפאס")
                system_id = _parse_positive_int(form.get("system_id"), "מערכת")
                assert gamepass_id is not None and system_id is not None
                await bot.services.roblox_creator.get_gamepass(bot, guild_id, discord_user_id, gamepass_id)
                await bot.services.systems.set_system_gamepass(system_id, str(gamepass_id))
                notice = f"הגיימפאס {gamepass_id} קושר למערכת שנבחרה."
            elif action == "send":
                gamepass_id = _parse_positive_int(form.get("gamepass_id"), "מזהה גיימפאס")
                channel_id = _parse_positive_int(form.get("channel_id"), "ערוץ")
                assert gamepass_id is not None and channel_id is not None
                gamepass_record = await bot.services.roblox_creator.get_gamepass(bot, guild_id, discord_user_id, gamepass_id)
                if not gamepass_record.is_for_sale:
                    raise ExternalServiceError("הגיימפאס הזה לא מוגדר כרגע למכירה.")
                linked_system = await _linked_system_for_gamepass(bot, gamepass_record.game_pass_id)
                if linked_system is None:
                    raise NotFoundError("צריך קודם לקשר את הגיימפאס למערכת.")
                channel = bot.get_channel(channel_id) or await bot.fetch_channel(channel_id)
                if not isinstance(channel, discord.TextChannel):
                    raise PermissionDeniedError("אפשר לשלוח את הודעת הגיימפאס רק לערוץ טקסט.")
                embed = _gamepass_embed(gamepass_record, linked_system)
                embed.title = f"קניית {linked_system.name}"
                embed.description = f"קנו את **{linked_system.name}** דרך הגיימפאס הזה.\n\nמחיר: **{_gamepass_price_label(gamepass_record)}**"
                view = discord.ui.View()
                view.add_item(discord.ui.Button(label="קניה ברובלוקס", style=discord.ButtonStyle.link, url=bot.services.roblox_creator.gamepass_url(gamepass_record.game_pass_id)))
                await channel.send(embed=embed, view=view)
                notice = f"הגיימפאס {gamepass_record.name} פורסם בערוץ שנבחר."
        gamepasses = await bot.services.roblox_creator.list_gamepasses(bot, guild_id, discord_user_id)
        gamepass_rows: list[str] = []
        for gamepass in gamepasses[:50]:
            linked_system = await _linked_system_for_gamepass(bot, gamepass.game_pass_id)
            display_name = await bot.services.systems.get_gamepass_display_name(str(gamepass.game_pass_id))
            gamepass_rows.append(f"<tr><td><strong>{_escape(gamepass.name)}</strong><br><span class='mono'>{gamepass.game_pass_id}</span></td><td>{_escape(_gamepass_price_label(gamepass))}</td><td>{'כן' if gamepass.is_for_sale else 'לא'}</td><td>{_escape(linked_system.name if linked_system else 'לא מקושר')}</td><td>{_escape(display_name or 'לא מוגדר')}</td></tr>")
        content = f"""
        {_notice_html(notice, success=success)}
        <div class="split-grid">
            <div class="card">
                <h2>יצירת גיימפאס חדש</h2>
                <form method="post" enctype="multipart/form-data">
                    <input type="hidden" name="action" value="create">
                    <div class="grid">
                        <label class="field field-wide"><span>שם</span><input type="text" name="name" required></label>
                        <label class="field"><span>מחיר</span><input type="number" min="1" name="price" required></label>
                        <label class="field"><span>שם תצוגה במשחק</span><input type="text" name="display_gamepass_name"></label>
                        <label class="field field-wide"><span>תיאור</span><textarea name="description"></textarea></label>
                        <label class="field"><span>קישור למערכת</span><select name="system_id">{_system_options(systems, None)}</select></label>
                        <label class="field"><span>תמונה</span><input type="file" name="image" accept="image/*"></label>
                        <label class="field"><span><input type="checkbox" name="for_sale" value="true" checked> למכירה מיד</span></label>
                        <label class="field"><span><input type="checkbox" name="regional_pricing" value="true" checked> תמחור אזורי</span></label>
                    </div>
                    <div class="actions"><button type="submit">צור גיימפאס</button></div>
                </form>
            </div>
            <div class="card stack">
                <div>
                    <h2>עדכון גיימפאס</h2>
                    <form method="post" enctype="multipart/form-data">
                        <input type="hidden" name="action" value="update">
                        <div class="grid">
                            <label class="field field-wide"><span>גיימפאס</span><select name="gamepass_id" required>{_gamepass_options(gamepasses, None)}</select></label>
                            <label class="field"><span>שם חדש</span><input type="text" name="name"></label>
                            <label class="field"><span>מחיר חדש</span><input type="number" min="1" name="price"></label>
                            <label class="field"><span>שם תצוגה במשחק</span><input type="text" name="display_gamepass_name"></label>
                            <label class="field"><span><input type="checkbox" name="clear_display_gamepass_name" value="true"> נקה שם תצוגה</span></label>
                            <label class="field"><span>למכירה</span><select name="for_sale_state">{_bool_options()}</select></label>
                            <label class="field"><span>תמחור אזורי</span><select name="regional_pricing_state">{_bool_options()}</select></label>
                            <label class="field field-wide"><span>תיאור</span><textarea name="description"></textarea></label>
                            <label class="field"><span>תמונה</span><input type="file" name="image" accept="image/*"></label>
                        </div>
                        <div class="actions"><button type="submit">עדכן גיימפאס</button></div>
                    </form>
                </div>
                <div>
                    <h2>קישור או שליחה</h2>
                    <form method="post" class="stack"><input type="hidden" name="action" value="connect"><div class="grid"><label class="field"><span>גיימפאס</span><select name="gamepass_id" required>{_gamepass_options(gamepasses, None)}</select></label><label class="field"><span>מערכת</span><select name="system_id" required>{_system_options(systems, None)}</select></label></div><div class="actions"><button type="submit">קשר למערכת</button></div></form>
                    <form method="post" class="stack"><input type="hidden" name="action" value="send"><div class="grid"><label class="field"><span>גיימפאס</span><select name="gamepass_id" required>{_gamepass_options(gamepasses, None)}</select></label><label class="field"><span>ערוץ</span><select name="channel_id" required>{_render_channel_options(channels, None)}</select></label></div><div class="actions"><button type="submit">שלח לערוץ</button></div></form>
                </div>
            </div>
        </div>
        <div class="table-wrap"><table><thead><tr><th>גיימפאס</th><th>מחיר</th><th>למכירה</th><th>מערכת</th><th>שם תצוגה</th></tr></thead><tbody>{''.join(gamepass_rows)}</tbody></table></div>
        """
        body = _admin_shell(session, current_path=request.path, title="ניהול גיימפאסים", intro="אותם כלים של owner gamepass commands, עכשיו דרך האתר.", content=content)
        return _page_response("ניהול גיימפאסים", body)
    except web.HTTPException:
        raise
    except SalesBotError as exc:
        return _error_response("ניהול גיימפאסים", str(exc), status=400)


async def special_system_compose_page(request: web.Request) -> web.Response:
    notice: str | None = None
    success = True
    form_title = request.query.get("title", "")
    form_description = ""
    selected_payment_methods: set[str] = set()
    price_values: dict[str, str] = {}
    selected_channel_id: int | None = None
    try:
        bot, session = await _require_admin_session(request)
        channels = await _list_text_channels(bot)
        if request.method == "POST":
            form = await request.post()
            action = str(form.get("action", "create")).strip().lower()
            if action == "toggle":
                special_system_id = _parse_positive_int(form.get("special_system_id"), "מערכת מיוחדת")
                assert special_system_id is not None
                requested_state = str(form.get("state", "")).strip().lower()
                if requested_state not in {"activate", "deactivate"}:
                    raise PermissionDeniedError("הפעולה שנבחרה על המערכת המיוחדת לא תקינה.")
                current_system = await bot.services.special_systems.get_special_system(special_system_id)
                updated_system = await bot.services.special_systems.set_active(
                    special_system_id,
                    is_active=requested_state == "activate",
                )
                await _refresh_special_system_public_message(
                    bot,
                    updated_system,
                    previous_record=current_system,
                )
                notice = "המערכת המיוחדת הופעלה מחדש ופורסמה." if requested_state == "activate" else "המערכת המיוחדת הושבתה והוסרה מהדף הציבורי."
            else:
                form_title = str(form.get("title", ""))
                form_description = str(form.get("description", ""))
                selected_payment_methods = {str(value) for value in form.getall("payment_method", [])}
                price_values = {key: str(form.get(f"price_{key}", "")) for key, _label in bot.services.special_systems.available_payment_methods()}
                selected_channel_id = _parse_positive_int(form.get("channel_id"), "ערוץ")
                assert selected_channel_id is not None
                images_uploads: list[tuple[str, bytes, str | None]] = []
                for field in form.getall("images", []):
                    upload = _extract_file_upload(field, image_only=True)
                    if upload is not None:
                        images_uploads.append(upload)
                payment_payload = [(key, price_values.get(key, "")) for key in selected_payment_methods]
                special_system = await bot.services.special_systems.create_special_system(
                    title=form_title,
                    description=form_description,
                    payment_methods=payment_payload,
                    images=images_uploads,
                    channel_id=selected_channel_id,
                    created_by=session.discord_user_id,
                )
                await _refresh_special_system_public_message(bot, special_system)
                notice = "המערכת המיוחדת נשמרה ופורסמה בהצלחה."
        existing_special_systems = await bot.services.special_systems.list_special_systems()
        existing_rows = "\n".join(
            f"""
            <tr>
                <td><strong>{_escape(item.title)}</strong><br><span class="mono">/{_escape(item.slug)}</span></td>
                <td><span class="badge{' rejected' if not item.is_active else ''}">{'פעילה' if item.is_active else 'לא פעילה'}</span></td>
                <td>{_escape(', '.join(f'{method.label}: {method.price}' for method in item.payment_methods))}</td>
                <td>{item.channel_id}</td>
                <td>
                    <div class="actions">
                        {'<a class="link-button ghost-button" href="' + _escape(_special_system_url(bot, item)) + '" target="_blank" rel="noreferrer">פתח דף קניה</a>' if item.is_active else ''}
                        <a class="link-button ghost-button" href="/admin/special-systems/{item.id}/edit">ערוך</a>
                        <form method="post" class="inline-form">
                            <input type="hidden" name="action" value="toggle">
                            <input type="hidden" name="special_system_id" value="{item.id}">
                            <input type="hidden" name="state" value="{'deactivate' if item.is_active else 'activate'}">
                            <button type="submit" class="ghost-button{' danger' if item.is_active else ''}">{'השבת' if item.is_active else 'הפעל מחדש'}</button>
                        </form>
                    </div>
                </td>
            </tr>
            """
            for item in existing_special_systems
        )
        content = f"""
        {_notice_html(notice, success=success)}
        <div class="split-grid">
            <div class="card">
                <h2>פרסום מערכת מיוחדת</h2>
                <form method="post" enctype="multipart/form-data">
                    <input type="hidden" name="action" value="create">
                    <div class="grid">
                        <label class="field field-wide"><span>כותרת</span><input type="text" name="title" value="{_escape(form_title)}" required></label>
                        <label class="field field-wide"><span>תיאור</span><textarea name="description" required>{_escape(form_description)}</textarea></label>
                        <div class="field field-wide"><span>אמצעי תשלום ומחיר לכל אמצעי</span><div class="stack">{_payment_method_editor(bot.services.special_systems, selected_payment_methods, price_values)}</div></div>
                        <label class="field field-wide"><span>תמונות</span><input type="file" name="images" accept="image/*" multiple></label>
                        <label class="field"><span>ערוץ לשליחה</span><select name="channel_id" required>{_render_channel_options(channels, selected_channel_id)}</select></label>
                    </div>
                    <div class="actions"><button type="submit">פרסם מערכת מיוחדת</button></div>
                </form>
            </div>
            <div class="card"><h2>מה הדף מייצר</h2><p>האתר ישלח הודעה עם כפתור <strong>קניה מיוחדת</strong>, יבנה דף הזמנה ציבורי בעברית, וישמור את הבקשות לרשימת האדמין.</p></div>
        </div>
        <div class="table-wrap"><table><thead><tr><th>מערכת</th><th>סטטוס</th><th>שיטות תשלום</th><th>ערוץ</th><th>פעולות</th></tr></thead><tbody>{existing_rows}</tbody></table></div>
        """
        body = _admin_shell(session, current_path=request.path, title="מערכות מיוחדות", intro="יצירת דף קניה מיוחד עם תמונות, מחירים וכפתור קניה יעודי.", content=content)
        return _page_response("מערכות מיוחדות", body)
    except web.HTTPException:
        raise
    except SalesBotError as exc:
        return _error_response("מערכות מיוחדות", str(exc), status=400)


async def special_system_edit_page(request: web.Request) -> web.Response:
    notice: str | None = None
    success = True
    try:
        bot, session = await _require_admin_session(request)
        channels = await _list_text_channels(bot)
        special_system_id = int(request.match_info["special_system_id"])
        current_system = await bot.services.special_systems.get_special_system(special_system_id)
        images = await bot.services.special_systems.list_special_system_images(current_system.id)
        form_title = current_system.title
        form_description = current_system.description
        selected_payment_methods = {method.key for method in current_system.payment_methods}
        price_values = {method.key: method.price for method in current_system.payment_methods}
        selected_channel_id: int | None = current_system.channel_id
        replace_images = False

        if request.method == "POST":
            form = await request.post()
            form_title = str(form.get("title", ""))
            form_description = str(form.get("description", ""))
            selected_payment_methods = {str(value) for value in form.getall("payment_method", [])}
            price_values = {key: str(form.get(f"price_{key}", "")) for key, _label in bot.services.special_systems.available_payment_methods()}
            selected_channel_id = _parse_positive_int(form.get("channel_id"), "ערוץ")
            assert selected_channel_id is not None
            replace_images = str(form.get("replace_images", "")).lower() in {"1", "true", "yes", "on"}
            images_uploads: list[tuple[str, bytes, str | None]] = []
            for field in form.getall("images", []):
                upload = _extract_file_upload(field, image_only=True)
                if upload is not None:
                    images_uploads.append(upload)
            payment_payload = [(key, price_values.get(key, "")) for key in selected_payment_methods]
            updated_system = await bot.services.special_systems.update_special_system(
                current_system.id,
                title=form_title,
                description=form_description,
                payment_methods=payment_payload,
                channel_id=selected_channel_id,
                replace_images=replace_images,
                images=images_uploads,
            )
            if updated_system.is_active:
                updated_system = await _refresh_special_system_public_message(
                    bot,
                    updated_system,
                    previous_record=current_system,
                )
            current_system = updated_system
            images = await bot.services.special_systems.list_special_system_images(current_system.id)
            notice = "המערכת המיוחדת עודכנה בהצלחה."

        public_url = _special_system_url(bot, current_system) if current_system.is_active else None
        message_url = _message_link(bot, current_system.channel_id, current_system.message_id)
        gallery_html = '<div class="gallery">' + ''.join(
            f'<img src="/special-system-images/{image.id}" alt="{_escape(image.asset_name)}">' for image in images
        ) + '</div>' if images else '<p class="muted">אין כרגע תמונות שמורות למערכת הזאת.</p>'
        content = f"""
        {_notice_html(notice, success=success)}
        <div class="split-grid">
            <div class="card stack">
                <div>
                    <h2>עריכת מערכת מיוחדת #{current_system.id}</h2>
                    <p class="muted">ה-slug הציבורי נשאר קבוע כדי לא לשבור קישורים קיימים.</p>
                </div>
                <div class="price-list">
                    <div class="price-item"><strong>Slug</strong><span class="mono">/{_escape(current_system.slug)}</span></div>
                    <div class="price-item"><strong>סטטוס</strong><span>{'פעילה' if current_system.is_active else 'לא פעילה'}</span></div>
                    <div class="price-item"><strong>ערוץ נוכחי</strong><span>{current_system.channel_id}</span></div>
                </div>
                <form method="post" enctype="multipart/form-data">
                    <div class="grid">
                        <label class="field field-wide"><span>כותרת</span><input type="text" name="title" value="{_escape(form_title)}" required></label>
                        <label class="field field-wide"><span>תיאור</span><textarea name="description" required>{_escape(form_description)}</textarea></label>
                        <div class="field field-wide"><span>אמצעי תשלום ומחיר לכל אמצעי</span><div class="stack">{_payment_method_editor(bot.services.special_systems, selected_payment_methods, price_values)}</div></div>
                        <label class="field field-wide"><span>תמונות חדשות</span><input type="file" name="images" accept="image/*" multiple></label>
                        <label class="field"><span>ערוץ לשליחה</span><select name="channel_id" required>{_render_channel_options(channels, selected_channel_id)}</select></label>
                        <label class="field"><span><input type="checkbox" name="replace_images" value="true"{' checked' if replace_images else ''}> החלף את כל התמונות הקיימות</span></label>
                    </div>
                    <div class="actions"><button type="submit">שמור שינויים</button><a class="link-button ghost-button" href="/admin/special-systems">חזרה לרשימה</a></div>
                </form>
            </div>
            <div class="card stack">
                <div><h2>תצוגה נוכחית</h2><p>אפשר להוסיף תמונות חדשות או להחליף את כל הגלריה הקיימת.</p></div>
                {gallery_html}
                <div class="actions">{'<a class="link-button ghost-button" href="' + _escape(public_url) + '" target="_blank" rel="noreferrer">פתח דף קניה</a>' if public_url else ''}{'<a class="link-button ghost-button" href="' + _escape(message_url) + '" target="_blank" rel="noreferrer">פתח הודעה בדיסקורד</a>' if message_url else ''}</div>
            </div>
        </div>
        """
        body = _admin_shell(session, current_path=request.path, title=f"עריכת מערכת מיוחדת #{current_system.id}", intro="עריכת פרטי המערכת המיוחדת ופרסום מחדש של ההודעה הציבורית לפי הצורך.", content=content)
        return _page_response(f"עריכת מערכת מיוחדת #{current_system.id}", body)
    except web.HTTPException:
        raise
    except SalesBotError as exc:
        return _error_response("עריכת מערכת מיוחדת", str(exc), status=400)


async def special_orders_list_page(request: web.Request) -> web.Response:
    try:
        bot, session = await _require_admin_session(request)
        notice: str | None = None
        if request.method == "POST":
            form = await request.post()
            action = str(form.get("action", "")).strip().lower()
            if action != "delete":
                raise PermissionDeniedError("הפעולה שנשלחה לבקשות המיוחדות לא תקינה.")
            order_id = _parse_positive_int(form.get("order_id"), "מזהה בקשה")
            assert order_id is not None
            deleted = await bot.services.special_systems.delete_order_request(order_id)
            notice = f"בקשה מיוחדת #{deleted.id} נמחקה ממסד הנתונים."
        elif str(request.query.get("deleted", "")).strip().isdigit():
            notice = f"בקשה מיוחדת #{_escape(request.query.get('deleted'))} נמחקה ממסד הנתונים."
        status_filter = str(request.query.get("status", "all")).strip().lower()
        statuses = None if status_filter == "all" else (status_filter,)
        orders = await bot.services.special_systems.list_order_requests(statuses=statuses)
        systems = {item.id: item for item in await bot.services.special_systems.list_special_systems()}
        rows = "\n".join(
            f"""
            <tr>
                <td><strong>#{order.id}</strong></td>
                <td>{_escape(systems.get(order.special_system_id).title if systems.get(order.special_system_id) else f'#{order.special_system_id}')}</td>
                <td><span class="mono">{order.user_id}</span><br>{_escape(order.discord_name)}</td>
                <td>{_escape(order.payment_method_label)}<br>{_escape(order.payment_price)}</td>
                <td>{_status_badge(order.status)}</td>
                <td><div class="table-actions"><a class="link-button ghost-button" href="/admin/special-orders/{order.id}">פתח</a><form method="post" class="inline-form"><input type="hidden" name="action" value="delete"><input type="hidden" name="order_id" value="{order.id}"><button type="submit" class="ghost-button danger">מחק</button></form></div></td>
            </tr>
            """
            for order in orders
        )
        if not rows:
            rows = '<tr><td colspan="6">אין כרגע בקשות שתואמות למסנן שבחרת.</td></tr>'
        content = f"""
        {_notice_html(notice, success=True)}
        <div class="actions"><a class="link-button ghost-button" href="/admin/special-orders?status=all">הכל</a><a class="link-button ghost-button" href="/admin/special-orders?status=pending">ממתינות</a><a class="link-button ghost-button" href="/admin/special-orders?status=accepted">התקבלו</a><a class="link-button ghost-button" href="/admin/special-orders?status=rejected">נדחו</a></div>
        <div class="table-wrap"><table><thead><tr><th>#</th><th>מערכת</th><th>לקוח</th><th>תשלום</th><th>סטטוס</th><th></th></tr></thead><tbody>{rows}</tbody></table></div>
        """
        body = _admin_shell(session, current_path=request.path, title="בקשות למערכות מיוחדות", intro="ריכוז כל הבקשות שהגיעו דרך דפי הקניה המיוחדים.", content=content)
        return _page_response("בקשות מיוחדות", body)
    except web.HTTPException:
        raise
    except SalesBotError as exc:
        return _error_response("בקשות מיוחדות", str(exc), status=400)


async def special_order_detail_page(request: web.Request) -> web.Response:
    notice: str | None = None
    success = True
    try:
        bot, session = await _require_admin_session(request)
        order_id = int(request.match_info["order_id"])
        order = await bot.services.special_systems.get_order_request(order_id)
        special_system = await bot.services.special_systems.get_special_system(order.special_system_id)
        if request.method == "POST":
            form = await request.post()
            action = str(form.get("action", "")).strip()
            if action == "delete":
                deleted = await bot.services.special_systems.delete_order_request(order.id)
                raise web.HTTPFound(f"/admin/special-orders?deleted={deleted.id}")
            if action not in {"accept", "reject"}:
                raise PermissionDeniedError("הפעולה שנבחרה לא תקינה.")
            if order.status != "pending":
                raise PermissionDeniedError("אפשר לאשר או לדחות רק בקשה שעדיין ממתינה לטיפול.")
            admin_reply = str(form.get("admin_reply", "")).strip() or None
            order = await bot.services.special_systems.resolve_order_request(order.id, reviewer_id=session.discord_user_id, status="accepted" if action == "accept" else "rejected", admin_reply=admin_reply)
            try:
                requester = await bot.fetch_user(order.user_id)
                if action == "accept":
                    await requester.send(admin_reply or "הבקשה שלך לקניית מערכת מיוחדת התקבלה.")
                else:
                    decline_message = "הבקשה שלך לקניית מערכת מיוחדת נדחתה"
                    if admin_reply:
                        decline_message += f"\n\n{admin_reply}"
                    await requester.send(decline_message)
            except discord.HTTPException:
                pass
            await _update_owner_order_message(bot, special_system, order)
            notice = "הבקשה עודכנה והלקוח קיבל הודעה ב-DM אם היה אפשר לשלוח לו."
        linked_roblox_label = "לא מחובר"
        if order.linked_roblox_sub:
            linked_roblox_label = " | ".join(part for part in (order.linked_roblox_display_name, order.linked_roblox_username, order.linked_roblox_sub) if part)
        buttons_html = '<button type="submit" name="action" value="delete" class="ghost-button danger">מחק בקשה</button>'
        if order.status == 'pending':
            buttons_html = '<button type="submit" name="action" value="accept">אשר בקשה</button><button type="submit" name="action" value="reject" class="ghost-button danger">דחה בקשה</button>' + buttons_html
        content = f"""
        {_notice_html(notice, success=success)}
        <div class="split-grid">
            <div class="card stack">
                <div><h2>פרטי הבקשה</h2></div>
                <div class="price-list">
                    <div class="price-item"><strong>מערכת מיוחדת</strong><span>{_escape(special_system.title)}</span></div>
                    <div class="price-item"><strong>סטטוס</strong><span>{_status_badge(order.status)}</span></div>
                    <div class="price-item"><strong>דיסקורד</strong><span>{_escape(order.discord_name)}<br><span class="mono">{order.user_id}</span></span></div>
                    <div class="price-item"><strong>רובלוקס שנשלח</strong><span>{_escape(order.roblox_name)}</span></div>
                    <div class="price-item"><strong>שיטת תשלום</strong><span>{_escape(order.payment_method_label)} | {_escape(order.payment_price)}</span></div>
                    <div class="price-item"><strong>חשבון רובלוקס מחובר</strong><span>{_escape(linked_roblox_label)}</span></div>
                    <div class="price-item"><strong>נשלח בתאריך</strong><span>{_escape(order.submitted_at)}</span></div>
                </div>
            </div>
            <div class="card">
                <h2>טיפול בבקשה</h2>
                <form method="post">
                    <div class="grid"><label class="field field-wide"><span>הודעה ללקוח</span><textarea name="admin_reply" placeholder="הודעה שתישלח ללקוח אם תאשר, או סיבה אם תדחה.">{_escape(order.admin_reply or '')}</textarea></label></div>
                    <div class="actions">{buttons_html}<a class="link-button ghost-button" href="/admin/special-orders">חזרה לרשימה</a></div>
                </form>
            </div>
        </div>
        """
        body = _admin_shell(session, current_path=request.path, title=f"בקשה מיוחדת #{order.id}", intro="בדיקת כל הפרטים לפני אישור, דחייה או מחיקה של בקשת הקניה.", content=content)
        return _page_response(f"בקשה מיוחדת #{order.id}", body)
    except web.HTTPException:
        raise
    except SalesBotError as exc:
        return _error_response("פרטי בקשה מיוחדת", str(exc), status=400)


async def custom_orders_list_page(request: web.Request) -> web.Response:
    try:
        bot, session = await _require_admin_session(request)
        notice: str | None = None
        if request.method == "POST":
            form = await request.post()
            action = str(form.get("action", "")).strip().lower()
            if action != "delete":
                raise PermissionDeniedError("הפעולה שנשלחה להזמנות האישיות לא תקינה.")
            order_id = _parse_positive_int(form.get("order_id"), "מזהה הזמנה")
            assert order_id is not None
            deleted = await bot.services.orders.delete_request(order_id)
            notice = f"הזמנה אישית #{deleted.id} נמחקה ממסד הנתונים."
        elif str(request.query.get("deleted", "")).strip().isdigit():
            notice = f"הזמנה אישית #{_escape(request.query.get('deleted'))} נמחקה ממסד הנתונים."
        status_filter = str(request.query.get("status", "all")).strip().lower()
        statuses = None if status_filter == "all" else (status_filter,)
        orders = await bot.services.orders.list_requests(statuses=statuses)
        requester_labels = await asyncio.gather(*(_discord_user_label(bot, order.user_id) for order in orders)) if orders else []
        rows = "\n".join(
            f"""
            <tr>
                <td><strong>#{order.id}</strong></td>
                <td>{_escape(requester_label)}<br><span class="mono">{order.user_id}</span></td>
                <td><strong>{_escape(order.requested_item)}</strong><br><span class="muted">{_escape(order.required_timeframe)}</span></td>
                <td>{_escape(order.payment_method)}<br>{_escape(order.offered_price)}</td>
                <td>{_escape(order.roblox_username or 'לא צוין')}</td>
                <td>{_status_badge(order.status)}</td>
                <td><div class="table-actions"><a class="link-button ghost-button" href="/admin/custom-orders/{order.id}">פתח</a><form method="post" class="inline-form"><input type="hidden" name="action" value="delete"><input type="hidden" name="order_id" value="{order.id}"><button type="submit" class="ghost-button danger">מחק</button></form></div></td>
            </tr>
            """
            for order, requester_label in zip(orders, requester_labels)
        )
        if not rows:
            rows = '<tr><td colspan="7">אין כרגע הזמנות שתואמות למסנן שבחרת.</td></tr>'
        content = f"""
        {_notice_html(notice, success=True)}
        <div class="actions"><a class="link-button ghost-button" href="/admin/custom-orders?status=all">הכל</a><a class="link-button ghost-button" href="/admin/custom-orders?status=pending">ממתינות</a><a class="link-button ghost-button" href="/admin/custom-orders?status=accepted">התקבלו</a><a class="link-button ghost-button" href="/admin/custom-orders?status=completed">הושלמו</a><a class="link-button ghost-button" href="/admin/custom-orders?status=rejected">נדחו</a></div>
        <div class="table-wrap"><table><thead><tr><th>#</th><th>לקוח</th><th>מה הוזמן</th><th>תשלום</th><th>רובלוקס</th><th>סטטוס</th><th></th></tr></thead><tbody>{rows}</tbody></table></div>
        """
        body = _admin_shell(session, current_path=request.path, title="הזמנות אישיות", intro="ריכוז כל ההזמנות האישיות שנשלחו דרך דף האתר החדש.", content=content)
        return _page_response("הזמנות אישיות", body)
    except web.HTTPException:
        raise
    except SalesBotError as exc:
        return _error_response("הזמנות אישיות", str(exc), status=400)


async def custom_order_detail_page(request: web.Request) -> web.Response:
    notice: str | None = None
    success = True
    try:
        bot, session = await _require_admin_session(request)
        order_id = int(request.match_info["order_id"])
        order = await bot.services.orders.get_request(order_id)

        if request.method == "POST":
            form = await request.post()
            action = str(form.get("action", "")).strip().lower()
            if action == "delete":
                deleted = await bot.services.orders.delete_request(order.id)
                raise web.HTTPFound(f"/admin/custom-orders?deleted={deleted.id}")
            if action not in {"accept", "reject", "complete"}:
                raise PermissionDeniedError("הפעולה שנבחרה להזמנה האישית לא תקינה.")
            if order.status in {"rejected", "completed"}:
                raise PermissionDeniedError("אי אפשר לשנות הזמנה שכבר נדחתה או הושלמה.")
            if order.status == "pending" and action == "complete":
                raise PermissionDeniedError("אפשר לסמן כהושלמה רק הזמנה שכבר התקבלה.")

            admin_reply = str(form.get("admin_reply", "")).strip() or None
            target_status = {
                "accept": "accepted",
                "reject": "rejected",
                "complete": "completed",
            }[action]
            order = await bot.services.orders.resolve_request(
                order.id,
                reviewer_id=session.discord_user_id,
                status=target_status,
                admin_reply=admin_reply,
            )
            await _notify_custom_order_requester(bot, order, admin_reply=admin_reply)
            await _update_owner_custom_order_message(bot, order)
            notice = {
                "accept": "ההזמנה התקבלה והלקוח קיבל עדכון ב-DM אם היה אפשר לשלוח.",
                "reject": "ההזמנה נדחתה והלקוח קיבל את הסיבה ב-DM אם היה אפשר לשלוח.",
                "complete": "ההזמנה סומנה כהושלמה והלקוח קיבל עדכון ב-DM אם היה אפשר לשלוח.",
            }[action]

        order_images = await bot.services.orders.list_request_images(order.id)
        requester_label = await _discord_user_label(bot, order.user_id)
        reviewer_label = await _discord_user_label(bot, order.reviewed_by) if order.reviewed_by is not None else None
        admin_note_label = ""
        if order.admin_reply:
            admin_note_label = "סיבת דחייה" if order.status == "rejected" else "הודעת אדמין"

        buttons_html = ""
        if order.status == "pending":
            buttons_html = '<button type="submit" name="action" value="accept">אשר הזמנה</button><button type="submit" name="action" value="reject" class="ghost-button danger">דחה הזמנה</button>'
        elif order.status == "accepted":
            buttons_html = '<button type="submit" name="action" value="complete">סמן כהושלמה</button><button type="submit" name="action" value="reject" class="ghost-button danger">דחה הזמנה</button>'

        review_meta = ""
        if order.reviewed_at:
            review_meta = f'<div class="price-item"><strong>טופלה בתאריך</strong><span>{_escape(order.reviewed_at)}</span></div>'
        if reviewer_label is not None:
            review_meta += f'<div class="price-item"><strong>טופלה על ידי</strong><span>{_escape(reviewer_label)}<br><span class="mono">{order.reviewed_by}</span></span></div>'
        if order.admin_reply and admin_note_label:
            review_meta += f'<div class="price-item"><strong>{_escape(admin_note_label)}</strong><span>{_escape(order.admin_reply)}</span></div>'

        delete_button_html = '<button type="submit" name="action" value="delete" class="ghost-button danger">מחק הזמנה</button>'
        content = f"""
        {_notice_html(notice, success=success)}
        <div class="split-grid">
            <div class="card stack">
                <div><h2>פרטי ההזמנה</h2></div>
                <div class="price-list">
                    <div class="price-item"><strong>סטטוס</strong><span>{_status_badge(order.status)}</span></div>
                    <div class="price-item"><strong>דיסקורד</strong><span>{_escape(requester_label)}<br><span class="mono">{order.user_id}</span></span></div>
                    <div class="price-item"><strong>מה הוזמן</strong><span>{_escape(order.requested_item)}</span></div>
                    <div class="price-item"><strong>דדליין שביקש הלקוח</strong><span>{_escape(order.required_timeframe)}</span></div>
                    <div class="price-item"><strong>שיטת תשלום</strong><span>{_escape(order.payment_method)}</span></div>
                    <div class="price-item"><strong>הצעת מחיר / תמורה</strong><span>{_escape(order.offered_price)}</span></div>
                    <div class="price-item"><strong>שם רובלוקס</strong><span>{_escape(order.roblox_username or 'לא צוין')}</span></div>
                    <div class="price-item"><strong>תמונות שצורפו</strong><span>{len(order_images)}</span></div>
                    <div class="price-item"><strong>נשלח בתאריך</strong><span>{_escape(order.submitted_at)}</span></div>
                    {review_meta}
                </div>
            </div>
            <div class="card">
                <h2>טיפול בהזמנה</h2>
                <form method="post">
                    <div class="grid"><label class="field field-wide"><span>הודעה ללקוח</span><textarea name="admin_reply" placeholder="הודעה שתישלח ללקוח אם תאשר, תדחה או תסמן כהושלמה.">{_escape(order.admin_reply or '')}</textarea></label></div>
                    <div class="actions">{buttons_html}{delete_button_html}<a class="link-button ghost-button" href="/admin/custom-orders">חזרה לרשימה</a></div>
                </form>
            </div>
        </div>
        {f'<div class="card stack gallery-section"><h2>תמונות שצורפו</h2>{_render_image_slider(_custom_order_gallery_urls(order_images), alt_text=f"הזמנה אישית {order.id}", empty_label="לא צורפו תמונות")}</div>' if order_images else ''}
        """
        body = _admin_shell(session, current_path=request.path, title=f"הזמנה אישית #{order.id}", intro="בדיקה, אישור, דחייה, סיום או מחיקה של הזמנה אישית שנשלחה מהאתר.", content=content)
        return _page_response(f"הזמנה אישית #{order.id}", body)
    except web.HTTPException:
        raise
    except SalesBotError as exc:
        return _error_response("פרטי הזמנה אישית", str(exc), status=400)


async def special_system_image_page(request: web.Request) -> web.Response:
    bot: SalesBot = request.app["bot"]
    try:
        image = await bot.services.special_systems.get_special_system_image(int(request.match_info["image_id"]))
        return web.Response(body=image.asset_bytes, content_type=image.content_type or "application/octet-stream")
    except SalesBotError as exc:
        return _error_response("תמונת מערכת מיוחדת", str(exc), status=404)


async def custom_order_image_page(request: web.Request) -> web.Response:
    try:
        bot, _session = await _require_admin_session(request)
        image = await bot.services.orders.get_request_image(int(request.match_info["image_id"]))
        content_type = image.content_type or mimetypes.guess_type(image.asset_name)[0] or "application/octet-stream"
        return web.Response(body=image.asset_bytes, content_type=content_type)
    except web.HTTPException:
        raise
    except SalesBotError as exc:
        return _error_response("תמונת הזמנה אישית", str(exc), status=404)


async def custom_orders_page(request: web.Request) -> web.Response:
    notice: str | None = None
    success = True
    requested_item = ""
    required_timeframe = ""
    selected_payment_method = ""
    offered_price = ""
    roblox_username = ""
    try:
        bot_ref, session = await _current_site_session(request)
        bot = bot_ref
        if session is not None:
            await _ensure_site_session_allowed(bot, session)

        if request.method == "POST":
            try:
                bot, session = await _require_active_site_session(request)
                form = await request.post()
                requested_item = str(form.get("requested_item", "")).strip()
                required_timeframe = str(form.get("required_timeframe", "")).strip()
                selected_payment_method = str(form.get("payment_method", "")).strip()
                offered_price = str(form.get("offered_price", "")).strip()
                roblox_username = str(form.get("roblox_username", "")).strip()
                uploaded_images = [
                    image
                    for image in (
                        _extract_file_upload(field, image_only=True)
                        for field in form.getall("images", [])
                    )
                    if image is not None
                ]
                if len(uploaded_images) > CUSTOM_ORDER_MAX_IMAGES:
                    raise PermissionDeniedError(f"אפשר לצרף עד {CUSTOM_ORDER_MAX_IMAGES} תמונות להזמנה האישית.")
                if not requested_item or not required_timeframe or not selected_payment_method or not offered_price or not roblox_username:
                    raise PermissionDeniedError("חובה למלא את כל השדות בטופס ההזמנה.")

                order = await bot.services.orders.create_request(
                    user_id=session.discord_user_id,
                    requested_item=requested_item,
                    required_timeframe=required_timeframe,
                    payment_method=selected_payment_method,
                    offered_price=offered_price,
                    roblox_username=roblox_username,
                    images=uploaded_images,
                )
                delivered_count, owner_message_id = await _send_custom_order_to_admins(bot, order)
                if owner_message_id is not None:
                    await bot.services.orders.set_owner_message(order.id, owner_message_id)
                if delivered_count <= 0:
                    LOGGER.warning("Custom order %s was saved but no admin DM could be delivered", order.id)

                success_html = """
                <div class="card stack">
                    <div><h2>ההזמנה נשלחה</h2><p>ההזמנה נשמרה ונשלחה לכל האדמינים שניתן היה להגיע אליהם. היא מחכה עכשיו ברשימת האדמין באתר.</p></div>
                    <div class="actions"><a class="link-button" href="/custom-orders">שלח הזמנה נוספת</a></div>
                </div>
                """
                body = _public_shell(
                    session,
                    current_path="/custom-orders",
                    title="הזמנה אישית",
                    intro="ההזמנה שלך נשמרה ונשלחה לאדמינים.",
                    login_path="/custom-orders",
                    section_label="הזמנות אישיות",
                    content=_notice_html("ההזמנה נשלחה בהצלחה. נחזור אליך ב-DM אחרי שנבדוק אותה.", success=True) + success_html,
                )
                return _page_response("הזמנה אישית", body)
            except web.HTTPRequestEntityTooLarge:
                notice = _custom_order_upload_limit_message()
                success = False
                bot_ref, session = await _current_site_session(request)
                bot = bot_ref
            except ValueError as exc:
                if "Maximum request body size" not in str(exc):
                    raise
                notice = _custom_order_upload_limit_message()
                success = False
                bot_ref, session = await _current_site_session(request)
                bot = bot_ref
            except SalesBotError as exc:
                notice = str(exc)
                success = False
                bot_ref, session = await _current_site_session(request)
                bot = bot_ref

        payment_methods_html = ''.join(
            f'<div class="price-item"><strong>{_escape(label)}</strong><span>אפשר לבחור בטופס</span></div>'
            for _key, label in bot.services.orders.available_payment_methods()
        )
        upload_fields_html = ''.join(
            f'''
            <label class="upload-slot{' is-hidden' if index > 0 else ''}" data-upload-slot>
                <strong>תמונה {index + 1}</strong>
                <input type="file" name="images" accept="image/*">
            </label>
            '''
            for index in range(CUSTOM_ORDER_MAX_IMAGES)
        )
        connected_account_html = ''
        if session is not None:
            connected_account_html = f'<div class="meta-card"><p><strong>חשבון דיסקורד מחובר:</strong> {_escape(_session_label(session))}</p></div>'
        content = f"""
        {_notice_html(notice, success=success)}
        <div class="split-grid">
            <div class="card stack">
                <div><h2>מה מקבלים בדף הזה</h2><p>אפשר לשלוח הזמנה אישית בלי לחבר חשבון רובלוקס לדיסקורד. כל מה שצריך הוא להתחבר עם דיסקורד ולמלא את הפרטים.</p></div>
                <div><h3>שיטות תשלום זמינות</h3><div class="price-list">{payment_methods_html}</div></div>
            </div>
            <div class="card">
                <h2>טופס הזמנה אישית</h2>
                <p class="muted">כל השדות חובה. שם הדיסקורד שלך נלקח אוטומטית מההתחברות לאתר.</p>
                {connected_account_html}
                <form method="post" enctype="multipart/form-data">
                    <div class="grid">
                        <label class="field field-wide"><span>מה אתה רוצה להזמין</span><textarea name="requested_item" required>{_escape(requested_item)}</textarea></label>
                        <label class="field"><span>תוך כמה זמן אתה צריך את זה</span><input type="text" name="required_timeframe" value="{_escape(required_timeframe)}" required></label>
                        <label class="field"><span>איך אתה משלם</span><select name="payment_method" required>{_order_payment_method_select_options(bot.services.orders, selected_payment_method)}</select></label>
                        <label class="field field-wide"><span>כמה אתה מוכן לשלם (או מה אתה מביא אם זה דברים במשחק)</span><textarea name="offered_price" required>{_escape(offered_price)}</textarea></label>
                        <label class="field"><span>מה השם שלך ברובלוקס</span><input type="text" name="roblox_username" value="{_escape(roblox_username)}" required></label>
                        <div class="field field-wide">
                            <span>תמונות לעיון האדמינים</span>
                            <div class="upload-slot-list" data-upload-sequence>
                                {upload_fields_html}
                            </div>
                            <p class="setting-hint">אפשר להעלות עד {CUSTOM_ORDER_MAX_IMAGES} תמונות. כל בחירה תפתח שדה נוסף, וביחד אפשר לשלוח עד {CUSTOM_ORDER_FORM_MAX_MB}MB.</p>
                        </div>
                    </div>
                    <div class="actions"><button type="submit">שלח הזמנה</button></div>
                </form>
            </div>
        </div>
        """
        body = _public_shell(
            session,
            current_path="/custom-orders",
            title="הזמנה אישית",
            intro="שלח כאן הזמנה אישית חדשה במקום הטופס הישן של דיסקורד.",
            login_path="/custom-orders",
            section_label="הזמנות אישיות",
            content=content,
        )
        return _page_response("הזמנה אישית", body)
    except web.HTTPException:
        raise
    except SalesBotError as exc:
        return _error_response("הזמנה אישית", str(exc), status=400)


async def account_payment_page(request: web.Request) -> web.Response:
    notice: str | None = None
    success = True
    roblox_username = ""
    roblox_password = ""
    profile_link = ""
    has_email = ""
    has_phone = ""
    has_two_factor = ""
    confirmed = False
    try:
        bot_ref, session = await _current_site_session(request)
        bot = bot_ref
        if session is not None:
            await _ensure_site_session_allowed(bot, session)

        if request.method == "POST":
            try:
                bot, session = await _require_active_site_session(request)
                form = await request.post()
                roblox_username = str(form.get("roblox_username", "")).strip()
                roblox_password = str(form.get("roblox_password", "")).strip()
                profile_link = str(form.get("profile_link", "")).strip()
                has_email = str(form.get("has_email", "")).strip().lower()
                has_phone = str(form.get("has_phone", "")).strip().lower()
                has_two_factor = str(form.get("has_two_factor", "")).strip().lower()
                confirmed = str(form.get("confirmed", "")).strip().lower() in {"1", "true", "yes", "on"}
                profile_image = _extract_file_upload(form.get("profile_image"), image_only=True)

                if not roblox_username or not roblox_password or not has_email or not has_phone or not has_two_factor:
                    raise PermissionDeniedError("חובה למלא את כל שדות החובה בטופס הזה.")
                if not confirmed:
                    raise PermissionDeniedError("חובה לאשר שאתה מבין שאין החזרות ושכל הפרטים נכונים.")

                delivered_count = await _send_account_payment_submission_to_admins(
                    bot,
                    session=session,
                    roblox_username=roblox_username,
                    roblox_password=roblox_password,
                    profile_link=profile_link or None,
                    profile_image=profile_image,
                    has_email=has_email == "yes",
                    has_phone=has_phone == "yes",
                    has_two_factor=has_two_factor == "yes",
                )
                if delivered_count <= 0:
                    raise ExternalServiceError("לא הצלחתי להעביר את פרטי המשתמש לאף אדמין ב-DM. נסה שוב בעוד רגע.")

                success_html = """
                <div class="card stack">
                    <div><h2>הטופס נשלח</h2><p>הפרטים הועברו לאדמינים בהצלחה. אחרי האימות המלא וההגעה של המשתמש ליוצרים, תקבלו את מה שסוכם.</p></div>
                    <div class="actions"><a class="link-button" href="/account-payment">שלח טופס נוסף</a></div>
                </div>
                """
                body = _public_shell(
                    session,
                    current_path="/account-payment",
                    title="שליחת משתמש בתור תשלום",
                    intro="הטופס נשלח לאדמינים בהצלחה.",
                    login_path="/account-payment",
                    section_label="תשלום במשתמש רובלוקס",
                    content=_notice_html("הטופס נשלח בהצלחה לאדמינים.", success=True) + success_html,
                )
                return _page_response("שליחת משתמש בתור תשלום", body)
            except SalesBotError as exc:
                notice = str(exc)
                success = False
                bot_ref, session = await _current_site_session(request)
                bot = bot_ref

        connected_account_html = ""
        if session is not None:
            connected_account_html = f'<div class="meta-card"><p><strong>חשבון דיסקורד מחובר:</strong> {_escape(_session_label(session))}</p></div>'

        content = f"""
        {_notice_html(notice, success=success)}
        <div class="split-grid">
            <div class="card stack">
                <div>
                    <h2>לפני שליחת הטופס</h2>
                    <p class="warning-note"><strong>הדף הזה הוא דף שליחת משתמש בתור תשלום. יש כמה דברים חשובים שתצטרך לדעת ובעת שליחת הטופס אתה מסכים להם</strong></p>
                    <p class="warning-note"><strong>אתה שולח כאן את הפרטים של המשתמש רובלוקס שאתה רוצה לתת לנו בתור תשלום</strong></p>
                    <p class="warning-note"><strong>אתה נותן את הסיסמא שלך למשתמש שאתה רוצה לתת לנו בתור תשלום</strong></p>
                    <p class="warning-note"><strong>אתה מסכים לכך שאין החזרות ורק לאחר האימות המלא וההגעה של המשתמש ליוצרים אתה תקבל את מה שהזמנת</strong></p>
                </div>
            </div>
            <div class="card">
                <h2>טופס שליחת משתמש</h2>
                <p class="muted">הטופס הזה דורש התחברות עם דיסקורד כדי שנדע מי שלח את הפרטים.</p>
                {connected_account_html}
                <form method="post" enctype="multipart/form-data">
                    <div class="grid">
                        <label class="field"><span>השם של המשתמש רובלוקס (שם!!!! לא כינוי!!!!)</span><input type="text" name="roblox_username" value="{_escape(roblox_username)}" required></label>
                        <label class="field"><span>סיסמא של המשתמש רובלוקס</span><input type="text" name="roblox_password" value="{_escape(roblox_password)}" required></label>
                        <label class="field field-wide"><span>קישור לפרופיל ברובלוקס במידה ואתה יכול לשלוח</span><input type="url" name="profile_link" value="{_escape(profile_link)}"></label>
                        <label class="field field-wide"><span>תמונה של הפרופיל במידה ואתה יכול להוסיף</span><input type="file" name="profile_image" accept="image/*"></label>
                        <label class="field"><span>האם יש על המשתמש מייל שלך</span><select name="has_email" required>{_yes_no_select_options(has_email)}</select></label>
                        <div class="field field-wide">
                            <span>האם יש מספר טלפון על המשתמש</span>
                            <select name="has_phone" required>{_yes_no_select_options(has_phone)}</select>
                            <p class="warning-note"><strong>מומלץ להוריד את המספר טלפון מהמשתמש לפני שתשלח את זה</strong></p>
                        </div>
                        <div class="field field-wide">
                            <span>האם יש אימות דו שלבי על המשתמש</span>
                            <select name="has_two_factor" required>{_yes_no_select_options(has_two_factor)}</select>
                            <p class="warning-note"><strong>אנא תוריד את האימות דו שלבי לפני שתשלח את המשתמש</strong></p>
                        </div>
                        <label class="meta-card check-card field-wide">
                            <span class="check-line warning-note">
                                <input type="checkbox" name="confirmed" value="true"{' checked' if confirmed else ''} required>
                                <strong>האם אתה מבין שאתה מביא לנו את המשתמש הזה והוא לא יחזור אלייך אחר כך, ובנוסף לכך אתה מאשר בכך שהבאת פרטים נכונים ולא זייפת אף פרט? (במידה ותזייף פרטים אתה תקבל בלאקליסט מהמשחק והשרת שלנו)</strong>
                            </span>
                        </label>
                    </div>
                    <div class="actions"><button type="submit">שלח את המשתמש לאדמינים</button></div>
                </form>
            </div>
        </div>
        """
        body = _public_shell(
            session,
            current_path="/account-payment",
            title="שליחת משתמש בתור תשלום",
            intro="שלח כאן את פרטי המשתמש שאתה מביא כתשלום, אחרי התחברות עם דיסקורד.",
            login_path="/account-payment",
            section_label="תשלום במשתמש רובלוקס",
            content=content,
        )
        return _page_response("שליחת משתמש בתור תשלום", body)
    except web.HTTPException:
        raise
    except SalesBotError as exc:
        return _error_response("שליחת משתמש בתור תשלום", str(exc), status=400)


async def special_system_page(request: web.Request) -> web.Response:
    notice: str | None = None
    success = True
    selected_payment_method = ""
    discord_name = ""
    roblox_name = ""
    try:
        bot: SalesBot = request.app["bot"]
        special_system = await bot.services.special_systems.get_special_system_by_slug(request.match_info["slug"])
        images = await bot.services.special_systems.list_special_system_images(special_system.id)
        bot_ref, session = await _require_active_site_session(request)
        assert bot_ref is bot
        linked_account: RobloxLinkRecord | None = None
        discord_name = _session_label(session)
        try:
            linked_account = await bot.services.oauth.get_link(session.discord_user_id)
        except NotFoundError:
            linked_account = None
        if request.method == "POST":
            form = await request.post()
            selected_payment_method = str(form.get("payment_method", "")).strip()
            discord_name = str(form.get("discord_name", "")).strip()
            roblox_name = str(form.get("roblox_name", "")).strip()
            if not discord_name or not roblox_name or not selected_payment_method:
                raise PermissionDeniedError("חובה למלא את כל השדות בטופס ההזמנה.")
            try:
                linked_account = await bot.services.oauth.get_link(session.discord_user_id)
            except NotFoundError:
                linked_account = None
            order = await bot.services.special_systems.create_order_request(special_system_id=special_system.id, user_id=session.discord_user_id, discord_name=discord_name, roblox_name=roblox_name, payment_method_key=selected_payment_method, linked_account=linked_account)
            owner = await bot.fetch_user(bot.settings.owner_user_id)
            owner_dm = owner.dm_channel or await owner.create_dm()
            owner_embed = await _owner_order_embed(special_system, order)
            view = discord.ui.View()
            view.add_item(discord.ui.Button(label="פתח את הבקשה באתר", style=discord.ButtonStyle.link, url=f"{bot.settings.public_base_url}/admin/special-orders/{order.id}"))
            owner_message = await owner_dm.send(content="יש בקשה לקניית מערכת מיוחדת חדשה", embed=owner_embed, view=view)
            await bot.services.special_systems.set_owner_message(order.id, owner_message.id)
            notice = "הבקשה נשלחה בהצלחה. נחזור אליך ב-DM אחרי שנבדוק אותה."
            success_html = f"""
            {_notice_html(notice, success=True)}
            <div class="card">
                <h2>הבקשה נשלחה</h2>
                <p>שלחנו לבעלים הודעה חדשה עם כל הפרטים, והבקשה מחכה עכשיו ברשימת האדמין.</p>
                <div class="actions">
                    <a class="link-button" href="/special-systems/{_escape(special_system.slug)}">שלח בקשה נוספת</a>
                </div>
            </div>
            """
            body = _public_shell(
                session,
                current_path="/special-systems",
                title=f"הזמנה מיוחדת - {special_system.title}",
                intro="הבקשה שלך התקבלה ונשמרה בבוט.",
                login_path=f"/special-systems/{special_system.slug}",
                content=success_html,
            )
            return _page_response(f"הזמנה מיוחדת - {special_system.title}", body)
        gallery_html = ""
        if images:
            gallery_html = f'<div class="card stack gallery-section"><h2>תמונות המערכת</h2>{_render_image_slider(_special_gallery_urls(images), alt_text=special_system.title, empty_label="אין כרגע תמונות תצוגה")}</div>'
        linked_label = "לא מחובר"
        if linked_account is not None:
            linked_label = " | ".join(part for part in (linked_account.roblox_display_name, linked_account.roblox_username, linked_account.roblox_sub) if part)
        content = f"""
        {_notice_html(notice, success=success)}
        <div class="split-grid">
            <div class="card stack">
                <div><h2>{_escape(special_system.title)}</h2><p>{_escape(special_system.description)}</p></div>
                <div><h3>אמצעי תשלום</h3><div class="price-list">{''.join(f'<div class="price-item"><strong>{_escape(method.label)}</strong><span>{_escape(method.price)}</span></div>' for method in special_system.payment_methods)}</div></div>
            </div>
            <div class="card">
                <h2>טופס הזמנה</h2>
                <p class="muted">אפשר לשלוח בקשה גם בלי חשבון רובלוקס מחובר. אם כבר חיברת רובלוקס, נצרף אותו אוטומטית לבקשה.</p>
                <div class="meta-card"><p><strong>חשבון רובלוקס מחובר:</strong> {_escape(linked_label)}</p></div>
                <form method="post">
                    <div class="grid">
                        <label class="field field-wide"><span>איזה שיטת תשלום אתה משלם</span><select name="payment_method" required>{_payment_method_select_options(special_system, selected_payment_method)}</select></label>
                        <label class="field"><span>מה השם שלך ברובלוקס</span><input type="text" name="roblox_name" value="{_escape(roblox_name)}" required></label>
                        <label class="field"><span>מה השם שלך בדיסקורד</span><input type="text" name="discord_name" value="{_escape(discord_name)}" required></label>
                    </div>
                    <div class="actions"><button type="submit">שלח בקשה</button></div>
                </form>
            </div>
        </div>
        {gallery_html}
        """
        body = _public_shell(
            session,
            current_path="/special-systems",
            title=f"הזמנה מיוחדת - {special_system.title}",
            intro="מלא את כל הפרטים כדי לשלוח בקשה חדשה לבוט. כל השדות חובה.",
            login_path=f"/special-systems/{special_system.slug}",
            content=content,
        )
        return _page_response(f"הזמנה מיוחדת - {special_system.title}", body)
    except web.HTTPException:
        raise
    except SalesBotError as exc:
        return _error_response("הזמנה מיוחדת", str(exc), status=400)
