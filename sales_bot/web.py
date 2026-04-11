from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from aiohttp import web

from sales_bot.exceptions import ExternalServiceError, NotFoundError

if TYPE_CHECKING:
    from sales_bot.bot import SalesBot


LOGGER = logging.getLogger(__name__)


def create_web_app(bot: "SalesBot") -> web.Application:
    app = web.Application()
    app["bot"] = bot
    app.router.add_get("/health", healthcheck)
    app.router.add_get("/oauth/roblox/callback", roblox_callback)
    app.router.add_post("/webhooks/paypal/simulate", paypal_webhook)
    return app


async def healthcheck(request: web.Request) -> web.Response:
    return web.json_response({"status": "ok"})


async def roblox_callback(request: web.Request) -> web.Response:
    bot: SalesBot = request.app["bot"]
    state = request.query.get("state", "")
    code = request.query.get("code", "")

    if not state or not code:
        return web.Response(text="Missing state or code.", status=400)

    try:
        user_id = await bot.services.oauth.consume_state(state)
        tokens = await bot.services.oauth.exchange_code(bot.http_session, code)
        profile = await bot.services.oauth.fetch_profile(bot.http_session, tokens["access_token"])
        record = await bot.services.oauth.link_account(user_id, profile)

        user = await bot.fetch_user(user_id)
        await user.send(
            f"Your Roblox account has been linked successfully as **{record.roblox_username or record.roblox_sub}**."
        )
    except (NotFoundError, ExternalServiceError) as exc:
        LOGGER.warning("Roblox OAuth callback failed: %s", exc)
        return web.Response(text=f"Link failed: {exc}", status=400)
    except Exception:
        LOGGER.exception("Unexpected Roblox OAuth callback failure")
        return web.Response(text="Unexpected OAuth error.", status=500)

    return web.Response(
        text="Roblox account linked successfully. You can close this window.",
        content_type="text/plain",
    )


async def paypal_webhook(request: web.Request) -> web.Response:
    bot: SalesBot = request.app["bot"]
    provided_token = request.headers.get("X-Webhook-Token", "")
    if provided_token != bot.settings.paypal_webhook_token:
        return web.json_response({"error": "unauthorized"}, status=401)

    payload: dict[str, Any] = await request.json()
    purchase_id = int(payload.get("purchase_id", 0))
    status = str(payload.get("status", "")).upper()

    if purchase_id <= 0 or status != "COMPLETED":
        return web.json_response({"error": "invalid payload"}, status=400)

    try:
        await bot.services.payments.complete_purchase(bot, purchase_id, payload)
    except NotFoundError as exc:
        return web.json_response({"error": str(exc)}, status=404)
    except Exception:
        LOGGER.exception("PayPal simulation webhook failed")
        return web.json_response({"error": "internal error"}, status=500)

    return web.json_response({"status": "completed", "purchase_id": purchase_id})
